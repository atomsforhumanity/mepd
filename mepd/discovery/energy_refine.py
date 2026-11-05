"""Energies of an explored network at a higher level of theory (e.g. DFT).

The explorer finds species and transition states with a fast method
(g-xTB). Its geometries are usually good, but small energy differences --
which of two TSs is lower, so which product forms -- can be wrong by more
than they are worth: on cyclopentadiene + methyl acrylate g-xTB puts the exo
TS 0.5 kcal/mol under the endo one, the reverse of experiment, and
wB97X-D/def2-TZVP single points at the same structures put it right.

Two modes, like the implicit-solvent correction (mepd solvent):

* "single-point" (default, cheap): every species and TS the kinetics use
  keeps its explorer geometry and thermal correction; only the electronic
  energy is replaced, G = E_high + (G_low - E_low). The geometries are not
  stationary points of the higher level, so this is an approximation; it
  is the usual one (e.g. "DFT//g-xTB").
* "reoptimize": every minimum is re-minimized at the higher level (each
  molecule of a multi-molecule species on its own), every TS is searched
  again from the explorer's and followed by an IRC that must reach the same
  two species, and thermal corrections come from the higher level's own
  Hessians. A TS not recovered keeps its single-point free energy (said in
  the notes). How far each structure moved is recorded (`rmsd_angstrom`:
  aligned RMSD from the explorer's geometry, largest over the molecules),
  and with `save_dir` both geometries are written there.

With `save_dir`, every single point and re-convergence is kept as soon
as it ends, and running again picks up from what was kept (at the same
level of theory only): a re-convergence at DFT can take hours per TS.

Which part of the network: the species and steps the explorer's own
free-energy kinetics use (reached from the seed within the barrier cutoff);
the rest carries no flux and keeps the explorer's energies. A species of
several molecules counts as those molecules apart, as in the explorer's free
energies (mepd.discovery.free_energy), so each molecule gets its own
higher-level energy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from qcref.constants import HARTREE_TO_KCAL_PER_MOL

from mepd.discovery.free_energy import FreeEnergies
from mepd.discovery.kinetics import Step, simulate

MODES = ("single-point", "reoptimize")


def kinetic_reach(steps: Sequence[Step], energies: Sequence[float], e0: float,
                  cutoff_kcal: Optional[float], mirror_of: Optional[Sequence[int]] = None
                  ) -> tuple[set[int], list[int]]:
    """(species, step indices) the kinetics treat: reached from species 0
    climbing at most `cutoff_kcal` (all of them without a cutoff).
    `mirror_of`: each species' mirror image (-1: not in the network,
    mepd.discovery.mirror); a species is reached when its mirror image is,
    through the twins of the steps that reach that one."""
    if cutoff_kcal is None:
        return set(range(len(energies))), list(range(len(steps)))
    from mepd.network_aware import accessibility

    acc = accessibility(steps, energies, e0)
    reach = {k for k, st in acc.items() if st.span <= cutoff_kcal} | {0}
    if mirror_of is not None:
        reach |= {mirror_of[k] for k in reach if mirror_of[k] >= 0}
    open_steps = [m for m, st in enumerate(steps) if st.a in reach and st.b in reach
                  and (st.ts_energy - e0) * HARTREE_TO_KCAL_PER_MOL <= cutoff_kcal]
    return reach, open_steps


def _fragments(structure, parts: list) -> list:
    """One Structure per molecule ((atoms, charge, multiplicity) from
    FreeEnergies.parts); the structure itself when it is one molecule."""
    from qcdata import Structure

    if len(parts) == 1:
        return [structure]
    symbols = [str(s) for s in structure.symbols]
    xyz = np.asarray(structure.geometry, dtype=float)
    return [Structure(symbols=[symbols[k] for k in p], geometry=xyz[p], charge=q, multiplicity=m)
            for p, q, m in parts]


def _energy(structure, engine) -> float:
    """Energy in Hartree on a fresh node (a node caches the energy of the
    first engine that saw it)."""
    from mepd.nodes.node import StructureNode

    return float(engine.compute_energies([StructureNode(structure=structure)])[0])


@dataclass
class Item:
    """One species or TS to refine."""
    kind: str                  # "species" | "ts"
    index: int                 # species index, or step index
    structure: object
    parts: list = field(default_factory=list)   # one Structure per molecule
    ends: tuple = ()           # TS: the structures of the step's two species


def _single_point(item: Item, low, high) -> dict:
    e_low = sum(_energy(s, low) for s in item.parts)
    e_high = sum(_energy(s, high) for s in item.parts)
    return {"e_low": e_low, "e_high": e_high, "delta_kcal": (e_high - e_low) * HARTREE_TO_KCAL_PER_MOL}


def _level(engine) -> str:
    """What sets an engine's energies (its type and scientific settings, not
    threads, memory or paths): a saved result is reused only at the same level."""
    import dataclasses

    import inspect

    skip = {"threads", "memory", "maxcore", "python", "executable", "timeout", "keep_workdirs",
            "disable_molecular_graphs"}
    simple = (str, int, float, bool, tuple, list, dict)

    def value(name):
        # read without running a property: an MLIP engine's `calculator` loads the model (CUDA in this process)
        if name in vars(engine):
            return vars(engine)[name]
        return None if isinstance(inspect.getattr_static(engine, name, None), property) else getattr(engine, name, None)

    if dataclasses.is_dataclass(engine):
        fields = {f.name: value(f.name) for f in dataclasses.fields(engine)}
        # settings an engine keeps outside its dataclass fields (an MLIP's model name)
        fields.update({k: v for k, v in vars(engine).items() if k not in fields and isinstance(v, simple)})
    else:
        fields = {k: v for k, v in vars(engine).items() if isinstance(v, simple)}
    fields = {k: v for k, v in fields.items() if k not in skip and not k.startswith("_")}
    return f"{type(engine).__name__}{sorted((k, repr(v)) for k, v in fields.items())}"


class _Saved:
    """Results kept as each one finishes (`save_dir`/<stage>/<name>.json,
    geometries included), so a run that stops -- a time limit, a crash -- keeps what
    it finished, and the same command picks up from there."""

    def __init__(self, save_dir: Optional[Path], level: str):
        self.dir, self.level = (Path(save_dir) if save_dir is not None else None), level

    def get(self, stage: str, name: str) -> Optional[dict]:
        import json

        if self.dir is None or not (self.dir / stage / f"{name}.json").exists():
            return None
        rec = json.loads((self.dir / stage / f"{name}.json").read_text())
        return rec["result"] if rec.get("level") == self.level else None

    def put(self, stage: str, name: str, result: dict) -> None:
        import json

        if self.dir is None:
            return
        d = self.dir / stage
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{name}.json.tmp"
        tmp.write_text(json.dumps({"level": self.level, "result": result}))
        tmp.replace(d / f"{name}.json")


def _stem(item: Item) -> str:
    return f"species_{item.index}" if item.kind == "species" else f"ts_{item.index}" + (
        "_bound" if item.kind == "ts_bound" else "")


def _reoptimize(item: Item, high, free_high: Optional[FreeEnergies]) -> dict:
    """Free energy (Hartree) of `item` at the higher level, re-converged."""
    import mepd
    from mepd.compare import same_molecule
    from mepd.nodes.node import StructureNode

    if item.kind == "species":
        total, thermal, structures = 0.0, 0.0, []
        for s in item.parts:
            start = StructureNode(structure=s)
            opt = mepd.optimize(StructureNode(structure=s), high)
            if not same_molecule(opt, start):
                raise RuntimeError("its bonds changed on minimization at the higher level")
            total += _energy(opt.structure, high)
            structures.append(opt.structure)
            if free_high is not None:
                thermal += free_high.corrections([opt.structure], [False])[0]
        return {"e_high": total, "g_high": total + thermal / HARTREE_TO_KCAL_PER_MOL, "structures": structures}
    ts = mepd.find_ts(StructureNode(structure=item.structure), high)
    # the ends only need to be told apart (by their bonds), not frequency-checked
    path = mepd.irc(ts, high, end_keywords={"check_frequencies": False}).path
    first, last = path.nodes[0], path.nodes[-1]
    a, b = (StructureNode(structure=s) for s in item.ends)
    if not ((same_molecule(first, a) and same_molecule(last, b)) or
            (same_molecule(first, b) and same_molecule(last, a))):
        raise RuntimeError("its IRC at the higher level connects other species")
    e = _energy(ts.structure, high)
    thermal = (free_high.corrections([ts.structure], [True], [(first.structure, last.structure)])[0]
               if free_high is not None else 0.0)
    return {"e_high": e, "g_high": e + thermal / HARTREE_TO_KCAL_PER_MOL, "structures": [ts.structure]}


def _rmsd_angstrom(a, b) -> float:
    """RMSD (Angstrom) of two geometries of the same atoms after rigid alignment."""
    from mepd.rigid_alignment import kabsch_align
    from mepd.units import ANGSTROM_TO_BOHR

    x = np.asarray(b.geometry, dtype=float)
    y = kabsch_align(np.asarray(a.geometry, dtype=float), x)
    return float(np.sqrt(((y - x) ** 2).sum(axis=1).mean())) / ANGSTROM_TO_BOHR


def refine_energies(species: Sequence, steps: Sequence[Step], *, low_engine, high_engine,
                    temperature: float, time_s: float, cutoff_kcal: Optional[float] = None,
                    mode: str = "single-point", free_energy: bool = True, standard_state: str = "1M",
                    workers: int = 1, on_note: Optional[Callable[[str], None]] = None,
                    save_dir: Optional[Path] = None, mirror_images: bool = True) -> dict:
    """`species`: Structures at the explorer's level (index 0 the seed);
    `steps`: its verified steps, each with files["ts"] (the TS geometry).
    Returns the low- and high-level energies of every refined species and
    TS, and the kinetics on both (see the module docstring). `mirror_images`:
    the kinetics count every structure with its mirror image, as the
    explorer's do (mepd.discovery.mirror)."""
    from mepd.cli_common import _fork_map
    from qcdata import Structure

    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    note = on_note or (lambda s: None)
    notes: list[str] = []

    def say(msg):
        notes.append(msg)
        note(msg)

    e_low = [_energy(s, low_engine) for s in species]
    e0 = e_low[0]
    steps = list(steps)
    finder = mirror_of = None
    if mirror_images:
        from mepd.discovery.mirror import MirrorFinder
        from mepd.nodes.node import StructureNode

        finder = MirrorFinder()
        mirror_of = finder.species_mirrors([StructureNode(structure=s) for s in species])
    reach, open_steps = kinetic_reach(steps, e_low, e0, cutoff_kcal, mirror_of)
    ref = species[0]
    ts_structs = {}
    for m in open_steps:
        fp = (steps[m].files or {}).get("ts")
        if not fp or not Path(fp).exists():
            raise FileNotFoundError(f"step {steps[m].label} has no TS geometry ({fp})")
        ts = Structure.open(str(fp))
        ts_structs[m] = ts.model_copy(update={"charge": ref.charge, "multiplicity": ref.multiplicity})

    # The explorer's own free energies (its level, its geometries); a TS's spectators apart, as in the species
    from mepd.discovery.free_energy import step_ends

    ts_ends = {m: step_ends(steps[m].files or {}, (species[steps[m].a], species[steps[m].b])) for m in open_steps}
    free_low = FreeEnergies(low_engine, temperature, standard_state, workers) if free_energy else None
    sp_idx = sorted(reach)
    if free_low is not None:
        corr = free_low.corrections([species[k] for k in sp_idx] + [ts_structs[m] for m in open_steps],
                                    [False] * len(sp_idx) + [True] * len(open_steps),
                                    [None] * len(sp_idx) + [ts_ends[m] for m in open_steps])
        if any(c is None for c in corr):
            raise RuntimeError("thermal corrections failed at the explorer's level: " + "; ".join(free_low.notes))
        sp_corr = dict(zip(sp_idx, corr[:len(sp_idx)]))
        ts_corr = dict(zip(open_steps, corr[len(sp_idx):]))
    else:
        sp_corr = {k: 0.0 for k in sp_idx}
        ts_corr = {m: 0.0 for m in open_steps}
    g_low = [e + sp_corr.get(k, 0.0) / HARTREE_TO_KCAL_PER_MOL for k, e in enumerate(e_low)]
    ts_low = {m: steps[m].ts_energy + ts_corr[m] / HARTREE_TO_KCAL_PER_MOL for m in open_steps}

    splitter = free_low or FreeEnergies(low_engine, temperature, standard_state)
    items = [Item("species", k, species[k],
                  _fragments(species[k], splitter.parts(species[k], False) if free_energy else [None]))
             for k in sp_idx]
    items += [Item("ts", m, ts_structs[m],
                   _fragments(ts_structs[m], splitter.parts(ts_structs[m], True, ts_ends[m]) if free_energy else [None]),
                   (species[steps[m].a], species[steps[m].b]))
              for m in open_steps]
    for n in dict.fromkeys(splitter.notes):
        say(f"free energies: {n}")
    # a TS with spectators is two channels, apart and bound (free_energy.both_channels): the bound one its own item
    channels = {m: c for m in open_steps
                if free_low is not None and (c := free_low.ts_channels(ts_structs[m], ts_ends[m])) is not None}
    bound_items = [Item("ts_bound", m, ts_structs[m], [ts_structs[m]]) for m in channels]

    # Single points everywhere first: they are what "reoptimize" falls back to
    threads = max(1, int(getattr(high_engine, "threads", 1) or 1))
    at_once = max(1, int(workers) // threads)

    saved = _Saved(save_dir, f"{_level(low_engine)} -> {_level(high_engine)}")

    def sp(item):
        done = saved.get("single_points", _stem(item))
        if done is not None:
            return done
        try:
            r = _single_point(item, low_engine, high_engine)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        saved.put("single_points", _stem(item), r)
        return r

    sps = list(_fork_map(sp, items + bound_items, at_once))
    sps, sps_bound = sps[:len(items)], sps[len(items):]
    failed = [(it, r) for it, r in zip(items + bound_items, sps + sps_bound) if isinstance(r, str)]
    if failed:
        raise RuntimeError("higher-level single points failed: " +
                           "; ".join(f"{it.kind} {it.index}: {r}" for it, r in failed[:5]))
    g_high = list(g_low)
    ts_high = dict(ts_low)
    rows = []
    for it, r in zip(items, sps):
        d = r["delta_kcal"] / HARTREE_TO_KCAL_PER_MOL
        if it.kind == "species":
            g_high[it.index] = g_low[it.index] + d
        else:
            ts_high[it.index] = ts_low[it.index] + d
        rows.append({"kind": it.kind, "index": it.index, "molecules": len(it.parts),
                     "e_low": r["e_low"], "e_high": r["e_high"], "how": "single point"})
    if channels:
        from mepd.discovery.free_energy import channels_shift

        d_apart = {it.index: r["delta_kcal"] for it, r in zip(items, sps) if it.kind == "ts"}
        for it, r in zip(bound_items, sps_bound):
            ts_high[it.index] = ts_low[it.index] + channels_shift(
                *channels[it.index], d_apart[it.index], r["delta_kcal"], temperature) / HARTREE_TO_KCAL_PER_MOL
    rows_bound = [{"kind": it.kind, "index": it.index, "molecules": 1, "e_low": r["e_low"], "e_high": r["e_high"],
                   "how": "single point"} for it, r in zip(bound_items, sps_bound)]

    if mode == "reoptimize":
        free_high = FreeEnergies(high_engine, temperature, standard_state, 1) if free_energy else None

        def reopt(item):
            from qcdata import Structure

            done = saved.get("reoptimized", _stem(item))
            if done is not None:
                if "error" in done:
                    return done["error"]
                return {**done, "structures": [Structure.from_xyz(x) for x in done["structures"]]}
            import sys
            import time

            t0 = time.time()
            try:
                r = _reoptimize(item, high_engine, free_high)
            except Exception as exc:
                r = f"{type(exc).__name__}: {exc}"
            saved.put("reoptimized", _stem(item), {"error": r} if isinstance(r, str) else
                      {**r, "structures": [st.to_xyz() for st in r["structures"]]})
            print(f"{_stem(item)}: {'failed' if isinstance(r, str) else 're-converged'} in {time.time() - t0:.0f} s",
                  file=sys.stderr, flush=True)
            return r

        # the TS searches take longest: start them first
        order = sorted(range(len(items)), key=lambda i: items[i].kind != "ts")
        by_item = dict(zip(order, _fork_map(reopt, [items[i] for i in order], at_once)))
        for it, row, r in zip(items, rows, (by_item[i] for i in range(len(items)))):
            label = f"species {it.index}" if it.kind == "species" else f"TS of step {steps[it.index].label}"
            if isinstance(r, str):
                say(f"{label}: not re-converged at the higher level ({r}); its single-point free energy is used")
                continue
            lows = it.parts if it.kind == "species" else [it.structure]   # a TS is re-converged whole
            row.update(how="reoptimized", e_high=r["e_high"],
                       rmsd_angstrom=max(_rmsd_angstrom(lo, hi) for lo, hi in zip(lows, r["structures"])))
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                stem = f"species_{it.index}" if it.kind == "species" else f"ts_{it.index}"
                files = []
                for j, (lo, hi) in enumerate(zip(lows, r["structures"])):
                    part = f"_m{j}" if len(lows) > 1 else ""
                    for level, st in (("low", lo), ("high", hi)):
                        name = f"{stem}{part}_{level}.xyz"
                        st.save(save_dir / name)
                        files.append(name)
                row["files"] = files
            if it.kind == "species":
                g_high[it.index] = r["g_high"]
            else:
                ts_high[it.index] = r["g_high"]

    # Out of reach (no flux): the explorer's energies, moved onto the higher level's scale by the seed's
    # shift (absolute energies of two methods differ by hundreds of Hartree); their steps stay closed.
    shift = g_high[0] - g_low[0]
    for k in range(len(species)):
        if k not in reach:
            g_high[k] = g_low[k] + shift

    def closed(m, table, offset):
        return table[m] if m in table else steps[m].ts_energy + offset + 1.0

    low_steps = [Step(st.a, st.b, closed(m, ts_low, 0.0), st.label, st.files) for m, st in enumerate(steps)]
    high_steps = [Step(st.a, st.b, closed(m, ts_high, shift), st.label, st.files) for m, st in enumerate(steps)]
    mirrors = None
    n_low = n_high = len(species)
    g_low_k, g_high_k = g_low, g_high
    if finder is not None:
        from mepd.nodes.node import StructureNode

        mirrors = finder([StructureNode(structure=s) for s in species], steps,
                                 [ts_structs.get(m) for m in range(len(steps))],
                                 (lambda s: [g for g, _, _ in splitter.parts(s, False)]) if free_energy else None)
        for msg in dict.fromkeys(mirrors.undecided):
            say(f"mirror images: {msg}")
        n_low, low_steps, g_low_k = mirrors.extend(low_steps, g_low, temperature)
        n_high, high_steps, g_high_k = mirrors.extend(high_steps, g_high, temperature)
    kin_low = simulate(n_low, low_steps, g_low_k, temperature=temperature, time_s=time_s)
    kin_high = simulate(n_high, high_steps, g_high_k, temperature=temperature, time_s=time_s)

    def rel(x):
        return (x - g_high[0]) * HARTREE_TO_KCAL_PER_MOL

    def rel_low(x):
        return (x - g_low[0]) * HARTREE_TO_KCAL_PER_MOL

    rows += rows_bound   # (after the re-convergence, which takes a TS whole)
    for row in rows:
        if row["kind"] == "species":
            row.update(g_low_kcal=rel_low(g_low[row["index"]]), g_high_kcal=rel(g_high[row["index"]]),
                       final_low=kin_low.final_concentration[row["index"]],
                       final_high=kin_high.final_concentration[row["index"]])
        else:
            st = steps[row["index"]]
            row.update(a=st.a, b=st.b, label=st.label, g_low_kcal=rel_low(ts_low[row["index"]]),
                       g_high_kcal=rel(ts_high[row["index"]]))
    return {"mode": mode, "basis": "free energy" if free_energy else "electronic", "temperature_K": temperature,
            "time_s": time_s, "standard_state": standard_state, "cutoff_kcal": cutoff_kcal,
            "refined": rows, "notes": notes,
            **({"mirror_images": {**mirrors.record(), "virtual_species": [
                {**v, "final_low": kin_low.final_concentration[v["index"]],
                 "final_high": kin_high.final_concentration[v["index"]]}
                for v in mirrors.record()["virtual_species"]]}} if mirrors is not None else {}),
            "kinetics": {"low": {"final_concentration": kin_low.final_concentration,
                                 "max_concentration": kin_low.max_concentration, "flux": kin_low.flux},
                         "high": {"final_concentration": kin_high.final_concentration,
                                  "max_concentration": kin_high.max_concentration, "flux": kin_high.flux}}}


def _resolve(fp, run_dir: Path) -> Optional[Path]:
    """A path from summary.json, as written (cluster or relative paths)."""
    if not fp:
        return None
    p = Path(fp)
    for cand in (p, run_dir / p, run_dir.parent / p):
        if cand.exists():
            return cand
    parts = p.parts
    for k in range(1, len(parts)):       # a cluster path: its tail under this run
        cand = run_dir / Path(*parts[k:])
        if cand.exists():
            return cand
    return None


def refine_run(run_dir: Path, *, low_engine, high_engine, mode: str = "single-point", workers: int = 1,
               standard_state: Optional[str] = None, on_note: Optional[Callable[[str], None]] = None,
               mirror_images: bool = True) -> dict:
    """refine_energies on a `mepd discovery expand` output folder."""
    import json

    from qcdata import Structure

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    steer = summary.get("steering", {})
    species = []
    for rec in summary["species"]:
        fp = _resolve(rec.get("file"), run_dir) or run_dir / "species" / f"species_{rec['index']}.xyz"
        species.append(Structure.open(str(fp)))
    ref = species[0]
    species = [s.model_copy(update={"charge": ref.charge, "multiplicity": ref.multiplicity}) for s in species]
    steps = []
    for st in summary.get("steps", []):
        files = dict(st.get("files") or {})
        if files.get("ts"):
            files["ts"] = str(_resolve(files["ts"], run_dir) or files["ts"])
        steps.append(Step(int(st["a"]), int(st["b"]), float(st["ts_energy"]), st.get("label", ""), files))
    out = refine_energies(
        species, steps, low_engine=low_engine, high_engine=high_engine,
        temperature=float(steer.get("temperature_K", 298.15)), time_s=float(steer.get("time_s", 3600.0)),
        cutoff_kcal=steer.get("cutoff_kcal"), mode=mode,
        standard_state=standard_state or steer.get("standard_state", "1M"), workers=workers, on_note=on_note,
        save_dir=run_dir / "refined", mirror_images=mirror_images)
    for row in out["refined"]:
        if row["kind"] == "species":
            row["smiles"] = summary["species"][row["index"]].get("smiles")
    return out
