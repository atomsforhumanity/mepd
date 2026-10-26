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
  the notes).

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
                  cutoff_kcal: Optional[float]) -> tuple[set[int], list[int]]:
    """(species, step indices) the kinetics treat: reached from species 0
    climbing at most `cutoff_kcal` (all of them without a cutoff)."""
    if cutoff_kcal is None:
        return set(range(len(energies))), list(range(len(steps)))
    from mepd.network_aware import accessibility

    acc = accessibility(steps, energies, e0)
    reach = {k for k, st in acc.items() if st.span <= cutoff_kcal} | {0}
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


def _reoptimize(item: Item, high, free_high: Optional[FreeEnergies]) -> dict:
    """Free energy (Hartree) of `item` at the higher level, re-converged."""
    import mepd
    from mepd.compare import same_molecule
    from mepd.nodes.node import StructureNode

    if item.kind == "species":
        total, thermal = 0.0, 0.0
        for s in item.parts:
            start = StructureNode(structure=s)
            opt = mepd.optimize(StructureNode(structure=s), high)
            if not same_molecule(opt, start):
                raise RuntimeError("its bonds changed on minimization at the higher level")
            total += _energy(opt.structure, high)
            if free_high is not None:
                thermal += free_high.corrections([opt.structure], [False])[0]
        return {"e_high": total, "g_high": total + thermal / HARTREE_TO_KCAL_PER_MOL, "structures": None}
    ts = mepd.find_ts(StructureNode(structure=item.structure), high)
    path = mepd.irc(ts, high).path
    first, last = path.nodes[0], path.nodes[-1]
    a, b = (StructureNode(structure=s) for s in item.ends)
    if not ((same_molecule(first, a) and same_molecule(last, b)) or
            (same_molecule(first, b) and same_molecule(last, a))):
        raise RuntimeError("its IRC at the higher level connects other species")
    e = _energy(ts.structure, high)
    thermal = free_high.corrections([ts.structure], [True])[0] if free_high is not None else 0.0
    return {"e_high": e, "g_high": e + thermal / HARTREE_TO_KCAL_PER_MOL, "structures": ts.structure}


def refine_energies(species: Sequence, steps: Sequence[Step], *, low_engine, high_engine,
                    temperature: float, time_s: float, cutoff_kcal: Optional[float] = None,
                    mode: str = "single-point", free_energy: bool = True, standard_state: str = "1M",
                    workers: int = 1, on_note: Optional[Callable[[str], None]] = None) -> dict:
    """`species`: Structures at the explorer's level (index 0 the seed);
    `steps`: its verified steps, each with files["ts"] (the TS geometry).
    Returns the low- and high-level energies of every refined species and
    TS, and the kinetics on both (see the module docstring)."""
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
    reach, open_steps = kinetic_reach(steps, e_low, e0, cutoff_kcal)
    ref = species[0]
    ts_structs = {}
    for m in open_steps:
        fp = (steps[m].files or {}).get("ts")
        if not fp or not Path(fp).exists():
            raise FileNotFoundError(f"step {steps[m].label} has no TS geometry ({fp})")
        ts = Structure.open(str(fp))
        ts_structs[m] = ts.model_copy(update={"charge": ref.charge, "multiplicity": ref.multiplicity})

    # The explorer's own free energies (its level, its geometries)
    free_low = FreeEnergies(low_engine, temperature, standard_state, workers) if free_energy else None
    sp_idx = sorted(reach)
    if free_low is not None:
        corr = free_low.corrections([species[k] for k in sp_idx] + [ts_structs[m] for m in open_steps],
                                    [False] * len(sp_idx) + [True] * len(open_steps))
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
    items += [Item("ts", m, ts_structs[m], [ts_structs[m]], (species[steps[m].a], species[steps[m].b]))
              for m in open_steps]
    for n in dict.fromkeys(splitter.notes):
        say(f"free energies: {n}")

    # Single points everywhere first: they are what "reoptimize" falls back to
    threads = max(1, int(getattr(high_engine, "threads", 1) or 1))
    at_once = max(1, int(workers) // threads)

    def sp(item):
        try:
            return _single_point(item, low_engine, high_engine)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    sps = list(_fork_map(sp, items, at_once))
    failed = [(it, r) for it, r in zip(items, sps) if isinstance(r, str)]
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

    if mode == "reoptimize":
        free_high = FreeEnergies(high_engine, temperature, standard_state, 1) if free_energy else None

        def reopt(item):
            try:
                return _reoptimize(item, high_engine, free_high)
            except Exception as exc:
                return f"{type(exc).__name__}: {exc}"

        for it, row, r in zip(items, rows, _fork_map(reopt, items, at_once)):
            label = f"species {it.index}" if it.kind == "species" else f"TS of step {steps[it.index].label}"
            if isinstance(r, str):
                say(f"{label}: not re-converged at the higher level ({r}); its single-point free energy is used")
                continue
            row.update(how="reoptimized", e_high=r["e_high"])
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
    kin_low = simulate(len(species), low_steps, g_low, temperature=temperature, time_s=time_s)
    kin_high = simulate(len(species), high_steps, g_high, temperature=temperature, time_s=time_s)

    def rel(x):
        return (x - g_high[0]) * HARTREE_TO_KCAL_PER_MOL

    def rel_low(x):
        return (x - g_low[0]) * HARTREE_TO_KCAL_PER_MOL

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
               standard_state: Optional[str] = None, on_note: Optional[Callable[[str], None]] = None) -> dict:
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
        standard_state=standard_state or steer.get("standard_state", "1M"), workers=workers, on_note=on_note)
    for row in out["refined"]:
        if row["kind"] == "species":
            row["smiles"] = summary["species"][row["index"]].get("smiles")
    return out
