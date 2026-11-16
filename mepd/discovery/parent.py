"""A finished `mepd discovery expand` run as the parent network of `mepd addmol` (docs/add-molecule.md): the
explorer's own molecule-level network (mepd.discovery.molecules.build: its molecules, its reactions, its rules
for what a TS counts as), on the run's own energies, with every IRC-verified step in it whether the run's
kinetics had it open or closed.

Nothing of the explorer's rules is written again here. This module reads the run's files, puts each box and
each TS on the run's free-energy scale (electronic energy + the run's recorded thermal correction and steering
shift; a correction the run never computed is computed now by mepd.discovery.free_energy.FreeEnergies, the
code the run used), and calls `molecules.build`, `Mirrors.extend`, `molecules.split`, `free_energy.step_ends`.

`ts_step`: a TS with an added molecule, given as a stationary point with its two ends, on the same scale:
G(TS, counted as the explorer counts a TS file) minus G of the separated molecules of its start.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

H_KCAL = 627.509474
BOHR = 0.529177210903
CLOSED = 1.0        # Hartree: a step not used is put this far above its ends (molecules.CLOSED_HARTREE is 0.5)


def is_run(path: Path) -> bool:
    path = Path(path)
    return (path / "summary.json").is_file() and (path / "species").is_dir()


def _here(run_dir: Path, recorded: Optional[str]) -> Optional[Path]:
    """A file the summary names (as the run saw it, e.g. run/ts/x.xyz) inside `run_dir`."""
    if not recorded:
        return None
    parts = Path(recorded).parts
    for k in range(len(parts)):
        fp = run_dir.joinpath(*parts[k:])
        if fp.exists():
            return fp
    return None


def profile_of(run_dir: Path) -> Optional[Path]:
    """The profile kept beside the run: a .toml with an engine_name in its folder or the one above."""
    run_dir = Path(run_dir)
    for fp in [*sorted(run_dir.glob("*.toml")), *sorted(run_dir.parent.glob("*.toml"))]:
        try:
            if "engine_name" in fp.read_text():
                return fp
        except OSError:
            continue
    return None


def _load(run_dir: Path):
    from qcdata import Structure

    from mepd.discovery.kinetics import Step
    from mepd.nodes.node import StructureNode

    summary = json.loads((run_dir / "summary.json").read_text())
    seed = Structure.open(str(run_dir / "species" / "species_0.xyz"))
    q = {"charge": int(seed.charge), "multiplicity": int(seed.multiplicity)}
    nodes = []
    for s in summary["species"]:
        st = Structure.open(str(run_dir / "species" / f"species_{s['index']}.xyz")).model_copy(update=q)
        node = StructureNode(structure=st)
        nodes.append(node)
    steps = []
    for st in summary["steps"]:
        files = dict(st.get("files") or {})
        for key in ("ts", "irc"):
            if files.get(key):
                fp = _here(run_dir, files[key])
                files[key] = str(fp) if fp else None
        steps.append(Step(int(st["a"]), int(st["b"]), float(st["ts_energy"]), st.get("label") or "", files))
    return summary, nodes, steps, q


def network(run_dir, engine=None, *, temperature: Optional[float] = None, workers: int = 1) -> dict:
    """The run's molecule-level network as mepd.addmol takes it, with each species' coordinates and each
    step's frames (the reacting molecules' atoms, in the TS's order).

    Steps: every IRC-verified step of the run with a TS file. Energies: free, relative to the seed's molecules
    (1 M or the run's standard state), at the run's temperature (another `temperature` recomputes every
    correction). A box or TS whose thermal correction the run did not record gets it from `engine` (the run's
    profile's) with the run's own code; without an engine, or where that fails, the step is left out and
    listed under "excluded" (reason "thermal_correction_missing"): the molecules' free energies are fitted over
    the boxes, and one box on an electronic energy would move them all."""
    from mepd.discovery import molecules as M
    from mepd.discovery.free_energy import FreeEnergies, optimized_on, step_ends
    from mepd.discovery.kinetics import Step
    from mepd.discovery.mirror import Mirrors

    run_dir = Path(run_dir)
    summary, nodes, steps, q = _load(run_dir)
    steering = summary.get("steering") or {}
    T_run = float(steering.get("temperature_K") or 298.15)
    T = float(temperature) if temperature is not None else T_run
    standard = "1atm" if str(steering.get("standard_state", "1M")).lower().startswith("1atm") else "1M"
    sp, st_rec = summary["species"], summary["steps"]
    n = len(nodes)

    def ts_structure(m):
        from qcdata import Structure

        fp = steps[m].files.get("ts")
        return Structure.open(fp).model_copy(update=q) if fp else None

    use = [m for m, st in enumerate(steps) if st.files.get("verified") is True and st.files.get("ts")
           and st.files.get("ts_kind") != "path_peak"]
    boxes = sorted({0} | {k for m in use for k in (steps[m].a, steps[m].b)})
    # thermal corrections: the run's where it recorded one (at its temperature), else computed now
    recorded = T == T_run
    c_box = {k: (sp[k].get("energy_terms") or {}).get("thermal_correction_kcal") if recorded else None for k in boxes}
    c_ts = {m: (st_rec[m].get("ts_energy_terms") or {}).get("thermal_correction_kcal") if recorded else None for m in use}
    computed = {"species": 0, "ts": 0, "failed": 0}
    ts_cache = {m: ts_structure(m) for m in use}
    ends = {m: step_ends(steps[m].files, (nodes[steps[m].a].structure, nodes[steps[m].b].structure)) for m in use}
    if engine is not None:
        free = FreeEnergies(engine, T, standard, workers)
        need_b = [k for k in boxes if c_box[k] is None]
        need_t = [m for m in use if c_ts[m] is None]
        got = free.corrections([nodes[k].structure for k in need_b] + [ts_cache[m] for m in need_t],
                               [False] * len(need_b) + [True] * len(need_t),
                               [None] * len(need_b) + [ends[m] for m in need_t],
                               [float(sp[k]["energy"]) for k in need_b] + [steps[m].ts_energy for m in need_t])
        for k, c in zip(need_b, got[:len(need_b)]):
            c_box[k] = c
            computed["species" if c is not None else "failed"] += 1
        for m, c in zip(need_t, got[len(need_b):]):
            c_ts[m] = c
            computed["ts" if c is not None else "failed"] += 1

    def shift(terms):
        # the steering level's shift the run's kinetics used: its single point, or the shift it lent a structure
        # outside its deciding windows (recorded relative to the seed's)
        terms = terms or {}
        return float(terms["steer_shift_kcal"]) if terms.get("steer") in ("single_point", "lent") and terms.get(
            "steer_shift_kcal") is not None else 0.0

    # a step with a correction missing is not used: put above its ends like every step that is not
    missing = [m for m in use if c_box[steps[m].a] is None or c_box[steps[m].b] is None or c_ts[m] is None]
    use = [m for m in use if m not in set(missing)] if c_box.get(0) is not None else []
    g_box = [float(sp[k]["energy"]) for k in range(n)]
    for k in boxes:
        if c_box[k] is not None:
            g_box[k] = float(sp[k]["energy"]) + (c_box[k] + shift(sp[k].get("energy_terms"))) / H_KCAL
    used = set(use)
    view = []
    for m, st in enumerate(steps):
        if m in used:
            g_ts = st.ts_energy + (c_ts[m] + shift(st_rec[m].get("ts_energy_terms"))) / H_KCAL
        else:
            g_ts = max(g_box[st.a], g_box[st.b]) + CLOSED
        view.append(Step(st.a, st.b, g_ts, st.label, st.files))
    # mirror images, as the run decided them
    rec = summary.get("mirror_images") if isinstance(summary.get("mirror_images"), dict) else {}
    mirrors = Mirrors(n, [v["mirror_of"] for v in rec.get("virtual_species", [])], list(rec.get("species_mirror") or []),
                      {int(k): v for k, v in (rec.get("chiral_conformers") or {}).items()},
                      [(t["step"], t["a"], t["b"]) for t in rec.get("added_twins", [])],
                      [tuple(p) for p in rec.get("found_twins", [])])
    _n_all, all_steps, all_g = mirrors.extend(view, g_box, T)

    parts_cache: dict = {}

    def parts(k):
        if k not in parts_cache:
            parts_cache[k] = (M.molecule_parts(nodes[k].structure), M.bonds_of(nodes[k].structure))
        return parts_cache[k][0]

    def bonds(k):
        parts(k)
        return parts_cache[k][1]

    def ts_info(m):
        # (as network_expansion._molecule_view tells it to molecules.build)
        ts = ts_cache.get(m) or ts_structure(m)
        fp = steps[m].files.get("ts")
        on = optimized_on(ts) if ts is not None else None
        if on is not None:
            has_ends = len(on) < len(ts.symbols)
        else:
            has_ends = bool(fp) and Path(fp).with_name(f"{Path(fp).stem}_ends.xyz").exists()
        return ts, has_ends, steps[m].files.get("verified") is not False

    net = M.build(n, parts, bonds, all_steps, all_g, mirrors.virtual_of, [t[0] for t in mirrors.twins],
                  mirrors.found_twins, ts_info)

    # species: each molecule's geometry from the lowest box that holds it
    where: dict = {}
    for k in sorted(boxes, key=lambda k: g_box[k]):
        for atoms, key in parts(k) or []:
            where.setdefault(key, (k, sorted(atoms)))
    species = []
    for i, key in enumerate(net.names):
        s = {"id": key, "name": key, "g": float(net.g[i]), "smiles": None if M.complex_info(key) else key,
             "charge": 0, "multiplicity": 1, "status": "verified", "level": "explorer"}
        hit = where.get(key) or where.get(M.canonical(key, mirror=True))
        if hit is not None:
            k, atoms = hit
            xyz = np.asarray(nodes[k].structure.geometry, dtype=float).reshape(-1, 3)[atoms] * BOHR
            s.update(atoms=[str(nodes[k].structure.symbols[a]) for a in atoms], coordinates=xyz.tolist(),
                     **({"mirror_image_shown": True} if key not in where else {}))
        species.append(s)
    out_steps = []
    for x in net.reactions:
        m = int(x["step"])
        st = steps[m]
        r, p = [net.names[i] for i in x["r"]], [net.names[i] for i in x["p"]]
        both = sorted((set(r) & set(p)))
        row = {"id": st.label + ("~mirror" if x["twin"] else ""), "reactants": r, "products": p, "shuttles": both,
               "label": " + ".join(r) + " -> " + " + ".join(p), "g_ts": float(x["g_ts"]), "status": "verified",
               "level": "explorer", "energies": "free", "run_step": st.label,
               "geometry": None}
        if x["twin"]:
            row["mirror_of"] = st.label
        else:
            row["geometry"] = _frames(M, x, st, ts_cache[m], ends[m], parts, bonds, q)
        out_steps.append(row)
    notes = [f"free energies: the run's own (electronic energy + its thermal correction and steering shift, "
             f"{standard}), molecules fitted over boxes (mepd.discovery.molecules)"]
    if any(computed.values()):
        notes.append(f"thermal corrections the run had not recorded, computed now: {computed['species']} boxes, "
                     f"{computed['ts']} TSs; {computed['failed']} failed")
    if missing:
        notes.append(f"{len(missing)} IRC-verified steps left out: a thermal correction of theirs is not recorded "
                     "and was not computed (give the run's profile)")
    return {"temperature": T, "energies": "free", "source": "explorer run", "species": species, "steps": out_steps,
            "energy_reference": "g and g_ts are free energies relative to the starting box's molecules, as the "
                                "explorer's molecule-level kinetics count them.",
            "excluded": [{"step": steps[j].label, **why} for j, why in sorted(net.left_out.items()) if j in used]
            + [{"step": steps[m].label, "reason": "thermal_correction_missing"} for m in missing],
            "notes": notes, "counts": dict(net.counts), "fit": dict(net.fit),
            "run": {"steps": len(steps), "verified": len(use), "species": n,
                    "seed": [net.names[i] for i in net.seed]}}


def _frames(M, x, st, ts, ends, parts, bonds, q) -> Optional[dict]:
    """The reacting molecules of a step in its TS's atom order: reactant end, TS, product end (the TS's IRC
    ends, else the ends its search started from), and the bonds that change."""
    from mepd.discovery.free_energy import bonds as bonds_at

    s = M.split(parts(st.a), parts(st.b), bonds(st.a), bonds(st.b))
    if s[0] != "ok" or ts is None or ends is None:
        return None
    react, spectators = set(s[3]), s[4]
    if x.get("with_neighbours"):
        for atoms in M._neighbours(ts, sorted(react), spectators):
            react |= set(atoms)
    atoms = sorted(react)
    sym = [str(ts.symbols[a]) for a in atoms]
    xyz = lambda structure: np.asarray(structure.geometry, dtype=float).reshape(-1, 3)[atoms] * BOHR   # noqa: E731
    try:
        first, last = xyz(ends[0]), xyz(ends[1])
    except Exception:
        return None
    # which end is the reactant side (box a): by the bonds among the reacting atoms
    idx = {a: i for i, a in enumerate(atoms)}
    pairs = lambda bs: {tuple(sorted((int(i), int(j)))) for i, j in bs}       # noqa: E731
    want = pairs((idx[i], idx[j]) for i, j in bonds(st.a) if i in idx and j in idx)
    if pairs(bonds_at(sym, last)) == want and pairs(bonds_at(sym, first)) != want:
        first, last = last, first
    b0, b1 = pairs(bonds_at(sym, first)), pairs(bonds_at(sym, last))
    return {"atoms": sym, "atom_indices_in_run": atoms, "charge": q["charge"], "multiplicity": q["multiplicity"],
            "reactant": first.tolist(), "ts": xyz(ts).tolist(), "product": last.tolist(),
            "bonds_broken": sorted(list(b) for b in b0 - b1), "bonds_formed": sorted(list(b) for b in b1 - b0)}


def query(run_dir, engine=None, *, target: str, initial: dict, time_s: float, temperature: Optional[float] = None,
          held=(), fraction: float = 0.5, workers: int = 1) -> dict:
    """The query file of an explorer run (mepd.addmol.query on `network`)."""
    from mepd import addmol

    net = network(run_dir, engine, temperature=temperature, workers=workers)
    fp = profile_of(Path(run_dir))
    return addmol.query(net, target=target, initial=initial, time_s=time_s, held=held, fraction=fraction,
                        level={"key": "explorer", "profile": fp.read_text() if fp else None})


def ts_step(ts: dict, reactant: list, product: list, engine, *, temperature: float = 298.15,
            standard_state: str = "1M") -> dict:
    """A step whose TS is a stationary point (verified, or a channel search's) on the parent's scale as a
    whole: G(TS) minus G of the separated molecules of its reactant end, quasi-RRHO, by the explorer's own
    free-energy code (mepd.discovery.free_energy.FreeEnergies; the TS counted as its file says it was optimized
    on, else by its two ends: the molecules that take no part apart). Never an electronic barrier from a
    complex on top of a separately estimated binding.

    ts: {"atoms", "coordinates" (Angstrom), "charge", "multiplicity", "comment" (the TS file's comment line,
    if any)}; reactant, product: the two ends' coordinates in the TS's atom order.
    Returns {"barrier": {"electronic", "free"} above the separated, minimized molecules of the reactant end,
    "reactants", "products": the molecules that take part ([{"smiles", "atoms": indices}]), "spectators",
    "counted": the atom counts the TS's correction counted, "method"}."""
    from qcdata import Structure

    from mepd.discovery import molecules as M
    from mepd.discovery.free_energy import FreeEnergies
    from mepd.nodes.node import StructureNode

    q = {"charge": int(ts.get("charge", 0)), "multiplicity": int(ts.get("multiplicity", 1))}
    sym = list(ts["atoms"])

    def structure(xyz, comment=None):
        s = Structure(symbols=sym, geometry=np.asarray(xyz, dtype=float) / BOHR, **q)
        return s.model_copy(update={"extras": {"xyz_comments": [comment]}}) if comment else s

    s_ts, s_r, s_p = structure(ts["coordinates"], ts.get("comment")), structure(reactant), structure(product)
    free = FreeEnergies(engine, float(temperature), standard_state)
    e_ts = float(engine.compute_energies([StructureNode(structure=s_ts)])[0])
    (c_ts,) = free.corrections([s_ts], [True], [(s_r, s_p)], [e_ts])
    if c_ts is None:
        raise RuntimeError("the TS's thermal correction failed: " + (free.notes[-1] if free.notes else "unknown"))
    counted = free.parts(s_ts, True, (s_r, s_p))
    taking_part = set(counted[0][0]) if len(counted) > 1 else set(range(len(sym)))
    # the separated molecules of the reactant end, each minimized alone
    g_apart = e_apart = 0.0
    for atoms, qi, mi in free.parts(s_r, False):
        xyz = np.asarray(s_r.geometry, dtype=float).reshape(-1, 3)[atoms]
        node = StructureNode(structure=Structure(symbols=[sym[a] for a in atoms], geometry=xyz, charge=qi,
                                                 multiplicity=mi))
        low = engine.compute_geometry_optimization(node)[-1]
        e = float(engine.compute_energies([low])[0])
        (c,) = free.corrections([low.structure], [False], None, [e])
        if c is None:
            raise RuntimeError("a separated molecule's thermal correction failed")
        e_apart += e
        g_apart += e + c / H_KCAL

    def side(s):
        mols = [{"smiles": key, "atoms": sorted(atoms)} for atoms, key in M.molecule_parts(s)]
        return ([m for m in mols if set(m["atoms"]) & taking_part],
                [m for m in mols if not set(m["atoms"]) & taking_part])

    (r_in, r_out), (p_in, _p_out) = side(s_r), side(s_p)
    return {"barrier": {"electronic": (e_ts - e_apart) * H_KCAL, "free": (e_ts - g_apart) * H_KCAL + c_ts},
            "reactants": r_in, "products": p_in, "spectators": r_out, "counted": [len(c[0]) for c in counted],
            "method": f"G(TS) - G(separated, minimized molecules of the reactant end), quasi-RRHO at {temperature:g} K, "
                      f"{standard_state}, {type(engine).__name__} Hessians (mepd.discovery.free_energy); the TS counted as "
                      + ("its reacting molecules, the others apart" if len(counted) > 1 else "one complex")}
