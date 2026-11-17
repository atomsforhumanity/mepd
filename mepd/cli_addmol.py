"""`mepd addmol`: add a molecule to a finished network (docs/add-molecule.md).

    mepd addmol adopt RUN WORKSPACE        a finished `mepd discovery expand` run into a workspace
    mepd addmol query WORKSPACE ...        the query file: the network and what limits a target
    mepd addmol test QUERY ...             candidates in, one result file each, updated as numbers come
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import typer

addmol_app = typer.Typer(help="Add a molecule to a finished network: what does it do to a target?",
                         no_args_is_help=True)


@addmol_app.command("adopt")
def adopt(
    run: Path = typer.Argument(..., exists=True, help="Output folder of a finished `mepd discovery expand` run "
                               "(summary.json, species/, ts/)."),
    workspace: Path = typer.Argument(..., help="Workspace folder (made if missing)."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="The run's profile (level of theory). "
                                          "Default: the profile .toml kept beside the run."),
) -> None:
    """A finished expansion into a workspace, as `mepd web` adopts a running one, and the energies of its
    mixtures' molecules (for the workspace's kinetics)."""
    from mepd.cli_common import _open_run_inputs
    from mepd.web import added
    from mepd.web.workspace import Workspace

    t0 = time.perf_counter()
    ws = Workspace(workspace)
    mgr = added.manager(ws)
    if inputs is not None:
        ws.write_profile(inputs.stem, inputs.read_text())
    job = added.adopt_run(mgr, run, profile=inputs.stem if inputs is not None else None)
    t1 = time.perf_counter()
    source = inputs or Path(job["level"]["source"])
    done = added.member_energies(ws, job, _open_run_inputs(source).engine)
    snap = ws.snapshot()
    typer.echo(json.dumps({
        "job": job["id"], "level": job["level"]["key"],
        "species": sum(1 for r in snap["structures"].values() if r.get("role") != "complex"),
        "mixtures": sum(1 for r in snap["structures"].values() if r.get("role") == "complex"),
        "edges_with_verified_ts": sum(1 for e in snap["edges"].values()
                                      if (e.get("origin") or {}).get("barrier_kcal") is not None
                                      and e["origin"].get("barrier_verified")),
        "molecules_minimized": done["done"], "failed": done["failed"],
        "adopt_s": round(t1 - t0, 1), "energies_s": round(time.perf_counter() - t1, 1)}, indent=1))


def _amounts(items: List[str]) -> dict:
    out = {}
    for item in items:
        name, _, amount = item.rpartition("=")
        if not name:
            raise typer.BadParameter(f"--initial {item!r}: give SPECIES=AMOUNT (mol/L)")
        out[name] = float(amount)
    return out


@addmol_app.command("query")
def query(
    workspace: Path = typer.Argument(..., exists=True, help="A workspace folder, or the output folder of a "
                                     "finished `mepd discovery expand` run (summary.json, species/, ts/): then "
                                     "the parent is the explorer's own molecule-level network."),
    target: str = typer.Option(..., "--target", help="The species to make more of: SMILES, name or id. A SMILES "
                               "without stereo marks counts its stereoisomers together."),
    initial: List[str] = typer.Option(..., "--initial", help="Starting amount, SPECIES=mol/L (repeat)."),
    held: List[str] = typer.Option([], "--held", help="A species kept at its starting amount (a constant feed; "
                                   "repeat)."),
    temperature: float = typer.Option(298.15, "--temperature", help="K."),
    time_s: float = typer.Option(86400.0, "--time", help="s."),
    fraction: float = typer.Option(0.5, "--fraction", help="The time to reach this fraction of the target's "
                                   "final amount is reported."),
    electronic: bool = typer.Option(False, "--electronic", help="Electronic energies instead of free energies "
                                    "(a workspace only)."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="An explorer run's profile, for the thermal "
                                          "corrections it did not record. Default: the .toml kept beside the run."),
    workers: int = typer.Option(1, "--workers", help="Processes for those thermal corrections."),
    output: Path = typer.Option(..., "--output", "-o", help="The query file (JSON)."),
) -> None:
    """The query file: conditions, species (SMILES, coordinates, amounts over time), steps (ends, TS, changing
    bonds, barriers) and what limits the target, readable without mepd."""
    from mepd import addmol
    from mepd.web import added
    from mepd.web.workspace import Workspace, WorkspaceError

    import os

    from mepd.orbitals import ORBITALS_ENV

    os.environ.pop(ORBITALS_ENV, None)      # (--output is a file here: no orbitals folder under its name)
    t0 = time.perf_counter()
    try:
        from mepd.discovery import parent
    except ImportError:
        parent = None
    try:
        if parent is not None and parent.is_run(workspace):
            from mepd.cli_common import _open_run_inputs

            profile = inputs or parent.profile_of(workspace)
            doc = parent.query(workspace, _open_run_inputs(profile).engine if profile else None, target=target,
                               initial=_amounts(initial), time_s=time_s, temperature=temperature, fraction=fraction,
                               workers=workers, held=held)
        else:
            ws = Workspace(workspace)
            doc = added.query(ws, added.manager(ws).jobs, target=target, initial=_amounts(initial),
                              temperature=temperature, time_s=time_s, free_energy=not electronic, fraction=fraction,
                              held=held)
    except (WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    addmol.write_json(output, doc)
    lim = doc["limits"]
    typer.echo(json.dumps({"query": str(output), "species": len(doc["species"]), "steps": len(doc["steps"]),
                           "target": doc["target"]["species"], "target_final": doc["target"]["final"],
                           "limits": lim.get("basis"), "why": lim.get("why"), "order": lim["order"][:5],
                           "wall_s": round(time.perf_counter() - t0, 1)}, indent=1))


def _candidates(smiles: List[str], files: Optional[Path], amount: Optional[float]) -> list:
    out = []
    for s in smiles:
        p = Path(s)
        if p.exists():
            from qcdata import Structure

            st = Structure.open(str(p))
            out.append({"name": p.stem, "atoms": list(st.symbols), "charge": int(st.charge),
                        "multiplicity": int(st.multiplicity),
                        "coordinates": [[float(v) * 0.529177210903 for v in row] for row in st.geometry]})
        else:
            out.append({"name": s, "smiles": s})
    if files is not None:
        loaded = json.loads(files.read_text())
        out += loaded if isinstance(loaded, list) else loaded["candidates"]
    names = set()
    for k, c in enumerate(out):
        if c.get("amount") is None:
            if amount is None:
                raise typer.BadParameter(f"candidate {c.get('name') or k}: no amount (give --amount, or \"amount\" in the file)")
            c["amount"] = amount
        c.setdefault("name", c.get("smiles") or f"candidate_{k}")
        if not (c.get("smiles") or (c.get("atoms") and c.get("coordinates"))):
            raise typer.BadParameter(f"candidate {c['name']}: give \"smiles\", or \"atoms\" and \"coordinates\"")
        if c["name"] in names:
            raise typer.BadParameter(f"two candidates are called {c['name']!r}")
        names.add(c["name"])
    return out


def _file_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:80] + ".json"


def _prepare(candidate: dict, engine) -> dict:
    """The candidate as the bystander layer takes it: its conformers (RDKit, then minimized at the level),
    lowest first, at most four; from coordinates, that structure minimized."""
    import numpy as np

    from mepd.conformers import generate_conformers
    from mepd.nodes.node import StructureNode

    if candidate.get("smiles"):
        from mepd.structures import smiles_to_structure

        st = smiles_to_structure(candidate["smiles"], candidate.get("charge"), candidate.get("multiplicity"))
        nodes = generate_conformers(StructureNode(structure=st))[:4]
    else:
        from qcdata import Structure

        st = Structure(symbols=list(candidate["atoms"]), geometry=np.asarray(candidate["coordinates"]) / 0.529177210903,
                       charge=int(candidate.get("charge", 0)), multiplicity=int(candidate.get("multiplicity", 1)))
        nodes = [StructureNode(structure=st)]
    best = [engine.compute_geometry_optimization(n)[-1] for n in nodes]
    energies = [float(e) for e in engine.compute_energies(best)]
    order = sorted(range(len(best)), key=lambda i: energies[i])
    return {"symbols": list(st.symbols), "charge": int(st.charge), "multiplicity": int(st.multiplicity),
            "conformers": [(np.asarray(best[i].structure.geometry) * 0.529177210903).tolist() for i in order]}


def _flat(smiles: Optional[str]) -> Optional[str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _c_key(molecule: dict) -> Optional[str]:
    """The added molecule's SMILES without stereo, from its geometry (what its molecules in a complex are told by)."""
    from mepd.discovery import molecules as M
    from mepd.web.variants import _structure

    return _flat(M.molecule_parts(_structure(molecule["symbols"], molecule["conformers"][0], molecule["charge"],
                                             molecule["multiplicity"]))[0][1])


def _species_ids(q: dict, mols: list, c_key: Optional[str], added: Optional[set] = None) -> list:
    """The species ids of a complex's molecules. `added`: the atoms of the molecule that is the added one (known
    from the TS's frames); without it, a molecule of the candidate's kind that is no species of the parent."""
    from mepd import addmol

    out = []
    for m in mols:
        if added is not None and set(m.get("atoms") or ()) == added:
            out.append(addmol.ADDED_ID)
            continue
        hit = addmol.match(q["species"], m["smiles"]) or ([addmol.ADDED_ID] if _flat(m["smiles"]) == c_key else [])
        if len(hit) != 1:
            raise ValueError(f"{m['smiles']} in its TS is no species of the network (a new species has to be "
                             "given under new_species)" if not hit else f"{m['smiles']} names several species")
        out.append(hit[0])
    return out


def _parent_of(q: dict, r: list, p: list) -> Optional[str]:
    """The parent's step between the same molecules without the added one (either way round; a molecule on both
    sides of it aside): its lowest."""
    from collections import Counter

    from mepd import addmol

    r, p = Counter(x for x in r if x != addmol.ADDED_ID), Counter(x for x in p if x != addmol.ADDED_ID)
    r, p = r - p, p - r
    same = []
    for st in q["steps"]:
        a, b = Counter(st["reactants"]), Counter(st["products"])
        a, b = a - b, b - a
        if (a, b) in ((r, p), (p, r)):
            same.append(st)
    return min(same, key=lambda st: st["g_ts"])["id"] if same else None


def _placed(q: dict, got: dict, molecule: dict, engine, added: Optional[tuple] = None) -> tuple:
    """A `ts_step` result as a step of the network: (the entry's fields, what a steering single point needs).
    Measured from the parent's own free energies of its molecules and the added molecule's lowest form over
    the candidate's conformers and the one this complex holds (mepd.discovery.parent.on_parent_scale)."""
    from mepd import addmol
    from mepd.discovery import parent
    from mepd.web.variants import _structure

    c_key = _c_key(molecule)
    # (`added`: the added molecule's atoms at the reactant and at the product end, where the TS's frames say which)
    a_r, a_p = added or (None, None)
    r, p, beside = (_species_ids(q, got[k], c_key, a) for k, a in (("reactants", a_r), ("products", a_p), ("spectators", a_r)))
    mine = [m for m, i in zip(got["reactants"] + got["spectators"], r + beside) if i == addmol.ADDED_ID]
    forms = [_structure(molecule["symbols"], c, molecule["charge"], molecule["multiplicity"])
             for c in molecule["conformers"]]
    forms += [_structure(m["symbols"], m["coordinates"], molecule["charge"], molecule["multiplicity"])
              for m in mine if m.get("coordinates")]
    low = parent.lowest_form(forms, engine, temperature=q["temperature"]) if mine else None
    placed = parent.on_parent_scale(got, r + beside, q, low["g_hartree"] if low else 0.0)
    fields = {"reactants": r, "products": p,
              "barrier": {"electronic": got["barrier"]["electronic"], "free": placed["free"]},
              "level": (q.get("level") or {}).get("key"), "reference": placed["reference"], "counted": got["counted"],
              "added_form": None if low is None else ("a conformer of the candidate" if low["index"] < len(
                  molecule["conformers"]) else "the form in this complex"),
              "free_from_molecules_minimized_alone": got["barrier"]["free"],
              "spectators": [m["smiles"] for m in got["spectators"]], "barrier_method": got["method"]}
    return fields, {"parts": got["ts_parts"], "added": low["structure"] if low else None, "n_added": len(mine)}


def _mirror_entry(q: dict, entry: dict, handed: bool) -> Optional[dict]:
    """The same step for the mirror-image twin of its parent step, for a candidate that is its own mirror image:
    the same numbers between the mirror-image species. None with a chiral candidate, or without a twin."""
    if handed or not entry.get("parent_step"):
        return None
    from mepd.discovery import molecules as M

    by_id = {st["id"]: st for st in q["steps"]}
    st = by_id.get(entry["parent_step"]) or {}
    twin = st.get("mirror") or st.get("mirror_of")
    if not twin or twin not in by_id:
        return None
    ids = {s["id"] for s in q["species"]}

    def image(x):
        m = M.canonical(x, mirror=True) if x in ids else x
        return m if m in ids else x

    twins = [by_id[x].get("mirror") or by_id[x].get("mirror_of") for x in entry.get("parent_steps") or [] if x in by_id]
    return {**entry, "id": f"{entry['id']}~mirror", "parent_step": twin, "mirror_image_of": entry["id"],
            **({"parent_steps": [x for x in twins if x]} if entry.get("parent_steps") else {}),
            "reactants": [image(x) for x in entry["reactants"]], "products": [image(x) for x in entry["products"]]}


def _ts_steps(q: dict, entries: list, molecule: dict, engine, failed: list, kept: dict) -> list:
    """The candidate's new steps given with a TS geometry and its two ends, each put on the parent's scale as a
    whole (mepd.discovery.parent.ts_step, `_placed`), and which species its reacting molecules are. An entry
    that already has a barrier and no TS is kept as given. `kept[id]`: what a steering single point of the step
    needs (its TS, how it is counted, the added molecule's reference form)."""
    import numpy as np

    from mepd.discovery import parent
    from mepd.web.variants import _structure

    done = []
    for n in entries:
        if not n.get("ts"):
            done.append(n)
            continue
        try:
            ends = n["ts"]["ends"]
            got = parent.ts_step({k: n["ts"].get(k) for k in ("atoms", "coordinates", "charge", "multiplicity", "comment")
                                  if n["ts"].get(k) is not None}, np.asarray(ends[0]), np.asarray(ends[1]), engine,
                                 temperature=q["temperature"])
            fields, steer = _placed(q, got, molecule, engine)
            kept[n.get("id")] = {"ts": _structure(n["ts"]["atoms"], n["ts"]["coordinates"], n["ts"].get("charge", 0),
                                                  n["ts"].get("multiplicity", 1)), **steer}
            # (the parent's own level: what the given `level` text said is kept beside it)
            done.append({"parent_step": _parent_of(q, fields["reactants"], fields["products"]),
                         **{k: v for k, v in n.items() if k not in ("ts", "level")}, **fields,
                         "level_given": n.get("level")})
        except Exception as exc:
            failed.append({"step": n.get("id"), "layer": n.get("layer") or "given", "status": "failed",
                           "why": f"{type(exc).__name__}: {exc}"})
    return done


def _steer(q: dict, est: dict, kept: dict, order: list, engine, steer_engine, lay: dict, failed: list, write) -> None:
    """The parent's steering level for the steps given with a TS (`kept`), in `order`: one single point of the
    TS (counted as its free energy counts it), of the added molecule's reference form and of the parent's
    starting box, by the run's own single-point code; the step's barrier moves by their difference and its level
    says so. The result file is rewritten after each step."""
    import numpy as np

    from mepd.discovery import parent
    from mepd.discovery.free_energy import FreeEnergies
    from mepd.web.variants import _structure

    t0 = time.perf_counter()
    ref = q.get("reference") or {}
    try:
        if not ref.get("seed_coordinates"):
            raise ValueError("the query has no starting box to measure the steering level from (a workspace's "
                             "network): its steps stay at the low level")
        seed = _structure(ref["seed_symbols"], ref["seed_coordinates"], ref.get("charge", 0), ref.get("multiplicity", 1))
        d_seed = parent.steer_shift(seed, FreeEnergies(engine, q["temperature"]).parts(seed, False), engine, steer_engine)
    except Exception as exc:
        lay["status"] = "failed"
        failed.append({"layer": "steer", "status": "failed", "why": f"{type(exc).__name__}: {exc}"})
        write("running")
        return
    d_added: dict = {}
    by_id = {n.get("id"): n for n in est["new_steps"]}
    for sid in order:
        n, k = by_id[sid], kept[sid]
        try:
            d_ts = parent.steer_shift(k["ts"], k["parts"], engine, steer_engine)
            d_c = 0.0
            if k["added"] is not None:
                key = np.round(np.asarray(k["added"].geometry, dtype=float), 4).tobytes()
                if key not in d_added:
                    d_added[key] = parent.steer_shift(k["added"], None, engine, steer_engine)
                d_c = d_added[key]
            shift = d_ts - k["n_added"] * d_c - d_seed
            n["barrier"] = {**n["barrier"], "free": n["barrier"]["free"] + shift}
            n["level"] = f"{n['level']}{parent.STEERED}"
            n["steer"] = {"shift_kcal": shift, "ts": d_ts, "added": d_c, "starting_box": d_seed}
        except Exception as exc:
            failed.append({"step": sid, "layer": "steer", "status": "failed", "why": f"{type(exc).__name__}: {exc}"})
        lay["done"] += 1
        lay["wall_s"] = round(time.perf_counter() - t0, 1)
        write("running")
    lay["status"] = "done"


_ENGINE = None
_INPUTS = None
_QUERY: dict = {}
QUICK = {"repeats": 1, "carry": 1, "polish": 0}     # the placer's quick setting (docs/bystander.md)


def _engine(profile):
    global _ENGINE, _INPUTS
    if _ENGINE is None:
        from mepd.cli_common import _open_run_inputs

        _INPUTS = _open_run_inputs(Path(profile))
        _ENGINE = _INPUTS.engine
    return _ENGINE


def _task(args):
    """One bystander estimate in a worker process (the engine is made once per process)."""
    profile, kind, ref, frames, molecule, temperature, starts, seed, setting = args
    from mepd import bystander as B

    try:
        return kind, ref, B.bystander(frames, molecule, _engine(profile), temperature=temperature, starts=starts,
                                      seed=seed, **setting)
    except Exception as exc:
        return kind, ref, {"status": "failed", "why": f"{type(exc).__name__}: {exc}", "binding": {}}


def _ts_row(q: dict, sid: str, label: str, layer: str, s_ts, first, last, c_atoms: set, molecule: dict, engine,
            start_keys: list, end_keys: list) -> dict:
    """A TS found with the molecule present, with its IRC's two ends (qcdata Structures, one atom order), read as
    the step `sid`: {"kind": "takes part" | "beside", "entry": the step on the parent's scale} when its IRC joins
    the step's two ends (the molecule's bonds changing between them or not), else {"kind": "elsewhere", "ends"}."""
    import numpy as np

    from mepd.discovery import molecules as M
    from mepd.discovery import parent

    BOHR = 0.529177210903
    ang = lambda st: np.asarray(st.geometry, dtype=float).reshape(-1, 3) * BOHR   # noqa: E731
    keys = lambda st: sorted(key for _a, key in M.molecule_parts(st))               # noqa: E731
    k0, k1 = keys(first), keys(last)
    if (k0, k1) == (end_keys, start_keys) and start_keys != end_keys:
        first, last, k0, k1 = last, first, k1, k0
    if (k0, k1) != (start_keys, end_keys):
        return {"kind": "elsewhere", "ends": [".".join(k0), ".".join(k1)]}
    sym = [str(x) for x in s_ts.symbols]
    ts = {"atoms": sym, "coordinates": ang(s_ts), "charge": int(s_ts.charge), "multiplicity": int(s_ts.multiplicity)}
    # Whether the molecule takes part is read in this TS's own frames, with the bonds its free energy is counted
    # with (mepd.discovery.free_energy): a whole molecule of the candidate's kind, the same atoms at both ends,
    # none of them in a bond that breaks or forms, is the added molecule beside the step; without one, it takes
    # part. (Earlier readings went wrong three ways: a looser bond criterion called a close contact a changed bond;
    # atom numbers from the scan's start called a molecule that had swapped a hydrogen with a neighbour in an
    # earlier step part of every later TS; and a molecule the free-energy code counts into the TS without any of
    # its bonds changing was called taking part. A candidate that is also a species of the parent is told from the
    # parent's own molecule of that kind the same way.)
    from mepd.discovery.free_energy import bonds

    pairs = lambda bs: {tuple(sorted((int(i), int(j)))) for i, j in bs}   # noqa: E731
    b0, b1 = pairs(bonds(sym, ang(first))), pairs(bonds(sym, ang(last)))
    moved = {i for pair in b0 ^ b1 for i in pair}
    c_key = _c_key(molecule)
    mine = lambda st: [frozenset(int(i) for i in a) for a, key in M.molecule_parts(st) if _flat(key) == c_key]  # noqa: E731
    at_r, at_p = mine(first), mine(last)
    still = [a for a in at_r if a in at_p and not a & moved]
    took = not still
    if took:        # the added one: a molecule of its kind in a changing bond (the first, at each end)
        a_r = next((a for a in at_r if a & moved), at_r[0] if at_r else None)
        a_p = next((a for a in at_p if a & moved), at_p[0] if at_p else None)
    else:
        a_r = a_p = still[-1]
    got = parent.ts_step(ts, ang(first), ang(last), engine, temperature=q["temperature"], whole=not took)
    fields, _steer_needs = _placed(q, got, molecule, engine,
                                   (set(a_r) if a_r else None, set(a_p) if a_p else None))
    return {"kind": "takes part" if took else "beside",
            "entry": {"id": f"{sid}|{label}", "parent_step": sid, "layer": layer, "status": "verified",
                      "label": (f"{sid}: the molecule takes part ({label})" if took
                                else f"{sid}: with the molecule beside it ({label})"),
                      "takes_part": took, "returned": True if took else None,
                      "bonds_broken": sorted(list(x) for x in b0 - b1), "bonds_formed": sorted(list(x) for x in b1 - b0),
                      **fields}}


def _verify_task(args):
    """One step verified with the molecule beside it, in a worker process: from the quick estimate's lowest
    complexes and from those of a second, independent quick estimate (two TS poses), each through
    mepd.cli_variant._reoptimize (ends minimized, the TS optimized with the molecule present, its IRC); each
    TS read with `_ts_row`. Returns {"step", "rows": the lowest row of each kind with the difference between
    the two poses as its spread, "elsewhere": [...], "notes", "wall_s"}."""
    profile, query_fp, sid, frames, molecule, first, out_dir, temperature = args
    import numpy as np
    from qcdata import Structure

    from mepd import bystander as B
    from mepd.cli_variant import _reoptimize
    from mepd.discovery import molecules as M
    from mepd.network_variants import adjacency
    from mepd.web.variants import _structure

    t0 = time.perf_counter()
    out = {"step": sid, "rows": [], "elsewhere": [], "notes": []}
    try:
        engine = _engine(profile)
        if query_fp not in _QUERY:
            _QUERY[query_fp] = json.loads(Path(query_fp).read_text())
        q = _QUERY[query_fp]
        sym = list(frames["symbols"]) + list(molecule["symbols"])
        n_x = len(frames["symbols"])
        charge, mult = int(frames["charge"]) + int(molecule["charge"]), int(frames["multiplicity"])
        poses = [first]
        try:
            again = B.bystander(frames, molecule, engine, temperature=temperature, seed=1, **QUICK)
            if again.get("status") == "ok" and all(k in again.get("complexes", {}) for k in ("r", "ts", "p")):
                same = np.abs(np.asarray(again["complexes"]["ts"]) - np.asarray(first["ts"])).max() < 0.1
                if not same:
                    poses.append(again["complexes"])
        except Exception as exc:
            out["notes"].append(f"second pose: {type(exc).__name__}: {exc}")
        pairs = lambda m: [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(m)))]   # noqa: E731
        keys = lambda x: sorted(key for _a, key in M.molecule_parts(_structure(sym, x, charge, mult)))   # noqa: E731
        found: dict = {}
        for k, cx in enumerate(poses):
            d = {"symbols": sym, "charge": charge, "multiplicity": mult, **{f: np.asarray(cx[f]).tolist() for f in ("r", "ts", "p")},
                 "bonds_r": pairs(adjacency(sym, np.asarray(cx["r"]))), "bonds_p": pairs(adjacency(sym, np.asarray(cx["p"])))}
            folder = Path(out_dir) / f"pose_{k}"
            try:
                summary = _reoptimize(d, _INPUTS, folder)
                ts_fp = folder / "reopt" / "ts.xyz"
                irc_fp = next((f for f in (folder / "reopt" / "irc.xyz", folder / "reopt" / "ts_irc.xyz") if f.exists()),
                              folder / "reopt" / "irc.xyz")
                if not (ts_fp.exists() and irc_fp.exists()):
                    out["notes"].append(f"pose {k}: {summary.get('headline') or summary.get('status')}")
                    continue
                irc = Structure.open_multi(str(irc_fp))
                q_ = {"charge": charge, "multiplicity": mult}
                row = _ts_row(q, sid, f"verified pose {k}", "verify", Structure.open(str(ts_fp)).model_copy(update=q_),
                              irc[0].model_copy(update=q_), irc[-1].model_copy(update=q_), set(range(n_x, len(sym))),
                              molecule, engine, keys(cx["r"]), keys(cx["p"]))
            except Exception as exc:
                out["notes"].append(f"pose {k}: {type(exc).__name__}: {exc}")
                continue
            if row["kind"] == "elsewhere":
                out["elsewhere"].append({"pose": k, "ends": row["ends"]})
                continue
            row["entry"].update(ts_file=str(ts_fp), irc_file=str(irc_fp), source="mepd variant --mode reoptimize's "
                                "function on the bystander estimate's complexes (TS optimized, IRC joins the step's ends)")
            found.setdefault(row["kind"], []).append(row["entry"])
        for kind, entries in found.items():
            entries.sort(key=lambda e: e["barrier"]["free"])
            low = entries[0]
            # two poses: the difference between them is the row's spread
            low["barrier"] = {**low["barrier"], "spread": (entries[1]["barrier"]["free"] - low["barrier"]["free"])
                              if len(entries) > 1 else None}
            low["spread"] = low["barrier"]["spread"]
            low["poses_verified"] = len(entries)
            out["rows"].append(low)
    except Exception as exc:
        out["notes"].append(f"{type(exc).__name__}: {exc}")
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


RELAY_EXPLORE = 8            # `mepd channels --explore-mechanisms`: mechanisms through the added molecule, by rank
RELAY_BUDGET_S = 300.0       # its --search-budget: no new search starts after this
RELAY_STOP_S = 1200.0        # a scan still running then is stopped; what it has verified by then counts
_SCANS: set = set()          # the running scans' process groups (stopped with the command)
# What the bystander estimate's barrier shift was measured to be worth (docs/bystander.md: g-xTB, gas; proline on
# aldol steps, water on HCOOH, AlCl3 on isoprene + methyl acrylate), for a page to show beside estimate rows.
ESTIMATE_UNCERTAINTY = {
    "barrier_shift_kcal": {"mean_abs_difference_to_verified": 2.0, "largest": 4.2, "same_sign": "9 of 10 steps"},
    "what": "the estimate (default setting) against the same step verified with the molecule present, ten steps",
    "quick_setting": "the quick setting against the default one over a 178-step network: rms 3.6 kcal/mol",
    "source": "docs/bystander.md"}
LANES = 3                    # scans side by side, each on its own cores (a scan keeps about a third of 32 busy)


def _reactions(q: dict) -> list:
    """The parent's net reactions: its steps with the molecules that are the same on both sides dropped (its own
    shuttles too), steps of one reaction (either way round) together. Each: {"key", "reactants", "products" (as
    its lowest step with frames is written), "steps" (ids), "lowest" (that step), "twin" (every step of it is a
    mirror-image twin), "atoms", "label"}. Only reactions with a step that has frames."""
    from collections import Counter

    name = {x["id"]: x.get("name") or x["id"] for x in q["species"]}
    size = {x["id"]: len(x.get("atoms") or []) for x in q["species"]}
    atoms_of = {x["id"]: list(x.get("atoms") or []) for x in q["species"]}
    found: dict = {}
    for st in q["steps"]:
        r, p = Counter(st["reactants"]), Counter(st["products"])
        r, p = sorted((r - p).elements()), sorted((p - r).elements())
        if not r or not p:
            continue
        key = min((tuple(r), tuple(p)), (tuple(p), tuple(r)))
        found.setdefault(key, []).append((st, r, p))
    out = []
    for key, members in found.items():
        framed = [m for m in members if m[0].get("geometry")]
        if not framed:
            continue
        low, r, p = min(framed, key=lambda m: m[0]["g_ts"])
        if sorted(a for x in r for a in atoms_of.get(x, ())) != sorted(a for x in p for a in atoms_of.get(x, ())):
            continue      # (a species whose key lost atoms: the two sides do not hold the same atoms; no scan)
        out.append({"key": key, "reactants": r, "products": p, "steps": [m[0]["id"] for m in members],
                    "lowest": low["id"], "twin": all(m[0].get("mirror_of") for m in members),
                    "atoms": sum(size.get(x, 0) for x in r),
                    "label": " + ".join(name[x] for x in r) + " -> " + " + ".join(name[x] for x in p)})
    return out


SCAN_CHANGE = 0.01     # (ii): a change of amount worth a scan, as a fraction of the limiting starting material


def _scan_worth(net: dict, molecule: dict, est: dict, initial: dict, target, reactions: list, flagged: set,
                handed: bool, conditions: Optional[dict] = None, among: Optional[set] = None) -> tuple:  # noqa: C901
    """Which of `reactions` are worth a scan, in order, on the kinetics as they stand (`est`: the molecule's rows
    so far). A reaction is worth one if, with no barrier above its ends, it would (i) lower the target's route
    peak, or (ii) bring some other species to a route peak below the target's present one and below what that
    species has now (something that could outrun the target or eat its material). Order: (i) by the resulting
    target peak, then (ii) by the resulting peak; ties (to 0.1 kcal/mol): reactions where the estimate or the
    verification saw the molecule react or take part (`flagged`), then fewer atoms; for a chiral molecule mirror
    twins after everything else. Returns (the target's route peak now, [(reaction, "target" | "other", the
    resulting peak, the species it is of)], [the reactions not worth a scan]). Route peaks: mepd.addmol.reach_peaks
    (at most two lots of molecules in hand).

    With `conditions` ({"time_s", "held"}; the run's own): (i) orders the scans while any reaction is left under it
    (`among`: the keys still to be scanned; the others only count as taken). When none is left, each reaction that
    passes (ii) by the route peak is put through the kinetics with its barrier above its ends removed: it is worth
    a scan only if the amount of some species at the end of the run then differs from what it is now by more than
    SCAN_CHANGE of the limiting starting material (the target's own amount counts, up or down). Those come as
    (reaction, "other", the change in M, the species), the largest change first."""
    import math

    from mepd import addmol

    variant = addmol.added(net, molecule, est)
    g = {x["id"]: float(x["g"]) for x in variant["species"]}
    goal = set(addmol._targets(target))
    now = addmol.peak_routes(variant, initial, target, limit=1)
    peak_now = now[0]["peak"] if now else math.inf
    ceiling = peak_now if math.isfinite(peak_now) else 1e6
    base = addmol.reach_peaks(variant, initial, ceiling)
    start = {x for x, v in initial.items() if float(v) > 0}
    worth, rest = [], []
    for rx in reactions:
        if not any(all(x in start or x in base for x in side) for side in (rx["reactants"], rx["products"])):
            rest.append(rx)                 # neither of its ends can be had below the target's route peak
            continue
        free = {"id": "(no barrier)", "reactants": rx["reactants"], "products": rx["products"],
                "g_ts": max(sum(g[x] for x in rx["reactants"]), sum(g[x] for x in rx["products"]))}
        got = addmol.reach_peaks({**variant, "steps": variant["steps"] + [free]}, initial, ceiling)
        t_new = min((got[x] for x in goal if x in got), default=math.inf)
        if t_new < peak_now - 0.05:
            worth.append((rx, "target", t_new, None))
            continue
        other = [(got[x], x) for x in got if x not in goal and got[x] < base.get(x, ceiling) - 0.05]
        if other:
            v, x = min(other)
            worth.append((rx, "other", v, x))
        else:
            rest.append(rx)
    order = lambda w: (bool(handed and w[0]["twin"]), w[1] != "target", round(w[2], 1),   # noqa: E731
                       w[0]["key"] not in flagged, w[0]["atoms"])
    if conditions is None:
        worth.sort(key=order)
        return peak_now, worth, rest
    first = [w for w in worth if w[1] == "target"]
    if any(among is None or w[0]["key"] in among for w in first):
        first.sort(key=order)
        return peak_now, first, rest + [w[0] for w in worth if w[1] != "target"]
    # nothing left under (i): (ii) in the kinetics
    limiting = min(float(v) for x, v in initial.items() if x != addmol.ADDED_ID and float(v) > 0)
    held = conditions.get("held") or ()
    k0 = addmol.run(variant, initial, conditions["time_s"], held)
    kept = []
    for rx, _kind, _v, _sp in (w for w in worth if w[1] != "target"):
        free = {"id": "(no barrier)", "reactants": rx["reactants"], "products": rx["products"],
                "g_ts": max(sum(g[x] for x in rx["reactants"]), sum(g[x] for x in rx["products"]))}
        k1 = addmol.run({**variant, "steps": variant["steps"] + [free]}, initial, conditions["time_s"], held)
        change = {x: float(k1["res"].c[-1, k1["pos"][x]] - k0["res"].c[-1, i]) for x, i in k0["pos"].items()}
        x = max(change, key=lambda x: abs(change[x]))
        if abs(change[x]) > SCAN_CHANGE * limiting:
            kept.append((rx, "other", change[x], x))
        else:
            rest.append(rx)
    kept.sort(key=lambda w: (bool(handed and w[0]["twin"]), -round(abs(w[2]) / limiting, 3),
                             w[0]["key"] not in flagged, w[0]["atoms"]))
    return peak_now, kept, rest


def _scan_ends(q: dict, rx: dict, molecule: dict, engine) -> dict:
    """The two ends of a reaction's scan: of its lowest parent step's own ends (the TS's IRC ends in the query),
    the molecules that change, the others removed; the added molecule placed beside each by the placer (quick
    setting). {"start", "end"}: qcdata Structures, the added molecule's atoms last. Nothing else is built."""
    from collections import Counter

    import numpy as np

    from mepd import bystander as B
    from mepd.discovery import molecules as M
    from mepd.web.variants import _structure

    st = next(x for x in q["steps"] if x["id"] == rx["lowest"])
    geo = st["geometry"]
    out = {}
    for name, frame, wanted in (("start", "reactant", rx["reactants"]), ("end", "product", rx["products"])):
        whole = _structure(geo["atoms"], geo[frame], geo["charge"], geo["multiplicity"])
        need = Counter(wanted)
        atoms = []
        for group, key in M.molecule_parts(whole):
            hit = _species_ids(q, [{"smiles": key}], None)[0]
            if need[hit] > 0:
                need[hit] -= 1
                atoms += sorted(group)
        if sum(need.values()):
            raise ValueError(f"the {frame} end of {st['id']} does not hold {' + '.join(wanted)}")
        sym = [geo["atoms"][i] for i in atoms]
        xyz = np.asarray(geo[frame], dtype=float)[atoms]
        res = B.bystander({"symbols": sym, "charge": int(geo["charge"]), "multiplicity": int(geo["multiplicity"]),
                           "r": xyz.tolist(), "ts": None, "p": None}, molecule, engine,
                          temperature=q["temperature"], **QUICK)
        if "r" not in (res.get("complexes") or {}):
            raise ValueError(f"the molecule could not be placed beside the {frame} end: {res.get('why') or res.get('status')}")
        out[name] = _structure(sym + list(molecule["symbols"]), res["complexes"]["r"],
                               int(geo["charge"]) + int(molecule["charge"]), int(geo["multiplicity"]))
    return out


def _relay_scan(q: dict, sid: str, molecule: dict, ends: dict, engine, profile: Path, folder: Path, rec: dict,
                on_row, started: float, cores: Optional[list] = None, compute=None) -> None:
    """One reaction scanned for transition states with the added molecule: `mepd channels` between its two ends
    with the molecule beside each (`ends`: {"start", "end"}: qcdata Structures, the molecule's atoms last; they
    need not number the reaction's atoms alike: channels maps them), mechanisms through the molecule explored
    (--explore-mechanisms, --explore-molecule). Its searches.json is read as it grows: every verified TS that
    joins the two ends is put on the parent's scale (`_ts_row`) and handed to `on_row(entry, takes_part)`; `rec`
    (the scan's line in the result file) follows. `cores`: the CPUs the scan is held to (taskset); `compute`: a
    lock held while this process's engine works (scans side by side share it)."""
    import contextlib
    import os
    import shutil
    import signal
    import subprocess
    import sys

    from qcdata import Structure

    from mepd.discovery import molecules as M
    from mepd.web.variants import _structure

    compute = compute or contextlib.nullcontext()
    charge, mult = int(ends["start"].charge), int(ends["start"].multiplicity)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("start", "end"):
        ends[name].save(str(folder / f"{name}.xyz"))
    c_key = _c_key(molecule)
    smiles = M.molecule_parts(_structure(molecule["symbols"], molecule["conformers"][0], molecule["charge"],
                                         molecule["multiplicity"]))[0][1]
    out = folder / "out"
    from mepd.cpus import available_cpus

    cpus = len(cores) if cores else available_cpus()   # (the mask, else the affinity mask and the container's limit)
    cmd = [sys.executable, "-m", "mepd.cli", "channels", "--start", str(folder / "start.xyz"), "--end",
           str(folder / "end.xyz"), "-i", str(profile), "--workers", str(cpus), "--parallel-workers", "1",
           "--search-budget", str(int(RELAY_BUDGET_S)), "--explore-mechanisms", str(RELAY_EXPLORE),
           "--explore-molecule", smiles, "-o", str(out)]
    if cores and shutil.which("taskset"):
        # (--workers does not bound a channels run: only its CPU mask does)
        cmd = ["taskset", "-c", ",".join(str(c) for c in cores)] + cmd
    rec.update(status="mapping", command=" ".join(cmd), run=str(folder), cores=cpus, t_started_s=round(
        time.perf_counter() - started, 1))
    t0 = time.perf_counter()
    start_keys = sorted(key for _a, key in M.molecule_parts(ends["start"]))
    end_keys = sorted(key for _a, key in M.molecule_parts(ends["end"]))
    seen: set = set()
    rows: list = []
    with open(folder / "channels.log", "w") as log:
        # (its own session: stopping it stops its workers too; it inherits this process's CPU mask)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        _SCANS.add(proc.pid)
        try:
            while True:
                ended = proc.poll() is not None
                data = None
                try:
                    data = json.loads((out / "searches.json").read_text())
                except (OSError, ValueError):
                    pass
                if data is not None:
                    searches = data.get("searches") or []
                    c_atoms = [set(m["atoms"]) for m in data.get("start_molecules") or [] if _flat(m.get("smiles")) == c_key]
                    counts = {k: sum(1 for x in searches if x.get("status") == k) for k in ("queued", "running", "searched")}
                    rec.update(status="searching" if not ended else rec["status"], searches={**counts, "total": len(searches)})
                    todo = sorted((ts for x in searches for ts in x.get("transition_states") or []
                                   if ts.get("status") == "verified" and ts.get("label") not in seen),
                                  key=lambda ts: ts.get("ts_energy_hartree") or 0.0)
                    for ts in todo:
                        seen.add(ts["label"])
                        touched = [m for m in ts.get("molecules") or [] if any(set(m["atoms"]) & c for c in c_atoms)]
                        took = bool(touched)
                        if not ts.get("joins_start_and_end"):
                            rec["elsewhere"] = rec.get("elsewhere", 0) + 1
                            if took and not all(m.get("returned") for m in touched):
                                # the molecule changed and did not come back: a step to a new species (not followed)
                                rec.setdefault("consumed", []).append({"ts": ts["label"], "irc_ends": ts.get("irc_ends")})
                            continue
                        e = ts.get("ts_energy_hartree")
                        if any(e is not None and abs(r - e) < 1e-5 for r in rows):
                            continue                    # the same TS again (another conformer pair found it)
                        try:
                            q_ = {"charge": charge, "multiplicity": mult}
                            irc = Structure.open_multi(str(out / ts["irc"]))
                            with compute:
                                row = _ts_row(q, sid, ts["label"], "relay",
                                              Structure.open(str(out / ts["ts"])).model_copy(update=q_),
                                              irc[0].model_copy(update=q_), irc[-1].model_copy(update=q_),
                                              set().union(*c_atoms) if c_atoms else set(), molecule, engine,
                                              start_keys, end_keys)
                        except Exception as exc:
                            rec.setdefault("failed_rows", []).append({"ts": ts["label"], "why": f"{type(exc).__name__}: {exc}"})
                            continue
                        rows.append(e if e is not None else 0.0)
                        if row["kind"] == "elsewhere":
                            rec["elsewhere"] = rec.get("elsewhere", 0) + 1
                            continue
                        took = row["kind"] == "takes part"
                        kind = "with_molecule" if took else "beside"
                        rec[kind] = rec.get(kind, 0) + 1
                        now = round(time.perf_counter() - started, 1)
                        if took and "first_with_molecule_s" not in rec:
                            rec["first_with_molecule_s"] = now
                        # every verified row of the scan: when it came and how high (to judge when a scan could stop)
                        rec.setdefault("arrivals", []).append({"t_s": now, "takes_part": took, "ts": ts["label"],
                                                               "free": row["entry"]["barrier"]["free"]})
                        on_row({**row["entry"], "source": "mepd channels (TS optimized, IRC joins the step's two ends)",
                                "ts_file": str(out / ts["ts"]), "irc_file": str(out / ts["irc"]), "t_s": now}, took)
                rec["wall_s"] = round(time.perf_counter() - t0, 1)
                on_row(None, None)
                if ended:
                    break
                if time.perf_counter() - t0 > RELAY_STOP_S:
                    os.killpg(proc.pid, signal.SIGTERM)       # (our own child's session)
                    rec["stopped"] = f"after {RELAY_STOP_S:g} s"
                    proc.wait(timeout=60)
                    continue
                time.sleep(3.0)
        finally:
            _SCANS.discard(proc.pid)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    if proc.returncode not in (0, None) and not rec.get("stopped") and not (out / "searches.json").exists():
        rec.update(status="failed", why=f"mepd channels ended with code {proc.returncode} (see {folder / 'channels.log'})")
    else:
        import re

        total = (rec.get("searches") or {}).get("total", 0)
        try:
            log_text = (folder / "channels.log").read_text(errors="replace")
        except OSError:
            log_text = ""
        lost = sum(int(n) for n in re.findall(r"Skipping (\d+) pair tree\(s\) whose root NEB failed", log_text))
        if lost:
            rec.setdefault("searches", {})["failed_at_once"] = lost
        found_any = rec.get("with_molecule") or rec.get("beside") or rec.get("elsewhere")
        if not found_any and (total == 0 or lost >= total):
            # nothing was searched: not the same as searching and finding nothing
            rec.update(status="failed", why_failed=(f"every path search failed at its first step ({lost} of {total})"
                                                    if total else "mepd channels started no path search"))
        else:
            rec["status"] = "done"
            if not rec.get("with_molecule"):
                rec["answer"] = "none found: no verified TS joins the reaction's two ends with the molecule taking part"


@addmol_app.command("test")
def test(
    query_file: Path = typer.Argument(..., exists=True, help="The query file (`mepd addmol query`)."),
    candidate: List[str] = typer.Option([], "--candidate", "-c", help="A candidate: SMILES, or an xyz file (repeat)."),
    candidates: Optional[Path] = typer.Option(None, "--candidates", help="Candidates as JSON: a list of {name, smiles "
                                              "| atoms + coordinates, amount, charge, multiplicity, pose, estimates} "
                                              "(docs/add-molecule.md)."),
    amount: Optional[float] = typer.Option(None, "--amount", help="mol/L of every candidate that gives none."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="Profile for the bystander layer. Default: "
                                          "the level's profile kept in the query."),
    steps: Optional[int] = typer.Option(None, "--steps", help="Bystander estimates for the first N steps of the "
                                        "query's order of work only (0: none, the kinetics of the estimates "
                                        "given with the candidates). Default: every step with geometries."),
    verify: Optional[int] = typer.Option(None, "--verify", help="Verify the first N steps of the order of work "
                                         "with the molecule beside them (TS optimized with it, IRC; 15-40 s per "
                                         "step and process). Default: every step with an estimate. 0: none."),
    spread: bool = typer.Option(False, "--estimate-spread", help="Estimates with the placer's default setting "
                                "(3 repeats, polish; about 7 times the quick setting's time): each estimate row "
                                "then carries the spread over its repeats."),
    scan: int = typer.Option(0, "--scan", help="The relay layer: scan up to N steps for transition states with "
                             "the molecule taking part (`mepd channels` with the molecule on both sides; minutes "
                             "per step on all the cores this command may use: pin it with taskset): the steps "
                             "that limit the target first, then those whose estimate or verification showed the "
                             "molecule reacting. 0: none."),
    scan_inputs: Optional[Path] = typer.Option(None, "--scan-inputs", help="Profile of the relay layer's path "
                                               "searches (same engine as the level; its path method). Default: "
                                               "the bystander layer's profile."),
    steer: Optional[Path] = typer.Option(None, "--steer", help="The parent run's steering profile: each step given "
                                         "with a TS is put on that level by single points (the TS, the candidate, "
                                         "the parent's starting box), the steps that carry the target's material "
                                         "first. Without it such steps stay at the parent's low level."),
    workers: int = typer.Option(1, "--workers", help="Processes for the bystander layer."),
    jobs: int = typer.Option(1, "--jobs", help="Candidates worked on at once, each with --workers processes (a "
                             "screen; with --scan each scan takes every core, so keep 1)."),
    output: Path = typer.Option(..., "--output", "-o", help="Folder: one result file per candidate, rewritten as "
                                "numbers come."),
) -> None:
    """Test candidates against a query: the layers that exist, cheapest first; one result file per candidate,
    rewritten as numbers come (the target's change, per-step barrier changes, new species, the molecule's
    fate, where each number is from)."""
    import os

    from mepd import addmol
    from mepd.orbitals import ORBITALS_ENV

    # (a run with --output keeps every SCF's orbitals for the web's viewer: not the thousands of this layer)
    os.environ.pop(ORBITALS_ENV, None)
    q = json.loads(query_file.read_text())
    if q.get("schema") != addmol.QUERY or q.get("version") != addmol.VERSION:
        raise typer.BadParameter(f"{query_file} is not a {addmol.QUERY} file of version {addmol.VERSION}")
    cands = _candidates(candidate, candidates, amount)
    output.mkdir(parents=True, exist_ok=True)
    profile = inputs
    if profile is None and (q.get("level") or {}).get("profile"):
        profile = output / "level_profile.toml"
        profile.write_text(q["level"]["profile"])
    with_ts = any(n.get("ts") for c in cands for n in ((c.get("estimates") or {}).get("new_steps") or []))
    needs = (steps is None or steps > 0) and any(st.get("geometry") for st in q["steps"])
    if (needs or with_ts or scan > 0) and profile is None:
        raise typer.BadParameter("the bystander layer, the relay layer and a step given with a TS need a profile: "
                                 "give --inputs (the query keeps none)")
    opt = {"query": str(query_file), "profile": str(profile) if profile else None, "steps": steps, "scan": scan,
           "verify": verify, "setting": {} if spread else dict(QUICK),
           "scan_profile": str(scan_inputs or profile) if (scan_inputs or profile) else None,
           "steer": str(steer) if steer else None, "workers": workers, "output": str(output)}
    if jobs > 1 and len(cands) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(jobs) as pool:
            summary = pool.map(_candidate_job, [(c, opt) for c in cands])
    else:
        summary = [_candidate(q, c, opt) for c in cands]
    typer.echo(json.dumps(summary, indent=1))


def _candidate_job(args) -> dict:
    cand, opt = args
    return _candidate(json.loads(Path(opt["query"]).read_text()), cand, opt)


def _candidate(q: dict, cand: dict, opt: dict) -> dict:
    """One candidate through the layers, its result file rewritten as numbers come: the estimates given with
    it; steps given with a TS (and their steering level); the bystander layer for the species that hold
    material and the steps to be scanned; the relay layer on those steps, one after another; the bystander
    layer for the rest of the parent."""
    import os

    from mepd import addmol
    from mepd.orbitals import ORBITALS_ENV

    os.environ.pop(ORBITALS_ENV, None)
    t0 = time.perf_counter()
    output, profile = Path(opt["output"]), opt["profile"]
    by_id = {st["id"]: st for st in q["steps"]}
    order = [o["step"] for o in q["limits"]["order"] if (by_id.get(o["step"]) or {}).get("geometry")]
    rest = sorted((st for st in q["steps"] if st.get("geometry") and st["id"] not in order),
                  key=lambda st: st["apparent_barrier"]["forward"] if st["apparent_barrier"]["forward"] is not None
                  else float("inf"))
    work = order + [st["id"] for st in rest]
    if opt["steps"] is not None:
        work = work[:max(0, opt["steps"])]
    level = (q.get("level") or {}).get("key")
    fp = output / _file_name(cand["name"])
    given = cand.pop("estimates", None) or {}
    est = {k: [{"layer": "given", **e} for e in given.get(k) or []]
           for k in ("species", "steps", "new_species", "new_steps")}
    failed: list = []
    layers = [{"name": "given", "status": "done", "what": "estimates that came with the candidate",
               "entries": sum(len(v) for v in est.values())}] if any(est.values()) else []
    n_by = len(work)
    lay = {"name": "bystander", "status": "running" if n_by else "skipped", "done": 0, "total": n_by, "level": level,
           "wall_s": 0.0, "what": "the molecule beside each step's ends and TS, the step's atoms held (mepd.bystander)"}
    layers.append(lay)
    # the stages still to come are in the file from the start ("queued")
    vlay = {"name": "verify", "status": "queued" if opt["verify"] != 0 else "skipped", "done": 0, "total": 0, "wall_s": 0.0,
            "level": level, "uncertainty": "each row's spread: the difference between its two verified poses",
            "what": "each step with the molecule beside it: the TS optimized with the molecule present from the "
                    "estimate's complex (two poses), its IRC, the ends as found; replaces the step's estimate"}
    rlay = {"name": "relay", "status": "queued" if opt["scan"] else "skipped", "done": 0, "total": 0, "wall_s": 0.0,
            "level": level, "what": f"mepd channels between a reaction's two ends with the molecule beside them "
                                    f"(--explore-mechanisms {RELAY_EXPLORE} --explore-molecule, --search-budget "
                                    f"{RELAY_BUDGET_S:g}): every verified TS on the parent's scale"}
    if n_by:
        lay["uncertainty"] = ESTIMATE_UNCERTAINTY
        layers += [vlay, rlay]
    extra: dict = {}
    timing: dict = {}

    def write(status):
        t = time.perf_counter()
        doc = addmol.result(q, cand, est, status=status, layers=layers, failed=failed)
        timing["kinetics_s"] = round(time.perf_counter() - t, 2)
        doc["timings"] = {**timing, "total_s": round(time.perf_counter() - t0, 1)}
        doc.update(extra)
        addmol.write_json(fp, doc)
        return doc

    engine = molecule = None
    handed = True
    if n_by or any(n.get("ts") for n in est["new_steps"]):
        from mepd.cli_common import _open_run_inputs
        from mepd.discovery.mirror import is_chiral
        from mepd.web.variants import _structure as _st

        engine = _open_run_inputs(Path(profile)).engine
        t = time.perf_counter()
        try:
            molecule = _prepare(cand, engine)
        except Exception as exc:
            failed.append({"layer": "candidate", "why": f"the candidate could not be built: {type(exc).__name__}: {exc}"})
            lay["status"] = "failed" if n_by else lay["status"]
            return _line(write("failed"), fp)
        timing["prepare_s"] = round(time.perf_counter() - t, 1)
        # the molecule as it was tested (its lowest conformer at the level), for a 3D view
        cand["geometry"] = {"atoms": molecule["symbols"], "coordinates": molecule["conformers"][0],
                            "conformers": len(molecule["conformers"])}
        handed = is_chiral(_st(molecule["symbols"], molecule["conformers"][0], molecule["charge"],
                               molecule["multiplicity"])) is not False
        cand["chiral"] = handed
        if any(n.get("ts") for n in est["new_steps"]):
            t = time.perf_counter()
            kept: dict = {}
            est["new_steps"] = _ts_steps(q, est["new_steps"], molecule, engine, failed, kept)
            # for a molecule that is its own mirror image, the same step between the mirror-image species
            est["new_steps"] += [m for m in (_mirror_entry(q, n, handed) for n in est["new_steps"] if n.get("id") in kept)
                                 if m is not None]
            timing["ts_steps_s"] = round(time.perf_counter() - t, 1)
            doc = write("running")
            todo = [n["id"] for n in est["new_steps"] if n.get("id") in kept]
            if opt["steer"] and todo:
                # the steps on the routes that carry the target's material first
                first = [x for r in ((doc.get("readout") or {}).get("routes") or {}).get("with") or [] for x in r["steps"]]
                todo.sort(key=lambda sid: first.index(sid) if sid in first else len(first))
                slay = {"name": "steer", "status": "running", "done": 0, "total": len(todo), "wall_s": 0.0,
                        "level": Path(opt["steer"]).name, "what": "the parent's steering level for the steps given with "
                                                                  "a TS: single points of the TS, the candidate and "
                                                                  "the parent's starting box"}
                layers.insert(1, slay)
                _steer(q, est, kept, todo, engine, _open_run_inputs(Path(opt["steer"])).engine, slay, failed, write)
                timing["steer_s"] = slay["wall_s"]
    doc = write("running" if n_by else "done")
    if not n_by:
        return _line(doc, fp)

    # A step and its mirror image are one step for a molecule that is its own mirror image: computed once, the
    # twin's rows a copy. With a chiral molecule they are two (diastereomeric) steps, each computed, the twins
    # after every step that is not one.
    import threading

    twin_of = {} if handed else {st["mirror_of"]: st["id"] for st in q["steps"]
                                 if st.get("mirror_of") and st["id"] in work and st["mirror_of"] in work}
    todo = [sid for sid in work if sid not in twin_of.values()]        # in the order of what limits the target
    todo.sort(key=lambda sid: bool(by_id[sid].get("mirror_of")))       # (stable: twins last)
    lay["total"] = len(todo)
    pose = cand.get("pose") or {}
    complexes: dict = {}
    state: dict = {sid: "none" for sid in todo}        # how each step ends: est / verified / takes part / ...
    reacted: list = []
    best: dict = {}                                    # (step, takes part) -> its lowest verified row
    scans: list = []
    kinetic_lock = threading.Lock()
    pool_done = [False]
    lock = threading.RLock()                           # the estimates, the result file
    compute = threading.Lock()                         # this process's engine (scans side by side share it)
    since = lambda: round(time.perf_counter() - t0, 1)                 # noqa: E731
    plain_write = write

    def write(status):                                 # noqa: F811  (one writer at a time; the route peak's history)
        with lock:
            doc = plain_write(status)
            peaks = ((doc.get("readout") or {}).get("peak_routes") or {})
            now = (peaks.get("with") or [{}])[0].get("peak")
            hist = extra.setdefault("route_peak", {"parent": (peaks.get("parent") or [{}])[0].get("peak"), "with": []})
            if now is not None and (not hist["with"] or abs(hist["with"][-1]["peak"] - now) > 1e-6):
                # with the read-out as it stood then (the page shows the answer over time without the kinetics)
                r = doc["readout"]
                tgt, fate = r.get("target") or {}, r.get("added") or {}
                hist["with"].append({
                    "t_s": since(), "peak": now, "rests_on": peaks["with"][0].get("rests_on"),
                    "estimated_steps": peaks["with"][0].get("estimated_steps"),
                    "readout": {"target": {"final": (tgt.get("with") or {}).get("final"),
                                           "time_to_fraction_s": (tgt.get("with") or {}).get("time_to_fraction_s"),
                                           "factor": tgt.get("factor"),
                                           "low": (tgt.get("spread") or {}).get("with_low"),
                                           "high": (tgt.get("spread") or {}).get("with_high")},
                                "fate": fate.get("fate"), "route_role": fate.get("route_role"),
                                "in_words": fate.get("in_words"), "turnovers": fate.get("turnovers")}})
            return doc

    def cut(signum, _frame):
        """Stopped from outside (a time limit) before it was complete: the file says so, the scans are stopped."""
        import signal as _signal

        extra["cut"] = {"t_s": since(), "why": "stopped by a signal before it was complete (a time limit)"}
        for item in layers:
            if item.get("status") in ("running", "queued"):
                item["status"] = "cut"
        for pid in list(_SCANS):
            try:
                os.killpg(pid, _signal.SIGTERM)
            except OSError:
                pass
        try:
            if lock.acquire(timeout=20):
                counts()
                plain_write("cut")
        finally:
            os._exit(124)

    if threading.current_thread() is threading.main_thread():
        import signal as _signal

        _signal.signal(_signal.SIGTERM, cut)

    def frames_of(sid):
        g = by_id[sid]["geometry"]
        return {"symbols": g["atoms"], "charge": g["charge"], "multiplicity": g["multiplicity"],
                "r": g["reactant"], "ts": g["ts"], "p": g["product"]}

    def counts():
        from collections import Counter

        c = Counter(state.values())
        extra["summary"] = {"steps": len(todo), "est": c["est"], "verified_beside": c["verified"],
                            "takes_part": c["takes part"], "leads_elsewhere": c["elsewhere"],
                            "failed": c["failed"] + c["reacted"], "reacted_in_the_estimate": c["reacted"],
                            "not_reached": c["none"],
                            "what": "how each parent step with frames ends: an estimate only; verified with the "
                                    "molecule beside it; a verified TS with the molecule taking part (its own, or "
                                    "its reaction's from a scan); its verification led to other species; no "
                                    "usable number"}

    def take(entry, took, rec=None, replaces_estimate=True, shared=()):
        """A verified row (from the verification or a scan) into the kinetics, if it is its step's lowest of its
        kind. From the verification, with the molecule beside the step, it replaces the step's estimate. A scan's
        row stands for every parent step of its reaction (`shared`)."""
        with lock:
            sid = entry["parent_step"]
            key = (sid, took, entry["layer"])
            if key in best and best[key]["barrier"]["free"] <= entry["barrier"]["free"]:
                return False
            drop = {best[key]["id"], best[key]["id"] + "~mirror"} if key in best else set()
            best[key] = entry
            for k in ("ts_file", "irc_file"):          # beside the result file: paths relative to its folder
                if entry.get(k) and os.path.isabs(str(entry[k])):
                    entry[k] = os.path.relpath(entry[k], output)
            if shared:
                entry["parent_steps"] = list(shared)
            twin = _mirror_entry(q, entry, handed)
            est["new_steps"][:] = [n for n in est["new_steps"] if n.get("id") not in drop] + [entry] + ([twin] if twin else [])
            if not took and replaces_estimate:
                est["steps"][:] = [e for e in est["steps"] if e.get("step") not in (sid, twin_of.get(sid))]
            for x in (shared or [sid]):
                if x in state and (took or (replaces_estimate and state[x] != "takes part")):
                    state[x] = "takes part" if took else "verified"
            timing.setdefault("first_verified_s", since())
            if took:
                timing.setdefault("first_relay_s", since())
            if rec is not None:
                kind = "with_molecule" if took else "beside"
                rec.setdefault("lowest", {})[kind] = entry["barrier"]["free"]
                rec.setdefault("lowest_arrived_s", {})[kind] = entry.get("t_s")
            return True

    # The order of all the work: reaction by reaction, by what the target would gain. The first derivation needs
    # only the parent's network.
    reactions = [rx for rx in _reactions(q) if any(x in state for x in rx["steps"])]
    if not handed:        # (a molecule that is its own mirror image: the twins take their originals' rows)
        reactions = [rx for rx in reactions if not rx["twin"]]
    net = {"temperature": q["temperature"], "energies": q["energies"],
           "species": [{k: x.get(k) for k in ("id", "name", "g", "smiles")} for x in q["species"]],
           "steps": [{k: x.get(k) for k in ("id", "label", "reactants", "products", "g_ts", "status", "level")}
                     for x in q["steps"]]}
    added_as = {"id": addmol.ADDED_ID, "name": cand.get("name")}
    start = {**q["conditions"]["initial"], addmol.ADDED_ID: float(cand["amount"])}
    target = q["target"]["species"]
    name_of = {x["id"]: x.get("name") or x["id"] for x in q["species"]}
    left_ends: list = []
    took_part: set = set()
    taken: set = set()
    of_step = {x: rx["key"] for rx in reactions for x in rx["steps"]}

    cond = {"time_s": q["conditions"]["time_s"], "held": q["conditions"].get("held") or ()}
    kinetic: dict = {"key": None, "kept": [], "rest": []}
    scan_rows = [0]                    # rows from scans so far (the kinetic test is made again when they change)

    def derive(final=False):
        """(the target's route peak now, the reactions worth a scan in order, those not worth one, the flagged).
        While a reaction would lower the target's route peak, those only; then the others, tested in the kinetics
        (made again only when a scan's rows changed the kinetics, or the estimates and verification ended)."""
        with lock:
            snap = json.loads(json.dumps(est, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
            flagged = {of_step[x] for x in set(reacted) | set(left_ends) | took_part if x in of_step}
            rest = [rx for rx in reactions if rx["key"] not in taken]
            key = (scan_rows[0], pool_done[0])
        peak, worth, unworthy = _scan_worth(net, added_as, snap, start, target, rest, flagged, handed)
        first = [w for w in worth if w[1] == "target"]
        if first and not final:
            return peak, first, unworthy + [w[0] for w in worth if w[1] != "target"], flagged
        with kinetic_lock:
            if kinetic["key"] != key or final:
                t = time.perf_counter()
                _p, kept, out = _scan_worth(net, added_as, snap, start, target, rest, flagged, handed, cond, set())
                kinetic.update(key=key, kept=kept, rest=out)
                told = extra["scan"].setdefault("kinetic_test", {"evaluations": 0, "wall_s": 0.0}) if "scan" in extra else {}
                told.update(evaluations=told.get("evaluations", 0) + 1, last_s=round(time.perf_counter() - t, 1),
                            wall_s=round(told.get("wall_s", 0.0) + time.perf_counter() - t, 1), last_t_s=since(),
                            last_tested=len([w for w in worth if w[1] != "target"]),
                            last_worth_a_scan=len(kept))
            kept = [w for w in kinetic["kept"] if w[0]["key"] not in taken]
            gone = {w[0]["key"] for w in kept} | {w[0]["key"] for w in first}
            return peak, first + kept, [rx for rx in rest if rx["key"] not in gone], flagged

    peak0, worth0, _unworthy0, _flags0 = derive() if reactions else (None, [], [], set())
    rank = {x: k for k, (rx, _kind, _v, _sp) in enumerate(worth0) for x in rx["steps"]}
    order_of = {sid: k for k, sid in enumerate(todo)}
    todo.sort(key=lambda sid: (bool(by_id[sid].get("mirror_of")), rank.get(sid, len(rank)), order_of[sid]))

    # The cores: two scan lanes from the start, the estimates and the verification on the rest, a third lane on
    # those when the verification ends (a scan keeps about ten cores of a 32-thread pod busy).
    from mepd.cpus import available_cpus

    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    cpus = cpus[:available_cpus()]
    scanning = bool(opt["scan"] and reactions)
    if scanning and len(cpus) >= 6:
        width = max(2, round(len(cpus) * 10 / 32))
        masks = [cpus[:width], cpus[width:2 * width], cpus[2 * width:3 * width] or cpus[2 * width:]]
        pool_cpus = cpus[2 * width:]
    elif scanning:
        masks, pool_cpus = [cpus[:max(1, len(cpus) // 2)]], cpus[max(1, len(cpus) // 2):] or cpus
    else:
        masks, pool_cpus = [], cpus
    n_pool = max(1, min(opt["workers"], len(pool_cpus)))
    active = [0]                       # scans running
    if scanning and hasattr(os, "sched_setaffinity"):
        try:        # this process (the kinetics, the rows' free energies) on the estimates' cores; the scans on theirs
            os.sched_setaffinity(0, set(pool_cpus))
        except OSError:
            pass
    threads: list = []
    t_rel = time.perf_counter()

    def lane(k):
        from mepd.cli_common import _open_run_inputs

        engine_k = _open_run_inputs(Path(profile)).engine
        while True:
            with lock:
                if len(taken) >= opt["scan"]:
                    return
            try:
                peak, worth, unworthy, flags = derive()
            except Exception as exc:         # (said in the file, not lost with the lane)
                with lock:
                    failed.append({"layer": "relay", "status": "failed", "t_s": since(),
                                   "why": f"lane {k}: the scan order could not be derived: {type(exc).__name__}: {exc}"})
                return
            with lock:
                worth = [w for w in worth if w[0]["key"] not in taken]
                if len(taken) >= opt["scan"]:
                    continue
                if not worth:
                    if active[0] == 0 and pool_done[0]:
                        extra["scan"].setdefault("ended", {
                            "t_s": since(), "why": "no reaction is left that, with no barrier, would lower the target's "
                                                   "route peak, or change the amount of any species at the end of "
                                                   f"the run by more than {SCAN_CHANGE:.0%} of the limiting starting "
                                                   "material"})
                        return
                    wait = True
                else:
                    wait = False
                    rx, kind, value, species = worth[0]
                    taken.add(rx["key"])
                    active[0] += 1
                    why = (f"with no barrier it would lower the target's route peak to {value:.1f} kcal/mol (now {peak:.1f})"
                           if kind == "target" else
                           f"with no barrier the amount of {name_of.get(species, species)} at the end of the run would "
                           f"change by {value:+.2g} M (more than {SCAN_CHANGE:.0%} of the limiting starting material)")
                    rec = {"step": rx["lowest"], "label": rx["label"], "reaction": rx["label"], "reactants": rx["reactants"],
                           "products": rx["products"], "steps": rx["steps"], "status": "queued", "lane": k,
                           "twin": rx["twin"], "flagged": rx["key"] in flags, "route_peak_now": peak, "arrivals": [],
                           "worth": kind, "route_peak_if_no_barrier": value if kind == "target" else peak,
                           "other_species": species, "other_species_change_if_no_barrier": value if kind == "other" else None,
                           "why": why + ("; the estimate or the verification saw the molecule react or take part"
                                         if rx["key"] in flags else "")}
                    scans.append(rec)
                    rlay["total"] = len(scans)
                    extra["scan"]["derivations"].append({
                        "t_s": since(), "lane": k, "route_peak_now": peak, "chosen": rx["label"],
                        "worth_a_scan": len(worth), "not_worth_a_scan": len(unworthy),
                        "top": [{"reaction": r["label"], "worth": kd, "species": sp,
                                 "peak_if_no_barrier" if kd == "target" else "change_if_no_barrier": v,
                                 "flagged": r["key"] in flags, "twin": r["twin"], "atoms": r["atoms"]}
                                for r, kd, v, sp in worth[:8]]})
            if wait:
                time.sleep(5.0)
                continue
            try:
                try:
                    with compute:
                        ends = _scan_ends(q, rx, molecule, engine_k)
                except Exception as exc:
                    rec.update(status="failed", why_failed=f"{type(exc).__name__}: {exc}")
                    continue
                last_write = [time.perf_counter()]

                def on_row(entry, took, rec=rec, rx=rx, last_write=last_write):
                    if rec.get("run") and os.path.isabs(str(rec["run"])):
                        rec["run"] = os.path.relpath(rec["run"], output)
                    changed = entry is not None and take(entry, took, rec, replaces_estimate=False, shared=rx["steps"])
                    if changed:
                        with lock:
                            scan_rows[0] += 1
                            if took:
                                took_part.update(rx["steps"])
                    if changed or time.perf_counter() - last_write[0] > 20.0:
                        with lock:
                            rlay["wall_s"] = round(time.perf_counter() - t_rel, 1)
                            counts()
                            write("running")
                        last_write[0] = time.perf_counter()

                _relay_scan(q, rx["lowest"], molecule, ends, engine_k, Path(opt["scan_profile"]),
                            output / (fp.stem + "_relay") / f"{len(scans):02d}_{_file_name(rx['lowest'])[:-5]}", rec, on_row,
                            t0, cores=masks[k], compute=compute)
            finally:
                with lock:
                    active[0] -= 1
                    rlay["done"] += 1

    if scanning:
        rlay.update(status="running", lanes=len(masks), what=rlay["what"] + f"; up to {len(masks)} scans side by side")
        extra["scan"] = {
            "rule": "one scan per net reaction of the parent (molecules the same on both sides dropped). A reaction is "
                    "worth a scan if, with no barrier above its ends, (i) it would lower the target's route peak; those "
                    "go first, by the resulting peak, derived again each time a lane frees. When none is left: (ii) a "
                    "reaction that would bring another species to a route peak below the target's present one and "
                    "below what that species has now is put through the kinetics with its barrier removed (same "
                    "conditions, same time, the molecule's rows in), and is worth a scan only if the amount of some "
                    f"species at the end of the run then differs from what it is now by more than {SCAN_CHANGE:.0%} of "
                    "the limiting starting material (the target's own amount counts, up or down); largest change "
                    "first; tested again when a scan's rows change the kinetics. Ties: reactions where the estimate "
                    "or the verification saw the molecule react or take part, then fewer atoms; for a chiral molecule "
                    "mirror twins after everything else. The stage ends when no such reaction is left, or after "
                    f"{opt['scan']} reactions",
            "change_worth_a_scan": SCAN_CHANGE,
            "steps": scans, "derivations": [], "reactions": len(reactions),
            "cores": {"lanes": masks, "estimates_and_verification": pool_cpus}}
        threads = [threading.Thread(target=lane, args=(k,), daemon=True) for k in range(len(masks))]
        for th in threads[:2]:
            th.start()                 # two lanes from the start; the third when the verification ends

    # a + b. Each step's quick estimate, then its verification with the molecule beside it, in the derived order,
    # on the cores the lanes leave: a verification that is ready goes before the next estimate.
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait as wait_for
    import multiprocessing as mp

    lay["what"] = ("the molecule beside each step's ends and TS, the step's atoms held (mepd.bystander, "
                   + ("default setting: 3 repeats" if not opt["setting"] else "quick setting: 1 repeat, no polish") + ")")
    n_verify = len(todo) if opt["verify"] is None else max(0, opt["verify"])
    vdir = output / (fp.stem + "_verify")
    pending_est, pending_ver, running = deque(todo), deque(), {}
    t_a, last = time.perf_counter(), time.perf_counter()
    t_b = None
    asked = 0

    def on_estimate(ref, out):
        nonlocal asked
        lay["done"] += 1
        lay["wall_s"] = round(time.perf_counter() - t_a, 1)
        timing.setdefault("first_bystander_s", since())
        entry = {"step": ref, "layer": "bystander", "level": level, "binding": out.get("binding") or {},
                 "poses": out.get("poses"), "wall_s": out.get("wall_s"), "free_method": out.get("free_method"),
                 "spread_method": out.get("spread_method"), "placer": out.get("placer"),
                 "repeats": out.get("repeats"), "t_s": since()}
        if all(k in (out.get("complexes") or {}) for k in ("r", "ts", "p")):
            complexes[ref] = out["complexes"]
        if out.get("status") == "ok":
            state[ref] = "est"
            est["steps"].append({**entry, "status": "est"})
            if ref in twin_of:
                est["steps"].append({**entry, "step": twin_of[ref], "status": "est", "mirror_image_of": ref})
            if ref in complexes and asked < n_verify:
                asked += 1
                pending_ver.append(ref)
                vlay.update(status="running", total=asked)
        else:
            state[ref] = "reacted" if out.get("status") == "reacted" else "failed"
            if out.get("status") == "reacted":
                reacted.append(ref)
            failed.append({**entry, "status": out.get("status"), "why": out.get("why")})
        if lay["done"] == lay["total"]:
            lay["status"] = "done"
            timing["bystander_s"] = lay["wall_s"]

    def on_verified(res):
        sid = res["step"]
        vlay["done"] += 1
        vlay["wall_s"] = round(time.perf_counter() - t_b, 1)
        changed = False
        for row in res["rows"]:
            row["t_s"] = since()
            changed |= take(row, bool(row["takes_part"]))
            if row["takes_part"]:
                took_part.add(sid)
        if not res["rows"]:
            if res["elsewhere"]:
                state[sid] = "elsewhere"
                left_ends.append(sid)
                failed.append({"step": sid, "layer": "verify", "status": "leads elsewhere", "t_s": since(),
                               "why": "the TS optimized with the molecule present joins other species; the step "
                                      "keeps its estimate", "ends": res["elsewhere"]})
            else:
                failed.append({"step": sid, "layer": "verify", "status": "failed", "t_s": since(),
                               "why": "; ".join(res["notes"]) or "no verified TS"})
        elif res["elsewhere"]:
            left_ends.append(sid)
        return changed

    with ProcessPoolExecutor(max_workers=n_pool, mp_context=mp.get_context("spawn"), initializer=_pin,
                             initargs=(pool_cpus if scanning else None,)) as pool:
        while pending_est or pending_ver or running:
            while len(running) < n_pool and (pending_ver or pending_est):
                if pending_ver:
                    sid = pending_ver.popleft()
                    t_b = t_b or time.perf_counter()
                    fut = pool.submit(_verify_task, (profile, opt["query"], sid, frames_of(sid), molecule, complexes[sid],
                                                     str(vdir / _file_name(sid)[:-5]), q["temperature"]))
                    running[fut] = ("verify", sid)
                else:
                    sid = pending_est.popleft()
                    fut = pool.submit(_task, (profile, "steps", sid, frames_of(sid), molecule, q["temperature"],
                                              ([{"frame": pose.get("frame", "ts"), "coords": pose["coordinates"]}]
                                               if pose.get("step") == sid else ()), 0, opt["setting"]))
                    running[fut] = ("estimate", sid)
            done, _ = wait_for(list(running), return_when=FIRST_COMPLETED)
            changed = False
            with lock:
                for fut in done:
                    kind, sid = running.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as exc:
                        res = (("steps", sid, {"status": "failed", "why": f"{type(exc).__name__}: {exc}"}) if kind == "estimate"
                               else {"step": sid, "rows": [], "elsewhere": [], "notes": [f"{type(exc).__name__}: {exc}"]})
                    if kind == "estimate":
                        on_estimate(res[1], res[2])
                    else:
                        changed |= on_verified(res)
                if changed or time.perf_counter() - last > 10.0:
                    counts()
                    write("running")
                    last = time.perf_counter()
    with lock:
        lay["status"] = "done"
        timing.setdefault("bystander_s", lay["wall_s"])
        if vlay["status"] == "running":
            vlay["status"] = "done"
            timing["verify_s"] = vlay["wall_s"]
        elif vlay["status"] == "queued":
            vlay["status"] = "skipped"
        pool_done[0] = True
        counts()
        doc = write("running")

    if threads:
        for th in threads[2:]:
            th.start()
        for th in threads:
            th.join()
        with lock:
            rlay["status"] = "done"
            rlay["wall_s"] = round(time.perf_counter() - t_rel, 1)
            timing["relay_s"] = rlay["wall_s"]
            extra["scan"].setdefault("ended", {"t_s": since(), "why": f"the limit of {opt['scan']} reactions"})
            _peak, worth, unworthy, _flags = derive(final=True)
            extra["scan"]["left"] = (
                [{"step": rx["lowest"], "reaction": rx["label"], "steps": rx["steps"],
                  "why": f"worth a scan, not reached: the limit of {opt['scan']} reactions was used up"} for rx, _k, _v, _s in worth]
                + [{"step": rx["lowest"], "reaction": rx["label"], "steps": rx["steps"],
                    "why": "not worth a scan: with no barrier it would neither lower the target's route peak nor "
                           f"change the amount of any species at the end of the run by more than {SCAN_CHANGE:.0%} of the "
                           "limiting starting material"} for rx in unworthy])
            extra["scan"]["not_scanned"] = (f"{len(extra['scan']['left'])} of {len(reactions)} reactions were not scanned: "
                                            f"{len(unworthy)} not worth a scan, {len(worth)} beyond the limit")
    if rlay["status"] == "queued":
        rlay["status"] = "skipped"
    counts()
    return _line(write("done"), fp)


def _pin(cpus) -> None:
    """A worker process held to these CPUs (the scan lanes have the others)."""
    import os

    if cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cpus))
        except OSError:
            pass


def _line(doc: dict, fp: Path) -> dict:
    r = doc.get("readout") or {}
    t = r.get("target") or {}
    return {"candidate": doc["candidate"]["name"], "result": str(fp), "status": doc["status"],
            "target_parent": (t.get("parent") or {}).get("final"), "target_with": (t.get("with") or {}).get("final"),
            "factor": t.get("factor"), "fate": (r.get("added") or {}).get("fate"),
            "resting_state": ((r.get("added") or {}).get("resting_state") or {}).get("name"),
            "failed": len(doc["failed"]), "timings": doc.get("timings")}
