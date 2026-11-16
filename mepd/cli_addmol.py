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
                               workers=workers)
        else:
            ws = Workspace(workspace)
            doc = added.query(ws, added.manager(ws).jobs, target=target, initial=_amounts(initial),
                              temperature=temperature, time_s=time_s, free_energy=not electronic, fraction=fraction)
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


def _ts_steps(q: dict, entries: list, molecule: dict, engine, failed: list) -> list:
    """The candidate's new steps given with a TS geometry and its two ends, each put on the parent's scale as a
    whole (mepd.discovery.parent.ts_step): its barrier above its separated molecules, and which species its
    reacting molecules are. An entry that already has a barrier and no TS is kept as given."""
    import numpy as np

    from mepd import addmol
    from mepd.discovery import molecules as M
    from mepd.discovery import parent
    from mepd.web.variants import _structure

    c_key = _flat(M.molecule_parts(_structure(molecule["symbols"], molecule["conformers"][0], molecule["charge"],
                                              molecule["multiplicity"]))[0][1])

    def ids(mols):
        out = []
        for m in mols:
            hit = addmol.match(q["species"], m["smiles"]) or (
                [addmol.ADDED_ID] if _flat(m["smiles"]) == c_key else [])
            if len(hit) != 1:
                raise ValueError(f"{m['smiles']} in its TS is no species of the network (a new species has to be "
                                 "given under new_species)" if not hit else f"{m['smiles']} names several species")
            out.append(hit[0])
        return out

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
            done.append({**{k: v for k, v in n.items() if k != "ts"}, "reactants": ids(got["reactants"]),
                         "products": ids(got["products"]), "barrier": got["barrier"], "counted": got["counted"],
                         "spectators": [m["smiles"] for m in got["spectators"]], "barrier_method": got["method"]})
        except Exception as exc:
            failed.append({"step": n.get("id"), "layer": n.get("layer") or "given", "status": "failed",
                           "why": f"{type(exc).__name__}: {exc}"})
    return done


_ENGINE = None


def _task(args):
    """One bystander estimate in a worker process (the engine is made once per process)."""
    global _ENGINE
    profile, kind, ref, frames, molecule, temperature, starts, seed = args
    from mepd import addmol
    from mepd.cli_common import _open_run_inputs

    if _ENGINE is None:
        _ENGINE = _open_run_inputs(Path(profile)).engine
    try:
        return kind, ref, addmol.bystander(frames, molecule, _ENGINE, temperature=temperature, starts=starts, seed=seed)
    except Exception as exc:
        return kind, ref, {"status": "failed", "why": f"{type(exc).__name__}: {exc}", "binding": {}}


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
    workers: int = typer.Option(1, "--workers", help="Processes for the bystander layer."),
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
    by_id = {st["id"]: st for st in q["steps"]}
    order = [o["step"] for o in q["limits"]["order"] if (by_id.get(o["step"]) or {}).get("geometry")]
    rest = sorted((st for st in q["steps"] if st.get("geometry") and st["id"] not in order),
                  key=lambda st: st["apparent_barrier"]["forward"] if st["apparent_barrier"]["forward"] is not None
                  else float("inf"))
    work = order + [st["id"] for st in rest]
    if steps is not None:
        work = work[:max(0, steps)]
    if work and profile is None:
        raise typer.BadParameter("the bystander layer needs a profile: give --inputs (the query keeps none)")
    total = sum(s["amount"]["initial"] for s in q["species"])
    held = [s for s in q["species"] if s.get("atoms") and (s["id"] in q["target"]["species"]
                                                       or s["amount"]["max"] >= 1e-6 * total)]
    level = (q.get("level") or {}).get("key")
    engine = None
    with_ts = any(n.get("ts") for c in cands for n in ((c.get("estimates") or {}).get("new_steps") or []))
    if with_ts and profile is None:
        raise typer.BadParameter("a step given with a TS needs a profile for its free energy: give --inputs")
    if work or with_ts:
        from mepd.cli_common import _open_run_inputs

        engine = _open_run_inputs(profile).engine
    summary = []
    for cand in cands:
        t0 = time.perf_counter()
        fp = output / _file_name(cand["name"])
        given = cand.pop("estimates", None) or {}
        est = {k: [{"layer": "given", **e} for e in given.get(k) or []]
               for k in ("species", "steps", "new_species", "new_steps")}
        failed: list = []
        layers = [{"name": "given", "status": "done", "what": "estimates that came with the candidate",
                   "entries": sum(len(v) for v in est.values())}] if any(est.values()) else []
        lay = {"name": "bystander", "status": "running" if work else "skipped", "done": 0,
               "total": (len(work) + len(held)) if work else 0, "level": level, "wall_s": 0.0,
               "what": "the molecule beside each step's ends and TS, the step's atoms held (stand-in placer: "
                       "random packed placements)"}
        layers.append(lay)
        timing = {}

        def write(status):
            t = time.perf_counter()
            doc = addmol.result(q, cand, est, status=status, layers=layers, failed=failed)
            timing["kinetics_s"] = round(time.perf_counter() - t, 2)
            doc["timings"] = {**timing, "total_s": round(time.perf_counter() - t0, 1)}
            addmol.write_json(fp, doc)
            return doc

        molecule = None
        if work or any(n.get("ts") for n in est["new_steps"]):
            t = time.perf_counter()
            try:
                molecule = _prepare(cand, engine)
            except Exception as exc:
                failed.append({"layer": "candidate", "why": f"the candidate could not be built: {type(exc).__name__}: {exc}"})
                lay["status"] = "failed" if work else lay["status"]
                summary.append(_line(write("failed"), fp))
                continue
            timing["prepare_s"] = round(time.perf_counter() - t, 1)
            t = time.perf_counter()
            est["new_steps"] = _ts_steps(q, est["new_steps"], molecule, engine, failed)
            timing["ts_steps_s"] = round(time.perf_counter() - t, 1)
        doc = write("running" if work else "done")
        if work:
            # A complex a given step starts from or ends in needs its binding.
            named = {x[:-len(addmol.ADDED_ID) - 1] for n in est["new_steps"] for x in n["reactants"] + n["products"]
                     if x.endswith("." + addmol.ADDED_ID)}
            held_c = held + [s for s in q["species"] if s["id"] in named and s not in held and s.get("atoms")]
            lay["total"] = len(work) + len(held_c)
            pose = cand.get("pose") or {}
            tasks = []
            for s in held_c:
                fr = {"symbols": s["atoms"], "charge": s.get("charge", 0), "multiplicity": s.get("multiplicity", 1),
                      "r": s["coordinates"], "ts": None, "p": None}
                tasks.append((str(profile), "species", s["id"], fr, molecule, q["temperature"], (), 0))
            for sid in work:
                g = by_id[sid]["geometry"]
                fr = {"symbols": g["atoms"], "charge": g["charge"], "multiplicity": g["multiplicity"],
                      "r": g["reactant"], "ts": g["ts"], "p": g["product"]}
                starts = ([{"frame": pose.get("frame", "ts"), "coords": pose["coordinates"]}]
                          if pose.get("step") == sid else ())
                tasks.append((str(profile), "steps", sid, fr, molecule, q["temperature"], starts, 0))
            t, last = time.perf_counter(), time.perf_counter()
            if workers > 1:
                import multiprocessing as mp

                pool = mp.get_context("spawn").Pool(workers)
                stream = pool.imap(_task, tasks)       # in the order of work
            else:
                pool, stream = None, map(_task, tasks)
            try:
                for kind, ref, out in stream:
                    lay["done"] += 1
                    lay["wall_s"] = round(time.perf_counter() - t, 1)
                    entry = {("species" if kind == "species" else "step"): ref, "layer": "bystander", "level": level,
                             "binding": out.get("binding") or {}, "poses": out.get("poses"),
                             "wall_s": out.get("wall_s"), "free_method": out.get("free_method"),
                             "spread_method": out.get("spread_method")}
                    if out["status"] == "ok":
                        if kind == "species":
                            entry["binding"] = entry["binding"]["r"]
                        est[kind].append({**entry, "status": "est"})
                    else:
                        failed.append({**entry, "status": out["status"], "why": out.get("why")})
                    if time.perf_counter() - last > 10.0:      # the file follows the layer
                        doc = write("running")
                        last = time.perf_counter()
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
            lay["status"] = "done"
            timing["bystander_s"] = lay["wall_s"]
            doc = write("done")
        summary.append(_line(doc, fp))
    typer.echo(json.dumps(summary, indent=1))


def _line(doc: dict, fp: Path) -> dict:
    r = doc.get("readout") or {}
    t = r.get("target") or {}
    return {"candidate": doc["candidate"]["name"], "result": str(fp), "status": doc["status"],
            "target_parent": (t.get("parent") or {}).get("final"), "target_with": (t.get("with") or {}).get("final"),
            "factor": t.get("factor"), "fate": (r.get("added") or {}).get("fate"),
            "resting_state": ((r.get("added") or {}).get("resting_state") or {}).get("name"),
            "failed": len(doc["failed"]), "timings": doc.get("timings")}
