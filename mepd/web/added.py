"""Add a molecule to a workspace's network (docs/add-molecule.md): the workspace side of mepd/addmol.py.

* `adopt_run`: a finished `mepd discovery expand` run into a workspace without a server, through the
  functions the web uses for a running one (JobManager._adopt_live_events, recheck_expansion_steps).
* `member_energies`: the energies of the molecules of its mixtures, which the web queues as jobs.
* `network`: the workspace's kinetic model (mepd.web.kinetics.build, free energies) as the plain network
  mepd.addmol works on; `step_frames`: a step's two ends and TS in one atom order.
* `put`, `variant_network`: the "added molecule" variant record and the network with the molecule in it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from mepd import addmol
from mepd import network_variants as NV
from mepd.web.workspace import WorkspaceError, new_id

BOHR = 0.529177210903
KIND = "added_molecule"


class _Quiet:
    def publish(self, *_a, **_k) -> None:
        pass


def manager(ws):
    """The workspace's jobs, with no server behind them."""
    from mepd.web.jobs import JobManager

    m = JobManager(ws, _Quiet())
    m.load()
    return m


# ---------------------------------------------------------------- adoption
def adopt_run(mgr, run_dir: Path, *, profile: Optional[str] = None, title: str = "") -> dict:
    """A finished `mepd discovery expand` run (its folder with summary.json, species/ and ts/) as an
    imported job of the workspace, its species as nodes and its steps on their edges: the run's
    summary replayed as the species events a running expansion sends, then the web's own adoption
    (every step the run kept, each pair's edge its lowest IRC-verified one). Returns the job.

    The level of theory is the workspace profile `profile`, else a profile .toml kept beside the run.
    The molecules of the run's mixtures still need their own energies (`member_energies`)."""
    from mepd.web import chem

    ws = mgr.ws
    run_dir = Path(run_dir).expanduser().resolve()
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
    except (OSError, ValueError):
        raise WorkspaceError(f"{run_dir} has no readable summary.json: not a finished expansion") from None
    species = {int(s["index"]): s for s in summary.get("species") or []}

    def xyz(k):
        return (run_dir / "species" / f"species_{k}.xyz").read_text()

    seed = chem.structures_from_xyz_text(xyz(0))[0]
    job = mgr.import_external(run_dir, op_key="graph-enumeration", charge=int(seed.charge),
                              multiplicity=int(seed.multiplicity), title=title, profile=profile)
    if job.get("level") is None:
        raise WorkspaceError("the run's level of theory is not known: give a workspace profile, or keep the "
                             "run's profile .toml beside its folder")
    rec = ws.add_structure(seed, energy=summary.get("seed_energy"), level=job["level"], optimized=True,
                           origin={"kind": "job", "job": job["id"], "entry": "min_0", "label": "Minimum 0", "frame": 0})
    job["targets"]["structures"] = [rec["id"]]
    parent = {}
    for r in summary.get("reactions") or []:
        if r.get("outcome") == "new_species" and r.get("target") is not None:
            parent.setdefault(int(r["target"]), int(r["source"]))
    live = mgr.job_dir(job["id"]) / "live"
    live.mkdir(parents=True, exist_ok=True)
    with open(live / "events.jsonl", "w") as fh:
        for k in sorted(species):
            if k == 0:
                continue
            s = species[k]
            fh.write(json.dumps({"event": "species", "index": k, "parent": parent.get(k, 0), "smiles": s.get("smiles"),
                                 "energy_hartree": s.get("energy"), "rel_energy_kcal": s.get("rel_energy_kcal"),
                                 "validation": s.get("validation"), "xyz": xyz(k)}) + "\n")
    # (no server: nothing would run the jobs the web queues for the molecules' energies, nor re-check live steps)
    job["members_queued_at"] = job["steps_rechecked_at"] = time.time()
    mgr._adopt_live_events(job)
    mgr._adopt_seed_energy(job)
    mgr.recheck_expansion_steps(job)
    return job


def member_energies(ws, job: dict, engine, *, progress=None) -> dict:
    """Minimize, at the job's level, every molecule of the job's mixtures that has no energy there (the web
    queues an optimization job for these), each from the geometry it has in the mixture.
    Returns {"done": n, "failed": [names]}."""
    from mepd.nodes.node import StructureNode

    key = (job.get("level") or {}).get("key")
    st = ws.snapshot()["structures"]
    nodes = set((job.get("live_nodes") or {}).values())
    need = []
    for sid in nodes:
        rec = st.get(sid) or {}
        for m in ([sid] if rec.get("role") != "complex" else rec.get("members") or []):
            mr = st.get(m)
            if mr is not None and m not in need and not (mr.get("energy") is not None
                                                         and (mr.get("level") or {}).get("key") == key):
                need.append(m)
    done, failed = 0, []
    for n, m in enumerate(need):
        try:
            node = StructureNode(structure=ws.load_structure(m))
            best = engine.compute_geometry_optimization(node)[-1]
            energy = float(engine.compute_energies([best])[0])
            ws.replace_geometry(m, best.structure, energy=energy, level=job["level"])
            done += 1
        except Exception as exc:      # one molecule that does not minimize leaves its reactions out, no more
            failed.append(f"{st[m].get('name') or m}: {type(exc).__name__}")
        if progress:
            progress(n + 1, len(need))
    return {"done": done, "failed": failed}


# ---------------------------------------------------------------- the model
def step_id(info: dict) -> str:
    return info["reaction"] if info.get("kind") == "reaction" else f"edge:{info['edge']}"


def network(ws, jobs: dict, level_key: Optional[str], temperature: float, *, free_energy: bool = True,
            view: Optional[str] = None) -> dict:
    """The workspace's network as mepd.addmol takes it (the model Analyze > Kinetics runs: IRC-verified
    steps, species with an energy at the level), with free energies at `temperature` unless
    `free_energy` is False. Step ids are reaction ids, or "edge:<id>" for a step between two molecules.
    `view`: a variant's id, for its network; an added-molecule variant's own records come from
    `variant_network`."""
    from mepd.web import kinetics as K
    from mepd.web.variants import network_view

    full = ws.snapshot()
    snap = network_view(full, view)
    net, ids, info, excluded = K.build(snap, jobs, level_key, full=full)
    report = None
    g, g_ts = net.g, np.array([s.g_ts for s in net.steps])
    if free_energy and net.steps:
        used = sorted({i for s in net.steps for i in s.reactants + s.products})
        keep = {old: new for new, old in enumerate(used)}
        from mepd import microkinetics as mk

        sub = mk.Network([net.names[i] for i in used], net.g[used],
                         [mk.Step([keep[i] for i in s.reactants], [keep[i] for i in s.products], s.g_ts, s.label,
                                  info=s.info) for s in net.steps])
        sub.geom = [net.geom[i] for i in used]
        energies_at, report = K.free_energies(ws, jobs, sub)
        g, g_ts = energies_at(float(temperature))
        net, ids = sub, [ids[i] for i in used]
    st = snap["structures"]
    species = [{"id": sid, "name": net.names[k], "g": float(g[k]), "smiles": st[sid].get("smiles"),
                "charge": st[sid].get("charge", 0), "multiplicity": st[sid].get("multiplicity", 1),
                "status": "verified", "level": level_key} for k, sid in enumerate(ids)]
    steps = [{"id": step_id(i), "label": i["label"], "reactants": [ids[x] for x in s.reactants],
              "products": [ids[x] for x in s.products], "shuttles": i.get("shuttles") or [], "g_ts": float(x),
              "status": "est" if i.get("estimate") else "verified", "level": level_key, "edge": i["edge"],
              "job": i.get("job")} for s, i, x in zip(net.steps, info, g_ts)]
    notes = []
    if report:
        notes.append(f"free energies: {report['method']}, 1 M standard state")
        if report["ts_without_geometry"]:
            notes.append(f"{len(report['ts_without_geometry'])} TS without a geometry: thermal correction of its "
                         "reactants")
    return {"temperature": float(temperature), "energies": "free" if report else "electronic", "species": species,
            "steps": steps, "excluded": excluded, "notes": notes}


def _xyz_frames(text: str) -> list:
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        if not lines[i].strip().isdigit():
            i += 1
            continue
        n = int(lines[i])
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        out.append(([r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])))
        i += n + 2
    return out


def step_frames(ws, jobs: dict, edge: dict, level_key: Optional[str]) -> Optional[dict]:
    """A step's reactant end, TS and product end in one atom order (Angstrom), oriented along the edge:
    {"symbols", "charge", "multiplicity", "r", "ts", "p"}; the TS kinetics uses and its IRC's two ends
    (mepd.web.variants._ts_frames; for an expansion's step its own ts/<label>.xyz and <label>_irc.xyz).
    None when the step has no such files."""
    from mepd.web.variants import _angstrom, _end_structure, _graph, _ts_frames

    st = ws.snapshot()["structures"]
    q = {"charge": int(st[edge["source"]].get("charge", 0)), "multiplicity": int(st[edge["source"]].get("multiplicity", 1))}
    try:
        found = _ts_frames(ws, jobs, edge, level_key)
    except Exception:
        found = None
    if found is not None:
        (r, ts, p), _e, _jid = found
        return {"symbols": list(r[0]), **q, "r": r[1], "ts": ts[1], "p": p[1]}
    o = edge.get("origin") or {}
    out = Path((jobs.get(o.get("job") or "") or {}).get("output_dir") or "")
    label = o.get("label")
    if not label or not (out / "ts" / f"{label}.xyz").exists() or not (out / "ts" / f"{label}_irc.xyz").exists():
        return None
    (ts,) = _xyz_frames((out / "ts" / f"{label}.xyz").read_text())[:1]
    irc = _xyz_frames((out / "ts" / f"{label}_irc.xyz").read_text())
    if len(irc) < 2 or irc[0][0] != ts[0]:
        return None
    r, p = irc[0], irc[-1]
    snap = ws.snapshot()
    src = _end_structure(ws, snap, edge["source"], edge)
    g_src, g_r, g_p = _graph(list(src.symbols), _angstrom(src)), _graph(*r), _graph(*p)
    if g_p == g_src and g_r != g_src:
        r, p = p, r
    return {"symbols": list(ts[0]), **q, "r": r[1], "ts": ts[1], "p": p[1]}


def bond_changes(symbols, r, p) -> dict:
    """The bonds that differ between two frames of one atom order, as atom-index pairs (0-based)."""
    a, b = NV.adjacency(symbols, np.asarray(r)), NV.adjacency(symbols, np.asarray(p))
    pairs = lambda m: [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(m)))]   # noqa: E731
    return {"broken": pairs(a & ~b), "formed": pairs(b & ~a)}


# ---------------------------------------------------------------- the variant record
def put(ws, molecule: dict, amount: float, estimates: dict, *, name: str = "", on: Optional[str] = None,
        vid: Optional[str] = None) -> dict:
    """The "added molecule" variant of the workspace's network (workspace["variants"][id], beside the
    substituent variants): the molecule, its amount and the estimates `mepd.addmol.added` builds the
    kinetics from. It replaces nothing of the parent. With `vid`: that record's estimates replaced (a
    later layer's numbers are in)."""
    old = ws.variants.get(vid) if vid else None
    rec = {"id": vid or new_id("v_"), "kind": KIND, "name": name or (old or {}).get("name") or
           f"+ {molecule.get('name') or molecule.get('smiles') or molecule['id']}",
           "on": on if old is None else old.get("on"), "added": {**molecule, "amount": float(amount)},
           "base": None, "edit": f"added {molecule.get('name') or molecule.get('smiles') or molecule['id']}",
           "estimates": estimates, "replaced": [], "replaced_reactions": [], "labels": {}, "species_parent": {},
           "reactions": {}, "created": (old or {}).get("created") or time.time(), "updated": time.time()}
    return ws.put_variant(rec)


def variant_network(ws, jobs: dict, level_key: Optional[str], temperature: float, vid: str, *,
                    free_energy: bool = True, parent: Optional[dict] = None) -> dict:
    """The network of an added-molecule variant: its parent's (`network`) with the molecule in it."""
    var = ws.variants.get(vid)
    if var is None or var.get("kind") != KIND:
        raise WorkspaceError(f"{vid!r} is not an added-molecule variant")
    parent = parent or network(ws, jobs, level_key, temperature, free_energy=free_energy, view=var.get("on"))
    return addmol.added(parent, var["added"], var["estimates"])
