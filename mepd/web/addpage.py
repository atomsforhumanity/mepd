"""The "Add a molecule" page of mepd web (docs/add-molecule.md): a thin layer over mepd.web.added and
`mepd addmol test`. Everything lives in files under `<workspace>/addmol/<query id>/`:

    query.json                 the query file (mepd.web.added.query): the network and what limits the target
    results/<name>.json        one result file per candidate, rewritten by the job as its layers finish

so a result file written by a command-line run and copied into `results/` is shown like any other.
The page reads them through `view` (one line per candidate) and `detail` (one candidate).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

from mepd import addmol
from mepd.web import added
from mepd.web.workspace import WorkspaceError

OP = "addmol-test"
PENDING = "pending-candidates.json"
# A parent step's barrier that moved by less than this (kcal/mol) is not listed as changed.
MOVED = 0.05

_cache: dict = {}


def _load(path: Path) -> dict:
    """A JSON file, re-read only when it changed (the page polls)."""
    key = str(path)
    stamp = path.stat().st_mtime_ns
    if _cache.get(key, (None,))[0] != stamp:
        _cache[key] = (stamp, json.loads(path.read_text()))
    return _cache[key][1]


def folder(ws, qid: Optional[str] = None) -> Path:
    base = ws.root / "addmol"
    if qid is None:
        return base
    if not qid.replace("_", "").isalnum():
        raise WorkspaceError(f"{qid!r} is not a query id")
    return base / qid


def _query(ws, qid: str) -> dict:
    fp = folder(ws, qid) / "query.json"
    if not fp.exists():
        raise WorkspaceError(f"no query {qid!r} in this workspace")
    return _load(fp)


def _result_path(ws, qid: str, name: str) -> Path:
    fp = folder(ws, qid) / "results" / name
    if "/" in name or "\\" in name or not name.endswith(".json") or not fp.exists():
        raise WorkspaceError(f"no result {name!r} for query {qid!r}")
    return fp


# ---------------------------------------------------------------- the query
def source_run(ws, jobs: dict) -> Optional[tuple]:
    """(job, run folder) when the workspace's network is one finished exploration: every edge with a
    verified TS is a step of the same `mepd discovery expand` run, and its folder is there. The parent
    of such a network is the explorer's own molecule-level network (its rules, its free energies, every
    verified step: mepd.discovery.parent), not its adoption into the workspace. None otherwise."""
    try:
        from mepd.discovery import parent
    except ImportError:
        return None
    owners = {(e.get("origin") or {}).get("job") for e in ws.snapshot()["edges"].values()
              if (e.get("origin") or {}).get("barrier_verified")}
    if len(owners) != 1:
        return None
    job = jobs.get(owners.pop() or "")
    if not job or job.get("op") != "graph-enumeration" or job.get("status") != "done":
        return None
    run = Path(job.get("output_dir") or "")
    return (job, run) if parent.is_run(run) else None


def make_query(ws, jobs: dict, *, target: str, initial: dict, temperature: float, time_s: float, held=(),
               free_energy: bool = True, together: bool = True, source: str = "auto", workers: int = 2) -> str:
    """Build the query file for these conditions (or keep the one that is there) and return its id.
    The id names the question (target, conditions, network), so asking it again finds its results.
    `target`, `held` and the keys of `initial` are structure ids; `together` counts the target's
    stereoisomers as one. `source`: "auto" (the exploration's own network where the workspace is one
    finished exploration, `source_run`; else the workspace's) or "workspace"."""
    # (`mepd addmol query` takes no constant feed for a run yet: a question with one is asked of the workspace's)
    found = source_run(ws, jobs) if source == "auto" and free_energy and not held else None
    key = json.dumps([target, sorted(initial.items()), sorted(held), float(temperature), float(time_s),
                      bool(free_energy), bool(together), "run" if found else "workspace"])
    qid = "q_" + hashlib.sha1(key.encode()).hexdigest()[:8]
    fp = folder(ws, qid) / "query.json"
    if fp.exists():
        return qid
    st = ws.snapshot()["structures"]
    missing = [x for x in [target, *initial, *held] if x not in st]
    if missing:
        raise WorkspaceError(f"unknown structure {missing[0]!r}")
    smiles = lambda sid: st[sid].get("smiles") or sid   # noqa: E731
    goal = target
    if together and st[target].get("smiles"):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(st[target]["smiles"])
        if mol is not None:
            Chem.RemoveStereochemistry(mol)
            goal = Chem.MolToSmiles(mol)          # (no stereo marks: every stereoisomer, counted together)
    if found:
        # `mepd addmol query RUN` in a process of its own: it builds the run's engine for the thermal
        # corrections the run did not record (engines set process-wide state; the server builds none).
        import subprocess
        import sys

        tmp = fp.with_name("query.making.json")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-m", "mepd.cli", "addmol", "query", str(found[1]), "--target",
                goal if goal != target else smiles(target), "--temperature", str(float(temperature)),
                "--time", str(float(time_s)), "--workers", str(workers), "--output", str(tmp)]
        for k, v in initial.items():
            argv += ["--initial", f"{smiles(k)}={float(v)}"]
        done = subprocess.run(argv, capture_output=True, text=True)
        if done.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            lines = [ln.strip(" │╭╰─╮╯") for ln in (done.stderr or done.stdout).strip().splitlines() if ln.strip(" │╭╰─╮╯")]
            raise WorkspaceError(f"the query could not be built: {lines[-1] if lines else 'mepd addmol query failed'}")
        doc = json.loads(tmp.read_text())
        tmp.unlink()
        kept = tmp.with_name(tmp.name + ".provenance.json")       # (what the command recorded about itself)
        if kept.exists():
            kept.replace(fp.with_name("query.provenance.json"))
    else:
        try:
            doc = added.query(ws, jobs, target=goal, initial=initial, temperature=temperature, time_s=time_s,
                              held=held, free_energy=free_energy)
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from None
    # What the page asked, in the workspace's own ids (an exploration's network names species by SMILES).
    doc["asked"] = {"target": target, "together": bool(together), "initial": dict(initial), "held": list(held),
                    "job": found[0]["id"] if found else None}
    fp.parent.mkdir(parents=True, exist_ok=True)
    addmol.write_json(fp, doc)
    return qid


def _edges(ws, q: dict) -> dict:
    """{step id: the workspace edge that step is on}: a workspace query names it; an exploration's step
    is found by its label among the edges its job put there."""
    out = {st["id"]: st["edge"] for st in q["steps"] if st.get("edge")}
    job = (q.get("asked") or {}).get("job")
    if job:
        by_label = {(e.get("origin") or {}).get("label"): eid for eid, e in ws.snapshot()["edges"].items()
                    if (e.get("origin") or {}).get("job") == job}
        out.update({st["id"]: by_label[st["run_step"]] for st in q["steps"] if st.get("run_step") in by_label})
    return out


def _nodes(ws, q: dict) -> dict:
    """{species id: the workspace structures that species is}: itself in a workspace query; by SMILES for an
    exploration's (which names species by SMILES, and counts mirror images as one)."""
    st = ws.snapshot()["structures"]
    out = {}
    for s in q["species"]:
        if s["id"] in st:
            out[s["id"]] = [s["id"]]
        elif s.get("smiles"):
            mine = [{"id": sid, "smiles": r.get("smiles")} for sid, r in st.items() if r.get("smiles") and r.get("role") != "complex"]
            out[s["id"]] = addmol.match(mine, s["smiles"])
    return out


def _names(q: dict, ws=None) -> dict:
    """{species id: name}; with the workspace, the name its structure has there (an exploration's network
    names species by SMILES)."""
    out = {s["id"]: s.get("name") or s["id"] for s in q["species"]}
    if ws is not None and (q.get("asked") or {}).get("job"):
        st = ws.snapshot()["structures"]
        for sid, mine in _nodes(ws, q).items():
            if mine and st[mine[0]].get("name"):
                out[sid] = st[mine[0]]["name"]
    return out


def _labels(q: dict, names: dict) -> dict:
    """{step id: its label in the names the page uses}: the query's own, or, for an exploration's network (whose
    labels are SMILES), written from the species' names."""
    if not (q.get("asked") or {}).get("job"):
        return {st["id"]: st.get("label") or st["id"] for st in q["steps"]}
    side = lambda ids: " + ".join(names.get(x, x) for x in ids)   # noqa: E731
    return {st["id"]: f"{side(st['reactants'])} -> {side(st['products'])}" for st in q["steps"]}


def _relabel(rows, labels: dict) -> None:
    for r in rows or []:
        if r and r.get("step") in labels:
            r["label"] = labels[r["step"]]


def _limits(q: dict, edges: dict, names: dict) -> dict:
    """What limits the target, with the names and barriers the page shows beside each step."""
    lim = q["limits"]
    steps = {st["id"]: st for st in q["steps"]}
    labels = _labels(q, names)

    def row(o: dict) -> dict:
        st = steps.get(o["step"]) or {}
        return {**o, "label": labels.get(o["step"], o["step"]), "barrier": (st.get("barrier") or {}).get("forward"),
                "status": st.get("status"), "edge": edges.get(o["step"]), "mirror": bool(st.get("mirror_of"))}

    return {"defined": lim.get("defined"), "basis": lim.get("basis"), "why": lim.get("why"),
            "cutoff_kcal": lim.get("cutoff_kcal"), "order": [row(o) for o in lim.get("order") or []],
            # (stabilizing a starting material always costs target, trivially: the page lists the others)
            "traps": [{**t, "name": names.get(t["species"], t["species"])} for t in lim.get("traps") or []
                      if not q["conditions"]["initial"].get(t["species"]) and t["species"] not in q["target"]["species"]],
            "routes": [{**r, "steps": [{**x, "label": labels.get(x["step"], x.get("label"))} for x in r["steps"]]}
                       for r in lim.get("routes") or []]}


def summary(q: dict, edges: Optional[dict] = None, names: Optional[dict] = None) -> dict:
    """The query as the page reads it: no coordinates, no time series."""
    names = names or _names(q)
    cond = q["conditions"]
    return {"target": {**q["target"], "names": [names.get(x, x) for x in q["target"]["species"]]},
            "conditions": {**cond, "initial": [{"id": k, "name": names.get(k, k), "amount": v}
                                               for k, v in cond["initial"].items()]},
            "energies": q.get("energies"), "level": (q.get("level") or {}).get("key"),
            "limits": _limits(q, edges or {}, names), "notes": q.get("notes") or [], "left_out": q.get("left_out"),
            "source": q.get("source") or "workspace", "asked": q.get("asked"),
            "counts": {"species": len(q["species"]), "steps": len(q["steps"]),
                       "with_geometry": sum(1 for st in q["steps"] if st.get("geometry"))}}


def queries(ws) -> list:
    """The questions asked in this workspace, the latest first: id, target, conditions, results."""
    out = []
    for fp in folder(ws).glob("*/query.json"):
        try:
            q = _load(fp)
        except (OSError, ValueError):
            continue
        s = summary(q, names=_names(q, ws))
        out.append({"id": fp.parent.name, "target": s["target"], "conditions": s["conditions"],
                    "results": len(list((fp.parent / "results").glob("*.json"))), "made": fp.stat().st_mtime})
    return sorted(out, key=lambda r: -r["made"])


# ---------------------------------------------------------------- candidates
def parse_candidates(text: str, amount: Optional[float] = None, filename: str = "") -> list:
    """Candidates from what was typed, pasted or loaded: a candidates file (JSON, docs/add-molecule.md),
    an xyz file (every frame one candidate), or lines of `SMILES [name] [amount in mol/L]`. Each comes
    back as the candidate `mepd addmol test` takes, plus `formula`, `natoms` and, if it cannot be used,
    `error`."""
    text = text.strip()
    stem = Path(filename).stem if filename else ""
    out = []
    if text[:1] in "[{":
        try:
            loaded = json.loads(text)
        except ValueError as exc:
            raise WorkspaceError(f"not a candidates file: {exc}") from None
        out = list(loaded if isinstance(loaded, list) else loaded.get("candidates") or [])
        if not all(isinstance(c, dict) for c in out):
            raise WorkspaceError("a candidates file is a list of {name, smiles | atoms + coordinates, amount}")
    elif text.split("\n", 1)[0].strip().isdigit():
        frames = added._xyz_frames(text)
        lines = text.splitlines()
        at = 0
        for k, (symbols, coords) in enumerate(frames):
            while not lines[at].strip().isdigit():
                at += 1
            comment = lines[at + 1].strip() if at + 1 < len(lines) else ""
            at += len(symbols) + 2
            charge = next((int(t.split("=")[1]) for t in comment.split() if t.startswith(("charge=", "qcdata_charge="))), 0)
            mult = next((int(t.split("=")[1]) for t in comment.split()
                         if t.startswith(("multiplicity=", "qcdata_multiplicity="))), 1)
            out.append({"name": (stem or "structure") + (f"_{k + 1}" if len(frames) > 1 else ""), "atoms": symbols,
                        "coordinates": coords.tolist(), "charge": charge, "multiplicity": mult})
    else:
        for line in text.splitlines():
            parts = line.split("#", 1)[0].split() if not line.lstrip().startswith("#") else []
            if not parts:
                continue
            cand = {"smiles": parts[0]}
            rest = parts[1:]
            if rest:
                try:
                    cand["amount"] = float(rest[-1])
                    rest = rest[:-1]
                except ValueError:
                    pass
            cand["name"] = " ".join(rest) or parts[0]
            out.append(cand)
    seen = set()
    for k, c in enumerate(out):
        c.setdefault("name", c.get("smiles") or f"candidate_{k + 1}")
        if c.get("amount") is None and amount is not None:
            c["amount"] = float(amount)
        problem = None
        if c.get("smiles"):
            from rdkit import Chem
            from rdkit.Chem.rdMolDescriptors import CalcMolFormula

            mol = Chem.MolFromSmiles(c["smiles"])
            if mol is None:
                problem = "this SMILES cannot be read"
            else:
                full = Chem.AddHs(mol)
                c["formula"], c["natoms"] = CalcMolFormula(full), full.GetNumAtoms()
        elif c.get("atoms") and c.get("coordinates") and len(c["atoms"]) == len(c["coordinates"]):
            c["natoms"] = len(c["atoms"])
        else:
            problem = "give a SMILES, or atoms and coordinates"
        if not problem and not (isinstance(c.get("amount"), (int, float)) and c["amount"] > 0):
            problem = "no amount (mol/L)"
        if not problem and c["name"] in seen:
            problem = "this name is used twice"
        seen.add(c["name"])
        if problem:
            c["error"] = problem
    return out


def file_name(name: str) -> str:
    from mepd.cli_addmol import _file_name

    return _file_name(name)


def start(manager, qid: str, candidates: list, *, steps: Optional[int] = None, workers: int = 2) -> dict:
    """Queue `mepd addmol test` for these candidates (one job; a result file each, in the query's
    `results/`). Refused: a candidate with an `error`, or one whose result is still being written."""
    ws = manager.ws
    q = _query(ws, qid)
    bad = [f"{c['name']}: {c['error']}" for c in candidates if c.get("error")]
    if bad or not candidates:
        raise WorkspaceError("; ".join(bad) or "no candidate given")
    busy = running(manager, qid)
    taken = [c["name"] for c in candidates if file_name(c["name"]) in busy]
    if taken:
        raise WorkspaceError(f"{', '.join(taken)}: still being tested")
    clean = [{k: v for k, v in c.items() if k not in ("formula", "natoms", "error")} for c in candidates]
    base = folder(ws, qid)
    (base / "results").mkdir(exist_ok=True)
    pending = base / PENDING           # (the job's builder moves it into the job's inputs)
    pending.write_text(json.dumps(clean, indent=1))
    names = ", ".join(c["name"] for c in clean[:3]) + (f" and {len(clean) - 3} more" if len(clean) > 3 else "")
    try:
        (job,) = manager.submit(OP, structure_ids=[(q.get("asked") or {}).get("target") or q["target"]["species"][0]],
                                edge_ids=[],
                                params={"query": qid, "steps": steps, "workers": workers,
                                        "names": [file_name(c["name"]) for c in clean]},
                                profile=None, label=f"Add a molecule: {names}")
    finally:
        pending.unlink(missing_ok=True)
    return job


def running(manager, qid: str) -> dict:
    """{result file name: job id} of the candidates a queued or running job is testing."""
    out = {}
    for j in manager.jobs.values():
        if j.get("op") == OP and j.get("status") in ("queued", "running") and (j.get("params") or {}).get("query") == qid:
            for n in (j.get("params") or {}).get("names") or []:
                out[n] = j["id"]
    return out


# ---------------------------------------------------------------- results
def _on_route(readout: dict) -> float:
    """The share of the target's material that came by routes with the added molecule in them."""
    share = (readout.get("added") or {}).get("route_share")
    if share is not None:
        return share
    return sum(r.get("share") or 0.0 for r in (readout.get("routes") or {}).get("with") or [] if r.get("with_added"))


def _through(row: dict) -> bool:
    """A parent step whose lowest version goes through the added molecule (the read-out's `group`; files
    written before it had one: by `through`)."""
    if row.get("group"):
        return row["group"] == "through"
    return row.get("through") is not None and row["through"] != row["step"]


def _moved(row: dict) -> bool:
    """A parent step whose own TS is unchanged and whose barrier moved because amounts moved."""
    if row.get("group"):
        return row["group"] == "amounts"
    return not _through(row) and abs(row.get("change") or 0.0) > MOVED


def line(doc: dict, order=()) -> dict:
    """One candidate in one line: what it does to the target, whether it is returned, where it rests,
    which layers are in. `best`: the step that changes most through the molecule, among the steps that limit
    the target (`order`: the query's order of work) where any of them does. `effect` orders candidates (larger: more of the target): the logarithm of the
    factor where both amounts are above the kinetics' resolution, else how far the lowest route's peak
    came down, kept below every resolved gain."""
    cand, r = doc.get("candidate") or {}, doc.get("readout")
    out = {"name": cand.get("name"), "smiles": cand.get("smiles"), "amount": cand.get("amount"),
           "status": doc.get("status"), "updated": doc.get("updated"),
           "layers": [{k: lay.get(k) for k in ("name", "status", "done", "total", "wall_s", "level", "what", "entries")}
                      for lay in doc.get("layers") or []],
           "failed": len(doc.get("failed") or []), "used": {"est": 0, "verified": 0}}
    for group in (doc.get("estimates") or {}).values():
        for e in group or []:
            if e.get("status") in out["used"]:
                out["used"][e["status"]] += 1
    # Why a whole layer gave nothing (not one step's failure).
    out["why"] = next((f.get("why") for f in doc.get("failed") or [] if not f.get("step") and not f.get("species")), None)
    if not r:
        return {**out, "readout": False, "effect": None}
    t, a = r["target"], r.get("added") or {}
    peak = {side: ((r.get("peak_routes") or {}).get(side) or [{}])[0].get("peak") for side in ("parent", "with")}
    through = [s for s in r["steps"] if _through(s) and s.get("change") is not None]
    limiting = [s for s in through if s["step"] in set(order)]
    best = min(limiting or through, key=lambda s: s["change"], default=None)
    worst = max((s for s in r["steps"] if s.get("change") is not None), key=lambda s: s["change"], default=None)
    drop = (peak["parent"] - peak["with"]) if peak["parent"] is not None and peak["with"] is not None else 0.0
    if t.get("factor") is not None and t["factor"] > 0:
        effect = 1000.0 + math.log10(t["factor"])
    elif t["with"].get("resolved") and not t["parent"].get("resolved"):
        effect = 2000.0 + math.log10(t["with"]["final"])        # the parent makes none: any amount is more
    else:
        effect = drop
    return {**out, "readout": True, "effect": effect,
            "target": {"parent": t["parent"], "with": t["with"], "factor": t.get("factor"),
                       "time_to_parent_amount_s": t.get("time_to_parent_amount_s")},
            "first_order": (r.get("first_order") or {}).get("factor"),
            "peak": peak, "on_route": _on_route(r),
            "fate": a.get("fate"), "returns": a.get("returns"), "shares": a.get("shares"),
            "resting_state": a.get("resting_state"),
            "through": len(through),
            "steps": [{k: s.get(k) for k in ("step", "label", "direction", "change", "status", "spread")}
                      for s in sorted(through, key=lambda s: s["change"])[:6]],
            "best": best and {**{k: best.get(k) for k in ("step", "label", "direction", "before", "after", "change",
                                                          "status", "spread")}, "limits": bool(limiting)},
            "raised": worst and worst["change"] > MOVED and {k: worst.get(k) for k in ("step", "label", "change", "status")}}


def view(manager, qid: str) -> dict:
    """What the page shows for one query: its summary and one line per candidate, the largest effect on
    the target first; candidates whose job has not written a file yet are listed as queued."""
    ws = manager.ws
    q = _query(ws, qid)
    busy = running(manager, qid)
    names = _names(q, ws)
    labels = _labels(q, names)
    order = [o["step"] for o in q["limits"].get("order") or []]
    rows = []
    for fp in sorted((folder(ws, qid) / "results").glob("*.json")):
        try:
            doc = _load(fp)
        except (OSError, ValueError):
            continue                       # (being replaced this instant: the next poll has it)
        if doc.get("schema") != addmol.RESULT:
            continue
        row = {**line(doc, order), "file": fp.name, "job": busy.get(fp.name)}
        _relabel([row.get("best"), row.get("raised") or None, *(row.get("steps") or [])], labels)
        if row["status"] == "running" and fp.name not in busy:
            # The file says running and no job is writing it: its job was cancelled or the server restarted.
            row["status"] = "interrupted"
        rows.append(row)
    have = {r["file"] for r in rows}
    rows.sort(key=lambda r: math.inf if r["effect"] is None else -r["effect"])
    # (one calculation tests its candidates one after another: those it has not reached yet wait)
    rows += [{"name": n[:-5], "file": n, "status": "queued", "job": j, "readout": False,
              "layers": [], "failed": 0, "effect": None} for n, j in busy.items() if n not in have]
    kept = {(v.get("addmol") or {}).get("result"): v["id"] for v in ws.variants.values()
            if (v.get("addmol") or {}).get("query") == qid}
    for r in rows:
        r["variant"] = kept.get(r["file"])
    return {"id": qid, **summary(q, _edges(ws, q), names), "results": rows}


def _source(doc: dict) -> dict:
    """{id of a step with the molecule (`through` of the read-out): the estimate its number comes from}."""
    est = doc.get("estimates") or {}
    out = {f"{e['step']}+{addmol.ADDED_ID}": e for e in est.get("steps") or [] if e.get("step")}
    out.update({e["id"]: e for e in est.get("new_steps") or [] if e.get("id")})
    return out


def detail(ws, qid: str, name: str) -> dict:
    """One candidate's result as the page reads it: the read-out with, for every changed step, the layer
    and level its number is from; the two groups of barrier changes (through the molecule; only
    because amounts moved); what is new; what failed and why. The estimates' raw bindings stay in the
    file."""
    q = _query(ws, qid)
    doc = _load(_result_path(ws, qid, name))
    r = doc.get("readout")
    steps = {st["id"]: st for st in q["steps"]}
    edges = _edges(ws, q)
    source = _source(doc)
    labels = _labels(q, _names(q, ws))
    out = {"file": name, **line(doc, [o["step"] for o in q["limits"].get("order") or []]), "candidate": {k: v for k, v in (doc.get("candidate") or {}).items()
                                                    if k not in ("coordinates", "pose", "atoms")},
           "failed": [{**{k: f.get(k) for k in ("layer", "status", "why", "step", "species")},
                       "label": labels.get(f.get("step")) or _names(q, ws).get(f.get("species"), f.get("species"))}
                      for f in doc.get("failed") or []],
           "timings": doc.get("timings"), "through_steps": [], "moved_steps": [], "new_steps": [], "new_species": []}
    if not r:
        return out

    def row(s: dict) -> dict:
        e = source.get(s.get("through")) or {}
        st = steps.get(s["step"]) or {}
        return {**s, "layer": e.get("layer"), "level": e.get("level"), "source": e.get("source"),
                "via": e.get("label") if e.get("id") else None, "geometry": bool(st.get("geometry")),
                "with_geometry": bool(e.get("complexes") or e.get("geometry")), "edge": edges.get(s["step"])}

    changed = [s for s in r["steps"] if s.get("change") is not None]
    out["through_steps"] = sorted((row(s) for s in changed if _through(s)), key=lambda s: s["change"])
    out["moved_steps"] = sorted(({**s, "edge": edges.get(s["step"])} for s in changed if _moved(s)),
                                key=lambda s: -abs(s["change"]))
    out["unchanged"] = len(changed) - len(out["through_steps"]) - len(out["moved_steps"])
    _relabel([out.get("best"), *out["through_steps"], *out["moved_steps"]], labels)
    out["new_steps"] = [{k: e.get(k) for k in ("id", "label", "parent_step", "status", "layer", "level", "source")}
                        | {"barrier": e.get("barrier"), "with_geometry": bool(e.get("geometry"))}
                        for e in (doc.get("estimates") or {}).get("new_steps") or []]
    # The species that hold the molecule, those that ever hold material first.
    out["new_species"] = sorted(r.get("new_species") or [], key=lambda s: -(s.get("max") or 0.0))[:40]
    out["new_species_total"] = len(r.get("new_species") or [])
    out["routes"] = (r.get("routes") or {}).get("with") or []
    out["peak_routes"] = r.get("peak_routes")
    out["notes"] = r.get("notes") or []
    out["first_order_detail"] = r.get("first_order")
    return out


def remove(manager, qid: str, name: str) -> None:
    fp = _result_path(manager.ws, qid, name)
    if name in running(manager, qid):
        raise WorkspaceError("this candidate is still being tested: cancel its calculation first")
    fp.unlink()
    _cache.pop(str(fp), None)


# ---------------------------------------------------------------- geometry for the 3D viewer
def _xyz(symbols, coords, comment: str) -> str:
    rows = "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, coords))
    return f"{len(symbols)}\n{comment}\n{rows}\n"


def geometry(ws, qid: str, step: str, name: Optional[str] = None, through: Optional[str] = None) -> dict:
    """A parent step's reactant end, TS and product end as xyz frames. With a candidate's result file and
    the version of the step its number is read from (`through` of the read-out): the frames with the
    molecule where that estimate kept them (`complexes` of a bystander estimate, the step's atoms first,
    then `molecule_symbols`; `geometry` of a step the molecule takes part in, in the form of the query's),
    else the parent's own frames and `with_molecule` False."""
    q = _query(ws, qid)
    st = next((s for s in q["steps"] if s["id"] == step), None)
    if st is None:
        raise WorkspaceError(f"no step {step!r} in this query")
    est = (_source(_load(_result_path(ws, qid, name))).get(through) or {}) if name and through else {}
    g, label = st.get("geometry"), st.get("label") or step
    words = (("reactant", "r", "reactant end"), ("ts", "ts", "TS"), ("product", "p", "product end"))
    if est.get("geometry"):
        eg = est["geometry"]
        frames = [_xyz(eg["atoms"], eg[k], f"{est.get('label') or label} · {w}") for k, _, w in words if eg.get(k)]
        return {"label": est.get("label") or label, "frames": frames, "ts": 1 if len(frames) == 3 else 0,
                "with_molecule": True}
    if g is None:
        raise WorkspaceError("this step has no TS and IRC files: nothing to show")
    cx, molecule = est.get("complexes") or {}, est.get("molecule_symbols") or []
    frames = []
    for key, short, word in words:
        if cx.get(short) and molecule and len(cx[short]) == len(g["atoms"]) + len(molecule):
            frames.append(_xyz(list(g["atoms"]) + list(molecule), cx[short], f"{label} · {word} with the molecule"))
        else:
            frames.append(_xyz(g["atoms"], g[key], f"{label} · {word}"))
    return {"label": label, "frames": frames, "ts": 1, "with_molecule": bool(cx and molecule)}


# ---------------------------------------------------------------- the variant (Explore's Network picker, the map)
def marks(q: dict, doc: dict, edges: dict, nodes: dict) -> dict:
    """What the network map marks for a candidate: per parent step whose lowest version goes through the
    molecule (by its edge) the change of its apparent barrier and whether that is an estimate; per parent
    species the complex with the molecule that holds material; what has no place on the parent's map (new
    species, steps without a parent step); and how many steps moved only because amounts moved (not
    marked: their own transition states are unchanged)."""
    r = doc.get("readout") or {}
    steps = {st["id"]: st for st in q["steps"]}
    names = _names(q)
    out = {"edges": {}, "species": {}, "new": [], "moved": 0, "status": doc.get("status"), "updated": doc.get("updated")}
    for s in r.get("steps") or []:
        edge = edges.get(s["step"])
        if edge is None or s.get("change") is None:
            continue
        if _through(s):
            out["edges"][edge] = {"change": round(s["change"], 2), "status": s.get("status"), "spread": s.get("spread")}
        elif _moved(s):
            out["moved"] += 1
    total = sum((r.get("conditions") or {}).get("initial", {}).values()) or 1.0
    for s in r.get("new_species") or []:
        base = s["id"].removesuffix("." + addmol.ADDED_ID)
        if s.get("kind") == "complex" and base in names:
            if (s.get("max") or 0.0) > addmol.RESOLUTION * total:
                for node in nodes.get(base) or []:
                    out["species"][node] = {"name": s.get("name"), "max": s.get("max"), "status": s.get("status")}
        else:
            out["new"].append({"name": s.get("name") or s["id"], "kind": "species", "status": s.get("status")})
    for e in (doc.get("estimates") or {}).get("new_steps") or []:
        if e.get("parent_step") not in steps:
            out["new"].append({"name": e.get("label") or e.get("id"), "kind": "step", "status": e.get("status")})
    return out


def keep(ws, qid: str, name: str) -> dict:
    """The candidate as an added-molecule variant of the workspace's network (mepd.web.added.put), so
    Explore's Network picker lists it and the map marks it. Keeping it again refreshes the record
    from the result file."""
    q = _query(ws, qid)
    doc = _load(_result_path(ws, qid, name))
    cand = doc["candidate"]
    old = next((v for v in ws.variants.values() if (v.get("addmol") or {}).get("query") == qid
                and v["addmol"].get("result") == name), None)
    rec = added.put(ws, {"id": addmol.ADDED_ID, "name": cand.get("name"), "smiles": cand.get("smiles")},
                    float(cand["amount"]), doc.get("estimates") or {}, vid=(old or {}).get("id"))
    rec["edit"] = [f"{cand['amount']:g} M of {cand.get('name') or cand.get('smiles')} added"]   # (a list, as the picker reads it)
    rec["addmol"] = {"query": qid, "result": name}
    rec["marks"] = marks(q, doc, _edges(ws, q), _nodes(ws, q))
    return ws.put_variant(rec)


def after_job(manager, job: dict) -> bool:
    """A finished test: the variants kept from its candidates follow their result files. True if any did."""
    ws = manager.ws
    qid = (job.get("params") or {}).get("query")
    names = set((job.get("params") or {}).get("names") or [])
    hit = [v["addmol"]["result"] for v in ws.variants.values()
           if (v.get("addmol") or {}).get("query") == qid and v["addmol"].get("result") in names]
    for name in hit:
        try:
            keep(ws, qid, name)
        except WorkspaceError:
            pass
    return bool(hit)
