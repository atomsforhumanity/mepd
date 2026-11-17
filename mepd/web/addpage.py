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


def start(manager, qid: str, candidates: list, *, steps: Optional[int] = None, verify: Optional[int] = None,
          scan: int = 25, side: bool = False, workers: int = 2) -> dict:
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
                                params={"query": qid, "steps": steps, "verify": verify, "scan": scan, "side": side, "workers": workers,
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


def _scan_steps(entry: dict) -> list:
    """The parent steps one scan stands for (one today; a scan per reaction covers several)."""
    return list(entry.get("steps") or ([entry["step"]] if entry.get("step") else []))


OVER = ("done", "failed", "cut", "interrupted")      # a test that will write no more


def progress(doc: dict) -> dict:
    """Where a test stands: the stages that are running (`stages`: any of them at once; `stage`, the
    first, for a one-line read), how every parent step ended so far (`summary` of the result file), the
    scans (asked, done, those running) and how many reactions were wanted for a scan but not scanned
    (`scan.left`). `open`: the answer can still change, or was never looked for on some steps, so
    "no effect found" is not "no effect"."""
    layers = doc.get("layers") or []
    stages = [{k: lay.get(k) for k in ("name", "done", "total")} for lay in layers if lay.get("status") == "running"]
    scan = doc.get("scan") or {}
    scans = scan.get("steps") or []
    live = [x.get("label") or ", ".join(_scan_steps(x)) for x in scans if x.get("status") in ("mapping", "searching")]
    unfinished = [x for x in scans if x.get("status") not in ("done", "failed")]
    summary = {k: v for k, v in (doc.get("summary") or {}).items() if k != "what"}
    left = len(scan.get("left") or [])
    return {"stage": stages[0] if stages else None, "stages": stages,
            "summary": summary or None,
            "scans": {"asked": len(scans), "done": sum(1 for x in scans if x.get("status") == "done"),
                      "failed": sum(1 for x in scans if x.get("status") == "failed"),
                      "running": live, "left": left, "of": scan.get("reactions")},
            "first": {k: (doc.get("timings") or {}).get(f"first_{k}_s") for k in ("bystander", "verified", "relay")},
            # The target's route settled: no reaction left that could lower it (the rows still to come may move it).
            "settled": doc.get("target_route_settled"),
            "open": bool(left or unfinished or doc.get("status") != "done"
                         or (summary.get("est") or 0) or (summary.get("not_reached") or 0))}


# ---------------------------------------------------------------- a finished test as it was at an earlier time
def _span(doc: dict, name: str) -> Optional[tuple]:
    """(first, last) arrival time of a layer's rows, kept or failed; None when none carries a time."""
    est = doc.get("estimates") or {}
    rows = [e for e in est.get("steps") or [] if e.get("layer") == name] + \
           [e for e in est.get("new_steps") or [] if e.get("layer") == name] + \
           [f for f in doc.get("failed") or [] if f.get("layer") == name]
    ts = [float(x["t_s"]) for x in rows if x.get("t_s") is not None]
    if name == "relay":
        for x in (doc.get("scan") or {}).get("steps") or []:
            if x.get("t_started_s") is not None:
                ts += [float(x["t_started_s"]), float(x["t_started_s"]) + float(x.get("wall_s") or 0.0)]
    return (min(ts), max(ts)) if ts else None


def duration(doc: dict) -> Optional[float]:
    """How long the test ran (s), when its rows carry arrival times: a replay is possible then."""
    spans = [x for x in (_span(doc, n) for n in ("bystander", "verify", "relay")) if x]
    total = (doc.get("timings") or {}).get("total_s")
    return max([float(total or 0.0)] + [b for _, b in spans]) if spans else None


def replay(q: dict, doc: dict, t: float) -> dict:
    """The result file as it stood `t` seconds into its test, built from the final file: the rows that had
    arrived by then (their `t_s`), the kinetics run again on those rows (mepd.addmol.result), each scan as
    far as it had come (its start, its arrivals), the route peak's history up to then, and every stage
    marked by whether its rows had begun or ended (no order of stages is assumed).

    What a final file no longer holds cannot be replayed: a row that a lower one of the same kind replaced
    (the lower one appears at its own time), the estimate of a step that was later verified, and counts
    of work done. `replay.kept` says so in numbers."""
    t = float(t)
    by = lambda rows: [e for e in rows or [] if e.get("t_s") is None or float(e["t_s"]) <= t]   # noqa: E731
    est = {k: by(v) for k, v in (doc.get("estimates") or {}).items()}
    failed = by(doc.get("failed"))
    spans = {n: _span(doc, n) for n in ("bystander", "verify", "relay")}
    layers = []
    for lay in doc.get("layers") or []:
        span = spans.get(lay.get("name"))
        if span is None:
            layers.append(lay)
            continue
        first, last = (0.0, span[1]) if lay["name"] == "bystander" else span
        status = "queued" if t < first else "running" if t < last else lay.get("status")
        kept = sum(1 for k in ("steps", "new_steps") for e in est[k] if e.get("layer") == lay["name"]) + \
            sum(1 for f in failed if f.get("layer") == lay["name"])
        layers.append({**{k: v for k, v in lay.items() if k not in ("done", "wall_s")}, "status": status,
                       "done": lay.get("done") if t >= last else None, "seen": kept,
                       "wall_s": lay.get("wall_s") if t >= last else None})
    out = addmol.result(q, doc.get("candidate") or {}, est, status="running" if t < (duration(doc) or 0.0) else doc.get("status"),
                        layers=layers, failed=failed)
    out["level_labels"] = doc.get("level_labels")
    out["timings"] = {k: v for k, v in (doc.get("timings") or {}).items() if k.startswith("first_") and v is not None and v <= t}
    peak = doc.get("route_peak") or {}
    out["route_peak"] = {**peak, "with": [x for x in peak.get("with") or [] if x["t_s"] <= t]}
    scans = []
    for x in (doc.get("scan") or {}).get("steps") or []:
        start = x.get("t_started_s")
        if start is None or t < start:
            continue
        if t >= start + float(x.get("wall_s") or 0.0):
            scans.append(x)
            continue
        came = [a for a in x.get("arrivals") or [] if a["t_s"] <= t]
        low = lambda part: min((a["free"] for a in came if bool(a.get("takes_part")) == part and a.get("free") is not None), default=None)   # noqa: E731
        scans.append({**{k: v for k, v in x.items() if k not in ("searches", "elsewhere", "consumed", "failed_rows", "answer")},
                      "status": "searching", "wall_s": t - start,
                      "with_molecule": sum(1 for a in came if a.get("takes_part")) if x.get("arrivals") is not None else None,
                      "beside": sum(1 for a in came if not a.get("takes_part")) if x.get("arrivals") is not None else None,
                      "lowest": {k: v for k, v in (("with_molecule", low(True)), ("beside", low(False))) if v is not None}})
    scan = doc.get("scan") or {}
    over = t >= (duration(doc) or 0.0)
    ended = scan.get("ended") or {}
    done_scanning = over or (ended.get("t_s") is not None and t >= float(ended["t_s"]))
    side = scan.get("side_reactions") or {}
    out["scan"] = {**{k: v for k, v in scan.items() if k not in ("steps", "left", "not_scanned", "derivations", "ended",
                                                                 "side_reactions", "kinetic_test")}, "steps": scans,
                   "left": scan.get("left") if done_scanning else [],
                   "not_scanned": scan.get("not_scanned") if done_scanning else None,
                   "ended": ended if done_scanning else None,
                   "side_reactions": side if side and (over or t >= float(side.get("evaluated_t_s") or 0.0)) else None}
    settled = doc.get("target_route_settled")
    out["target_route_settled"] = settled if settled and t >= float(settled["t_s"]) else None
    out["scope"] = doc.get("scope")
    # The answer as the run itself had it then (its read-out at the last change of the route peak): the rows a
    # later, lower row replaced are gone from the final file, so the kinetics run again on it cannot give it.
    seen = [x for x in peak.get("with") or [] if x["t_s"] <= t and x.get("readout")]
    if seen and out.get("readout") and not over:
        then, r = seen[-1], out["readout"]
        tg, a = then["readout"].get("target") or {}, r.get("added") or {}
        total = sum((q["conditions"].get("initial") or {}).values()) or 1.0
        if tg.get("final") is not None:
            r["target"]["with"] = {**r["target"]["with"], "final": tg["final"], "time_to_fraction_s": tg.get("time_to_fraction_s"),
                                   "resolved": tg["final"] > addmol.RESOLUTION * total}
            r["target"]["spread"] = {"with_low": tg.get("low"), "with_high": tg.get("high"), "factor_low": None,
                                     "factor_high": None, "what": "as the run had it then"} if tg.get("low") is not None else None
        r["added"] = {**a, **{k: then["readout"][k] for k in ("fate", "route_role", "in_words", "turnovers") if k in then["readout"]},
                      "returns": then["readout"].get("turnovers", a.get("returns"))}
        r["peak_routes"] = {**(r.get("peak_routes") or {}),
                            "with": [{**(((r.get("peak_routes") or {}).get("with") or [{}])[0]), "peak": then["peak"],
                                      "rests_on": then.get("rests_on")}]}
    # How the steps stood then, from the rows in by then.
    steps = {st["id"] for st in q["steps"] if st.get("geometry")}
    part = {i for e in est.get("new_steps") or [] if e.get("takes_part") for i in (e.get("parent_steps") or [e.get("parent_step")])}
    beside = {e.get("parent_step") for e in est.get("new_steps") or [] if e.get("takes_part") is False} - part
    quick = {e.get("step") for e in est.get("steps") or []} - part - beside
    gone = {f.get("step") for f in failed if f.get("status") == "leads elsewhere"} - part - beside
    out["summary"] = None if over and doc.get("summary") else {
        "steps": (doc.get("summary") or {}).get("steps") or len(steps), "est": len(quick & steps), "verified_beside": len(beside & steps), "takes_part": len(part & steps),
        "leads_elsewhere": len(gone & steps), "failed": 0, "not_reached": 0}
    if out["summary"] is None:
        out["summary"] = doc.get("summary")
    final = doc.get("estimates") or {}
    out["replay"] = {"t_s": t, "of": duration(doc), "over": over,
                     "kept": {"estimates": len(final.get("steps") or []), "verified": len(final.get("new_steps") or [])}}
    return out


_replays: dict = {}


def at_time(fp: Path, q: dict, t: Optional[float]) -> dict:
    """The result file, or (`t`) its replay at that time, kept for the times last asked for."""
    doc = _load(fp)
    if t is None or duration(doc) is None:
        return doc
    key = (str(fp), fp.stat().st_mtime_ns, round(float(t), 1))
    if key not in _replays:
        if len(_replays) > 64:
            _replays.clear()
        _replays[key] = replay(q, doc, float(t))
    return _replays[key]


def line(doc: dict, order=()) -> dict:
    """One candidate in one line: what it does to the target, whether it is returned, where it rests,
    which layers are in. `best`: the step that changes most through the molecule, among the steps that limit
    the target (`order`: the query's order of work) where any of them does. `effect` orders candidates (larger: more of the target): the logarithm of the
    factor where both amounts are above the kinetics' resolution, else how far the lowest route's peak
    came down, kept below every resolved gain."""
    cand, r = doc.get("candidate") or {}, doc.get("readout")
    out = {"name": cand.get("name"), "smiles": cand.get("smiles"), "amount": cand.get("amount"),
           "status": doc.get("status"), "updated": doc.get("updated"),
           "layers": [{k: lay.get(k) for k in ("name", "status", "done", "total", "wall_s", "level", "what", "entries", "seen", "lanes")}
                      for lay in doc.get("layers") or []],
           "failed": len(doc.get("failed") or []), "used": {"est": 0, "verified": 0}, "progress": progress(doc),
           "settled": doc.get("target_route_settled")}
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
    rests_on = ((r.get("peak_routes") or {}).get("with") or [{}])[0].get("rests_on")
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
                       "time_to_parent_amount_s": t.get("time_to_parent_amount_s"), "spread": t.get("spread")},
            "first_order": (r.get("first_order") or {}).get("factor"),
            "peak": peak, "peak_rests_on": rests_on, "on_route": _on_route(r),
            "fate": a.get("fate"), "returns": a.get("returns"), "shares": a.get("shares"),
            "resting_state": a.get("resting_state"),
            "route_role": a.get("route_role"), "in_words": a.get("in_words"),
            "through": len(through),
            "steps": [{**{k: s.get(k) for k in ("step", "label", "direction", "change", "status", "spread")},
                       "limits": s["step"] in set(order)}
                      # (the steps that limit the target first: a far-off step can change by more and matter less)
                      for s in sorted(through, key=lambda s: (s["step"] not in set(order), s["change"]))[:6]],
            "best": best and {**{k: best.get(k) for k in ("step", "label", "direction", "before", "after", "change",
                                                          "status", "spread")}, "limits": bool(limiting),
                              # (its number is a scan's of the whole reaction, found from another of its steps)
                              "via_reaction": (best.get("through") or "").split("|")[0] not in ("", best["step"])},
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


def _reaction_words(text: Optional[str], names: dict) -> Optional[str]:
    """A reaction as the core writes it ("A + B -> C", species ids) in the names the page uses."""
    if not text or "->" not in text:
        return text
    side = lambda part: " + ".join(names.get(x.strip(), x.strip()) for x in part.split(" + "))   # noqa: E731
    left, right = text.split("->", 1)
    return f"{side(left)} -> {side(right)}"


def _species_words(text: str, names: dict, molecule: Optional[str]) -> list:
    """An IRC end as the explorer writes it ("A.B.C", SMILES) in the names the page uses."""
    out = []
    for part in (text or "").split("."):
        hit = addmol.match([{"id": k, "smiles": k} for k in names], part) if part else []
        same = molecule and addmol.match([{"id": "m", "smiles": molecule}], part)
        out.append("the molecule" if same else names.get(hit[0], part) if hit else part)
    return out


def detail(manager, qid: str, name: str, t: Optional[float] = None) -> dict:
    """One candidate's result as the page reads it: the read-out with, for every changed step, the layer
    and level its number is from; the two groups of barrier changes (through the molecule; only
    because amounts moved); what is new; what failed and why. The estimates' raw bindings stay in the
    file."""
    ws = manager.ws
    q = _query(ws, qid)
    fp = _result_path(ws, qid, name)
    whole = _load(fp)
    doc = at_time(fp, q, t)
    r = doc.get("readout")
    steps = {st["id"]: st for st in q["steps"]}
    edges = _edges(ws, q)
    source = _source(doc)
    labels = _labels(q, _names(q, ws))
    out = {"file": name, **line(doc, [o["step"] for o in q["limits"].get("order") or []]), "candidate": {k: v for k, v in (doc.get("candidate") or {}).items()
                                                    if k not in ("coordinates", "pose", "atoms")},
           "failed": [{**{k: f.get(k) for k in ("layer", "status", "why", "step", "species")},
                       "label": labels.get(f.get("step")) or _names(q, ws).get(f.get("species"), f.get("species"))}
                      for f in doc.get("failed") or [] if f.get("status") != "leads elsewhere"],
           "timings": doc.get("timings"), "through_steps": [], "moved_steps": [], "new_steps": [], "new_species": []}
    out["job"] = running(manager, qid).get(name)
    out["duration_s"] = duration(whole)
    out["replay"] = doc.get("replay")
    out["route_peak"] = doc.get("route_peak")
    out["cut"] = doc.get("cut")
    quick = next((lay for lay in doc.get("layers") or [] if lay.get("name") == "bystander"), {})
    out["uncertainty"] = quick.get("uncertainty") if isinstance(quick.get("uncertainty"), dict) else None
    out["level_words"] = (doc.get("level_labels") or {}).get((q.get("level") or {}).get("key")) or (q.get("level") or {}).get("label")
    out["solvent"] = "solvation" in ((q.get("level") or {}).get("profile") or "")
    if out["status"] == "running" and out["job"] is None and not doc.get("replay"):   # (cut short, as in `view`)
        out["status"] = "interrupted"
        out["layers"] = [{**lay, "status": "stopped early"} if lay.get("status") == "running" else lay
                         for lay in out["layers"]]
    if not r:
        return out

    level_labels = doc.get("level_labels") or {}
    rank = {o["step"]: k for k, o in enumerate(q["limits"].get("order") or [])}
    names_ = _names(q, ws)
    # Species whose free energy the parent's fit fixes only together with others: a step that makes or uses one
    # (its net amount changes) rests on a share of a sum, not on a free energy of its own.
    together = {sp["id"] for sp in q["species"] if sp.get("g_fixed_only_together")}

    def alone(step: dict) -> list:
        c = {}
        for x in step.get("reactants") or []:
            c[x] = c.get(x, 0) - 1
        for x in step.get("products") or []:
            c[x] = c.get(x, 0) + 1
        return [names_.get(x, x) for x, n in c.items() if n and x in together]

    def row(s: dict) -> dict:
        e = source.get(s.get("through")) or {}
        st = steps.get(s["step"]) or {}
        # What kind of number: a quick estimate; a verified TS with the molecule beside the step; one with it taking part.
        kind = ("takes part" if e.get("takes_part") else "beside") if e.get("status") == "verified" else "estimate"
        files = None
        if e.get("ts_file"):
            files = "there" if _file(fp, e["ts_file"], look=True) else "gone"
        via = (s.get("through") or "").split("|")[0]
        return {**s, "fixed_together": alone(st), "via_step": via if e.get("layer") == "relay" and via and via != s["step"] else None,
                "via_label": labels.get(via) if e.get("layer") == "relay" and via != s["step"] else None,
                "layer": e.get("layer"), "level": level_labels.get(e.get("level"), e.get("level")),
                "source": e.get("source"), "kind": kind, "poses_verified": e.get("poses_verified"),
                "returned": e.get("returned"), "t_s": e.get("t_s"), "rank": rank.get(s["step"]),
                "via": e.get("label") if e.get("id") and e.get("layer") not in ("verify", "relay") else None,
                "geometry": bool(st.get("geometry")), "files": files,
                "with_geometry": bool(e.get("complexes") or e.get("geometry") or files == "there"),
                "edge": edges.get(s["step"])}

    changed = [s for s in r["steps"] if s.get("change") is not None]
    # (in the read-out's own order: the page orders them, and keeps that order still while a test runs)
    out["through_steps"] = [row(s) for s in changed if _through(s)]
    out["moved_steps"] = sorted(({**s, "edge": edges.get(s["step"])} for s in changed if _moved(s)),
                                key=lambda s: -abs(s["change"]))
    out["unchanged"] = len(changed) - len(out["through_steps"]) - len(out["moved_steps"])
    _relabel([out.get("best"), *out["through_steps"], *out["moved_steps"]], labels)
    # Steps with the molecule that are no version of a parent step, and versions that came with the candidate
    # (the verified versions of parent steps are the rows of the table above).
    out["new_steps"] = [{k: e.get(k) for k in ("id", "label", "parent_step", "status", "layer", "source")}
                        | {"barrier": e.get("barrier"), "with_geometry": bool(e.get("geometry")),
                           "level": level_labels.get(e.get("level"), e.get("level"))}
                        for e in (doc.get("estimates") or {}).get("new_steps") or []
                        if e.get("layer") not in ("verify", "relay") or e.get("parent_step") not in steps]
    # The species that hold the molecule, those that ever hold material first.
    out["new_species"] = sorted(r.get("new_species") or [], key=lambda s: -(s.get("max") or 0.0))[:40]
    out["new_species_total"] = len(r.get("new_species") or [])
    out["routes"] = (r.get("routes") or {}).get("with") or []
    out["peak_routes"] = r.get("peak_routes")
    out["notes"] = r.get("notes") or []
    out["first_order_detail"] = r.get("first_order")
    out["route_peak_with_added"] = (r.get("added") or {}).get("route_peak_with_added")
    out["mixed_levels"] = r.get("mixed_levels")
    scan = doc.get("scan") or {}
    out["scans"] = [{**{k: x.get(k) for k in ("status", "why", "why_failed", "searches", "with_molecule", "beside", "lowest",
                                               "elsewhere", "failed_rows", "wall_s", "t_started_s", "lane", "twin",
                                               "first_with_molecule_s", "stopped", "answer", "cores",
                                               "route_peak_now", "route_peak_if_no_barrier")},
                     "steps": _scan_steps(x), "labels": [labels.get(i, i) for i in _scan_steps(x)],
                     "consumed": len(x.get("consumed") or [])} for x in scan.get("steps") or []]
    grouped: dict = {}
    for x in scan.get("left") or []:
        grouped.setdefault(x.get("why") or "not scanned", []).append(labels.get(x.get("step"), x.get("reaction") or x.get("step")))
    out["scan_left"] = [{"why": why, "n": len(names), "labels": names[:6]} for why, names in grouped.items()]
    out["fixed_together_note"] = next((n for n in q.get("notes") or [] if "g_fixed_only_together" in n), None)
    out["scan_rule"], out["not_scanned"] = scan.get("rule"), scan.get("not_scanned")
    out["scan_reactions"] = scan.get("reactions")
    names = names_
    out["scan_ended"] = scan.get("ended")
    out["settled"] = doc.get("target_route_settled")
    out["scope"] = doc.get("scope")
    side = scan.get("side_reactions")
    out["side_reactions"] = side and {**{k: side.get(k) for k in ("asked_for", "ask_with", "what", "change_worth_listing")},
                                      "listed": [{**x, "label": _reaction_words(x.get("reaction"), names),
                                                  "species_name": names.get(x.get("species"), x.get("species"))}
                                                 for x in side.get("listed") or []]}
    # A takes-part row that lies above the same reaction with the molecule beside it is no catalysis.
    out["takes_part_above"] = [{"reaction": _reaction_words(x.get("reaction"), names), "takes_part": x["lowest"]["with_molecule"],
                                "beside": x["lowest"]["beside"]} for x in scan.get("steps") or []
                               if (x.get("lowest") or {}).get("with_molecule") is not None and (x.get("lowest") or {}).get("beside") is not None
                               and x["lowest"]["with_molecule"] > x["lowest"]["beside"]]
    # Where a verification led when it did not join the step's own ends: each pose's two IRC ends, in names.
    smiles = (doc.get("candidate") or {}).get("smiles")
    out["elsewhere"] = [{"step": f.get("step"), "label": labels.get(f.get("step"), f.get("step")), "t_s": f.get("t_s"),
                         "poses": [{"pose": x.get("pose"), "ends": [_species_words(e, names, smiles) for e in x.get("ends") or []]}
                                   for x in f.get("ends") or []]}
                        for f in doc.get("failed") or [] if f.get("status") == "leads elsewhere"]
    return out


def remove(manager, qid: str, name: str) -> None:
    fp = _result_path(manager.ws, qid, name)
    if name in running(manager, qid):
        raise WorkspaceError("this candidate is still being tested: cancel its calculation first")
    fp.unlink()
    _cache.pop(str(fp), None)


# ---------------------------------------------------------------- geometry for the 3D viewer
def _file(result: Path, recorded: str, *, look: bool = False) -> Optional[Path]:
    """A file a row names (`ts_file`, `irc_file`): as recorded if absolute, else relative to the result
    file's folder; where it is not there but `files.tgz` beside the result holds it (a pod run), taken out
    of the archive into its place. `look`: only say whether it could be had (the archive is not read: a
    relative path with an archive beside counts)."""
    fp = Path(recorded)
    if not fp.is_absolute():
        fp = result.parent / fp
    if fp.is_file():
        return fp
    own = result.with_name(result.stem + ".files.tgz")      # (several pod outputs in one folder: one archive each)
    tgz = own if own.is_file() else result.parent / "files.tgz"
    if Path(recorded).is_absolute() or not tgz.is_file():
        return None
    if look:
        return tgz
    import tarfile

    want = {recorded.lstrip("./"), "./" + recorded.lstrip("./")}
    with tarfile.open(tgz) as tar:
        for member in tar:
            if member.name in want and member.isfile():
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_bytes(tar.extractfile(member).read())
                return fp
    return None


def _xyz(symbols, coords, comment: str) -> str:
    rows = "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, coords))
    return f"{len(symbols)}\n{comment}\n{rows}\n"


def geometry(ws, qid: str, step: str, name: Optional[str] = None, through: Optional[str] = None) -> dict:  # noqa: C901
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
    if est.get("ts_file"):
        # A verified TS with the molecule: its own file, with the two ends of its IRC where that is there too.
        result = _result_path(ws, qid, name)
        ts = _file(result, est["ts_file"])
        irc = _file(result, est["irc_file"]) if est.get("irc_file") else None
        if ts is None:
            if g is None:
                raise WorkspaceError(f"this transition state was written to {est['ts_file']}, which is not there any more")
            return {"label": label, "ts": 1, "with_molecule": False, "missing": est["ts_file"],
                    "frames": [_xyz(g["atoms"], g[k], f"{label} · {w}") for k, _, w in words]}
        (top,) = added._xyz_frames(ts.read_text())[:1]
        path = added._xyz_frames(irc.read_text()) if irc is not None else []
        frames = [_xyz(*top, f"{label} · TS with the molecule")]
        if len(path) >= 2:
            frames = [_xyz(*path[0], f"{label} · one end of its IRC"), frames[0], _xyz(*path[-1], f"{label} · the other end")]
        return {"label": label, "frames": frames, "ts": 1 if len(frames) == 3 else 0, "with_molecule": True,
                "ends": "irc"}
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
    """What the network map marks for a candidate: per parent step that matters for the target and whose
    lowest version goes through the molecule (by its edge) the change of its apparent barrier and whether
    that is an estimate. A step matters when it is one of those that limit the target (the query's order
    of work) or lies on the easiest route to the target with the molecule; once every step is verified
    with the molecule beside it, most steps "go through" it, and marking them all marks nothing
    (`more`: how many others do). Per parent
    species the complex with the molecule that holds material; what has no place on the parent's map (new
    species, steps without a parent step); and how many steps moved only because amounts moved (not
    marked: their own transition states are unchanged)."""
    r = doc.get("readout") or {}
    steps = {st["id"]: st for st in q["steps"]}
    names = _names(q)
    out = {"edges": {}, "species": {}, "new": [], "moved": 0, "more": 0, "status": doc.get("status"),
           "updated": doc.get("updated")}
    route = {x.get("step") for x in (((r.get("peak_routes") or {}).get("with") or [{}])[0].get("steps") or [])}
    key = {o["step"] for o in q["limits"].get("order") or []}
    for s in r.get("steps") or []:
        edge = edges.get(s["step"])
        if edge is None or s.get("change") is None:
            continue
        if _through(s) and not (s["step"] in key or s["step"] in route or s.get("through") in route):
            out["more"] += 1
        elif _through(s):
            mark = {"change": round(s["change"], 2), "status": s.get("status"), "spread": s.get("spread")}
            # (two steps on one edge: the larger lowering is drawn)
            if edge not in out["edges"] or mark["change"] < out["edges"][edge]["change"]:
                out["edges"][edge] = mark
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
