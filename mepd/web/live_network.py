"""The live network view's data: an explorer run (`mepd discovery expand`) as
a timeline the page can play, growing, correcting itself and moving material.

Each refresh of the explorer's live kinetics (live/kinetics.json, schema of
the explorer team's step_list/conc/energy_source fields) becomes one compact
frame, holding only what can matter on screen:
- the species that ever hold material (their energy, peak and final amount,
  whether their energy is DFT-corrected);
- the open steps between them (barriers, source, IRC check, net flux);
- counts of everything else.

Correction events (barrier moved, step replaced, a species re-referenced to
a lower conformer, a step closed or reopened) come from live/events.jsonl, and
the selectivity against experiment from live/selectivity.json when a
benchmark annotator writes one.

A running job is recorded as it goes (JobManager appends each new kinetics
refresh to live/network_frames.jsonl); a finished replay folder, or an
example packed with `pack-replay`, is read the same way.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Optional

FRAMES = "network_frames.jsonl"
SELECTIVITY_FRAMES = "selectivity_frames.jsonl"
MAX_SPECIES = 60          # shown per frame, by peak amount (the seed always); size shows relevance
MAX_STEPS = 90            # open steps among them, by |flux|
MIN_PEAK = 0.0            # every found species may show; the cap keeps the most relevant


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def _sig(x):
    return None if x is None else float(f"{float(x):.3g}")


def compact(kin: dict) -> Optional[dict]:
    """One live kinetics refresh -> one frame (None if it is not the new schema)."""
    species = kin.get("species") or []
    steps = kin.get("step_list")
    if steps is None or not species:
        return None
    t_start = kin.get("t_start") or kin.get("t_wall") or 0.0
    # Mirror images the kinetics add (enantiomers not found yet) carry real material: fold them in.
    extra = {}
    for m in kin.get("mirror_species") or []:
        twin = str(m.get("mirror_of"))
        e = extra.setdefault(twin, [0.0, 0.0])
        e[0] += float(m.get("max_concentration") or 0)
        e[1] += float(m.get("final") or 0)
    rows = []
    for s in species:
        sid = str(s.get("id", s.get("index")))
        peak = float(s.get("max_concentration") or 0) + extra.get(sid, [0, 0])[0]
        fin = float(s.get("final") or 0) + extra.get(sid, [0, 0])[1]
        rows.append((sid, s, peak, fin))
    seed = rows[0][0]
    shown = [r for r in rows if r[0] == seed or r[2] >= MIN_PEAK]
    shown.sort(key=lambda r: (r[0] != seed, -r[2]))
    shown = shown[:MAX_SPECIES]
    keep = {r[0] for r in shown}
    sp = [[sid, _r(s.get("G", s.get("rel_kcal"))), _sig(peak), _sig(fin), 1 if s.get("populated") else 0,
           1 if s.get("energy_source") == "dft" else 0] for sid, s, peak, fin in shown]
    smiles = {sid: s.get("smiles", "") for sid, s, _, _ in shown}
    open_steps = [st for st in steps if st.get("state") == "open" and st.get("dG_fwd") is not None
                  and str(st.get("from")) in keep and str(st.get("to")) in keep and str(st.get("from")) != str(st.get("to"))]
    open_steps.sort(key=lambda st: -abs(st.get("flux") or 0))
    st_rows = [[str(st["id"]), str(st["from"]), str(st["to"]), _r(st.get("dG_fwd")), _r(st.get("dG_rev")),
                {"dft": "d", "borrowed": "b"}.get(st.get("energy_source"), "g"),
                {True: 1, False: 0, "unchecked": -1}.get(st.get("irc_ok"), None), _sig(st.get("flux"))]
               for st in open_steps[:MAX_STEPS]]
    # Steps no longer in the network (closed out of reach, replaced, dropped)
    # among the shown species: drawn faint, so corrections stay visible.
    gone = [[str(st["id"]), str(st["from"]), str(st["to"]), st.get("state")] for st in steps
            if st.get("state") in ("closed", "replaced", "dropped")
            and str(st.get("from")) in keep and str(st.get("to")) in keep and str(st.get("from")) != str(st.get("to"))][:150]
    states = {}
    for st in steps:
        states[st.get("state", "open")] = states.get(st.get("state", "open"), 0) + 1
    basis = kin.get("basis") or ""
    return {
        "t": _r((kin.get("t_wall") or t_start) - t_start, 1),
        "steered": "single points" in basis,
        "basis": basis.split(";")[0],
        "species": sp, "steps": st_rows, "gone": gone, "smiles": smiles,
        "counts": {"species": len(species), "steps": len(steps), **states},
    }


def compact_legacy(kin: dict, known_steps: list) -> Optional[dict]:
    """A refresh from explorer runs before the step_list schema: species by
    index, and the steps the run reported (events.jsonl "step" lines) once
    both their species are found. These carry no per-step flux, so their
    edges show barriers only."""
    species = kin.get("species") or []
    if not species or "index" not in species[0]:
        return None
    kin = {**kin, "species": [{**s, "id": s["index"]} for s in species]}
    have = {str(s["index"]) for s in species}
    # One edge per pair of species (the lowest of its channels, an IRC-checked one before an unchecked one),
    # lowest barriers first.
    best = {}
    for st in known_steps:
        a, b = str(st["a"]), str(st["b"])
        if a not in have or b not in have or a == b:
            continue
        fwd, rev = (st["dG"][0], st["dG"][1]) if a < b else (st["dG"][1], st["dG"][0])
        key = (min(a, b), max(a, b))
        low = min(x for x in (fwd, rev) if x is not None) if (fwd, rev) != (None, None) else None
        # A TS below one of its ends (ts_below_ends) is no barrier of that pair: after the real ones, else
        # the lowest (most negative) of them fill the frame's step cap and push every real route out.
        rank = (bool(st.get("unchecked")), low is None, ts_below_ends((fwd, rev)), low or 0)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, {"id": st["label"], "from": key[0], "to": key[1], "dG_fwd": fwd, "dG_rev": rev,
                                "state": "open", "energy_source": "gxtb",
                                **({"irc_ok": "unchecked"} if st.get("unchecked") else {})})
    kin["step_list"] = [v for _, v in sorted(best.values(), key=lambda x: x[0][1:])]
    f = compact(kin)
    if f is not None:
        f["legacy"] = True
    return f


def _legacy_steps(lines: Iterable[str]) -> list:
    out = []
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "step" and ev.get("barrier_kcal") and None not in (ev.get("a"), ev.get("b")):
            bk = ev["barrier_kcal"]
            out.append({"label": ev.get("label") or f"{ev['a']}_{ev['b']}", "a": ev["a"], "b": ev["b"],
                        "dG": [bk[0], bk[1] if len(bk) > 1 else None]})
    return out


def _changes(lines: Iterable[str], t_start: float) -> list:
    """The explorer's correction events, compact: [t, kind, ids, detail]."""
    out = []
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") != "change":
            continue
        kind, old, new = ev.get("kind"), ev.get("old") or {}, ev.get("new") or {}
        if kind == "barrier_moved":
            detail = {"from": _r(old.get("dG_fwd")), "to": _r(new.get("dG_fwd")), "src": new.get("energy_source"),
                      "why": ev.get("why", "")}
        elif kind == "species_rereferenced":
            detail = {"from": _r(old.get("E_rel_kcal")), "to": _r(new.get("E_rel_kcal"))}
        elif kind == "step_replaced":
            detail = {"by": new.get("replaced_by") or []}
        else:
            detail = {}
        out.append([_r((ev.get("t_wall") or t_start) - t_start, 1), kind, [str(i) for i in ev.get("ids") or []], detail])
    return out


def _selectivity(data: Optional[dict], t_start: float, entries: Optional[list] = None) -> Optional[dict]:
    if not data and not entries:
        return None
    data = data or {}
    rows = entries if entries is not None else data.get("entries") or []
    return {
        "system": data.get("system"),
        "reference": data.get("reference"),
        "products": sorted({k for e in rows for k in (e.get("products") or {})}),
        # Which species each product is (the page names those nodes after it).
        "species": {k: [str(i) for i in (v.get("species") or [])]
                    for e in rows[-1:] for k, v in (e.get("products") or {}).items()},
        "series": [[_r((e.get("t_wall") or t_start) - t_start, 1), _sig(e.get("ratio")), e.get("settled_at") is not None,
                    {k: _sig(v.get("amount")) for k, v in (e.get("products") or {}).items()}] for e in rows],
    }


BELOW_KCAL = -0.1   # a barrier under this: the TS lies below that end (the edge cards' negative-barrier rule)


def ts_below_ends(barriers) -> bool:
    """A step whose TS lies below one of its own ends (a barrier, either way, under BELOW_KCAL): not that
    reaction's TS (unchecked; a separate question why the run found it)."""
    return any(b is not None and b < BELOW_KCAL for b in list(barriers or [])[:2])


def _unfit(frames: list) -> list:
    """In place: each frame's steps whose TS lies below one of their ends (ts_below_ends) without their
    barriers, and marked (a 9th entry, 1): the page draws them as a TS that doesn't fit its ends."""
    for f in frames:
        f["steps"] = [s if not ts_below_ends(s[3:5]) else [*s[:3], None, None, *s[5:7], s[7] if len(s) > 7 else None, 1]
                      for s in f.get("steps") or []]
    return frames


def timeline(frames: list, events: list, selectivity: Optional[dict], *, source: dict) -> dict:
    """Frames share one smiles table; the end state is the last steered frame
    (a trailing unsteered refresh is not the answer). A step whose TS lies below one of its ends draws
    without a barrier (_unfit)."""
    _unfit(frames)
    smiles = {}
    for f in frames:
        smiles.update({k: v for k, v in f.pop("smiles", {}).items() if v or k not in smiles})
    # A run without DFT steering has nothing to wait for: none of its frames is "pending".
    if frames and not any(f["steered"] for f in frames):
        for f in frames:
            f["steered"] = True
    final = max((i for i, f in enumerate(frames) if f["steered"]), default=len(frames) - 1)
    return {"source": source, "frames": frames, "smiles": smiles, "names": names_of(smiles), "events": events,
            "selectivity": selectivity, "final": final}


def drawable(smiles: Optional[str]) -> bool:
    """A SMILES a depiction can be made from: not empty, not the reactor's
    placeholder for a species no Lewis structure fits ("C7H17OP (unusual bonding #df15)")."""
    return bool(smiles) and not str(smiles).startswith("?") and "unusual bonding" not in str(smiles)


def names_of(smiles: dict, formulas: Optional[dict] = None) -> dict:
    """Everyday names where every molecule of a mixture has one ("water + CO");
    for a species no Lewis structure fits, its formula ("C7H17OP · unusual bonding")."""
    from mepd.web.chem import common_name

    out = {}
    for sid, smi in smiles.items():
        if not drawable(smi):
            f = (formulas or {}).get(sid) or (str(smi).split(" (unusual")[0] if smi else "")
            if f:
                out[sid] = f"{f} · unusual bonding"
            continue
        parts = [common_name(x) for x in (smi or "").split(".")] if smi else []
        if parts and all(parts):
            out[sid] = " + ".join(parts)
    return out


def from_snapshots(snap_dir: Path, selectivity_file: Optional[Path] = None) -> dict:
    """A replay folder: kinetics_<mtime_ns>.json versions + events.jsonl."""
    snap_dir = Path(snap_dir)
    frames, t_start = [], None
    ev_fp = snap_dir / "events.jsonl"
    ev_lines = ev_fp.read_text().splitlines() if ev_fp.exists() else []
    legacy_steps = None
    snaps = sorted(snap_dir.glob("kinetics_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    t0_ns = int(snaps[0].stem.split("_")[-1]) if snaps else 0
    for fp in snaps:
        try:
            d = json.loads(fp.read_text())
        except ValueError:
            continue
        kin = d.get("data", d)
        t_start = t_start or kin.get("t_start")
        f = compact(kin)
        if f is None:
            if legacy_steps is None:
                legacy_steps = _legacy_steps(ev_lines)
            f = compact_legacy(kin, legacy_steps)
            if f is not None:    # no t_wall in these: the snapshot's own time
                f["t"] = _r((int(fp.stem.split("_")[-1]) - t0_ns) / 1e9, 1)
        if f is not None:
            frames.append(f)
    events = _changes(ev_lines, t_start or 0.0)
    sel = None
    if selectivity_file and Path(selectivity_file).exists():
        d = json.loads(Path(selectivity_file).read_text())
        sel = _selectivity(d.get("data", d), t_start or 0.0)
    return timeline(frames, events, sel, source={"kind": "replay", "path": snap_dir.name})


PERCEIVED = "perceived_smiles.json"


def _events(job_dir: Path) -> list:
    fp = Path(job_dir) / "live" / "events.jsonl"
    return fp.read_text().splitlines() if fp.exists() else []


def _perceived(job_dir: Path, lines: list) -> dict:
    """SMILES the explorer left empty (radicals, odd valences), perceived from
    each species' xyz in events.jsonl (the seed from inputs/seed.xyz); cached."""
    job_dir = Path(job_dir)
    fp = job_dir / "live" / PERCEIVED
    try:
        known = json.loads(fp.read_text()) if fp.exists() else {}
    except ValueError:
        known = {}
    todo = {}
    seed = job_dir / "inputs" / "seed.xyz"
    if "0" not in known and seed.exists():
        todo["0"] = seed.read_text()
    for line in lines:
        if '"species"' not in line[:40]:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        sid = str(ev.get("index"))
        if ev.get("event") == "species" and not ev.get("smiles") and ev.get("xyz") and sid not in known:
            todo[sid] = ev["xyz"]
    if not todo:
        return known
    from qcdata import Structure
    from mepd.web.chem import perceive_smiles

    for sid, xyz in todo.items():
        try:
            known[sid] = perceive_smiles(Structure.from_xyz(xyz)) or ""
        except Exception:
            known[sid] = ""
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(known))
    except OSError:
        pass
    return known


def fill_species_smiles(job_dir: Path, kin: dict) -> None:
    """In place: a kinetics refresh's empty species SMILES, perceived (the job page's tables)."""
    species = kin.get("species") or []
    if not any(not s.get("smiles") for s in species):
        return
    perceived = _perceived(job_dir, _events(job_dir))
    for s in species:
        if not s.get("smiles"):
            s["smiles"] = perceived.get(str(s.get("id", s.get("index"))), "")


def _fill_smiles(f: dict, perceived: dict) -> dict:
    for sid, smi in list(f.get("smiles", {}).items()):
        if not smi and perceived.get(sid):
            f["smiles"][sid] = perceived[sid]
    return f


def _frame_of(job_dir: Path, kin: dict, lines: list) -> Optional[dict]:
    """One refresh as a frame, in either schema (the older one takes its steps from events.jsonl)."""
    f = compact(kin)
    if f is None:
        f = compact_legacy(kin, _legacy_steps(lines))
    return _fill_smiles(f, _perceived(job_dir, lines)) if f is not None else None


def record(job_dir: Path, kin: dict, t0: Optional[float] = None) -> bool:
    """Append one live kinetics refresh of a running job to its frames (t from
    the refresh's own clock, else the job's start `t0`)."""
    import time

    lines = _events(job_dir)
    f = _frame_of(job_dir, kin, lines)
    if f is None:
        return False
    t_start = kin.get("t_start")
    if not kin.get("t_wall") and t0:
        f["t"], t_start = _r(time.time() - t0, 1), t0
    live = Path(job_dir) / "live"
    live.mkdir(parents=True, exist_ok=True)
    with open(live / FRAMES, "a") as fh:
        fh.write(json.dumps({**f, "t_start": t_start}, separators=(",", ":")) + "\n")
    return True


def _summary_steps(job_dir: Path) -> Optional[list]:
    """A finished (or stopped) run's own steps, each under the species its IRC really
    joins: what its live step events (reported under the pair searched, before the
    IRC check) turned out to be. None while it runs."""
    try:
        out = Path(json.loads((Path(job_dir) / "job.json").read_text()).get("output_dir") or "")
        steps = json.loads((out / "summary.json").read_text()).get("steps")
    except (OSError, ValueError):
        return None
    if steps is None:
        return None
    # A step the run had no time to IRC-check (files.verified False) stays in the network, left out of
    # its final kinetics (files.left_out_of_final_kinetics): drawn as unconfirmed.
    return [{"label": st["label"], "a": st["a"], "b": st["b"],
             "dG": (list(st.get("barrier_kcal") or []) + [None, None])[:2],
             "unchecked": (st.get("files") or {}).get("verified") is False,
             "left_out": (st.get("files") or {}).get("left_out_of_final_kinetics")} for st in steps if st.get("label")]


def _growing(job_dir: Path, lines: list, t0: Optional[float], steps: Optional[list] = None) -> Optional[dict]:
    """The network as events.jsonl has it now, on top of the last kinetics
    refresh: species and steps found since then show at once (no amounts yet)."""
    import time

    try:
        raw = json.loads((Path(job_dir) / "live" / "kinetics.json").read_text())
        kin = dict(raw.get("data", raw))
    except (OSError, ValueError):
        kin = {"basis": "", "species": []}
    if kin.get("step_list") is not None:
        return None     # the new schema refreshes itself
    species = list(kin.get("species") or [])
    have = {str(s.get("index")) for s in species}
    if "0" not in have:
        species.insert(0, {"index": 0, "smiles": "", "rel_kcal": 0.0, "max_concentration": 1.0, "final": 1.0,
                           "populated": True})
        have.add("0")
    for line in lines:
        if '"species"' not in line[:40]:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        sid = str(ev.get("index"))
        if ev.get("event") == "species" and sid not in have:
            have.add(sid)
            species.append({"index": ev.get("index"), "smiles": ev.get("smiles") or "", "rel_kcal": ev.get("rel_energy_kcal"),
                            "max_concentration": 0.0, "final": 0.0, "populated": False})
    kin["species"] = species
    f = compact_legacy(kin, steps if steps is not None else _legacy_steps(lines))
    if f is None:
        return None
    f["growing"] = True
    f["steered"] = "single points" in (kin.get("basis") or "")
    f["t"] = _r(time.time() - t0, 1) if t0 else 0.0
    return _fill_smiles(f, _perceived(job_dir, lines))


def record_selectivity(job_dir: Path, data: dict) -> None:
    """The annotator's selectivity.json is rewritten on each refresh: keep each version's latest entry."""
    entries = (data or {}).get("entries") or []
    if not entries:
        return
    head = {k: v for k, v in data.items() if k != "entries"}
    with open(Path(job_dir) / "live" / SELECTIVITY_FRAMES, "a") as fh:
        fh.write(json.dumps({**head, "entry": entries[-1]}, separators=(",", ":")) + "\n")


def species_xyz(job_dir: Path, sid: str) -> Optional[str]:
    """A species' geometry: its final one (output/species), else as the run first
    reported it (live/events.jsonl), the seed's from the job's input."""
    job_dir = Path(job_dir)
    try:
        out = Path(json.loads((job_dir / "job.json").read_text()).get("output_dir") or job_dir / "output")
    except (OSError, ValueError):
        out = job_dir / "output"
    fp = out / "species" / f"species_{sid}.xyz"
    if fp.exists():
        return fp.read_text()
    if str(sid) == "0" and (job_dir / "inputs" / "seed.xyz").exists():
        return (job_dir / "inputs" / "seed.xyz").read_text()
    for line in _events(job_dir):
        if '"species"' in line[:40] and f'"index": {sid},' in line:
            try:
                return json.loads(line).get("xyz")
            except ValueError:
                return None
    return None


def species_formula(job_dir: Path, sid: str) -> Optional[str]:
    from mepd.web.depict_graph import formula

    xyz = species_xyz(job_dir, sid)
    return formula(xyz) if xyz else None


def species_graph_svg(job_dir: Path, sid: str) -> Optional[str]:
    """A species drawn from its geometry (mepd.web.depict_graph), cached in live/graphs/: of several
    molecules, those that differ from the run's start (species 0)."""
    from mepd.web.depict_graph import depict_xyz_svg

    cache = Path(job_dir) / "live" / "graphs" / f"{sid}.changed.svg"
    if cache.exists():
        return cache.read_text()
    xyz = species_xyz(job_dir, sid)
    svg = depict_xyz_svg(xyz, start=species_xyz(job_dir, "0") if str(sid) != "0" else None) if xyz else None
    if svg:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(svg)
    return svg


def _md_frame(run_dir: Path, index: int) -> Optional[str]:
    """Frame `index` of a reactor run's MD (md/segment_*.xyz in order)."""
    k = 0
    for fp in sorted((run_dir / "md").glob("segment_*.xyz")):
        lines = fp.read_text().splitlines()
        i = 0
        while i < len(lines):
            head = lines[i].strip()
            if not head.isdigit():
                i += 1
                continue
            n = int(head)
            if k == index:
                return "\n".join(lines[i:i + n + 2])
            k += 1
            i += n + 2
    return None


def reactor_products(job_dir: Path, limit: int = 12) -> Optional[dict]:
    """What the reactor's MD has made so far, before the run optimizes any of it:
    each distinct product side of its events, how often, and where to draw it."""
    try:
        out = Path(json.loads((Path(job_dir) / "job.json").read_text()).get("output_dir") or "")
    except (OSError, ValueError):
        return None
    runs = sorted((out / "reactor").glob("run_*/live_events.json"))
    if not runs:
        return None
    seen: dict = {}
    n_events = 0
    loaded = []
    for fp in runs:
        try:
            loaded.append((fp, json.loads(fp.read_text()).get("events") or []))
        except (OSError, ValueError):
            continue
    # The start itself (what the runs' first events begin from) is no product.
    starts = {(ev.get("label") or "").split(" -> ", 1)[0] for _, evs in loaded for ev in evs[:1]}
    for fp, evs in loaded:
        for ev in evs:
            label = ev.get("label") or ""
            if " -> " not in label or ev.get("direction") == "backward":
                continue
            prod = label.split(" -> ", 1)[1]
            if prod in starts:
                continue
            n_events += 1
            if prod not in seen:
                seen[prod] = {"products": prod, "count": 0, "run": fp.parent.name, "event": ev.get("event"),
                              "unusual": "unusual bonding" in prod}
            seen[prod]["count"] += 1
    items = sorted(seen.values(), key=lambda x: -x["count"])
    return {"runs": len(runs), "events": n_events, "distinct": len(items), "products": items[:limit]}


def reactor_product_svg(job_dir: Path, run: str, event: int) -> Optional[str]:
    """One reactor event's atoms at its product frame, drawn from the geometry; cached."""
    from mepd.web.depict_graph import depict_xyz_svg

    if "/" in run or run.startswith("."):
        return None
    cache = Path(job_dir) / "live" / "graphs" / f"reactor_{run}_{int(event)}.svg"
    if cache.exists():
        return cache.read_text()
    try:
        out = Path(json.loads((Path(job_dir) / "job.json").read_text()).get("output_dir") or "")
        run_dir = out / "reactor" / run
        ev = next(e for e in json.loads((run_dir / "live_events.json").read_text()).get("events") or []
                  if e.get("event") == int(event))
    except (OSError, ValueError, StopIteration):
        return None
    frame = _md_frame(run_dir, int(ev.get("product_frame") or 0))
    if frame is None:
        return None
    lines = frame.splitlines()
    atoms = [lines[2 + a] for a in ev.get("atoms") or range(len(lines) - 2) if 2 + a < len(lines)]
    svg = depict_xyz_svg(f"{len(atoms)}\n\n" + "\n".join(atoms), charge=int(ev.get("charge") or 0))
    if svg:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(svg)
    return svg


def propose_phase(job_dir: Path, species: int) -> Optional[str]:
    """What a running propose job (the explorer's live/propose_<species>.json) is doing: its reactor's MD
    (trajectories started since the job began, in <output>/reactor/run_*, and those done), minimizing its
    products, or their Hessian checks."""
    try:
        d = json.loads((Path(job_dir) / "live" / f"propose_{int(species)}.json").read_text())
        d = d.get("data", d)
    except (OSError, ValueError):
        return None
    phase, total = d.get("phase"), d.get("total")
    if phase == "generate":
        runs = []
        try:
            out = Path(json.loads((Path(job_dir) / "job.json").read_text()).get("output_dir") or "")
        except (OSError, ValueError):
            out = None
        if out and d.get("runs_from") is not None:
            runs = [r for r in (out / "reactor").glob("run_*")
                    if r.name[4:].isdigit() and int(r.name[4:]) >= int(d["runs_from"])]
        if not runs:
            return "generating proposals"
        done = sum((r / "md" / "done").exists() for r in runs)
        return f"MD: {done}/{len(runs)} trajector{'y' if len(runs) == 1 else 'ies'} done"
    if phase == "minimize":
        return f"minimizing {total} product{'' if total == 1 else 's'}"
    if phase == "hessian":
        return f"Hessian checks {d.get('done', 0)}/{total}"
    return None


def activity(job_dir: Path) -> dict:
    """What the explorer is doing now (its queue, each propose job with its phase) and the simulations
    under the network (one live stream each: proposed reactions, path searches)."""
    live = Path(job_dir) / "live"
    out = {"queue": None, "sims": []}
    try:
        q = json.loads((live / "queue.json").read_text())
        out["queue"] = q.get("data", q)
        for r in out["queue"].get("running") or []:
            what = str(r.get("what") or "")
            if r.get("kind") == "propose" and what.startswith("species "):
                k = what[len("species "):].split(" ")[0]
                r["phase"] = propose_phase(job_dir, int(k)) if k.isdigit() else None
    except (OSError, ValueError, AttributeError):
        pass
    sims = []
    for fp in sorted(live.glob("rxn*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        try:
            d = json.loads(fp.read_text())
        except (OSError, ValueError):
            continue
        plot = d.get("plot") or {}
        sims.append({"stream": fp.stem, "label": d.get("label"), "caption": plot.get("caption") or d.get("caption"),
                     "from": plot.get("reactant_smiles"), "to": plot.get("product_smiles"), "source": plot.get("source"),
                     "kind": d.get("kind")})
    out["sims"] = sims
    out["reactor"] = reactor_products(job_dir)
    return out


def _explored(frames: list, lines: list) -> list:
    """Every species the run found (a frame holds only the most populated): [id, the frame time it first
    counted, the species it was found from (its live event; None for one an IRC end made)]. Species are
    only ever appended, so species k counts from the first frame with more than k."""
    parents = {}
    for line in lines:
        if '"species"' not in line[:40]:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "species":
            parents.setdefault(str(ev.get("index")), ev.get("parent"))
    out, k = [], 0
    for f in frames:
        while k < (f.get("counts") or {}).get("species", 0):
            p = parents.get(str(k))
            out.append([str(k), f["t"], None if p is None else str(p)])
            k += 1
    return out


def from_job(job_dir: Path, *, running: bool = False, t0: Optional[float] = None) -> dict:
    """A job's timeline: its recorded frames, and while it runs the network as
    events.jsonl has it now (so it grows from the first species on)."""
    live = Path(job_dir) / "live"
    fp = live / FRAMES
    frames, t_start = [], None
    for line in (fp.read_text().splitlines() if fp.exists() else []):
        try:
            f = json.loads(line)
        except ValueError:
            continue
        t_start = t_start or f.pop("t_start", None)
        f.pop("t_start", None)
        frames.append(f)
    lines = _events(job_dir)
    final_steps = None if running else _summary_steps(job_dir)
    if final_steps is not None:
        # The run's live events reported steps under the pair searched; its summary has each under
        # the species its IRC joins, and none it dropped: no frame shows a step that was never real.
        real = {st["label"]: {str(st["a"]), str(st["b"])} for st in final_steps}
        for f in frames:
            f["steps"] = [x for x in f["steps"] if real.get(x[0]) == {x[1], x[2]}]
            f["gone"] = [x for x in f.get("gone") or [] if real.get(x[0]) == {x[1], x[2]}]
        g = _growing(job_dir, lines, t0, steps=final_steps) if lines else None
        if g is not None:
            g.pop("growing", None)
            g["t"] = max(g["t"], frames[-1]["t"] if frames else 0.0)
            g["steered"] = frames[-1]["steered"] if frames else g["steered"]
            frames.append(g)
    elif running or not frames:
        g = _growing(job_dir, lines, t0) if lines else None
        if g is not None and (not frames or g["counts"]["species"] > frames[-1]["counts"]["species"]
                              or g["counts"]["steps"] > frames[-1]["counts"]["steps"]):
            frames.append(g)
    events = _changes(lines, t_start or t0 or 0.0)
    sel = None
    sfp = live / SELECTIVITY_FRAMES
    if sfp.exists():
        rows = [json.loads(x) for x in sfp.read_text().splitlines() if x.strip()]
        if rows:
            sel = _selectivity(rows[-1], t_start or 0.0, entries=[r["entry"] for r in rows])
    tl = timeline(frames, events, sel, source={"kind": "job"})
    # Species drawn from their geometry (no SMILES fits them): their formulas name them.
    formulas = {sid: species_formula(job_dir, sid) for sid, smi in tl["smiles"].items() if not drawable(smi)}
    tl["names"] = {**names_of(tl["smiles"], formulas), **{k: v for k, v in tl["names"].items() if v}}
    tl["graph_ids"] = sorted(k for k, v in tl["smiles"].items() if not drawable(v))
    # Species the run took from an earlier one (`--known`): shown as reused, not found again.
    known = set()
    for line in lines:
        if '"known"' in line and '"species"' in line[:40]:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "species" and ev.get("known"):
                known.add(str(ev.get("index")))
    tl["known_ids"] = sorted(known)
    tl["explored"] = _explored(frames, lines)
    tl["activity"] = activity(job_dir)
    return tl


def pack_replay(snap_dir: Path, slug: str, title: str, tag: str, selectivity_file: Optional[Path] = None,
                blurb: str = "") -> Path:
    """An Examples entry that opens straight into the live network view."""
    from mepd.web.showcase_tool import SHOWCASE

    tl = from_snapshots(Path(snap_dir), selectivity_file)
    SHOWCASE.mkdir(exist_ok=True)
    out = SHOWCASE / f"{slug}.timeline.json.gz"
    with gzip.open(out, "wt") as fh:
        json.dump(tl, fh, separators=(",", ":"))
    (SHOWCASE / f"{slug}.json").write_text(json.dumps({"slug": slug, "title": title, "tag": tag, "blurb": blurb,
                                                       "kind": "replay"}, indent=1) + "\n")
    return out


def example_timeline(slug: str) -> Optional[dict]:
    from mepd.web.showcase_tool import SHOWCASE

    fp = SHOWCASE / f"{slug}.timeline.json.gz"
    if "/" in slug or slug.startswith(".") or not fp.exists():
        return None
    with gzip.open(fp, "rt") as fh:
        tl = json.load(fh)
    if "names" not in tl:    # packed before names were part of the timeline
        tl["names"] = names_of(tl.get("smiles") or {})
    _unfit(tl.get("frames") or [])
    return tl


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pack an explorer replay as a live-network example.")
    ap.add_argument("snapshots", type=Path)
    ap.add_argument("slug")
    ap.add_argument("--title", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--blurb", default="")
    ap.add_argument("--selectivity", type=Path)
    a = ap.parse_args()
    path = pack_replay(a.snapshots, a.slug, a.title, a.tag, a.selectivity, a.blurb)
    print(f"{path} {path.stat().st_size / 1e6:.2f} MB")


# ---------------------------------------------------------------------------------------------------
# Reactor events as they happen (the live views' overlay): each reaction event the reactor's MD finds
# (<output>/reactor/run_kk/live_events.json, rewritten after each MD segment), with when it happened,
# the species whose MD it was, and the species it became once that is found. Served incrementally:
# events are numbered in the order this server first saw them; a client asks for those after its last
# (`since`) and starts again when the server's `epoch` changes (a restart renumbers them).

import time as _time

EPOCH = f"{_time.time():.0f}"
SEGMENT_FRAMES = 125   # frames per segment file the reactor writes (reactor_generator.LIVE_FRAMES)
_REACTOR: dict = {}


def _canon_multiset(smiles: str) -> Optional[tuple]:
    """A state as its sorted canonical molecules (None if one does not parse)."""
    from mepd.web.chem import _canonical

    out = []
    for m in str(smiles).split("|")[0].split("."):
        m = m.strip()
        if not m:
            continue
        c = _CANON.get(m)
        if c is None:
            c = _CANON[m] = _canonical(m) or ""
        if not c:
            return None
        out.append(c)
    return tuple(sorted(out))


_CANON: dict = {}


def _side(text: str) -> list:
    """One side of a reactor event's label ("2 [H] + C=O"), molecule by molecule, canonical."""
    out = []
    for part in text.split(" + "):
        part = part.strip()
        n, _, smi = part.partition(" ")
        k = int(n) if n.isdigit() and smi else 1
        out += [_canon_multiset(smi if n.isdigit() and smi else part)] * k
    return out


def _first_frame(run: Path):
    """(symbols, coordinates) of a run's first MD frame, or None."""
    fp = run / "md" / "segment_000.xyz"
    try:
        with open(fp) as fh:
            n = int(fh.readline())
            fh.readline()
            rows = [fh.readline().split() for _ in range(n)]
        return [r[0] for r in rows], [[float(v) for v in r[1:4]] for r in rows]
    except (OSError, ValueError, IndexError):
        return None


def _xyz_rows(text: str):
    lines = [x.split() for x in str(text).splitlines()[2:] if x.strip()]
    return [r[0] for r in lines], [[float(v) for v in r[1:4]] for r in lines]


def _centred(xyz):
    n = max(1, len(xyz))
    c = [sum(p[i] for p in xyz) / n for i in range(3)]
    return [[p[0] - c[0], p[1] - c[1], p[2] - c[2]] for p in xyz]


def _species_states(job_dir: Path) -> dict:
    """What the run has found so far: species id -> (smiles, symbols, centred xyz, parent)."""
    job_dir = Path(job_dir)
    out = {}
    seed = job_dir / "inputs" / "seed.xyz"
    if seed.exists():
        sy, xyz = _xyz_rows(seed.read_text())
        out["0"] = ["", sy, _centred(xyz), None]
    for line in _events(job_dir):
        if '"species"' not in line[:40]:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "species" and ev.get("xyz"):
            sy, xyz = _xyz_rows(ev["xyz"])
            out[str(ev.get("index"))] = [ev.get("smiles") or "", sy, _centred(xyz), None if ev.get("parent") is None else str(ev["parent"])]
    try:   # every species' SMILES, also those without an event (made by an IRC end)
        raw = json.loads((job_dir / "live" / "kinetics.json").read_text())
        for s in (raw.get("data", raw).get("species") or []):
            sid = str(s.get("id", s.get("index")))
            if sid in out:
                out[sid][0] = out[sid][0] or s.get("smiles") or ""
            else:
                out[sid] = [s.get("smiles") or "", None, None, None]
    except (OSError, ValueError, AttributeError):
        pass
    perceived = {}
    try:
        perceived = json.loads((job_dir / "live" / PERCEIVED).read_text())
    except (OSError, ValueError):
        pass
    for sid, v in out.items():
        v[0] = v[0] or perceived.get(sid, "")
    return out


def _source_of(run: Path, states: dict) -> Optional[str]:
    """The species whose MD this run is: its first frame is that species' geometry, centred."""
    got = _first_frame(run)
    if got is None:
        return None
    sy, xyz = got
    best, best_d = None, 0.05
    for sid, (_, ssy, sxyz, _) in states.items():
        if ssy != sy or sxyz is None:
            continue
        d = max(abs(a - b) for p, q in zip(xyz, sxyz) for a, b in zip(p, q))
        if d < best_d:
            best, best_d = sid, d
    return best


def _source_by_time(job_dir: Path, out: Path, run: Path, index: int) -> Optional[str]:
    """The species whose propose job started this run: the one running when its first MD segment was
    written (network_aware.jsonl's job lines, once that job ended), else, while it runs, the propose job
    the explorer runs now (live/queue.json; with several, live/propose_<k>.json's runs_from)."""
    import re

    try:
        t = (run / "md" / "segment_000.xyz").stat().st_mtime
    except OSError:
        t = _time.time()
    best = None
    fp = out / "network_aware.jsonl"
    if fp.exists():
        for line in fp.read_text().splitlines():
            if '"propose"' not in line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            m = re.match(r"species (\d+)", str(e.get("what") or ""))
            if e.get("kind") == "propose" and m and e["t"] - (e.get("run_s") or 0) - 1 <= t <= e["t"] + 1:
                start = e["t"] - (e.get("run_s") or 0)
                if best is None or start > best[0]:
                    best = (start, m.group(1))
    if best:
        return best[1]
    live = Path(job_dir) / "live"
    try:
        q = json.loads((live / "queue.json").read_text())
        running = [re.match(r"species (\d+)", str(r.get("what") or "")) for r in (q.get("data", q).get("running") or [])
                   if r.get("kind") == "propose"]
        running = [m.group(1) for m in running if m]
    except (OSError, ValueError, AttributeError):
        running = []
    if len(running) == 1:
        return running[0]
    froms = []
    for k in running:
        try:
            d = json.loads((live / f"propose_{k}.json").read_text())
            froms.append((int(d.get("data", d).get("runs_from")), k))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    froms = [x for x in froms if x[0] <= index]
    return max(froms)[1] if froms else (running[0] if running else None)


def _runs_of(job: dict) -> list:
    """(run id, run folder, the folder the event movie route takes it from) for each reactor run of a
    network expansion (output/reactor/run_kk) or a reacting mixture's seeds (output/seeds/<s>/run/reactor/run_kk)."""
    out = Path(job.get("output_dir") or "")
    if job.get("op") == "network-mixture":
        res = []
        for sd in sorted((out / "seeds").glob("*")):
            for k, d in enumerate(sorted((sd / "run" / "reactor").glob("run_*"))):
                res.append((f"{sd.name}/{k}", d, sd.name))
        return res
    return [(str(k), d, None) for k, d in enumerate(sorted((out / "reactor").glob("run_*")))]


def reactor_events(job_dir: Path, job: dict, since: int = 0, limit: int = 4000) -> dict:
    """The reactor events of a job after the first `since` this server has numbered (see above).
    Each: {i, t (s from the job's start, when its MD segment was written), run, seed, k, label, atoms
    (count), src (species id or the mixture molecule it started from), prod (the species or molecule it
    became, once there is one), changes (bonds formed and broken)}."""
    job_dir = Path(job_dir)
    key = str(job_dir)
    st = _REACTOR.setdefault(key, {"seen": {}, "events": [], "src": {}, "state": {}, "prod_pending": []})
    t0 = float(job.get("started") or 0.0)
    mixture = job.get("op") == "network-mixture"
    states = None
    seeds_mol = {}
    if mixture:
        try:
            mx = json.loads((Path(job["output_dir"]) / "mixture.json").read_text())
            seeds_mol = {s["name"]: s.get("molecules") or [] for s in mx.get("seeds") or []}
            mols = {_canon_multiset(m["molecule"]): m["molecule"] for m in mx.get("molecules") or []}
        except (OSError, ValueError, KeyError):
            mols = {}
    fresh = []
    for rid, run, seed in _runs_of(job):
        fp = run / "live_events.json"
        try:
            m = fp.stat().st_mtime
        except OSError:
            continue
        seen = st["seen"].get(rid)
        if seen is not None and seen[0] == m:
            continue
        try:
            evs = json.loads(fp.read_text()).get("events") or []
        except (OSError, ValueError):
            continue
        n0 = seen[1] if seen else 0
        st["seen"][rid] = (m, len(evs))
        if rid not in st["src"]:
            if mixture:
                if seeds_mol.get(seed):
                    st["src"][rid] = seeds_mol[seed][0]
            else:
                states = states if states is not None else _species_states(job_dir)
                st["src"][rid] = _source_of(run, states) or _source_by_time(job_dir, Path(job["output_dir"]), run, int(rid))
        for ev in evs[n0:]:
            seg = run / "md" / f"segment_{int(ev.get('product_frame', 0)) // SEGMENT_FRAMES:03d}.xyz"
            try:
                t = seg.stat().st_mtime
            except OSError:
                t = m
            label = ev.get("label") or ""
            fresh.append({"t": round(t - t0, 1) if t0 else 0.0, "run": rid.split("/")[-1], "seed": seed, "rid": rid,
                          "k": ev.get("event"), "label": label, "atoms": len(ev.get("atoms") or []),
                          "src": st["src"].get(rid), "prod": None,
                          "changes": [[c["atoms"][0], c["atoms"][1], 1 if c.get("formed") else 0] for c in ev.get("bond_changes") or []][:8]})
    if mixture:   # a seed's molecules are known once mixture.json lists it
        for e in st["events"] + fresh:
            if e["src"] is None and seeds_mol.get(e["seed"]):
                e["src"] = seeds_mol[e["seed"]][0]
                st["src"][e["rid"]] = e["src"]
                if e.get("i") is not None:
                    st.setdefault("src_late", set()).add(e["i"])
    if fresh:
        fresh.sort(key=lambda e: e["t"])
        for e in fresh:
            e["i"] = len(st["events"])
            st["events"].append(e)
            st["prod_pending"].append(e["i"])
    # what each event became: the state after it (the run's start, less its reactants, plus its products)
    # once that is a species of the run (or, in a mixture, a product that is one of its molecules)
    if st["prod_pending"]:
        if mixture:
            for i in list(st["prod_pending"]):
                e = st["events"][i]
                sides = e["label"].split(" -> ")
                prods = [p.strip() for p in sides[1].split(" + ")] if len(sides) == 2 else []
                reac = set(p.strip() for p in sides[0].split(" + ")) if sides else set()
                for p in prods:
                    c = _canon_multiset(p)
                    if c and c in mols and p not in reac:
                        e["prod"] = mols[c]
                        break
                if e["prod"] is not None:
                    st["prod_pending"].remove(i)
        else:
            states = states if states is not None else _species_states(job_dir)
            by_state = {}
            for sid, (smi, _, _, _) in states.items():
                c = _canon_multiset(smi) if smi else None
                if c and c not in by_state:
                    by_state[c] = sid
            after_of = st.setdefault("after", {})
            for i in list(st["prod_pending"]):
                e = st["events"][i]
                if i not in after_of:   # the state after it, worked out once, in the run's own order
                    rs = st["state"].get(e["rid"])
                    if rs is None and e["src"] is not None and states.get(e["src"], [""])[0]:
                        rs = list(_canon_multiset(states[e["src"]][0]) or [])
                    sides = e["label"].split(" -> ")
                    reac = _side(sides[0]) if len(sides) == 2 else [None]
                    prods = _side(sides[1]) if len(sides) == 2 else [None]
                    after = None
                    if rs is not None and not any(x is None or len(x) != 1 for x in reac + prods):
                        left = list(rs)
                        if all(x[0] in left and not left.remove(x[0]) for x in reac):
                            after = tuple(sorted(left + [x[0] for x in prods]))
                            st["state"][e["rid"]] = list(after)
                    after_of[i] = after
                after = after_of[i]
                if after is None:
                    st["prod_pending"].remove(i)
                    continue
                hit = by_state.get(after)
                if hit is not None and hit != e["src"]:
                    e["prod"] = hit
                    st["prod_pending"].remove(i)
            st["prod_pending"] = st["prod_pending"][-3000:]
    evs = st["events"]
    since = max(0, int(since))
    # (products matched since: the client updates those it has)
    return {"epoch": EPOCH, "n": len(evs), "events": evs[since:since + limit],
            "prods": {e["i"]: e["prod"] for e in evs[max(0, since - 400):since] if e["prod"] is not None},
            "srcs": {i: evs[i]["src"] for i in sorted(st.get("src_late") or ()) if i < since},
            "runs": len(st["seen"]), "running": sum(1 for _, run, _ in _runs_of(job) if not (run / "md" / "done").exists())}
