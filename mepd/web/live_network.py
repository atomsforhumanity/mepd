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
                {True: 1, False: 0}.get(st.get("irc_ok"), None), _sig(st.get("flux"))]
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
    # One edge per pair of species (the lowest of its channels), lowest barriers first.
    best = {}
    for st in known_steps:
        a, b = str(st["a"]), str(st["b"])
        if a not in have or b not in have or a == b:
            continue
        fwd, rev = (st["dG"][0], st["dG"][1]) if a < b else (st["dG"][1], st["dG"][0])
        key = (min(a, b), max(a, b))
        low = min(x for x in (fwd, rev) if x is not None) if (fwd, rev) != (None, None) else None
        if key not in best or (low is not None and low < best[key][0]):
            best[key] = (low, {"id": st["label"], "from": key[0], "to": key[1], "dG_fwd": fwd, "dG_rev": rev,
                               "state": "open", "energy_source": "gxtb"})
    kin["step_list"] = [v for _, v in sorted(best.values(), key=lambda x: (x[0] is None, x[0] or 0))]
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


def timeline(frames: list, events: list, selectivity: Optional[dict], *, source: dict) -> dict:
    """Frames share one smiles table; the end state is the last steered frame
    (a trailing unsteered refresh is not the answer)."""
    smiles = {}
    for f in frames:
        smiles.update(f.pop("smiles", {}))
    final = max((i for i, f in enumerate(frames) if f["steered"]), default=len(frames) - 1)
    return {"source": source, "frames": frames, "smiles": smiles, "names": names_of(smiles), "events": events,
            "selectivity": selectivity, "final": final}


def names_of(smiles: dict) -> dict:
    """Everyday names where every molecule of a mixture has one ("water + CO")."""
    from mepd.web.chem import common_name

    out = {}
    for sid, smi in smiles.items():
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


def record(job_dir: Path, kin: dict) -> bool:
    """Append one live kinetics refresh of a running job to its frames."""
    f = compact(kin)
    if f is None:
        return False
    live = Path(job_dir) / "live"
    live.mkdir(parents=True, exist_ok=True)
    with open(live / FRAMES, "a") as fh:
        fh.write(json.dumps({**f, "t_start": kin.get("t_start")}, separators=(",", ":")) + "\n")
    return True


def record_selectivity(job_dir: Path, data: dict) -> None:
    """The annotator's selectivity.json is rewritten on each refresh: keep each version's latest entry."""
    entries = (data or {}).get("entries") or []
    if not entries:
        return
    head = {k: v for k, v in data.items() if k != "entries"}
    with open(Path(job_dir) / "live" / SELECTIVITY_FRAMES, "a") as fh:
        fh.write(json.dumps({**head, "entry": entries[-1]}, separators=(",", ":")) + "\n")


def from_job(job_dir: Path) -> Optional[dict]:
    live = Path(job_dir) / "live"
    fp = live / FRAMES
    if not fp.exists():
        return None
    frames, t_start = [], None
    for line in fp.read_text().splitlines():
        try:
            f = json.loads(line)
        except ValueError:
            continue
        t_start = t_start or f.pop("t_start", None)
        f.pop("t_start", None)
        frames.append(f)
    ev = live / "events.jsonl"
    events = _changes(ev.read_text().splitlines(), t_start or 0.0) if ev.exists() else []
    sel = None
    sfp = live / SELECTIVITY_FRAMES
    if sfp.exists():
        rows = [json.loads(x) for x in sfp.read_text().splitlines() if x.strip()]
        if rows:
            sel = _selectivity(rows[-1], t_start or 0.0, entries=[r["entry"] for r in rows])
    return timeline(frames, events, sel, source={"kind": "job"})


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
