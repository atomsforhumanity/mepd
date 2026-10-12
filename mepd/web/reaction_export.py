"""A reaction's page as files to share (Analyze › a reaction › Export).

Two forms of the same data (`reaction_bundle`):

- `export_html`: one self-contained page -- no server, no internet (3Dmol is
  inlined): the equation with depictions, the energy diagram with every
  channel found for the reaction, the reactant complex, each channel's TS
  and IRC and the product complex in 3D, the energies and where they come
  from. Opens from disk, mails as one file, and embeds in an <iframe> (it
  fits its width).
- `export_zip`: the raw data for SI and other tools -- an XYZ per structure,
  each channel's TS and IRC, energies as CSV and JSON, the diagram as SVG,
  the profiles the barriers were computed with, a README, and the page.

A channel is one way the reaction goes that a search on its edge found and
IRC-verified: each direct channel of a Reaction channels / Sample more
paths run, or the route of a TS search (one TS, or several steps through
intermediates). Exits to other products (off-target) are not this
reaction's and are only counted.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

KCAL = 627.509474
_VENDOR = Path(__file__).parent / "static" / "vendor" / "3Dmol-min.js"
_SEARCH_OPS = ("ts", "tsopt", "channels", "channels-more")
MAX_DIAGRAM_CHANNELS = 8   # categorical hues; more are listed in the table, not drawn
# Validated categorical palette (light on white, dark on #1f1e1c): fixed order, never cycled.
_HUES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_HUES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def _mepd_version() -> str:
    try:
        from importlib.metadata import version

        return version("mepd")
    except Exception:
        return "unknown"


def _read(fp: Path) -> Optional[str]:
    try:
        return Path(fp).read_text()
    except OSError:
        return None


def route_ts_xyz(job: dict, job_dir: Path, result_of: Callable[[], dict], reaction: Optional[int] = None) -> Optional[str]:
    """The structure behind a job's barrier (xyz): its IRC-verified TS; else
    its first optimized TS; else the highest point of its path (an
    unverified barrier). A nanoreactor job: that reaction's own TS. None
    if it has none."""
    fp = Path(job_dir) / "route_ts.xyz"
    if fp.is_file():
        return fp.read_text()
    if job["op"] == "nanoreactor":
        from mepd.web.nanoreactor import _json

        data = _json(Path(job["output_dir"]) / "network.json") or {}
        rx = next((r for r in data.get("reactions") or [] if r["id"] == reaction), None)
        return _read(Path(((rx or {}).get("ts") or {}).get("files", {}).get("ts") or "")) if rx else None
    result = result_of()
    if (result.get("route_ts") or {}).get("xyz"):
        return result["route_ts"]["xyz"]
    for kind in ("ts", "path"):
        for g in result.get("groups") or []:
            if g.get("kind") != kind or not g.get("entries"):
                continue
            e = g["entries"][0]
            k = e.get("ts_index") or 0 if kind == "path" else 0
            if e.get("frames"):
                return e["frames"][min(k, len(e["frames"]) - 1)]["xyz"]
    return None


def _ladder_from_graph(r: dict, structures: dict) -> Optional[dict]:
    """Every species and both complexes at one level of theory: kcal/mol
    from the separated reactants (as the reaction card draws it)."""
    recs = [structures.get(i) for i in r["reactants"] + r["products"]]
    cx = [structures.get(i) for i in r.get("complexes") or []]
    if not all(x and x.get("energy") is not None for x in recs):
        return None
    key = (recs[0].get("level") or {}).get("key")
    if not all((x.get("level") or {}).get("key") == key for x in recs + [c for c in cx if c]):
        return None
    base = sum(structures[i]["energy"] for i in r["reactants"])

    def e(x):
        return (x["energy"] - base) * KCAL if x and x.get("energy") is not None and \
            (x.get("level") or {}).get("key") == key else None
    return {"reactants": 0.0, "reactant_complex": e(cx[0] if cx else None),
            "product_complex": e(cx[1] if len(cx) > 1 else None),
            "products": (sum(structures[i]["energy"] for i in r["products"]) - base) * KCAL}


# ------------------------------------------------------------------ channels

def _step(entry: dict, irc: Optional[dict] = None) -> Optional[dict]:
    """One TS of a channel from a result entry: an IRC (frames, ts_index)
    or a lone TS (`irc`: its IRC entry, if any)."""
    path = irc if irc is not None else entry if len(entry.get("frames") or []) > 1 else None
    frames = (path or entry).get("frames") or []
    if not frames:
        return None
    k = (path or entry).get("ts_index")
    k = 0 if k is None or not 0 <= k < len(frames) else k
    out = {"label": entry.get("label") or entry["id"], "ts_xyz": frames[k]["xyz"],
           "ts_hartree": frames[k].get("energy_hartree"), "barrier_search_kcal": entry.get("barrier_kcal"),
           "irc": None}
    if path is not None:
        out["irc"] = {"ts_index": k, "frames": [f["xyz"] for f in frames],
                      "energy_kcal": [f.get("energy_kcal") for f in frames],
                      "energy_hartree": [f.get("energy_hartree") for f in frames]}
    return out


def _result_channels(result: dict) -> tuple[list[dict], int]:
    """(channels, off-target exits) of one search's page result."""
    groups: dict[str, list] = {}
    for g in result.get("groups") or []:
        groups.setdefault(g.get("kind"), []).extend(g.get("entries") or [])
    out = []
    for e in groups.get("channel", []):                       # direct channels (Reaction channels)
        st = _step(e)
        if st:
            out.append({"kind": "direct", "label": e.get("label") or e["id"], "verified": True, "steps": [st]})
    routes: dict[str, list] = {}
    for e in groups.get("alternate", []):                     # multi-step channels, one entry per step
        m = re.match(r"^(alt_\d+)_step_\d+$", e["id"])
        routes.setdefault(m.group(1) if m else "first search", []).append(e)
    for key, entries in routes.items():
        steps = [s for s in (_step(e) for e in entries) if s]
        if steps:
            out.append({"kind": "multi-step", "label": f"{key.replace('_', ' ')} ({len(steps)} steps)",
                        "verified": True, "steps": steps})
    offtarget = len(groups.get("offtarget", []))
    if out or "channel" in groups:
        return out, offtarget
    # A TS search (or TS optimization): the TSs on its start -> end route, in order.
    ircs = {e["id"]: e for e in groups.get("irc", [])}
    steps = [s for s in (_step(e, ircs.get(f"{e['id']}_irc")) for e in groups.get("ts", [])) if s]
    if steps:
        verified = result.get("barrier_verified", True) is not False
        out.append({"kind": "direct" if len(steps) == 1 else "multi-step",
                    "label": "TS search" if len(steps) == 1 else f"TS search route ({len(steps)} steps)",
                    "verified": verified, "steps": steps})
    elif result.get("barrier_kcal") is not None:            # only the path's highest point: unverified
        path = next((e for e in groups.get("path", []) if e.get("ts_index") is not None), None)
        st = _step(path) if path else None
        if st:
            out.append({"kind": "path maximum", "label": "path maximum (no TS/IRC confirms it)", "verified": False,
                        "steps": [st]})
    return out, offtarget


def _channels(r: dict, S: dict, edge: Optional[dict], jobs: dict, ladder: dict, *,
              page_of: Callable[[dict], dict], result_of: Callable[[dict], dict]) -> tuple[list, int, list]:
    """Every channel found for the reaction (all searches on its edge),
    placed on the diagram: (channels, off-target exits, the searches)."""
    if not edge:
        return [], 0, []
    pages: dict[str, dict] = {}
    for j in jobs.values():
        if j["op"] in _SEARCH_OPS and edge["id"] in (j.get("targets") or {}).get("edges", []) and j["status"] == "done":
            page = page_of(j)
            pages.setdefault(page["id"], page)
    found, offtarget, seen = [], 0, set()
    for page in sorted(pages.values(), key=lambda j: j.get("created") or 0):
        try:
            result = result_of(page)
        except Exception:
            continue
        chans, off = _result_channels(result)
        offtarget += off
        level = page.get("level") or {}
        for c in chans:
            top = max(c["steps"], key=lambda s: s["ts_hartree"] if s["ts_hartree"] is not None else float("-inf"))
            key = (level.get("key"), round(top["ts_hartree"], 5) if top["ts_hartree"] is not None else c["label"])
            if key in seen:                     # the same TS reached by two searches
                continue
            seen.add(key)
            found.append({**c, "level": level.get("label"), "level_key": level.get("key"), "job": page["id"],
                          "job_title": page.get("title"), "top": top})
    # Placement: the TS's own energy against the separated reactants (same level of theory), else
    # (another level, or no reactant energies) the reactant complex plus the search's barrier.
    rc = ladder.get("reactant_complex")
    for c in found:
        recs = [S.get(i) or {} for i in r["reactants"]]
        same = all(x.get("energy") is not None and (x.get("level") or {}).get("key") == c["level_key"] for x in recs)
        e_ts = c["top"]["ts_hartree"]
        if same and e_ts is not None and c["level_key"] is not None:
            c["ts_kcal"] = (e_ts - sum(x["energy"] for x in recs)) * KCAL
            c["placed"] = "absolute"
        else:
            b = c["top"]["barrier_search_kcal"]
            c["ts_kcal"] = rc + b if rc is not None and b is not None else None
            c["placed"] = "search reference"
        c["barrier_kcal"] = (c["ts_kcal"] - rc if c["placed"] == "absolute" and rc is not None and c["ts_kcal"] is not None
                             else c["top"]["barrier_search_kcal"])
        # As its search reports it: from the lowest reactant geometry that search found (a conformer
        # it sampled can lie below this reaction's reactant complex).
        c["barrier_search_kcal"] = c["top"]["barrier_search_kcal"]
        del c["top"]
    found.sort(key=lambda c: (not c["verified"], c["ts_kcal"] if c["ts_kcal"] is not None else float("inf")))
    for n, c in enumerate(found, start=1):
        c["n"], c["id"], c["name"] = n, f"ch{n}", f"Channel {n}"
    return found, offtarget, list(pages.values())


def reaction_bundle(ws, jobs: dict, rid: str, *, job_dir: Callable[[str], Path],
                    result_of: Callable[[dict], dict], page_of: Callable[[dict], dict] = lambda j: j) -> dict:
    """Everything the export shows of reaction `rid`, as plain data: its
    species, complexes, energy ladder and every channel found for it."""
    from mepd.web import chem
    from mepd.web.workspace import WorkspaceError

    snap = ws.snapshot()
    r = (snap.get("reactions") or {}).get(rid)
    if r is None:
        raise WorkspaceError(f"no reaction {rid!r}")
    S = snap["structures"]
    edge = snap["edges"].get(r.get("edge") or "")

    def struct(sid: str, role: str) -> dict:
        rec = S.get(sid) or {}
        xyz = _read(ws.structure_path(sid)) if sid in S else None
        smiles = rec.get("smiles")
        return {"id": sid, "role": role, "name": rec.get("name") or sid, "smiles": smiles,
                "charge": rec.get("charge"), "multiplicity": rec.get("multiplicity"),
                "energy_hartree": rec.get("energy"), "level": (rec.get("level") or {}).get("label"),
                "xyz": xyz, "svg": chem.depict_svg(smiles, 180, 130) if smiles else None}

    def side(ids: list, role: str) -> list:
        counts: dict = {}
        for i in ids:
            counts[i] = counts.get(i, 0) + 1
        return [{**struct(i, role), "count": n, "shuttle": i in (r.get("shuttles") or [])} for i, n in counts.items()]

    complexes = [struct(c, role) for c, role in zip(r.get("complexes") or [], ("reactant complex", "product complex"))]
    ladder = dict(r.get("ladder") or {})
    if ladder.get("reactant_complex") is None:
        ladder = _ladder_from_graph(r, S) or ladder
    channels, offtarget, searches = _channels(r, S, edge, jobs, ladder, page_of=page_of, result_of=result_of)
    origin = r.get("origin") or {}
    if not channels and origin.get("nanoreactor"):
        # The nanoreactor's own TS search for this reaction (its --connect): the TS alone.
        nano = jobs.get(origin.get("job") or "")
        if nano is not None:
            from mepd.web.nanoreactor import _json

            data = _json(Path(nano["output_dir"]) / "network.json") or {}
            rx = next((x for x in data.get("reactions") or [] if x["id"] == origin.get("index")), None)
            ts = (rx or {}).get("ts") or {}
            xyz = _read(Path((ts.get("files") or {}).get("ts") or "")) if ts else None
            if xyz and ts.get("barrier_kcal") is not None:
                rc = ladder.get("reactant_complex")
                channels = [{"kind": "direct", "label": ts.get("label") or "nanoreactor TS search", "verified": True,
                             "steps": [{"label": "TS", "ts_xyz": xyz, "ts_hartree": None,
                                        "barrier_search_kcal": ts["barrier_kcal"], "irc": None}],
                             "level": (nano.get("level") or {}).get("label"),
                             "level_key": (nano.get("level") or {}).get("key"), "job": nano["id"],
                             "job_title": nano.get("title"), "placed": "search reference",
                             "ts_kcal": rc + ts["barrier_kcal"] if rc is not None else None,
                             "barrier_kcal": ts["barrier_kcal"], "n": 1, "id": "ch1", "name": "Channel 1"}]
                searches = [nano]
    lowest = next((c for c in channels if c["verified"] and c["barrier_kcal"] is not None), None)
    source = jobs.get(origin.get("job") or "")
    profiles = []
    for j in searches:
        text = _read(job_dir(j["id"]) / "inputs" / "profile.toml")
        if text and all(text != p["toml"] for p in profiles):
            profiles.append({"job": j["id"], "job_title": j.get("title"), "toml": text})
    return {
        "id": rid, "label": r.get("label") or rid,
        "reactants": side(r["reactants"], "reactant"), "products": side(r["products"], "product"),
        "complexes": complexes, "ladder": ladder, "channels": channels, "offtarget": offtarget,
        "lowest": lowest["id"] if lowest else None,
        "delta_e_kcal": r.get("delta_e_kcal"), "complex_delta_e_kcal": r.get("complex_delta_e_kcal"),
        "relaxed_from": r.get("relaxed_from"),
        "source": ("composed by hand" if origin.get("kind") == "composed"
                   else f"{source['title']}" if source else origin.get("kind") or ""),
        "seen": {"count": r.get("count"), "reverse": r.get("reverse_count")} if origin.get("nanoreactor") else None,
        "profiles": profiles,
        "exported": time.strftime("%Y-%m-%d %H:%M %Z"), "mepd_version": _mepd_version(),
    }


# ------------------------------------------------------------------ files

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("_")[:60] or "structure"


def _many(side: list) -> bool:
    return len(side) > 1 or side[0]["count"] > 1


def _levels(b: dict) -> list[dict]:
    """The diagram's points, kcal/mol from the separated reactants: the ends,
    the complexes, and each channel's (highest) TS."""
    L = b["ladder"]
    rows = [{"key": "reactants", "label": "separated reactants" if _many(b["reactants"]) else "reactant",
             "kcal": L.get("reactants")}]
    if _many(b["reactants"]):
        rows.append({"key": "reactant_complex", "label": "reactant complex", "kcal": L.get("reactant_complex")})
    for c in b["channels"]:
        rows.append({"key": c["id"], "label": f"{c['name']} TS" + ("" if c["verified"] else " (path maximum)"),
                     "kcal": c["ts_kcal"], "channel": c["n"]})
    if not b["channels"]:
        rows.append({"key": "ts", "label": "TS (not searched)", "kcal": None})
    if _many(b["products"]):
        rows.append({"key": "product_complex", "label": "product complex", "kcal": L.get("product_complex")})
    rows.append({"key": "products", "label": "separated products" if _many(b["products"]) else "product",
                 "kcal": L.get("products")})
    return rows


def _hue_style() -> str:
    """Per-channel colours as classes, light and dark (inline or standalone SVG)."""
    light = "".join(f".c{k + 1}{{stroke:{h}}}.f{k + 1}{{fill:{h}}}" for k, h in enumerate(_HUES_LIGHT))
    dark = "".join(f".c{k + 1}{{stroke:{h}}}.f{k + 1}{{fill:{h}}}" for k, h in enumerate(_HUES_DARK))
    return f"{light}@media (prefers-color-scheme:dark){{{dark}}}"


def energy_diagram_svg(b: dict, width: int = 640, height: int = 220) -> str:
    """The energy diagram as a standalone SVG, kcal/mol from the separated
    reactants: the ends and complexes, then one TS per channel (the first
    MAX_DIAGRAM_CHANNELS), each reached from the reactant side and going on
    to the product side, in its channel's colour. Text is in ink
    (currentColor); a channel's colour only marks its lines."""
    rows = _levels(b)
    drawn_ch = [r for r in rows if r.get("channel") and r["kcal"] is not None][:MAX_DIAGRAM_CHANNELS]
    keep = {r["key"] for r in drawn_ch}
    cols = [r for r in rows if not r.get("channel") or r["key"] in keep]
    known = [r["kcal"] for r in cols if r["kcal"] is not None]
    if len(known) < 2:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 40" role="img">'
                f'<text x="{width / 2}" y="24" text-anchor="middle" fill="currentColor" font-size="13" '
                f'font-family="system-ui, sans-serif">No energies for this reaction yet.</text></svg>')
    top, bottom = 28, 44
    half = max(14.0, min(28.0, (width - 80) / (len(cols) * 2.6)))   # half a level's width
    pad = max(46.0, half + 20)
    lo, hi = min(known), max(known)
    y = lambda v: top + (height - top - bottom) * (0.5 if hi == lo else (hi - v) / (hi - lo))
    x = lambda i: pad + i * ((width - 2 * pad) / max(1, len(cols) - 1))
    pts = [{"x": x(i), "y": y(r["kcal"]) if r["kcal"] is not None else None, **r} for i, r in enumerate(cols)]
    lowest = next((c for c in b["channels"] if c["id"] == b.get("lowest")), None)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Energy along the reaction, kcal/mol, one transition state per channel" '
             f'font-family="system-ui, sans-serif" font-size="11"><style>{_hue_style()}</style>']
    mid = next((k for k, q in enumerate(pts) if q.get("channel") or q["key"] == "ts"), len(pts) - 1)
    before = [p for p in pts[:mid] if not p.get("channel")]
    after = [p for p in pts if not p.get("channel") and p not in before and p["key"] != "ts"]
    chans = [p for p in pts if p.get("channel")]

    def link(a, c, cls="", dash="4 3"):
        if a["y"] is None or c["y"] is None:
            return
        style = f'class="{cls}" stroke-width="1.5"' if cls else 'stroke="currentColor" stroke-opacity="0.4"'
        parts.append(f'<line x1="{a["x"] + half:.1f}" y1="{a["y"]:.1f}" x2="{c["x"] - half:.1f}" y2="{c["y"]:.1f}" '
                     f'{style} stroke-dasharray="{dash}"/>')
    ends = [p for p in before if p["y"] is not None]
    for a, c in zip(ends, ends[1:]):
        link(a, c)
    ends2 = [p for p in after if p["y"] is not None]
    for a, c in zip(ends2, ends2[1:]):
        link(a, c)
    last_r = ends[-1] if ends else None
    first_p = ends2[0] if ends2 else None
    for p in chans:
        if last_r:
            link(last_r, p, f'c{p["channel"]}', "none")
        if first_p:
            link(p, first_p, f'c{p["channel"]}', "none")
    if not chans and last_r and first_p:
        link(last_r, first_p)
    bold = lambda t: ' font-weight="600"' if t == "lowest" else ""
    for p in pts:
        lines = p["label"].split(" (")[0]
        if p.get("channel"):
            words = [f'ch {p["channel"]}'] + (["lowest"] if lowest and p["key"] == lowest["id"] else [])
        else:
            w = lines.split(" ")
            words = [" ".join(w[:1]), " ".join(w[1:])] if len(w) > 1 else w
        if p["y"] is None:
            parts.append(f'<text x="{p["x"]:.1f}" y="{(height - bottom + top) / 2:.1f}" text-anchor="middle" '
                         f'fill="currentColor" fill-opacity="0.5">—</text>')
        else:
            cls = f'class="c{p["channel"]}"' if p.get("channel") else 'stroke="currentColor"'
            tip = html.escape(f'{p["label"]}: {p["kcal"]:+.1f} kcal/mol')
            parts.append(f'<g><title>{tip}</title>'
                         f'<rect x="{p["x"] - half - 2:.1f}" y="{p["y"] - 18:.1f}" width="{2 * half + 4:.1f}" height="26" '
                         f'fill="transparent"/>'
                         f'<line x1="{p["x"] - half:.1f}" x2="{p["x"] + half:.1f}" y1="{p["y"]:.1f}" y2="{p["y"]:.1f}" '
                         f'{cls} stroke-width="3" stroke-linecap="round"/>'
                         f'<text x="{p["x"]:.1f}" y="{p["y"] - 7:.1f}" text-anchor="middle" fill="currentColor" '
                         f'font-weight="600">{p["kcal"]:+.1f}</text></g>')
        parts.append(f'<text x="{p["x"]:.1f}" y="{height - bottom + 18}" text-anchor="middle" fill="currentColor" '
                     f'fill-opacity="0.75">'
                     + "".join(f'<tspan x="{p["x"]:.1f}" dy="{0 if k == 0 else 13}"'
                               f'{bold(t)}>{html.escape(t)}</tspan>'
                               for k, t in enumerate(words)) + "</text>")
    hidden = len([c for c in b["channels"] if c["ts_kcal"] is not None]) - len(chans)
    if hidden > 0:
        parts.append(f'<text x="{width - 6}" y="12" text-anchor="end" fill="currentColor" fill-opacity="0.6">'
                     f'+{hidden} more channel{"s" if hidden > 1 else ""} in the table</text>')
    parts.append("</svg>")
    return "".join(parts)


def _structures(b: dict) -> list[tuple[str, str, Optional[str]]]:
    """(file name, title, xyz) of every structure the export carries."""
    out = []
    for sp in b["reactants"] + b["products"]:
        out.append((f"{sp['role']}_{_safe(sp['name'])}.xyz", f"{sp['role']}: {sp['name']}", sp["xyz"]))
    for c in b["complexes"]:
        out.append((f"{c['role'].replace(' ', '_')}.xyz", c["role"], c["xyz"]))
    for c in b["channels"]:
        for k, st in enumerate(c["steps"], start=1):
            tag = f"_step{k}" if len(c["steps"]) > 1 else ""
            out.append((f"ts_{c['id']}{tag}.xyz", f"{c['name']}{f' step {k}' if tag else ''} TS", st["ts_xyz"]))
    return [x for x in out if x[2]]


def _ircs(b: dict) -> list[tuple[str, str, dict]]:
    out = []
    for c in b["channels"]:
        for k, st in enumerate(c["steps"], start=1):
            if st.get("irc"):
                tag = f"_step{k}" if len(c["steps"]) > 1 else ""
                out.append((f"irc_{c['id']}{tag}.xyz", f"{c['name']}{f' step {k}' if tag else ''} IRC", st["irc"]))
    return out


def _irc_xyz(irc: dict) -> str:
    out = []
    for k, (xyz, e) in enumerate(zip(irc["frames"], irc["energy_hartree"])):
        lines = xyz.splitlines()
        lines[1] = f"IRC frame {k}" + (f" E={e:.10f} Eh" if e is not None else "") + \
            (" (TS)" if k == irc.get("ts_index") else "")
        out.append("\n".join(lines) + "\n")
    return "".join(out)


def _placed_note(c: dict) -> str:
    return ("" if c["placed"] == "absolute" else
            "placed from its search's own reference (another level of theory, or no reactant energies)")


def _energies_csv(b: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["point", "kcal/mol from separated reactants"])
    for r in _levels(b):
        w.writerow([r["label"], "" if r["kcal"] is None else f"{r['kcal']:.3f}"])
    w.writerow([])
    w.writerow(["channel", "kind", "TS kcal/mol from separated reactants", "barrier kcal/mol from reactant complex",
                "barrier kcal/mol as its search reports it", "IRC-verified", "steps", "level of theory", "found by", "note"])
    for c in b["channels"]:
        w.writerow([c["name"], c["kind"], "" if c["ts_kcal"] is None else f"{c['ts_kcal']:.3f}",
                    "" if c["barrier_kcal"] is None else f"{c['barrier_kcal']:.3f}",
                    "" if c.get("barrier_search_kcal") is None else f"{c['barrier_search_kcal']:.3f}",
                    "yes" if c["verified"] else "no",
                    len(c["steps"]), c.get("level") or "", c.get("job_title") or "", _placed_note(c)])
    w.writerow([])
    w.writerow(["species", "role", "SMILES", "charge", "multiplicity", "energy (Hartree)", "level of theory"])
    for sp in b["reactants"] + b["products"] + b["complexes"]:
        w.writerow([sp["name"], sp["role"], sp.get("smiles") or "", sp.get("charge"), sp.get("multiplicity"),
                    "" if sp.get("energy_hartree") is None else f"{sp['energy_hartree']:.10f}", sp.get("level") or ""])
    return buf.getvalue()


def _readme(b: dict) -> str:
    lines = [f"# {b['label']}", "", f"Exported from mepd {b['mepd_version']} on {b['exported']}.", ""]
    if b["channels"]:
        lines.append(f"{len(b['channels'])} channel{'s' if len(b['channels']) > 1 else ''} found for this reaction"
                     + (f" ({b['offtarget']} more searches led to other products, not included)" if b["offtarget"] else "")
                     + ":")
        lines.append("")
        for c in b["channels"]:
            bar = f"{c['barrier_kcal']:.1f} kcal/mol from the reactant complex" if c["barrier_kcal"] is not None else "barrier unknown"
            lines.append(f"- {c['name']} ({c['kind']}{', lowest' if c['id'] == b.get('lowest') else ''}): {bar}, "
                         f"{'IRC-verified' if c['verified'] else 'not IRC-verified'}"
                         + (f", {c['level']}" if c.get("level") else "") + (f"; {_placed_note(c)}" if _placed_note(c) else ""))
        lines.append("")
    else:
        lines += ["No TS search on this reaction yet.", ""]
    if b.get("delta_e_kcal") is not None:
        lines += [f"Reaction energy: {b['delta_e_kcal']:+.1f} kcal/mol (separated products minus separated reactants).", ""]
    lines += ["Energies are electronic energies (no ZPE or thermal corrections), kcal/mol relative to the separated "
              "reactants, all over the same atoms.", "",
              "## Files", "",
              "- `report.html`: the interactive page (opens offline; embed it with an <iframe>)",
              "- `reaction.json`: everything below as data",
              "- `energies.csv`: the diagram's points, each channel, and every species' energy",
              "- `energy_diagram.svg`: the diagram as a vector figure"]
    lines += [f"- `structures/{name}`: {title}" for name, title, _ in _structures(b)]
    lines += [f"- `{name}`: {title}, one frame per point (energy in each comment line)" for name, title, _ in _ircs(b)]
    for k, p in enumerate(b["profiles"]):
        lines.append(f"- `{_profile_name(k)}`: the mepd profile (level of theory and settings) of {p['job_title']}")
    return "\n".join(lines) + "\n"


def _profile_name(k: int) -> str:
    return "profile.toml" if k == 0 else f"profile_{k + 1}.toml"


def _json_data(b: dict) -> dict:
    """The bundle without the depictions (images, not data)."""
    strip = lambda sp: {k: v for k, v in sp.items() if k != "svg"}
    return {**b, "reactants": [strip(s) for s in b["reactants"]], "products": [strip(s) for s in b["products"]],
            "complexes": [strip(s) for s in b["complexes"]], "energy_levels": _levels(b)}


def export_zip(b: dict) -> bytes:
    buf = io.BytesIO()
    root = _safe(b["label"]) or b["id"]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{root}/README.md", _readme(b))
        z.writestr(f"{root}/report.html", export_html(b))
        z.writestr(f"{root}/reaction.json", json.dumps(_json_data(b), indent=2, default=str))
        z.writestr(f"{root}/energies.csv", _energies_csv(b))
        z.writestr(f"{root}/energy_diagram.svg", energy_diagram_svg(b))
        for name, _, xyz in _structures(b):
            z.writestr(f"{root}/structures/{name}", xyz)
        for name, _, irc in _ircs(b):
            z.writestr(f"{root}/{name}", _irc_xyz(irc))
        for k, p in enumerate(b["profiles"]):
            z.writestr(f"{root}/{_profile_name(k)}", p["toml"])
    return buf.getvalue()


# ------------------------------------------------------------------ the page

_CSS = """
:root{--bg:#fbfaf7;--card:#fff;--ink:#1c1b19;--muted:#6b6862;--line:#e6e2da;--accent:#c2410c;--chip:#f3f0ea;
  color-scheme:light;font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#161514;--card:#1f1e1c;--ink:#ece9e3;--muted:#a19d95;--line:#33312d;
  --accent:#fb923c;--chip:#2a2825;color-scheme:dark}}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--ink)}
main{max-width:980px;margin:0 auto;padding:20px 16px 32px}
h1{font-size:1.35rem;margin:0 0 4px;line-height:1.3}h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin:22px 0 8px;font-weight:600}
.meta{color:var(--muted);font-size:.85rem}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.eq{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:14px 0}
.sp{display:flex;flex-direction:column;align-items:center;min-width:90px}
.sp .img{background:#fff;border-radius:8px;padding:2px 6px}.sp .img svg{display:block;width:130px;height:auto;max-width:100%}
.sp .nm{font-size:.85rem;text-align:center;max-width:160px;overflow-wrap:anywhere}.op{font-size:1.4rem;color:var(--muted)}
.badge{font-size:.7rem;background:var(--chip);border-radius:99px;padding:1px 7px;color:var(--muted)}
.stats{display:flex;flex-wrap:wrap;gap:10px}.stat{flex:1 1 150px;background:var(--card);border:1px solid var(--line);
  border-radius:10px;padding:10px 12px}.stat b{display:block;font-size:1.4rem;font-variant-numeric:tabular-nums}
.stat span{font-size:.8rem;color:var(--muted)}.diagram{color:var(--ink);overflow-x:auto}
.diagram svg{display:block;width:100%;max-width:760px;min-width:520px;height:auto;margin:0 auto}
.picker{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}
.picker select{font:inherit;font-size:.85rem;padding:3px 6px;border-radius:7px;border:1px solid var(--line);
  background:var(--card);color:var(--ink);max-width:100%}
.tabs{display:flex;flex-wrap:wrap;gap:4px}.tabs button{font:inherit;font-size:.85rem;border:1px solid var(--line);
  background:var(--card);color:var(--ink);border-radius:7px;padding:4px 10px;cursor:pointer}
.tabs button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
#viewer{position:relative;width:100%;height:340px;border-radius:8px;overflow:hidden;background:var(--chip)}
.bar{display:flex;align-items:center;gap:8px;margin-top:8px}.bar input{flex:1;min-width:0}.bar button{font:inherit;
  border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:7px;padding:3px 10px;cursor:pointer}
#irc-plot{display:block;width:100%;height:90px;margin-top:6px;color:var(--ink)}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums}code,.mono{font-family:ui-monospace,Menlo,monospace;font-size:.8rem}
tr.pick{cursor:pointer}tr.pick:hover td{background:var(--chip)}tr.on td{background:var(--chip)}
.sw{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.files a{display:inline-block;margin:2px 8px 2px 0;color:var(--accent)}details pre{white-space:pre-wrap;font-size:.75rem;
  background:var(--chip);padding:10px;border-radius:8px;max-height:280px;overflow:auto}
.note{font-size:.8rem;color:var(--muted)}footer{margin-top:26px;font-size:.75rem;color:var(--muted)}
"""

_JS = r"""
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const rc = D.complexes.find((c) => c.role === 'reactant complex'), pc = D.complexes.find((c) => c.role === 'product complex');
const viewsOf = (ch) => {
  const v = [];
  if (rc && rc.xyz) v.push({key:'reactant-complex', label:'Reactant complex', xyz:[rc.xyz]});
  if (ch) ch.steps.forEach((s, k) => {
    const tag = ch.steps.length > 1 ? ` (step ${k + 1})` : '';
    v.push({key: k ? `ts${k + 1}` : 'ts', label: (ch.verified ? 'TS' : 'Path maximum') + tag, xyz:[s.ts_xyz]});
    if (s.irc) v.push({key: k ? `irc${k + 1}` : 'irc', label:'IRC' + tag, xyz:s.irc.frames, irc:s.irc});
  });
  if (pc && pc.xyz) v.push({key:'product-complex', label:'Product complex', xyz:[pc.xyz]});
  if (!v.length) for (const s of D.reactants.concat(D.products)) if (s.xyz) v.push({key:s.id + s.role, label:s.name, xyz:[s.xyz]});
  return v;
};
const box = document.getElementById('views');
if (typeof $3Dmol === 'undefined' || !viewsOf(D.channels[0]).length) { box.hidden = true; return; }
const v = $3Dmol.createViewer(document.getElementById('viewer'), {backgroundAlpha:0});
const tabs = document.getElementById('tabs'), sel = document.getElementById('channel');
const bar = document.getElementById('irc-bar'), slider = document.getElementById('irc-slider'), label = document.getElementById('irc-label');
const play = document.getElementById('irc-play'), plot = document.getElementById('irc-plot'), line = document.getElementById('irc-line');
let ch = D.channels[0] || null, views = [], cur = null, frame = 0, timer = null;
function draw(k, refit){
  v.removeAllModels(); v.addModel(cur.xyz[k], 'xyz'); v.setStyle({}, {stick:{radius:0.14}, sphere:{scale:0.25}});
  if (refit) v.zoomTo(); v.render();
  if (cur.irc){ const e = cur.irc.energy_kcal[k]; slider.value = k;
    label.textContent = `frame ${k + 1}/${cur.xyz.length}` + (e != null ? ` · ${e.toFixed(1)} kcal/mol` : '') + (k === cur.irc.ts_index ? ' · TS' : '');
    const m = document.getElementById('irc-mark'), x = (8 + 584 * k / Math.max(1, cur.xyz.length - 1)).toFixed(1);
    m.setAttribute('x1', x); m.setAttribute('x2', x); }
}
function stop(){ clearInterval(timer); timer = null; play.textContent = 'Play'; }
function show(i){
  cur = views[i]; stop();
  [...tabs.children].forEach((b, j) => b.classList.toggle('on', j === i));
  bar.hidden = !cur.irc; plot.toggleAttribute('hidden', !cur.irc);   // (an SVG element has no .hidden)
  if (cur.irc){
    slider.max = cur.xyz.length - 1;
    const ys = cur.irc.energy_kcal, ok = ys.filter((y) => y != null), lo = Math.min(...ok), hi = Math.max(...ok);
    line.setAttribute('points', ys.map((y, k) => y == null ? '' : `${(8 + 584 * k / Math.max(1, ys.length - 1)).toFixed(1)},${(6 + 78 * (1 - (y - lo) / ((hi - lo) || 1))).toFixed(1)}`).join(' '));
    document.getElementById('irc-note').textContent = `IRC energy from ${lo.toFixed(1)} to ${hi.toFixed(1)} kcal/mol (relative to the reactant side of its search)`;
  }
  document.getElementById('irc-note').hidden = !cur.irc;
  frame = cur.irc && cur.irc.ts_index != null ? cur.irc.ts_index : 0;
  draw(frame, true);
}
function pick(c, key){
  ch = c; views = viewsOf(ch); tabs.textContent = '';
  views.forEach((w, i) => { const b = document.createElement('button'); b.textContent = w.label; b.onclick = () => show(i); tabs.appendChild(b); });
  if (sel && ch) sel.value = ch.id;
  document.querySelectorAll('tr.pick').forEach((tr) => tr.classList.toggle('on', ch && tr.dataset.ch === ch.id));
  const want = views.findIndex((w) => w.key === key);
  show(want >= 0 ? want : Math.max(0, views.findIndex((w) => w.key === 'ts')));
}
if (sel) sel.onchange = () => pick(D.channels.find((c) => c.id === sel.value), cur && cur.key);
document.querySelectorAll('tr.pick').forEach((tr) => { tr.onclick = () => { pick(D.channels.find((c) => c.id === tr.dataset.ch), cur && cur.key); box.scrollIntoView({behavior:'smooth', block:'nearest'}); }; });
slider.oninput = () => { stop(); frame = +slider.value; draw(frame, false); };
play.onclick = () => { if (timer) { stop(); return; }
  play.textContent = 'Pause'; timer = setInterval(() => { frame = (frame + 1) % cur.xyz.length; draw(frame, false); }, 90); };
new ResizeObserver(() => { v.resize(); v.render(); }).observe(document.getElementById('viewer'));
// A link can open one channel and view: report.html#ch2-irc, #ch1-ts, #irc, #product-complex ...
const m = location.hash.slice(1).match(/^(?:(ch\d+)-?)?(.*)$/);
pick(D.channels.find((c) => c.id === m[1]) || D.channels.find((c) => c.id === D.lowest) || D.channels[0] || null, m[2]);
// Files: each structure as its own download (works when the page is opened from disk).
const files = document.getElementById('files');
const add = (name, text) => { const a = document.createElement('a'); a.textContent = name; a.href = URL.createObjectURL(new Blob([text], {type:'chemical/x-xyz'})); a.download = name; files.appendChild(a); };
for (const s of D.reactants.concat(D.products, D.complexes)) if (s.xyz) add(`${s.role.replace(/ /g,'_')}_${s.name.replace(/[^A-Za-z0-9._+-]+/g,'_')}.xyz`, s.xyz);
for (const c of D.channels) c.steps.forEach((s, k) => { const tag = c.steps.length > 1 ? `_step${k + 1}` : '';
  add(`ts_${c.id}${tag}.xyz`, s.ts_xyz); if (s.irc) add(`irc_${c.id}${tag}.xyz`, s.irc.frames.join('')); });
})();
"""


def export_html(b: dict) -> str:
    """One self-contained page: no server and no internet needed."""
    e = html.escape
    kcal = lambda v: "—" if v is None else f"{v:+.1f}"
    hartree = lambda v: "" if v is None else f"{v:.8f}"

    def species(sp: dict) -> str:
        img = sp.get("svg") or f'<span class="mono">{e(sp.get("smiles") or sp["name"])}</span>'
        tag = ' <span class="badge">shuttle</span>' if sp.get("shuttle") else ""
        count = f"{sp['count']} × " if sp["count"] > 1 else ""
        return (f'<div class="sp"><div class="img">{img}</div><div class="nm">'
                f'{count}{e(sp["name"])}{tag}</div></div>')
    side = lambda xs: '<span class="op">+</span>'.join(species(sp) for sp in xs)
    chans = b["channels"]
    lowest = next((c for c in chans if c["id"] == b.get("lowest")), None)
    stats = []
    if lowest is not None and lowest["barrier_kcal"] is not None:
        stats.append((f'{lowest["barrier_kcal"]:.1f}', f'lowest ΔE‡, kcal/mol from the reactant complex ({lowest["name"]})'))
    if chans:
        stats.append((str(len(chans)), "channels found" + (f" · {b['offtarget']} more led elsewhere" if b["offtarget"] else "")))
    if b.get("delta_e_kcal") is not None:
        stats.append((f'{b["delta_e_kcal"]:+.1f}', "ΔE kcal/mol, separated products − reactants"))
    if b.get("seen"):
        stats.append((str(b["seen"]["count"] or 0), "times seen in the MD"
                      + (f' (reverse {b["seen"]["reverse"]}×)' if b["seen"].get("reverse") else "")))
    hue = lambda n: f'<span class="sw" style="background:{_HUES_LIGHT[(n - 1)] if n <= len(_HUES_LIGHT) else "#888"}"></span>' \
        if n <= MAX_DIAGRAM_CHANNELS else '<span class="sw"></span>'
    steps_of = lambda c: f" · {len(c['steps'])} steps" if len(c["steps"]) > 1 else ""
    ch_rows = "".join(
        f'<tr class="pick" data-ch="{c["id"]}"><td>{hue(c["n"])}{e(c["name"])}'
        f'{" <span class=badge>lowest</span>" if lowest and c["id"] == lowest["id"] else ""}</td>'
        f'<td>{e(c["kind"])}{steps_of(c)}</td>'
        f'<td class="num">{kcal(c["barrier_kcal"])}</td><td class="num">{kcal(c.get("barrier_search_kcal"))}</td>'
        f'<td class="num">{kcal(c["ts_kcal"])}</td>'
        f'<td>{"yes" if c["verified"] else "no"}</td><td>{e(c.get("level") or "")}</td>'
        f'<td>{e(c.get("job_title") or "")}{" *" if c["placed"] != "absolute" else ""}</td></tr>' for c in chans)
    approx = any(c["placed"] != "absolute" for c in chans)
    ch_table = (f'<div class="card tablewrap"><table><thead><tr><th>Channel</th><th>Kind</th>'
                f'<th class="num">ΔE‡ from RC</th><th class="num">ΔE‡ (its search)</th><th class="num">TS vs reactants</th>'
                f'<th>IRC-verified</th><th>Level</th>'
                f'<th>Found by</th></tr></thead><tbody>{ch_rows}</tbody></table></div>'
                f'<p class="note">kcal/mol. ΔE‡ from RC: from this reaction\'s reactant complex (the diagram); its search: '
                f'from the lowest reactant geometry that search found, as its result page reports it. '
                f'Click a channel to see its TS and IRC.'
                + (" * Placed from its search's own reference (another level of theory, or no reactant energies): "
                   "compare it with the others with care." if approx else "")
                + (f" {b['offtarget']} more saddle point(s) the searches found lead to other products and are not "
                   "included." if b["offtarget"] else "") + "</p>") if chans else \
        '<p class="note">No TS search on this reaction yet.</p>'
    kcal2 = lambda v: "—" if v is None else f"{v:+.2f}"
    rows = "".join(f'<tr><td>{e(r["label"])}</td><td class="num">{kcal2(r["kcal"])}</td></tr>' for r in _levels(b))
    sp_rows = "".join(
        f'<tr><td>{e(sp["name"])}</td><td>{e(sp["role"])}</td><td class="mono">{e(sp.get("smiles") or "")}</td>'
        f'<td class="num">{hartree(sp.get("energy_hartree"))}</td>'
        f'<td>{e(sp.get("level") or "")}</td></tr>' for sp in b["reactants"] + b["products"] + b["complexes"])
    levels = sorted({x for x in [c.get("level") for c in chans] + [sp.get("level") for sp in b["reactants"] + b["products"]] if x})
    title = b["label"].replace("->", "→")
    data = json.dumps(_json_data(b), default=str).replace("</", "<\\/")
    three = _read(_VENDOR) or ""
    picker = (f'<label class="note" for="channel">Channel</label><select id="channel">'
              + "".join(f'<option value="{c["id"]}">{e(c["name"])} · {e(c["kind"])} · ΔE‡ {kcal(c["barrier_kcal"])}</option>'
                        for c in chans) + "</select>") if len(chans) > 1 else ""
    profiles = "".join(f'<details><summary class="note">The mepd profile of {e(p["job_title"] or p["job"])}</summary>'
                       f'<pre>{e(p["toml"])}</pre></details>' for p in b["profiles"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)} · mepd</title><meta name="generator" content="mepd {e(b["mepd_version"])}">
<style>{_CSS}{_hue_style().replace(".c", ".diagram .c").replace(".f", ".diagram .f")}</style></head><body><main>
<h1>{e(title)}</h1>
<div class="meta">{e(b["source"])}{f" · sampled in the MD as {e(b['relaxed_from'])}" if b.get("relaxed_from") else ""}
 · level of theory: {e(", ".join(levels) or "not recorded")}</div>
<div class="eq">{side(b["reactants"])}<span class="op">→</span>{side(b["products"])}</div>
<div class="stats">{"".join(f'<div class="stat"><b>{v}</b><span>{e(t)}</span></div>' for v, t in stats)}</div>
<h2>Energy diagram</h2>
<div class="card diagram">{energy_diagram_svg(b)}</div>
<p class="note">Electronic energies (no ZPE or thermal corrections), kcal/mol from the separated reactants, all over the
 same atoms; one transition state per channel (a multi-step channel: its highest step).</p>
<h2>Channels</h2>{ch_table}
<section id="views"><h2>Structures</h2><div class="card"><div class="picker">{picker}<div class="tabs" id="tabs"></div></div>
<div id="viewer"></div>
<div class="bar" id="irc-bar" hidden><button id="irc-play">Play</button><input type="range" id="irc-slider" min="0" value="0">
<span class="note" id="irc-label"></span></div>
<svg id="irc-plot" viewBox="0 0 600 90" preserveAspectRatio="none" hidden><polyline id="irc-line" fill="none"
 stroke="currentColor" stroke-width="1.5" vector-effect="non-scaling-stroke" points=""/><line id="irc-mark" x1="8" x2="8"
 y1="0" y2="90" stroke="#c2410c" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>
<p class="note" id="irc-note" hidden></p></div></section>
<h2>Energies</h2>
<div class="card tablewrap"><table><thead><tr><th>Point</th><th class="num">kcal/mol</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="card tablewrap" style="margin-top:10px"><table><thead><tr><th>Structure</th><th>Role</th><th>SMILES</th><th class="num">Energy (Eh)</th><th>Level</th></tr></thead>
<tbody>{sp_rows}</tbody></table></div>
<h2>Files</h2><div class="files" id="files"></div>{profiles}
<footer>Exported from mepd {e(b["mepd_version"])} on {e(b["exported"])}. 3D viewer: 3Dmol.js (BSD-3-Clause).</footer>
</main>
<script type="application/json" id="data">{data}</script>
<script>{three}</script>
<script>{_JS}</script>
</body></html>
"""
