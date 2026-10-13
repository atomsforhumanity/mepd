"""A whole workspace's results as files (Analyze › Reactions › Export all).

Two forms of the same data: every reaction Analyze lists (the workspace's
reactions, and each edge between two species that a search ran on), each
as `reaction_export.reaction_bundle` builds it.

- `export_html`: one self-contained page to browse them: an index of every
  reaction (lowest barrier, reaction energy, channels, level of theory),
  each opening its own reaction page (the per-reaction export: energy
  diagram, every channel's TS and IRC in 3D). One copy of 3Dmol.js serves
  every page; no server and no internet needed. The CSV is in it too.
- `export_csv`: one row per elementary step of every channel found, plus one
  row for each reaction with none, in fixed columns with units in their
  names -- for loading into other tools in bulk. Geometries are inline (an
  element list and flat Angstrom coordinates, the TS and both ends of its step in
  one atom order), so a row stands alone.
"""
from __future__ import annotations

import csv
import html
import io
import json
import time
from pathlib import Path
from typing import Callable, Optional

from mepd.web import reaction_export as rx

KCAL = rx.KCAL
_SEARCH_OPS = rx._SEARCH_OPS
_VIEWER_SLOT = "/*__MEPD_3DMOL__*/"

# Fixed order: ingestion code can rely on it. Add new columns at the end.
CSV_COLUMNS = [
    "reaction_id", "reaction_label", "reaction_smiles", "reactant_smiles", "product_smiles",
    "n_reactant_molecules", "n_product_molecules", "charge", "multiplicity",
    "status", "n_channels", "n_offtarget",
    "channel_rank", "channel_kind", "channel_is_lowest", "channel_irc_verified", "channel_n_steps",
    "channel_barrier_kcal_mol", "channel_barrier_search_kcal_mol", "channel_ts_rel_reactants_kcal_mol",
    "channel_energy_placement",
    "step_index", "step_is_highest", "step_ts_energy_hartree",
    "step_forward_barrier_kcal_mol", "step_reverse_barrier_kcal_mol",
    "step_start_energy_hartree", "step_end_energy_hartree",
    "reaction_energy_kcal_mol", "complex_reaction_energy_kcal_mol",
    "reactants_energy_hartree", "products_energy_hartree",
    "level_of_theory", "level_key",
    "elements", "ts_coords_angstrom", "step_start_coords_angstrom", "step_end_coords_angstrom",
    "found_by_job", "mepd_version", "exported_at",
]
CSV_COLUMN_HELP = {
    "status": "channels: at least one IRC-verified channel; unverified: only path maxima / unverified TSs; "
              "not_searched: no finished search",
    "channel_barrier_kcal_mol": "from this reaction's reactant complex (or the reactant, unimolecular)",
    "channel_barrier_search_kcal_mol": "as the search reports it: from the lowest reactant geometry it found",
    "channel_ts_rel_reactants_kcal_mol": "the channel's highest TS against the separated reactants",
    "channel_energy_placement": "absolute: TS and reactants at one level; search_reference: from the search's barrier",
    "step_forward_barrier_kcal_mol": "this step's TS minus its start (empty without one)",
    "step_start_coords_angstrom": "the step's start: its IRC end on the reactant side (verified), else the path's "
                                  "first point (a path maximum); step_end likewise",
    "reaction_energy_kcal_mol": "separated products minus separated reactants",
    "elements": "space-separated, the atom order of all three geometries",
    "ts_coords_angstrom": "x y z per atom, space-separated, flattened",
}


def workspace_reaction_ids(snap: dict, jobs: dict) -> list[str]:
    """What Analyze lists: every reaction, then each edge no reaction stands
    for whose two ends are species and that a search ran on (or that carries
    a barrier of its own)."""
    reactions = snap.get("reactions") or {}
    of_edge = {r.get("edge") for r in reactions.values() if r.get("edge")}
    S = snap["structures"]
    species = lambda sid: sid in S and S[sid].get("role") not in ("ts", "complex")
    searched = {e for j in jobs.values() if j["op"] in _SEARCH_OPS for e in (j.get("targets") or {}).get("edges", [])}
    edges = [e for e in (snap.get("edges") or {}).values()
             if e["id"] not in of_edge and species(e["source"]) and species(e["target"])
             and (e["id"] in searched or (e.get("origin") or {}).get("barrier_kcal") is not None
                  or (e.get("origin") or {}).get("qmmm_ts"))]
    by_time = lambda x: x.get("created") or 0
    return [r["id"] for r in sorted(reactions.values(), key=by_time)] + \
        [f"edge:{e['id']}" for e in sorted(edges, key=by_time)]


def workspace_bundles(ws, jobs: dict, *, job_dir: Callable[[str], Path], result_of: Callable[[dict], dict],
                      page_of: Callable[[dict], dict] = lambda j: j) -> dict:
    """Every listed reaction's bundle, plus the workspace's own facts."""
    snap = ws.snapshot()
    bundles = []
    for rid in workspace_reaction_ids(snap, jobs):
        b = rx.reaction_bundle(ws, jobs, rid, job_dir=job_dir, result_of=result_of, page_of=page_of)
        bundles.append(b)
    return {"name": Path(ws.root).name, "reactions": bundles,
            "exported": time.strftime("%Y-%m-%d %H:%M %Z"), "mepd_version": rx._mepd_version()}


# ------------------------------------------------------------------ CSV

def _geometry(xyz: Optional[str]) -> tuple[list[str], str]:
    """(elements, flat coordinates) of an xyz text."""
    if not xyz:
        return [], ""
    lines = xyz.strip().splitlines()
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return [], ""
    els, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        if len(parts) < 4:
            return [], ""
        els.append(parts[0])
        coords += parts[1:4]
    return els, " ".join(coords)


def _smiles(side: list) -> str:
    return ".".join(sp.get("smiles") or "" for sp in side for _ in range(sp.get("count") or 1))


def _num(v, fmt: str = ".4f") -> str:
    return "" if v is None else format(v, fmt)


def _total(side: list, key: str):
    vals = [(sp.get(key), sp.get("count") or 1) for sp in side]
    return None if any(v is None for v, _ in vals) else sum(v * n for v, n in vals)


def _charge_mult(b: dict) -> tuple:
    """The whole system's charge and spin multiplicity: the reactant complex's
    when there is one, else the reactants' (charges add; one molecule's
    multiplicity, else unknown)."""
    if b["complexes"]:
        c = b["complexes"][0]
        return c.get("charge"), c.get("multiplicity")
    charge = _total(b["reactants"], "charge")
    n = sum(sp.get("count") or 1 for sp in b["reactants"])
    return charge, (b["reactants"][0].get("multiplicity") if n == 1 else None)


def _delta_e(b: dict):
    """Separated products minus separated reactants (kcal/mol): the reaction's
    own, else its energy ladder's (all species at one level)."""
    if b.get("delta_e_kcal") is not None:
        return b["delta_e_kcal"]
    L = b.get("ladder") or {}
    return None if L.get("products") is None or L.get("reactants") is None else L["products"] - L["reactants"]


def _status(b: dict) -> str:
    if any(c["verified"] for c in b["channels"]):
        return "channels"
    return "unverified" if b["channels"] else "not_searched"


def csv_rows(b: dict, *, exported: str, version: str) -> list[dict]:
    charge, mult = _charge_mult(b)
    base = {
        "reaction_id": b["id"], "reaction_label": b["label"].replace("->", "→"),
        "reaction_smiles": f"{_smiles(b['reactants'])}>>{_smiles(b['products'])}",
        "reactant_smiles": _smiles(b["reactants"]), "product_smiles": _smiles(b["products"]),
        "n_reactant_molecules": sum(sp.get("count") or 1 for sp in b["reactants"]),
        "n_product_molecules": sum(sp.get("count") or 1 for sp in b["products"]),
        "charge": "" if charge is None else charge, "multiplicity": "" if mult is None else mult,
        "status": _status(b), "n_channels": len(b["channels"]), "n_offtarget": b.get("offtarget") or 0,
        "reaction_energy_kcal_mol": _num(_delta_e(b)),
        "complex_reaction_energy_kcal_mol": _num(b.get("complex_delta_e_kcal")),
        "reactants_energy_hartree": _num(_total(b["reactants"], "energy_hartree"), ".10f"),
        "products_energy_hartree": _num(_total(b["products"], "energy_hartree"), ".10f"),
        "mepd_version": version, "exported_at": exported,
    }
    if not b["channels"]:
        return [{k: base.get(k, "") for k in CSV_COLUMNS}]
    rows = []
    for c in b["channels"]:
        highest = max(range(len(c["steps"])), key=lambda k: c["steps"][k]["ts_hartree"]
                      if c["steps"][k]["ts_hartree"] is not None else float("-inf"))
        for k, st in enumerate(c["steps"]):
            els, ts = _geometry(st["ts_xyz"])
            irc = st.get("irc") or {}
            frames, eh = irc.get("frames") or [], irc.get("energy_hartree") or []
            e0, e1 = (eh[0], eh[-1]) if len(eh) > 1 else (None, None)
            ets = st["ts_hartree"]
            row = {**base,
                   "channel_rank": c["n"], "channel_kind": c["kind"].replace(" ", "_"),
                   "channel_is_lowest": int(c["id"] == b.get("lowest")), "channel_irc_verified": int(bool(c["verified"])),
                   "channel_n_steps": len(c["steps"]),
                   "channel_barrier_kcal_mol": _num(c.get("barrier_kcal")),
                   "channel_barrier_search_kcal_mol": _num(c.get("barrier_search_kcal")),
                   "channel_ts_rel_reactants_kcal_mol": _num(c.get("ts_kcal")),
                   "channel_energy_placement": (c.get("placed") or "").replace(" ", "_"),
                   "step_index": k, "step_is_highest": int(k == highest),
                   "step_ts_energy_hartree": _num(ets, ".10f"),
                   "step_forward_barrier_kcal_mol": _num((ets - e0) * KCAL if None not in (ets, e0) else None),
                   "step_reverse_barrier_kcal_mol": _num((ets - e1) * KCAL if None not in (ets, e1) else None),
                   "step_start_energy_hartree": _num(e0, ".10f"), "step_end_energy_hartree": _num(e1, ".10f"),
                   "level_of_theory": c.get("level") or "", "level_key": c.get("level_key") or "",
                   "elements": " ".join(els), "ts_coords_angstrom": ts,
                   "step_start_coords_angstrom": _geometry(frames[0])[1] if len(frames) > 1 else "",
                   "step_end_coords_angstrom": _geometry(frames[-1])[1] if len(frames) > 1 else "",
                   "found_by_job": c.get("job") or ""}
            rows.append({k: row.get(k, "") for k in CSV_COLUMNS})
    return rows


def export_csv(w: dict) -> str:
    buf = io.StringIO()
    out = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    out.writeheader()
    for b in w["reactions"]:
        out.writerows(csv_rows(b, exported=w["exported"], version=w["mepd_version"]))
    return buf.getvalue()


# ------------------------------------------------------------------ HTML

def _index_row(b: dict) -> dict:
    lowest = next((c for c in b["channels"] if c["id"] == b.get("lowest")), None)
    levels = sorted({c.get("level") for c in b["channels"] if c.get("level")})
    return {"id": b["id"], "label": b["label"].replace("->", "→"),
            "smiles": f"{_smiles(b['reactants'])}>>{_smiles(b['products'])}",
            "barrier": lowest["barrier_kcal"] if lowest else None,
            "unverified": None if lowest or not b["channels"] else
            min((c["barrier_kcal"] for c in b["channels"] if c["barrier_kcal"] is not None), default=None),
            "de": _delta_e(b), "channels": len(b["channels"]), "offtarget": b.get("offtarget") or 0,
            "status": _status(b), "levels": ", ".join(levels), "source": b.get("source") or ""}


def export_html(w: dict) -> str:
    """One self-contained page: the index, and each reaction's page on click."""
    e = html.escape
    pages = {b["id"]: rx.export_html(b, viewer_js=_VIEWER_SLOT) for b in w["reactions"]}
    rows = [_index_row(b) for b in w["reactions"]]
    data = json.dumps({"rows": rows, "pages": pages, "csv": export_csv(w),
                       "name": w["name"], "slot": _VIEWER_SLOT}).replace("</", "<\\/")
    three = (rx._read(rx._VENDOR) or "").replace("</script", "<\\/script")
    n = len(rows)
    n_ok = sum(r["status"] == "channels" for r in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(w["name"])} · mepd</title><meta name="generator" content="mepd {e(w["mepd_version"])}">
<style>{rx._CSS}{_CSS}</style></head><body>
<main id="index">
<h1>{e(w["name"])}</h1>
<div class="meta">{n} reaction{"" if n == 1 else "s"} · {n_ok} with an IRC-verified channel ·
 exported from mepd {e(w["mepd_version"])} on {e(w["exported"])}</div>
<div class="tools"><input id="q" type="search" placeholder="Filter by name or SMILES" aria-label="Filter reactions">
<select id="only" aria-label="Show"><option value="">All</option><option value="channels">With a verified channel</option>
<option value="unverified">Unverified only</option><option value="not_searched">Not searched</option></select>
<button id="csv" title="One row per elementary step of every channel, fixed columns with units: for bulk loading">Download CSV</button></div>
<div class="card tablewrap"><table id="t"><thead><tr>
<th data-k="label">Reaction</th><th data-k="barrier" class="num">Lowest ΔE‡</th><th data-k="de" class="num">ΔE</th>
<th data-k="channels" class="num">Channels</th><th data-k="levels">Level</th><th data-k="source">Found by</th>
</tr></thead><tbody></tbody></table></div>
<p class="note">kcal/mol. Lowest ΔE‡: the lowest IRC-verified channel, from the reactant complex (≈ and a question mark:
 only unverified barriers). ΔE: separated products − reactants. Electronic energies, no ZPE or thermal corrections.
 Click a reaction to open its page; click a column to sort.</p>
</main>
<div id="page" hidden><div class="pagebar"><button id="back" title="All reactions">← All</button><span id="pagetitle"></span>
<span class="grow"></span><button id="prev" title="Previous reaction">‹</button><button id="next" title="Next reaction">›</button></div>
<iframe id="frame" title="Reaction"></iframe></div>
<script type="application/json" id="data">{data}</script>
<script type="text/plain" id="three">{three}</script>
<script>{_JS}</script>
</body></html>
"""


_CSS = """
.tools{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 10px}
.tools input{flex:1 1 220px;min-width:0}
.tools input,.tools select,.tools button,.pagebar button{font:inherit;padding:5px 9px;border:1px solid var(--line);
 border-radius:4px;background:var(--card);color:var(--ink)}
.tools button,.pagebar button{cursor:pointer}
#t tbody tr{cursor:pointer}#t tbody tr:hover{background:var(--line)}
#t th[data-k]{cursor:pointer;user-select:none}
#t .sm{display:block;font-size:11px;color:var(--muted);word-break:break-all}
#t td,#t th{white-space:normal}#t td:first-child{overflow-wrap:anywhere;min-width:12em}#t td.num,#t th.num{white-space:nowrap}
#page{position:fixed;inset:0;display:flex;flex-direction:column;background:var(--bg)}
#page[hidden]{display:none}
.pagebar{display:flex;align-items:center;gap:8px;padding:8px 16px;border-bottom:1px solid var(--line)}
.pagebar .grow{flex:1}#pagetitle{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#frame{flex:1;width:100%;border:0;background:var(--bg)}
"""

_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const three = document.getElementById('three').textContent;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const kcal = (v) => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(1);
let sortK = null, sortDir = 1, shown = D.rows;
function barrierCell(r) {
  if (r.barrier != null) return kcal(r.barrier);
  if (r.unverified != null) return `<span title="not IRC-verified">≈${kcal(r.unverified)}?</span>`;
  return r.channels ? '—' : '<span class="note">not searched</span>';
}
function render() {
  const q = $('q').value.toLowerCase(), only = $('only').value;
  shown = D.rows.filter((r) => (!only || r.status === only) && (!q || (r.label + ' ' + r.smiles).toLowerCase().includes(q)));
  if (sortK) {
    const v = (r) => sortK === 'barrier' ? (r.barrier ?? r.unverified) : r[sortK];
    shown = [...shown].sort((a, b) => {
      const x = v(a), y = v(b);
      if (x == null || x === '') return 1; if (y == null || y === '') return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
    });
  }
  $('t').tBodies[0].innerHTML = shown.map((r) => `<tr data-id="${esc(r.id)}"><td>${esc(r.label)}<span class="sm mono">${esc(r.smiles)}</span></td>
    <td class="num">${barrierCell(r)}</td><td class="num">${kcal(r.de)}</td>
    <td class="num">${r.channels}${r.offtarget ? ` <span class="note" title="searches that led to other products">+${r.offtarget}</span>` : ''}</td>
    <td>${esc(r.levels)}</td><td>${esc(r.source)}</td></tr>`).join('') ||
    '<tr><td colspan="6" class="note">No reaction matches.</td></tr>';
}
function open(id) {
  const page = D.pages[id];
  if (!page) { show(null); return; }
  const r = D.rows.find((x) => x.id === id);
  $('pagetitle').textContent = r ? r.label : id;
  $('frame').srcdoc = page.split(D.slot).join(three);
  $('index').hidden = true; $('page').hidden = false;
  const k = shown.findIndex((x) => x.id === id);
  $('prev').disabled = k <= 0; $('next').disabled = k < 0 || k >= shown.length - 1;
}
function show(id) {
  if (id) { open(id); } else { $('page').hidden = true; $('index').hidden = false; $('frame').removeAttribute('srcdoc'); }
}
const route = () => show(decodeURIComponent((location.hash.match(/^#r=(.+)$/) || [])[1] || '') || null);
const go = (id) => { location.hash = id ? 'r=' + encodeURIComponent(id) : ''; };
$('t').tBodies[0].addEventListener('click', (e) => { const tr = e.target.closest('tr[data-id]'); if (tr) go(tr.dataset.id); });
$('t').tHead.addEventListener('click', (e) => {
  const k = e.target.closest('th')?.dataset.k; if (!k) return;
  sortDir = sortK === k ? -sortDir : 1; sortK = k; render();
});
$('q').addEventListener('input', render); $('only').addEventListener('change', render);
$('back').addEventListener('click', () => go(null));
const step = (d) => { const id = decodeURIComponent(location.hash.slice(3)); const k = shown.findIndex((x) => x.id === id);
  if (shown[k + d]) go(shown[k + d].id); };
$('prev').addEventListener('click', () => step(-1)); $('next').addEventListener('click', () => step(1));
$('csv').addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([D.csv], {type: 'text/csv'}));
  a.download = (D.name || 'workspace') + '_reactions.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
window.addEventListener('hashchange', route);
render(); route();
"""
