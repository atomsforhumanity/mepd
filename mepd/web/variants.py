"""Network variants in a workspace (mepd/network_variants.py): edit a
species in Design, carry the edit through every reaction of the network,
recompute each affected reaction (`mepd variant`: a fast estimate, then a
re-optimization), and model the variant's kinetics next to the parent's.

A variant is a layer of the workspace, not a copy: its species, complexes,
edges and reactions are ordinary records tagged `variant: <id>` (and
`parent: <the record they stand for>`), so every operation runs on them as
on any other. `workspace["variants"][<id>]` keeps the edit, which parent
species and reactions it replaces, and the reactions it could not carry.

* The parent network is every record without a `variant` tag.
* A variant's network is the parent's, minus the species it replaces and
  the reactions those take part in, plus its own records (`network_view`).

Geometries of a parent reaction, in one atom order: the TS that kinetics
uses for its edge (that job's IRC through it: both ends and the TS), else
the edge's two ends (the pinned conformers). Two ends from different
sources may number their atoms differently: when the bonds that change
between them look implausible (more than `MAX_BOND_CHANGES`), SLAPMapper
renumbers the end to match the start, and the reaction says so.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from mepd import network_variants as NV
from mepd.web.workspace import WorkspaceError, is_species, new_id

BOHR = 0.529177210903
MAX_BOND_CHANGES = 6


def _angstrom(structure) -> np.ndarray:
    return np.asarray(structure.geometry, dtype=float) * BOHR


def _read_xyz(fp: Path):
    from qcdata import Structure

    return Structure.open(str(fp))


def _frames_from_xyz(text: str):
    lines = text.strip().splitlines()
    n = int(lines[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def _graph(symbols, coords) -> str:
    return NV.molecule_key(symbols, coords, NV.adjacency(symbols, coords))[0]


def _end_structure(ws, snap: dict, sid: str, edge: dict):
    """The geometry a job on this edge uses for one end: the pinned conformer, else the lowest."""
    cid = (edge.get("conformers") or {}).get(sid)
    fp = ws.conformer_path(sid, cid) if cid else ws.structure_path(sid)
    return _read_xyz(fp)


def _ts_frames(ws, jobs: dict, edge: dict, level_key: Optional[str]):
    """(r, ts, p frames as (symbols, Å), energies, job id) of the TS kinetics
    uses for this edge, oriented source -> target; None if it has none
    (or its IRC frames are not available)."""
    from mepd.web import results as R
    from mepd.web.kinetics import edge_ts

    e_ts, verified, jid, _ = edge_ts(edge, jobs, level_key, False)
    if e_ts is None or jid is None or jid not in jobs:
        return None
    # The TS's IRC may be in another job on this edge with the same TS (a
    # TS search's "Sample more paths" runs keep their paths in their own
    # folders): look in all of them, the one kinetics named first.
    family = [jid] + [k for k, j in jobs.items() if k != jid and edge["id"] in j["targets"]["edges"]
                      and j["status"] == "done" and (j.get("level") or {}).get("key") == level_key]
    best = None
    for k_job in family:
        try:
            result = R.collect_cached(jobs[k_job], ws.jobs_dir / k_job)
        except Exception:
            continue
        for g in result.get("groups") or []:
            for e in g.get("entries") or []:
                fr = e.get("frames") or []
                k = e.get("ts_index")
                if len(fr) < 3 or k is None or not 0 < k < len(fr) - 1:
                    continue
                en = fr[k].get("energy_hartree")
                if en is None or abs(en - e_ts) > 1e-5:
                    continue
                best = (fr[0], fr[k], fr[-1])
                jid = k_job
                break
            if best:
                break
        if best:
            break
    if best is None:
        return None
    frames = [_frames_from_xyz(f["xyz"]) for f in best]
    energies = [f.get("energy_hartree") for f in best]
    sym = frames[0][0]
    if any(f[0] != sym for f in frames):
        return None
    # Orient: the IRC's first frame is the edge's source side.
    snap = ws.snapshot()
    src = _end_structure(ws, snap, edge["source"], edge)
    tgt = _end_structure(ws, snap, edge["target"], edge)
    g_src, g_tgt = _graph(list(src.symbols), _angstrom(src)), _graph(list(tgt.symbols), _angstrom(tgt))
    g0, g1 = _graph(sym, frames[0][1]), _graph(sym, frames[2][1])
    if (g0, g1) == (g_tgt, g_src) and (g0, g1) != (g_src, g_tgt):
        frames, energies = frames[::-1], energies[::-1]
    elif (g0, g1) != (g_src, g_tgt):
        return None
    return frames, energies, jid


def _bond_changes(sym, a, b) -> int:
    return int((NV.adjacency(sym, a) != NV.adjacency(sym, b)).sum() // 2)


def _end_frames(ws, snap: dict, edge: dict):
    """(r, p frames, note) from the edge's two ends; the end renumbered by
    SLAPMapper when its order does not fit the start's."""
    s = _end_structure(ws, snap, edge["source"], edge)
    t = _end_structure(ws, snap, edge["target"], edge)
    sym_s, sym_t = list(s.symbols), list(t.symbols)
    xs, xt = _angstrom(s), _angstrom(t)
    if sym_s == sym_t and _bond_changes(sym_s, xs, xt) <= MAX_BOND_CHANGES:
        return (sym_s, xs), (sym_t, xt), ""
    from mepd import atom_mapping as AM

    if not AM.HAS_SLAPMAPPER:
        raise WorkspaceError("its two ends number their atoms differently and SLAPMapper (mepd[aam]) is not "
                             "installed to match them")
    m = AM.suggest_atom_mapping(s, t)
    if m is None:
        raise WorkspaceError("SLAPMapper could not match the atoms of its two ends")
    t2 = AM.realign_end_to_start(m, t)
    sym2, x2 = list(t2.symbols), _angstrom(t2)
    if sym2 != sym_s:
        raise WorkspaceError("its two ends do not have the same atoms")
    return (sym_s, xs), (sym2, x2), "remapped"


def parent_network(ws, jobs: dict, level_key: Optional[str], network: Optional[str] = None):
    """(species, reactions, info, problems) of the parent network (or of
    the variant `network`, for an edit on top of it), as
    mepd.network_variants takes them. `info[rid]`: the edge, the parent TS
    job and its frames' energies."""
    from mepd.web.kinetics import edge_ts

    snap = network_view(ws.snapshot(), network)
    st = snap["structures"]
    species = {}
    for sid, rec in st.items():
        if not is_species(rec) or rec.get("qmmm"):
            continue
        try:
            s = ws.load_structure(sid)
        except Exception:
            continue
        species[sid] = NV.Species(sid, list(s.symbols), _angstrom(s), int(rec["charge"]), int(rec["multiplicity"]),
                                  name=rec.get("name") or sid)
    reactions, info, problems = [], {}, {}

    def add(rid, edge, reactants, products, label):
        if any(s not in species for s in reactants + products):
            problems[rid] = "a species of it is not a plain molecule (QM/MM, or missing)"
            return
        a = st.get(edge["source"]) or {}
        frames = _ts_frames(ws, jobs, edge, level_key)
        note, energies, jid = "", None, None
        try:
            if frames is not None:
                (r, ts, p), energies, jid = frames
            else:
                r, p, note = _end_frames(ws, snap, edge)
                ts = None
                if edge_ts(edge, jobs, level_key, False)[0] is not None:
                    # Kinetics has a TS for it, but its IRC does not run between these ends.
                    note = "; ".join(x for x in (note, "the parent's TS is one step of a longer route") if x)
        except WorkspaceError as exc:
            problems[rid] = str(exc)
            return
        reactions.append(NV.Reaction(rid, list(reactants), list(products), r=r, ts=ts, p=p,
                                     charge=int(a.get("charge", 0)), multiplicity=int(a.get("multiplicity", 1)),
                                     label=label, note=note))
        info[rid] = {"edge": edge["id"], "job": jid, "energies": energies, "label": label, "note": note}

    for rid, r in (snap.get("reactions") or {}).items():
        edge = snap["edges"].get(r.get("edge") or "")
        if edge is None:
            problems[rid] = "no TS endpoints"
            continue
        add(rid, edge, r["reactants"], r["products"], r.get("label") or rid)
    complexes = {sid for sid, rec in st.items() if rec.get("role") == "complex"}
    for eid, e in snap["edges"].items():
        if e["source"] in complexes or e["target"] in complexes:
            continue
        if not (is_species(st.get(e["source"])) and is_species(st.get(e["target"]))):
            continue
        add(f"edge:{eid}", e, [e["source"]], [e["target"]],
            f"{st[e['source']]['name']} -> {st[e['target']]['name']}")
    return species, reactions, info, problems


def _design_edit(ws, design: dict) -> tuple[str, NV.Edit]:
    from mepd.web import design as D

    src = design.get("source") or {}
    if src.get("kind") != "graph" or not src.get("parent_molblock"):
        raise WorkspaceError("Apply to network needs a design opened from a species in Explore (Edit in Design)")
    base = src["structure"]
    rec = ws.structure(base)
    if not is_species(rec) or rec.get("qmmm"):
        raise WorkspaceError(f"{rec['name']} is not a plain molecule: only species can be edited across a network")
    if design.get("reaction"):
        raise WorkspaceError("Apply to network works on one molecule, not a reaction design")
    try:
        edit = NV.derive_edit(D.read(src["parent_molblock"]), D.read(design["molblock"]))
    except NV.VariantError as exc:
        raise WorkspaceError(str(exc)) from None
    return base, edit


def _on(ws, design: dict) -> Optional[str]:
    """The variant the design was opened in (its edit goes on top of it), or None."""
    on = (design.get("source") or {}).get("network") or None
    if on and on not in ws.variants:
        raise WorkspaceError("the variant this design was opened in no longer exists: load the species again")
    return on


def plan(ws, jobs: dict, design: dict, level_key: Optional[str]):
    base, edit = _design_edit(ws, design)
    species, reactions, info, problems = parent_network(ws, jobs, level_key, _on(ws, design))
    if base not in species:
        raise WorkspaceError("the edited species is not part of the network")
    try:
        p = NV.propagate(species, reactions, base, edit)
    except NV.VariantError as exc:
        raise WorkspaceError(str(exc)) from None
    return base, edit, p, species, reactions, info, problems


def preview(ws, jobs: dict, design: dict, level_key: Optional[str]) -> dict:
    """What Apply to network would do (no calculations): the new species,
    each parent reaction's variants and their status, the jobs it starts."""
    base, edit, p, species, reactions, info, problems = plan(ws, jobs, design, level_key)
    st = ws.snapshot()["structures"]
    labels = {}
    out_species = []
    for label, v in p.species.items():
        smi = v.smiles
        labels[label] = smi or label
        out_species.append({"label": label, "parent": v.parent, "parent_name": species[v.parent].name,
                            "smiles": smi, "charge": v.charge})
    rows = {}
    for vr in p.reactions:
        row = rows.setdefault(vr.parent, {"parent": vr.parent, "label": info.get(vr.parent, {}).get("label", vr.parent),
                                          "note": info.get(vr.parent, {}).get("note", ""), "variants": []})
        row["variants"].append({"status": vr.status, "why": vr.why, "placement": vr.placement,
                                "clash": round(vr.clash, 2),
                                "text": " + ".join(labels.get(x, st.get(x, {}).get("name", x)) for x in vr.reactants)
                                + " → " + " + ".join(labels.get(x, st.get(x, {}).get("name", x)) for x in vr.products)})
    touched = set(p.replaced)
    unreached = {rid: why for rid, why in {**problems, **p.problems}.items()
                 if rid not in rows and _involves(rid, reactions, touched, ws)}
    n_jobs = sum(1 for vr in p.reactions if vr.status != "blocked")
    on = _on(ws, design)
    return {"base": base, "base_name": species[base].name, "edit": edit.describe(), "species": out_species,
            "on": on, "on_name": ws.variants[on]["name"] if on else None,
            "reactions": list(rows.values()), "shared": [info.get(r, {}).get("label", r) for r in p.shared],
            "replaced": [species[s].name for s in p.replaced],
            "problems": [{"reaction": rid, "label": _label(ws, rid), "why": why} for rid, why in unreached.items()],
            "warnings": p.warnings, "jobs": n_jobs, "new_species": len(p.species)}


def _label(ws, rid: str) -> str:
    snap = ws.snapshot()
    if rid.startswith("edge:"):
        return rid
    return (snap.get("reactions", {}).get(rid) or {}).get("label") or rid


def _involves(rid, reactions, touched, ws) -> bool:
    for r in reactions:
        if r.id == rid:
            return bool(set(r.reactants + r.products) & touched)
    rec = (ws.snapshot().get("reactions") or {}).get(rid)
    return bool(rec and set(rec.get("reactants", []) + rec.get("products", [])) & touched)


# ------------------------------------------------------------------ creating the variant

def _structure(symbols, coords, charge, multiplicity):
    from qcdata import Structure

    return Structure(symbols=list(symbols), geometry=np.asarray(coords, dtype=float) / BOHR,
                     charge=int(charge), multiplicity=int(multiplicity))


def create(ws, jobs: dict, design: dict, level_key: Optional[str], *, name: str = "",
           vid: Optional[str] = None) -> dict:
    """Add the variant's records (or, with `vid`, the reactions an existing
    variant does not have yet) and return {variant, species to minimize,
    edges to compute}."""
    base, edit, p, species, reactions, info, problems = plan(ws, jobs, design, level_key)
    by_id = {r.id: r for r in reactions}
    old = (ws.variants.get(vid) if vid else None) or {}
    vid = vid or new_id("v_")
    folder = ws.root / "variants" / vid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "edit.json").write_text(json.dumps(edit.to_json()))
    known_keys = {r["key"] for rs in (old.get("reactions") or {}).values() for r in rs}
    sid_of = dict(old.get("labels") or {})
    new_species = []
    origin = {"kind": "variant", "variant": vid}

    # Species: the edited one and every one the edit comes out in.
    for label, v in p.species.items():
        key = json.dumps(v.key)
        if key in sid_of:
            continue
        st = _structure(v.symbols, v.coords, v.charge, v.multiplicity)
        made = ws.add_or_merge(st, smiles=v.smiles, origin={**origin, "parent": v.parent,
                                                            "label": f"{species[v.parent].name} ({name or vid})"})
        rec = made["rec"]
        if not made["merged"]:
            ws.tag_variant([rec["id"]], vid, {rec["id"]: v.parent})
            new_species.append(rec["id"])
        sid_of[key] = rec["id"]
    label_sid = {label: sid_of[json.dumps(v.key)] for label, v in p.species.items()}
    species_parent = dict(old.get("species_parent") or {})
    species_parent.update({label_sid[label]: v.parent for label, v in p.species.items()})

    edges_out, rx_out = [], {k: list(v) for k, v in (old.get("reactions") or {}).items()}
    blocked = list(old.get("blocked") or [])

    def first(vr):
        # Of variant reactions that are the same reaction (from two parent
        # records of one step), the one kept starts from a parent TS, the lowest.
        e = (info[vr.parent].get("energies") or [None, None, None])[1]
        return ({"ok": 0, "no_ts": 1}.get(vr.status, 2), e if e is not None else math.inf)

    for vr in sorted(p.reactions, key=first):
        key = json.dumps(vr.key)
        if key in known_keys:
            continue
        known_keys.add(key)
        parent = by_id[vr.parent]
        reactants = [label_sid.get(x, x) for x in vr.reactants]
        products = [label_sid.get(x, x) for x in vr.products]
        if vr.status == "blocked":
            blocked.append({"parent": vr.parent, "why": vr.why, "placement": vr.placement, "key": key})
            continue
        cx = []
        for frame, members in ((vr.r, reactants), (vr.p, products)):
            st = _structure(vr.symbols, frame, vr.charge, vr.multiplicity)
            cx.append(ws.add_complex(st, members, origin={**origin, "parent": vr.parent}, variant=vid))
        edge = ws.add_edge(cx[0]["id"], cx[1]["id"], label=f"{info[vr.parent]['label']} ({name or vid})",
                           origin={**origin, "parent_reaction": vr.parent,
                                   "parent_edge": info[vr.parent]["edge"], "parent_job": info[vr.parent]["job"]},
                           reaction=False)
        spec_fp = folder / f"{edge['id']}.json"
        spec_fp.write_text(json.dumps(vr.spec(parent, info[vr.parent]["energies"])))
        ws.tag_variant([edge["id"]], vid, {edge["id"]: info[vr.parent]["edge"]}, kind="edges")
        ws.patch_edge_origin(edge["id"], spec=str(spec_fp), status=vr.status, placement=vr.placement,
                             entry=vr.entry, note=info[vr.parent]["note"])
        rec = ws.put_reaction(reactants=reactants, products=products,
                              origin={**origin, "job": None, "index": f"{vid}:{key}", "parent": vr.parent},
                              label=f"{info[vr.parent]['label']} ({name or vid})", complexes=[cx[0]["id"], cx[1]["id"]],
                              edge=edge["id"], count=0, reverse_count=0, delta_e_kcal=None, events=[],
                              variant=vid, parent=vr.parent)
        rx_out.setdefault(vr.parent, []).append({"rid": rec["id"], "edge": edge["id"], "key": key,
                                                  "status": vr.status, "placement": vr.placement})
        edges_out.append(edge["id"])
    replaced = sorted(set(old.get("replaced") or []) | set(p.replaced))
    variant = {
        "id": vid, "name": name or old.get("name") or f"{species[base].name} edit",
        "on": _on(ws, design), "base": base, "edit": edit.describe(), "design_molblock": design["molblock"],
        "parent_molblock": design["source"]["parent_molblock"],
        "created": old.get("created") or time.time(), "updated": time.time(),
        "labels": sid_of, "species_parent": species_parent, "replaced": replaced,
        "replaced_reactions": sorted(set(old.get("replaced_reactions") or [])
                                     | {vr.parent for vr in p.reactions}),
        "reactions": rx_out, "blocked": blocked, "shared": p.shared,
        "problems": {**problems, **p.problems}, "warnings": p.warnings,
    }
    ws.put_variant(variant)
    return {"variant": variant, "species": new_species, "edges": edges_out}


# ------------------------------------------------------------------ kinetics: which records are in which network

def chain(snap: dict, vid: Optional[str]) -> list:
    """The variant and those it is built on, innermost first ([] for the parent)."""
    out, variants = [], snap.get("variants") or {}
    while vid:
        var = variants.get(vid)
        if var is None:
            raise WorkspaceError(f"unknown variant {vid!r}")
        if vid in out:
            raise WorkspaceError(f"variant {vid!r} is built on itself")
        out.append(vid)
        vid = var.get("on")
    return out


def network_view(snap: dict, vid: Optional[str]) -> dict:
    """The snapshot as one network sees it: the parent's (no variant
    records), or a variant's (the parent minus what it replaces, plus its
    own records; one built on another variant starts from that one's)."""
    out = json.loads(json.dumps(snap))
    keep = set(chain(snap, vid))
    # A variant's species, also those that merged into another variant's
    # record (the same molecule: iodide from two variants that make it).
    mine = {sid for v in keep for sid in (snap["variants"][v].get("labels") or {}).values()}
    drop_s = {sid for sid, r in snap["structures"].items()
              if r.get("variant") and r["variant"] not in keep and sid not in mine}
    drop_e = {eid for eid, e in snap["edges"].items() if e.get("variant") and e["variant"] not in keep}
    drop_r = {rid for rid, r in (snap.get("reactions") or {}).items() if r.get("variant") and r["variant"] not in keep}
    for v in keep:
        var = snap["variants"][v]
        drop_s |= set(var.get("replaced") or [])
        for rid in var.get("replaced_reactions") or []:
            if rid.startswith("edge:"):
                drop_e.add(rid[5:])
            else:
                drop_r.add(rid)
    for sid in drop_s:
        out["structures"].pop(sid, None)
    for eid in drop_e:
        out["edges"].pop(eid, None)
    for rid in drop_r:
        out.get("reactions", {}).pop(rid, None)
    return out


def estimate_ts(edge: dict, full: dict, jobs: dict, level_key: Optional[str]) -> Optional[float]:
    """A variant edge's fast estimate as a TS energy (Hartree): the parent
    reaction's TS height above its reactants, shifted by the estimate, on
    top of the variant reactants' own energies. None until the fast
    estimate and every species energy are in."""
    from mepd.web.kinetics import edge_ts

    o = edge.get("origin") or {}
    est = o.get("estimate") or {}
    if est.get("barrier_shift") is None or est.get("status") != "ok":
        return None
    st = full["structures"]
    pr = o.get("parent_reaction")
    pe = full["edges"].get(o.get("parent_edge") or "")
    if pe is None:
        return None
    e_ts, _, _, _ = edge_ts(pe, jobs, level_key, False)
    if e_ts is None:
        return None
    if pr and not pr.startswith("edge:"):
        prx = (full.get("reactions") or {}).get(pr)
        if prx is None:
            return None
        p_reactants = prx["reactants"]
    else:
        p_reactants = [pe["source"]]
    vrx = next((r for r in (full.get("reactions") or {}).values() if r.get("edge") == edge["id"]), None)
    v_reactants = vrx["reactants"] if vrx else [edge["source"]]

    def total(ids):
        es = []
        for s in ids:
            rec = st.get(s) or {}
            if rec.get("energy") is None or (level_key and (rec.get("level") or {}).get("key") != level_key):
                return None
            es.append(rec["energy"])
        return sum(es)

    a, b = total(p_reactants), total(v_reactants)
    if a is None or b is None:
        return None
    return b + (e_ts - a) + est["barrier_shift"] / 627.509474


# ------------------------------------------------------------------ jobs

def after_job(manager, job: dict, result: dict) -> list:
    """A variant job finished: record a fast estimate on its edge and start
    the re-optimization; after a re-optimization that did not reach a
    verified TS, start the parent's path search on the variant's ends.
    Returns the jobs submitted."""
    ws = manager.ws
    eids = job["targets"]["edges"]
    if not eids or job["status"] != "done":
        return []
    eid = eids[0]
    try:
        edge = ws.edge(eid)
    except WorkspaceError:
        return []
    var = (result.get("variant") or {})
    submitted = []
    if job["op"] == "variant-fast":
        fast = var.get("fast") or {}
        ws.patch_edge_origin(eid, estimate={"status": fast.get("status"), "barrier_shift": fast.get("barrier_shift"),
                                            "reaction_shift": fast.get("reaction_shift"), "job": job["id"],
                                            "why": fast.get("why")})
        submitted += manager.submit("variant-reopt", structure_ids=[], edge_ids=[eid], params={},
                                    profile=job.get("profile"), label=f"Re-optimize {edge.get('label', '')}")
    elif job["op"] == "variant-reopt":
        reopt = var.get("reopt") or {}
        out = Path(job["output_dir"]) / "reopt"
        _adopt_ends(ws, edge, out, job)
        if reopt.get("status") not in ("ok", "ends_changed") or not reopt.get("barrier_verified"):
            ws.patch_edge_origin(eid, fallback={"why": reopt.get("headline") or reopt.get("status")})
            submitted += _fallback(manager, edge, job)
    return submitted


def _adopt_ends(ws, edge: dict, out: Path, job: dict) -> None:
    """The minimized ends become the edge's conformers (a path search then
    starts from real minima)."""
    level = job.get("level")
    pins = {}
    for sid, fp in ((edge["source"], out / "start.xyz"), (edge["target"], out / "end.xyz")):
        if not fp.exists():
            continue
        try:
            st = _read_xyz(fp)
            rec = ws.structure(sid)
            st = st.model_copy(update={"charge": rec["charge"], "multiplicity": rec["multiplicity"]})
            en = float(np.loadtxt(fp.with_suffix(".energies")).ravel()[0])
        except Exception:
            continue
        cid, _ = ws.add_conformer(sid, st, energy=en, level=level, optimized=True,
                                  origin={"kind": "variant-reopt", "job": job["id"]})
        pins[sid] = cid
    if pins:
        ws.update_edge(edge["id"], conformers=pins)


def _fallback(manager, edge: dict, job: dict) -> list:
    """The parent reaction's own path search, on the variant's ends."""
    o = edge.get("origin") or {}
    parent = manager.jobs.get(o.get("parent_job") or "")
    for _ in range(10):
        # A variant built on another: back to the search of the reaction it all started from.
        if parent is None or parent["op"] not in ("variant-fast", "variant-reopt") or not parent["targets"]["edges"]:
            break
        try:
            up = manager.ws.edge(parent["targets"]["edges"][0]).get("origin") or {}
        except WorkspaceError:
            break
        parent = manager.jobs.get(up.get("parent_job") or "")
    op, params = "ts", None
    if parent is not None and parent["op"] == "channels-more":
        # "Sample more paths": the channels run it extended is the search to repeat.
        parent = manager.jobs.get(parent.get("source_job") or "") or parent
    if parent is not None and parent["op"] in ("ts", "channels"):
        op, params = parent["op"], parent.get("params")
    try:
        return manager.submit(op, structure_ids=[], edge_ids=[edge["id"]], params=params,
                              profile=job.get("profile"), label=f"Path search {edge.get('label', '')} "
                              "(re-optimization did not connect)")
    except WorkspaceError:
        return manager.submit("ts", structure_ids=[], edge_ids=[edge["id"]], params=None,
                              profile=job.get("profile"), label=f"Path search {edge.get('label', '')}")


def start_jobs(manager, made: dict, profile: Optional[str]) -> list:
    """The fast estimate for each new variant reaction (its re-optimization
    follows when it finishes). A reaction whose parent has no usable TS
    goes to the re-optimization at once: it minimizes the substituted ends,
    and the path search starts from them when it is done (after_job).
    Jobs start in the order they are queued, so the minutes-long path
    searches line up behind every estimate and re-optimization queued by then."""
    ws = manager.ws
    out, searches = [], []
    for eid in made["edges"]:
        edge = ws.edge(eid)
        if (edge.get("origin") or {}).get("status") == "no_ts":
            searches.append(edge)
            continue
        out += manager.submit("variant-fast", structure_ids=[], edge_ids=[eid], params={}, profile=profile,
                              label=f"Fast estimate {edge.get('label', '')}")
    for edge in searches:
        out += manager.submit("variant-reopt", structure_ids=[], edge_ids=[edge["id"]], params={}, profile=profile,
                              label=f"Minimize the ends of {edge.get('label', '')}")
    return out


def delete(ws, vid: str) -> None:
    var = ws.variants.get(vid)
    if var is None:
        raise WorkspaceError(f"unknown variant {vid!r}")
    on_it = [v["name"] for v in ws.variants.values() if v.get("on") == vid]
    if on_it:
        raise WorkspaceError(f"{', '.join(on_it)} {'is' if len(on_it) == 1 else 'are'} built on {var['name']}: "
                             "delete that first")
    snap = ws.snapshot()
    sids = [sid for sid, r in snap["structures"].items() if r.get("variant") == vid]
    # Species another variant also has (the same molecule merged into this
    # one's record) stay, as that variant's.
    for other, v in ws.variants.items():
        if other == vid:
            continue
        theirs = set((v.get("labels") or {}).values()) & set(sids)
        if theirs:
            parent_of = {sid: p for sid, p in (v.get("species_parent") or {}).items() if sid in theirs}
            ws.tag_variant(sorted(theirs), other, parent_of)
            sids = [sid for sid in sids if sid not in theirs]
    eids = [eid for eid, e in snap["edges"].items() if e.get("variant") == vid]
    ws.delete_many(sids, eids)
    ws.drop_variant(vid)
    shutil.rmtree(ws.root / "variants" / vid, ignore_errors=True)


def compare_inputs(snap: dict, vid: str, initial: dict, held: list, target: Optional[str]) -> tuple:
    """The parent's conditions as the variant sees them: amounts of a
    replaced species go to its variant (split evenly over several), a
    target to its variant (the first, if several); through each variant it
    is built on in turn. Returns (initial, held, target, match {parent id:
    [variant ids]}, notes)."""
    levels = chain(snap, vid)
    if not levels:
        raise WorkspaceError(f"unknown variant {vid!r}")
    match: dict = {}                           # parent id -> its variants in this network
    amounts = {sid: {sid: float(c)} for sid, c in (initial or {}).items()}
    notes = []
    for v in reversed(levels):                  # outermost (on the parent) first
        step: dict = {}
        for sid, parent in (snap["variants"][v].get("species_parent") or {}).items():
            step.setdefault(parent, []).append(sid)
        inner = {x for xs in match.values() for x in xs}
        for root in match:
            match[root] = [x for cur in match[root] for x in (step.get(cur) or [cur])]
        for parent, vs in step.items():
            if parent not in inner:
                match[parent] = list(vs)
        for root, parts in amounts.items():
            new = {}
            for cur, c in parts.items():
                vs = step.get(cur)
                if not vs:
                    new[cur] = new.get(cur, 0.0) + c
                    continue
                if len(vs) > 1:
                    notes.append(f"{snap['structures'].get(cur, {}).get('name', cur)}: its amount is split over "
                                 f"its {len(vs)} variants")
                for x in vs:
                    new[x] = new.get(x, 0.0) + c / len(vs)
            amounts[root] = new
    new_initial: dict = {}
    for parts in amounts.values():
        for x, c in parts.items():
            new_initial[x] = new_initial.get(x, 0.0) + c
    new_held = [x for s in held or [] for x in match.get(s, [s])]
    new_target = match.get(target, [target])[0] if target else None
    return new_initial, new_held, new_target, match, notes
