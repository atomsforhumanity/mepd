"""The live network view's data: an explorer run (`mepd discovery expand`) as
a timeline the page can play, growing, correcting itself and moving material.

Each refresh of the explorer's live kinetics (live/kinetics.json, schema of
the explorer team's step_list/conc/energy_source fields) becomes one compact
frame, holding only what can matter on screen:
- the species that ever hold material (their energy, peak and final amount,
  whether their energy is DFT-corrected);
- the open steps between them (barriers, source, IRC check, net flux);
- counts of everything else;
and, when read, what forms per molecule rather than per box state (molecule_yields).

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


MAX_YIELDS = 16           # molecules per frame, by amount (from a billionth of the start up)
MIN_YIELD = 1e-9


def _formula_of(smiles: str) -> Optional[str]:
    from rdkit import Chem, rdBase
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return CalcMolFormula(mol) if mol is not None else None


def _ring_positions(mol) -> Optional[str]:
    """Where a one-ring molecule's two substituted ring atoms sit, the shorter way round: "1,4" (two
    atoms apart is "1,3", next to each other "1,2"); None for any other molecule."""
    ri = mol.GetRingInfo()
    if ri.NumRings() != 1:
        return None
    ring = list(ri.AtomRings()[0])
    held = [k for k, a in enumerate(ring) if any(nb.GetIdx() not in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors())]
    if len(held) != 2:
        return None
    d = abs(held[0] - held[1])
    return f"1,{min(d, len(ring) - d) + 1}"


def _endo_exo(mol) -> Optional[str]:
    """endo or exo (IUPAC) of the substituents on a bridged bicyclic part, from the stereo the molecule has
    (assigned from its geometry): a substituent on a bridge is exo when it points to the shortest of the
    other two bridges, endo when away. Only bridge atoms with one substituent count; None unless they all
    agree, or without a bridged part, a shortest bridge or a 3D embedding."""
    from itertools import combinations

    from rdkit import Chem
    from rdkit.Chem import AllChem

    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    for r1, r2 in combinations(rings, 2):
        shared = r1 & r2
        if len(shared) < 3:      # fused (two atoms in common) or apart
            continue
        heads = {a for a in shared if any(nb.GetIdx() in (r1 | r2) - shared for nb in mol.GetAtomWithIdx(a).GetNeighbors())}
        if len(heads) != 2:
            continue
        bridges = [r1 - shared, r2 - shared, shared - heads]
        part = r1 | r2
        found = []
        for k, bridge in enumerate(bridges):
            others = sorted((b for j, b in enumerate(bridges) if j != k), key=len)
            if len(others[0]) == len(others[1]):
                continue
            for a in bridge:
                out = [nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() not in part]
                if len(out) == 1:
                    found.append((out[0], others[0], others[1]))
        if not found:
            return None
        h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(h, randomSeed=7) != 0:
            return None
        xyz = h.GetConformer().GetPositions()
        near = lambda x, b: min(((xyz[x] - xyz[i]) ** 2).sum() for i in b)
        words = {"exo" if near(x, short) < near(x, long) else "endo" for x, short, long in found}
        return words.pop() if len(words) == 1 else None
    return None


def _double_bond_stereo(mol) -> Optional[str]:
    """E and Z of the molecule's stereo double bonds (CIP), as "E,Z"; None without any."""
    from rdkit.Chem import rdCIPLabeler

    rdCIPLabeler.AssignCIPLabels(mol)
    codes = sorted(b.GetProp("_CIPCode") for b in mol.GetBonds() if b.HasProp("_CIPCode"))
    return ",".join(codes) if codes else None


def _ring_kind(mol) -> str:
    """"open chain"; a ring by its size and what else than carbon is in it ("6-ring", "5-ring with O");
    several rings by their count ("3 rings")."""
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return "open chain"
    if len(rings) > 1:
        return f"{len(rings)} rings"
    other = sorted({mol.GetAtomWithIdx(a).GetSymbol() for a in rings[0]} - {"C"})
    return f"{len(rings[0])}-ring" + (f" with {' and '.join(other)}" if other else "")


def isomer_labels(keys: list) -> dict:
    """Labels that tell apart molecules with one formula and no everyday name, in plain words, from their
    graphs and the stereo their geometry gave them: the formula, then only as much as tells them apart,
    in this order: endo or exo, E or Z, where a ring's two substituents sit ("1,4-isomer"), open chain or
    the ring's size and make-up ("6-ring", "5-ring with O"); failing those, in their place, the SMILES
    (without stereo when that is enough). One told apart
    only by what the others are gets its own first trait ("open chain").
    {molecule: (label, the part that tells it apart)}; a molecule alone with its formula: (formula, None)."""
    from rdkit import Chem, rdBase

    by = {}
    for k in keys:
        by.setdefault(_formula_of(k), []).append(k)
    out = {}
    for formula, ks in by.items():
        if len(ks) == 1:
            out[ks[0]] = (formula, None)
            continue
        traits = {}
        for k in ks:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(k)
            ring = _ring_positions(mol) if mol is not None else None
            traits[k] = [_endo_exo(mol) if mol is not None and "@" in k else None,
                         _double_bond_stereo(mol) if mol is not None and ("/" in k or "\\" in k) else None,
                         f"{ring}-isomer" if ring else None,
                         _ring_kind(mol) if mol is not None else None]
        tell = {k: [] for k in ks}

        def colliding():
            groups = {}
            for k in ks:
                groups.setdefault(tuple(tell[k]), []).append(k)
            return [g for g in groups.values() if len(g) > 1]

        for j in range(len(traits[ks[0]])):
            for g in colliding():
                if len({traits[k][j] for k in g}) > 1:
                    for k in g:
                        if traits[k][j]:
                            tell[k].append(traits[k][j])
        # one ring's size said for one isomer is said for every isomer told by where its ring's substituents sit
        placed = [k for k in ks if traits[k][2] in tell[k]]
        if any(traits[k][3] in tell[k] for k in placed):
            for k in placed:
                if traits[k][3] not in tell[k]:
                    tell[k].append(traits[k][3])
        for g in colliding():
            flat = {}
            for k in g:
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(k)
                flat[k] = Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else k
            for k in g:
                tell[k] = [flat[k] if len(set(flat.values())) == len(g) else k]
        for k in ks:   # one told apart by what the others are: what it is itself, when that is still unique
            first = next((x for x in traits[k] if x), None)
            if not tell[k] and first and all(tell[o] != [first] for o in ks):
                tell[k] = [first]
        for k in ks:
            words = ", ".join(tell[k]) or None
            out[k] = (f"{formula}, {words}" if words else formula, words)
    return out


def _mirror_smiles(smiles: str) -> str:
    """The mirror image of a stereo SMILES (every tetrahedral centre inverted; double bonds keep E/Z)."""
    return smiles.replace("@@", "\0").replace("@", "@@").replace("\0", "@")


def molecule_yields(tl: dict, stereo: Optional[dict] = None, mirror_images: bool = True) -> dict:
    """In place: what forms, per molecule rather than per box state. A species is a whole box (every
    molecule of the start, reacted or not); in each frame a molecule's amount, per starting box, is the sum
    over the species of what each holds at the end of the simulated time times the copies of that molecule
    in it (2 H2 in a box count twice), so it can exceed 1. Molecules by canonical SMILES, with the stereo of
    each species' geometry where it has any (`stereo`: {species: stereo SMILES}, from stereo_smiles_of); a
    molecule and its mirror image are one when the run counts mirror images (the explorer's default), as
    its kinetics do. The start's own molecules, wherever they are, are what is left of the start.

    Each frame gets `yields` ([molecule, amount, the species holding most of it], the MAX_YIELDS largest
    from MIN_YIELD up)
    and `left` ([molecule, amount] for the start's molecules); the timeline `molecules` ({molecule: {name,
    formula, label and tell (one without a name that shares its formula: isomer_labels), species, mirror (it
    stands for itself and its mirror image)}}) and `start` (the start's molecules, as
    many times as the box holds each)."""
    from collections import Counter

    from mepd.web.chem import common_name

    smiles, frames = tl.get("smiles") or {}, tl.get("frames") or []
    stereo = stereo or {}
    seed = frames[0]["species"][0][0] if frames and frames[0].get("species") else "0"
    one, chiral, parts = {}, set(), {}

    def key_of(m):
        if not mirror_images or "@" not in m:
            return m
        if m not in one:
            twin = _canonical_one(_mirror_smiles(m))
            one[m] = min(m, twin) if twin else m
            if twin and twin != m:
                chiral.add(one[m])
        return one[m]

    def molecules_of(sid):
        if sid not in parts:
            smi = stereo.get(sid) or smiles.get(sid) or ""
            ms = _canon_multiset(smi) if drawable(smi) else None
            parts[sid] = Counter(key_of(m) for m in ms) if ms else Counter({f"#{sid}": 1})
        return parts[sid]

    start = molecules_of(seed)
    table = {}
    for f in frames:
        amount, main = Counter(), {}
        for s in f.get("species") or []:
            fin = s[3] or 0.0
            if fin <= 0:
                continue
            for m, n in molecules_of(s[0]).items():
                amount[m] += fin * n
                if fin * n > main.get(m, (0.0, None))[0]:
                    main[m] = (fin * n, s[0])
        made = sorted((m for m in amount if m not in start and amount[m] >= MIN_YIELD), key=lambda m: -amount[m])[:MAX_YIELDS]
        f["yields"] = [[m, _sig(amount[m]), main[m][1]] for m in made]
        f["left"] = [[m, _sig(amount[m])] for m in sorted(start) if m in amount]
        for m in made:
            table.setdefault(m, main[m][1])
    for m in start:
        table.setdefault(m, seed)
    names = {m: (tl.get("names") or {}).get(sid) if m.startswith("#") else common_name(m) for m, sid in table.items()}
    labels = isomer_labels([m for m in table if not names[m] and not m.startswith("#")])
    tl["molecules"] = {m: {"name": names[m], "formula": None if m.startswith("#") else _formula_of(m),
                           **({"label": labels[m][0], "tell": labels[m][1]} if m in labels and labels[m][1] else {}),
                           "species": sid, **({"mirror": True} if m in chiral else {})} for m, sid in table.items()}
    tl["start"] = sorted(start.elements())
    return tl


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


STEREO = "stereo_smiles.json"
_MAY_STEREO: dict = {}


def _species_stereo(job_dir: Path, smiles: dict, lines: list) -> dict:
    """{species: its SMILES with the stereo of its geometry} for the species with a molecule that can
    have stereoisomers: from its final geometry (output/species), else as the run reported it
    (events.jsonl; the start's from inputs/seed.xyz). Cached in live/stereo_smiles.json by SMILES and
    geometry, so each species is matched once (and again when the run ends)."""
    job_dir = Path(job_dir)
    fp = job_dir / "live" / STEREO
    try:
        cache = json.loads(fp.read_text()) if fp.exists() else {}
    except ValueError:
        cache = {}
    try:
        out = Path(json.loads((job_dir / "job.json").read_text()).get("output_dir") or job_dir / "output")
    except (OSError, ValueError):
        out = job_dir / "output"

    def may(m):
        if m not in _MAY_STEREO:
            _MAY_STEREO[m] = _may_have_stereo_smiles(m)
        return _MAY_STEREO[m]

    want = {sid: smi for sid, smi in smiles.items() if drawable(smi) and any(may(m) for m in smi.split("."))}
    todo = {}
    for sid, smi in want.items():
        final = out / "species" / f"species_{sid}.xyz"
        src = "final" if final.exists() else "live"
        if (cache.get(sid) or [None, None])[:2] != [smi, src]:
            todo[sid] = (smi, src, final)
    if todo:
        found = {}
        need = {sid for sid, (_, src, _) in todo.items() if src == "live"}
        for line in lines if need else []:
            if '"species"' not in line[:40]:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "species" and str(ev.get("index")) in need and ev.get("xyz"):
                found[str(ev["index"])] = ev["xyz"]
        seed = job_dir / "inputs" / "seed.xyz"
        if "0" in need and "0" not in found and seed.exists():
            found["0"] = seed.read_text()
        for sid, (smi, src, final) in todo.items():
            xyz = final.read_text() if src == "final" else found.get(sid)
            if xyz:          # (no geometry yet: asked again next time)
                cache[sid] = [smi, src, stereo_smiles_of(smi, xyz) or ""]
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(cache))
        except OSError:
            pass
    return {sid: cache[sid][2] for sid, smi in want.items() if sid in cache and cache[sid][0] == smi and cache[sid][2]}


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


def _summary_species(job_dir: Path) -> Optional[dict]:
    """A finished run's own species and final amounts: {species: {smiles, rel_kcal, peak, final}} from
    summary.json, the virtual mirror images its kinetics add counted with the species they mirror (as a
    live refresh's mirror_species are). None without amounts."""
    try:
        out = Path(json.loads((Path(job_dir) / "job.json").read_text()).get("output_dir") or "")
        summ = json.loads((out / "summary.json").read_text())
    except (OSError, ValueError):
        return None
    species = summ.get("species") or []
    if not species or any(s.get("final_concentration") is None for s in species):
        return None
    got = {str(s["index"]): {"smiles": s.get("smiles") or "", "rel_kcal": s.get("rel_energy_kcal"),
                             "peak": float(s.get("max_concentration") or 0), "final": float(s["final_concentration"])}
           for s in species}
    for v in (summ.get("mirror_images") or {}).get("virtual_species") or []:
        twin = got.get(str(v.get("mirror_of")))
        if twin is not None:
            twin["peak"] += float(v.get("max_concentration") or 0)
            twin["final"] += float(v.get("final_concentration") or 0)
    return got


def _growing(job_dir: Path, lines: list, t0: Optional[float], steps: Optional[list] = None,
             t_end: Optional[float] = None, summary: Optional[dict] = None) -> Optional[dict]:
    """The network as events.jsonl has it now, on top of the last kinetics
    refresh: species and steps found since then show at once (no amounts yet).
    `summary` (a finished run's, _summary_species): its species and what each holds at the end, in place
    of the last refresh's (a species found after it holds its share too)."""
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
    thr = float(kin.get("threshold") or 0.01)
    for sid, x in (summary or {}).items():
        if sid not in have:
            have.add(sid)
            species.append({"index": int(sid), "smiles": x["smiles"], "rel_kcal": x["rel_kcal"]})
    for k, sp in enumerate(species):
        x = (summary or {}).get(str(sp.get("index")))
        if x is not None:
            species[k] = {**sp, "max_concentration": x["peak"], "final": x["final"], "populated": x["peak"] >= thr}
    kin["species"] = species
    f = compact_legacy(kin, steps if steps is not None else _legacy_steps(lines))
    if f is None:
        return None
    f["growing"] = True
    f["steered"] = "single points" in (kin.get("basis") or "")
    f["t"] = _r((t_end or time.time()) - t0, 1) if t0 else 0.0   # (a finished run: when it ended, not now)
    return _fill_smiles(f, _perceived(job_dir, lines))


def record_selectivity(job_dir: Path, data: dict) -> None:
    """The annotator's selectivity.json is rewritten on each refresh: keep each version's latest entry."""
    entries = (data or {}).get("entries") or []
    if not entries:
        return
    head = {k: v for k, v in data.items() if k != "entries"}
    with open(Path(job_dir) / "live" / SELECTIVITY_FRAMES, "a") as fh:
        fh.write(json.dumps({**head, "entry": entries[-1]}, separators=(",", ":")) + "\n")


def species_xyz(job_dir: Path, sid: str, irc: bool = True) -> Optional[str]:
    """A species' geometry: its final one (output/species), else as the run first
    reported it (live/events.jsonl), the seed's from the job's input, else (one an IRC end made,
    while the run is on) that IRC end."""
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
    lines = _events(job_dir)
    for line in lines:
        if '"species"' in line[:40] and f'"index": {sid},' in line:
            try:
                return json.loads(line).get("xyz")
            except ValueError:
                return None
    return _irc_end_xyz(job_dir, out, str(sid), lines) if irc else None


def _irc_end_xyz(job_dir: Path, out: Path, sid: str, lines: list) -> Optional[str]:
    """A species an IRC end made (the run reports no species event for it, and writes output/species only
    when it ends): the end of one of its steps' IRCs that is not the step's other species."""
    from mepd.web.step_check import _frames

    for line in lines:
        if '"step"' not in line[:30]:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        a, b = str(ev.get("a")), str(ev.get("b"))
        if ev.get("event") != "step" or sid not in (a, b) or a == b:
            continue
        fp = out / "ts" / f"{ev.get('label')}_irc.xyz"
        try:
            frames = _frames(fp.read_text())
        except OSError:
            continue
        other = species_xyz(job_dir, b if sid == a else a, irc=False)
        if len(frames) < 2 or not other:
            continue
        key = _xyz_key(other)
        pick = [x for x in (frames[0], frames[-1]) if _xyz_key(x) != key]
        if len(pick) == 1 and key is not None:
            return pick[0]
    return None


def _xyz_key(xyz: str) -> Optional[tuple]:
    sy, xyz_ = _xyz_rows(xyz)
    mols = _molecules(sy, xyz_) if sy else None
    return _key(mols) if mols else None


_FEED: dict = {}


def mixture_feed(job_dir: Path, job: dict) -> list:
    """A reacting mixture's feed before its first merge (no mixture.json yet), from its command: each --feed
    (a SMILES, or an xyz file of one molecule, the copy the job keeps under inputs/, perceived), its
    --concentration (in order; 1 each without) and a name (the file's, which the workspace's replaces)."""
    argv = [str(x) for x in job.get("argv") or []]
    feeds = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--feed"]
    conc = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--concentration"]
    out = []
    for k, f in enumerate(feeds):
        smi, name = f, f
        if f.endswith(".xyz"):
            name = Path(f).name
            try:
                xyz = (Path(job_dir) / "inputs" / name).read_text()
            except OSError:
                continue
            if xyz not in _FEED:
                from qcdata import Structure

                from mepd.web.chem import perceive_smiles
                try:
                    _FEED[xyz] = perceive_smiles(Structure.from_xyz(xyz)) or ""
                except Exception:  # noqa: BLE001 -- not perceived: left out
                    _FEED[xyz] = ""
            smi = _FEED[xyz]
        if smi:
            try:
                c = float(conc[k]) if k < len(conc) else 1.0
            except ValueError:
                c = 1.0
            out.append({"name": name, "molecule": smi, "concentration": c})
    return out


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


def conditions_of(job_dir: Path, basis: str = "") -> dict:
    """The run's temperature (K), simulated time (s) and time budget (s), from its command line (the
    temperature, without one, from the kinetics' basis "free energy (1500 K, 1atm)")."""
    import re

    try:
        argv = [str(x) for x in json.loads((Path(job_dir) / "job.json").read_text()).get("argv") or []]
    except (OSError, ValueError):
        argv = []

    def opt(name):
        try:
            return float(argv[argv.index(name) + 1])
        except (ValueError, IndexError):
            return None
    t = opt("--temperature")
    if t is None:
        m = re.search(r"([\d.]+)\s*K\b", basis or "")
        t = float(m.group(1)) if m else None
    return {"T": t, "time_s": opt("--time"), "budget_s": opt("--time-budget")}


def from_job(job_dir: Path, *, running: bool = False, t0: Optional[float] = None, t_end: Optional[float] = None,
             since: Optional[int] = None) -> dict:
    """A job's timeline: its recorded frames, and while it runs the network as
    events.jsonl has it now (so it grows from the first species on). `t_end`: when a finished run
    ended. `since` (while it runs): only the frames from that one on (`recorded`: how many frames are
    recorded; the last one after them, if any, is the network as it stands, replaced on every poll)."""
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
        g = _growing(job_dir, lines, t0, steps=final_steps, t_end=t_end, summary=_summary_species(job_dir)) if lines else None
        recorded = len(frames)
        if g is not None:
            g.pop("growing", None)
            g["t"] = max(g["t"], frames[-1]["t"] if frames else 0.0)
            g["steered"] = frames[-1]["steered"] if frames else g["steered"]
            frames.append(g)
    else:
        recorded = len(frames)
    if final_steps is None and (running or not frames):
        g = _growing(job_dir, lines, t0, t_end=None if running else t_end) if lines else None
        if g is not None and (not frames or g["counts"]["species"] > frames[-1]["counts"]["species"]
                              or g["counts"]["steps"] > frames[-1]["counts"]["steps"]):
            frames.append(g)
    # A finished run's frames end when it ended: a frame recorded later (a server that first read the
    # finished run's summary hours after, from the run's start) is that end, not hours of more growth.
    origin = t_start or t0
    if not running and t_end and origin and t_end > origin:
        cap = float(t_end) - float(origin)
        for f in frames:
            if f.get("t", 0.0) > cap:
                f["t"] = round(cap, 1)
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
    tl["conditions"] = conditions_of(job_dir, frames[-1].get("basis", "") if frames else "")
    try:
        job = json.loads((Path(job_dir) / "job.json").read_text())
    except (OSError, ValueError):
        job = {}
    flags = [str(x) for x in job.get("argv") or []] + str(job.get("command") or "").split()
    molecule_yields(tl, _species_stereo(job_dir, tl["smiles"], lines), mirror_images="--no-mirror-images" not in flags)
    tl["recorded"] = recorded
    if since is not None and running:
        tl["since"] = since = max(0, min(int(since), recorded))
        tl["frames"] = tl["frames"][since:]
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
    return molecule_yields(tl)


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


def _canonical_one(m: str) -> str:
    """One molecule's canonical SMILES (cached; "" if it does not parse)."""
    from mepd.web.chem import _canonical

    c = _CANON.get(m)
    if c is None:
        c = _CANON[m] = _canonical(m) or ""
    return c


def _canon_multiset(smiles: str) -> Optional[tuple]:
    """A state as its sorted canonical molecules (None if one does not parse)."""
    out = []
    for m in str(smiles).split("|")[0].split("."):
        m = m.strip()
        if not m:
            continue
        c = _canonical_one(m)
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
    # (by folder name: an index into the sorted folders shifts when run_100 sorts in before run_11, or a run
    # shows up out of order)
    if job.get("op") == "network-mixture":
        return [(f"{sd.name}/{d.name}", d, sd.name) for sd in sorted((out / "seeds").glob("*"))
                for d in sorted((sd / "run" / "reactor").glob("run_*"))]
    return [(d.name, d, None) for d in sorted((out / "reactor").glob("run_*"))]


def reactor_events(job_dir: Path, job: dict, since: int = 0, limit: int = 4000, budget_s: float = 1.0) -> dict:
    """The reactor events of a job after the first `since` this server has numbered (see above).
    Each: {i, t (s from the job's start, when its MD segment was written), run, seed, k, label, atoms
    (count), src (species id or the mixture molecule it started from), prod (the species or molecule it
    made, only once that is verified on the event's own frames (see _Verifier); events are verified a
    few at a time, `budget_s` per call, and a match found later comes with a later call), changes}."""
    job_dir = Path(job_dir)
    key = str(job_dir)
    st = _REACTOR.setdefault(key, {"seen": {}, "events": [], "src": {}, "raw": {}, "todo": [], "prods": {},
                                    "verifier": _Verifier(), "skeletons": {}, "version": 0, "tried": {}})
    t0 = float(job.get("started") or 0.0)
    mixture = job.get("op") == "network-mixture"
    states = None
    seeds_mol = {}
    if mixture:
        try:
            mx = json.loads((Path(job["output_dir"]) / "mixture.json").read_text())
            seeds_mol = {s["name"]: s.get("molecules") or [] for s in mx.get("seeds") or []}
            for m_ in mx.get("molecules") or []:
                if m_["molecule"] not in st["skeletons"]:
                    st["skeletons"][m_["molecule"]] = _skeleton_of_smiles(m_["molecule"].split("|")[0])
                    st["version"] += 1
        except (OSError, ValueError, KeyError):
            pass
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
                st["src"][rid] = _source_of(run, states) or _source_by_time(job_dir, Path(job["output_dir"]), run, int(run.name[4:]))
        for ev in evs[n0:]:
            seg = run / "md" / f"segment_{int(ev.get('product_frame', 0)) // SEGMENT_FRAMES:03d}.xyz"
            try:
                t = seg.stat().st_mtime
            except OSError:
                t = m
            label = ev.get("label") or ""
            fresh.append({"_raw": {"atoms": ev.get("atoms"), "product_frame": ev.get("product_frame"), "label": label, "run": run},
                          "t": round(t - t0, 1) if t0 else 0.0, "run": run.name, "seed": seed, "rid": rid,
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
            st["raw"][e["i"]] = e.pop("_raw")
            st["events"].append(e)
            st["todo"].append(e["i"])
    # What each event made, verified on its own frames (_Verifier, mixture_product), within the budget.
    if not mixture:
        states = states if states is not None else _species_states(job_dir)
        v = st["verifier"]
        n_before = len(v.changed)
        v.index(states, Path(job["output_dir"]))
        if len(v.changed) != n_before:
            st["version"] += 1
    deadline = _time.time() + budget_s
    left = []
    for n, i in enumerate(st["todo"]):
        if _time.time() > deadline:
            left += st["todo"][n:]
            break
        if st["tried"].get(i) == st["version"]:   # nothing new to match it against since
            left.append(i)
            continue
        raw, e = st["raw"][i], st["events"][i]
        if mixture:
            verdict, hit = mixture_product(raw["run"], raw, st["skeletons"])
        else:
            if i not in st["prods"]:
                st["prods"][i] = None if _is_exchange(raw["label"]) else (_event_products(raw["run"], raw) or None)
                if st["prods"][i] is not None and _label_key(raw["label"]) != _key(st["prods"][i]):
                    st["prods"][i] = None   # the frames and the reactor's label disagree on what formed
            prods = st["prods"][i]
            verdict, hit = "none", None
            if prods is not None:
                for sid in st["verifier"].by_key.get(_key(prods), []):
                    if st["verifier"].drawn.get(sid) != _key(prods):
                        continue   # its card draws other molecules (its SMILES and its geometry disagree)
                    vv = _same_molecules(prods, st["verifier"].changed[sid])
                    if vv == "same":
                        verdict, hit = "same", sid
                        break
            elif i in st["prods"]:
                verdict = "never"
        if verdict == "same":
            e["prod"] = hit
            st.setdefault("late", []).append(i)
            st["prods"].pop(i, None)
        elif verdict != "never" and not (mixture and verdict == "cant"):
            st["tried"][i] = st["version"]
            left.append(i)
    st["todo"] = left
    evs = st["events"]
    since = max(0, int(since))
    # (products matched since: the client updates those it has)
    return {"epoch": EPOCH, "n": len(evs), "events": evs[since:since + limit],
            "prods": {i: evs[i]["prod"] for i in (st.get("late") or [])[-2000:] if i < since},
            "srcs": {i: evs[i]["src"] for i in sorted(st.get("src_late") or ()) if i < since},
            "runs": len(st["seen"]), "running": sum(1 for _, run, _ in _runs_of(job) if not (run / "md" / "done").exists())}


# ---------------------------------------------------------------------------------------------------
# Which species (or mixture molecule) a reactor event made, verified on the event itself. A link is
# shown only when it holds:
# - the molecules the event's reacting atoms form just after its product frame (the bond graph seen in
#   most of the next dozen MD frames, on their averaged geometry) are what the reactor's label says;
# - they are, one for one, the molecules that make the species differ from the run's start (a species
#   is a whole box: its molecules less the start's, as its card draws it), compared as the network
#   compares species: the same bond graph (hydrogens included), and for a molecule that can have
#   stereoisomers the same stereochemistry (mepd.compare, from 3D; "can't tell" is no link);
# - the event is not an exchange (its products are its reactants).
# A mixture molecule has no geometry here: its bond graph must be the product's, and a molecule that
# can have stereoisomers is not claimed.

_Z = {"H": 1, "He": 2, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12, "Si": 14, "P": 15,
      "S": 16, "Cl": 17, "K": 19, "Br": 35, "I": 53}


def _bare(mol):
    """A molecule as its bare bond graph: hydrogens as atoms, every bond single, no charges, radicals or
    aromaticity (atom order kept)."""
    from rdkit import Chem

    m = Chem.RWMol(Chem.AddHs(mol) if any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms()) else mol)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)
        a.SetIsAromatic(False)
        a.SetNumRadicalElectrons(0)
    for bd in m.GetBonds():
        bd.SetBondType(Chem.BondType.SINGLE)
        bd.SetIsAromatic(False)
    return m.GetMol()


def _skeleton(mol) -> Optional[str]:
    """A molecule's bond graph as a canonical string: hydrogens as atoms, every bond single, no charges
    (bond orders and charges are perception; the graph is what the network compares first)."""
    from rdkit import Chem

    try:
        return Chem.MolToSmiles(_bare(mol), canonical=True, isomericSmiles=False, allHsExplicit=True)
    except Exception:
        return None


def stereo_smiles_of(smiles: str, xyz: str) -> Optional[str]:
    """`smiles` (a species' molecules, with the network's bond orders and charges) with the stereo of its
    geometry: matched atom for atom onto the geometry's bond graph (bonds from distances), stereocentres
    and double bonds assigned from 3D (RDKit). None when the two graphs differ."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDetermineBonds

    try:
        with rdBase.BlockLogs():
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            rows = [r.split()[:4] for r in xyz.strip().splitlines()[2:] if len(r.split()) >= 4]
            geo = Chem.MolFromXYZBlock(f"{len(rows)}\n\n" + "".join(" ".join(r) + "\n" for r in rows))
            rdDetermineBonds.DetermineConnectivity(geo)
            a, b = _bare(mol), _bare(geo)
            if a.GetNumAtoms() != b.GetNumAtoms() or a.GetNumBonds() != b.GetNumBonds():
                return None
            match = b.GetSubstructMatch(a)
            if len(match) != a.GetNumAtoms():
                return None
            conf, at = Chem.Conformer(mol.GetNumAtoms()), geo.GetConformer()
            for i, j in enumerate(match):
                conf.SetAtomPosition(i, at.GetAtomPosition(j))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            Chem.AssignStereochemistryFrom3D(mol)
            return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def _skeleton_of_smiles(smiles: str) -> Optional[str]:
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles) or Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return _skeleton(mol)


def _may_have_stereo_smiles(smiles: str) -> bool:
    """Whether a molecule (from its SMILES) can have stereoisomers (a possible stereocentre or stereo
    double bond); True when that cannot be told."""
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True
    try:
        return len(Chem.FindPotentialStereo(Chem.AddHs(mol))) > 0
    except Exception:
        return True


class _Mol:
    """One molecule of a geometry: formula, bond-graph skeleton, and lazily its StructureNode and
    whether it can have stereoisomers."""

    def __init__(self, symbols, coords, skeleton):
        from collections import Counter

        self.symbols, self.coords, self.skeleton = list(symbols), coords, skeleton
        c = Counter(self.symbols)
        self.formula = "".join(f"{k}{c[k]}" for k in sorted(c))
        self._node = None
        self._stereo = None

    @property
    def node(self):
        if self._node is None:
            from qcdata import Structure

            from mepd.nodes.node import StructureNode

            xyz = f"{len(self.symbols)}\n\n" + "".join(f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, (x, y, z) in zip(self.symbols, self.coords))
            st = Structure.from_xyz(xyz)
            electrons = sum(_Z.get(x, 6) for x in self.symbols)
            self._node = StructureNode(structure=st.model_copy(update={"charge": 0, "multiplicity": 2 if electrons % 2 else 1}))
        return self._node

    @property
    def may_have_stereo(self) -> bool:
        """From its own geometry (bond orders perceived): a possible stereocentre or stereo double bond;
        True when bond orders cannot be perceived."""
        if self._stereo is None:
            from rdkit import Chem
            from rdkit.Chem import rdDetermineBonds

            try:
                mol = Chem.MolFromXYZBlock(f"{len(self.symbols)}\n\n" + "".join(
                    f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, (x, y, z) in zip(self.symbols, self.coords)))
                electrons = sum(_Z.get(x, 6) for x in self.symbols)
                if electrons % 2:
                    raise ValueError("odd electron count")
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                self._stereo = len(Chem.FindPotentialStereo(mol)) > 0
            except Exception:
                self._stereo = True
        return self._stereo


def _molecules(symbols: list, coords) -> Optional[list]:
    """The separate molecules of a geometry (bonds perceived from distances), as _Mol."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    try:
        xyz = f"{len(symbols)}\n\n" + "".join(f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, (x, y, z) in zip(symbols, coords))
        mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineConnectivity(mol)
        out = []
        for idx, frag in zip(Chem.GetMolFrags(mol), Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)):
            out.append(_Mol([symbols[i] for i in idx], [coords[i] for i in idx], _skeleton(frag)))
        return out
    except Exception:
        return None


def _key(mols: list) -> tuple:
    return tuple(sorted(m.skeleton or "?" for m in mols))


def _same_molecules(a: list, b: list) -> str:
    """'same' when `a` and `b` pair up one for one as the same molecule, as the network compares species
    (mepd.compare: heavy-atom bond graphs, then stereochemistry from 3D), 'differ' when they cannot,
    'cant' when only an undecided stereochemistry stands in the way. (The skeletons only pick candidates.)"""
    from mepd.compare import compare_structures

    if _key(a) != _key(b):
        return "differ"
    left, undecided = list(b), False
    for ma in a:
        hit = None
        for k, mb in enumerate(left):
            if mb.skeleton != ma.skeleton:
                continue
            c = compare_structures(ma.node, mb.node)
            if c.same:
                hit = k
                break
            undecided = undecided or c.undetermined
        if hit is None:
            return "cant" if undecided else "differ"
        left.pop(hit)
    return "same"


def _changed_molecules(start: list, species: list) -> list:
    """A species' molecules less the start's, as its box depiction draws them (those that differ)."""
    left = list(species)
    for ms in start:
        for k, mb in enumerate(left):
            if mb.skeleton == ms.skeleton and _same_molecules([ms], [mb]) == "same":
                left.pop(k)
                break
    return left


def _event_products(run: Path, ev: dict, window: int = 12) -> Optional[list]:
    """The molecules an event's reacting atoms form just after its product frame: the bond graph seen in
    most of the next `window` MD frames (one hot frame can stretch a bond past its cutoff), on their
    averaged geometry."""
    from collections import Counter

    from mepd.web.nanoreactor import _segment

    atoms = list(ev.get("atoms") or [])
    if not atoms:
        return None
    pf = int(ev.get("product_frame", 0))
    frames, syms = [], None
    for f in range(pf, pf + window + 1, 2):
        fp = run / "md" / f"segment_{f // SEGMENT_FRAMES:03d}.xyz"
        try:
            sy, xyz = _segment(fp)
            frames.append(xyz[f % SEGMENT_FRAMES][atoms])
            syms = [sy[a] for a in atoms]
        except (OSError, IndexError, ValueError):
            continue
    if not frames:
        return None
    per = [(_molecules(syms, fr), fr) for fr in frames]
    per = [(m, fr) for m, fr in per if m]
    if not per:
        return None
    mode = Counter(_key(m) for m, _ in per).most_common(1)[0][0]
    import numpy as np

    avg = np.mean([fr for m, fr in per if _key(m) == mode], axis=0)
    mols = _molecules(syms, avg)
    return mols if mols and _key(mols) == mode else next(m for m, _ in per if _key(m) == mode)


def _is_exchange(label: str) -> bool:
    sides = str(label).split(" -> ")
    return len(sides) == 2 and sorted(x for x in _side(sides[0]) if x) == sorted(x for x in _side(sides[1]) if x)


def _label_key(label: str) -> Optional[tuple]:
    """The products the reactor's label names, as bond-graph skeletons."""
    sides = str(label).split(" -> ")
    if len(sides) != 2:
        return None
    out = []
    for part in sides[1].split(" + "):
        part = part.strip()
        n, _, smi = part.partition(" ")
        k = int(n) if n.isdigit() and smi else 1
        sk = _skeleton_of_smiles(smi if n.isdigit() and smi else part)
        if sk is None:
            return None
        out += [sk] * k
    return tuple(sorted(out))


class _Verifier:
    """Per job: the start's and each species' changed molecules (indexed by their skeletons)."""

    def __init__(self):
        self.start = None
        self.changed: dict = {}     # species id -> changed molecules
        self.by_key: dict = {}      # their skeletons -> [species ids]
        self.drawn: dict = {}       # species id -> the skeletons its card draws (its SMILES less the start's)

    def index(self, states: dict, out: Path) -> None:
        if self.start is None:
            st = states.get("0")
            sy, xyz = (st[1], st[2]) if st and st[1] else _species_file(out, "0")
            self.start = _molecules(sy, xyz) if sy else None
        if not self.start:
            return
        for sid, st in states.items():
            if sid in self.changed or sid == "0":
                continue
            sy, xyz = (st[1], st[2]) if st[1] else _species_file(out, sid)
            mols = _molecules(sy, xyz) if sy else None
            if not mols:
                continue
            ch = _changed_molecules(self.start, mols)
            self.changed[sid] = ch
            self.by_key.setdefault(_key(ch), []).append(sid)
            # A card draws a species from its SMILES (by the molecules that differ from the start's), or from
            # its geometry when it has none: a link must agree with that drawing too.
            smi, start_smi = st[0] if st else "", (states.get("0") or [""])[0]
            if smi and drawable(smi) and start_smi:
                from collections import Counter

                drawn = Counter(_skeleton_of_smiles(x) for x in smi.split("|")[0].split(".")) \
                    - Counter(_skeleton_of_smiles(x) for x in start_smi.split("|")[0].split("."))
                self.drawn[sid] = tuple(sorted(drawn.elements())) if None not in drawn else None
            else:
                self.drawn[sid] = _key(ch)

    def check(self, run: Path, ev: dict) -> tuple:
        """('same', species) when verified, else (verdict, None): 'differ', 'cant' or 'none' (no candidate)."""
        if _is_exchange(ev.get("label") or ""):
            return "none", None
        prods = _event_products(run, ev)
        if not prods:
            return "none", None
        if _label_key(ev.get("label") or "") != _key(prods):
            return "cant", None   # the frames and the reactor's label disagree on what formed
        verdicts = []
        for sid in self.by_key.get(_key(prods), []):
            if self.drawn.get(sid) != _key(prods):
                continue
            v = _same_molecules(prods, self.changed[sid])
            if v == "same":
                return "same", sid
            verdicts.append(v)
        return ("cant" if "cant" in verdicts else "differ" if verdicts else "none"), None


def _species_file(out: Path, sid: str):
    fp = Path(out) / "species" / f"species_{sid}.xyz"
    try:
        return _xyz_rows(fp.read_text())
    except OSError:
        return None, None


def mixture_product(run: Path, ev: dict, skeletons: dict) -> tuple:
    """(verdict, molecule) for a mixture: a product molecule of the event (from its frames) whose bond graph
    is a mixture molecule's, that is not one of its reactants; one that can have stereoisomers is not
    claimed ('cant'). `skeletons`: molecule SMILES -> skeleton."""
    if _is_exchange(ev.get("label") or ""):
        return "none", None
    prods = _event_products(run, ev)
    if not prods or _label_key(ev.get("label") or "") != _key(prods):
        return "none", None
    sides = str(ev.get("label") or "").split(" -> ")
    reac = set(_skeleton_of_smiles(x.strip().partition(" ")[2] if x.strip().partition(" ")[0].isdigit() else x.strip())
               for x in sides[0].split(" + "))
    for m in prods:
        if m.skeleton in reac:
            continue
        for smi, sk in skeletons.items():
            if sk and sk == m.skeleton:
                return ("cant", None) if _may_have_stereo_smiles(smi.split("|")[0]) else ("same", smi)
    return "none", None
