"""Every Nanoreactor run of a session as one list of reactions (Reactions › Nanoreactor runs): each reaction
once, however many runs found it and whichever way each run wrote it, with how often each run saw it (labelled
by the run's conditions), whether it was refined, and the session reaction that stands for it (its TS search).

A run is read from its output folder: network.json once its analysis is done, else what it has written so far
(live_events.json, rewritten after each simulated ps; live_network.json, the reactions refined while the MD
runs). Run longer continues a run in the same folder, so a folder is one run. A reaction's identity is the
nanoreactor's own key (each side's species as SMILES|charge|spin, direction-free). A reaction with a species
whose bonding could not be read (SMILES '?...' or 'unusual bonding') stays with its own run: two of those need
not be the same molecule.

The runs and their session reactions are unchanged: this only reads them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_CACHE: dict = {}          # output folder -> (file mtimes, its reactions)
_ENDS = re.compile(r"\|-?\d+\|\d+$")


def _split_side(side: str) -> list:
    """'A|0|1+[NH4+]|1|1' -> ['A|0|1', '[NH4+]|1|1']: a '+' ends a species only after its '|charge|spin'."""
    out, cur = [], ""
    for part in side.split("+"):
        cur = f"{cur}+{part}" if cur else part
        if _ENDS.search(cur):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def key_sides(key: str) -> tuple[list, list]:
    a, _, b = key.partition(">")
    return _split_side(a), _split_side(b)


def _smiles(species_key: str) -> str:
    return species_key.rsplit("|", 2)[0]


def _label_sides(label: str) -> tuple[list, list]:
    a, _, b = str(label).partition(" -> ")
    return a.split(" + "), b.split(" + ")


def as_key(key: str, label: str) -> bool:
    """Whether `label` ('A + B -> C', SMILES) is written the way `key` is."""
    return sorted(_label_sides(label)[0]) == sorted(_smiles(s) for s in key_sides(key)[0])


def _unclear(smiles_list) -> bool:
    return any(str(s).startswith("?") or "unusual bonding" in str(s) for s in smiles_list)


def _json(fp: Path):
    try:
        return json.loads(Path(fp).read_text())
    except (OSError, ValueError):
        return None


def _sp_key(sp: dict) -> str:
    return sp.get("key") or f"{sp['smiles']}|{sp['charge']}|{sp['multiplicity']}"


def _rx_key(rx: dict, keys: dict) -> str:
    if rx.get("key"):
        return rx["key"]
    a = "+".join(sorted(keys[i] for i in rx["reactants"]))
    b = "+".join(sorted(keys[i] for i in rx["products"]))
    return min(f"{a}>{b}", f"{b}>{a}")


def _reason(rx: dict) -> Optional[str]:
    c = rx.get("complex") or {}
    return c.get("reason") or ("no TS endpoints" if c.get("error") else None)


def _row(n=0, n_rev=0, first_fs=None, unclear=False, written_as_key=True) -> dict:
    return {"n": n, "n_rev": n_rev, "delta_e": None, "refined": False, "reason": None, "first_fs": first_fs,
            "unclear": unclear, "written_as_key": written_as_key}


def _from_reaction(row: dict, key: str, rx: dict) -> None:
    """A run's reaction record into its row, counted in the key's direction; `written_as_key`: how the run
    (and so its session reaction) writes it."""
    fwd = as_key(key, rx.get("label", ""))
    n, n_rev = int(rx.get("count") or 0), int(rx.get("reverse_count") or 0)
    if row["n"] + row["n_rev"] == 0:
        row["n"], row["n_rev"] = (n, n_rev) if fwd else (n_rev, n)
    de = rx.get("delta_e_kcal")
    if de is not None:
        row["delta_e"], row["refined"] = (de if fwd else -de), True
    row["reason"] = row["reason"] or _reason(rx)
    row["written_as_key"] = fwd
    if row["first_fs"] is None:
        row["first_fs"] = rx.get("first_fs")


def run_reactions(out: Path) -> tuple[dict, bool]:
    """One run's reactions by key, counted in the key's own direction ({key: row}), and whether its analysis
    is final. Cached by the files' times."""
    out = Path(out)
    files = [out / "network.json", out / "live_network.json", out / "live_events.json"]
    stamps = tuple(fp.stat().st_mtime if fp.exists() else 0.0 for fp in files)
    hit = _CACHE.get(str(out))
    if hit and hit[0] == stamps:
        return hit[1]
    got: dict = {}
    final = _json(files[0])
    done = bool(final and final.get("reactions") is not None)
    if done:
        sources = [final]
    else:
        # So far: the settled events (each with its reaction's key), then what is refined already.
        for e in (_json(files[2]) or {}).get("events") or []:
            key = e.get("key")
            if not key or e.get("tentative") or e.get("reaction") is None:
                continue
            a, b = _label_sides(e.get("label", ""))
            row = got.setdefault(key, _row(first_fs=e.get("start_fs"), unclear=_unclear(a + b),
                                           written_as_key=as_key(key, e.get("label", ""))))
            row["n" if as_key(key, e.get("label", "")) else "n_rev"] += 1
        sources = [_json(files[1]) or {}]
    for data in sources:
        keys = {sp["id"]: _sp_key(sp) for sp in data.get("species") or []}
        smiles = {sp["id"]: sp.get("smiles") for sp in data.get("species") or []}
        for rx in data.get("reactions") or []:
            ids = rx.get("reactants", []) + rx.get("products", [])
            if not all(i in keys for i in ids):
                continue
            key = _rx_key(rx, keys)
            _from_reaction(got.setdefault(key, _row(unclear=_unclear(smiles[i] for i in ids))), key, rx)
    _CACHE[str(out)] = (stamps, (got, done))
    return got, done


def _flip(row: dict) -> dict:
    neg = lambda v: None if v is None else -v   # noqa: E731
    return {**row, "smiles": [row["smiles"][1], row["smiles"][0]], "n": row["n_rev"], "n_rev": row["n"],
            "delta_e": neg(row["delta_e"]),
            "members": [{**m, "n": m["n_rev"], "n_rev": m["n"], "delta_e": neg(m["delta_e"])} for m in row["members"]]}


def merge(runs: list[dict]) -> list[dict]:
    """runs: [{"run": id, "reactions": {key: row of run_reactions}, "ws": {key: (session reaction id, has TS
    endpoints)}}] -> one row per reaction: {key, smiles: [left, right], n, n_rev, delta_e, stage ('refined' |
    'seen'), reason, first_fs, members: [{run, n, n_rev, delta_e, refined, reason, reaction, edge}], rep, runs}.

    The representative (`rep`) is the session reaction the row shows (its card, its TS search): one with TS
    endpoints, from the run that saw it most. The row is written its way, so its counts, ΔE and barrier read in
    one direction; without one, the way it was seen most."""
    rows: dict = {}
    for r in runs:
        for key, x in r["reactions"].items():
            rid, edge = (r.get("ws") or {}).get(key, (None, False))
            k = f"{r['run']}:{key}" if x.get("unclear") else key
            row = rows.setdefault(k, {"key": k, "rxkey": key, "members": []})
            row["members"].append({"run": r["run"], "n": x["n"], "n_rev": x["n_rev"], "delta_e": x["delta_e"],
                                   "refined": x["refined"], "reason": x["reason"], "first_fs": x["first_fs"],
                                   "reaction": rid, "edge": bool(edge), "as_key": x["written_as_key"]})
    out = []
    for row in rows.values():
        ms = row["members"]
        n, n_rev = sum(m["n"] for m in ms), sum(m["n_rev"] for m in ms)
        rep = max((m for m in ms if m["reaction"]), key=lambda m: (m["edge"], m["n"] + m["n_rev"]), default=None)
        refined = [m for m in ms if m["delta_e"] is not None]
        left, right = key_sides(row["rxkey"])
        first = [m["first_fs"] for m in ms if m["first_fs"] is not None]
        shown = {"key": row["key"], "smiles": [[_smiles(s) for s in left], [_smiles(s) for s in right]],
                 "n": n, "n_rev": n_rev,
                 "delta_e": max(refined, key=lambda m: m["n"] + m["n_rev"])["delta_e"] if refined else None,
                 "stage": "refined" if refined else "seen",
                 "reason": next((m["reason"] for m in ms if m["reason"]), None),
                 "first_fs": min(first) if first else None,
                 "members": [{f: m[f] for f in ("run", "n", "n_rev", "delta_e", "refined", "reason", "reaction", "edge")}
                             for m in ms],
                 "rep": rep["reaction"] if rep else None, "runs": len({m["run"] for m in ms})}
        out.append(_flip(shown) if (not rep["as_key"] if rep else n_rev > n) else shown)
    return out


def _label_runs(runs: list[dict]) -> None:
    """Each run's label (its conditions): its temperature, and whatever else differs between the runs (time,
    squeeze, spin, the box); a run steered by hand is 'steered'. Its name adds #k among repeats."""
    md = [r for r in runs if not r["steered"]]
    vary = {f: len({r[f] for r in md}) > 1 for f in ("time_ps", "compress", "multiplicity")}
    boxes: dict = {}
    for r in md:
        boxes.setdefault(r["_box"], len(boxes) + 1)
    for r in runs:
        if r["steered"]:
            r["label"] = "steered"
            continue
        bits = [f"{r['temperature']:g} K"]
        if vary["time_ps"]:
            bits.append(f"{r['time_ps']:g} ps")
        if vary["compress"]:
            bits.append(f"squeeze {r['compress']:g}")
        if vary["multiplicity"] and r["multiplicity"]:
            bits.append(f"spin {r['multiplicity']}")
        if len(boxes) > 1:
            bits.append(f"box {boxes[r['_box']]}")
        r["label"] = " · ".join(bits)
    # `name`: the run itself, numbered among the runs with its conditions (repeats).
    seen: dict = {}
    for r in runs:
        seen[r["label"]] = seen.get(r["label"], 0) + 1
        r["name"] = r["label"]
    k: dict = {}
    for r in runs:
        if seen[r["label"]] > 1:
            k[r["label"]] = k.get(r["label"], 0) + 1
            r["name"] = f"{r['label']} #{k[r['label']]}"


def collect(ws, jobs: list[dict]) -> dict:
    """The session's Nanoreactor runs and their reactions, merged (for /api/nanoreactor/reactions)."""
    folders: dict = {}
    for j in sorted(jobs, key=lambda j: j.get("created") or 0):
        if j.get("op") in ("nanoreactor", "nanoreactor-more") and j.get("output_dir"):
            folders.setdefault(str(Path(j["output_dir"]).resolve()), []).append(j)
    snap = ws.snapshot()
    reactions, edges = snap.get("reactions") or {}, snap.get("edges") or {}
    runs, data = [], []
    for out, group in folders.items():
        base = next((j for j in group if j["op"] == "nanoreactor"), None)
        if base is None:
            continue
        p = base.get("params") or {}
        ids = {j["id"] for j in group}
        got, final = run_reactions(Path(out))
        wsr = {}
        for rid, rec in reactions.items():
            o = rec.get("origin") or {}
            if o.get("job") in ids and o.get("key") in got:
                wsr[o["key"]] = (rid, bool(rec.get("edge") and rec["edge"] in edges))
        busy = [j for j in group if j.get("status") in ("running", "queued")]
        more = sum(float((j.get("params") or {}).get("more_ps") or 0) for j in group if j["op"] == "nanoreactor-more")
        runs.append({"run": base["id"], "title": base.get("title"),
                     "status": busy[-1]["status"] if busy else group[-1].get("status"), "final": final,
                     "temperature": float(p.get("temperature") or 0), "steered": bool(p.get("trajectory")),
                     "time_ps": float(p.get("time_ps") or 0) + more, "compress": float(p.get("compress") or 0),
                     "multiplicity": p.get("multiplicity"), "created": base.get("created"),
                     "_box": (tuple((base.get("targets") or {}).get("structures") or []), str(p.get("copies")))})
        data.append({"run": base["id"], "reactions": got, "ws": wsr})
    _label_runs(runs)
    for r in runs:
        r.pop("_box")
    return {"runs": runs, "reactions": merge(data)}
