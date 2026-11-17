"""Read a finished `mepd addmol test` run's verified rows again from its own TS and IRC files, with the code
as it is now, and give the result file that follows.

    reread_rows.py QUERY.json RESULT.json FILES_DIR PROFILE.toml OUT.json [--rows all|relay+takes] [--workers N]

No search is run: every row the result file kept (verification and relay scans; `ts_file`, `irc_file` under
FILES_DIR, as packed by addmol_task.sh) is put on the parent's scale again by mepd.cli_addmol._ts_row: whether
the molecule takes part, which molecule is the added one, the barrier. A row whose files are missing, or whose
IRC no longer reads as its step, is kept as it was and counted. The read-out is then computed from the rows
(mepd.addmol.result). OUT.json is RESULT.json with those rows and that read-out, and `reread`: what changed.
`--rows relay+takes`: only the relay rows and the rows that said "takes part" (a large molecule's Hessians are slow).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

FIELDS = ("label", "takes_part", "returned", "bonds_broken", "bonds_formed", "reactants", "products", "level",
          "reference", "counted", "added_form", "free_from_molecules_minimized_alone", "spectators", "barrier_method")


def one(args):
    query_fp, entry, files, profile, molecule = args
    from qcdata import Structure

    from mepd import cli_addmol as c
    from mepd.discovery import molecules as M

    try:
        engine = c._engine(profile)
        if query_fp not in c._QUERY:
            c._QUERY[query_fp] = json.loads(Path(query_fp).read_text())
        q = c._QUERY[query_fp]
        ts_fp, irc_fp = Path(files) / entry["ts_file"], Path(files) / entry["irc_file"]
        ends = ts_fp.parent if entry["layer"] == "verify" else ts_fp.parent.parent.parent
        keys = lambda f: sorted(k for _a, k in M.molecule_parts(Structure.open(str(ends / f))))   # noqa: E731
        ts = Structure.open(str(ts_fp))
        irc = Structure.open_multi(str(irc_fp))
        q_ = {"charge": int(ts.charge), "multiplicity": int(ts.multiplicity)}
        label = entry["id"].split("|", 1)[1]
        row = c._ts_row(q, entry["parent_step"], label, entry["layer"], ts, irc[0].model_copy(update=q_),
                        irc[-1].model_copy(update=q_), set(), molecule, engine, keys("start.xyz"), keys("end.xyz"))
        if row["kind"] == "elsewhere":
            return entry["id"], None, "its IRC ends are not the ends kept beside it"
        return entry["id"], row["entry"], None
    except Exception as exc:
        return entry["id"], None, f"{type(exc).__name__}: {exc}"


def main(query_fp, result_fp, files, profile, out_fp, rows="all", workers=4):
    import multiprocessing as mp

    from mepd import addmol
    from mepd import cli_addmol as c
    from mepd.cli_common import _open_run_inputs

    q = json.loads(Path(query_fp).read_text())
    doc = json.loads(Path(result_fp).read_text())
    cand = doc["candidate"]
    molecule = c._prepare({"smiles": cand["smiles"], "amount": cand["amount"]}, _open_run_inputs(Path(profile)).engine)
    entries = doc["estimates"]["new_steps"]
    todo = [n for n in entries if n.get("ts_file") and not n.get("mirror_image_of")
            and (rows == "all" or n["layer"] == "relay" or n.get("takes_part"))]
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=mp.get_context("spawn")) as pool:
        got = {i: (e, why) for i, e, why in pool.map(one, [(query_fp, n, files, profile, molecule) for n in todo])}
    changes = {"rows_read_again": len(todo), "kept_as_they_were": [], "takes_part_to_beside": [], "beside_to_takes_part": [],
               "named_added_now": 0, "barrier_changed_by_more_than_0.1": []}
    by_id = {n["id"]: n for n in entries}
    for n in todo:
        new, why = got[n["id"]]
        if new is None:
            changes["kept_as_they_were"].append({"row": n["id"], "why": why})
            continue
        if n["takes_part"] != new["takes_part"]:
            changes["takes_part_to_beside" if n["takes_part"] else "beside_to_takes_part"].append(n["id"])
        if addmol.ADDED_ID in new["reactants"] and addmol.ADDED_ID not in n["reactants"]:
            changes["named_added_now"] += 1
        if abs(new["barrier"]["free"] - n["barrier"]["free"]) > 0.1:
            changes["barrier_changed_by_more_than_0.1"].append([n["id"], n["barrier"]["free"], new["barrier"]["free"]])
        n.update({k: new[k] for k in FIELDS if k in new})
        n["barrier"] = {**n["barrier"], **new["barrier"]}
        twin = by_id.get(n["id"] + "~mirror")
        if twin is not None:
            image = c._mirror_entry(q, n, False)
            if image is not None:
                twin.update({k: image[k] for k in FIELDS + ("barrier",) if k in image})
    out = addmol.result(q, cand, doc["estimates"], status=doc["status"], layers=doc["layers"], failed=doc["failed"])
    for k, v in doc.items():
        out.setdefault(k, v)
    out["reread"] = {"what": "the rows of this file read again from the run's TS and IRC files (reread_rows.py); no "
                             "search was run; the read-out follows from the rows", "of": str(Path(result_fp).name), **changes}
    addmol.write_json(out_fp, out)
    r, o = out["readout"], doc["readout"]
    print(json.dumps({"changes": {k: (v if isinstance(v, int) else len(v)) for k, v in changes.items()},
                      "target_before": o["target"]["with"]["final"], "target_now": r["target"]["with"]["final"],
                      "parent": r["target"]["parent"]["final"], "fate_before": o["added"]["in_words"],
                      "fate_now": r["added"]["in_words"],
                      "route_peak_before": o["peak_routes"]["with"][0]["peak"],
                      "route_peak_now": r["peak_routes"]["with"][0]["peak"]}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = {"rows": "all", "workers": 4}
    while len(a) > 5:
        opt[a[5].lstrip("-")] = a[6]
        del a[5:7]
    main(*a, **opt)
