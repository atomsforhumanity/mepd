"""What a local run did with the whole-box TSs that have a molecule taking no part (from the run's own files).

    python benchmarks/diagnostics/thermo/run_check.py OUT_DIR REFERENCE_DIR CHECK_DIR

OUT_DIR: local_run.py's folder (run/, expand.log, network_aware.jsonl, result_*.json). REFERENCE_DIR: an unpacked
recorded run of the same task (run/summary.json, expand.log, network_aware.jsonl). Writes CHECK_DIR/run_check.md
and .json: how it ended; tracebacks and warning kinds against the reference; every verified TS's file record
against what its step says it was counted as; the whole-box TSs whose IRC ends leave a molecule as it was, their
pairing jobs in the event log and their `_sub` steps (file record, counted_as, the queue's own IRC record); steps
removed that involve a `_sub`; anything paired twice or again; job seconds by kind for both runs. Reads no
kinetics result.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def kinds(texts):
    return Counter(re.sub(r"[-+]?\d+(\.\d+)?", "#", t)[:70] for t in texts)


def jobs(fp):
    tot, n, bad = defaultdict(float), Counter(), Counter()
    events = [json.loads(x) for x in Path(fp).read_text().splitlines() if x.strip()] if Path(fp).exists() else []
    for e in events:
        if e.get("event") == "job":
            tot[e["kind"]] += e.get("run_s") or 0.0
            n[e["kind"]] += 1
            bad[e["kind"]] += not e.get("ok")
    return events, {k: {"jobs": n[k], "seconds": round(tot[k], 1), "failed": bad[k]} for k in sorted(n)}


def main(out_dir, ref_dir, check_dir):
    from qcdata import Structure

    import mepd.discovery.free_energy as fe

    out, ref, chk = Path(out_dir), Path(ref_dir), Path(check_dir)
    chk.mkdir(parents=True, exist_ok=True)
    s = json.loads((out / "run/summary.json").read_text())
    r = json.loads((ref / "run/summary.json").read_text())
    legs = [json.loads(fp.read_text()) for fp in sorted(out.glob("result_*.json"))]
    log, rlog = (out / "expand.log").read_text(), (ref / "expand.log").read_text()
    res = {"legs": [{k: x[k] for k in ("exit", "killed", "wall_s", "resume", "workers")} for x in legs],
           "stopped": (s.get("rounds") or [{}])[-1].get("stopped"), "tracebacks": log.count("Traceback"),
           "reference_tracebacks": rlog.count("Traceback")}
    wk, rk = kinds(s.get("warnings") or []), kinds(r.get("warnings") or [])
    lk = kinds([x for x in log.splitlines() if x.startswith("WARNING")])
    rlk = kinds([x for x in rlog.splitlines() if x.startswith("WARNING")])
    res["summary_warning_kinds_new"] = {k: v for k, v in wk.items() if k not in rk}
    res["log_warning_kinds_new"] = {k: v for k, v in lk.items() if k not in rlk}
    res["summary_warnings"], res["reference_summary_warnings"] = sum(wk.values()), sum(rk.values())
    sp = {x["index"]: x for x in s["species"]}
    steps = {st["label"]: st for st in s["steps"]}
    mism, no_record, n_ver, records = [], [], 0, Counter()
    population, subs = [], []
    for st in s["steps"]:
        files = st.get("files") or {}
        fp = out / files["ts"] if files.get("ts") else None
        if files.get("verified") is not True or fp is None or not fp.exists():
            continue
        n_ver += 1
        ts = Structure.open(str(fp))
        on = fe.optimized_on(ts)
        n = len(ts.symbols)
        records["none" if on is None else "whole" if len(on) == n else "subsystem"] += 1
        species = tuple(Structure.open(str(out / sp[k]["file"])) for k in (st["a"], st["b"]))
        ends = fe.step_ends({"ts": str(fp), "irc": str(out / files["irc"]) if files.get("irc") else None}, species)
        groups, source = fe.counted_as(ts, ends)
        want = [len(g) for g in groups] if groups else [n]
        ca = (st.get("ts_energy_terms") or {}).get("counted_as")
        if (st.get("ts_energy_terms") or {}).get("thermal_correction_kcal") is not None:
            if ca is None:
                no_record.append(st["label"])
            elif ca["molecules"] != want or ca["from"] != source or (on is not None and ca["molecules"][0] != len(on)):
                mism.append({"step": st["label"], "file": want, "record": ca})
        if on is not None and len(on) == n and "rotamer_of" not in files and files.get("irc"):
            bare = ends[0].model_copy(update={"extras": {}})
            parts = fe.FreeEnergies(None, 298.15).parts(bare, True, ends)
            if len(parts) > 1:
                population.append({"step": st["label"], "a": st["a"], "b": st["b"], "subsystem": len(parts[0][0]),
                                   "left_out": n - len(parts[0][0])})
        if st["label"].endswith("_sub"):
            subs.append({"step": st["label"], "a": st["a"], "b": st["b"], "optimized_on": on if on is None else len(on),
                         "atoms": n, "counted_as": ca, "verified": files.get("verified"), "rotamer_of": files.get("rotamer_of"),
                         "irc": st.get("irc"), "ts_group": st.get("ts_group"),
                         "parent_ts_group": (steps.get(st["label"][:-4]) or {}).get("ts_group"),
                         "parent_counted_as": ((steps.get(st["label"][:-4]) or {}).get("ts_energy_terms") or {}).get("counted_as")})
    unverified_subs = [st["label"] for st in s["steps"] if st["label"].endswith("_sub")
                       and (st.get("files") or {}).get("verified") is not True]
    events, mine = jobs(out / "network_aware.jsonl")
    _, theirs = jobs(ref / "network_aware.jsonl")
    pj = [e for e in events if e.get("event") == "job" and e.get("kind") == "rotamer"
          and "on its reacting molecules" in (e.get("what") or "")]
    by_what = Counter(e["what"] for e in pj)
    removed = [x for x in s.get("steps_removed") or [] if str(x.get("label", "")).endswith("_sub")
               or str(x.get("of", "")).endswith("_sub")]
    files_sub = sorted(p.stem for p in (out / "run/ts").glob("*_sub.xyz"))
    have = {x["step"] for x in subs} | set(unverified_subs)
    res.update({
        "verified_ts": n_ver, "ts_file_records": dict(records), "counted_as_mismatches": mism,
        "corrected_without_counted_as": no_record, "whole_box_with_unchanged_molecule": population,
        "pairing_jobs": {"run": len(pj), "ok": sum(1 for e in pj if e.get("ok")),
                         "failed": [{"what": e["what"], "why": e.get("why")} for e in pj if not e.get("ok")],
                         "seconds": round(sum(e.get("run_s") or 0 for e in pj), 1),
                         "run_more_than_once": {k: v for k, v in by_what.items() if v > 1}},
        "sub_ts_files": files_sub, "sub_steps": subs, "sub_steps_unverified": unverified_subs,
        "sub_files_without_a_step": [x for x in files_sub if x not in have],
        "population_without_sub_file": [p["step"] for p in population if p["step"] + "_sub" not in files_sub],
        "removed_involving_sub": removed,
        "sub_of_sub": [st["label"] for st in s["steps"] if "_sub_sub" in st["label"]]
        + [p for p in files_sub if "_sub_sub" in p],
        "merged_in_kinetics": [x["step"] for x in subs if x["ts_group"] is not None and x["ts_group"] == x["parent_ts_group"]],
        "job_seconds": mine, "reference_job_seconds": theirs})
    (chk / "run_check.json").write_text(json.dumps(res, indent=1))
    L = [f"Legs: {res['legs']}; stop reason {res['stopped']}; tracebacks {res['tracebacks']} (reference "
         f"{res['reference_tracebacks']}); summary warnings {res['summary_warnings']} (reference "
         f"{res['reference_summary_warnings']}); kinds not in the reference: summary {res['summary_warning_kinds_new']}, "
         f"log {res['log_warning_kinds_new']}", "",
         f"Verified TSs {n_ver}; file records {dict(records)}; counted otherwise than the file says: {len(mism)}; "
         f"with a correction but no counted_as: {len(no_record)}", "",
         f"Whole-box verified TSs whose IRC ends leave a molecule as it was: {len(population)}; pairing jobs run "
         f"{len(pj)} (ok {res['pairing_jobs']['ok']}, {res['pairing_jobs']['seconds']} s), run more than once "
         f"{len(res['pairing_jobs']['run_more_than_once'])}; `_sub` TS files {len(files_sub)}; `_sub` steps verified "
         f"{len(subs)}, unverified {len(unverified_subs)}; files without a step {len(res['sub_files_without_a_step'])}; "
         f"removed steps involving a `_sub` {len(removed)}: {dict(Counter(x['reason'] for x in removed))}; `_sub` of a "
         f"`_sub`: {len(res['sub_of_sub'])}; in one kinetics TS group with its parent: {len(res['merged_in_kinetics'])}", "",
         "| `_sub` step | pair | optimized on (of atoms) | counted_as | parent counted_as | verified | queue's IRC record |",
         "|---|---|---|---|---|---|---|"]
    for x in subs:
        L.append(f"| {x['step']} | {x['a']}-{x['b']} | {x['optimized_on']} ({x['atoms']}) | {x['counted_as']} | "
                 f"{x['parent_counted_as']} | {x['verified']} | {x['irc']} |")
    L += ["", "Pairing jobs that failed or returned nothing:"] + [f"- {e['what']}: {e['why']}" for e in res["pairing_jobs"]["failed"]]
    L += ["", "Removed steps involving a `_sub`:"] + [f"- {json.dumps(x)}" for x in removed]
    L += ["", "| job kind | jobs | seconds | failed | reference jobs | reference seconds | reference failed |", "|" + "---|" * 7]
    for k in sorted(set(mine) | set(theirs)):
        a, b = mine.get(k, {}), theirs.get(k, {})
        L.append(f"| {k} | {a.get('jobs', 0)} | {a.get('seconds', 0)} | {a.get('failed', 0)} | {b.get('jobs', 0)} | "
                 f"{b.get('seconds', 0)} | {b.get('failed', 0)} |")
    (chk / "run_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
