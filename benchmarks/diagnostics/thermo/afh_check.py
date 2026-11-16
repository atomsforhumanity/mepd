"""Two run sets of the same tasks (before, after): thermal corrections and provisional closures against the
cutoff + 15, the pairing's jobs, and the whole-box TSs with a bystander that got no subsystem TS.

    python benchmarks/diagnostics/thermo/afh_check.py BEFORE_ROOT AFTER_ROOT OUT_DIR

Each root holds <run>/run/summary.json, the files it names and <run>/network_aware.jsonl. Reads records, TS files
and event logs; computes no energy. Per run:
  - steps over the cutoff + 15 kcal/mol on electronic energy: how many, how many open, how many in a window;
    steps closed provisionally; for the `before` set, whether the cache key of a TS's correction as its job made
    it (its step's ends file) equals the one the main process asks for (its IRC's ends): a correction the job
    computed under another key is never found, and the step is then closed provisionally;
  - `after`: the pairing jobs (count, worker seconds), the `_sub` steps, their IRC jobs, the `_sub` steps open
    and of those the ones whose parent lies over the cutoff + 15;
  - `after`: verified whole-box TSs whose IRC ends leave a molecule as it was and that have no `_sub` step, by
    reason, and how many of them are open.
No product ratio or score is read.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

H = 627.5094740631


def main(before, after, out_dir):
    from qcdata import Structure

    import mepd.discovery.free_energy as fe

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res, lines = {}, []
    for name, root in (("before", Path(before)), ("after", Path(after))):
        lines += [f"## {name}: {root.name}", "",
                  "| run | steps | over cutoff + 15 (electronic) | of those open | in a window | closed provisionally | "
                  "pairing jobs (worker s) | `_sub` steps (parent over + 15) | IRC jobs of `_sub` (s) | open `_sub` "
                  "(parent over + 15) | no correction: job's key is not the main's / other |", "|" + "---|" * 11]
        for fp in sorted(root.glob("*/run/summary.json")):
            d = fp.parent.parent
            s = json.loads(fp.read_text())
            e0, cut = s["seed_energy"], s["steering"]["cutoff_kcal"]
            steps = {x["label"]: x for x in s["steps"]}

            def over(x):
                return (x["ts_energy"] - e0) * H > cut + 15.0
            ev = [json.loads(x) for x in (d / "network_aware.jsonl").read_text().splitlines() if x.strip()]
            jobs = [e for e in ev if e.get("event") == "job"]
            pj = [e for e in jobs if e["kind"] == "rotamer" and "on its reacting" in (e.get("what") or "")]
            ij = [e for e in jobs if e["kind"] == "irc" and "_sub" in (e.get("what") or "")]
            subs = [x for x in s["steps"] if x["label"].endswith("_sub") and x["label"][:-4] in steps]
            ov = [x for x in s["steps"] if over(x)]
            key = Counter()
            free = fe.FreeEnergies(None, 298.15)
            for x in s["steps"]:
                if (x.get("ts_energy_terms") or {}).get("thermal_correction_kcal") is not None:
                    continue
                f = x["files"]
                ts, ef = d / f["ts"], (d / f["ts"]).with_name(Path(f["ts"]).stem + "_ends.xyz")
                irc = d / f["irc"] if f.get("irc") else None
                if not (ts.exists() and ef.exists() and irc is not None and irc.exists()):
                    key["other"] += 1
                    continue
                st = Structure.open(str(ts))
                a, b = Structure.open_multi(str(ef)), Structure.open_multi(str(irc))
                key["differs" if free._key(st, True, (a[0], a[-1])) != free._key(st, True, (b[0], b[-1]))
                    else "other"] += 1
            row = {"steps": len(s["steps"]), "over": len(ov), "over_open": sum(1 for x in ov if x.get("closed") is None),
                   "over_open_window": sum(1 for x in ov if x.get("closed") is None and x.get("deciding_window")),
                   "provisional": sum(1 for x in s["steps"] if (x.get("closed") or {}).get("provisional")),
                   "pairing_jobs": len(pj), "pairing_s": round(sum(e.get("run_s") or 0 for e in pj), 1),
                   "sub": len(subs), "sub_parent_over": sum(1 for x in subs if over(steps[x["label"][:-4]])),
                   "sub_irc_jobs": len(ij), "sub_irc_s": round(sum(e.get("run_s") or 0 for e in ij), 1),
                   "sub_open": sum(1 for x in subs if x.get("closed") is None),
                   "sub_open_parent_over": sum(1 for x in subs if x.get("closed") is None and over(steps[x["label"][:-4]])),
                   "no_correction_key_differs": key["differs"], "no_correction_other": key["other"]}
            if name == "after":
                done = {e["what"].split(" · ")[1].split(" on its")[0]: e for e in pj}
                removed = {x["label"]: x for x in s.get("steps_removed") or []}
                why, why_open = Counter(), Counter()
                for x in s["steps"]:
                    f = x["files"]
                    if f.get("verified") is not True or not f.get("irc") or f.get("rotamer_of") \
                            or f.get("ts_kind") == "path_peak" or not (d / f["ts"]).exists():
                        continue
                    ts = Structure.open(str(d / f["ts"]))
                    on = fe.optimized_on(ts)
                    if on is None or len(on) != len(ts.symbols):
                        continue
                    ends = fe.step_ends({"ts": str(d / f["ts"]), "irc": str(d / f["irc"])}, ())
                    if len(free.parts(ends[0].model_copy(update={"extras": {}}), True, ends)) == 1:
                        continue
                    lab = x["label"]
                    if lab + "_sub" in steps:
                        w = "has a `_sub` step"
                    elif lab not in done:
                        w = "no pairing job in the event log"
                    elif not done[lab].get("ok"):
                        w = "job failed: " + str(done[lab].get("why"))[:60]
                    elif not os.path.exists(d / f"run/ts/{lab}_sub.xyz"):
                        w = "job kept nothing (the reacting core moved)"
                    elif lab + "_sub" in removed:
                        w = "`_sub` removed: " + removed[lab + "_sub"]["reason"]
                    else:
                        w = "`_sub` file without a step"
                    why[w] += 1
                    if x.get("closed") is None:
                        why_open[w] += 1
                row["whole_box_with_bystander"] = dict(why)
                row["of_those_open"] = dict(why_open)
            res[f"{name}:{d.name}"] = row
            lines.append(f"| {d.name} | {row['steps']} | {row['over']} | {row['over_open']} | {row['over_open_window']} | "
                         f"{row['provisional']} | {row['pairing_jobs']} ({row['pairing_s']}) | {row['sub']} "
                         f"({row['sub_parent_over']}) | {row['sub_irc_jobs']} ({row['sub_irc_s']}) | {row['sub_open']} "
                         f"({row['sub_open_parent_over']}) | {row['no_correction_key_differs']} / "
                         f"{row['no_correction_other']} |")
        lines.append("")
        if name == "after":
            lines += ["Verified whole-box TSs with a molecule that takes no part, by what became of them (open ones "
                      "in brackets):", ""]
            for k, row in res.items():
                if k.startswith("after:") and row.get("whole_box_with_bystander"):
                    lines.append(f"- {k[6:]}: " + "; ".join(
                        f"{w}: {v} [{row['of_those_open'].get(w, 0)}]" for w, v in sorted(
                            row["whole_box_with_bystander"].items(), key=lambda kv: -kv[1])))
    (out_dir / "afh_check.json").write_text(json.dumps(res, indent=1))
    (out_dir / "afh_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
