"""What the explorer's recording costs its main process: from a cProfile dump of a `mepd discovery expand` run
(`python -m cProfile -o PROFILE.out -m mepd.cli discovery expand ...`), the cumulative seconds and calls of the
functions that only record (network_expansion's _final_record, _stamp_opened, _left, _irc_why, _now, _rec), beside
the whole run's. (The one recording call in the workers, a proposal's stereo key in the propose job, is not in the
main process's profile.)

    python benchmarks/known/record_cost.py PROFILE.out OUT.json
"""
import json
import pstats
import sys

RECORDING = ("_final_record", "_stamp_opened", "_left", "_irc_why", "_now", "_rec")


def main(argv):
    stats = pstats.Stats(argv[0]).stats
    rows = {}
    for (fname, _line, func), (_cc, ncalls, _tt, cumulative, _callers) in stats.items():
        if func in RECORDING and fname.endswith("network_expansion.py"):
            rows[func] = {"calls": ncalls, "cumulative_s": round(cumulative, 4)}
    total = max(v[3] for v in stats.values())
    # _final_record and _left call the others: the sum of these two and _stamp_opened, _irc_why bounds the cost
    # (plus _now and _rec where they are called directly)
    bound = sum(r["cumulative_s"] for r in rows.values())
    out = {"profile": argv[0], "run_main_process_s": round(total, 2), "recording": rows,
           "recording_upper_bound_s": round(bound, 4), "share_of_main_process": round(bound / total, 6)}
    with open(argv[1], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
