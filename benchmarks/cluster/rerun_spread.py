"""The same-seed rerun spread of the settle time from two arms of the same run code (as score.control_spread does
for control reruns: per system the largest |settle(A, r) - settle(B, r)| over the replicates both ran; a pair with
a run that never settled is left out and counted), written as results/NAME/aggregate.json so that
gate.py --spread-from NAME uses it.

    python benchmarks/cluster/rerun_spread.py RESULT_A:CONFIG RESULT_B:CONFIG --as NAME
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recall_noise import RESULTS, runs_of  # noqa: E402


def main(argv: list) -> int:
    name = argv[argv.index("--as") + 1]
    (ra, ca), (rb, cb) = (a.split(":", 1) for a in argv[:2])
    A, B = runs_of(ra, ca), runs_of(rb, cb)
    spread, pairs, never = {}, {}, {}
    for key in sorted(set(A) & set(B)):
        sy = key[0]
        x = (A[key].get("settle") or {}).get("settle_s") if A[key].get("ok") else None
        y = (B[key].get("settle") or {}).get("settle_s") if B[key].get("ok") else None
        if x is None or y is None:
            never[sy] = never.get(sy, 0) + 1
            continue
        pairs.setdefault(sy, []).append([key[1], round(x, 1), round(y, 1)])
        spread[sy] = max(spread.get(sy, 0.0), abs(x - y))
    out = {"control_settle_spread_s": spread, "pairs": pairs, "pairs_never_settled": never,
           "from": [argv[0], argv[1]], "definition": "max |settle(A, r) - settle(B, r)| per system, same seeds"}
    (RESULTS / name).mkdir(parents=True, exist_ok=True)
    (RESULTS / name / "aggregate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
