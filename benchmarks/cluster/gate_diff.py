"""Two gate.py --json outputs of the same comparison side by side (e.g. before / after a scoring change): per
system the verdict, the accuracy check's medians and tolerance, full recall and the settle medians, and what moved.

    python benchmarks/cluster/gate_diff.py OLD_GATE_JSON NEW_GATE_JSON
"""
import json
import sys


def row(g: dict) -> dict:
    c = g.get("checks") or {}
    acc, rec, st = c.get("accuracy") or {}, c.get("recall") or {}, c.get("settle") or {}
    return {"pass": g.get("pass"), "acc": (acc.get("base_median"), acc.get("cand_median"), acc.get("tolerance")),
            "recall": (rec.get("base_full"), rec.get("cand_full")),
            "settle": (st.get("base_median_s"), st.get("cand_median_s"), st.get("tolerance_s")), "why": g.get("why")}


def _f(x):
    return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def main(argv: list) -> int:
    old, new = (json.load(open(p)) for p in argv[:2])
    print(f"base {new['base']['run']}:{new['base']['config']}  candidate {new['candidate']['run']}:{new['candidate']['config']}")
    moved = []
    for sy in sorted(set(old["systems"]) | set(new["systems"])):
        a, b = (row(x["systems"][sy]) if sy in x["systems"] else None for x in (old, new))
        line = f"{sy:16s} " + " | ".join(
            f"{name}: pass {_f(r['pass'])}, err {'/'.join(_f(v) for v in r['acc'])}, recall "
            f"{'/'.join(_f(v) for v in r['recall'])}, settle {'/'.join(_f(v) for v in r['settle'])}"
            if r else f"{name}: -" for name, r in (("old", a), ("new", b)))
        print(line)
        r2 = lambda r: tuple(None if v is None else round(v, 2) for v in r["acc"])   # moved: by > 0.005 kcal/mol
        if a is None or b is None or a["pass"] != b["pass"] or r2(a) != r2(b):
            moved.append({"system": sy, "old": a, "new": b})
    print(f"overall: old {'PASS' if old.get('pass') else 'FAIL'}, new {'PASS' if new.get('pass') else 'FAIL'}")
    print("moved (verdict or accuracy numbers): " + (", ".join(m["system"] for m in moved) or "none"))
    for m in moved:
        print(" ", m["system"], "why old:", (m["old"] or {}).get("why"), "new:", (m["new"] or {}).get("why"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
