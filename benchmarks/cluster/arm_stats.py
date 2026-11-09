"""Per system of one arm: median settle (as scored, and on the partial basis), the share settled within 30 min,
wall time of the runs that converged, wall time of the runs that stopped at max_species (time to the cap), median
CPU hours; with --occ (team28/scripts/occupancy.py's output for these runs) the seed propose job's duration and
post-MD phase (median and range). With --seed-species OTHER_RESULT:CONFIG: per replicate, whether the seed's
products (expand.log: the "species k SMILES dE" lines right after the first "reactor: ... distinct product(s)" line, the seed's) are the same
SMILES with the same dE (0.1 kcal/mol) in both arms.

    python benchmarks/cluster/arm_stats.py RESULT:CONFIG [--occ OCC.json] [--seed-species RESULT:CONFIG]
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recall_noise import RESULTS, runs_of  # noqa: E402


def seed_products(o: dict) -> list | None:
    log = RESULTS / o["run"] / o["id"] / "expand.log"
    if not log.is_file():
        return None
    lines = log.read_text(errors="replace").splitlines()
    last = next((i for i, x in enumerate(lines) if x.startswith("reactor:") and "distinct product(s) from" in x), None)
    if last is None:
        return None
    out = []
    for x in lines[last + 1:]:
        m = re.match(r"^\S*\s*species (\d+) (\S+)\s+ΔE=([-+]?\d+\.\d)", x.strip())
        if not m:
            break
        out.append((m.group(2), m.group(3)))
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


def main(argv: list) -> int:
    spec = argv[0]
    occ = json.loads(Path(argv[argv.index("--occ") + 1]).read_text()) if "--occ" in argv else []
    occ_by = {Path(r["run"]).name: r for r in occ}
    other = runs_of(*argv[argv.index("--seed-species") + 1].split(":", 1)) if "--seed-species" in argv else None
    runs = runs_of(*spec.split(":", 1))
    out = {}
    for sy in sorted({s for s, _ in runs}):
        rs = [o for (s, _), o in sorted(runs.items()) if s == sy]
        settle = [(o.get("settle") or {}).get("settle_s") for o in rs]
        part = [(o.get("settle") or {}).get("settle_s_partial_basis") for o in rs]
        conv = [o.get("wall_s") for o in rs if str(o.get("stopped")).startswith("converged")]
        cap = [o.get("wall_s") for o in rs if "max species" in str(o.get("stopped"))]
        seed = [(occ_by.get(o["id"]) or {}).get("seed_propose") or {} for o in rs]
        g = {"n": len(rs), "settle_median_s": med(settle), "settle_partial_median_s": med(part),
             "settled_30min": sum(x is not None and x <= 1800 for x in settle) / len(rs),
             "converged_wall_s": sorted(round(x) for x in conv), "capped_wall_s": sorted(round(x) for x in cap),
             "cpu_h_median": med([o.get("cpu_h") for o in rs])}
        if occ:
            g["seed_propose_s"] = {"median": med([x.get("dur") for x in seed]), "values": [x.get("dur") for x in seed]}
            g["seed_post_md_s"] = {"median": med([x.get("post_md_s") for x in seed]),
                                   "values": [x.get("post_md_s") for x in seed]}
        if other is not None:
            same = []
            for o in rs:
                p = other.get((sy, o["replicate"]))
                a, b = seed_products(o), seed_products(p) if p else None
                same.append(None if a is None or b is None else a == b)
            g["seed_products_identical"] = same
        out[sy] = g
    print(json.dumps({"arm": spec, "systems": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
