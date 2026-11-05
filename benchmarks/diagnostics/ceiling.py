"""Ceiling of a benchmark at the explorer's own levels, from a network spec (JSON).

    python ceiling.py SPEC.json OUT.json

SPEC: {"profile": g-xTB profile, "steer": ORCA steering profile (or null), "temperature": K, "time_s": s,
       "standard_state": "1atm"|"1M", "cache": per-structure cache (JSON),
       "species": [{"label", "xyz", "frame" (default 0), "product" (benchmark label, "" for the seed)}],   # [0] = seed
       "steps": [{"label", "xyz", "a", "b", "irc" (optional: first/last frame = ends), "channel"}],
       "ratio": [numerator product, denominator product]}

Writes, for four variants (explorer-level with steering, without steering; each with and without mirror images),
the explorer's kinetics (explorer_level.evaluate) and the selectivity num:den from the final concentrations and from
the summed rate constants, and RT ln of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from explorer_level import TS, ExplorerLevel, Network, Species, evaluate, kcal_ratio, load  # noqa: E402


def build(spec: dict, base: Path) -> Network:
    def p(x):
        return x if Path(x).is_absolute() else str(base / x)

    species = [Species(s["label"], load(p(s["xyz"]), s.get("frame", 0)), s.get("product", "")) for s in spec["species"]]
    steps = []
    for t in spec["steps"]:
        ends = None
        if t.get("irc"):
            ends = (load(p(t["irc"]), 0), load(p(t["irc"]), -1))
        steps.append(TS(t["label"], load(p(t["xyz"])), int(t["a"]), int(t["b"]), ends, t.get("channel", "")))
    return Network(species, steps)


def main(spec_path: str, out_path: str):
    spec = json.loads(Path(spec_path).read_text())
    base = Path(spec_path).resolve().parent
    level = ExplorerLevel(spec["profile"] if Path(spec["profile"]).is_absolute() else str(base / spec["profile"]),
                          (spec["steer"] if not spec.get("steer") or Path(spec["steer"]).is_absolute()
                           else str(base / spec["steer"])),
                          spec["temperature"], spec["standard_state"], str(base / spec["cache"]),
                          spec.get("workers", 1))
    net = build(spec, base)
    num, den = spec["ratio"]
    out = {"spec": spec, "variants": {}}
    for steer in ([True, False] if level.steer is not None else [False]):
        for mirrors in (True, False):
            rec = evaluate(level, net, spec["time_s"], steer=steer, mirrors=mirrors)
            f = rec["final"]
            r_final = f.get(num, 0.0) / f[den] if f.get(den, 0.0) > 0 else float("inf")
            ks = rec["rate_sums"]
            r_rate = ks.get(num, 0.0) / ks[den] if ks.get(den, 0.0) > 0 else float("inf")
            rec["selectivity_final"] = r_final
            rec["selectivity_rates"] = r_rate
            rec["rtln_final_kcal"] = kcal_ratio(r_final, level.T) if 0 < r_final < float("inf") else None
            name = ("steered" if steer else "gxtb") + ("+mirrors" if mirrors else "")
            out["variants"][name] = rec
            print(f"{name:<18} {num}:{den} = {r_final:9.4g} (final), {r_rate:9.4g} (rate sums); "
                  + "; ".join(f"{s['label']} dG {s['dG_kcal']:.2f} dE {s['dE_kcal']:.2f}" for s in rec["steps"]))
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
