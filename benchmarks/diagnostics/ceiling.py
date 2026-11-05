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


def _prefill(level, net, workers: int) -> None:
    """Compute every structure's thermal correction and steering single point in `workers` processes first (the
    cache then serves evaluate()); one process writes the cache."""
    from multiprocessing import get_context

    from explorer_level import geo_key

    keep = spec_steer = getattr(net, "steer_species", None)
    items = [(s.st, False, None) for k, s in enumerate(net.species) if keep is None or k in set(keep) | {0}] \
        + [(t.st, True, t.ends) for t in net.steps]

    def todo_keys(st, ts, ends):
        groups = level.free._groups(st, ts, ends)
        return ("corr:" + level.free._key(st, ts, ends) + f":{level.T:g}:{level.ss}",
                "sp:" + geo_key(st, (ts,) + ((groups,) if groups and len(groups) > 1 else ())))

    jobs = [(i, it) for i, it in enumerate(items) if any(level.cache.get(k) is None for k in todo_keys(*it))]
    if not jobs:
        return
    global _LEVEL
    _LEVEL = level
    with get_context("fork").Pool(workers) as pool:
        for i, res in pool.imap_unordered(_one_item, jobs):
            for k, v in res.items():
                level.cache.put(k, v)


_LEVEL = None


def _one_item(job):
    i, (st, ts, ends) = job
    level = _LEVEL
    from mepd.discovery.energy_refine import Item, _fragments, _single_point

    from explorer_level import geo_key

    out = {}
    groups = level.free._groups(st, ts, ends)
    ck = "corr:" + level.free._key(st, ts, ends) + f":{level.T:g}:{level.ss}"
    if level.cache.get(ck) is None:
        v = level.free._correction(st, ts, ends)
        out[ck] = {"value": list(v) if isinstance(v, tuple) else float(v)}
    sk = "sp:" + geo_key(st, (ts,) + ((groups,) if groups and len(groups) > 1 else ()))
    if level.steer is not None and level.cache.get(sk) is None:
        parts = _fragments(st, level.parts(st, ts, ends))
        out[sk] = _single_point(Item("ts" if ts else "species", 0, st, parts), level.engine, level.steer)
    ek = "e:" + geo_key(st)
    if level.cache.get(ek) is None:
        from mepd.discovery.energy_refine import _energy

        out[ek] = {"value": _energy(st, level.engine)}
    return i, out


def main(spec_path: str, out_path: str, prefill: int = 1):
    spec = json.loads(Path(spec_path).read_text())
    base = Path(spec_path).resolve().parent
    level = ExplorerLevel(spec["profile"] if Path(spec["profile"]).is_absolute() else str(base / spec["profile"]),
                          (spec["steer"] if not spec.get("steer") or Path(spec["steer"]).is_absolute()
                           else str(base / spec["steer"])),
                          spec["temperature"], spec["standard_state"], str(base / spec["cache"]),
                          spec.get("workers", 1))
    net = build(spec, base)
    net.steer_species = spec.get("steer_species")
    if prefill > 1:
        _prefill(level, net, prefill)
    num, den = spec["ratio"]
    out = {"spec": spec, "variants": {}}
    for steer in ([True, False] if level.steer is not None else [False]):
        for mirrors in (True, False):
            rec = evaluate(level, net, spec["time_s"], steer=steer, mirrors=mirrors,
                           steer_species=spec.get("steer_species"))
            dens = den if isinstance(den, list) else [den]      # a list: the sum of those products ("the rest")
            f = rec["final"]
            fd = sum(f.get(d, 0.0) for d in dens)
            r_final = f.get(num, 0.0) / fd if fd > 0 else float("inf")
            ks = rec["rate_sums"]
            kd = sum(ks.get(d, 0.0) for d in dens)
            r_rate = ks.get(num, 0.0) / kd if kd > 0 else float("inf")
            rec["selectivity_final"] = r_final
            rec["selectivity_rates"] = r_rate
            rec["rtln_final_kcal"] = kcal_ratio(r_final, level.T) if 0 < r_final < float("inf") else None
            name = ("steered" if steer else "gxtb") + ("+mirrors" if mirrors else "")
            out["variants"][name] = rec
            print(f"{name:<18} {num}:{'+'.join(dens)} = {r_final:9.4g} (final), {r_rate:9.4g} (rate sums); "
                  + "; ".join(f"{s['label']} dG {s['dG_kcal']:.2f} dE {s['dE_kcal']:.2f}" for s in rec["steps"]))
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 1
    if "--prefill" in args:
        i = args.index("--prefill")
        n = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    main(args[0], args[1], n)
