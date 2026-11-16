"""Size of the standard low-frequency treatments on the groups whose members are converged TSs (no treatment chosen).

    python benchmarks/diagnostics/thermo/soft_options.py VERIFIED.json OUT_DIR

From verified.py's output: the groups all of whose members had the saddle step applied. Per member, the
vibrational free energy (kcal/mol, the run's temperature) of its five lowest real modes where the saddle step
ended (the other modes agree between members), under: harmonic; quasi-RRHO with cutoff 50 cm^-1 (the explorer's)
and 100 cm^-1; modes below 100 cm^-1 raised to 100. Per group and treatment: the spread between members.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treatments import g_vib  # noqa: E402


def main(src, out_dir):
    d = json.loads(Path(src).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hows = ("harmonic", "qRRHO 50", "qRRHO 100", "raise to 100")
    lines = ["| group | recorded spread | " + " | ".join(hows) + " | lowest three real modes of each member, cm-1 |",
             "|" + "---|" * (len(hows) + 3)]
    res = []
    for g in sorted(d["groups"], key=lambda g: -g["spread_kcal"]):
        ms = g["members"]
        if any("saddle" not in m for m in ms):
            continue
        row = {"group": f"{g['system']} {g['what']} {g['energy']:.6f}", "recorded": g["spread_kcal"],
               "temperature": g["temperature"]}
        for how in hows:
            v = [g_vib(m["saddle"]["lowest6"], g["temperature"], how) for m in ms]   # g_vib drops the first (TS) mode
            row[how] = max(v) - min(v)
        row["modes"] = [[round(x, 1) for x in m["saddle"]["lowest6"][1:4]] for m in ms]
        res.append(row)
        lines.append(f"| {row['group']} | {row['recorded']:.3f} | " + " | ".join(f"{row[h]:.3f}" for h in hows) +
                     " | " + "; ".join(" ".join(map(str, x)) for x in row["modes"]) + " |")
    (out_dir / "soft_options.json").write_text(json.dumps(res, indent=1))
    (out_dir / "soft_options.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
