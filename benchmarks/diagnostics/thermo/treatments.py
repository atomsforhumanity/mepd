"""How much standard low-frequency treatments differ between the copies of one TS (no treatment is chosen here).

    python benchmarks/diagnostics/thermo/treatments.py DIR/ts_terms.json OUT_DIR

From the frequencies ts_terms.py recorded for each copy of a TS, at the geometry given and at the converged
saddle, the vibrational free energy (ZPE + thermal energy - T S, kcal/mol) under:
  harmonic       every mode a harmonic oscillator;
  qRRHO 50       Grimme's interpolation to a free rotor's entropy, cutoff 50 cm^-1 (the explorer's, xtb's default);
  qRRHO 100      the same, cutoff 100 cm^-1 (the value of the 2012 paper);
  raise to 100   modes below 100 cm^-1 counted as 100 cm^-1 (Ribeiro, Marenich, Cramer, Truhlar 2011).
The TS mode (most negative) is left out; another imaginary mode counts with its absolute value, as the explorer
does. Prints each treatment's value per copy and its spread, given and converged. Computes no structure.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from scipy.constants import Avogadro, Boltzmann, Planck, c, gas_constant as R

B_AV = 1e-44


def g_vib(freqs, T, how):
    f = sorted(float(x) for x in freqs)[1:]
    kT, total = Boltzmann * T, 0.0
    for nu in (abs(x) for x in f):
        if how == "raise to 100":
            nu = max(nu, 100.0)
        hv = Planck * c * nu * 100.0
        x = hv / kT
        s = R * (x / math.expm1(x) - math.log(-math.expm1(-x)))
        if how.startswith("qRRHO"):
            mu = Planck / (8 * math.pi ** 2 * c * nu * 100.0)
            mu = mu * B_AV / (mu + B_AV)
            s_fr = R * (0.5 + math.log(math.sqrt(8 * math.pi ** 3 * mu * kT / Planck ** 2)))
            w = 1.0 / (1.0 + (float(how.split()[1]) / nu) ** 4)
            s = w * s + (1 - w) * s_fr
        total += (0.5 * hv * Avogadro + hv * Avogadro / math.expm1(x) - T * s) / 4184.0
    return total


def main(src, out_dir):
    d = json.loads(Path(src).read_text())
    T = d["temperature"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(d["ts"])
    rows = {}
    lines = [f"Vibrational free energy at {T:g} K (kcal/mol) of {', '.join(names)}.", "",
             "| treatment | geometry | " + " | ".join(names) + " | spread |", "|---|---|" + "---|" * (len(names) + 1)]
    for how in ("harmonic", "qRRHO 50", "qRRHO 100", "raise to 100"):
        for where in ("given", "converged"):
            v = [g_vib(d["ts"][n]["frequencies_cm"] if where == "given" else d["ts"][n]["saddle"]["frequencies_cm"],
                       T, how) for n in names]
            rows[f"{how} | {where}"] = v
            lines.append(f"| {how} | {where} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {max(v) - min(v):.3f} |")
    lines += ["", "Lowest real mode (cm^-1), given: " + ", ".join(f"{sorted(d['ts'][n]['frequencies_cm'])[1]:.1f}"
                                                                    for n in names) + "; converged: " +
              ", ".join(f"{sorted(d['ts'][n]['saddle']['frequencies_cm'])[1]:.1f}" for n in names)]
    (out_dir / "treatments.json").write_text(json.dumps({"temperature": T, "copies": names, "g_vib": rows}, indent=1))
    (out_dir / "treatments.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
