"""Model ladder for a unimolecular gas-phase selectivity: RRHO TST, tunnelling, pressure fall-off.

    python falloff.py DEF.json OUT.json

DEF: {"ladder_dir": dft_ladder workdir, "extra_dirs": [...], "level": electronic level name in "sp" (or "dft" for the
      optimization level), "temperatures": [K...], "T_ref": K, "pressures_atm": [...], "P_ref": atm,
      "reactant": [ladder names of the reactant conformers; the first is the floor],
      "channels": {product: [{"ts": name, "degeneracy": L, "products": [ladder names of the separated products]}]},
      "ratio": [num, den], "well": {"lj_sigma_A", "lj_eps_K"}, "bath": {"mass_amu", "lj_sigma_A", "lj_eps_K"},
      "de_down_cm": [values], "de": grain cm^-1, "e_max_cm": top of the grid}

Everything from the ladder's DFT frequencies and geometries; electronic energies at `level`. Prints and writes:
k_inf (RRHO TST, the same partition functions as the RRKM), with Wigner and Eckart tunnelling; strong-collision and
master-equation (exponential down) rate constants at each pressure; and the selectivity num:den for each.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rrkm  # noqa: E402
from explorer_level import read_frames  # noqa: E402
from ladder_table import load_records  # noqa: E402

HARTREE_CM = 219474.6313705


def zpe_cm(freqs):
    return 0.5 * sum(f for f in freqs if f > 0)


def main(def_path: str, out_path: str):
    D = json.loads(Path(def_path).read_text())
    base = Path(def_path).resolve().parent
    recs = load_records(base / D["ladder_dir"], [base / e for e in D.get("extra_dirs", [])])
    lev = D["level"]

    def energy(name):
        r = recs[name]
        return r["e_opt"] if lev == "dft" else r["sp"][lev]["energy"]

    def geom(name):
        sym, X = read_frames(base / D["ladder_dir"] / f"{name}_opt.xyz")[0]
        return sym, X

    def real(name):
        f = [x for x in recs[name]["freqs_cm"]]
        return [x for x in f if x > 0]

    r0 = D["reactant"][0]
    e_ref = energy(r0) * HARTREE_CM + zpe_cm(real(r0))
    conformers = []
    for name in D["reactant"]:
        sym, X = geom(name)
        off = energy(name) * HARTREE_CM + zpe_cm(real(name)) - e_ref
        conformers.append((real(name), rrkm.moments(sym, X), off, D.get("sigma", {}).get(name, 1), 1))
    sym0, _ = geom(r0)
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    mass = sum(pt.GetMostCommonIsotopeMass(s) for s in sym0)
    well = rrkm.Well(conformers, mass, D["well"]["lj_sigma_A"], D["well"]["lj_eps_K"])
    bath = rrkm.Bath(**D.get("bath", {}))
    channels, meta = [], {}
    for prod, items in D["channels"].items():
        for it in items:
            t = it["ts"]
            sym, X = geom(t)
            f_all = recs[t]["freqs_cm"]
            imag = min(f_all)
            e0 = energy(t) * HARTREE_CM + zpe_cm(real(t)) - e_ref
            v_rev = None
            if it.get("products"):
                # the reaction energy at `product_level` (default: the same level) when the products lack `level`
                plev = D.get("product_level", lev)

                def e_at(name, lv):
                    r = recs[name]
                    return r["e_opt"] if lv == "dft" else r["sp"][lv]["energy"]

                rxn = (sum(e_at(p, plev) * HARTREE_CM + zpe_cm(real(p)) for p in it["products"])
                       - (e_at(r0, plev) * HARTREE_CM + zpe_cm(real(r0))))
                v_rev = e0 - rxn
            ch = rrkm.Channel(f"{prod}|{t}", real(t), rrkm.moments(sym, X), e0, it.get("degeneracy", 1), 1, imag, v_rev)
            channels.append(ch)
            meta[ch.name] = {"product": prod, "e0_kcal": e0 / rrkm.KCAL_TO_CM, "imag_cm": imag,
                             "v_rev_kcal": None if v_rev is None else v_rev / rrkm.KCAL_TO_CM,
                             "degeneracy": it.get("degeneracy", 1)}
    model = rrkm.Model(well, channels, de=D.get("de", 10.0), e_max_cm=D.get("e_max_cm", 60000.0))
    num, den = D["ratio"]

    dens = den if isinstance(den, list) else [den]

    def sel(kdict):
        kn = sum(v for k, v in kdict.items() if k.split("|")[0] == num)
        kd = sum(v for k, v in kdict.items() if k.split("|")[0] in dens)
        return kn / kd

    out = {"level": lev, "channels": meta, "reactant_offsets_kcal": [c[2] / rrkm.KCAL_TO_CM for c in conformers],
           "rows": []}
    for T in D["temperatures"]:
        rt = rrkm.R_KCAL * T if hasattr(rrkm, "R_KCAL") else 1.987204259e-3 * T
        row = {"T": T}
        kinf = {c.name: model.k_tst(c, T) for c in channels}
        kinf_mic = {c.name: model.k_inf(c, T) for c in channels}
        wig = {c.name: kinf[c.name] * rrkm.wigner(c.imag_cm, T) for c in channels}
        eck = {c.name: kinf[c.name] * (rrkm.eckart_kappa(c.e0_cm, c.v_rev_cm, c.imag_cm, T)
                                       if c.v_rev_cm is not None else 1.0) for c in channels}
        row["k_inf"] = kinf
        row["sel_inf_tst"] = sel(kinf)
        row["sel_inf_microcanonical"] = sel(kinf_mic)
        row["sel_inf_wigner"] = sel(wig)
        row["sel_inf_eckart"] = sel(eck)
        row["kappa_wigner"] = {c.name: rrkm.wigner(c.imag_cm, T) for c in channels}
        row["kappa_eckart"] = {c.name: (rrkm.eckart_kappa(c.e0_cm, c.v_rev_cm, c.imag_cm, T)
                                        if c.v_rev_cm is not None else None) for c in channels}
        row["pressures"] = []
        me_here = T == D.get("T_ref", T)          # the master equation (slow) only at the reference temperature
        for P in D["pressures_atm"]:
            pr = {"P_atm": P}
            sc = model.strong_collision(T, P, bath)
            pr["omega"] = sc.pop("omega")
            pr["sel_strong"] = sel(sc)
            pr["k_strong"] = sc
            sct = model.strong_collision(T, P, bath, tunnel=True)
            sct.pop("omega")
            pr["sel_strong_eckart"] = sel(sct)
            for a in (D.get("de_down_cm", []) if me_here else []):
                me = model.master_equation(T, P, bath, a)
                lam = me.pop("lambda")
                me.pop("omega")
                pr[f"sel_me_{a:g}"] = sel(me)
                pr[f"k_me_{a:g}"] = me
                met = model.master_equation(T, P, bath, a, tunnel=True)
                met.pop("lambda")
                met.pop("omega")
                pr[f"sel_me_{a:g}_eckart"] = sel(met)
            row["pressures"].append(pr)
        out["rows"].append(row)
        print(f"T {T:6.0f}  k_inf {num}:{den} TST {row['sel_inf_tst']:7.3g} (micro {row['sel_inf_microcanonical']:7.3g})"
              f"  +Wigner {row['sel_inf_wigner']:7.3g}  +Eckart {row['sel_inf_eckart']:7.3g}  RTln(TST) "
              f"{rt * math.log(row['sel_inf_tst']):5.2f}")
        for pr in row["pressures"]:
            extra = "  ".join(f"ME{a:g} {pr[f'sel_me_{a:g}']:7.3g} (+Eck {pr[f'sel_me_{a:g}_eckart']:7.3g})"
                              for a in D.get("de_down_cm", []) if f"sel_me_{a:g}" in pr)
            print(f"     P {pr['P_atm']:6.2f} atm  strong {pr['sel_strong']:7.3g} (+Eck {pr['sel_strong_eckart']:7.3g})"
                  f"  {extra}   k_strong/k_inf " + " ".join(f"{k.split('|')[0]} {pr['k_strong'][k] / row['k_inf'][k]:.3g}"
                                                         for k in pr["k_strong"]))
    for k, v in meta.items():
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
    Path(out_path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
