"""One TS found by several runs: why do its thermal corrections differ?

    python benchmarks/diagnostics/thermo/ts_terms.py --profile PROFILE.toml --temperature 1500 \
        --standard-state 1atm --repeats 5 --out DIR  NAME=TS.xyz ...

For each TS file (a run's own; its step's ends are read from <stem>_irc.xyz beside it, as the explorer does):
  - the explorer's own correction with the Hessian at the geometry in the file (FreeEnergies._correction with
    FreeEnergies._saddle switched off: the code before the TS was converged first), `--repeats` times, each with
    a new engine, and the same after a rigid rotation and an atom reorder of the same geometry;
  - the correction of the code as it is (the TS converged to HESSIAN_FMAX first), its terms there, and how far
    the geometry moved;
  - how it is perceived: the molecules it counts as, xtb's rotational number and linearity, chirality;
  - its gradient norm and frequencies at the explorer's level, and the correction term by term:
    ZPE, vibrational energy, vibrational entropy per mode (quasi-RRHO and plain harmonic), rotation, translation.
And between the files: the best-fit RMSD over atom orders and mirror images, the sorted-distance difference, and
the reacting distances. Nothing is optimized, built or changed; a diagnostic only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

BOHR = 1.8897259886


def load(fp):
    from qcdata import Structure

    return Structure.open(str(fp))


def kabsch_rmsd(p, q, allow_mirror=False):
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    if d < 0 and not allow_mirror:
        s[-1] = -s[-1]
    return math.sqrt(max(0.0, (p ** 2).sum() + (q ** 2).sum() - 2 * s.sum()) / len(p))


def best_rmsd(sym_a, xa, sym_b, xb):
    """Lowest RMSD (Angstrom) over the atom orders that keep elements, proper and with a mirror image."""
    els = sorted(set(sym_a))
    idx_a = {e: [i for i, s in enumerate(sym_a) if s == e] for e in els}
    idx_b = {e: [i for i, s in enumerate(sym_b) if s == e] for e in els}
    best = {"proper": (9e9, None), "mirror": (9e9, None)}
    for perms in itertools.product(*[itertools.permutations(idx_b[e]) for e in els]):
        order = [0] * len(sym_a)
        for e, pb in zip(els, perms):
            for i, j in zip(idx_a[e], pb):
                order[i] = j
        for name, mirror in (("proper", False), ("mirror", True)):
            r = kabsch_rmsd(xa, xb[order] * (np.array([1, 1, -1]) if mirror else 1.0))
            if r < best[name][0]:
                best[name] = (r, order)
    return best


def terms(freqs, symbols, xyz_A, T, rot, linear, standard_state, rotor_cutoff=50.0):
    """mepd.thermo.qrrho, term by term (kcal/mol), with the per-mode vibrational parts."""
    from scipy.constants import Avogadro, Boltzmann, Planck, c, gas_constant as R

    from mepd.thermo import B_AV, qrrho

    kT = Boltzmann * T
    f = sorted(float(x) for x in freqs)
    dropped = None
    if f and f[0] < 0:
        dropped, f = f[0], f[1:]
    per_mode = []
    for nu0 in f:
        nu = abs(nu0)
        hv = Planck * c * nu * 100.0
        x = hv / kT
        zpe = 0.5 * hv * Avogadro / 4184.0
        e = hv * Avogadro / math.expm1(x) / 4184.0
        s_ho = R * (x / math.expm1(x) - math.log(-math.expm1(-x)))
        mu = Planck / (8 * math.pi ** 2 * c * nu * 100.0)
        mu_eff = mu * B_AV / (mu + B_AV)
        s_fr = R * (0.5 + math.log(math.sqrt(8 * math.pi ** 3 * mu_eff * kT / Planck ** 2)))
        w = 1.0 / (1.0 + (rotor_cutoff / nu) ** 4)
        per_mode.append({"cm": nu0, "zpe": zpe, "e_vib": e, "w_ho": w, "minus_TS_ho": -T * s_ho / 4184.0,
                         "minus_TS_free_rotor": -T * s_fr / 4184.0,
                         "minus_TS_qrrho": -T * (w * s_ho + (1 - w) * s_fr) / 4184.0,
                         "g_qrrho": zpe + e - T * (w * s_ho + (1 - w) * s_fr) / 4184.0,
                         "g_ho": zpe + e - T * s_ho / 4184.0})
    full = qrrho(freqs, symbols, xyz_A, T, rotational_number=rot, linear=linear, ts=True, standard_state=standard_state)
    novib = qrrho([], symbols, xyz_A, T, rotational_number=rot, linear=linear, standard_state=standard_state)
    return {"ts_mode_dropped_cm": dropped, "g_corr": full["g_corr"], "n_imaginary": full["n_imaginary"],
            "zpe": sum(m["zpe"] for m in per_mode), "e_vib": sum(m["e_vib"] for m in per_mode),
            "minus_TS_vib_qrrho": sum(m["minus_TS_qrrho"] for m in per_mode),
            "minus_TS_vib_ho": sum(m["minus_TS_ho"] for m in per_mode),
            "g_rot_trans_el": novib["g_corr"], "h_rot_trans": novib["h"], "s_rot_trans_cal": novib["s"],
            "g_corr_harmonic": novib["g_corr"] + sum(m["g_ho"] for m in per_mode), "modes": per_mode}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ts", nargs="+", help="NAME=TS.xyz")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--temperature", type=float, default=1500.0)
    ap.add_argument("--standard-state", default="1atm")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import mepd
    from qcdata import Structure

    from mepd.discovery.free_energy import FreeEnergies, step_ends
    from mepd.discovery.mirror import is_chiral
    from mepd.nodes.node import StructureNode
    from mepd.thermo import frequencies as xtb_analysis

    saddle = FreeEnergies._saddle
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res, geo = {}, {}
    for item in a.ts:
        name, fp = item.split("=", 1)
        st = load(fp)
        sym = [str(s) for s in st.symbols]
        xyz = np.asarray(st.geometry, dtype=float).reshape(-1, 3) / BOHR
        geo[name] = (sym, xyz)
        ends = step_ends({"ts": fp, "irc": str(Path(fp).with_name(Path(fp).stem + "_irc.xyz"))}, ())
        r = {"file": Path(fp).name, "ends_from": "irc" if ends else None}
        FreeEnergies._saddle = saddle
        free = FreeEnergies(mepd.engine_from_profile(a.profile), a.temperature, a.standard_state)
        r["correction_converged_first"] = free._correction(st, True, ends if ends else None)
        tight, moved = free._saddle(StructureNode(structure=st))
        xt = np.asarray(tight.structure.geometry, dtype=float).reshape(-1, 3) / BOHR
        wt = sorted(mepd.frequencies(tight, free.engine).wavenumbers_cm.tolist())
        gt = np.asarray(free.engine.compute_gradients([tight])[0], dtype=float).reshape(-1, 3)
        r["saddle"] = {"energy_change_kcal": moved, "rmsd_from_given_A": kabsch_rmsd(xt, xyz),
                       "fmax_eV_A": float(np.linalg.norm(gt, axis=1).max() * 27.211386 * BOHR),
                       "frequencies_cm": wt, "xyz_A": xt.tolist()}
        FreeEnergies._saddle = lambda self, node: (node, 0.0)
        corr = []
        for _ in range(a.repeats):
            free = FreeEnergies(mepd.engine_from_profile(a.profile), a.temperature, a.standard_state)
            corr.append(free._correction(st, True, ends if ends else None))
        r["correction_repeats"] = corr
        free = FreeEnergies(mepd.engine_from_profile(a.profile), a.temperature, a.standard_state)
        r["parts"] = [list(p) for p in free.parts(st, True, ends if ends else None)]
        r["notes"] = free.notes
        # the same geometry rotated, and with its atoms in another order (no ends: one molecule, as perceived above)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        rot_st = Structure(symbols=sym, geometry=(xyz @ q.T) * BOHR, charge=st.charge, multiplicity=st.multiplicity)
        perm = rng.permutation(len(sym))
        perm_st = Structure(symbols=[sym[k] for k in perm], geometry=xyz[perm] * BOHR, charge=st.charge,
                            multiplicity=st.multiplicity)
        r["correction_no_ends"] = free._correction(st, True, None)
        r["correction_rotated"] = free._correction(rot_st, True, None)
        r["correction_reordered"] = free._correction(perm_st, True, None)
        engine = mepd.engine_from_profile(a.profile)
        node = StructureNode(structure=st)
        r["energy"] = float(mepd.energies([node], engine)[0])
        g = np.asarray(engine.compute_gradients([node])[0], dtype=float)
        r["gradient_max_eV_A"] = float(np.abs(g).max() * 27.211386 * BOHR)
        r["gradient_rms_eV_A"] = float(np.sqrt((g ** 2).mean()) * 27.211386 * BOHR)
        w = sorted(mepd.frequencies(node, engine).wavenumbers_cm.tolist())
        r["frequencies_cm"] = w
        shape = xtb_analysis(sym, xyz, charge=int(st.charge), multiplicity=int(st.multiplicity))
        r["rotational_number"], r["linear"] = shape["rotational_number"], shape["linear"]
        try:
            ff = xtb_analysis(sym, xyz, charge=int(st.charge), multiplicity=int(st.multiplicity),
                              method_args=["--gfnff"])
            r["rotational_number_gfnff"] = ff["rotational_number"]
        except Exception as exc:
            r["rotational_number_gfnff"] = f"failed: {type(exc).__name__}"
        r["chiral"] = bool(is_chiral(st))
        r["terms"] = terms(w, sym, xyz, a.temperature, shape["rotational_number"], shape["linear"], a.standard_state)
        r["saddle"]["terms"] = terms(wt, sym, xt, a.temperature, shape["rotational_number"], shape["linear"],
                                     a.standard_state)
        c = [k for k, s in enumerate(sym) if s == "C"]
        o = [k for k, s in enumerate(sym) if s == "O"]
        h = [k for k, s in enumerate(sym) if s == "H"]
        d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        r["distances_A"] = {"sorted_all": sorted(d[np.triu_indices(len(sym), 1)].tolist()),
                            "C-C": sorted(d[i, j] for i in c for j in c if i < j),
                            "C-O": sorted(d[i, j] for i in c for j in o),
                            "O-H": sorted(d[i, j] for i in o for j in h)[:4],
                            "C-H": sorted(d[i, j] for i in c for j in h)[:6]}
        res[name] = r
        geo[name + "@saddle"] = (sym, xt)
    pairs = {}
    names = list(res)
    for i, j in itertools.combinations(names, 2):
        b = best_rmsd(*geo[i], *geo[j])
        da = np.array(res[i]["distances_A"]["sorted_all"])
        db = np.array(res[j]["distances_A"]["sorted_all"])
        pairs[f"{i}|{j}"] = {"rmsd_proper_A": b["proper"][0], "rmsd_mirror_A": b["mirror"][0],
                             "max_sorted_distance_diff_A": float(np.abs(da - db).max()),
                             "dE_Eh": res[i]["energy"] - res[j]["energy"],
                             "saddles_rmsd_proper_A": best_rmsd(*geo[i + "@saddle"], *geo[j + "@saddle"])["proper"][0],
                             "saddles_rmsd_mirror_A": best_rmsd(*geo[i + "@saddle"], *geo[j + "@saddle"])["mirror"][0]}
    (out / "ts_terms.json").write_text(json.dumps({"temperature": a.temperature, "standard_state": a.standard_state,
                                                   "ts": res, "pairs": pairs}, indent=1))
    for n in names:
        r, t = res[n], res[n]["terms"]
        print(f"== {n} ({r['file']})  E={r['energy']:.7f}  |g|max={r['gradient_max_eV_A']:.5f} eV/A  "
              f"sigma={r['rotational_number']} (gfnff {r['rotational_number_gfnff']}) linear={r['linear']} "
              f"chiral={r['chiral']} parts={[len(p[0]) for p in r['parts']]}")
        print("   correction repeats:", " ".join(f"{v:.4f}" for v in r["correction_repeats"]),
              f"| no ends {r['correction_no_ends']:.4f} rotated {r['correction_rotated']:.4f} "
              f"reordered {r['correction_reordered']:.4f}")
        print("   freqs:", " ".join(f"{v:.1f}" for v in r["frequencies_cm"]))
        print(f"   zpe {t['zpe']:.3f}  e_vib {t['e_vib']:.3f}  -TS_vib {t['minus_TS_vib_qrrho']:.3f} "
              f"(harmonic {t['minus_TS_vib_ho']:.3f})  rot+trans G {t['g_rot_trans_el']:.3f}  "
              f"g_corr {t['g_corr']:.3f} (harmonic {t['g_corr_harmonic']:.3f})")
        print("   per mode g (qRRHO):", " ".join(f"{m['cm']:.0f}:{m['g_qrrho']:.2f}" for m in t["modes"]))
        sd = r["saddle"]
        print(f"   converged first: correction {r['correction_converged_first']:.4f}  (moved {sd['rmsd_from_given_A']:.4f} A, "
              f"{sd['energy_change_kcal']:.5f} kcal/mol, fmax {sd['fmax_eV_A']:.1e}; lowest "
              + " ".join(f"{v:.1f}" for v in sd["frequencies_cm"][:3]) + f"; qRRHO {sd['terms']['g_corr']:.3f}, "
              f"harmonic {sd['terms']['g_corr_harmonic']:.3f})")
        print("   distances:", {k: [round(x, 3) for x in v] for k, v in r["distances_A"].items() if k != "sorted_all"})
    for k, v in pairs.items():
        print(k, {kk: round(vv, 6) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
