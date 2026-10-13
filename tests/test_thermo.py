import math
import shutil

import numpy as np
import pytest

from mepd import thermo

# Ethylene oxide (C2v, rotational number 2), Angstrom.
EO = (["C", "C", "O", "H", "H", "H", "H"],
      np.array([[-0.7333, -0.3896, 0.0], [0.7333, -0.3896, 0.0], [0.0, 0.7983, 0.0],
                [-1.2588, -0.6274, 0.9195], [-1.2588, -0.6274, -0.9195],
                [1.2588, -0.6274, 0.9195], [1.2588, -0.6274, -0.9195]]))
EO_FREQS = [812.25, 963.36, 1004.06, 1021.96, 1082.84, 1095.06, 1120.58, 1132.42, 1324.84, 1423.08,
            1486.06, 3067.0, 3074.0, 3150.0, 3163.0]


def test_one_molar_standard_state_shifts_g_by_rt_ln_24():
    sym, xyz = EO
    atm = thermo.qrrho(EO_FREQS, sym, xyz, 298.15, rotational_number=2, standard_state="1atm")
    one_m = thermo.qrrho(EO_FREQS, sym, xyz, 298.15, rotational_number=2)
    rt = 8.314462618 * 298.15 / 4184
    assert one_m["g_corr"] - atm["g_corr"] == pytest.approx(rt * math.log(24.465), abs=0.005)
    assert atm["zpe"] == pytest.approx(sum(EO_FREQS) / 2 * 2.859144e-3, rel=1e-4)   # cm^-1 -> kcal/mol


def test_a_ts_leaves_out_its_imaginary_mode_and_soft_modes_are_rotors():
    sym, xyz = EO
    base = thermo.qrrho(EO_FREQS, sym, xyz, 298.15, rotational_number=2)
    ts = thermo.qrrho([-900.0] + EO_FREQS, sym, xyz, 298.15, rotational_number=2, ts=True)
    assert ts["g_corr"] == pytest.approx(base["g_corr"]) and ts["n_imaginary"] == 1
    # A 10 cm^-1 mode: harmonic entropy would be huge; quasi-RRHO keeps it near a free rotor's.
    soft = thermo.qrrho(EO_FREQS + [10.0], sym, xyz, 298.15, rotational_number=2)
    assert 0 < (soft["s"] - base["s"]) < 6.0           # cal/mol/K; harmonic would give ~ 8


def test_xtb_free_energy_is_reproduced():
    if not (shutil.which("xtb") or shutil.which("gxtb")):
        pytest.skip("no xtb")
    sym, xyz = EO
    fr = thermo.frequencies(sym, xyz)
    ours = thermo.qrrho(fr["freqs"], sym, xyz, 298.15, rotational_number=fr["rotational_number"],
                        linear=fr["linear"], standard_state="1atm")
    assert fr["rotational_number"] == 2
    assert ours["g_corr"] == pytest.approx(fr["xtb_g298_eh"] * thermo.HARTREE_KCAL, abs=0.1)
