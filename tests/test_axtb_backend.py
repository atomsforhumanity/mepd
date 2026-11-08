"""GXTBCalculator(backend="axtb"): g-xTB 2.0.1 from axtb in this process, checked against the g-xTB
program where one is installed."""
from __future__ import annotations

import multiprocessing
import os
import shutil

import numpy as np
import pytest

axtb = pytest.importorskip("axtb")

from qcdata import Structure  # noqa: E402

from mepd.engines.gxtb import GXTBCalculator  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402

# A distorted formamide (not a stationary point) and the HCN <-> HNC saddle region.
FORMAMIDE = """6

C   0.000   0.420   0.000
O   1.180   0.650   0.050
N  -0.920   1.380  -0.020
H  -0.330  -0.620   0.040
H  -1.900   1.150   0.030
H  -0.640   2.360   0.070
"""


# Ethane: big enough for faer to multiply matrices in parallel (axtb's vibrational analysis then
# starts rayon's global pool; methane and formamide stay sequential).
ETHANE = """8

C  -0.748   0.042   0.011
C   0.751   0.001   0.001
H  -1.130  -0.636   0.814
H  -1.189   1.025   0.200
H  -1.200  -0.333  -0.938
H   1.084  -0.737  -0.773
H   1.227   0.958  -0.265
H   1.204  -0.320   0.950
"""


def _node(xyz: str = FORMAMIDE) -> StructureNode:
    return StructureNode(structure=Structure.from_xyz(xyz))


def _program() -> GXTBCalculator | None:
    exe = os.getenv("GXTB_EXECUTABLE") or shutil.which("gxtb-xtb")
    return GXTBCalculator(executable=exe, n_parallel=1) if exe else None


needs_program = pytest.mark.skipif(_program() is None, reason="no g-xTB program (gxtb-xtb or $GXTB_EXECUTABLE)")


@needs_program
@pytest.mark.parametrize("conventions, tol_e, tol_g", [("gxtb", 1e-9, 1e-7), (None, 1e-8, 1e-4)])
def test_energy_and_gradient_match_the_program(conventions, tol_e, tol_g):
    # "gxtb" reproduces the program's SCF step by step; the default conventions converge to the same
    # state along another path, so their gradients differ by the SCF's own convergence error.
    ref = _program()
    ref.extra_args = ["--acc", "0.01"]      # the axtb backend's default SCF accuracy (axtb_backend.ACCURACY)
    ax = GXTBCalculator(backend="axtb", axtb_conventions=conventions, n_parallel=1)
    a, b = _node(), _node()
    e_ref, g_ref = ref.compute_energies([a])[0], a.gradient
    e_ax, g_ax = ax.compute_energies([b])[0], b.gradient
    assert abs(e_ax - e_ref) < tol_e
    assert np.abs(np.asarray(g_ax) - np.asarray(g_ref)).max() < tol_g


@needs_program
def test_frequencies_match_the_program():
    # At a minimum: away from one, the program's projection (geometric center, before mass
    # weighting) and the mass-weighted one here differ by up to ~20 cm^-1 in soft modes.
    ax_eng = GXTBCalculator(backend="axtb", n_parallel=1)
    minimum = ax_eng.compute_geometry_optimization(_node(), {"fmax": 1e-4})[-1]
    ref = _program()._compute_hessian_result(minimum.copy())
    ax = ax_eng._compute_hessian_result(minimum.copy())
    f_ref, f_ax = np.sort(ref.results.freqs_wavenumber), np.sort(ax.results.freqs_wavenumber)
    assert len(f_ax) == len(f_ref) == 3 * 6 - 6
    assert np.abs(f_ax - f_ref).max() < 1.0      # the program's numerical Hessian (--acc 0.001)
    modes = np.asarray(ax.results.normal_modes_cartesian)
    assert modes.shape == (12, 6, 3)
    assert np.allclose(np.linalg.norm(modes.reshape(12, -1), axis=1), 1.0)


def test_hessian_is_symmetric_and_matches_gradient_differences():
    eng = GXTBCalculator(backend="axtb", n_parallel=1, extra_args=["--acc", "1e-6"])
    node = _node()
    h = eng.compute_hessian(node)
    assert np.allclose(h, h.T, atol=1e-10)
    x0 = np.asarray(node.structure.geometry, dtype=float)
    step = 1e-3
    for k in (0, 7, 13):
        cols = []
        for sign in (1, -1):
            x = x0.copy().reshape(-1)
            x[k] += sign * step
            s = node.structure.model_copy(update={"geometry": x.reshape(-1, 3)})
            cols.append(np.asarray(eng.compute_gradients([StructureNode(structure=s)])[0]).reshape(-1))
        assert np.abs((cols[0] - cols[1]) / (2 * step) - h[:, k]).max() < 1e-5


# A methyl formate IRC end from the scoreboard (sb-t11 methyl_formate r0, ch_14_15_3, first end, unminimized),
# 1e-3 eV/Angstrom from a minimum: at the program's SCF accuracy (1.0) its gradients are off by ~1e-4 Eh/bohr
# near that minimum, and qcopt took 3000 steps of 6e-8 bohr without reaching 1e-3 eV/Angstrom.
STALLED_END = """8

C  -0.13048586113018570 0.82638120508557267 -0.63158958588896341
O   0.33297317255590825 0.62450503273393010 0.67091335629957016
C  -0.54816062225775986 0.55994364146377240 1.65174908170331647
O   0.25774782666901480 -1.01058244716271828 -1.28201480947908153
H   0.54334785454241097 -0.81683686785019716 -2.18230811213430043
H  -1.20268330013034697 0.62798735694769581 -0.59957594161902716
H   1.04841243024736297 -1.16527278222517183 -0.74869469204531924
H   0.08105929314821832 0.58271664590564853 2.55740232518638511
"""


def test_scf_accuracy_defaults_to_0_01_and_acc_overrides_it():
    assert GXTBCalculator(backend="axtb")._axtb_settings["accuracy"] == 0.01
    assert GXTBCalculator(backend="axtb", extra_args=["--acc", "1.0"])._axtb_settings["accuracy"] == 1.0


def test_qcopt_minimizes_an_end_the_loose_scf_stalled():
    pytest.importorskip("qcopt")
    import mepd

    eng = GXTBCalculator(backend="axtb", axtb_conventions="gxtb", n_parallel=1)
    end = mepd.minimize(_node(STALLED_END), eng, check_frequencies=False, maxiter=200)
    assert np.linalg.norm(np.asarray(end.gradient).reshape(-1, 3), axis=1).max() < 1e-3 / 27.211386 * 0.529177


def test_options_it_cannot_apply_raise():
    with pytest.raises(ValueError, match="does not take"):
        GXTBCalculator(backend="axtb", extra_args=["--gfn", "2"])
    with pytest.raises(ValueError, match="minimizer"):
        GXTBCalculator(backend="axtb", minimizer="xtb")
    with pytest.raises(ValueError, match="g-xTB only"):
        GXTBCalculator(backend="axtb", add_gxtb_flag=False)
    with pytest.raises(ValueError, match="backend"):
        GXTBCalculator(backend="tblite")


def test_scf_failure_takes_the_retry_ladder(caplog):
    # 3 SCF iterations never converge: the ladder's 1000 K / 1000 iterations does.
    eng = GXTBCalculator(backend="axtb", n_parallel=1, extra_args=["--iterations", "3"])
    with caplog.at_level("WARNING"):
        e = eng.compute_energies([_node()])[0]
    assert np.isfinite(e)
    assert "did not converge" in caplog.text


def _child_hessian(q):
    eng = GXTBCalculator(backend="axtb", n_parallel=1)
    q.put(float(np.asarray(eng._compute_hessian_result(_node(ETHANE)).results.freqs_wavenumber)[-1]))


def test_forked_child_after_a_hessian_does_not_hang():
    """mepd forks workers (channel_search, network expansion): a Hessian and frequencies in the
    parent must leave nothing a forked child waits on."""
    eng = GXTBCalculator(backend="axtb", n_parallel=1)
    eng._compute_hessian_result(_node(ETHANE))
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_hessian, args=(q,))
    p.start()
    p.join(60)
    alive = p.is_alive()
    if alive:
        p.kill()
    assert not alive, "forked child hung"
    assert q.get(timeout=5) > 1000


@needs_program
def test_kept_orbitals_are_axtbs_and_count_as_gxtb(tmp_path, monkeypatch):
    """A job's kept SCF ($MEPD_ORBITALS_DIR) from backend "axtb" is axtb's Molden file (spherical,
    normalized primitives): kept as method "axtb", found by a g-xTB lookup, and read with its own
    conventions it gives the program's HOMO."""
    from mepd import orbitals

    for name, eng in (("axtb", GXTBCalculator(backend="axtb", n_parallel=1)), ("program", _program())):
        monkeypatch.setenv("MEPD_ORBITALS_DIR", str(tmp_path / name))
        eng.compute_energies([_node()])
    s = Structure.from_xyz(FORMAMIDE)
    symbols, coords = list(s.symbols), np.asarray(s.geometry) * 0.529177210903
    ax = orbitals.find_run_orbitals([tmp_path / "axtb"], symbols, coords, "gxtb")
    ref = orbitals.find_run_orbitals([tmp_path / "program"], symbols, coords, "gxtb")
    assert (ax["method"], ax["label"], ref["method"]) == ("axtb", "g-xTB", "gxtb")
    assert orbitals.find_run_orbitals([tmp_path / "axtb"], symbols, coords, "gfn2") is None

    rng = np.random.default_rng(0)
    pts = coords.mean(axis=0) / 0.529177210903 + rng.normal(scale=2.0, size=(400, 3))

    def homo(hit):
        m = hit["method"]
        mol = orbitals.parse_molden((tmp_path / hit["dir"] / hit["file"]).read_text(), orbitals.D_NORM[m])
        top = max((mo for mo in mol["mos"] if mo["occ"] > 0.5), key=lambda mo: mo["energy"])
        return orbitals.evaluate(mol, [top["coef"]], pts, orbitals.D_NORM[m], orbitals.NORMALIZED_PRIMS[m])[0]

    a, b = homo(ax), homo(ref)
    assert np.abs(np.abs(a) - np.abs(b)).max() < 1e-3 * np.abs(b).max()
