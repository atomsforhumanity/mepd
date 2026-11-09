"""The xtb engine's axtb backend: GFN2-xTB in process (xtb 6.7.1's conventions), the xtb program for GFN1 and for
point charges, reproducible SCFs, the program's SCC ladder, and analytic Hessians."""
import shutil

import numpy as np
import pytest

from mepd.engines import axtb_backend
from mepd.engines.gfnff import XTBEngine, default_backend

pytestmark = pytest.mark.skipif(axtb_backend.version() is None, reason="needs axtb")

WATER = ["O", "H", "H"], np.array([[0.0, 0.0, 0.221], [0.0, 1.430, -0.883], [0.0, -1.430, -0.883]])
METHYL = ["C", "H", "H", "H"], np.array([[0.0, 0.0, 0.04], [2.04, 0.0, 0.0], [-1.02, 1.767, 0.0], [-1.02, -1.767, 0.0]])


def test_gfn2_runs_in_axtb_by_default_and_gfn1_runs_the_program(monkeypatch):
    assert default_backend("gfn2") == "axtb" and default_backend("gfn1") == "xtb"
    assert XTBEngine().backend == "axtb" and XTBEngine(method="gfn1").backend == "xtb"
    assert XTBEngine(executable="/somewhere/xtb").backend == "xtb"      # a program named explicitly
    monkeypatch.setenv("MEPD_XTB_BACKEND", "xtb")
    assert XTBEngine().backend == "xtb"
    with pytest.raises(ValueError, match="GFN1"):
        XTBEngine(method="gfn1", backend="axtb")


def test_axtb_gives_xtb_671s_gfn2_and_the_same_numbers_call_after_call():
    eng = XTBEngine(backend="axtb", n_parallel=1)
    e1, g1, pcg = eng.energy_gradient(*WATER)
    assert pcg is None and g1.shape == (3, 3) and eng.last_charges.shape == (3,)
    eng.energy_gradient(WATER[0], WATER[1] + 0.05)            # another geometry in between
    e2, g2, _ = eng.energy_gradient(*WATER)
    assert e1 == e2 and np.array_equal(g1, g2)                 # each SCF from GFN2's own start, as the program
    import axtb

    ref = axtb.Calculator("gfn2", [8, 1, 1], WATER[1], conventions="xtb-6.7.1").singlepoint(WATER[1])
    assert abs(ref["energy"] - e1) < 1e-10
    e, g, _ = eng.energy_gradient(*METHYL, multiplicity=2)    # an open shell
    assert np.isfinite(e) and g.shape == (4, 3)


@pytest.mark.skipif(shutil.which("xtb") is None, reason="needs the xtb program")
def test_axtb_agrees_with_the_xtb_program():
    ax = XTBEngine(backend="axtb", n_parallel=1)
    pr = XTBEngine(backend="xtb", n_parallel=1)
    for (sym, x), mult in ((WATER, 1), (METHYL, 2)):
        ea, ga, _ = ax.energy_gradient(sym, x, 0, mult)
        ep, gp, _ = pr.energy_gradient(sym, x, 0, mult)
        assert abs(ea - ep) < 1e-8 and np.abs(ga - gp).max() < 1e-6


def test_point_charges_run_the_program(monkeypatch):
    eng = XTBEngine(backend="axtb", n_parallel=1)
    called = {}

    def fake_run(self, workdir, symbols, coords, charge, grad=True, uhf=0, point_charges=None, etemp=None):
        called["pc"] = point_charges
        (workdir / "pcgrad").write_text("0 0 0\n")
        return -1.0, np.zeros((len(symbols), 3))

    monkeypatch.setattr(XTBEngine, "_run", fake_run)
    monkeypatch.setattr(XTBEngine, "xtb_program", lambda self: "xtb")
    e, _, pcg = eng.energy_gradient(*WATER, point_charges=[(0.4, (5.0, 0.0, 0.0))])
    assert e == -1.0 and called["pc"] and pcg.shape == (1, 3)


def test_an_scf_that_needs_smearing_holds_that_temperature(monkeypatch):
    eng = XTBEngine(backend="axtb", n_parallel=1)
    seen = []

    def fake_point(numbers, positions, *, etemp=None, **kw):
        seen.append(etemp)
        return {"energy": -1.0, "gradient": np.zeros((len(numbers), 3)), "charges": None,
                "etemp": 1000.0 if etemp is None else None}

    monkeypatch.setattr(axtb_backend, "gfn2_point", fake_point)
    eng.energy_gradient(*WATER)
    eng.energy_gradient(*WATER)
    assert seen == [None, 1000.0] and eng._scc_floor == 1000.0 and eng._scc_epoch == 1


def test_the_hessian_is_axtbs_analytic_one():
    from qcdata import Structure

    from mepd.nodes.node import StructureNode

    eng = XTBEngine(backend="axtb", n_parallel=1)
    assert eng.analytic_hessian
    node = StructureNode(structure=Structure(symbols=WATER[0], geometry=WATER[1]))
    h = eng.compute_hessian(node)
    x, step = WATER[1].reshape(-1), 1e-3
    fd = np.array([(eng.energy_gradient(WATER[0], (x + step * e).reshape(3, 3))[1].reshape(-1)
                    - eng.energy_gradient(WATER[0], (x - step * e).reshape(3, 3))[1].reshape(-1)) / (2 * step)
                   for e in np.eye(9)])
    assert h.shape == (9, 9) and np.abs(h - 0.5 * (fd + fd.T)).max() < 1e-4
