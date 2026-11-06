"""The proposal MD (reactor, collide, the interactive Reactor): Langevin dynamics on axtb."""
import multiprocessing

import numpy as np
import pytest

axtb = pytest.importorskip("axtb")

from mepd.discovery.langevin import ETEMPS, LangevinMD  # noqa: E402

# formic acid (Z), Angstrom: O0 C1 O2(=) H3(on O0) H4(on C1)
SYM = ["O", "C", "O", "H", "H"]
XYZ = np.array([[0.74, -0.62, -0.08], [-0.41, 0.06, -0.16], [-0.56, 1.17, 0.27], [1.40, -0.07, 0.37],
                [-1.17, -0.55, -0.67]]) / 0.529177210903


def test_gfn2_on_axtb_is_tblites_gfn2():
    tblite = pytest.importorskip("tblite.interface")
    md = LangevinMD(SYM, XYZ, 0, 1, temperature=300.0)
    e, g = md.forces(XYZ, biased=False)
    ref = tblite.Calculator("GFN2-xTB", np.array([8, 6, 8, 1, 1]), XYZ)
    ref.set("verbosity", 0)
    ref.set("temperature", ETEMPS[0] * 3.166811563e-6)
    r = ref.singlepoint()
    assert e == pytest.approx(float(r.get("energy")), abs=1e-8)
    assert np.allclose(g, r.get("gradient"), atol=1e-6)


def test_the_method_is_an_option_and_gxtb_runs():
    md = LangevinMD(SYM, XYZ, 0, 1, temperature=1000.0, method="gxtb")
    assert md.method == "gxtb-v2.0.1"
    md.start()
    assert all(md.step() for _ in range(5))
    with pytest.raises(ValueError, match="method"):
        LangevinMD(SYM, XYZ, 0, 1, temperature=300.0, method="pm6")


def test_an_scf_that_fails_is_retried_at_another_smearing(monkeypatch):
    md = LangevinMD(SYM, XYZ, 0, 1, temperature=300.0)
    real = md._calc

    class Fails:
        def singlepoint(self, x, gradient=True):
            raise RuntimeError("SCF not converged in 250 iterations")

    monkeypatch.setattr(md, "_calc", lambda etemp: Fails() if etemp == ETEMPS[0] else real(etemp))
    e, _ = md.forces(XYZ, biased=False)
    assert np.isfinite(e) and md.scf_retries == 1 and set(md._calcs) == {ETEMPS[1]}
    monkeypatch.setattr(md, "_calc", lambda etemp: Fails())
    with pytest.raises(RuntimeError):            # every smearing failed: the step's SCF failure (step() retries)
        md.forces(XYZ, biased=False)


def _child(q, method):
    md = LangevinMD(SYM, XYZ, 0, 1, temperature=1500.0, method=method, seed=1)
    md.start()
    q.put(sum(md.step() for _ in range(50)))


@pytest.mark.parametrize("method", ["gfn2", "gxtb-v2.0.1"])
def test_md_in_a_forked_child_after_axtb_in_the_parent(method):
    """The explorer forks its queue workers, and a proposal job may run the MD in one: axtb used in the parent
    (an MD, and a Hessian on several threads) must leave nothing a forked child waits on."""
    parent = LangevinMD(SYM, XYZ, 0, 1, temperature=1500.0, method=method)
    parent.start()
    parent.step()
    axtb.Calculator("gxtb-v2.0.1", [8, 6, 8, 1, 1], XYZ, threads=2, conventions="gxtb").hessian(XYZ)
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, method))
    p.start()
    p.join(120)
    alive = p.is_alive()
    if alive:
        p.kill()
    assert not alive, "forked child hung"
    assert q.get(timeout=5) == 50


def test_uma_runs_on_mepds_fairchem_engine_and_its_model_server(monkeypatch):
    """md_method "uma": energies and gradients from FAIRChemEngine._predict (Hartree, Hartree/bohr), in this process
    or through the model server whose address the generators pass (their trajectories' steps batched on one GPU)."""
    import mepd.discovery.langevin as lg

    seen = []

    class Engine:                     # a harmonic stand-in for UMA: no weights or GPU needed
        server = None

        def _predict(self, atoms_list):
            seen.append((self.server, len(atoms_list[0]), atoms_list[0].info.get("charge")))
            x = atoms_list[0].positions / 0.529177210903
            return [0.5 * float((x ** 2).sum())], [x.copy()]

    def engine(server=None):
        eng = Engine()
        eng.server = server
        return eng

    monkeypatch.setattr(lg, "_uma_engine", engine)
    md = LangevinMD(SYM, XYZ, 0, 1, temperature=300.0, method="uma", server="/tmp/model.sock")
    e, g = md.forces(XYZ, biased=False)
    assert e == pytest.approx(0.5 * (XYZ ** 2).sum()) and np.allclose(g, XYZ)
    assert seen == [("/tmp/model.sock", 5, 0)]
    md.start()
    assert md.step() and md.scf_retries == 0
