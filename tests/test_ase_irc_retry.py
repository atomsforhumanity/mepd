"""Sella's IRC can stop on its first steps off the saddle
(IRCInnerLoopConvergenceFailure -- every g-xTB IRC tried did). The ASE
engine then reruns that direction from the TS with keep_going, which steps
on past it; any other failure still raises."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")


class IRCInnerLoopConvergenceFailure(Exception):
    pass


def _fake_irc(calls: list, fail_always: bool = False):
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import Trajectory

    class FakeIRC:
        def __init__(self, atoms, logfile=None, trajectory=None, keep_going=False, **kw):
            self.atoms, self.trajectory, self.keep_going = atoms, trajectory, keep_going

        def run(self, fmax, steps, direction):
            calls.append((direction, self.keep_going))
            if fail_always or not self.keep_going:
                raise IRCInnerLoopConvergenceFailure()
            with Trajectory(self.trajectory, "w") as t:
                for d in (0.05, 0.10):
                    a = self.atoms.copy()
                    a.positions[0] += d if direction == "forward" else -d
                    a.calc = SinglePointCalculator(a, energy=-1.0, forces=np.zeros((len(a), 3)))
                    t.write(a)

    return FakeIRC


def _ts():
    from qcconst.constants import ANGSTROM_TO_BOHR
    from qcdata import Structure

    from mepd.nodes.node import StructureNode

    xyz = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]) * ANGSTROM_TO_BOHR
    return StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=xyz, charge=0, multiplicity=1))


def test_an_inner_loop_failure_is_retried_with_keep_going(monkeypatch):
    from ase.calculators.emt import EMT

    import mepd.engines.ase as ase_engine

    calls: list = []
    monkeypatch.setattr(ase_engine, "SellaIRC", _fake_irc(calls))
    chain = ase_engine.ASEEngine(calculator=EMT()).compute_irc_chain(_ts())
    assert calls == [("reverse", False), ("reverse", True), ("forward", False), ("forward", True)]
    assert len(chain) == 5                     # 2 frames each way, and the TS


def test_a_failure_that_keep_going_cannot_fix_still_raises(monkeypatch):
    from ase.calculators.emt import EMT

    import mepd.engines.ase as ase_engine
    from mepd.errors import ElectronicStructureError

    calls: list = []
    monkeypatch.setattr(ase_engine, "SellaIRC", _fake_irc(calls, fail_always=True))
    with pytest.raises(ElectronicStructureError, match="IRCInnerLoopConvergenceFailure"):
        ase_engine.ASEEngine(calculator=EMT()).compute_irc_chain(_ts())
    assert calls == [("reverse", False), ("reverse", True)]


def _rotation(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def test_rigid_spin_is_stripped_from_the_frames_and_gradients_turn_with_them():
    from mepd.engines.ase import _align_frames

    ref = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    stretched = ref.copy()
    stretched[1, 0] = 1.0
    grad = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # The second and third frames are the stretched one, spun and shifted: only rigid motion after it.
    frames = [stretched, stretched @ _rotation(30).T + 2.0, stretched @ _rotation(60).T]
    grads = [grad, grad @ _rotation(30).T, grad @ _rotation(60).T]
    out, out_grads = _align_frames(frames, grads, ref)
    for f, g in zip(out, out_grads):
        assert np.allclose(f, out[0], atol=1e-8) and np.allclose(g, out_grads[0], atol=1e-8)
    assert np.isclose(np.linalg.norm(out[0][1] - out[0][0]), 1.0)   # the real motion (the stretch) is kept


def test_irc_convergence_ignores_the_rotational_eigenvalues():
    """Sella's test (lowest Hessian eigenvalue > 0) fails at a minimum when noise pushes a
    rotation's ~0 eigenvalue negative; the vibrational curvature is what counts."""
    from types import SimpleNamespace

    from mepd.engines.ase import _rigid_basis, _sella_irc_class

    pos = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    rigid = _rigid_basis(pos)
    assert rigid.shape == (9, 6)
    u, _, _ = np.linalg.svd(rigid, full_matrices=True)
    vib = u[:, 6:]
    hessian = vib @ np.diag([0.3, 0.5, 0.7]) @ vib.T - 1e-4 * rigid @ rigid.T   # a minimum, noisy rotations

    class Base:
        def converged(self, forces=None):
            return "sella's own"

    def irc(constraints=(), vib_evals=None):
        h = hessian if vib_evals is None else vib @ np.diag(vib_evals) @ vib.T
        obj = _sella_irc_class(Base)()
        obj.first, obj.fmax = False, 0.1
        obj.atoms = SimpleNamespace(constraints=list(constraints), get_positions=lambda: pos)
        obj.pes = SimpleNamespace(H=SimpleNamespace(evals=np.linalg.eigvalsh(h), asarray=lambda: h),
                                  converged=lambda fmax: (True,))
        return obj.converged()

    assert np.linalg.eigvalsh(hessian)[0] < 0 and irc() is True
    assert irc(vib_evals=[-0.2, 0.5, 0.7]) is False            # still on a downhill mode
    assert irc(constraints=["fixed"]) == "sella's own"          # fixed atoms: rotations are real motions
