"""ASE minimizations restart when the engine's surface changes under them, and a step that keeps
calculating is cut off (mepd.engines.ase._EvaluationGuard).

On mepd.afh.org a MINMAX search stood still after GFN2-xTB raised its electronic temperature in the middle
of a minimization ("recomputing the other 1 structure(s) of this batch at it"): ASE's L-BFGS line search
kept the start of the step from the old surface and took thousands of calculations instead of tens."""
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from qcdata import Structure

from mepd.engines import ase as mase
from mepd.engines.ase import ASEEngine
from mepd.nodes.node import StructureNode


class _TwoSurfaces(Calculator):
    """Harmonic bonds along an H4 chain. After `switch_at` calculations the surface changes (every energy
    shifted, the force constant scaled) and `engine._scc_epoch` goes up, as an xtb engine's does when it
    raises its electronic temperature."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, switch_at=None, **kw):
        super().__init__(**kw)
        self.n, self.switch_at = 0, switch_at
        self.engine = SimpleNamespace(_scc_epoch=0)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.n += 1
        if self.switch_at is not None and self.n == self.switch_at:
            self.engine._scc_epoch += 1
        k, shift = (1.03, 2e-3) if self.engine._scc_epoch else (1.0, 0.0)
        x = self.atoms.positions
        e, f = shift, np.zeros_like(x)
        for i in range(len(x) - 1):
            d = x[i + 1] - x[i]
            r = np.linalg.norm(d)
            e += k * (r - 0.74) ** 2
            g = 2 * k * (r - 0.74) * d / r
            f[i] += g
            f[i + 1] -= g
        self.results = {"energy": e, "forces": f}


def _node():
    xyz = np.array([[0, 0, 0], [1.1, 0.1, 0], [2.0, -0.2, 0.1], [3.1, 0.3, -0.2]], dtype=float)
    return StructureNode(structure=Structure(symbols=["H"] * 4, geometry=xyz / 0.529177210903, charge=0,
                                             multiplicity=1), has_molecular_graph=False)


def test_a_surface_change_restarts_the_minimization(caplog):
    calc = _TwoSurfaces(switch_at=4)
    eng = ASEEngine(calculator=calc, geometry_optimizer="LBFGSLineSearch")
    with caplog.at_level(logging.WARNING, logger="mepd.engines.ase"):
        traj = eng.compute_geometry_optimization(_node(), keywords={"fmax": 1e-4})
    assert "the engine's surface changed" in caplog.text
    assert caplog.text.count("restarting the optimization") == 1
    x = np.asarray(traj[-1].structure.geometry).reshape(-1, 3) * 0.529177210903
    assert np.allclose(np.linalg.norm(np.diff(x, axis=0), axis=1), 0.74, atol=1e-3)
    assert "get_property" not in calc.__dict__   # the guard leaves the calculator as it found it


def test_a_steady_surface_never_restarts(caplog):
    eng = ASEEngine(calculator=_TwoSurfaces(), geometry_optimizer="LBFGSLineSearch")
    with caplog.at_level(logging.WARNING, logger="mepd.engines.ase"):
        eng.compute_geometry_optimization(_node(), keywords={"fmax": 1e-4})
    assert "restarting" not in caplog.text


def test_a_step_that_keeps_calculating_is_cut_off():
    calc = _TwoSurfaces()
    guard = mase._EvaluationGuard(calc, calc.engine)
    guard.start()
    atoms = _node()
    from mepd.qcdata_structure_helpers import structure_to_ase_atoms

    a = structure_to_ase_atoms(atoms.structure)
    a.calc = calc
    try:
        with pytest.raises(mase._RestartOptimizer, match="one step took over"):
            for i in range(mase._MAX_EVALS_PER_STEP + 5):
                a.positions[0, 0] += 1e-3   # a new point each time, as a line search asks for
                a.get_potential_energy()
        guard.step_done()
        a.positions[0, 0] += 1e-3
        a.get_potential_energy()   # a finished step starts the count again
    finally:
        guard.release()
    assert "get_property" not in calc.__dict__
