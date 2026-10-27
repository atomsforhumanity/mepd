"""Harmonic distance restraints (#26): `[[restraints]]` in a profile hold a
distance while the rest optimizes, on plain engines and on QM/MM (where they
go into the QM/MM energy, so microiterations keep them)."""
import numpy as np
import pytest
from qcdata import Structure
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.errors import InputsError
from mepd.nodes.node import StructureNode
from mepd.restraints import (Restraint, RestrainedEngine, parse_restraints, restraint_energy_gradient,
                             restraint_report)


def test_the_gradient_is_the_energys_derivative():
    rs = [Restraint(0, 1, 1.10, 500.0), Restraint(1, 2, 2.0, 50.0)]
    x = np.random.default_rng(3).normal(size=(3, 3)) * 1.5
    e, g = restraint_energy_gradient(rs, x)
    h, num = 1e-5, np.zeros_like(x)
    for a in range(3):
        for c in range(3):
            xp, xm = x.copy(), x.copy()
            xp[a, c] += h
            xm[a, c] -= h
            num[a, c] = (restraint_energy_gradient(rs, xp)[0] - restraint_energy_gradient(rs, xm)[0]) / (2 * h)
    assert e > 0 and np.allclose(g, num, atol=1e-7)
    # At the target: no energy, no force.
    y = np.array([[0.0, 0.0, 0.0], [1.10 * ANGSTROM_TO_BOHR, 0, 0]])
    e0, g0 = restraint_energy_gradient([Restraint(0, 1, 1.10)], y)
    assert e0 == pytest.approx(0) and np.abs(g0).max() < 1e-12


@pytest.mark.parametrize("bad, msg", [
    ({"atoms": [1], "distance": 1.0}, "two different"),
    ({"atoms": [1, 1], "distance": 1.0}, "two different"),
    ({"atoms": [0, 1]}, "target distance"),
    ({"atoms": [0, 1], "distance": 1.0, "k": 3}, "unknown key"),
    ({"atoms": [0, 1], "distance": 1.0, "force_constant": -1}, "force_constant"),
])
def test_bad_restraints_are_refused_with_what_to_write(bad, msg):
    with pytest.raises(InputsError, match=msg):
        parse_restraints([bad])


def test_a_restrained_optimization_holds_the_distance():
    from mepd.inputs import RunInputs

    ri = RunInputs(engine_name="ase", ase_engine_kwds={"calculator": "ase.calculators.lj:LennardJones",
                                                       "calculator_kwds": {"sigma": 3.4, "epsilon": 0.0104,
                                                                           "rc": 10.0},
                                                       "geometry_optimizer": "LBFGS"},
                   restraints=[{"atoms": [0, 1], "distance": 5.0}])
    assert isinstance(ri.engine, RestrainedEngine)
    x = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0]]) * ANGSTROM_TO_BOHR
    node = StructureNode(structure=Structure(symbols=["Ar"] * 3, geometry=x), has_molecular_graph=False)
    traj = ri.engine.compute_geometry_optimization(node, keywords={"fmax": 1e-4, "maxiter": 500})
    y = np.asarray(traj[-1].coords) / ANGSTROM_TO_BOHR
    assert np.linalg.norm(y[0] - y[1]) == pytest.approx(5.0, abs=0.02)     # held (LJ pulls it to ~3.8)
    assert np.linalg.norm(y[0] - y[2]) == pytest.approx(3.82, abs=0.05)    # the rest relaxed
    (line,) = restraint_report(parse_restraints(ri.restraints), traj[-1].coords)
    assert line.startswith("atoms 0-1: 5.0")
    assert ri.to_dict()["restraints"] == [{"atoms": [0, 1], "distance": 5.0}]


def test_qmmm_restraints_go_into_the_qmmm_energy_and_only_on_the_qm_region():
    from mepd.engines.qmmm import QMMMEngine
    from mepd.inputs import RunInputs
    from mepd.qmmm import QMMMRegion
    from tests.test_qmmm import QM_PROPANOL, _propanol, _xtb

    _xtb()
    s = _propanol()
    region = QMMMRegion.build(s, QM_PROPANOL, active_radius=None)
    with pytest.raises(InputsError, match="QM region only"):
        RunInputs(engine_name="xtb", qmmm={"reference": s.to_xyz(), "qm_atoms": QM_PROPANOL},
                  restraints=[{"atoms": [0, 3], "distance": 1.5}])
    plain = QMMMEngine(base=RunInputs(engine_name="xtb").engine, region=region)
    held = QMMMEngine(base=RunInputs(engine_name="xtb").engine, region=region,
                      restraints=[{"atoms": [2, 3], "distance": 1.6}])
    a, b = plain.evaluate([StructureNode(structure=s)])[0], held.evaluate([StructureNode(structure=s)])[0]
    e, g = restraint_energy_gradient(held.restraints, s.geometry)
    assert b["restraints"] == pytest.approx(e) and e > 0
    assert b["energy"] == pytest.approx(a["energy"] + e)
    assert np.allclose(b["gradient"], a["gradient"] + g)
