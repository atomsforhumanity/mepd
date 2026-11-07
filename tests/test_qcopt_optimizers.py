"""qcopt (AFH's optimizer) as the g-xTB engine's minimizer, TS optimizer and IRC
(`minimizer = "qcopt"`, `ts_optimizer = "qcopt"`), on axtb's g-xTB."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qcopt")
pytest.importorskip("axtb")

from qcdata import Structure  # noqa: E402

import mepd  # noqa: E402
from mepd.engines.gxtb import GXTBCalculator  # noqa: E402
from mepd.errors import GeometryOptimizationNotConvergedError  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402

# A distorted formamide (not a stationary point).
FORMAMIDE = """6

C   0.000   0.420   0.000
O   1.180   0.650   0.050
N  -0.920   1.380  -0.020
H  -0.330  -0.620   0.040
H  -1.900   1.150   0.030
H  -0.640   2.360   0.070
"""

# HCN <-> HNC: the hydrogen half-way round, near the saddle.
HCN_TS_GUESS = """3

C   0.000   0.000   0.000
N   1.180   0.000   0.000
H   0.750   1.050   0.000
"""


def _node(xyz: str) -> StructureNode:
    return StructureNode(structure=Structure.from_xyz(xyz))


def _engine(**kw) -> GXTBCalculator:
    return GXTBCalculator(backend="axtb", minimizer="qcopt", ts_optimizer="qcopt", n_parallel=1, **kw)


def _largest_force_ev(node: StructureNode) -> float:
    from qcref import constants as c

    g = np.asarray(node.gradient, dtype=float)
    return float(np.linalg.norm(g, axis=1).max() / float(c.EV_TO_HARTREE) * float(c.ANGSTROM_TO_BOHR))


def _h_on(node: StructureNode) -> str:
    """Which heavy atom the hydrogen sits on: "C" (HCN) or "N" (HNC)."""
    x = np.asarray(node.coords, dtype=float)
    return "C" if np.linalg.norm(x[2] - x[0]) < np.linalg.norm(x[2] - x[1]) else "N"


def test_minimization_reaches_a_true_minimum_with_its_energies_cached():
    engine = _engine()
    start = _node(FORMAMIDE)
    traj = engine.compute_geometry_optimization(start, {"fmax": 0.001})
    end = traj[-1]
    assert len(traj) > 2
    assert _largest_force_ev(end) < 0.001
    assert end.energy < engine.compute_energies([_node(FORMAMIDE)])[0]
    # the cached energy is the engine's own at that geometry
    fresh = StructureNode(structure=end.structure)
    # (to SCF convergence: a fresh calculation starts from another guess)
    assert abs(engine.compute_energies([fresh])[0] - end.energy) < 1e-8
    assert mepd.minimize(start, engine, fmax=0.001).frequencies.n_imaginary(10.0) == 0


def test_unconverged_minimization_raises_where_it_stopped():
    engine = _engine()
    with pytest.raises(GeometryOptimizationNotConvergedError) as info:
        engine.compute_geometry_optimization(_node(FORMAMIDE), {"fmax": 1e-6, "maxiter": 3})
    err = info.value
    assert err.steps == 3 and isinstance(err.structure, StructureNode)
    assert not np.allclose(err.structure.coords, _node(FORMAMIDE).coords)


def test_ts_search_and_irc_connect_hcn_and_hnc():
    engine = _engine()
    ts = mepd.find_ts(_node(HCN_TS_GUESS), engine)
    assert ts.frequencies.n_imaginary(10.0) == 1
    assert _largest_force_ev(ts) < 0.001
    result = mepd.irc(ts, engine)
    first, last = result.path.nodes[0], result.path.nodes[-1]
    assert sorted([_h_on(first), _h_on(last)]) == ["C", "N"]
    # the IRC path runs downhill from the TS on both sides
    energies = np.asarray(engine.compute_energies(result.irc_path), dtype=float)
    k = int(np.argmax(energies))
    assert 0 < k < len(energies) - 1
    assert np.allclose(result.irc_path.nodes[k].coords, ts.coords)


def test_irc_takes_sella_step_and_force_keywords():
    engine = _engine()
    ts = engine.compute_transition_state(_node(HCN_TS_GUESS), {"fmax": 0.001})
    fine = engine.compute_irc_chain(ts, {"dx": 0.05, "fmax": 0.05, "keep_going": True})
    coarse = engine.compute_irc_chain(ts, {"dx": 0.2, "fmax": 0.05})
    assert len(fine.nodes) > len(coarse.nodes)


def test_sella_only_options_raise():
    engine = _engine()
    with pytest.raises(ValueError, match="Sella options"):
        engine.compute_transition_state(_node(HCN_TS_GUESS), {"optimizer_kwds": {"eta": 1e-4}})
    with pytest.raises(ValueError, match="Sella options"):
        engine.compute_irc_chain(_node(HCN_TS_GUESS), {"eta": 1e-4})
    with pytest.raises(ValueError, match="ts_optimizer"):
        GXTBCalculator(backend="axtb", ts_optimizer="berny")


def test_minimization_with_hessian_every_uses_the_engines_hessians(monkeypatch):
    engine = _engine()
    calls = []
    original = type(engine).compute_hessian

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(engine), "compute_hessian", counted)
    traj = engine.compute_geometry_optimization(_node(FORMAMIDE), {"fmax": 0.001, "hessian_every": 2})
    assert _largest_force_ev(traj[-1]) < 0.001
    # the starting Hessian and one every 2 steps, all from the engine
    assert len(calls) >= 1 + (len(traj) - 1) // 2 >= 2


def test_geometric_coordsys_from_the_cli_is_ignored_like_sella_does():
    """mepd's CLI adds coordsys="cart" (a geomeTRIC option) to every minimization."""
    engine = _engine()
    traj = engine.compute_geometry_optimization(_node(FORMAMIDE), {"fmax": 0.001, "coordsys": "cart", "maxit": 500})
    assert _largest_force_ev(traj[-1]) < 0.001


def test_irc_that_runs_out_of_steps_still_gives_a_path():
    # qcopt >= 0.2 reports such an IRC as unconverged; mepd minimizes the ends itself
    engine = _engine()
    ts = engine.compute_transition_state(_node(HCN_TS_GUESS), {"fmax": 0.001})
    chain = engine.compute_irc_chain(ts, {"dx": 0.05, "steps": 3})
    assert len(chain.nodes) >= 5  # both branches and the TS
