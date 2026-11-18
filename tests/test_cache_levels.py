"""Results cached on nodes carry the engine level that computed them (mepd.engines.engine.cache_level).

Before, an engine returned whatever energy a node had cached, whichever engine had computed it: a node with a
gas-phase energy came back unsolvated from a SolvatedEngine, and barrier_kcal(ts, r, engine) used cached
energies from any level."""
from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
from qcdata import Structure

from mepd.compute import barrier_kcal
from mepd.engines.engine import Engine, cache_level
from mepd.nodes.node import StructureNode


def _node(x: float) -> StructureNode:
    geom = np.array([[0.0, 0.0, 0.0], [1.43 + x, 0.0, 0.95], [-1.43, 0.0, 0.95]])
    return StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=geom, charge=0, multiplicity=1))


@dataclasses.dataclass
class _Toy(Engine):
    """E = scale * x of atom 1 (Hartree); counts the points it computes."""
    scale: float = 1.0
    n_parallel: int = 1
    calls: int = 0

    def _run(self, nodes):
        from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
        from mepd.nodes.nodehelpers import update_node_cache

        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        self.calls += len(todo)
        res = []
        for n in todo:
            g = np.zeros_like(n.coords)
            g[1, 0] = self.scale
            res.append(FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                {"energy": self.scale * float(n.coords[1, 0]), "gradient": g})}))
        update_node_cache(todo, res)
        return nodes

    def compute_energies(self, chain):
        return np.array([n.energy for n in self._run(list(chain))])

    def compute_gradients(self, chain):
        return np.array([n.gradient for n in self._run(list(chain))])


class _FakeCorrection:
    """dG_solv = -0.01 * x of atom 1."""
    def __init__(self, solvent, model="alpb", **kw):
        self.solvent, self.model, self.fallbacks = solvent, model, []

    def compute(self, nodes):
        dg = np.array([-0.01 * n.coords[1, 0] for n in nodes])
        grads = []
        for n in nodes:
            g = np.zeros_like(n.coords)
            g[1, 0] = -0.01
            grads.append(g)
        return dg, np.array(grads)


def test_a_node_cached_by_another_level_is_recomputed():
    gas, other = _Toy(scale=1.0), _Toy(scale=2.0)
    n = _node(0.1)
    x = n.coords[1, 0]
    assert gas.compute_energies([n])[0] == pytest.approx(x)
    assert other.compute_energies([n])[0] == pytest.approx(2 * x)        # not the cached gas-phase value
    assert gas.compute_energies([n])[0] == pytest.approx(x)              # and back
    assert other.compute_gradients([n])[0][1, 0] == pytest.approx(2.0)


def test_the_same_level_reuses_the_cache():
    eng = _Toy(scale=1.0)
    n = _node(0.1)
    eng.compute_energies([n])
    eng.compute_gradients([n])
    eng.compute_energies([n])
    assert eng.calls == 1
    # An equal engine (another object, e.g. in another process) is the same level.
    twin = _Toy(scale=1.0)
    twin.compute_energies([n])
    assert twin.calls == 0
    # Pool width and counters are not part of the level.
    assert cache_level(_Toy(scale=1.0, n_parallel=8)) == cache_level(_Toy(scale=1.0))


def test_a_solvated_engine_does_not_return_a_gas_phase_energy(monkeypatch):
    import mepd.solvation as S

    monkeypatch.setattr(S, "SolvationCorrection", _FakeCorrection)
    base = _Toy(scale=1.0)
    solv = S.SolvatedEngine(base, "water")
    n = _node(0.1)
    x = n.coords[1, 0]
    base.compute_energies([n])                                             # gas phase first
    assert solv.compute_energies([n])[0] == pytest.approx(0.99 * x)
    assert base.compute_energies([n])[0] == pytest.approx(x)              # and the reverse
    m = _node(0.2)
    solv.compute_gradients([m])
    calls = base.calls
    assert solv.compute_energies([m])[0] == pytest.approx(0.99 * m.coords[1, 0])
    assert base.calls == calls                                             # same level: from the cache


def test_settings_changed_by_replace_are_another_level():
    """g-xTB in solvent through axtb is the same engine class with `axtb_solvation` set (dataclasses.replace)."""
    gas = _Toy(scale=1.0)
    cache_level(gas)                                                       # computed and kept on first use
    solv = dataclasses.replace(gas, scale=0.99)
    assert cache_level(solv) != cache_level(gas)
    n = _node(0.1)
    gas.compute_energies([n])
    assert solv.compute_energies([n])[0] == pytest.approx(0.99 * n.coords[1, 0])


def test_an_untagged_cache_is_trusted():
    """A cache set outside engines (read from a file) has no level and is used as before."""
    n = _node(0.1)
    n._cached_energy = -1.0
    n._cached_gradient = np.zeros_like(n.coords)
    eng = _Toy(scale=1.0)
    assert eng.compute_energies([n])[0] == -1.0
    assert eng.calls == 0
    assert n._cached_level is None


def test_the_tag_travels_with_copies_and_pickles_and_is_cleared_by_new_coords():
    eng = _Toy(scale=1.0)
    n = _node(0.1)
    eng.compute_gradients([n])
    assert n._cached_level == cache_level(eng)
    assert n.copy()._cached_level == n._cached_level
    assert pickle.loads(pickle.dumps(n))._cached_level == n._cached_level
    assert StructureNode.from_serializable(n.to_serializable())._cached_level == n._cached_level
    moved = n.update_coords(n.coords + 0.01)
    assert moved._cached_energy is None and moved._cached_level is None


def test_optimization_results_are_tagged():
    @dataclasses.dataclass
    class _Opt(_Toy):
        def compute_geometry_optimization(self, node, keywords=None):
            out = node.update_coords(node.coords * 1.0)
            self._run([out])
            return [out]

    eng = _Opt(scale=1.0)
    traj = eng.compute_geometry_optimization(_node(0.1))
    assert traj[-1]._cached_level == cache_level(eng)
    other = _Toy(scale=3.0)
    assert other.compute_energies([traj[-1]])[0] == pytest.approx(3 * traj[-1].coords[1, 0])


def test_barrier_kcal_uses_the_engine_it_is_given():
    gas, solv = _Toy(scale=1.0), _Toy(scale=0.5)
    ts, r = _node(0.3), _node(0.1)
    gas.compute_energies([ts, r])
    hartree_to_kcal = 627.509474
    b_gas = barrier_kcal(ts, r)
    b_solv = barrier_kcal(ts, r, engine=solv)
    assert b_gas == pytest.approx((ts.coords[1, 0] - r.coords[1, 0]) * hartree_to_kcal, rel=1e-6)
    assert b_solv == pytest.approx(0.5 * b_gas, rel=1e-6)
    with pytest.raises(ValueError, match="pass an engine"):
        barrier_kcal(_node(0.5), r)
