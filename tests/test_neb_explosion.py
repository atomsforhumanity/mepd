"""A path step that blows an image up is rejected, the step shrunk, and the
branch fails after repeated blow-ups instead of being split on the wreck (#23)."""
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

import mepd.neb as neb_module
from mepd.chain import Chain
from mepd.errors import PathDivergedError
from mepd.inputs import ChainInputs
from mepd.neb import NEB, _step_exploded
from mepd.nodes.node import StructureNode


def _chain(energies, gmax=0.01):
    nodes = []
    for i, e in enumerate(energies):
        n = StructureNode(structure=Structure(geometry=np.array([[0.0, 0.0, 0.0], [1.4 + 0.1 * i, 0.0, 0.0]]),
                                              symbols=["H", "H"], charge=0, multiplicity=1))
        n._cached_energy = e
        n._cached_gradient = np.full((2, 3), gmax)
        nodes.append(n)
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def test_ordinary_steps_pass_and_blow_ups_are_named():
    before = _chain([-1.0, -0.98, -1.0])
    assert _step_exploded(before, _chain([-1.0, -0.99, -1.0])) is None
    assert "rose" in _step_exploded(before, _chain([-1.0, 1.0e7, -1.0]))
    assert "gradient" in _step_exploded(before, _chain([-1.0, -0.98, -1.0], gmax=1.6e8))
    assert "finite" in _step_exploded(before, _chain([-1.0, np.nan, -1.0]))
    # A model potential in its own units: a rise small against the path's span is a step.
    assert _step_exploded(_chain([0.0, 30.0, -5.0]), _chain([0.0, 31.5, -5.0])) is None
    # A large but proportionate gradient (an early, rough path) is not a blow-up.
    assert _step_exploded(_chain([-1.0, -0.9, -1.0], gmax=0.5), _chain([-1.0, -0.9, -1.0], gmax=2.0)) is None


def _neb(monkeypatch, energies_after):
    """An NEB whose every trial step lands on `energies_after`."""
    calls = []

    class Engine:
        def compute_gradients(self, chain):
            calls.append(chain)
            if len(calls) > 1:   # a trial step
                for n, e in zip(chain.nodes, energies_after):
                    n._cached_energy, n._cached_gradient = e, np.full((2, 3), 0.01)
            return np.array([n._cached_gradient for n in chain.nodes])

        def compute_energies(self, chain):
            return np.array([n._cached_energy for n in chain.nodes])

    opt = SimpleNamespace(timestep=1.0, g_old=None, reset=lambda: None,
                          optimize_step=lambda chain, chain_gradients: chain.copy())
    params = SimpleNamespace(use_geodesic_tangent=False, v=False, negative_steps_thre=10, positive_steps_thre=10)
    monkeypatch.setattr(neb_module.ch, "compute_NEB_gradient",
                        lambda chain, geodesic_tangent=False: np.zeros((len(chain), 2, 3)))
    start = _chain([-1.0, -0.98, -1.0])
    return NEB(initial_chain=start, optimizer=opt, parameters=params, engine=Engine()), start, opt


def test_an_exploding_step_is_retried_smaller_then_the_branch_fails(monkeypatch):
    neb, start, opt = _neb(monkeypatch, [-1.0, 1.5e7, -1.0])
    with pytest.raises(PathDivergedError, match="diverged"):
        neb.update_chain(start)
    assert opt.timestep == pytest.approx(0.5 ** neb_module.MAX_NRETRIES)


def test_an_ordinary_step_is_taken(monkeypatch):
    neb, start, opt = _neb(monkeypatch, [-1.0, -0.99, -1.0])
    new = neb.update_chain(start)
    assert new.energies[1] == pytest.approx(-0.99) and opt.timestep == 1.0
