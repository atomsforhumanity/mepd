"""NEB keeps the NEB gradient of the chains it just projected (team42/overhead): the step's
gradient correlation reuses them instead of projecting both chains again. The search must be
exactly the one that recomputes them."""
import numpy as np
import pytest
from qcdata import Structure

pytest.importorskip("ase")
from ase.calculators.emt import EMT  # noqa: E402

import mepd.chainhelpers as ch  # noqa: E402
from mepd.chain import Chain  # noqa: E402
from mepd.engines.ase import ASEEngine  # noqa: E402
from mepd.inputs import RunInputs  # noqa: E402
from mepd.msmep import MSMEP  # noqa: E402
from mepd.neb import NEB  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402

A = 1.8897259886


def _node(xyz):
    return StructureNode(structure=Structure(symbols=["H", "C", "N"], geometry=np.array(xyz) * A,
                                             charge=0, multiplicity=1))


START = _node([[0.35, 0.0, -1.02], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
END = _node([[0.35, 0.0, 2.12], [0.0, 0.0, 0.0], [0.0, 0.0, 1.17]])


def _search():
    ri = RunInputs(path_min_method="NEB", path_min_inputs={"max_steps": 12, "do_elem_step_checks": False})
    ri.engine = ASEEngine(calculator=EMT())
    ri.gi_inputs.nimages = 6
    msmep = MSMEP(ri)
    chain = msmep._create_interpolation(Chain.model_validate(
        {"nodes": [START.copy(), END.copy()], "parameters": ri.chain_inputs}))
    neb = msmep._construct_path_minimizer(initial_chain=chain)
    assert isinstance(neb, NEB)
    try:
        neb.optimize_chain()
    except Exception:
        pass   # not converged in 12 steps: the trajectory is what is compared
    return neb


def test_memoized_neb_gradient_gives_the_same_search(monkeypatch):
    calls = []
    real = ch.compute_NEB_gradient
    monkeypatch.setattr(ch, "compute_NEB_gradient", lambda *a, **k: calls.append(1) or real(*a, **k))
    memo = _search()
    n_memo = len(calls)
    monkeypatch.setattr(NEB, "_neb_gradient", lambda self, chain: real(
        chain, geodesic_tangent=self.parameters.use_geodesic_tangent))
    monkeypatch.setattr(ch, "_gradient_correlation", ch._gradient_correlation)
    plain = _search()
    assert memo.grad_calls_made == plain.grad_calls_made
    assert len(memo.chain_trajectory) == len(plain.chain_trajectory) > 3
    for a, b in zip(memo.chain_trajectory, plain.chain_trajectory):
        np.testing.assert_array_equal(a.coordinates, b.coordinates)
    assert n_memo <= len(memo.chain_trajectory) + 2    # about one projection per step, not three
