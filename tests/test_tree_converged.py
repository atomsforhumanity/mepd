"""tree.json's `converged`: whether a node's path search reached its own stopping criterion, as the path
minimizer itself reports it (`PathMinimizer.search_converged`). It used to be true for every searched node,
also for a search stopped at its iteration limit."""
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

pytest.importorskip("ase")
from ase.calculators.emt import EMT  # noqa: E402

from mepd import tree_log  # noqa: E402
from mepd.chain import Chain  # noqa: E402
from mepd.engines.ase import ASEEngine  # noqa: E402
from mepd.inputs import RunInputs  # noqa: E402
from mepd.msmep import MSMEP  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402

A = 1.8897259886


def _node(xyz):
    return StructureNode(structure=Structure(symbols=["H", "C", "N"], geometry=np.array(xyz) * A,
                                             charge=0, multiplicity=1))


START = _node([[0.35, 0.0, -1.02], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
END = _node([[0.35, 0.0, 2.12], [0.0, 0.0, 0.0], [0.0, 0.0, 1.17]])


def _msmep(method, **path_min_inputs):
    ri = RunInputs(path_min_method=method, path_min_inputs={"do_elem_step_checks": False, **path_min_inputs})
    ri.engine = ASEEngine(calculator=EMT())
    ri.gi_inputs.nimages = 7
    return MSMEP(ri)


def _record(msmep) -> dict:
    """The tree.json record of one search of START -> END through MSMEP, as a run writes it."""
    chain = Chain.model_validate({"nodes": [START.copy(), END.copy()], "parameters": msmep.inputs.chain_inputs})
    minimizer, _ = msmep.run_minimize_chain(chain)
    return tree_log.node_record(SimpleNamespace(data=minimizer, children=[], leaf_status=""))


def test_tree_log_writes_what_the_minimizer_reports_and_infers_nothing():
    def rec(**data):
        return tree_log.node_record(SimpleNamespace(data=SimpleNamespace(chain_trajectory=[1, 2], **data),
                                                    children=[], leaf_status=""))["converged"]

    # msmep's own `converged` (optimize_chain returned without raising) is not the answer
    assert rec(search_converged=False, converged=True) is False
    assert rec(search_converged=True, converged=False) is True
    assert rec(converged=True) is None                       # the minimizer says nothing: unknown, not true
    assert rec(search_converged=None) is None
    # a node without a path search (skipped, pruned, failed before its search) has no answer
    assert tree_log.node_record(SimpleNamespace(data=None, children=[], leaf_status="identical_endpoints"))[
        "converged"] is None


def test_a_freezing_string_stopped_at_its_iteration_limit_is_not_converged():
    assert _record(_msmep("FNEB", max_grow_iter=1, max_min_iter=3))["converged"] is False


def test_a_complete_freezing_string_is_converged():
    assert _record(_msmep("FNEB", max_grow_iter=40, max_min_iter=10))["converged"] is True


def test_a_neb_stopped_at_its_step_limit_is_not_converged():
    assert _record(_msmep("NEB", max_steps=2))["converged"] is False


def test_a_neb_that_meets_its_criteria_is_converged():
    # criteria loose enough to be met at once: the flag follows NEB's own test, whatever its thresholds are
    loose = dict(rms_grad_thre=1e3, max_rms_grad_thre=1e3, ts_grad_thre=1e3, ts_spring_thre=1e3, barrier_thre=1e3,
                 en_thre=1e3, climb=False)
    assert _record(_msmep("NEB", max_steps=50, **loose))["converged"] is True


def test_gsm_reports_its_own_end_not_that_it_returned(monkeypatch):
    from mepd.pathminimizers.gsm import GSM

    def run(converged):
        gsm = GSM.__new__(GSM)
        gsm.initial_chain = Chain.model_validate({"nodes": [START.copy(), END.copy()],
                                                  "parameters": RunInputs().chain_inputs})
        gsm.engine = SimpleNamespace(compute_energies=lambda nodes: [n.__setattr__("_cached_energy", 0.0) for n in nodes])
        gsm.parameters = SimpleNamespace(seed_with_geodesic_interpolation=False, do_elem_step_checks=False)
        gsm.grad_calls_made = 0

        def attempt(chain, *a):
            gsm.gsm_attempts.append({"converged": converged})
            return chain, [], False

        gsm._execute_gsm_attempt_with_fallback = attempt
        gsm.optimize_chain()
        return gsm.search_converged

    assert run(False) is False      # out of iterations
    assert run(True) is True


def test_fsm_ts_takes_the_flag_of_the_method_that_made_the_path(monkeypatch):
    import test_fsm_ts as t

    def run(ends, string_complete, fallback_flag=None):
        """test_fsm_ts's stubbed FSM-TS run, with its stub string and stub NEB reporting these flags."""
        def setattr_(target, value=None, *rest, **kw):
            monkeypatch.setattr(target, value, *rest, **kw)
            if target == "mepd.pathminimizers.fneb.FreezingNEB":
                value.search_converged = string_complete
            if target == "mepd.neb.NEB":
                value.search_converged = fallback_flag

        return t._run(SimpleNamespace(setattr=setattr_), ends)[0]

    # no fallback: the TS converged and the pseudo-IRC decided; what is left is whether the string was complete
    assert run([t.P, t.R], True).search_converged is True          # elementary
    assert run([t.P, t.R], False).search_converged is False        # string cut at max_grow_iter
    assert run([t.X, t.R], True).search_converged is True          # split at a new species
    # both sides lead to the reactant: NEB searches the pair; its flag is the node's
    m = run([t.R, t.R], True, fallback_flag=False)
    assert m.path_method_used == "NEB" and m.search_converged is False
    assert run([t.R, t.R], True, fallback_flag=True).search_converged is True


def test_mlpgi_optimizer_says_whether_its_last_stage_ended_on_its_own_criterion():
    from mepd import mlp_geodesic as gi
    from test_mlpgi import _quad_energy

    q = np.linspace(-1, 1, 7)
    frames = np.stack([np.array([[x, 0.8 * (1 - x ** 2), 0.0], [0.0, 0.0, 0.0]]) for x in q])
    cut = gi.GeodesicOptimizer(frames, ["C", "H"], _quad_energy, align=False,
                               config=gi.OptimizerConfig(fire_stage1_iter=2, fire_stage2_iter=2,
                                                         refinement_step_interval=0))
    cut.optimize()
    assert cut.converged is False
    full = gi.GeodesicOptimizer(frames, ["C", "H"], _quad_energy, align=False)
    full.optimize()
    assert full.converged is True


def test_the_installed_geometric_has_what_the_neb_force_test_reads():
    """GeometricNEB.search_converged applies geomeTRIC's own force test to the returned band."""
    geometric_neb = pytest.importorskip("geometric.neb")
    import inspect

    from geometric.params import NEBParams

    params = NEBParams()
    assert isinstance(params.maxg, float) and isinstance(params.avgg, float)
    source = inspect.getsource(geometric_neb)      # set by the band classes OptimizeChain returns
    assert "self.maxg" in source and "self.avgg" in source
