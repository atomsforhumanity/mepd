"""FSM-TS: FSM, then the TS and a pseudo-IRC decide the pair's fate.

The engine steps (FSM growth, TS optimization, Hessian, minimizations, NEB,
GSM) are stubbed so each verdict -- elementary, split, fallback -- is driven
exactly; the species comparison is mepd's own (molecular graphs).
"""
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode
from mepd.pathminimizers import fsm_ts
from mepd.pathminimizers.fsm_ts import FSMTS

A2B = 1.8897259886


def _water(h1=(0.758, 0.0, 0.504), h2=(-0.758, 0.0, 0.504)):
    xyz = np.array([[0.0, 0.0, 0.0], h1, h2]) * A2B
    return StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=xyz, charge=0, multiplicity=1))


R = _water()                                                  # H2O
P = _water(h1=(2.6, 0.0, 0.45))                               # OH + H
X = _water(h1=(2.6, 0.0, 0.45), h2=(-2.6, 0.0, 0.45))         # O + 2 H (a new species)
TS = _water(h1=(1.3, 0.0, 0.48))


class _Engine:
    """What FSM-TS calls on an engine; the TS is set per test (None: TS search fails)."""

    def __init__(self, ts=TS):
        self.ts = ts

    def compute_transition_state(self, node, keywords=None):
        if self.ts is None:
            raise RuntimeError("geomeTRIC did not converge")
        return self.ts

    def compute_energies(self, nodes):
        for n in nodes:
            n._cached_energy = -76.0


def _run(monkeypatch, ends, *, freqs=(-900.0, 120.0), engine=None, ancestors=None, gsm=None, **params):
    """FSMTS on R -> P with the pseudo-IRC minima `ends` ([plus, minus]).
    The fallback is NEB unless `gsm` is given: then GSM, whose stub returns
    `gsm` (an exception is raised)."""
    params.setdefault("fallback", "GSM" if gsm is not None else "NEB")
    ri = RunInputs(path_min_method="FSM-TS", path_min_inputs=params)
    chain = Chain.model_validate({"nodes": [R.copy(), TS.copy(), P.copy()], "parameters": ri.chain_inputs})
    for node, e in zip(chain.nodes, (-76.40, -76.30, -76.35)):   # the TS guess is the highest image
        node._cached_energy = e
    if ancestors:
        chain._split_ancestors = ancestors   # as MSMEP tags a child pair

    class FakeFSM:
        def __init__(self, **kw):
            assert kw["parameters"].do_elem_step_checks is False   # FSM's own check is off
            self.chain_trajectory, self.optimized = [chain], chain
            self.grad_calls_made = self.geom_grad_calls_made = 0

        def optimize_chain(self):
            return None

    neb_runs = []

    class FakeNEB:
        def __init__(self, initial_chain, parameters, optimizer, engine):
            neb_runs.append(parameters)
            self.chain_trajectory, self.optimized = [initial_chain], initial_chain

        def optimize_chain(self):
            return "NEB's verdict"

    monkeypatch.setattr("mepd.pathminimizers.fneb.FreezingNEB", FakeFSM)
    monkeypatch.setattr("mepd.neb.NEB", FakeNEB)

    class FakeGSM:
        def __init__(self, initial_chain, engine, parameters, gi_inputs):
            neb_runs.append(("GSM", parameters))
            self.chain_trajectory, self.optimized = [initial_chain], initial_chain
            self.grad_calls_made, self.geom_grad_calls_made = 5, 0

        def optimize_chain(self):
            if isinstance(gsm, Exception):
                raise gsm
            return gsm

    monkeypatch.setattr("mepd.pathminimizers.gsm.GSM", FakeGSM)
    monkeypatch.setattr(FSMTS, "_gsm_executable", lambda self: "/opt/gsm" if gsm is not None else None)
    monkeypatch.setattr("mepd.discovery.vri._optimize_nodes", lambda engine, nodes, kw: list(ends))
    monkeypatch.setattr(FSMTS, "_modes", lambda self, ts: SimpleNamespace(
        freqs=np.array(freqs), lowest_mode_cart=np.ones((3, 3))))
    m = FSMTS(initial_chain=chain, engine=engine or _Engine(), optimizer=SimpleNamespace(),
              parameters=ri.path_min_inputs, gi_inputs=ri.gi_inputs)
    return m, m.optimize_chain(), neb_runs


def test_both_ends_reached_is_an_elementary_step_and_keeps_the_ts(monkeypatch):
    m, res, neb = _run(monkeypatch, [P, R])
    assert res.is_elem_step and not neb
    assert m.ts_node is not None and m.path_method_used == "FSM"
    assert m.pseudo_irc["verdict"] == "elementary" and m.pseudo_irc["ends"] == ["product", "reactant"]
    assert [m._same(n, s) for n, s in zip(res.minimization_results, (R, P))] == [True, True]   # reactant side first
    # The result ends on [reactant, converged TS, product] (FSM's path kept before it): whatever takes a path's
    # highest image as its TS (--use-tsopt, channels, the web) starts from the TS the pseudo-IRC verified.
    final = m.chain_trajectory[-1]
    assert len(final) == 3 and m.optimized is final and len(m.chain_trajectory) == 2
    assert np.allclose(final[1].coords, TS.coords) and np.allclose(final[0].coords, R.coords)
    assert final.get_ts_node() is final[1]


def test_the_ts_optimization_is_converged_tightly_and_ts_fmax_overrides_it(monkeypatch):
    # At 0.01 eV/Å a soft mode can be unsettled, so a real TS looked like a saddle of order 2.
    class Recording(_Engine):
        def compute_transition_state(self, node, keywords=None):
            self.keywords = keywords
            return super().compute_transition_state(node, keywords)

    for params, fmax in (({}, 0.001), ({"ts_fmax": 0.005}, 0.005)):
        eng = Recording()
        _run(monkeypatch, [P, R], engine=eng, **params)
        assert eng.keywords == {"maxiter": 500, "fmax": fmax}


def test_a_new_species_on_one_side_splits_there(monkeypatch):
    m, res, neb = _run(monkeypatch, [R, X])
    assert not res.is_elem_step and res.splitting_criterion == "maxima" and not neb
    assert len(m.chain_trajectory) == 1   # a split keeps FSM's path (its TS does not join R and P)
    assert len(res.minimization_results) == 1 and m._same(res.minimization_results[0], X)
    assert m.ts_node is None


def test_two_new_species_split_at_both_in_path_order(monkeypatch):
    near_p = _water(h1=(2.6, 0.0, 0.45), h2=(-0.758, 0.0, 2.9))   # O-H with the other H elsewhere: also new
    m, res, _ = _run(monkeypatch, [X, near_p])
    assert res.splitting_criterion == "maxima" and len(res.minimization_results) in (1, 2)
    assert all(not m._same(n, R) and not m._same(n, P) for n in res.minimization_results)


@pytest.mark.parametrize("ends,side", [((R, R), "reactant"), ((P, P), "product")])
def test_a_ts_whose_sides_both_fall_to_one_end_is_discarded_for_neb(monkeypatch, ends, side):
    m, res, neb = _run(monkeypatch, list(ends), neb_inputs={"max_steps": 77})
    assert res == "NEB's verdict" and m.path_method_used == "NEB" and m.ts_node is None
    assert f"minimize to the {side}" in m.fallback_reason
    # NEB runs with NEB's own settings, the shared ones, and the user's overrides.
    assert neb[0].max_steps == 77 and hasattr(neb[0], "climb") and neb[0].skip_identical_graphs is True


def test_a_ts_leading_back_to_a_species_split_above_falls_back_instead_of_splitting(monkeypatch):
    # Pair R -> P came from splitting X -> P; a TS joining R and X leads back up the tree.
    m, res, neb = _run(monkeypatch, [R, X], ancestors=[(X, P)])
    assert res == "NEB's verdict" and neb and "came from" in m.fallback_reason
    assert m.pseudo_irc["ends"] == ["reactant", "earlier"]
    # The same ends with no such ancestry: a split at X, as before.
    m, res, neb = _run(monkeypatch, [R, X])
    assert res.splitting_criterion == "maxima" and not neb


def test_a_failed_ts_search_or_a_ts_with_no_imaginary_mode_falls_back_to_neb(monkeypatch):
    m, res, _ = _run(monkeypatch, [P, R], engine=_Engine(ts=None))
    assert res == "NEB's verdict" and "TS optimization failed" in m.fallback_reason
    m, res, _ = _run(monkeypatch, [P, R], freqs=(35.0, 120.0))
    assert res == "NEB's verdict" and "ended at a minimum" in m.fallback_reason
    # Two imaginary modes: a second-order saddle, not a TS. Noise above the cutoff (-20 cm^-1) does not count.
    m, res, _ = _run(monkeypatch, [P, R], freqs=(-900.0, -150.0, 80.0))
    assert res == "NEB's verdict" and "saddle of order 2" in m.fallback_reason
    m, res, _ = _run(monkeypatch, [P, R], freqs=(-900.0, -12.0, 80.0))
    assert res.is_elem_step and m.pseudo_irc["n_imaginary"] == 1


def test_with_the_fallback_off_the_fsm_path_gets_fsms_usual_check(monkeypatch):
    seen = []
    monkeypatch.setattr(fsm_ts, "check_if_elem_step", lambda chain, engine, **kw: seen.append(len(chain)) or "FSM check")
    m, res, neb = _run(monkeypatch, [R, R], neb_fallback=False)
    assert res == "FSM check" and seen == [3] and not neb and m.path_method_used == "FSM"


def test_gsm_is_the_fallback_with_its_own_settings_the_shared_ones_and_overrides(monkeypatch):
    m, res, runs = _run(monkeypatch, [R, R], gsm="GSM's verdict", gsm_inputs={"nnodes": 20})
    assert res == "GSM's verdict" and m.path_method_used == "GSM" and m.ts_node is None
    (name, p), = runs
    assert name == "GSM" and p.nnodes == 20 and hasattr(p, "max_opt_iters") and p.skip_identical_graphs is True
    assert m.grad_calls_made == 5


def test_a_failing_gsm_hands_the_pair_to_neb_and_no_binary_means_neb(monkeypatch):
    m, res, runs = _run(monkeypatch, [R, R], gsm=RuntimeError("gsm crashed"))
    assert res == "NEB's verdict" and m.path_method_used == "NEB" and "GSM failed" in m.fallback_reason
    assert runs[0][0] == "GSM" and len(runs) == 2
    m, res, runs = _run(monkeypatch, [R, R], fallback="GSM")   # no binary
    assert res == "NEB's verdict" and m.path_method_used == "NEB" and len(runs) == 1


def test_fallback_none_or_the_old_switch_off_keeps_the_fsm_path(monkeypatch):
    monkeypatch.setattr(fsm_ts, "check_if_elem_step", lambda chain, engine, **kw: "FSM check")
    for off in ({"fallback": "none"}, {"neb_fallback": False, "fallback": "GSM"}):
        m, res, runs = _run(monkeypatch, [R, R], **off)
        assert res == "FSM check" and not runs and m.path_method_used == "FSM"


def test_fsm_ts_is_a_path_method_with_its_aliases():
    from mepd.msmep import PATH_METHODS, _normalize_path_method

    assert "FSM-TS" in PATH_METHODS
    for name in ("FSM-TS", "fsm_ts", "fsmts", "FNEB-TS"):
        assert _normalize_path_method(name) == "FSM-TS"
    p = RunInputs(path_min_method="fsm-ts").path_min_inputs
    assert p.min_images == 3 and p.max_min_iter == 3 and p.max_grow_iter == 20 and p.pseudo_irc_amplitude == 0.3 and p.fallback == "GSM"


# ---- run --use-tsopt after FSM-TS (issue #22): its verified TS, not a new search from the string


def test_use_tsopt_takes_the_ts_fsm_ts_verified(monkeypatch):
    from mepd.cli import _verified_fsm_ts

    m, result, _ = _run(monkeypatch, [R, P])
    leaf = SimpleNamespace(index=3, data=m)
    assert _verified_fsm_ts(leaf) is m.ts_node
    m.ts_node = None   # a fallback leaf: no verified TS
    assert _verified_fsm_ts(leaf) is None


def test_use_tsopt_reads_the_verified_ts_of_a_tree_from_disk(tmp_path):
    import json

    from mepd.cli import _verified_fsm_ts
    from mepd.inputs import ChainInputs

    chain = Chain.model_validate({"nodes": [R.copy(), TS.copy(), P.copy()], "parameters": ChainInputs()})
    leaf = SimpleNamespace(index=2, data=SimpleNamespace(chain_trajectory=[chain]))
    for verdict, expected in (("elementary", True), ("split", False)):
        (tmp_path / "tree.json").write_text(json.dumps(
            {"version": 1, "nodes": [{"index": 2, "pseudo_irc": {"verdict": verdict}}]}))
        got = _verified_fsm_ts(leaf, tmp_path)
        assert (got is chain[1]) is expected
    (tmp_path / "tree.json").unlink()
    assert _verified_fsm_ts(leaf, tmp_path) is None   # no record (e.g. an NEB tree): search as before


def test_an_already_optimized_ts_is_not_searched_again(monkeypatch, tmp_path):
    from mepd.cli_common import _optimize_ts_and_irc

    def no_search(*a, **k):
        raise AssertionError("the TS was searched again")

    monkeypatch.setattr("mepd.compute.find_ts", no_search)
    ts = TS.copy()
    ts._cached_energy = -76.3
    ri = SimpleNamespace(engine=_Engine(), ts_inputs=None)
    out = _optimize_ts_and_irc(ts, ri, tmp_path, run_irc=False, label="ts_leaf_1", already_optimized=True)
    assert out.ts_node is ts and (tmp_path / "ts_leaf_1.xyz").exists()
