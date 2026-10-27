"""Network-aware MSMEP (mepd/network_aware.py): pieces of a split whose
route from the start must climb more than the barrier cutoff are not
searched, and elementary steps above it get no TS search."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

from mepd.chain import Chain
from mepd.elementarystep import ElemStepResults
from mepd.engines.engine import Engine
from mepd.inputs import ChainInputs
from mepd.network_aware import RouteState, eyring_cutoff_kcal, resolve_cutoff
from mepd.nodes.node import StructureNode
from mepd.units import HARTREE_TO_KCAL
from tests.test_msmep_same_pair_split_limit import _msmep


def _species(x: float) -> Structure:
    # A distinct species per x (the third atom far away: a different graph).
    return Structure(geometry=np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [x, 0.0, 0.0]]),
                     symbols=["C", "O", "H"], charge=0, multiplicity=1)


class _EnergyByX(Engine):
    """Energy (kcal/mol, stored in Hartree) looked up by the third atom's x."""

    def __init__(self, table):
        self.table = table

    def _e(self, node):
        x = round(float(node.structure.geometry[2][0]), 3)
        return self.table[x] / HARTREE_TO_KCAL

    def compute_energies(self, chain):
        out = []
        for n in chain:
            n._cached_energy = self._e(n)
            out.append(n._cached_energy)
        return out

    def compute_gradients(self, chain):
        self.compute_energies(chain)
        return [np.zeros_like(n.coords) for n in chain]


def _chain(*xs) -> Chain:
    return Chain.model_validate({"nodes": [StructureNode(structure=_species(x)) for x in xs],
                                 "parameters": ChainInputs()})


def _result(elem: bool) -> ElemStepResults:
    return ElemStepResults(is_elem_step=elem, is_concave=True, splitting_criterion="minima",
                           minimization_results=[], number_grad_calls=0, new_structures=[])


def _ts_at(monkeypatch, table, x=None, fail=False):
    """mepd.compute.find_ts stand-in: the confirmed TS is the species at `x`."""
    import mepd.compute

    def find_ts(guess, engine, **kw):
        if fail:
            raise RuntimeError("no saddle")
        node = StructureNode(structure=_species(x))
        node._cached_energy = table[x] / HARTREE_TO_KCAL
        return node

    monkeypatch.setattr(mepd.compute, "find_ts", find_ts)


def _run(monkeypatch, table, pieces, cutoff=30.0, leaf_peak=None):
    """Root A -> B splits once into `pieces` (lists of x); every piece is then
    elementary, its path peaking at `leaf_peak` (an x in `table`) if given."""
    msmep = _msmep()
    msmep.inputs.engine = _EnergyByX(table)
    msmep.inputs.path_min_inputs.barrier_cutoff_kcal = cutoff
    msmep.inputs.path_min_inputs.skip_identical_graphs = False   # toy species: told apart by geometry
    searched = []

    def minimize(input_chain):
        searched.append([round(float(n.structure.geometry[2][0]), 3) for n in (input_chain[0], input_chain[-1])])
        if len(searched) == 1:
            return SimpleNamespace(chain_trajectory=[input_chain]), _result(False)
        xs = [float(input_chain[0].structure.geometry[2][0]), float(input_chain[-1].structure.geometry[2][0])]
        path = _chain(xs[0], leaf_peak, xs[1]) if leaf_peak is not None else _chain(*xs)
        msmep.inputs.engine.compute_energies(path)
        return SimpleNamespace(chain_trajectory=[path]), _result(True)

    monkeypatch.setattr(msmep, "run_minimize_chain", minimize)
    monkeypatch.setattr(msmep, "make_sequence_of_chains",
                        lambda chain, split_method, minimization_results: [_chain(*p) for p in pieces])
    tree = msmep.run_recursive_minimize(_chain(2.0, 6.0), max_depth=1)
    return tree, searched


def test_cutoff_from_temperature_and_time():
    assert eyring_cutoff_kcal(333.15, 86400) == pytest.approx(27.3, abs=0.1)
    assert eyring_cutoff_kcal(373.15, 86400) > eyring_cutoff_kcal(333.15, 86400) > eyring_cutoff_kcal(298.15, 86400)
    pmi = SimpleNamespace(cutoff_temperature=333.15, cutoff_time_s=86400, cutoff_margin_kcal=10.0)
    assert resolve_cutoff(pmi) == pytest.approx(37.3, abs=0.1)
    assert resolve_cutoff(SimpleNamespace(barrier_cutoff_kcal=25.0, cutoff_temperature=333.15)) == 25.0
    assert resolve_cutoff(SimpleNamespace()) is None
    assert resolve_cutoff(SimpleNamespace(barrier_cutoff_kcal=0)) is None


def test_route_span_is_measured_from_the_deepest_preceding_well():
    s = RouteState.at(0.0).through(-20.0)       # fall into a well
    s = s.over(5.0).through(-10.0)              # then a barrier 25 above that well
    assert s.well == -20.0 and s.span == pytest.approx(25.0)
    assert RouteState.at(12.0).span == 12.0     # a start above the floor was climbed to


def test_a_piece_heading_high_is_searched_but_a_piece_starting_out_of_reach_is_not(monkeypatch, tmp_path):
    """A piece from a reachable start is searched even when it ends high (it can find
    intermediates on its way); a piece starting above floor + cutoff is not."""
    monkeypatch.setenv("MEPD_NETWORK_AWARE_LOG", str(tmp_path / "na.jsonl"))
    table = {2.0: 0.0, 6.0: -5.0, 4.0: 50.0}   # A=0, B=-5, intermediate C at +50
    tree, searched = _run(monkeypatch, table, [[2.0, 4.0], [4.0, 6.0]])
    assert searched == [[2.0, 6.0], [2.0, 4.0]]          # A -> C searched, C -> B not
    first, second = tree.children
    assert first.leaf_status == "above_cutoff" and first.skip_ts
    assert second.leaf_status == "pruned_above_cutoff" and second.data is None
    events = [json.loads(l) for l in (tmp_path / "na.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["split", "leaf", "pruned"]
    assert events[2]["span_kcal"] == pytest.approx(50.0)


def test_a_piece_from_an_unreachable_start_is_not_searched(monkeypatch):
    table = {2.0: 0.0, 6.0: -5.0, 4.0: 50.0, 5.0: 10.0}
    tree, searched = _run(monkeypatch, table, [[2.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    assert searched == [[2.0, 6.0], [2.0, 4.0], [5.0, 6.0]]     # only the piece starting at +50 is skipped
    assert tree.children[1].leaf_status == "pruned_above_cutoff"


def test_a_low_route_is_searched_and_its_steps_keep_their_ts(monkeypatch):
    table = {2.0: 0.0, 6.0: -5.0, 4.0: 10.0, 3.0: 20.0}
    tree, searched = _run(monkeypatch, table, [[2.0, 4.0], [4.0, 6.0]], leaf_peak=3.0)
    assert searched == [[2.0, 6.0], [2.0, 4.0], [4.0, 6.0]]
    assert all(not getattr(c, "skip_ts", False) for c in tree.children)
    assert tree.children[-1].na_route["span_kcal"] == pytest.approx(20.0)


def test_a_step_above_the_cutoff_is_kept_without_a_ts_search_and_does_not_block_the_next_piece(monkeypatch):
    table = {2.0: 0.0, 6.0: -5.0, 4.0: 10.0, 3.0: 45.0, 3.5: 41.0}   # path peak +45, confirmed TS +41
    _ts_at(monkeypatch, table, 3.5)
    tree, searched = _run(monkeypatch, table, [[2.0, 4.0], [4.0, 6.0]], leaf_peak=3.0)
    first, second = tree.children
    assert first.leaf_status == "above_cutoff" and first.skip_ts and first.data is not None
    # the next piece starts at +10, within reach by some other route: searched (here its own
    # step peaks at the same +41 TS, so it is above the cutoff too)
    assert searched == [[2.0, 6.0], [2.0, 4.0], [4.0, 6.0]] and second.leaf_status == "above_cutoff"


def test_a_climb_out_of_this_paths_deep_well_does_not_mark_a_step(monkeypatch):
    """Another route may reach the same TS (+5 above the floor) from a shallower well:
    only the TS's own height above the floor can make a step irrelevant network-wide."""
    table = {2.0: 0.0, 6.0: -5.0, 4.0: -30.0, 3.0: 5.0}   # well at -30, next step peaks at +5
    _ts_at(monkeypatch, table, 3.0)
    tree, searched = _run(monkeypatch, table, [[2.0, 4.0], [4.0, 6.0]], leaf_peak=3.0)
    second = tree.children[1]
    assert not getattr(second, "skip_ts", False)
    assert second.na_route["span_kcal"] == pytest.approx(35.0)   # recorded, not used to prune


def test_without_a_cutoff_nothing_changes(monkeypatch):
    table = {2.0: 0.0, 6.0: -5.0, 4.0: 50.0}
    tree, searched = _run(monkeypatch, table, [[2.0, 4.0], [4.0, 6.0]], cutoff=0)
    assert len(searched) == 3
    assert all(getattr(c, "leaf_status", None) is None for c in tree.children)


def test_a_path_peak_above_the_cutoff_whose_ts_is_lower_is_not_cut(monkeypatch):
    """The early-stopped path's highest image overestimates the barrier: the
    optimized TS decides."""
    table = {2.0: 0.0, 6.0: -5.0, 4.0: 10.0, 3.0: 70.0, 3.5: 20.0}   # path peak +70, real TS +20
    _ts_at(monkeypatch, table, 3.5)
    tree, searched = _run(monkeypatch, table, [[2.0, 4.0], [4.0, 6.0]], leaf_peak=3.0)
    assert searched == [[2.0, 6.0], [2.0, 4.0], [4.0, 6.0]]
    assert all(not getattr(c, "skip_ts", False) for c in tree.children)


def test_an_unconfirmed_high_peak_cuts_nothing(monkeypatch):
    table = {2.0: 0.0, 6.0: -5.0, 4.0: 10.0, 3.0: 70.0}
    _ts_at(monkeypatch, table, fail=True)
    tree, searched = _run(monkeypatch, table, [[2.0, 4.0], [4.0, 6.0]], leaf_peak=3.0)
    assert searched == [[2.0, 6.0], [2.0, 4.0], [4.0, 6.0]]
    assert all(not getattr(c, "skip_ts", False) for c in tree.children)
