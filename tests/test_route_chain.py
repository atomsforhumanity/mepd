"""A multi-step route reads as one chain: start -> intermediates -> end, each
step turned to run from the species the step before it ended on."""
from types import SimpleNamespace

from mepd.web import results


def _patch(monkeypatch):
    # Nodes: species name + energy (Hartree); connectivity = same species name.
    monkeypatch.setattr(results, "_node_energy", lambda n: n.e)
    monkeypatch.setattr(results, "_same_connectivity", lambda a, b: a.sp == b.sp)
    monkeypatch.setattr(results, "_smiles", lambda n: n.sp)


def node(sp, e):
    return SimpleNamespace(sp=sp, e=e)


def test_steps_chain_through_intermediates_whatever_way_each_irc_was_stored(monkeypatch):
    _patch(monkeypatch)
    h = 1 / results.HARTREE_TO_KCAL          # 1 kcal/mol in Hartree
    start, end = node("A", 0.0), node("C", -5 * h)
    step1 = [node("A", 0.0), node("TS1", 20 * h), node("B", 10 * h)]
    step2 = [node("C", -5 * h), node("TS2", 30 * h), node("B", 11 * h)]   # stored end -> intermediate
    chain = results._route_chain([("s1", "TS 1", step1), ("s2", "TS 2", step2)], start, end, floor=0.0)
    assert [(c["from"], c["to"]) for c in chain] == [("start", "B"), ("B", "end")]
    assert chain[1]["from_kcal"] == 11 and chain[1]["to_kcal"] == -5    # turned around
    assert chain[1]["ts_kcal"] == 30 and chain[1]["barrier_kcal"] == 19   # from the species before it


def test_a_step_without_both_ends_gives_no_chain(monkeypatch):
    _patch(monkeypatch)
    start, end = node("A", 0.0), node("C", 0.0)
    assert results._route_chain([("s1", "TS 1", [node("TS", 1.0)])], start, end, floor=0.0) == []
