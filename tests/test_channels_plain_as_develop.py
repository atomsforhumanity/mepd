"""Plain `mepd channels` (no molecule asked for) keeps develop's order of mechanisms and its pair folder names;
with --explore-molecule the mechanisms are queued in the order they were ranked (team41 decision 23a)."""
import json
import types

import numpy as np
from qcdata import Structure

import mepd.atom_mapping_selection as sel
from mepd.channel_search import _expand_pairs_by_mechanism
from mepd.nodes.node import StructureNode


def _h2(x):
    return Structure(symbols=["H", "H"], geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]), charge=0, multiplicity=1)


def _choices(*_a, **_k):
    # the ranking puts "z-first" first (SLAPMapper's own), "a-second" after it (an explored one)
    return [types.SimpleNamespace(key=k, score=s, n_variants=1, order=o,
                                  winner=types.SimpleNamespace(end_structure=_h2(x), atom_map=object()))
            for k, s, o, x in (("a-second", 2.0, 1, 1.5), ("z-first", 1.0, 0, 1.6))]


def _run(tmp_path, molecule, monkeypatch):
    monkeypatch.setattr(sel, "select_per_mechanism", _choices)
    ri = types.SimpleNamespace(atom_mapping_inputs=types.SimpleNamespace(metric="snap", explore_molecule=molecule))
    structures = [StructureNode(structure=_h2(1.4)), StructureNode(structure=_h2(1.45))]
    out = tmp_path / ("with" if molecule else "plain")
    out.mkdir()
    _, cands, summary = _expand_pairs_by_mechanism(structures, [(0, 1)], True, ri, pairs_per_mechanism=1,
                                                   workers=1, output=out)
    table = json.loads((out / "pair_mechanisms.json").read_text())
    return cands, {r["mechanism"]: r["pair"] for r in table}, list(summary["path_searches_per_mechanism"])


def test_plain_channels_order_and_folder_names_are_develops(tmp_path, monkeypatch):
    cands, pairs, keys = _run(tmp_path, None, monkeypatch)
    assert keys == ["a-second", "z-first"]                       # by name
    assert pairs == {"a-second": "pair_0_2", "z-first": "pair_0_3"} and cands == [(0, 2), (0, 3)]


def test_with_a_molecule_asked_for_the_ranked_order_is_used(tmp_path, monkeypatch):
    cands, pairs, keys = _run(tmp_path, "O", monkeypatch)
    assert keys == ["z-first", "a-second"]
    assert pairs == {"z-first": "pair_0_2", "a-second": "pair_0_3"} and cands == [(0, 2), (0, 3)]
