"""MINMAX (`mepd channels --method minmax`, mepd.minmax): one path search per
elementary step, on the built pair with the lowest GI + xtb peak; a split's
pieces each get a pair chosen the same way before they are searched."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("rdkit")


def _embedded(smiles: str, seed: int = 1):
    from qcref.constants import ANGSTROM_TO_BOHR
    from qcdata import Structure
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return Structure(symbols=[a.GetSymbol() for a in m.GetAtoms()],
                     geometry=m.GetConformer().GetPositions() * ANGSTROM_TO_BOHR, charge=0, multiplicity=1)


def _node(smiles: str, seed: int = 1):
    from mepd.nodes.node import StructureNode

    return StructureNode(structure=_embedded(smiles, seed))


def _search(tmp_path, monkeypatch, peaks: dict):
    """A MinmaxSearch whose pools are the input geometry alone and whose
    built pairs are stubbed: for each side, one pair per entry of
    `peaks[side]` (its GI + xtb peak), every pair being the two minima as given."""
    import mepd.channel_search as cs
    import mepd.conformers as conformers
    from mepd.built_pair_mapping import MechanismPair
    from mepd.channel_search import ChannelsOptions
    from mepd.inputs import RunInputs
    from mepd.minmax import MinmaxSearch

    monkeypatch.setattr(conformers, "generate_conformers", lambda node, inputs, stats: [node])
    calls = []

    def built(pool, side, a, b, run_inputs, workers=1):
        calls.append((side, len(pool)))
        return ([(0, MechanismPair(key=f"mech {n}", score=p, start=a.structure, end=b.structure, n_variants=1,
                                   floor=-200.0)) for n, p in enumerate(peaks[side])], len(peaks[side]))

    monkeypatch.setattr(cs, "_built_pairs_by_mechanism", built)
    options = ChannelsOptions(method="minmax", minimize_ends=False)
    return MinmaxSearch(RunInputs(), options, tmp_path, None, workers=1), calls


def test_only_the_pair_with_the_lowest_peak_is_chosen(tmp_path, monkeypatch):
    search, calls = _search(tmp_path, monkeypatch, {"start": [-150.0, -170.0], "end": [-160.0]})
    a, b = _node("CC=O"), _node("C=CO")
    start, end = search.choose(a, b, 0, 0)
    assert calls == [("start", 1), ("end", 1)]           # partners built from both sides' conformers
    rec = search.records[0]
    assert rec["n_built_pairs"] == 3 and rec["chosen"]["built_from"] == "start"
    assert rec["chosen"]["peak_kcal"] == -170.0 and rec["chosen"]["runner_up_peak_kcal"] == -160.0
    saved = json.loads((tmp_path / "minmax" / "n0" / "built_pairs" / "built_pairs.json").read_text())
    assert [e["score"] for e in saved["entries"]] == [-170.0, -160.0, -150.0]
    assert [e["searched"] for e in saved["entries"]] == [True, False, False]
    assert (tmp_path / "minmax" / "n0" / "conformers" / "start_pool.xyz").exists()


def test_a_species_pool_is_sampled_once(tmp_path, monkeypatch):
    search, _ = _search(tmp_path, monkeypatch, {"start": [-150.0], "end": [-160.0]})
    a, b = _node("CC=O"), _node("C=CO")
    search.choose(a, b, 0, 0)
    # The same species in the same numbering, as either end and in another geometry: its pool is reused.
    search.choose(a, _node("CC=O", seed=3), 1, 1)
    assert search.records[1]["conformers"]["start"].get("reused") is True
    assert search.records[1]["conformers"]["end"].get("reused") is True
    # Another species is sampled.
    search.choose(a, _node("CCO"), 2, 1)
    assert not search.records[2]["conformers"]["end"].get("reused")


def test_nothing_built_searches_the_minima_as_given(tmp_path, monkeypatch):
    search, _ = _search(tmp_path, monkeypatch, {"start": [], "end": []})
    a, b = _node("CC=O"), _node("C=CO")
    assert search.choose(a, b, 0, 0) == (a, b)
    assert search.records[0]["chosen"] is None and "as given" in search.records[0]["note"]


def test_a_split_piece_is_searched_from_its_own_best_pair(tmp_path, monkeypatch):
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs

    search, calls = _search(tmp_path, monkeypatch, {"start": [-150.0], "end": [-140.0]})
    seen = []
    monkeypatch.setattr(search, "chain", lambda s, e: seen.append((s, e)) or Chain.model_validate(
        {"nodes": [s, e], "parameters": ChainInputs()}))
    piece = Chain.model_validate({"nodes": [_node("CC=O"), _node("C=CO")], "parameters": ChainInputs()})
    piece._split_ancestors = [(_node("CCO"), _node("C=O.C"))]
    new = search.piece_chain(piece, 1, 4)
    assert new is not piece and len(seen) == 1 and search.records[4]["depth"] == 1
    assert len(new._split_ancestors) == 1          # MSMEP's cycle limits follow the piece


def test_conformer_changes_and_cycles_are_left_to_msmep(tmp_path, monkeypatch):
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs

    search, calls = _search(tmp_path, monkeypatch, {"start": [-150.0], "end": [-140.0]})
    same = Chain.model_validate({"nodes": [_node("CCO", 1), _node("CCO", 5)], "parameters": ChainInputs()})
    assert search.piece_chain(same, 1, 1) is same and "conformer change" in search.records[1]["note"]
    a, b = _node("CC=O"), _node("C=CO")
    loop = Chain.model_validate({"nodes": [a, b], "parameters": ChainInputs()})
    loop._split_ancestors = [(_node("C=CO", 2), _node("CC=O", 2))]      # the same two species, higher up
    assert search.piece_chain(loop, 2, 2) is loop and "cycle" in search.records[2]["note"]
    assert calls == []


def test_msmep_hands_every_split_piece_to_piece_chain(monkeypatch):
    """The hook MINMAX uses: each piece of a split goes through _piece_chain
    (depth, tree index) before it is searched; the base MSMEP keeps it."""
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs, RunInputs
    from mepd.msmep import MSMEP
    from mepd.TreeNode import TreeNode

    pieces = [Chain.model_validate({"nodes": [_node("CC=O"), _node("C=CO")], "parameters": ChainInputs()})
              for _ in range(2)]

    def step(self, chain, index, tree_depth=0, max_depth=None):
        node = TreeNode(data=None, children=[], index=index)
        return node, (pieces if index == 0 else [])

    monkeypatch.setattr(MSMEP, "_guarded_step", step)
    got = []

    class Recording(MSMEP):
        def _piece_chain(self, chain_frag, tree_depth, tree_node_index):
            got.append((tree_depth, tree_node_index, chain_frag is pieces[len(got)]))
            return super()._piece_chain(chain_frag, tree_depth, tree_node_index)

    tree = Recording(inputs=RunInputs()).run_recursive_minimize(pieces[0])
    assert got == [(1, 1, True), (1, 2, True)] and [c.index for c in tree.children] == [1, 2]


def test_minmax_needs_atom_mapping_and_takes_no_built_pair_list():
    from mepd.channel_search import ChannelsOptions
    from mepd.errors import InputsError

    ChannelsOptions(method="minmax").validate()
    with pytest.raises(InputsError, match="needs"):
        ChannelsOptions(method="minmax", atom_mapping=False).validate()
    with pytest.raises(InputsError, match="not used"):
        ChannelsOptions(method="minmax", built_pairs="x", pair_from="start+end").validate()
    with pytest.raises(InputsError, match="Known"):
        ChannelsOptions(method="other").validate()


def test_the_web_minmax_method_runs_channels_with_method_minmax():
    from mepd.web.operations import OPERATIONS, ChannelsParams, generic_flags, is_minmax

    op = OPERATIONS["channels"]
    labels = [m["label"] for m in op.methods]
    assert "MINMAX" in labels
    assert next(m for m in op.methods if m["label"] == "MINMAX")["fixed"] == {"search": "minmax"}
    flags = generic_flags(ChannelsParams(search="minmax"))
    assert "--minmax-explore" in flags and flags[flags.index("--minmax-explore") + 1] == "5"
    for gone in ("--pair-from", "--pairs-per-mechanism", "--atom-mapping-metric", "--explore-mechanisms",
                 "--max-pairs", "--relax-mechanisms"):
        assert gone not in flags
    assert "--minmax-explore" not in generic_flags(ChannelsParams())
    assert is_minmax({"argv": ["channels", "--method", "minmax"]}) and not is_minmax({"argv": ["channels"]})
