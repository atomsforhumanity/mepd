"""`mepd channels`: path searches are queued mechanism by mechanism, in the order the mechanisms were ranked."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qcdata.models.structure import Structure

import mepd.channel_search as cs
from mepd.atom_mapping import reorder_structure
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode

DATA = Path(__file__).parent / "data" / "mapping"


def _keto_enol_with_water():
    start = Structure.open(DATA / "keto_enol_1water_start.xyz")
    end = Structure.open(DATA / "keto_enol_1water_end.xyz")
    return start, reorder_structure(end, list(np.random.default_rng(0).permutation(len(end.symbols))))


def test_select_per_mechanism_numbers_slapmappers_mechanism_first_then_the_explored_ones_as_ranked():
    from mepd.atom_mapping import suggest_mechanism_candidates
    from mepd.atom_mapping_selection import select_per_mechanism

    start, end = _keto_enol_with_water()
    run_inputs = RunInputs()
    run_inputs.atom_mapping_inputs.explore_mechanisms = 4
    choices = select_per_mechanism(start, end, "snap", run_inputs)
    ranked = list(suggest_mechanism_candidates(start, end, explore=4))
    assert [c.key for c in sorted(choices, key=lambda c: c.order)] == ranked
    assert not ranked[0].startswith("explored:") and all(k.startswith("explored:") for k in ranked[1:])


def test_searches_are_queued_in_the_mechanisms_ranked_order_not_by_name(tmp_path, monkeypatch):
    s = Structure.open(DATA / "keto_enol_1water_start.xyz")
    structures = [StructureNode(structure=s) for _ in range(4)]      # 2 start + 2 end conformers
    ranked = ["break b | form b", "explored: break z | form z", "explored: break a | form a"]

    def choices(start, end, metric, run_inputs, max_variants_per_mechanism=None):
        return [SimpleNamespace(key=key, score=3.0 - n, n_variants=1, order=n,
                                winner=SimpleNamespace(end_structure=end.model_copy(), atom_map=[n]))
                for n, key in enumerate(ranked)]

    monkeypatch.setattr("mepd.atom_mapping_selection.select_per_mechanism", choices)
    run_inputs = SimpleNamespace(atom_mapping_inputs=SimpleNamespace(metric="endpoint-rmsd", n_candidates=4))
    _, cands, summary = cs._expand_pairs_by_mechanism(structures, [(0, 2), (0, 3), (1, 2), (1, 3)], True, run_inputs,
                                                      pairs_per_mechanism=2, workers=1, output=tmp_path)
    table = json.loads((tmp_path / "pair_mechanisms.json").read_text())
    assert [r["mechanism"] for r in table] == [k for k in ranked for _ in range(2)]
    assert cands == [(r["start_conformer"], int(r["pair"].rsplit("_", 1)[1])) for r in table]   # the queue's order
    assert summary["path_searches_per_mechanism"] == {k: 2 for k in ranked}                     # none dropped


def test_explore_molecule_needs_explore_mechanisms_and_a_molecule_of_the_start(tmp_path):
    import pytest

    from mepd.errors import InputsError

    with pytest.raises(InputsError, match="--explore-mechanisms"):
        cs.ChannelsOptions(explore_molecule="O").validate()
    start, end = _keto_enol_with_water()
    options = cs.ChannelsOptions(explore_mechanisms=3, explore_molecule="CCO")
    with pytest.raises(InputsError, match="no molecule of the start is CCO"):
        cs.run_channels(StructureNode(structure=start), StructureNode(structure=end), RunInputs(), options, tmp_path)
    run_inputs = RunInputs()
    cs.apply_options(run_inputs, cs.ChannelsOptions(explore_mechanisms=3, explore_molecule="O"))
    assert run_inputs.atom_mapping_inputs.explore_molecule == "O"
