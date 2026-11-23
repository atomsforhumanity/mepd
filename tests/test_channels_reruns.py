"""A rerun of `mepd channels` into the same folder (team47 audit, defects 5, 6, 7): a pair folder is reused
only for the two structures it was searched for, a label never names two searches, the classification is
written afresh, and a transition state left without its IRC gets it. No quantum chemistry: path searches,
transition-state optimizations and IRCs are replaced by hand-made structures.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata.models.structure import Structure

from mepd import channel_search, cli_channels, cli_common
from mepd.chain import Chain
from mepd.channel_search import _discover_channels, _ts_and_irc_once, read_channels
from mepd.cli_common import TsIrcResult, _load_structure_from_smiles_or_xyz
from mepd.inputs import ChainInputs, RunInputs
from mepd.nodes.node import StructureNode

KCAL = 627.5094740631


def _species(smiles: str, energy_kcal: float = 0.0) -> StructureNode:
    node = StructureNode(structure=_load_structure_from_smiles_or_xyz(smiles, None, None))
    node._cached_energy = energy_kcal / KCAL
    return node


def _chain(nodes) -> Chain:
    return Chain.model_validate({"nodes": list(nodes), "parameters": ChainInputs()})


def _discover(tmp_path, monkeypatch, start, end, steps: dict):
    """_discover_channels on `steps` ({label: (ts node, irc chain) or None}), with nothing computed."""
    tasks = [(label, start) for label in steps]
    monkeypatch.setattr(channel_search, "_collect_ts_guess_tasks", lambda *a, **k: tasks)

    def once(node, label, ts_dir, run_inputs, charge, multiplicity):
        found = steps[label]
        return None if found is None else TsIrcResult(ts_node=found[0], irc_chain=found[1])

    monkeypatch.setattr(channel_search, "_ts_and_irc_once", once)
    monkeypatch.setattr(channel_search, "_load_ts_and_irc_from_disk", lambda *a, **k: None)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return _discover_channels(tmp_path, start, end, RunInputs(), int(start.structure.charge), 1, workers=1)


# --- (a) a pair folder belongs to the two structures it was searched for --------------------------------


class _Searches:
    """Stands in for MSMEP: records each search started and writes a one-node tree of the pair."""

    def __init__(self, monkeypatch):
        self.started = started = []

        class FakeMSMEP:
            def __init__(self, inputs):
                pass

            def run_recursive_minimize(self, chain):
                started.append(chain)
                return SimpleNamespace(write_to_disk=lambda tree_dir: (
                    tree_dir.mkdir(parents=True, exist_ok=True), (tree_dir / "adj_matrix.txt").write_text("1\n"),
                    chain.write_to_disk(tree_dir / "node_0.xyz")))

        monkeypatch.setattr("mepd.msmep.MSMEP", FakeMSMEP)
        monkeypatch.setattr("mepd.interpolation.initial_chain", lambda chain, *a, **k: chain)

    def run(self, structures, folder, pairs=((0, 1),)):
        cli_common._run_msmep_pairs(structures, list(pairs), folder, RunInputs(), parallel=False,
                                    parallel_workers=None, workers=1)
        return len(self.started)


def test_a_stored_tree_of_other_structures_is_not_taken_as_this_pair(tmp_path, monkeypatch):
    """The audit's test: a folder from before the identity file, holding another pair's tree."""
    old_start, old_end = _species("CC=O"), _species("C=CO")
    tree = tmp_path / "pair_0_1" / "tree"
    tree.mkdir(parents=True)
    (tree / "adj_matrix.txt").write_text("1\n")
    _chain([old_start, old_end]).write_to_disk(tree / "node_0.xyz")

    searched = []

    class FakeMSMEP:
        def __init__(self, inputs):
            pass

        def run_recursive_minimize(self, chain):
            searched.append(chain)
            raise RuntimeError("stop here: the search was started, which is all this test asks")

    monkeypatch.setattr("mepd.msmep.MSMEP", FakeMSMEP)
    monkeypatch.setattr("mepd.interpolation.initial_chain", lambda chain, *a, **k: chain)
    new_start, new_end = _species("C1CO1"), _species("C=CO")      # another species altogether at index 0
    cli_common._run_msmep_pairs([new_start, new_end], [(0, 1)], tmp_path, RunInputs(), parallel=False,
                                parallel_workers=None, workers=1)
    assert len(searched) == 1


def test_a_finished_pair_is_skipped_on_a_rerun_with_the_same_structures(tmp_path, monkeypatch):
    searches = _Searches(monkeypatch)
    pair = [_species("CC=O"), _species("C=CO")]
    assert searches.run(pair, tmp_path / "pairs") == 1
    assert json.loads((tmp_path / "pairs" / "pair_0_1" / "pair_identity.json").read_text())["sha256"]
    # the same structures again, as a rerun reads them back from an xyz file (12 decimals): no new search
    reread = [StructureNode(structure=n.structure.model_copy(update={
        "geometry": np.round(np.asarray(n.structure.geometry), 10)})) for n in pair]
    assert searches.run(reread, tmp_path / "pairs") == 1
    assert not (tmp_path / "pairs-replaced").exists()


def test_a_folder_from_before_the_identity_file_is_still_resumed(tmp_path, monkeypatch):
    """Runs written before this fix have no pair_identity.json: their trees are reused when the stored root
    path's ends are the pair's structures (network completion, the explorer and `channels` resume that way)."""
    searches = _Searches(monkeypatch)
    pair = [_species("CC=O"), _species("C=CO")]
    searches.run(pair, tmp_path / "pairs")
    (tmp_path / "pairs" / "pair_0_1" / "pair_identity.json").unlink()
    assert searches.run(pair, tmp_path / "pairs") == 1            # skipped: same structures
    turned = [StructureNode(structure=n.structure.model_copy(update={
        "geometry": np.asarray(n.structure.geometry)[:, [1, 2, 0]] + 3.0})) for n in pair]
    assert searches.run(turned, tmp_path / "pairs") == 1          # rigidly moved: still the same pair


def test_a_folder_of_another_pair_is_moved_aside_with_its_transition_state_files(tmp_path, monkeypatch, caplog):
    searches = _Searches(monkeypatch)
    pairs_dir, ts_dir = tmp_path / "pairs", tmp_path / "ts"
    searches.run([_species("CC=O"), _species("C=CO")], pairs_dir)
    ts_dir.mkdir()
    for name in ("ts_pair_0_1_leaf_0.xyz", "ts_pair_0_1_leaf_0_irc.xyz", "ts_pair_0_10_leaf_0.xyz"):
        (ts_dir / name).write_text("kept as written\n")
    # the rerun's pools changed: index 0 is now another conformer (here: another geometry of the species)
    other = _species("CC=O")
    other = StructureNode(structure=other.structure.model_copy(update={
        "geometry": np.asarray(other.structure.geometry) * 1.05}))
    assert searches.run([other, _species("C=CO")], pairs_dir) == 2
    aside = list((tmp_path / "pairs-replaced").iterdir())
    assert len(aside) == 1 and aside[0].name.startswith("pair_0_1.")
    assert (aside[0] / "tree" / "adj_matrix.txt").exists()                      # nothing deleted
    assert sorted(p.name for p in (aside[0] / "ts").iterdir()) == ["ts_pair_0_1_leaf_0.xyz",
                                                                   "ts_pair_0_1_leaf_0_irc.xyz"]
    assert [p.name for p in ts_dir.iterdir()] == ["ts_pair_0_10_leaf_0.xyz"]   # another pair's file stays
    assert "holds a finished search of other structures" in caplog.text
    assert (pairs_dir / "pair_0_1" / "tree" / "adj_matrix.txt").exists()        # the new search's tree
    assert searches.run([other, _species("C=CO")], pairs_dir) == 2              # and it is resumed next time


def _expand(tmp_path, monkeypatch, n_end: int) -> dict:
    """_expand_pairs_by_mechanism for 1 start conformer and `n_end` end conformers; mechanism A needs a
    renumbered product, mechanism B keeps the input numbering. Returns {pair label: (start, end, mechanism)}."""
    s = _load_structure_from_smiles_or_xyz("CCO", None, None)
    structures = [StructureNode(structure=s.model_copy()) for _ in range(1 + n_end)]
    candidates = [(0, j) for j in range(1, 1 + n_end)]

    def choices(start, end, metric, run_inputs, max_variants_per_mechanism=None):
        return [SimpleNamespace(key="mech A", score=0.1, n_variants=1,
                                winner=SimpleNamespace(end_structure=end.model_copy(), atom_map=[1])),
                SimpleNamespace(key="mech B", score=0.2, n_variants=1,
                                winner=SimpleNamespace(end_structure=end, atom_map=None)),
                SimpleNamespace(key="mech C", score=0.3, n_variants=1,
                                winner=SimpleNamespace(end_structure=end, atom_map=None))]

    monkeypatch.setattr("mepd.atom_mapping_selection.select_per_mechanism", choices)
    run_inputs = SimpleNamespace(atom_mapping_inputs=SimpleNamespace(metric="endpoint-rmsd", n_candidates=4))
    cli_channels._expand_pairs_by_mechanism(structures, candidates, True, run_inputs, pairs_per_mechanism=0,
                                            workers=1, output=tmp_path)
    table = json.loads((tmp_path / "pair_mechanisms.json").read_text())
    out: dict = {}
    for r in table:
        if not r.get("not_in_this_run"):
            out.setdefault(r["pair"], set()).add((r["start_conformer"], r["end_structure"], r["mechanism"]))
    return out


def _clashes(first: dict, second: dict) -> dict:
    """Labels of both runs that name another (start, end) pair or, for a renumbered product, another search."""
    return {label: (first[label], second[label]) for label in first if label in second
            and first[label] != second[label]}


def test_a_label_never_names_two_different_searches_across_reruns(tmp_path, monkeypatch):
    """The audit's test (with a third mechanism, which shares the input numbering's label with B)."""
    first = _expand(tmp_path, monkeypatch, n_end=2)
    second = _expand(tmp_path, monkeypatch, n_end=4)
    assert not _clashes(first, second), f"labels re-used for another search: {_clashes(first, second)}"
    # mechanisms B and C keep the input's numbering and still share one search per pair
    assert first["pair_0_1"] == {(0, 1, "mech B"), (0, 1, "mech C")} == second["pair_0_1"]
    # and it stays so on later reruns, larger or smaller
    third = _expand(tmp_path, monkeypatch, n_end=6)
    fourth = _expand(tmp_path, monkeypatch, n_end=2)
    for earlier, later in ((first, third), (second, third), (first, fourth), (second, fourth), (third, fourth)):
        assert not _clashes(earlier, later)


def test_an_unchanged_rerun_keeps_every_label(tmp_path, monkeypatch):
    first = _expand(tmp_path, monkeypatch, n_end=3)
    assert _expand(tmp_path, monkeypatch, n_end=3) == first
    grown = _expand(tmp_path, monkeypatch, n_end=5)
    assert _expand(tmp_path, monkeypatch, n_end=5) == grown       # also after labels were retired


# --- (b) the classification is written afresh ------------------------------------------------------------


def test_a_rerun_leaves_no_stale_classification(tmp_path, monkeypatch):
    start, mid, end = _species("CC=O"), _species("C=CO", 5.0), _species("C1CO1", 20.0)
    ts1, ts2 = _species("CC=O", 60.0), _species("CC=O", 70.0)
    step_mid_end = (ts2, _chain([mid, ts2, end]))
    step_start_mid = (ts1, _chain([start, ts1, mid]))
    # first run: only mid <-> end is found: an off-target exit (nothing joins it to the start)
    _discover(tmp_path, monkeypatch, start, end, {"ts_pair_0_2_leaf_0": step_mid_end})
    first = read_channels(tmp_path, charge=0, multiplicity=1)
    assert (len(first.direct), len(first.multistep), len(first.offtarget)) == (0, 0, 1)
    (tmp_path / "ts" / "kept.txt").write_text("not a classification folder\n")
    # rerun with one more search: start <-> mid. Now start -> mid -> end is a two-step route.
    _discover(tmp_path, monkeypatch, start, end, {"ts_pair_0_2_leaf_0": step_mid_end,
                                                  "ts_pair_0_3_leaf_0": step_start_mid})
    second = read_channels(tmp_path, charge=0, multiplicity=1)
    assert (len(second.direct), len(second.multistep), len(second.offtarget)) == (0, 1, 0)
    assert (tmp_path / "ts" / "kept.txt").exists()


# --- (c) a transition state without its IRC --------------------------------------------------------------


def _stored_ts(tmp_path, monkeypatch, failure_note=None):
    """ts/<label>.xyz on disk without an IRC; returns (calls of the follow-up, result of _ts_and_irc_once)."""
    ts = _species("CC=O", 60.0)
    ts_dir = tmp_path / "ts"
    ts_dir.mkdir()
    _chain([ts]).write_to_disk(ts_dir / "ts_pair_0_1_leaf_0.xyz")
    if failure_note is not None:
        (ts_dir / "ts_pair_0_1_leaf_0_irc_failed.txt").write_text(failure_note)
    calls = []
    monkeypatch.setattr(channel_search, "_optimize_ts_and_irc",
                        lambda node, run_inputs, output, **kw: calls.append(kw) or TsIrcResult(
                            ts_node=ts, irc_chain=_chain([ts, ts])))
    return calls, _ts_and_irc_once(ts, "ts_pair_0_1_leaf_0", ts_dir, RunInputs(), 0, 1)


def test_a_transition_state_without_irc_gets_its_irc_on_a_rerun(tmp_path, monkeypatch):
    calls, result = _stored_ts(tmp_path, monkeypatch)
    assert calls and result.irc_chain is not None
    assert calls[0]["already_optimized"] and calls[0]["run_irc"]      # the stored TS is not optimized again


def test_an_irc_that_failed_under_these_numerics_is_not_run_again(tmp_path, monkeypatch):
    note = f"ConvergenceError: no\n{cli_common.IRC_FAILED_EPOCH_PREFIX}{cli_common._irc_epoch()}\n"
    calls, result = _stored_ts(tmp_path, monkeypatch, failure_note=note)
    assert not calls and result.irc_chain is None


@pytest.mark.parametrize("note", ["ConvergenceError: from a run before notes carried their numerics\n",
                                  "ConvergenceError: no\nirc numerics: energies=1,irc=0\n"])
def test_an_irc_that_failed_under_other_numerics_is_run_again(tmp_path, monkeypatch, note):
    calls, result = _stored_ts(tmp_path, monkeypatch, failure_note=note)
    assert calls and result.irc_chain is not None


def test_the_irc_failure_note_says_which_numerics_it_failed_under(tmp_path):
    from mepd.cli_common import _optimize_ts_and_irc

    ts = _species("CC=O", 60.0)

    class Engine:
        def compute_transition_state(self, node, keywords=None):
            return node

        def compute_irc_chain(self, ts_node, keywords=None):
            raise RuntimeError("no IRC today")

        def compute_energies(self, nodes):
            return [n.energy for n in nodes]

    result = _optimize_ts_and_irc(ts, SimpleNamespace(engine=Engine(), ts_inputs=None), tmp_path, run_irc=True,
                                  label="ts_pair_0_1_leaf_0", already_optimized=True)
    assert result.irc_chain is None
    lines = (tmp_path / "ts_pair_0_1_leaf_0_irc_failed.txt").read_text().splitlines()
    assert "no IRC today" in lines[0]                                   # the first line stays the reason
    assert lines[1] == cli_common.IRC_FAILED_EPOCH_PREFIX + cli_common._irc_epoch()
    assert cli_common._irc_failure_is_current(tmp_path / "ts_pair_0_1_leaf_0_irc_failed.txt")
