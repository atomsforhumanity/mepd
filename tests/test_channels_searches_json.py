"""`mepd channels`: <output>/searches.json says, while the run goes, what each path search has found."""
import json
from pathlib import Path

import numpy as np
from qcdata.models.structure import Structure

import mepd.channel_search as cs
from mepd.atom_mapping import realign_end_to_start, reorder_structure, suggest_atom_mapping
from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode

DATA = Path(__file__).parent / "data" / "mapping"


def _keto_enol_with_water():
    start = Structure.open(DATA / "keto_enol_1water_start.xyz")
    end = Structure.open(DATA / "keto_enol_1water_end.xyz")
    return start, reorder_structure(end, list(np.random.default_rng(0).permutation(len(end.symbols))))


def test_searches_json_reports_each_transition_state_as_soon_as_its_irc_is_written(tmp_path, monkeypatch):
    start, end = _keto_enol_with_water()
    mapped = realign_end_to_start(suggest_atom_mapping(start, end), end)
    from mepd.atom_mapping import catalytic_participants, explore_mechanisms, mechanism_key

    relay = next(x for x in (realign_end_to_start(m, end)
                             for m in explore_mechanisms(start, end, [suggest_atom_mapping(start, end)]))
                 if catalytic_participants(start, x) == ["H2O"])
    structures = [StructureNode(structure=x) for x in (start, mapped, relay)]
    key = mechanism_key(start, relay)
    (tmp_path / "pair_mechanisms.json").write_text(json.dumps([
        {"pair": "pair_0_1", "mechanism": mechanism_key(start, mapped)},
        {"pair": "pair_0_2", "mechanism": f"explored: {key} [catalytic: H2O]"}]))
    monkeypatch.setattr(cs, "_pair_ts_guess_tasks", lambda pair_dir, q, m: [(f"ts_{pair_dir.name}_leaf_0", None)])
    table = cs._SearchTable(tmp_path, structures, 1, [(0, 1), (0, 2)], structures[0], structures[1], 0, 1)
    table.write()
    doc = json.loads((tmp_path / "searches.json").read_text())
    assert [s["status"] for s in doc["searches"]] == ["queued", "queued"] and not doc["finished"]
    assert sorted(len(m["atoms"]) for m in doc["start_molecules"]) == [3, 10]
    water = next(m for m in doc["start_molecules"] if len(m["atoms"]) == 3)
    assert doc["searches"][1]["molecules"] == [
        {**m, "returned": m == water} for m in doc["start_molecules"]]   # the mapping passes an H through the water

    # the relay's search finishes and its TS gets an IRC that ends in the relay's two structures
    (tmp_path / "pairs" / "pair_0_1").mkdir(parents=True)
    tree = tmp_path / "pairs" / "pair_0_2" / "tree"
    tree.mkdir(parents=True)
    (tree / "adj_matrix.txt").write_text("0\n")
    ts_dir = tmp_path / "ts"
    ts_dir.mkdir()
    Chain.model_validate({"nodes": [structures[0]], "parameters": ChainInputs()}).write_to_disk(
        ts_dir / "ts_pair_0_2_leaf_0.xyz")
    table.write()
    doc = json.loads((tmp_path / "searches.json").read_text())
    # a tree whose search has not been reported finished may still be being written: not read yet
    assert [s["status"] for s in doc["searches"]] == ["running", "running"]
    table.done("pair", (0, 2))
    doc = json.loads((tmp_path / "searches.json").read_text())
    assert [s["status"] for s in doc["searches"]] == ["running", "searched"]
    assert doc["searches"][1]["transition_states"][0]["status"] == "ts_pending"     # its follow-up not done
    Chain.model_validate({"nodes": [structures[2], structures[0]], "parameters": ChainInputs()}).write_to_disk(
        ts_dir / "ts_pair_0_2_leaf_0_irc.xyz")           # the IRC as it ran: from the end's side
    table.write()
    assert json.loads((tmp_path / "searches.json").read_text())["searches"][1]["transition_states"][0]["status"] \
        == "ts_pending"                                  # (an IRC file is read once its follow-up has finished)
    table.done("task", ("ts_pair_0_2_leaf_0", None))
    assert json.loads((tmp_path / "searches.json").read_text())["searches"][1]["transition_states"][0]["status"] \
        == "verified"
    table.write(finished=True)
    doc = json.loads((tmp_path / "searches.json").read_text())
    ts = doc["searches"][1]["transition_states"][0]
    assert doc["finished"] and doc["searches"][0]["status"] == "failed"
    assert ts["status"] == "verified" and ts["joins_start_and_end"] and ts["is_searched_mechanism"]
    assert ts["mechanism_found"] == key and ts["irc"] == "ts/ts_pair_0_2_leaf_0_irc.xyz"
    assert [m for m in ts["molecules"] if m["returned"]] == [{**water, "returned": True}]
    assert len(ts["bonds_broken"]) == 2 and len(ts["bonds_formed"]) == 2
    assert sum(a in water["atoms"] or b in water["atoms"] for a, b in ts["bonds_broken"] + ts["bonds_formed"]) == 3


def test_follow_up_done_is_called_after_each_pair_and_task(tmp_path):
    from mepd.cli_common import PairFollowUp, _fork_pairs_with_follow_up

    calls = []
    follow = PairFollowUp(tasks=lambda d: [d.name], run=lambda t: None, done=lambda kind, item: calls.append((kind, item)))
    _fork_pairs_with_follow_up(lambda pair: None, [(0, 1), (0, 2), (0, 3)], tmp_path, 2, follow)
    assert sorted(calls) == [("pair", (0, 1)), ("pair", (0, 2)), ("pair", (0, 3)),
                             ("task", "pair_0_1"), ("task", "pair_0_2"), ("task", "pair_0_3")]


def test_a_tree_being_written_is_not_read_and_not_remembered(tmp_path, monkeypatch):
    """The table was written while a worker had written a tree's adj_matrix.txt and not yet its nodes: it read
    no TS guesses and kept that empty answer for good."""
    from types import SimpleNamespace as S

    out = tmp_path
    (out / "pairs" / "pair_0_1" / "tree").mkdir(parents=True)
    t = cs._SearchTable(out, [S(energy=-1.0, structure=None), S(energy=-0.9, structure=None)], 1, [(0, 1)],
                        S(structure=None), S(structure=None), 0, 1)
    (out / "pairs" / "pair_0_1" / "tree" / "adj_matrix.txt").write_text("0\n")
    calls = []
    monkeypatch.setattr(cs, "_pair_ts_guess_tasks", lambda pair_dir, q, m: calls.append(1) or [("ts_pair_0_1_leaf_1", None)])
    t.write()
    assert calls == [] and json.loads((out / "searches.json").read_text())["searches"][0]["status"] == "running"
    t.done("pair", (0, 1))
    d = json.loads((out / "searches.json").read_text())
    assert calls == [1] and [x["label"] for x in d["searches"][0]["transition_states"]] == ["ts_pair_0_1_leaf_1"]


def test_follow_ups_take_freed_lanes_before_queued_searches_and_no_more_processes_work_than_workers(tmp_path):
    """`--workers 2`: two processes work at any time (searches and their TS/IRC follow-ups together), and a
    finished search's follow-up starts before the next queued search."""
    import time

    from mepd.cli_common import PairFollowUp, _fork_pairs_with_follow_up

    def span(name, seconds):
        (tmp_path / f"{name}.start").write_text(str(time.time()))
        time.sleep(seconds)
        (tmp_path / f"{name}.end").write_text(str(time.time()))

    follow = PairFollowUp(tasks=lambda d: [d.name], run=lambda name: span(f"task_{name}", 0.4), during=2, after=2,
                          share_lanes=True)
    _fork_pairs_with_follow_up(lambda pair: span(f"pair_{pair[0]}_{pair[1]}", 0.4), [(0, k) for k in range(5)],
                               tmp_path, 2, follow)
    spans = {fp.stem: (float(fp.read_text()), float(fp.with_suffix(".end").read_text()))
             for fp in tmp_path.glob("*.start")}
    assert len(spans) == 10
    assert max(sum(a <= t < b for a, b in spans.values()) for t, _ in spans.values()) == 2
    assert spans["task_pair_0_0"][0] < spans["pair_0_2"][0] and spans["task_pair_0_1"][0] < spans["pair_0_3"][0]


def test_without_shared_lanes_follow_ups_run_beside_the_searches_as_on_develop(tmp_path):
    """Plain `mepd channels` (no molecule asked for): up to `during` follow-ups beside the `workers` searches."""
    import time

    from mepd.cli_common import PairFollowUp, _fork_pairs_with_follow_up

    def span(name, seconds):
        (tmp_path / f"{name}.start").write_text(str(time.time()))
        time.sleep(seconds)
        (tmp_path / f"{name}.end").write_text(str(time.time()))

    follow = PairFollowUp(tasks=lambda d: [d.name], run=lambda name: span(f"task_{name}", 0.6), during=2, after=2)
    _fork_pairs_with_follow_up(lambda pair: span(f"pair_{pair[0]}_{pair[1]}", 0.4), [(0, k) for k in range(5)],
                               tmp_path, 2, follow)
    spans = {fp.stem: (float(fp.read_text()), float(fp.with_suffix(".end").read_text()))
             for fp in tmp_path.glob("*.start")}
    assert len(spans) == 10 and max(sum(a <= t < b for a, b in spans.values()) for t, _ in spans.values()) > 2
