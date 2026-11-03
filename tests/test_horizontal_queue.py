"""The connect queue: steps reach the caller as they are found; pieces below a split run only when
the caller's kinetics put material at one of their ends; unreachable ones are iced, and a later call
thaws them once material reaches them."""
import threading
from pathlib import Path
from types import SimpleNamespace

from mepd.discovery.horizontal import HorizontalConnector


class Node(SimpleNamespace):
    pass


def _connector(tmp_path):
    hc = object.__new__(HorizontalConnector)
    hc.ts_dir, hc.workers, hc.max_pieces, hc.max_depth, hc.nimages = Path(tmp_path), 2, 40, 6, 11
    hc.say = lambda *a: None
    hc.job_inputs = SimpleNamespace(path_min_inputs=SimpleNamespace(barrier_cutoff_kcal=None, cutoff_temperature=None,
                                                                     network_floor_hartree=None))
    hc._lock = threading.Lock()
    hc._running = SimpleNamespace(value=0)
    hc._fast_ts = lambda i, j, a, b: None                  # every single TS fails: path searches
    hc._root_chain = lambda a, b: [a, b]
    hc._leaf_ts = lambda chain, label: SimpleNamespace(energy=-1.0)
    hc.iced, hc._pieces, hc._next_index, hc.free, hc._jobs, hc._live = [], {}, {}, None, {}, None
    return hc


def test_pieces_wait_for_material_and_run_when_it_comes(tmp_path):
    A, B, I, J = (Node(name=n) for n in "ABIJ")
    hc = _connector(tmp_path)

    def piece(chain, index, depth):          # A-B splits at I: A-I (leaf), I-J-B: I-J (leaf), J-B (leaf)
        a, b = chain
        if (a.name, b.name) == ("A", "B"):
            return SimpleNamespace(index=index), [[A, I], [I, B]]
        if (a.name, b.name) == ("I", "B"):
            return SimpleNamespace(index=index), [[I, J], [J, B]]
        return SimpleNamespace(index=index, data=SimpleNamespace(chain_trajectory=[chain])), []

    hc._piece = piece
    found, asked = [], []
    hot = {"A"}

    def on_found(rec):
        found.append((rec["start"].name, rec["end"].name))
        hot.add(rec["end"].name) if rec["start"].name in hot else None

    def relevant(chain, final):
        asked.append(tuple(n.name for n in chain))
        return True if {chain[0].name, chain[-1].name} & hot else None

    out = hc([(0, 1)], [A, B], on_found=on_found, relevant=relevant)
    assert ("A", "I") in found                      # reported as found, before the call returned
    assert ("I", "J") in found                      # I became populated by A-I, so I-B and then I-J ran
    assert ("J", "B") in found                      # J populated via I-J
    assert len(out) == len(found)


def _split_into_i_j(hc):
    A, B, I, J = (Node(name=n) for n in "ABIJ")

    def piece(chain, index, depth):          # A-B splits into I-J only (no end the source reaches)
        a, b = chain
        if (a.name, b.name) == ("A", "B"):
            return SimpleNamespace(index=index), [[I, J]]
        return SimpleNamespace(index=index, data=SimpleNamespace(chain_trajectory=[chain])), []

    hc._piece = piece
    return A, B, I, J


def test_pieces_nothing_reaches_are_iced_not_dropped(tmp_path):
    hc = _connector(tmp_path)
    A, B, I, J = _split_into_i_j(hc)
    found, done = [], []
    out = hc([(0, 1)], [A, B], on_found=lambda r: found.append(r), relevant=lambda chain, final: None,
             on_pair_done=done.append)
    assert found == [] and out == []           # I-J never ran
    assert [(pair, [n.name for n in args[0]], args[2]) for pair, args in hc.iced] == [((0, 1), ["I", "J"], 1)]
    assert done == [(0, 1)]                    # the pair has nothing running: resumable as finished


def test_a_later_call_thaws_an_iced_piece_once_material_reaches_it(tmp_path):
    hc = _connector(tmp_path)
    A, B, I, J = _split_into_i_j(hc)
    hc([(0, 1)], [A, B], relevant=lambda chain, final: None)
    assert len(hc.iced) == 1
    # still out of reach: iced again, not run
    assert hc([], [A, B], relevant=lambda chain, final: None) == [] and len(hc.iced) == 1
    # a step found since sends flux to I: the next call (even with no pairs of its own) runs I-J
    out = hc([], [A, B], relevant=lambda chain, final: True if chain[0].name == "I" else None)
    assert [(r["start"].name, r["end"].name) for r in out] == [("I", "J")]
    assert hc.iced == []
    assert out[0]["label"] == "pair_0_1_leaf_1"    # its own number in the pair's search, as when it was split


def test_iced_pieces_and_numbering_survive_a_resume(tmp_path):
    import pickle

    hc = _connector(tmp_path)
    A, B, I, J = _split_into_i_j(hc)
    hc([(0, 1)], [A, B], relevant=lambda chain, final: None)
    state = pickle.loads(pickle.dumps(hc.queue_state()))
    fresh = _connector(tmp_path)
    _split_into_i_j(fresh)
    fresh.restore(state)
    assert len(fresh.iced) == 1 and fresh._next_index == {(0, 1): 2}
    out = fresh([], [A, B], relevant=lambda chain, final: True)
    assert [(r["start"].name, r["end"].name) for r in out] == [("I", "J")]


def test_own_jobs_and_pairs_added_while_the_queue_runs(tmp_path):
    """The explorer's proposals run as jobs in the queue's pool; their products' pairs join the running
    call; when the queue drains, the caller may add more before the call ends."""
    hc = _connector(tmp_path)
    A, B, C = (Node(name=n) for n in "ABC")
    hc._fast_ts = lambda i, j, a, b: (SimpleNamespace(energy=-1.0), f"fast_{i}_{j}", None)
    hc.register("propose", lambda name: name + "!", slots=2)
    nodes, got, drained = [A, B], [], []

    def proposed(out):
        got.append(out)
        nodes.append(C)                      # a new species, and its pair, while the queue runs
        hc.add_pairs([(0, 2)])

    def on_drain():
        if drained:
            return False
        drained.append(True)
        hc.submit("propose", ("again",), got.append)
        return True

    out = hc([(0, 1)], nodes, start=lambda: hc.submit("propose", ("A",), proposed), on_drain=on_drain)
    assert got == ["A!", "again!"]
    assert sorted(r["pair"] for r in out) == [(0, 1), (0, 2)]
    assert hc._live is None                  # submit/add_pairs only work while a call runs


def test_the_queue_sets_one_floor_for_every_piece(tmp_path):
    hc = _connector(tmp_path)
    hc.job_inputs.path_min_inputs.barrier_cutoff_kcal = 30.0
    seed = Node(name="A", _cached_energy=-5.0, energy=-5.0)
    hc._piece = lambda chain, index, depth: (SimpleNamespace(index=index), [])
    hc([(0, 1)], [seed, Node(name="B")])
    assert hc.job_inputs.path_min_inputs.network_floor_hartree == -5.0


def test_core_share_follows_the_running_jobs_and_stays_a_number_where_it_is_written():
    """An engine's n_parallel in the explorer's queues: the cores over the jobs running now. Copies of the
    run's inputs share it; written out (stats.json, a spawned process) it is the number it is then."""
    import copy
    import json
    import multiprocessing
    import pickle

    from mepd.discovery.horizontal import _CoreShare

    running = multiprocessing.get_context("fork").RawValue("i", 0)
    share = _CoreShare(48, running)
    assert int(share) == 48
    running.value = 6
    assert int(share) == 8 and min(20, share) == 8 and share // 2 == 4
    assert copy.deepcopy(share) is share
    assert pickle.loads(pickle.dumps(share)) == 8
    assert json.dumps({"engine_parallel": int(share)}) == '{"engine_parallel": 8}'
