"""The connect queue: steps reach the caller as they are found; pieces below a split run only when
the caller's kinetics put material at one of their ends; unreachable ones are dropped at the end."""
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
    return hc


def test_pieces_wait_for_material_and_unreached_ones_are_dropped(tmp_path):
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


def test_pieces_nothing_reaches_are_dropped(tmp_path):
    A, B, I, J = (Node(name=n) for n in "ABIJ")
    hc = _connector(tmp_path)

    def piece(chain, index, depth):          # A-B splits into I-J only (no end the source reaches)
        a, b = chain
        if (a.name, b.name) == ("A", "B"):
            return SimpleNamespace(index=index), [[I, J]]
        return SimpleNamespace(index=index, data=SimpleNamespace(chain_trajectory=[chain])), []

    hc._piece = piece
    found = []
    out = hc([(0, 1)], [A, B], on_found=lambda r: found.append(r), relevant=lambda chain, final: None)
    assert found == [] and out == []           # I-J never ran


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
