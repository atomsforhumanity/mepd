"""Reactions › Nanoreactor runs (mepd.web.reactor_list): every run's reactions as one list. One reaction found
by runs that wrote it in opposite directions is one row, counted and signed one way (its representative's);
species whose bonding could not be read keep their reactions per run; Run longer is the same run; a run still
going is read from its live files."""
import json

from mepd.web import reactor_list as rl

ALD = {"smiles": "CC=O", "charge": 0, "multiplicity": 1}
OH = {"smiles": "[OH-]", "charge": -1, "multiplicity": 1}
W = {"smiles": "O", "charge": 0, "multiplicity": 1}
ENO = {"smiles": "C=C[O-]", "charge": -1, "multiplicity": 1}
KEY = "C=C[O-]|-1|1+O|0|1>CC=O|0|1+[OH-]|-1|1"


def _species(*sps):
    return [{"id": i, **sp} for i, sp in enumerate(sps)]


def _run(tmp_path, name, reactions, species=(ALD, OH, W, ENO), live_events=None, live_network=None, final=True):
    out = tmp_path / name
    out.mkdir()
    if final:
        (out / "network.json").write_text(json.dumps({"species": _species(*species), "reactions": reactions,
                                                      "events": []}))
    if live_events is not None:
        (out / "live_events.json").write_text(json.dumps({"events": live_events}))
    if live_network is not None:
        (out / "live_network.json").write_text(json.dumps(live_network))
    return out


def _rx(i, reactants, products, label, count, reverse=0, de=None, reason=None):
    return {"id": i, "reactants": reactants, "products": products, "label": label, "count": count,
            "reverse_count": reverse, "delta_e_kcal": de, "first_fs": 100.0 * (i + 1),
            "complex": {"reason": reason} if reason else {}}


FWD = "CC=O + [OH-] -> O + C=C[O-]"
REV = "O + C=C[O-] -> CC=O + [OH-]"


def test_a_reaction_written_both_ways_is_one_row_read_its_representatives_way(tmp_path):
    a, _ = rl.run_reactions(_run(tmp_path, "a", [_rx(0, [0, 1], [2, 3], FWD, 3, 1, -28.0)]))
    b, _ = rl.run_reactions(_run(tmp_path, "b", [_rx(0, [2, 3], [0, 1], REV, 1, 0, 28.0)]))
    assert a[KEY]["n"] == 1 and a[KEY]["n_rev"] == 3 and a[KEY]["delta_e"] == 28.0     # in the key's direction
    assert not a[KEY]["written_as_key"] and b[KEY]["written_as_key"]

    # No session reaction with TS endpoints: shown the way it was seen most (3 times, against 1 + 1).
    (row,) = rl.merge([{"run": "a", "reactions": a}, {"run": "b", "reactions": b}])
    assert [sorted(s) for s in row["smiles"]] == [["CC=O", "[OH-]"], ["C=C[O-]", "O"]]
    assert (row["n"], row["n_rev"], row["delta_e"], row["runs"]) == (3, 2, -28.0, 2)
    assert row["stage"] == "refined" and row["rep"] is None

    # Run b's session reaction has TS endpoints: the row reads its way (its barrier, its card).
    (row,) = rl.merge([{"run": "a", "reactions": a, "ws": {KEY: ("r_a", False)}},
                       {"run": "b", "reactions": b, "ws": {KEY: ("r_b", True)}}])
    assert row["rep"] == "r_b" and [sorted(s) for s in row["smiles"]] == [["C=C[O-]", "O"], ["CC=O", "[OH-]"]]
    assert (row["n"], row["n_rev"], row["delta_e"]) == (2, 3, 28.0)
    assert {m["run"]: (m["n"], m["n_rev"]) for m in row["members"]} == {"a": (1, 3), "b": (1, 0)}


def test_species_that_could_not_be_read_keep_their_reactions_per_run(tmp_path):
    odd = {"smiles": "?C2H3O", "charge": -1, "multiplicity": 1}
    rx = _rx(0, [0, 1], [2, 3], "CC=O + [OH-] -> O + ?C2H3O", 1)
    a, _ = rl.run_reactions(_run(tmp_path, "a", [rx], species=(ALD, OH, W, odd)))
    b, _ = rl.run_reactions(_run(tmp_path, "b", [rx], species=(ALD, OH, W, odd)))
    rows = rl.merge([{"run": "a", "reactions": a}, {"run": "b", "reactions": b}])
    assert len(rows) == 2 and {r["key"].split(":")[0] for r in rows} == {"a", "b"}
    assert all(r["stage"] == "seen" for r in rows)


def test_a_run_still_going_is_read_from_its_live_files(tmp_path):
    events = [{"key": KEY, "label": FWD, "reaction": 0, "start_fs": 50.0},
              {"key": KEY, "label": FWD, "reaction": 0, "start_fs": 90.0},
              {"key": KEY, "label": REV, "reaction": 0, "start_fs": 120.0},
              {"key": KEY, "label": FWD, "reaction": 0, "start_fs": 130.0, "tentative": True},   # may still change
              {"key": None, "label": "O + O -> O + O", "reaction": None}]                        # a swap
    live = {"species": _species(ALD, OH, W, ENO), "reactions": [_rx(0, [0, 1], [2, 3], FWD, 2, 1, -28.0)]}
    got, final = rl.run_reactions(_run(tmp_path, "a", [], live_events=events, live_network=live, final=False))
    assert not final and got[KEY]["n_rev"] == 2 and got[KEY]["n"] == 1 and got[KEY]["refined"]
    seen, _ = rl.run_reactions(_run(tmp_path, "b", [], live_events=events[:1], final=False))
    assert seen[KEY]["refined"] is False and seen[KEY]["delta_e"] is None


class _WS:
    def __init__(self, reactions=None, edges=None):
        self.data = {"reactions": reactions or {}, "edges": edges or {}, "structures": {}}

    def snapshot(self):
        return self.data


def _job(jid, out, op="nanoreactor", created=0, **params):
    return {"id": jid, "op": op, "output_dir": str(out), "status": "done", "created": created, "title": jid,
            "targets": {"structures": ["s1", "s2"]},
            "params": {"temperature": 2000.0, "time_ps": 5.0, "compress": 0.6, "multiplicity": 1, "copies": "2, 1",
                       **params}}


def test_runs_are_labelled_by_what_differs_and_run_longer_is_the_same_run(tmp_path):
    a = _run(tmp_path, "a", [_rx(0, [0, 1], [2, 3], FWD, 3, 0, -28.0)])
    b = _run(tmp_path, "b", [_rx(0, [2, 3], [0, 1], REV, 1, 0, 28.0)])
    c = _run(tmp_path, "c", [_rx(0, [0, 1], [2, 3], FWD, 2, 0, -28.0)])
    jobs = [_job("j_a", a, created=1), _job("j_b", b, created=2, temperature=2500.0),
            _job("j_c", c, created=3, trajectory="traj.xyz"),
            _job("j_a_more", a, op="nanoreactor-more", created=4, more_ps=5.0)]
    ws = _WS({"r1": {"origin": {"job": "j_a", "key": KEY}, "edge": "e1"}}, {"e1": {}})
    out = rl.collect(ws, jobs)
    assert [(r["run"], r["label"], r["time_ps"]) for r in out["runs"]] == [
        ("j_a", "2000 K · 10 ps", 10.0), ("j_b", "2500 K · 5 ps", 5.0), ("j_c", "steered", 5.0)]
    (row,) = out["reactions"]
    # Read run a's way (its session reaction): a 3 and c 2 that way, b's 1 the other way.
    assert row["rep"] == "r1" and row["runs"] == 3 and (row["n"], row["n_rev"]) == (5, 1)
    assert row["delta_e"] == -28.0


def test_a_box_that_differs_is_named(tmp_path):
    a = _run(tmp_path, "a", [])
    b = _run(tmp_path, "b", [])
    out = rl.collect(_WS(), [_job("j_a", a, created=1), _job("j_b", b, created=2, copies="4, 1")])
    assert [r["label"] for r in out["runs"]] == ["2000 K · box 1", "2000 K · box 2"]


def test_repeats_share_their_conditions_and_are_numbered(tmp_path):
    runs = [_job(f"j_{k}", _run(tmp_path, f"r{k}", []), created=k, seed=k) for k in range(3)]
    out = rl.collect(_WS(), runs)
    assert [(r["label"], r["name"]) for r in out["runs"]] == [("2000 K", "2000 K #1"), ("2000 K", "2000 K #2"),
                                                              ("2000 K", "2000 K #3")]
