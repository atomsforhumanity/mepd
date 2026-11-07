"""Verified steps of a flux-steered expansion fill in their graph edges:
barrier (in the edge's direction), TS and IRC, instead of staying "proposed"."""
import json

from mepd.web import chem
from mepd.web.jobs import JobManager
from mepd.web.workspace import Workspace
from tests.test_web import HCN_XYZ, WATER_BENT_XYZ, WATER_XYZ


class Bus:
    def publish(self, event, data):
        pass


def _ws(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ids = [ws.add_structure(chem.structures_from_xyz_text(x, 0, 1)[0], origin={"kind": "xyz"})["id"]
           for x in (WATER_XYZ, WATER_BENT_XYZ, HCN_XYZ)]
    return ws, JobManager(ws, Bus()), ids


def test_step_event_turns_a_proposed_edge_into_a_computed_one(tmp_path):
    ws, jobs, (s0, s1, s2) = _ws(tmp_path)
    ws.add_edge(s0, s1, origin={"kind": "job", "job": "j", "proposed": True})
    job = {"id": "j", "live_nodes": {"0": s0, "1": s1, "2": s2}}
    known = ws.snapshot()["structures"]
    step = {"event": "step", "a": 1, "b": 0, "label": "pair_0_1_leaf_0", "barrier_kcal": [30.0, 20.0]}
    assert jobs._adopt_step(job, step, job["live_nodes"], known)
    (edge,) = ws.snapshot()["edges"].values()
    o = edge["origin"]
    # The edge runs s0 -> s1, the step 1 -> 0: its barrier from s0 is the step's reverse one.
    assert not o.get("proposed") and o["barrier_kcal"] == 20.0 and o["entry"] == "pair_0_1_leaf_0_irc" and o["has_ts"]
    # A higher-barrier TS for the same pair does not replace it; a lower one does.
    assert not jobs._adopt_step(job, {**step, "label": "other", "barrier_kcal": [40.0, 35.0]}, job["live_nodes"], known)
    assert jobs._adopt_step(job, {**step, "label": "better", "barrier_kcal": [15.0, 5.0]}, job["live_nodes"], known)
    assert ws.snapshot()["edges"][edge["id"]]["origin"]["barrier_kcal"] == 5.0
    # A step between species with no edge yet gets one.
    assert jobs._adopt_step(job, {"event": "step", "a": 1, "b": 2, "label": "x", "barrier_kcal": [9.0, 8.0]},
                            job["live_nodes"], known)
    assert len(ws.snapshot()["edges"]) == 2


def test_finished_expansion_backfills_steps_from_its_summary(tmp_path):
    from mepd.web.app import adopt_expansion_steps

    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    ws.add_edge(s0, s1, origin={"kind": "job", "job": "j", "proposed": True})
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"steps": [
        {"a": 0, "b": 1, "label": "pair_0_1_leaf_0", "barrier_kcal": [12.0, 3.0], "files": {}}]}))
    job = {"id": "j", "output_dir": str(out), "live_nodes": {"0": s0, "1": s1}}
    assert adopt_expansion_steps(jobs, job)
    (edge,) = ws.snapshot()["edges"].values()
    assert edge["origin"]["barrier_kcal"] == 12.0 and not edge["origin"].get("proposed")
    assert not adopt_expansion_steps(jobs, job)      # nothing new the second time


def test_the_finished_summary_replaces_live_barriers(tmp_path):
    from mepd.web.app import adopt_expansion_steps

    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    job = {"id": "j", "live_nodes": {"0": s0, "1": s1}}
    live = {"event": "step", "a": 0, "b": 1, "label": "early", "barrier_kcal": [5.0, 1.0]}
    jobs._adopt_step(job, live, job["live_nodes"], ws.snapshot()["structures"])
    out = tmp_path / "out"
    out.mkdir()
    # Species 0 was found lower later: the same TS is now 8 kcal/mol above it. Two TSs join the pair;
    # the lower one (by TS energy) is kept.
    (out / "summary.json").write_text(json.dumps({"steps": [
        {"a": 0, "b": 1, "label": "early", "ts_energy": -1.00, "barrier_kcal": [8.0, 1.0]},
        {"a": 0, "b": 1, "label": "high", "ts_energy": -0.90, "barrier_kcal": [70.0, 63.0]}]}))
    assert adopt_expansion_steps(jobs, {**job, "output_dir": str(out)})
    (edge,) = ws.snapshot()["edges"].values()
    assert edge["origin"]["barrier_kcal"] == 8.0 and edge["origin"]["label"] == "early"


def test_steps_keep_their_ts_energy_for_kinetics(tmp_path):
    """Kinetics measures every barrier from the TS's own energy: a step's edge carries it,
    from the live event as from the finished summary (it used to keep only the barrier)."""
    from mepd.web.app import adopt_expansion_steps

    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    job = {"id": "j", "live_nodes": {"0": s0, "1": s1}}
    jobs._adopt_step(job, {"event": "step", "a": 0, "b": 1, "label": "s", "ts_energy": -76.3, "barrier_kcal": [9.0, 4.0]},
                     job["live_nodes"], ws.snapshot()["structures"])
    (edge,) = ws.snapshot()["edges"].values()
    assert edge["origin"]["ts_energy_hartree"] == -76.3
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"steps": [{"a": 0, "b": 1, "label": "s", "ts_energy": -76.31,
                                                             "barrier_kcal": [8.0, 4.0]}]}))
    assert adopt_expansion_steps(jobs, {**job, "output_dir": str(out)})
    assert ws.snapshot()["edges"][edge["id"]]["origin"]["ts_energy_hartree"] == -76.31


def test_old_edges_get_their_ts_energy_from_the_runs_step_events(tmp_path):
    """Edges adopted before the energy was kept, from steps a stopped run's summary leaves out."""
    from mepd.web.app import backfill_expansion_ts

    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    ws.add_edge(s0, s1, origin={"kind": "job", "job": "j", "has_ts": True, "label": "ch_0_1_3", "barrier_kcal": 9.0,
                                "ts_energy_hartree": None})
    live = jobs.job_dir("j") / "live"
    live.mkdir(parents=True)
    (live / "events.jsonl").write_text(json.dumps({"event": "step", "a": 0, "b": 1, "label": "ch_0_1_3",
                                                   "ts_energy": -76.2, "barrier_kcal": [9.0, 3.0]}) + "\n")
    assert backfill_expansion_ts(jobs, {"id": "j"})
    (edge,) = ws.snapshot()["edges"].values()
    assert edge["origin"]["ts_energy_hartree"] == -76.2
    assert not backfill_expansion_ts(jobs, {"id": "j"})


def test_the_expansions_optimized_seed_gives_the_seed_node_its_energy(tmp_path):
    """A seed without an energy at the run's level (a reactor molecule, a drawing) would drop
    out of kinetics with everything it starts: the run's own optimized seed (species 0) fills it in."""
    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    out = tmp_path / "out"
    (out / "species").mkdir(parents=True)
    (out / "species" / "species_0.xyz").write_text(WATER_XYZ)
    (out / "summary_partial.json").write_text(json.dumps({"seed_energy": -76.4}))
    level = {"profile": "default", "key": "k1", "label": "gxtb"}
    job = {"id": "j", "op": "graph-enumeration", "output_dir": str(out), "live_nodes": {"0": s0}, "level": level,
           "charge": 0, "multiplicity": 1, "targets": {"structures": [s0], "edges": []}}
    assert ws.snapshot()["structures"][s0]["energy"] is None
    assert jobs._adopt_seed_energy(job)
    rec = ws.snapshot()["structures"][s0]
    assert rec["energy"] == -76.4 and rec["level"]["key"] == "k1"
    assert not jobs._adopt_seed_energy(job)           # once


def test_kinetics_names_the_molecules_whose_missing_energy_keeps_reactions_out(tmp_path):
    from mepd.web.kinetics import analyze

    ws, jobs, (s0, s1, s2) = _ws(tmp_path)
    snap = ws.snapshot()
    st = snap["structures"]
    st[s0]["energy"], st[s0]["level"] = -76.4, {"key": "k1"}
    snap["edges"] = {"e1": {"id": "e1", "source": s0, "target": s2,
                            "origin": {"kind": "job", "job": "j", "has_ts": True, "barrier_kcal": 9.0, "ts_energy_hartree": -76.3}}}
    out = analyze(snap, {"j": {"id": "j", "status": "done", "targets": {"edges": []}, "level": {"key": "k1"}}}, "k1",
                  initial={s0: 1.0}, held=[], temperature=300, time_s=1.0, target=None)
    assert "no energy at this level" in out["error"]
    assert [m["id"] for m in out["missing_energy"]] == [s2]
