"""Verified steps of a flux-steered expansion fill in their graph edges:
barrier (in the edge's direction), TS and IRC, instead of staying "proposed"."""
import json

from mepd.web import chem
from mepd.web.jobs import JobManager
from mepd.web.workspace import Workspace
from tests.test_web import HCN_XYZ, WATER_BENT_XYZ, WATER_XYZ, client  # noqa: F401  (the app fixture)


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


def _irc(path, *frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(frames))


def test_a_finished_run_puts_each_step_on_the_pair_its_irc_joins(tmp_path):
    """Live step events come under the pair searched, before the IRC check; the summary has
    each under its real ends, and none the run dropped. A step on the wrong edge comes off it."""
    from mepd.web.app import adopt_expansion_steps

    ws, jobs, (s0, s1, s2) = _ws(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    job = {"id": "j", "op": "graph-enumeration", "output_dir": str(out), "live_nodes": {"0": s0, "1": s1, "2": s2},
           "seed_adopted": True}
    known = ws.snapshot()["structures"]
    # Reported live: "moved" between 0 and 1 (really 0-2), "dropped" between 1 and 2 (its IRC ran elsewhere).
    for label, a, b in (("moved", 0, 1), ("dropped", 1, 2)):
        jobs._adopt_step(job, {"event": "step", "a": a, "b": b, "label": label, "ts_energy": -76.0,
                               "barrier_kcal": [-50.0, 3.0]}, job["live_nodes"], known)
    assert all(e["origin"]["barrier_verified"] is False for e in ws.snapshot()["edges"].values())   # unchecked yet
    (out / "summary.json").write_text(json.dumps({"species": [], "steps": [
        {"a": 0, "b": 2, "label": "moved", "ts_energy": -76.0, "barrier_kcal": [12.0, 3.0],
         "files": {"verified": True, "pair": [0, 1]}}]}))
    assert adopt_expansion_steps(jobs, job)
    edges = {frozenset((e["source"], e["target"])): e["origin"] for e in ws.snapshot()["edges"].values()}
    assert edges[frozenset((s0, s2))]["label"] == "moved" and edges[frozenset((s0, s2))]["barrier_verified"] is True
    assert edges[frozenset((s0, s1))].get("proposed") and "joins other species" in edges[frozenset((s0, s1))]["headline"]
    assert edges[frozenset((s1, s2))].get("proposed") and "dropped" in edges[frozenset((s1, s2))]["headline"]


def test_a_live_step_whose_irc_reads_as_its_two_species_is_verified(tmp_path):
    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    out = tmp_path / "out"
    job = {"id": "j", "op": "graph-enumeration", "output_dir": str(out), "live_nodes": {"0": s0, "1": s1},
           "charge": 0, "multiplicity": 1}
    st = ws.snapshot()["structures"]
    # water and water pulled apart (OH + H): an IRC from one to the other reads as those two
    _irc(out / "ts" / "w_irc.xyz", WATER_XYZ.strip() + "\n", WATER_BENT_XYZ.strip() + "\n")
    jobs._adopt_step(job, {"event": "step", "a": 0, "b": 1, "label": "w", "ts_energy": -76.0, "barrier_kcal": [5.0, 5.0]},
                     job["live_nodes"], st)
    (edge,) = ws.snapshot()["edges"].values()
    assert edge["origin"]["barrier_verified"] is True


def test_kinetics_leaves_unchecked_expansion_steps_out(tmp_path):
    from mepd.web.kinetics import edge_ts

    edge = {"id": "e", "origin": {"job": "j", "barrier_kcal": 5.0, "ts_energy_hartree": -1.0, "barrier_verified": False}}
    jobs = {"j": {"id": "j", "status": "done", "targets": {"edges": []}, "level": {"key": "k"}}}
    assert edge_ts(edge, jobs, "k", include_unverified=False)[0] is None
    assert edge_ts(edge, jobs, "k", include_unverified=True)[0] == -1.0


def test_species_no_lewis_structure_fits_are_drawn_from_their_geometry():
    from mepd.web.depict_graph import depict_xyz_svg, formula
    from mepd.web.live_network import drawable, names_of

    ylide = """5

P 0.0 0.0 0.0
C 1.68 0.0 0.0
C -0.6 1.5 0.0
C -0.6 -0.75 1.3
C -0.6 -0.75 -1.3
"""
    assert formula(ylide) == "C4P"
    assert depict_xyz_svg(ylide).lstrip().startswith("<svg")
    assert not drawable("C7H17OP (unusual bonding #df15)") and not drawable("") and drawable("CC=O")
    assert names_of({"3": "C7H17OP (unusual bonding #df15)"}) == {"3": "C7H17OP · unusual bonding"}


def test_species_identity_is_the_molecular_graph():
    from mepd.web.step_check import same_species

    assert same_species("[OH].[H]", "[H+].[OH-]")          # one geometry, two Lewis readings
    assert same_species("C=C.O", "O.C=C")
    assert not same_species("CC=O", "C=CO")                 # where the H sits makes another species
    assert not same_species("C=[O+][CH2-]", "C1CO1")


def test_an_expansions_mixtures_get_their_molecules_energies_queued_once(tmp_path):
    """Kinetics counts a reaction from its molecules apart; the run optimizes each mixture whole.
    The molecules missing an energy at the run's level get one minimization job per charge and spin."""
    ws, jobs, (s0, s1, s2) = _ws(tmp_path)
    from tests.test_web import HCN_XYZ, WATER_XYZ

    (both,) = chem.structures_from_xyz_text(
        "6\n\n" + "\n".join(WATER_XYZ.strip().splitlines()[2:]) + "\n"
        + "\n".join(f"{a} {float(x) + 6:.3f} {y} {z}" for a, x, y, z in (ln.split() for ln in HCN_XYZ.strip().splitlines()[2:])) + "\n",
        0, 1)
    cx = ws.add_structure(both, origin={"kind": "xyz"}, energy=-170.0, level={"key": "k1"})
    assert cx["role"] == "complex"
    submitted = []
    jobs.submit = lambda op, **kw: submitted.append((op, sorted(kw["structure_ids"]))) or [{"id": "j_opt"}]
    job = {"id": "j", "op": "graph-enumeration", "level": {"key": "k1"}, "profile": None,
           "live_nodes": {"0": cx["id"]}, "targets": {"structures": [cx["id"]], "edges": []}}
    assert jobs._queue_member_energies(job, force=True)
    assert submitted == [("optimize", sorted(cx["members"]))]
    assert not jobs._queue_member_energies(job, force=True)       # asked once


def test_a_live_step_whose_irc_joins_other_species_moves_there(tmp_path):
    """The TS search for one pair found another step's TS (the Wittig: searched ylide + acetone -> products,
    its IRC runs products -> oxaphosphetane): it leaves the searched edge and joins the pair its IRC joins."""
    ws, jobs, (s0, s1, s2) = _ws(tmp_path)
    out = tmp_path / "out"
    st = ws.snapshot()["structures"]
    for sid, e in ((s0, -76.40), (s1, -76.35), (s2, -93.0)):
        ws._data["structures"][sid]["energy"] = e
    job = {"id": "j", "op": "graph-enumeration", "output_dir": str(out), "live_nodes": {"0": s0, "1": s1, "2": s2},
           "charge": 0, "multiplicity": 1}
    # reported between 0 and 2; its IRC runs water (0) -> water pulled apart (1)
    _irc(out / "ts" / "x_irc.xyz", WATER_XYZ.strip() + "\n", WATER_BENT_XYZ.strip() + "\n")
    known = ws.snapshot()["structures"]
    assert jobs._adopt_step(job, {"event": "step", "a": 0, "b": 2, "label": "x", "ts_energy": -76.30,
                                  "barrier_kcal": [1.0, 2.0]}, job["live_nodes"], known)
    edges = {frozenset((e["source"], e["target"])): e["origin"] for e in ws.snapshot()["edges"].values()}
    assert frozenset((s0, s2)) not in edges or edges[frozenset((s0, s2))].get("proposed")
    moved = edges[frozenset((s0, s1))]
    assert moved["label"] == "x" and moved["barrier_verified"] is True
    assert abs(moved["barrier_kcal"] - (0.10 * 627.5094740631 if moved.get("barrier_from", s0) == s0 else 0.05 * 627.5094740631)) < 0.1


def test_a_completed_step_of_a_running_expansion_opens_with_its_energy_profile(client, tmp_path):
    """A start that came without an energy at the run's level (drawn, built) gets the run's only when it ends
    (species/species_0.xyz): mid-run, or after a stopped or failed run, its completed steps measured from the
    start's node energy had none ("No energy profile"). They are measured from the run's own optimized start
    instead (summary_partial.json's seed_energy, the value the finished run writes to species_0)."""
    from tests.test_web import _add

    s0 = _add(client, WATER_XYZ)[0]["id"]
    s1 = _add(client, WATER_BENT_XYZ)[0]["id"]
    out = tmp_path / "run_out"
    (out / "ts").mkdir(parents=True)
    frames = [WATER_XYZ.replace("water\n", "\n"), WATER_XYZ.replace("0.758 0.000 0.504", "1.40 0.000 0.470").replace("water\n", "\n"),
              WATER_BENT_XYZ.replace("water, one H pulled off\n", "\n")]
    (out / "ts" / "fast_0_1_irc.xyz").write_text("".join(frames))
    (out / "ts" / "fast_0_1_irc.energies").write_text("-76.40\n-76.30\n-76.35\n")
    (out / "summary_partial.json").write_text(json.dumps({"partial": True, "seed_energy": -76.40}))
    jobs = client.app.state.sessions.current.jobs
    job = {"id": "j_running", "op": "graph-enumeration", "status": "running", "title": "expansion", "created": 0,
           "output_dir": str(out), "targets": {"structures": [s0], "edges": []}, "live_nodes": {"0": s0, "1": s1}}
    jobs.jobs[job["id"]] = job
    ws = client.app.state.sessions.current.ws
    assert ws.snapshot()["structures"][s0].get("energy") is None
    edge = ws.add_edge(s0, s1, origin={"kind": "job", "job": "j_running", "label": "fast_0_1", "has_ts": True})
    got = client.get(f"/api/edges/{edge['id']}/step-path").json()
    kcal = [f["energy_kcal"] for f in got["frames"]]
    assert kcal[0] == 0.0 and abs(kcal[1] - 0.1 * 627.5094740631) < 1e-6 and got["ts_index"] == 1
    assert "run's own" in got["reference"]
    # from the other end: that node is no start, and has no energy either: nothing made up
    back = client.get(f"/api/edges/{edge['id']}/step-path", params={"start": s1}).json()
    assert all(f["energy_kcal"] is None for f in back["frames"])


def test_orbitals_follow_an_expansions_own_step_on_its_edge(client, tmp_path, monkeypatch):
    """Explore's reaction from a network expansion: Orbitals along this path read that step's own IRC (as the
    player shows it, source -> target), even while the run goes on, and are listed on that reaction."""
    from tests.test_web import _add
    from mepd.web import operations

    monkeypatch.setattr(operations.get_operation("orbitals"), "available", True)
    s0 = _add(client, WATER_XYZ)[0]["id"]
    s1 = _add(client, WATER_BENT_XYZ)[0]["id"]
    out = tmp_path / "run_out"
    (out / "ts").mkdir(parents=True)
    frames = [WATER_XYZ.replace("water\n", "\n"), WATER_XYZ.replace("0.758 0.000 0.504", "1.40 0.000 0.470").replace("water\n", "\n"),
              WATER_BENT_XYZ.replace("water, one H pulled off\n", "\n")]
    (out / "ts" / "fast_0_1_irc.xyz").write_text("".join(frames))
    (out / "ts" / "fast_0_1_irc.energies").write_text("-76.40\n-76.30\n-76.35\n")
    jobs = client.app.state.sessions.current.jobs
    job = {"id": "j_running", "op": "graph-enumeration", "status": "running", "title": "expansion", "created": 0,
           "output_dir": str(out), "targets": {"structures": [s0], "edges": []}, "live_nodes": {"0": s0, "1": s1},
           "charge": 0, "multiplicity": 1}
    jobs.jobs[job["id"]] = job
    ws = client.app.state.sessions.current.ws
    edge = ws.add_edge(s0, s1, origin={"kind": "job", "job": "j_running", "label": "fast_0_1", "has_ts": True})
    r = client.post("/api/jobs", json={"op": "orbitals", "source_job": "j_running", "dry_run": True,
                                       "params": {"edge": edge["id"], "basis": "sto-3g", "frames": 3}})
    assert r.status_code == 200, r.text
    rec = r.json()[0]
    assert rec["targets"] == {"structures": [s0, s1], "edges": [edge["id"]]}
    assert "--ts-frame" in rec["argv"] and rec["argv"][rec["argv"].index("--ts-frame") + 1] == "2"
    # without an edge, an expansion is no source of orbitals; nor is a reaction of another run
    assert client.post("/api/jobs", json={"op": "orbitals", "source_job": "j_running", "dry_run": True,
                                          "params": {}}).status_code == 400
    s2 = _add(client, HCN_XYZ)[0]["id"]
    other = ws.add_edge(s0, s2, origin={"kind": "job", "job": "j_other", "label": "fast_0_1"})
    assert client.post("/api/jobs", json={"op": "orbitals", "source_job": "j_running", "dry_run": True,
                                          "params": {"edge": other["id"]}}).status_code == 400


def test_kinetics_say_when_they_go_beyond_what_was_explored(tmp_path):
    """Kinetics hotter than the expansion's own cross barriers it never searched past, and material that ends in
    species it never expanded goes on in ways nobody looked for: both are said, with the mass that ends there.
    A mixture the run expanded ("ketene + water") counts for its molecules; at the run's own conditions, nothing."""
    from mepd.web.kinetics import coverage

    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({
        "steering": {"mode": "access", "cutoff_kcal": 87.9, "temperature_K": 1000.0, "time_s": 3600.0},
        "rounds": [{"sources": [0, 1]}]}))
    job = {"id": "j_x", "op": "graph-enumeration", "status": "done", "title": "expansion", "output_dir": str(out),
           "live_nodes": {"0": "s_a", "1": "s_kw", "2": "s_co"}, "params": {"flux_threshold": 0.01}}
    snap = {"structures": {"s_a": {"smiles": "CC(=O)O"}, "s_kw": {"smiles": "C=C=O.O"}, "s_k": {"smiles": "C=C=O"},
                           "s_w": {"smiles": "O"}, "s_co": {"smiles": "[C-]#[O+]"}}}
    sp = lambda i, c0, final, mx, m: {"id": i, "name": i, "c0": c0, "final": final, "max": mx, "mass": m}
    species = [sp("s_a", 1.0, 0.0, 1.0, 60.0), sp("s_k", 0, 0.5, 0.6, 42.0), sp("s_w", 0, 0.5, 0.6, 18.0),
               sp("s_co", 0, 0.0, 0.002, 28.0)]
    at_run = coverage(snap, {"j_x": job}, species, 1000.0, 3600.0, {"s_a"})
    assert at_run["beyond"] is False and at_run["frontier"] == []
    assert abs(at_run["searched_kcal"] - 87.9) < 1e-9 and at_run["reach_kcal"] < 87.9
    species[1:] = [sp("s_k", 0, 0.0, 0.6, 42.0), sp("s_w", 0, 0.5, 0.6, 18.0), sp("s_co", 0, 1.0, 1.0, 28.0)]
    hot = coverage(snap, {"j_x": job}, species, 1500.0, 3600.0, {"s_a"})
    assert hot["beyond"] is True and hot["reach_kcal"] > 110
    assert [f["id"] for f in hot["frontier"]] == ["s_co"] and abs(hot["frontier_share"] - 28.0 / 60.0) < 1e-9
    # no kinetics-steered exploration behind the network: nothing to compare with
    assert coverage(snap, {}, species, 1500.0, 3600.0, {"s_a"}) is None
