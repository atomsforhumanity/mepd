"""The live network view's data (mepd.web.live_network): explorer kinetics
refreshes become compact frames, corrections compact events, and the end state
is never a trailing unsteered refresh."""
import json

from mepd.web import live_network as ln


def _kin(t, basis="free energy (353 K); single points on 3 of 3 structures", final_b=0.3):
    return {
        "basis": basis, "t_wall": 1000.0 + t, "t_start": 1000.0,
        "species": [
            {"id": "0", "smiles": "C=C.C=C", "G": 0.0, "max_concentration": 1.0, "final": 0.1, "populated": True, "energy_source": "dft"},
            {"id": "1", "smiles": "C1CCC1", "G": -20.0, "max_concentration": 0.6, "final": 0.6, "populated": True, "energy_source": "dft"},
            {"id": "2", "smiles": "C=CC=C", "G": -5.0, "max_concentration": final_b, "final": final_b, "populated": True, "energy_source": "gxtb"},
        ],
        "mirror_species": [{"id": "m1", "mirror_of": "1", "max_concentration": 0.1, "final": 0.1}],
        "step_list": [
            {"id": "a", "from": "0", "to": "1", "dG_fwd": 25.0, "dG_rev": 45.0, "energy_source": "dft", "irc_ok": True, "state": "open", "flux": 0.7},
            {"id": "b", "from": "0", "to": "2", "dG_fwd": 28.0, "dG_rev": 33.0, "energy_source": "gxtb", "irc_ok": None, "state": "open", "flux": 0.3},
            {"id": "c", "from": "1", "to": "2", "dG_fwd": 60.0, "dG_rev": 75.0, "energy_source": "gxtb", "state": "closed", "flux": None},
        ],
    }


def test_a_refresh_becomes_a_compact_frame():
    f = ln.compact(_kin(12.0))
    assert f["t"] == 12.0 and f["steered"]
    ids = [s[0] for s in f["species"]]
    assert ids[0] == "0"                                   # the seed first
    one = next(s for s in f["species"] if s[0] == "1")
    assert one[2] == 0.7                                   # its mirror image's material folded in
    assert {s[0] for s in f["steps"]} == {"a", "b"}        # only open steps carry
    assert f["gone"] == [["c", "1", "2", "closed"]]
    assert f["counts"]["closed"] == 1 and f["counts"]["open"] == 2
    assert ln.compact({"species": [{"id": "0"}]}) is None   # not the live-network schema


def test_a_job_records_its_refreshes_and_ends_on_a_steered_one(tmp_path):
    for t in (0.0, 30.0):
        assert ln.record(tmp_path, _kin(t))
    ln.record(tmp_path, _kin(60.0, basis="free energy (353 K)"))       # unsteered tail
    (tmp_path / "live" / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "species", "index": 1},
        {"event": "change", "kind": "barrier_moved", "t_wall": 1031.0, "ids": ["a"], "old": {"dG_fwd": 27.0},
         "new": {"dG_fwd": 25.0, "energy_source": "dft"}, "why": "its TS's steering single point came back"},
    ]) + "\n")
    tl = ln.from_job(tmp_path)
    assert len(tl["frames"]) == 3 and tl["final"] == 1
    assert tl["events"] == [[31.0, "barrier_moved", ["a"], {"from": 27.0, "to": 25.0, "src": "dft",
                                                           "why": "its TS's steering single point came back"}]]
    assert tl["smiles"]["1"] == "C1CCC1"


def test_older_explorer_runs_replay_from_their_step_events(tmp_path):
    """Runs before the step_list schema: species by index, steps from events.jsonl
    (one edge per pair, its lowest barrier), no flux; everyday names come along."""
    species = [{"index": 0, "smiles": "O=CO", "rel_kcal": 0.0, "max_concentration": 1.0, "final": 0.0, "populated": True},
               {"index": 1, "smiles": "O.[C-]#[O+]", "rel_kcal": -5.0, "max_concentration": 0.9, "final": 0.9, "populated": True},
               {"index": 2, "smiles": "", "rel_kcal": 40.0, "max_concentration": 0.0, "final": 0.0, "populated": False}]
    for k, ns in ((1, 1_000_000_000), (2, 31_000_000_000)):
        (tmp_path / f"kinetics_{ns}.json").write_text(json.dumps(
            {"kind": "x", "data": {"basis": "free energy (1500 K); single points on 2 of 2 structures", "steps": 2,
                                   "species": species[: k + 1]}}))
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "step", "a": 0, "b": 1, "label": "s01", "barrier_kcal": [60.0, 70.0]},
        {"event": "step", "a": 1, "b": 0, "label": "s10", "barrier_kcal": [65.0, 55.0]},   # same pair, lower
        {"event": "step", "a": 0, "b": 2, "label": "s02", "barrier_kcal": [80.0, 40.0]},
    ]) + "\n")
    tl = ln.from_snapshots(tmp_path)
    first, last = tl["frames"]
    assert last["legacy"] and last["t"] == 30.0
    assert [s[0] for s in first["steps"]] == ["s10"]                  # s02 waits for species 2
    s10 = next(s for s in last["steps"] if s[0] == "s10")
    assert (s10[1], s10[2], s10[3], s10[4]) == ("0", "1", 55.0, 65.0)  # oriented 0 -> 1
    assert tl["names"]["0"] == "formic acid" and tl["names"]["1"] == "water + CO"


def test_every_packed_replay_opens():
    from mepd.web.showcase_tool import listing

    replays = [e for e in listing() if e.get("kind") == "replay"]
    assert len(replays) >= 2
    for e in replays:
        tl = ln.example_timeline(e["slug"])
        assert tl["frames"] and "names" in tl and 0 <= tl["final"] < len(tl["frames"])


def _legacy_job(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "seed.xyz").write_text("3\n\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    live = tmp_path / "live"
    live.mkdir()
    (live / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "species", "index": 1, "parent": 0, "smiles": "", "rel_energy_kcal": 3.0,
         "xyz": "2\n\nH 0 0 0\nH 0.74 0 0\n"},
        {"event": "step", "a": 0, "b": 1, "label": "fast_0_1", "barrier_kcal": [30.0, 27.0]},
    ]) + "\n")
    (live / "queue.json").write_text(json.dumps({"kind": "queue", "data": {"workers": 4, "busy": 1,
                                                 "running": [{"kind": "ts", "what": "0→1"}], "finished": {}}}))
    (live / "rxn0001.json").write_text(json.dumps({"kind": "morph", "label": "#1 from species 0",
                                                   "plot": {"reactant_smiles": "O", "product_smiles": "[H][H]", "source": 0}}))
    return live


def test_a_running_network_shows_from_its_first_species_before_any_kinetics(tmp_path):
    """develop's explorer (older kinetics schema): the page grows from events.jsonl
    at once, perceives the SMILES the explorer left empty, and says what runs now."""
    _legacy_job(tmp_path)
    tl = ln.from_job(tmp_path, running=True, t0=0.0)
    f = tl["frames"][-1]
    assert f["growing"] and {s[0] for s in f["species"]} == {"0", "1"}
    assert [s[0] for s in f["steps"]] == ["fast_0_1"]
    assert tl["smiles"]["0"] == "O" and tl["smiles"]["1"] == "[H][H]"   # from the seed / the event's xyz
    assert f["steered"]                                                 # no DFT steering in this run: nothing pending
    assert tl["activity"]["queue"]["busy"] == 1
    assert tl["activity"]["sims"][0]["stream"] == "rxn0001"


def test_older_kinetics_refreshes_are_recorded_with_the_jobs_clock(tmp_path):
    _legacy_job(tmp_path)
    kin = {"basis": "free energy (1000 K, 1M)", "steps": 1, "species": [
        {"index": 0, "smiles": "", "rel_kcal": 0.0, "max_concentration": 1.0, "final": 0.4, "populated": True},
        {"index": 1, "smiles": "", "rel_kcal": 3.0, "max_concentration": 0.6, "final": 0.6, "populated": True}]}
    import time
    assert ln.record(tmp_path, kin, t0=time.time() - 42)
    tl = ln.from_job(tmp_path)
    f = tl["frames"][0]
    assert 41 <= f["t"] <= 60 and f.get("legacy")
    assert tl["smiles"]["1"] == "[H][H]"


def test_the_job_pages_kinetics_table_gets_perceived_smiles(tmp_path):
    _legacy_job(tmp_path)
    kin = {"species": [{"index": 0, "smiles": ""}, {"index": 1, "smiles": ""}, {"index": 2, "smiles": "C"}]}
    ln.fill_species_smiles(tmp_path, kin)
    assert [s["smiles"] for s in kin["species"]] == ["O", "[H][H]", "C"]


def test_a_step_the_run_had_no_time_to_check_is_drawn_unconfirmed():
    """A run out of time leaves its IRC-unchecked steps out of the final kinetics (summary.json:
    files.verified False): the final network keeps them as unconfirmed (irc -1), and a pair with a
    checked step shows that one even when an unchecked one is lower."""
    kin = {"basis": "free energy (1500 K)", "species": [
        {"index": 0, "smiles": "O=CO", "rel_kcal": 0.0, "max_concentration": 1.0, "final": 0.5, "populated": True},
        {"index": 1, "smiles": "O.[C-]#[O+]", "rel_kcal": -5.0, "max_concentration": 0.5, "final": 0.5, "populated": True},
        {"index": 2, "smiles": "O=C=O.[H][H]", "rel_kcal": -10.0, "max_concentration": 0.2, "final": 0.2, "populated": True}]}
    steps = [{"label": "ok01", "a": 0, "b": 1, "dG": [60.0, 65.0], "unchecked": False},
             {"label": "low01", "a": 0, "b": 1, "dG": [50.0, 55.0], "unchecked": True},
             {"label": "u02", "a": 0, "b": 2, "dG": [70.0, 80.0], "unchecked": True}]
    f = ln.compact_legacy(kin, steps)
    irc = {s[0]: s[6] for s in f["steps"]}
    assert irc == {"ok01": None, "u02": -1}


METHANOL = ["C 0 0 0", "O 1.42 0 0", "H -0.36 1.03 0", "H -0.36 -0.51 0.89", "H -0.36 -0.51 -0.89", "H 1.75 0.9 0"]
FORMALDEHYDE_H2 = ["C 0 0 0", "O 1.21 0 0", "H -0.55 0.94 0", "H -0.55 -0.94 0", "H 4.0 0 0", "H 4.74 0 0"]


def _xyz(rows, frames=1):
    return "".join(f"{len(rows)}\n\n" + "\n".join(rows) + "\n" for _ in range(frames))


def test_reactor_events_arrive_once_each_and_name_only_verified_species(tmp_path):
    """The live views' reactor overlay: each MD event once, timed by the segment that holds its product
    frame, from the species whose propose job ran then. It names the species it made only when that is
    verified on its own frames: the molecules its atoms form are the molecules that make the species differ
    from the start (as the network compares species); an exchange (its products are its reactants) makes
    nothing, and nor does an event whose products are only part of what a species changed."""
    import os

    job_dir, out = tmp_path / "job", tmp_path / "job" / "output"
    run = out / "reactor" / "run_00"
    (run / "md").mkdir(parents=True)
    (job_dir / "live").mkdir(parents=True)
    (job_dir / "inputs").mkdir()
    (job_dir / "inputs" / "seed.xyz").write_text(_xyz(METHANOL))
    (run / "md" / "segment_000.xyz").write_text(_xyz(METHANOL, 125))
    (run / "md" / "segment_001.xyz").write_text(_xyz(FORMALDEHYDE_H2, 30))
    t0 = 1000.0
    os.utime(run / "md" / "segment_000.xyz", (t0 + 2, t0 + 2))
    os.utime(run / "md" / "segment_001.xyz", (t0 + 5, t0 + 5))
    (out / "network_aware.jsonl").write_text(json.dumps({"t": t0 + 9, "event": "job", "kind": "propose", "what": "species 0",
                                                         "run_s": 9.0, "wait_s": 0.0, "ok": True}) + "\n")
    (job_dir / "live" / "kinetics.json").write_text(json.dumps({"data": {"species": [{"index": 0, "smiles": "CO"}]}}))
    made = {"event": 0, "reactant_frame": 120, "product_frame": 130, "atoms": [0, 1, 2, 3, 4, 5],
            "label": "CO -> C=O + [H][H]", "bond_changes": [{"frame": 125, "atoms": [0, 4], "formed": False}]}
    exchange = {"event": 1, "reactant_frame": 126, "product_frame": 132, "atoms": [0, 1, 2, 3], "label": "C=O -> C=O"}
    (run / "live_events.json").write_text(json.dumps({"events": [made, exchange]}))
    job = {"op": "graph-enumeration", "output_dir": str(out), "started": t0}
    got = ln.reactor_events(job_dir, job, 0)
    assert got["n"] == 2 and got["runs"] == 1
    e = got["events"][0]
    assert (e["t"], e["src"], e["prod"], e["run"], e["k"]) == (5.0, "0", None, "run_00", 0)   # segment_001 holds frame 130
    assert ln.reactor_events(job_dir, job, 2)["events"] == []                                # nothing new: nothing twice
    # runs are known by folder: run_100 sorting in before run_11 (or a run turning up late) moves nothing
    for name in ("run_11", "run_100"):
        (out / "reactor" / name / "md").mkdir(parents=True)
        (out / "reactor" / name / "live_events.json").write_text(json.dumps({"events": [dict(exchange, event=0)]}))
    more = ln.reactor_events(job_dir, job, 2)["events"]
    assert sorted(x["run"] for x in more) == ["run_100", "run_11"] and ln.reactor_events(job_dir, job, 4)["events"] == []
    # a species of formaldehyde + H2 and H2 again (it changed more than the event made): not that one
    (job_dir / "live" / "events.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"event": "species", "index": 2, "parent": 0, "smiles": "C=O.[H][H].[H][H]",
         "xyz": _xyz(FORMALDEHYDE_H2[:4] + ["H 4.0 0 0", "H 4.74 0 0", "H 8.0 0 0", "H 8.74 0 0"])}]) + "\n")
    assert ln.reactor_events(job_dir, job, 4)["prods"] == {}
    # the species it made turns up: the next poll names it, for the event the client already has; never the exchange
    with open(job_dir / "live" / "events.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "species", "index": 1, "parent": 0, "smiles": "C=O.[H][H]",
                             "xyz": _xyz(FORMALDEHYDE_H2)}) + "\n")
    later = ln.reactor_events(job_dir, job, 4)
    assert later["events"] == [] and later["prods"] == {0: "1"}


def test_a_running_timeline_sends_the_new_frames_with_the_runs_conditions(tmp_path):
    """While a run is followed, a poll asks only for the frames it lacks (`since`); the timeline says how many
    are recorded and the run's temperature, simulated time and time budget (from its command line)."""
    for t in (0.0, 30.0, 60.0):
        assert ln.record(tmp_path, _kin(t))
    (tmp_path / "job.json").write_text(json.dumps({"argv": ["discovery", "expand", "seed.xyz", "--temperature", "353.15",
                                                            "--time", "3600", "--time-budget", "1800"]}))
    full = ln.from_job(tmp_path, running=True)
    assert full["recorded"] == 3 and len(full["frames"]) == 3
    assert full["conditions"] == {"T": 353.15, "time_s": 3600.0, "budget_s": 1800.0}
    tail = ln.from_job(tmp_path, running=True, since=2)
    assert tail["since"] == 2 and [f["t"] for f in tail["frames"]] == [60.0] and tail["recorded"] == 3
    assert ln.from_job(tmp_path, running=True, since=9)["frames"] == []             # (asked past the end)
    assert len(ln.from_job(tmp_path, running=False, since=2)["frames"]) == 3       # a finished run: all of it


def test_a_finished_runs_last_moment_is_when_it_ended(tmp_path):
    """A finished run's network as it stood at the end is timed when the run ended, not when it is looked at."""
    (tmp_path / "live").mkdir()
    (tmp_path / "live" / "events.jsonl").write_text(json.dumps({"event": "species", "index": 1, "smiles": "CO", "parent": 0}) + "\n")
    tl = ln.from_job(tmp_path, running=False, t0=1000.0, t_end=1090.0)
    assert tl["frames"][-1]["t"] == 90.0


def test_a_finished_runs_end_holds_its_summarys_amounts(tmp_path):
    """The last live kinetics can predate a finished run's last species (isoprene + methyl acrylate: the meta
    adduct's mirror twin, a fifth of the material, has no species event at all): the end holds the run's own
    species and final amounts (summary.json), its virtual mirror images counted with the species they mirror."""
    out = tmp_path / "output"
    out.mkdir()
    (tmp_path / "job.json").write_text(json.dumps({"output_dir": str(out)}))
    live = tmp_path / "live"
    live.mkdir()
    (live / "kinetics.json").write_text(json.dumps({"basis": "free energy (418 K, 1M)", "threshold": 0.01, "species": [
        {"index": 0, "smiles": "C=CC(=C)C.C=CC(=O)OC", "rel_kcal": 0.0, "max_concentration": 1.0, "final": 0.5, "populated": True},
        {"index": 1, "smiles": "COC(=O)C1CC=C(C)CC1", "rel_kcal": -41.0, "max_concentration": 0.5, "final": 0.5, "populated": True}]}))
    (live / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "species", "index": 1, "smiles": "COC(=O)C1CC=C(C)CC1", "parent": 0},
        {"event": "species", "index": 2, "smiles": "COC(=O)C1CCC=C(C)C1", "parent": 0},      # after the last refresh
    ]) + "\n")                                                                                # (3: no event at all)
    (out / "summary.json").write_text(json.dumps({
        "steps": [{"label": "a", "a": 0, "b": 1, "barrier_kcal": [25.0, 66.0]}, {"label": "b", "a": 0, "b": 2, "barrier_kcal": [26.0, 67.0]},
                  {"label": "c", "a": 0, "b": 3, "barrier_kcal": [26.0, 66.0]}],
        "species": [{"index": 0, "max_concentration": 1.0, "final_concentration": 0.1},
                    {"index": 1, "max_concentration": 0.25, "final_concentration": 0.25},
                    {"index": 2, "max_concentration": 0.2, "final_concentration": 0.2},
                    {"index": 3, "smiles": "COC(=O)C1CCC=C(C)C1", "rel_energy_kcal": -40.0, "max_concentration": 0.2,
                     "final_concentration": 0.2}],
        "mirror_images": {"virtual_species": [{"index": 4, "mirror_of": 1, "max_concentration": 0.25, "final_concentration": 0.25}]}}))
    end = ln.from_job(tmp_path, running=False, t0=0.0, t_end=60.0)["frames"][-1]
    held = {s[0]: (s[1], s[2], s[3], s[4]) for s in end["species"]}
    assert held == {"0": (0.0, 1.0, 0.1, 1), "1": (-41.0, 0.5, 0.5, 1), "2": (None, 0.2, 0.2, 1), "3": (-40.0, 0.2, 0.2, 1)}
    assert {s[0] for s in end["steps"]} == {"a", "b", "c"}


def test_a_species_an_irc_end_made_is_drawn_while_the_run_is_on(tmp_path):
    """The run reports no species event for an IRC end and writes output/species only at its end: until then
    the species' geometry is its step's IRC end that is not the step's other species."""
    out = tmp_path / "output"
    (out / "ts").mkdir(parents=True)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "job.json").write_text(json.dumps({"output_dir": str(out)}))
    methanol = "6\n\nC 0 0 0\nO 1.42 0 0\nH -0.36 1.03 0\nH -0.36 -0.51 0.89\nH -0.36 -0.51 -0.89\nH 1.75 0.9 0"
    split = "6\n\nC 0 0 0\nO 1.21 0 0\nH -0.55 0.94 0\nH -0.55 -0.94 0\nH 4.0 0 0\nH 4.74 0 0"
    (tmp_path / "inputs" / "seed.xyz").write_text(methanol + "\n")
    (out / "ts" / "fast_0_3_irc.xyz").write_text(methanol + "\n" + split + "\n")
    (tmp_path / "live").mkdir()
    (tmp_path / "live" / "events.jsonl").write_text(json.dumps({"event": "step", "a": 0, "b": 3, "label": "fast_0_3"}) + "\n")
    xyz = ln.species_xyz(tmp_path, "3")
    assert xyz is not None and "H 4.74 0 0" in xyz
    assert ln.species_xyz(tmp_path, "4") is None


def test_an_event_movie_can_be_a_short_preview(tmp_path):
    """The live views' ghost of a reactor event asks for a few frames: evenly spaced, the event's reactant and
    product frames kept and its bond changes on the nearest kept frame."""
    from mepd.web.nanoreactor import event_view

    (tmp_path / "md").mkdir()
    (tmp_path / "md" / "segment_000.xyz").write_text("".join(f"2\n\nH 0 0 0\nH {0.74 + 0.01 * k:.2f} 0 0\n" for k in range(125)))
    (tmp_path / "md" / "schedule.json").write_text(json.dumps({"dump_fs": 2.0}))
    (tmp_path / "live_events.json").write_text(json.dumps({"events": [{"event": 0, "atoms": [0, 1], "reactant_frame": 50,
        "product_frame": 70, "label": "[H][H] -> 2 [H]", "bond_changes": [{"frame": 61, "atoms": [0, 1], "formed": False}]}]}))
    full = event_view(tmp_path, 0)
    short = event_view(tmp_path, 0, max_frames=10)
    assert len(full["frames"]) == 81 and len(short["frames"]) <= 12
    assert full["frames"][full["reactant_frame"]] == short["frames"][short["reactant_frame"]]
    assert full["frames"][full["product_frame"]] == short["frames"][short["product_frame"]]
    assert 0 <= short["changes"][0]["frame"] < len(short["frames"])


def test_what_forms_is_counted_per_molecule_not_per_box_state():
    """A species is a whole box. Two formic acids: 90 % of the boxes end with one dehydrated (water + CO +
    the other formic acid), 10 % with both (2 water + 2 CO). Per starting pair: CO and water 0.9 + 2 x 0.1 =
    1.1 each (a molecule in both states counted from both, twice where a box holds two), formic acid
    left 0.9; the species that holds most of each comes along (for the view's link)."""
    tl = {"smiles": {"0": "OC=O.OC=O", "1": "O.OC=O.[C-]#[O+]", "2": "O.O.[C-]#[O+].[C-]#[O+]"}, "names": {},
          "frames": [{"t": 0.0, "species": [["0", 0.0, 1.0, 0.0, 1, 0], ["1", -5.0, 0.9, 0.9, 1, 0],
                                            ["2", -9.0, 0.1, 0.1, 1, 0]]}]}
    ln.molecule_yields(tl)
    f = tl["frames"][0]
    assert sorted(f["yields"]) == [["O", 1.1, "1"], ["[C-]#[O+]", 1.1, "1"]]
    assert f["left"] == [["O=CO", 0.9]]
    assert tl["start"] == ["O=CO", "O=CO"]
    assert tl["molecules"]["[C-]#[O+]"]["name"] == "CO" and tl["molecules"]["O=CO"]["name"] == "formic acid"


def _embedded_xyz(smiles, seed=7):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return Chem.MolToXYZBlock(m)


def test_molecules_keep_their_geometrys_stereo_and_meet_their_mirror_image(tmp_path):
    """The explorer's SMILES carry no stereo (endo and exo furan-maleimide adducts read the same): a species'
    molecules take it from its geometry. The run counts every chiral species with its mirror image (its
    default), so the two hands of a molecule are one, said to be (`mirror`); with --no-mirror-images, two."""
    from rdkit import Chem

    endo, exo = "O=C1NC(=O)[C@H]2[C@@H]1[C@H]1C=C[C@@H]2O1", "O=C1NC(=O)[C@H]2[C@@H]1[C@@H]1C=C[C@H]2O1"
    plain = "O=C1NC(=O)C2C3C=CC(O3)C12"
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    assert canon(ln.stereo_smiles_of(plain, _embedded_xyz(endo))) == canon(endo)
    assert canon(ln.stereo_smiles_of(plain, _embedded_xyz(exo))) == canon(exo)
    assert ln.stereo_smiles_of("CCO", _embedded_xyz(endo)) is None                  # not this geometry's graph

    para_r, para_s = "COC(=O)[C@@H]1CC=C(C)CC1", "COC(=O)[C@H]1CC=C(C)CC1"
    (tmp_path / "live").mkdir()
    lines = [json.dumps({"event": "species", "index": k, "smiles": "COC(=O)C1CC=C(C)CC1", "xyz": _embedded_xyz(s)})
             for k, s in ((1, para_r), (2, para_s))]
    smiles = {"0": "C=CC(=C)C.C=CC(=O)OC", "1": "COC(=O)C1CC=C(C)CC1", "2": "COC(=O)C1CC=C(C)CC1"}
    stereo = ln._species_stereo(tmp_path, smiles, lines)
    assert set(stereo) == {"1", "2"} and canon(stereo["1"]) == canon(para_r) and canon(stereo["2"]) == canon(para_s)
    assert (tmp_path / "live" / ln.STEREO).exists()

    def tl():
        return {"smiles": smiles, "names": {}, "frames": [{"t": 0.0, "species": [
            ["0", 0.0, 1.0, 0.5, 1, 0], ["1", -41.0, 0.25, 0.25, 1, 0], ["2", -41.0, 0.25, 0.25, 1, 0]]}]}
    one = ln.molecule_yields(tl(), stereo)
    [(m, amount, _)] = one["frames"][0]["yields"]
    assert amount == 0.5 and one["molecules"][m]["mirror"] and canon(m) in (canon(para_r), canon(para_s))
    two = ln.molecule_yields(tl(), stereo, mirror_images=False)
    assert sorted(a for _, a, _ in two["frames"][0]["yields"]) == [0.25, 0.25]
    assert two["frames"][0]["left"] == [["C=CC(=C)C", 0.5], ["C=CC(=O)OC", 0.5]]


def test_isomers_get_labels_that_tell_them_apart():
    """Molecules with one formula and no everyday name are told apart in plain words, from their graphs and
    the stereo their geometry gave them, by only as much as it takes: endo or exo, E or Z, where a ring's
    two substituents sit, open chain or the ring's size; failing those, the SMILES."""
    def tells(keys):
        return {k: v[1] for k, v in ln.isomer_labels(keys).items()}

    para, meta, chain = "COC(=O)[C@@H]1CC=C(C)CC1", "COC(=O)[C@@H]1CCC=C(C)C1", "C=CC(=C)CCCC(=O)OC"
    assert tells([para, meta, chain]) == {para: "1,4-isomer", meta: "1,3-isomer", chain: "open chain"}
    assert ln.isomer_labels([para, meta])[para] == ("C9H14O2, 1,4-isomer", "1,4-isomer")
    cyclobutane = "C=C[C@]1(C)C[C@H](C(=O)OC)C1"                        # 1,3 too, on a 4-ring
    assert tells([para, meta, cyclobutane]) == {para: "1,4-isomer, 6-ring", meta: "1,3-isomer, 6-ring",
                                                cyclobutane: "1,3-isomer, 4-ring"}
    endo, exo = "O=C1NC(=O)[C@H]2[C@@H]1[C@H]1C=C[C@@H]2O1", "O=C1NC(=O)[C@H]2[C@@H]1[C@@H]1C=C[C@H]2O1"
    assert tells([endo, exo]) == {endo: "endo", exo: "exo"}
    f_endo, f_exo = "COC(=O)[C@@H]1C[C@@H]2C=C[C@H]1C2", "COC(=O)[C@@H]1C[C@H]2C=C[C@@H]1C2"
    assert tells([f_endo, f_exo]) == {f_endo: "endo", f_exo: "exo"}
    ze, ee = "COC(=O)/C=C\\C=C\\C(=O)OC", "COC(=O)/C=C/C=C/C(=O)OC"
    assert tells([ze, ee]) == {ze: "E,Z", ee: "E,E"}
    branched = "C=CC(=C)C[C@@H](C)C(=O)OC"                               # two open chains: their SMILES
    assert tells([chain, branched]) == {chain: "C=CC(=C)CCCC(=O)OC", branched: "C=CC(=C)CC(C)C(=O)OC"}
    assert ln.isomer_labels(["CCO"]) == {"CCO": ("C2H6O", None)}            # alone with its formula
    tl = ln.molecule_yields({"smiles": {"0": "C=CC(=C)C.C=CC(=O)OC", "1": para, "2": meta}, "names": {}, "frames": [
        {"t": 0.0, "species": [["0", 0.0, 1.0, 0.1, 1, 0], ["1", -41.0, 0.5, 0.5, 1, 0], ["2", -41.0, 0.4, 0.4, 1, 0]]}]})
    mol = {m: (x.get("label"), x.get("tell")) for m, x in tl["molecules"].items() if x.get("tell")}
    assert sorted(mol.values()) == [("C9H14O2, 1,3-isomer", "1,3-isomer"), ("C9H14O2, 1,4-isomer", "1,4-isomer")]
    assert tl["molecules"]["C=CC(=C)C"] == {"name": "isoprene", "formula": "C5H8", "species": "0"}


def test_a_finished_runs_frames_end_when_it_ended(tmp_path):
    """A frame recorded after the run finished (a server that first read its end hours later) is the run's
    end, not hours more of growth: its time is the finish, so the replay's clock ends with the run."""
    for t in (0.0, 30.0):
        assert ln.record(tmp_path, _kin(t))
    assert ln.record(tmp_path, _kin(5000.0))                          # recorded long after the end
    tl = ln.from_job(tmp_path, t0=1000.0, t_end=1100.0)
    assert [f["t"] for f in tl["frames"]] == [0.0, 30.0, 100.0]
    assert [f["t"] for f in ln.from_job(tmp_path, running=True, t0=1000.0)["frames"]][-1] == 5000.0   # running: as recorded
