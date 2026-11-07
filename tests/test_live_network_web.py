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
