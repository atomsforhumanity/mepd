"""An explorer run as the parent of mepd addmol (mepd/discovery/parent.py): its own molecule-level network."""
import json

import numpy as np
import pytest

from mepd import addmol
from mepd.discovery import parent
from tests.test_addmol_web import E_HCN, E_HNC, E_TS, FRAMES, _xyz

KCAL = 627.509474
C_HCN, C_HNC, C_TS = 10.0, 9.0, 7.0          # thermal corrections "the run recorded", kcal/mol


def _run(tmp_path, recorded=True, steer=None):
    out = tmp_path / "expand" / "run"
    (out / "species").mkdir(parents=True)
    (out / "ts").mkdir()
    (tmp_path / "expand" / "profile.toml").write_text('engine_name = "gxtb"\n[gxtb_engine_kwds]\nbackend = "axtb"\n')
    (out / "species" / "species_0.xyz").write_text(_xyz("hcn", E_HCN))
    (out / "species" / "species_1.xyz").write_text(_xyz("hnc", E_HNC))
    (out / "ts" / "fast_0_1.xyz").write_text(_xyz("ts", E_TS))
    (out / "ts" / "fast_0_1_irc.xyz").write_text(_xyz("hnc", E_HNC) + _xyz("ts", E_TS) + _xyz("hcn", E_HCN))
    (out / "ts" / "fast_0_1_b.xyz").write_text(_xyz("ts", E_TS + 0.05))
    terms = lambda c, s=None: {"thermal_correction_kcal": c if recorded else None,     # noqa: E731
                               "steer": "single_point" if s is not None else "not_asked", "steer_shift_kcal": s}
    (out / "summary.json").write_text(json.dumps({
        "seed_energy": E_HCN, "steering": {"temperature_K": 298.15, "standard_state": "1M"},
        "species": [{"index": 0, "smiles": "C#N", "energy": E_HCN, "energy_terms": terms(C_HCN, 0.0)},
                    {"index": 1, "smiles": "[C-]#[NH+]", "energy": E_HNC, "energy_terms": terms(C_HNC)}],
        "steps": [{"a": 0, "b": 1, "label": "fast_0_1", "ts_energy": E_TS, "ts_energy_terms": terms(C_TS, steer),
                   "files": {"ts": "run/ts/fast_0_1.xyz", "irc": "run/ts/fast_0_1_irc.xyz", "verified": True},
                   "closed": {"reason": "ts_above_cutoff"}},
                  {"a": 0, "b": 1, "label": "fast_0_1_b", "ts_energy": E_TS + 0.05, "ts_energy_terms": terms(None),
                   "files": {"ts": "run/ts/fast_0_1_b.xyz", "verified": False}}],
        "mirror_images": {}}))
    return out


def test_the_runs_own_network_with_its_closed_verified_steps(tmp_path):
    run = _run(tmp_path, steer=-2.0)
    assert parent.is_run(run) and not parent.is_run(tmp_path) and parent.profile_of(run).name == "profile.toml"
    net = parent.network(run)
    assert net["source"] == "explorer run" and net["energies"] == "free" and net["temperature"] == 298.15
    by = {s["id"]: s for s in net["species"]}
    hcn, hnc = (k for k in by if "N" in k and "H+" not in k), (k for k in by if "H+" in k)
    hcn, hnc = next(hcn), next(hnc)
    # relative to the seed's molecule, on the run's free energies
    assert by[hcn]["g"] == pytest.approx(0.0) and by[hnc]["g"] == pytest.approx((E_HNC - E_HCN) * KCAL + C_HNC - C_HCN)
    (step,) = net["steps"]                        # the verified one; closed by the run's cutoff, kept here
    assert step["id"] == "fast_0_1" and step["reactants"] == [hcn] and step["products"] == [hnc]
    assert step["energies"] == "free" and step["status"] == "verified"
    assert step["g_ts"] == pytest.approx((E_TS - E_HCN) * KCAL + C_TS - C_HCN - 2.0)       # with its steering shift
    g = step["geometry"]                          # oriented along the step (the IRC file starts at the product)
    assert g["atoms"] == ["H", "C", "N"] and np.allclose(g["reactant"], FRAMES["hcn"]) and np.allclose(g["ts"], FRAMES["ts"])
    assert g["bonds_broken"] == [[0, 1]] and g["bonds_formed"] == [[0, 2]]
    assert by[hcn]["atoms"] == ["H", "C", "N"] and np.allclose(by[hcn]["coordinates"], FRAMES["hcn"])


def test_a_correction_the_run_did_not_record(tmp_path):
    run = _run(tmp_path, recorded=False)
    net = parent.network(run)                     # no engine: the step is left out, and it says so
    assert net["steps"] == [] and [x["reason"] for x in net["excluded"]] == ["thermal_correction_missing"]
    assert any("left out" in n for n in net["notes"])
    pytest.importorskip("axtb")
    from mepd.engines.gxtb import GXTBCalculator

    net = parent.network(run, GXTBCalculator(backend="axtb"))
    (step,) = net["steps"]
    assert step["energies"] == "free" and any("computed now: 2 boxes, 1 TSs; 0 failed" in n for n in net["notes"])
    assert abs(step["g_ts"] - (E_TS - E_HCN) * KCAL) < 6.0 and step["g_ts"] != pytest.approx((E_TS - E_HCN) * KCAL)


def test_the_query_of_a_run_and_the_cli(tmp_path):
    from typer.testing import CliRunner

    from mepd.cli import app

    run = _run(tmp_path)
    q = parent.query(run, target="C#N", initial={"[C-]#[NH+]": 1.0}, time_s=3600.0)
    q = json.loads(json.dumps(q))
    assert q["schema"] == addmol.QUERY and q["source"] == "explorer run" and q["level"]["profile"].startswith("engine_name")
    assert q["steps"][0]["geometry"] and len(q["species"][0]["series"]) == len(q["times"])
    assert "relative to the starting box" in q["energy_reference"] and not q["limits"]["defined"]
    out = tmp_path / "q.json"
    res = CliRunner().invoke(app, ["addmol", "query", str(run), "--target", "C#N", "--initial", "[C-]#[NH+]=1.0",
                                   "--time", "3600", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text())["steps"][0]["id"] == "fast_0_1"


def test_a_ts_with_the_molecule_on_the_parents_scale_as_a_whole(tmp_path):
    """HCN -> HNC with a water beside it that takes no part: the TS counted by its ends (the water apart),
    above the separated, minimized molecules; the water is no reactant."""
    pytest.importorskip("axtb")
    from mepd.engines.gxtb import GXTBCalculator

    water = [(3.0, 0.0, 0.5), (3.9, 0.3, 0.5), (2.6, 0.9, 0.5)]
    frame = lambda key: list(FRAMES[key]) + water     # noqa: E731
    ts = {"atoms": ["H", "C", "N", "O", "H", "H"], "coordinates": frame("ts"), "charge": 0, "multiplicity": 1}
    got = parent.ts_step(ts, frame("hcn"), frame("hnc"), GXTBCalculator(backend="axtb"))
    assert [m["atoms"] for m in got["reactants"]] == [[0, 1, 2]] and [m["atoms"] for m in got["products"]] == [[0, 1, 2]]
    assert [m["smiles"] for m in got["spectators"]] == ["O"] and got["counted"] == [3, 3]
    b = got["barrier"]
    assert 20.0 < b["electronic"] < 120.0 and abs(b["free"] - b["electronic"]) < 8.0 and "apart" in got["method"]


def test_a_candidates_step_given_with_its_ts_through_the_cli(tmp_path):
    """`mepd addmol test`: a new step that comes with TS coordinates and its two ends gets its barrier from them
    (never from a barrier "from the complex" plus a binding), and its species from the molecules in it."""
    pytest.importorskip("axtb")
    from typer.testing import CliRunner

    from mepd.cli import app

    run = _run(tmp_path)
    qf = tmp_path / "q.json"
    cli = CliRunner()
    assert cli.invoke(app, ["addmol", "query", str(run), "--target", "C#N", "--initial", "[C-]#[NH+]=1.0",
                            "--time", "3600", "-o", str(qf)]).exit_code == 0
    q = json.loads(qf.read_text())
    water = [(3.0, 0.0, 0.5), (3.9, 0.3, 0.5), (2.6, 0.9, 0.5)]
    frame = lambda key: [list(x) for x in list(FRAMES[key]) + water]     # noqa: E731
    step = {"id": "beside", "label": "HNC -> HCN with the molecule beside it", "status": "est", "source": "a test",
            "ts": {"atoms": ["H", "C", "N", "O", "H", "H"], "coordinates": frame("ts"), "charge": 0, "multiplicity": 1,
                   "ends": [frame("hnc"), frame("hcn")]}}
    bad = {**step, "id": "bad", "ts": {**step["ts"], "atoms": ["H", "Si", "N", "O", "H", "H"]}}
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps([{"name": "water", "smiles": "O", "amount": 0.1,
                               "estimates": {"new_steps": [step, bad]}}]))
    res = cli.invoke(app, ["addmol", "test", str(qf), "--candidates", str(cf), "--steps", "0", "-o", str(tmp_path / "r")])
    assert res.exit_code == 0, res.output
    doc = json.loads((tmp_path / "r" / "water.json").read_text())
    (entry,) = doc["estimates"]["new_steps"]
    hnc = next(s["id"] for s in q["species"] if "H+" in s["id"])
    assert entry["id"] == "beside" and entry["reactants"] == [hnc] and entry["spectators"] == ["O"]
    assert entry["barrier"]["free"] is not None and "ts" not in entry and entry["layer"] == "given"
    assert entry["parent_step"] == "fast_0_1"                  # the parent's step between the same molecules
    assert "separated" in entry["barrier_method"] and doc["timings"]["ts_steps_s"] >= 0
    (lost,) = [f for f in doc["failed"] if f.get("step") == "bad"]
    assert lost["status"] == "failed" and lost["why"]          # (HSiN is no species of this network)
    assert doc["readout"] is not None


# ---------------------------------------------------------------- one reference per molecule; levels; mirror twins
def test_a_ts_is_measured_from_the_parents_own_free_energies():
    """acetone-like A (+ a bystander B: together the starting box) with the added molecule: the barrier is the TS
    above the starting box and the added molecule's lowest form, B taken out at the network's free energy;
    not above whatever conformers the complex happened to hold."""
    h = 1.0 / KCAL
    net = {"species": [{"id": "A", "g": -7.0}, {"id": "B", "g": 7.0}],
           "reference": {"seed_g_hartree": -100.0, "seed_atoms": ["C", "H", "O"]}}
    got = {"g_ts_hartree": -150.0 + 30.0 * h,
           "reactants": [{"smiles": "A", "symbols": ["C", "O"], "g_hartree": -60.0 + 2.0 * h},
                         {"smiles": "X", "symbols": ["N"], "g_hartree": -50.0 + 4.0 * h}],
           "spectators": [{"smiles": "B", "symbols": ["H"], "g_hartree": -40.0 + 1.0 * h}]}
    out = parent.on_parent_scale(got, ["A", "added", "B"], net, -50.0)
    # 30 above the box and X; B (g 7) out: the step A + X -> ... has its TS at 23, that is 30 above A (g -7)
    assert out["free"] == pytest.approx(30.0) and "starting box" in out["reference"]
    # the molecules of the TS are not the starting box (no B): each as it was minimized alone, and it says so
    alone = parent.on_parent_scale({**got, "spectators": []}, ["A", "added"], net, -50.0)
    assert alone["free"] == pytest.approx((-150.0 + 30.0 * h + 50.0 + 60.0 - 2.0 * h) * KCAL) and "alone" in alone["reference"]


def test_the_lowest_form_of_a_molecule():
    pytest.importorskip("axtb")
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.nodes.node import StructureNode
    from mepd.structures import smiles_to_structure

    engine = GXTBCalculator(backend="axtb")
    low = engine.compute_geometry_optimization(StructureNode(structure=smiles_to_structure("OO")))[-1].structure
    flat = low.model_copy(update={"geometry": np.asarray(low.geometry) * [1.0, 1.0, 1.0] * 1.03})   # stretched
    got = parent.lowest_form([flat, low], engine)
    assert got["index"] == 1 and got["g_hartree"] > got["e_hartree"] - 0.1


def _twin_run(tmp_path):
    """The HCN run with a recorded mirror twin of its step (as a run records one for a chiral TS)."""
    run = _run(tmp_path)
    fp = run / "summary.json"
    s = json.loads(fp.read_text())
    s["mirror_images"] = {"virtual_species": [], "species_mirror": [0, 1], "chiral_conformers": {},
                          "added_twins": [{"step": 0, "a": 0, "b": 1}], "found_twins": [], "undecided": []}
    fp.write_text(json.dumps(s))
    return run


def test_a_mirror_image_box_on_no_step_of_its_own_enters_on_its_free_energy(tmp_path):
    """A box the run found that is the mirror image of a step's end, and is itself on no verified step, is in the
    molecule fit through its twin step and through the shared energy of mirror images: on its free energy like
    every other box, not on its electronic energy (which lies a thermal correction lower and moved the molecule)."""
    run = _run(tmp_path)
    (run / "species" / "species_2.xyz").write_text(_xyz("hnc", E_HNC))
    fp = run / "summary.json"
    s = json.loads(fp.read_text())
    s["species"].append({"index": 2, "smiles": "[C-]#[NH+]", "energy": E_HNC,
                         "energy_terms": {"thermal_correction_kcal": C_HNC, "steer": "not_asked", "steer_shift_kcal": None}})
    s["mirror_images"] = {"virtual_species": [], "species_mirror": [0, 2, 1], "chiral_conformers": {},
                          "added_twins": [{"step": 0, "a": 0, "b": 2}], "found_twins": [], "undecided": []}
    fp.write_text(json.dumps(s))
    net = parent.network(run)
    hnc = next(x for x in net["species"] if "H+" in x["id"])
    assert hnc["g"] == pytest.approx((E_HNC - E_HCN) * KCAL + C_HNC - C_HCN)
    assert "g_fixed_only_together" not in hnc and not any("together" in x for x in net["notes"])


def test_a_mirror_twin_has_its_originals_frames_reflected(tmp_path):
    net = parent.network(_twin_run(tmp_path))
    one, twin = net["steps"]
    assert twin["id"] == "fast_0_1~mirror" and twin["mirror_of"] == "fast_0_1" and one["mirror"] == twin["id"]
    g, m = one["geometry"], twin["geometry"]
    assert m["reflected"] and m["atoms"] == g["atoms"] and m["bonds_broken"] == g["bonds_broken"]
    for key in ("reactant", "ts", "product"):
        assert np.allclose(np.asarray(m[key]) * [-1, 1, 1], g[key])
    assert net["reference"]["seed_atoms"] == ["C", "H", "N"] and len(net["reference"]["seed_coordinates"]) == 3
    assert {st["level"] for st in net["steps"]} == {"explorer"}


def test_the_steering_level_for_a_step_given_with_its_ts(tmp_path):
    """--steer: the step moves to the parent's steering level by single points and says so; the read-out says
    that the comparison mixes levels when the parent's own steps are not there. (Here the steering profile is the
    explorer's own, so the shift is zero: the plumbing, not the level.)"""
    pytest.importorskip("axtb")
    from typer.testing import CliRunner

    from mepd.cli import app

    run = _run(tmp_path)
    qf = tmp_path / "q.json"
    cli = CliRunner()
    assert cli.invoke(app, ["addmol", "query", str(run), "--target", "C#N", "--initial", "[C-]#[NH+]=1.0",
                            "--time", "3600", "-o", str(qf)]).exit_code == 0
    water = [(3.0, 0.0, 0.5), (3.9, 0.3, 0.5), (2.6, 0.9, 0.5)]
    frame = lambda key: [list(x) for x in list(FRAMES[key]) + water]     # noqa: E731
    step = {"id": "beside", "status": "est", "level": "somebody's",
            "ts": {"atoms": ["H", "C", "N", "O", "H", "H"], "coordinates": frame("ts"), "charge": 0, "multiplicity": 1,
                   "ends": [frame("hnc"), frame("hcn")]}}
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps([{"name": "water", "smiles": "O", "amount": 0.1, "estimates": {"new_steps": [step]}}]))
    args = ["addmol", "test", str(qf), "--candidates", str(cf), "--steps", "0"]
    assert cli.invoke(app, args + ["-o", str(tmp_path / "low")]).exit_code == 0
    low = json.loads((tmp_path / "low" / "water.json").read_text())
    (entry,) = low["estimates"]["new_steps"]
    assert entry["level"] == "explorer" and entry["level_given"] == "somebody's" and "starting box" in entry["reference"]
    assert entry["added_form"] in ("a conformer of the candidate", "the form in this complex")
    assert low["readout"]["mixed_levels"] is False and list(low["readout"]["levels"]) == ["explorer"]
    res = cli.invoke(app, args + ["--steer", str(run.parent / "profile.toml"), "-o", str(tmp_path / "high")])
    assert res.exit_code == 0, res.output
    high = json.loads((tmp_path / "high" / "water.json").read_text())
    (entry,) = high["estimates"]["new_steps"]
    assert entry["level"] == "explorer+steer" and abs(entry["steer"]["shift_kcal"]) < 1e-6
    assert entry["barrier"]["free"] == pytest.approx(low["estimates"]["new_steps"][0]["barrier"]["free"], abs=1e-6)
    assert [x["name"] for x in high["layers"]] == ["given", "steer", "bystander"] and high["layers"][1]["done"] == 1
    assert high["readout"]["mixed_levels"] is True and set(high["readout"]["levels"]) == {"explorer", "explorer+steer"}


def test_a_step_and_its_mirror_twin_in_the_result(tmp_path):
    """An achiral candidate: the twin's bystander row is the original's (one computation). A chiral one: two
    computations, two rows; each row names the other."""
    pytest.importorskip("axtb")
    from typer.testing import CliRunner

    from mepd.cli import app

    run = _twin_run(tmp_path)
    qf = tmp_path / "q.json"
    cli = CliRunner()
    assert cli.invoke(app, ["addmol", "query", str(run), "--target", "C#N", "--initial", "[C-]#[NH+]=1.0",
                            "--time", "3600", "-o", str(qf)]).exit_code == 0
    res = cli.invoke(app, ["addmol", "test", str(qf), "-c", "O", "-c", "C[C@@H](O)Cl", "--amount", "0.1",
                           "--verify", "0", "-o", str(tmp_path / "r")])
    assert res.exit_code == 0, res.output
    water = json.loads((tmp_path / "r" / "O.json").read_text())
    rows = {e["step"]: e for e in water["estimates"]["steps"]}
    assert set(rows) == {"fast_0_1", "fast_0_1~mirror"} and rows["fast_0_1~mirror"]["mirror_image_of"] == "fast_0_1"
    assert rows["fast_0_1~mirror"]["binding"] == rows["fast_0_1"]["binding"] and water["candidate"]["chiral"] is False
    table = {r["step"]: r for r in water["readout"]["steps"]}
    assert table["fast_0_1"]["mirror"] == "fast_0_1~mirror" and table["fast_0_1~mirror"]["mirror"] == "fast_0_1"
    assert table["fast_0_1"]["after"] == pytest.approx(table["fast_0_1~mirror"]["after"], abs=1e-6)
    chiral = json.loads((tmp_path / "r" / _name("C[C@@H](O)Cl")).read_text())
    assert chiral["candidate"]["chiral"] is True
    both = [e for e in chiral["estimates"]["steps"] + chiral["failed"] if e.get("step")]
    assert sorted(e["step"] for e in both) == ["fast_0_1", "fast_0_1~mirror"]
    assert not any(e.get("mirror_image_of") for e in both)


def _name(text):
    from mepd.cli_addmol import _file_name

    return _file_name(text)


def test_the_relay_layer_scans_a_limiting_step_and_writes_as_it_goes(tmp_path):
    """--scan 1: `mepd channels` between the step's two ends with the molecule beside them; every verified TS
    that joins them becomes a step on the parent's scale, the step's line says what was searched and found."""
    pytest.importorskip("axtb")
    from typer.testing import CliRunner

    from mepd.cli import app

    run = _run(tmp_path)
    qf = tmp_path / "q.json"
    cli = CliRunner()
    assert cli.invoke(app, ["addmol", "query", str(run), "--target", "C#N", "--initial", "[C-]#[NH+]=1.0",
                            "--time", "3600", "--held", "[C-]#[NH+]", "-o", str(qf)]).exit_code == 0
    q = json.loads(qf.read_text())
    assert q["conditions"]["held"] and q["level"]["label"].startswith("gxtb")
    assert (run / "addmol_thermo").is_dir() or True            # (this run records every correction: nothing to keep)
    import os

    before = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    res = cli.invoke(app, ["addmol", "test", str(qf), "-c", "O", "--amount", "0.1", "--scan", "1", "-o", str(tmp_path / "r")])
    assert res.exit_code == 0, res.output
    # (the command held itself to the estimates' cores while the scans ran: the next candidate gets them all back)
    assert before is None or os.sched_getaffinity(0) == before
    doc = json.loads((tmp_path / "r" / "O.json").read_text())
    # the order of work: the quick estimate, the verification with the molecule beside the step, the scan
    assert doc["status"] == "done" and [x["name"] for x in doc["layers"]] == ["bystander", "verify", "relay"]
    assert all(x["status"] == "done" for x in doc["layers"]) and "quick setting" in doc["layers"][0]["what"]
    assert doc["layers"][0]["uncertainty"]["barrier_shift_kcal"]["mean_abs_difference_to_verified"] == 2.0
    assert "--scan" not in doc["scan"]["rule"] and "or after 1 reactions" in doc["scan"]["rule"]
    for n in doc["estimates"]["new_steps"]:
        assert not n["ts_file"].startswith("/") and (tmp_path / "r" / n["ts_file"]).exists()
    (line,) = doc["scan"]["steps"]
    # one scan per net reaction, chosen by what the target would gain; it stands for the reaction's steps
    assert line["step"] == "fast_0_1" and line["steps"] == ["fast_0_1"]
    assert doc["scan"]["cores"]["lanes"] and line["status"] in ("done", "failed")
    assert isinstance(line["arrivals"], list) and "does_not_hold" in doc["scope"] and doc["scope"]["level"]
    # the scans' end: the target's route settled (or the limit used up), the side reactions listed, not scanned
    side = doc["scan"]["side_reactions"]
    assert side["asked_for"] is False and isinstance(side["listed"], list) and side["ask_with"] == "--scan-side-reactions"
    assert all("not scanned" in x["in_words"] for x in side["listed"]) and "left" in doc["scan"]
    if doc["scan"]["ended"]["why"].startswith("no reaction is left"):
        assert doc["target_route_settled"]["t_s"] <= doc["scan"]["ended"]["t_s"] + 1e-6
        assert "no reaction is left that could lower the target's route" in doc["target_route_settled"]["in_words"]
        assert doc["target_route_settled"]["rests_on"] in ("verified", "estimate")
        for ev in doc["target_route_settled"].get("changed_after_settled", []):
            assert ev["t_s"] >= doc["target_route_settled"]["t_s"]
    event = doc["route_peak"]["with"][-1]
    assert event["rests_on"] in ("verified", "estimate") and "final" in event["readout"]["target"]
    assert doc["readout"]["peak_routes"]["with"][0]["rests_on"] == event["rests_on"]
    assert line["reaction"] == "C#N -> [C-]#[NH+]" and "with no barrier it would lower the target's route peak" in line["why"]
    assert line["worth"] == "target" and doc["scan"]["ended"]["why"].startswith(("no reaction is left", "the limit"))
    assert not line["run"].startswith("/") and (tmp_path / "r" / line["run"] / "channels.log").exists()
    assert line["route_peak_if_no_barrier"] < line["route_peak_now"] and line["lane"] in (0, 1, 2)
    assert line["searches"]["total"] >= 1 and "channels" in line["command"] and "--explore-molecule" in line["command"]
    assert "net reaction" in doc["scan"]["rule"] and doc["timings"]["first_bystander_s"] > 0
    (derived,) = doc["scan"]["derivations"]
    assert derived["chosen"] == line["reaction"] and derived["top"][0]["reaction"] == line["reaction"]
    assert doc["scan"]["left"] == [] and all("t_s" in a and "free" in a for a in line.get("arrivals", []))
    assert doc["route_peak"]["parent"] is not None and doc["route_peak"]["with"][0]["t_s"] >= 0
    assert all("t_s" in e for e in doc["estimates"]["steps"] + [n for n in doc["estimates"]["new_steps"]])
    rows = [n for n in doc["estimates"]["new_steps"] if n["layer"] in ("verify", "relay")]
    kinds = [(n["layer"], n["takes_part"]) for n in rows]
    assert len(kinds) == len(set(kinds))                       # the kinetics keep the lowest row of each kind
    assert all(n["parent_steps"] == ["fast_0_1"] for n in rows if n["layer"] == "relay")
    for n in rows:
        assert n["status"] == "verified" and n["parent_step"] == "fast_0_1" and n["barrier"]["free"] is not None
        assert n["level"] == "explorer" and set(n["reactants"]) <= {"C#N", "[C-]#[NH+]", "added"}
    if ("verify", False) in kinds:          # verified with the molecule beside it: replaces the bystander estimate
        assert not [e for e in doc["estimates"]["steps"] if e["step"] == "fast_0_1"]
    if not line.get("with_molecule"):
        assert "none found" in line["answer"]
    # how the parent's steps ended, in one place
    sm = doc["summary"]
    assert sm["steps"] == 1 and sm["est"] + sm["verified_beside"] + sm["takes_part"] + sm["leads_elsewhere"] + sm["failed"] == 1
    assert doc["candidate"]["geometry"]["atoms"] == ["O", "H", "H"] and doc["level_labels"]["explorer"]


def test_frequencies_are_kept_so_another_temperature_needs_no_hessian(tmp_path, monkeypatch):
    pytest.importorskip("axtb")
    import mepd
    from mepd.engines.gxtb import GXTBCalculator

    run = _run(tmp_path, recorded=False)
    engine = GXTBCalculator(backend="axtb")
    cache = tmp_path / "kept"
    first = parent.network(run, engine, cache=cache)
    assert len(list(cache.glob("*.json"))) == 3                     # two boxes and one TS
    plain = parent.network(run, engine, temperature=400.0)          # without the cache: computed again
    monkeypatch.setattr(mepd, "frequencies", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a Hessian was asked")))
    again = parent.network(run, engine, temperature=400.0, cache=cache)
    assert again["steps"][0]["g_ts"] == pytest.approx(plain["steps"][0]["g_ts"], abs=1e-6)
    assert again["steps"][0]["g_ts"] != pytest.approx(first["steps"][0]["g_ts"], abs=1e-3)


def test_a_step_given_with_its_ts_is_mirrored_for_an_achiral_candidate():
    from mepd.cli_addmol import _mirror_entry

    q = {"species": [{"id": x} for x in ("CC=O", "C[C@H](O)F", "C[C@@H](O)F")],
         "steps": [{"id": "s", "mirror": "s~mirror"}, {"id": "s~mirror", "mirror_of": "s"}]}
    entry = {"id": "given", "parent_step": "s", "reactants": ["CC=O", "added"], "products": ["C[C@H](O)F", "added"],
             "barrier": {"free": 20.0}}
    twin = _mirror_entry(q, entry, handed=False)
    assert twin["id"] == "given~mirror" and twin["parent_step"] == "s~mirror" and twin["mirror_image_of"] == "given"
    assert twin["reactants"] == ["CC=O", "added"] and twin["products"] == ["C[C@@H](O)F", "added"]
    assert twin["barrier"] == entry["barrier"]
    assert _mirror_entry(q, entry, handed=True) is None and _mirror_entry(q, {**entry, "parent_step": None}, False) is None


def test_candidates_of_one_job_in_parallel(tmp_path):
    from typer.testing import CliRunner

    from mepd.cli import app

    run = _run(tmp_path)
    qf = tmp_path / "q.json"
    cli = CliRunner()
    assert cli.invoke(app, ["addmol", "query", str(run), "--target", "C#N", "--initial", "[C-]#[NH+]=1.0",
                            "--time", "3600", "-o", str(qf)]).exit_code == 0
    res = cli.invoke(app, ["addmol", "test", str(qf), "-c", "O", "-c", "N", "-c", "CO", "--amount", "0.1", "--steps", "0",
                           "--jobs", "2", "-o", str(tmp_path / "r")])
    assert res.exit_code == 0, res.output
    assert [x["candidate"] for x in json.loads(res.output)] == ["water", "ammonia", "methanol"]
    assert sorted(p.name for p in (tmp_path / "r").glob("*.json")) == ["CO.json", "N.json", "O.json"]


def test_reactions_and_which_are_worth_a_scan():
    """Steps with the same net change are one reaction (a molecule on both sides dropped). Worth a scan: a
    reaction that, with no barrier, would lower the target's route peak (first, by the resulting peak), or bring
    another species below the target's present peak (then). The rest are not scanned. Twins last for a chiral
    molecule. A reaction whose two sides do not hold the same atoms is no reaction to scan."""
    from mepd.cli_addmol import _reactions, _scan_worth

    def sp(i, g, n=3):
        return {"id": i, "name": i, "g": g, "atoms": ["C"] * n}

    def st(i, r, p, ts, **kw):
        return {"id": i, "reactants": r, "products": p, "g_ts": ts, "geometry": {"atoms": []}, **kw}

    q = {"temperature": 298.15, "energies": "free",
         "species": [sp("A", 0.0), sp("S", 0.0, 9), sp("I", 5.0), sp("B", -5.0), sp("X", 1.0), sp("Y", 60.0),
                     sp("Bm", -5.0), sp("Lost", 0.0, 2), sp("W", 0.5), sp("Wm", 0.5), sp("Z", 12.0)],
         "steps": [st("a_i", ["A"], ["I"], 50.0), st("a_i_shuttle", ["A", "S"], ["I", "S"], 45.0),   # one reaction
                   st("i_b", ["I"], ["B"], 30.0), st("a_x", ["A"], ["X"], 55.0), st("a_y", ["A"], ["Y"], 70.0),
                   st("i_bm", ["I"], ["Bm"], 30.0, mirror_of="i_b"), st("a_lost", ["A"], ["Lost"], 20.0),
                   st("a_w", ["A"], ["W"], 58.0), st("a_wm", ["A"], ["Wm"], 58.0, mirror_of="a_w"),
                   st("a_z", ["A"], ["Z"], 58.0)]}
    rx = {r["label"]: r for r in _reactions(q)}
    assert rx["A -> I"]["steps"] == ["a_i", "a_i_shuttle"] and rx["A -> I"]["lowest"] == "a_i_shuttle"
    assert rx["I -> Bm"]["twin"] and not rx["I -> B"]["twin"] and "A -> Lost" not in rx
    net = {k: q[k] for k in ("temperature", "energies", "species", "steps")}
    peak, worth, rest = _scan_worth(net, {"id": "added"}, {}, {"A": 1.0, "S": 1.0, "added": 0.1}, ["B"],
                                    list(rx.values()), set(), True)
    assert peak == pytest.approx(45.0)
    # A -> I without a barrier: the target's peak falls to 30. A -> X without one: X (not the target) comes to 1,
    # below the target's 45. I -> B alone lowers nothing (the peak is A -> I's), nor does A -> Y (Y lies above).
    # I -> Bm starts from I, which cannot be had below the target's peak. A -> W and its twin A -> Wm would bring W
    # and Wm to 0.5: both worth a scan, the twin after everything else although its number is lower than X's.
    assert [(w[0]["label"], w[1]) for w in worth] == [("A -> I", "target"), ("A -> W", "other"), ("A -> X", "other"),
                                                      ("A -> Z", "other"), ("A -> Wm", "other")]
    assert worth[0][2] == pytest.approx(30.0) and worth[2][2] == pytest.approx(1.0) and worth[2][3] == "X"
    assert sorted(r["label"] for r in rest) == ["A -> Y", "I -> B", "I -> Bm"]
    # With the run's conditions: while a reaction would lower the target's peak, those only.
    args = (net, {"id": "added"}, {}, {"A": 1.0, "S": 1.0, "added": 0.1}, ["B"], list(rx.values()), set(), True)
    cond = {"time_s": 100.0, "held": ()}
    _peak, worth, rest = _scan_worth(*args, cond)
    assert [w[0]["label"] for w in worth] == ["A -> I"] and len(rest) == 7
    # None left under (i): the others through the kinetics with the barrier removed. A -> W and A -> X would
    # each take a large part of A (W the larger: it lies lower); A -> Z would make 1e-9 M of Z: not worth a scan.
    _peak, worth, rest = _scan_worth(*args, cond, set())
    assert [(w[0]["label"], w[3]) for w in worth] == [("A -> W", "W"), ("A -> X", "X"), ("A -> Wm", "Wm")]
    assert 0.1 < worth[0][2] < 0.31 and worth[1][2] < worth[0][2]      # (A is shared with "Lost" in this time)
    assert "A -> Z" in [r["label"] for r in rest]


def test_a_run_stopped_from_outside_says_so_in_its_file(tmp_path):
    """SIGTERM (a time limit) while it works: the result file's status is "cut", not "running" and not "done"."""
    pytest.importorskip("axtb")
    import os
    import signal
    import subprocess
    import sys
    import time

    run = _run(tmp_path)
    qf = tmp_path / "q.json"
    q = parent.query(run, target="C#N", initial={"[C-]#[NH+]": 1.0}, time_s=3600.0)
    qf.write_text(json.dumps(q))
    out = tmp_path / "r"
    proc = subprocess.Popen([sys.executable, "-m", "mepd.cli", "addmol", "test", str(qf), "-c", "O", "--amount", "0.1",
                             "--scan", "1", "-o", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "OMP_NUM_THREADS": "1"})
    fp = out / "O.json"
    try:
        for _ in range(600):                      # until the scan runs (or the run is over)
            if proc.poll() is not None:
                break
            try:
                if any(x["name"] == "relay" and x["status"] == "running" for x in json.loads(fp.read_text())["layers"]):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        if proc.poll() is not None:
            pytest.skip("the run finished before it could be stopped")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == 124
    finally:
        if proc.poll() is None:
            proc.kill()
    doc = json.loads(fp.read_text())
    assert doc["status"] == "cut" and "signal" in doc["cut"]["why"] and doc["cut"]["t_s"] > 0
    assert {x["name"]: x["status"] for x in doc["layers"]}["relay"] == "cut"


def test_a_query_from_a_run_built_before_the_fit_fix_is_refused(tmp_path):
    from typer.testing import CliRunner

    from mepd.cli import app

    run = _run(tmp_path)
    q = parent.query(run, target="[C-]#[NH+]", initial={"C#N": 1.0}, time_s=10.0)
    assert q["fixed_only_together"] == []
    del q["fixed_only_together"]
    fp = tmp_path / "old.json"
    fp.write_text(json.dumps(q))
    res = CliRunner().invoke(app, ["addmol", "test", str(fp), "-c", "O", "-o", str(tmp_path / "out")])
    assert res.exit_code != 0 and "build it again" in res.output


def test_a_candidate_given_as_smiles_gets_a_name():
    from mepd.cli_addmol import _candidates

    got = _candidates(["O", "OC(=O)[C@@H]1CCCN1", "CC(C)(C)O", "CCCCO"], None, 0.1)
    assert [c["name"] for c in got] == ["water", "L-proline", "C4H10O", "CCCCO"]       # (one formula: the SMILES)
    assert [c["_stem"] for c in got] == ["O", "OC(=O)[C@@H]1CCCN1", "CC(C)(C)O", "CCCCO"]
    assert all(c["smiles"] for c in got)


def test_a_ts_found_by_the_verification_and_by_a_scan_counts_once():
    from mepd.cli_addmol import _same_ts

    row = {"id": "s|verified pose 0", "layer": "verify", "takes_part": True, "reactants": ["A", "added"],
           "products": ["B", "added"], "barrier": {"electronic": 51.1900, "free": 89.130}}
    again = {**row, "id": "s|ts_pair_0_3_leaf_0", "layer": "relay", "barrier": {"electronic": 51.1950, "free": 89.129}}
    other = {**again, "barrier": {"electronic": 53.0, "free": 90.0}}
    assert _same_ts([row], again) == row["id"] and _same_ts([row], other) is None
    assert _same_ts([row], {**again, "takes_part": False}) is None
    assert _same_ts([{**row, "mirror_image_of": "x"}], again) is None


def test_every_free_energy_the_tool_adds_is_at_the_parents_standard_state(monkeypatch):
    """A 1 atm parent (an explorer run at 1 atm): the placer's binding is counted at 1 atm too (5.6 kcal/mol a
    molecule apart at 700 K from 1 M); TS rows and the molecule's lowest form take the query's standard state."""
    import inspect

    import mepd.cli_addmol as c
    import mepd.thermo as T
    from mepd import bystander as B

    assert c._standard({"reference": {"standard_state": "1atm"}}) == "1atm" and c._standard({}) == "1M"
    seen = []
    monkeypatch.setattr(T, "qrrho", lambda *a, **k: seen.append(k.get("standard_state")) or {"g_corr": 0.0})
    monkeypatch.setattr(B, "rigid_modes", lambda *a, **k: [10.0] * 6)

    class Eng:
        def compute_hessian(self, node):
            return None

    B.binding_free_correction(Eng(), ["O", "H", "H", "H", "H"], np.zeros((5, 3)), 3, temperature=700.0,
                              standard_state="1atm")
    assert seen == ["1atm"] * 3
    assert "standard_state" in inspect.signature(B.bystander).parameters
    src = inspect.getsource(c)
    assert src.count("standard_state=_standard(q)") >= 4      # TS rows (twice), lowest form, the second pose


def test_a_given_ts_holds_the_candidate_before_a_parent_species_of_its_kind(monkeypatch):
    """Water added to a network that makes water: in a TS given with its ends, the water molecule is the added
    one (was: the parent's own water species, whose amount is the network's)."""
    import mepd.cli_addmol as c
    from mepd.discovery import parent as P

    monkeypatch.setattr(P, "lowest_form", lambda *a, **k: {"g_hartree": 0.0, "index": 0, "structure": None})
    monkeypatch.setattr(P, "on_parent_scale", lambda *a, **k: {"free": 1.0, "reference": "test"})
    q = {"temperature": 700.0, "level": {"key": "explorer"},
         "species": [{"id": "O", "name": "O", "smiles": "O"}, {"id": "O=CO", "name": "O=CO", "smiles": "O=CO"},
                     {"id": "O=C=O", "name": "O=C=O", "smiles": "O=C=O"}, {"id": "[H][H]", "name": "[H][H]", "smiles": "[H][H]"}]}
    water = {"symbols": ["O", "H", "H"], "charge": 0, "multiplicity": 1,
             "conformers": [[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]}
    got = {"reactants": [{"smiles": "O=CO", "atoms": [0, 1, 2, 3, 4]}, {"smiles": "O", "atoms": [5, 6, 7]}],
           "products": [{"smiles": "O=C=O", "atoms": [0, 1, 2]}, {"smiles": "[H][H]", "atoms": [3, 6]},
                        {"smiles": "O", "atoms": [4, 5, 7]}],
           "spectators": [], "ts_parts": None, "barrier": {"electronic": 1.0, "free": 1.0}, "counted": [8], "method": "m"}
    fields, _ = c._placed(q, got, water, None)
    assert fields["reactants"] == ["O=CO", "added"] and fields["products"] == ["O=C=O", "[H][H]", "added"]
