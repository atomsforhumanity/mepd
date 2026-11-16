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


def test_the_bystander_layer_tells_mirror_frames_apart_only_for_a_chiral_molecule():
    """A chiral species and its mirror image: an achiral molecule placed in mirror-image poses binds both alike
    (numerical noise); a chiral one, the same hand beside both, does not."""
    pytest.importorskip("axtb")
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.nodes.node import StructureNode
    from mepd.structures import smiles_to_structure

    engine = GXTBCalculator(backend="axtb")
    ang = lambda st: np.asarray(st.geometry, dtype=float).reshape(-1, 3) * 0.529177210903     # noqa: E731
    opt = lambda smi: engine.compute_geometry_optimization(StructureNode(structure=smiles_to_structure(smi)))[-1].structure   # noqa: E731
    host = opt("C[C@H](O)F")
    mirror = np.array([-1.0, 1.0, 1.0])
    frame = {"symbols": [str(x) for x in host.symbols], "charge": 0, "multiplicity": 1, "r": ang(host), "ts": None, "p": None}
    twin = {**frame, "r": ang(host) * mirror}

    def molecule(smi):
        st = opt(smi)
        return {"symbols": [str(x) for x in st.symbols], "charge": 0, "multiplicity": 1, "conformers": [ang(st)]}

    def beside(fr, mol, pose):
        return addmol.bystander(fr, mol, engine, poses=0, starts=[{"frame": "r", "coords": pose}])

    water = molecule("O")
    first = addmol.bystander(frame, water, engine, poses=2)
    pose = np.asarray(first["complexes"]["r"])[len(frame["symbols"]):]
    a, b = beside(frame, water, pose), beside(twin, water, pose * mirror)
    assert a["binding"]["r"]["electronic"] == pytest.approx(b["binding"]["r"]["electronic"], abs=0.02)
    # a chiral molecule: the same molecule (not its mirror image) beside the mirror frame, put where its mirror
    # image would sit
    chiral = molecule("C[C@@H](O)Cl")
    first = addmol.bystander(frame, chiral, engine, poses=2)
    pose = np.asarray(first["complexes"]["r"])[len(frame["symbols"]):]
    want = pose * mirror                               # the mirror image's place: the same hand fitted onto it
    x = np.asarray(chiral["conformers"][0])
    pc, qc = x - x.mean(axis=0), want - want.mean(axis=0)
    u, _s, vt = np.linalg.svd(pc.T @ qc)
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    a, b = beside(frame, chiral, pose), beside(twin, chiral, pc @ rot + want.mean(axis=0))
    # (with the one given pose and no generic ones, a pose that is already relaxed reads "unmoved": the binding
    # is there all the same)
    assert abs(a["binding"]["r"]["electronic"] - b["binding"]["r"]["electronic"]) > 0.05


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
                           "-o", str(tmp_path / "r")])
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
