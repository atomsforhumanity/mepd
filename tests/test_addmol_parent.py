"""An explorer run as the parent of mepd addmol (mepd/discovery/parent.py): its own molecule-level network."""
import json

import numpy as np
import pytest

from mepd import addmol
from mepd.discovery import parent
from tests.test_addmol_web import E_HCN, E_HNC, E_TS, FRAMES, _xyz

KCAL = 627.509474
C_HCN, C_HNC, C_TS = 10.0, 9.0, 7.0          # thermal corrections "the run recorded", kcal/mol


def _run(tmp_path, recorded=True, steer=0.0):
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
    net = parent.network(run)                     # no engine: electronic differences, and it says so
    (step,) = net["steps"]
    assert step["energies"] == "electronic" and step["g_ts"] == pytest.approx((E_TS - E_HCN) * KCAL)
    assert any("electronic differences" in n for n in net["notes"])
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
