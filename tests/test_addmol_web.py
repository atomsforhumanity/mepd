"""Add a molecule, the workspace side (mepd/web/added.py): a finished expansion adopted without a server,
the network mepd.addmol works on, a step's frames, the variant record."""
import json

import numpy as np
import pytest

from mepd import addmol
from mepd.web import added
from mepd.web.workspace import Workspace, WorkspaceError

# H, C, N in one order: HCN, a bent TS, HNC
FRAMES = {"hcn": [(0.0, 0.0, -1.07), (0.0, 0.0, 0.0), (0.0, 0.0, 1.16)],
          "ts": [(1.10, 0.0, 0.60), (0.0, 0.0, 0.0), (0.0, 0.0, 1.19)],
          "hnc": [(0.0, 0.0, 2.17), (0.0, 0.0, 0.0), (0.0, 0.0, 1.17)]}
E_HCN, E_TS, E_HNC = -93.40, -93.32, -93.38


def _xyz(key, energy):
    rows = "\n".join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip("HCN", FRAMES[key]))
    return f"3\nqcdata_charge=0 qcdata_multiplicity=1 E={energy}\n{rows}\n"


@pytest.fixture
def run(tmp_path):
    """A finished `discovery expand` run of two species and one IRC-verified step, as the explorer writes it."""
    out = tmp_path / "expand" / "run"
    (out / "species").mkdir(parents=True)
    (out / "ts").mkdir()
    (tmp_path / "expand" / "profile.toml").write_text('engine_name = "gxtb"\n')
    (out / "species" / "species_0.xyz").write_text(_xyz("hcn", E_HCN))
    (out / "species" / "species_1.xyz").write_text(_xyz("hnc", E_HNC))
    (out / "ts" / "fast_0_1.xyz").write_text(_xyz("ts", E_TS))
    # (the IRC as the run happened to write it: from the product side)
    (out / "ts" / "fast_0_1_irc.xyz").write_text(_xyz("hnc", E_HNC) + _xyz("ts", E_TS) + _xyz("hcn", E_HCN))
    kcal = 627.509474
    (out / "summary.json").write_text(json.dumps({
        "seed_energy": E_HCN,
        "species": [{"index": 0, "smiles": "C#N", "energy": E_HCN, "rel_energy_kcal": 0.0},
                    {"index": 1, "smiles": "[C-]#[NH+]", "energy": E_HNC, "rel_energy_kcal": (E_HNC - E_HCN) * kcal}],
        "reactions": [{"source": 0, "target": 1, "outcome": "new_species"}],
        "steps": [{"a": 0, "b": 1, "label": "fast_0_1", "ts_energy": E_TS,
                   "barrier_kcal": [(E_TS - E_HCN) * kcal, (E_TS - E_HNC) * kcal], "files": {"verified": True}},
                  {"a": 0, "b": 1, "label": "fast_0_1_b", "ts_energy": E_TS + 0.05,
                   "barrier_kcal": [(E_TS + 0.05 - E_HCN) * kcal, (E_TS + 0.05 - E_HNC) * kcal],
                   "files": {"verified": False}}]}))
    return out


@pytest.fixture
def adopted(tmp_path, run):
    ws = Workspace(tmp_path / "ws")
    mgr = added.manager(ws)
    job = added.adopt_run(mgr, run)
    return ws, mgr, job


def test_a_finished_expansion_is_adopted_without_a_server(adopted, tmp_path):
    ws, mgr, job = adopted
    snap = ws.snapshot()
    assert sorted(r["smiles"] for r in snap["structures"].values()) == ["C#N", "[C-]#[NH+]"]
    assert all(r["energy"] is not None and r["level"]["key"] == job["level"]["key"] for r in snap["structures"].values())
    (edge,) = snap["edges"].values()
    o = edge["origin"]
    assert o["label"] == "fast_0_1" and o["barrier_verified"] and o["ts_energy_hartree"] == E_TS
    assert not any(j["status"] == "queued" for j in mgr.jobs.values())       # nothing waits for a server
    # the workspace opens again with it
    again = added.manager(Workspace(tmp_path / "ws"))
    assert [j["op"] for j in again.jobs.values()] == ["graph-enumeration"]


def test_a_folder_that_is_no_finished_run_is_refused(tmp_path):
    with pytest.raises(WorkspaceError, match="summary.json"):
        added.adopt_run(added.manager(Workspace(tmp_path / "ws")), tmp_path)


def test_kinetics_finds_the_ts_geometry_of_an_expansions_step(adopted):
    """(it used to look only for ts*.xyz with an .energies file: an expansion's TS took its reactants' correction)"""
    from mepd.web.kinetics import ts_geometry

    _ws, _mgr, job = adopted
    symbols, xyz = ts_geometry(job, E_TS)
    assert symbols == ["H", "C", "N"] and xyz[0] == pytest.approx(FRAMES["ts"][0])
    assert ts_geometry(job, E_TS + 0.5) is None


def test_the_network_and_a_steps_frames(adopted):
    ws, mgr, job = adopted
    key = job["level"]["key"]
    net = added.network(ws, mgr.jobs, key, 298.15, free_energy=False)
    (step,) = net["steps"]
    by = {s["smiles"]: s for s in net["species"]}
    assert net["energies"] == "electronic" and step["status"] == "verified" and step["id"].startswith("edge:")
    assert step["reactants"] == [by["C#N"]["id"]] and step["products"] == [by["[C-]#[NH+]"]["id"]]
    assert step["g_ts"] - by["C#N"]["g"] == pytest.approx((E_TS - E_HCN) * 627.509474)
    assert [x["reason"] for x in net["excluded"]] == []
    fr = added.step_frames(ws, mgr.jobs, ws.edge(step["edge"]), key)
    # oriented along the step (the run's IRC file starts at the product), in the TS's atom order
    assert fr["symbols"] == ["H", "C", "N"] and fr["charge"] == 0
    assert np.allclose(fr["r"], FRAMES["hcn"]) and np.allclose(fr["ts"], FRAMES["ts"]) and np.allclose(fr["p"], FRAMES["hnc"])
    assert added.bond_changes(fr["symbols"], fr["r"], fr["p"]) == {"broken": [[0, 1]], "formed": [[0, 2]]}


def test_the_added_molecule_variant_record(adopted):
    ws, mgr, job = adopted
    key = job["level"]["key"]
    parent = added.network(ws, mgr.jobs, key, 298.15, free_energy=False)
    step = parent["steps"][0]
    est = {"steps": [{"step": step["id"], "status": "est", "level": "test",
                      "binding": {"r": {"electronic": -4.0}, "ts": {"electronic": -9.0}, "p": {"electronic": -1.0}}}]}
    var = added.put(ws, {"id": "water", "name": "water", "smiles": "O"}, 0.1, est)
    rec = ws.variants[var["id"]]
    assert rec["edit"] == ["added water"]
    assert rec["kind"] == "added_molecule" and rec["added"]["amount"] == 0.1 and rec["replaced"] == []
    net = added.variant_network(ws, mgr.jobs, key, 298.15, var["id"], free_energy=False)
    # (an estimate beside the step: its version with the molecule, no bound species)
    assert net["added"] == "water" and len(net["species"]) == 3 and len(net["steps"]) == 2
    with_c = next(s for s in net["steps"] if s.get("parent_step") == step["id"])
    assert with_c["g_ts"] == pytest.approx(step["g_ts"] - 9.0) and with_c["status"] == "est"
    # the parent's own network is what it was: the variant replaces nothing
    assert added.network(ws, mgr.jobs, key, 298.15, free_energy=False)["steps"] == parent["steps"]
    # a later layer's numbers go into the same record
    est["steps"][0]["binding"]["ts"]["electronic"] = -11.0
    assert added.put(ws, rec["added"], 0.1, est, vid=var["id"])["id"] == var["id"] and len(ws.variants) == 1
    out = addmol.readout(parent, added.variant_network(ws, mgr.jobs, key, 298.15, var["id"], free_energy=False),
                         {step["reactants"][0]: 1.0}, 0.1, step["products"][0], 3600.0)
    assert out["steps"][0]["through"] == with_c["id"]
    with pytest.raises(WorkspaceError):
        added.variant_network(ws, mgr.jobs, key, 298.15, "nope")


# ---------------------------------------------------------------- the query file and the CLI
def test_the_query_file_is_plain_and_complete(adopted):
    ws, mgr, _job = adopted
    doc = added.query(ws, mgr.jobs, target="[C-]#[NH+]", initial={"C#N": 1.0}, temperature=1500.0, time_s=1e-8,
                      free_energy=False)
    doc = json.loads(json.dumps(doc))                     # plain JSON, nothing of mepd's in it
    assert doc["schema"] == "mepd-addmol-query" and doc["version"] == 1 and doc["units"]["length"] == "angstrom"
    assert doc["energies"] == "electronic" and "absolute" in doc["energy_reference"]
    hcn = next(s for s in doc["species"] if s["smiles"] == "C#N")
    assert hcn["atoms"] == ["H", "C", "N"] and len(hcn["coordinates"]) == 3 and hcn["amount"]["initial"] == 1.0
    assert len(hcn["series"]) == len(doc["times"]) and hcn["status"] == "verified" and hcn["level"] == doc["level"]["key"]
    (step,) = doc["steps"]
    g = step["geometry"]
    assert g["atoms"] == ["H", "C", "N"] and g["bonds_broken"] == [[0, 1]] and g["bonds_formed"] == [[0, 2]]
    assert np.allclose(g["ts"], FRAMES["ts"]) and step["barrier"]["forward"] == pytest.approx((E_TS - E_HCN) * 627.509474)
    assert step["status"] == "verified" and step["extent"] > 0
    # at 1500 K the step runs (1.5 % in this time): its degree of control is defined, and it is the one to lower
    assert doc["target"]["resolved"] and doc["limits"]["defined"]
    assert doc["limits"]["order"][0]["step"] == step["id"] and doc["limits"]["order"][0]["role"] == "lower"
    assert step["control"] == pytest.approx(doc["limits"]["control"]["steps"][step["id"]])
    assert doc["level"]["profile"].startswith("engine_name")
    with pytest.raises(WorkspaceError, match="no species"):
        added.query(ws, mgr.jobs, target="CCO", initial={"C#N": 1.0}, temperature=300.0, time_s=1.0, free_energy=False)


def test_the_cli_from_a_run_to_a_result_file(run, tmp_path):
    """adopt, query, test with estimates that come with the candidate (no bystander layer: --steps 0)."""
    from typer.testing import CliRunner

    from mepd.cli import app

    cli = CliRunner()
    ws = tmp_path / "ws2"
    # (every molecule of this run already has its energy: nothing to minimize)
    res = cli.invoke(app, ["addmol", "adopt", str(run), str(ws)])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["edges_with_verified_ts"] == 1
    qf = tmp_path / "query.json"
    res = cli.invoke(app, ["addmol", "query", str(ws), "--target", "C#N", "--initial", "[C-]#[NH+]=1.0",
                           "--temperature", "300", "--time", "3600", "--electronic", "-o", str(qf)])
    assert res.exit_code == 0, res.output
    q = json.loads(qf.read_text())
    step = q["steps"][0]
    assert not q["limits"]["defined"] and q["limits"]["order"][0]["step"] == step["id"]
    hcn, hnc = step["reactants"][0], step["products"][0]          # (the step is stored HCN -> HNC: uphill)
    cands = [{"name": "relay cat", "smiles": "O", "amount": 0.1, "estimates": {"new_steps": [
                 {"id": "relay", "label": "HCN -> HNC through the molecule", "reactants": [hcn, "added"],
                  "products": [hnc, "added"], "barrier": {"electronic": 25.0}, "parent_step": step["id"],
                  "status": "est", "level": "from a test", "source": "made up"}]}},
             {"name": "nothing", "smiles": "N", "amount": 0.1}]
    cf = tmp_path / "cands.json"
    cf.write_text(json.dumps(cands))
    out = tmp_path / "results"
    res = cli.invoke(app, ["addmol", "test", str(qf), "--candidates", str(cf), "--steps", "0", "-o", str(out)])
    assert res.exit_code == 0, res.output
    lines = json.loads(res.output)
    assert [x["candidate"] for x in lines] == ["relay cat", "nothing"]
    relay = json.loads((out / "relay_cat.json").read_text())
    assert relay["status"] == "done" and relay["readout"]["added"]["fate"] == "regenerated"
    assert relay["readout"]["target"]["with"]["final"] > 0.01 > relay["readout"]["target"]["parent"]["final"]
    row = relay["readout"]["steps"][0]
    assert row["through"] == "relay" and row["status"] == "est" and row["change"] < -10.0
    assert relay["readout"]["first_order"]["factor"] is None and "none of the target" in relay["readout"]["first_order"]["why"]
    assert [x["name"] for x in relay["layers"]] == ["given", "bystander"] and relay["layers"][1]["status"] == "skipped"
    assert relay["estimates"]["new_steps"][0]["layer"] == "given" and relay["estimates"]["new_steps"][0]["source"] == "made up"
    assert relay["readout"]["peak_routes"]["with"][0]["peak"] < relay["readout"]["peak_routes"]["parent"][0]["peak"]
    nothing = json.loads((out / "nothing.json").read_text())
    assert nothing["readout"]["added"]["fate"] == "no part" and nothing["readout"]["steps"][0]["change"] == 0.0
    # a file that is no query, a candidate without an amount
    assert cli.invoke(app, ["addmol", "test", str(cf), "-c", "O", "--amount", "1", "-o", str(out)]).exit_code != 0
    assert cli.invoke(app, ["addmol", "test", str(qf), "-c", "O", "--steps", "0", "-o", str(out)]).exit_code != 0


def test_the_bystander_layer_beside_a_step(run, tmp_path):
    """mepd.addmol.bystander is mepd.bystander's placer (its own tests: tests/test_bystander.py): HCN -> HNC with
    water, quick setting; bindings to the three frames, the shifts, the step's atoms held."""
    pytest.importorskip("axtb")
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.nodes.node import StructureNode
    from mepd.structures import smiles_to_structure

    engine = GXTBCalculator(backend="axtb")
    frames = {"symbols": ["H", "C", "N"], "charge": 0, "multiplicity": 1,
              "r": FRAMES["hcn"], "ts": FRAMES["ts"], "p": FRAMES["hnc"]}
    w = engine.compute_geometry_optimization(StructureNode(structure=smiles_to_structure("O")))[-1].structure
    water = {"symbols": list(w.symbols), "charge": 0, "multiplicity": 1,
             "conformers": [np.asarray(w.geometry) * 0.529177210903]}
    out = addmol.bystander(frames, water, engine, repeats=1, carry=1, polish=0)
    assert out["status"] in ("ok", "reacted") and out["placer"] == "mixed"
    if out["status"] == "ok":
        assert set(out["binding"]) == {"r", "ts", "p"}
        assert out["barrier_shift"]["electronic"] == pytest.approx(
            out["binding"]["ts"]["electronic"] - out["binding"]["r"]["electronic"])
        assert np.allclose(np.array(out["complexes"]["ts"])[:3], FRAMES["ts"])        # the step's atoms are held
    json.dumps(out)                                                                    # plain numbers and lists
