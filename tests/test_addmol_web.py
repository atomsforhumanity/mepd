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
    assert rec["kind"] == "added_molecule" and rec["added"]["amount"] == 0.1 and rec["replaced"] == []
    net = added.variant_network(ws, mgr.jobs, key, 298.15, var["id"], free_energy=False)
    assert net["added"] == "water" and len(net["species"]) == 5 and len(net["steps"]) == 4
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
