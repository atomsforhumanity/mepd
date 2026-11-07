"""`mepd discovery expand --known <folder>`: a follow-up expansion takes what an earlier one at the same level
found and computed (species, steps, steering single points, Hessians) instead of computing it again."""
import json

import numpy as np
import pytest
from typer.testing import CliRunner

import mepd.cli_common as cli_common
from mepd.cli import app
from mepd.cli_common import _load_structure_from_smiles_or_xyz
from mepd.discovery.known import read_known
from mepd.discovery.network_expansion import expand_network
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode
from test_network_expansion import _AsyncConnector, _GraphEngine, _SteerEngine, _ts_file_connector


def _node(smiles):
    return StructureNode(structure=_load_structure_from_smiles_or_xyz(smiles, None, None))


def _write_known(folder, res, level="L1", free=False):
    """`res` as `discovery expand` leaves it in `folder` (the files read_known reads; the run itself kept its
    steering single points and state there)."""
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs

    (folder / "species").mkdir(parents=True, exist_ok=True)
    for k, sp in enumerate(res.species):
        Chain.model_validate({"nodes": [sp.node.copy()], "parameters": ChainInputs()}).write_to_disk(
            folder / "species" / f"species_{k}.xyz")
    summary = {
        "species": [{"index": k, "smiles": s.smiles, "energy": float(s.node.energy), "loosely_minimized": s.loose,
                     "validation": s.validation, "file": str(folder / "species" / f"species_{k}.xyz")}
                    for k, s in enumerate(res.species)],
        "reactions": [{"source": e.source, "target": e.target, "outcome": e.outcome} for e in res.edges],
        "steps": [{"a": st.a, "b": st.b, "ts_energy": st.ts_energy, "label": st.label, "files": st.files}
                  for st in res.steps],
        "rounds": res.rounds,
        "steering": {"kinetics_energy": "free" if free else "electronic", "temperature_K": 298.15,
                     "standard_state": "1M"}}
    (folder / "summary.json").write_text(json.dumps(summary, default=str))
    (folder / "provenance.json").write_text(json.dumps({"level_key": level, "argv": []}))


def _first_run(tmp_path, **kw):
    """HCN -> HNC, a TS 5 kcal/mol over HNC: flux into HNC, which is expanded too; its steering single points
    (2 species, 1 TS) and state kept in the run's folder."""
    folder = tmp_path / "run1"
    res = expand_network(_node("C#N"), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="flux",
                         connect=_ts_file_connector(tmp_path, 5.0), steer_engine=_SteerEngine(),
                         steer_save_dir=folder / "steer_single_points", state_file=folder / "expand_state.pkl", **kw)
    assert [s.smiles for s in res.species] == ["C#N", "[C-]#[NH+]"] and len(res.steps) == 1
    _write_known(folder, res, free=bool((kw.get("kinetics") or {}).get("free_energy")))
    return folder, res


class _Counting:
    def __init__(self, inner):
        self.inner, self.pairs = inner, []

    def __call__(self, pairs, nodes):
        self.pairs += list(pairs)
        return self.inner(pairs, nodes)


def _follow_up(tmp_path, seed, folders, level="L1", **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    runs, skipped = read_known(folders, level)
    eng, high = _GraphEngine(), _SteerEngine()
    connect = _Counting(_ts_file_connector(tmp_path, 5.0))
    res = expand_network(seed, eng, rounds=3, allow_zwitterions=True, steer="flux", connect=connect,
                         steer_engine=high, steer_save_dir=tmp_path / "run2" / "steer_single_points",
                         known={"runs": runs, "skipped": skipped, "dir": tmp_path / "run2" / "ts"}, **kw)
    return res, eng, high, connect


def test_a_follow_up_from_a_known_product_takes_its_species_steps_and_single_points(tmp_path):
    folder, first = _first_run(tmp_path)
    hnc = StructureNode(structure=first.species[1].node.structure)        # the product, as a seed of its own
    res, eng, high, connect = _follow_up(tmp_path, hnc, [folder])
    assert eng.optimized == 0          # the seed is a known species, HNC's proposal (HCN) too: no minimization
    assert connect.pairs == []         # the known step's pair gets no new TS search
    assert high.calls == 0             # the known single points are not computed again
    assert [s.smiles for s in res.species] == ["[C-]#[NH+]", "C#N"]
    assert res.species[0].node.energy == first.species[1].node.energy and res.species[1].known == str(folder)
    (st,) = res.steps                  # the reaction back to the earlier seed is in the network
    assert {st.a, st.b} == {0, 1} and st.files["known"] == str(folder) and st.label.startswith("known0_")
    assert st.ts_energy == first.steps[0].ts_energy
    assert res.known["reused"] == {"species": 1, "steps": 1, "single_points": 3, "hessians": 0}
    assert res.known["used"][0]["seed_is"] == 1 and res.known["used"][0]["atom_order"] == "same"
    assert res.rounds[0]["sources"] == [0]          # the seed is expanded; HCN, which the earlier run expanded, not
    assert res.kinetics.flux[1] > 0.01              # the kinetics run on the known network

    # without --known the same seed does it all again
    again, eng2, high2, connect2 = _follow_up(tmp_path / "plain", hnc, [])
    assert eng2.optimized > 0 and connect2.pairs and high2.calls == 3


def test_a_known_run_in_another_atom_order_is_mapped_onto_the_seed(tmp_path):
    folder, first = _first_run(tmp_path)
    s = first.species[1].node.structure
    order = [2, 0, 1]
    seed = StructureNode(structure=s.model_copy(update={"symbols": [s.symbols[i] for i in order],
                                                         "geometry": np.asarray(s.geometry)[order]}))
    res, eng, high, connect = _follow_up(tmp_path, seed, [folder])
    assert res.known["used"][0]["atom_order"] == "mapped"
    assert [str(x) for x in res.species[1].node.symbols] == [str(x) for x in seed.structure.symbols]
    assert eng.optimized == 0 and connect.pairs == [] and high.calls == 0    # its single points re-keyed too
    ts = StructureNode(structure=_load_structure_from_smiles_or_xyz(res.steps[0].files["ts"], 0, 1))
    assert [str(x) for x in ts.symbols] == [str(x) for x in seed.structure.symbols]


def test_a_run_at_another_level_is_skipped_and_said(tmp_path):
    folder, first = _first_run(tmp_path)
    hnc = StructureNode(structure=first.species[1].node.structure)
    res, eng, high, connect = _follow_up(tmp_path, hnc, [folder], level="L2")
    assert res.known["used"] == [] and "another level of theory" in res.known["skipped"][0]["why"]
    assert all(sp.known is None for sp in res.species) and connect.pairs and high.calls == 3


def test_a_run_whose_species_are_not_the_seed_is_skipped(tmp_path):
    folder, _ = _first_run(tmp_path)
    res, *_ = _follow_up(tmp_path, _node("CC#N"), [folder])
    assert res.known["used"] == [] and "atoms can't be mapped" in res.known["skipped"][0]["why"]


def test_known_hessians_are_not_computed_again(tmp_path, monkeypatch):
    from mepd.discovery.free_energy import FreeEnergies

    calls = []

    def correction(self, structure, ts, ends=None):
        calls.append(ts)
        return 1.0 if ts else 0.5

    monkeypatch.setattr(FreeEnergies, "_correction", correction)
    kin = {"free_energy": True}
    folder, first = _first_run(tmp_path, kinetics=kin)
    assert len(calls) == 3                         # 2 species, 1 TS
    calls.clear()
    res, *_ = _follow_up(tmp_path, StructureNode(structure=first.species[1].node.structure), [folder],
                         kinetics=kin)
    assert calls == [] and res.known["reused"]["hessians"] == 3
    assert res.rounds[0]["kinetics"]["energies"].startswith("free energy")
    calls.clear()
    other = {"free_energy": True, "temperature": 400.0}     # another temperature: its corrections don't apply
    res, *_ = _follow_up(tmp_path / "hot", StructureNode(structure=first.species[1].node.structure), [folder],
                         kinetics=other)
    assert calls and res.known["reused"]["hessians"] == 0 and "298.15 K" in res.known["used"][0]["hessians_skipped"]


def test_async_a_known_unverified_step_gets_its_irc_check_and_the_live_view_says_known(tmp_path, monkeypatch):
    folder, first = _first_run(tmp_path)
    summary = json.loads((folder / "summary.json").read_text())
    summary["steps"][0]["files"]["verified"] = False                    # a candidate only
    (folder / "summary.json").write_text(json.dumps(summary))
    live = tmp_path / "live"
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(live))
    checked = []

    def verify(items):
        checked.extend(ts for _, ts in items)
        return [(i, _node("[C-]#[NH+]"), _node("C#N")) for i, _ in items]

    runs, skipped = read_known([folder], "L1")
    connect = _AsyncConnector()
    res = expand_network(StructureNode(structure=first.species[1].node.structure), _GraphEngine(), steer="access",
                         connect=connect, verify=verify, access_cutoff_kcal=60.0, kinetics={"threshold": 1e-6},
                         schedule="async", allow_zwitterions=True,
                         known={"runs": runs, "skipped": skipped, "dir": tmp_path / "run2" / "ts"})
    assert checked == [str(tmp_path / "run2" / "ts" / "known0_0-1.xyz")]
    assert (0, 1) not in connect.pairs and (1, 0) not in connect.pairs
    assert res.steps[0].files["verified"] is True
    events = [json.loads(x) for x in (live / "events.jsonl").read_text().splitlines()]
    assert [e["index"] for e in events if e.get("event") == "species" and e.get("known")] == [1]
    assert [e["label"] for e in events if e.get("event") == "step" and e.get("known")] == ["known0_0-1"]


def test_resuming_with_other_known_runs_is_refused(tmp_path):
    from test_network_expansion import _LiveConnector

    folder, first = _first_run(tmp_path)
    runs, skipped = read_known([folder], "L1")
    hnc = StructureNode(structure=first.species[1].node.structure)
    kw = dict(rounds=2, steer="access", access_cutoff_kcal=60.0, allow_zwitterions=True,
              kinetics={"threshold": 1e-6}, state_file=tmp_path / "state.pkl")
    expand_network(hnc, _GraphEngine(), connect=_LiveConnector(), **kw,
                   known={"runs": runs, "skipped": skipped, "dir": tmp_path / "ts"})
    with pytest.raises(ValueError, match="known"):
        expand_network(hnc, _GraphEngine(), connect=_LiveConnector(), resume=True, **kw)
    # the same known runs resume
    expand_network(hnc, _GraphEngine(), connect=_LiveConnector(), resume=True, **kw,
                   known={"runs": runs, "skipped": skipped, "dir": tmp_path / "ts"})


def test_cli_known_folders_are_used_or_skipped_and_recorded(tmp_path, monkeypatch):
    from mepd.web.workspace import level_key

    def run_inputs(_path):
        ri = RunInputs()
        ri.engine = _GraphEngine()
        return ri

    monkeypatch.setattr(cli_common, "_open_run_inputs", run_inputs)
    assert "--known" in CliRunner().invoke(app, ["discovery", "expand", "--help"]).output
    one, other = tmp_path / "one", tmp_path / "other"
    for out in (one, other):
        res = CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "--rounds", "2",
                                       "--steer", "window", "-o", str(out)])
        assert res.exit_code == 0, res.output
    (one / "provenance.json").write_text(json.dumps({"level_key": level_key(None), "argv": []}))
    (other / "provenance.json").write_text(json.dumps({"level_key": "elsewhere", "argv": []}))
    out = tmp_path / "two"
    res = CliRunner().invoke(app, ["discovery", "expand", str(one / "species" / "species_1.xyz"), "--allow-zwitterions",
                                   "--steer", "window", "--known", str(one), "--known", str(other), "-o", str(out)])
    assert res.exit_code == 0, res.output
    summary = json.loads((out / "summary.json").read_text())
    assert [u["folder"] for u in summary["known"]["used"]] == [str(one)]
    assert summary["known"]["skipped"][0]["folder"] == str(other)
    assert summary["known"]["reused"]["species"] == 1
    assert summary["species"][1]["known"] == str(one) and "known" not in summary["species"][0]


def test_saved_steering_points_are_reused_with_other_thread_counts(tmp_path):
    from mepd.discovery.energy_refine import _Saved

    level = "GXTBCalculator[('backend', \"'axtb'\"), ('n_parallel', '8'), ('n_threads', '1')] -> OrcaEngine[('basis', \"'def2-TZVP'\")]"
    _Saved(tmp_path, level).put("single_points", "k", {"delta_kcal": 1.0})
    assert _Saved(tmp_path, level.replace("'8'", "'32'")).get("single_points", "k") == {"delta_kcal": 1.0}
    assert _Saved(tmp_path, level.replace("def2-TZVP", "def2-SVP")).get("single_points", "k") is None
