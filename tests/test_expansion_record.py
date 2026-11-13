"""What a run records for the measuring instruments (docs/network-expansion.md, "What a run records"): each
step's free-energy barrier on the kinetics' basis, why the kinetics leave a step out and the numbers compared, IRC
outcomes, why a path peak stands in for a TS, when species and steps were found and first used, and which energies
the answer rests on. Recording only: the networks these runs find are the ones the other tests pin."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure
from typer.testing import CliRunner

import mepd.cli_common as cli_common
from mepd.cli import app
from mepd.discovery import molecules as M
from mepd.discovery.kinetics import Step
from mepd.discovery.network_expansion import expand_network
from mepd.inputs import RunInputs
from test_molecules import A2B, H, _parts_net
from test_network_expansion import (
    _connector, _filed_connector, _GraphEngine, _node, _run_unchecked, _ShiftEngine, _SteerEngine,
    _ts_file_connector, _unverified_connector,
    test_a_path_piece_whose_ts_belongs_to_another_motion_keeps_its_path_peak as _path_peak_run,
)

KCAL = 627.5094740631


def _zero_corrections(monkeypatch):
    from mepd.discovery.free_energy import FreeEnergies

    monkeypatch.setattr(FreeEnergies, "corrections",
                        lambda self, structures, ts, ends=None, energies=None: [0.0] * len(structures))


def test_an_open_step_has_its_free_energy_barrier_and_a_step_out_of_reach_says_why(monkeypatch, tmp_path):
    """Open: the barrier from each end on the kinetics' basis (what the rates used, unclamped), no closing record.
    Closed by the reach test: no barrier, what the run never computed, and the reason with the energy compared,
    its value and the cutoff."""
    _zero_corrections(monkeypatch)
    run = lambda cutoff: expand_network(_node("C#N"), _ShiftEngine(), rounds=1, allow_zwitterions=True,
                                        steer="access", connect=_filed_connector(5.0, tmp_path),
                                        access_cutoff_kcal=cutoff, kinetics={"free_energy": True})
    res = run(30.0)
    assert len(res.record["steps"]) == len(res.steps) and len(res.record["species"]) == len(res.species)
    for rec, used in zip(res.record["steps"], res.kinetics.barriers_kcal):
        assert rec["closed"] is None and rec["free_energy_barrier_missing"] is None
        assert rec["free_energy_barrier_kcal"] == pytest.approx(list(used))
        assert rec["ts_energy_terms"] == {"thermal_correction_kcal": 0.0, "steer": None, "steer_shift_kcal": None}
        assert rec["t_opened"] >= rec["t_found"] > 0 and rec["ts_group"] == 0
    assert all(sp["energy_terms"]["thermal_correction_kcal"] == 0.0 for sp in res.record["species"])
    res = run(3.0)
    (st,), (rec,) = res.steps, res.record["steps"]
    e0 = float(res.species[0].node.energy)
    assert res.kinetics.barriers_kcal[0] is None and rec["free_energy_barrier_kcal"] is None
    assert rec["free_energy_barrier_missing"] == [f"thermal correction of species {st.b}"]
    closed = rec["closed"]
    assert closed["reason"] == "end_out_of_reach" and closed["by"] == "box kinetics" and closed["species"] == st.b
    assert closed["cutoff_kcal"] == 3.0 and closed["value_kcal"] > 3.0 and "electronic" in closed["energy_used"]
    assert closed["ts_above_seed_kcal"] == pytest.approx((st.ts_energy - e0) * KCAL)
    assert rec["t_opened"] is None


def test_a_ts_above_the_cutoff_between_species_within_reach_says_so(monkeypatch, tmp_path):
    """Two TSs between one pair, one above the cutoff: that step is closed on its own electronic TS energy, with
    the value and the cutoff; its barrier is not made up (its TS's thermal correction was never computed)."""
    import mepd

    _zero_corrections(monkeypatch)
    low = _filed_connector(5.0, tmp_path)

    def connect(pairs, nodes):
        out = low(pairs, nodes)
        for i, j in pairs:
            ts = nodes[i].copy()
            ts.structure = ts.structure.model_copy(update={"geometry": np.asarray(ts.coords) + [0.0, 9.0, 0.0]})
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + 40.0 / KCAL
            fp = tmp_path / f"high_{i}_{j}.xyz"
            mepd.write_xyz(fp, ts)
            out.append({"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"high-{i}-{j}", "files": {"ts": str(fp)}})
        return out

    res = expand_network(_node("C#N"), _ShiftEngine(), rounds=1, allow_zwitterions=True, steer="access",
                         connect=connect, access_cutoff_kcal=30.0, kinetics={"free_energy": True})
    m = next(m for m, st in enumerate(res.steps) if st.label.startswith("high"))
    rec, e0 = res.record["steps"][m], float(res.species[0].node.energy)
    assert rec["closed"]["reason"] == "ts_above_cutoff" and rec["closed"]["cutoff_kcal"] == 30.0
    assert rec["closed"]["value_kcal"] == pytest.approx((res.steps[m].ts_energy - e0) * KCAL)
    assert rec["free_energy_barrier_kcal"] is None
    assert rec["free_energy_barrier_missing"] == ["thermal correction of the TS"]
    assert [r["closed"] for n, r in enumerate(res.record["steps"]) if n != m] == [None] * (len(res.steps) - 1)


def test_a_failed_irc_is_recorded_with_its_reason_and_a_checked_step_with_its_ends(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fast_0_1_irc.failed").write_text("RuntimeError: the IRC did not converge\n")
    res = expand_network(_node("C#N"), _GraphEngine(), rounds=1, allow_zwitterions=True, steer="access",
                         connect=_unverified_connector(5.0), verify=lambda items: [(i, None, None) for i, _ in items],
                         access_cutoff_kcal=30.0)
    assert res.steps == []
    (gone,) = res.steps_removed
    assert gone["reason"] == "irc_failed" and gone["label"] == "fast_0_1" and (gone["a"], gone["b"]) == (0, 1)
    assert gone["irc"]["ok"] is False and gone["irc"]["why"] == "RuntimeError: the IRC did not converge"
    holder = {}

    def verify(items):
        return [(i, holder["r"].species[0].node.copy(), holder["r"].species[1].node.copy()) for i, _ in items]

    import mepd.discovery.network_expansion as ne

    class Capture(ne.ExpansionResult):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder["r"] = self

    monkeypatch.setattr(ne, "ExpansionResult", Capture)
    res = expand_network(_node("C#N"), _GraphEngine(), rounds=1, allow_zwitterions=True, steer="access",
                         connect=_unverified_connector(5.0), verify=verify, access_cutoff_kcal=30.0)
    (rec,) = res.record["steps"]
    assert rec["irc"]["ok"] is True and rec["irc"]["ends"] == [0, 1] and rec["irc"]["new_species"] == []
    assert rec["irc"]["t"] >= rec["t_found"] and res.steps_removed == []


def test_a_path_peak_says_why_it_stands_in_for_a_ts():
    """A piece whose leaf TS's IRC ended in one species keeps its path's highest image: the step says so, and the
    TSs that went (the pair's single TS whose IRC failed, the leaf TS) are in steps_removed with their reasons."""
    res = _path_peak_run(refine=lambda items: [None for _ in items])
    assert res.steps and all(st.files.get("ts_kind") == "path_peak" for st in res.steps)
    for st, rec in zip(res.steps, res.record["steps"]):
        why = rec["placeholder_reason"]
        assert why["why"] == "leaf_ts_irc_went_elsewhere" and why["irc_ends"] == [st.a, st.a]
        assert why["ts"] == st.label.removesuffix("_peak")
    reasons = {r["label"]: r["reason"] for r in res.steps_removed}
    assert set(reasons.values()) == {"irc_failed", "irc_ends_one_species"}
    assert all(reasons[st.label.removesuffix("_peak")] == "irc_ends_one_species" for st in res.steps)


class _FailingSteer(_SteerEngine):
    """The steering level fails on the TSs of _ts_file_connector (pushed far along +x)."""

    def compute_energies(self, nodes):
        if any(float(np.asarray(n.coords)[0, 0]) > 40.0 for n in nodes):
            raise RuntimeError("SCF not converged")
        return super().compute_energies(nodes)


def test_a_failed_steering_single_point_is_recorded_for_the_structure_it_left_on_the_explorers_energy(tmp_path):
    res = expand_network(_node("C#N"), _GraphEngine(), rounds=1, allow_zwitterions=True, steer="flux",
                         connect=_ts_file_connector(tmp_path, 5.0), steer_engine=_FailingSteer(),
                         steer_save_dir=tmp_path / "steer")
    (fallback,) = res.record["steer_fallbacks"]
    assert fallback["kind"] == "ts" and fallback["step"] == 0 and "SCF not converged" in fallback["why"]
    terms = res.record["steps"][0]["ts_energy_terms"]
    assert terms["steer"] == "failed" and terms["steer_shift_kcal"] is None and "SCF" in terms["steer_error"]
    assert [sp["energy_terms"]["steer"] for sp in res.record["species"]] == ["single_point", "single_point"]
    assert res.record["species"][0]["energy_terms"]["steer_shift_kcal"] == 0.0
    ok = expand_network(_node("C#N"), _GraphEngine(), rounds=1, allow_zwitterions=True, steer="flux",
                        connect=_ts_file_connector(tmp_path, 5.0), steer_engine=_SteerEngine(),
                        steer_save_dir=tmp_path / "steer2")
    assert ok.record["steer_fallbacks"] == []
    assert ok.record["steps"][0]["ts_energy_terms"]["steer_shift_kcal"] == pytest.approx(-55.0)


def test_a_step_no_irc_checked_is_closed_with_that_reason_and_keeps_the_barrier_it_had(monkeypatch, tmp_path):
    """Out of time: the unchecked step and the one only it reached are closed, each with its reason; their terms
    were all computed, so their barriers on the kinetics' basis are given beside the null kinetics barrier."""
    from mepd import search_control

    calls = []
    monkeypatch.setattr(search_control, "stop_requested", lambda: calls.append(1) or len(calls) > 2)
    res = _run_unchecked(tmp_path)
    reasons = {tuple(st.files.get("pair") or ()): rec["closed"] and rec["closed"]["reason"]
               for st, rec in zip(res.steps, res.record["steps"])}
    assert reasons[(0, 2)] == "no_irc_check" and reasons[(0, 3)] == "reached_only_through_unchecked_steps"
    rt_ln2 = 1.987204259e-3 * 298.15 * np.log(2.0)    # (a chiral conformer counts with its mirror image)
    for m, (st, rec) in enumerate(zip(res.steps, res.record["steps"])):
        used = res.kinetics.barriers_kcal[m]
        assert rec["free_energy_barrier_missing"] is None
        if used is not None:
            assert rec["free_energy_barrier_kcal"] == pytest.approx(list(used))
            assert rec["closed"] is None or rec["closed"]["by"] == "molecule kinetics"
        else:
            raw = [(st.ts_energy - float(res.species[k].node.energy)) * KCAL for k in (st.a, st.b)]
            assert rec["free_energy_barrier_kcal"] == pytest.approx(raw, abs=rt_ln2 + 1e-6)
            assert rec["closed"]["by"] == "box kinetics"
        assert rec["reacting"] is not None and rec["irc"] is None
    assert all(rec["molecules"] and sp.t_found > 0 for sp, rec in zip(res.species, res.record["species"]))


def test_a_product_above_the_cutoff_says_why_its_pair_was_not_searched():
    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="access", connect=_connector(5.0),
                         access_cutoff_kcal=0.1, max_products=20)
    (why,) = [e.not_searched for e in res.edges if e.not_searched]
    assert why["reason"] == "product_above_cutoff" and why["cutoff_kcal"] == 0.1 and why["value_kcal"] > 0.1
    assert all(e.t > 0 for e in res.edges)


def test_the_molecule_network_says_which_open_box_steps_are_no_reaction_of_it_and_why():
    boxes = [[((0, 1), "C")], [((0, 1), "C=")]]
    parts, bonds = _parts_net(boxes)
    steps = [Step(0, 1, 10.0 / H, "peak", {"ts_kind": "path_peak"}), Step(0, 1, 12.0 / H, "ts", {}),
             Step(0, 0, 3.0 / H, "conformers", {}), Step(0, 1, 400.0 / H, "far", {}),
             Step(0, 1, 9.0 / H, "closed", {"closed": True})]
    net = M.build(2, parts, bonds, steps, [0.0, 2.0 / H], ts_info=lambda m: (None, True, True))
    assert net.left_out == {0: {"reason": "path_peak"}, 2: {"reason": "no_reaction"},
                            3: {"reason": "ts_above_closed_threshold", "value_kcal": pytest.approx(398.0),
                                "cutoff_kcal": pytest.approx(0.5 * H)}}
    boxes = [[((0, 1), "C"), ((2,), "O")], [((0, 1), "C="), ((2,), "O")]]
    parts, bonds = _parts_net(boxes)
    ts = Structure(symbols=["C", "C", "O"], geometry=np.array([[0, 0, 0], [1.5, 0, 0], [2.6, 0, 0]]) * A2B)
    net = M.build(2, parts, bonds, [Step(0, 1, 10.0 / H, "s", {})], [0.0, 2.0 / H], ts_info=lambda m: (ts, False, False))
    assert net.left_out == {0: {"reason": "whole_box_ts_unchecked"}}
    assert "left_out" not in net.record()          # the summary's molecule record is as it was


def test_a_leaf_ts_that_failed_leaves_its_reason_on_the_path_peak(tmp_path):
    """The queue: a pair's fast TS fails, its path search's leaf TS fails too, and the piece is kept at its
    highest image; the record handed to the explorer says why (both errors)."""
    from mepd.discovery.horizontal import HorizontalConnector

    a, b = SimpleNamespace(name="A"), SimpleNamespace(name="B")
    hc = object.__new__(HorizontalConnector)
    hc.ts_dir, hc.workers, hc.max_pieces, hc.max_depth, hc.nimages = Path(tmp_path), 1, 40, 6, 11
    hc.say = lambda *a: None
    hc.job_inputs = SimpleNamespace(path_min_inputs=SimpleNamespace(barrier_cutoff_kcal=None, cutoff_temperature=None,
                                                                     network_floor_hartree=None))
    hc._lock, hc._running = threading.Lock(), SimpleNamespace(value=0)
    hc.iced, hc._pieces, hc._next_index, hc.free, hc._jobs, hc._live = [], {}, {}, None, {}, None
    hc.stop_when = None

    def fail(what):
        def job(*args):
            raise RuntimeError(what)
        return job

    hc._fast_ts, hc._leaf_ts = fail("fast TS: 2 imaginary modes"), fail("leaf TS: not converged")
    hc._root_chain = lambda a, b: ([a, b], None)
    hc._piece = lambda chain, index, depth: (SimpleNamespace(index=index, data=SimpleNamespace(chain_trajectory=[chain])), [])
    hc._path_peak = lambda chain, label, cut=None: SimpleNamespace(energy=-1.0)
    (rec,) = hc([(0, 1)], [a, b])
    assert rec["files"]["ts_kind"] == "path_peak"
    assert rec["placeholder_reason"] == {"why": "leaf_ts_failed", "error": "RuntimeError: leaf TS: not converged",
                                         "fast_ts_error": "RuntimeError: fast TS: 2 imaginary modes"}


def test_live_events_carry_their_time(tmp_path, monkeypatch):
    from mepd import progress

    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path))
    progress.append_live_event({"event": "species", "index": 1})
    (line,) = (tmp_path / "events.jsonl").read_text().splitlines()
    assert json.loads(line)["t"] > 0 and json.loads(line)["index"] == 1


def test_summary_carries_the_record(tmp_path, monkeypatch):
    """summary.json: the new fields beside the old ones, on every species, step and proposed reaction; the levels'
    identity; the partial summary's steps with their times."""
    import mepd.discovery.network_expansion as ne

    _zero_corrections(monkeypatch)
    real, partial = ne.expand_network, {}

    def expand(*a, **kw):
        def checkpoint(res):
            kw["checkpoint"](res)
            partial.update(json.loads((tmp_path / "out" / "summary_partial.json").read_text()))
        return real(*a, **{**kw, "connect": _filed_connector(5.0, tmp_path), "verify": None, "refine": None,
                           "checkpoint": checkpoint})

    def run_inputs(_path):
        ri = RunInputs()
        ri.engine = _ShiftEngine()
        return ri

    monkeypatch.setattr(cli_common, "_open_run_inputs", run_inputs)
    monkeypatch.setattr(ne, "expand_network", expand)
    profile = tmp_path / "profile.toml"
    profile.write_text('engine_name = "gxtb"\n')
    out = tmp_path / "out"
    res = CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "--steer", "access",
                                   "--barrier-cutoff", "30", "--kinetics-energy", "free", "-i", str(profile),
                                   "-o", str(out)])
    assert res.exit_code == 0, res.output
    summary = json.loads((out / "summary.json").read_text())
    for sp in summary["species"]:
        assert sp["t_found"] > 0 and set(sp["energy_terms"]) == {"thermal_correction_kcal", "steer", "steer_shift_kcal"}
        assert [m["atoms"] for m in sp["molecules"]] == [[0, 1, 2]]
    (st,) = summary["steps"]
    assert st["kinetics_barrier_kcal"] == pytest.approx(st["free_energy_barrier_kcal"]) and st["closed"] is None
    assert {"free_energy_barrier_missing", "ts_energy_terms", "reacting", "ts_group", "t_found", "t_opened", "irc",
            "placeholder_reason"} <= set(st)
    assert st["reacting"]["atoms"] == [0, 1, 2] and st["reacting"]["spectators"] == []
    assert all({"t", "batch", "proposed_key", "not_searched"} <= set(r) for r in summary["reactions"])
    assert summary["steps_removed"] == [] and "record_error" not in summary
    level = summary["steering"]["level"]
    assert level["steer"] is None and level["explorer"]["engine"] == "_ShiftEngine"
    import hashlib

    assert level["explorer"]["profile_sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    assert summary["steering"]["steer_fallbacks"] == []
    assert partial["steps"][0]["t_found"] == st["t_found"] and partial["species"][0]["t_found"] > 0
