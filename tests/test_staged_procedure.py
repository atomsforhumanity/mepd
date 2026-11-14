"""procedure="staged" of the network expansion (docs/network-expansion.md, "The staged procedure"): the access
cutoff on the explorer's free energies, steering single points for the deciding window only. The classic
procedure is what every other test pins."""
import numpy as np
import pytest

import mepd
from mepd.discovery import network_expansion as NE
from mepd.discovery.network_expansion import expand_network
from test_network_expansion import _AsyncConnector, _GraphEngine, _node, _SteerEngine

KCAL = 627.5094740631


class _FiledAsync(_AsyncConnector):
    """_AsyncConnector whose TSs lie `ts_kcal` above their higher end and are written to files (the free
    energies read a TS from its file)."""

    def __init__(self, folder, ts_kcal=5.0, verified=True, kind=None):
        super().__init__()
        self.folder, self.ts_kcal, self.verified, self.kind = folder, ts_kcal, verified, kind

    def _found(self, i, j, nodes):
        rec = super()._found(i, j, nodes)
        up = self.ts_kcal(i, j) if callable(self.ts_kcal) else self.ts_kcal
        ts = rec["ts"]     # (the later species' geometry moved 50 Bohr along x: what _SteerEngine tells a TS by)
        ts.structure = ts.structure.model_copy(
            update={"geometry": np.asarray(nodes[max(i, j)].coords) + [50.0 + len(self.pairs), 0.0, 0.0]})
        ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + up / KCAL
        fp = self.folder / f"ts_{i}_{j}_{len(self.pairs)}.xyz"
        mepd.write_xyz(fp, rec["ts"], comments=[f"E={rec['ts'].energy:.10f}"])
        return {**rec, "label": fp.stem, "files": {"ts": str(fp), "verified": self.verified,
                                                   **({"ts_kind": self.kind} if self.kind else {})}}


def _thermal(monkeypatch, ts_kcal, species_kcal=0.0):
    """Thermal corrections: `ts_kcal` for every TS, `species_kcal` for every species; returns the list of
    structures a Hessian was asked for (True: a TS)."""
    from mepd.discovery.free_energy import FreeEnergies

    asked = []

    def correction(self, structure, ts, ends=None, energy=None):
        asked.append(bool(ts))
        return float(ts_kcal if ts else species_kcal(structure) if callable(species_kcal) else species_kcal)

    monkeypatch.setattr(FreeEnergies, "_correction", correction)
    return asked


def _run(tmp_path, procedure, cutoff, ts_kcal=5.0, **kw):
    return expand_network(_node("C#N"), _GraphEngine(), connect=_FiledAsync(tmp_path, ts_kcal), steer="access",
                          access_cutoff_kcal=cutoff, max_products=5, schedule="async", allow_zwitterions=True,
                          kinetics={"threshold": 1e-6, "free_energy": True}, procedure=procedure, **kw)


def test_staged_needs_the_async_queue_and_free_energy_kinetics(tmp_path):
    with pytest.raises(ValueError, match="procedure must be"):
        _run(tmp_path, "other", 30.0)
    with pytest.raises(ValueError, match="free-energy kinetics"):
        expand_network(_node("C#N"), _GraphEngine(), connect=_FiledAsync(tmp_path), steer="access",
                       access_cutoff_kcal=30.0, schedule="async", procedure="staged")


def test_a_ts_over_the_cutoff_on_electronic_energy_and_under_it_on_free_energy_stays_open(monkeypatch, tmp_path):
    """TSs 5 kcal/mol up, cutoff 3, thermal correction -4: closed by the classic gate (electronic), open in the
    staged one (free energy 1), with the barrier the kinetics used."""
    _thermal(monkeypatch, -4.0)
    classic = _run(tmp_path / "c", "classic", 3.0) if (tmp_path / "c").mkdir() is None else None
    assert classic.steps and all(r["closed"] is not None for r in classic.record["steps"])
    staged = _run(tmp_path, "staged", 3.0)
    open_ = [r for r in staged.record["steps"] if r["closed"] is None]
    assert open_ and all(b is not None for b in staged.kinetics.barriers_kcal[:1])
    st, rec = next((s, r) for s, r in zip(staged.steps, staged.record["steps"]) if r["closed"] is None and s.a == 0)
    e = [float(s.node.energy) for s in staged.species]
    assert rec["free_energy_barrier_kcal"][0] == pytest.approx((st.ts_energy - e[0]) * KCAL - 4.0)
    assert 1 in staged.rounds[0]["sources"] and 1 not in classic.rounds[0]["sources"]   # and its product expanded


def test_the_gate_measures_free_energies_from_the_seeds(monkeypatch, tmp_path):
    """Every structure 50 kcal/mol up in thermal correction, the seed too: the heights above the seed are the
    electronic ones, and a TS 5 up is open under a cutoff of 30."""
    _thermal(monkeypatch, 50.0, 50.0)
    res = _run(tmp_path, "staged", 30.0)
    assert res.steps and all(r["closed"] is None for r in res.record["steps"])


def test_only_irc_verified_transition_states_are_in_a_window(monkeypatch, tmp_path):
    """A step no IRC checked, and a path's highest image standing in for a TS, are in no deciding window: no
    single point but the seed's is asked for, and the open ones are on a lent shift."""
    _thermal(monkeypatch, 0.0)
    for name, kw in (("unverified", {"verified": False}), ("peak", {"kind": "path_peak"})):
        (tmp_path / name).mkdir()
        high = _SteerEngine()
        res = expand_network(_node("C#N"), _GraphEngine(), connect=_FiledAsync(tmp_path / name, **kw), steer="access",
                             access_cutoff_kcal=30.0, max_products=5, schedule="async", allow_zwitterions=True,
                             kinetics={"threshold": 1e-6, "free_energy": True}, procedure="staged",
                             steer_engine=high, steer_save_dir=tmp_path / name / "steer")
        assert res.steps and all(r["deciding_window"] is None for r in res.record["steps"]), name
        assert res.record["staged"]["single_points"] == {"species": 1, "ts": 0, "computed": 1}, name
        assert all(r["shift_lent"] for r in res.record["steps"] if r["closed"] is None), name


def test_a_ts_under_the_cutoff_on_electronic_energy_and_over_it_on_free_energy_is_closed(monkeypatch, tmp_path):
    """The reverse: TSs 5 up, cutoff 8, thermal correction +6: open in the classic gate, closed in the staged
    one, the record saying on which energy."""
    _thermal(monkeypatch, 6.0)
    (tmp_path / "c").mkdir()
    classic = _run(tmp_path / "c", "classic", 8.0)
    assert any(r["closed"] is None for r in classic.record["steps"])
    staged = _run(tmp_path, "staged", 8.0)
    assert staged.steps and all(r["closed"] is not None for r in staged.record["steps"])
    closed = staged.record["steps"][0]["closed"]
    assert closed["reason"] == "ts_above_cutoff" and not closed.get("provisional")
    assert closed["value_kcal"] > 8.0 == closed["cutoff_kcal"] and closed["energy_used"].startswith("free")
    assert staged.record["staged"]["provisional_closures"]["steps"] == []


def test_a_structure_without_a_hessian_is_asked_for_one_within_the_margin_and_closed_provisionally_above_it(
        monkeypatch, tmp_path):
    """No thermal correction yet (the stub queue computes none): 10 above the seed with cutoff 3 is within the
    cutoff + 15, so its Hessian is computed and the free energy (10 - 8) opens it; 30 above is closed
    provisionally, without a Hessian, and the summary says so."""
    asked = _thermal(monkeypatch, -8.0)
    res = _run(tmp_path, "staged", 3.0, ts_kcal=10.0)
    assert any(asked) and any(r["closed"] is None for r in res.record["steps"])
    asked.clear()
    (tmp_path / "far").mkdir()
    res = _run(tmp_path / "far", "staged", 3.0, ts_kcal=30.0)
    assert not any(asked)                                   # no TS Hessian was computed
    assert res.steps and all(r["closed"] and r["closed"]["provisional"] for r in res.record["steps"])
    closed = res.record["steps"][0]["closed"]
    assert closed["cutoff_kcal"] == 3.0 + NE.PROVISIONAL_MARGIN_KCAL and closed["value_kcal"] > closed["cutoff_kcal"]
    assert "electronic" in closed["energy_used"]
    assert res.record["staged"]["provisional_closures"]["steps"] == [st.label for st in res.steps]


def test_a_proposed_products_free_energy_decides_whether_its_pair_is_searched(monkeypatch, tmp_path):
    """HNC lies at HCN's electronic energy, within the cutoff + 15: its Hessian is a job of the queue before its
    pair is searched. 40 kcal/mol up in free energy (cutoff 30) the pair is neither searched nor sampled, and the
    reaction says why; at the seed's free energy it is searched. Classic asks for no such job."""
    from mepd.discovery.network_expansion import graph_edges
    from mepd.nodes.node import StructureNode

    seed_bonds = graph_edges(_node("C#N"))
    up = [40.0]
    asked = _thermal(monkeypatch, 0.0, lambda s: 0.0 if graph_edges(StructureNode(structure=s)) == seed_bonds else up[0])
    runs = {}
    for name, procedure in (("closed", "staged"), ("open", "staged"), ("classic", "classic")):
        (tmp_path / name).mkdir()
        up[0] = 40.0 if name == "closed" else 0.0
        connect = _FiledAsync(tmp_path / name)
        res = expand_network(_node("C#N"), _GraphEngine(), connect=connect, steer="access", access_cutoff_kcal=30.0,
                             max_products=5, schedule="async", allow_zwitterions=True, procedure=procedure,
                             kinetics={"threshold": 1e-6, "free_energy": True})
        runs[name] = (res, connect)
    res, connect = runs["closed"]
    from mepd.discovery.horizontal import SEED_RANK

    # a seed pair's product: its Hessian job is of the kind that ranks with the seed's own reactions
    assert ("thermo_seed", "species 1") in [(kind, label) for kind, label, _ in connect.at_submit]
    assert connect.ranks["thermo_seed"] == SEED_RANK and connect.ranks["thermo"] > SEED_RANK
    assert connect.pairs == [] and res.steps == [] and len(res.species) == 2
    (edge,) = [e for e in res.edges if e.target == 1]
    assert edge.not_searched["reason"] == "product_above_cutoff" and edge.not_searched["value_kcal"] == \
        pytest.approx(40.0, abs=1.0)
    assert edge.not_searched["energy_used"].startswith("free") and "provisional" not in edge.not_searched
    assert res.species[1].loose is False                       # the tight minimum its Hessian was taken at
    res, connect = runs["open"]
    assert connect.pairs[0] == (0, 1) and res.steps
    res, connect = runs["classic"]
    assert all(not kind.startswith("thermo") for kind, _, _ in connect.at_submit) and connect.pairs[0] == (0, 1)
    assert "thermo" not in connect.ranks


# --- step 2: steering single points for the deciding window only ---------------------------------------------

def test_the_deciding_window_is_each_populated_species_steps_near_its_lowest_barrier():
    g = [0.0, -10.0 / KCAL, 0.0]
    steps = [(0, 0, 1, 50.0 / KCAL), (1, 0, 2, 55.0 / KCAL), (2, 0, 2, 70.0 / KCAL), (3, 1, 2, 30.0 / KCAL)]
    width = 1.987204259e-3 * 298.15 * np.log(100.0) + NE.LEVEL_MARGIN_KCAL
    w = NE.deciding_window(steps, g, {0}, 298.15)
    assert set(w) == {0, 1} and w[0]["width_kcal"] == pytest.approx(width)       # 70 is 20 over the lowest
    assert w[1]["of"] == [{"species": 0, "barrier_kcal": pytest.approx(55.0), "lowest_kcal": pytest.approx(50.0)}]
    # species 1 populated too: its lowest barrier is step 3's (40); step 0 (60 from it) is in the seed's window only
    w = NE.deciding_window(steps, g, {0, 1}, 298.15)
    assert set(w) == {0, 1, 3} and [x["species"] for x in w[0]["of"]] == [0] and w[3]["of"][0]["barrier_kcal"] == \
        pytest.approx(40.0)
    # the window widens with the temperature (RT ln 100), not the margin: at 1500 K 19.7, still under 20; at 1600 K over
    assert set(NE.deciding_window(steps, g, {0}, 1500.0)) == {0, 1}
    assert set(NE.deciding_window(steps, g, {0}, 1600.0)) == {0, 1, 2}
    assert NE.deciding_window(steps, g, {5}, 298.15) == {}                        # a species without steps


def test_a_lent_shift_is_the_median_of_the_refined_neighbours_never_zero_beside_them():
    known = {1: 2.0, 2: 4.0, 5: -10.0}
    lent = NE.lent_shifts(known, {3: [1, 2], 4: [9], 6: [], 7: [1, 2, 5]})
    assert lent == {3: 3.0, 4: 2.0, 6: 2.0, 7: 2.0}      # no refined neighbour: the median of all that are
    assert NE.lent_shifts({}, {3: [1]}) == {3: 0.0}       # nothing refined at all


def test_a_species_on_a_lent_shift_is_flagged_on_its_steps_and_the_kinetics_use_the_shift(monkeypatch, tmp_path):
    """A species without a single point is lent a shift (here 7 for every structure that is lent one): the
    kinetics' barrier from it carries that shift, and its open steps name it (end_shift_lent). The windows are
    those of every populated species, not the seed's alone."""
    _thermal(monkeypatch, 0.0)
    monkeypatch.setattr(NE, "lent_shifts", lambda known, neighbours: {i: 7.0 / KCAL for i in neighbours})
    res = _ladder(tmp_path, "staged", _SteerEngine())
    rec, sp = res.record["steps"], res.record["species"]
    lent = {k for k, r in enumerate(sp) if r["energy_terms"].get("shift_lent")}
    assert lent and all(sp[k]["energy_terms"]["steer_shift_kcal"] == pytest.approx(7.0) for k in lent)
    e = [float(x.node.energy) for x in res.species]
    seen = 0
    for st, r in zip(res.steps, rec):
        if r["closed"] is not None:
            assert r["end_shift_lent"] is None
            continue
        assert r["end_shift_lent"] == [k for k in (st.a, st.b) if k in lent]
        for side, k in enumerate((st.a, st.b)):
            if k in lent:
                seen += 1
                want = ((st.ts_energy - e[k]) * KCAL + r["ts_energy_terms"]["steer_shift_kcal"]
                        - sp[k]["energy_terms"]["steer_shift_kcal"])
                assert r["free_energy_barrier_kcal"][side] == pytest.approx(want, abs=0.5)
    assert seen
    of = {x["species"] for r in rec if r["deciding_window"] for x in r["deciding_window"]["of"]}
    assert 0 in of and of - {0}


def _ladder(tmp_path, procedure, high):
    """Ethanol's products from the seed with TSs 15, 25, 35, ... kcal/mol over their higher end: at 298 K the
    window (RT ln 100 + 6 = 8.7) holds the first one."""
    return expand_network(_node("CCO"), _GraphEngine(), steer="access", access_cutoff_kcal=60.0, max_products=8,
                          connect=_FiledAsync(tmp_path, lambda i, j: 5.0 + 10.0 * max(i, j)), schedule="async",
                          kinetics={"threshold": 2.0, "free_energy": True}, procedure=procedure,
                          steer_engine=high, steer_save_dir=tmp_path / "steer")


def test_single_points_go_to_the_seed_and_the_deciding_window_and_the_rest_is_lent_a_shift(monkeypatch, tmp_path):
    """Staged: single points for the seed, the window's TSs and their ends; an open step outside the window is on
    the median shift of the refined TSs out of its species (here -55, _SteerEngine's shift of every TS) and
    flagged; classic computes one per structure in reach."""
    _thermal(monkeypatch, 0.0)
    (tmp_path / "c").mkdir()
    all_points = _SteerEngine()
    classic = _ladder(tmp_path / "c", "classic", all_points)
    high = _SteerEngine()
    res = _ladder(tmp_path, "staged", high)
    rec = res.record["steps"]
    window = [m for m, r in enumerate(rec) if r["deciding_window"]]
    out = [m for m, r in enumerate(rec) if r["closed"] is None and not r["deciding_window"]]
    assert window and out
    # the seed's window: its lowest step alone (the next is 10 kcal/mol up)
    assert [m for m in window if any(x["species"] == 0 for x in rec[m]["deciding_window"]["of"])] == [0]
    for m in window:
        assert rec[m]["ts_energy_terms"]["steer"] == "single_point" and rec[m]["shift_lent"] is False
        assert rec[m]["ts_energy_terms"]["steer_shift_kcal"] == pytest.approx(-55.0)
        for k in (res.steps[m].a, res.steps[m].b):
            assert res.record["species"][k]["energy_terms"]["steer"] == "single_point"
    # outside the window: on a lent shift, unless an earlier window (fewer steps known) asked for its single point
    lent = [m for m in out if rec[m]["ts_energy_terms"]["steer"] == "lent"]
    assert lent and all(rec[m]["ts_energy_terms"]["steer"] == "single_point" for m in out if m not in lent)
    for m in lent:
        terms = rec[m]["ts_energy_terms"]
        assert terms["shift_lent"] is True and rec[m]["shift_lent"] is True
        assert terms["steer_shift_kcal"] == pytest.approx(-55.0)       # not 0 beside the shifted ones
    staged = res.record["staged"]
    points = {"species": sum(r["energy_terms"]["steer"] == "single_point" for r in res.record["species"]),
              "ts": sum(r["ts_energy_terms"]["steer"] == "single_point" for r in rec)}
    assert {k: staged["single_points"][k] for k in points} == points and points["ts"] >= len(window)
    assert staged["single_points"]["computed"] >= points["species"] + points["ts"]
    assert staged["shift_lent"]["steps"] == [res.steps[m].label for m in lent]
    assert staged["deciding_window_steps"] == [res.steps[m].label for m in window]
    assert high.calls < all_points.calls           # (one call of the higher level per single point)
    assert len(res.steps) >= len(classic.steps) > 0
    assert all(r["closed"] is None or r["shift_lent"] is None for r in rec)
    lent_species = [k for k, r in enumerate(res.record["species"]) if r["energy_terms"].get("shift_lent")]
    assert lent_species == staged["shift_lent"]["species"] and lent_species


def test_the_windows_single_points_start_before_the_seeds_own_jobs(monkeypatch, tmp_path):
    from mepd.discovery.horizontal import SEED_RANK

    _thermal(monkeypatch, 0.0)
    ranks = {}
    for procedure in ("classic", "staged"):
        (tmp_path / procedure).mkdir()
        connect = _FiledAsync(tmp_path / procedure)
        expand_network(_node("C#N"), _GraphEngine(), steer="access", access_cutoff_kcal=60.0, max_products=5,
                       connect=connect, schedule="async", allow_zwitterions=True, procedure=procedure,
                       kinetics={"threshold": 1e-6, "free_energy": True}, steer_engine=_SteerEngine(),
                       steer_save_dir=tmp_path / procedure / "steer")
        ranks[procedure] = connect.ranks["steer"]
    assert ranks["classic"] == 0 and ranks["staged"] < SEED_RANK


class _VerifyingQueue(_AsyncConnector):
    """The queue with IRC checks: every pair's TS is written unverified to a file (5 kcal/mol over its ends);
    `ends` keeps what its IRC will reach."""

    def __init__(self, folder):
        super().__init__()
        self.folder, self.ends = folder, {}

    def _found(self, i, j, nodes):
        rec = super()._found(i, j, nodes)
        fp = self.folder / f"fast_{i}_{j}.xyz"
        mepd.write_xyz(fp, rec["ts"], comments=[f"E={rec['ts'].energy:.10f}"])
        self.ends[str(fp)] = (nodes[i].copy(), nodes[j].copy())
        return {**rec, "pair": (i, j), "label": fp.stem, "files": {"ts": str(fp), "verified": False, "pair": [i, j]}}


def test_the_window_is_read_on_the_populated_set_of_the_same_kinetics(monkeypatch, tmp_path):
    """The kinetics that first populate HNC also give its window: the window is made again on the populated set
    they leave, with the same steps, not at the next kinetics."""
    from mepd.discovery import molecules

    calls, real, simulate = [], NE.deciding_window, molecules.simulate

    def window(steps, g, populated, temperature, *a):
        calls.append((len(steps), set(populated)))
        return real(steps, g, populated, temperature, *a)

    def solved(*a, **kw):
        calls.append("kinetics")
        return simulate(*a, **kw)

    monkeypatch.setattr(NE, "deciding_window", window)
    monkeypatch.setattr(molecules, "simulate", solved)
    _thermal(monkeypatch, 0.0)
    _run(tmp_path, "staged", 30.0)
    first = next(k for k, c in enumerate(calls) if c != "kinetics" and c[1] == {0, 1})
    n = calls[first][0]
    # the seed's window, the kinetics that populate HNC, then at once: the window on {0, 1} (it moved) and the
    # view made again with it, before the next kinetics
    assert n == 1 and calls[first - 2:first + 3] == [(1, {0}), "kinetics", (1, {0, 1}), (1, {0, 1}), "kinetics"]


def test_an_irc_check_makes_the_kinetics_look_again_so_the_window_gets_its_single_points(monkeypatch, tmp_path):
    """Nothing follows the IRC check of the only step (no conformer pair, no rotamer): the check itself must
    make the live kinetics look again, so the now verified TS's single point is a job of the queue."""
    import time
    from pathlib import Path

    _thermal(monkeypatch, 0.0)
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)     # (no live kinetics held back for coming too soon)
    connect = _VerifyingQueue(tmp_path)

    def verify(items):
        out = []
        for idx, fp in items:
            start, end = connect.ends[fp]
            mepd.write_xyz(Path(fp).with_name(f"{Path(fp).stem}_irc.xyz"), [start, end])
            out.append((idx, start.copy(), end.copy()))
        return out

    refine = lambda items: []
    refine.pairs, refine.width = (lambda item: {"start": [], "end": [], "pairs": []}), 1
    res = expand_network(_node("C#N"), _GraphEngine(), connect=connect, verify=verify, refine=refine, steer="access",
                         access_cutoff_kcal=30.0, max_products=5, schedule="async", allow_zwitterions=True,
                         kinetics={"threshold": 1e-6, "free_energy": True}, procedure="staged",
                         steer_engine=_SteerEngine(), steer_save_dir=tmp_path / "steer")
    assert [st.label for st in res.steps] == ["fast_0_1"] and res.record["steps"][0]["deciding_window"]
    assert sum(kind == "steer" and label.startswith("ts ") for kind, label, _ in connect.at_submit) == 1


def test_a_resumed_run_asks_again_for_the_hessian_of_a_product_whose_pair_waited(monkeypatch, tmp_path):
    """A staged run killed while a proposed product's Hessian job was out: resumed, the pair waits for the
    Hessian again (not searched on the provisional rule), and the product's free energy closes it."""
    from mepd.discovery.network_expansion import graph_edges
    from mepd.nodes.node import StructureNode

    class Killed(BaseException):
        pass

    class Dying(_FiledAsync):
        def _own(self, kind, args):
            if kind.startswith("thermo"):
                raise Killed()
            return super()._own(kind, args)

    seed_bonds = graph_edges(_node("C#N"))
    _thermal(monkeypatch, 0.0, lambda s: 0.0 if graph_edges(StructureNode(structure=s)) == seed_bonds else 40.0)
    run = lambda connect, **kw: expand_network(
        _node("C#N"), _GraphEngine(), connect=connect, steer="access", access_cutoff_kcal=30.0, max_products=5,
        schedule="async", allow_zwitterions=True, procedure="staged", state_file=tmp_path / "state.pkl",
        kinetics={"threshold": 1e-6, "free_energy": True}, **kw)
    with pytest.raises(Killed):
        run(Dying(tmp_path))
    connect = _FiledAsync(tmp_path)
    res = run(connect, resume=True)
    assert ("thermo_seed", "species 1") in [(kind, label) for kind, label, _ in connect.at_submit]
    assert connect.pairs == [] and res.steps == []
    (edge,) = [e for e in res.edges if e.target == 1]
    assert edge.not_searched["reason"] == "product_above_cutoff" and edge.not_searched["energy_used"].startswith("free")


def test_a_product_over_the_cutoff_on_its_electronic_energy_but_within_the_margin_gets_its_hessian(monkeypatch, tmp_path):
    """Cutoff -5: HNC, at HCN's electronic energy, is over it but within the 15 kcal/mol margin, so it is not
    closed provisionally: its Hessian is asked for, and the free energy (0 above the seed's) closes it."""
    _thermal(monkeypatch, 0.0)
    connect = _FiledAsync(tmp_path)
    res = expand_network(_node("C#N"), _GraphEngine(), connect=connect, steer="access", access_cutoff_kcal=-5.0,
                         max_products=5, schedule="async", allow_zwitterions=True, procedure="staged",
                         kinetics={"threshold": 1e-6, "free_energy": True})
    assert ("thermo_seed", "species 1") in [(kind, label) for kind, label, _ in connect.at_submit]
    (edge,) = [e for e in res.edges if e.target == 1]
    assert edge.not_searched["energy_used"].startswith("free") and "provisional" not in edge.not_searched
    assert edge.not_searched["cutoff_kcal"] == -5.0 and connect.pairs == []
