"""Add a molecule to a network (mepd/addmol.py): small networks with known answers."""
import math

import pytest

from mepd import addmol as A
from mepd import microkinetics as mk

T = 298.15
RT = mk.rt_kcal(T)
PRE = 6.2124e12                      # k_B T / h at 298.15 K, 1/s


def k_of(barrier):
    return PRE * math.exp(-barrier / RT)


def network(steps, species=("A", "B"), g=None):
    g = g or {}
    return {"temperature": T, "energies": "free",
            "species": [{"id": s, "name": s, "g": g.get(s, 0.0)} for s in species],
            "steps": [{"id": i, "label": i, "reactants": r, "products": p, "g_ts": ts, "status": "verified"}
                      for i, r, p, ts in steps]}


def both(x):
    return {"electronic": x, "free": x, "spread": 0.5}


PARENT = network([("ab", ["A"], ["B"], 30.0)], g={"B": -10.0})
C = {"id": "C", "name": "cat"}


def beside(r, ts, p, status="est"):
    """The molecule beside the step A -> B: its binding to the TS (an estimate), and to A and to B as verified
    bound species (an estimate of the molecule beside a held end makes no bound species)."""
    out = {"steps": [{"step": "ab", "status": status, "binding": {"r": both(r), "ts": both(ts), "p": both(p)}}]}
    if status == "est":
        out["species"] = [{"species": x, "status": "verified", "binding": both(b)} for x, b in (("A", r), ("B", p))]
    return out


def total(net, k, sid):
    """A species' amount at the end, free and in its complex."""
    out = k["res"].c[-1, k["pos"][sid]]
    cx = A.complex_id(sid, "C")
    return float(out + (k["res"].c[-1, k["pos"][cx]] if cx in k["pos"] else 0.0))


def test_a_catalyst_that_saturates():
    """A.C is the resting state: the rate is k_cat [C]total, whatever the amount of A."""
    v = A.added(PARENT, C, beside(-6.0, -12.0, -2.0))
    ids = {s["id"] for s in v["species"]}
    assert {"C", "A.C", "B.C"} <= ids
    k_cat, t = k_of(30.0 - 12.0 + 6.0), 1.0e4
    made = {}
    for a0, c0 in ((1.0, 0.01), (2.0, 0.01), (1.0, 0.005)):
        k = A.run(v, {"A": a0, "C": c0}, t)
        made[a0, c0] = total(v, k, "B")
        assert made[a0, c0] == pytest.approx(k_cat * c0 * t, rel=0.05)
    assert made[2.0, 0.01] == pytest.approx(made[1.0, 0.01], rel=0.02)          # zero order in A
    k = A.run(v, {"A": 1.0, "C": 0.01}, t)
    f = A.fate(v, k)
    assert f["fate"] == "regenerated" and f["resting_state"]["id"] == "A.C" and f["resting_state"]["share"] > 0.95
    assert f["returns"] == pytest.approx(k_cat * t, rel=0.05)
    # its apparent barrier is the TS above the resting complex (C's amount is in the complex's)
    j = next(i for i, st in enumerate(v["steps"]) if st["id"] == "ab+C")
    assert k["apparent"][j][0] == pytest.approx(24.0 - RT * math.log(0.01), abs=0.1)


def test_binding_to_the_reactant_is_not_counted_twice():
    """From dilute A the rate is (kT/h) exp(-(barrier + binding(ts)) / RT) [A][C]: the TS is counted from the
    separated molecules, so a deeper reactant complex does not make it faster."""
    t, a0, c0 = 1000.0, 1e-7, 1e-3
    made = []
    for b_r in (-2.0, -1.0, 0.0):
        v = A.added(PARENT, C, beside(b_r, -12.0, 0.0))
        made.append(total(v, A.run(v, {"A": a0, "C": c0}, t), "B"))
    want = a0 * (1 - math.exp(-k_of(30.0 - 12.0) * c0 * t))
    assert 0.1 * a0 < want < 0.9 * a0
    assert made == pytest.approx([want] * 3, rel=0.05)


def test_a_product_that_inhibits():
    """C stays on the product: one turnover, then nothing."""
    free = A.added(PARENT, C, beside(0.0, -12.0, 0.0))
    held = A.added(PARENT, C, beside(0.0, -12.0, -14.0))
    t = 3.0e3
    kf, kh = A.run(free, {"A": 1.0, "C": 0.01}, t), A.run(held, {"A": 1.0, "C": 0.01}, t)
    assert total(free, kf, "B") > 0.5
    assert total(held, kh, "B") < 0.03
    f = A.fate(held, kh)
    assert f["resting_state"]["id"] == "B.C" and f["shares"]["bound"] > 0.9 and f["fate"] == "regenerated"
    assert 0.5 < f["returns"] < 3.0


RELAY = {"new_steps": [{"id": "relay", "label": "A -> B through C", "reactants": ["A", "C"], "products": ["B", "C"],
                        "barrier": both(20.0), "parent_step": "ab", "status": "est", "source": "a test"}]}


def test_a_relay_returns_the_molecule():
    v = A.added(PARENT, C, RELAY)
    t, c0 = 20.0, 0.1
    out = A.readout(PARENT, v, {"A": 1.0}, c0, "B", t)
    want = 1 - math.exp(-k_of(20.0) * c0 * t)
    assert out["target"]["with"]["final"] == pytest.approx(want, rel=0.02)
    assert out["target"]["factor"] == pytest.approx(want / (k_of(30.0) * t), rel=0.02)
    f = out["added"]
    assert f["fate"] == "regenerated" and f["shares"]["free"] == pytest.approx(1.0, abs=1e-6)
    assert f["returns"] == pytest.approx(want / c0, rel=0.02) and f["resting_state"]["id"] == "C"
    (route,) = out["routes"]["with"]
    assert route["steps"] == ["relay"] and route["with_added"] and route["share"] == pytest.approx(1.0, abs=1e-3)
    (row,) = out["steps"]
    # before: 30 from A at 1 M; after: the relay, 20 from A and C at their (time-averaged) amounts
    assert row["before"] == pytest.approx(30.0, abs=0.3) and row["through"] == "relay" and row["status"] == "est"
    assert row["after"] == pytest.approx(20.0 - RT * math.log(c0), abs=0.6)
    assert out["target"]["with"]["time_to_fraction_s"] == pytest.approx(
        -math.log(1 - 0.5 * want) / (k_of(20.0) * c0), rel=0.1)
    # the lowest-peak route says what it rests on: an estimate here, verified rows once the relay is verified
    (peak,) = out["peak_routes"]["with"]
    assert peak["rests_on"] == "estimate" and peak["estimated_steps"] == ["relay"]
    sure = {"new_steps": [{**RELAY["new_steps"][0], "status": "verified"}]}
    (peak,) = A.readout(PARENT, A.added(PARENT, C, sure), {"A": 1.0}, c0, "B", t)["peak_routes"]["with"]
    assert peak["rests_on"] == "verified" and peak["estimated_steps"] == []


def test_a_consumed_reagent():
    est = {"new_species": [{"id": "D", "name": "adduct", "formed_from": ["A", "C"], "energy": both(-10.0),
                            "status": "verified"}],
           "new_steps": [{"id": "add", "reactants": ["A", "C"], "products": ["D"], "barrier": both(15.0),
                          "status": "verified"}]}
    v = A.added(PARENT, C, est)
    k = A.run(v, {"A": 1.0, "C": 0.2}, 100.0)
    f = A.fate(v, k)
    assert k["res"].c[-1, k["pos"]["D"]] == pytest.approx(0.2, rel=1e-3)
    assert f["fate"] == "consumed" and f["shares"]["covalent"] == pytest.approx(1.0, abs=1e-3)
    assert f["returns"] == 0.0 and f["resting_state"]["id"] == "D"


def test_a_covalent_cycle_is_a_catalyst():
    """A + C -> I, I -> B + C: C is held in I but comes back with every B."""
    est = {"new_species": [{"id": "I", "formed_from": ["A", "C"], "energy": both(-3.0), "status": "est"}],
           "new_steps": [{"id": "on", "reactants": ["A", "C"], "products": ["I"], "barrier": both(14.0), "status": "est"},
                         {"id": "off", "reactants": ["I"], "products": ["B", "C"], "barrier": both(19.0), "status": "est"}]}
    v = A.added(PARENT, C, est)
    k = A.run(v, {"A": 1.0, "C": 0.01}, 60.0)
    f = A.fate(v, k)
    assert f["fate"] == "regenerated" and f["resting_state"]["id"] == "I" and f["returns"] > 3
    (route,) = A.flux_routes(v, k, "B")
    assert route["steps"] == ["on", "off"]


def test_an_inert_binder_and_a_molecule_that_does_nothing():
    fast = network([("ab", ["A"], ["B"], 22.0)], g={"B": -10.0})
    t = 1.0 / k_of(22.0)
    binder = A.added(fast, C, {"species": [{"species": "A", "binding": both(-8.0), "status": "verified"}]})
    out = A.readout(fast, binder, {"A": 0.1}, 1.0, "B", t)
    assert out["added"]["fate"] == "bound and inert" and out["added"]["resting_state"]["id"] == "C"
    # (C is in excess: most of it stays free, but it holds nearly all of A)
    out = A.readout(fast, binder, {"A": 1.0}, 0.5, "B", t)
    assert out["added"]["fate"] == "bound and inert" and out["added"]["shares"]["bound"] > 0.9
    assert out["target"]["factor"] < 0.9                      # A held in the complex: less B
    nothing = A.added(fast, C, {"species": [{"species": "A", "binding": both(5.0), "status": "verified"}]})
    # an estimated binding makes no bound species at all
    est_only = A.added(fast, C, {"species": [{"species": "A", "binding": both(-8.0), "status": "est"}]})
    assert [x["id"] for x in est_only["species"]] == ["A", "B", "C"] and any("only a verified" in n for n in est_only["notes"])
    out = A.readout(fast, nothing, {"A": 1.0}, 0.5, "B", t)
    assert out["added"]["fate"] == "no part" and out["target"]["factor"] == pytest.approx(1.0, abs=1e-3)


def test_estimates_that_failed_are_left_out():
    v = A.added(PARENT, C, beside(-6.0, -12.0, -2.0, status="reacted"))
    assert [s["id"] for s in v["species"]] == ["A", "B", "C"] and len(v["steps"]) == 1


def test_an_electronic_number_stands_in_and_is_said():
    est = {"species": [{"species": "A", "status": "verified", "binding": {"electronic": -9.0, "free": 2.0}}],
           "steps": [{"step": "ab", "status": "est",
                      "binding": {"r": {"electronic": -9.0, "free": 2.0}, "ts": {"electronic": -12.0}, "p": {"electronic": 0.0}}}]}
    v = A.added(PARENT, C, est)
    cx = next(s for s in v["species"] if s["id"] == "A.C")
    assert cx["binding"] == 2.0                                # the free number where there is one
    assert next(st for st in v["steps"] if st["id"] == "ab+C")["g_ts"] == 18.0
    assert any("no free-energy estimate" in n for n in v["notes"])
    assert A.added({**PARENT, "energies": "electronic"}, C, est)["notes"] == []


# ---------------------------------------------------------------- sensitivity
def test_degree_of_control_orders_the_work():
    """A -> B and A -> D with equal barriers, run to the end: B = k1 / (k1 + k2), so X = +1/2 and -1/2."""
    net = network([("ab", ["A"], ["B"], 20.0), ("ad", ["A"], ["D"], 20.0)], species=("A", "B", "D"),
                  g={"B": -20.0, "D": -20.0})
    lim = A.limits(net, {"A": 1.0}, "B", 2000.0)
    assert lim["defined"] and lim["basis"] == "degree of control"
    assert lim["control"]["steps"]["ab"] == pytest.approx(0.5, abs=0.02)
    assert lim["control"]["steps"]["ad"] == pytest.approx(-0.5, abs=0.02)
    assert {o["step"]: o["role"] for o in lim["order"]} == {"ab": "lower", "ad": "do not lower"}
    assert [t["species"] for t in lim["traps"]] == []          # (both products are far downhill: no trap)


def test_a_parent_that_makes_none_of_the_target_gives_its_lowest_peak_routes():
    """A -> I -> B (TSs 50 and 40 above A) beats the direct A -> B (55); X is not defined."""
    net = network([("direct", ["A"], ["B"], 55.0), ("ai", ["A"], ["I"], 50.0), ("ib", ["I"], ["B"], 40.0)],
                  species=("A", "I", "B"), g={"I": 10.0, "B": -5.0})
    lim = A.limits(net, {"A": 1.0}, "B", 86400.0)
    assert not lim["defined"] and lim["control"] is None and "none of the target" in lim["why"]
    first, second = lim["routes"]
    assert [s["step"] for s in first["steps"]] == ["ai", "ib"] and first["peak"] == pytest.approx(50.0)
    assert [s["height"] for s in first["steps"]] == pytest.approx([50.0, 40.0])
    assert [s["step"] for s in second["steps"]] == ["direct"] and second["peak"] == pytest.approx(55.0)
    cut = lim["cutoff_kcal"]
    assert cut == pytest.approx(24.4, abs=0.1)
    assert [(o["step"], round(o["above_cutoff"] + cut, 6)) for o in lim["order"]] == [("direct", 55.0), ("ai", 50.0),
                                                                                    ("ib", 40.0)]
    assert A.first_order(lim, {"steps": {"ai": -5.0}}, T) is None


def test_a_route_counts_two_products_of_one_step_once():
    """A + A2 -> E + F (uphill by 20), then E + F -> B: the second TS is 54 above the mixture, not 74."""
    net = network([("ef", ["A", "A2"], ["E", "F"], 52.0), ("b", ["E", "F"], ["B"], 54.0)],
                  species=("A", "A2", "E", "F", "B"), g={"E": 12.0, "F": 8.0, "B": -3.0})
    (route,) = A.peak_routes(net, {"A": 1.0, "A2": 1.0}, "B")
    assert [s["height"] for s in route["steps"]] == pytest.approx([52.0, 54.0])
    # amounts move the heights: 0.1 M of A2 costs RT ln 10 on the way up
    (route,) = A.peak_routes(net, {"A": 1.0, "A2": 0.1}, "B")
    assert route["peak"] == pytest.approx(54.0 + RT * math.log(10.0))


def test_a_route_does_not_profit_from_a_side_reaction():
    """A -> J is far downhill; with J in hand everything after would look lower. It is not on the way to B."""
    net = network([("junk", ["A"], ["J"], 30.0), ("ai", ["A"], ["I"], 50.0), ("ib", ["I"], ["B"], 40.0)],
                  species=("A", "J", "I", "B"), g={"J": -40.0, "I": 10.0, "B": -5.0})
    (route,) = A.peak_routes(net, {"A": 1.0}, "B")
    assert [s["step"] for s in route["steps"]] == ["ai", "ib"] and route["peak"] == pytest.approx(50.0)


def test_a_target_of_several_species_counts_them_together():
    """Two mirror-image products from one start: together they are all of it, and no TS controls the sum."""
    net = network([("r", ["A"], ["R"], 20.0), ("s", ["A"], ["S"], 20.0)], species=("A", "R", "S"),
                  g={"R": -20.0, "S": -20.0})
    lim = A.limits(net, {"A": 1.0}, ["R", "S"], 2000.0)
    assert lim["defined"] and abs(lim["control"]["steps"]["r"]) < 0.02 and abs(lim["control"]["steps"]["s"]) < 0.02
    k = A.run(net, {"A": 1.0}, 2000.0)
    assert sorted(r["steps"][0] for r in A.flux_routes(net, k, ["R", "S"])) == ["r", "s"]
    assert [s["step"] for s in A.peak_routes(net, {"A": 1.0}, ["R", "S"])[0]["steps"]] in (["r"], ["s"])


def test_the_first_order_estimate_beside_the_full_kinetics():
    net = network([("ab", ["A"], ["B"], 24.0)], g={"B": -10.0})
    est = beside(0.0, -3.0, 0.0)
    est["steps"][0]["binding"]["r"] = est["steps"][0]["binding"]["p"] = {"electronic": 5.0, "free": 5.0}
    for x in est["species"]:
        x["binding"] = {"electronic": 5.0, "free": 5.0}
    v = A.added(net, C, est)
    want = 1 + 0.1 * math.exp(3.0 / RT)
    for t, close in ((0.001 / k_of(24.0), True), (3.0 / k_of(24.0), False)):
        lim = A.limits(net, {"A": 1.0}, "B", t)
        fo = A.first_order(lim, A.shifts(net, "C", est, 0.1), T)
        full = A.readout(net, v, {"A": 1.0}, 0.1, "B", t)["target"]["factor"]
        assert fo["terms"][0]["id"] == "ab" and fo["terms"][0]["shift"] == pytest.approx(-RT * math.log(want))
        if close:          # low conversion: the amount is linear in the rate constant
            assert fo["factor"] == pytest.approx(want, rel=0.05) and full == pytest.approx(want, rel=0.1)
        else:              # near the end the amount saturates: first order no longer holds, and it shows
            assert full < 1.1 and abs(math.log(fo["factor"]) - math.log(full)) > 0.05


def test_flux_splits_over_two_routes():
    """A -> B directly and through I, nine to one."""
    d = RT * math.log(9.0)
    net = network([("direct", ["A"], ["B"], 20.0), ("ai", ["A"], ["I"], 20.0 + d), ("ib", ["I"], ["B"], 5.0)],
                  species=("A", "I", "B"), g={"I": -1.0, "B": -30.0})
    k = A.run(net, {"A": 1.0}, 100.0)
    r1, r2 = A.flux_routes(net, k, "B")
    assert r1["steps"] == ["direct"] and r1["share"] == pytest.approx(0.9, abs=0.01)
    assert r2["steps"] == ["ai", "ib"] and r2["share"] == pytest.approx(0.1, abs=0.01)


def test_time_to_an_amount():
    assert A.time_to([0.0, 1.0, 10.0, 100.0], [0.0, 0.0, 1.0, 2.0], 0.5) == pytest.approx(math.sqrt(10.0))
    assert A.time_to([0.0, 1.0], [0.0, 1.0], 2.0) is None and A.time_to([0.0, 1.0], [0.0, 1.0], 0.0) is None


def test_a_version_written_the_other_way_is_read_the_parents_way():
    """The parent step is stored B -> A; the relay is given as A -> B, and the material goes A -> B."""
    back = network([("ba", ["B"], ["A"], 30.0)], g={"B": -10.0})
    v = A.added(back, C, {"new_steps": [{**RELAY["new_steps"][0], "parent_step": "ba"}]})
    (row,) = A.readout(back, v, {"A": 1.0}, 0.1, "B", 20.0)["steps"]
    k = A.run(v, {"A": 1.0, "C": 0.1}, 20.0)
    j = next(i for i, st in enumerate(v["steps"]) if st["id"] == "relay")
    # read the way the material goes: A -> B, the reverse of how the parent step is written
    assert row["direction"] == "reverse" and row["barrier"] == pytest.approx(30.0)
    assert row["through"] == "relay" and row["after"] == pytest.approx(k["apparent"][j][0])
    assert row["before"] == pytest.approx(30.0, abs=0.3)


def test_the_result_file_of_a_candidate(tmp_path):
    """From a query (here only its kinetics part) and estimates: tagged numbers, the read-out, what failed."""
    net = network([("ab", ["A"], ["B"], 24.0)], g={"B": -10.0})
    t = 0.001 / k_of(24.0)
    k = A.run(net, {"A": 1.0}, t)
    query = {**net, "schema": A.QUERY, "version": A.VERSION, "limits": A.limits(net, {"A": 1.0}, ["B"], t, k=k),
             "conditions": {"temperature": T, "time_s": t, "initial": {"A": 1.0}, "held": []},
             "target": {"species": ["B"], "fraction": 0.5}}
    est = {"steps": [{"step": "ab", "status": "est", "layer": "bystander", "level": "L",
                      "binding": {"r": both(5.0), "ts": both(-3.0), "p": both(5.0)}}]}
    doc = A.result(query, {"name": "cat", "smiles": "O", "amount": 0.1}, est,
                   failed=[{"step": "x", "layer": "bystander", "status": "reacted", "why": "it changed bonds"}])
    assert doc["schema"] == A.RESULT and doc["version"] == 1 and doc["status"] == "done"
    want = 1 + 0.1 * math.exp(3.0 / RT)
    assert doc["readout"]["target"]["factor"] == pytest.approx(want, rel=0.1)
    fo = doc["readout"]["first_order"]
    assert fo["factor"] == pytest.approx(want, rel=0.05) and fo["kinetics_factor"] == doc["readout"]["target"]["factor"]
    assert doc["readout"]["added"]["fate"] == "regenerated" and len(doc["failed"]) == 1
    assert doc["estimates"]["steps"][0]["status"] == "est"
    # the estimates' spread (0.5 on every binding), moved together: the target's amount between two bounds
    sp = doc["readout"]["target"]["spread"]
    assert sp["with_low"] < doc["readout"]["target"]["with"]["final"] < sp["with_high"]
    assert sp["factor_high"] / sp["factor_low"] == pytest.approx(math.exp(1.0 / RT), rel=0.15)
    A.write_json(tmp_path / "r.json", doc)
    import json

    assert json.loads((tmp_path / "r.json").read_text())["readout"]["steps"][0]["through"] == "ab+added"
    # estimates the kinetics cannot use do not lose the file: it says what failed
    bad = A.result(query, {"name": "cat", "amount": 0.1},
                   {"new_steps": [{"id": "n", "reactants": ["nope"], "products": ["B"], "barrier": both(1.0), "status": "est"}]})
    assert bad["readout"] is None and "unknown species" in bad["failed"][0]["why"]


def test_naming_species():
    sp = [{"id": "s1", "name": "aldol R", "smiles": "CC(=O)C[C@@H](C)O"}, {"id": "s2", "name": "aldol S", "smiles": "CC(=O)C[C@H](C)O"},
          {"id": "s3", "name": "acetone", "smiles": "CC(C)=O"}]
    assert A.match(sp, "CC(=O)CC(C)O") == ["s1", "s2"]           # no stereo marks: both
    assert A.match(sp, "C[C@H](O)CC(C)=O") in (["s1"], ["s2"])   # with them: one
    assert A.match(sp, "acetone") == ["s3"] and A.match(sp, "s2") == ["s2"] and A.match(sp, "CCO") == []


def test_the_barrier_table_has_two_groups_and_the_fate_two_facts():
    """A -> B -> D: a relay for A -> B. The first row goes through the molecule; B -> D's own TS is what it was,
    its apparent barrier moved because B is there now."""
    net = network([("ab", ["A"], ["B"], 30.0), ("bd", ["B"], ["D"], 28.0)], species=("A", "B", "D"),
                  g={"B": -1.0, "D": -3.0})
    v = A.added(net, C, RELAY)
    out = A.readout(net, v, {"A": 1.0}, 0.1, "B", 20.0)
    rows = {r["step"]: r for r in out["steps"]}
    assert rows["ab"]["group"] == "through" and rows["bd"]["group"] == "amounts" and rows["bd"]["through"] == "bd"
    f = out["added"]
    assert f["on_route"] is True and f["route_share"] == pytest.approx(1.0, abs=1e-3) and f["route_peak_with_added"]
    assert f["route_peak"] == pytest.approx(20.0 - RT * math.log(0.1), abs=0.01)
    assert f["turnovers"] == f["returns"] > 0.1 and f["fate"] == "regenerated"


def test_a_slow_catalyst_on_the_route_is_not_no_part():
    """The relay is the only way to B, and far too slow for a turnover in the time: on the route, few turnovers."""
    slow = {"new_steps": [{**RELAY["new_steps"][0], "barrier": both(27.0)}]}
    hard = network([("ab", ["A"], ["B"], 45.0)], g={"B": -10.0})
    f = A.readout(hard, A.added(hard, C, slow), {"A": 1.0}, 0.1, "B", 100.0)["added"]
    assert f["on_route"] is True and f["turnovers"] < 1e-3 and f["fate"] == "regenerated"
    # no steps with it at all: neither fact holds
    idle = A.readout(hard, A.added(hard, C, {}), {"A": 1.0}, 0.1, "B", 100.0)["added"]
    assert not idle["on_route"] and idle["fate"] == "no part" and idle["route_peak_with_added"] is False
    assert idle["route_role"] is None and "not on the routes" in idle["in_words"]
    assert f["route_role"] == "takes part" and "takes part in a step of the routes" in f["in_words"]
    # beside the TS of the route's only step (an estimate of the molecule beside it), taking part in nothing:
    # on the route, and said as that
    near = A.readout(hard, A.added(hard, C, {"steps": [{"step": "ab", "status": "est", "binding": {
        "r": both(0.0), "ts": both(-12.0), "p": both(0.0)}}]}), {"A": 1.0}, 0.1, "B", 100.0)["added"]
    assert near["on_route"] is True and near["route_role"] == "beside" and near["route_share_taking_part"] == 0.0
    assert "beside the transition states" in near["in_words"] and "taking part in none" in near["in_words"]


def test_a_row_for_a_reaction_is_compared_with_the_reactions_lowest_parent_ts():
    """A row found for a net reaction stands for all its parent steps; a shuttle step of that reaction (S on both
    sides, its TS higher) is read from the reaction's own lowest parent TS, so a row above that is no gain."""
    shuttle = network([("ab", ["A"], ["B"], 30.0), ("ab_s", ["A", "S"], ["B", "S"], 50.0)],
                      species=("A", "B", "S"), g={"B": -10.0})
    row = {"new_steps": [{"id": "scan", "reactants": ["A", "C"], "products": ["B", "C"], "barrier": both(40.0),
                          "parent_step": "ab", "parent_steps": ["ab", "ab_s"], "status": "verified"}]}
    out = A.readout(shuttle, A.added(shuttle, C, row), {"A": 1.0, "S": 1.0}, 0.1, "B", 20.0)
    by = {r["step"]: r for r in out["steps"]}
    for sid in ("ab", "ab_s"):
        assert by[sid]["before"] == pytest.approx(30.0, abs=0.05) and by[sid]["group"] != "through"
        assert abs(by[sid]["change"]) < 0.3 and by[sid]["reaction_steps"] == ["ab", "ab_s"]
    assert by["ab_s"]["before_this_step"] == pytest.approx(50.0, abs=0.05)
    # a row below the reaction's lowest TS is a gain through the molecule, for each of its steps
    row["new_steps"][0]["barrier"] = both(20.0)
    out = A.readout(shuttle, A.added(shuttle, C, row), {"A": 1.0, "S": 1.0}, 0.1, "B", 20.0)
    by = {r["step"]: r for r in out["steps"]}
    assert by["ab_s"]["group"] == "through" and by["ab_s"]["through"] == "scan"
    assert by["ab_s"]["change"] == pytest.approx(by["ab"]["change"]) and by["ab"]["change"] < -5


def test_a_shuttle_step_with_its_own_row_is_read_from_its_reactions_lowest_parent_ts():
    """A row of the shuttle step alone (from its own verification) is still compared with the reaction's best."""
    shuttle = network([("ab", ["A"], ["B"], 30.0), ("ab_s", ["A", "S"], ["B", "S"], 50.0)],
                      species=("A", "B", "S"), g={"B": -10.0})
    row = {"new_steps": [{"id": "ab_s|v", "reactants": ["A", "C"], "products": ["B", "C"], "barrier": both(35.0),
                          "parent_step": "ab_s", "status": "verified"}]}
    by = {r["step"]: r for r in A.readout(shuttle, A.added(shuttle, C, row), {"A": 1.0, "S": 1.0}, 0.1, "B", 20.0)["steps"]}
    assert by["ab_s"]["before"] == pytest.approx(30.0, abs=0.05) and abs(by["ab_s"]["change"]) < 0.3


def test_a_row_beside_a_steered_parent_step_counts_only_when_steered_too():
    steered = network([("ab", ["A"], ["B"], 30.0)], g={"B": -10.0})
    steered["steps"][0]["level"] = "explorer+steer"
    row = {"new_steps": [{"id": "relay", "reactants": ["A", "C"], "products": ["B", "C"], "barrier": both(20.0),
                          "parent_step": "ab", "status": "verified", "level": "explorer"}]}
    v = A.added(steered, C, row)
    assert [st["id"] for st in v["steps"]] == ["ab"] and any("not compared" in n for n in v["notes"])
    row["new_steps"][0]["level"] = "explorer+steer"
    assert [st["id"] for st in A.added(steered, C, row)["steps"]] == ["ab", "relay"]
    # beside a parent step at the explorer level an explorer row counts, as before
    assert [st["id"] for st in A.added(PARENT, C, {"new_steps": [{**row["new_steps"][0], "level": "explorer"}]})["steps"]] \
        == ["ab", "relay"]
