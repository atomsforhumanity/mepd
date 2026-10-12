"""Microkinetics with steps of any order: equilibria, shuttles, steady
states and degrees of rate control."""
import math

import numpy as np
import pytest

from mepd import microkinetics as mk


def test_isomerization_reaches_its_equilibrium_constant():
    net = mk.Network(["A", "B"], [0.0, -1.0], [mk.Step([0], [1], 15.0)])
    res = mk.simulate(net, [1.0, 0.0], 298.15, 1e6)
    K = math.exp(1.0 / mk.rt_kcal(298.15))
    assert res.c[-1, 1] / res.c[-1, 0] == pytest.approx(K, rel=1e-3)
    assert res.c[-1].sum() == pytest.approx(1.0, rel=1e-6)       # mass conserved
    assert res.extent[0] == pytest.approx(res.c[-1, 1], rel=1e-4)


def test_association_obeys_mass_action_at_one_molar_standard_state():
    net = mk.Network(["A", "B", "C"], [0.0, 0.0, -2.0], [mk.Step([0, 1], [2], 10.0)])
    res = mk.simulate(net, [0.5, 0.3, 0.0], 298.15, 1e6)
    a, b, c = res.c[-1]
    assert c / (a * b) == pytest.approx(math.exp(2.0 / mk.rt_kcal(298.15)), rel=1e-3)


def test_a_shuttle_speeds_its_step_and_is_not_consumed():
    # A -> B directly (barrier 25) or with water W (A + W -> B + W, barrier 18).
    direct = mk.Network(["A", "B", "W"], [0.0, -5.0, 0.0], [mk.Step([0], [1], 25.0)])
    both = mk.Network(["A", "B", "W"], [0.0, -5.0, 0.0], [mk.Step([0], [1], 25.0), mk.Step([0, 2], [1, 2], 18.0)])
    t = 1.0
    slow = mk.simulate(direct, [1.0, 0.0, 1.0], 298.15, t).c[-1, 1]
    fast = mk.simulate(both, [1.0, 0.0, 1.0], 298.15, t)
    assert fast.c[-1, 1] > 100 * slow
    assert fast.c[-1, 2] == pytest.approx(1.0, rel=1e-9)
    assert fast.extent[1] > 50 * fast.extent[0]                   # the material went through the shuttle


def test_constant_feed_reaches_a_steady_state_and_rate_control_sums_to_one():
    # A (fed) -> I -> P, P drained to a held-empty sink: a steady formation rate of P.
    net = mk.Network(["A", "I", "P", "S"], [0.0, -2.0, -10.0, -40.0],
                     [mk.Step([0], [1], 18.0), mk.Step([1], [2], 20.0), mk.Step([2], [3], 5.0)])
    held = [0, 3]
    res = mk.simulate(net, [1.0, 0.0, 0.0, 0.0], 298.15, 1e8, held=held)
    assert res.steady
    drc = mk.degree_of_control(net, [1.0, 0.0, 0.0, 0.0], 298.15, 1e8, target=3, what="rate", held=held)
    assert sum(drc["steps"]) == pytest.approx(1.0, abs=0.03)
    assert np.argmax(drc["steps"]) == 1                            # I -> P is the rate-determining TS


def test_temperature_sweep_gives_an_arrhenius_activation_energy():
    net = mk.Network(["A", "B"], [0.0, -10.0], [mk.Step([0], [1], 20.0)])
    out = mk.sweep(net, [1.0, 0.0], [300, 320, 340], 1e-6, target=1)
    # at very short times the formation rate is k [A]: apparent Ea = barrier + RT
    assert out["apparent_ea_kcal"][1] == pytest.approx(20.0 + mk.rt_kcal(320), abs=0.3)


def test_a_hot_catalytic_cycle_with_fast_equilibria_integrates():
    """The CO2 + epoxide cycle of a workspace (bromide opens the ring, CO2
    inserts, the ring closes; the ammonium rides along), electronic
    energies in kcal/mol. At 1000 K its fast steps run at ~1e12 M/s each
    way, and tracking their extents' rounding noise made BDF crawl, then
    give up ("the kinetics did not integrate")."""
    g = [-96514.34, -118372.65, -214895.35, -134400.88, -1711819.90, -1830222.37, -1615326.41]
    steps = [([0, 1], [2], -214830.28), ([6, 3, 0], [4, 3], -1846198.95), ([4, 3, 1], [5, 3], -1964618.51),
             ([5, 3], [6, 2, 3], -1964613.19), ([1, 4, 3], [0, 1, 3, 6], -1964606.52)]
    net = mk.Network(["EO", "CO2", "EC", "NMe4+", "BrCH2CH2O-", "BrCH2CH2OCO2-", "Br-"], g,
                     [mk.Step(a, b, t) for a, b, t in steps])
    res = mk.simulate(net, [1, 1, 0, 1, 0, 0, 1], 1000.0, 3600.0)
    assert res.c[-1, [4, 5, 6]].sum() == pytest.approx(1.0, rel=1e-6)     # bromine conserved
    assert res.c[-1, 3] == pytest.approx(1.0)                               # the ammonium only rides along


def test_long_runs_end_exactly_at_the_requested_time():
    # logspace's last point can round past the end of the span (a year: 3.15e7 s)
    net = mk.Network(["A", "B"], [0.0, -1.0], [mk.Step([0], [1], 25.0)])
    res = mk.simulate(net, [1.0, 0.0], 298.15, 3.15e7)
    assert res.times[-1] == 3.15e7
