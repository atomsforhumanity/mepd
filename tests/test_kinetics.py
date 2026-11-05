import numpy as np
import pytest
from qcref.constants import HARTREE_TO_KCAL_PER_MOL
from scipy.constants import Boltzmann, Planck, gas_constant

from mepd.discovery.kinetics import Step, rate_constants, select_for_expansion, simulate

T = 298.15
RT = gas_constant * T / 4184.0  # kcal/mol


def _ha(kcal):
    return kcal / HARTREE_TO_KCAL_PER_MOL


def test_eyring_rates_obey_detailed_balance():
    energies = [0.0, _ha(-3.0)]
    (kf, kr), = rate_constants([Step(0, 1, _ha(20.0))], energies, T)
    assert kf == pytest.approx(Boltzmann * T / Planck * np.exp(-20.0 / RT))
    assert kf / kr == pytest.approx(np.exp(3.0 / RT))


def test_two_state_matches_the_analytic_solution():
    energies = [0.0, _ha(-1.0)]
    step = Step(0, 1, _ha(21.0))
    (kf, kr), = rate_constants([step], energies, T)
    t = 2.0 / (kf + kr)
    res = simulate(2, [step], energies, temperature=T, time_s=t)
    k = kf + kr
    cb = kf / k * (1 - np.exp(-k * t))
    assert res.final_concentration[1] == pytest.approx(cb, rel=1e-5)
    # Inflow into B = integral of kf * c_A; c_A = 1 - c_B.
    ca_int = t - kf / k * (t - (1 - np.exp(-k * t)) / k)
    assert res.flux[1] == pytest.approx(kf * ca_int, rel=1e-5)


def test_long_times_reach_boltzmann_equilibrium_across_a_chain():
    energies = [0.0, _ha(-2.0), _ha(1.0)]
    steps = [Step(0, 1, _ha(15.0)), Step(1, 2, _ha(16.0))]
    res = simulate(3, steps, energies, temperature=T, time_s=1e6)
    boltz = np.exp(-np.array([0.0, -2.0, 1.0]) / RT)
    assert np.allclose(res.final_concentration, boltz / boltz.sum(), rtol=1e-4)


def test_only_species_reached_by_enough_flux_are_selected():
    # seed -> A (fast) -> B (slow: 30 kcal/mol, nothing gets there in an hour)
    energies = [0.0, _ha(-5.0), _ha(-10.0)]
    steps = [Step(0, 1, _ha(18.0)), Step(1, 2, _ha(-5.0 + 30.0))]
    res = simulate(3, steps, energies, temperature=T, time_s=3600.0)
    assert res.flux[1] > 0.9 and res.flux[2] < 1e-4  # k = 6.5e-10 /s: ~2e-6 in an hour
    assert select_for_expansion(res, {0}, 0.01) == [1]


def test_no_steps_means_no_flux():
    res = simulate(3, [], [0.0, 0.0, 0.0])
    assert res.flux == [0.0, 0.0, 0.0] and res.final_concentration == [1.0, 0.0, 0.0]


def test_barrierless_and_slow_steps_together_conserve_material_and_finish_fast():
    """Rates from ~1e13 /s (barrierless) to ~1e-80 /s over 6 h: the Radau solve on K y took hours on
    networks like this one (K's diagonal loses the slow rates to rounding) and created material."""
    import time

    h = 1 / 627.5094740631
    rng = np.random.default_rng(0)
    n = 55
    e = np.r_[0.0, rng.uniform(-32, 90, n - 1)] * h
    steps = []
    for _ in range(90):
        a, b = rng.choice(n, 2, replace=False)
        ts = max(e[a], e[b]) + rng.choice([0.0, rng.uniform(0, 3), rng.uniform(10, 60)]) * h
        steps.append(Step(int(a), int(b), float(ts), "", {}))
    for i in range(1, 8):
        steps.append(Step(0, i, float(max(e[0], e[i]) + rng.uniform(15, 25) * h), "", {}))
    t0 = time.time()
    k = simulate(n, steps, e.tolist(), temperature=333, time_s=21600)
    assert time.time() - t0 < 30
    assert sum(k.final_concentration) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("time_s", [1e-3, 2e-3, 3e-3, 7e-3, 2e-2, 7200.0])
def test_a_network_slower_than_its_time_span_still_integrates(time_s):
    """Only slow steps (just after the first TS): the samples run from time_s to time_s, and geomspace can put
    one a rounding error past the end, which solve_ivp refuses (2 ms methyl formate runs crashed)."""
    from mepd.conditions import simulate_first_order

    step, energies = Step(0, 1, -99.8, "x", {}), [-100.0, -100.01]
    res = simulate(2, [step], energies, temperature=1500, time_s=time_s)
    (kf, kr), = rate_constants([step], energies, 1500)
    exact = (kr + kf * np.exp(-(kf + kr) * time_s)) / (kf + kr)        # A <-> B from pure A
    assert res.final_concentration[0] == pytest.approx(exact, rel=1e-5)
    out = simulate_first_order(2, [(0, 1, 1e-12 / time_s, 1e-13 / time_s)], [1.0, 0.0], time_s=time_s)
    assert out["t"][0] == 0.0 and out["t"][-1] == time_s and out["final"][0] == pytest.approx(1.0, abs=1e-9)
