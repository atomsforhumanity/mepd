"""Microkinetics on an expanding reaction network, to decide which species
are worth expanding next (flux-steered exploration).

Every verified elementary step A <-> B (a TS whose IRC connects A and B)
gets Eyring rate constants in both directions from the one TS energy,

    k(A->B) = (kB T / h) exp(-(E_TS - E_A) / RT),

so detailed balance holds by construction (energies are electronic, each
species at its lowest conformer found -- the barrier floor rule). Starting
from the seed at concentration 1, the linear first-order network
dc/dt = K c is integrated to `time_s`. Each proposed product counts as one
species, a pair of fragments included (as in the expansion itself), so all
steps are first order.

A species' *concentration flux* is the total amount that flowed into it
over the simulated time, integral_0^T sum_i k(i->j) c_i(t) dt. Species
whose flux reaches `threshold` are expanded in the next round; the network
stops growing once none does. This is the concentration-flux criterion of
Bensberg & Reiher (Isr. J. Chem. 63, e202200123 (2023)), the same idea as
the rate-based model enlargement of Susnow, Dean, Green & Peczak (J. Phys.
Chem. A 101, 3731 (1997)) used by RMG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.constants import Boltzmann, Planck, gas_constant

from qcref.constants import HARTREE_TO_KCAL_PER_MOL

_KCAL_TO_J = 4184.0


@dataclass
class Step:
    """One verified elementary step between species `a` and `b`."""
    a: int
    b: int
    ts_energy: float  # Hartree
    label: str = ""
    files: dict = field(default_factory=dict)
    # recording only (summary.json; docs/network-expansion.md, "What a run records"): when the step was found and
    # first used, its IRC's outcome, why it is a path peak. No decision reads it.
    rec: dict = field(default_factory=dict)


@dataclass
class KineticsResult:
    time_s: float
    temperature: float
    flux: list[float]          # integrated inflow per species
    max_concentration: list[float]
    final_concentration: list[float]
    barriers_kcal: list[Optional[tuple[float, float]]]  # (forward a->b, reverse b->a) per step; None: closed
    # with mirror images (mepd.discovery.mirror): the species the explorer found come first, the
    # mirror images it did not find after them; `mirrors` is Mirrors.record()
    n_found: Optional[int] = None
    mirrors: Optional[dict] = None


def rate_constants(steps: Sequence[Step], energies: Sequence[float], temperature: float) -> list[tuple[float, float]]:
    """(k(a->b), k(b->a)) in 1/s per step. The TS is never taken below
    either end (a TS energy below a minimum means the minimum's lowest
    conformer was found elsewhere), keeping detailed balance exact."""
    prefactor = Boltzmann * temperature / Planck
    rt = gas_constant * temperature / _KCAL_TO_J  # kcal/mol
    out = []
    for s in steps:
        ea, eb = float(energies[s.a]), float(energies[s.b])
        ts = max(float(s.ts_energy), ea, eb)
        out.append(tuple(prefactor * np.exp(-(ts - e) * HARTREE_TO_KCAL_PER_MOL / rt) for e in (ea, eb)))
    return out


# A barrier below this (kcal/mol) is reported, never absorbed: with one level
# of theory, minimized endpoints and a dense enough path it cannot happen.
NEGATIVE_BARRIER_TOL = -0.1


def raw_barriers(steps: Sequence[Step], energies: Sequence[float]) -> list[tuple[float, float]]:
    """(TS - E_a, TS - E_b) in kcal/mol as computed: may be negative, which
    signals a problem (see negative_barrier_warnings)."""
    return [((s.ts_energy - energies[s.a]) * HARTREE_TO_KCAL_PER_MOL,
             (s.ts_energy - energies[s.b]) * HARTREE_TO_KCAL_PER_MOL) for s in steps]


def negative_barrier_warning(label: str, a: int, b: int, forward: float, reverse: float,
                             verified: bool = True) -> Optional[str]:
    """The warning for a step whose TS lies below one of its ends, or None. `verified` False: no IRC has
    checked the step (files["verified"] False), so its ends are only the pair its TS search started between."""
    bad = [(k, v) for k, v in ((a, forward), (b, reverse)) if v < NEGATIVE_BARRIER_TOL]
    if not bad:
        return None
    where = ", ".join(f"{-v:.1f} kcal/mol below species {k}" for k, v in bad)
    if not verified:
        # sb-t24 (60 runs): 280 of the 281 steps with a TS below an end had no IRC; IRCs of them (g-xTB, as run)
        # went elsewhere every time -- a rotation of another species or of an end, or a step between others --
        # with the species' and TSs' energies all on one level (recomputed: 0.0 kcal/mol apart)
        return (f"Negative barrier for unverified step {label or f'{a} -> {b}'}: its TS is {where}, so it is no TS "
                "between these two species but the saddle of some other motion the TS optimization reached "
                "instead (no IRC has checked it). It takes no part in the final kinetics; if the kinetics need "
                "the step, its IRC files it under the species it connects, or drops it.")
    return (f"Negative barrier for step {label or f'{a} -> {b}'}: its TS is {where}. That cannot happen with one "
            "level of theory, minimized endpoints and a dense enough path, so something is wrong: an endpoint not "
            "minimized at this level (e.g. a lower conformer found later), energies from different levels, or a "
            "path too sparse to catch the real maximum. The kinetics treat this TS as level with that end; the "
            "raw barrier is what is reported.")


def negative_barrier_warnings(steps: Sequence[Step], energies: Sequence[float]) -> list[str]:
    out = []
    for s, (f, r) in zip(steps, raw_barriers(steps, energies)):
        w = negative_barrier_warning(getattr(s, "label", ""), s.a, s.b, f, r,
                                     (getattr(s, "files", None) or {}).get("verified") is not False)
        if w:
            out.append(w)
    return out


def barriers(steps: Sequence[Step], energies: Sequence[float]) -> list[tuple[float, float]]:
    """The barriers the rate constants use: the TS never below either end.
    For reporting, use raw_barriers and negative_barrier_warnings -- this
    clamp must never hide a negative barrier from the user."""
    return [((max(s.ts_energy, energies[s.a], energies[s.b]) - energies[s.a]) * HARTREE_TO_KCAL_PER_MOL,
             (max(s.ts_energy, energies[s.a], energies[s.b]) - energies[s.b]) * HARTREE_TO_KCAL_PER_MOL)
            for s in steps]


def simulate(n_species: int, steps: Sequence[Step], energies: Sequence[float], *, temperature: float = 298.15,
             time_s: float = 3600.0, start: int = 0, n_samples: int = 80) -> KineticsResult:
    """Integrate dc/dt = K c (plus the running inflow of every species)
    from c = e_start to `time_s` with an implicit (Radau) solver -- rates
    here span many orders of magnitude. The right-hand side is summed step
    by step (each step's net flux leaves one species and enters the other),
    which conserves material exactly; K's diagonal, a sum of rates from
    ~1e13 /s down to ~1e-5 /s, does not (its rounding error alone creates
    or destroys material over hours), and the solver could then take hours
    on one network. K is only the Jacobian."""
    from scipy.integrate import solve_ivp

    ks = rate_constants(steps, energies, temperature)
    K = np.zeros((n_species, n_species))
    for s, (kf, kr) in zip(steps, ks):
        if s.a == s.b:
            continue
        K[s.b, s.a] += kf
        K[s.a, s.a] -= kf
        K[s.a, s.b] += kr
        K[s.b, s.b] -= kr
    inflow = np.where(np.eye(n_species, dtype=bool), 0.0, K)  # k(i->j) at [j, i]
    n = n_species
    J = np.zeros((2 * n, 2 * n))
    J[:n, :n] = K
    J[n:, :n] = inflow  # d(flux_j)/dt = sum_i k(i->j) c_i
    y0 = np.zeros(2 * n)
    y0[start] = 1.0
    if not steps or not np.any(K):
        c = y0[:n]
        return KineticsResult(time_s, temperature, [0.0] * n, c.tolist(), c.tolist(), barriers(steps, energies))
    fastest = max(max(k) for k in ks)
    t_eval = np.unique(np.concatenate([[0.0], np.geomspace(min(1e-3 / fastest, time_s), time_s, n_samples)]))
    t_eval = t_eval[t_eval <= time_s]      # geomspace(t, t, n) can land a rounding error past the end
    t_eval[-1] = time_s
    live = [(s.a, s.b, kf, kr) for s, (kf, kr) in zip(steps, ks) if s.a != s.b]
    ia = np.array([x[0] for x in live])
    ib = np.array([x[1] for x in live])
    kfs = np.array([x[2] for x in live])
    krs = np.array([x[3] for x in live])

    def rhs(t, y):
        c = y[:n]
        fwd, rev = kfs * c[ia], krs * c[ib]
        d = np.zeros(2 * n)
        np.add.at(d, ib, fwd - rev)
        np.add.at(d, ia, rev - fwd)
        np.add.at(d, n + ib, fwd)   # inflow into each species, accumulated
        np.add.at(d, n + ia, rev)
        return d

    sol = solve_ivp(rhs, (0.0, time_s), y0, method="Radau", jac=J, t_eval=t_eval, rtol=1e-8, atol=1e-14)
    c = np.clip(sol.y[:n], 0.0, None)
    return KineticsResult(time_s, temperature, np.clip(sol.y[n:, -1], 0.0, None).tolist(),
                          c.max(axis=1).tolist(), c[:, -1].tolist(), barriers(steps, energies))


def select_for_expansion(result: KineticsResult, expanded: set[int], threshold: float,
                         limit: Optional[int] = None) -> list[int]:
    """Species not expanded yet whose concentration flux reached
    `threshold`, most flux first."""
    n = len(result.flux) if result.n_found is None else result.n_found   # an added mirror image is no species to expand
    picks = sorted((j for j, f in enumerate(result.flux[:n]) if j not in expanded and f >= threshold),
                   key=lambda j: -result.flux[j])
    return picks[:limit] if limit else picks
