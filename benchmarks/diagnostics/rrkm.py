"""Unimolecular rate constants beyond the explorer's model: tunnelling and pressure fall-off.

The explorer's kinetics are canonical TST at the high-pressure limit (Eyring with free energies, no tunnelling,
no collisional energy transfer). Gas-phase shock-tube experiments on small molecules are often in fall-off, where the
branching between two channels of different thresholds is set by the microcanonical competition. This module gives
the simplest defensible model of that:

- RRKM k_i(E) = L_i * F_rot,i * N_i(E - E0_i) / (h rho(E)); harmonic vibrations counted directly (Beyer-Swinehart,
  grain `dE` cm^-1); the external 2D (J) rotor adiabatic with the ratio of rotational partition functions
  (F_rot = Q_rot,TS / Q_rot,R, which makes the high-pressure limit equal canonical TST exactly), the K rotor folded
  into F_rot too; L_i the reaction path degeneracy (mirror images). Reactant: several conformers (wells in fast
  equilibrium at these energies) summed with their zero-point offsets.
- Tunnelling: Wigner (canonical) and the asymmetric Eckart barrier (Johnston & Heicklen 1962), canonical and,
  for the microcanonical rates, N_i(E) convolved with the Eckart transmission probability (Miller 1979).
- Collisions: strong collisions (Lindemann-Hinshelwood-RRKM: every collision thermalises), omega = beta_c Z_LJ [M],
  Z_LJ from Lennard-Jones parameters with the Neufeld collision integral; and a weak-collision 1D master equation
  (exponential-down model, <dE>_down given), solved for its slowest eigenvector (steady state) to give each
  channel's phenomenological rate constant.
Everything in cm^-1 internally; energies relative to the reactant's lowest zero-point level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

H = 6.62607015e-34
KB = 1.380649e-23
C_CM = 2.99792458e10          # cm/s
NA = 6.02214076e23
AMU = 1.66053906660e-27
KCAL_TO_CM = 349.7550878       # cm^-1 per kcal/mol
K_CM = 0.6950348004            # cm^-1 per K (k_B/hc)


def beyer_swinehart(freqs_cm: Sequence[float], de: float, n: int) -> np.ndarray:
    """Harmonic vibrational density of states per grain (states / cm^-1) on 0, de, 2de, ... (n grains)."""
    rho = np.zeros(n)
    rho[0] = 1.0
    for f in freqs_cm:
        k = max(1, int(round(float(f) / de)))
        for i in range(k, n):
            rho[i] += rho[i - k]
    return rho / de


def sum_of_states(rho: np.ndarray, de: float) -> np.ndarray:
    return np.cumsum(rho) * de


def q_vib(freqs_cm, T) -> float:
    """Harmonic vibrational partition function from the zero-point level."""
    x = np.asarray(freqs_cm, float) / (K_CM * T)
    return float(np.prod(1.0 / (1.0 - np.exp(-x))))


def q_rot(moments_amu_A2: Sequence[float], T: float, sigma: int = 1) -> float:
    """Classical rigid-rotor partition function (3D, or linear when the smallest moment ~ 0)."""
    I = np.sort(np.asarray(moments_amu_A2, float)) * AMU * 1e-20
    if I[0] < 1e-3 * I[2]:
        return 8 * math.pi ** 2 * I[2] * KB * T / (sigma * H ** 2)
    return (math.sqrt(math.pi * np.prod(I)) / sigma) * (8 * math.pi ** 2 * KB * T / H ** 2) ** 1.5


def moments(symbols, xyz_A) -> list:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    m = np.array([pt.GetMostCommonIsotopeMass(s) for s in symbols])
    X = np.asarray(xyz_A, float)
    r = X - (m[:, None] * X).sum(0) / m.sum()
    I = np.zeros((3, 3))
    for mi, ri in zip(m, r):
        I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    return sorted(np.linalg.eigvalsh(I).tolist())


def wigner(imag_cm: float, T: float) -> float:
    u = abs(imag_cm) / (K_CM * T)
    return 1.0 + u * u / 24.0


def eckart_probability(e_cm: np.ndarray, v1_cm: float, v2_cm: float, imag_cm: float) -> np.ndarray:
    """Transmission probability through an asymmetric Eckart barrier at energies e (measured from the reactant
    side's zero, so the barrier top is at v1; the product side lies at v1 - v2). Johnston & Heicklen,
    J. Phys. Chem. 66, 532 (1962), in the form of Truhlar & Kuppermann."""
    e = np.asarray(e_cm, float)
    nu = abs(imag_cm)
    v1, v2 = float(v1_cm), float(v2_cm)
    out = np.zeros_like(e)
    ok = e > max(0.0, v1 - v2) + 1e-9          # above both asymptotes
    if not np.any(ok):
        return out
    E = e[ok]
    # dimensionless (energies in units of h*nu; all cm^-1 so ratios are fine)
    alpha1, alpha2 = 2 * math.pi * v1 / nu, 2 * math.pi * v2 / nu
    xi = E / v1
    a = 2 * np.sqrt(alpha1 * xi) / (1 / math.sqrt(alpha1) + 1 / math.sqrt(alpha2))
    b = 2 * np.sqrt(np.clip((1 + xi) * alpha1 - alpha2, 0, None)) / (1 / math.sqrt(alpha1) + 1 / math.sqrt(alpha2))
    d = 2 * math.sqrt(max(alpha1 * alpha2 - 4 * math.pi ** 2 / 16, 0.0)) / (1 / math.sqrt(alpha1) + 1 / math.sqrt(alpha2))
    # P = 1 - (cosh(2pi(a-b)) + cosh(2pi d)) / (cosh(2pi(a+b)) + cosh(2pi d)), computed stably
    def lcosh(x):
        x = np.abs(x)
        return x + np.log1p(np.exp(-2 * x)) - math.log(2)
    num = np.logaddexp(lcosh(2 * math.pi * (a - b)), lcosh(2 * math.pi * d) * np.ones_like(a))
    den = np.logaddexp(lcosh(2 * math.pi * (a + b)), lcosh(2 * math.pi * d) * np.ones_like(a))
    out[ok] = np.clip(1.0 - np.exp(num - den), 0.0, 1.0)
    return out


def eckart_kappa(v1_cm: float, v2_cm: float, imag_cm: float, T: float, de: float = 1.0) -> float:
    """Canonical Eckart tunnelling factor: kappa = exp(V1/kT)/kT * int P(E) exp(-E/kT) dE (E from the reactant)."""
    kT = K_CM * T
    e = np.arange(0.0, v1_cm + 40 * kT, de)
    P = eckart_probability(e, v1_cm, v2_cm, imag_cm)
    w = np.exp(-(e - v1_cm) / kT)
    return float(np.trapezoid(P * w, e) / kT)


@dataclass
class Channel:
    name: str
    freqs_cm: list                 # real TS frequencies (the reaction mode left out)
    moments: list                  # TS moments of inertia, amu A^2
    e0_cm: float                   # TS zero-point level above the reactant's lowest zero-point level
    degeneracy: float = 1.0        # reaction path degeneracy (mirror images; symmetry numbers)
    sigma: int = 1                 # TS rotational symmetry number
    imag_cm: float = 0.0
    v_rev_cm: Optional[float] = None   # ZPE-corrected barrier from the product side (Eckart); None: no tunnelling


@dataclass
class Well:
    conformers: list               # [(freqs_cm, moments, zpe offset cm^-1 from the lowest, sigma, multiplicity)]
    mass_amu: float
    lj_sigma_A: float
    lj_eps_K: float


@dataclass
class Bath:
    mass_amu: float = 39.948
    lj_sigma_A: float = 3.47
    lj_eps_K: float = 114.0


def collision_rate(well: Well, bath: Bath, T: float, P_atm: float) -> float:
    """Lennard-Jones collision frequency (1/s) at T and P (ideal gas, bath dominant)."""
    sig = 0.5 * (well.lj_sigma_A + bath.lj_sigma_A) * 1e-10
    eps = math.sqrt(well.lj_eps_K * bath.lj_eps_K)
    ts = T / eps
    omega22 = 1.16145 * ts ** -0.14874 + 0.52487 * math.exp(-0.77320 * ts) + 2.16178 * math.exp(-2.43787 * ts)
    mu = well.mass_amu * bath.mass_amu / (well.mass_amu + bath.mass_amu) * AMU
    z = math.pi * sig ** 2 * math.sqrt(8 * KB * T / (math.pi * mu)) * omega22
    conc = P_atm * 101325.0 / (KB * T)
    return z * conc


@dataclass
class Model:
    well: Well
    channels: list
    de: float = 10.0
    e_max_cm: float = 60000.0
    _cache: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.e_max_cm / self.de) + 1

    def rho_reactant(self) -> np.ndarray:
        if "rho" not in self._cache:
            n = self.n
            tot = np.zeros(n)
            for freqs, mom, off, sigma, mult in self.well.conformers:
                r = beyer_swinehart(freqs, self.de, n) * mult
                s = int(round(off / self.de))
                tot[s:] += r[:n - s]
            self._cache["rho"] = tot
        return self._cache["rho"]

    def q_reactant(self, T: float) -> float:
        """Vibrational x rotational partition function of the reactant well (conformers summed)."""
        return sum(q_vib(f, T) * q_rot(m, T, s) * mult * math.exp(-off / (K_CM * T))
                   for f, m, off, s, mult in self.well.conformers)

    def q_rot_ref(self, T: float) -> float:
        """The reactant's rotational partition function used in F_rot (Boltzmann average over conformers)."""
        w = [mult * q_vib(f, T) * math.exp(-off / (K_CM * T)) for f, m, off, s, mult in self.well.conformers]
        qr = [q_rot(m, T, s) for f, m, off, s, mult in self.well.conformers]
        return float(np.dot(w, qr) / np.sum(w))

    def n_ts(self, ch: Channel, tunnel: bool) -> np.ndarray:
        """Sum of states of the TS on the reactant's grid (energy from the reactant's zero-point level)."""
        key = ("N", ch.name, tunnel)
        if key in self._cache:
            return self._cache[key]
        n = self.n
        rho_ts = beyer_swinehart(ch.freqs_cm, self.de, n)
        e = np.arange(n) * self.de
        if tunnel and ch.v_rev_cm is not None and ch.imag_cm:
            # N(E) = sum over TS states eps of P(E - eps) with P the 1D Eckart probability (barrier top at e0)
            P = eckart_probability(e, ch.e0_cm, ch.v_rev_cm, ch.imag_cm)   # P as a function of E above reactant
            # convolution: N(E) = int rho_ts(eps) P(E - eps) d eps (eps from the TS zero-point level = e0 offset)
            N = np.convolve(rho_ts * self.de, P)[:n]
        else:
            s = int(round(ch.e0_cm / self.de))
            N = np.zeros(n)
            N[s:] = sum_of_states(rho_ts, self.de)[:n - s]
        self._cache[key] = N
        return N

    def k_E(self, ch: Channel, T: float, tunnel: bool) -> np.ndarray:
        """RRKM k(E) (1/s) on the grid, with F_rot = Q_rot,TS / Q_rot,R at T (adiabatic external rotation)."""
        rho = self.rho_reactant()
        frot = q_rot(ch.moments, T, ch.sigma) / self.q_rot_ref(T)
        N = self.n_ts(ch, tunnel)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = ch.degeneracy * frot * N * C_CM / np.where(rho > 0, rho, np.inf)
        return np.nan_to_num(k)

    def k_inf(self, ch: Channel, T: float, tunnel: bool = False) -> float:
        """High-pressure limit from the microcanonical rates (= canonical TST with the same partition functions)."""
        e = np.arange(self.n) * self.de
        f = self.rho_reactant() * np.exp(-e / (K_CM * T))
        return float(np.sum(self.k_E(ch, T, tunnel) * f) / np.sum(f))

    def k_tst(self, ch: Channel, T: float) -> float:
        """Canonical TST, closed form (check)."""
        q = q_vib(ch.freqs_cm, T) * q_rot(ch.moments, T, ch.sigma)
        return ch.degeneracy * KB * T / H * q / self.q_reactant(T) * math.exp(-ch.e0_cm / (K_CM * T))

    def strong_collision(self, T: float, P_atm: float, bath: Bath, beta_c: float = 1.0, tunnel: bool = False) -> dict:
        """k_i(T, P) (1/s) under strong collisions: int k_i(E) w f(E) / (w + sum_j k_j(E)) dE."""
        e = np.arange(self.n) * self.de
        f = self.rho_reactant() * np.exp(-e / (K_CM * T))
        f /= f.sum()
        w = beta_c * collision_rate(self.well, bath, T, P_atm)
        ks = {ch.name: self.k_E(ch, T, tunnel) for ch in self.channels}
        tot = sum(ks.values())
        return {name: float(np.sum(k * w * f / (w + tot))) for name, k in ks.items()} | {"omega": w}

    def master_equation(self, T: float, P_atm: float, bath: Bath, de_down_cm: float, tunnel: bool = False,
                        e_cut: Optional[float] = None) -> dict:
        """1D master equation, exponential-down energy transfer (<dE>_down = alpha), collision rate Z_LJ [M].
        The phenomenological rate constants from the slowest eigenvector (steady-state distribution)."""
        de = self.de
        kT = K_CM * T
        n_full = self.n
        # coarse grid for the ME: grains of max(de, alpha/10)
        step = max(1, int(round(max(de, de_down_cm / 10.0) / de)))
        idx = np.arange(0, n_full, step)
        e = idx * de
        rho = np.add.reduceat(self.rho_reactant(), idx) * de / (step * de)   # mean density per grain
        g = rho * np.exp(-e / kT)
        ks = {ch.name: np.add.reduceat(self.k_E(ch, T, tunnel), idx) / step for ch in self.channels}
        omega = collision_rate(self.well, bath, T, P_atm)
        keep = g > 0
        e, g = e[keep], g[keep]
        ks = {k: v[keep] for k, v in ks.items()}
        m = len(e)
        # P[i, j] = probability that a collision takes grain j to grain i. Down (E_i <= E_j): C_j exp(-(E_j-E_i)/a);
        # up (E_i > E_j) from detailed balance: P[i, j] g_j = P[j, i] g_i = C_i exp(-(E_i-E_j)/a) g_i.
        # The normalisation constants C_j (sum_i P[i, j] = 1) by recursion from the top grain down.
        a = float(de_down_cm)
        D = np.exp(-np.abs(e[:, None] - e[None, :]) / a)          # symmetric kernel
        C = np.zeros(m)
        lg = np.log(g)
        for j in range(m - 1, -1, -1):
            down = D[:j + 1, j].sum()
            up = float(np.sum(C[j + 1:] * D[j + 1:, j] * np.exp(lg[j + 1:] - lg[j]))) if j + 1 < m else 0.0
            C[j] = max((1.0 - up) / down, 1e-12 / down)
        Pm = np.where(e[:, None] <= e[None, :], C[None, :] * D, C[:, None] * D * np.exp(lg[:, None] - lg[None, :]))
        ktot = sum(ks.values())
        M = omega * (Pm - np.eye(m)) - np.diag(ktot)
        # symmetrise with sqrt(g) for a stable eigensolve
        s = np.sqrt(np.where(g > 0, g, 1e-300))
        A = (M * s[None, :]) / s[:, None]
        A = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(A)
        lam = -w[-1]
        x = V[:, -1] * s
        x = np.abs(x) / np.abs(x).sum()
        return {ch.name: float(np.sum(ks[ch.name] * x)) for ch in self.channels} | {"lambda": float(lam),
                                                                                    "omega": omega}
