"""Langevin dynamics (BAOAB) on GFN2-xTB (tblite, the wavefunction carried
from step to step) inside the nanoreactor's soft logfermi wall: the one
integrator of the interactive Reactor (mepd.web.sandbox), the collide
generator and the reactor generator. What steers the atoms (the Reactor's
drags, Collide's pulls) comes in as `bias`, an extra gradient."""
from __future__ import annotations

import math
import os
from typing import Callable, Optional

import numpy as np

BOHR = 0.529177210903            # Angstrom
KCAL = 627.509474                # kcal/mol per Hartree
MAX_PULL = 0.08                  # Hartree/bohr cap on a pull's force (~100 kcal/mol/A)
KB = 3.166811563e-6 / 1.066551   # amu bohr^2/fs^2 per K
ACC = 0.93766                    # (Eh/bohr)/amu -> bohr/fs^2


def capped(f: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(f)
    return f * (MAX_PULL / norm) if norm > MAX_PULL else f


def pull_pairs(x: np.ndarray, pairs, k: float) -> np.ndarray:
    """The gradient of pulling each pair's two atoms toward their moving
    midpoint with a spring `k` (Eh/bohr^2), the force capped."""
    g = np.zeros_like(x)
    for i, j in pairs:
        mid = 0.5 * (x[i] + x[j])
        for a in (i, j):
            g[a] += capped(k * (x[a] - mid))
    return g


class LangevinMD:
    """`pos`/`vel` in bohr and bohr/fs; `wall`: its radius in Angstrom around
    the origin (None: no wall); `T`, `dt` (fs) and `friction` (1/fs) may change
    between steps."""

    def __init__(self, symbols, pos_bohr, charge: int, multiplicity: int, *, temperature: float,
                 wall: Optional[float] = None, seed: int = 0, dt: float = 0.5, friction: float = 0.02,
                 bias: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from rdkit import Chem
        from tblite.interface import Calculator

        from mepd.discovery.nanoreactor import _masses

        table = Chem.GetPeriodicTable()
        self.pos = np.asarray(pos_bohr, dtype=float)
        self.masses = _masses(symbols)
        self.calc = Calculator("GFN2-xTB", np.array([table.GetAtomicNumber(s) for s in symbols]), self.pos,
                               charge=float(charge), uhf=int(multiplicity) - 1)
        self.calc.set("verbosity", 0)
        self.calc.set("temperature", 3000.0 * 3.166811563e-6)   # Fermi smearing: bonds break without SCF failures
        self.wall, self.wall_force = wall, 20.0 / KCAL * BOHR      # 20 kcal/mol/A, in Eh/bohr
        self.T, self.dt, self.friction, self.bias = float(temperature), dt, friction, bias
        self.rng = np.random.default_rng(seed)
        self.vel = self.energy = self.grad = None
        self.failures = 0
        self._res = None

    def forces(self, x, biased: bool = True):
        from mepd.discovery.nanoreactor import _wall

        self.calc.update(x)
        try:
            r = self.calc.singlepoint(self._res) if self._res is not None else self.calc.singlepoint()
        except Exception:
            r = self.calc.singlepoint()          # without the previous wavefunction
        self._res = r
        e, g = float(r.get("energy")), np.array(r.get("gradient"))
        if self.wall is not None:
            ew, gw = _wall(x, self.wall / BOHR, self.wall_force)
            e, g = e + ew, g + gw
        if biased and self.bias is not None:
            g = g + self.bias(x)
        return e, g

    def start(self) -> None:
        """Velocities from the Maxwell-Boltzmann distribution at `T`, and the first forces."""
        self.vel = self.rng.normal(size=self.pos.shape) * np.sqrt(KB * self.T / self.masses)[:, None]
        self.energy, self.grad = self.forces(self.pos)

    def step(self) -> bool:
        """One step; False if it failed (an SCF): the step is taken back and the
        velocities halved. Raises after more than 20 failures in a row."""
        m = self.masses[:, None]
        c1 = math.exp(-self.friction * self.dt)
        c2 = np.sqrt((1 - c1 * c1) * KB * self.T / self.masses)[:, None]
        old = (self.pos.copy(), self.vel.copy(), self.grad.copy())
        try:
            self.vel -= 0.5 * self.dt * ACC * self.grad / m
            self.pos = self.pos + 0.5 * self.dt * self.vel
            self.vel = c1 * self.vel + c2 * self.rng.normal(size=self.vel.shape)
            self.pos = self.pos + 0.5 * self.dt * self.vel
            self.energy, self.grad = self.forces(self.pos)
            self.vel -= 0.5 * self.dt * ACC * self.grad / m
            if not np.all(np.isfinite(self.pos)):
                raise FloatingPointError("non-finite positions")
        except Exception:
            self.pos, self.vel, self.grad = old
            self.vel *= 0.5
            self._res = None
            self.failures += 1
            if self.failures > 20:
                raise
            return False
        self.failures = 0
        return True

    def relax(self, n: int) -> None:
        """`n` steepest-descent steps without the bias, each at most 0.1 bohr per
        atom: no new chemistry (a quench, or the strain out of a packing)."""
        for _ in range(int(n)):
            _, g = self.forces(self.pos, biased=False)
            step = -g / self.masses[:, None] * 20.0
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            self.pos = self.pos + np.where(norm > 0.1, step * 0.1 / np.maximum(norm, 1e-12), step)
