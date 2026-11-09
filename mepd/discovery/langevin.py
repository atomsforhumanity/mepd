"""Langevin dynamics (BAOAB) on GFN2-xTB or g-xTB (axtb, in this process, one
calculator per trajectory; GFN2-xTB carries its wavefunction from step to
step) inside the nanoreactor's soft logfermi wall: the one
integrator of the interactive Reactor (mepd.web.sandbox), the collide
generator and the reactor generator. What steers the atoms (the Reactor's
drags, Collide's pulls) comes in as `bias`, an extra gradient."""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

BOHR = 0.529177210903            # Angstrom
KCAL = 627.509474                # kcal/mol per Hartree
METHODS = ("gfn2", "gxtb-v2.0.1", "uma")    # GFN2-xTB and g-xTB on axtb, UMA (fairchem)
UMA_MODEL = "uma-s-1p2"
# Fermi smearing (K): the MD's own, then for a step whose SCF fails to converge (g-xTB at 3000 K oscillates for
# some torn geometries, e.g. CO + OH + H, where 6000 K and 1000 K converge in a few ms)
ETEMPS = (3000.0, 6000.0, 1000.0)
MAX_PULL = 0.08                  # Hartree/bohr cap on a pull's force (~100 kcal/mol/A)
KB = 3.166811563e-6 / 1.066551   # amu bohr^2/fs^2 per K
ACC = 0.93766                    # (Eh/bohr)/amu -> bohr/fs^2


def capped(f: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(f)
    return f * (MAX_PULL / norm) if norm > MAX_PULL else f


def pull_pairs(x: np.ndarray, pairs, k: float, masses=None) -> np.ndarray:
    """The gradient of pulling each pair's two atoms toward their moving
    midpoint with a spring `k` (Eh/bohr^2), the force capped (MAX_PULL). A
    pair (i, j, A, B) of atoms of two molecules pushes the whole molecules
    together instead: that force, along the line between i and j, on each
    molecule (atom lists A and B), spread over its atoms by mass (`masses`),
    so nothing is dragged out of either. i and j only set the direction:
    which atoms meet is up to the molecules' orientation and the dynamics.
    While i and j are further apart than 2 MAX_PULL / k (1.9 A at
    k = 100 kcal/mol/A^2) the force is the cap, whatever `k`."""
    g = np.zeros_like(x)
    for i, j, *mols in pairs:
        mid = 0.5 * (x[i] + x[j])
        if not mols:
            for a in (i, j):
                g[a] += capped(k * (x[a] - mid))
            continue
        if masses is None:
            raise ValueError("pull_pairs: a pair of molecules (i, j, A, B) needs the atoms' masses")
        f = capped(k * (x[i] - mid))
        for atoms, sign in zip(mols, (1.0, -1.0)):
            atoms = list(atoms)
            w = np.asarray(masses, dtype=float)[atoms]
            g[atoms] += sign * f * (w / w.sum())[:, None]
    return g


def held_distances(atoms, ref_bohr, k: float):
    """The restraint holding `atoms` at their distances to each other in
    `ref_bohr` (e.g. a QM/MM model's anchors): xtb's $constrain on an atom
    list, E = k (r - r0)^2 over every pair, `k` in Eh/bohr^2 as xtb's force
    constant. x -> (energy, gradient); None for fewer than two atoms."""
    atoms = sorted({int(a) for a in atoms})
    if len(atoms) < 2:
        return None
    i, j = (np.asarray(atoms)[p] for p in np.triu_indices(len(atoms), 1))
    ref = np.asarray(ref_bohr, dtype=float)
    r0 = np.linalg.norm(ref[i] - ref[j], axis=1)

    def restraint(x):
        d = x[i] - x[j]
        r = np.linalg.norm(d, axis=1)
        f = (2.0 * k * (r - r0) / np.maximum(r, 1e-12))[:, None] * d
        g = np.zeros_like(x)
        np.add.at(g, i, f)
        np.add.at(g, j, -f)
        return float(k * np.sum((r - r0) ** 2)), g
    return restraint


_UMA: dict = {}


def _uma_engine(server: Optional[str] = None):
    """This process's FAIRChemEngine for the MD (loaded once), or one asking the model server at `server`."""
    from mepd.engines.fairchem import FAIRChemEngine

    eng = _UMA.get(server)
    if eng is None:
        eng = _UMA[server] = FAIRChemEngine(model=UMA_MODEL)
        eng.server = server
    return eng


def uma_server() -> str:
    """The address of this process's UMA model server for MD (started once; it exits with this process)."""
    from mepd.engines.fairchem import serve

    if "served" not in _UMA:
        from mepd.engines.fairchem import FAIRChemEngine

        _UMA["served"] = serve(FAIRChemEngine(model=UMA_MODEL))
    return _UMA["served"].server


class LangevinMD:
    """`pos`/`vel` in bohr and bohr/fs; `wall`: its radius in Angstrom around
    the origin (None: no wall); `T`, `dt` (fs) and `friction` (1/fs) may change
    between steps. `method`: one of METHODS. GFN2-xTB and g-xTB run through
    axtb on one thread (GFN2-xTB in tblite's conventions: tblite's energies to
    ~1e-10 Eh; g-xTB in the program's), with Fermi smearing (ETEMPS: bonds
    break without SCF failures). "uma": mepd's FAIRChemEngine (UMA_MODEL,
    the omol head), in this process or, with `server` (an address from
    `uma_server`), on a model server that batches the steps of every
    trajectory that asks at once. `restraint`: x -> (energy, gradient) held in
    every step and in `relax` (unlike `bias`), e.g. `held_distances`."""

    def __init__(self, symbols, pos_bohr, charge: int, multiplicity: int, *, temperature: float,
                 wall: Optional[float] = None, seed: int = 0, dt: float = 0.5, friction: float = 0.02,
                 bias: Optional[Callable[[np.ndarray], np.ndarray]] = None, method: str = "gfn2",
                 server: Optional[str] = None, restraint: Optional[Callable] = None):
        from rdkit import Chem

        from mepd.discovery.nanoreactor import _masses

        method = {"gxtb": "gxtb-v2.0.1"}.get(method, method)
        if method not in METHODS:
            raise ValueError(f"LangevinMD method must be one of {', '.join(METHODS)}, not {method!r}.")
        table = Chem.GetPeriodicTable()
        self.pos = np.asarray(pos_bohr, dtype=float)
        self.masses = _masses(symbols)
        self.method = method
        uhf = int(multiplicity) - 1
        if method == "uma":
            from qcdata import Structure

            from mepd.qcdata_structure_helpers import structure_to_ase_atoms

            self._uma = _uma_engine(server)
            self._atoms = structure_to_ase_atoms(Structure(symbols=list(symbols), geometry=self.pos,
                                                           charge=int(charge), multiplicity=int(multiplicity)))
        self._calc_args = ([table.GetAtomicNumber(s) for s in symbols],
                           dict(charge=float(charge), uhf=uhf or None, threads=1,
                                conventions="tblite" if method == "gfn2" else "gxtb"))
        self._calcs = {}          # one axtb Calculator per smearing, made when first needed
        self.wall, self.wall_force = wall, 20.0 / KCAL * BOHR      # 20 kcal/mol/A, in Eh/bohr
        self.T, self.dt, self.friction, self.bias = float(temperature), dt, friction, bias
        self.restraint = restraint
        self.rng = np.random.default_rng(seed)
        self.vel = self.energy = self.grad = None
        self.failures = 0
        self.scf_retries = 0      # steps whose SCF converged only at another smearing

    def _calc(self, etemp: float):
        calc = self._calcs.get(etemp)
        if calc is None:
            import axtb

            numbers, kwds = self._calc_args
            calc = self._calcs[etemp] = axtb.Calculator(self.method, numbers, np.ascontiguousarray(self.pos),
                                                        etemp=etemp, **kwds)
        return calc

    def _reset(self) -> None:
        """The next SCF from the method's own guess, not the last wavefunction."""
        for calc in self._calcs.values():
            calc.reset_guess()

    def forces(self, x, biased: bool = True):
        from mepd.discovery.nanoreactor import _wall

        x = np.ascontiguousarray(x, dtype=float)
        if self.method == "uma":
            self._atoms.positions = x * BOHR
            energies, gradients = self._uma._predict([self._atoms])
            r = {"energy": energies[0], "gradient": gradients[0]}
        for k, etemp in enumerate(ETEMPS if self.method != "uma" else ()):
            try:
                r = self._calc(etemp).singlepoint(x, gradient=True)
                break
            except Exception:
                if k == len(ETEMPS) - 1:
                    raise
                self.scf_retries += 1
        e, g = float(r["energy"]), np.asarray(r["gradient"], dtype=float).reshape(x.shape)
        if self.wall is not None:
            ew, gw = _wall(x, self.wall / BOHR, self.wall_force)
            e, g = e + ew, g + gw
        if self.restraint is not None:
            er, gr = self.restraint(x)
            e, g = e + er, g + gr
        if biased and self.bias is not None:
            g = g + self.bias(x)
        return e, g

    def start(self) -> None:
        """Velocities from the Maxwell-Boltzmann distribution at `T`, and the first forces."""
        self.vel = self.rng.normal(size=self.pos.shape) * np.sqrt(KB * self.T / self.masses)[:, None]
        self.energy, self.grad = self.forces(self.pos)

    def step(self) -> bool:
        """One step; False if it failed (an SCF): the step is taken back and the
        velocities halved. Raises after more than 20 failures in a row. The
        bias is added outside the retry: a pull that raises is no SCF failure,
        and raises at once."""
        m = self.masses[:, None]
        c1 = math.exp(-self.friction * self.dt)
        c2 = np.sqrt((1 - c1 * c1) * KB * self.T / self.masses)[:, None]
        old = (self.pos.copy(), self.vel.copy(), self.grad.copy())
        try:
            self.vel -= 0.5 * self.dt * ACC * self.grad / m
            self.pos = self.pos + 0.5 * self.dt * self.vel
            self.vel = c1 * self.vel + c2 * self.rng.normal(size=self.vel.shape)
            self.pos = self.pos + 0.5 * self.dt * self.vel
            energy, grad = self.forces(self.pos, biased=False)
            if not np.all(np.isfinite(self.pos)):
                raise FloatingPointError("non-finite positions")
        except Exception:
            self.pos, self.vel, self.grad = old
            self.vel *= 0.5
            self._reset()
            self.failures += 1
            if self.failures > 20:
                raise
            return False
        self.energy = energy
        self.grad = grad + self.bias(self.pos) if self.bias is not None else grad
        self.vel -= 0.5 * self.dt * ACC * self.grad / m
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
