"""Harmonic distance restraints: E = k/2 (r - r0)^2 between two atoms, added
to an engine's energies and gradients.

For holding a bond while the rest relaxes, e.g. to test whether a Michaelis
complex is a minimum with its C-H kept at 1.10 Å (frozen atoms cannot do it
for QM atoms: a QM/MM region's QM atoms always move). In a profile:

    [[restraints]]
    atoms = [12, 45]          # 0-based atom indices
    distance = 1.10           # Å
    force_constant = 500      # kcal/mol/Å² (default)

Restraints act on everything run with the profile (optimizations, paths, TS
searches, IRCs): the energies reported include the restraint energy, which is
small once the restraint holds. On a QM/MM system they go into the QM/MM
energy itself, so microiterative optimizations keep working; there they may
involve only atoms the outer optimizer moves (QM atoms and link partners).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.engines.engine import Engine
from mepd.engines.modified import ModifiedEngine
from mepd.errors import InputsError

KCAL_MOL_PER_HARTREE = 627.509474
DEFAULT_FORCE_CONSTANT = 500.0   # kcal/mol/Å²: ~0.03 Å off target per 15 kcal/mol/Å of opposing force


@dataclass(frozen=True)
class Restraint:
    i: int
    j: int
    distance: float                      # Å
    force_constant: float = DEFAULT_FORCE_CONSTANT   # kcal/mol/Å²

    @property
    def k_au(self) -> float:
        """Force constant in Eh/bohr²."""
        return self.force_constant / KCAL_MOL_PER_HARTREE / ANGSTROM_TO_BOHR ** 2

    @property
    def r0_bohr(self) -> float:
        return self.distance * ANGSTROM_TO_BOHR


def parse_restraints(items) -> list[Restraint]:
    """Restraints from a profile's `[[restraints]]` tables (or Restraint objects)."""
    out = []
    for k, item in enumerate(items or []):
        if isinstance(item, Restraint):
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise InputsError(f"restraints[{k}] must be a table with atoms = [i, j] and distance (Å), not {item!r}")
        unknown = set(item) - {"atoms", "distance", "force_constant"}
        if unknown:
            raise InputsError(f"restraints[{k}]: unknown key(s) {', '.join(sorted(unknown))} "
                              "(use atoms, distance, force_constant)")
        atoms = item.get("atoms")
        if not isinstance(atoms, (list, tuple)) or len(atoms) != 2 or int(atoms[0]) == int(atoms[1]):
            raise InputsError(f"restraints[{k}]: atoms must be two different 0-based atom indices, not {atoms!r}")
        if "distance" not in item:
            raise InputsError(f"restraints[{k}]: give the target distance in Å (distance = ...)")
        r = Restraint(int(atoms[0]), int(atoms[1]), float(item["distance"]),
                      float(item.get("force_constant", DEFAULT_FORCE_CONSTANT)))
        if r.i < 0 or r.j < 0 or r.distance <= 0:
            raise InputsError(f"restraints[{k}]: atom indices must be >= 0 and the distance > 0 Å")
        if r.force_constant <= 0:
            raise InputsError(f"restraints[{k}]: force_constant must be > 0 kcal/mol/Å²")
        out.append(r)
    return out


def restraint_energy_gradient(restraints: list[Restraint], coords_bohr) -> tuple[float, np.ndarray]:
    """(energy Eh, gradient Eh/bohr) of the restraints at one geometry."""
    x = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    e, g = 0.0, np.zeros_like(x)
    for r in restraints:
        if max(r.i, r.j) >= len(x):
            raise InputsError(f"restraint on atoms {r.i}, {r.j}: the structure has only {len(x)} atoms")
        d = x[r.i] - x[r.j]
        dist = float(np.linalg.norm(d))
        dev = dist - r.r0_bohr
        e += 0.5 * r.k_au * dev ** 2
        if dist > 1e-12:
            f = r.k_au * dev * d / dist
            g[r.i] += f
            g[r.j] -= f
    return e, g


def restraint_report(restraints: list[Restraint], coords_bohr) -> list[str]:
    """One line per restraint: target, actual distance, energy."""
    x = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    lines = []
    for r in restraints:
        dist = float(np.linalg.norm(x[r.i] - x[r.j])) / ANGSTROM_TO_BOHR
        e = 0.5 * r.force_constant * (dist - r.distance) ** 2
        lines.append(f"atoms {r.i}-{r.j}: {dist:.3f} Å (target {r.distance:.3f}, {e:.2f} kcal/mol)")
    return lines


@dataclass
class RestrainedEngine(ModifiedEngine):
    """`base` plus harmonic distance restraints."""

    base: Engine
    restraints: Optional[list] = None

    def __post_init__(self):
        self.restraints = parse_restraints(self.restraints)
        self._inherit()

    def _term(self, nodes):
        if not self.restraints:
            return None
        parts = [restraint_energy_gradient(self.restraints, n.coords) for n in nodes]
        return np.array([p[0] for p in parts]), np.array([p[1] for p in parts])
