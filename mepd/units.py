"""Unit conventions and conversion constants.

Inside mepd, coordinates are Bohr and energies are Hartree (qcdata's
convention): `node.coords`, `node.energy`, `engine.compute_energies(...)`.
Files are Angstrom (xyz). A number in any other unit says so in its name,
e.g. `barrier_kcal` or `rmsd_angstrom`.

Use these constants instead of literals, so every script converts the same way.
"""
from qcconst.constants import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
)

HARTREE_TO_KCAL = HARTREE_TO_KCAL_PER_MOL
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_PER_MOL

__all__ = [
    "ANGSTROM_TO_BOHR",
    "BOHR_TO_ANGSTROM",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL",
    "HARTREE_TO_KCAL_PER_MOL",
    "KCAL_TO_HARTREE",
]
