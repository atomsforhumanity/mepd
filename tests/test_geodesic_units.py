"""Geodesic interpolation works in Angstrom (its radii, pair cutoffs and
nudges are Angstrom) while mepd's coordinates are Bohr: run_geodesic must
convert on the way in and back on the way out."""
from __future__ import annotations

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from qcdata import Structure

from mepd import chainhelpers as ch
from mepd.geodesic_interpolation2.morsegeodesic import run_geodesic_get_smoother
from mepd.nodes.node import StructureNode

# Acetaldehyde -> vinyl alcohol (1,3-H shift), Angstrom.
SYMBOLS = ["C", "C", "O", "H", "H", "H", "H"]
ETHANAL = np.array([[-1.27, -0.10, 0.00], [0.20, 0.10, 0.00], [0.80, 1.16, 0.00], [-1.80, 0.85, 0.00],
                    [-1.53, -0.67, 0.89], [-1.53, -0.67, -0.89], [0.78, -0.85, 0.00]])
VINYL_ALCOHOL = np.array([[-1.23, -0.15, 0.00], [0.10, 0.05, 0.00], [0.75, 1.24, 0.00], [1.70, 1.10, 0.00],
                          [-1.65, -1.15, 0.00], [-1.90, 0.70, 0.00], [0.75, -0.82, 0.00]])


def _node(xyz):
    return StructureNode(structure=Structure(symbols=SYMBOLS, geometry=xyz * ANGSTROM_TO_BOHR,
                                             charge=0, multiplicity=1))


def test_run_geodesic_is_the_angstrom_geodesic_in_bohr():
    chain, smoother = ch.run_geodesic([_node(ETHANAL), _node(VINYL_ALCOHOL)], nimages=7, return_smoother=True)
    ends = np.array([_node(ETHANAL).coords, _node(VINYL_ALCOHOL).coords]) * BOHR_TO_ANGSTROM
    ref = run_geodesic_get_smoother((SYMBOLS, ends), nimages=7)
    got = np.array([n.coords for n in chain.nodes])
    np.testing.assert_allclose(got, ref.path / BOHR_TO_ANGSTROM, atol=1e-10)
    np.testing.assert_allclose(smoother.path, got)
    assert smoother.nrij == ref.nrij


def test_bonds_read_as_unstretched_in_the_geodesic_metric():
    """A bond at its normal length has w close to 1 (C-H read 0.2 when Bohr
    went in), and the 3 A pair cutoff reaches the O...methyl H pairs (2.6 A,
    beyond the 1.6 A it amounted to in Bohr)."""
    _, smoother = ch.run_geodesic([_node(ETHANAL), _node(VINYL_ALCOHOL)], nimages=5, return_smoother=True)
    pairs = [tuple(int(i) for i in p) for p in smoother.rij_list_np]
    assert 0.8 < smoother.w[0][pairs.index((0, 4))] < 1.2   # C-H
    assert (2, 4) in pairs
