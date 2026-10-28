"""Mirror images in the explorer's kinetics (mepd.discovery.mirror)."""
import numpy as np
from qcdata import Structure
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.discovery.kinetics import Step, simulate
from mepd.discovery.mirror import MirrorFinder, is_chiral, mirrored
from mepd.nodes.node import StructureNode

SYMBOLS = ["C", "H", "F", "Cl", "Br"]
# CHFClBr: tetrahedral (chiral), and flat (a mirror plane: achiral)
TETRAHEDRAL = np.array([[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [0.78, -0.78, -0.78],
                        [-1.02, 1.02, -1.02], [-1.10, -1.10, 1.10]])
FLAT = np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [0.0, 1.35, 0.0], [-1.77, 0.0, 0.0], [0.0, -1.93, 0.0]])
# the H gone: CFClBr + H, flat and apart (another species, also achiral)
APART = FLAT + np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def _st(xyz):
    return Structure(symbols=SYMBOLS, geometry=np.asarray(xyz) * ANGSTROM_TO_BOHR, charge=0, multiplicity=1)


def _node(xyz):
    return StructureNode(structure=_st(xyz))


def test_chirality_of_a_geometry():
    assert is_chiral(_st(TETRAHEDRAL)) is True
    assert is_chiral(_st(FLAT)) is False
    methane = Structure(symbols=["C", "H", "H", "H", "H"], charge=0, multiplicity=1,
                        geometry=np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                                           [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]) * ANGSTROM_TO_BOHR)
    assert is_chiral(methane) is False   # its hydrogens permute onto the mirror image


def _final(n, steps, energies):
    return simulate(n, steps, energies, temperature=300.0, time_s=10.0).final_concentration


def test_a_chiral_ts_between_achiral_species_counts_twice():
    species = [_node(FLAT), _node(APART)]
    steps = [Step(0, 1, 0.025, "s", {})]
    mirrors = MirrorFinder()(species, steps, [_st(TETRAHEDRAL)])
    assert mirrors.species_mirror == [0, 1] and not mirrors.virtual_of and not mirrors.chiral_conformers
    assert mirrors.twins == [(0, 0, 1)]
    n, all_steps, g = mirrors.extend(steps, [0.0, 0.0], 300.0)
    assert n == 2 and len(all_steps) == 2 and all_steps[1].ts_energy == steps[0].ts_energy
    # the same as one channel of twice the rate: what counting the twin means
    k1 = simulate(2, steps, [0.0, 0.0], temperature=300.0, time_s=1.0)
    k2 = simulate(n, all_steps, g, temperature=300.0, time_s=1.0)
    assert k2.flux[1] > 1.9 * k1.flux[1]


def test_a_twin_already_found_is_not_added_again():
    species = [_node(FLAT), _node(APART)]
    steps = [Step(0, 1, 0.025, "s", {}), Step(1, 0, 0.025, "t", {})]
    mirrors = MirrorFinder()(species, steps, [_st(TETRAHEDRAL), mirrored(_st(TETRAHEDRAL))])
    assert mirrors.twins == [] and mirrors.found_twins == [(0, 1)]


def test_an_achiral_ts_has_no_twin():
    species = [_node(FLAT), _node(APART)]
    steps = [Step(0, 1, 0.025, "s", {})]
    assert MirrorFinder()(species, steps, [_st(FLAT + [[0, 0, 0], [0.5, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])]).twins == []


def test_the_enantiomer_of_a_chiral_product_joins_the_kinetics():
    species = [_node(FLAT), _node(TETRAHEDRAL)]
    ts = TETRAHEDRAL * 0.9 + FLAT * 0.1
    steps = [Step(0, 1, 0.025, "s", {})]
    mirrors = MirrorFinder()(species, steps, [_st(ts)])
    assert mirrors.virtual_of == [1] and mirrors.species_mirror == [0, 2, 1]
    assert mirrors.twins == [(0, 0, 2)]
    n, all_steps, g = mirrors.extend(steps, [0.0, -0.01], 300.0)
    c = _final(n, all_steps, g)
    assert n == 3 and abs(c[1] - c[2]) < 1e-9 and c[1] > 0.4   # a racemate


def test_a_chiral_conformer_of_an_achiral_molecule_counts_both_minima():
    # twisted H2O2-like: an achiral molecule (no stereo centre) in a chiral conformer
    sym = ["O", "O", "H", "H"]
    xyz = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [-0.3, 0.92, 0.2], [1.75, 0.2, 0.92]])
    node = StructureNode(structure=Structure(symbols=sym, geometry=xyz * ANGSTROM_TO_BOHR, charge=0, multiplicity=1))
    mirrors = MirrorFinder()([node], [], [])
    assert mirrors.species_mirror == [0] and mirrors.chiral_conformers == {0: 1}
    _, _, g = mirrors.extend([], [0.0], 300.0)
    rt_ln2_kcal = 0.0019872 * 300.0 * np.log(2.0)
    assert abs(g[0] * 627.5094740631 + rt_ln2_kcal) < 1e-3
