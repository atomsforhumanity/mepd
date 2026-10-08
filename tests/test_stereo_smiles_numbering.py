"""The stereochemistry check (`_is_connectivity_identical`) compares SMILES
built from 3D. RDKit perceiving the bonds from 3D can join close fragments,
and which way a borderline case falls depended on the atom numbering alone:
this reactant (KAIST RXN_0000114217, a two-molecule complex with a P...P
contact) read as two molecules or as one with P=P, so built partners were
rejected as a stereo change they did not have. The SMILES is now built on
mepd's own bond graph, and a real stereo change must still be seen."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pynauty")

XYZ = Path(__file__).parent / "data" / "smiles_numbering_dependent.xyz"


def _renumbered(structure, order):
    from qcdata import Structure

    return Structure(symbols=[structure.symbols[i] for i in order], geometry=np.asarray(structure.geometry)[order],
                     charge=structure.charge, multiplicity=structure.multiplicity)


def test_one_smiles_whatever_the_numbering():
    from qcdata import Structure
    from qcinf import structure_to_smiles

    from mepd.nodes.node import StructureNode
    from mepd.nodes.nodehelpers import _stereo_smiles

    s = Structure.open(XYZ)
    rng = np.random.default_rng(0)
    orders = [rng.permutation(len(s.symbols)).tolist() for _ in range(30)]
    assert len({structure_to_smiles(_renumbered(s, o)) for o in orders}) > 1     # the problem, still in RDKit
    ours = {_stereo_smiles(StructureNode(structure=_renumbered(s, o)))[0] for o in orders}
    assert len(ours) == 1 and "." in ours.pop()     # one answer, and two molecules


def test_a_renumbered_structure_is_identical_to_itself():
    from qcdata import Structure

    from mepd.nodes.node import StructureNode
    from mepd.nodes.nodehelpers import _is_connectivity_identical

    s = Structure.open(XYZ)
    rng = np.random.default_rng(1)
    for _ in range(10):
        o = rng.permutation(len(s.symbols)).tolist()
        r = _renumbered(s, o)
        inv = np.argsort(o).tolist()
        back = _renumbered(r, inv)      # same numbering as s, reached through another one
        assert _is_connectivity_identical(StructureNode(structure=back), StructureNode(structure=s),
                                          verbose=False, collect_comparison=False)
        assert _is_connectivity_identical(StructureNode(structure=r), StructureNode(structure=_renumbered(s, o)),
                                          verbose=False, collect_comparison=False)


def _embedded(smiles: str):
    from qcconst.constants import ANGSTROM_TO_BOHR
    from qcdata import Structure
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    return Structure(symbols=[a.GetSymbol() for a in m.GetAtoms()],
                     geometry=m.GetConformer().GetPositions() * ANGSTROM_TO_BOHR, charge=0, multiplicity=1)


@pytest.mark.parametrize("a, b", [("C[C@H](N)O", "C[C@@H](N)O"), (r"C/C=C/C", r"C/C=C\C")])
def test_a_stereo_change_is_still_a_change(a, b):
    """An inverted centre and cis vs trans, same atoms in the same order."""
    from mepd.nodes.node import StructureNode
    from mepd.nodes.nodehelpers import _is_connectivity_identical

    x, y = _embedded(a), _embedded(b)
    assert list(x.symbols) == list(y.symbols)
    assert _is_connectivity_identical(StructureNode(structure=x), StructureNode(structure=x),
                                      verbose=False, collect_comparison=False)
    assert not _is_connectivity_identical(StructureNode(structure=x), StructureNode(structure=y),
                                          verbose=False, collect_comparison=False)
