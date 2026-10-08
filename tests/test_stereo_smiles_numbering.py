"""The stereochemistry check (`_is_connectivity_identical`) compares SMILES
built from 3D. RDKit's bond orders from 3D can depend on the atom numbering
alone: this reactant (KAIST RXN_0000114217, a two-molecule complex with a
hypervalent-capable P) gave two SMILES across renumberings, so built
partners were rejected as a stereo change they did not have."""

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
    assert len({_stereo_smiles(StructureNode(structure=_renumbered(s, o))) for o in orders}) == 1


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
