"""Design's Move tool: a dragged atom goes exactly where it was put, and
nothing else moves or relaxes (hand-made geometries stay as drawn)."""
import numpy as np
import pytest

from mepd.web import design
from mepd.web.workspace import WorkspaceError


def _methane_block():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    return Chem.MolToMolBlock(m)


def _coords(block):
    lines = block.split("\n")
    n = int(lines[3][:3])
    return np.array([[float(x) for x in l.split()[:3]] for l in lines[4:4 + n]])


def test_move_puts_one_atom_there_and_leaves_the_rest():
    block = _methane_block()
    before = _coords(block)
    out = design.edit(block, {"op": "move", "atom": 1, "xyz": [1.5, 0.2, -0.3]})
    after = _coords(out["molblock"])
    assert np.allclose(after[1], [1.5, 0.2, -0.3], atol=1e-3)
    others = [i for i in range(len(before)) if i != 1]
    assert np.allclose(after[others], before[others], atol=1e-3)
    assert out["changed"] == [1]


@pytest.mark.parametrize("xyz", [None, [1, 2], ["a", 0, 0], [float("nan"), 0, 0]])
def test_move_refuses_bad_coordinates(xyz):
    with pytest.raises(WorkspaceError):
        design.edit(_methane_block(), {"op": "move", "atom": 0, "xyz": xyz})
