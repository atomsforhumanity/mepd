"""An uploaded ion pair splits into ions, not radicals (mepd#14)."""
import numpy as np
import pytest
from qcdata import Structure
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.web import compose


def _ion_pair():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("c1cc[nH+]cc1.[O-]C(=O)C(F)(F)F"))
    AllChem.EmbedMolecule(m, randomSeed=3)
    x = m.GetConformer().GetPositions()
    cation, anion = (list(f) for f in Chem.GetMolFrags(m))
    x[anion] += x[cation].mean(0) - x[anion].mean(0) + [4.5, 0, 0]
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    hint = np.zeros(len(syms))
    hint[cation], hint[anion] = 1 / len(cation), -1 / len(anion)
    return Structure(symbols=syms, geometry=x * ANGSTROM_TO_BOHR, charge=0, multiplicity=1), hint


def test_partial_charges_turn_two_radicals_into_an_ion_pair(monkeypatch):
    s, hint = _ion_pair()
    monkeypatch.setattr(compose, "_partial_charges", lambda *a: None)
    assert sorted((f.charge, f.multiplicity) for f in compose._fragments(s)) == [(0, 2), (0, 2)]
    monkeypatch.setattr(compose, "_partial_charges", lambda *a: hint)
    assert sorted((f.charge, f.multiplicity, len(f.symbols)) for f in compose._fragments(s)) == [(-1, 1, 7), (1, 1, 12)]


def test_ion_pair_with_xtb():
    from mepd import programs

    if not programs.xtb_executable(download=False):
        pytest.skip("no xtb")
    s, _ = _ion_pair()
    assert sorted(f.charge for f in compose._fragments(s)) == [-1, 1]
