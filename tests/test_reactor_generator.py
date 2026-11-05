"""The reactor generator: the interactive Reactor (piston and random Collide), unattended."""
import numpy as np
import pytest

from mepd.discovery.collide_generator import _bonds
from mepd.discovery.reactor_generator import collision_pair

# formic acid (Z), Angstrom: O0 C1 O2(=) H3(on O0) H4(on C1)
SYM = ["O", "C", "O", "H", "H"]
XYZ = np.array([[0.74, -0.62, -0.08], [-0.41, 0.06, -0.16], [-0.56, 1.17, 0.27], [1.40, -0.07, 0.37],
                [-1.17, -0.55, -0.67]])


def test_pulls_in_one_molecule_reach_every_pair_two_bonds_apart():
    rng = np.random.default_rng(0)
    pairs = {collision_pair(SYM, XYZ, rng) for _ in range(300)}
    assert {(4, 0), (4, 3), (3, 2)} <= pairs          # dehydration, H2 loss, the H shift
    assert (1, 0) not in pairs and (4, 1) not in pairs  # bonded atoms are never pulled


def test_pulls_between_molecules_take_two_heavy_atoms_of_different_molecules():
    xyz = np.vstack([XYZ, XYZ + [6.0, 0, 0]])
    rng = np.random.default_rng(1)
    picks = [collision_pair(SYM * 2, xyz, rng) for _ in range(200)]
    inter = [(a, b) for a, b in picks if (a < 5) != (b < 5)]
    assert inter and all(SYM[a % 5] != "H" and SYM[b % 5] != "H" for a, b in inter)


def test_the_reactor_finds_both_formic_acid_channels():
    pytest.importorskip("tblite")
    from types import SimpleNamespace

    from mepd.discovery.reactor_generator import reactor_products

    s = SimpleNamespace(symbols=SYM, geometry=XYZ * 1.8897259886, charge=0, multiplicity=1,
                        model_copy=lambda update: SimpleNamespace(**update))
    out = reactor_products(s, temperature=1500, time_ps=5, n_seeds=16, parallel=4, seed=0)
    graphs = {_bonds(SYM, np.asarray(p.geometry) / 1.8897259886) for p in out}
    assert frozenset({(1, 0), (2, 1), (4, 3)}) in graphs                                   # CO2 + H2
    assert any((2, 1) in g and (3, 0) in g and (4, 0) in g and (1, 0) not in g for g in graphs)   # CO + H2O


def _embedded(smiles):
    from qcdata import Structure
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return Structure(symbols=[a.GetSymbol() for a in mol.GetAtoms()],
                     geometry=mol.GetConformer().GetPositions() / 0.529177210903)


def test_two_diastereomers_of_one_bond_graph_are_two_products():
    from mepd.discovery.nanoreactor_generator import stereo_key

    cis, trans = _embedded("C[C@H]1C[C@@H]1C"), _embedded("C[C@H]1C[C@H]1C")
    assert stereo_key(cis) != stereo_key(trans)


def test_batches_stop_once_few_reactions_of_the_seed_give_a_product_seen_once(monkeypatch):
    import mepd.discovery.reactor_generator as rg
    from qcdata import Structure

    batches = []

    def fake(jobs, width):            # every trajectory finds the same two products: converged after one batch
        batches.append(len(jobs))
        return [{"states": [(1, "A", XYZ.tolist())], "keys": [["A", True]] * 10 + [["B", True]] * 10,
                 "events": 20, "torn": 0} for _ in jobs]

    monkeypatch.setattr(rg, "_run_in_processes", fake)
    s = Structure(symbols=SYM, geometry=XYZ / 0.529177210903)
    assert len(rg.reactor_products(s, n_seeds=2, parallel=1)) == 1 and batches == [2]

    batches.clear()
    monkeypatch.setattr(rg, "_run_in_processes", lambda jobs, width: batches.append(1) or [
        {"states": [], "keys": [[f"new{len(batches)}", True]], "events": 1, "torn": 0}])
    rg.reactor_products(s, n_seeds=1, parallel=1, max_batches=3)
    assert len(batches) == 3      # every event a new product: never converged, stops at max_batches

    batches.clear()                # one product, but too few reactions of the seed to tell (< 1/unseen)
    monkeypatch.setattr(rg, "_run_in_processes", lambda jobs, width: batches.append(1) or [
        {"states": [], "keys": [["A", True], ["C", False]], "events": 2, "torn": 0}])
    rg.reactor_products(s, n_seeds=1, parallel=1, max_batches=3)
    assert len(batches) == 3


def test_an_event_that_only_renumbers_identical_atoms_is_no_product(monkeypatch):
    pytest.importorskip("tblite")
    import mepd.discovery.reactor_generator as rg

    # formic acid's two hydrogens trade places: new bonds by atom number, the same molecule
    swapped = XYZ.copy()
    swapped[[3, 4]] = swapped[[4, 3]]
    frames = np.array([XYZ] * 100 + [swapped] * 100)
    monkeypatch.setattr(rg, "_trajectory", lambda *a, **k: frames)
    out = rg._states(SYM, XYZ, 0, 1, 20, {})
    assert out["events"] >= 1 and out["keys"] == [] and out["states"] == []
