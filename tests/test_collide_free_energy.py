"""The collide generator (the Reactor's Collide, unattended) and free-energy kinetics."""
import numpy as np
import pytest

from mepd.discovery.collide_generator import _bonds, collision_moves
from mepd.discovery.free_energy import FreeEnergies, molecules

# formic acid (Z), Angstrom: O0 C1 O2(=) H3(on O0) H4(on C1)
SYM = ["O", "C", "O", "H", "H"]
XYZ = np.array([[0.74, -0.62, -0.08], [-0.41, 0.06, -0.16], [-0.56, 1.17, 0.27], [1.40, -0.07, 0.37],
                [-1.17, -0.55, -0.67]])


def test_moves_pull_pairs_two_or_more_bonds_apart_and_push_a_bond_next_to_them():
    moves = collision_moves(SYM, XYZ)
    pulls = {(i, j) for i, j, push in moves if push is None}
    assert (4, 0) in pulls and (4, 3) in pulls          # H(C) -> O(H): dehydration; H...H: H2 loss
    assert (1, 0) not in pulls                          # bonded: never pulled
    # the four-centre elimination: pull H(C) onto O(H) while the C-O(H) bond is dragged apart
    assert (4, 0, (1, 0)) in moves


def test_collisions_find_both_formic_acid_channels():
    pytest.importorskip("tblite")
    from types import SimpleNamespace

    from mepd.discovery.collide_generator import collide_products

    s = SimpleNamespace(symbols=SYM, geometry=XYZ * 1.8897259886, charge=0, multiplicity=1,
                        model_copy=lambda update: SimpleNamespace(**update))
    out = collide_products(s, n_collisions=32, parallel=4, seed=0)
    graphs = {_bonds(SYM, np.asarray(p.geometry) / 1.8897259886) for p in out}
    assert frozenset({(2, 1), (4, 3)}) | {(1, 0)} in graphs              # CO2 + H2
    assert any((2, 1) in g and (3, 0) in g and (4, 0) in g and (1, 0) not in g for g in graphs)   # CO + H2O


def test_a_species_of_two_molecules_counts_them_apart(monkeypatch):
    # CO + H2O as one species: two molecules, each its own correction
    sym = ["O", "C", "O", "H", "H"]
    xyz = np.array([[0, 0, 0], [3.0, 0, 0], [4.13, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], dtype=float)
    assert sorted(map(sorted, molecules(sym, xyz))) == [[0, 3, 4], [1, 2]]
    fe = FreeEnergies(engine=None, temperature=1500.0)
    seen = []
    monkeypatch.setattr(fe, "_one", lambda s, x, q, m, ts: seen.append((tuple(s), q, m, ts)) or -40.0)
    from types import SimpleNamespace

    from qcdata import Structure

    st = Structure(symbols=sym, geometry=xyz * 1.8897259886, charge=0, multiplicity=1)
    monkeypatch.setattr("mepd.energies", lambda nodes, eng: [0.0] * len(nodes))
    assert fe._correction(st, ts=False) == -80.0
    assert sorted(seen) == [(("C", "O"), 0, 1, False), (("O", "H", "H"), 0, 1, False)]
    seen.clear()
    assert fe._correction(st, ts=True) == -40.0 and len(seen) == 1    # a TS is one molecule
