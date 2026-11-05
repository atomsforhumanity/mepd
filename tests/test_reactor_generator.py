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
    inter = [p for p in picks if (p[0] < 5) != (p[1] < 5)]
    assert inter and all(SYM[a % 5] != "H" and SYM[b % 5] != "H" for a, b, *_ in inter)
    # the two whole molecules go with the pair: they are pulled together, not the two atoms
    assert all(len(p) == 4 and p[0] in p[2] and p[1] in p[3] and sorted(p[2] + p[3]) == list(range(10)) for p in inter)
    assert all(len(p) == 2 for p in picks if (p[0] < 5) == (p[1] < 5))


def test_a_pull_between_molecules_moves_them_whole():
    from mepd.discovery.langevin import pull_pairs
    from mepd.discovery.nanoreactor import _masses

    sym = SYM * 2
    x = np.vstack([XYZ, XYZ + [6.0, 1.0, 0]]) / 0.529177210903
    m = _masses(sym)
    atom = pull_pairs(x, [(1, 6)], 0.05)
    whole = pull_pairs(x, [(1, 6, tuple(range(5)), tuple(range(5, 10)))], 0.05, m)
    assert np.allclose(whole[:5].sum(0), atom[1]) and np.allclose(whole[5:].sum(0), atom[6])   # the same force
    accel = whole / m[:, None]                 # every atom of a molecule accelerated alike: nothing pulled out of it
    assert np.allclose(accel[:5], accel[0]) and np.allclose(accel[5:], accel[5])
    assert np.count_nonzero(atom.any(axis=1)) == 2    # within one molecule a pull still acts on the two atoms
    with pytest.raises(ValueError, match="masses"):
        pull_pairs(x, [(1, 6, tuple(range(5)), tuple(range(5, 10)))], 0.05)


def test_pushes_between_two_molecules_lose_no_step_of_a_trajectory(monkeypatch):
    pytest.importorskip("tblite")
    import mepd.discovery.reactor_generator as rg

    picks, pick = [], rg.collision_pair
    monkeypatch.setattr(rg, "collision_pair", lambda *a, **kw: picks.append(pick(*a, **kw)) or picks[-1])
    xyz = np.vstack([XYZ, XYZ + [5.0, 0.0, 0.0]])
    # a push every ~25 fs for 0.2 ps; a pull that raised was taken for an SCF failure and cut the trajectory short
    frames = rg._trajectory(SYM * 2, xyz, 0, 1, temperature=300.0, time_ps=0.2, compress=1.0, period_ps=1.0,
                            duty=0.75, pull_fs=20.0, gap_fs=5.0, k=100.0, intra_hops=2, seed=0)
    assert any(p is not None and len(p) == 4 for p in picks)       # pushes between the two molecules happened
    assert len(frames) == round(200.0 / rg.TRAJ_FS) + 1             # and every frame came back


def test_pulls_between_molecules_prefer_no_species():
    """Two waters and a methane: every molecule is picked alike, so two waters meet in about a third of the
    pulls between molecules (the Reactor's own pick avoided pairing two waters; user 2026-11-05: no species
    preferred in an unattended run)."""
    sym = ["O", "H", "H"] * 2 + ["C", "H", "H", "H", "H"]
    water = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    methane = np.array([[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63],
                        [0.63, -0.63, -0.63]])
    xyz = np.vstack([water, water + [6.0, 0, 0], methane + [0, 6.0, 0]])
    rng = np.random.default_rng(3)
    mol = lambda i: 0 if i < 3 else (1 if i < 6 else 2)
    inter = [(a, b) for a, b, *_ in (collision_pair(sym, xyz, rng) for _ in range(3000)) if mol(a) != mol(b)]
    both_water = sum(mol(a) < 2 and mol(b) < 2 for a, b in inter) / len(inter)
    assert both_water == pytest.approx(1 / 3, abs=0.05)


def test_the_reactor_finds_both_formic_acid_channels():
    pytest.importorskip("tblite")
    from types import SimpleNamespace

    from mepd.discovery.reactor_generator import reactor_products

    s = SimpleNamespace(symbols=SYM, geometry=XYZ * 1.8897259886, charge=0, multiplicity=1,
                        model_copy=lambda update: SimpleNamespace(**update))
    # 2200 K (the default): at 1500 K CO + H2O forms in only ~3-9% of trajectories, much of it from pulling the H2
    # of CO2 + H2 back onto CO2, which whole molecules pushed together rarely do
    out = reactor_products(s, temperature=2200, time_ps=5, n_seeds=16, parallel=4, seed=0)
    graphs = {_bonds(SYM, np.asarray(p.geometry) / 1.8897259886) for p in out}
    assert frozenset({(1, 0), (2, 1), (4, 3)}) in graphs                                   # CO2 + H2
    # CO + H2O, the water on either oxygen (products are told apart by SMILES, so one of the two is proposed)
    bond = lambda a, b: (max(a, b), min(a, b))
    assert any({bond(c, 1), bond(3, w), bond(4, w)} <= g and bond(1, w) not in g
               for g in graphs for c, w in ((2, 0), (0, 2)))


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
        return [{"states": [(1, "A", XYZ.tolist())], "keys": [["A", True, 5.0]] * 10 + [["B", True, 5.0]] * 10,
                 "events": 20, "torn": 0} for _ in jobs]

    monkeypatch.setattr(rg, "_run_in_processes", fake)
    s = Structure(symbols=SYM, geometry=XYZ / 0.529177210903)
    assert len(rg.reactor_products(s, n_seeds=2, parallel=1)) == 1 and batches == [2]

    batches.clear()
    monkeypatch.setattr(rg, "_run_in_processes", lambda jobs, width: batches.append(1) or [
        {"states": [], "keys": [[f"new{len(batches)}", True, 5.0]], "events": 1, "torn": 0}])
    rg.reactor_products(s, n_seeds=1, parallel=1, max_batches=3)
    assert len(batches) == 3      # every event a new product: never converged, stops at max_batches

    batches.clear()                # one product, but too few reactions of the seed to tell (< 1/unseen)
    monkeypatch.setattr(rg, "_run_in_processes", lambda jobs, width: batches.append(1) or [
        {"states": [], "keys": [["A", True, 5.0], ["C", False, 5.0]], "events": 2, "torn": 0}])
    rg.reactor_products(s, n_seeds=1, parallel=1, max_batches=3)
    assert len(batches) == 3



def test_products_above_the_cutoff_are_proposed_but_dont_hold_up_convergence(monkeypatch):
    import mepd.discovery.reactor_generator as rg
    from qcdata import Structure

    batches = []

    def fake(jobs, width):   # one product within reach again and again, and a new one far above it every time
        batches.append(1)
        hi = f"high{len(batches)}"   # 2 of 22 events seen once: 9%
        return [{"states": [(1, "A", XYZ.tolist()), (2, hi, (XYZ + 0.1).tolist())],
                 "keys": [["A", True, 5.0]] * 20 + [[hi, True, 150.0], [hi + "b", True, 150.0]],
                 "events": 22, "torn": 0}]

    monkeypatch.setattr(rg, "_run_in_processes", fake)
    s = Structure(symbols=SYM, geometry=XYZ / 0.529177210903)
    rg.reactor_products(s, n_seeds=1, parallel=1, max_batches=3)
    assert len(batches) == 3                       # without a cutoff: a new product every batch
    batches.clear()
    out = rg.reactor_products(s, n_seeds=1, parallel=1, max_batches=3, cutoff_kcal=80.0)
    assert len(batches) == 1 and len(out) == 2     # converged at once; the high product still proposed


def test_the_explorer_passes_the_reactor_what_is_left_of_the_access_cutoff():
    from types import SimpleNamespace

    from mepd.discovery.network_expansion import _generator_cutoff

    sp = SimpleNamespace(rel_energy_kcal=20.0)
    assert _generator_cutoff("reactor", {"n_seeds": 4}, 88.0, sp) == {"n_seeds": 4, "cutoff_kcal": 68.0}
    assert _generator_cutoff("reactor", {}, None, sp) == {}
    assert _generator_cutoff("collide", {}, 88.0, object()) == {}


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
