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



def test_a_collision_of_two_molecules_moves_them_whole_and_no_move_is_a_template(monkeypatch):
    import mepd.discovery.collide_generator as cg
    import mepd.discovery.langevin as lg

    xyz = np.vstack([XYZ, XYZ + [6.0, 0.0, 0.0]])
    # one move per atom pair (H included, within a molecule two or more bonds apart), each a single pull and
    # at most one bond pushed apart: no move of two pulls at once (the [4+2] template is gone)
    assert all(len(m) == 3 for m in cg.collision_moves(SYM * 2, xyz))
    pulled, real = [], lg.pull_pairs
    monkeypatch.setattr(lg, "pull_pairs",
                        lambda x, pairs, k, masses=None: pulled.append(pairs) or real(x, pairs, k, masses))

    class MD:                       # the dynamics left out: only what is pulled matters here
        dt = 0.5

        def __init__(self, symbols, pos, *args, bias=None, **kwargs):
            self.pos, self.bias = np.asarray(pos), bias

        def start(self):
            pass

        def step(self):
            self.bias(self.pos)
            return True

        def relax(self, n):
            pass

    monkeypatch.setattr(lg, "LangevinMD", MD)
    end = cg._collide(SYM * 2, xyz, 0, 1, (1, 6), 1000.0, 1.0, 0.0, 100.0, 0)
    assert not isinstance(end, str), end          # _collide returns an error as text
    (i, j, A, B), = pulled[0]
    assert (i, j) == (1, 6) and sorted(A) == list(range(5)) and sorted(B) == list(range(5, 10))
    pulled.clear()
    end = cg._collide(SYM * 2, xyz, 0, 1, (4, 0), 1000.0, 1.0, 0.0, 100.0, 0)    # within one molecule: the atoms
    assert not isinstance(end, str), end
    assert pulled[0] == [(4, 0)]


def test_collisions_find_both_formic_acid_channels():
    pytest.importorskip("axtb")
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


# formic acid -> CO + H2O next to a spectator water (O5 H6 H7), Angstrom
_W = np.array([[6.0, 0, 0], [6.96, 0, 0], [5.76, 0.93, 0]])
_CO_H2O = np.array([[3.0, 0, 0], [-0.41, 0.06, -0.16], [-0.56, 1.17, 0.27], [3.96, 0, 0], [2.76, 0.93, 0]])
TS_SYM = SYM + ["O", "H", "H"]


def _st(xyz):
    from qcdata import Structure

    return Structure(symbols=TS_SYM, geometry=np.asarray(xyz) * 1.8897259886, charge=0, multiplicity=1)


def test_a_ts_counts_its_spectators_apart_and_its_reacting_complex_whole(monkeypatch):
    from mepd.discovery.free_energy import ts_groups

    start, end = np.vstack([XYZ, _W]), np.vstack([_CO_H2O, _W])
    # the TS where the water has left: a distance split would cut the reacting complex in two, its ends don't
    assert ts_groups(TS_SYM, end, [start, end]) == [[0, 1, 2, 3, 4], [5, 6, 7]]
    assert ts_groups(TS_SYM, start, [start, start]) is None                  # nothing reacts
    swapped = end[[5, 1, 2, 3, 4, 0, 6, 7]]          # an end in another atom order: the two waters swap O
    assert ts_groups(TS_SYM, end, [start, swapped]) is None

    fe = FreeEnergies(engine=None, temperature=1000.0)
    seen = []
    monkeypatch.setattr(fe, "_one", lambda s, x, q, m, ts: seen.append((tuple(s), ts)) or -40.0)
    monkeypatch.setattr("mepd.energies", lambda nodes, eng: [0.0] * len(nodes))
    ts = _st(end)
    assert fe._correction(ts, True, (_st(start), _st(end))) == (-80.0, -40.0)     # apart, and bound
    assert seen == [(tuple(SYM), True), (("O", "H", "H"), False), (tuple(TS_SYM), True)]   # mode: TS only
    seen.clear()
    assert fe._correction(ts, True) == -40.0 and seen == [(tuple(TS_SYM), True)]   # no ends: whole
    assert fe._key(ts, True, (_st(start), _st(end))) != fe._key(ts, True)
    seen.clear()
    assert fe._correction(ts, True, (_st(start), _st(swapped))) == -40.0 and len(seen) == 1
    assert any("don't match its bonds" in n for n in fe.notes)


def test_step_ends_come_from_the_irc_then_the_ts_job_then_the_species(tmp_path):
    import mepd
    from mepd.discovery.free_energy import step_ends
    from mepd.nodes.node import StructureNode

    a, b, mid = _st(np.vstack([XYZ, _W])), _st(np.vstack([_CO_H2O, _W])), _st(np.vstack([XYZ, _W]) + 0.01)
    ts = tmp_path / "s.xyz"
    mepd.write_xyz(ts, StructureNode(structure=mid))
    assert step_ends({"ts": str(ts)}, ("A", "B")) == ("A", "B")
    mepd.write_xyz(tmp_path / "s_ends.xyz", [StructureNode(structure=b), StructureNode(structure=a)])
    e0, e1 = step_ends({"ts": str(ts)}, ("A", "B"))
    assert np.allclose(e0.geometry, b.geometry, atol=1e-5) and np.allclose(e1.geometry, a.geometry, atol=1e-5)
    mepd.write_xyz(tmp_path / "irc.xyz", [StructureNode(structure=s) for s in (a, mid, b)])
    e0, e1 = step_ends({"ts": str(ts), "irc": str(tmp_path / "irc.xyz")}, ("A", "B"))
    assert np.allclose(e0.geometry, a.geometry, atol=1e-5) and np.allclose(e1.geometry, b.geometry, atol=1e-5)


def test_a_ts_with_spectators_takes_the_lower_of_apart_and_bound(monkeypatch):
    """Apart (spectators at the standard state) or bound (the whole complex: an H-bond, ion pair or dative
    contact that lowers the TS), whichever is lower -- not the rates added, which would count a spectator that
    does nothing for the TS twice (apart already holds the configurations with it next to the TS)."""
    from mepd.discovery.free_energy import both_channels

    T = 1000.0
    assert both_channels(-50.0, -50.0, T) == -50.0                              # no rate doubling
    assert both_channels(-50.0, -80.0, T) == pytest.approx(-80.0, abs=1e-5)     # a bound catalyst wins
    assert both_channels(-80.0, -50.0, T) == pytest.approx(-80.0, abs=1e-5)     # spectators apart win

    fe = FreeEnergies(engine=None, temperature=T)
    start, end = np.vstack([XYZ, _W]), np.vstack([_CO_H2O, _W])
    ends = (_st(start), _st(end))
    monkeypatch.setattr(fe, "_correction", lambda s, t, e=None: (-60.0, -70.0) if e is not None else -70.0)
    ts = _st(end)
    assert fe.corrections([ts], [True], [ends])[0] == pytest.approx(both_channels(-60.0, -70.0, T))
    assert fe.ts_channels(ts, ends) == (-60.0, -70.0)
    assert fe.corrections([ts], [True])[0] == -70.0 and fe.ts_channels(ts, None) is None   # no spectators


def test_single_points_move_each_channel_of_a_ts_on_its_own():
    from mepd.discovery.free_energy import both_channels, channels_shift

    T = 1500.0
    # the same shift for both: the combination moves by it
    assert channels_shift(-60.0, -62.0, 4.0, 4.0, T) == pytest.approx(4.0)
    # DFT binds the spectator 20 kcal/mol harder: the bound channel takes over
    d = channels_shift(-60.0, -55.0, 0.0, -20.0, T)
    assert -60.0 + 0.0 > both_channels(-60.0, -55.0, T) + d == pytest.approx(both_channels(-60.0, -75.0, T))
