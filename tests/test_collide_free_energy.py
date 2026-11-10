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
    assert fe._correction(ts, True, (_st(start), _st(end))) == -80.0     # apart only: no bound complex
    assert seen == [(tuple(SYM), True), (("O", "H", "H"), False)]        # mode: TS only
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


def test_a_ts_with_spectators_is_counted_from_its_own_energy_with_no_calculation_on_the_whole(monkeypatch):
    """A TS the explorer computed on its subsystem (network_expansion.subsystem): its correction is relative to
    the energy it was given (E_subsystem(TS) + env), from its reacting complex's and each spectator's own
    energies -- no single point, and no Hessian, of the whole box."""
    from mepd.discovery.free_energy import HARTREE_TO_KCAL

    fe = FreeEnergies(engine=None, temperature=1000.0)
    sizes = []
    monkeypatch.setattr(fe, "_one", lambda s, x, q, m, ts: sizes.append(len(s)) or -40.0)
    monkeypatch.setattr("mepd.energies", lambda nodes, eng: sizes.extend(len(n.symbols) for n in nodes)
                        or [-1.0] * len(nodes))
    start, end = np.vstack([XYZ, _W]), np.vstack([_CO_H2O, _W])
    corr = fe._correction(_st(end), True, (_st(start), _st(end)), energy=-2.5)
    assert corr == pytest.approx(-80.0 + (-2.0 + 2.5) * HARTREE_TO_KCAL)
    assert len(TS_SYM) not in sizes and sorted(sizes) == [3, 3, 5, 5]     # each piece's Hessian and energy only
    # an earlier version's correction (apart, bound) counts as the lower
    fe._cache[fe._key(_st(end), True, (_st(start), _st(end)))] = (-60.0, -70.0)
    assert fe.corrections([_st(end)], [True], [(_st(start), _st(end))]) == [-70.0]


# glycine -> CH2=NH + H2O + CO, the TS whose GFN2-xTB SCF does not converge (Angstrom): xtb --bhess fails there
GLY_TS_SYM = ["N", "C", "C", "O", "O", "H", "H", "H", "H", "H"]
GLY_TS = np.array([[-1.03090, 0.19451, 0.30488], [-1.37377, -0.84223, -0.32760], [2.09112, 1.14423, 0.05358],
                   [1.22001, 1.63990, 0.66316], [1.27103, -0.33598, -0.75534], [-0.01599, 0.48320, 0.20783],
                   [-1.66005, 0.74138, 0.87941], [-0.56078, -1.29406, -0.88937], [-2.38325, -1.22950, -0.30182],
                   [2.00267, -0.70065, -1.23095]])


def test_a_geometry_where_gfn2_fails_takes_its_shape_from_gfn_ff(monkeypatch):
    """The shape (linear?, symmetry number) the correction needs comes from xtb's analysis in a GFN2-xTB run; where
    GFN2's SCF fails (a TS), from the same analysis in a GFN-FF run, so the structure still has a correction (and the
    engine's own Hessian gives the vibrations either way)."""
    from types import SimpleNamespace

    import mepd.thermo as thermo

    real, calls = thermo.frequencies, []

    def frequencies(symbols, xyz, **kw):
        calls.append(list(kw.get("method_args", ())))
        if not kw.get("method_args"):
            raise RuntimeError("xtb --bhess failed (exit 128): scf: Self consistent charge iterator did not converge")
        return {"rotational_number": 1, "linear": False}

    monkeypatch.setattr(thermo, "frequencies", frequencies)
    w = [-300.0] + [100.0 * k for k in range(1, 3 * len(GLY_TS_SYM) - 6)]
    monkeypatch.setattr("mepd.frequencies", lambda node, engine: SimpleNamespace(wavenumbers_cm=np.array(w)))
    fe = FreeEnergies(engine=None, temperature=1000.0)
    g = fe._one(GLY_TS_SYM, GLY_TS, 0, 1, True)
    assert calls == [[], ["--gfnff"]] and np.isfinite(g)
    monkeypatch.setattr(thermo, "frequencies", real)
    import shutil

    if not shutil.which("xtb"):
        pytest.skip("no xtb")
    shape = real(GLY_TS_SYM, GLY_TS, method_args=["--gfnff"])     # GFN-FF runs where GFN2's SCF does not
    assert shape["rotational_number"] == 1 and not shape["linear"]
