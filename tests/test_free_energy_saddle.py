"""A TS's thermal correction is taken at its saddle (free_energy.FreeEnergies._saddle): the same TS found by
several runs, each stopped at another point within the searches' force threshold, gets one correction."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import mepd.discovery.free_energy as fe
from mepd.discovery.free_energy import BOHR_PER_ANGSTROM, EV_PER_HARTREE, HESSIAN_FMAX, FreeEnergies

SYM = ["O", "C", "O", "H", "H"]
XYZ = np.array([[0.74, -0.62, -0.08], [-0.41, 0.06, -0.16], [-0.56, 1.17, 0.27], [1.40, -0.07, 0.37],
                [-1.17, -0.55, -0.67]])
EV_A = EV_PER_HARTREE * BOHR_PER_ANGSTROM


class Engine:
    """Forces of a given size (eV/Angstrom) on the first atom; nothing else."""
    analytic_hessian = True

    def __init__(self, force):
        self.force, self.asked = force, 0

    def compute_gradients(self, nodes):
        self.asked += 1
        g = np.zeros((len(nodes), len(SYM), 3))
        g[:, 0, 0] = self.force / EV_A
        return g


def _patched(monkeypatch, energies=(-1.0, -1.00002), fail=False, second=300.0):
    """mepd's optimizer, energies and frequencies replaced: the TS optimizer shifts the geometry by 1 bohr."""
    calls = {"find_ts": [], "hessian_at": []}

    def find_ts(node, engine, **kw):
        calls["find_ts"].append(kw)
        if fail:
            raise mepd.ConvergenceError("out of steps")
        out = node.copy()
        out.structure = out.structure.model_copy(update={"geometry": np.asarray(node.structure.geometry) + 1.0})
        return out

    def frequencies(node, engine):
        calls["hessian_at"].append(np.asarray(node.structure.geometry, dtype=float).reshape(-1, 3).copy())
        return SimpleNamespace(wavenumbers_cm=np.array([-1500.0, second, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0,
                                                        3000.0]))

    import mepd

    monkeypatch.setattr(mepd, "find_ts", find_ts)
    monkeypatch.setattr(mepd, "frequencies", frequencies)
    monkeypatch.setattr(mepd, "energies", lambda nodes, engine: list(energies)[:len(nodes)])
    monkeypatch.setattr("mepd.thermo.frequencies", lambda *a, **k: {"rotational_number": 1, "linear": False})
    return calls


def test_a_converged_ts_is_converged_further_before_its_hessian_and_its_energy_change_counts(monkeypatch):
    calls = _patched(monkeypatch)
    given = FreeEnergies(Engine(5e-5), 1500.0)._one(SYM, XYZ, 0, 1, True, True)      # already below HESSIAN_FMAX
    assert not calls["find_ts"]
    np.testing.assert_allclose(calls["hessian_at"][-1], XYZ * BOHR_PER_ANGSTROM)

    moved = FreeEnergies(Engine(8e-4), 1500.0)._one(SYM, XYZ, 0, 1, True, True)      # where a search stops (fmax 0.001)
    assert len(calls["find_ts"]) == 1 and calls["find_ts"][0]["fmax"] == HESSIAN_FMAX
    assert calls["find_ts"][0]["check_frequencies"] is False
    hessians = len(calls["hessian_at"])
    np.testing.assert_allclose(calls["hessian_at"][-1], XYZ * BOHR_PER_ANGSTROM + 1.0)   # the Hessian: at the saddle
    assert hessians == 2      # one for `given` above, one for this TS: the saddle's own, not taken again
    # the same frequencies and (shifted) geometry: the corrections differ by the optimization's energy change
    assert moved - given == pytest.approx(-0.00002 * fe.HARTREE_TO_KCAL, abs=1e-9)


@pytest.mark.parametrize("force, ts, analytic", [(0.2, True, True),      # no stationary point of the engine
                                                 (8e-4, False, True),    # a minimum
                                                 (8e-4, True, False)])   # another engine
def test_what_is_no_converged_ts_of_an_analytic_hessian_engine_keeps_its_geometry(monkeypatch, force, ts, analytic):
    calls = _patched(monkeypatch)
    engine = Engine(force)
    engine.analytic_hessian = analytic
    free = FreeEnergies(engine, 1500.0)
    free._one(SYM, XYZ, 0, 1, ts, True)
    assert not calls["find_ts"] and not free.notes
    assert engine.asked == (1 if ts and analytic else 0)
    np.testing.assert_allclose(calls["hessian_at"][-1], XYZ * BOHR_PER_ANGSTROM)


@pytest.mark.parametrize("energies, second, why", [((-1.0, -1.0005), 300.0, "(-0.31 kcal/mol)"),
                                                   ((-1.0, -1.00002), -40.0, "(2 negative modes)"),
                                                   ((-1.0, -0.994), 300.0, "(+3.77 kcal/mol)")])
def test_a_ts_that_converges_to_another_structure_keeps_its_geometry_and_says_so(monkeypatch, energies, second, why):
    """The correction is added to the energy the TS came with, and the Hessian must be a first-order saddle's:
    what the optimizer ends on counts only with that energy (0.05 kcal/mol) and exactly one negative mode. A
    recorded TS ended on another motion's saddle 4.7 kcal/mol lower; a minimum is walked up a rotor (+3.7)."""
    calls = _patched(monkeypatch, energies=energies, second=second)
    free = FreeEnergies(Engine(8e-4), 1500.0)
    free._one(SYM, XYZ, 0, 1, True, True)
    assert len(calls["find_ts"]) == 1
    np.testing.assert_allclose(calls["hessian_at"][-1], XYZ * BOHR_PER_ANGSTROM)   # the Hessian: where it was given
    assert len(free.notes) == 1 and "another structure " + why in free.notes[0]


def test_only_what_a_ts_optimization_produced_is_converged_further(monkeypatch):
    """_correction: the molecule a TS's file says it was optimized on. A file without the record (a path's highest
    image kept in place of a TS, an older run's TS) and a molecule other than the one optimized keep their
    geometry, whatever their forces."""
    from qcdata import Structure

    from mepd.discovery.free_energy import optimized_on_comment

    calls = _patched(monkeypatch)
    monkeypatch.setattr("mepd.energies", lambda nodes, engine: [-1.0, -1.00002][:len(nodes)])

    def st(comment=None):
        return Structure(symbols=SYM, geometry=XYZ * BOHR_PER_ANGSTROM, charge=0, multiplicity=1,
                         extras={"xyz_comments": comment.split()} if comment else {})

    engine = Engine(8e-4)
    free = FreeEnergies(engine, 1500.0)
    free._correction(st(), True)                                        # no record: a force passes, nothing runs
    free._correction(st("E=-1.0"), True)
    free._correction(st(f"E=-1.0 {optimized_on_comment()}"), False)      # not a TS
    assert not calls["find_ts"] and engine.asked == 0
    free._correction(st(f"E=-1.0 {optimized_on_comment()}"), True)
    assert len(calls["find_ts"]) == 1


def test_a_minimum_is_not_walked_up_to_a_saddle():
    """Ethanol minimized as the explorer minimizes a species passes the force gate (0.001 eV/Angstrom); the TS
    optimizer takes it 0.66 Angstrom up a rotor to +3.7 kcal/mol. Without the record of a TS optimization it is
    not touched; with one (were it ever written for such a structure) the result is refused."""
    pytest.importorskip("axtb")
    from qcdata import Structure
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from mepd.discovery.free_energy import optimized_on_comment
    from mepd.discovery.network_expansion import _optimize
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.nodes.node import StructureNode
    from mepd.thermo import _xtb

    try:
        _xtb()
    except Exception:
        pytest.skip("no xtb program (the rotational symmetry number)")
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    engine = GXTBCalculator(backend="axtb", axtb_conventions="gxtb", n_parallel=1)
    node = StructureNode(structure=Structure(symbols=[a.GetSymbol() for a in m.GetAtoms()],
                                             geometry=m.GetConformer().GetPositions() * BOHR_PER_ANGSTROM,
                                             charge=0, multiplicity=1))
    minimum = _optimize(engine, [node], 500, None)[0].structure
    free = FreeEnergies(engine, 298.15)
    plain = free._correction(minimum.model_copy(update={"extras": {}}), True)
    assert not free.notes
    said = minimum.model_copy(update={"extras": {"xyz_comments": [optimized_on_comment()]}})
    kept = free._correction(said, True)
    assert len(free.notes) == 1 and "another structure (+" in free.notes[0]      # the TS optimizer ran: refused
    assert kept == pytest.approx(plain, abs=1e-6)


def test_a_ts_the_optimizer_fails_on_keeps_its_geometry_and_says_so(monkeypatch):
    calls = _patched(monkeypatch, fail=True)
    free = FreeEnergies(Engine(8e-4), 1500.0)
    kept = free._one(SYM, XYZ, 0, 1, True, True)
    np.testing.assert_allclose(calls["hessian_at"][-1], XYZ * BOHR_PER_ANGSTROM)
    assert kept == FreeEnergies(Engine(5e-5), 1500.0)._one(SYM, XYZ, 0, 1, True, True)
    assert free.notes == ["a TS could not be converged further before its Hessian (ConvergenceError)"]


# One acetic acid -> ketene + water TS as two explorer runs wrote it (g-xTB through axtb, searches converged to
# 0.001 eV/Angstrom; energies -228.9742566535 and -228.9742564837 Hartree). Angstrom.
ACETIC_SYM = ["C", "C", "O", "O", "H", "H", "H", "H"]
ACETIC_A = np.array([[-0.93026199203631810, -0.08360469251256332, -0.17982411309721458],
                     [-0.00572004083140730, 0.94835360516144984, -0.00520885207367517],
                     [0.54306942027267824, 1.93494316516900278, 0.13653969558303253],
                     [1.35138683878031585, -0.71481897352478407, -0.22548031952816183],
                     [-1.47617968847702419, -0.27145102765709694, 0.73720899355185510],
                     [-1.59010494459852159, 0.11691958841662443, -1.01489757415258497],
                     [-0.00747735858480875, -0.92039048730013395, -0.33884882639877495],
                     [2.19794409150200742, -1.15079028558868535, -0.20549355923278861]])
ACETIC_B = np.array([[-0.79479057307676948, -0.24704932909309160, -0.46095986366940861],
                     [-0.10389221820943832, 0.95028327614171626, -0.26229344519616304],
                     [0.27500881334491056, 2.02270945225515764, -0.23656598052610189],
                     [1.14807342096360299, -0.38462736486586696, 0.88910194235130646],
                     [1.90281927620260971, -0.62703817960542663, 1.41707076479258065],
                     [-1.81185044982753163, -0.17698673752560612, -0.09494575021071894],
                     [-0.75360880513743733, -0.54994369249924779, -1.50043955051728140],
                     [0.01716305182618648, -0.87077781622714534, 0.26323776789422348]])


def test_one_ts_as_two_runs_stopped_at_it_gets_one_correction(monkeypatch):
    """The two geometries are 0.03 Angstrom apart along a mode of 28 cm^-1 that is 31 cm^-1 at one and 7i at the
    other: the corrections at the geometries given are 1.55 kcal/mol apart at 1500 K (the three runs' product
    ratios a factor 1.35), at the saddle 0.02."""
    pytest.importorskip("axtb")
    from qcdata import Structure

    from mepd.engines.gxtb import GXTBCalculator
    from mepd.thermo import _xtb

    try:
        _xtb()
    except Exception:
        pytest.skip("no xtb program (the rotational symmetry number)")
    engine = GXTBCalculator(backend="axtb", axtb_conventions="gxtb", n_parallel=1)
    a, b = (Structure(symbols=ACETIC_SYM, geometry=x * BOHR_PER_ANGSTROM, charge=0, multiplicity=1,
                      extras={"xyz_comments": ["optimized_on=whole"]})          # as a TS search writes them
            for x in (ACETIC_A, ACETIC_B))
    free = FreeEnergies(engine, 1500.0, "1atm")
    at_saddle = [free._correction(s, True) for s in (a, b)]
    assert abs(at_saddle[0] - at_saddle[1]) < 0.05 and not free.notes
    assert at_saddle[0] == pytest.approx(-101.49, abs=0.03)
    monkeypatch.setattr(FreeEnergies, "_saddle", lambda self, node: (node, 0.0))
    as_given = [free._correction(s, True) for s in (a, b)]
    assert as_given[0] == pytest.approx(-101.524, abs=0.01) and as_given[0] - as_given[1] > 1.5
    # a minimum is not touched: the same number with and without
    assert free._correction(a, False) == pytest.approx(FreeEnergies(engine, 1500.0, "1atm")._correction(a, False),
                                                       abs=1e-6)
