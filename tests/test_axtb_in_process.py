"""axtb in this process for g-xTB in implicit solvent (mepd#40), the GFN-FF environment of QM/MM
(mepd#41) and the analytic QM-region Hessian of QM/MM TS searches (mepd#42)."""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

axtb = pytest.importorskip("axtb")

from qcdata import Structure  # noqa: E402

from mepd.engines.gxtb import GXTBCalculator  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402

needs_xtb = pytest.mark.skipif(not (shutil.which("xtb") or os.getenv("XTB_EXECUTABLE")), reason="needs xtb")

from mepd.engines import axtb_backend  # noqa: E402

# Older axtb builds (the image's former pin) have neither; mepd then falls back to the xtb program.
pytestmark = pytest.mark.skipif(not (axtb_backend.supports("composite") and axtb_backend.supports("gfnff")),
                                reason="this axtb has no solvation composite or GFN-FF (axtb >= 0.1.0)")


def _node(structure) -> StructureNode:
    return StructureNode(structure=structure, has_molecular_graph=False)


def _acetate() -> Structure:
    xyz = """7

C   -1.4400   0.0300   0.0000
C    0.0700  -0.0200   0.0000
O    0.6700   1.0700   0.0000
O    0.6400  -1.1300   0.0000
H   -1.8100  -0.9900   0.0000
H   -1.8000   0.5600   0.8800
H   -1.8000   0.5600  -0.8800
"""
    return Structure.from_xyz(xyz).model_copy(update={"charge": -1})


def _central_differences(gradient, x0: np.ndarray, h: float = 1e-4) -> np.ndarray:
    flat = x0.reshape(-1)
    cols = []
    for k in range(flat.size):
        g = []
        for sign in (1.0, -1.0):
            y = flat.copy()
            y[k] += sign * h
            g.append(np.asarray(gradient(y.reshape(x0.shape)), dtype=float).reshape(-1))
        cols.append((g[0] - g[1]) / (2 * h))
    hess = np.array(cols)
    return 0.5 * (hess + hess.T)


# ------------------------------------------------------------------ mepd#40: g-xTB in solvent
def test_a_solvation_table_puts_axtb_gxtb_in_solvent_as_one_composite():
    from mepd.solvation import SolvatedEngine, solvate_engine, uses_correction

    eng = solvate_engine(GXTBCalculator(backend="axtb"), {"solvent": "dmso", "model": "gbsa"})
    assert isinstance(eng, GXTBCalculator) and eng.axtb_solvation == {"solvent": "dmso", "model": "gbsa"}
    assert uses_correction(eng)                     # still the GFN2-xTB composite, not a native model
    # The program route on request: the engine wrapped, the pair from xtb.
    old = solvate_engine(GXTBCalculator(backend="axtb"), {"solvent": "dmso", "backend": "xtb"})
    assert isinstance(old, SolvatedEngine) and old.backend == "xtb" and uses_correction(old)
    with pytest.raises(ValueError, match="backend"):
        GXTBCalculator(backend="xtb", axtb_solvation={"solvent": "water"})


def test_the_composite_is_gxtb_plus_the_gfn2_solvation_free_energy():
    s = _acetate()
    eng = GXTBCalculator(backend="axtb", axtb_solvation={"solvent": "water"})
    n = _node(s)
    eng.compute_gradients([n])
    gas = GXTBCalculator(backend="axtb")
    g = _node(s)
    gas.compute_gradients([g])
    numbers, x = [int(z) for z in s.atomic_numbers], np.asarray(s.geometry)
    pair = [axtb.singlepoint(numbers, x, charge=-1.0, method="gfn2", gradient=True, accuracy=0.01, **kw)
            for kw in ({"alpb": "water"}, {})]
    dg = pair[0]["energy"] - pair[1]["energy"]
    assert -0.15 < dg < -0.05                       # acetate in water: ~ -60 kcal/mol
    assert abs(n.energy - (g.energy + dg)) < 1e-7
    assert np.allclose(n.gradient, g.gradient + pair[0]["gradient"] - pair[1]["gradient"], atol=1e-6)


@needs_xtb
def test_the_composite_reproduces_the_three_program_runs():
    """With the programs' conventions and SCF accuracy, the in-process composite is the old route's
    numbers (g-xTB + two xtb GFN2 runs, SolvatedEngine) to round-off (Colton: <= 1e-10 Eh)."""
    from mepd.solvation import SolvatedEngine

    kw = dict(backend="axtb", axtb_conventions="gxtb", extra_args=["--acc", "1.0"], n_parallel=1)
    new = GXTBCalculator(**kw, axtb_solvation={"solvent": "hexane", "model": "alpb"})
    old = SolvatedEngine(GXTBCalculator(**kw), "hexane", backend="xtb")
    a, b = _node(_acetate()), _node(_acetate())
    ga, gb = new.compute_gradients([a])[0], old.compute_gradients([b])[0]
    assert abs(a.energy - b.energy) < 1e-9
    assert np.abs(ga - gb).max() < 1e-9


def test_the_composite_hessian_is_analytic_and_matches_its_gradient():
    eng = GXTBCalculator(backend="axtb", axtb_solvation={"solvent": "water"}, extra_args=["--acc", "0.001"])
    n = _node(_acetate())
    hess = eng.compute_hessian(n)
    fd = _central_differences(lambda x: eng.compute_gradients([n.update_coords(x)])[0], np.asarray(n.coords))
    assert np.abs(hess - fd).max() < 1e-4
    res = eng._compute_hessian_result(n)
    assert len(res.results.freqs_wavenumber) == 3 * 7 - 6


def test_the_correction_for_other_engines_runs_its_pair_in_axtb():
    from mepd.solvation import SolvationCorrection

    corr = SolvationCorrection("water")
    assert corr.backend == "axtb"
    dg, grad = corr.compute([_node(_acetate())])
    assert -0.15 < dg[0] < -0.05 and grad.shape == (1, 7, 3)
    # The ladder is the program route's, rung for rung.
    assert SolvationCorrection._axtb_ladder()[1] == {"max_iter": 1000, "etemp": 1000.0}
    assert len(SolvationCorrection._axtb_ladder()) == len(SolvationCorrection.SCC_FALLBACKS)
    with pytest.raises(ValueError, match="CPCM-X"):
        SolvationCorrection("water", "cpcmx", backend="axtb")


@needs_xtb
def test_the_axtb_pair_matches_the_xtb_pair():
    from mepd.solvation import SolvationCorrection

    nodes = [_node(_acetate())]
    new = SolvationCorrection("thf", "gbsa", backend="axtb", conventions="xtb-6.7.1").compute(nodes)
    old = SolvationCorrection("thf", "gbsa", backend="xtb").compute([_node(_acetate())])
    assert abs(new[0][0] - old[0][0]) < 1e-7          # the program pair runs at xtb's looser default SCF
    assert np.abs(new[1] - old[1]).max() < 5e-5


# ------------------------------------------------------------------ mepd#41: GFN-FF in process
def _water_cluster(n: int = 4) -> Structure:
    w = np.array([[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]])
    pos = [w + np.array([i, j, 0.0]) * 5.6 for i in range(n) for j in range(n)]
    return Structure(symbols=["O", "H", "H"] * n * n, geometry=np.vstack(pos))


def test_gfnff_runs_in_axtb_by_default_and_keeps_the_reference_topology(monkeypatch):
    from mepd.engines.gfnff import GFNFFEngine, default_backend

    assert default_backend() == "axtb" and default_backend("gfn2") == "xtb"
    monkeypatch.setenv("MEPD_GFNFF_BACKEND", "xtb")
    assert default_backend() == "xtb"
    monkeypatch.delenv("MEPD_GFNFF_BACKEND")
    s = _water_cluster()
    eng = GFNFFEngine(n_parallel=2, topology_reference=s.to_xyz())
    assert eng.backend == "axtb" and eng.executable is None        # no xtb program looked for
    x = np.asarray(s.geometry) + np.random.default_rng(0).normal(scale=0.05, size=np.shape(s.geometry))
    (e1, g1), (e2, _) = eng.energy_gradients([(list(s.symbols), x, 0, None, 0), (list(s.symbols), x + 0.02, 0, None, 0)])
    assert e1 != e2 and g1.shape == (48, 3)
    # The same geometry with the topology of another one is another number: the topology is kept.
    calc = axtb.Calculator("gfnff", [8, 1, 1] * 16, x, conventions="xtb-6.7.1")
    assert abs(calc.singlepoint(x)["energy"] - e1) < 1e-6
    with pytest.raises(ValueError, match="GFN-FF"):
        GFNFFEngine(method="gfn2", backend="axtb")


def test_a_topology_is_set_up_once_per_concurrent_caller(monkeypatch):
    """Each batch runs on a new thread pool; the topology's calculators outlive its threads."""
    from mepd.engines import axtb_backend
    from mepd.engines.gfnff import GFNFFEngine

    made = []
    real = axtb.Calculator
    monkeypatch.setattr(axtb, "Calculator", lambda *a, **k: made.append(a[0]) or real(*a, **k))
    monkeypatch.setattr(axtb_backend, "_GFNFF_POOL", {})
    s = _water_cluster(3)
    eng = GFNFFEngine(n_parallel=2, topology_reference=s.to_xyz())
    x = np.asarray(s.geometry)
    for batch in range(4):
        eng.energy_gradients([(list(s.symbols), x + 0.01 * (batch + k), 0, None, 0) for k in range(2)])
    assert 1 <= len(made) <= 2


@needs_xtb
def test_gfnff_in_axtb_matches_the_xtb_program(tmp_path, monkeypatch):
    from mepd.engines.gfnff import GFNFFEngine

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))     # gfnff_topo files of the program route
    s = _water_cluster(3)
    x = np.asarray(s.geometry) + np.random.default_rng(1).normal(scale=0.05, size=np.shape(s.geometry))
    out = []
    for backend in ("axtb", "xtb"):
        eng = GFNFFEngine(backend=backend, n_parallel=1, topology_reference=s.to_xyz())
        out.append(eng.energy_gradient(list(s.symbols), x, 0))
    # xtb reads Angstrom from a file and converts with CODATA 2018, mepd with 2022: ~1e-9 relative.
    assert abs(out[0][0] - out[1][0]) < 1e-7
    assert np.abs(out[0][1] - out[1][1]).max() < 1e-7


def test_gfnff_hessian_is_analytic():
    from mepd.engines.gfnff import GFNFFEngine

    s = _water_cluster(2)
    eng = GFNFFEngine(n_parallel=1, topology_reference=s.to_xyz())
    n = _node(s.model_copy(update={"geometry": np.asarray(s.geometry) + 0.03}))
    hess = eng.compute_hessian(n)
    fd = _central_differences(lambda x: eng.energy_gradient(list(s.symbols), x, 0)[1], np.asarray(n.coords))
    assert np.abs(hess - fd).max() < 1e-5


def test_qmmm_profile_key_picks_the_gfnff_backend():
    from mepd.qmmm import region_from_inputs

    region = _solvated_region()
    data = {**region.to_dict(), "gfnff_backend": "xtb"}
    assert region_from_inputs(data).natoms == region.natoms     # the key is the engine's, not the region's
    from mepd.engines.qmmm import QMMMEngine

    eng = QMMMEngine(base=GXTBCalculator(backend="axtb"), region=region)
    assert eng._low.engine.backend == "axtb"


@needs_xtb
def test_an_axtb_without_the_features_falls_back_to_the_programs(monkeypatch):
    from mepd.engines.gfnff import default_backend
    from mepd.solvation import SolvatedEngine, solvate_engine

    monkeypatch.setattr(axtb_backend, "_FEATURES", {"composite": False, "gfnff": False})
    assert default_backend() == "xtb"
    eng = solvate_engine(GXTBCalculator(backend="axtb"), {"solvent": "water"})
    assert isinstance(eng, SolvatedEngine)
    with pytest.raises(ValueError, match="composite"):
        GXTBCalculator(backend="axtb", axtb_solvation={"solvent": "water"})


# ------------------------------------------------------------------ mepd#42: QM-region Hessians
def _solvated_region():
    """Acetate in 8 waters, the QM region cut at its C-C bond (CO2- QM, CH3 MM: one link atom)."""
    from mepd.qmmm import QMMMRegion

    s = _acetate()
    xyz = np.asarray(s.geometry)
    waters = []
    w = np.array([[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]])
    for k, d in enumerate([(6, 0, 0), (-7, 0, 0), (0, 6.5, 0), (0, -6.5, 0), (0, 0, 6), (0, 0, -6), (5, 5, 0),
                           (-5, -5, 3)]):
        waters.append(w + np.array(d, dtype=float) + xyz.mean(axis=0))
    geom = np.vstack([xyz, *waters])
    system = Structure(symbols=list(s.symbols) + ["O", "H", "H"] * 8, geometry=geom, charge=-1)
    return QMMMRegion.build(system, [1, 2, 3], qm_charge=-1, active_radius=None)


def test_model_jacobian_maps_gradients_as_the_link_spreading_does():
    region = _solvated_region()
    assert region.links == [(1, 0)]
    g_model = np.random.default_rng(2).normal(size=(4, 3))          # 3 QM atoms + 1 link H
    jac = region.model_jacobian()
    assert np.allclose((jac.T @ g_model.reshape(-1)).reshape(-1, 3), region.model_gradient_to_full(g_model))
    sub = region.model_jacobian([1, 0])                             # columns in the order asked
    assert np.allclose(sub, jac[:, [3, 4, 5, 0, 1, 2]])


def test_qm_region_hessian_from_analytic_hessians_matches_differences_of_the_qmmm_gradient():
    from mepd.engines.qmmm import QMMMEngine

    region = _solvated_region()
    eng = QMMMEngine(base=GXTBCalculator(backend="axtb", extra_args=["--acc", "0.001"]), region=region,
                     restraints=[{"atoms": [2, 3], "distance": 2.2, "force_constant": 50.0}])
    assert eng.analytic_hessian_block
    system = Structure.from_xyz(region.reference).model_copy(update={"charge": -1})
    node = _node(system)
    atoms = [0, 1, 2, 3]                                             # QM atoms and the link's MM partner
    block = eng.hessian_block(node, atoms, step=5e-3)
    dof = [3 * a + c for a in atoms for c in range(3)]
    x0 = np.asarray(node.coords)

    def grad(sub):
        x = x0.copy().reshape(-1)
        x[dof] = sub.reshape(-1)
        return eng.compute_gradients([node.update_coords(x.reshape(-1, 3))])[0].reshape(-1)[dof]

    fd = _central_differences(grad, x0.reshape(-1)[dof], h=5e-3)
    assert np.abs(block - fd).max() < 1e-4


def test_frozen_atoms_and_microiterations_take_the_analytic_block(monkeypatch):
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.engines.microiter import MicroIterations
    from mepd.engines.qmmm import QMMMEngine

    region = _solvated_region()
    qmmm = QMMMEngine(base=GXTBCalculator(backend="axtb"), region=region)
    calls = []
    real = QMMMEngine.hessian_block
    monkeypatch.setattr(QMMMEngine, "hessian_block", lambda self, *a, **k: calls.append(a[1]) or real(self, *a, **k))
    node = _node(Structure.from_xyz(region.reference).model_copy(update={"charge": -1}))
    frozen = FrozenAtomsEngine(base=qmmm, frozen=[], hessian_atoms=list(region.qm_atoms))
    res = frozen._compute_hessian_result(node)
    assert calls == [[1, 2, 3]] and len(res.results.freqs_wavenumber) == 9
    micro = MicroIterations(qmmm)
    w, _ = micro.qm_modes(node.structure, np.asarray(node.coords))
    assert calls[-1] == micro.outer and len(w) == 3 * len(micro.outer)
    assert "analytic" in micro.qm_hessian_cost()
    # Electrostatic embedding (TIP3P/AMBER charges): the QM Hessian would need the point charges'
    # response, which axtb lacks; no block.
    eng = QMMMEngine(base=GXTBCalculator(backend="axtb"), region=region)
    eng._low = type("Low", (), {"electrostatic": True})()
    assert not eng.analytic_hessian_block and real(eng, node, [1, 2, 3]) is None
    # A QM engine without analytic Hessians: no block either.
    assert QMMMEngine(base=GXTBCalculator(backend="xtb"), region=region).hessian_block(node, [1, 2, 3]) is None
