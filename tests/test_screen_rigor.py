"""Optimizations report what they reached: an unconverged run raises, a
minimization that stops on a saddle escapes it (or says it did not), a TS must
have exactly one imaginary mode, and a stereochemistry that cannot be compared
is not a match."""
from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.lj import LennardJones
from qcdata import Structure

import mepd
from mepd.engines.ase import ASEEngine
from mepd.errors import GeometryOptimizationNotConvergedError, WrongStationaryPointError
from mepd.nodes.node import StructureNode
from mepd.units import ANGSTROM_TO_BOHR


def _argon(xyz_angstrom) -> StructureNode:
    xyz = np.asarray(xyz_angstrom, dtype=float)
    return StructureNode(structure=Structure(symbols=["Ar"] * len(xyz), geometry=xyz * ANGSTROM_TO_BOHR,
                                             charge=0, multiplicity=1), has_molecular_graph=False)


# A deep, short Lennard-Jones well: soft argon (0.01 eV) gives saddle modes of a
# few cm^-1, below mepd's imaginary-mode noise cutoff.
SIGMA, EPSILON = 2.0, 1.0
R_MIN = SIGMA * 2 ** (1 / 6)


def _lj() -> ASEEngine:
    # plain LBFGS: ASE's line-search variant fails on this steep toy surface
    return ASEEngine(calculator=LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=8.0), geometry_optimizer="LBFGS")


def test_an_optimization_that_runs_out_of_steps_raises_instead_of_returning_its_last_frame():
    trimer = _argon([[0, 0, 0], [2.6, 0, 0], [1.2, 1.8, 0]])
    with pytest.raises(GeometryOptimizationNotConvergedError, match="did not converge in 1 steps"):
        mepd.optimize(trimer, _lj(), fmax=1e-4, steps=1)


class _FoldEngine:
    """A scripted surface: planar structures (all z = 0) are saddles with one
    out-of-plane imaginary mode; folding up (+z on atom 0) is downhill, folding
    down is less so. Minimization moves nothing (it is already 'converged')."""

    def compute_geometry_optimization(self, node, keywords=None):
        return [node.copy()]

    def compute_energies(self, nodes):
        out = []
        for n in nodes:
            z = float(np.asarray(n.coords)[0, 2])
            n._cached_energy = -abs(z) - (0.5 * z if z > 0 else 0.0)
            out.append(n._cached_energy)
        return out

    def _compute_hessian_result(self, node):
        from types import SimpleNamespace

        planar = abs(float(np.asarray(node.coords)[0, 2])) < 1e-9
        mode = np.zeros((len(node.coords), 3))
        mode[0, 2] = 1.0
        freqs = [-150.0, 80.0, 90.0] if planar else [40.0, 80.0, 90.0]
        return SimpleNamespace(results=SimpleNamespace(freqs_wavenumber=freqs,
                                                       normal_modes_cartesian=[mode, mode * 0, mode * 0]))


def test_minimize_escapes_a_saddle_along_its_imaginary_mode_and_keeps_the_lower_side():
    flat = _argon([[0, 0, 0], [R_MIN, 0, 0], [0, R_MIN, 0]])
    out = mepd.minimize(flat, _FoldEngine(), displacement_angstrom=0.1)
    assert out.saddle_escapes == 1 and out.frequencies.n_imaginary() == 0
    assert float(np.asarray(out.coords)[0, 2]) == pytest.approx(0.1 * ANGSTROM_TO_BOHR)   # the lower side


def test_minimize_says_so_when_it_cannot_leave_the_saddle():
    flat = _argon([[0, 0, 0], [R_MIN, 0, 0], [0, R_MIN, 0]])
    with pytest.raises(WrongStationaryPointError, match="saddle point"):
        mepd.minimize(flat, _FoldEngine(), escape_saddles=0)


def test_minimize_on_a_real_surface_reaches_a_minimum():
    eng = _lj()
    trimer = mepd.minimize(_argon([[0, 0, 0], [2.6, 0, 0], [1.2, 1.8, 0]]), eng, fmax=1e-4)
    assert trimer.frequencies.n_imaginary() == 0


def test_find_ts_rejects_a_converged_structure_without_exactly_one_imaginary_mode(monkeypatch):
    from types import SimpleNamespace

    node = _argon([[0, 0, 0], [R_MIN, 0, 0]])
    for freqs, ok in (([-400.0, 50.0], True), ([30.0, 50.0], False), ([-400.0, -60.0, 50.0], False),
                      ([-400.0, -5.0, 50.0], True)):     # -5 cm^-1 is below the noise cutoff
        engine = SimpleNamespace(
            compute_transition_state=lambda node: node,
            _compute_hessian_result=lambda n, f=freqs: SimpleNamespace(
                results=SimpleNamespace(freqs_wavenumber=f, normal_modes_cartesian=None)))
        if ok:
            ts = mepd.find_ts(node, engine)
            assert ts.frequencies.n_imaginary() == 1
        else:
            with pytest.raises(WrongStationaryPointError, match="imaginary modes"):
                mepd.find_ts(node, engine)
    assert mepd.find_ts(node, SimpleNamespace(compute_transition_state=lambda node: node),
                        check_frequencies=False) is node


HCN = "3\n\nC 0 0 0\nN 0 0 1.16\nH 0 0 -1.07\n"


def test_a_stereochemistry_that_cannot_be_compared_is_not_a_match(tmp_path, monkeypatch):
    (tmp_path / "hcn.xyz").write_text(HCN)
    a = mepd.load(tmp_path / "hcn.xyz", charge=0, multiplicity=1)
    b = mepd.load(tmp_path / "hcn.xyz", charge=0, multiplicity=1)
    assert mepd.compare_structures(a, b).same and mepd.same_molecule(a, b)

    import mepd.nodes.nodehelpers as helpers

    def broken(node, bond_orders=True):
        raise RuntimeError("no SMILES today")

    monkeypatch.setattr(helpers, "_stereo_smiles", broken)
    # OpenBabel, the second opinion, can confirm "same" ...
    c = mepd.compare_structures(a, b)
    assert c.same and c.method == "openbabel" and "stereo SMILES failed" in c.reason
    # ... but its "different" is not decisive, and neither is its failure
    import mepd.compare as cmp

    names = iter(["C#N", "[C-]#[NH+]"] * 10)
    monkeypatch.setattr(cmp, "_openbabel_stereo_smiles", lambda node: next(names))
    c = mepd.compare_structures(a, b)
    assert c.stereo is None and "not decisive" in c.reason
    monkeypatch.setattr(cmp, "_openbabel_stereo_smiles", broken)
    c = mepd.compare_structures(a, b)
    assert c.graph and c.stereo is None and c.undetermined and "OpenBabel failed" in c.reason
    assert not mepd.same_molecule(a, b)                      # "can't tell" never passes
    assert mepd.same_molecule(a, b, stereo=False)            # the graphs alone still match
    hit = mepd.connects([a, b], a, b)
    assert not hit.start and not hit.end and hit.undetermined == (True, True)


def test_missing_bond_orders_leave_e_z_undecided(tmp_path, monkeypatch):
    (tmp_path / "hcn.xyz").write_text(HCN)
    a = mepd.load(tmp_path / "hcn.xyz", charge=0, multiplicity=1)
    import mepd.nodes.nodehelpers as helpers

    real = helpers._stereo_smiles
    monkeypatch.setattr(helpers, "_stereo_smiles", lambda node, bond_orders=True: (real(node, False)[0], False))
    import mepd.compare as cmp

    monkeypatch.setattr(cmp, "_openbabel_stereo_smiles", lambda node: (_ for _ in ()).throw(RuntimeError("off")))
    c = mepd.compare_structures(a, a)
    assert c.stereo is None and "bond orders could not be assigned" in c.reason


def test_geometric_minimizer_reaches_the_lennard_jones_minimum():
    from mepd.irc_geometric import minimize_with_geometric

    dimer = _argon([[0, 0, 0], [2.5, 0, 0]])
    last = minimize_with_geometric(_lj(), dimer, fmax=1e-4)[-1]
    r = float(np.linalg.norm(np.subtract(*np.asarray(last.coords))) / ANGSTROM_TO_BOHR)
    assert r == pytest.approx(R_MIN, abs=2e-3)


def test_find_ts_continues_an_unconverged_optimization_from_where_it_stopped():
    from types import SimpleNamespace

    start = _argon([[0, 0, 0], [R_MIN, 0, 0]])
    halfway = _argon([[0, 0, 0], [R_MIN + 0.1, 0, 0]])
    calls = []

    def ts_opt(node):
        calls.append(np.asarray(node.coords).copy())
        if len(calls) == 1:
            err = GeometryOptimizationNotConvergedError(msg="out of steps")
            err.structure = halfway
            raise err
        return node

    one = SimpleNamespace(results=SimpleNamespace(freqs_wavenumber=[-400.0, 50.0], normal_modes_cartesian=None))
    engine = SimpleNamespace(compute_transition_state=ts_opt, _compute_hessian_result=lambda n: one)
    ts = mepd.find_ts(start, engine)
    assert len(calls) == 2 and np.allclose(calls[1], np.asarray(halfway.coords))   # resumed, not restarted
    assert np.allclose(np.asarray(ts.coords), np.asarray(halfway.coords))
    calls.clear()
    with pytest.raises(GeometryOptimizationNotConvergedError):
        mepd.find_ts(start, engine, continue_unconverged=0)


def test_irc_ends_are_minimized_and_the_path_runs_minimum_to_minimum():
    from mepd.chain import Chain

    eng = _lj()
    a = _argon([[0, 0, 0], [2.6, 0, 0], [1.2, 1.8, 0]])
    b = _argon([[0, 0, 0], [2.5, 0, 0], [1.3, 1.9, 0]])
    mid = _argon([[0, 0, 0], [2.55, 0, 0], [1.25, 1.85, 0]])
    eng.compute_irc_chain = lambda ts, keywords=None: Chain.model_validate({"nodes": [a, mid, b]})
    res = mepd.irc(mid, eng)
    assert len(res.path.nodes) == 5 and len(res.irc_path.nodes) == 3
    for end in res.ends:
        assert end.frequencies.n_imaginary() == 0
    raw = mepd.irc(mid, eng, minimize_ends=False)
    assert len(raw.path.nodes) == 3


def test_find_ts_defaults_to_mepd_ts_convergence():
    """mepd.find_ts without fmax/maxiter uses `mepd ts`'s 0.001 eV/Å and 1000 steps,
    not an engine's looser optimizer default (mepd#6); a given value wins, and an
    engine whose method takes no keywords is called without them."""
    import mepd
    from mepd.inputs import TSInputs
    from mepd.nodes.node import StructureNode
    from qcdata import Structure

    node = StructureNode(structure=Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 1.4]]))
    seen = []

    class Takes:
        def compute_transition_state(self, node, keywords=None):
            seen.append(keywords)
            return node

    class NoKeywords:
        def compute_transition_state(self, node):
            seen.append("called")
            return node

    mepd.find_ts(node, Takes(), check_frequencies=False)
    mepd.find_ts(node, Takes(), check_frequencies=False, fmax=0.05)
    mepd.find_ts(node, NoKeywords(), check_frequencies=False)
    assert seen[0] == {"fmax": TSInputs.fmax, "maxiter": TSInputs.maxiter} == {"fmax": 0.001, "maxiter": 1000}
    assert seen[1] == {"fmax": 0.05, "maxiter": 1000}
    assert seen[2] == "called"
