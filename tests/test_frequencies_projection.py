"""build_hessian_result_from_matrix projects translations and rotations out instead of dropping the six
smallest |eigenvalues| (the default analysis for every engine without its own: ORCA, Psi4, MLIPs, solvated,
restrained and forced wrappers)."""
from __future__ import annotations

import numpy as np
import pytest
from qcdata import Structure

from mepd.discovery.vri import rigid_body_basis
from mepd.engines.engine import _HESSIAN_EIGENVALUE_TO_CM2, build_hessian_result_from_matrix
from mepd.nodes.node import StructureNode

SYMBOLS = ["C", "H", "H", "H", "Cl"]
MASSES = np.array([12.011, 1.008, 1.008, 1.008, 35.45])


def _geometry() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [1.9, 0.9, 0.3], [-1.6, 1.2, -0.2], [-0.2, -2.0, 0.4], [0.3, 0.1, 3.4]])


def _eig(nu_cm: float) -> float:
    """Mass-weighted Hessian eigenvalue of a mode of `nu_cm` (negative: imaginary)."""
    return np.sign(nu_cm) * nu_cm ** 2 / _HESSIAN_EIGENVALUE_TO_CM2


def _hessian(vibrations_cm, rigid_noise_cm) -> np.ndarray:
    """A Cartesian Hessian with the given vibrations, plus noise along the rigid-body directions (as a
    finite-difference Hessian, or one away from a stationary point, has)."""
    geom = _geometry()
    basis = rigid_body_basis(geom, MASSES)                       # (15, 6), mass-weighted, orthonormal
    q, _ = np.linalg.qr(np.column_stack([basis, np.random.default_rng(1).normal(size=(15, 9))]))
    vib = q[:, 6:]                                               # 9 directions orthogonal to rigid-body motion
    k = vib @ np.diag([_eig(v) for v in vibrations_cm]) @ vib.T
    k += basis @ np.diag([_eig(v) for v in rigid_noise_cm]) @ basis.T
    sm = np.repeat(np.sqrt(MASSES), 3)
    return k * np.outer(sm, sm)


def _node() -> StructureNode:
    return StructureNode(structure=Structure(symbols=SYMBOLS, geometry=_geometry(), charge=0, multiplicity=1))


def test_a_soft_imaginary_mode_survives_rigid_body_noise():
    vibrations = [-15.0, 180.0, 420.0, 700.0, 1000.0, 1200.0, 1400.0, 3000.0, 3100.0]
    noise = [30.0, -25.0, 40.0, 35.0, -45.0, 50.0]               # larger than the soft mode
    out = build_hessian_result_from_matrix(_node(), _hessian(vibrations, noise))
    freqs = sorted(out.results.freqs_wavenumber)
    assert len(freqs) == 9
    assert sum(f < 0 for f in freqs) == 1                         # dropping the 6 smallest |lambda| lost it
    assert freqs == pytest.approx(sorted(vibrations), abs=1e-6)


def test_modes_are_cartesian_and_match_the_frequencies():
    vibrations = [-400.0, 180.0, 420.0, 700.0, 1000.0, 1200.0, 1400.0, 3000.0, 3100.0]
    out = build_hessian_result_from_matrix(_node(), _hessian(vibrations, [0.0] * 6))
    for mode in out.results.normal_modes_cartesian:
        mode = np.asarray(mode)
        assert mode.shape == (5, 3)
        assert np.linalg.norm(mode) == pytest.approx(1.0)
    # The imaginary mode is first and has no rigid-body component.
    imag = np.asarray(out.results.normal_modes_cartesian[0]).reshape(-1) * np.repeat(np.sqrt(MASSES), 3)
    basis = rigid_body_basis(_geometry(), MASSES)
    assert np.linalg.norm(basis.T @ (imag / np.linalg.norm(imag))) < 1e-8


def test_a_linear_molecule_keeps_its_bending_modes():
    geom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.0, -2.2]])
    masses = np.array([12.011, 15.999, 15.999])
    basis = rigid_body_basis(geom, masses)
    assert basis.shape[1] == 5
    q, _ = np.linalg.qr(np.column_stack([basis, np.random.default_rng(2).normal(size=(9, 4))]))
    vib = q[:, 5:]
    nus = [20.0, 20.5, 1300.0, 2400.0]                           # two very soft bends: kept
    sm = np.repeat(np.sqrt(masses), 3)
    hess = (vib @ np.diag([_eig(v) for v in nus]) @ vib.T) * np.outer(sm, sm)
    node = StructureNode(structure=Structure(symbols=["C", "O", "O"], geometry=geom, charge=0, multiplicity=1))
    freqs = sorted(build_hessian_result_from_matrix(node, hess).results.freqs_wavenumber)
    assert freqs == pytest.approx(nus, abs=1e-6)
