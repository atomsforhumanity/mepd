"""The NEB tangent has no rigid translation or rotation, so g_perp and the spring force carry none either.

Neighbouring images differ by a little rigid rotation as well as by the reaction. Left in the tangent, the rigid
parts of g_perp and of the spring force cancelled only in their sum (whose rigid motion is projected out), and
the g_perp and TS-spring convergence tests never passed on a converged band (team42: 12,249 band gradients
instead of 466 on bs09)."""
from __future__ import annotations

import numpy as np
import pytest
from qcdata import Structure

import mepd.chainhelpers as ch
from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode

WATER = np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 1.11], [-1.43, 0.0, 1.11]])


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _image(stretch: float, theta: float) -> np.ndarray:
    g = WATER.copy()
    g[1] *= 1 + stretch
    g[2] *= 1 + stretch
    g = g - g.mean(axis=0)
    return g @ _rot(theta).T


def _rigid_parts(coords: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Net translation and torque about the centroid of a per-atom vector."""
    s = coords - coords.mean(axis=0)
    return float(np.linalg.norm(v.sum(axis=0))), float(np.linalg.norm(np.cross(s, v).sum(axis=0)))


def _chain(geoms, energies, grads) -> Chain:
    nodes = []
    for g, e, gr in zip(geoms, energies, grads):
        n = StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=g, charge=0, multiplicity=1))
        n._cached_energy, n._cached_gradient = e, gr
        nodes.append(n)
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def test_the_tangent_has_no_rigid_motion():
    x0, x1 = _image(0.0, 0.0), _image(0.05, 0.08)
    raw = x1 - x0
    assert _rigid_parts(x0, raw)[1] > 1e-3                         # the images rotate as well as stretch
    t = ch._tangent_without_rigid_motion(_chain([x0, x1, x1], [0, 1, 2], [np.zeros((3, 3))] * 3).nodes[0], raw)
    trans, torque = _rigid_parts(x0, t)
    assert trans == pytest.approx(0.0, abs=1e-12) and torque == pytest.approx(0.0, abs=1e-12)
    # The internal part (the stretch) is what is left.
    assert np.linalg.norm(t) > 0.5 * np.linalg.norm(_image(0.05, 0.0) - x0)


def test_g_perp_and_the_spring_force_carry_no_rigid_parts():
    geoms = [_image(0.0, 0.0), _image(0.04, 0.06), _image(0.08, 0.12)]
    energies = [-1.0, -0.95, -0.9]
    # A gradient with no rigid part (a symmetric stretch force, no net force or torque at its image).
    grads = []
    for g in geoms:
        s = g - g.mean(axis=0)
        f = s.copy()
        f[0] *= 0.0
        f = f - f.mean(axis=0)
        f = ch.project_rigid_body_forces(g, f)
        grads.append(0.01 * f)
    chain = _chain(geoms, energies, grads)
    gperp, spring = ch.pe_grads_spring_forces_nudged(chain=chain)
    mid = geoms[1]
    for v in (gperp[0], spring[0]):
        trans, torque = _rigid_parts(mid, np.asarray(v))
        assert trans < 1e-10 and torque < 1e-10


def test_a_band_without_rigid_motion_keeps_its_tangent():
    geoms = [_image(0.0, 0.0), _image(0.04, 0.0), _image(0.08, 0.0)]
    chain = _chain(geoms, [-1.0, -0.95, -0.9], [np.zeros((3, 3))] * 3)
    raw = geoms[2] - geoms[1]
    t = ch._tangent_without_rigid_motion(chain.nodes[1], raw)
    assert np.allclose(t, raw, atol=1e-12)
