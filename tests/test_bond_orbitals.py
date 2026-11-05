"""Bond orbitals (mepd.bond_orbitals, /api/bond-orbitals): intrinsic bond
orbitals of a geometry and followed along a path. Checks: the textbook
bonds and lone pairs for every program's files, the localization's optimum,
a turned frame's orbitals turned with it, and the SN2 path's two curly
arrows (the C-Cl bond becomes a chloride lone pair, the other chloride's
lone pair becomes the new C-Cl bond) with everything else unchanged."""
from pathlib import Path

import numpy as np
import pytest

from mepd import bond_orbitals as bo
from mepd import fragment_orbitals as fo
from mepd import orbitals

WATER = (["O", "H", "H"], np.array([[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]))
ETHYLENE = (["C", "C", "H", "H", "H", "H"],
            np.array([[0.667, 0, 0], [-0.667, 0, 0], [1.237, 0.923, 0], [1.237, -0.923, 0],
                      [-1.237, 0.923, 0], [-1.237, -0.923, 0]]))
CH3CL = (["C", "Cl", "H", "H", "H"],
         np.array([[0, 0, 0], [1.77, 0, 0], [-0.36, 1.02, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]]))
SN2 = ["C", "Cl", "Cl", "H", "H", "H"]


def _has(method):
    try:
        orbitals._executable(method)
        return True
    except Exception:
        return False


def _kinds(B) -> list:
    """Sorted (number of atoms, atoms) of every orbital."""
    return sorted((len(o["atoms"]), tuple(sorted(a for a, _ in o["atoms"]))) for o in B.orbitals)


def sn2_frame(t: float) -> np.ndarray:
    """Cl- + CH3Cl -> ClCH3 + Cl- by interpolation, t from -1 to 1: C-Cl1 1.8 -> 3.2 A, C-Cl2 3.2 -> 1.8 A,
    the methyl umbrella turning inside out."""
    d1, d2 = 1.8 + (t + 1) * 0.7, 3.2 - (t + 1) * 0.7
    hx = -0.36 * t
    r = 1.03 * np.sqrt(1 - (hx / 1.09) ** 2)
    return np.array([[0, 0, 0], [-d1, 0, 0], [d2, 0, 0],
                     [hx, r, 0], [hx, -r / 2, r * 0.866], [hx, -r / 2, -r * 0.866]])


@pytest.mark.parametrize("method", ["gfn2", "gxtb"])
def test_bonds_and_lone_pairs(method):
    if not _has(method):
        pytest.skip(f"{method} not available")
    syms, xyz = WATER
    B = bo.BondOrbitals(orbitals.run_molden(syms, xyz, 0, 1, method), method, 0, syms)
    assert not B.all_electron
    assert _kinds(B) == [(1, (0,)), (1, (0,)), (2, (0, 1)), (2, (0, 2))]   # two lone pairs, two O-H bonds
    assert abs(B.charges.sum()) < 1e-6 and B.charges[0] < -0.3 < 0 < B.charges[1]
    syms, xyz = ETHYLENE
    B = bo.BondOrbitals(orbitals.run_molden(syms, xyz, 0, 1, method), method, 0, syms)
    # sigma and pi between the carbons, four C-H bonds (g-xTB writes valence charges as atomic numbers:
    # the given elements are used).
    assert _kinds(B) == [(2, (0, 1)), (2, (0, 1)), (2, (0, 2)), (2, (0, 3)), (2, (1, 4)), (2, (1, 5))]
    # The IBOs are orthonormal and span the filled orbitals.
    S = fo.overlap(fo.basis(B.mol, method), fo.basis(B.mol, method))
    assert np.abs(B.C @ S @ B.C.T - np.eye(len(B.C))).max() < 1e-8


def test_terachem_all_electron_with_minao():
    """An all-electron file (TeraChem B3LYP/6-31G*): MINAO reference, core orbitals left out, charges add up."""
    text = (Path(__file__).parent / "data" / "terachem_ch3cl_b3lyp_631gs.molden").read_text()
    syms = CH3CL[0]
    B = bo.BondOrbitals(text, "terachem", 0, syms)
    assert B.all_electron
    # Cl: three lone pairs; C-Cl; three C-H. No core orbitals (C 1s, Cl 1s-2p).
    assert _kinds(B) == [(1, (1,)), (1, (1,)), (1, (1,)), (2, (0, 1)), (2, (0, 2)), (2, (0, 3)), (2, (0, 4))]
    assert abs(B.charges.sum()) < 1e-3


@pytest.mark.parametrize("name", ["gfn2_ch3cl", "gxtb2_ch3cl", "gxtb2_ceh4"])
def test_spherical_molden_files_give_the_same_bond_orbitals(name):
    """axtb's files with spherical d and f functions give the bond orbitals of the Cartesian file of the same
    calculation. CeH4 (g-xTB, Ce s, d and f valence shells): four Ce-H bonds, polarized to H, and
    charges that add up (Ce's 54 core electrons left out)."""
    data = Path(__file__).parent / "data"
    syms = CH3CL[0] if name.endswith("ch3cl") else ["Ce", "H", "H", "H", "H"]
    a, b = (bo.BondOrbitals((data / f"axtb_{name}_{kind}.molden").read_text(), "axtb", 0, syms)
            for kind in ("spherical", "cartesian"))
    assert np.abs(a.pop - b.pop).max() < 1e-8 and np.abs(a.charges - b.charges).max() < 1e-8
    assert np.abs(np.array([o["centre"] for o in a.orbitals]) - [o["centre"] for o in b.orbitals]).max() < 1e-6
    if name.endswith("ch3cl"):
        assert _kinds(a) == [(1, (1,)), (1, (1,)), (1, (1,)), (2, (0, 1)), (2, (0, 2)), (2, (0, 3)), (2, (0, 4))]
    else:
        assert _kinds(a) == [(2, (0, 1)), (2, (0, 2)), (2, (0, 3)), (2, (0, 4))]
        assert all(o["atoms"][0][0] != 0 and o["atoms"][0][1] > 0.6 for o in a.orbitals)   # polarized to H
        assert abs(a.charges.sum()) < 1e-3 and 1.0 < a.charges[0] < 2.5


def test_localization_reaches_the_optimum():
    """Every pair rotation is at the maximum of the functional (a scan finds nothing better), and the
    functional beats the canonical orbitals'."""
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    syms, xyz = CH3CL
    text = orbitals.run_molden(syms, xyz, 0, 1, "gfn2")
    B = bo.BondOrbitals(text, "gfn2", 0, syms)
    Q, atom_of = B.Q.T, B.ref_atom
    best = bo.functional(Q, atom_of)
    rng = np.random.default_rng(1)
    for _ in range(6):
        i, j = rng.choice(Q.shape[1], 2, replace=False)
        for phi in np.linspace(-np.pi / 4, np.pi / 4, 61):
            R = Q.copy()
            R[:, i], R[:, j] = np.cos(phi) * Q[:, i] + np.sin(phi) * Q[:, j], -np.sin(phi) * Q[:, i] + np.cos(phi) * Q[:, j]
            assert bo.functional(R, atom_of) <= best + 1e-9
    mol = orbitals.parse_molden(text)
    C = np.array([m["coef"] for m in sorted(mol["mos"], key=lambda m: m["energy"]) if m["occ"] > 1]).T
    b1 = fo.basis(mol, "gfn2")
    S1 = fo.overlap(b1, b1)
    S12, S2, ref_atom = bo._reference(mol, b1, S1, False)
    A = bo.iaos(S1, S12, S2, C)
    canonical = A.T @ S1 @ C
    assert np.abs(canonical.T @ canonical - np.eye(C.shape[1])).max() < 1e-8   # the IAOs hold the filled orbitals
    assert bo.functional(canonical, ref_atom) < best


def test_a_turned_frame_gets_its_orbitals_turned():
    """The same geometry computed turned and moved: its IBOs, turned back onto the shown frame, are the
    unturned ones (the same populations; the same functions, up to sign, for every orbital not mixed with an
    equivalent one)."""
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    from scipy.spatial.transform import Rotation

    x = sn2_frame(-0.3)
    R = Rotation.from_euler("xyz", [30, 50, 70], degrees=True).as_matrix()
    xr = (x - x.mean(axis=0)) @ R + 2.0
    B0 = bo.BondOrbitals(orbitals.run_molden(SN2, x, -1, 1, "gfn2"), "gfn2", -1, SN2)
    _, Rk = orbitals._aligned(x, xr)
    Bt = bo.BondOrbitals(orbitals.run_molden(SN2, xr, -1, 1, "gfn2"), "gfn2", -1, SN2, shown_A=x,
                         align={"R": Rk, "from": x.mean(axis=0), "to": xr.mean(axis=0)})
    assert np.allclose(np.sort(B0.pop, axis=0), np.sort(Bt.pop, axis=0), atol=1e-3)
    S = fo.overlap(fo.basis(B0.mol, "gfn2"), fo.basis(Bt.mol, "gfn2"))
    best = np.abs(B0.C @ S @ Bt.C.T).max(axis=1)
    assert (best > 1 - 1e-4).sum() >= len(best) - 2      # Cl's two lone pairs around the axis may mix


def test_sn2_path_has_two_curly_arrows():
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    frames = [bo.BondOrbitals(orbitals.run_molden(SN2, sn2_frame(t), -1, 1, "gfn2"), "gfn2", -1, SN2)
              for t in np.linspace(-1, 1, 15)]
    F = bo.follow(frames)
    assert len(F["active"]) == 2
    arrows = {(tuple(a for a, _ in F["orbitals"][t]["start"]), tuple(a for a, _ in F["orbitals"][t]["end"]))
              for t in F["active"]}
    # C-Cl1 bond -> Cl1 lone pair; Cl2 lone pair -> C-Cl2 bond (atoms listed by share, the larger first).
    assert arrows == {((1, 0), (1,)), ((2,), (2, 0))}
    for t in F["active"]:
        o = F["orbitals"][t]
        assert 0.3 < o["max_moved"] < 0.5 and o["moved"][0] == 0
        assert all(c is not None for c in o["centre"]) and o["matched"] > 0.9
    quiet = [o for o in F["orbitals"] if o["id"] not in F["active"]]
    assert len(quiet) == 9 and max(o["max_moved"] for o in quiet) < 0.02
    # The leaving chloride ends up with the charge: IAO charges -0.24 -> -0.99.
    q = np.array(F["charges"])
    assert q[0][1] > -0.4 and q[-1][1] < -0.9 and np.allclose(q.sum(axis=1), -1, atol=5e-3)   # (rounded per atom)
    # A frame that failed is skipped, the rest still followed.
    F2 = bo.follow(frames[:5] + [None] + frames[6:])
    assert F2["active"] == F["active"] and F2["orbitals"][F["active"][0]]["local"][5] is None


def _xyz_text(syms, xyz):
    rows = [f"{s} {x:.5f} {y:.5f} {z:.5f}" for s, (x, y, z) in zip(syms, xyz)]
    return f"{len(rows)}\n\n" + "\n".join(rows) + "\n"


def test_the_endpoints(tmp_path):
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    ws = tmp_path / "ws"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text('engine_name = "xtb"\n')
    with TestClient(create_app(ws, max_concurrent=1)) as c:
        r = c.post("/api/bond-orbitals", json={"xyz": _xyz_text(*WATER)})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["label"] == "GFN2-xTB" and len(out["orbitals"]) == 4 and out["drawn"]["cube"].startswith("mepd")
        first = out["drawn"]["index"]
        assert out["orbitals"][first]["energy_ev"] == max(o["energy_ev"] for o in out["orbitals"])  # the highest
        other = c.post("/api/bond-orbitals", json={"xyz": _xyz_text(*WATER), "orbital": (first + 1) % 4}).json()
        assert other["drawn"]["index"] == (first + 1) % 4 and other["drawn"]["cube"] != out["drawn"]["cube"]
        frames = [_xyz_text(SN2, sn2_frame(t)) for t in np.linspace(-1, 1, 9)]
        p = c.post("/api/bond-orbitals/path", json={"frames": frames, "charge": -1})
        assert p.status_code == 200, p.text
        F = p.json()
        assert len(F["active"]) == 2 and F["failed"] == [] and F["active_from"] == bo.ACTIVE
        kept = sorted((ws / "orbitals" / "bonds").glob("*.json.gz"))
        assert len(kept) == 1 + 9
        # A frame's orbital as followed: its number there, drawn on the coarser grid of a path.
        t = F["active"][0]
        k = F["orbitals"][t]["local"][4]
        d = c.post("/api/bond-orbitals", json={"xyz": frames[4], "charge": -1, "orbital": k, "follow": True}).json()
        assert d["drawn"]["index"] == k and d["source"] == "single point"
        # Following again reads the kept frames.
        from mepd.web import bond_store

        bond_store._MEM.clear()
        before = {f: f.stat().st_mtime_ns for f in kept}
        assert c.post("/api/bond-orbitals/path", json={"frames": frames, "charge": -1}).json()["active"] == F["active"]
        assert all(f.stat().st_mtime_ns == m for f, m in before.items())
        assert c.post("/api/bond-orbitals/path", json={"frames": frames[:2] + [_xyz_text(*WATER)]}).status_code == 400


def test_diels_alder_irc_three_arrows_followed_smoothly():
    """A g-xTB IRC of butadiene + ethylene (every 4th frame of 164; the TS is aromatic): the three pi bonds
    move into the new sigma bonds and the new pi bond, each followed with overlaps above IboView's 0.8 and
    without flipping between localizations near the TS (exponent 4 did: see EXPONENT)."""
    if not _has("gxtb"):
        pytest.skip("g-xTB not available")
    text = (Path(__file__).parent / "data" / "diels_alder_irc_gxtb.xyz").read_text().splitlines()
    n = int(text[0])
    geoms = [orbitals._parse_xyz("\n".join(text[k:k + n + 2])) for k in range(0, len(text), n + 2)]
    syms = geoms[0][0]
    F = bo.follow([bo.BondOrbitals(orbitals.run_molden(s, np.array(x), 0, 1, "gxtb"), "gxtb", 0, s) for s, x in geoms])
    assert len(F["active"]) == 3
    carbons = {i for i, s in enumerate(syms) if s == "C"}
    pairs = set()
    for t in F["active"]:
        o = F["orbitals"][t]
        start, end = {a for a, _ in o["start"]}, {a for a, _ in o["end"]}
        assert len(start) == len(end) == 2 and start | end <= carbons and start != end
        pairs.add(frozenset(end))
        moved = np.array(o["moved"])
        assert o["matched"] > 0.8 and abs(o["max_moved"] - moved[-1]) < 0.05      # no overshoot near the TS
        assert np.abs(np.diff(moved)).max() < 0.4
    assert len(pairs) == 3
    quiet = [o for o in F["orbitals"] if o["id"] not in F["active"]]
    assert max(o["max_moved"] for o in quiet) < 0.1


def test_compiled_integrals_and_localization_match_the_reference_code():
    """The compiled overlap and dipole integrals equal mepd.fragment_orbitals.overlap and dipoles for every
    program's conventions (and the MINAO cross overlaps run); the compiled localization reaches the same
    functional as the plain one."""
    text = (Path(__file__).parent / "data" / "terachem_ch3cl_b3lyp_631gs.molden").read_text()
    cases = [("terachem", text)]
    for m in ("gfn2", "gxtb"):
        if _has(m):
            cases.append((m, orbitals.run_molden(*CH3CL, 0, 1, m)))
    for method, t in cases:
        b = fo.basis(orbitals.parse_molden(t), method)
        S, D = bo.integrals(b, dip=True)
        assert np.abs(S - fo.overlap(b, b)).max() < 1e-12 and np.abs(D - bo.dipoles(b)).max() < 1e-12
        B = bo.BondOrbitals(t, method, 0, CH3CL[0])
        Q = B.Q.T
        U, _ = bo.localize(Q, B.ref_atom)
        plain = bo._localize_kernel
        bo._localize_kernel = None
        try:
            V, _ = bo.localize(Q, B.ref_atom)
        finally:
            bo._localize_kernel = plain
        assert abs(bo.functional(Q @ U, B.ref_atom) - bo.functional(Q @ V, B.ref_atom)) < 1e-8
