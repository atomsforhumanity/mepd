"""Fragment orbitals (mepd.fragment_orbitals, /api/fragment-orbitals): exact
overlaps between molecules' own orbitals. Checks: orbitals come out
orthonormal through the analytic overlap for every program's conventions,
and the Woodward-Hoffmann textbook case -- suprafacial [2+2] HOMO/LUMO
overlap vanishes by symmetry, Diels-Alder's does not."""
from pathlib import Path

import numpy as np
import pytest

from mepd import fragment_orbitals as fo
from mepd import orbitals

ETHYLENE = (["C", "C", "H", "H", "H", "H"],
            np.array([[0.667, 0, 0], [-0.667, 0, 0], [1.237, 0.923, 0], [1.237, -0.923, 0],
                      [-1.237, 0.923, 0], [-1.237, -0.923, 0]]))
# s-cis butadiene, terminal carbons 2.8 A apart along x, in the xy plane.
BUTADIENE = (["C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
             np.array([[-1.40, 0.0, 0], [-0.73, 1.17, 0], [0.73, 1.17, 0], [1.40, 0.0, 0],
                       [-2.48, 0.0, 0], [-0.88, -0.93, 0], [-1.28, 2.11, 0], [1.28, 2.11, 0],
                       [2.48, 0.0, 0], [0.88, -0.93, 0]]))


def _has(method):
    try:
        orbitals._executable(method)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("method", ["gfn2", "gxtb"])
def test_orbitals_are_orthonormal_through_the_overlap(method):
    if not _has(method):
        pytest.skip(f"{method} not available")
    syms = ["C", "Cl", "H", "H", "H"]   # Cl: d functions, scaled differently by the two programs
    xyz = np.array([[0, 0, 0], [1.77, 0, 0], [-0.36, 1.02, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]])
    side = fo._Side(orbitals.run_molden(syms, xyz, 0, 1, method), method)
    M = side.C @ fo.overlap(side.basis, side.basis) @ side.C.T
    assert np.abs(M - np.eye(len(M))).max() < 1e-6


def test_terachem_orbitals_are_orthonormal_through_the_overlap():
    """Normalized primitives and 6-31G*'s Cartesian d shells."""
    text = (Path(__file__).parent / "data" / "terachem_ch3cl_b3lyp_631gs.molden").read_text()
    side = fo._Side(text, "terachem")
    M = side.C @ fo.overlap(side.basis, side.basis) @ side.C.T
    assert np.abs(M - np.eye(len(M))).max() < 1e-4


def test_symmetry_allowed_and_forbidden():
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    e_syms, e_xyz = ETHYLENE
    # Two ethylenes face to face: [2+2] suprafacial, HOMO -> LUMO overlap zero by symmetry.
    r = fo.analyze(e_syms * 2, np.vstack([e_xyz, e_xyz + [0, 0, 2.5]]), 0, 1, "gfn2", cubes=False)
    assert len(r["fragments"]) == 2 and len(r["pairs"]) == 2
    assert all(t["s"] < 1e-6 for p in r["pairs"] for t in p["table"] if (t["occ"], t["virt"]) == ("homo", "lumo"))
    # Butadiene above ethylene (Diels-Alder): allowed, and bigger the closer they are.
    b_syms, b_xyz = BUTADIENE
    s = []
    for d in (3.0, 2.3):
        r = fo.analyze(b_syms + e_syms, np.vstack([b_xyz + [0, 0, d], e_xyz]), 0, 1, "gfn2", cubes=False)
        hl = {(p["donor"], p["acceptor"]): next(t["s"] for t in p["table"] if (t["occ"], t["virt"]) == ("homo", "lumo"))
              for p in r["pairs"]}
        s.append(hl[(0, 1)])
        assert hl[(0, 1)] > 0.02 and hl[(1, 0)] > 0.01
    assert s[1] > s[0]


def test_fragments_follow_the_reference_and_charges_sum():
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    # Hydroxide near methyl chloride: two molecules, charges -1 and 0.
    syms = ["C", "Cl", "H", "H", "H", "O", "H"]
    xyz = np.array([[0, 0, 0], [1.80, 0, 0], [-0.36, 1.02, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89],
                    [-2.6, 0, 0], [-3.55, 0, 0]])
    r = fo.analyze(syms, xyz, -1, 1, "gfn2", whole_molden=orbitals.run_molden(syms, xyz, -1, 1, "gfn2"))
    charges = {f["formula"]: f["charge"] for f in r["fragments"]}
    assert sum(charges.values()) == -1 and len(charges) == 2
    oh = [f["formula"] for f in r["fragments"]].index("HO")
    assert r["pairs"][0]["donor"] == oh                       # OH- gives...
    assert r["pairs"][0]["occ"] == "homo-2"                     # ...from its sigma lone pair: the pi pair (HOMO) is orthogonal
    lone = {lv["name"]: lv for lv in r["fragments"][oh]["levels"]}
    assert lone["homo"]["degeneracy"] == 2
    assert r["whole"]["homo"] and r["drawn"]["occ_cube"] and r["drawn"]["s"] == pytest.approx(r["pairs"][0]["s"], abs=1e-3)
    # Closer than bonding (a TS-like geometry): cut by the reference's bonding, still two molecules.
    close = xyz.copy()
    close[5:] += [0.9, 0, 0]
    r2 = fo.analyze(syms, close, -1, 1, "gfn2", reference_A=xyz, cubes=False)
    assert len(r2["fragments"]) == 2
    assert {k: r["cut"][k] for k in ("same", "here")} == {"same": True, "here": 2}
    assert r2["cut"]["same"] is False and r2["cut"]["here"] == 1      # bonded here: the bar warns
    with pytest.raises(orbitals.OrbitalError):
        fo.analyze(syms[:5], xyz[:5], 0, 1, "gfn2")      # one molecule


def test_the_endpoints(tmp_path):
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    ws = tmp_path / "ws"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text('engine_name = "xtb"\n')
    b_syms, b_xyz = BUTADIENE
    e_syms, e_xyz = ETHYLENE

    def xyz_text(d):
        rows = [f"{s} {x:.4f} {y:.4f} {z:.4f}" for s, (x, y, z) in zip(b_syms + e_syms, np.vstack([b_xyz + [0, 0, d], e_xyz]))]
        return f"{len(rows)}\n\n" + "\n".join(rows) + "\n"

    with TestClient(create_app(ws, max_concurrent=1)) as c:
        r = c.post("/api/fragment-orbitals", json={"xyz": xyz_text(2.5)})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["label"] == "GFN2-xTB" and len(out["pairs"]) == 2 and "whole" in out and "drawn" in out
        assert out["cut"]["from"] == "shown"
        other = c.post("/api/fragment-orbitals", json={"xyz": xyz_text(2.5), "pick": {
            "donor": 1, "acceptor": 0, "occ": "homo", "virt": "lumo"}}).json()["drawn"]
        assert (other["donor"], other["acceptor"], other["occ"]) == (1, 0, "homo")
        frags = [f["atoms"] for f in out["fragments"]]
        prof = c.post("/api/fragment-orbitals/profile", json={"frames": [xyz_text(d) for d in (3.5, 3.0, 2.5)],
                                                              "fragments": frags}).json()["pairs"]
        s = [next(t[2] for p in fr if p["donor"] == 0 for t in p["table"] if t[:2] == ["homo", "lumo"]) for fr in prof]
        assert s[0] < s[1] < s[2]
        # Kept on disk: every frame of the profile, read back without building again.
        kept = list((ws / "orbitals" / "fragments").glob("*.json.gz"))
        assert len(kept) >= 4
        from mepd.web import fragment_store

        before = {p: p.stat().st_mtime_ns for p in kept}
        prof2 = c.post("/api/fragment-orbitals/profile", json={"frames": [xyz_text(d) for d in (3.5, 3.0, 2.5)],
                                                               "fragments": frags}).json()["pairs"]
        assert prof2 == prof and all(p.stat().st_mtime_ns == t for p, t in before.items())
        assert fragment_store.WORKERS >= 1
        assert c.post("/api/fragment-orbitals", json={"xyz": "3\n\nO 0 0 0\nH 0 0.76 0.5\nH 0 -0.76 0.5\n"}).status_code == 400
        # A path stored product-first: its last frame cuts it (here a bonded-looking 1.5 A frame first).
        r = c.post("/api/fragment-orbitals", json={"xyz": xyz_text(1.5), "reference": [xyz_text(1.5), xyz_text(3.5)]})
        assert r.status_code == 200 and len(r.json()["fragments"]) == 2
        cut = r.json()["cut"]
        assert {k: cut[k] for k in ("same", "here", "from", "options")} == {"same": False, "here": 1, "from": "end",
                                                                             "options": ["end"]}   # bonded here
        start = c.post("/api/fragment-orbitals", json={"xyz": xyz_text(2.5), "reference": [xyz_text(3.5), xyz_text(1.5)]})
        cut = start.json()["cut"]
        assert {k: cut[k] for k in ("same", "here", "from", "options")} == {"same": True, "here": 2, "from": "start",
                                                                             "options": ["start"]}
        # Both ends two molecules: the start by default, the end when asked.
        both = [xyz_text(3.5), xyz_text(3.0)]
        assert c.post("/api/fragment-orbitals", json={"xyz": xyz_text(3.2), "reference": both}).json()["cut"]["from"] == "start"
        end = c.post("/api/fragment-orbitals", json={"xyz": xyz_text(3.2), "reference": both, "cut": "end"}).json()["cut"]
        assert end["from"] == "end" and end["options"] == ["start", "end"]


def test_two_radicals_pair_their_somos():
    """CH3* + H* (a C-H bond forming): the bond is SOMO + SOMO, offered first, and it is the big overlap."""
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    syms = ["C", "H", "H", "H", "H"]
    xyz = np.array([[0, 0, 0], [1.08, 0, 0], [-0.54, 0.935, 0], [-0.54, -0.935, 0], [0, 0, 1.6]])
    r = fo.analyze(syms, xyz, 0, 1, "gfn2", cubes=False)
    assert sorted(f["multiplicity"] for f in r["fragments"]) == [2, 2]
    first = r["pairs"][0]
    assert (first["occ"], first["virt"]) == ("somo", "somo") and first["s"] > 0.2
    others = [t["s"] for p in r["pairs"] for t in p["table"] if (t["occ"], t["virt"]) != ("somo", "somo")]
    assert first["s"] > max(others)
    h = next(f for f in r["fragments"] if f["formula"] == "H")
    assert [lv["name"] for lv in h["levels"]] [:2] == ["somo", "somo"]   # one level, on both sides


# Ammonia with two waters (a proton-relay-like arrangement), molecules 2-3 A apart.
NH3_2H2O = (["N", "H", "H", "H", "O", "H", "H", "O", "H", "H"],
            np.array([[0, 0, 0], [-0.38, 0.94, 0.0], [-0.38, -0.47, 0.81], [-0.38, -0.47, -0.81],
                      [2.9, 0, 0], [1.95, 0, 0], [3.2, 0.9, 0],
                      [4.6, 2.2, 0], [3.9, 1.6, 0], [5.4, 1.7, 0]]))


def test_molecules_grouped_into_fragments():
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    syms, xyz = NH3_2H2O
    alone = fo.analyze(syms, xyz, 0, 1, "gfn2", cubes=False)
    assert [m["formula"] for m in alone["cut"]["molecules"]] == ["H3N", "H2O", "H2O"] and len(alone["fragments"]) == 3
    grouped = fo.analyze(syms, xyz, 0, 1, "gfn2", groups=[[0], [1, 2]], cubes=False)
    f = grouped["fragments"]
    assert [x["formula"] for x in f] == ["H3N", "H2O + H2O"] and f[1]["members"] == [1, 2]
    assert f[1]["atoms"] == [4, 5, 6, 7, 8, 9] and f[1]["charge"] == 0 and f[1]["multiplicity"] == 1
    assert grouped["cut"]["groups"] == [[0], [1, 2]] and len(grouped["pairs"]) == 2
    for bad in ([[0], [1]], [[0, 1], [1, 2]], [[0, 1, 2]]):
        with pytest.raises(orbitals.OrbitalError):
            fo.analyze(syms, xyz, 0, 1, "gfn2", groups=bad, cubes=False)


def test_groups_through_the_endpoint(tmp_path):
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    ws = tmp_path / "ws"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text('engine_name = "xtb"\n')
    syms, xyz = NH3_2H2O
    text = f"{len(syms)}\n\n" + "\n".join(f"{s} {x:.4f} {y:.4f} {z:.4f}" for s, (x, y, z) in zip(syms, xyz)) + "\n"
    with TestClient(create_app(ws, max_concurrent=1)) as c:
        r = c.post("/api/fragment-orbitals", json={"xyz": text, "groups": [[0], [1, 2]]})
        assert r.status_code == 200, r.text
        assert [x["formula"] for x in r.json()["fragments"]] == ["H3N", "H2O + H2O"]
        assert c.post("/api/fragment-orbitals", json={"xyz": text, "groups": [[0, 1, 2]]}).status_code == 400
        assert c.post("/api/fragment-orbitals", json={"xyz": text, "groups": [["x"]]}).status_code == 400


def test_a_bare_cation_as_drawn_is_an_acceptor():
    """Mg2+ next to water: kept apart (given, as Design draws it), no filled orbitals, and the default
    donation is water's sigma lone pair into Mg's 3s (its LUMO), not its 3p or 3d sets."""
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    syms = ["O", "H", "H", "Mg"]
    xyz = np.array([[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0], [0, -2.05, 0]])
    assert len(fo.fragments_of(syms, xyz, 2)) == 1                       # by distance: one "molecule"
    r = fo.analyze(syms, xyz, 2, 1, "gfn2", fragments=[{"atoms": [0, 1, 2], "charge": 0}, {"atoms": [3], "charge": 2}],
                   cubes=False)
    mg = next(f for f in r["fragments"] if f["formula"] == "Mg")
    assert mg["charge"] == 2 and not any(lv["occupied"] for lv in mg["levels"])
    assert len(r["pairs"]) == 1 and r["fragments"][r["pairs"][0]["acceptor"]]["formula"] == "Mg"
    assert (r["pairs"][0]["occ"], r["pairs"][0]["virt"]) == ("homo-1", "lumo") and r["cut"]["groups"] is None
