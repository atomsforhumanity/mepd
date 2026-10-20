"""The orbital explorer's data (mepd.orbital_levels, /api/orbital-explorer):
levels per frame, the MO-diagram contributions, and orbitals followed along a
path by overlap, with their sign carried."""
import numpy as np
import pytest

from mepd import fragment_orbitals as fo
from mepd import orbital_levels as ol
from mepd import orbitals
from tests.test_fragment_orbitals import BUTADIENE, ETHYLENE, _has

pytestmark = pytest.mark.skipif(not _has("gfn2"), reason="xtb not available")


def _approach(ds):
    b, bx = BUTADIENE
    e, ex = ETHYLENE
    return b + e, [np.vstack([bx + [0, 0, d], ex]) for d in ds]


def test_a_path_is_cut_once_and_followed_through():
    syms, frames = _approach((4.0, 3.2, 2.6, 2.2))
    r = ol.analyze_path(syms, frames, 0, 1, "gfn2")
    assert [f["formula"] for f in r["fragments"]] == ["C4H6", "C2H4"]
    assert all(fr is not None for fr in r["frames"])
    # Butadiene's HOMO is one line from the first frame to the last.
    homo = r["frames"][0]["fragments"][0]["homo"]
    line = next(ln for ln in r["lines"]["fragments"][0] if ln["points"][0][:2] == [0, homo])
    assert [p[0] for p in line["points"]] == [0, 1, 2, 3]
    # Ethylene's LUMO mixes into the filled whole orbitals up close (the donation), not 4 A apart.
    e_lumo = r["frames"][0]["fragments"][1]["homo"] + 1

    def lumo_in_filled(fr):
        return sum(c["w"] for o in fr["whole"]["orbitals"] if o["rel"] <= 0
                   for c in o.get("from", []) if (c["fragment"], c["i"]) == (1, e_lumo))
    assert lumo_in_filled(r["frames"][0]) < 0.01 < lumo_in_filled(r["frames"][-1])
    # The fragments' overlaps grow as they approach.
    s = [max((abs(o["s"]) for o in fr["overlaps"] if o["a"] == [0, homo] and o["b"] == [1, e_lumo]), default=0)
         for fr in r["frames"]]
    assert s[0] < s[-1]


def test_a_flipped_scf_keeps_its_line_and_carries_the_sign():
    syms, frames = _approach((3.0, 2.9, 2.8))
    sides = [fo._Side(orbitals.run_molden(syms, f, 0, 1, "gfn2"), "gfn2") for f in frames]
    plain = {ln["points"][0][1]: ln["points"] for ln in ol.follow(sides) if len(ln["points"]) == 3}
    sides[1].C = -sides[1].C                       # the middle SCF came back as -psi everywhere
    flipped = {ln["points"][0][1]: ln["points"] for ln in ol.follow(sides) if len(ln["points"]) == 3}
    assert len(plain) >= 2 * ol.WINDOW - 1 and plain.keys() == flipped.keys()
    for i, pts in plain.items():
        assert [p[1] for p in flipped[i]] == [p[1] for p in pts]           # the same orbitals followed
        assert [p[2] for p in flipped[i]] == [pts[0][2], -pts[1][2], pts[2][2]]   # the flip carried


def test_one_molecule_has_only_the_whole_system():
    syms, frames = _approach((1.6,))
    r = ol.analyze_path(syms, frames, 0, 1, "gfn2")
    assert r["fragments"] == [] and r["frames"][0]["whole"]["orbitals"]


def test_the_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    ws = tmp_path / "ws"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text('engine_name = "xtb"\n')
    syms, frames = _approach((3.5, 2.8))
    texts = [f"{len(syms)}\n\n" + "\n".join(f"{s} {x:.4f} {y:.4f} {z:.4f}" for s, (x, y, z) in zip(syms, f)) + "\n"
             for f in frames]
    with TestClient(create_app(ws, max_concurrent=1)) as c:
        r = c.post("/api/orbital-explorer", json={"frames": texts})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["label"] == "GFN2-xTB" and len(out["frames"]) == 2 and len(out["fragments"]) == 2
        again = c.post("/api/orbital-explorer", json={"frames": texts}).json()
        assert again == out                                                  # kept
        homo = out["frames"][1]["fragments"][0]["homo"]
        cube = c.get(f"/api/orbital-explorer/{out['key']}/cube", params={"frame": 1, "system": "0", "i": homo, "coarse": 1})
        assert cube.status_code == 200 and cube.json()["cube"].count("\n") > 100
        assert c.get(f"/api/orbital-explorer/{out['key']}/cube", params={"frame": 1, "system": "../x", "i": 0}).status_code == 400
        single = c.post("/api/orbital-explorer", json={"xyz": texts[0]}).json()
        assert len(single["frames"]) == 1
