"""Frontier orbitals (mepd.orbitals, /api/orbitals): xtb's molden orbitals
evaluated on a grid must come out normalized for both programs, whose
Cartesian d functions are scaled differently."""
import shutil

import numpy as np
import pytest

from mepd import orbitals

CH3CL = (["C", "Cl", "H", "H", "H"],
         [[0, 0, 0], [1.77, 0, 0], [-0.36, 1.02, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]])


def _has(method):
    try:
        orbitals._executable(method)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("method", ["gfn2", "gxtb"])
def test_every_orbital_comes_out_normalized(method):
    if not _has(method):
        pytest.skip(f"{method} not available")
    out = orbitals.frontier_orbitals(*CH3CL, 0, 1, method, extra=2)    # Cl: d functions
    assert set(out["orbitals"]) >= {"homo", "lumo", "homo-1", "lumo+1", "homo-2", "lumo+2"}
    for name, orb in out["orbitals"].items():
        assert orb["norm"] == pytest.approx(1.0, abs=0.01), name
        assert sum(orb["atoms"]) == pytest.approx(1.0, abs=1e-3)
        assert orb["cube"].count("\n") > 100
    assert out["gap_ev"] == pytest.approx(out["orbitals"]["lumo"]["energy_ev"] - out["orbitals"]["homo"]["energy_ev"], abs=1e-3)
    assert out["orbitals"]["homo"]["occupation"] > 0.5 and out["orbitals"]["lumo"]["occupation"] < 0.5


def test_where_the_orbitals_sit():
    """Formaldehyde: the HOMO is oxygen's lone pair, the LUMO the C=O pi*."""
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    out = orbitals.frontier_orbitals(["C", "O", "H", "H"], [[0, 0, 0], [1.21, 0, 0], [-0.55, 0.94, 0], [-0.55, -0.94, 0]],
                                     method="gfn2")
    homo, lumo = out["orbitals"]["homo"]["atoms"], out["orbitals"]["lumo"]["atoms"]
    assert np.argmax(homo) == 1 and homo[1] > 0.5
    assert lumo[0] + lumo[1] > 0.9 and lumo[0] > 0.3


def test_open_shell_takes_the_highest_partly_occupied_orbital():
    """xtb writes an open shell's orbitals spin-averaged: OH's degenerate pi pair holds 1.5 each (the SOMO)."""
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    out = orbitals.frontier_orbitals(["O", "H"], [[0, 0, 0], [0.97, 0, 0]], 0, 2, "gfn2")
    assert out["open_shell"] and out["orbitals"]["homo"]["occupation"] == pytest.approx(1.5)
    assert out["orbitals"]["lumo"]["occupation"] == 0


def test_the_endpoint(tmp_path):
    if not _has("gfn2"):
        pytest.skip("xtb not available")
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    ws = tmp_path / "ws"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text('engine_name = "xtb"\n')     # not g-xTB: GFN2-xTB orbitals
    xyz = "3\nwater\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
    with TestClient(create_app(ws, max_concurrent=1)) as c:
        r = c.post("/api/orbitals", json={"xyz": xyz})
        assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip"
        out = r.json()
        assert out["method"] == "gfn2" and {"homo", "lumo"} <= set(out["orbitals"]) and out["gap_ev"] > 0
        assert c.post("/api/orbitals", json={"xyz": "not a geometry"}).status_code == 400
