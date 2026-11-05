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


def _terachem_molden():
    from pathlib import Path

    return (Path(__file__).parent / "data" / "terachem_ch3cl_b3lyp_631gs.molden").read_text()


def test_terachem_orbitals_are_read_with_its_conventions():
    """TeraChem's molden (B3LYP/6-31G*, Cl d functions): no Sym= lines, normalized primitives."""
    out = orbitals.orbitals_from_molden(_terachem_molden(), "terachem", extra=2)
    assert out["orbitals"]["homo"]["energy_ev"] == pytest.approx(-8.06, abs=0.01)
    for name, orb in out["orbitals"].items():   # (6-31G*'s tight functions on the default grid: within 2 %)
        assert orb["norm"] == pytest.approx(1.0, abs=0.02), name


def test_a_terachem_session_gets_terachem_orbitals(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    seen = {}

    def fake_run(symbols, coords, charge, multiplicity, engine_kwds, timeout_s=600.0):
        seen.update(engine_kwds)
        return _terachem_molden()

    monkeypatch.setattr(orbitals, "run_terachem_molden", fake_run)
    orbitals._cached.cache_clear()
    ws = tmp_path / "ws"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text(
        'engine_name = "terachem"\n[terachem_engine_kwds]\nmethod = "b3lyp"\nbasis = "6-31gs"\n')
    xyz = "5\nch3cl\nC 0 0 0\nCl 1.77 0 0\nH -0.36 1.02 0\nH -0.36 -0.51 0.89\nH -0.36 -0.51 -0.89\n"
    with TestClient(create_app(ws, max_concurrent=1)) as c:
        out = c.post("/api/orbitals", json={"xyz": xyz}).json()
    assert out["method"] == "terachem" and out["label"] == "TeraChem b3lyp/6-31gs"
    assert seen == {"method": "b3lyp", "basis": "6-31gs"}


def test_terachem_runs_when_installed():
    if not shutil.which("terachem"):
        pytest.skip("TeraChem not installed")
    try:
        out = orbitals.frontier_orbitals(*CH3CL, 0, 1, "terachem", engine_kwds={"method": "hf", "basis": "sto-3g"})
    except orbitals.OrbitalError as exc:
        pytest.skip(f"TeraChem cannot run here: {exc}")
    assert out["label"] == "TeraChem hf/sto-3g" and out["orbitals"]["homo"]["norm"] == pytest.approx(1.0, abs=0.01)


def test_a_calculation_keeps_its_orbitals_and_the_viewer_reads_them(tmp_path, monkeypatch):
    """With MEPD_ORBITALS_DIR set, every g-xTB SCF keeps its molden file; asking for that
    geometry then reads it (no new calculation), and only the newest KEEP_RUNS are kept."""
    if not _has("gxtb"):
        pytest.skip("g-xTB not available")
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.nodes.node import StructureNode
    from mepd.web import chem

    monkeypatch.setenv("MEPD_ORBITALS_DIR", str(tmp_path / "orbitals"))
    monkeypatch.setattr(orbitals, "KEEP_RUNS", 3)
    monkeypatch.setattr(orbitals, "PRUNE_EVERY", 1)   # prune on every save (the default lets it overshoot a little)
    s = chem.structure_from_smiles("C=O")
    eng = GXTBCalculator()
    for k in range(4):
        eng.compute_gradients([StructureNode(structure=s.model_copy(update={"geometry": s.geometry + 0.02 * k}))])
    kept = sorted((tmp_path / "orbitals").glob("*.molden"))
    assert len(kept) == 3
    monkeypatch.setattr(orbitals, "run_molden", lambda *a, **k: pytest.fail("ran a new single point"))
    last = (s.geometry + 0.06) * 0.529177210903
    out = orbitals.frontier_orbitals(list(s.symbols), last, 0, 1, "gxtb", run_dirs=[tmp_path / "orbitals"])
    assert out["source"] == "run" and out["label"] == "g-xTB" and out["orbitals"]["homo"]["norm"] == pytest.approx(1.0, abs=0.01)
    # The first was dropped (the others are it shifted, a rigid match: so ask for an exact one).
    assert orbitals.find_run_orbitals([tmp_path / "orbitals"], list(s.symbols), s.geometry * 0.529177210903,
                                      rigid=False) is None


def test_no_orbitals_dir_means_nothing_is_kept(tmp_path, monkeypatch):
    if not _has("gxtb"):
        pytest.skip("g-xTB not available")
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.nodes.node import StructureNode
    from mepd.web import chem

    monkeypatch.delenv("MEPD_ORBITALS_DIR", raising=False)
    eng = GXTBCalculator()
    assert eng._orbitals_kind(["gxtb", "x.xyz"]) is None
    eng.compute_gradients([StructureNode(structure=chem.structure_from_smiles("O"))])


def test_the_keeper_gives_every_structure_its_orbitals_ready(tmp_path, monkeypatch):
    """The session's store: every structure gets orbitals built ahead, and /api/orbitals then sends
    them as stored -- no calculation when they are shown."""
    if not _has("gxtb"):
        pytest.skip("g-xTB not available")
    from fastapi.testclient import TestClient

    from mepd.web import orbital_store
    from mepd.web.app import create_app
    from mepd.web.workspace import Workspace

    ws_dir = tmp_path / "ws"
    (ws_dir / "profiles").mkdir(parents=True)
    (ws_dir / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    ws = Workspace(ws_dir)
    from mepd.web import chem
    s = chem.structure_from_smiles("C=CC=O")
    rec = ws.add_or_merge(s, name="acrolein", smiles="C=CC=O", optimized=False, origin={"kind": "smiles"})["rec"]
    monkeypatch.setattr(orbital_store, "SETTLE_S", 0.0)
    keeper = orbital_store.OrbitalKeeper(ws)
    assert keeper.sweep() == 1 and keeper.sweep() == 0                     # once, then it is there
    stored = list((ws_dir / "orbitals").glob("*.json.gz"))
    assert len(stored) == 1
    monkeypatch.setattr(orbitals, "run_molden", lambda *a, **k: pytest.fail("ran a new single point"))
    xyz = ws.structure_path(rec["id"]).read_text()
    with TestClient(create_app(ws_dir, max_concurrent=1)) as c:
        out = c.post("/api/orbitals", json={"xyz": xyz}).json()
    assert out["label"] == "g-xTB" and out["source"] == "single point" and out["orbitals"]["homo"]["norm"] == pytest.approx(1, abs=0.01)


def test_a_cli_run_keeps_its_orbitals_under_its_output(tmp_path, monkeypatch):
    import os

    from mepd.cli import _keep_orbitals

    monkeypatch.delenv(orbitals.ORBITALS_ENV, raising=False)
    assert _keep_orbitals(["ts", "--output", str(tmp_path / "out")])
    assert os.environ[orbitals.ORBITALS_ENV] == str(tmp_path / "out" / "orbitals")
    assert not _keep_orbitals(["ts", "--output", str(tmp_path / "other")])   # the web job's choice stands


def test_a_turned_frame_gets_its_kept_orbitals_turned_with_it(tmp_path, monkeypatch):
    """IRC frames have their rigid motion taken out: the same geometry, turned and moved. It still
    matches the kept SCF, and the orbitals come out as for that frame itself (same energies, same
    share on each atom)."""
    if not _has("gxtb"):
        pytest.skip("g-xTB not available")
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.nodes.node import StructureNode
    from mepd.web import chem

    monkeypatch.setenv("MEPD_ORBITALS_DIR", str(tmp_path / "orbitals"))
    s = chem.structure_from_smiles("C=CC=O")
    GXTBCalculator().compute_gradients([StructureNode(structure=s)])
    xyz = np.asarray(s.geometry) * 0.529177210903
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    turned = xyz @ R.T + [1.5, -2.0, 0.3]
    hit = orbitals.find_run_orbitals([tmp_path / "orbitals"], list(s.symbols), turned, "gxtb")
    assert hit is not None and hit["align"] is not None
    assert orbitals.find_run_orbitals([tmp_path / "orbitals"], list(s.symbols), turned, "gxtb", rigid=False) is None
    run = orbitals.frontier_orbitals(list(s.symbols), turned, 0, 1, "gxtb", run_dirs=[tmp_path / "orbitals"])
    fresh = orbitals.frontier_orbitals(list(s.symbols), turned, 0, 1, "gxtb")
    assert run["source"] == "run" and fresh["source"] == "single point"
    for k in ("homo", "lumo"):
        assert run["orbitals"][k]["energy_ev"] == pytest.approx(fresh["orbitals"][k]["energy_ev"], abs=1e-3)
        assert run["orbitals"][k]["atoms"] == pytest.approx(fresh["orbitals"][k]["atoms"], abs=0.02)


def test_a_jobs_frames_are_built_once_and_kept(tmp_path, monkeypatch):
    if not _has("gxtb"):
        pytest.skip("g-xTB not available")
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.nodes.node import StructureNode
    from mepd.web import chem

    monkeypatch.setenv("MEPD_ORBITALS_DIR", str(tmp_path / "orbitals"))
    s = chem.structure_from_smiles("C=O")
    GXTBCalculator().compute_gradients([StructureNode(structure=s)])
    xyz = np.asarray(s.geometry) * 0.529177210903
    first = orbitals.frame_build(tmp_path / "orbitals", list(s.symbols), xyz)
    assert first is not None and len(list((tmp_path / "orbitals" / "built").glob("*.json.gz"))) == 1
    monkeypatch.setattr(orbitals, "orbitals_from_molden", lambda *a, **k: pytest.fail("built again"))
    orbitals._from_file.cache_clear()
    assert orbitals.frame_build(tmp_path / "orbitals", list(s.symbols), xyz) == first      # a file read
    assert orbitals.frame_build(tmp_path / "orbitals", list(s.symbols), xyz + [[0, 0, 1.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]) is None


# axtb (GFN2-xTB, g-xTB) writes Molden files with spherical d and f functions by default and with
# Cartesian ones on request. tests/data has both files of three calculations (axtb 0.1.0,
# `axtb.singlepoint(..., molden=True)` and `molden="cartesian"`): CH3Cl with GFN2-xTB and with
# g-xTB 2.0.1 (Cl d functions) and CeH4 with g-xTB 2.0.1 (Ce d and f functions).
AXTB_PAIRS = ["gfn2_ch3cl", "gxtb2_ch3cl", "gxtb2_ceh4"]


def _axtb_molden(name, kind):
    from pathlib import Path

    return (Path(__file__).parent / "data" / f"axtb_{name}_{kind}.molden").read_text()


def _values(text, d_norm="own", normalized_prims=True, points=None):
    """Every orbital of a molden file on a grid around it (or at `points`), and the points."""
    mol = orbitals.parse_molden(text, d_norm)
    if points is None:
        xyz = np.array([p for _, p in mol["atoms"]])
        origin, h, n = orbitals.grid_for(xyz, max_points=32 ** 3)
        axes = [origin[i] + h * np.arange(n[i]) for i in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    coefs = [m["coef"] for m in mol["mos"]]
    return orbitals.evaluate(mol, coefs, points, d_norm, normalized_prims).astype(float), points


@pytest.mark.parametrize("name", AXTB_PAIRS)
def test_spherical_and_cartesian_molden_files_give_the_same_orbitals(name):
    """Every orbital of a spherical file ([5D]/[7F]) equals the same orbital of the Cartesian
    file of the same calculation on a grid, and the frontier orbitals come out the same."""
    sph, cart = _axtb_molden(name, "spherical"), _axtb_molden(name, "cartesian")
    assert "[5D]" in sph and "[6D]" in cart and ("[7F]" in sph) == (name == "gxtb2_ceh4")
    v_sph, pts = _values(sph)
    v_cart, _ = _values(cart, points=pts)
    assert len(v_sph) == len(v_cart) > 10
    assert (np.abs(v_sph - v_cart).max(axis=1) / np.abs(v_cart).max(axis=1)).max() < 1e-7
    a = orbitals.orbitals_from_molden(sph, "axtb", extra=2, max_points=24 ** 3)
    b = orbitals.orbitals_from_molden(cart, "axtb", extra=2, max_points=24 ** 3)
    assert a["gap_ev"] == b["gap_ev"] and set(a["orbitals"]) == set(b["orbitals"])
    for k, orb in a["orbitals"].items():
        assert orb["energy_ev"] == b["orbitals"][k]["energy_ev"] and orb["norm"] == b["orbitals"][k]["norm"], k
        assert np.allclose(orb["atoms"], b["orbitals"][k]["atoms"], atol=1e-6), k


@pytest.mark.parametrize("name", ["gfn2_ch3cl", "gxtb2_ceh4"])
def test_axtb_orbitals_come_out_normalized(name):
    """Normalized primitives and functions (the Molden convention): every occupied orbital has
    norm 1 on a fine grid (Ce's tight functions need one)."""
    mol = orbitals.parse_molden(_axtb_molden(name, "spherical"))
    xyz = np.array([p for _, p in mol["atoms"]])
    origin, h, n = orbitals.grid_for(xyz, spacing=0.08, margin=6.0, max_points=110 ** 3)
    axes = [origin[i] + h * np.arange(n[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    occupied = [m["coef"] for m in mol["mos"] if m["occ"] > 0.5]
    norm = sum((orbitals.evaluate(mol, occupied, chunk, "own", True).astype(float) ** 2).sum(axis=1)
               for chunk in np.array_split(pts, 20)) * h ** 3
    assert np.abs(norm - 1).max() < 0.01


@pytest.mark.parametrize("flags", ["[7F]\n[5D]", "[5D]", "[5D7F]", "[5d]\n[7f]\n[9g]", "[5D]\n[7F]"])
def test_every_spelling_of_the_spherical_flags(flags):
    """[5D] alone means 5D and 7F (the format description), PySCF writes [5d][7f][9g], axtb
    [7F][5D]: all read the same."""
    sph = _axtb_molden("gxtb2_ceh4", "spherical")
    v_ref, pts = _values(sph)
    v, _ = _values(sph.replace("[7F]\n[5D]", flags), points=pts)
    assert np.array_equal(v, v_ref)


def test_spherical_flags():
    none = {0: False, 1: False, 2: False, 3: False, 4: False}
    assert orbitals._spherical_shells("[5D]\n") == {**none, 2: True, 3: True}
    assert orbitals._spherical_shells("[5D10F]\n") == {**none, 2: True}
    assert orbitals._spherical_shells("[7F]\n") == {**none, 3: True}
    assert orbitals._spherical_shells(" [5d7f]\n[9G]\n") == {**none, 2: True, 3: True, 4: True}
    assert orbitals._spherical_shells(_axtb_molden("gfn2_ch3cl", "cartesian")) == none
    for l in (2, 3):
        own, axis = orbitals._pure_to_cart(l, "own"), orbitals._pure_to_cart(l, "axis")
        assert own.shape == axis.shape == (len(orbitals._CART[l]), 2 * l + 1)
        assert np.allclose(own[0], axis[0])          # x^l: the same component either way
    with pytest.raises(orbitals.OrbitalError, match="g functions"):     # spherical g is not read
        orbitals.parse_molden("[Molden Format]\n[Atoms] AU\nX 1 1 0 0 0\n[GTO]\n1 0\ng 1 1.0\n1.0 1.0\n\n[9G]\n[MO]\n"
                              "Sym= 1a\nEne= -1.0\nSpin= Alpha\nOccup= 2.0\n1 1.0\n")


def test_axtb_files_are_read_with_the_molden_convention():
    """axtb writes normalized primitives with every Cartesian component normalized (as TeraChem);
    reading the spherical file with either component normalization gives the same orbitals."""
    assert orbitals.D_NORM["axtb"] == "own" and orbitals.NORMALIZED_PRIMS["axtb"] is True
    sph = _axtb_molden("gxtb2_ch3cl", "spherical")
    v_own, pts = _values(sph, "own")
    v_axis, _ = _values(sph, "axis", points=pts)
    assert (np.abs(v_own - v_axis).max(axis=1) / np.abs(v_own).max(axis=1)).max() < 1e-6
    with pytest.raises(orbitals.OrbitalError, match="axtb"):   # read from kept files, never run here
        orbitals.run_molden(*CH3CL, 0, 1, "axtb")
