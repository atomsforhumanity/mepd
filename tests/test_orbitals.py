"""Frontier orbitals (mepd.orbitals, /api/orbitals): xtb's molden orbitals
evaluated on a grid must come out normalized for both programs, whose
Cartesian d functions are scaled differently."""
import json
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


def test_the_lookup_reads_only_live_entries_and_follows_a_growing_index(tmp_path):
    """An index lists every SCF ever kept, the folder only the newest: entries whose molden file is gone when
    the index is first read are skipped; lines appended later are read on from where the last read stopped;
    exact, turned and different geometries are told apart as before."""
    from scipy.spatial.transform import Rotation

    d = tmp_path / "orbitals"
    d.mkdir()
    rng = np.random.default_rng(0)
    syms = ["C", "O", "H", "H"]
    geoms = [rng.normal(size=(4, 3)) * 1.2 for _ in range(6)]

    def keep(k, exists=True):
        name = f"{1000 + k}-1.molden"
        if exists:
            (d / name).write_text("kept")
        with open(d / "index.jsonl", "a") as fh:
            fh.write(json.dumps({"file": name, "method": "gxtb", "label": "g-xTB", "source": "run", "symbols": syms,
                                 "coords": geoms[k].round(5).tolist()}) + "\n")

    for k in range(4):
        keep(k, exists=k >= 2)                       # the first two were pruned long ago
    assert [e["file"] for e in orbitals._index(d / "index.jsonl")] == ["1002-1.molden", "1003-1.molden"]
    assert orbitals.find_run_orbitals([d], syms, geoms[0]) is None
    hit = orbitals.find_run_orbitals([d], syms, geoms[3] + 1e-4)
    assert hit["file"] == "1003-1.molden" and hit["align"] is None
    R = Rotation.from_euler("xyz", [20, 70, -40], degrees=True).as_matrix()
    turned = (geoms[2] - geoms[2].mean(axis=0)) @ R + 3.0
    hit = orbitals.find_run_orbitals([d], syms, turned)
    assert hit["file"] == "1002-1.molden" and hit["align"] is not None
    moved = turned + 0.0
    moved[0] += 0.05                                  # a different geometry: no match, turned or not
    assert orbitals.find_run_orbitals([d], syms, moved) is None
    assert orbitals.find_run_orbitals([d], syms, turned, rigid=False) is None
    keep(4)                                           # a running calculation keeps another
    assert orbitals.find_run_orbitals([d], syms, geoms[4])["file"] == "1004-1.molden"
    assert orbitals.find_run_orbitals([d], syms[::-1], geoms[4]) is None   # other atoms
    (d / "1004-1.molden").unlink()                    # pruned after it was read: checked when found
    assert orbitals.find_run_orbitals([d], syms, geoms[4]) is None


def test_orbitals_follow_by_overlap_and_say_what_happens(tmp_path):
    """mepd.orbital_path on a short path with stand-in frames: lines follow each orbital, their character is read
    from Mulliken shares, and the story names the frontier at both ends."""
    import json

    import numpy as np
    import pytest

    pytest.importorskip("linus")
    from mepd import orbital_path as op

    # H2 stretched along a path, within bonding distance (H-H 0.70 -> 0.88 A): the bonding orbital rises.
    frames = [np.array([[0, 0, 0], [0, 0, r]]) for r in np.linspace(0.70, 0.88, 6)]
    doc = op.analyze(["H", "H"], frames, 0, 1, "pbe0", "sto-3g", tmp_path, ts_index=3, count=6, threads=1,
                     log=lambda m: None)
    assert doc["status"] == "done" and all(f is not None for f in doc["frames"])
    homo = [ln for ln in doc["lines"] if all(p[1] == doc["frames"][p[0]]["homo"] for p in ln["points"])]
    assert homo and len(homo[0]["points"]) == 6                        # followed through every frame
    first = doc["frames"][0]["orbitals"]
    sigma = next(o for o in first if o["rel"] == 0)
    sigma_star = next(o for o in first if o["rel"] == 1)
    assert sigma["what"]["kind"] == "bond" and sigma_star["what"]["kind"] == "antibond"
    # near equilibrium the closed shell is stable: said first, then the frontier
    assert doc["story"][0]["kind"] == "stability" and doc["story"][0]["ok"] is True
    assert doc["story"][1]["kind"] == "frontier" and doc["story"][1]["text"] == "HOMO: σ H1–H2 all along"
    assert "H1–H2 σ*" in doc["story"][2]["detail"]
    assert json.loads((tmp_path / "orbitals.json").read_text())["status"] == "done"
    cube = op.orbital_cube(tmp_path, 0, sigma["i"], max_points=12 ** 3)
    assert cube.splitlines()[2].split()[0] == "2"                       # two atoms in the cube


def test_the_closed_shell_check_finds_the_coulson_fischer_point_of_h2(tmp_path):
    """H2 pulled apart at HF/STO-3G (Szabo & Ostlund, Modern Quantum Chemistry, 3.8.7): the restricted solution is
    stable at equilibrium and becomes unstable to the unrestricted (broken-symmetry) one past the Coulson-Fischer
    point near 1.2 A; far apart, the natural orbitals tend to one electron each (a pure diradical, y -> 1)."""
    import numpy as np
    import pytest

    pytest.importorskip("linus")
    from mepd import orbital_path as op

    R = [0.74, 1.0, 1.15, 1.3, 2.0, 3.5]
    frames = [np.array([[0, 0, 0], [0, 0, r]]) for r in R]
    doc = op.analyze(["H", "H"], frames, 0, 1, "hf", "sto-3g", tmp_path, count=len(R), threads=1, log=lambda m: None)
    st = [f["stability"] for f in doc["frames"]]
    assert [s["unstable"] for s in st] == [False, False, False, True, True, True]
    assert st[0]["n_homo"] == 2.0 and st[0]["n_lumo"] == 0.0 and st[0]["y"] == 0.0
    assert st[3]["de_kcal"] < -1 and 0.1 < st[3]["n_lumo"] < 0.5
    assert abs(st[-1]["n_homo"] - 1) < 0.05 and abs(st[-1]["n_lumo"] - 1) < 0.05 and st[-1]["y"] > 0.9
    assert abs(st[-1]["s2"] - 1.0) < 0.02                      # the broken-symmetry singlet: half triplet
    head = doc["story"][0]
    assert head["kind"] == "stability" and head["ok"] is False and head["text"].startswith("Closed-shell picture fails at frames 4–6")
    assert op.diradical_character(0.0, 2.0) == 0.0 and abs(op.diradical_character(1.0, 1.0) - 1.0) < 1e-12


def test_an_open_shell_follows_both_spins_and_names_its_somo(tmp_path):
    """Linear H + H2 -> H2 + H, a doublet (unrestricted): the unpaired electron moves from one end H to the other
    through the nonbonding sigma orbital of linear H3, with a node at the central H at the symmetric TS (e.g. Albright,
    Burdett & Whangbo, Orbital Interactions in Chemistry); alpha and beta are followed apart, <S^2> stays near 0.75."""
    import numpy as np
    import pytest

    pytest.importorskip("linus")
    from mepd import orbital_path as op

    frames = []
    for s in np.linspace(-2.0, 2.0, 7):
        r1, r2 = 0.74 + 0.19 * np.exp(s), 0.74 + 0.19 * np.exp(-s)
        frames.append(np.array([[0, 0, 0], [r1, 0, 0], [r1 + r2, 0, 0]]))
    doc = op.analyze(["H", "H", "H"], frames, 0, 2, "pbe0", "sto-3g", tmp_path, ts_index=3, count=7, threads=1,
                     log=lambda m: None)
    assert doc["spin"] == "unrestricted" and not doc["stability_checked"]
    assert {ln["spin"] for ln in doc["lines"]} == {"a", "b"}
    ts = doc["frames"][3]
    somo = next(o for o in ts["orbitals"] if o.get("somo"))
    assert somo["what"]["kind"] == "radical" and sorted(somo["what"]["atoms"]) == [0, 2]      # on the two ends
    assert ts["beta"]["homo"] == 0 and all(0.75 <= f["s2"] < 0.8 for f in doc["frames"])
    start = next(o for o in doc["frames"][0]["orbitals"] if o.get("somo"))
    assert start["what"]["kind"] == "radical" and start["what"]["atoms"] == [2]               # on the free H (H3)
    texts = [x["text"] for x in doc["story"]]
    assert texts[0].startswith("⟨S²⟩ 0.75") and any(t.startswith("α SOMO: • H3 → • H1,H3 at the TS → • H1") for t in texts)
    assert op.orbital_cube(tmp_path, 3, ts["beta"]["orbitals"][0]["i"], max_points=10 ** 3, spin="beta")


def test_a_delocalized_pi_orbital_is_named_as_one(tmp_path):
    """s-cis butadiene's frontier orbitals (Huckel psi2 and psi3) are pi systems over all four carbons."""
    import numpy as np
    import pytest

    pytest.importorskip("linus")
    from mepd import orbital_path as op

    C2, C3 = np.array([-0.74, 0, 0]), np.array([0.74, 0, 0])
    c, s = np.cos(np.radians(124)), np.sin(np.radians(124))
    C1 = C2 + 1.34 * np.array([c, s, 0])
    C4 = np.array([-C1[0], C1[1], 0])
    rot = lambda v, a: np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1], 0])
    H2 = C2 + 1.09 * rot(np.array([1, 0, 0]), np.radians(-117))
    u = (C2 - C1) / np.linalg.norm(C2 - C1)
    Ha, Hb = C1 + 1.08 * rot(u, np.radians(121.5)), C1 + 1.08 * rot(u, np.radians(-121.5))
    m = lambda v: np.array([-v[0], v[1], v[2]])
    xyz = np.array([C1, C2, C3, C4, H2, m(H2), Ha, Hb, m(Ha), m(Hb)])
    f = op.Frame(op.linus_molden(["C"] * 4 + ["H"] * 6, xyz, 0, 1, "pbe0", "sto-3g", 1)[0], ["C"] * 4 + ["H"] * 6)
    homo, lumo = f.character(f.side.homo), f.character(f.side.homo + 1)
    assert homo["kind"] == "pi" and homo["atoms"] == [0, 1, 2, 3] and not homo["empty"]
    assert lumo["kind"] == "pi" and lumo["atoms"] == [0, 1, 2, 3] and lumo["empty"]
