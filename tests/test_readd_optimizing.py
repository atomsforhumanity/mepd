"""A molecule added again is not left "optimizing" for good: the second geometry's minimization relaxes onto the
conformer the structure already has, and that path never cleared the status, so every calculation on the molecule
was refused ("still being optimized"). Seen on mepd.afh.org, 2026-11-09."""
import json

from fastapi.testclient import TestClient

from mepd.web import chem
from mepd.web.app import apply_optimization, create_app
from mepd.web.workspace import Workspace
from tests.test_web import WATER_XYZ

WATER_2 = WATER_XYZ.replace("0.758 0.000 0.504", "0.700 0.100 0.520", 1)
LEVEL = {"profile": None, "key": "k1", "label": "gxtb"}


def _s(xyz):
    return chem.structures_from_xyz_text(xyz, 0, 1)[0]


def _finished_optimize(tmp_path, sid, conformer, xyz, energy, status="done"):
    out = tmp_path / "job" / "output"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"structures": [{"index": 0, "converged": True, "energy": energy}]}))
    (out / "opt_0.xyz").write_text(xyz)
    return {"id": "j_x", "op": "optimize", "status": status, "output_dir": str(out), "level": LEVEL,
            "targets": {"structures": [sid], "edges": []}, "target_conformers": [conformer]}


def test_a_second_copy_that_relaxes_onto_the_first_leaves_the_molecule_ready(tmp_path):
    ws = Workspace(tmp_path / "ws")
    first = ws.add_or_merge(_s(WATER_XYZ), origin={"kind": "xyz"}, energy=-76.0, level=LEVEL)
    sid = first["rec"]["id"]
    second = ws.add_or_merge(_s(WATER_2), origin={"kind": "xyz"})      # the same molecule again, unoptimized
    assert second["merged"] and not second["duplicate"]
    ws.set_status([sid], "optimizing")                                 # what queue_optimization does
    job = _finished_optimize(tmp_path, sid, second["conformer"], WATER_XYZ, -76.0)
    apply_optimization(ws, job)
    rec = ws.structure(sid)
    assert len(rec["conformers"]) == 1                                 # relaxed onto the first: one kept
    assert rec["status"] == "ready"


def test_another_minimization_still_running_keeps_it_optimizing(tmp_path):
    ws = Workspace(tmp_path / "ws")
    first = ws.add_or_merge(_s(WATER_XYZ), origin={"kind": "xyz"}, energy=-76.0, level=LEVEL)
    sid = first["rec"]["id"]
    second = ws.add_or_merge(_s(WATER_2), origin={"kind": "xyz"})
    ws.set_status([sid], "optimizing")
    apply_optimization(ws, _finished_optimize(tmp_path, sid, second["conformer"], WATER_XYZ, -76.0),
                       busy=frozenset({sid}))
    assert ws.structure(sid)["status"] == "optimizing"


def test_a_stale_optimizing_status_does_not_block_a_calculation(tmp_path, monkeypatch):
    """Workspaces already stuck (before the fix): with no minimization queued or running, the status settles
    when a calculation is asked for, instead of refusing it."""
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        (rec,) = c.post("/api/structures", json={"text": WATER_XYZ, "optimize": False}).json()
        ws = c.app.state.sessions.current.ws
        ws.set_status([rec["id"]], "optimizing")
        r = c.post("/api/jobs", json={"op": "conformers", "structures": [rec["id"]], "params": {}, "dry_run": True})
        assert r.status_code == 200, r.text
        assert ws.structure(rec["id"])["status"] == "ready"
