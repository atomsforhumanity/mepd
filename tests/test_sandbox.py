"""The interactive reactor (mepd.web.sandbox): live GFN2-xTB dynamics that
follow pulls and wall changes, and its web endpoints."""
import time

import numpy as np
import pytest

pytest.importorskip("tblite.interface")

from mepd.web import chem, sandbox  # noqa: E402


def _wait(box, since, seconds):
    f = {"seq": since}
    t0 = time.time()
    while time.time() - t0 < seconds:
        f = box.frame(f["seq"])
    return f


def test_the_dynamics_follow_a_pull_and_the_wall():
    box = sandbox.start("t_pull", [chem.structure_from_smiles("O")] * 3, temperature=300)
    try:
        f = _wait(box, 0, 2.5)
        assert f.get("pos") and f["t_fs"] > 0 and not f.get("error")
        target = (np.array(f["pos"][0]) + [3.0, 0, 0]).tolist()
        box.command({"op": "pull", "atom": 0, "target": target, "k": 50})
        f = _wait(box, f["seq"], 2.0)
        assert np.linalg.norm(np.array(f["pos"][0]) - target) < 1.0 and "0" in {str(k) for k in f["pulls"]}
        box.command({"op": "release_all"})
        box.command({"op": "radius", "value": 1.0})               # clamped to the 3 Angstrom minimum
        f = _wait(box, f["seq"], 2.0)
        assert f["radius"] == 3.0 and not f["pulls"]
    finally:
        sandbox.stop("t_pull")
    assert not box.proc.is_alive()


def test_the_endpoints_start_steer_snapshot_and_stop(tmp_path):
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    ws = tmp_path / "ws"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(ws, max_concurrent=1)) as c:
        (w,) = c.post("/api/structures", json={"text": "O water", "optimize": False}).json()
        box = c.post("/api/sandbox", json={"counts": {w["id"]: 3}, "temperature": 300}).json()
        sid = box["id"]
        try:
            assert len(box["symbols"]) == 9
            f = {"seq": 0}
            for _ in range(10):
                f = c.get(f"/api/sandbox/{sid}/frame", params={"since": f["seq"]}).json()
            assert f["pos"]
            assert c.post(f"/api/sandbox/{sid}/command", json={"op": "temperature", "value": 500}).json()["ok"]
            assert c.post(f"/api/sandbox/{sid}/command", json={"op": "piston", "value": True}).json()["ok"]
            assert c.post(f"/api/sandbox/{sid}/command", json={"op": "rm -rf"}).status_code == 400
            snap = c.post(f"/api/sandbox/{sid}/snapshot").json()
            assert snap["natoms"] == 9 and snap["role"] in ("complex", "minimum")
            for _ in range(10):                                         # enough trajectory to analyze
                f = c.get(f"/api/sandbox/{sid}/frame", params={"since": f["seq"]}).json()
            out = c.post(f"/api/sandbox/{sid}/analyze").json()         # Stop & analyze: a nanoreactor job on it
            job = c.get(f"/api/jobs/{out['job']}").json()["job"]
            assert job["op"] == "nanoreactor" and "--trajectory" in job["argv"]
            traj = job["argv"][job["argv"].index("--trajectory") + 1]
            assert traj.endswith("trajectory.xyz") and (ws / "sandbox" / sid / "first_frame.xyz").exists()
            assert c.get(f"/api/sandbox/{sid}").status_code == 404     # stopped and handed over
            c.post(f"/api/jobs/{out['job']}/cancel")
        finally:
            c.delete(f"/api/sandbox/{sid}")
        assert c.get(f"/api/sandbox/{sid}").status_code == 404


def test_the_demo_reactor_is_per_visitor_and_never_evicts_another(tmp_path):
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app
    from mepd.web.demo import DemoPolicy

    root = tmp_path / "demo"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    policy = DemoPolicy(max_atoms=10, sandbox={"total": 2, "per_owner": 1, "idle_s": 60, "max_s": 60})
    app = create_app(root, demo=policy, demo_password="pw")

    def visitor():
        c = TestClient(app, follow_redirects=False)
        c.__enter__()
        r = c.post("/login", data={"password": "pw"})
        c.cookies.set("mepd_visitor", r.cookies.get("mepd_visitor"))
        (w,) = c.post("/api/structures", json={"text": "O water", "optimize": False}).json()
        return c, w["id"]

    (a, wa), (b, wb), (c, wc) = visitor(), visitor(), visitor()
    started = []
    try:
        r = a.post("/api/sandbox", json={"counts": {wa: 11}})          # 33 atoms > 3 x max_atoms
        assert r.status_code == 400 and "at most 30" in r.json()["detail"]
        first = a.post("/api/sandbox", json={"counts": {wa: 2}}).json()["id"]
        second = a.post("/api/sandbox", json={"counts": {wa: 2}}).json()["id"]   # replaces a's first
        started += [first, second]
        assert a.get(f"/api/sandbox/{first}").status_code == 404
        assert sandbox.get(second).max_s == 60
        assert b.get(f"/api/sandbox/{second}").status_code == 404        # not b's to see or stop
        assert b.delete(f"/api/sandbox/{second}").status_code == 404
        started.append(b.post("/api/sandbox", json={"counts": {wb: 2}}).json()["id"])
        r = c.post("/api/sandbox", json={"counts": {wc: 2}})             # full: refused, nobody evicted
        assert r.status_code == 400 and "in use" in r.json()["detail"]
        assert a.get(f"/api/sandbox/{second}").status_code == 200
        r = a.post("/api/jobs", json={"op": "nanoreactor", "structures": [wa],
                                      "params": {"trajectory": "/etc/passwd"}})
        assert r.status_code == 400
    finally:
        for sid in started:
            sandbox.stop(sid)
        for client in (a, b, c):
            client.__exit__(None, None, None)


def test_reaction_events_are_found_as_the_nanoreactor_finds_them():
    """Hydrogen atoms make H2: the event list (nanoreactor detection on the
    evenly spaced trajectory) says so, in the nanoreactor's own labels."""
    # A fixed packing: from some random ones (2 in 12) the atoms do not meet within 15 s.
    box = sandbox.start("t_ev", [chem.structure_from_smiles("[H]")] * 4, temperature=400, seed=0)
    try:
        t0 = time.time()
        while time.time() - t0 < 15 and not box.events_view()["events"]:
            box.frame(0, wait=0.5)
        view = box.events_view()
        assert view["frame_fs"] == sandbox.TRAJ_FS and view["n_frames"] > 10
        assert any(e["label"] == "2 [H] -> [H][H]" for e in view["events"])
    finally:
        sandbox.stop("t_ev")


def test_a_sandbox_starts_from_a_geometry_as_it_is():
    from mepd.web.compose import place

    w = chem.structure_from_smiles("O")
    box = sandbox.start_from("t_geo", place([w, w]), temperature=300)
    try:
        f = _wait(box, 0, 1.5)
        assert len(f["pos"]) == 6 and box.describe()["temperature"] == 300.0
        assert 2.0 < f["radius"] < 8.0                                    # just beyond its farthest atom
    finally:
        sandbox.stop("t_geo")


def test_the_piston_cycle_matches_the_nanoreactor():
    """Wide for 75 % of a 1 ps cycle, a 100 fs ramp, then 0.7 x the radius;
    settings that cannot be a cycle keep the previous ones."""
    p = sandbox.piston_settings({"value": True}, None)
    assert p == sandbox.PISTON
    p["t0"] = 100.0
    r = [sandbox.piston_radius(t, 10.0, p) for t in (100, 849, 900, 950, 1000, 1099, 1100)]
    assert r == pytest.approx([10.0, 10.0, 8.5, 7.0, 7.0, 7.0, 10.0])
    assert sandbox.piston_radius(500, 10.0, None) == 10.0
    q = sandbox.piston_settings({"value": True, "period_ps": 0.25, "compress": 2.0, "duty": "x"}, p)
    assert q == {"period_ps": 0.25, "compress": 0.7, "duty": 0.75}
    assert sandbox.piston_settings({"value": False}, p) is None


def test_the_piston_moves_the_wall_in_the_dynamics():
    box = sandbox.start("t_piston", [chem.structure_from_smiles("O")] * 3, temperature=300)
    try:
        f = _wait(box, 0, 1.5)
        wide = f["radius"]
        assert f["wall_now"] == pytest.approx(wide) and f["piston"] is None
        box.command({"op": "piston", "value": True, "period_ps": 0.4, "compress": 0.5, "duty": 0.5})
        seen = set()
        t0 = time.time()
        while time.time() - t0 < 8 and not {1.0, 0.5} <= seen:
            f = box.frame(f["seq"])
            seen.add(round(f["wall_now"] / wide, 1))
        assert f["piston"] == {"period_ps": 0.4, "compress": 0.5, "duty": 0.5}
        assert {1.0, 0.5} <= seen and f["radius"] == wide                   # the Wall stays the user's
    finally:
        sandbox.stop("t_piston")


def test_a_reactor_exits_when_its_server_is_killed():
    """Only the server's reader thread stops an idle reactor; a server killed
    outright (SIGKILL) must not leave its MD process stepping on forever."""
    import os
    import signal
    import subprocess
    import sys
    import textwrap

    server = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
        import time
        from mepd.web import chem, sandbox
        box = sandbox.start("t_orphan", [chem.structure_from_smiles("O")] * 2, temperature=300)
        print(box.proc.pid, flush=True)
        time.sleep(600)
    """)], stdout=subprocess.PIPE, text=True)
    try:
        pid = int(server.stdout.readline())
        time.sleep(3)                                # past the start-up, into the MD loop
        os.kill(server.pid, signal.SIGKILL)
        server.wait(10)
        t0 = time.time()
        while time.time() - t0 < 15:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("the reactor outlived its server")
    finally:
        if server.poll() is None:
            server.kill()


def test_collide_pushes_whole_molecules_as_the_network_generator_does():
    """Collide is reactor_generator's own move: two molecules picked at random,
    pushed together as wholes (langevin.pull_pairs with (i, j, A, B)), or two
    atoms of one molecule; the frames say which, and auto repeats it."""
    box = sandbox.start("t_collide", [chem.structure_from_smiles("O")] * 3, temperature=300)
    try:
        f = _wait(box, 0, 2.0)
        box.command({"op": "auto_collide", "value": True})
        seen = []
        t0 = time.time()
        while time.time() - t0 < 6 and len(seen) < 3:
            f = box.frame(f["seq"])
            c = f.get("collide")
            if c and c.get("i") is not None:
                seen.append(c)
        assert seen and all(c["auto"] for c in seen)
        between = [c for c in seen if "a" in c]
        for c in between:   # whole molecules: each water's three atoms, the two atoms one of each
            assert len(c["a"]) == 3 and len(c["b"]) == 3 and c["i"] in c["a"] and c["j"] in c["b"]
            assert not set(c["a"]) & set(c["b"])
        box.command({"op": "release_all"})            # stops the pushes and the repeats
        f = _wait(box, f["seq"], 1.0)
        assert not (f.get("collide") or {}).get("auto")
    finally:
        sandbox.stop("t_collide")
