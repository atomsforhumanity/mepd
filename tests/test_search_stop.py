"""Stop & analyze (mepd.search_control): a stop request ends the running path
search with its current path and keeps every search from splitting further;
what was found stays and is analyzed."""
import numpy as np
import pytest

from mepd import search_control
from mepd.chain import Chain
from mepd.engines.flower import FlowerPotential
from mepd.inputs import ChainInputs, NEBInputs, RunInputs
from mepd.msmep import MSMEP
from mepd.nodes.node import XYNode
from mepd.optimizers.cg import ConjugateGradient


def test_a_stop_ends_only_a_path_search(tmp_path, monkeypatch):
    fp = tmp_path / "stop"
    monkeypatch.setenv(search_control.STOP_ENV, str(fp))
    search_control.check("chain")                       # nothing asked yet
    fp.touch()
    search_control.check("chain")                       # outside a path search (TS optimization, IRC): no-op
    with pytest.raises(search_control.SearchStopped) as stop, search_control.interruptible():
        search_control.check("chain")
    assert stop.value.chain == "chain"
    assert search_control.search_over(None) == "stopped"
    monkeypatch.delenv(search_control.STOP_ENV)
    assert search_control.search_over(None) is None


def _flower(max_steps=200):
    coords = np.linspace([-2.59807434, -1.499999], [2.5980755, 1.49999912], 15)
    coords[1:-1] += [-1, 1]
    chain_inputs = ChainInputs(k=10, delta_k=9, interpolation="linear", node_ene_thre=10)
    neb = NEBInputs(barrier_thre=5, v=False, max_steps=max_steps, climb=False, do_elem_step_checks=True,
                    early_stop_force_thre=0.1)
    chain = Chain.model_validate({"nodes": [XYNode(structure=xy) for xy in coords], "parameters": chain_inputs})
    run_inputs = RunInputs(path_min_inputs=neb.__dict__, chain_inputs=chain_inputs.__dict__)
    run_inputs.engine = FlowerPotential()
    run_inputs.optimizer = ConjugateGradient(timestep=0.1)
    return MSMEP(run_inputs), chain


def test_stopping_mid_search_keeps_its_path_as_a_stopped_leaf(monkeypatch):
    steps = []

    def asked():   # the stop arrives at the search's 5th step
        steps.append(1)
        return len(steps) >= 5
    monkeypatch.setattr(search_control, "stop_requested", asked)
    m, chain = _flower()
    history = m.run_recursive_minimize(chain, max_depth=2)
    (leaf,) = history.ordered_leaves
    assert leaf.leaf_status == "stopped" and not leaf.children
    out = history.output_chain
    first = leaf.data.chain_trajectory[0]    # MSMEP's interpolation, before any step
    assert len(out) == len(first) and np.all(np.isfinite(out.energies))
    assert not np.allclose(out.coordinates, first.coordinates)     # the path as it was at that step
    assert getattr(leaf.data, "stopped", False) and len(leaf.data.chain_trajectory) <= 7


def test_without_a_stop_the_same_search_runs_on():
    m, chain = _flower()
    history = m.run_recursive_minimize(chain, max_depth=2)
    assert all(getattr(n, "leaf_status", None) != "stopped" for n in history.ordered_leaves)


def test_the_web_stops_only_running_stoppable_jobs_through_their_own_file(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mepd.web import operations as ops
    from mepd.web.app import create_app
    from mepd.web.jobs import STOP_FILE

    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    assert ops.OPERATIONS["channels-more"].stoppable and ops.OPERATIONS["channels"].stoppable
    assert ops.OPERATIONS["ts"].stoppable and not ops.OPERATIONS["optimize"].stoppable
    assert ops.OPERATIONS["channels-more"].describe()["stoppable"] is True
    app = create_app(tmp_path / "ws", max_concurrent=1)
    with TestClient(app) as client:
        client.get("/api/state")
        manager = app.state.sessions.current.jobs
        job = {"id": "j_test", "op": "channels-more", "status": "running", "title": "t", "targets": {}}
        manager.jobs["j_test"] = job
        manager.job_dir("j_test").mkdir(parents=True, exist_ok=True)
        r = client.post("/api/jobs/j_test/stop-analyze")
        assert r.status_code == 200 and r.json()["stopping"] is True
        assert (manager.job_dir("j_test") / STOP_FILE).exists()
        job["status"] = "done"
        assert "not running" in client.post("/api/jobs/j_test/stop-analyze").json()["detail"]
        manager.jobs["j_opt"] = {**job, "id": "j_opt", "op": "optimize", "status": "running"}
        assert "cannot be stopped early" in client.post("/api/jobs/j_opt/stop-analyze").json()["detail"]
