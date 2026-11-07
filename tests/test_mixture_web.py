"""The web's reacting mixture (`mepd discovery mixture`): its command, and its page's data."""
import json

import pytest

from mepd.web.operations import OPERATIONS, _cli_options
from mepd.web.workspace import WorkspaceError


class Ctx:
    output_dir = "/out"
    structures = [{"id": "s_cp"}, {"id": "s_ma"}]

    def snapshot_structure(self, rec, name):
        return f"/in/{name}.xyz"

    def common_flags(self):
        return ["--charge", "0", "--inputs", "/in/profile.toml"]


def test_the_mixture_command_has_every_feed_its_concentration_and_the_growth_choice():
    op = OPERATIONS["network-mixture"]
    p = op.parse_params({"grow_by": "reactor", "concentrations": "0.92, 0.5", "copies": 2, "generations": 1,
                         "seed_minutes": 5})
    argv = op.build(Ctx(), p)
    assert argv[:2] == ["discovery", "mixture"]
    assert [argv[i + 1] for i, w in enumerate(argv) if w == "--feed"] == ["/in/feed_0.xyz", "/in/feed_1.xyz"]
    assert [argv[i + 1] for i, w in enumerate(argv) if w == "--concentration"] == ["0.92", "0.5"]
    assert "--expand-args=--generator reactor --time-budget 300" in argv
    assert "--continuous" in argv and argv[argv.index("--inputs") + 1] == "/in/profile.toml"
    known = _cli_options(("discovery", "mixture"))
    assert not [w for w in argv if w.startswith("--") and w.split("=")[0] not in known]
    # one concentration for all
    argv = op.build(Ctx(), op.parse_params({"concentrations": "1"}))
    assert [argv[i + 1] for i, w in enumerate(argv) if w == "--concentration"] == ["1", "1"]
    with pytest.raises(WorkspaceError):
        op.build(Ctx(), op.parse_params({"concentrations": "1, 2, 3"}))


def test_the_demo_caps_a_mixture():
    from mepd.web.demo import DemoPolicy

    policy = DemoPolicy()
    policy.check_op("network-mixture", {"workers": 4, "copies": 2})
    for bad in ({"workers": 16}, {"copies": 3}, {"generations": 4}, {"seed_minutes": 120}):
        with pytest.raises(WorkspaceError):
            policy.check_op("network-mixture", bad)


def test_the_mixture_page_reads_mixture_json_and_the_seeds_before_it(tmp_path):
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    app = create_app(tmp_path / "ws")
    with TestClient(app) as c:
        c.get("/api/state")
        mgr = next(iter(app.state.sessions._open.values())).jobs if hasattr(app.state, "sessions") else None
        if mgr is None:
            pytest.skip("no session manager handle")
        out = tmp_path / "out"
        (out / "seeds" / "g0-a").mkdir(parents=True)
        mgr.jobs["j_m"] = {"id": "j_m", "op": "network-mixture", "status": "running", "output_dir": str(out),
                           "targets": {"structures": [], "edges": []}}
        d = c.get("/api/jobs/j_m/mixture").json()
        assert d["mixture"] is None and d["seeds"] == [{"name": "g0-a", "status": "running"}]
        (out / "mixture.json").write_text(json.dumps({"feed": [{"name": "water", "molecule": "O", "concentration": 1}],
                                                      "molecules": [{"molecule": "O"}], "reactions": [], "seeds": []}))
        d = c.get("/api/jobs/j_m/mixture").json()
        assert d["mixture"]["feed"][0]["molecule"] == "O" and d["names"]["O"] == "water"
