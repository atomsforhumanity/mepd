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


def test_the_mixture_page_reads_mixture_json_and_the_seeds_and_feed_before_it(tmp_path):
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
        # before the first merge the page draws the feed: from the command (an xyz file the job keeps, or SMILES)
        inputs = mgr.job_dir("j_m") / "inputs"
        inputs.mkdir(parents=True)
        (inputs / "feed_0.xyz").write_text("3\n\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
        mgr.jobs["j_m"].update(argv=["discovery", "mixture", "--feed", "/elsewhere/feed_0.xyz", "--concentration", "2",
                                     "--feed", "CC", "--concentration", "0.5", "--temperature", "300", "--time", "60"],
                               params={"temperature": 300.0, "time_s": 60.0, "standard_state": "1M"})
        d = c.get("/api/jobs/j_m/mixture").json()
        assert [(f["molecule"], f["concentration"]) for f in d["feed"]] == [("O", 2.0), ("CC", 0.5)]
        assert d["settings"] == {"temperature_K": 300.0, "time_s": 60.0, "standard_state": "1M"}
        assert d["names"] == {"O": "water", "CC": "ethane"}
        (out / "mixture.json").write_text(json.dumps({"feed": [{"name": "water", "molecule": "O", "concentration": 1}],
                                                      "molecules": [{"molecule": "O"}], "reactions": [], "seeds": []}))
        d = c.get("/api/jobs/j_m/mixture").json()
        assert d["mixture"]["feed"][0]["molecule"] == "O" and d["names"]["O"] == "water" and "feed" not in d
        # molecules without an everyday name: their formula, and what tells apart those that share one
        para, meta = "COC(=O)C1CC=C(C)CC1", "COC(=O)C1CCC=C(C)C1"
        (out / "mixture.json").write_text(json.dumps({"feed": [{"name": "water", "molecule": "O", "concentration": 1}],
                                                      "molecules": [{"molecule": "O"}, {"molecule": para}, {"molecule": meta}],
                                                      "reactions": [], "seeds": []}))
        d = c.get("/api/jobs/j_m/mixture").json()
        assert d["labels"] == {para: {"formula": "C9H14O2", "label": "C9H14O2, 1,4-isomer", "tell": "1,4-isomer"},
                               meta: {"formula": "C9H14O2", "label": "C9H14O2, 1,3-isomer", "tell": "1,3-isomer"}}


def test_a_real_mixture_reads_by_its_names_with_mirror_images_counted(tmp_path):
    """What a real `discovery mixture` writes: the feed named by its snapshot file, a chiral TS's step listed
    again as its mirror image, and a species of two forms ("A|B")."""
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    app = create_app(tmp_path / "ws")
    with TestClient(app) as c:
        c.get("/api/state")
        mgr = next(iter(app.state.sessions._open.values())).jobs if hasattr(app.state, "sessions") else None
        if mgr is None:
            pytest.skip("no session manager handle")
        sid = c.post("/api/structures", json={"text": "OC=O", "name": "formic acid", "optimize": False}).json()[0]["id"]
        out = tmp_path / "out"
        out.mkdir()
        mgr.jobs["j_r"] = {"id": "j_r", "op": "network-mixture", "status": "done", "output_dir": str(out),
                           "targets": {"structures": [sid], "edges": []}}
        carbene = "[H]O/[C-]=[O+]/[H]|[H]O/[C-]=[O+]\\[H]"
        step = {"reactants": ["[H]OC([H])=O"], "products": [carbene], "barrier_kcal": 79.0,
                "reverse_barrier_kcal": 46.5, "net_extent": 0.004, "source": "547f/fast_0_3"}
        (out / "mixture.json").write_text(json.dumps({
            "feed": [{"name": str(tmp_path / "jobs/j_r/inputs/feed_0.xyz"), "molecule": "[H]OC([H])=O",
                      "concentration": 1}],
            "molecules": [{"molecule": "[H]OC([H])=O"}, {"molecule": carbene}],
            "reactions": [step, {**step, "mirror_image": True}, {**step, "source": "547f/fast_0_3_rot0", "barrier_kcal": 80.0}],
            "seeds": []}))
        d = c.get("/api/jobs/j_r/mixture").json()
        assert d["mixture"]["feed"][0]["name"] == "formic acid"
        # a chiral TS's mirror image: one reaction, both channels' conversion, saying so
        rx = d["mixture"]["reactions"]
        assert [r["source"] for r in rx] == ["547f/fast_0_3", "547f/fast_0_3_rot0"]
        assert rx[0]["channels"] == 2 and rx[0]["mirror_images"] and rx[0]["net_extent"] == pytest.approx(0.008)
        assert "channels" not in rx[1] and d["mixture"]["channels"] == 3
        assert c.get("/api/depict", params={"smiles": carbene}).status_code == 200
