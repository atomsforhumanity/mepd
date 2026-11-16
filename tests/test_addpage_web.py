"""The Add a molecule page of mepd web (mepd/web/addpage.py): the query kept in the workspace, candidates read
from text, result files as lines and detail, the geometry for the 3D view, the variant the map marks, the job."""
import json

import pytest

from mepd import addmol
from mepd.web import added, addpage
from mepd.web.workspace import Workspace, WorkspaceError

# H, C, N in one order: HCN, a bent TS, HNC (the run of tests/test_addmol_web.py)
FRAMES = {"hcn": [(0.0, 0.0, -1.07), (0.0, 0.0, 0.0), (0.0, 0.0, 1.16)],
          "ts": [(1.10, 0.0, 0.60), (0.0, 0.0, 0.0), (0.0, 0.0, 1.19)],
          "hnc": [(0.0, 0.0, 2.17), (0.0, 0.0, 0.0), (0.0, 0.0, 1.17)]}
E_HCN, E_TS, E_HNC = -93.40, -93.32, -93.38
KCAL = 627.509474


def _xyz(key, energy):
    rows = "\n".join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip("HCN", FRAMES[key]))
    return f"3\nqcdata_charge=0 qcdata_multiplicity=1 E={energy}\n{rows}\n"


@pytest.fixture
def mgr(tmp_path):
    """A workspace with one adopted run: HCN, HNC and the IRC-verified step between them."""
    out = tmp_path / "expand" / "run"
    (out / "species").mkdir(parents=True)
    (out / "ts").mkdir()
    (tmp_path / "expand" / "profile.toml").write_text('engine_name = "gxtb"\n')
    (out / "species" / "species_0.xyz").write_text(_xyz("hcn", E_HCN))
    (out / "species" / "species_1.xyz").write_text(_xyz("hnc", E_HNC))
    (out / "ts" / "fast_0_1.xyz").write_text(_xyz("ts", E_TS))
    (out / "ts" / "fast_0_1_irc.xyz").write_text(_xyz("hcn", E_HCN) + _xyz("ts", E_TS) + _xyz("hnc", E_HNC))
    (out / "summary.json").write_text(json.dumps({
        "seed_energy": E_HCN,
        "species": [{"index": 0, "smiles": "C#N", "energy": E_HCN, "rel_energy_kcal": 0.0},
                    {"index": 1, "smiles": "[C-]#[NH+]", "energy": E_HNC, "rel_energy_kcal": (E_HNC - E_HCN) * KCAL}],
        "reactions": [{"source": 0, "target": 1, "outcome": "new_species"}],
        "steps": [{"a": 0, "b": 1, "label": "fast_0_1", "ts_energy": E_TS,
                   "barrier_kcal": [(E_TS - E_HCN) * KCAL, (E_TS - E_HNC) * KCAL], "files": {"verified": True}}]}))
    m = added.manager(Workspace(tmp_path / "ws"))
    added.adopt_run(m, out)
    return m


def ask(mgr, temperature=300.0, time_s=3600.0):
    """HNC from HCN (uphill: at 300 K none forms, at 1500 K some does)."""
    ids = {r["smiles"]: sid for sid, r in mgr.ws.snapshot()["structures"].items()}
    return addpage.make_query(mgr.ws, mgr.jobs, target=ids["[C-]#[NH+]"], initial={ids["C#N"]: 1.0},
                              temperature=temperature, time_s=time_s, free_energy=False)


def put_result(mgr, qid, name, estimates=None, **kw):
    q = json.loads((addpage.folder(mgr.ws, qid) / "query.json").read_text())
    est = {"species": [], "steps": [], "new_species": [], "new_steps": [], **(estimates or {})}
    doc = addmol.result(q, {"name": name, "smiles": "O", "amount": 0.1}, est, **kw)
    fp = addpage.folder(mgr.ws, qid) / "results" / addpage.file_name(name)
    fp.parent.mkdir(exist_ok=True)
    addmol.write_json(fp, doc)
    return q, fp.name


def relay(q, barrier=25.0, status="est", **more):
    st = q["steps"][0]
    return {"new_steps": [{"id": "relay", "label": "HCN -> HNC through the molecule", "parent_step": st["id"],
                           "reactants": [st["reactants"][0], "added"], "products": [st["products"][0], "added"],
                           "barrier": {"electronic": barrier, "spread": 1.5}, "spread": 1.5, "status": status,
                           "layer": "given", "level": "a test", "source": "made up", **more}]}


# ---------------------------------------------------------------- the query
def test_a_question_is_kept_and_found_again(mgr):
    qid = ask(mgr)
    fp = addpage.folder(mgr.ws, qid) / "query.json"
    stamp = fp.stat().st_mtime_ns
    assert ask(mgr) == qid and fp.stat().st_mtime_ns == stamp           # asked again: the same file
    assert ask(mgr, temperature=1500.0, time_s=1e-8) != qid             # other conditions: another question
    v = addpage.view(mgr, qid)
    assert v["id"] == qid and v["results"] == [] and v["energies"] == "electronic"
    assert v["target"]["names"] and not v["target"]["resolved"]
    assert [m["amount"] for m in v["conditions"]["initial"]] == [1.0] and v["conditions"]["initial"][0]["name"]
    # none forms at 300 K: the easiest route, each step with its height and its name, against the cutoff
    lim = v["limits"]
    assert not lim["defined"] and "none of the target" in lim["why"] and lim["cutoff_kcal"] > 0
    assert lim["routes"][0]["steps"][0]["height"] == pytest.approx((E_TS - E_HCN) * KCAL)
    assert lim["order"][0]["label"] and lim["order"][0]["edge"] and lim["order"][0]["status"] == "verified"
    assert "coordinates" not in json.dumps(v) and "series" not in json.dumps(v)     # the page reads no geometry here
    assert [x["id"] for x in addpage.queries(mgr.ws)][0] in (qid, ask(mgr, temperature=1500.0, time_s=1e-8))
    assert len(addpage.queries(mgr.ws)) == 2
    with pytest.raises(WorkspaceError, match="no query"):
        addpage.view(mgr, "q_nope")
    with pytest.raises(WorkspaceError):
        addpage.folder(mgr.ws, "../x")


def test_where_the_target_forms_the_sensitivity_order_is_shown(mgr):
    v = addpage.view(mgr, ask(mgr, temperature=1500.0, time_s=1e-8))
    lim = v["limits"]
    assert v["target"]["resolved"] and lim["defined"] and lim["order"][0]["role"] == "lower"
    assert lim["order"][0]["control"] > 0.5 and lim["order"][0]["barrier"] == pytest.approx((E_TS - E_HCN) * KCAL)
    assert lim["traps"] == []          # (the starting material and the target are not listed as traps)


# ---------------------------------------------------------------- candidates
def test_candidates_from_lines_a_file_or_a_structure():
    out = addpage.parse_candidates("OC(=O)[C@@H]1CCCN1 L-proline 0.2\n# a comment\nCC(=O)O\nO water\n", 0.1)
    assert [(c["name"], c["amount"]) for c in out] == [("L-proline", 0.2), ("CC(=O)O", 0.1), ("water", 0.1)]
    assert out[0]["formula"] == "C5H9NO2" and out[0]["natoms"] == 17 and not any(c.get("error") for c in out)
    # what cannot be used says why, and the rest is still read
    bad = addpage.parse_candidates("not-a-smiles x\nO a\nN a\nC", None)
    assert [c.get("error") for c in bad] == ["this SMILES cannot be read", "no amount (mol/L)", "no amount (mol/L)",
                                             "no amount (mol/L)"]
    assert addpage.parse_candidates("O a\nN a", 1.0)[1]["error"] == "this name is used twice"
    # a candidates file, as a model writes it: poses and estimates ride along
    text = json.dumps([{"name": "m1", "smiles": "O", "amount": 0.05, "pose": {"step": "s", "frame": "ts", "coordinates": []},
                        "estimates": {"new_steps": []}}, {"smiles": "N"}])
    out = addpage.parse_candidates(text, 0.3)
    assert out[0]["pose"]["step"] == "s" and out[0]["amount"] == 0.05 and out[1]["name"] == "N" and out[1]["amount"] == 0.3
    # an xyz file: every frame a candidate, named after the file
    xyz = "3\ncharge=-1\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
    out = addpage.parse_candidates(xyz + xyz, 0.1, "cat.xyz")
    assert [c["name"] for c in out] == ["cat_1", "cat_2"] and out[0]["atoms"] == ["O", "H", "H"] and out[0]["charge"] == -1
    assert out[0]["natoms"] == 3 and len(out[0]["coordinates"]) == 3
    with pytest.raises(WorkspaceError, match="candidates file"):
        addpage.parse_candidates("[1, 2]", 0.1)
    with pytest.raises(WorkspaceError, match="not a candidates file"):
        addpage.parse_candidates("{oops", 0.1)


def test_one_job_tests_the_candidates_of_one_click(mgr):
    qid = ask(mgr)
    cands = addpage.parse_candidates("O water\nN ammonia", 0.1)
    job = addpage.start(mgr, qid, cands, steps=3, workers=2)
    base = addpage.folder(mgr.ws, qid)
    argv = job["argv"]
    assert argv[:3] == ["addmol", "test", str(base / "query.json")] and argv[-2:] == ["--output", str(base / "results")]
    assert argv[argv.index("--workers") + 1] == "2" and argv[argv.index("--steps") + 1] == "3"
    sent = json.loads(open(argv[argv.index("--candidates") + 1]).read())
    assert [c["name"] for c in sent] == ["water", "ammonia"] and "formula" not in sent[0]      # kept with the job
    assert not (base / addpage.PENDING).exists() and job["op"] == "addmol-test" and job["status"] == "queued"
    assert addpage.running(mgr, qid) == {"water.json": job["id"], "ammonia.json": job["id"]}
    # listed before the job has written anything, and not startable twice
    rows = addpage.view(mgr, qid)["results"]
    assert [(r["name"], r["status"], r["readout"]) for r in rows] == [("water", "queued", False), ("ammonia", "queued", False)]
    with pytest.raises(WorkspaceError, match="still being tested"):
        addpage.start(mgr, qid, cands)
    with pytest.raises(WorkspaceError, match="cannot be read"):
        addpage.start(mgr, qid, addpage.parse_candidates("xx(", 0.1))
    with pytest.raises(WorkspaceError, match="no candidate"):
        addpage.start(mgr, qid, [])


# ---------------------------------------------------------------- results
def test_candidates_are_ranked_and_each_line_says_what_it_is_built_on(mgr):
    qid = ask(mgr)
    q, relay_file = put_result(mgr, qid, "relay", None)
    put_result(mgr, qid, "relay", relay(q))
    put_result(mgr, qid, "better", relay(q, barrier=20.0, status="verified"))
    put_result(mgr, qid, "nothing")
    put_result(mgr, qid, "broken", {"new_steps": [{"id": "x", "reactants": ["nowhere"], "products": ["added"],
                                                   "barrier": {"electronic": 1.0}, "status": "est"}]})
    put_result(mgr, qid, "half", relay(q, barrier=27.0), status="running",
               layers=[{"name": "bystander", "status": "running", "done": 1, "total": 3}])
    rows = addpage.view(mgr, qid)["results"]
    assert [r["name"] for r in rows] == ["better", "relay", "half", "nothing", "broken"]
    better, rel, half, nothing, broken = rows
    # (HNC lies too high to hold a resolved amount at 300 K: ranked by how far the easiest route came down)
    assert better["target"]["factor"] is None and better["effect"] > rel["effect"] > half["effect"] > 0
    assert better["used"] == {"est": 0, "verified": 1} and rel["used"] == {"est": 1, "verified": 0}
    assert rel["best"]["change"] < -10 and rel["best"]["spread"] == 1.5 and rel["best"]["status"] == "est"
    assert rel["through"] == 1 and rel["steps"][0]["step"] == q["steps"][0]["id"] and rel["on_route"] == pytest.approx(1.0)
    assert rel["peak"]["with"] < rel["peak"]["parent"] and rel["returns"] > 0
    # a molecule that does nothing is a result; one whose kinetics cannot be built says why
    assert nothing["readout"] and nothing["through"] == 0 and nothing["best"] is None and nothing["fate"] == "no part"
    assert nothing["used"] == {"est": 0, "verified": 0} and nothing["effect"] == 0.0
    assert not broken["readout"] and "unknown species nowhere" in broken["why"] and broken["effect"] is None
    # a file that says running with no job writing it was cut short
    assert half["status"] == "interrupted" and half["layers"][0]["done"] == 1
    assert all(r["variant"] is None and r["job"] is None for r in rows)
    cut = addpage.detail(mgr, qid, half["file"])
    assert cut["status"] == "interrupted" and cut["layers"][0]["status"] == "stopped early" and cut["job"] is None
    # where the target forms, a gain is a factor, and any factor ranks above a route that only came down
    hot = ask(mgr, temperature=1500.0, time_s=1e-8)
    qh, _ = put_result(mgr, hot, "relay", None)
    put_result(mgr, hot, "relay", relay(qh, barrier=20.0))
    (row,) = addpage.view(mgr, hot)["results"]
    assert row["target"]["factor"] > 1.5 and row["effect"] > 1000


def test_one_candidate_in_detail(mgr):
    qid = ask(mgr)
    q, _ = put_result(mgr, qid, "x")
    st = q["steps"][0]
    beside = {"steps": [{"step": st["id"], "status": "est", "layer": "bystander", "level": "lvl",
                         "binding": {k: {"electronic": v, "spread": 0.7} for k, v in (("r", -1.0), ("ts", -30.0), ("p", -1.0))}}]}
    _, name = put_result(mgr, qid, "beside", beside, failed=[{"step": st["id"], "layer": "bystander", "status": "unmoved",
                                                               "why": "the molecule did not move"}])
    d = addpage.detail(mgr, qid, name)
    (row,) = d["through_steps"]
    assert row["through"] == f"{st['id']}+added" and row["layer"] == "bystander" and row["level"] == "lvl"
    assert row["change"] == pytest.approx(row["after"] - row["before"]) and row["spread"] == 0.7 and row["status"] == "est"
    assert row["geometry"] and not row["with_geometry"] and row["edge"] == st["edge"]
    assert d["moved_steps"] == [] and d["unchanged"] == 0 and d["new_steps"] == []
    assert d["failed"] == [{"layer": "bystander", "status": "unmoved", "why": "the molecule did not move",
                            "step": st["id"], "species": None, "label": st["label"]}]
    assert "binding" not in json.dumps(d["through_steps"]) and "coordinates" not in d["candidate"]
    # a step the molecule takes part in: listed as new, and the parent step reads through it
    _, name = put_result(mgr, qid, "relay", relay(q))
    d = addpage.detail(mgr, qid, name)
    assert d["through_steps"][0]["via"] == "HCN -> HNC through the molecule" and d["through_steps"][0]["layer"] == "given"
    assert d["new_steps"][0]["barrier"]["electronic"] == 25.0 and d["new_steps"][0]["source"] == "made up"
    assert d["peak_routes"]["with"][0]["peak"] < d["peak_routes"]["parent"][0]["peak"]
    # a result whose kinetics could not be built still reads (and is plain JSON)
    _, name = put_result(mgr, qid, "broken", {"new_steps": [{"id": "x", "reactants": ["nowhere"], "products": ["added"],
                                                             "barrier": {"electronic": 1.0}, "status": "est"}]})
    d = json.loads(json.dumps(addpage.detail(mgr, qid, name), allow_nan=False))
    assert not d["readout"] and d["effect"] is None and "nowhere" in d["failed"][0]["why"]
    addpage.remove(mgr, qid, name)
    # nothing at all
    d = addpage.detail(mgr, qid, "x.json")
    assert d["through_steps"] == [] and d["unchanged"] == 1 and d["readout"]
    with pytest.raises(WorkspaceError, match="no result"):
        addpage.detail(mgr, qid, "../query.json")
    addpage.remove(mgr, qid, "x.json")
    assert [r["name"] for r in addpage.view(mgr, qid)["results"]] == ["beside", "relay"]


def test_the_geometry_shows_the_molecule_only_where_the_estimate_kept_it(mgr):
    qid = ask(mgr)
    q, _ = put_result(mgr, qid, "x")
    st = q["steps"][0]
    g = addpage.geometry(mgr.ws, qid, st["id"])
    assert len(g["frames"]) == 3 and g["ts"] == 1 and not g["with_molecule"] and g["frames"][1].startswith("3\n")
    water = [[5.0, 0.0, 0.0], [5.96, 0.0, 0.0], [4.76, 0.93, 0.0]]
    beside = {"steps": [{"step": st["id"], "status": "est", "layer": "bystander", "molecule_symbols": ["O", "H", "H"],
                         "binding": {k: {"electronic": v} for k, v in (("r", -1.0), ("ts", -30.0), ("p", -1.0))},
                         "complexes": {k: [list(x) for x in FRAMES[f]] + water
                                       for k, f in (("r", "hcn"), ("ts", "ts"), ("p", "hnc"))}}]}
    _, name = put_result(mgr, qid, "beside", beside)
    with_c = addpage.geometry(mgr.ws, qid, st["id"], name, f"{st['id']}+added")
    assert with_c["with_molecule"] and with_c["frames"][1].startswith("6\n") and "with the molecule" in with_c["frames"][1]
    # the same step read through another estimate (a relay that came with no geometry): the parent's own frames
    other = addpage.geometry(mgr.ws, qid, st["id"], name, "relay")
    assert not other["with_molecule"] and other["frames"][1].startswith("3\n")
    assert addpage.detail(mgr, qid, name)["through_steps"][0]["with_geometry"]
    with pytest.raises(WorkspaceError, match="no step"):
        addpage.geometry(mgr.ws, qid, "nope")


# ---------------------------------------------------------------- the variant and the map
def test_a_kept_candidate_is_a_variant_the_map_can_mark(mgr):
    qid = ask(mgr)
    q, _ = put_result(mgr, qid, "x")
    st = q["steps"][0]
    est = {**relay(q), "species": [{"species": st["reactants"][0], "status": "verified", "binding": {"electronic": -3.0}}],
           "new_species": [{"id": "adduct", "name": "an adduct", "formed_from": [st["reactants"][0], "added"],
                            "energy": {"electronic": 5.0}, "status": "est"}]}
    _, name = put_result(mgr, qid, "cat", est, status="running")
    rec = addpage.keep(mgr.ws, qid, name)
    assert rec["kind"] == "added_molecule" and rec["addmol"] == {"query": qid, "result": name}
    assert rec["edit"] == ["0.1 M of cat added"] and rec["replaced"] == []       # (a list: Explore's picker joins it)
    m = rec["marks"]
    assert m["edges"] == {st["edge"]: {"change": pytest.approx(m["edges"][st["edge"]]["change"]), "status": "est",
                                       "spread": 1.5}} and m["edges"][st["edge"]]["change"] < -10
    assert list(m["species"]) == [st["reactants"][0]] and m["species"][st["reactants"][0]]["status"] == "verified"
    assert m["new"] == [{"name": "an adduct", "kind": "species", "status": "est"}] and m["status"] == "running"
    assert addpage.view(mgr, qid)["results"][0]["variant"] == rec["id"]
    # the kinetics of the record are those of the result file
    net = added.variant_network(mgr.ws, mgr.jobs, q["level"]["key"], 300.0, rec["id"], free_energy=False)
    assert any(s["id"] == "relay" for s in net["steps"])
    # its job finished with a verified step: the record follows the file, under the same id
    put_result(mgr, qid, "cat", {**est, **relay(q, status="verified")})
    assert addpage.after_job(mgr, {"params": {"query": qid, "names": [name]}})
    again = mgr.ws.variants[rec["id"]]
    assert len(mgr.ws.variants) == 1 and again["marks"]["status"] == "done"
    assert again["marks"]["edges"][st["edge"]]["status"] == "verified"
    assert not addpage.after_job(mgr, {"params": {"query": qid, "names": ["other.json"]}})
    # a molecule that changes nothing marks nothing
    doc = json.loads((addpage.folder(mgr.ws, qid) / "results" / "x.json").read_text())
    assert addpage.marks(q, doc, addpage._edges(mgr.ws, q), addpage._nodes(mgr.ws, q))["edges"] == {}


# ---------------------------------------------------------------- the endpoints
def test_the_page_endpoints(mgr, tmp_path):
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    ids = {r["smiles"]: sid for sid, r in mgr.ws.snapshot()["structures"].items()}
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        assert c.get("/api/addmol").json() == {"queries": []}
        r = c.post("/api/addmol/query", json={"target": ids["[C-]#[NH+]"], "initial": {ids["C#N"]: 1.0},
                                              "temperature": 300.0, "time_s": 3600.0, "free_energy": False})
        assert r.status_code == 200, r.text
        qid = r.json()["id"]
        assert r.json()["limits"]["routes"] and [q["id"] for q in c.get("/api/addmol").json()["queries"]] == [qid]
        f = c.get(f"/api/addmol/{qid}/query.json")
        assert f.json()["schema"] == "mepd-addmol-query" and "attachment" in f.headers["content-disposition"]
        assert c.get("/api/addmol/q_nope/query.json").status_code == 404 and c.get("/api/addmol/q_nope").status_code == 400
        # reading candidates starts nothing
        r = c.post(f"/api/addmol/{qid}/test", json={"text": "O water\nxx( bad", "amount": 0.1, "dry_run": True})
        assert [(x["name"], x.get("error")) for x in r.json()["candidates"]] == [("water", None), ("bad", "this SMILES cannot be read")]
        assert c.get(f"/api/addmol/{qid}").json()["results"] == []
        assert c.post(f"/api/addmol/{qid}/test", json={"text": "xx( bad", "amount": 0.1}).status_code == 400
        q, name = put_result(mgr, qid, "cat", None)
        put_result(mgr, qid, "cat", relay(q))
        assert c.get(f"/api/addmol/{qid}").json()["results"][0]["best"]["status"] == "est"
        assert c.get(f"/api/addmol/{qid}/results/{name}").json()["through_steps"][0]["through"] == "relay"
        g = c.get(f"/api/addmol/{qid}/geometry", params={"step": q["steps"][0]["id"], "result": name, "through": "relay"})
        assert len(g.json()["frames"]) == 3 and not g.json()["with_molecule"]
        vid = c.post(f"/api/addmol/{qid}/results/{name}/keep").json()["variant"]
        assert c.get("/api/state").json()["workspace"]["variants"][vid]["marks"]["edges"]
        assert c.delete(f"/api/addmol/{qid}/results/{name}").json() == {"ok": True}
        assert c.get(f"/api/addmol/{qid}/results/{name}").status_code == 400


# ---------------------------------------------------------------- an exploration's own network
def test_a_workspace_that_is_one_exploration_is_asked_through_its_run(mgr):
    """(decision of team41: the parent of an explorer run is its own molecule-level network, not its adoption)"""
    pytest.importorskip("mepd.discovery.parent")
    job, run = addpage.source_run(mgr.ws, mgr.jobs)
    assert job["op"] == "graph-enumeration" and (run / "summary.json").exists()
    # its steps are named by the run's labels and its species by SMILES: the map still finds their edge and nodes
    snap = mgr.ws.snapshot()
    (eid,) = snap["edges"]
    ids = {r["smiles"]: sid for sid, r in snap["structures"].items()}
    q = {"asked": {"job": job["id"]}, "steps": [{"id": "fast_0_1", "run_step": "fast_0_1"}, {"id": "other", "run_step": "x"}],
         "species": [{"id": "C#N", "smiles": "C#N"}, {"id": "CC", "smiles": "CC"}]}
    assert addpage._edges(mgr.ws, q) == {"fast_0_1": eid}
    assert addpage._nodes(mgr.ws, q) == {"C#N": [ids["C#N"]], "CC": []}
    # an exploration that is still running is not asked this way
    assert addpage.source_run(mgr.ws, {job["id"]: {**job, "status": "running"}}) is None
    assert addpage.source_run(mgr.ws, {}) is None
    # a question with a constant feed, or on electronic energies, is asked of the workspace's network
    qid = ask(mgr)
    assert addpage.view(mgr, qid)["source"] == "workspace" and addpage.view(mgr, qid)["asked"]["job"] is None
