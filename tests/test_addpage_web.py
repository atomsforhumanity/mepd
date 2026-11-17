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
    job = addpage.start(mgr, qid, cands, steps=3, workers=2, side=True)
    base = addpage.folder(mgr.ws, qid)
    argv = job["argv"]
    assert argv[:3] == ["addmol", "test", str(base / "query.json")] and argv[-2:] == ["--output", str(base / "results")]
    assert argv[argv.index("--workers") + 1] == "2" and argv[argv.index("--steps") + 1] == "3"
    # the scans: an upper bound of 25 reactions by default; side reactions only when asked for
    assert argv[argv.index("--scan") + 1] == "25" and "--scan-side-reactions" in argv
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


# ---------------------------------------------------------------- a test as it runs: stages, kinds of rows, scans
def staged(q, tmp_path, *, relay_status="searching", with_file=True):
    """A result file as `mepd addmol test` writes it mid-scan: one step verified with the molecule beside it,
    one TS with it taking part (a scan's), the scan still searching, one step left unscanned."""
    st = q["steps"][0]
    ts = tmp_path / "ts.xyz"
    if with_file:
        ts.write_text("4\nts\n" + "".join(f"{a} {k} 0 0\n" for k, a in enumerate("HCNO")))
        (tmp_path / "irc.xyz").write_text(("4\nend\n" + "".join(f"{a} {k} 0 0\n" for k, a in enumerate("HCNO"))) * 3)
    row = {"parent_step": st["id"], "status": "verified", "reactants": [st["reactants"][0], "added"],
           "products": [st["products"][0], "added"], "level": "lvl", "spread": 0.4, "poses_verified": 2}
    est = {"new_steps": [
        {**row, "id": f"{st['id']}|verified pose 0", "layer": "verify", "takes_part": False,
         "barrier": {"electronic": 40.0, "spread": 0.4}, "ts_file": str(tmp_path / "gone.xyz")},
        {**row, "id": f"{st['id']}|ts_pair_1", "layer": "relay", "takes_part": True, "returned": True,
         "barrier": {"electronic": 22.0}, "spread": None, "ts_file": str(ts), "irc_file": str(tmp_path / "irc.xyz")}]}
    layers = [{"name": "bystander", "status": "done", "done": 1, "total": 1}, {"name": "verify", "status": "done", "done": 1, "total": 1},
              {"name": "relay", "status": "running", "done": 0, "total": 1}]
    extra = {"level_labels": {"lvl": "a level in words"},
             "summary": {"steps": 1, "est": 0, "verified_beside": 0, "takes_part": 1, "leads_elsewhere": 0, "failed": 0,
                         "reacted_in_the_estimate": 0, "not_reached": 0, "what": "..."},
             "timings": {"first_bystander_s": 3.0, "first_verified_s": 20.0, "first_relay_s": 90.0},
             "scan": {"rule": "the limiting steps first", "not_scanned": "0 other steps",
                      "steps": [{"step": st["id"], "status": relay_status, "why": "limits the target",
                                 "searches": {"searched": 3, "total": 9}, "with_molecule": 1, "lowest": {"with_molecule": 22.0},
                                 "elsewhere": 2, "consumed": [{"ts": "x"}], "wall_s": 61.0, "command": "long"}],
                      "left": [{"step": st["id"], "why": "reacted in the estimate; beyond --scan"},
                               {"step": "other", "why": "reacted in the estimate; beyond --scan"}]}}
    return est, layers, extra


def put_staged(mgr, qid, name, tmp_path, status="running", **kw):
    q, _ = put_result(mgr, qid, "blank")
    est, layers, extra = staged(q, tmp_path, **kw)
    _, file = put_result(mgr, qid, name, est, status=status, layers=layers)
    fp = addpage.folder(mgr.ws, qid) / "results" / file
    fp.write_text(json.dumps({**json.loads(fp.read_text()), **extra}))
    return q, file


def test_a_running_test_says_its_stage_and_what_was_not_scanned(mgr, tmp_path):
    qid = ask(mgr)
    q, file = put_staged(mgr, qid, "cat", tmp_path)
    doc = json.loads((addpage.folder(mgr.ws, qid) / "results" / file).read_text())
    p = addpage.progress(doc)
    assert p["stage"] == {"name": "relay", "done": 0, "total": 1} and p["summary"]["takes_part"] == 1 and "what" not in p["summary"]
    assert p["scans"] == {"asked": 1, "done": 0, "failed": 0, "running": [q["steps"][0]["id"]], "left": 2, "of": None}
    assert p["stages"] == [p["stage"]]
    assert p["first"] == {"bystander": 3.0, "verified": 20.0, "relay": 90.0} and p["open"]
    # finished, every scan done, nothing left: only then is the answer closed
    done = {**doc, "status": "done", "scan": {"steps": [{**doc["scan"]["steps"][0], "status": "done"}], "left": []}}
    assert not addpage.progress(done)["open"]
    assert addpage.progress({**done, "scan": doc["scan"]})["open"]                     # (steps left unscanned)
    assert addpage.progress({**done, "summary": {**doc["summary"], "est": 2}})["open"]   # (estimates only)
    assert addpage.progress({"status": "done"}) == {"stage": None, "stages": [], "summary": None, "open": False, "settled": None, "first": {
        "bystander": None, "verified": None, "relay": None}, "scans": {"asked": 0, "done": 0, "failed": 0, "running": [], "left": 0, "of": None}}
    # two stages running at once are both said; a test cut from outside stays open
    both = {**doc, "layers": [{"name": "verify", "status": "running", "done": 3, "total": 9}, {"name": "relay", "status": "running", "done": 0, "total": 2}]}
    assert [x["name"] for x in addpage.progress(both)["stages"]] == ["verify", "relay"]
    assert addpage.progress({**done, "status": "cut"})["open"]
    row = next(r for r in addpage.view(mgr, qid)["results"] if r["name"] == "cat")
    assert row["progress"]["scans"]["left"] == 2 and "spread" in row["target"]


def test_rows_say_what_kind_of_number_they_are_and_scans_are_listed(mgr, tmp_path):
    qid = ask(mgr)
    q, file = put_staged(mgr, qid, "cat", tmp_path)
    st = q["steps"][0]
    d = addpage.detail(mgr, qid, file)
    (row,) = d["through_steps"]
    # the lowest version of the step is the scan's TS with the molecule taking part; its files are there
    assert row["kind"] == "takes part" and row["layer"] == "relay" and row["through"] == f"{st['id']}|ts_pair_1"
    assert row["files"] == "there" and row["with_geometry"] and row["level"] == "a level in words" and row["rank"] == 0
    assert d["new_steps"] == []                                 # (versions of a parent step are rows of the table)
    (scan,) = d["scans"]
    assert scan["steps"] == [st["id"]] and scan["status"] == "searching" and scan["consumed"] == 1 and "command" not in scan
    assert scan["searches"] == {"searched": 3, "total": 9} and scan["lowest"] == {"with_molecule": 22.0}
    assert d["scan_left"] == [{"why": "reacted in the estimate; beyond --scan", "n": 2, "labels": [st["label"], "other"]}]
    assert d["scan_rule"] == "the limiting steps first" and d["not_scanned"] == "0 other steps"
    # a scan that stands for several parent steps (one per reaction) is read the same way
    assert addpage._scan_steps({"steps": ["a", "b"], "step": "a"}) == ["a", "b"] and addpage._scan_steps({}) == []
    # an estimate reads as one
    beside = {"steps": [{"step": st["id"], "status": "est", "layer": "bystander", "level": "lvl",
                         "binding": {k: {"electronic": v} for k, v in (("r", -1.0), ("ts", -30.0), ("p", -1.0))}}]}
    _, name = put_result(mgr, qid, "quick", beside)
    assert addpage.detail(mgr, qid, name)["through_steps"][0]["kind"] == "estimate"


def test_a_verified_ts_is_shown_from_its_own_file_or_said_to_be_gone(mgr, tmp_path):
    qid = ask(mgr)
    q, file = put_staged(mgr, qid, "cat", tmp_path)
    st = q["steps"][0]
    g = addpage.geometry(mgr.ws, qid, st["id"], file, f"{st['id']}|ts_pair_1")
    assert g["with_molecule"] and g["ends"] == "irc" and len(g["frames"]) == 3 and g["frames"][1].startswith("4\n")
    # the other row's file was lost with the machine it ran on: the parent's own TS, and the path, said plainly
    gone = addpage.geometry(mgr.ws, qid, st["id"], file, f"{st['id']}|verified pose 0")
    assert not gone["with_molecule"] and gone["missing"].endswith("gone.xyz") and gone["frames"][1].startswith("3\n")


# ---------------------------------------------------------------- a finished test replayed by its rows' arrival times
def timed(mgr, qid, tmp_path):
    """The staged result, finished, every row with the time it arrived: the estimate's failure at 5 s, the verified
    row at 20 s, the scan from 60 s with its TS at 90 s (and a higher one at 70 s that the file no longer holds)."""
    q, file = put_staged(mgr, qid, "cat", tmp_path, status="done", relay_status="done")
    fp = addpage.folder(mgr.ws, qid) / "results" / file
    doc = json.loads(fp.read_text())
    st = q["steps"][0]
    doc["estimates"]["new_steps"][0]["t_s"], doc["estimates"]["new_steps"][1]["t_s"] = 20.0, 90.0
    doc["failed"] = [{"step": st["id"], "layer": "verify", "status": "leads elsewhere", "t_s": 25.0, "why": "joins other species",
                      "ends": [{"pose": 1, "ends": ["C#N.O", "C#N.O"]}]}]
    doc["layers"] = [{"name": "bystander", "status": "done", "done": 1, "total": 1,
                      "uncertainty": {"barrier_shift_kcal": {"mean_abs_difference_to_verified": 2.0, "largest": 4.2, "same_sign": "9 of 10 steps"}}},
                     {"name": "verify", "status": "done", "done": 1, "total": 1}, {"name": "relay", "status": "done", "done": 1, "total": 1, "lanes": 2}]
    doc["scan"]["steps"][0].update(t_started_s=60.0, wall_s=40.0, steps=[st["id"], "twin"], lane=1,
                                   arrivals=[{"t_s": 70.0, "takes_part": True, "ts": "a", "free": 30.0},
                                             {"t_s": 90.0, "takes_part": True, "ts": "b", "free": 22.0}])
    doc["route_peak"] = {"parent": 50.0, "with": [{"t_s": 6.0, "peak": 50.0}, {"t_s": 20.0, "peak": 40.0}, {"t_s": 90.0, "peak": 22.0}]}
    doc["timings"] = {"first_bystander_s": 5.0, "first_verified_s": 20.0, "first_relay_s": 70.0, "total_s": 100.0}
    fp.write_text(json.dumps(doc))
    return q, file, doc


def test_a_finished_test_is_replayed_from_its_rows_arrival_times(mgr, tmp_path):
    qid = ask(mgr)
    q, file, doc = timed(mgr, qid, tmp_path)
    st = q["steps"][0]
    assert addpage.duration(doc) == 100.0 and addpage.duration({"timings": {"total_s": 9.0}}) is None
    end = addpage.detail(mgr, qid, file)
    assert end["replay"] is None and end["duration_s"] == 100.0 and end["through_steps"][0]["kind"] == "takes part"
    assert end["uncertainty"]["barrier_shift_kcal"]["largest"] == 4.2          # (read from the file, not from the page)
    assert end["route_peak"]["with"][-1] == {"t_s": 90.0, "peak": 22.0}
    # 10 s in: nothing verified yet; the verification and the scan have not begun
    d = addpage.detail(mgr, qid, file, 10.0)
    assert d["status"] == "running" and d["replay"]["t_s"] == 10.0 and not d["replay"]["over"] and d["through_steps"] == []
    assert [(x["name"], x["status"]) for x in d["layers"]] == [("bystander", "done"), ("verify", "queued"), ("relay", "queued")]
    assert d["scans"] == [] and d["elsewhere"] == [] and d["route_peak"]["with"] == [{"t_s": 6.0, "peak": 50.0}]
    assert d["progress"]["first"] == {"bystander": 5.0, "verified": None, "relay": None} and d["scan_left"] == []
    # 30 s in: the verified row beside the step is in and counted by the kinetics; its other pose led elsewhere
    d = addpage.detail(mgr, qid, file, 30.0)
    (row,) = d["through_steps"]
    assert row["kind"] == "beside" and row["through"] == f"{st['id']}|verified pose 0" and row["t_s"] == 20.0
    assert d["progress"]["summary"]["verified_beside"] == 1 and d["progress"]["summary"]["takes_part"] == 0
    (gone,) = d["elsewhere"]
    assert gone["label"] == st["label"] and gone["poses"][0]["pose"] == 1 and len(gone["poses"][0]["ends"]) == 2
    assert "the molecule" in gone["poses"][0]["ends"][0]                       # (the candidate's own SMILES, in words)
    # 75 s in: the scan is searching; its first TS has arrived (30.0), the lower one has not; verify is done
    d = addpage.detail(mgr, qid, file, 75.0)
    (scan,) = d["scans"]
    assert scan["status"] == "searching" and scan["with_molecule"] == 1 and scan["lowest"] == {"with_molecule": 30.0}
    assert scan["steps"] == [st["id"], "twin"] and scan["lane"] == 1 and scan["wall_s"] == 15.0 and scan["searches"] is None
    assert [(x["name"], x["status"]) for x in d["layers"]][1:] == [("verify", "done"), ("relay", "running")]
    assert d["through_steps"][0]["kind"] == "beside"       # (the 30.0 row was replaced in the file: the replay cannot show it)
    assert d["peak"]["with"] > addpage.detail(mgr, qid, file, 95.0)["peak"]["with"]
    # at the end the replay is the file
    last = addpage.detail(mgr, qid, file, 100.0)
    assert last["replay"]["over"] and last["status"] == "done" and last["through_steps"][0]["through"] == end["through_steps"][0]["through"]
    assert last["scans"][0]["status"] == "done" and last["scan_left"] == end["scan_left"]


def test_a_rows_files_are_read_beside_the_result_or_from_its_archive(mgr, tmp_path):
    import tarfile

    qid = ask(mgr)
    q, file, doc = timed(mgr, qid, tmp_path)
    st = q["steps"][0]
    fp = addpage.folder(mgr.ws, qid) / "results" / file
    # a pod run: paths relative to the result file, the files in files.tgz beside it
    doc["estimates"]["new_steps"][1].update(ts_file="scan/ts.xyz", irc_file="scan/irc.xyz")
    fp.write_text(json.dumps(doc))
    (tmp_path / "scan").mkdir()
    for n in ("ts.xyz", "irc.xyz"):
        (tmp_path / "scan" / n).write_text((tmp_path / n).read_text())
    with tarfile.open(fp.parent / "files.tgz", "w:gz") as tar:
        tar.add(tmp_path / "scan" / "ts.xyz", arcname="./scan/ts.xyz")
        tar.add(tmp_path / "scan" / "irc.xyz", arcname="./scan/irc.xyz")
    assert addpage.detail(mgr, qid, file)["through_steps"][0]["files"] == "there"
    g = addpage.geometry(mgr.ws, qid, st["id"], file, f"{st['id']}|ts_pair_1")
    assert g["with_molecule"] and len(g["frames"]) == 3 and (fp.parent / "scan" / "ts.xyz").is_file()   # (taken out once)
    # a relative path with nothing beside it is gone, and said so
    doc["estimates"]["new_steps"][1].update(ts_file="nowhere/ts.xyz")
    fp.write_text(json.dumps(doc))
    assert addpage.geometry(mgr.ws, qid, st["id"], file, f"{st['id']}|ts_pair_1")["missing"] == "nowhere/ts.xyz"



# ---------------------------------------------------------------- settled, side reactions, the control
def test_settled_side_reactions_and_takes_part_above_beside(mgr, tmp_path):
    qid = ask(mgr)
    q, file, doc = timed(mgr, qid, tmp_path)
    st = q["steps"][0]
    fp = addpage.folder(mgr.ws, qid) / "results" / file
    a, b = st["reactants"][0], st["products"][0]
    doc["target_route_settled"] = {"t_s": 80.0, "route_peak": 22.0, "in_words": "no reaction is left that could lower it"}
    doc["scan"]["ended"] = {"t_s": 80.5, "why": "no reaction is left that could lower the target's route"}
    doc["scan"]["side_reactions"] = {"asked_for": False, "ask_with": "--scan-side-reactions", "evaluated_t_s": 95.0,
                                     "listed": [{"step": st["id"], "reaction": f"{a} -> {b}", "steps": [st["id"]], "species": a,
                                                 "change": 0.5, "in_words": "could divert material; not scanned"}]}
    doc["scan"]["steps"][0]["lowest"] = {"with_molecule": 30.0, "beside": 25.0}
    doc["route_peak"]["with"][-1]["readout"] = {"target": {"final": 0.01, "low": 0.005, "high": 0.02, "time_to_fraction_s": 9.0},
                                                "fate": "regenerated", "route_role": "takes part", "in_words": "takes part", "turnovers": 3.0}
    doc["scope"] = {"level": "lvl", "phase": "gas", "does_not_hold": "routes through species not in the network"}
    fp.write_text(json.dumps(doc))
    end = addpage.detail(mgr, qid, file)
    names = addpage._names(q, mgr.ws)
    assert end["settled"]["t_s"] == 80.0 and end["progress"]["settled"]["route_peak"] == 22.0 and end["scope"]["phase"] == "gas"
    (side,) = end["side_reactions"]["listed"]
    assert side["label"] == f"{names[a]} -> {names[b]}" and side["species_name"] == names[a] and not end["side_reactions"]["asked_for"]
    assert end["takes_part_above"] == [{"reaction": None, "takes_part": 30.0, "beside": 25.0}]
    # replayed: before the moment it settled there is no "settled"; the side reactions come a little later
    assert addpage.detail(mgr, qid, file, 70.0)["settled"] is None and addpage.detail(mgr, qid, file, 70.0)["side_reactions"] is None
    at85 = addpage.detail(mgr, qid, file, 85.0)
    assert at85["settled"]["t_s"] == 80.0 and at85["scan_ended"]["t_s"] == 80.5 and at85["side_reactions"] is None
    assert addpage.detail(mgr, qid, file, 96.0)["side_reactions"]["listed"]
    # the answer at 95 s is the run's own read-out at its last peak event (90 s), not the kinetics run again
    at95 = addpage.detail(mgr, qid, file, 95.0)
    assert at95["target"]["with"]["final"] == 0.01 and at95["target"]["spread"]["with_low"] == 0.005 and at95["fate"] == "regenerated"


def test_a_step_that_makes_a_jointly_fixed_species_is_marked(mgr, tmp_path):
    qid = ask(mgr)
    q, file, doc = timed(mgr, qid, tmp_path)
    st = q["steps"][0]
    qf = addpage.folder(mgr.ws, qid) / "query.json"
    qq = json.loads(qf.read_text())
    for sp in qq["species"]:
        if sp["id"] == st["products"][0]:
            sp["g_fixed_only_together"] = True
    qq["notes"] = qq.get("notes", []) + ["1 species are never seen apart in the boxes of the fit (g_fixed_only_together on each)"]
    qf.write_text(json.dumps(qq))
    d = addpage.detail(mgr, qid, file)
    assert d["through_steps"][0]["fixed_together"] == [addpage._names(qq, mgr.ws)[st["products"][0]]]
    assert "g_fixed_only_together" in d["fixed_together_note"]


def test_each_result_may_bring_its_own_archive(mgr, tmp_path):
    import tarfile

    qid = ask(mgr)
    q, file, doc = timed(mgr, qid, tmp_path)
    st = q["steps"][0]
    fp = addpage.folder(mgr.ws, qid) / "results" / file
    doc["estimates"]["new_steps"][1].update(ts_file="own/ts.xyz", irc_file="own/irc.xyz")
    fp.write_text(json.dumps(doc))
    with tarfile.open(fp.with_name(fp.stem + ".files.tgz"), "w:gz") as tar:
        tar.add(tmp_path / "ts.xyz", arcname="./own/ts.xyz")
        tar.add(tmp_path / "irc.xyz", arcname="./own/irc.xyz")
    assert addpage.geometry(mgr.ws, qid, st["id"], file, f"{st['id']}|ts_pair_1")["with_molecule"]



def test_steered_rows_reactions_read_as_one_and_the_flag_list(mgr, tmp_path):
    """(core round 5-6: `fixed_only_together` on the query, `reaction_steps` on read-out rows, `+steer` levels)"""
    qid = ask(mgr)
    q, file, doc = timed(mgr, qid, tmp_path)
    st = q["steps"][0]
    qf = addpage.folder(mgr.ws, qid) / "query.json"
    qq = json.loads(qf.read_text())
    qq["fixed_only_together"] = [st["products"][0]]
    qq["reference"] = {"standard_state": "1atm"}
    qf.write_text(json.dumps(qq))
    fp = addpage.folder(mgr.ws, qid) / "results" / file
    rows = doc["estimates"]["new_steps"]
    rows[1].update(level="lvl+steer", steer={"shift_kcal": -1.5})
    rows.append({**rows[0], "id": "x|left", "steer": {"not_steered": "no single point for the complex"}})
    fp.write_text(json.dumps(doc))
    d = addpage.detail(mgr, qid, file)
    (row,) = d["through_steps"]
    assert row["steered"] and row["steer_shift"] == -1.5 and row["fixed_together"]
    assert d["not_steered"] == [{"id": "x|left", "label": st["label"], "why": "no single point for the complex"}]
    assert addpage.view(mgr, qid)["standard_state"] == "1atm"
