"""A workspace's results as one browsable HTML page and one CSV for bulk loading."""
import csv
import io
import json
import re

import pytest
from fastapi.testclient import TestClient

from mepd.web import workspace_export as wx
from mepd.web.app import create_app
from mepd.web.reaction_export import edge_reaction

WATER = "3\nwater\nO 0.000 0.000 0.000\nH 0.758 0.000 0.504\nH -0.758 0.000 0.504\n"
BENT = "3\nbent\nO 0.000 0.000 0.000\nH 2.600 0.000 0.450\nH -0.758 0.000 0.504\n"


@pytest.fixture(autouse=True)
def _isolated_recent_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))


def _species(sid, role, name, smiles, e, count=1):
    return {"id": sid, "role": role, "name": name, "smiles": smiles, "charge": 0, "multiplicity": 1,
            "energy_hartree": e, "level": "GFN2-xTB", "xyz": WATER, "svg": "<svg/>", "count": count, "shuttle": False}


def _bundle(rid="r_1", channels=True) -> dict:
    """As reaction_bundle returns it: 2 water -> two products, a direct channel and a 2-step one."""
    irc = lambda e_ts: {"ts_index": 1, "frames": [WATER, BENT, WATER], "energy_kcal": [0.0, 12.0, -3.0],
                        "energy_hartree": [-5.47, e_ts, -5.48]}
    step = lambda e_ts, b: {"label": "s", "ts_xyz": BENT, "ts_hartree": e_ts, "barrier_search_kcal": b, "irc": irc(e_ts)}
    common = {"verified": True, "level": "GFN2-xTB", "level_key": "k", "job": "j_1", "job_title": "TS",
              "placed": "absolute"}
    chans = [
        {**common, "kind": "direct", "label": "c0", "steps": [step(-5.45, 12.0)], "ts_kcal": 5.7, "barrier_kcal": 12.0,
         "barrier_search_kcal": 12.0, "n": 1, "id": "ch1", "name": "Channel 1"},
        {**common, "kind": "multi-step", "label": "alt", "steps": [step(-5.44, 18.0), step(-5.43, 20.0)],
         "ts_kcal": 13.7, "barrier_kcal": 20.0, "barrier_search_kcal": 20.0, "n": 2, "id": "ch2", "name": "Channel 2"}]
    return {
        "id": rid, "label": "2 O -> [OH] + [OH2+]",
        "reactants": [_species("s1", "reactant", "water", "O", -5.07, count=2)],
        "products": [_species("s3", "product", "OH", "[OH]", -4.41), _species("s4", "product", "H3O", "[OH3]", -5.0)],
        "complexes": [], "channels": chans if channels else [], "offtarget": 1, "lowest": "ch1" if channels else None,
        "ladder": {"reactants": 0.0, "reactant_complex": None, "product_complex": None, "products": 4.0},
        "delta_e_kcal": None, "complex_delta_e_kcal": None, "relaxed_from": None, "source": "composed by hand",
        "seen": None, "profiles": [], "exported": "2026-10-13", "mepd_version": "test"}


def _workspace(*bundles):
    return {"name": "demo ws", "reactions": list(bundles), "exported": "2026-10-13 10:00", "mepd_version": "test"}


def test_the_csv_has_one_row_per_step_in_fixed_columns_with_geometries_inline():
    rows = list(csv.DictReader(io.StringIO(wx.export_csv(_workspace(_bundle(), _bundle("r_2", channels=False))))))
    assert list(rows[0]) == wx.CSV_COLUMNS
    # Channel 1 (one step), channel 2 (two steps), and one row for the reaction nothing was found for.
    assert [(r["reaction_id"], r["channel_rank"], r["step_index"]) for r in rows] == \
        [("r_1", "1", "0"), ("r_1", "2", "0"), ("r_1", "2", "1"), ("r_2", "", "")]
    first, step2, empty = rows[0], rows[2], rows[3]
    assert first["reaction_smiles"] == "O.O>>[OH].[OH3]" and first["n_reactant_molecules"] == "2"
    assert first["channel_is_lowest"] == "1" and first["channel_irc_verified"] == "1" and first["status"] == "channels"
    assert first["channel_barrier_kcal_mol"] == "12.0000" and first["step_ts_energy_hartree"] == "-5.4500000000"
    # Step barriers from its own IRC ends; reaction energy from the ladder when the reaction has none.
    assert float(first["step_forward_barrier_kcal_mol"]) == pytest.approx(0.02 * wx.KCAL, abs=1e-3)
    assert float(first["step_reverse_barrier_kcal_mol"]) == pytest.approx(0.03 * wx.KCAL, abs=1e-3)
    assert first["reaction_energy_kcal_mol"] == "4.0000" and first["reactants_energy_hartree"] == "-10.1400000000"
    # The TS and both step ends in one atom order: elements once, 3 numbers per atom each.
    assert first["elements"] == "O H H"
    for col in ("ts_coords_angstrom", "step_start_coords_angstrom", "step_end_coords_angstrom"):
        assert len(first[col].split()) == 9
    assert first["ts_coords_angstrom"].split()[3] == "2.600"
    assert step2["step_is_highest"] == "1" and rows[1]["step_is_highest"] == "0"
    assert empty["status"] == "not_searched" and empty["n_channels"] == "0" and empty["ts_coords_angstrom"] == ""
    # Two molecules: their charges add, the multiplicity is not one molecule's.
    assert first["charge"] == "0" and first["multiplicity"] == ""


def test_the_page_is_self_contained_with_one_viewer_copy_and_every_reaction_page():
    page = wx.export_html(_workspace(_bundle(), _bundle("r_2", channels=False)))
    assert not re.search(r'(src|href)="https?://', page)
    assert page.count("$3Dmol") >= 1 and page.count('id="three"') == 1
    data = json.loads(page.split('<script type="application/json" id="data">')[1].split("</script>")[0])
    assert [r["id"] for r in data["rows"]] == ["r_1", "r_2"] and set(data["pages"]) == {"r_1", "r_2"}
    # Each reaction page has a slot where the one copy of 3Dmol goes, not a copy of its own.
    assert all(data["slot"] in p and "$3Dmol.createViewer" not in p.split("<script>")[1] for p in data["pages"].values())
    assert data["rows"][0]["barrier"] == 12.0 and data["rows"][1]["status"] == "not_searched"
    assert data["csv"].startswith(",".join(wx.CSV_COLUMNS))


def test_what_is_exported_is_what_analyze_lists():
    snap = {
        "structures": {"a": {"role": None, "name": "A"}, "b": {"role": None, "name": "B"},
                       "c": {"role": "complex"}, "t": {"role": "ts"}},
        "reactions": {"r_1": {"id": "r_1", "edge": "e_c", "created": 1}},
        "edges": {"e_ab": {"id": "e_ab", "source": "a", "target": "b", "created": 2},     # searched
                  "e_ba": {"id": "e_ba", "source": "b", "target": "a", "created": 3},     # never searched
                  "e_c": {"id": "e_c", "source": "c", "target": "b", "created": 4},       # a reaction stands for it
                  "e_t": {"id": "e_t", "source": "a", "target": "t", "created": 5}}}      # a TS end
    jobs = {"j": {"op": "ts", "targets": {"edges": ["e_ab", "e_t"]}}}
    assert wx.workspace_reaction_ids(snap, jobs) == ["r_1", "edge:e_ab"]
    r = edge_reaction(snap, "edge:e_ab")
    assert r["reactants"] == ["a"] and r["products"] == ["b"] and r["label"] == "A -> B" and r["edge"] == "e_ab"
    assert edge_reaction(snap, "r_1") is None and edge_reaction(snap, "edge:nope") is None


def test_the_endpoint_serves_both_formats(tmp_path):
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as client:
        r = client.get("/api/workspace/export?format=csv")
        assert r.status_code == 200 and r.text.strip() == ",".join(wx.CSV_COLUMNS)
        assert 'filename="ws_reactions.csv"' in r.headers["content-disposition"]
        r = client.get("/api/workspace/export?format=html")
        assert r.status_code == 200 and "0 reactions" in r.text and 'filename="ws.html"' in r.headers["content-disposition"]
        assert client.get("/api/workspace/export?format=pdf").status_code == 400


def test_requests_while_an_export_builds_share_that_build(tmp_path, monkeypatch):
    """An impatient double click (or both formats at once) builds the export once."""
    import threading
    import time

    calls = []

    def slow(ws, jobs, **kw):
        calls.append(1)
        time.sleep(0.5)
        return _workspace(_bundle())
    monkeypatch.setattr(wx, "workspace_bundles", slow)
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as client:
        out = []
        get = lambda f: out.append(client.get(f"/api/workspace/export?format={f}").status_code)
        threads = [threading.Thread(target=get, args=(f,)) for f in ("csv", "csv", "html")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert out == [200, 200, 200] and len(calls) == 1
        client.get("/api/workspace/export?format=csv")   # a later request builds afresh (the workspace may have changed)
        assert len(calls) == 2
