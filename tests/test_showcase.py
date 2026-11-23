"""Home's Examples door: shipped studies list, unpack as a workspace of their
own with paths pointing at the copy, and stay small."""
import json

from mepd.web import showcase_tool


def test_examples_are_listed_and_small():
    items = showcase_tool.listing()
    assert items, "at least one example ships with mepd"
    for x in items:
        assert {"slug", "title", "tag", "kind"} <= set(x)
        name = f"{x['slug']}.timeline.json.gz" if x["kind"] == "replay" else f"{x['slug']}.tar.gz"
        size = (showcase_tool.SHOWCASE / name).stat().st_size
        assert size < 3_000_000, f"{x['slug']} is {size / 1e6:.1f} MB: trim it (orbitals, reactor trajectories)"


def test_unpacked_example_points_at_its_copy(tmp_path):
    slug = next(x["slug"] for x in showcase_tool.listing() if x["kind"] == "workspace")
    dest = showcase_tool.unpack(slug, tmp_path / "copy")
    assert (dest / "workspace.json").exists()
    for jf in dest.glob("jobs/*/job.json"):
        text = jf.read_text()
        assert "@WORKSPACE@" not in text
        job = json.loads(text)
        if job.get("output_dir"):
            assert job["output_dir"].startswith(str(dest))


def test_free_name_skips_taken_ones(tmp_path):
    (tmp_path / "example-sn2").mkdir()
    assert showcase_tool.free_name(tmp_path, "sn2").name == "example-sn2-2"


def _fake_copy(path, jobs=()):
    path.mkdir(parents=True)
    (path / "workspace.json").write_text("{}")
    for j in jobs:
        (path / "jobs" / j).mkdir(parents=True)
        (path / "jobs" / j / "job.json").write_text("{}")


def test_existing_copy_is_the_first_that_still_has_the_job(tmp_path):
    assert showcase_tool.existing_copy(tmp_path, "sn2") is None
    _fake_copy(tmp_path / "example-sn2")                         # the job deleted from this one
    _fake_copy(tmp_path / "example-sn2-2", jobs=["j_1"])
    assert showcase_tool.existing_copy(tmp_path, "sn2").name == "example-sn2"
    assert showcase_tool.existing_copy(tmp_path, "sn2", "j_1").name == "example-sn2-2"
    assert showcase_tool.existing_copy(tmp_path, "sn2", "j_2") is None


def test_pack_leaves_out_orbital_cubes_and_caches(tmp_path, monkeypatch):
    import tarfile

    monkeypatch.setattr(showcase_tool, "SHOWCASE", tmp_path / "showcase")
    ws = tmp_path / "ws"
    out = ws / "jobs" / "j_orb" / "output"
    (out / "frames").mkdir(parents=True)
    (out / "cubes").mkdir()
    (ws / "orbitals").mkdir()
    (ws / "workspace.json").write_text("{}")
    (ws / "jobs" / "j_orb" / "job.json").write_text(json.dumps({"output_dir": str(out)}))
    (out / "frames" / "0.molden.gz").write_bytes(b"x")
    (out / "cubes" / "0_3.json.gz").write_bytes(b"x")
    (ws / "orbitals" / "cache.json.gz").write_bytes(b"x")
    arc = showcase_tool.pack(ws, "t", "T", "tag")
    with tarfile.open(arc) as tar:
        names = tar.getnames()
    assert any(n.endswith("frames/0.molden.gz") for n in names)
    assert not any("cubes" in n or n.startswith("./orbitals") for n in names)


def test_landing_points_at_the_sn2_examples_ts_search(tmp_path):
    """Welcome.js's Molecule and Path pictures open this job of the shipped example: its TS search (the
    first search and the channels run sampled after it)."""
    import re
    from pathlib import Path

    js = (Path(showcase_tool.__file__).parent / "static" / "js" / "components" / "Welcome.js").read_text()
    ts = re.search(r"const SN2 = \{ slug: 'sn2', ts: '(j_\w+)' \}", js).group(1)
    dest = showcase_tool.unpack("sn2", tmp_path / "copy")
    assert json.loads((dest / "jobs" / ts / "job.json").read_text())["op"] == "ts"
    assert any(json.loads(f.read_text()).get("extends") == ts for f in dest.glob("jobs/*/job.json"))  # More paths


def test_open_example_reuses_the_copy_that_has_the_job(tmp_path, monkeypatch):
    """The landing's "Open it" twice: one copy, opened again (Examples without reuse: a new copy)."""
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        first = c.post("/api/examples/sn2/open", json={"reuse": True, "job": "j_85e27dbc"})
        assert first.status_code == 200, first.text
        assert first.json()["current"].endswith("example-sn2")
        again = c.post("/api/examples/sn2/open", json={"reuse": True, "job": "j_21e0d1ec"})
        assert again.json()["current"] == first.json()["current"]
        assert "j_21e0d1ec" in {j["id"] for j in c.get("/api/state").json()["jobs"]}
        fresh = c.post("/api/examples/sn2/open")
        assert fresh.json()["current"].endswith("example-sn2-2")
