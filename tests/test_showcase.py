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


def test_replay_examples_play_as_live_network_timelines():
    from mepd.web import live_network

    for x in showcase_tool.listing():
        if x["kind"] != "replay":
            continue
        tl = live_network.example_timeline(x["slug"])
        assert tl["frames"] and 0 <= tl["final"] < len(tl["frames"])
        assert tl["frames"][tl["final"]]["steered"], "the end state is a steered frame"
        ids = {s[0] for f in tl["frames"] for s in f["species"]}
        assert ids <= set(tl["smiles"])
