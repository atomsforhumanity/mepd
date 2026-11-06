"""Home's Examples door: shipped studies list, unpack as a workspace of their
own with paths pointing at the copy, and stay small."""
import json

from mepd.web import showcase_tool


def test_examples_are_listed_and_small():
    items = showcase_tool.listing()
    assert items, "at least one example ships with mepd"
    for x in items:
        assert {"slug", "title", "tag"} <= set(x)
        size = (showcase_tool.SHOWCASE / f"{x['slug']}.tar.gz").stat().st_size
        assert size < 3_000_000, f"{x['slug']} is {size / 1e6:.1f} MB: trim it (orbitals, reactor trajectories)"


def test_unpacked_example_points_at_its_copy(tmp_path):
    slug = showcase_tool.listing()[0]["slug"]
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
