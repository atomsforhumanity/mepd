"""Sample more paths on a channels run with built endpoints: it goes down the
run's ranked built pairs, or searches the entries asked for; the result
lists the ranked pairs."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from mepd.web.operations import ChannelsMoreParams, WorkspaceError, _channels_more_built  # noqa: E402
from mepd.web.results import built_pairs_view  # noqa: E402


def _run(tmp_path, n: int, n_searched: int) -> dict:
    out = tmp_path / "out"
    (out / "built_pairs").mkdir(parents=True)
    entries = [{"label": f"built_{k}", "rank": k, "built_from": "start", "conformer": k, "mechanism": "m",
                "mechanism_text": "break C1-H1, form H1-O1", "score": -100.0 + k, "barrier_kcal": 10.0 + k,
                "n_variants": 1, "floor": -110.0, "searched": k < n_searched} for k in range(n)]
    (out / "built_pairs" / "built_pairs.json").write_text(json.dumps({"metric": "snap-gi-xtb", "entries": entries}))
    return {"output_dir": str(out), "argv": ["channels", "--pair-from", "start+end", "--atom-mapping-candidates", "3",
                                             "--max-pairs", "3", "--output", str(out)]}


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_more_goes_down_the_ranked_list(tmp_path):
    src = _run(tmp_path, 8, 3)
    argv = _channels_more_built(src, ChannelsMoreParams())
    assert _flag(argv, "--atom-mapping-candidates") == "6"     # 3 more by default
    assert _flag(argv, "--built-pairs").endswith("built_pairs") and "--search-built" not in argv
    assert argv.count("--atom-mapping-candidates") == 1 and "--max-pairs" not in argv
    with pytest.raises(WorkspaceError, match="more than 3"):
        _channels_more_built(src, ChannelsMoreParams(path_searches=2))


def test_a_chosen_entry_is_searched_by_its_label(tmp_path):
    argv = _channels_more_built(_run(tmp_path, 8, 3), ChannelsMoreParams(search_built="built_6"))
    assert _flag(argv, "--search-built") == "built_6"
    with pytest.raises(WorkspaceError, match="built_9"):
        _channels_more_built(_run(tmp_path / "b", 8, 3), ChannelsMoreParams(search_built="built_9"))


def test_nothing_left_and_runs_without_a_list(tmp_path):
    with pytest.raises(WorkspaceError, match="every built pair"):
        _channels_more_built(_run(tmp_path, 3, 3), ChannelsMoreParams())
    assert _channels_more_built({"output_dir": str(tmp_path / "none"), "argv": []}, ChannelsMoreParams()) is None


def test_result_lists_the_ranked_pairs(tmp_path):
    _run(tmp_path, 4, 1)
    view = built_pairs_view(tmp_path / "out")
    assert [e["searched"] for e in view["entries"]] == [True, False, False, False]
    assert view["entries"][2]["barrier_kcal"] == 12.0 and "floor" not in view["entries"][0]
    assert built_pairs_view(tmp_path / "missing") is None


def test_more_counts_per_mechanism_when_the_run_did(tmp_path):
    src = _run(tmp_path, 8, 0)
    fp = tmp_path / "out" / "built_pairs" / "built_pairs.json"
    data = json.loads(fp.read_text())
    for e in data["entries"]:
        e["mechanism"], e["searched"] = ("a" if e["rank"] % 2 else "b"), e["rank"] < 4
    fp.write_text(json.dumps(data))
    src["argv"].append("--built-cap-per-mechanism")
    argv = _channels_more_built(src, ChannelsMoreParams())
    assert _flag(argv, "--atom-mapping-candidates") == "5" and "--built-cap-per-mechanism" in argv
