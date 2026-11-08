"""score.partial_steps (mirror-image twins of a partial network) and score.queue_jobs (cancelled jobs)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402


def test_a_partial_network_keeps_the_found_twin_of_a_step_it_has():
    summary = {"steps": [{"label": "a"}, {"label": "b"}, {"label": "a_twin"}, {"label": "c"}],
               "mirror_images": {"found_twins": [[0, 2]]}}
    assert score.partial_steps(summary, {"a"}) == {0, 2}          # the twin found later counts from the start
    assert score.partial_steps(summary, {"a_twin", "c"}) == {0, 2, 3}
    assert score.partial_steps(summary, {"b"}) == {1}
    assert score.partial_steps({"steps": [{"label": "a"}]}, {"a"}) == {0}   # no mirror record


def test_jobs_marked_cancelled_are_not_failures(tmp_path):
    na = tmp_path / "network_aware.jsonl"
    events = [{"event": "phase", "phase": "cycle_start", "t": 0.0},
              {"event": "job", "kind": "irc", "t": 500.0, "ok": False, "cancelled": True},     # species cap stop
              {"event": "job", "kind": "irc", "t": 600.0, "ok": False, "why": "IRC failed"},
              {"event": "job", "kind": "fast", "t": 1799.5, "ok": False},                      # at the deadline
              {"event": "job", "kind": "fast", "t": 700.0, "ok": True}]
    na.write_text("\n".join(json.dumps(e) for e in events))
    failed, cancelled = score.queue_jobs(na, 1800)
    assert [e["t"] for e in failed] == [600.0]
    assert sorted(e["t"] for e in cancelled) == [500.0, 1799.5]
