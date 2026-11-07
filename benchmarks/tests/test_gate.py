"""gate.compare's two-stage recall rule (required channels; Fisher p < 0.05 fails, a smaller drop is inconclusive
until both sides have 15 replicates)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
import gate  # noqa: E402


def entry(n: int, connected: dict, full: int, required=("x",)) -> dict:
    stat = {"median": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0, "n_inf": 0}
    return {"n": n, "required": list(required), "full_recall_fraction": full / n,
            "recall": {c: {"found": 1.0, "connected": k / n} for c, k in connected.items()},
            "worst_abs_error_kcal": dict(stat), "settle_s": {**stat, "median": 100.0}}


def test_one_replicate_less_is_inconclusive_not_a_fail():
    v = gate.compare({"s": entry(5, {"x": 5}, 5)}, {"s": entry(5, {"x": 4}, 4)})["s"]
    assert v["status"] == "inconclusive" and not v["pass"]
    assert set(v["checks"]["recall"]["inconclusive"]) == {"x", "full recall"}


def test_four_of_five_lost_fails():
    v = gate.compare({"s": entry(5, {"x": 5}, 5)}, {"s": entry(5, {"x": 1}, 1)})["s"]
    assert v["status"] == "fail"          # Fisher one-sided p = 0.024


def test_a_channel_not_required_does_not_count():
    b = entry(5, {"x": 5, "side": 5}, 5)
    c = entry(5, {"x": 5, "side": 0}, 5)
    assert gate.compare({"s": b}, {"s": c})["s"]["status"] == "pass"


def test_with_15_a_side_a_non_significant_drop_passes_and_a_significant_one_fails():
    assert gate.compare({"s": entry(15, {"x": 15}, 15)}, {"s": entry(15, {"x": 13}, 13)})["s"]["status"] == "pass"
    assert gate.compare({"s": entry(15, {"x": 15}, 15)}, {"s": entry(15, {"x": 10}, 10)})["s"]["status"] == "fail"


def test_accuracy_and_settle_still_fail_on_their_own():
    c = entry(5, {"x": 5}, 5)
    c["settle_s"]["median"] = 1000.0
    assert gate.compare({"s": entry(5, {"x": 5}, 5)}, {"s": c})["s"]["status"] == "fail"
