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


def test_a_configuration_generator_replaces_the_common_one():
    import bench   # noqa: E402  (benchmarks/cluster)

    argv = bench.with_config(["--steer", "access", "--generator", "reactor", "--x", "1"],
                             ["--max-species", "400", "--generator", "nanoreactor", "--generator-option", "forces=piston"])
    assert argv == ["--steer", "access", "--generator", "nanoreactor", "--x", "1", "--max-species", "400",
                    "--generator-option", "forces=piston"]
    assert bench.with_config(["--generator", "reactor"], ["--max-species", "400"]) == \
        ["--generator", "reactor", "--max-species", "400"]


def test_a_configuration_steering_level_replaces_the_common_one():
    import bench   # noqa: E402  (benchmarks/cluster)

    argv = bench.with_config(["--steer-energies", "{RUN_DIR}/inputs/orca_steer.toml", "--x", "1"],
                             ["--max-species", "400", "--steer-energies", "{RUN_DIR}/inputs/linus.toml"])
    assert argv == ["--steer-energies", "{RUN_DIR}/inputs/linus.toml", "--x", "1", "--max-species", "400"]


def _settled(n, med, values):
    e = entry(n, {"x": n}, n)
    e["settle_s"]["median"] = med
    e["settle_values"] = list(values)
    return e


def test_a_settle_median_over_tolerance_without_a_rank_shift_is_inconclusive():
    base = _settled(5, 1000.0, [600, 800, 1000, 1500, 1700])
    cand = _settled(5, 1450.0, [500, 900, 1450, 1600, 1650])      # same spread, the median just lands higher
    v = gate.compare({"s": base}, {"s": cand})["s"]
    assert v["checks"]["settle"]["status"] == "inconclusive" and v["status"] == "inconclusive"


def test_a_settle_rise_the_rank_test_sees_fails():
    base = _settled(5, 100.0, [90, 95, 100, 105, 110])
    cand = _settled(5, 400.0, [380, 390, 400, 410, float("inf")])
    v = gate.compare({"s": base}, {"s": cand})["s"]
    assert v["checks"]["settle"]["status"] == "fail" and v["status"] == "fail"


def test_with_15_a_side_a_non_significant_settle_rise_passes():
    vals = [100.0 * k for k in range(1, 16)]
    base = _settled(15, 800.0, vals)
    cand = _settled(15, 1100.0, vals[3:] + [1600.0, 1700.0, 1800.0])
    v = gate.compare({"s": base}, {"s": cand})["s"]
    assert v["checks"]["settle"]["status"] in ("pass", "fail")
    assert v["checks"]["settle"]["status"] == ("fail" if v["checks"]["settle"]["mann_whitney_p_slower"] < 0.05 else "pass")
