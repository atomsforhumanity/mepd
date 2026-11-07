"""score.live_first_passage: first passage on a live kinetics snapshot's own network."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402


def _snapshot(secondary_open: bool) -> dict:
    """Seed 0 -> 1 (channel X, fast) and 0 -> 2 (channel Y, slower); 1 -> 3 (secondary, fast), 3 -> 2 (fast):
    what the seed forms first is X; the amounts at the end are mostly Y when the secondary chemistry is open."""
    sp = [{"index": k, "id": str(k), "G": g} for k, g in enumerate((0.0, -20.0, -30.0, -25.0))]
    steps = [("0", "1", 30.0, "open"), ("0", "2", 33.0, "open"),
             ("1", "3", 20.0, "open" if secondary_open else "closed"), ("3", "2", 10.0, "open")]
    return {"species": sp, "mirror_species": [],
            "step_list": [{"id": f"s{i}", "from": a, "to": b, "dG_fwd": f, "state": st}
                          for i, (a, b, f, st) in enumerate(steps)]}


def test_first_passage_ignores_what_a_channel_product_forms_next():
    T, t = 500.0, 1e3
    closed = score.live_first_passage(_snapshot(False), {1, 2}, T, t)
    opened = score.live_first_passage(_snapshot(True), {1, 2}, T, t)
    assert closed[1] / closed[2] == pytest.approx(opened[1] / opened[2], rel=1e-6)
    assert closed[1] > 10 * closed[2]          # X first: 3 kcal/mol lower TS at 500 K
    amounts = score.live_first_passage(_snapshot(True), set(), T, t)   # nothing absorbing: the amounts
    assert amounts[2] > amounts[1]             # the secondary chemistry turns X into Y


def test_mirror_images_fold_into_their_species_and_absorb():
    d = _snapshot(False)
    d["mirror_species"] = [{"id": "m1", "mirror_of": "1", "G": -20.0}]
    d["step_list"].append({"id": "s0 (mirror image)", "from": "0", "to": "m1", "dG_fwd": 30.0, "state": "open"})
    out = score.live_first_passage(d, {1, 2}, 500.0, 1e3)
    ref = score.live_first_passage(_snapshot(False), {1, 2}, 500.0, 1e3)
    assert out[1] / out[2] == pytest.approx(2 * ref[1] / ref[2], rel=1e-3)   # two routes to X


def test_no_live_network_no_first_passage():
    assert score.live_first_passage({"species": [{"index": 0, "id": "0", "G": 0.0}]}, {0}, 300.0, 1.0) is None
