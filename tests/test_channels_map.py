"""The channels path map: every live path placed in bond-progress or
mapping-independent distance coordinates, over a fitted surface."""
import json

import numpy as np

from mepd.web.channels_pes import channels_map

H2K = 627.509474


def _xyz(coords, symbols):
    return f"{len(symbols)}\n\n" + "".join(f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, (x, y, z) in zip(symbols, coords))


def _write_stream(live, name, frames, energies_kcal, ref_hartree=-1.0, active=True):
    live.mkdir(parents=True, exist_ok=True)
    (live / f"{name}.json").write_text(json.dumps({"finished": False, "monitors": {"branch-0": {
        "active": active, "geometry": {"frames": frames},
        "plot": {"y": energies_kcal, "energy_ref_hartree": ref_hartree}}}}))


def _exchange(n=7):
    # H1-H2 + H3 -> H1 + H2-H3 along a line.
    out = []
    for t in np.linspace(0, 1, n):
        out.append(_xyz([(0, 0, 0), (0.74 + 1.2 * t, 0, 0), (0.74 + 1.2 + 0.74, 0, 0)], ["H", "H", "H"]))
    return out


def test_bond_progress_runs_from_reactant_to_product(tmp_path):
    job = tmp_path / "job"
    _write_stream(job / "live", "pair_0_1", _exchange(), [0, 5, 12, 15, 12, 6, -1])
    _write_stream(job / "live", "pair_1_1", _exchange(), [0, 4, 9, 11, 9, 4, -2], ref_hartree=-1.0 + 2 / H2K)
    m = channels_map(job, "bonds", running=True)
    assert len(m["paths"]) == 2
    assert any("no gradient" in w and "pair_0_1/branch-0" in w for w in m["warnings"])   # none here: said, per path
    p = next(p for p in m["paths"] if p["pair"] == "pair_0_1")
    assert p["points"][0][:2] == [0.0, 0.0] and p["points"][-1][:2] == [1.0, 1.0]
    assert p["bonds"]["kind"] == "break-form" and p["bonds"]["x"] == ["H1–H2"] and p["bonds"]["y"] == ["H2–H3"]
    # One common energy scale: the second pair starts 2 kcal/mol higher.
    q = next(p for p in m["paths"] if p["pair"] == "pair_1_1")
    assert abs(q["points"][0][2] - 2.0) < 1e-6 and p["active"]
    assert len(m["E"]) == len(m["y"]) and len(m["E"][0]) == len(m["x"])
    assert not channels_map(job, "bonds", running=False)["paths"][0]["active"]   # a stopped run relaxes nothing


def test_only_formed_bonds_are_split_against_each_other(tmp_path):
    # Two bonds form (C0-C2 then C1-C3): asynchronous -- the first closes before the second.
    frames = []
    for t in np.linspace(0, 1, 6):
        d1 = 3.0 - 1.5 * min(1.0, 2 * t)
        d2 = 3.0 - 1.5 * max(0.0, 2 * t - 1)
        frames.append(_xyz([(0, 0, 0), (0, 1.5, 0), (d1, 0, 0), (d2, 1.5, 0)], ["C", "C", "C", "C"]))
    job = tmp_path / "job"
    _write_stream(job / "live", "pair_0_0", frames, [0, 10, 20, 15, 5, -3])
    m = channels_map(job, "bonds")
    (p,) = m["paths"]
    assert p["bonds"]["kind"] == "formed" and "first bond(s) formed" in m["axes"]["x"]
    xs = [q[0] for q in p["points"]]
    ys = [q[1] for q in p["points"]]
    assert xs[2] > 0.7 and ys[2] < 0.1           # bowed toward an edge: stepwise


def test_distance_coordinates_ignore_atom_numbering(tmp_path):
    frames = _exchange()
    job = tmp_path / "job"
    _write_stream(job / "live", "pair_0_1", frames, [0, 5, 12, 15, 12, 6, -1])
    renumbered = ["\n".join([f.splitlines()[0], "", f.splitlines()[4], f.splitlines()[2], f.splitlines()[3]]) + "\n"
                  for f in frames]
    _write_stream(job / "live", "pair_0_2", renumbered, [0, 5, 12, 15, 12, 6, -1])
    m = channels_map(job, "distance")
    a, b = (p["points"] for p in m["paths"])
    assert np.allclose([q[:2] for q in a], [q[:2] for q in b], atol=1e-6)   # same structures, same place


def test_gradients_are_projected_exactly_onto_the_bond_coordinates(tmp_path):
    """E = a*d12 + b*d23 is linear in the two bond lengths, so dE/dq along
    each normalized bond coordinate is exactly a*(d_end - d_start) (and b*...)."""
    from mepd.web.channels_pes import BOHR, _ddist, _projected_gradient

    X = np.array([[0.0, 0, 0], [1.3, 0.1, 0], [2.6, -0.1, 0.2]])
    a, b = 0.02, -0.03                        # Hartree / Angstrom
    s12, e12, s23, e23 = 0.74, 1.94, 1.94, 0.74
    B = np.array([(_ddist(X, 0, 1) / (e12 - s12)).reshape(-1), (_ddist(X, 1, 2) / (e23 - s23)).reshape(-1)])
    g_angstrom = (a * _ddist(X, 0, 1) + b * _ddist(X, 1, 2)).reshape(-1)
    g_q = _projected_gradient(B, g_angstrom * BOHR)            # payload units: Hartree / bohr
    assert np.allclose(g_q, [a * (e12 - s12) * H2K, b * (e23 - s23) * H2K])


def test_live_gradients_reach_the_fit(tmp_path):
    frames = _exchange()
    job = tmp_path / "job"
    live = job / "live"
    live.mkdir(parents=True)
    grads = [[0.001 * (k - 3)] * 9 for k in range(len(frames))]
    (live / "pair_0_1.json").write_text(json.dumps({"finished": False, "monitors": {"branch-0": {
        "active": True, "geometry": {"frames": frames, "gradients": grads},
        "plot": {"y": [0, 5, 12, 15, 12, 6, -1], "energy_ref_hartree": -1.0}}}}))
    m = channels_map(job, "bonds")
    assert not any("no gradient" in w for w in m["warnings"]) and m["E"]


def test_optimized_ts_are_marked_with_how_close_the_chain_came(tmp_path):
    frames = _exchange()
    job = tmp_path / "job"
    _write_stream(job / "live", "pair_0_1", frames, [0, 5, 12, 15, 12, 6, -1])
    out = job / "output"
    (out / "ts").mkdir(parents=True)
    ts = frames[3].replace("\n\n", "\nFrame 1\n", 1)                   # the chain's own highest image as the TS
    (out / "ts" / "ts_pair_0_1_leaf_0.xyz").write_text(ts)
    (out / "ts" / "ts_pair_0_1_leaf_0.energies").write_text(f"{-1.0 + 14 / H2K}\n")
    (out / "ts" / "ts_pair_0_1_leaf_0_irc.xyz").write_text(ts)       # an IRC file is not a TS
    (out / "channels" / "channel_0").mkdir(parents=True)
    (out / "channels" / "channel_0" / "members.txt").write_text("ts_pair_0_1_leaf_0\n")
    m = channels_map(job, "bonds", output_dir=out)
    (mark,) = m["ts"]
    assert mark["pair"] == "pair_0_1" and mark["kind"] == "direct" and mark["group"] == "channel_0"
    assert abs(mark["e"] - 14.0) < 1e-6 and np.allclose(mark["q"], m["paths"][0]["points"][3][:2])
    assert mark["closest"]["image"] == 3 and mark["closest"]["rmsd"] < 1e-6
    assert channels_map(job, "bonds")["ts"] == []                    # no output folder given: no marks


def test_irc_frame_places_chains_along_and_off_the_true_path(tmp_path):
    frames = _exchange(9)
    job = tmp_path / "job"
    energies = [0, 5, 12, 18, 20, 18, 12, 5, -1]
    _write_stream(job / "live", "pair_0_1", frames, energies)
    # A second chain displaced sideways from the path.
    off = [f.replace(" 0.0000 0.0000\n", " 0.4000 0.0000\n", 1) for f in frames]
    _write_stream(job / "live", "pair_0_2", off, energies)
    out = job / "output"
    (out / "ts").mkdir(parents=True)
    (out / "ts" / "ts_pair_0_1_leaf_0.xyz").write_text(frames[4])
    (out / "ts" / "ts_pair_0_1_leaf_0.energies").write_text(f"{-1.0 + 20 / H2K}\n")
    (out / "ts" / "ts_pair_0_1_leaf_0_irc.xyz").write_text("".join(frames[::-1]))     # written product first
    (out / "ts" / "ts_pair_0_1_leaf_0_irc.energies").write_text(
        "\n".join(str(-1.0 + e / H2K) for e in energies[::-1]) + "\n")
    m = channels_map(job, "irc", output_dir=out)
    assert m["irc"]["ts"] == "ts_pair_0_1_leaf_0" and [c["label"] for c in m["irc_choices"]] == ["ts_pair_0_1_leaf_0"]
    on = next(p for p in m["paths"] if p["pair"] == "pair_0_1")["points"]
    xs = [q[0] for q in on]
    assert all(abs(q[1]) < 1e-6 for q in on)              # the IRC's own frames lie on it
    assert xs == sorted(xs) and xs[0] < 0 < xs[-1]         # reactant first (reoriented), TS at 0
    assert abs(xs[4]) < 1e-6
    offp = next(p for p in m["paths"] if p["pair"] == "pair_0_2")["points"]
    assert all(q[1] > 0.01 for q in offp[1:-1])            # the displaced chain sits off the IRC
    assert channels_map(job, "irc")["warnings"]            # no output folder: says an IRC is needed


def test_only_missing_interior_gradients_are_worth_a_warning(tmp_path):
    """A GSM string's pinned endpoints have no gradient; that alone is normal."""
    frames = _exchange()
    job = tmp_path / "job"
    live = job / "live"
    live.mkdir(parents=True)
    grads = [None] + [[0.001] * 9 for _ in frames[1:-1]] + [None]
    (live / "pair_0_1.json").write_text(json.dumps({"finished": False, "monitors": {"branch-0": {
        "active": True, "geometry": {"frames": frames, "gradients": grads},
        "plot": {"y": [0, 5, 12, 15, 12, 6, -1], "energy_ref_hartree": -1.0}}}}))
    assert not any("no gradient" in w for w in channels_map(job, "bonds")["warnings"])


def test_no_surface_between_chains_and_no_false_valleys(tmp_path):
    """Between two explored chains nothing is known: the map must not show
    a (typically lower) interpolated energy there, and nowhere may it show
    an energy lower than the computed images around it."""
    job = tmp_path / "job"
    base = _exchange(9)
    far = [f.replace(" 0.0000 0.0000\n", " 1.5000 0.0000\n", 1) for f in base]    # a second chain, well apart
    e = [0, 8, 20, 30, 34, 30, 20, 8, -2]
    _write_stream(job / "live", "pair_0_1", base, e)
    _write_stream(job / "live", "pair_0_2", far, [v + 3 for v in e])
    m = channels_map(job, "distance")
    E = np.array([[np.nan if v is None else v for v in row] for row in m["E"]])
    pts = np.array([q for p in m["paths"] for q in p["points"]])
    lowest = float(np.nanmin(pts[:, 2]))
    assert np.nanmin(E) >= lowest - 1e-6                           # never below what was computed
    # A grid cell far from every image is blank.
    X, Y = np.meshgrid(m["x"], m["y"])
    d = np.min(np.hypot(X.ravel()[:, None] - pts[:, 0][None], Y.ravel()[:, None] - pts[:, 1][None]), axis=1)
    far_cells = d.reshape(E.shape) > 0.5 * np.hypot(np.ptp(m["x"]), np.ptp(m["y"])) / 4
    assert far_cells.any() and np.isnan(E[far_cells]).all()


def test_chain_verdicts_reach_the_map(tmp_path, monkeypatch):
    """MSMEP's verdict on each chain (elementary / split / unresolved) is
    written to the live stream and shown per path (split chains dashed)."""
    from mepd import progress

    live = tmp_path / "job" / "live"
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(live))
    printer = progress.ProgressPrinter(use_rich=False)
    monkeypatch.setattr(progress, "get_progress_printer", lambda: printer)
    frames = _exchange()
    _write_stream(live, "pair_0_1", frames, [0, 5, 12, 15, 12, 6, -1])
    data = json.loads((live / "pair_0_1.json").read_text())
    state = printer._state_for_monitor("branch-0")
    state["chain_geometry"] = data["monitors"]["branch-0"]["geometry"]
    state["chain_plot_payload"] = data["monitors"]["branch-0"]["plot"]
    with progress.progress_monitor("branch-0"):
        progress.set_monitor_verdict("split")
    payload = printer._monitors_payload()
    assert payload["branch-0"]["verdict"] == "split"
    (live / "pair_0_1.json").write_text(json.dumps({"finished": False, "monitors": {"branch-0": {
        **data["monitors"]["branch-0"], "verdict": "split"}, "branch-1": {**data["monitors"]["branch-0"], "verdict": "elementary"}}}))
    m = channels_map(tmp_path / "job", "bonds")
    assert {p["monitor"]: p["verdict"] for p in m["paths"]} == {"branch-0": "split", "branch-1": "elementary"}


def _hop(a, b, n=5, y=0.0):
    """H1 at 0 and H3 at 3 A, H2 moving from x=a to x=b (bonded to H1 at 0.74, to H3 at 2.26, to neither between)."""
    return [_xyz([(0, 0, 0), (a + (b - a) * t, y * np.sin(np.pi * t), 0), (3.0, 0, 0)], ["H", "H", "H"])
            for t in np.linspace(0, 1, n)]


def _write_monitors(live, name, monitors):
    live.mkdir(parents=True, exist_ok=True)
    (live / f"{name}.json").write_text(json.dumps({"finished": True, "monitors": {
        mid: {"active": False, "verdict": verdict, "geometry": {"frames": frames},
              "plot": {"y": energies, "energy_ref_hartree": -1.0}}
        for mid, (frames, energies, verdict) in monitors.items()}}))


def test_each_pair_reads_as_one_route_and_pairs_through_the_same_minima_are_one_mechanism(tmp_path):
    """A split path is replaced by its elementary pieces, chained end to start
    whatever order the live data lists them in; minima are named by bonding
    pattern (the intermediate keeps its name across pairs); a dead end the
    search also relaxed is not on the route; routes are lettered by their
    highest point."""
    live = tmp_path / "job" / "live"
    # Two steps through an intermediate (H2 on its own between H1 and H3), listed out of order, plus a
    # step from the intermediate back to the reactant that leads nowhere new.
    _write_monitors(live, "pair_0_3", {
        "branch-0": (_hop(0.74, 2.26, 7), [0, 20, 40, 30, 40, 20, -5], "split"),
        "branch-1": (_hop(1.5, 2.26), [30, 38, 41, 20, -5], "elementary"),
        "branch-2": (_hop(0.74, 1.5), [0, 15, 35, 33, 30], "elementary"),
        "branch-3": (_hop(1.5, 0.74, y=0.4), [30, 50, 60, 20, 0], "elementary"),
    })
    # The same mechanism in another pair, higher; and a one-step pair, lowest of all.
    _write_monitors(live, "pair_1_4", {
        "branch-0": (_hop(0.74, 2.26, 7), [0, 25, 45, 30, 45, 20, -5], "split"),
        "branch-1": (_hop(0.74, 1.5), [0, 20, 44, 33, 30], "elementary"),
        "branch-2": (_hop(1.5, 2.26), [30, 38, 42, 20, -5], "elementary"),
    })
    _write_monitors(live, "pair_2_5", {"branch-0": (_hop(0.74, 2.26, 7, y=0.3), [0, 10, 25, 30, 25, 10, -5], "elementary")})
    m = channels_map(tmp_path / "job", "distance", running=False)
    routes = {r["pair"]: r for r in m["routes"]}
    assert [r["pair"] for r in m["routes"]] == ["pair_2_5", "pair_0_3", "pair_1_4"]     # lowest highest point first
    assert routes["pair_2_5"]["minima"] == ["R", "P"] and routes["pair_2_5"]["mechanism"] == "A"
    two = routes["pair_0_3"]
    assert two["minima"] == ["R", "I1", "P"] and two["complete"] and two["mechanism"] == "B"
    assert [s["path"] for s in two["steps"]] == ["pair_0_3/branch-2", "pair_0_3/branch-1"]
    assert [s["peak"] for s in two["steps"]] == [35.0, 41.0] and two["top"] == 41.0
    assert two["steps"][1]["barrier"] == 11.0                                # from the intermediate, 30
    assert routes["pair_1_4"]["minima"] == ["R", "I1", "P"] and routes["pair_1_4"]["mechanism"] == "B"
    on_route = {s["path"] for r in m["routes"] for s in r["steps"]}
    assert {p["id"] for p in m["paths"]} - on_route == {"pair_0_3/branch-0", "pair_0_3/branch-3", "pair_1_4/branch-0"}
    # The minima sit where their chains end, and every image knows how far along its chain it is.
    assert two["points"][1]["q"] == next(p for p in m["paths"] if p["id"] == "pair_0_3/branch-2")["points"][-1][:2]
    p = next(p for p in m["paths"] if p["id"] == "pair_2_5/branch-0")
    assert p["s"][0] == 0.0 and p["s"][-1] == 1.0 and all(a < b for a, b in zip(p["s"], p["s"][1:]))


def test_a_route_still_being_searched_stops_where_it_has_got_to(tmp_path):
    live = tmp_path / "job" / "live"
    _write_monitors(live, "pair_0_3", {
        "branch-0": (_hop(0.74, 2.26, 7), [0, 20, 40, 30, 40, 20, -5], "split"),
        "branch-1": (_hop(0.74, 1.5), [0, 15, 35, 33, 30], "elementary"),
    })
    (r,) = channels_map(tmp_path / "job", "bonds", running=True)["routes"]
    assert r["minima"] == ["R", "I1"] and not r["complete"] and len(r["steps"]) == 1
