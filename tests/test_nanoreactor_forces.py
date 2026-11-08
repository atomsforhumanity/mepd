"""One MD generator, the nanoreactor, with forces switched on and off: forces=piston,collide is the reactor,
forces=collide is collide, forces=piston the reactor's MD without pulls. Every trajectory is streamed for the web's
Reactor view (output/reactor/run_kk) with its forces and pulls; streaming changes no product."""
import json

import numpy as np
import pytest
from qcdata import Structure

from mepd.discovery.generators import get_generator, resolve_forces

HCOOH = [("C", 0.000, 0.420, 0.000), ("O", 1.150, -0.180, 0.000), ("O", -1.090, -0.310, 0.000),
         ("H", 0.000, 1.510, 0.000), ("H", -1.880, 0.250, 0.000)]


def _formic():
    return Structure(symbols=[a[0] for a in HCOOH], geometry=np.array([a[1:] for a in HCOOH]) * 1.8897259886,
                     charge=0, multiplicity=1)


@pytest.mark.parametrize("options, name, passed", [
    (["forces=piston,collide", "time_ps=3"], "reactor", ["time_ps=3"]),
    (["forces=collide,piston"], "reactor", []),
    (["forces=collide", "n_collisions=4"], "collide", ["n_collisions=4"]),
    (["forces=piston", "seed=2"], "reactor", ["seed=2", "forces=piston"]),
    (["forces=piston", "embedded=true"], "nanoreactor", ["embedded=true"]),
])
def test_forces_pick_the_generator_that_runs_them(options, name, passed):
    assert resolve_forces("nanoreactor", options) == (name, passed)


def test_older_spellings_are_left_as_they_were():
    for name, options in (("nanoreactor", ["n_seeds=8"]), ("reactor", ["forces=piston"]), ("collide", []),
                          ("bond-rules", [])):
        assert resolve_forces(name, options) == (name, options)


@pytest.mark.parametrize("options, message", [
    (["forces=piston,heat"], "forces are one or more of"),
    (["forces="], "forces are one or more of"),
    (["forces=piston", "gap_fs=100"], "forces=piston has no pulls"),
    (["forces=piston,collide", "embedded=true"], "piston only"),
])
def test_forces_that_cannot_run_are_refused(options, message):
    with pytest.raises(ValueError, match=message):
        resolve_forces("nanoreactor", options)


def test_the_older_nanoreactor_refuses_forces_outside_the_cli():
    with pytest.raises(ValueError, match="resolved by `mepd discovery expand`"):
        get_generator("nanoreactor").propose(_formic(), max_products=1, options={"forces": "piston"})


def _runs(work):
    return sorted(work.glob("run_*"))


def _frames(run):
    """A run's streamed frames, segment_000.xyz first: [(symbols, xyz)]."""
    out = []
    for fp in sorted((run / "md").glob("segment_*.xyz")):
        lines = fp.read_text().splitlines()
        while lines:
            n = int(lines[0])
            rows = [ln.split() for ln in lines[2:2 + n]]
            out.append(([r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows])))
            lines = lines[2 + n:]
    return out


def test_the_piston_alone_squeezes_the_wall_and_pulls_nothing(tmp_path):
    from mepd.discovery.reactor_generator import _trajectory

    sy = [a[0] for a in HCOOH]
    xyz = np.array([a[1:] for a in HCOOH])
    kw = dict(temperature=1500.0, time_ps=0.4, compress=0.7, period_ps=0.2, duty=0.5, pull_fs=50.0, gap_fs=20.0,
              k=100.0, intra_hops=2, seed=3, md_method="gfn2")
    _trajectory(sy, xyz, 0, 1, workdir=str(tmp_path / "piston"), forces="piston", **kw)
    _trajectory(sy, xyz, 0, 1, workdir=str(tmp_path / "both"), **kw)
    piston, = _runs(tmp_path / "piston")
    both, = _runs(tmp_path / "both")
    sp, sb = (json.loads((r / "md" / "schedule.json").read_text()) for r in (piston, both))
    # the frames, in order from t = 0 (the start geometry, centred), dump_fs apart, in the input's atom order
    frames = _frames(both)
    assert sb["dump_fs"] == 2.0 and len(frames) == 400 / 2 + 1
    assert frames[0][0] == sy
    assert np.allclose(frames[0][1], xyz - xyz.mean(axis=0), atol=1e-4)
    # a pull's times count from frame 0 of segment_000 (frame t/dump_fs; MD steps are 0.5 fs, so not always a whole frame)
    assert all(0 <= p["t0_fs"] < 400 for p in sb["pulls"])
    assert all(0 <= a < len(sy) for p in sb["pulls"] for a in p["atoms"])
    assert sp["forces"] == ["piston"] and sp["pulls"] == []
    assert min(r for _, r in sp["segments"]) < max(r for _, r in sp["segments"])   # the wall moves
    assert sb["forces"] == ["piston", "collide"] and sb["pulls"]
    first = sb["pulls"][0]
    assert first["t1_fs"] - first["t0_fs"] == pytest.approx(50.0) and len(first["atoms"]) == 2
    assert (piston / "md" / "done").exists() and (piston / "live_events.json").exists()


def test_collide_streams_each_collision_and_gives_the_same_products(tmp_path):
    gen = get_generator("collide")
    opts = {"n_collisions": 3, "parallel": 1, "seed": 5, "pull_fs": 100.0, "relax_fs": 60.0}
    plain = gen.propose(_formic(), max_products=10, options=opts)
    streamed = gen.propose(_formic(), max_products=10, options={**opts, "workdir": str(tmp_path)})
    assert [np.asarray(p.geometry).tolist() for p in plain] == [np.asarray(p.geometry).tolist() for p in streamed]
    runs = _runs(tmp_path)
    assert len(runs) == 3
    for r in runs:
        sched = json.loads((r / "md" / "schedule.json").read_text())
        assert sched["forces"] == ["collide"] and sched["time_ps"] == pytest.approx(0.16)
        assert sched["relax_fs"] == 60.0
        (pull,) = sched["pulls"]
        assert (pull["t0_fs"], pull["t1_fs"]) == (0.0, 100.0)
        assert (r / "md" / "done").exists() and (r / "live_events.json").exists()
        frames = _frames(r)
        assert len(frames) == 160 / 2 + 1         # 160 fs from t = 0 (where the pull starts), a frame every 2 fs
        assert all(f[0] == [a[0] for a in HCOOH] for f in frames)


def test_the_cli_streams_collide_into_the_output_folder(tmp_path):
    from mepd.discovery.cli_expand import _generator_options

    for name in ("collide", "reactor", "nanoreactor"):
        assert _generator_options(name, [], tmp_path)["workdir"] == str((tmp_path / "reactor").resolve())
    assert "workdir" not in _generator_options("bond-rules", [], tmp_path)
