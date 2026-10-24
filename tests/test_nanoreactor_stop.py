"""Nanoreactor Stop & analyze: the MD ends where it is, what it wrote is
analyzed; and the refinement after the MD runs species and reactions at once."""
import json
import sys
import threading
import time

import numpy as np

from mepd import search_control
from mepd.discovery import nanoreactor as nr


class _MorseEngine:
    """H atoms with Morse bonds between every pair; touches `stop_at_call`'s file at that gradient call."""

    def __init__(self, stop_file=None, stop_at_call=None):
        self.calls, self.stop_file, self.stop_at_call = 0, stop_file, stop_at_call

    def compute_gradients(self, nodes):
        self.calls += 1
        if self.stop_at_call is not None and self.calls == self.stop_at_call:
            self.stop_file.touch()
        out = []
        for node in nodes:
            x = np.asarray(node.structure.geometry).reshape(-1, 3)
            d = x[:, None] - x[None]
            r = np.linalg.norm(d, axis=2) + np.eye(len(x))
            ex = np.exp(-(r - 1.4))
            e = 0.17 * (1 - ex) ** 2 - 0.17
            np.fill_diagonal(e, 0.0)
            dedr = 2 * 0.17 * ex * (1 - ex)
            np.fill_diagonal(dedr, 0.0)
            node._cached_energy = float(e.sum() / 2)
            out.append(((dedr / r)[:, :, None] * d).sum(axis=1))
        return np.array(out)


def _h2s(n=4):
    rng = np.random.default_rng(1)
    return ["H"] * (2 * n), np.vstack([np.array([[0, 0, 0], [0.74, 0, 0]]) + rng.uniform(-2.5, 2.5, 3)
                                       for _ in range(n)])


def _settings(**kw):
    """4 piston segments of 25 frames (no ramp)."""
    return nr.ReactorSettings(temperature=1000.0, time_ps=0.2, radius=5.0, period_ps=0.1, duty=0.5, ramp_fs=0.0,
                              dump_fs=2.0, step_fs=0.5, **kw)


def test_a_stop_ends_the_engine_md_with_the_frames_so_far_and_a_rerun_redoes_that_segment(tmp_path, monkeypatch):
    stop = tmp_path / "stop"
    monkeypatch.setenv(search_control.STOP_ENV, str(stop))
    symbols, coords = _h2s()
    s = _settings(method="level")
    seen = []
    engine = _MorseEngine(stop)
    md = tmp_path / "md"

    def on_event(event, payload):
        seen.append((event, payload))
        if event == "md_segment" and payload["segment"] == 2:
            engine.stop_at_call = engine.calls + 4 * 10      # 4 steps per frame: 10 frames into the 3rd segment
    traj = nr.run_engine_md(symbols, coords, charge=0, multiplicity=1, settings=s, engine=engine, workdir=md,
                            on_event=on_event)
    _, frames, _ = nr.read_xyz_frames(traj)
    assert 50 < len(frames) < 75                       # two whole segments and part of the third
    stopped = json.loads((md / "stopped.json").read_text())
    assert stopped["segment"] == 2 and abs(stopped["time_ps"] - len(frames) * 0.002) < 1e-9
    assert [e for e, _ in seen if e == "md_stopped"] == ["md_stopped"]
    assert (md / "segment_002.partial.xyz").exists() and not (md / "segment_002.xyz").exists()
    # Run longer (or Resume): the stopped segment runs again from its start, and the stop's leftovers go.
    stop.unlink()
    traj = nr.run_engine_md(symbols, coords, charge=0, multiplicity=1, settings=s, engine=_MorseEngine(), workdir=md)
    assert len(nr.read_xyz_frames(traj)[1]) == 100
    assert not (md / "stopped.json").exists() and not list(md.glob("*.partial.xyz"))


_FAKE_XTB = r'''#!{python}
import pathlib, shutil, sys, time
args = sys.argv[1:]
src = pathlib.Path(args[0]).read_text()
if "--opt" in args:
    pathlib.Path("xtbopt.xyz").write_text(src)
    sys.exit(0)
inp = pathlib.Path(args[args.index("--input") + 1]).read_text()
n = int(src.split()[0])
body = "".join(src.splitlines(keepends=True)[2:2 + n])
t = float(inp.split("time=")[1].split()[0]) * 1000.0
dump = float(inp.split("dump=")[1].split()[0])
with open("xtb.trj", "w") as trj:
    for k in range(int(round(t / dump))):
        trj.write(f"{{n}}\n energy: -1.0\n{{body}}")
        trj.flush()
        time.sleep(0.05)
pathlib.Path("mdrestart").write_text("restart\n")
'''


def test_a_stop_kills_the_running_xtb_md_segment_and_keeps_its_frames(tmp_path, monkeypatch):
    exe = tmp_path / "xtb"
    exe.write_text(_FAKE_XTB.format(python=sys.executable))
    exe.chmod(0o755)
    stop = tmp_path / "stop"
    monkeypatch.setenv(search_control.STOP_ENV, str(stop))
    symbols, coords = _h2s()
    s = _settings(method="gfn2")       # 4 segments of 25 frames, about 1.3 s each

    def stop_in_segment_1(event, payload):
        if event == "md_segment" and payload["segment"] == 1:
            threading.Timer(0.6, stop.touch).start()

    t0 = time.monotonic()
    traj = nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=s, workdir=tmp_path / "md",
                             executable=str(exe), on_event=stop_in_segment_1)
    assert time.monotonic() - t0 < 5.0              # not the 4 segments' 5+ s
    _, frames, _ = nr.read_xyz_frames(traj)
    assert 25 < len(frames) < 50
    assert json.loads((tmp_path / "md" / "stopped.json").read_text())["segment"] == 1


def test_refinement_runs_species_and_reactions_at_once_and_records_them_here(monkeypatch):
    """Each species and reaction is refined on a copy in a worker thread; the
    records, their energy changes and the *_refined events (which write
    network.json) are updated from the calling thread only."""
    from types import SimpleNamespace as NS

    threads, lock = set(), threading.Lock()

    def fake_species(rec, cuts, engine, sp_dir, maxiter, name=None, live=False):
        with lock:
            threads.add(threading.current_thread().name)
        time.sleep(0.2)
        rec.energy, rec.file, rec.note = -1.0 - rec.id, f"sp_{rec.id}.xyz", ""

    def fake_reaction(rxn, cuts, engine, d, maxiter, live=False):
        with lock:
            threads.add(threading.current_thread().name)
        time.sleep(0.2)
        rxn.complex = {"reactant": "r.xyz", "product": "p.xyz", "delta_e_kcal": 1.0}

    monkeypatch.setattr(nr, "refine_species_one", fake_species)
    monkeypatch.setattr(nr, "refine_reaction_one", fake_reaction)
    monkeypatch.setattr(nr, "_species_cuts", lambda *a: [])
    monkeypatch.setattr(nr, "_reaction_cuts", lambda *a: [])
    species = [nr.SpeciesRecord(id=i, smiles=f"S{i}", charge=0, multiplicity=1, formula="H", first_fs=0.0) for i in range(4)]
    reactions = [nr.ReactionRecord(id=k, reactants=[k], products=[k + 1], shuttles=[], label=f"S{k} -> S{k + 1}",
                                 first_fs=0.0)
                 for k in range(3)]

    class GXTBCalculator:          # known to be thread-safe, by name
        pass

    main = threading.current_thread()
    events = []
    t0 = time.monotonic()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as out:
        nr.refine(["H"], np.zeros((1, 1, 3)), None, species, reactions, GXTBCalculator(), Path(out),
                  on_event=lambda e, p: events.append((e, threading.current_thread() is main)), workers=7)
    assert time.monotonic() - t0 < 0.6 and len(threads) > 1          # 7 pieces of 0.2 s, at once
    assert [s.energy for s in species] == [-1.0, -2.0, -3.0, -4.0]
    assert all(r.complex["reactant"] == "r.xyz" for r in reactions)
    assert [round(r.delta_e_kcal, 3) for r in reactions] == [round(-627.5094740631, 3)] * 3
    assert all(on_main for e, on_main in events if e.endswith("_refined"))
    assert sorted(e for e, _ in events if e.endswith("_refined")) == ["reaction_refined"] * 3 + ["species_refined"] * 4

    # An engine not known to be thread-safe: one at a time, as before.
    threads.clear()
    with tempfile.TemporaryDirectory() as out:
        nr.refine(["H"], np.zeros((1, 1, 3)), None, species, reactions, NS(), Path(out), workers=7)
    assert threads == {main.name}


_FAILING_XTB = r'''#!{python}
import os, pathlib, sys
args = sys.argv[1:]
src = pathlib.Path(args[0]).read_text()
if "--opt" in args:
    pathlib.Path("xtbopt.xyz").write_text(src)
    sys.exit(0)
calls = pathlib.Path("calls")
k = int(calls.read_text()) if calls.exists() else 0
calls.write_text(str(k + 1))
inp = pathlib.Path(args[args.index("--input") + 1]).read_text()
with open("runs.log", "a") as log:
    log.write(f"{{args[0]}} {{inp.split('restart=')[1].split()[0]}} {{pathlib.Path('xtbrestart').exists()}}\n")
n = int(src.split()[0])
body = "".join(src.splitlines(keepends=True)[2:2 + n])
t = float(inp.split("time=")[1].split()[0]) * 1000.0
dump = float(inp.split("dump=")[1].split()[0])
fail = {{int(c) for c in os.environ.get("FAKE_XTB_FAIL", "").split(",") if c}}
mode = os.environ.get("FAKE_XTB_MODE", "zero")
with open("xtb.trj", "w") as trj:
    for f in range(int(round(t / dump))):
        # 10 frames in (-1: at once, every time) the SCC fails (energy 0, gnorm 0, and xtb goes on),
        # or lands on a wrong solution (tens of Eh up, and xtb stops: "MD is unstable").
        bad = (k in fail and f >= (0 if mode == "at_once" else 10)) or -1 in fail
        if bad and mode == "crash":
            print("abnormal termination of xtb", file=sys.stderr)
            sys.exit(128)
        if bad and mode == "jump":
            trj.write(f"{{n}}\n energy: -60.0 gnorm: 20.0\n{{body}}")
            print("MD is unstable, emergency exit")
            break
        e = 0.0 if bad else -100.0 - 0.01 * f
        trj.write(f"{{n}}\n energy: {{e:.12f}} gnorm: {{abs(e):.12f}}\n{{body}}")
pathlib.Path("xtbrestart").write_text("charges\n")
pathlib.Path("mdrestart").write_text("restart\n")
'''


def _failing_xtb(tmp_path):
    exe = tmp_path / "xtb"
    exe.write_text(_FAILING_XTB.format(python=sys.executable))
    exe.chmod(0o755)
    return str(exe)


def test_an_scc_failure_in_the_xtb_md_is_cut_off_and_the_segment_continues_from_its_last_good_frame(
        tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_XTB_FAIL", "1,2")      # segment 1 fails, and so does its first retry
    symbols, coords = _h2s()
    seen = []
    md = tmp_path / "md"
    traj = nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=_settings(method="gfn2"),
                             workdir=md, executable=_failing_xtb(tmp_path), on_event=lambda e, p: seen.append((e, p)))
    _, frames, comments = nr.read_xyz_frames(traj)
    assert len(frames) == 100                       # 4 segments of 25 frames: none lost, none of zero force
    assert not any("energy: 0.0" in c for c in comments)
    assert len(nr.read_xyz_frames(md / "segment_001.xyz")[1]) == 25
    runs = (md / "runs.log").read_text().split("\n")
    # Segment 1 ran from the restart, then twice more from its last good frame
    # with fresh velocities and no stale charges; segment 2 from segment 1's end.
    assert runs[1:5] == ["reactor.xyz true True", "resume.xyz false False", "resume.xyz false False",
                         "reactor.xyz true True"]
    assert sum("broke down" in p.get("message", "") for e, p in seen if e == "warning") == 2


def test_an_energy_jump_and_unstable_stop_in_the_xtb_md_is_cut_off_and_continued(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_XTB_FAIL", "2")
    monkeypatch.setenv("FAKE_XTB_MODE", "jump")
    symbols, coords = _h2s()
    seen = []
    md = tmp_path / "md"
    traj = nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=_settings(method="gfn2"),
                             workdir=md, executable=_failing_xtb(tmp_path), on_event=lambda e, p: seen.append((e, p)))
    _, frames, comments = nr.read_xyz_frames(traj)
    assert len(frames) == 100 and not any("-60.0" in c for c in comments)
    assert (md / "runs.log").read_text().split("\n")[2:5] == ["reactor.xyz true True", "resume.xyz false False",
                                                               "reactor.xyz true True"]
    assert [p["message"] for e, p in seen if e == "warning" and "broke down" in p["message"]] == [
        "xtb's MD broke down in segment 2 (energy jumped by +40.1 Eh in one frame); continued from the last good "
        "frame"]


def test_an_xtb_md_whose_scc_keeps_failing_ends_early_instead_of_failing_or_falling_apart(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_XTB_FAIL", "-1")
    symbols, coords = _h2s()
    seen = []
    traj = nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=_settings(method="gfn2"),
                             workdir=tmp_path / "md", executable=_failing_xtb(tmp_path),
                             on_event=lambda e, p: seen.append((e, p)))
    assert len(nr.read_xyz_frames(traj)[1]) == 0                  # no frame of zero forces handed on
    warnings = [p["message"] for e, p in seen if e == "warning"]
    assert any("ended early" in w for w in warnings)
    assert [e for e, _ in seen].count("md_segment") == 3            # 3 segments without progress, then it ends


def test_a_segment_that_keeps_breaking_down_is_skipped_and_the_next_starts_where_it_got_to(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_XTB_FAIL", "1,2,3,4,5,6")    # segment 1: every try fails at once
    monkeypatch.setenv("FAKE_XTB_MODE", "at_once")
    symbols, coords = _h2s()
    seen = []
    md = tmp_path / "md"
    traj = nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=_settings(method="gfn2"),
                             workdir=md, executable=_failing_xtb(tmp_path), on_event=lambda e, p: seen.append((e, p)))
    _, frames, comments = nr.read_xyz_frames(traj)
    assert len(frames) == 75 and not any("energy: 0.0" in c for c in comments)     # segment 1 skipped
    assert (md / "segment_001.cut.xyz").exists() and not (md / "segment_001.mdrestart").exists()
    assert (md / "runs.log").read_text().split("\n")[7] == "segment_001.cut.xyz false False"
    assert any("the rest of that segment is skipped" in p["message"] for e, p in seen if e == "warning")
    # A rerun into the folder (resume) keeps the skip and runs nothing again.
    n_runs = len((md / "runs.log").read_text().split("\n"))
    nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=_settings(method="gfn2"),
                      workdir=md, executable=_failing_xtb(tmp_path))
    assert len((md / "runs.log").read_text().split("\n")) == n_runs


def test_an_xtb_md_run_that_aborts_mid_segment_is_a_breakdown_too(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_XTB_FAIL", "2")
    monkeypatch.setenv("FAKE_XTB_MODE", "crash")
    symbols, coords = _h2s()
    seen = []
    traj = nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=_settings(method="gfn2"),
                             workdir=tmp_path / "md", executable=_failing_xtb(tmp_path),
                             on_event=lambda e, p: seen.append((e, p)))
    assert len(nr.read_xyz_frames(traj)[1]) == 100
    assert any("abnormal termination" in p["message"] for e, p in seen if e == "warning")



def test_a_squeeze_xtb_cannot_converge_opens_the_piston_for_the_rest_of_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_XTB_FAIL", "1,2,3,4,5,6")    # segment 1 (the narrow wall) fails at once, every try
    monkeypatch.setenv("FAKE_XTB_MODE", "at_once")
    symbols, coords = _h2s()
    seen = []
    md = tmp_path / "md"
    nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=_settings(method="gfn2"),
                      workdir=md, executable=_failing_xtb(tmp_path), on_event=lambda e, p: seen.append((e, p)))
    radii = [p["radius"] for e, p in seen if e == "md_segment"]
    assert radii[:2] == [5.0, 3.5] and abs(radii[3] - 3.5 * 1.15) < 1e-9      # the next squeeze is gentler
    assert abs(json.loads((md / "piston.json").read_text())["floor"] - 3.5 * 1.15) < 1e-9
    assert any("piston now closes only to 4.02" in p["message"] for e, p in seen if e == "warning")


def test_anchors_are_held_together_and_the_piston_pushes_only_the_molecules_without_one(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_XTB_FAIL", raising=False)
    symbols, coords = _h2s()                 # 4 H2: atoms 0-1, 2-3, 4-5, 6-7
    md = tmp_path / "md"
    nr.run_reactor_md(symbols, coords, charge=0, multiplicity=1, settings=_settings(method="gfn2"),
                      workdir=md, executable=_failing_xtb(tmp_path), anchors=[2, 0])
    for name in ("opt.inp", "md.inp"):
        inp = (md / name).read_text()
        assert inp.split("sphere: ")[1].split("\n")[0].split(", ")[1] == "5,6,7,8"    # the piston
        assert inp.split("sphere: ")[2].split("\n")[0] == f"{5.0 * nr.ANGSTROM_TO_BOHR:.6f}, 2,4"  # wide, always
        assert "$constrain" in inp and "atoms: 1,3\n" in inp
