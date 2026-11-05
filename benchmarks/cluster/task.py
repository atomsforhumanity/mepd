#!/usr/bin/env python3
"""One benchmark task in an afh pod (written into the run by `bench.py build`; stdlib only).

Reads $TASK_JSON (one line of tasks.jsonl, see bench.py), runs `mepd discovery expand` on the
task's seed in the task's node-local scratch directory with the run's code snapshot
($RUN_DIR/code, copied to scratch first), and publishes into out/:

  result.json          what ran and how it ended: argv, seeds, start/end epoch, wall, CPU
                       (user+sys of the expand process tree), exit code, killed (why), stop reason
  summary.json         the run's summary (also inside run.tgz)
  expand.log           expand's stdout and stderr
  network_aware.jsonl  the explorer's event log (MEPD_NETWORK_AWARE_LOG)
  run.tgz              the run directory without orbitals/
  snapshots.tgz        every version of live/kinetics.json and run/summary_partial.json the run
                       wrote, as kinetics_<mtime_ns>.json / summary_partial_<mtime_ns>.json
                       (the settle time is read from them), plus live/events.jsonl

The live kinetics only exist while a live viewer is attached, so MEPD_DRIVE_CHAIN_DIR points at
./live (what `mepd web` jobs do); a thread copies each new version of the two files once a second.
A hard timeout (task "hard_timeout", seconds) stops expand's whole process group; so does the
pod's SIGTERM (afh task_timeout). Exit 0 when summary.json exists, else 1.
"""
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

POLL_S = 1.0
WATCH = {"kinetics": Path("live/kinetics.json"), "summary_partial": Path("run/summary_partial.json")}


def snapshotter(stop: threading.Event, snap: Path) -> None:
    """Copy each new version (new mtime) of the watched files into `snap`."""
    seen = {}
    while True:
        for name, fp in WATCH.items():
            try:
                st = fp.stat()
            except OSError:
                continue
            if seen.get(name) == st.st_mtime_ns:
                continue
            try:
                data = fp.read_bytes()
                json.loads(data)        # written atomically, but a half-read file is retried next tick
            except (OSError, ValueError):
                continue
            seen[name] = st.st_mtime_ns
            (snap / f"{name}_{st.st_mtime_ns}.json").write_bytes(data)
        if stop.is_set():
            return
        stop.wait(POLL_S)


def main() -> int:
    raw = os.environ.get("TASK_JSON", "task.json")       # afh: the path of task.json (or the JSON itself)
    task = json.loads(Path(raw).read_text() if not raw.lstrip().startswith("{") else raw)
    run_dir = Path(os.environ["RUN_DIR"])
    nthreads = int(os.environ.get("NTHREADS", "1"))
    out = Path("out")
    snap = Path("snapshots")
    for d in (out, snap, Path("tmp"), Path("live")):
        d.mkdir(exist_ok=True)
    shutil.copytree(run_dir / task.get("code", "code"), "code")   # imports from local scratch, not NFS
    shutil.copy(run_dir / task["seed_xyz"], "seed.xyz")
    argv = [a.replace("{RUN_DIR}", str(run_dir)) for a in task["argv"]]
    cmd = [sys.executable, "-m", "mepd.cli", "discovery", "expand", "seed.xyz", *argv,
           "--workers", str(nthreads), "-o", "run"]
    env = dict(os.environ, PYTHONPATH=str(Path("code").resolve()), PYTHONUNBUFFERED="1",
               TMPDIR=str(Path("tmp").resolve()), MEPD_DRIVE_CHAIN_DIR=str(Path("live").resolve()),
               MEPD_NETWORK_AWARE_LOG=str((out / "network_aware.jsonl").resolve()),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    stop = threading.Event()
    watcher = threading.Thread(target=snapshotter, args=(stop, snap), daemon=True)
    killed = {"why": None}

    start = time.time()
    with open(out / "expand.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    def kill_group(why: str) -> None:
        if killed["why"] is None:
            killed["why"] = why
        for sig, grace in ((signal.SIGTERM, 60.0), (signal.SIGKILL, 0.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            t_end = time.time() + grace
            while grace and proc.poll() is None and time.time() < t_end:
                time.sleep(1.0)
            if proc.poll() is not None:
                return

    signal.signal(signal.SIGTERM, lambda *_: kill_group("SIGTERM (afh task_timeout)"))
    watcher.start()
    hard = float(task.get("hard_timeout") or 0)
    while proc.poll() is None:
        if hard and time.time() - start > hard:
            kill_group(f"hard timeout {hard:.0f} s")
            break
        time.sleep(1.0)
    rc = proc.wait()
    end = time.time()
    try:
        os.killpg(proc.pid, signal.SIGKILL)      # stragglers of expand's session (none normally)
    except ProcessLookupError:
        pass
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    stop.set()
    watcher.join(timeout=30)

    summary = Path("run/summary.json")
    stopped = None
    if summary.is_file():
        shutil.copy(summary, out / "summary.json")
        try:
            stopped = (json.loads(summary.read_text()).get("rounds") or [{}])[-1].get("stopped")
        except ValueError:
            pass
    commit_fp = Path("code/COMMIT")
    result = {"id": task["id"], "system": task["system"], "config": task["config"],
              "replicate": task["replicate"], "seeds": task.get("seeds"), "argv": cmd,
              "code_commit": commit_fp.read_text().strip() if commit_fp.is_file() else None,
              "harness_commit": task.get("harness_commit"), "budget_s": task.get("budget"),
              "start": start, "end": end, "wall_s": end - start, "cpu_s": ru.ru_utime + ru.ru_stime,
              "exit": rc, "killed": killed["why"], "stopped": stopped, "summary": summary.is_file(),
              "nthreads": nthreads, "host": os.environ.get("HOSTNAME")}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    if Path("live/events.jsonl").is_file():
        shutil.copy("live/events.jsonl", snap / "events.jsonl")
    with tarfile.open(out / "snapshots.tgz", "w:gz") as tf:
        tf.add(snap, arcname="snapshots")
    if Path("run").is_dir():
        with tarfile.open(out / "run.tgz", "w:gz") as tf:
            tf.add("run", arcname="run", filter=lambda ti: None if ti.name.startswith("run/orbitals") else ti)
    return 0 if summary.is_file() else 1


if __name__ == "__main__":
    sys.exit(main())
