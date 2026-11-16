"""One benchmark task run locally as the cluster's task.py runs it (same command, environment and files).

    python benchmarks/diagnostics/thermo/local_run.py TASKS.jsonl TASK_ID RUN_DIR OUT_DIR WORKERS
        [--kill-after SECONDS] [--resume]

RUN_DIR: the run set's folder (its inputs/ are what {RUN_DIR} in the task's arguments points at). In OUT_DIR:
seed.xyz, run/, live/, tmp/, expand.log (appended on --resume), network_aware.jsonl, result.json (per leg:
result_<n>.json). mepd is the one on PYTHONPATH (printed). --kill-after: the whole process group is killed
(SIGKILL, as a crash) after that many seconds. --resume: the same command with --resume.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def main(tasks, task_id, run_dir, out_dir, workers, *flags):
    task = next(t for t in map(json.loads, Path(tasks).read_text().splitlines()) if t["id"] == task_id)
    run_dir, out = Path(run_dir).resolve(), Path(out_dir).resolve()
    kill_after = float(flags[flags.index("--kill-after") + 1]) if "--kill-after" in flags else None
    resume = "--resume" in flags
    for d in ("tmp", "live"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if not resume:
        shutil.copy(run_dir / task["seed_xyz"], out / "seed.xyz")
    argv = [a.replace("{RUN_DIR}", str(run_dir)) for a in task["argv"]]
    cmd = [sys.executable, "-m", "mepd.cli", "discovery", "expand", "seed.xyz", *argv, "--workers", str(workers),
           "-o", "run"] + (["--resume"] if resume else [])
    env = dict(os.environ, PYTHONUNBUFFERED="1", TMPDIR=str(out / "tmp"), MEPD_DRIVE_CHAIN_DIR=str(out / "live"),
               MEPD_NETWORK_AWARE_LOG=str(out / "network_aware.jsonl"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    probe = subprocess.run([sys.executable, "-c", "import mepd; print(mepd.__file__)"], env=env, capture_output=True,
                           text=True, cwd=out)
    print("mepd:", probe.stdout.strip(), flush=True)
    start = time.time()
    with open(out / "expand.log", "a") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=out, start_new_session=True)
    killed = None
    while proc.poll() is None:
        if kill_after is not None and time.time() - start > kill_after:
            killed = f"SIGKILL after {kill_after:g} s"
            os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(1.0)
    rc = proc.wait()
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    n = len(list(out.glob("result_*.json")))
    (out / f"result_{n}.json").write_text(json.dumps({
        "id": task_id, "argv": cmd, "mepd_file": probe.stdout.strip(), "start": start, "end": time.time(),
        "wall_s": time.time() - start, "exit": rc, "killed": killed, "workers": int(workers), "resume": resume,
        "summary": (out / "run/summary.json").is_file()}, indent=1))
    print(f"exit {rc}, killed {killed}, wall {time.time() - start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
