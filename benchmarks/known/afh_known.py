"""The --known follow-up benchmark on the AFH cluster (F0007 with the scoreboard's settings).

    P=/home/user/astera/mepd/.venv/bin/python; K=benchmarks/known/afh_known.py
    $P $K stage1 RUN_DIR                    # the seed (Cp + methyl acrylate complex), budget 1800 s
    $P $K stage2 RUN_DIR STAGE1_TASK_OUTPUT # two follow-ups from stage 1's endo adduct, same RNG seeds:
                                            #   kn-plain (no --known) and kn-known (--known <stage 1's run folder>)
    $P $K analyze STAGE2_RUN_DIR OUT.json   # wall time, TS searches / single points / Hessians run vs reused,
                                            #   the endo adduct's onward chemistry and its back reaction

Each task is the scoreboard's: its pod wrapper task.py (team2/scoreboard, SCOREBOARD_COMMIT) runs `mepd discovery
expand` with bench.py's F0007 arguments (COMMON + system_argv + replicate 0's seeds) from this commit's mepd/
(code/, git archive), inside the explorer team's recorder (recorded.py, provenance into out/provenance/).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCOREBOARD = Path("/home/user/astera/data/worktrees/team-scoreboard")
SCOREBOARD_COMMIT = "4b257f5"
BIN = Path("/home/user/astera/data/runs/explorer-team/bin")
IMAGE = "192.168.8.6:5000/mepd-orca-uma-axtb:2026-11-05"
BUDGET = 1800.0
THREADS = 32
# bench.py's F0007 task (COMMON, system_argv, seed_argv(0)) at SCOREBOARD_COMMIT
ARGV = ["--charge", "0", "--multiplicity", "1", "--steer", "access", "--schedule", "async", "--generator", "reactor",
        "--steer-energies", "{RUN_DIR}/inputs/orca_steer.toml", "--inputs", "{RUN_DIR}/inputs/f0007_mlpgi.toml",
        "--standard-state", "1M", "--cutoff-temperature", "353.15", "--cutoff-time", "23400", "--temperature",
        "353.15", "--time", "23400", "--time-budget", f"{BUDGET:g}", "--generator-option", "seed=0",
        "--channels-option", "random_seed=0"]


def _git(*args, cwd=REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _run_dir(d: Path, name: str, tasks: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    commit = _git("rev-parse", "HEAD")
    code = d / "code"
    if not code.exists():
        code.mkdir()
        archive = subprocess.run(["git", "archive", commit, "mepd"], cwd=REPO, capture_output=True, check=True).stdout
        subprocess.run(["tar", "x", "-C", str(code)], input=archive, check=True)
        (code / "COMMIT").write_text(commit + "\n")
    (d / "inputs").mkdir(exist_ok=True)
    for f in ("f0007_mlpgi.toml", "orca_steer.toml", "f0007_start_complex.xyz"):
        (d / "inputs" / f).write_text(_git("show", f"{SCOREBOARD_COMMIT}:benchmarks/{f}", cwd=SCOREBOARD) + "\n")
    (d / "task.py").write_text(_git("show", f"{SCOREBOARD_COMMIT}:benchmarks/cluster/task.py", cwd=SCOREBOARD) + "\n")
    for f in ("recorded.py", "manifest.py"):
        shutil.copy(BIN / f, d / f)
    (d / "run.yaml").write_text(f'''description: "explorer-team reuse: --known follow-up benchmark ({name}) on F0007, mepd {commit[:7]}, \
scoreboard task.py {SCOREBOARD_COMMIT}, 32 threads, budget {BUDGET:g} s (benchmarks/known/afh_known.py)"
image: {IMAGE}
command: "exec python3 $RUN_DIR/recorded.py out --binary orca -- python3 $RUN_DIR/task.py"
threads_per_task: {THREADS}
cpus_per_pod: {THREADS}
memory_per_pod: 32Gi
shards: {len(tasks)}
parallelism: {len(tasks)}
task_timeout: {int(3.5 * BUDGET)}
priority: high
notify: never
env: {{SELLA_DISABLE_GPU: 1, OMP_NUM_THREADS: 1, MKL_NUM_THREADS: 1, OPENBLAS_NUM_THREADS: 1}}
''')
    (d / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (d / "manifest.json").write_text(json.dumps({"run": d.name, "commit": commit, "image": IMAGE,
                                                 "scoreboard_commit": _git("rev-parse", SCOREBOARD_COMMIT, cwd=SCOREBOARD),
                                                 "tasks": tasks}, indent=1))


def _task(tid: str, seed_xyz: str, extra=()) -> dict:
    return {"id": tid, "system": "f0007", "config": tid, "replicate": 0,
            "seeds": {"reactor_seed": 0, "channels_random_seed": 0}, "seed_xyz": seed_xyz, "budget": BUDGET,
            "hard_timeout": 3.0 * BUDGET, "argv": ARGV + list(extra)}


def stage1(d: Path) -> None:
    _run_dir(d, "stage 1: the seed", [_task("kn-seed", "inputs/f0007_start_complex.xyz")])


def _score():
    sys.path.insert(0, str(SCOREBOARD / "benchmarks"))
    import score   # noqa: E402  (the scoreboard's matching of channels to species; worktree at SCOREBOARD_COMMIT)

    return score


def _endo(run: Path) -> int:
    """The lowest endo adduct of a run folder (the scoreboard's own channel matching)."""
    score = _score()
    sysdef = score.load_systems()["f0007"]
    s = json.loads((run / "summary.json").read_text())
    sp = s["species"]
    chans = score.channel_species(sysdef, [x.get("smiles") or "" for x in sp],
                                  lambda k: run / "species" / f"species_{k}.xyz")
    return min(chans["endo"], key=lambda k: sp[k]["energy"])


def stage2(d: Path, stage1_output: Path) -> None:
    assert _git("rev-parse", "HEAD", cwd=SCOREBOARD).startswith(SCOREBOARD_COMMIT)
    tmp = Path(tempfile.mkdtemp())
    with tarfile.open(stage1_output / "run.tgz") as tf:
        tf.extractall(tmp)
    run1 = tmp / "run"
    k = _endo(run1)
    for tid in ("kn-plain", "kn-known"):
        t = d / "tasks" / tid
        t.mkdir(parents=True, exist_ok=True)
        shutil.copy(run1 / "species" / f"species_{k}.xyz", t / "seed.xyz")
    shutil.copytree(run1, d / "tasks" / "kn-known" / "known1", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("orbitals", "reactor"))
    (d / "endo.json").write_text(json.dumps({"stage1_output": str(stage1_output), "endo_species": k}))
    _run_dir(d, f"stage 2: follow-ups from stage 1's endo adduct (species {k})",
             [_task("kn-plain", "tasks/kn-plain/seed.xyz"), _task("kn-known", "tasks/kn-known/seed.xyz", ["--known", "known1"])])
    shutil.rmtree(tmp)


def _facts(out: Path) -> dict:
    """One finished task: its wall time, what it computed and what it took as known, its kinetics."""
    import pickle

    score = _score()
    tmp = Path(tempfile.mkdtemp())
    with tarfile.open(out / "run.tgz") as tf:
        tf.extractall(tmp)
    run = tmp / "run"
    s = json.loads((run / "summary.json").read_text())
    res = json.loads((out / "result.json").read_text())
    ts = [p for p in (run / "ts").glob("*.xyz") if not p.name.endswith(("_irc.xyz", "_ends.xyz"))]
    pts = [json.loads(p.read_text()) for p in (run / "steer_single_points" / "single_points").glob("*.json")]
    with open(run / "expand_state.pkl", "rb") as fh:
        free_cache = pickle.load(fh)["free_cache"]
    reused = (s.get("known") or {}).get("reused") or {"species": 0, "steps": 0, "single_points": 0, "hessians": 0}
    sp = s["species"]
    cp_ma = tuple(sorted(score._canon(x) for x in ("C1=CCC=C1", "C=CC(=O)OC")))
    back = [k for k, x in enumerate(sp) if score.fragments(x.get("smiles") or "") == cp_ma]
    steps0 = [st for st in s["steps"] if 0 in (st["a"], st["b"])]

    def rec(st):
        other = st["b"] if st["a"] == 0 else st["a"]
        bar = st.get("kinetics_barrier_kcal")
        return {"to": other, "smiles": sp[other].get("smiles"), "label": st["label"],
                "known": bool(st["files"].get("known")), "verified": st["files"].get("verified", True),
                "barrier_kcal_from_seed": (bar[0] if st["a"] == 0 else bar[1]) if bar else None,
                "raw_barrier_kcal_from_seed": st["barrier_kcal"][0] if st["a"] == 0 else st["barrier_kcal"][1]}

    onward = sorted([rec(st) for st in steps0], key=lambda r: (r["barrier_kcal_from_seed"] is None,
                                                               r["barrier_kcal_from_seed"] or 0.0))
    populated = [{"index": k, "smiles": x.get("smiles"), "final": x.get("final_concentration"),
                  "max": x.get("max_concentration"), "known": bool(x.get("known"))}
                 for k, x in enumerate(sp) if (x.get("max_concentration") or 0) >= 0.01 or k == 0]
    out_rec = {
        "task": out.name, "wall_s": res.get("wall_s"), "cpu_h": (res.get("cpu_s") or 0) / 3600, "stopped": res.get("stopped"),
        "code_commit": res.get("code_commit"), "mepd_file": res.get("mepd_file"), "host": res.get("host"),
        "species": len(sp), "steps": len(s["steps"]),
        "ts_searches_run": len([p for p in ts if not p.name.startswith("known")]),
        "ts_taken_known": len([p for p in ts if p.name.startswith("known")]),
        "irc_run": len([p for p in (run / "ts").glob("*_irc.xyz") if not p.name.startswith("known")]),
        "single_points_run": sum(1 for p in pts if "known_from" not in p["result"]),
        "single_points_reused": reused["single_points"], "hessians_reused": reused["hessians"],
        "hessians_run": len(free_cache) - reused["hessians"],
        "species_reused": reused["species"], "steps_reused": reused["steps"],
        "known": {k: v for k, v in (s.get("known") or {}).items() if k != "reused"},
        "kinetics_energies": (s.get("rounds") or [{}])[-1].get("kinetics", {}).get("energies"),
        "seed_final": sp[0].get("final_concentration"),
        "populated": populated,
        "back_to_cp_ma": [r for r in onward if r["to"] in back],
        "lowest_steps_from_seed": onward[:8],
        "cp_ma_species": back}
    shutil.rmtree(tmp)
    return out_rec


def analyze(d: Path, dst: Path) -> None:
    rows = [_facts(d / "output" / tid) for tid in ("kn-plain", "kn-known") if (d / "output" / tid).is_dir()]
    dst.write_text(json.dumps(rows, indent=1, default=str))
    for r in rows:
        print(json.dumps({k: r[k] for k in ("task", "wall_s", "stopped", "species", "steps", "ts_searches_run",
                                            "ts_taken_known", "irc_run", "single_points_run", "single_points_reused",
                                            "hessians_run", "hessians_reused", "species_reused", "steps_reused")}))
        print("  back to Cp + MA:", r["back_to_cp_ma"])
        print("  populated:", r["populated"])


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "stage1":
        stage1(Path(rest[0]))
    elif cmd == "stage2":
        stage2(Path(rest[0]), Path(rest[1]))
    elif cmd == "analyze":
        analyze(Path(rest[0]), Path(rest[1]))
    else:
        raise SystemExit(__doc__)
