"""Benchmark runs of `mepd discovery expand` on the AFH cluster: build, submit, pull, score.

    P=/home/user/astera/mepd/.venv/bin/python; B=benchmarks/cluster/bench.py
    $P $B build  RUN --commit SHA [--config NAME[=ARGS]]... [--systems a,b] [--replicates 5]
                 [--group 'a,b:default,T1800' --group 'c,d:default'] (instead of --systems/--config)
                 [--budget 1800] [--threads 32] [--parallelism 8] [--memory 64Gi] [--control]
    $P $B submit RUN           # afh push + afh submit
    $P $B pull   RUN           # afh pull into the run's folder
    $P $B score  RUN           # unpack and score every finished task, write aggregate.json + table.md
    $P $B gate   BASE_RUN:CONFIG CAND_RUN:CONFIG     # see gate.py

A run is one afh run (folder `<scoreboard>/afh/RUN/`, results in `<scoreboard>/results/RUN/`):
  run.yaml, task.py (the pod-side wrapper), tasks.jsonl (one task per system x configuration x
  replicate), code/ (git archive of the commit's mepd/, with code/COMMIT), inputs/ (seeds, profiles
  and the ORCA steering profile from this branch's benchmarks/), manifest.json.

Every task: `--steer access --schedule async --generator reactor --steer-energies orca_steer.toml`
(wB97X-D3BJ/def2-TZVP single points), the system's profile, seed and kinetics conditions
(score.SYSTEMS), `--time-budget B --workers $NTHREADS`, then the configuration's own arguments,
then the replicate's seeds.

Configurations: a name from CONFIGS, or NAME=ARGS with the extra expand arguments spelled out
(e.g. 'T1600=--generator-option temperature=1600').

Replicates differ only in their RNG seeds, S = 1000 x replicate (replicate 0 = the code's defaults):
  --generator-option seed=S       the reactor's Langevin MD and Collide pulls: trajectory k of batch b
                                  uses seed S + b*n + k (n = trajectories per batch = workers), so
                                  seeds 1000 apart never share a trajectory for n*max_batches < 1000;
  --channels-option random_seed=S channel sampling's RDKit conformer embedding.
Fixed in every replicate: the profile's gi_inputs.random_seed (0) and everything else. Not seeded:
the async queue's timing (which job finishes first), so a replicate rerun with its seeds is not
guaranteed to repeat; `--control` adds one rerun of replicate 0 per system for the first
configuration to measure that.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
TEAM = Path("/home/user/astera/data/runs/explorer-team/scoreboard")
AFH_DIR, RESULTS = TEAM / "afh", TEAM / "results"
IMAGE = "192.168.8.6:5000/mepd-orca-uma:2026-10-28a"

sys.path.insert(0, str(BENCH))
import score  # noqa: E402  (benchmarks/score.py: the suite, metrics and audit)

CONFIGS = {
    "default": [],                                                  # the code's own reactor settings
    "T1800": ["--generator-option", "temperature=1800"],
    "T2200": ["--generator-option", "temperature=2200"],
}
COMMON = ["--steer", "access", "--schedule", "async", "--generator", "reactor",
          "--steer-energies", "{RUN_DIR}/inputs/orca_steer.toml"]


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def _config(spec: str) -> tuple[str, list]:
    name, sep, args = spec.partition("=")
    if sep:
        return name, shlex.split(args)
    if name not in CONFIGS:
        raise SystemExit(f"unknown configuration {name!r}: one of {sorted(CONFIGS)} or NAME=ARGS")
    return name, list(CONFIGS[name])


def system_argv(name: str, sysdef: dict, budget: float) -> list:
    c = sysdef["conditions"]
    return ["--charge", str(sysdef.get("charge", 0)), "--multiplicity", str(sysdef.get("multiplicity", 1)), *COMMON,
            "--inputs", "{RUN_DIR}/inputs/" + Path(sysdef["profile"]).name, "--standard-state", c["standard_state"],
            "--cutoff-temperature", f"{c['temperature_K']:g}", "--cutoff-time", f"{c['time_s']:g}",
            "--temperature", f"{c['temperature_K']:g}", "--time", f"{c['time_s']:g}", "--time-budget", f"{budget:g}"]


def seed_argv(replicate: int) -> tuple[dict, list]:
    s = 1000 * int(replicate)
    return ({"reactor_seed": s, "channels_random_seed": s},
            ["--generator-option", f"seed={s}", "--channels-option", f"random_seed={s}"])


def build(run: str, commit: str, groups: list, replicates: int, budget: float, threads: int,
          parallelism: int, memory: str, control: bool, hard_factor: float = 3.0) -> Path:
    """`groups`: [(systems, [(config name, args)])] in priority order (pods start in task order)."""
    suite = score.load_systems()
    systems = list(dict.fromkeys(n for g in groups for n in g[0]))
    configs = list({c: a for g in groups for c, a in g[1]}.items())
    sha = _git("rev-parse", commit)
    harness = _git("rev-parse", "HEAD") + ("+dirty" if _git("status", "--porcelain", "--", "benchmarks") else "")
    d = AFH_DIR / run
    if d.exists():
        raise SystemExit(f"{d} exists: pick another run name")
    (d / "inputs").mkdir(parents=True)
    (d / "code").mkdir()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", sha, "mepd"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(d / "code")], input=archive, check=True)
    (d / "code" / "COMMIT").write_text(sha + "\n")
    shutil.copy(HERE / "task.py", d / "task.py")
    for name in systems:
        sd = suite[name]
        for f in (sd["seed"], sd["profile"]):
            shutil.copy(BENCH / f, d / "inputs" / Path(f).name)
    shutil.copy(BENCH / "orca_steer.toml", d / "inputs" / "orca_steer.toml")
    tasks = []
    # pods start in task order: group by group (priority), and in a group replicate by replicate, every system
    # (as given: long ones first) and configuration in each, so all arms meet the same cluster load and a run
    # cut short has every arm
    order = [(r, name, c) for g_systems, g_configs in groups for r in range(replicates)
             for name in g_systems for c in g_configs]
    for r, name, (cname, cargs) in order:
        seeds, sargs = seed_argv(r)
        tasks.append({"id": f"{name}-{cname}-r{r}", "system": name, "config": cname, "replicate": r,
                      "seeds": seeds, "seed_xyz": f"inputs/{Path(suite[name]['seed']).name}", "budget": budget,
                      "hard_timeout": hard_factor * budget,
                      "argv": system_argv(name, suite[name], budget) + cargs + sargs,
                      "harness_commit": harness})
    if control:   # replicate 0 again: what the seeds don't fix (queue timing); the first group, last in order
        cname, cargs = groups[0][1][0]
        for name in groups[0][0]:
            seeds, sargs = seed_argv(0)
            tasks.append({"id": f"{name}-{cname}-r0b", "system": name, "config": cname, "replicate": 0,
                          "control_of": f"{name}-{cname}-r0", "seeds": seeds,
                          "seed_xyz": f"inputs/{Path(suite[name]['seed']).name}", "budget": budget,
                          "hard_timeout": hard_factor * budget,
                          "argv": system_argv(name, suite[name], budget) + cargs + sargs, "harness_commit": harness})
    (d / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    timeout = int(hard_factor * budget + 900)
    what = "; then ".join(f"{','.join(gs)} x {','.join(c for c, _ in gc)}" for gs, gc in groups)
    desc = (f"scoreboard {run}: discovery expand on explorer {sha[:7]}, {what}, {replicates} replicates each"
            f"{' + controls' if control else ''}, {threads} threads, budget {budget:g} s "
            f"(explorer-team phase 0, benchmarks/cluster/bench.py)")
    (d / "run.yaml").write_text(
        f'description: "{desc}"\nimage: {IMAGE}\ncommand: "exec python3 $RUN_DIR/task.py"\n'
        f"threads_per_task: {threads}\ncpus_per_pod: {threads}\nmemory_per_pod: {memory}\n"
        f"shards: {len(tasks)}\nparallelism: {parallelism}\ntask_timeout: {timeout}\npriority: batch\n"
        "notify: auto\nenv: {SELLA_DISABLE_GPU: 1, OMP_NUM_THREADS: 1, MKL_NUM_THREADS: 1, OPENBLAS_NUM_THREADS: 1}\n")
    (d / "manifest.json").write_text(json.dumps(
        {"run": run, "commit": sha, "harness_commit": harness, "image": IMAGE, "configs": dict(configs),
         "systems": {n: suite[n] for n in systems}, "replicates": replicates, "budget_s": budget,
         "threads": threads, "control": control, "tasks": len(tasks), "suite_source": score.SUITE_SOURCE},
        indent=1))
    print(f"{d}: {len(tasks)} tasks, explorer {sha[:7]}, harness {harness[:7]}")
    return d


def afh(*args, check=True):
    print("+ afh", " ".join(args), flush=True)
    return subprocess.run(["afh", *args], check=check)


def score_run(runs, rescore: bool = False, name: str | None = None) -> dict:
    """Unpack and score every finished task of the run(s); aggregate per (system, configuration) into
    results/<name> (default: the run's own name; several runs of one commit = one result set)."""
    import tarfile

    runs = [runs] if isinstance(runs, str) else list(runs)
    name = name or runs[0]
    suite = score.load_systems()
    manifests, scores = [], []
    for run in runs:
        scores += _score_tasks(run, suite, rescore, manifests)
    commits = sorted({m["commit"] for m in manifests})
    agg = score.aggregate(scores, suite)
    agg["run"], agg["runs"], agg["commit"] = name, runs, commits[0] if len(commits) == 1 else commits
    agg["harness_commit"] = sorted({m["harness_commit"] for m in manifests})
    agg["suite_source"] = score.SUITE_SOURCE
    res = RESULTS / name
    res.mkdir(parents=True, exist_ok=True)
    (res / "aggregate.json").write_text(json.dumps(agg, indent=1, default=str))
    (res / "table.md").write_text(score.table(agg, scores))
    print(score.table(agg, scores))
    return agg


def _score_tasks(run: str, suite: dict, rescore: bool, manifests: list) -> list:
    import tarfile

    d, res = AFH_DIR / run, RESULTS / run
    manifests.append(json.loads((d / "manifest.json").read_text()))
    tasks = [json.loads(line) for line in (d / "tasks.jsonl").read_text().splitlines() if line.strip()]
    scores = []
    for t in tasks:
        src = next((p for p in (d / "output" / t["id"], d / "errors" / t["id"]) if p.is_dir()), None)
        if src is None:
            continue
        dst = res / t["id"]
        if (dst / "score.json").is_file() and not rescore:
            scores.append(json.loads((dst / "score.json").read_text()))
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for tgz in ("run.tgz", "snapshots.tgz"):
            if (src / tgz).is_file() and not (dst / tgz.split(".")[0]).is_dir():
                with tarfile.open(src / tgz) as tf:     # (the reactor's trajectories stay in the archive)
                    tf.extractall(dst, members=[m for m in tf if not m.name.startswith("run/reactor/")],
                                  filter="data")
        for f in ("result.json", "_meta.json", "expand.log", "network_aware.jsonl"):
            if (src / f).is_file():
                shutil.copy(src / f, dst / f)
        o = score.score_task(t, suite[t["system"]], dst)
        o["afh"] = "output" if src.parent.name == "output" else "errors"
        o["run"] = run
        (dst / "score.json").write_text(json.dumps(o, indent=1, default=str))
        scores.append(o)
    return scores


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("run")
    b.add_argument("--commit", required=True)
    b.add_argument("--config", action="append", default=[])
    b.add_argument("--systems", default="")
    b.add_argument("--group", action="append", default=[],
                   help="SYSTEMS:CONFIGS (comma-separated each), repeatable, in priority order; replaces --systems/--config")
    b.add_argument("--replicates", type=int, default=5)
    b.add_argument("--budget", type=float, default=1800.0)
    b.add_argument("--threads", type=int, default=32)
    b.add_argument("--parallelism", type=int, default=8)
    b.add_argument("--memory", default="64Gi")
    b.add_argument("--control", action="store_true")
    for name in ("submit", "pull", "status"):
        sub.add_parser(name).add_argument("run")
    s = sub.add_parser("score")
    s.add_argument("run", nargs="+")
    s.add_argument("--rescore", action="store_true")
    s.add_argument("--as", dest="name", default=None, help="results name for several runs (default: the first)")
    g = sub.add_parser("gate")
    g.add_argument("base")
    g.add_argument("cand")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if a.group:
            groups = [([x for x in g.split(":")[0].split(",") if x], [_config(c) for c in g.split(":")[1].split(",")])
                      for g in a.group]
        else:
            groups = [([x for x in a.systems.split(",") if x] or list(score.load_systems()),
                       [_config(c) for c in (a.config or ["default"])])]
        build(a.run, a.commit, groups, a.replicates, a.budget, a.threads, a.parallelism, a.memory, a.control)
    elif a.cmd == "submit":
        afh("push", str(AFH_DIR / a.run), a.run)
        afh("submit", a.run)
    elif a.cmd == "pull":
        afh("pull", a.run, str(AFH_DIR / a.run))
    elif a.cmd == "status":
        afh("status", a.run, check=False)
    elif a.cmd == "score":
        score_run(a.run, a.rescore, a.name)
    elif a.cmd == "gate":
        import gate
        return gate.main([a.base, a.cand])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
