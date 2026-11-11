"""Benchmark runs of `mepd discovery expand` on the AFH cluster: build, submit, pull, score.

    P=/home/user/astera/mepd/.venv/bin/python; B=benchmarks/cluster/bench.py
    $P $B build  RUN --commit SHA [--config NAME[=ARGS]]... [--systems a,b] [--replicates 5]
                 [--group 'a,b:default,T1800' --group 'c,d:default'] (instead of --systems/--config)
                 [--budget 1800] [--threads 32] [--parallelism 8] [--memory 64Gi] [--control]
    $P $B submit RUN           # afh push + afh submit
    $P $B pull   RUN           # afh pull into the run's folder
    $P $B score  RUN [RUN2 --as NAME]  # unpack and score every finished task, write aggregate.json + table.md
    $P $B report NAME          # table + gate both ways between two configurations + audit + controls (report.md)
    $P $B local  RUN [--threads 4]   # run RUN's tasks here, one at a time, filed as the cluster's (checks only)
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
IMAGE = "192.168.8.6:5000/mepd-orca-uma-axtb:2026-11-05"   # mepd-orca-uma + axtb (g-xTB in process, develop 37250d5)

sys.path.insert(0, str(BENCH))
import score  # noqa: E402  (benchmarks/score.py: the suite, metrics and audit)

CONFIGS = {
    "default": [],                                                  # the code's own reactor settings
    "cap400": ["--max-species", "400"],
    "T1500": ["--generator-option", "temperature=1500"],
    "T1800": ["--generator-option", "temperature=1800"],
    "T2200": ["--generator-option", "temperature=2200"],
    # F0007 only: the g-xTB program instead of axtb (the profiles use axtb since 37250d5), for the equivalence check
    "program": ["--inputs", "{RUN_DIR}/inputs/f0007_mlpgi_program.toml"],
    # team21's unified generator, spelled as the old reactor (coordinator 2026-11-08): replaces COMMON's --generator
    "nano400": ["--max-species", "400", "--generator", "nanoreactor", "--generator-option", "forces=piston,collide"],
    # team22: linus DFT steering (RI-JK, wB97X-D3BJ/def2-TZVP, memory_gb 1.5) in place of ORCA (coordinator 2026-11-08)
    "linus400": ["--max-species", "400", "--steer-energies", "{RUN_DIR}/inputs/linus_steer_mem1.5.toml"],
}
REPLACING = ("--generator", "--steer-energies")   # a configuration's value for these replaces COMMON's
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


def system_argv(name: str, sysdef: dict, budget: float, concentration: bool = False) -> list:
    """`concentration`: the code has --concentration (molecule-level kinetics): the system's starting
    concentration (score._concentration) is passed."""
    c = sysdef["conditions"]
    return ["--charge", str(sysdef.get("charge", 0)), "--multiplicity", str(sysdef.get("multiplicity", 1)), *COMMON,
            "--inputs", "{RUN_DIR}/inputs/" + Path(sysdef["profile"]).name, "--standard-state", c["standard_state"],
            "--cutoff-temperature", f"{c['temperature_K']:g}", "--cutoff-time", f"{c['time_s']:g}",
            "--temperature", f"{c['temperature_K']:g}", "--time", f"{c['time_s']:g}", "--time-budget", f"{budget:g}",
            *(["--concentration", f"{c['concentration']:g}"] if concentration and c.get("concentration") is not None
              else [])]


def _has_concentration(commit: str) -> bool:
    """Whether `commit`'s discovery expand takes --concentration."""
    try:
        return '"--concentration"' in _git("show", f"{commit}:mepd/discovery/cli_expand.py")
    except subprocess.CalledProcessError:
        return False


def with_config(argv: list, cargs: list) -> list:
    """argv + a configuration's arguments; a configuration's value of an option in REPLACING replaces COMMON's
    (in place)."""
    argv, cargs = list(argv), list(cargs)
    for opt in REPLACING:
        if opt in cargs and opt in argv:
            i, j = argv.index(opt), cargs.index(opt)
            argv[i + 1] = cargs[j + 1]
            del cargs[j:j + 2]
    return argv + cargs


def seed_argv(replicate: int) -> tuple[dict, list]:
    s = 1000 * int(replicate)
    return ({"reactor_seed": s, "channels_random_seed": s},
            ["--generator-option", f"seed={s}", "--channels-option", f"random_seed={s}"])


def build(run: str, commit: str, groups: list, replicates: int, budget: float, threads: int,
          parallelism: int, memory: str, control: bool, hard_factor: float = 3.0, image: str = IMAGE,
          interleave: bool = False, first_replicate: int = 0) -> Path:
    """`groups`: [(systems, [(config name, args)], replicates or None, commit or None)] in priority order (pods
    start in task order); None = `replicates` / `commit`. A group with its own commit runs from code-<sha7>/ and
    labels its configurations <config>-<sha7>, so several commits share one run (and its queue)."""
    suite = score.load_systems()
    groups = [tuple(g) + (None,) * (4 - len(g)) for g in groups]
    sha = _git("rev-parse", commit)
    shas = {g[3]: _git("rev-parse", g[3]) for g in groups if g[3]}
    label = lambda c, g: c if not g[3] else f"{c}-{shas[g[3]][:7]}"
    groups = [(g[0], [(label(c, g), a) for c, a in g[1]], g[2], g[3]) for g in groups]
    systems = list(dict.fromkeys(n for g in groups for n in g[0]))
    configs = list({c: a for g in groups for c, a in g[1]}.items())
    harness = _git("rev-parse", "HEAD") + ("+dirty" if _git("status", "--porcelain", "--", "benchmarks") else "")
    d = AFH_DIR / run
    if d.exists():
        raise SystemExit(f"{d} exists: pick another run name")
    (d / "inputs").mkdir(parents=True)
    (d / "code").mkdir()
    for dirname, full in [("code", sha)] + [(f"code-{x[:7]}", x) for x in sorted(set(shas.values()))]:
        (d / dirname).mkdir(exist_ok=True)
        archive = subprocess.run(["git", "-C", str(REPO), "archive", full, "mepd"], capture_output=True,
                                 check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d / dirname)], input=archive, check=True)
        (d / dirname / "COMMIT").write_text(full + "\n")
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
    if interleave:   # replicate by replicate, system by system, every group's configurations side by side
        order = [(r, name, c, g[3]) for r in range(max(g[2] or replicates for g in groups)) for name in systems
                 for g in groups if name in g[0] and r < (g[2] or replicates) for c in g[1]]
    else:
        order = [(r, name, c, g_commit) for g_systems, g_configs, g_reps, g_commit in groups
                 for r in range(g_reps or replicates) for name in g_systems for c in g_configs]
    # more replicates of an earlier run: numbered (and seeded, 1000 x r) from first_replicate on
    order = [(r + first_replicate, name, c, g) for r, name, c, g in order]
    conc = {c: _has_concentration(c) for c in {sha, *shas.values()}}
    for r, name, (cname, cargs), g_commit in order:
        seeds, sargs = seed_argv(r)
        tasks.append({"id": f"{name}-{cname}-r{r}", "system": name, "config": cname, "replicate": r,
                      "seeds": seeds, "seed_xyz": f"inputs/{Path(suite[name]['seed']).name}", "budget": budget,
                      "hard_timeout": hard_factor * budget,
                      "argv": with_config(system_argv(name, suite[name], budget,
                                                      conc[shas[g_commit] if g_commit else sha]), cargs) + sargs,
                      "harness_commit": harness,
                      **({"code": f"code-{shas[g_commit][:7]}"} if g_commit else {})})
    if control:   # replicate 0 again: what the seeds don't fix (queue timing); the first group, last in order
        assert not groups[0][3], "controls for a group with its own commit: not supported"
        cname, cargs = groups[0][1][0]
        for name in groups[0][0]:
            seeds, sargs = seed_argv(0)
            tasks.append({"id": f"{name}-{cname}-r0b", "system": name, "config": cname, "replicate": 0,
                          "control_of": f"{name}-{cname}-r0", "seeds": seeds,
                          "seed_xyz": f"inputs/{Path(suite[name]['seed']).name}", "budget": budget,
                          "hard_timeout": hard_factor * budget,
                          "argv": with_config(system_argv(name, suite[name], budget, conc[sha]), cargs) + sargs,
                          "harness_commit": harness})
    for t in tasks:     # input files a configuration names ({RUN_DIR}/inputs/X): from benchmarks/ or cluster/suite/
        for a in t["argv"]:
            if a.startswith("{RUN_DIR}/inputs/") and not (d / "inputs" / a.split("/")[-1]).is_file():
                src = next((p for p in (BENCH / a.split("/")[-1], HERE / "suite" / a.split("/")[-1]) if p.is_file()), None)
                if src is None:
                    raise SystemExit(f"{a}: no such input in benchmarks/ or benchmarks/cluster/suite/")
                shutil.copy(src, d / "inputs" / src.name)
    (d / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    timeout = int(hard_factor * budget + 900)
    what = "; then ".join(f"{','.join(gs)} x {','.join(c for c, _ in gc)}" + (f" x {gr}" if gr else "")
                          for gs, gc, gr, _ in groups)
    desc = (f"scoreboard {run}: discovery expand on explorer {sha[:7]}, {what}, {replicates} replicates each"
            f"{' + controls' if control else ''}, {threads} threads, budget {budget:g} s "
            f"(explorer-team phase 0, benchmarks/cluster/bench.py)")
    (d / "run.yaml").write_text(
        f'description: "{desc}"\nimage: {image}\ncommand: "exec python3 $RUN_DIR/task.py"\n'
        f"threads_per_task: {threads}\ncpus_per_pod: {threads}\nmemory_per_pod: {memory}\n"
        f"shards: {len(tasks)}\nparallelism: {parallelism}\ntask_timeout: {timeout}\npriority: batch\n"
        "notify: auto\nenv: {SELLA_DISABLE_GPU: 1, OMP_NUM_THREADS: 1, MKL_NUM_THREADS: 1, OPENBLAS_NUM_THREADS: 1}\n")
    (d / "manifest.json").write_text(json.dumps(
        {"run": run, "commit": sha if not shas else sorted({sha, *shas.values()} if any(
            not g[3] for g in groups) else set(shas.values())), "harness_commit": harness, "image": image,
         "configs": dict(configs),
         "systems": {n: suite[n] for n in systems}, "replicates": replicates, "budget_s": budget,
         "threads": threads, "control": control, "tasks": len(tasks), "suite_source": score.SUITE_SOURCE},
        indent=1))
    print(f"{d}: {len(tasks)} tasks, explorer {sha[:7]}, harness {harness[:7]}")
    return d


def local(run: str, threads: int) -> None:
    """Run RUN's pending tasks one at a time on this machine with the pod wrapper (NTHREADS = `threads`),
    filed like the cluster's: afh/RUN/output/<id> (exit 0) or errors/<id>, with a _meta.json. For checks of
    the pipeline at a few cores; not the 32-core "quick" definition."""
    import os
    import socket
    import tempfile
    import time

    d = AFH_DIR / run
    tasks = [json.loads(line) for line in (d / "tasks.jsonl").read_text().splitlines() if line.strip()]
    for t in tasks:
        if (d / "output" / t["id"]).is_dir() or (d / "errors" / t["id"]).is_dir():
            continue
        with tempfile.TemporaryDirectory(prefix=f"sb-{t['id']}-",
                                         dir=os.environ.get("SB_LOCAL_SCRATCH") or None) as wd:
            env = dict(os.environ, TASK_JSON=json.dumps(t), RUN_DIR=str(d), NTHREADS=str(threads), TASK_ID=t["id"])
            t0 = time.time()
            print(f"local {run}: {t['id']} ...", flush=True)
            rc = subprocess.run([sys.executable, str(d / "task.py")], cwd=wd, env=env).returncode
            dst = d / ("output" if rc == 0 else "errors") / t["id"]
            dst.parent.mkdir(exist_ok=True)
            shutil.copytree(Path(wd) / "out", dst)
            res = json.loads((dst / "result.json").read_text()) if (dst / "result.json").is_file() else {}
            secs = time.time() - t0
            (dst / "_meta.json").write_text(json.dumps(
                {"id": t["id"], "exit_code": rc, "seconds": round(secs, 2), "host": f"local:{socket.gethostname()}",
                 "cpu_seconds": res.get("cpu_s"), "cores": None if not res.get("cpu_s") else res["cpu_s"] / secs,
                 "nthreads": threads}, indent=1))
            print(f"local {run}: {t['id']} exit {rc} in {secs:.0f} s", flush=True)


POD_CAP = 8   # all of jan's pods together (32 threads each: 256 threads), coordinator 2026-11-09


def jan_pods() -> dict:
    """{run: pods} committed on the cluster now by every one of jan's runs, whichever agent or script made them:
    for each unfinished Job (label afh/user=jan) min(parallelism, its shards not yet done), and each Running or
    Pending pod of jan's that belongs to no Job (afh debug) as one."""
    jobs = json.loads(subprocess.run(["afh", "ssh", "kubectl -n jobs get jobs -l afh/user=jan -o json"],
                                     capture_output=True, text=True, check=True).stdout)["items"]
    out = {}
    for j in jobs:
        st, sp = j.get("status") or {}, j["spec"]
        if any(c.get("type") in ("Complete", "Failed") and c.get("status") == "True" for c in st.get("conditions") or []):
            continue
        rem = int(sp.get("completions") or 0) - int(st.get("succeeded") or 0) - int(st.get("failed") or 0)
        n = max(min(int(sp.get("parallelism") or 0), rem), int(st.get("active") or 0))
        if n > 0:
            out[(j["metadata"].get("labels") or {}).get("afh/run") or j["metadata"]["name"]] = n
    pods = subprocess.run(["afh", "ssh", "kubectl -n jobs get pods -l afh/user=jan --no-headers "
                           "-o custom-columns=P:.status.phase,J:.metadata.labels.job-name"],
                          capture_output=True, text=True, check=True).stdout.split("\n")
    loose = sum(1 for line in pods if line.split()[:1] and line.split()[0] in ("Running", "Pending")
                and line.split()[1:2] == ["<none>"])
    if loose:
        out["(pods outside a Job)"] = loose
    return out


def pod_guard(run: str, pods_for_run: int, cap: int = POD_CAP) -> tuple[bool, str]:
    """Whether `run` may have `pods_for_run` pods now: every other committed pod of jan's plus these <= cap."""
    live = jan_pods()
    others = sum(v for k, v in live.items() if k != run)
    ok = others + pods_for_run <= cap
    return ok, (f"pod guard: jan has {others} pods committed outside {run} ({live}); {run} wants {pods_for_run}; "
                f"{others + pods_for_run} {'<=' if ok else '>'} {cap}: {'go' if ok else 'wait'}")


def _run_pods(run: str) -> int:
    """The pods a submit of `run` would start: min(parallelism in run.yaml, its tasks)."""
    y = (AFH_DIR / run / "run.yaml").read_text()
    par = next((int(l.split(":", 1)[1]) for l in y.splitlines() if l.startswith("parallelism:")), POD_CAP)
    n = sum(1 for l in (AFH_DIR / run / "tasks.jsonl").read_text().splitlines() if l.strip())
    return min(par, n)


def image_components(reg: str, name: str, tag: str) -> dict:
    """The axtb and linus commits the image was built from, read from its config's labels (afh.axtb, afh.linus,
    written at build time, mepd-orca-uma-axtb:2026-11-09-main on); "no label" for images built before."""
    accept = ("-H 'Accept: application/vnd.docker.distribution.manifest.v2+json' "
              "-H 'Accept: application/vnd.oci.image.index.v1+json' -H 'Accept: application/vnd.oci.image.manifest.v1+json'")
    get = lambda ref: subprocess.run(["afh", "ssh", f"curl -s {accept} http://{reg}/v2/{name}/manifests/{ref}"],
                                     capture_output=True, text=True).stdout
    try:
        man = json.loads(get(tag))
        if "manifests" in man:   # an OCI index (buildx with attestations): the linux/amd64 image
            ref = next(x["digest"] for x in man["manifests"]
                       if (x.get("platform") or {}).get("architecture") == "amd64")
            man = json.loads(get(ref))
        cfg = man["config"]["digest"]
        blob = subprocess.run(["afh", "ssh", f"curl -s http://{reg}/v2/{name}/blobs/{cfg}"], capture_output=True,
                              text=True).stdout
        labels = (json.loads(blob).get("config") or {}).get("Labels") or {}
    except (ValueError, KeyError, TypeError):
        labels = {}
    return {k: labels.get(f"afh.{k}", "no label") for k in ("axtb", "linus")}


def record_digest(run: str, after_submit: bool) -> str:
    """The registry digest of the run's image tag, into manifest.json (BRIEF: provenance at run time)."""
    import datetime as dt

    m = json.loads((AFH_DIR / run / "manifest.json").read_text())
    reg, _, rest = m["image"].partition("/")
    name, _, tag = rest.partition(":")
    out = subprocess.run(["afh", "ssh", f"curl -s -I -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' "
                          f"http://{reg}/v2/{name}/manifests/{tag}"], capture_output=True, text=True).stdout
    digest = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.lower().startswith("docker-content-digest")), None)
    if not digest:
        raise SystemExit(f"no digest for {m['image']}")
    m.update(image_digest=digest, digest_read_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             image_components=image_components(reg, name, tag),
             **({"digest_read_after_submit": True} if after_submit else {}))
    (AFH_DIR / run / "manifest.json").write_text(json.dumps(m, indent=1))
    print(f"{run}: {m['image']} = {digest}")
    return digest


def afh(*args, check=True):
    print("+ afh", " ".join(args), flush=True)
    return subprocess.run(["afh", *args], check=check)


def score_run(runs, rescore: bool = False, name: str | None = None, reselect: bool = False,
              resettle: bool = False) -> dict:
    """Unpack and score every finished task of the run(s); aggregate per (system, configuration) into
    results/<name> (default: the run's own name). Several runs = one result set; a later run's task replaces an
    earlier one's of the same id, and `commits_by_system` says which commit each system's runs used."""
    import tarfile

    runs = [runs] if isinstance(runs, str) else list(runs)
    name = name or runs[0]
    suite = score.load_systems()
    manifests, scores = [], []
    by_id: dict = {}
    for run in runs:     # a later run's task replaces an earlier run's task of the same id (a rerun at a fix)
        for o in _score_tasks(run, suite, rescore, manifests, reselect or resettle, resettle):
            by_id[o["id"]] = o
    scores = list(by_id.values())
    commits = sorted({c for m in manifests for c in (m["commit"] if isinstance(m["commit"], list) else [m["commit"]])})
    agg = score.aggregate(scores, suite)
    agg["commits_by_system"] = {sy: sorted({str(o.get("code_commit"))[:7] for o in scores if o["system"] == sy})
                                for sy in sorted({o["system"] for o in scores})}
    agg["run"], agg["runs"], agg["commit"] = name, runs, commits[0] if len(commits) == 1 else commits
    agg["harness_commit"] = sorted({m["harness_commit"] for m in manifests})
    agg["suite_source"] = score.SUITE_SOURCE
    res = RESULTS / name
    res.mkdir(parents=True, exist_ok=True)
    (res / "aggregate.json").write_text(json.dumps(agg, indent=1, default=str))
    (res / "table.md").write_text(score.table(agg, scores))
    print(score.table(agg, scores))
    return agg


# the keys score.score_run's selectivity part writes (reselect replaces only these)
SELECTIVITY_KEYS = ("targets", "worst_error_kcal", "worst_abs_error_kcal", "within_unattributable",
                    "worst_error_final_amounts_kcal", "selectivity_measure", "first_passage_from",
                    "first_passage_error")


def _reselect(args):
    """The selectivity part of one scored task again (score.py's current measure), the rest of its score.json kept:
    the audit doesn't depend on it. With `resettle`, the settle time too (it follows the scored selectivity)."""
    sysdef, dst, resettle = args
    o = json.loads((dst / "score.json").read_text())
    if not o.get("ok"):
        return o
    result = json.loads((dst / "result.json").read_text()) if (dst / "result.json").is_file() else {}
    snaps = dst / "snapshots" if resettle and result.get("start") and (dst / "snapshots").is_dir() else None
    new = score.score_run(sysdef, dst / "run", result, snaps, do_audit=False)
    before = {"selectivity": [t.get("selectivity") for t in o.get("targets") or []],
              "worst_error_kcal": o.get("worst_error_kcal"),
              **({"settle_s": (o.get("settle") or {}).get("settle_s")} if snaps else {})}
    o.update({k: new.get(k) for k in SELECTIVITY_KEYS + (("settle",) if snaps else ())})
    o["reselected"] = {"instrument": score.instrument(), "before": before}
    (dst / "score.json").write_text(json.dumps(o, indent=1, default=str))
    return o


def _score_tasks(run: str, suite: dict, rescore: bool, manifests: list, reselect: bool = False,
                 resettle: bool = False) -> list:
    import tarfile

    d, res = AFH_DIR / run, RESULTS / run
    manifests.append(json.loads((d / "manifest.json").read_text()))
    tasks = [json.loads(line) for line in (d / "tasks.jsonl").read_text().splitlines() if line.strip()]
    scores = []
    if reselect and not rescore:   # 4 processes (local <= 4 cores), one thread each
        from concurrent.futures import ProcessPoolExecutor

        todo = [(suite[t["system"]], res / t["id"], resettle) for t in tasks if (res / t["id"] / "score.json").is_file()]
        with ProcessPoolExecutor(4) as ex:
            done = {str(o["id"]): o for o in ex.map(_reselect, todo)}
    for t in tasks:
        src = next((p for p in (d / "output" / t["id"], d / "errors" / t["id"]) if p.is_dir()), None)
        if src is None:
            continue
        dst = res / t["id"]
        if (dst / "score.json").is_file() and not rescore:
            scores.append(done[t["id"]] if reselect else json.loads((dst / "score.json").read_text()))
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


def report(name: str) -> str:
    """The baseline report of a scored result set: the table, the gate both ways between two configurations
    (systems run in both), audit flag totals with examples, the mirror-image TS spread, and each control
    rerun next to the replicate it repeats."""
    import gate

    res = RESULTS / name
    agg = json.loads((res / "aggregate.json").read_text())
    by_id: dict = {}     # as score_run: a later run's task replaces an earlier one's of the same id
    for run in agg.get("runs", [name]):
        for fp in sorted((RESULTS / run).glob("*/score.json")):
            o = json.loads(fp.read_text())
            by_id[o["id"]] = o
    scores = list(by_id.values())
    out = [f"# {name}: explorer {agg.get('commit')}, harness {agg.get('harness_commit')}, suite {agg.get('suite_source')}",
           "", (res / "table.md").read_text()]
    groups = {k: g for k, g in agg.items() if isinstance(g, dict) and "system" in g}
    configs = sorted({g["config"] for g in groups.values()})
    if "default" in configs and "T1800" in configs:
        configs = ["T1800", "default"]
    if len(configs) == 2:
        per = {c: {g["system"]: g for g in groups.values() if g["config"] == c} for c in configs}
        both = sorted(set(per[configs[0]]) & set(per[configs[1]]))
        for b, c in ((configs[0], configs[1]), (configs[1], configs[0])):
            v = gate.compare({k: per[b][k] for k in both}, {k: per[c][k] for k in both},
                             agg.get("control_settle_spread_s"))
            out.append(f"## gate: base {b} -> candidate {c} ({', '.join(both)})")
            out += [f"- {k}: {'PASS' if x['pass'] else 'FAIL'}" + (f" ({'; '.join(x['why'])})" if x["why"] else "")
                    for k, x in v.items()]
            out.append(f"- overall: {'PASS' if all(x['pass'] for x in v.values()) else 'FAIL'}")
            out.append("")
    out.append("## audit flags (all runs, controls included)")
    tot: dict = {}
    for o in scores:
        for k, v in ((o.get("audit") or {}).get("flags") or {}).items():
            for f in v:
                tot.setdefault(k, []).append((o["id"], f))
    for k, v in sorted(tot.items()):
        ex = "; ".join(f"{i}: {json.dumps(f, default=str)[:160]}" for i, f in v[:3])
        out.append(f"- {k}: {len(v)} in {len({i for i, _ in v})} runs. e.g. {ex}")
    out.append("")
    out.append("## mirror-image TS pairs, |dG| (kcal/mol) per system")
    for system in sorted({o["system"] for o in scores}):
        ds = [abs(p["dG_kcal"]) for o in scores if o["system"] == system
              for p in ((o.get("audit") or {}).get("info") or {}).get("mirror_ts_pairs", []) if p.get("dG_kcal") is not None]
        if ds:
            out.append(f"- {system}: {len(ds)} pairs, median {sorted(ds)[len(ds) // 2]:.2f}, max {max(ds):.2f}, "
                       f"{sum(d > 0.2 for d in ds)} above 0.2")
    out.append("")
    out.append("## where the time goes (medians over replicates): worker-seconds per job kind (share), wall phases")
    for key, g in sorted(groups.items()):
        jt = g.get("job_time") or {}
        kinds = sorted((jt.get("by_kind") or {}).items(), key=lambda kv: -(kv[1].get("run_s") or 0))
        ph = jt.get("phases_s") or {}
        fmt = lambda x: "-" if x is None else f"{x:.0f}"
        out.append(f"- {key}: " + ", ".join(f"{k} {v['run_s']:.0f} s ({100 * (v['share'] or 0):.0f} %, n {v['n']:.0f})"
                                            for k, v in kinds)
                   + f" | worker total {fmt(jt.get('worker_s'))} s, main callbacks {fmt(jt.get('main_callbacks_s'))} s"
                   + f" | wall: before queue {fmt(ph.get('before_queue'))}, queue {fmt(ph.get('queue'))}, "
                     f"after {fmt(ph.get('after_queue'))} s")
        w = jt.get("wall") or {}
        if w:
            sole = sorted((w.get("sole_s") or {}).items(), key=lambda kv: -(kv[1] or 0))
            out.append(f"  - queue wall {fmt(w.get('span_s'))} s: only one kind running: "
                       + ", ".join(f"{k} {v:.0f} s" for k, v in sole if v)
                       + f"; nothing running {fmt(w.get('idle_s'))} s; some job of the kind running: "
                       + ", ".join(f"{k} {v:.0f}" for k, v in sorted((w.get('busy_s') or {}).items(),
                                                                       key=lambda kv: -(kv[1] or 0)) if v))
    out.append("")
    by_id = {o["id"]: o for o in scores}
    pairs = [(by_id[i.replace("-program-", "-default-")], by_id[i]) for i in by_id
             if "-program-" in i and i.replace("-program-", "-default-") in by_id]
    if pairs:
        out.append("## equivalence: g-xTB program vs axtb (same seeds)")
        for a, b in pairs:
            g = lambda r: (f"species {r.get('species')}, steps {r.get('steps')}, sel "
                           f"{'; '.join(format(t['selectivity'], '.3g') if t.get('selectivity') is not None else '-' for t in r.get('targets', []))}, "
                           f"found {r.get('recall', {}).get('found')}, connected {r.get('recall', {}).get('connected')}, "
                           f"wall {r.get('wall_s') or 0:.0f} s, CPU {r.get('cpu_h') or 0:.2f} h, "
                           f"job s {((r.get('job_time') or {}).get('worker_s') or 0):.0f}, {r.get('stopped')}")
            out.append(f"- axtb {a['id']}: {g(a)}")
            out.append(f"- program {b['id']}: {g(b)}")
        out.append("")
    out.append("## controls (replicate 0 rerun with the same seeds)")
    for o in scores:
        if not o.get("control_of") or o["control_of"] not in by_id:
            continue
        a = by_id[o["control_of"]]
        f = lambda x: "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))
        row = lambda r: (f"species {r.get('species')}, steps {r.get('steps')}, sel "
                         f"{'; '.join(f(t['selectivity']) for t in r.get('targets', []))}, settle "
                         f"{f((r.get('settle') or {}).get('settle_s'))}, {r.get('stopped')}")
        out.append(f"- {a['id']}: {row(a)}  |  rerun: {row(o)}")
    text = "\n".join(out) + "\n"
    (res / "report.md").write_text(text)
    return text


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
                   help="SYSTEMS:CONFIGS[:REPLICATES[:COMMIT]] (comma-separated lists), repeatable, in priority "
                        "order; replaces --systems/--config. A COMMIT of its own labels the configurations <config>-<sha7>")
    b.add_argument("--image", default=IMAGE)
    b.add_argument("--first-replicate", type=int, default=0,
                   help="number (and seed) replicates from this one, to add replicates to an earlier run")
    b.add_argument("--interleave", action="store_true",
                   help="order tasks replicate by replicate and system by system across all groups")
    b.add_argument("--replicates", type=int, default=5)
    b.add_argument("--budget", type=float, default=1800.0)
    b.add_argument("--threads", type=int, default=32)
    b.add_argument("--parallelism", type=int, default=8)
    b.add_argument("--memory", default="64Gi")
    b.add_argument("--control", action="store_true")
    for name in ("submit", "pull", "status", "digest"):
        sp = sub.add_parser(name)
        sp.add_argument("run")
        if name == "submit":
            sp.add_argument("--force", action="store_true", help="afh submit --force (e.g. memory above afh's estimate)")
            sp.add_argument("--dry-run", action="store_true", help="only the pod guard's decision (exit 0 go, 75 wait)")
            sp.add_argument("--wait", action="store_true", help="wait (60 s steps) until the pod guard says go")
            sp.add_argument("--pod-cap", type=int, default=POD_CAP, help="jan's pods in all (default 8 = 256 threads)")
    pg = sub.add_parser("podguard", help="exit 0 if RUN may have PODS pods now, 75 if not (jan's live pods)")
    pg.add_argument("run")
    pg.add_argument("pods", type=int)
    pg.add_argument("--pod-cap", type=int, default=POD_CAP)
    s = sub.add_parser("score")
    s.add_argument("run", nargs="+")
    s.add_argument("--rescore", action="store_true")
    s.add_argument("--reselect", action="store_true",
                   help="recompute only the selectivity part of scored tasks (score.py's current measure)")
    s.add_argument("--resettle", action="store_true", help="--reselect and the settle time")
    s.add_argument("--as", dest="name", default=None, help="results name for several runs (default: the first)")
    g = sub.add_parser("gate")
    g.add_argument("base")
    g.add_argument("cand")
    sub.add_parser("report").add_argument("name")
    lo = sub.add_parser("local")
    lo.add_argument("run")
    lo.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if a.group:
            groups = [([x for x in g.split(":")[0].split(",") if x], [_config(c) for c in g.split(":")[1].split(",")],
                       int(g.split(":")[2]) if len(g.split(":")) > 2 and g.split(":")[2] else None,
                       g.split(":")[3] if len(g.split(":")) > 3 else None) for g in a.group]
        else:
            groups = [([x for x in a.systems.split(",") if x] or list(score.load_systems()),
                       [_config(c) for c in (a.config or ["default"])])]
        build(a.run, a.commit, groups, a.replicates, a.budget, a.threads, a.parallelism, a.memory, a.control,
              image=a.image, interleave=a.interleave, first_replicate=a.first_replicate)
    elif a.cmd == "submit":
        import time

        want = _run_pods(a.run)
        while True:
            ok, msg = pod_guard(a.run, want, a.pod_cap)
            print(msg)
            if ok or a.dry_run or not a.wait:
                break
            time.sleep(60)
        if a.dry_run:
            return 0 if ok else 75
        if not ok:
            return 75
        record_digest(a.run, after_submit=False)
        afh("push", str(AFH_DIR / a.run), a.run)
        afh("submit", a.run, *(["--force"] if a.force else []))
    elif a.cmd == "podguard":
        ok, msg = pod_guard(a.run, a.pods, a.pod_cap)
        print(msg)
        return 0 if ok else 75
    elif a.cmd == "digest":       # for runs submitted before digests were recorded
        record_digest(a.run, after_submit=True)
    elif a.cmd == "pull":
        afh("pull", a.run, str(AFH_DIR / a.run))
    elif a.cmd == "status":
        afh("status", a.run, check=False)
    elif a.cmd == "score":
        score_run(a.run, a.rescore, a.name, a.reselect, a.resettle)
    elif a.cmd == "local":
        local(a.run, a.threads)
    elif a.cmd == "report":
        print(report(a.name))
    elif a.cmd == "gate":
        import gate
        return gate.main([a.base, a.cand])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
