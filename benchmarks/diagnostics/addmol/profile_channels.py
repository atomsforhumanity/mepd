"""Where a `mepd channels` run spends its wall time and its cores.

    python benchmarks/diagnostics/addmol/profile_channels.py PROFILE_DIR -- channels --reaction ... -o OUT
    python benchmarks/diagnostics/addmol/profile_channels.py --report PROFILE_DIR OUT

Run mode: `mepd <arguments after -->` in this process, with (1) timers around the stages' functions (the
functions are called unchanged; each process of the run keeps its own totals in PROFILE_DIR/agg.<pid>.json,
and every path search, TS optimisation and IRC is one line of PROFILE_DIR/events.jsonl with its start and
length) and (2) a sampler process that reads /proc every 2 s: the processes and threads below this one, how
many are running, and the CPU time they used since the last sample (PROFILE_DIR/cpu.csv).

Report mode: the table of wall time by stage (from OUT/stats.json and the timers), CPU use over time, and
when each channel's TS had its IRC (the file times under OUT/ts, against the run's start).
Standard library only besides mepd; reads and times, changes nothing.
"""
import functools
import json
import os
import sys
import time
from pathlib import Path

# (module, attribute path, name in the table, whether each call is logged as an event)
TIMED = [
    ("mepd.channel_search", "_minimize_endpoints", "minimize the two SMILES ends", True),
    ("mepd.channel_search", "_check_endpoint_atom_mapping", "endpoint atom-mapping check", True),
    ("mepd.channel_search", "_minimize_conformer_pool", "conformers: minimisation", True),
    ("mepd.sampling", "generate_seed_pairs", "conformers: sampling", True),
    ("mepd.channel_search", "_expand_pairs_by_mechanism", "mapping stage (parent, wall)", True),
    ("mepd.atom_mapping_selection", "select_per_mechanism", "mapping: one pair, all of it", False),
    ("mepd.atom_mapping", "suggest_mechanism_candidates", "mapping: mechanisms of one pair", False),
    ("mepd.atom_mapping", "SlapMapper.get_maps", "mapping: SLAPMapper", False),
    ("mepd.atom_mapping", "explore_mechanisms", "mapping: explore_mechanisms", False),
    ("mepd.atom_mapping", "expand_mapping_fully", "mapping: symmetry variants", False),
    ("mepd.atom_mapping", "_snap_variants", "mapping: snap", False),
    ("mepd.atom_mapping", "catalytic_participants", "mapping: catalytic label", False),
    ("mepd.atom_mapping", "mechanism_key", "mapping: mechanism key", False),
    ("mepd.atom_mapping", "realign_end_to_start", "mapping: reordered end structures", False),
    ("mepd.atom_mapping_selection", "snap_choice", "mapping: scoring (snap choice)", False),
    ("mepd.atom_mapping_selection", "_interpolate", "mapping: scoring, GI", False),
    ("mepd.atom_mapping_selection", "_relaxed_choice", "mapping: scoring, relaxed paths", False),
    ("mepd.msmep", "MSMEP.run_parallel_recursive_minimize", "path search (one pair)", True),
    ("mepd.msmep", "MSMEP.run_recursive_minimize", "path search (one pair)", True),
    ("mepd.compute", "find_ts", "TS optimisation", True),
    ("mepd.compute", "irc", "IRC", True),
    ("mepd.channel_search", "_discover_channels", "final TS/IRC pass and classification (parent, wall)", True),
]


def _install(out: Path) -> None:
    import importlib

    agg: dict = {}
    state = {"pid": os.getpid(), "last": 0.0}

    def flush(force=False):
        now = time.time()
        if os.getpid() != state["pid"]:   # a forked worker: its own totals
            agg.clear()
            state["pid"] = os.getpid()
        if force or now - state["last"] > 2.0:
            state["last"] = now
            tmp = out / f"agg.{os.getpid()}.json.tmp"
            tmp.write_text(json.dumps(agg))
            os.replace(tmp, out / f"agg.{os.getpid()}.json")

    def timed(fn, name, event):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            if os.getpid() != state["pid"]:
                flush()
            t0w, t0 = time.time(), time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                n, s = agg.get(name, (0, 0.0))
                agg[name] = (n + 1, s + dt)
                if event:
                    with open(out / "events.jsonl", "a") as f:
                        f.write(json.dumps({"name": name, "pid": os.getpid(), "t0": t0w, "seconds": dt}) + "\n")
                flush(force=event)
        return wrapper

    for module, path, name, event in TIMED:
        try:
            obj = importlib.import_module(module)
            *parents, attr = path.split(".")
            for p in parents:
                obj = getattr(obj, p)
            setattr(obj, attr, timed(getattr(obj, attr), name, event))
        except Exception as exc:
            print(f"profile_channels: {module}.{path} not timed ({type(exc).__name__}: {exc})", file=sys.stderr)


def _sampler(root: int, out: Path, every: float = 2.0) -> None:
    tick = os.sysconf("SC_CLK_TCK")
    last: dict = {}
    t_last = time.time()
    with open(out / "cpu.csv", "w") as f:
        f.write("t,processes,threads,running,cores_used\n")
        while True:
            time.sleep(every)
            procs = {}
            for d in os.listdir("/proc"):
                if not d.isdigit():
                    continue
                try:
                    rest = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
                except OSError:
                    continue
                procs[int(d)] = (int(rest[1]), rest[0], int(rest[11]) + int(rest[12]), int(rest[17]))
            if root not in procs:
                return
            kids: dict = {}
            for pid, (ppid, *_r) in procs.items():
                kids.setdefault(ppid, []).append(pid)
            tree, todo = [], [root]
            while todo:
                p = todo.pop()
                tree.append(p)
                todo += kids.get(p, [])
            tree = [p for p in tree if p != os.getpid()]
            now = time.time()
            used = sum(procs[p][2] - last[p] for p in tree if p in last) / tick / max(now - t_last, 1e-9)
            last = {p: procs[p][2] for p in tree}
            t_last = now
            f.write(f"{now:.1f},{len(tree)},{sum(procs[p][3] for p in tree)},"
                    f"{sum(procs[p][1] == 'R' for p in tree)},{used:.2f}\n")
            f.flush()


def run(out: Path, argv: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "start.json").write_text(json.dumps({"t0": time.time(), "argv": argv, "cpus": sorted(os.sched_getaffinity(0))}))
    root = os.getpid()
    if os.fork() == 0:
        try:
            _sampler(root, out)
        finally:
            os._exit(0)
    _install(out)
    from mepd.cli import app

    sys.argv = ["mepd", *argv]
    try:
        app()
    finally:
        (out / "end.json").write_text(json.dumps({"t1": time.time()}))


def report(prof: Path, out: Path) -> None:
    t0 = json.loads((prof / "start.json").read_text())["t0"]
    t1 = json.loads((prof / "end.json").read_text())["t1"] if (prof / "end.json").is_file() else None
    stats = json.loads((out / "stats.json").read_text()) if (out / "stats.json").is_file() else {}
    total: dict = {}
    for fp in prof.glob("agg.*.json"):
        for name, (n, s) in json.loads(fp.read_text()).items():
            a = total.setdefault(name, [0, 0.0, 0])
            a[0] += n; a[1] += s; a[2] += 1
    print(f"wall {(t1 or time.time()) - t0:.0f} s; cpus {json.loads((prof / 'start.json').read_text())['cpus']}")
    print("stats.json:", {k: stats.get(k) for k in ("workers", "parallel_workers", "engine_parallel", "n_pairs",
                                                    "n_mechanisms", "n_path_searches", "pair_atom_mapping_seconds",
                                                    "msmep_seconds", "ts_discovery_seconds", "total_seconds",
                                                    "search_budget_reached")})
    for side in ("start", "end"):
        c = (stats.get("conformers") or {}).get(side, {})
        print(f"  conformers {side}: sampling {c.get('seconds')} s, minimisation {c.get('minimize_seconds')} s, "
              f"final {c.get('n_final')}")
    print(f"{'timer':52s} {'calls':>8s} {'summed s':>10s} {'processes':>9s}")
    for _, _, name, _ in TIMED:
        if name in total and name not in getattr(report, "_done", set()):
            report._done = getattr(report, "_done", set()) | {name}
            n, s, k = total[name]
            print(f"{name:52s} {n:8d} {s:10.1f} {k:9d}")
    rows = [l.split(",") for l in (prof / "cpu.csv").read_text().splitlines()[1:]] if (prof / "cpu.csv").is_file() else []
    if rows:
        ev = [json.loads(l) for l in (prof / "events.jsonl").read_text().splitlines()] if (prof / "events.jsonl").is_file() else []
        marks = {e["name"]: e["t0"] - t0 for e in reversed(ev)}
        print("CPU over time (cores used, mean over each tenth of the run; processes max; threads max):")
        n = max(1, len(rows) // 10)
        for k in range(0, len(rows), n):
            part = rows[k:k + n]
            print(f"  {float(part[0][0]) - t0:7.0f}-{float(part[-1][0]) - t0:7.0f} s: "
                  f"{sum(float(r[4]) for r in part) / len(part):5.2f} cores, "
                  f"{max(int(r[1]) for r in part):3d} processes, {max(int(r[2]) for r in part):4d} threads, "
                  f"running max {max(int(r[3]) for r in part)}")
        print(f"  whole run: {sum(float(r[4]) for r in rows) / len(rows):.2f} cores mean, "
              f"{max(int(r[1]) for r in rows)} processes max, {max(int(r[2]) for r in rows)} threads max")
        print("first start of each event kind (s after start):", {k: round(v, 1) for k, v in sorted(marks.items(), key=lambda kv: kv[1])})
    # when each channel's TS had its IRC
    ts = out / "ts"
    for kind in ("channels", "alternate-channels"):
        for c in sorted((out / kind).glob("*")) if (out / kind).is_dir() else []:
            for m in sorted(c.rglob("members.txt")):
                labels = m.read_text().split()
                times = [(ts / f"{lab}_irc.xyz").stat().st_mtime - t0 for lab in labels if (ts / f"{lab}_irc.xyz").is_file()]
                if times:
                    print(f"  {m.parent.relative_to(out)}: first member's IRC written {min(times):.0f} s after start "
                          f"({len(labels)} member(s))")


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        k = sys.argv.index("--")
        run(Path(sys.argv[1]).resolve(), sys.argv[k + 1:])
