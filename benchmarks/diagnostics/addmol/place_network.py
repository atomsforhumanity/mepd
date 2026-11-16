"""The bystander estimate over every IRC-verified step of an expand run, in parallel: the wall time a user waits.

    python benchmarks/diagnostics/addmol/place_network.py RUN_DIR PROFILE SMILES OUT_DIR [--workers 8]
        [--placer screen-sp] [--poses 6] [--repeats 3] [--conformers 3] [--final-profile P2] [--max-steps 0]

Steps are worked in the order of their TS height above the seed, lowest first (a stand-in for the order the
kinetics' sensitivities give), `--workers` processes of one thread each (mepd.bystander.bystander_steps).
Written: steps.jsonl (one line per step as it finishes, with the seconds since the start), run.json (wall
time, time to the first and to the tenth result, core-seconds, statuses).
"""
import argparse
import functools
import json
import os
import platform
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("profile")
    ap.add_argument("smiles")
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--placer", default="screen-sp")
    ap.add_argument("--poses", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--conformers", type=int, default=3)
    ap.add_argument("--final-profile", default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    import mepd
    from mepd import bystander as B

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = mepd.engine_from_profile(a.profile)
    mol = B.molecule_from_smiles(a.smiles, engine, n=a.conformers)
    t_mol = time.perf_counter() - t0
    summary = json.loads((a.run_dir / "summary.json").read_text())
    rows = [s for s in summary["steps"] if (s.get("files") or {}).get("verified") and s["files"].get("irc")]
    rows.sort(key=lambda s: (s.get("closed") or {}).get("ts_above_seed_kcal") or s["ts_energy"])
    if a.max_steps:
        rows = rows[:a.max_steps]
    steps = {}
    for s in rows:
        ts = mepd.load(str(a.run_dir / "ts" / Path(s["files"]["ts"]).name))
        irc = mepd.read_frames(str(a.run_dir / "ts" / Path(s["files"]["irc"]).name))
        steps[s["label"]] = {"symbols": list(ts.structure.symbols), "charge": int(ts.structure.charge),
                             "multiplicity": int(ts.structure.multiplicity), "r": B._xyz(irc[0]).tolist(),
                             "ts": B._xyz(ts).tolist(), "p": B._xyz(irc[-1]).tolist()}
    t_load = time.perf_counter() - t0 - t_mol
    fh = (a.out / "steps.jsonl").open("w")
    times = []

    def on_result(key, res):
        times.append(time.perf_counter() - t0)
        fh.write(json.dumps({"label": key, "t": times[-1], **{k: v for k, v in res.items() if k != "symbols"}},
                            default=float) + "\n")
        fh.flush()

    out = B.bystander_steps(steps, mol, functools.partial(mepd.engine_from_profile, a.profile), workers=a.workers,
                            make_final=functools.partial(mepd.engine_from_profile, a.final_profile) if a.final_profile else None,
                            on_result=on_result, placer=a.placer, poses=a.poses, repeats=a.repeats)
    fh.close()
    status = {}
    for r in out.values():
        status[r["status"]] = status.get(r["status"], 0) + 1
    (a.out / "run.json").write_text(json.dumps({
        "steps": len(steps), "atoms": len(mol["symbols"]) + len(next(iter(steps.values()))["symbols"]),
        "workers": a.workers, "wall_s": time.perf_counter() - t0, "molecule_s": t_mol, "load_s": t_load,
        "first_result_s": min(times) if times else None, "tenth_result_s": sorted(times)[9] if len(times) > 9 else None,
        "core_s": sum(r.get("wall_s", 0.0) for r in out.values()), "status": status,
        "gradient_calls": sum(r.get("gradient_calls") or 0 for r in out.values()),
        "cpu": platform.processor(), "cpu_count": os.cpu_count(), "mepd_file": mepd.__file__,
        "args": {k: str(v) for k, v in vars(a).items()}}, indent=1))
    print(json.dumps(json.loads((a.out / "run.json").read_text()), indent=1))


if __name__ == "__main__":
    main()
