"""Does carrying more poses, or relaxing the lowest ones further, cut the spread of the bystander shift?

    python benchmarks/diagnostics/addmol/place_refine.py RUN_DIR PROFILE SMILES OUT.json --labels a,b
        --variants "placer:poses:carry:polish:fmax,..." [--repeats 5] [--conformers 3] [--workers 8]

Per step, variant and repeat (seed = repeat) one mepd.bystander.search on one thread; from each repeat's
lowest eligible pose per frame its binding energies and shifts; per step and variant their standard
deviation and range over the repeats, the mean seconds and gradient calls of a repeat.
"""
import argparse
import json
import statistics as S
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

H = 627.509474


def task(a):
    run_dir, profile, label, variant, rep, confs_xyz, csym, cq = a
    import mepd
    from mepd import bystander as B
    from place_measure import load_step

    placer, poses, carry, polish, fmax = variant.split(":")
    engine = mepd.engine_from_profile(profile)
    calls = {"n": 0}
    grad = engine.compute_gradients

    def counted(chain):
        calls["n"] += 1
        return grad(chain)
    engine.compute_gradients = counted
    st, frames = load_step(Path(run_dir), label)
    symx = list(frames["ts"].structure.symbols)
    charge = int(frames["ts"].structure.charge) + cq
    cnodes = [B._node(csym, x, cq) for x in confs_xyz]
    e_c = [B._energy(engine, c) for c in cnodes]
    e_x = {f: B._energy(engine, frames[f]) for f in frames}
    t0 = time.perf_counter()
    need_q = placer in ("mixed", "seeded", "screen-ff")
    q = {f: B.gfn2_charges(frames[f]) for f in frames} if need_q else None
    confs = [(csym, np.asarray(x), B.gfn2_charges(c) if need_q else None, e) for x, c, e in zip(confs_xyz, cnodes, e_c)]
    done = B.search({f: (symx, B._xyz(frames[f])) for f in frames}, confs, engine, placer=placer, poses=int(poses),
                    seed=rep, carry=int(carry), charges=q, charge=charge, fmax=float(fmax), polish=int(polish))
    b = {}
    for f in frames:
        low = B.lowest(done[f])
        if low is not None:
            b[f] = (low["energy"] - e_x[f] - min(e_c)) * H
    return {"label": label, "variant": variant, "repeat": rep, "binding": b, "wall_s": time.perf_counter() - t0,
            "gradient_calls": calls["n"], "relaxations": sum(len(v) for v in done.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("profile")
    ap.add_argument("smiles")
    ap.add_argument("out")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--conformers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import multiprocessing as mp

    import mepd
    from mepd import bystander as B

    confs = B.conformers(a.smiles, mepd.engine_from_profile(a.profile), n=a.conformers)
    csym, cq = list(confs[0].structure.symbols), int(confs[0].structure.charge)
    jobs = [(a.run_dir, a.profile, lab, v, rep, [B._xyz(c) for c in confs], csym, cq)
            for lab in a.labels.split(",") for v in a.variants.split(",") for rep in range(a.repeats)]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        res = list(pool.map(task, jobs))
    rows = []
    lowest = {}
    for r in res:
        for f, v in r["binding"].items():
            lowest[(r["label"], f)] = min(lowest.get((r["label"], f), 9e9), v)
    for lab in a.labels.split(","):
        for v in a.variants.split(","):
            rs = [r for r in res if r["label"] == lab and r["variant"] == v and len(r["binding"]) == 3]
            bs = [r["binding"]["ts"] - r["binding"]["r"] for r in rs]
            above = [r["binding"][f] - lowest[(lab, f)] for r in rs for f in r["binding"]]
            rows.append({"label": lab, "variant": v, "barrier_shift": bs, "std": S.pstdev(bs) if len(bs) > 1 else None,
                         "range": max(bs) - min(bs) if bs else None, "mean_above_lowest": S.mean(above) if above else None,
                         "wall_s": S.mean(r["wall_s"] for r in rs) if rs else None,
                         "gradient_calls": S.mean(r["gradient_calls"] for r in rs) if rs else None,
                         "pooled_shift": lowest[(lab, "ts")] - lowest[(lab, "r")]})
            x = rows[-1]
            print(lab, v, [round(b, 1) for b in bs], "std", None if x["std"] is None else round(x["std"], 2),
                  "above", round(x["mean_above_lowest"], 2), "s", round(x["wall_s"], 1), "pooled", round(x["pooled_shift"], 1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "tasks": res, "args": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
