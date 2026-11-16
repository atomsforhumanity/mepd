"""Placers of the bystander estimate compared: spread over independent repeats against cost.

    python benchmarks/diagnostics/addmol/place_measure.py RUN_DIR PROFILE SMILES OUT_DIR --labels a,b,...
        [--placers random,seeded,screen-ff,screen-sp] [--poses 24] [--prefixes 3,6,12,24] [--repeats 5]
        [--conformers 3] [--workers 8] [--relax-substrate] [--final-profile P2] [--carry 1]

For each named IRC-verified step of an expand run (RUN_DIR/summary.json; frames = first IRC frame, TS, last
IRC frame, whatever atoms the run's box holds) and each placer and repeat (seed = repeat), one process on one
thread relaxes `--poses` placed poses per frame (mepd.bystander.search, no carrying), then, for each prefix n
of them, carries the best pose of each frame to the other two (mepd.bystander.carry_poses, upto=n). Every
relaxed pose is written with its energy, flags and the seconds it took, so the estimate at any n, with and
without carrying, and its cost follow from the file (place_analyze.py). With --final-profile, single points
at that level on every eligible relaxed pose. Nothing is compared with a target; no pose is set by hand.
"""
import argparse
import copy
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

H = 627.509474


def load_step(run_dir: Path, label: str):
    import mepd

    s = json.loads((run_dir / "summary.json").read_text())
    st = next(x for x in s["steps"] if x["label"] == label)
    ts = mepd.load(str(run_dir / "ts" / Path(st["files"]["ts"]).name))
    irc = mepd.read_frames(str(run_dir / "ts" / Path(st["files"]["irc"]).name))
    return st, {"r": irc[0], "ts": ts, "p": irc[-1]}


def task(a):
    run_dir, profile, final_profile, label, placer, rep, poses, prefixes, carry, relax_sub, confs_xyz, csym, cq = a
    import mepd
    from mepd import bystander as B

    engine = mepd.engine_from_profile(profile)
    calls = {"n": 0}
    grad = engine.compute_gradients

    def counted(chain):
        calls["n"] += 1
        return grad(chain)
    engine.compute_gradients = counted
    st, frames = load_step(Path(run_dir), label)
    symx = list(frames["ts"].structure.symbols)
    n_x, charge = len(symx), int(frames["ts"].structure.charge) + cq
    t0 = time.perf_counter()
    cnodes = [B._node(csym, x, cq) for x in confs_xyz]
    e_c = [B._energy(engine, c) for c in cnodes]
    need_q = placer in ("mixed", "seeded", "screen-ff")
    q = {f: B.gfn2_charges(frames[f]) for f in frames} if need_q else None
    confs = [(csym, np.asarray(x), B.gfn2_charges(c) if need_q else None, e) for x, c, e in zip(confs_xyz, cnodes, e_c)]
    X = {f: (symx, B._xyz(frames[f])) for f in frames}
    reaction_atoms = None
    if relax_sub:
        from mepd.network_variants import adjacency
        from mepd.substituents import relax_group

        ra, rb = adjacency(symx, X["r"][1]), adjacency(symx, X["p"][1])
        reaction_atoms = sorted(int(i) for i in np.nonzero((ra != rb).any(axis=1))[0])
        e_x = {f: float(relax_group(engine, frames[f], [i for i in range(n_x) if i not in set(reaction_atoms)]).energy)
               for f in frames}
    else:
        e_x = {f: B._energy(engine, frames[f]) for f in frames}
    t_setup, c_setup = time.perf_counter() - t0, calls["n"]
    marks = []
    done = B.search(X, confs, engine, placer=placer, poses=poses, seed=rep, carry=0, relax_substrate=relax_sub,
                    reaction_atoms=reaction_atoms, charges=q, charge=charge,
                    on_pose=lambda f, p: marks.append(calls["n"]))
    t_search, c_search = time.perf_counter() - t0 - t_setup, calls["n"] - c_setup
    symbols = symx + list(csym)
    held = set(range(n_x)) if not relax_sub else set(reaction_atoms)
    free = [i for i in range(len(symbols)) if i not in held]
    from mepd.network_variants import adjacency

    bonds_c = adjacency(csym, confs_xyz[0])
    carried = {}
    for n in prefixes:
        c0, t1 = calls["n"], time.perf_counter()
        d = {f: list(ps) for f, ps in done.items()}
        added = B.carry_poses(X, d, engine, symbols, free, bonds_c, n_x, carry, charge=charge, multiplicity=1,
                              fmax=0.05, steps=300, t0=t1, upto=n)
        carried[n] = {"poses": added, "wall_s": time.perf_counter() - t1, "gradient_calls": calls["n"] - c0}
    final = None
    if final_profile:
        fe = mepd.engine_from_profile(final_profile)
        t1 = time.perf_counter()
        fe_c = min(B._energy(fe, B._node(csym, x, cq)) for x in confs_xyz)
        fe_x = {f: B._energy(fe, B._node(symx, X[f][1], charge - cq)) for f in frames}
        n_sp = 0
        for group in [done] + [c["poses"] for c in carried.values()]:
            for f, ps in group.items():
                for p in ps:
                    if "failed" not in p and B._eligible(p) and "final_energy" not in p:
                        p["final_energy"] = B._energy(fe, B._node(symbols, p["xyz"], charge))
                        n_sp += 1
        final = {"molecule_energy": fe_c, "step_energies": fe_x, "wall_s": time.perf_counter() - t1, "single_points": n_sp}

    def slim(p, keep_xyz):
        o = {k: v for k, v in p.items() if k != "xyz"}
        if keep_xyz and "xyz" in p:
            o["xyz"] = np.round(p["xyz"], 5).tolist()
        return o

    def best_idx(ps):
        ok = [i for i, p in enumerate(ps) if "failed" not in p and B._eligible(p)]
        return {min(ok, key=lambda i: ps[i]["energy"])} if ok else set()

    out_poses = {}
    for f, ps in done.items():
        keep = set()
        for n in prefixes:
            keep |= best_idx(ps[:n])
        out_poses[f] = [slim(p, i in keep) for i, p in enumerate(ps)]
    for n in carried:
        carried[n]["poses"] = {f: [slim(p, True) for p in ps] for f, ps in carried[n]["poses"].items()}
    return {"label": label, "placer": placer, "repeat": rep, "poses": out_poses, "carried": carried, "final": final,
            "molecule_energy": min(e_c), "step_energies": e_x, "n_step_atoms": n_x, "symbols": symbols,
            "charge": charge, "setup_s": t_setup, "search_s": t_search, "search_gradient_calls": c_search,
            "reaction_atoms": reaction_atoms, "reacting": st.get("reacting"), "a": st["a"], "b": st["b"],
            "parent_barrier_kcal": [(e_x["ts"] - e_x["r"]) * H, (e_x["ts"] - e_x["p"]) * H]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("profile")
    ap.add_argument("smiles")
    ap.add_argument("out", type=Path)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--placers", default="random,seeded,screen-ff,screen-sp")
    ap.add_argument("--poses", type=int, default=24)
    ap.add_argument("--prefixes", default="3,6,12,24")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--conformers", type=int, default=3)
    ap.add_argument("--carry", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--relax-substrate", action="store_true")
    ap.add_argument("--final-profile", default=None)
    a = ap.parse_args()
    import mepd
    from mepd import bystander as B

    a.out.mkdir(parents=True, exist_ok=True)
    engine = mepd.engine_from_profile(a.profile)
    t0 = time.perf_counter()
    confs = B.conformers(a.smiles, engine, n=a.conformers)
    t_conf = time.perf_counter() - t0
    mepd.write_xyz(a.out / "conformers.xyz", confs)
    csym, cq = list(confs[0].structure.symbols), int(confs[0].structure.charge)
    prefixes = [int(x) for x in a.prefixes.split(",")]
    jobs = [(str(a.run_dir), a.profile, a.final_profile, lab, pl, rep, a.poses, prefixes, a.carry, a.relax_substrate,
             [B._xyz(c) for c in confs], csym, cq)
            for lab in a.labels.split(",") for pl in a.placers.split(",") for rep in range(a.repeats)]
    t0 = time.perf_counter()
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool, \
            (a.out / "tasks.jsonl").open("w") as fh:
        for res in pool.map(task, jobs):
            fh.write(json.dumps(res, default=float) + "\n")
            fh.flush()
            print(res["label"], res["placer"], res["repeat"], f"{res['search_s']:.1f}s", flush=True)
    (a.out / "run.json").write_text(json.dumps({
        "run_dir": str(a.run_dir), "profile": a.profile, "final_profile": a.final_profile, "smiles": a.smiles,
        "conformers": len(confs), "conformer_energies": [float(c.energy) for c in confs], "conformers_s": t_conf,
        "wall_s": time.perf_counter() - t0, "workers": a.workers, "threads_per_task": 1, "host_cpu": platform.processor(),
        "cpu_count": os.cpu_count(), "mepd_file": mepd.__file__, "args": {k: str(v) for k, v in vars(a).items()}}, indent=1))


if __name__ == "__main__":
    main()
