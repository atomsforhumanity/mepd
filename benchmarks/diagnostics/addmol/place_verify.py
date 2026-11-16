"""The bystander estimate of a step against the same step verified with the molecule present.

    python benchmarks/diagnostics/addmol/place_verify.py RUN_DIR PROFILE SMILES OUT_DIR --labels a,b,...
        [--placer seeded] [--poses 6] [--repeats 3] [--conformers 3] [--workers 8] [--seed 0] [--relax-substrate]

Per step (one process, one thread): (1) mepd.bystander.bystander on the step's frames (first IRC frame, TS,
last IRC frame of RUN_DIR's record); (2) the parent step through `mepd variant --mode reoptimize`'s function
(mepd.cli_variant._reoptimize: ends minimized, TS optimized, IRC), so parent and complex are measured by one
protocol; (3) the same function on the estimate's three lowest complexes: the TS optimized with the molecule
present from the estimate's TS complex, its IRC, the ends as found; (4) quasi-RRHO (mepd.thermo.qrrho, 1 M,
the engine's Hessian, the TS's lowest mode left out) of the relaxed complexes, the relaxed parent structures
and the molecule, for the full binding free-energy correction beside the estimate's. Written per step:
the estimate, both verifications' summaries, shifts, bond changes of the verified ends split into the
molecule's own, the step's own and pairs across, thermal corrections, seconds of each part.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

H = 627.509474


def pairs(adj):
    return [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(adj)))]


def g_corr(engine, node, ts=False):
    import mepd
    from mepd.thermo import qrrho
    from mepd import bystander as B

    fr = mepd.frequencies(node, engine)
    w = [float(x) for x in fr.wavenumbers_cm]
    q = qrrho(w, list(node.structure.symbols), B._xyz(node), 298.15, ts=ts,
              multiplicity=int(node.structure.multiplicity))
    return {"g_corr": q["g_corr"], "lowest_cm": w[:3], "n_imaginary": q["n_imaginary"]}


def task(a):
    run_dir, profile, label, placer, poses, repeats, seed, relax_sub, confs_xyz, csym, cq, out, carry, polish = a
    import mepd
    from mepd import bystander as B
    from mepd.cli_common import _open_run_inputs
    from mepd.cli_variant import _reoptimize
    from mepd.network_variants import adjacency
    from place_measure import load_step

    out = Path(out) / label
    out.mkdir(parents=True, exist_ok=True)
    ri = _open_run_inputs(Path(profile))
    engine = ri.engine
    st, frames = load_step(Path(run_dir), label)
    symx = list(frames["ts"].structure.symbols)
    n_x, qx = len(symx), int(frames["ts"].structure.charge)
    confs = [B._node(csym, x, cq) for x in confs_xyz]
    rec = {"label": label, "reacting": st.get("reacting"), "recorded_barrier_kcal": st.get("barrier_kcal")}
    t = time.perf_counter()
    est = B.bystander({"symbols": symx, "charge": qx, "multiplicity": 1, **{f: B._xyz(frames[f]).tolist() for f in frames}},
                      {"symbols": csym, "charge": cq, "multiplicity": 1, "conformers": [np.asarray(x).tolist() for x in confs_xyz]},
                      engine, placer=placer, poses=poses, repeats=repeats, seed=seed, relax_substrate=relax_sub,
                      carry=carry, polish=polish)
    rec["estimate"], rec["estimate_s"] = est, time.perf_counter() - t
    X = {f: B._xyz(frames[f]) for f in frames}
    t = time.perf_counter()
    dpar = {"symbols": symx, "charge": qx, "multiplicity": 1, **{f: X[f].tolist() for f in X},
            "bonds_r": pairs(adjacency(symx, X["r"])), "bonds_p": pairs(adjacency(symx, X["p"]))}
    try:
        rec["parent"] = _reoptimize(dpar, ri, out / "parent")
    except Exception as exc:
        rec["parent"] = {"status": "error", "why": f"{type(exc).__name__}: {exc}"}
    rec["parent_s"] = time.perf_counter() - t
    if all(f in est.get("complexes", {}) for f in ("r", "ts", "p")):
        sym = est["symbols"]
        C = {f: np.asarray(est["complexes"][f]) for f in ("r", "ts", "p")}
        t = time.perf_counter()
        d = {"symbols": sym, "charge": est["charge"], "multiplicity": 1, **{f: C[f].tolist() for f in C},
             "bonds_r": pairs(adjacency(sym, C["r"])), "bonds_p": pairs(adjacency(sym, C["p"]))}
        try:
            rec["with_c"] = _reoptimize(d, ri, out / "with_c")
        except Exception as exc:
            rec["with_c"] = {"status": "error", "why": f"{type(exc).__name__}: {exc}"}
        rec["with_c_s"] = time.perf_counter() - t
        # what the verified structures are, split by whose bonds changed
        want = {"r": adjacency(symx, X["r"]), "p": adjacency(symx, X["p"])}
        bc = adjacency(csym, confs_xyz[0])
        found = {}
        for name, fp in (("end_r", "start.xyz"), ("end_p", "end.xyz"), ("irc", "ts_irc.xyz"), ("irc2", "irc.xyz")):
            fp = out / "with_c" / "reopt" / fp
            if not fp.exists():
                continue
            fr = mepd.read_frames(str(fp))
            for tag, node in ((name, fr[0]),) if name.startswith("end") else ((name + "_first", fr[0]), (name + "_last", fr[-1])):
                adj = adjacency(sym, B._xyz(node))
                found[tag] = {"step_bonds": "r" if np.array_equal(adj[:n_x, :n_x], want["r"]) else
                              "p" if np.array_equal(adj[:n_x, :n_x], want["p"]) else "other",
                              "molecule_bonds_kept": bool(np.array_equal(adj[n_x:, n_x:], bc)),
                              "cross_bonds": pairs(adj[:n_x, n_x:])}
        rec["verified_structures"] = found
        if rec["parent"].get("barrier_kcal") is not None and rec["with_c"].get("barrier_kcal") is not None:
            rec["verified_barrier_shift_kcal"] = rec["with_c"]["barrier_kcal"] - rec["parent"]["barrier_kcal"]
            if rec["with_c"].get("reaction_kcal") is not None and rec["parent"].get("reaction_kcal") is not None:
                rec["verified_reaction_shift_kcal"] = rec["with_c"]["reaction_kcal"] - rec["parent"]["reaction_kcal"]
        # full quasi-RRHO of the relaxed structures
        t = time.perf_counter()
        th = {}
        try:
            th["molecule"] = g_corr(engine, confs[0])
            for who in ("parent", "with_c"):
                for f, fn, ts in (("r", "start.xyz", False), ("p", "end.xyz", False), ("ts", "ts.xyz", True)):
                    fp = out / who / "reopt" / fn
                    if fp.exists():
                        th[f"{who}_{f}"] = g_corr(engine, mepd.load(str(fp)), ts=ts)
            for f in ("r", "ts", "p"):
                if f"parent_{f}" in th and f"with_c_{f}" in th:
                    th[f"binding_g_corr_{f}"] = th[f"with_c_{f}"]["g_corr"] - th[f"parent_{f}"]["g_corr"] - th["molecule"]["g_corr"]
        except Exception as exc:
            th["error"] = f"{type(exc).__name__}: {exc}"
        rec["thermo"], rec["thermo_s"] = th, time.perf_counter() - t
    (out / "record.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("profile")
    ap.add_argument("smiles")
    ap.add_argument("out", type=Path)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--placer", default="seeded")
    ap.add_argument("--poses", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--carry", type=int, default=3)
    ap.add_argument("--polish", type=int, default=2)
    ap.add_argument("--conformers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--relax-substrate", action="store_true")
    a = ap.parse_args()
    import multiprocessing as mp

    import mepd
    from mepd import bystander as B

    a.out.mkdir(parents=True, exist_ok=True)
    confs = B.conformers(a.smiles, mepd.engine_from_profile(a.profile), n=a.conformers)
    csym, cq = list(confs[0].structure.symbols), int(confs[0].structure.charge)
    jobs = [(str(a.run_dir), a.profile, lab, a.placer, a.poses, a.repeats, a.seed, a.relax_substrate,
             [B._xyz(c) for c in confs], csym, cq, str(a.out), a.carry, a.polish) for lab in a.labels.split(",")]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        for rec in pool.map(task, jobs):
            e, w = rec["estimate"], rec.get("with_c", {})
            print(rec["label"], e["status"], e["barrier_shift"]["electronic"], "| verified", w.get("status"),
                  rec.get("verified_barrier_shift_kcal"), f"| {rec['estimate_s']:.0f}s est, {rec.get('with_c_s', 0):.0f}s verify",
                  flush=True)
    (a.out / "run.json").write_text(json.dumps({"wall_s": time.perf_counter() - t0, "mepd_file": mepd.__file__,
                                                "args": {k: str(v) for k, v in vars(a).items()}}, indent=1))


if __name__ == "__main__":
    main()
