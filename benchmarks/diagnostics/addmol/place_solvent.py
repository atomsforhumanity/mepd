"""Gas-phase search with solvated single points against solvated throughout, for the bystander estimate.

    python benchmarks/diagnostics/addmol/place_solvent.py RUN_DIR GAS_PROFILE SOLV_PROFILE SMILES OUT_DIR
        --labels a,b [--solv-smiles S2] [--placer seeded] [--poses 6] [--repeats 3] [--carry 3] [--polish 2]
        [--conformers 3] [--workers 8]

Per step, one process on one thread runs mepd.bystander.bystander twice: (1) search at GAS_PROFILE with single
points at SOLV_PROFILE on the three lowest poses of each frame and repeat (the molecule's conformers minimized
in the gas phase); (2) search at SOLV_PROFILE throughout (conformers minimized there; --solv-smiles: another
protonation form for this arm, e.g. a zwitterion that is no minimum in the gas phase). The parent frames are
the gas-phase run's in both. Written per step: both results with their seconds. The first lines of run.json
give the molecule's forms: energies of every conformer at both levels.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def task(a):
    run_dir, gas, solv, label, mol_gas, mol_solv, kw, out = a
    import mepd
    from mepd import bystander as B
    from place_measure import load_step

    st, fr = load_step(Path(run_dir), label)
    frames = {"symbols": list(fr["ts"].structure.symbols), "charge": int(fr["ts"].structure.charge), "multiplicity": 1,
              **{f: B._xyz(fr[f]).tolist() for f in fr}}
    eg, es = mepd.engine_from_profile(gas), mepd.engine_from_profile(solv)
    t = time.perf_counter()
    a1 = B.bystander(frames, mol_gas, eg, final_engine=es, **kw)
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    a2 = B.bystander(frames, mol_solv, es, **kw)
    t2 = time.perf_counter() - t
    rec = {"label": label, "reacting": st.get("reacting"), "gas_search": a1, "gas_search_s": t1,
           "solvated_search": a2, "solvated_search_s": t2}
    Path(out, f"{label}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("gas")
    ap.add_argument("solv")
    ap.add_argument("smiles")
    ap.add_argument("out", type=Path)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--solv-smiles", default=None)
    ap.add_argument("--placer", default="seeded")
    ap.add_argument("--poses", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--carry", type=int, default=3)
    ap.add_argument("--polish", type=int, default=2)
    ap.add_argument("--conformers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import multiprocessing as mp

    import mepd
    from mepd import bystander as B

    a.out.mkdir(parents=True, exist_ok=True)
    eg, es = mepd.engine_from_profile(a.gas), mepd.engine_from_profile(a.solv)
    mol_gas = B.molecule_from_smiles(a.smiles, eg, n=a.conformers)
    mol_solv = B.molecule_from_smiles(a.solv_smiles or a.smiles, es, n=a.conformers)
    forms = {}
    for name, mol in (("gas_minimized", mol_gas), ("solvent_minimized", mol_solv)):
        nodes = [B._node(mol["symbols"], c, mol["charge"]) for c in mol["conformers"]]
        forms[name] = {"smiles": mol["smiles"], "gas_energy": [B._energy(eg, n) for n in nodes],
                       "solvated_energy": [B._energy(es, n) for n in nodes]}
    kw = {"placer": a.placer, "poses": a.poses, "repeats": a.repeats, "carry": a.carry, "polish": a.polish}
    jobs = [(a.run_dir, a.gas, a.solv, lab, mol_gas, mol_solv, kw, str(a.out)) for lab in a.labels.split(",")]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        for r in pool.map(task, jobs):
            g, s = r["gas_search"], r["solvated_search"]
            print(r["label"], "gas", g["status"], g["barrier_shift"]["electronic"], "gas+sp",
                  (g.get("final") or {}).get("barrier_shift", {}).get("electronic"), "solv", s["status"],
                  s["barrier_shift"]["electronic"], f"{r['gas_search_s']:.0f}s / {r['solvated_search_s']:.0f}s", flush=True)
    (a.out / "run.json").write_text(json.dumps({"forms": forms, "wall_s": time.perf_counter() - t0,
                                                "mepd_file": mepd.__file__, "args": {k: str(v) for k, v in vars(a).items()}}, indent=1))


if __name__ == "__main__":
    main()
