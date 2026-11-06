"""Reactor trajectories through the generator's own trajectory + quench + key: sample.py XYZ OUT N SEED0 WIDTH key=val...
Writes OUT/s_XXXX.json: events [(frame, key, from_seed)], states xyz per new key, wall times."""
import sys, os, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def read(fp):
    L = open(fp).read().splitlines(); n = int(L[0])
    return [l.split()[0] for l in L[2:2+n]], np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2+n]])

def one(a):
    xyz, out, seed, opts = a
    if os.environ.get('MD'):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import md_axtb; md_axtb.install(os.environ['MD'])
    from qcdata import Structure
    from qcref.constants import ANGSTROM_TO_BOHR
    if os.environ.get('RG') == 'cyc':
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import rg_cyc as rg
    else:
        from mepd.discovery import reactor_generator as rg
    from mepd.discovery.nanoreactor import DetectSettings, bond_history, detect_events
    from mepd.discovery.nanoreactor_generator import stereo_key
    sy, x0 = read(xyz)
    o = {k: v for k, v in rg.OPTIONS.items() if k not in ("n_seeds", "unseen", "max_batches", "parallel", "cutoff_kcal")}
    o.update(opts); o["seed"] = seed; o["workdir"] = ""; o["intra_hops"] = int(o["intra_hops"])
    r0 = sys.modules['md_axtb'].RETRIES[0] if 'md_axtb' in sys.modules else 0
    t0 = time.time(); fr = rg._trajectory(sy, x0, 0, 1, **o); t_md = time.time() - t0
    det = DetectSettings(); hist = bond_history(sy, fr, det, rg.TRAJ_FS); evs = detect_events(len(sy), hist, det, rg.TRAJ_FS)
    st = Structure(symbols=sy, geometry=x0 * ANGSTROM_TO_BOHR, charge=0, multiplicity=1)
    seedk = prev = stereo_key(st); _, e0 = rg._quench(sy, x0, 0, 1); b0 = {a for p in hist.initial for a in p}
    rec, states, torn, nq = [], {}, 0, 0
    t1 = time.time()
    for ev in sorted(evs, key=lambda e: (e.product_frame, e.start)):
        if b0 - {a for p in hist.bonds_at(ev.product_frame) for a in p}:
            torn += 1; continue
        q, e = rg._quench(sy, fr[ev.product_frame], 0, 1); nq += 1
        k = stereo_key(st.model_copy(update={"geometry": q * ANGSTROM_TO_BOHR}))
        if k is None or k == prev: continue
        rec.append([int(ev.product_frame), k, prev == seedk, None if e is None else e - e0]); prev = k
        states.setdefault(k, q.tolist())
    retries = sys.modules["md_axtb"].RETRIES[0] - r0 if "md_axtb" in sys.modules else None
    r = {"seed": seed, "seed_key": seedk, "md": os.environ.get("MD") or "tblite-gfn2", "scf_retries": retries, "frames": len(fr), "events": rec, "n_events": len(evs), "torn": torn,
         "t_md": t_md, "t_quench": time.time() - t1, "n_quench": nq, "states": states}
    json.dump(r, open(f"{out}/s_{seed:04d}.json", "w"))
    return seed, t_md, time.time() - t1

if __name__ == "__main__":
    xyz, out, n, s0, w = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    opts = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in sys.argv[6:]}
    os.makedirs(out, exist_ok=True)
    json.dump({"xyz": xyz, "opts": opts}, open(f"{out}/meta.json", "w"))
    with ProcessPoolExecutor(w) as ex:
        for seed, a, b in ex.map(one, [(xyz, out, s0 + k, opts) for k in range(n)]):
            print(f"seed {seed}: md {a:.1f} s quench {b:.1f} s", flush=True)
