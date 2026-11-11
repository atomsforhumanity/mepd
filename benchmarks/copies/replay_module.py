"""team36: the replay of replay_molecules.py with mepd.discovery.molecules (the code the explorer runs), to check the
module gives the replay's answer on the recorded runs.

    replay_module.py OUT_DIR RUN_DIR [RUN_DIR ...] [--jobs N]

Per run: the box network (score._kinetics_network + mirror images) through molecules.build (TS info from the run's
files: <ts>_ends.xyz beside a TS, files["verified"]), molecules.simulate at the run's T and time from c0 = 1 per
seed molecule with every channel product absorbing, channel amounts by the non-negative split of
replay_molecules.py; writes OUT_DIR/replay_module.json."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import mepd  # noqa: F401  (this worktree's, before knobs puts the scoreboard's first)

sys.path.insert(0, "/home/user/astera/data/runs/explorer-team/team30/scripts")
import knobs  # noqa: E402

S = knobs.S


def one(run: str) -> dict:
    from mepd.discovery import molecules as M
    from mepd.discovery.kinetics import Step as BStep
    from mepd.discovery.mirror import Mirrors
    from mepd.qcdata_structure_helpers import read_multiple_structure_from_file

    systems = S.load_systems()
    R = knobs.Run(Path(run), systems)
    s = R.s
    base = R.evaluate()
    g0, ts0, opened, _ = S._kinetics_network(s)
    steps = s["steps"]
    T, t_end = float(s["steering"]["temperature_K"]), float(s["steering"]["time_s"])
    rec = s.get("mirror_images") or {}
    n0 = len(g0)
    paths = S.Paths(s, Path(run) / "run")
    ks = [BStep(st["a"], st["b"], ts0[m] + (0.0 if m in opened else 1.0), st.get("label", ""),
                {**dict(st.get("files") or {}), **({} if m in opened else {"closed": True})})
          for m, st in enumerate(steps)]
    g = list(g0)
    virtual_of, twin_of, found = [], [], []
    if rec:
        mir = Mirrors(n0, [v["mirror_of"] for v in rec["virtual_species"]], rec["species_mirror"],
                      {int(k): v for k, v in rec["chiral_conformers"].items()},
                      [(t["step"], t["a"], t["b"]) for t in rec["added_twins"]], [tuple(p) for p in rec["found_twins"]])
        _, ks, g = mir.extend(ks, g, T)
        virtual_of = [v["mirror_of"] for v in rec["virtual_species"]]
        twin_of = [t["step"] for t in rec["added_twins"]]
        found = [tuple(p) for p in rec["found_twins"]]
    structs = {}

    def struct(k):
        if k not in structs:
            structs[k] = read_multiple_structure_from_file(str(paths(s["species"][k]["file"])), 0, 1)[0]
        return structs[k]

    def ts_info(m):
        fp = paths((steps[m].get("files") or {}).get("ts"))
        if fp is None or not Path(fp).is_file():
            return None, False, (steps[m].get("files") or {}).get("verified") is not False
        ts = read_multiple_structure_from_file(str(fp), 0, 1)[0]
        return ts, Path(fp).with_name(Path(fp).stem + "_ends.xyz").is_file(), \
            (steps[m].get("files") or {}).get("verified") is not False

    net = M.build(n0, lambda k: M.molecule_parts(struct(k)), lambda k: M.bonds_of(struct(k)), ks, g,
                  virtual_of, twin_of, found, ts_info)
    chans = R.sysdef["channels"]

    def matches(k, p, racemic):
        if S._has_stereo(p):
            return k in {S._iso(p)} | ({S._iso(p, mirror=True)} if racemic else set())
        return S._canon(k) == S._canon(p)

    def pid(p):
        return ("iso", S._iso(p)) if S._has_stereo(p) else ("canon", S._canon(p))

    cm = {(n, p): [i for i, k in enumerate(net.names) if matches(k, p, chans[n].get("racemic"))]
          for n in chans for p in chans[n]["products"]}
    absorbing = sorted({i for v in cm.values() for i in v} - set(net.seed))
    kin = M.simulate(net, net.c0(1.0), T, t_end, absorbing)
    from scipy.optimize import nnls

    pidx = {}
    for n in chans:
        for p in chans[n]["products"]:
            pidx.setdefault(pid(p), len(pidx))
    names = list(chans)
    A = np.zeros((len(pidx), len(names)))
    b = np.zeros(len(pidx))
    for c, n in enumerate(names):
        for p in chans[n]["products"]:
            A[pidx[pid(p)], c] += 1
    seen = set()
    for (n, p), idx in cm.items():
        for i in idx:
            if (i, pid(p)) not in seen:
                seen.add((i, pid(p)))
                b[pidx[pid(p)]] += kin.final[i]
    sc = max(b.max(), 1e-300)
    nv, _ = nnls(A, b / sc)
    amt = dict(zip(names, nv * sc))
    sels = []
    for t in R.sysdef["targets"]:
        x = sum(amt[c] for c in t["num"])
        y = sum(amt[c] for c in t["den"])
        sels.append(x / y if y > 0 else (math.inf if x > 0 else None))
    return {"run": R.name, "system": R.system, "box": [t["selectivity"] for t in base.get("targets", [])],
            "module": sels, "counts": net.counts, "fit": net.fit, "n_molecules": len(net.names)}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv[1:])
    print("mepd:", mepd.__file__, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r, f in [(r, ex.submit(one, r)) for r in a.runs]:
            try:
                x = f.result()
            except Exception as exc:
                import traceback
                x = {"run": Path(r).name, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()}
            res.append(x)
            print(json.dumps({k: x.get(k) for k in ("run", "error", "box", "module", "counts")}, default=str), flush=True)
    (out / "replay_module.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
