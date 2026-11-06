"""Oracle test: the ceiling's structures re-optimized at GFN2-xTB (axtb), then the explorer's ceiling on them.

    python oracle.py SYSTEM_DIR [--workers N] [--no-ceiling] [--prefill N]

The user's principle: GFN2-xTB or g-xTB only provide the geometries (and Hessians) the steering level decides on.
This re-runs every TS of SYSTEM_DIR/ceiling_spec.json, and its mirror twin, at GFN2-xTB:
- mepd.find_ts from the g-xTB TS (Sella, exact Hessian, fmax 0.001 eV/A, exactly one imaginary mode), then mepd.irc
  with minimized ends; the ends' SMILES against the g-xTB TS's IRC ends (survival = found, one imaginary mode,
  same connectivity); all-atom RMSD to the g-xTB TS after the best rotation (same atom order);
- the seed re-minimized at GFN2; product species = the GFN2 IRC product ends;
- timing on axtb, one thread: the GFN2 TS search + IRC from the g-xTB TS, and the g-xTB (axtb, gxtb conventions)
  TS search + IRC from the GFN2 TS (the same displacement the other way), plus one energy+gradient and one analytic
  Hessian at each level on each TS.
Then SYSTEM_DIR/oracle_gfn2/ceiling_spec.json (the same network on GFN2 structures; profile = GFN2 through axtb;
steering single points as before, products unsteered) and, unless --no-ceiling, its ceiling
(oracle_gfn2/ceiling_explorer_level.json): thermal corrections from GFN2 Hessians, as the explorer would make them
with a GFN2 oracle.

GFN2 through mepd: the g-xTB engine's axtb backend with axtb's method switched to "gfn2" in this process
(mepd.engines.axtb_backend.METHOD); explorer code is not changed.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from explorer_level import HARTREE, angstrom, load, read_frames, xyz_text  # noqa: E402

GFN2_PROFILE = """engine_name = "gxtb"
program = "xtb"
[gxtb_engine_kwds]
backend = "axtb"
"""
GXTB_PROFILE = GFN2_PROFILE + 'axtb_conventions = "gxtb"\n'
UMA_PROFILE = """engine_name = "mlip"
[mlip_engine_kwds]
model = "uma-s-1p2"
device = "{device}"
"""
ORACLE = {"name": "gfn2", "device": "cuda"}     # set by main: "gfn2" (axtb) or "uma" (mepd's mlip engine)


def oracle_engine(outd: Path):
    """The oracle's engine (GFN2: axtb with the method switched; UMA: the mlip engine on ORACLE's device)."""
    if ORACLE["name"] == "gfn2":
        use_gfn2(True)
        return engine(GFN2_PROFILE, outd, ".oracle.toml")
    use_gfn2(False)
    return engine(UMA_PROFILE.format(device=ORACLE["device"]), outd, ".oracle.toml")


def use_gfn2(on: bool = True) -> None:
    import mepd.engines.axtb_backend as ab

    ab.METHOD = "gfn2" if on else "gxtb-v2.0.1"
    ab._LOCAL.__dict__.pop("cache", None)


def engine(profile_text: str, d: Path, name: str):
    """The engine of a profile. Each process writes its own copy first and reads it back whole (forked workers
    sharing one file read it half-written, got no [gxtb_engine_kwds] and so the g-xTB program instead of GFN2)."""
    from mepd.inputs import RunInputs

    p = d / f"{name}.{os.getpid()}"
    p.write_text(profile_text)
    eng = RunInputs.open(str(p)).engine
    (d / name).write_text(profile_text)          # the copy the ceiling spec names (rewritten whole each time)
    return eng


def rmsd(X, Y) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(X.T @ Y)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dd]) @ U.T
    return float(np.sqrt((((X @ R.T) - Y) ** 2).sum() / len(X)))


def search(guess_st, eng, out: Path, tag: str) -> dict:
    """TS search + IRC at `eng` from `guess_st`; writes out/<tag>_ts.xyz and _irc.xyz."""
    import mepd
    from mepd.nodes.node import StructureNode

    from collect_ts import smiles

    rec = {}
    t0 = time.time()
    try:
        ts = mepd.find_ts(StructureNode(structure=guess_st), eng)
        rec["t_ts"] = time.time() - t0
        e = float(mepd.energies([ts], eng)[0])
        w = ts.frequencies.wavenumbers_cm
        rec.update(energy=e, imag_cm=float(w[0]), n_imag=int((w < -10).sum()), lowest_cm=[float(x) for x in w[:3]])
        (out / f"{tag}_ts.xyz").write_text(xyz_text(ts.structure, f"E {e:.10f}"))
        t1 = time.time()
        res = mepd.irc(ts, eng)
        rec["t_irc"] = time.time() - t1
        a, b = res.ends
        ea, eb = (float(x) for x in mepd.energies([a, b], eng))
        (out / f"{tag}_irc.xyz").write_text(xyz_text(a.structure, f"E {ea:.10f}") + xyz_text(b.structure, f"E {eb:.10f}"))
        rec.update(ends=[smiles(a.structure), smiles(b.structure)], end_energies=[ea, eb], ok=True)
    except Exception as exc:
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:500])
    rec["t_total"] = time.time() - t0
    return rec


def per_call(st, eng) -> dict:
    import mepd
    from mepd.nodes.node import StructureNode

    t0 = time.time()
    for _ in range(3):
        eng.compute_gradients([StructureNode(structure=st)])
    t1 = time.time()
    mepd.frequencies(StructureNode(structure=st), eng)
    t2 = time.time()
    return {"energy_gradient_s": (t1 - t0) / 3, "hessian_s": t2 - t1}


def one_ts(job) -> dict:
    """job: (label, g-xTB ts.xyz, g-xTB irc.xyz or None, g-xTB result.json or None, outdir)."""
    label, ts_path, irc_path, res_path, outd = job
    outd = Path(outd)
    fp = outd / f"{label}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    rec = {"label": label, "gxtb_ts": ts_path}
    try:
        g_st = load(ts_path)
        if res_path and Path(res_path).exists():
            r = json.loads(Path(res_path).read_text())
            rec["gxtb_ends"] = r.get("ends")
            rec["gxtb_energy"] = r.get("energy")
            rec["gxtb_imag_cm"] = r.get("imag_cm")
        K = ORACLE["name"]
        e2 = oracle_engine(outd)
        rec[K] = search(g_st, e2, outd, label)
        rec["engine"] = repr(e2)[:200]
        if K == "gfn2":
            import mepd.engines.axtb_backend as ab

            rec["axtb_method"] = ab.METHOD
            rec["backend"] = getattr(e2, "backend", None)
        rec[f"{K}_call"] = per_call(g_st, e2)
        use_gfn2(False)
        eg = engine(GXTB_PROFILE, outd, ".gxtb.toml")
        rec["gxtb_call"] = per_call(g_st, eg)
        if rec[K].get("ok"):
            g2 = load(outd / f"{label}_ts.xyz")
            rec["rmsd_A"] = rmsd(angstrom(g2), angstrom(g_st))
            if K == "gfn2":
                rec["gxtb_from_gfn2"] = search(g2, eg, outd, f"{label}__gxtb_from_gfn2")
            ends2 = sorted(rec[K]["ends"])
            rec["same_connectivity"] = (ends2 == sorted(rec.get("gxtb_ends") or [])) if rec.get("gxtb_ends") else None
        rec["survives"] = bool(rec[K].get("ok") and rec[K].get("n_imag") == 1 and rec.get("same_connectivity"))
    except Exception as exc:
        rec.update(error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
    finally:
        use_gfn2(False)
    fp.write_text(json.dumps(rec, indent=1))
    return rec


def twins_of(sysdir: Path, ts_dir: str) -> list:
    """The mirror twin(s) of a ceiling TS among SYSTEM_DIR/ts_refined/distinct.json groups."""
    fp = sysdir / "ts_refined" / "distinct.json"
    if not fp.exists():
        return []
    groups = json.loads(fp.read_text())
    mine = next((g for g in groups if ts_dir in g["members"] or g["name"] == ts_dir), None)
    if mine is None:
        return []
    return [g["name"] for g in groups if g.get("mirror_of") == mine["name"]
            or (mine.get("mirror_of") and g["name"] == mine["mirror_of"])]


def main(argv):
    workers, prefill, do_ceiling = 1, 1, True
    if "--workers" in argv:
        i = argv.index("--workers")
        workers = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--prefill" in argv:
        i = argv.index("--prefill")
        prefill = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--no-ceiling" in argv:
        do_ceiling = False
        argv.remove("--no-ceiling")
    if "--oracle" in argv:
        i = argv.index("--oracle")
        ORACLE["name"] = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if "--device" in argv:
        i = argv.index("--device")
        ORACLE["device"] = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    for flag, store, key in (("--export-sp", EXPORT, "dir"), ("--import-sp", IMPORT, "file")):
        if flag in argv:
            i = argv.index(flag)
            store[key] = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
    K = ORACLE["name"]
    if K == "uma":
        workers = 1          # one process holds the GPU model (no forked workers)
    sysdir = Path(argv[0]).resolve()
    out = sysdir / f"oracle_{K}"
    out.mkdir(exist_ok=True)
    spec = json.loads((sysdir / "ceiling_spec.json").read_text())
    jobs, roles = [], {}
    for st in spec["steps"]:
        tsd = Path(st["xyz"]).parent
        res = sysdir / tsd / "result.json"
        jobs.append((st["label"], str(sysdir / st["xyz"]), None, str(res), str(out)))
        roles[st["label"]] = "ceiling"
        for k, tw in enumerate(twins_of(sysdir, tsd.name)):
            lab = f"{st['label']}__twin{k}"
            jobs.append((lab, str(sysdir / "ts_refined" / tw / "ts.xyz"), None,
                         str(sysdir / "ts_refined" / tw / "result.json"), str(out)))
            roles[lab] = "mirror twin"
    if workers > 1:
        from multiprocessing import get_context

        with get_context("fork").Pool(workers) as pool:
            recs = pool.map(one_ts, jobs, chunksize=1)
    else:
        recs = [one_ts(j) for j in jobs]
    # the seed at GFN2
    import mepd
    from mepd.nodes.node import StructureNode

    e2 = oracle_engine(out)
    seed_fp = out / f"seed_{K}.xyz"
    if not seed_fp.exists():
        t0 = time.time()
        s = mepd.minimize(StructureNode(structure=load(sysdir / spec["species"][0]["xyz"])), e2)
        e = float(mepd.energies([s], e2)[0])
        seed_fp.write_text(xyz_text(s.structure, f"seed at {K} E {e:.10f} ({time.time() - t0:.1f} s)"))
        rmsd_seed = rmsd(angstrom(s.structure), angstrom(load(sysdir / spec["species"][0]["xyz"])))
        (out / "seed.json").write_text(json.dumps({"rmsd_A": rmsd_seed}))
    # the GFN2 network: every surviving ceiling TS, its product end as the product species
    from collect_ts import smiles

    by = {r["label"]: r for r in recs}
    species = [{"label": spec["species"][0]["label"], "xyz": str(seed_fp)}]
    steps = []
    used = {}
    for st in spec["steps"]:
        # the conformer at this oracle: the ceiling's TS, or its mirror twin when only the twin survived (the mirror
        # images then supply the other face, as they do for the explorer)
        cands = [st["label"]] + sorted(l for l in by if l.startswith(st["label"] + "__twin"))
        lab = next((c for c in cands if by[c].get("survives")), None)
        if lab is None:
            continue
        used[st["label"]] = lab
        r = by[lab]
        fr = read_frames(out / f"{lab}_irc.xyz")
        seed_smi = smiles(load(sysdir / spec["species"][0]["xyz"]))
        k = [i for i, sm in enumerate(r[K]["ends"]) if sm != seed_smi]
        k = k[0] if k else 1
        sym, X = fr[k]
        from explorer_level import structure

        pfile = out / f"{st['label']}_product.xyz"
        pfile.write_text(xyz_text(structure(sym, X), f"{K} IRC product end"))
        prod = spec["species"][st["b"]].get("product", "")
        species.append({"label": f"{st['label']}_product", "xyz": str(pfile), "product": prod})
        steps.append({**st, "xyz": str(out / f"{lab}_ts.xyz"), "irc": str(out / f"{lab}_irc.xyz"),
                      "a": 0, "b": len(species) - 1})
    gspec = {**spec, "profile": str(out / ".oracle.toml"), "cache": str(out / "ceiling_cache.json"),
             "species": species, "steps": steps}
    gspec.pop("steer_species", None)   # every species steered, as the explorer does (GFN2 product energies are far off)
    if not Path(gspec["steer"]).is_absolute():
        gspec["steer"] = str(sysdir / gspec["steer"])
    (out / "ceiling_spec.json").write_text(json.dumps(gspec, indent=1))
    summary = {"n_ts": len(recs), "survived": sum(bool(r.get("survives")) for r in recs),
               "lost": [r["label"] for r in recs if not r.get("survives")], "roles": roles,
               "network_uses": used, "conformers_lost": [st["label"] for st in spec["steps"] if st["label"] not in used]}
    (out / "survival.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    for r in recs:
        g = r.get(K, {})
        print(f"{r['label']:<28} {roles[r['label']]:<12} survives={r.get('survives')} imag {g.get('imag_cm', 0):8.1f} "
              f"n_imag {g.get('n_imag')} rmsd {r.get('rmsd_A', float('nan')):.3f} A  ends {g.get('ends')} "
              f"t({K} ts+irc) {g.get('t_total', 0):.1f} s  t(g-xTB ts+irc) {r.get('gxtb_from_gfn2', {}).get('t_total', 0):.1f} s "
              f"{r.get('error', '') or g.get('error', '')}")
    if "--run-todo" in sys.argv:            # (handled in __main__)
        pass
    if IMPORT.get("file"):
        from explorer_level import Cache

        c = Cache(out / "ceiling_cache.json")
        for k, v in json.loads(Path(IMPORT["file"]).read_text()).items():
            c.put(k, v)
    if do_ceiling:
        import ceiling

        prefill_threads(out / "ceiling_spec.json", max(1, prefill))
        ceiling.main(str(out / "ceiling_spec.json"), str(out / "ceiling_explorer_level.json"), 1)


def prefill_threads(spec_path: Path, workers: int) -> None:
    """Thermal corrections and oracle energies in this process (the oracle may be a GPU model: no forking), the
    steering single points (ORCA subprocesses) on `workers` threads; all into the ceiling's cache."""
    from concurrent.futures import ThreadPoolExecutor

    import ceiling
    from explorer_level import ExplorerLevel, geo_key
    from mepd.discovery.energy_refine import _energy, _fragments

    spec = json.loads(spec_path.read_text())
    if ORACLE["name"] == "gfn2":
        use_gfn2(True)
    level = ExplorerLevel(spec["profile"], spec["steer"], spec["temperature"], spec["standard_state"], spec["cache"])
    net = ceiling.build(spec, spec_path.parent)
    items = [(s.st, False, None) for s in net.species] + [(t.st, True, t.ends) for t in net.steps]
    todo = []
    for st, ts, ends in items:
        level.energy(st)
        level.correction(st, ts, ends)
        groups = level.free._groups(st, ts, ends)
        key = "sp:" + geo_key(st, (ts,) + ((groups,) if groups and len(groups) > 1 else ()))
        if level.cache.get(key) is None:
            parts = _fragments(st, level.parts(st, ts, ends))
            todo.append((key, parts, sum(_energy(p, level.engine) for p in parts)))

    def high(item):
        key, parts, e_low = item
        e_high = sum(_energy(p, level.steer) for p in parts)
        return key, {"e_low": e_low, "e_high": e_high, "delta_kcal": (e_high - e_low) * HARTREE}

    if EXPORT.get("dir"):
        # the steering single points elsewhere (a GPU oracle here, ORCA on the cluster): sp_todo.json
        rows = [{"key": key, "e_low": e_low,
                 "parts": [{"xyz": xyz_text(p, ""), "charge": int(p.charge), "multiplicity": int(p.multiplicity)}
                           for p in parts]} for key, parts, e_low in todo]
        Path(EXPORT["dir"], "sp_todo.json").write_text(json.dumps(rows))
        print(f"{len(rows)} steering single points exported to {EXPORT['dir']}/sp_todo.json")
        raise SystemExit(0)
    with ThreadPoolExecutor(workers) as pool:
        for key, val in pool.map(high, todo):
            level.cache.put(key, val)


EXPORT: dict = {}
IMPORT: dict = {}


def run_todo(todo_path: str, steer_profile: str, workers: int) -> None:
    """ORCA single points for an exported sp_todo.json -> sp_done.json beside it ({key: {e_low, e_high, delta_kcal}})."""
    from concurrent.futures import ThreadPoolExecutor

    from explorer_level import structure
    from mepd.discovery.energy_refine import _energy
    from mepd.inputs import RunInputs

    steer = RunInputs.open(steer_profile).engine
    rows = json.loads(Path(todo_path).read_text())

    def one(r):
        e_high = 0.0
        for p in r["parts"]:
            sym, X = read_frames_text(p["xyz"])
            e_high += _energy(structure(sym, X, p["charge"], p["multiplicity"]), steer)
        return r["key"], {"e_low": r["e_low"], "e_high": e_high, "delta_kcal": (e_high - r["e_low"]) * HARTREE}

    with ThreadPoolExecutor(workers) as pool:
        done = dict(pool.map(one, rows))
    Path(todo_path).with_name("sp_done.json").write_text(json.dumps(done, indent=0))


def read_frames_text(text: str):
    lines = text.splitlines()
    n = int(lines[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if sys.argv[1] == "--run-todo":          # oracle.py --run-todo sp_todo.json orca_steer.toml WORKERS
        run_todo(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main(sys.argv[1:])
