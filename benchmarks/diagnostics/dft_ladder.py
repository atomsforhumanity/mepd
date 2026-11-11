"""Level ladder for a small benchmark: DFT re-optimization + frequencies, IRC check, higher-level single points.

    python dft_ladder.py SPEC.json

SPEC: {"workdir": dir for results (relative to the spec), "temperature": K, "standard_state": "1atm"|"1M",
       "opt": {"method", "basis", "keywords", "threads", "maxcore"}   # the DFT level (ORCA engine of mepd, via qccodec)
       "sp": [{"name", "method", "basis", "keywords", "threads", "maxcore"}, ...],   # single points at the DFT geometry
       "structures": {name: {"xyz": file, "frame": 0, "kind": "min"|"ts", "irc_expect": [smiles, smiles] (TS only)}}}

Per structure (each cached in workdir/<name>.json, so a rerun continues): DFT minimization (mepd.minimize, Hessian
checked) or TS search (mepd.find_ts: ORCA OptTS from an analytic Hessian, exactly one imaginary mode), the DFT
thermal correction exactly as the explorer computes one (FreeEnergies._one: qRRHO from that level's Hessian), for a TS
the IRC (mepd.irc, Sella on ORCA gradients, ends minimized) and its ends' SMILES, then every single point of "sp" at
the DFT geometry and, for comparison, the DFT single point at the input (g-xTB) geometry.
Every calculation goes through mepd's ORCA engine, i.e. qccodec's ORCA codec.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from explorer_level import HARTREE, load, xyz_text  # noqa: E402


def orca(cfg: dict):
    from mepd.engines.orca import OrcaEngine

    exe = cfg.get("executable")   # else OrcaEngine finds it: $MEPD_ORCA, /opt/orca/orca, `orca` on PATH
    return OrcaEngine(method=cfg["method"], basis=cfg["basis"], keywords=cfg.get("keywords", ""),
                      threads=int(cfg.get("threads", 1)), maxcore=int(cfg.get("maxcore", 3000)),
                      cpcm_solvent=cfg.get("cpcm_solvent"), **({"executable": exe} if exe else {}))


def one(name: str, s: dict, spec: dict, base: Path) -> dict:
    import mepd
    from mepd.discovery.free_energy import FreeEnergies
    from mepd.nodes.node import StructureNode

    from collect_ts import smiles

    wd = base / spec["workdir"]
    wd.mkdir(parents=True, exist_ok=True)
    fp = wd / f"{name}.json"
    rec = json.loads(fp.read_text()) if fp.exists() else {"name": name, "kind": s["kind"]}

    def save():
        fp.write_text(json.dumps(rec, indent=1))

    xyz = s["xyz"] if Path(s["xyz"]).is_absolute() else str(base / s["xyz"])
    start = load(xyz, s.get("frame", 0), s.get("charge", 0), s.get("multiplicity", 1))
    opt = orca(spec["opt"])
    T, ss = float(spec["temperature"]), spec["standard_state"]
    try:
        if s["kind"] == "sp":       # single points only, at the given geometry (no optimization, no Hessian)
            rec["opt_xyz"] = xyz_text(start, name)
        if "e_opt_at_input" not in rec and s["kind"] != "sp":
            rec["e_opt_at_input"] = float(opt.compute_energies([StructureNode(structure=start)])[0])
            save()
        if "opt_xyz" not in rec:
            t0 = time.time()
            node = StructureNode(structure=start)
            if s["kind"] == "ts":
                res = mepd.find_ts(node, opt)
            else:
                res = mepd.minimize(node, opt) if len(start.symbols) > 1 else node
            rec["opt_xyz"] = xyz_text(res.structure, name)
            rec["e_opt"] = float(opt.compute_energies([StructureNode(structure=res.structure)])[0])
            fr = getattr(res, "frequencies", None)
            rec["freqs_cm"] = [float(x) for x in fr.wavenumbers_cm] if fr is not None else []
            rec["opt_seconds"] = time.time() - t0
            save()
        from explorer_level import read_frames, structure

        (wd / f"{name}_opt.xyz").write_text(rec["opt_xyz"])
        sym, X = read_frames(wd / f"{name}_opt.xyz")[0]
        st = structure(sym, X, s.get("charge", 0), s.get("multiplicity", 1))
        if "corr_kcal" not in rec and s["kind"] != "sp":
            free = FreeEnergies(opt, T, ss)
            rec["corr_kcal"] = float(free._one(sym, X, int(st.charge), int(st.multiplicity), s["kind"] == "ts"))
            rec["corr_T"], rec["corr_ss"] = T, ss
            save()
        if s["kind"] == "ts" and "irc_ends" not in rec and spec.get("irc", True):
            try:
                res = mepd.irc(StructureNode(structure=st), opt)
                a, b = res.ends
                rec["irc_ends"] = [smiles(a.structure), smiles(b.structure)]
                rec["irc_end_energies"] = [float(x) for x in opt.compute_energies([a, b])]
                (wd / f"{name}_irc_ends.xyz").write_text(xyz_text(a.structure, "end a") + xyz_text(b.structure, "end b"))
                exp = s.get("irc_expect")
                rec["irc_ok"] = (sorted(rec["irc_ends"]) == sorted(exp)) if exp else None
            except Exception as exc:
                rec["irc_error"] = f"{type(exc).__name__}: {exc}"
            save()
        rec.setdefault("sp", {})
        for lev in spec.get("sp", []):
            if lev["name"] in rec["sp"]:
                continue
            t0 = time.time()
            e = float(orca(lev).compute_energies([StructureNode(structure=st)])[0])
            rec["sp"][lev["name"]] = {"energy": e, "seconds": time.time() - t0}
            save()
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["trace"] = traceback.format_exc()[-3000:]
        save()
    return rec


def main(spec_path: str, only=None, workers: int = 1):
    spec = json.loads(Path(spec_path).read_text())
    base = Path(spec_path).resolve().parent
    names = [n for n in spec["structures"] if only is None or n in only]
    if workers > 1:
        from multiprocessing import get_context

        with get_context("fork").Pool(workers) as pool:
            recs = pool.starmap(one, [(n, spec["structures"][n], spec, base) for n in names], chunksize=1)
    else:
        recs = [one(n, spec["structures"][n], spec, base) for n in names]
    for r in recs:
        print(r["name"], r.get("e_opt"), r.get("corr_kcal"), r.get("irc_ends"), r.get("irc_ok"),
              {k: v["energy"] for k, v in r.get("sp", {}).items()}, r.get("error", ""))


if __name__ == "__main__":
    args = sys.argv[1:]
    w = 1
    if "--workers" in args:
        i = args.index("--workers")
        w = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    main(args[0], set(args[1:]) or None, w)
