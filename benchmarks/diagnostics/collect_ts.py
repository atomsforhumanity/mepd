"""Refine TS candidates at the explorer's level, check them, classify them by their IRC ends, deduplicate.

    python collect_ts.py PROFILE OUTDIR GUESS.xyz [GUESS.xyz ...] [--workers N]

Each guess: mepd.find_ts with the engine of PROFILE (the protocol's defaults: fmax 0.001 eV/A, exact Hessian,
exactly one imaginary mode below -10 cm-1), then mepd.irc with both ends minimized. Written to OUTDIR/<name>/:
ts.xyz (energy in the comment), irc.xyz (the minimized ends: first and last frame), result.json (energy, imaginary
frequency, the ends' SMILES). OUTDIR/index.json lists every candidate; a guess already done is skipped.

Deduplication (`distinct`): the explorer's own `_same_ts` rule -- same ends, |dE| < 0.1 kcal/mol and superimposable
(mepd.discovery.mirror.superimposable, permutations of identical atoms) -- and mirror twins marked (superimposable
on the mirror image).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from explorer_level import HARTREE, angstrom, load, read_frames, structure, xyz_text  # noqa: E402


def smiles(st) -> str:
    from mepd.discovery.network_expansion import _display_smiles
    from rdkit import Chem

    s = _display_smiles(st) or "?"
    m = Chem.MolFromSmiles(s, sanitize=False)
    if m is not None:
        try:
            return ".".join(sorted(Chem.MolToSmiles(m, isomericSmiles=False).split(".")))
        except Exception:
            pass
    return s


def refine_one(args) -> dict:
    profile, guess, out = args
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    import mepd
    from mepd.inputs import RunInputs
    from mepd.nodes.node import StructureNode

    rec = {"guess": str(guess), "name": out.name}
    try:
        eng = RunInputs.open(profile).engine
        node = StructureNode(structure=load(guess))
        ts = mepd.find_ts(node, eng)
        e = float(mepd.energies([ts], eng)[0])
        w = ts.frequencies.wavenumbers_cm
        rec.update(energy=e, imag_cm=float(w[0]), lowest_cm=[float(x) for x in w[:4]])
        (out / "ts.xyz").write_text(xyz_text(ts.structure, f"E {e:.10f}"))
        res = mepd.irc(ts, eng)
        a, b = res.ends
        ea, eb = (float(x) for x in mepd.energies([a, b], eng))
        (out / "irc.xyz").write_text(xyz_text(a.structure, f"E {ea:.10f}") + xyz_text(b.structure, f"E {eb:.10f}"))
        rec.update(end_energies=[ea, eb], ends=[smiles(a.structure), smiles(b.structure)], ok=True)
    except Exception as exc:
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])
    (out / "result.json").write_text(json.dumps(rec, indent=1))
    return rec


def distinct(recs: list, outdir: Path) -> list:
    """Group refined TSs: [{rep, members, mirror_of}] with the explorer's same-TS rule."""
    from mepd.discovery.mirror import superimposable

    groups = []
    for r in sorted((r for r in recs if r.get("ok")), key=lambda r: r["energy"]):
        st = load(outdir / r["name"] / "ts.xyz")
        X = angstrom(st)
        ends = tuple(sorted(r["ends"]))
        hit = None
        for g in groups:
            if g["ends"] != ends or abs(g["energy"] - r["energy"]) * HARTREE >= 0.1:
                continue
            if superimposable([str(s) for s in st.symbols], g["X"], X):
                hit = g
                break
        if hit is not None:
            hit["members"].append(r["name"])
            continue
        g = {"name": r["name"], "energy": r["energy"], "ends": ends, "X": X, "members": [r["name"]],
             "imag_cm": r["imag_cm"], "mirror_of": None, "symbols": [str(s) for s in st.symbols]}
        Y = X.copy()
        Y[:, 0] *= -1
        for o in groups:
            if o["ends"] == ends and abs(o["energy"] - r["energy"]) * HARTREE < 0.1 and \
                    superimposable(g["symbols"], o["X"], Y):
                g["mirror_of"] = o["name"]
                break
        g["chiral"] = not superimposable(g["symbols"], X, Y)
        groups.append(g)
    return [{k: v for k, v in g.items() if k not in ("X", "symbols")} for g in groups]


def main(argv):
    workers = 1
    if "--workers" in argv:
        i = argv.index("--workers")
        workers = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    profile, outdir, guesses = argv[0], Path(argv[1]), argv[2:]
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for gpath in guesses:
        gp = Path(gpath)
        # unique per guess file: its last path parts plus a short hash of the whole path
        import hashlib

        tag = hashlib.sha1(str(gp.resolve()).encode()).hexdigest()[:6]
        name = "__".join(gp.parts[-4:]).replace(".xyz", "").replace("/", "_") + "__" + tag
        jobs.append((profile, str(gp), str(outdir / name)))
    if workers > 1:
        from multiprocessing import get_context

        with get_context("fork").Pool(workers) as pool:
            recs = pool.map(refine_one, jobs, chunksize=1)
    else:
        recs = [refine_one(j) for j in jobs]
    idx = outdir / "index.json"
    old = json.loads(idx.read_text()) if idx.exists() else []
    names = {r["name"] for r in recs}
    allrecs = [r for r in old if r["name"] not in names] + recs
    idx.write_text(json.dumps(allrecs, indent=1))
    groups = distinct(allrecs, outdir)
    (outdir / "distinct.json").write_text(json.dumps(groups, indent=1))
    e0 = min((g["energy"] for g in groups), default=0.0)
    for g in groups:
        print(f"{g['name'][:60]:<60} {' >> '.join(g['ends']):<40} {(g['energy'] - e0) * HARTREE:7.2f} "
              f"{g['imag_cm']:8.1f} chiral={g['chiral']} mirror_of={g['mirror_of']} n={len(g['members'])}")
    bad = [r for r in allrecs if not r.get("ok")]
    print(f"{len(allrecs)} candidates, {len(allrecs) - len(bad)} refined, {len(groups)} distinct; failed: "
          + ", ".join(f"{r['name']} ({r.get('error', '')[:80]})" for r in bad))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(sys.argv[1:])
