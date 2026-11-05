"""Benchmark harness for `mepd discovery expand` (reactor generator, access steering, async schedule).

    python benchmarks/score.py run   BENCH OUT [--src WORKTREE] [--budget S] [--workers N] [-- extra expand args]
    python benchmarks/score.py score BENCH OUT          # score a finished run directory again

BENCH is `hcooh` or `f0007`. `run` prints, and saves as OUT/score.json, one line: wall and CPU time,
species and steps, the selectivity read from summary.json against experiment, and whether the
products the selectivity needs were found and connected to the seed.

How the selectivity is read (the same for both benchmarks):
- the kinetics of the run's last round (`rounds[-1].kinetics`), i.e. the run's own temperature,
  time and free energies; mirror-image virtual species are added to the species they mirror;
- `final_concentration` at the run's kinetics time (what an experiment measures as yield);
  the ratio from the integrated inflow (`flux`) is printed too as `sel_flux`;
- HCOOH: dehydration : decarboxylation = sum_k c_k * n_CO(k) / sum_k c_k * n_CO2(k), counting the
  CO and CO2 molecules in each species' SMILES (so a cluster species with one CO counts once);
- F0007: endo : exo from the norbornene ester adduct species (no-stereo SMILES match), each
  classified endo/exo from its geometry (ester carbon nearer the C=C than the CH2 bridge = endo).
`err` is RT ln(sel / nearest experimental bound) in kcal/mol at the kinetics temperature
(0 inside the experimental range): the free-energy error of the selectivity.
"""
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = "/home/user/astera/mepd/.venv/bin/python"
ORCA = "/home/user/.local/opt/orca-6.0.0/orca"
R_KCAL = 1.987204e-3

COMMON = ["--charge", "0", "--multiplicity", "1", "--steer", "access", "--schedule", "async",
          "--generator", "reactor", "--generator-option", "temperature=1800"]
BENCHES = {
    # experiment: dehydration (CO + H2O) : decarboxylation (CO2 + H2) = 6.3-10
    "hcooh": dict(seed="hcooh.xyz", expt=(6.3, 10.0), args=[
        "--inputs", str(HERE / "hcooh_profile.toml"), "--standard-state", "1atm",
        "--cutoff-temperature", "1500", "--cutoff-time", "1e-3", "--temperature", "1500", "--time", "1e-3"]),
    # experiment: endo:exo = 77:23
    "f0007": dict(seed="f0007_start_complex.xyz", expt=(77 / 23, 77 / 23), args=[
        "--inputs", str(HERE / "f0007_mlpgi.toml"), "--standard-state", "1M",
        "--cutoff-temperature", "353", "--cutoff-time", "23400", "--temperature", "353", "--time", "23400"]
        + (["--steer-energies", str(HERE / "f0007_orca_steer.toml")] if os.access(ORCA, os.X_OK) else [])),
}


def _canon(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else smi


def _species_amounts(s):
    """(final concentration, flux) per found species, mirror images added to their parent."""
    kin = s["rounds"][-1].get("kinetics") if s.get("rounds") else None
    if not kin or not kin.get("final_concentration"):
        return None
    mir = {v["index"]: v["mirror_of"] for v in (kin.get("mirror_images") or s.get("mirror_images") or {})
           .get("virtual_species", [])}
    c, f = {}, {}
    for k, (ck, fk) in enumerate(zip(kin["final_concentration"], kin["flux"])):
        p = mir.get(k, k)
        c[p] = c.get(p, 0.0) + ck
        f[p] = f.get(p, 0.0) + fk
    return c, f


def _reachable(s):
    adj = {}
    for st in s.get("steps", []):
        adj.setdefault(st["a"], set()).add(st["b"])
        adj.setdefault(st["b"], set()).add(st["a"])
    seen, todo = {0}, [0]
    while todo:
        for n in adj.get(todo.pop(), ()):
            if n not in seen:
                seen.add(n)
                todo.append(n)
    return seen


def stereo(fp):
    """endo or exo for a 5-norbornene-2-carboxylate: the ester carbon nearer the C=C than the CH2 bridge = endo
    (as build_tables.stereo of the F0007 finding)."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    m = Chem.MolFromXYZFile(str(fp))
    rdDetermineBonds.DetermineBonds(m, charge=0)
    x = m.GetConformer().GetPositions()
    cc = [a for b in m.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE and b.IsInRing()
          for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())]
    carbons = lambda a: {n.GetIdx() for n in m.GetAtomWithIdx(a).GetNeighbors() if n.GetSymbol() == "C"}
    bridge = [a.GetIdx() for a in m.GetAtoms() if a.IsInRing() and a.GetSymbol() == "C"
              and a.GetTotalNumHs(includeNeighbors=True) == 2 and all(carbons(n) & set(cc) for n in carbons(a.GetIdx()))]
    ester = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "C" and not a.IsInRing() and a.GetDegree() == 3]
    if len(cc) != 2 or len(bridge) != 1 or len(ester) != 1:
        return ""
    e = x[ester[0]]
    return "endo" if np.linalg.norm(e - x[cc].mean(0)) < np.linalg.norm(e - x[bridge[0]]) else "exo"


def _channels(bench, s, run):
    """{channel name: {species index: weight}} for the selectivity numerator and denominator."""
    frags = [[_canon(f) for f in sp["smiles"].split(".")] for sp in s["species"]]
    if bench == "hcooh":
        co, co2 = _canon("[C-]#[O+]"), _canon("O=C=O")
        return {"dehydration": {k: fr.count(co) for k, fr in enumerate(frags) if co in fr},
                "decarboxylation": {k: fr.count(co2) for k, fr in enumerate(frags) if co2 in fr}}
    adduct = _canon("COC(=O)C1CC2C=CC1C2")
    out = {"endo": {}, "exo": {}}
    for k, fr in enumerate(frags):
        if adduct in fr:
            st = stereo(run / "species" / os.path.basename(s["species"][k]["file"]))
            if st:
                out[st][k] = 1
    return out


def score(bench, run, wall=None, cpu=None, src=None):
    run = Path(run)
    s = json.loads((run / "summary.json").read_text())
    lo, hi = BENCHES[bench]["expt"]
    ch = _channels(bench, s, run)
    (num_name, num), (den_name, den) = ch.items()
    reach = _reachable(s)
    amounts = _species_amounts(s)
    temp = s.get("steering", {}).get("temperature_K")
    out = dict(bench=bench, src=src, run=str(run), wall_s=wall, cpu_s=cpu, species=len(s["species"]),
               steps=len(s.get("steps", [])), stopped=(s.get("rounds") or [{}])[-1].get("stopped"),
               kinetics_T=temp, kinetics_time_s=s.get("steering", {}).get("time_s"), expt=[lo, hi],
               found={num_name: bool(num), den_name: bool(den)},
               connected={num_name: any(k in reach for k in num), den_name: any(k in reach for k in den)},
               species_of={num_name: sorted(num), den_name: sorted(den)})
    sel = {}
    if amounts:
        for i, key in enumerate(("final_concentration", "flux")):
            a = sum(w * amounts[i].get(k, 0.0) for k, w in num.items())
            b = sum(w * amounts[i].get(k, 0.0) for k, w in den.items())
            out[f"{num_name}_{key}"], out[f"{den_name}_{key}"] = a, b
            sel[key] = a / b if b > 0 else (math.inf if a > 0 else None)
        out["conversion"] = 1.0 - amounts[0].get(0, 0.0)
    out["sel"], out["sel_flux"] = sel.get("final_concentration"), sel.get("flux")
    r = out["sel"]
    out["in_range"] = r is not None and lo * 0.999 <= r <= hi * 1.001
    if r is not None and temp and 0 < r < math.inf:
        out["err_kcal"] = 0.0 if out["in_range"] else R_KCAL * temp * math.log(r / (lo if r < lo else hi))
    else:
        out["err_kcal"] = None
    return out


def line(o):
    f = lambda x, p=".3g": "-" if x is None else format(x, p)
    yn = lambda d: ",".join(f"{k}:{'Y' if v else 'N'}" for k, v in d.items())
    ex = f"{o['expt'][0]:.3g}-{o['expt'][1]:.3g}" if o["expt"][0] != o["expt"][1] else f"{o['expt'][0]:.3g}"
    return (f"bench={o['bench']} src={o['src'] or '-'} wall={f(o['wall_s'], '.0f')}s cpu={f(o['cpu_s'], '.0f')}s "
            f"species={o['species']} steps={o['steps']} sel={f(o['sel'])} sel_flux={f(o['sel_flux'])} expt={ex} "
            f"err={f(o['err_kcal'], '+.2f')}kcal conv={f(o.get('conversion'))} found={yn(o['found'])} "
            f"connected={yn(o['connected'])} stopped='{o['stopped']}'")


def main(argv):
    cmd, bench, out = argv[:3]
    rest = argv[3:]
    extra = rest[rest.index("--") + 1:] if "--" in rest else []
    opts = rest[:rest.index("--")] if "--" in rest else rest
    opt = dict(zip(opts[::2], opts[1::2]))
    src = Path(opt.get("--src", HERE.parent)).resolve()
    out = Path(out).resolve()
    if cmd == "score":
        o = json.loads((out / "score.json").read_text()) if (out / "score.json").exists() else {}
        o = score(bench, out / "run" if (out / "run").is_dir() else out, o.get("wall_s"), o.get("cpu_s"), o.get("src"))
    else:
        b = BENCHES[bench]
        rev = subprocess.run(["git", "-C", str(src), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                             text=True).stdout.strip() + "@" + subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   PYTHONPATH=str(src), MEPD_NETWORK_AWARE_LOG=str(out / "network_aware.jsonl"))
        args = [PY, "-m", "mepd.cli", "discovery", "expand", str(HERE / b["seed"]), *COMMON, *b["args"],
                "--time-budget", opt.get("--budget", "1800"), "--workers", opt.get("--workers", "4"),
                "-o", str(out / "run"), *extra]
        (out / "argv.json").write_text(json.dumps(dict(src=str(src), rev=rev, argv=args), indent=1))
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.time()
        with open(out / "stdout.log", "w") as log:
            rc = subprocess.run(args, cwd=str(out), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        if not (out / "run" / "summary.json").exists():
            print(f"bench={bench} src={rev} rc={rc} wall={wall:.0f}s cpu={cpu:.0f}s: no summary.json "
                  f"(see {out / 'stdout.log'})")
            return 1
        o = score(bench, out / "run", wall, cpu, rev)
        o["returncode"] = rc
    (out / "score.json").write_text(json.dumps(o, indent=1))
    print(line(o))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
