"""Exploration error: an explorer run against the ceiling, channel by channel.

    python explorer_vs_ceiling.py CEILING.json CHANNELS.json RUN_DIR [RUN_DIR ...] [--json OUT]

CEILING.json: ceiling.py output (its "steered+mirrors" variant: each channel's TSs and free barriers).
CHANNELS.json: {"T": K, "channels": {label: [sorted no-stereo SMILES of the product molecules]}, "ratio": [num, den|[...]],
                "ts_dir": optional dir of the ceiling's TS xyz files is not needed: geometries come from the ceiling spec}
RUN_DIR: an explorer output folder with summary.json (a scoreboard task folder with run/summary.json also works).

Per channel and run:
- proposed: a species whose molecules are exactly the channel's products (the scoreboard's "found");
- connected: a step with a kinetics barrier joins the seed (species 0) to such a species directly, or a route through
  open steps reaches it (then `route` = yes, direct = no);
- the direct steps' free barriers in the run's final kinetics (each with its mirror twin, if the kinetics added one),
  their effective barrier -RT ln sum exp(-dG/RT), against the ceiling's; and whether the run's lowest direct TS is the
  ceiling's lowest TS (superimposable, or its mirror image) or another, higher one;
- the run's selectivity (final concentrations, mirror images folded in) against the ceiling's.
Error contributions in kcal/mol (RT ln at the run's T): exploration = RT ln(sel_run / sel_ceiling).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

R_KCAL = 1.987204259e-3


def canon(smi: str) -> list:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for part in (smi or "").split("."):
        m = Chem.MolFromSmiles(part)
        out.append(Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else part)
    return sorted(out)


def norbornene_class(fp) -> str:
    """endo/exo of a 5-norbornene-2-carboxylate geometry: the ester carbon nearer the ring C=C midpoint than the
    CH2 bridge = endo (the rule of benchmarks/score.py stereo_class and the F0007 Finding); '' otherwise."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    try:
        m = Chem.MolFromXYZFile(str(fp))
        rdDetermineBonds.DetermineBonds(m, charge=0)
    except Exception:
        return ""
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


def bicyclic_class(fp) -> str:
    """endo/exo of a [4+2] bicyclic adduct (norbornene or 7-oxanorbornene) with the dienophile's carbonyl carbons as
    the substituent: their centroid nearer the ring C=C midpoint than the one-atom bridge (CH2 or O) = endo."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    try:
        m = Chem.MolFromXYZFile(str(fp))
        rdDetermineBonds.DetermineBonds(m, charge=0)
    except Exception:
        return ""
    x = m.GetConformer().GetPositions()
    cc = [a for b in m.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE and b.IsInRing()
          and b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C"
          for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())]
    if len(cc) != 2:
        return ""
    heads = {n.GetIdx() for a in cc for n in m.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() not in cc and n.IsInRing()}
    bridge = [a.GetIdx() for a in m.GetAtoms() if a.IsInRing() and a.GetIdx() not in heads
              and {n.GetIdx() for n in a.GetNeighbors()} >= heads and len(heads) == 2]
    carbonyl = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "C" and any(
        n.GetSymbol() == "O" and m.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for n in a.GetNeighbors())]
    if len(bridge) != 1 or not carbonyl:
        return ""
    e = x[carbonyl].mean(0)
    return "endo" if np.linalg.norm(e - x[cc].mean(0)) < np.linalg.norm(e - x[bridge[0]]) else "exo"


def species_file(sp: dict, run_dir: Path):
    fp = Path(sp.get("file") or "")
    if fp.exists():
        return fp
    for c in (run_dir / "species" / fp.name, run_dir / "run" / "species" / fp.name):
        if c.exists():
            return c
    return None


def load_run(d: Path) -> dict:
    for c in (d / "summary.json", d / "run" / "summary.json"):
        if c.exists():
            return json.loads(c.read_text())
    raise FileNotFoundError(d)


def reach(n: int, steps: list) -> set:
    adj = {}
    for st in steps:
        if st.get("kinetics_barrier_kcal") is None:
            continue
        adj.setdefault(st["a"], set()).add(st["b"])
        adj.setdefault(st["b"], set()).add(st["a"])
    seen, todo = {0}, [0]
    while todo:
        k = todo.pop()
        for j in adj.get(k, ()):
            if j not in seen:
                seen.add(j)
                todo.append(j)
    return seen


def analyse(ceiling: dict, chdef: dict, run_dir: Path) -> dict:
    s = load_run(run_dir)
    T = float(chdef["T"])
    rt = R_KCAL * T
    kin = s["rounds"][-1].get("kinetics") or {}
    mir = kin.get("mirror_images") or s.get("mirror_images") or {}
    twins = {}
    for t in mir.get("added_twins", []):
        twins[t["step"]] = twins.get(t["step"], 0) + 1
    species_ch = {}
    geo_class = chdef.get("geometry_class", {})        # {channel: "endo"|"exo"} (norbornene esters)
    for sp in s["species"]:
        c = canon(sp.get("smiles") or "")
        for lab, prods in chdef["channels"].items():
            if c == sorted(canon(".".join(prods))):
                if lab in geo_class:
                    fp = species_file(sp, run_dir)
                    cls = norbornene_class if chdef.get("classifier", "norbornene") == "norbornene" else bicyclic_class
                    if fp is None or cls(fp) != geo_class[lab]:
                        continue
                species_ch[sp["index"]] = lab
    connected = reach(len(s["species"]), s["steps"])
    cv = ceiling["variants"]["steered+mirrors"]
    out = {"run": str(run_dir), "channels": {}}
    for lab in chdef["channels"]:
        idx = [k for k, c in species_ch.items() if c == lab]
        direct = []
        for m, st in enumerate(s["steps"]):
            if 0 in (st["a"], st["b"]) and ({st["a"], st["b"]} - {0}) & set(idx):
                kb = st.get("kinetics_barrier_kcal")
                dg = None if kb is None else (kb[0] if st["a"] == 0 else kb[1])
                direct.append({"label": st["label"], "dG": dg, "copies": 1 + twins.get(m, 0)})
        open_d = [d for d in direct if d["dG"] is not None]
        eff = (-rt * math.log(sum(d["copies"] * math.exp(-d["dG"] / rt) for d in open_d))) if open_d else None
        c_steps = [x for x in cv["steps"] if x["channel"] == lab]
        c_copies = {}
        for t in cv.get("mirrors", {}).get("added_twins", []):
            c_copies[t["step"]] = c_copies.get(t["step"], 0) + 1
        c_eff = -rt * math.log(sum((1 + c_copies.get(x["index"], 0)) * math.exp(-x["dG_kcal"] / rt) for x in c_steps)) \
            if c_steps else None
        out["channels"][lab] = {
            "proposed": bool(idx), "connected": bool(set(idx) & connected), "direct_steps": direct,
            "run_eff_barrier": eff, "ceiling_eff_barrier": c_eff,
            "excess_kcal": (eff - c_eff) if (eff is not None and c_eff is not None) else None,
            "run_lowest_dG": min((d["dG"] for d in open_d), default=None),
            "ceiling_lowest_dG": min((x["dG_kcal"] for x in c_steps), default=None)}
    # the run's selectivity, mirror images folded
    num, den = chdef["ratio"]
    dens = den if isinstance(den, list) else [den]
    vmap = {v["index"]: v["mirror_of"] for v in mir.get("virtual_species", [])}
    amt = {}
    for k, c in enumerate(kin.get("final_concentration") or []):
        p = vmap.get(k, k)
        lab = species_ch.get(p)
        if lab:
            amt[lab] = amt.get(lab, 0.0) + c
    dsum = sum(amt.get(d, 0.0) for d in dens)
    sel = amt.get(num, 0.0) / dsum if dsum > 0 else (float("inf") if amt.get(num) else None)
    c_sel = ceiling["variants"]["steered+mirrors"]["selectivity_final"]
    out["selectivity"] = sel
    out["ceiling_selectivity"] = c_sel
    out["exploration_error_kcal"] = (rt * math.log(sel / c_sel)) if sel not in (None, 0.0, float("inf")) else None
    return out


def main(argv):
    js = None
    if "--json" in argv:
        i = argv.index("--json")
        js = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    ceiling = json.loads(Path(argv[0]).read_text())
    chdef = json.loads(Path(argv[1]).read_text())
    res = []
    for d in argv[2:]:
        try:
            r = analyse(ceiling, chdef, Path(d))
        except Exception as exc:
            r = {"run": d, "error": f"{type(exc).__name__}: {exc}"}
        res.append(r)
        if "error" in r:
            print(d, r["error"])
            continue
        ch = "  ".join(f"{k}: P{'Y' if v['proposed'] else 'N'} C{'Y' if v['connected'] else 'N'} "
                       f"dG‡ {v['run_eff_barrier'] if v['run_eff_barrier'] is None else round(v['run_eff_barrier'], 2)}"
                       f" (ceil {v['ceiling_eff_barrier'] if v['ceiling_eff_barrier'] is None else round(v['ceiling_eff_barrier'], 2)})"
                       for k, v in r["channels"].items())
        sel = r["selectivity"]
        print(f"{'/'.join(Path(d).parts[-3:])[-40:]:<40} sel {sel if sel is None else round(sel, 3)} (ceil {r['ceiling_selectivity']:.3g}) "
              f"expl.err {r['exploration_error_kcal'] if r['exploration_error_kcal'] is None else round(r['exploration_error_kcal'], 2)}  | {ch}")
    if js:
        Path(js).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
