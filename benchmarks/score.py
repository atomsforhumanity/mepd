"""Scoring `mepd discovery expand` runs against experiment (explorer team, phase 0: the scoreboard).

    P=/home/user/astera/mepd/.venv/bin/python
    $P benchmarks/score.py score SYSTEM RUNDIR [--start EPOCH]   # score one run folder (prints, writes nothing)
    $P benchmarks/score.py systems                                # the suite as loaded

Cluster runs (build, submit, pull, score, aggregate, gate): benchmarks/cluster/bench.py and gate.py.

The suite (`load_systems`): benchmarks/systems.yaml (the curator's) when it exists, else PROVISIONAL below.
Each system: its seed, charge and multiplicity, profile, kinetics conditions (temperature, time, standard
state; the access cutoff uses the same), its experimental channels (product sets as SMILES; F0007's
two diastereomers also by a geometric classifier) and its experimental selectivities (ratios of channel
amounts with a range).

Definitions (BRIEF.md, "Shared definitions"), as computed here:
- species amounts: the run's last kinetics (`summary.json` rounds[-1].kinetics), `final_concentration`
  at the kinetics time (the yield an experiment measures); the virtual mirror images the kinetics add
  are counted with the species they mirror.
- a species belongs to a channel when its molecules are exactly the channel's product set (RDKit
  canonical SMILES without stereo of each molecule of the species' SMILES, as a multiset), and, for a
  classified channel, its geometry gives the class (endo/exo).
- recall: found = a species of the channel exists; connected = one is reachable from the seed (species 0)
  through steps the kinetics use, i.e. steps with a kinetics barrier (not closed out of reach).
  `linked` = reachable through any step, closed ones too.
- selectivity of a target = sum of the amounts of the numerator channels' species / the denominator's.
  error = RT ln(selectivity / nearest bound of the experimental range) at the system's temperature,
  kcal/mol, 0 inside the range; +-inf when one side has no amount; worst = largest |error|.
- settle time: the earliest time (s after expand started) from which, to the end of the run, every
  channel is found and connected and every target's selectivity is within 0.5 kcal/mol (RT ln) of the
  run's final one. Read from the live kinetics the run wrote (MEPD_DRIVE_CHAIN_DIR/kinetics.json, every
  version kept by the task wrapper) and its partial summaries (summary_partial.json): at a snapshot,
  amounts = the live kinetics' `final` per species (a chiral species whose mirror image is not yet a
  species counts twice: its virtual twin carries as much); a channel is connected when the latest partial
  summary's steps reach it through steps within the access cutoff (energy_refine.kinetic_reach, as the
  kinetics close steps) or when the live kinetics give it a population > 1e-30 (no closed step carries
  that much). No snapshot after the last change: the end of the run. Recall incomplete at the end: None.
- wall, CPU (user+sys of the expand process tree), stop reason: the task wrapper's result.json.

The invariant audit (flags, never fixes): duplicate species, duplicate TS steps, mirror-image TS pairs
whose free energies differ by > 0.2 kcal/mol (spread reported), channel products found but not
connected, barriers reported for closed steps, time budget overrun > 2 %, negative barriers, IRC ends
that are not their step's species. Structure comparisons use mepd from this branch (develop 53c68d4):
`nanoreactor_generator.stereo_key`, `mirror.superimposable`, `compare.same_molecule`.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))     # the audit's structure comparisons: this branch's mepd
R_KCAL = 1.987204e-3
HARTREE_KCAL = 627.5094740631
SETTLE_KCAL = 0.5
QUICK_S = 1800.0

# Provisional suite until the curator's benchmarks/systems.yaml exists. Targets marked UNCONFIRMED are
# placeholders from REFERENCES.md (playgroud/exp_benchmarks), not pinned from primary sources.
PROVISIONAL = {
    "hcooh": {
        "seed": "hcooh.xyz", "charge": 0, "multiplicity": 1, "profile": "hcooh_profile.toml",
        "conditions": {"temperature_K": 1500.0, "time_s": 1e-3, "standard_state": "1atm"},
        "channels": {"dehydration": {"products": ["[C-]#[O+]", "O"]},
                     "decarboxylation": {"products": ["O=C=O", "[H][H]"]}},
        "targets": [{"num": ["dehydration"], "den": ["decarboxylation"], "range": [1.0, 20.0],
                     "status": "UNCONFIRMED: REFERENCES.md lists ~10 -> ~1 near 2000 K (Hsu 1982, secondary), "
                               "~20 (Elwardany 2015, snippet), ~100 (Saito 1984, unconfirmed); the curator pins it"}],
        "held_out": False,
    },
    "methyl_formate": {
        "seed": "cluster/suite/methyl_formate.xyz", "charge": 0, "multiplicity": 1, "profile": "hcooh_profile.toml",
        "conditions": {"temperature_K": 1400.0, "time_s": 2e-3, "standard_state": "1atm"},
        "channels": {"methanol_co": {"products": ["CO", "[C-]#[O+]"]},
                     "formaldehyde2": {"products": ["C=O", "C=O"]},
                     "methane_co2": {"products": ["C", "O=C=O"]}},
        "targets": [{"num": ["methanol_co"], "den": ["formaldehyde2"], "range": [19.0, math.inf],
                     "status": "UNCONFIRMED: REFERENCES.md 'CH3OH+CO dominant (~95%), others minor (<5% each)'; "
                               "Ren 2013 (doi:10.1016/j.proci.2012.05.071) k1/k2 at 1400 K = 25 from the quoted "
                               "Arrhenius fits (PDF not checked); the curator pins it"},
                    {"num": ["methanol_co"], "den": ["methane_co2"], "range": [19.0, math.inf],
                     "status": "UNCONFIRMED: as above (k3 exponent unconfirmed)"}],
        "held_out": False,
    },
    "f0007": {
        "seed": "f0007_start_complex.xyz", "charge": 0, "multiplicity": 1, "profile": "f0007_mlpgi.toml",
        "conditions": {"temperature_K": 353.0, "time_s": 23400.0, "standard_state": "1M"},
        "channels": {"endo": {"products": ["COC(=O)C1CC2C=CC1C2"], "classify": "endo"},
                     "exo": {"products": ["COC(=O)C1CC2C=CC1C2"], "classify": "exo"}},
        "targets": [{"num": ["endo"], "den": ["exo"], "range": [77 / 23, 77 / 23],
                     "status": "Lording et al., Chem. Sci. 11, 11915 (2020), doi:10.1039/d0sc04553e: endo:exo "
                               "77:23 in benzene at 80 C (NMR), as verified in Finding F0007"}],
        "held_out": False,
    },
}
SUITE_SOURCE = "provisional (benchmarks/score.py PROVISIONAL)"


def load_systems() -> dict:
    """The suite: the curator's benchmarks/systems.yaml when it exists, else PROVISIONAL."""
    global SUITE_SOURCE
    fp = HERE / "systems.yaml"
    if not fp.is_file():
        return PROVISIONAL
    import yaml

    raw = yaml.safe_load(fp.read_text())
    SUITE_SOURCE = f"benchmarks/systems.yaml"
    systems = raw.get("systems", raw)
    out = {}
    for name, sd in systems.items():
        sd = dict(sd)
        for t in sd.get("targets", []):
            t["num"] = [t["num"]] if isinstance(t["num"], str) else list(t["num"])
            t["den"] = [t["den"]] if isinstance(t["den"], str) else list(t["den"])
            t["range"] = [math.inf if str(v).lower() in ("inf", "infinity", ".inf") else float(v) for v in t["range"]]
        out[name] = sd
    return out


# ----------------------------------------------------------------------------------------------- reading
def _canon(smi: str) -> str:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else smi


_FRAG_CACHE: dict = {}


def fragments(smiles: str) -> tuple:
    """The species' molecules as a sorted tuple of canonical no-stereo SMILES."""
    if smiles not in _FRAG_CACHE:
        _FRAG_CACHE[smiles] = tuple(sorted(_canon(f) for f in smiles.split("."))) if smiles else ()
    return _FRAG_CACHE[smiles]


class Paths:
    """Maps the run's own (pod) paths in summary.json onto the local copy of its run folder."""

    def __init__(self, summary: dict, run_dir: Path):
        self.run = Path(run_dir)
        fp = (summary.get("output_files") or {}).get("species") or ""
        self.prefix = str(Path(fp).parent) if fp else None

    def __call__(self, fp):
        if not fp:
            return None
        if self.prefix and str(fp).startswith(self.prefix + "/"):
            return self.run / str(fp)[len(self.prefix) + 1:]
        p = Path(fp)
        return p if p.exists() else self.run / Path(*p.parts[-2:])


def stereo_class(fp) -> str:
    """endo or exo for a 5-norbornene-2-carboxylate: the ester carbon nearer the C=C than the CH2 bridge =
    endo (as build_tables.stereo of the F0007 finding); '' if the geometry isn't one."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    m = Chem.MolFromXYZFile(str(fp))
    if m is None:
        return ""
    try:
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


def channel_species(sysdef: dict, smiles: list, species_file) -> dict:
    """{channel: [species index]} for species SMILES `smiles` (index = position); `species_file(k)`:
    species k's geometry file, for classified channels."""
    want = {name: tuple(sorted(_canon(p) for p in ch["products"])) for name, ch in sysdef["channels"].items()}
    out = {name: [] for name in want}
    cls_cache = {}
    for k, smi in enumerate(smiles):
        fr = fragments(smi or "")
        for name, prod in want.items():
            if fr != prod:
                continue
            cls = sysdef["channels"][name].get("classify")
            if cls:
                if k not in cls_cache:
                    fp = species_file(k)
                    cls_cache[k] = stereo_class(fp) if fp and Path(fp).is_file() else ""
                if cls_cache[k] != cls:
                    continue
            out[name].append(k)
    return out


def folded_amounts(summary: dict) -> dict | None:
    """{species: (final concentration, flux)} of the last kinetics, virtual mirror images added to the
    species they mirror; None without kinetics."""
    kin = (summary.get("rounds") or [{}])[-1].get("kinetics")
    if not kin or not kin.get("final_concentration"):
        return None
    mir = {v["index"]: v["mirror_of"] for v in (kin.get("mirror_images") or summary.get("mirror_images") or {})
           .get("virtual_species", [])}
    out = {}
    for k, (c, f) in enumerate(zip(kin["final_concentration"], kin["flux"])):
        p = mir.get(k, k)
        c0, f0 = out.get(p, (0.0, 0.0))
        out[p] = (c0 + c, f0 + f)
    return out


def reachable(n: int, steps: list, use=lambda st: True) -> set:
    adj = {}
    for st in steps:
        if use(st):
            adj.setdefault(st["a"], set()).add(st["b"])
            adj.setdefault(st["b"], set()).add(st["a"])
    seen, todo = {0}, [0]
    while todo:
        for k in adj.get(todo.pop(), ()):
            if k not in seen:
                seen.add(k)
                todo.append(k)
    return seen


def open_steps(summary: dict, cutoff_kcal=None) -> list:
    """Indices of the steps within reach (energy_refine.kinetic_reach on the summary's energies), as the
    explorer's kinetics decide which steps they keep open."""
    from mepd.discovery.energy_refine import kinetic_reach
    from mepd.discovery.kinetics import Step

    sp = summary.get("species") or []
    steps = summary.get("steps") or []
    if not sp or not steps:
        return []
    e0 = float(summary.get("seed_energy") if summary.get("seed_energy") is not None else sp[0]["energy"])
    cutoff = cutoff_kcal if cutoff_kcal is not None else (summary.get("steering") or {}).get("cutoff_kcal")
    n = len(sp)
    mir = (summary.get("mirror_images") or {}).get("species_mirror")
    mirror_of = [j if (j is not None and 0 <= j < n) else -1 for j in mir[:n]] if mir else None
    ks = [Step(st["a"], st["b"], float(st["ts_energy"]), st.get("label", ""), {}) for st in steps]
    _, opened = kinetic_reach(ks, [float(s["energy"]) for s in sp], e0, cutoff, mirror_of)
    return opened


# ----------------------------------------------------------------------------------------------- metrics
def target_values(sysdef: dict, chans: dict, amount) -> list:
    """Per target: (selectivity or None, numerator, denominator); `amount(k)`: species k's amount."""
    out = []
    for t in sysdef["targets"]:
        a = sum(amount(k) for ch in t["num"] for k in chans.get(ch, []))
        b = sum(amount(k) for ch in t["den"] for k in chans.get(ch, []))
        sel = a / b if b > 0 else (math.inf if a > 0 else None)
        out.append((sel, a, b))
    return out


def error_kcal(sel, lo: float, hi: float, temp: float):
    """RT ln(sel / nearest bound), 0 inside [lo, hi]; +-inf for inf/0; None for no amount at all."""
    if sel is None:
        return None
    if lo * (1 - 1e-9) <= sel <= hi * (1 + 1e-9):
        return 0.0
    if sel == 0:
        return -math.inf
    if math.isinf(sel):
        return math.inf
    return R_KCAL * temp * math.log(sel / (lo if sel < lo else hi))


def _target_name(t: dict) -> str:
    return "+".join(t["num"]) + ":" + "+".join(t["den"])


def _within(sel, ref, temp) -> bool:
    if sel is None or ref is None:
        return False
    if math.isinf(ref) or ref == 0 or math.isinf(sel) or sel == 0:
        return sel == ref
    return abs(R_KCAL * temp * math.log(sel / ref)) <= SETTLE_KCAL


def settle(sysdef: dict, run_dir: Path, snap_dir: Path, summary: dict, final: dict, t0: float, t_end: float,
           species_file) -> dict:
    """The settle time (see the module docstring) and the selectivity's time course."""
    from mepd.discovery.energy_refine import kinetic_reach
    from mepd.discovery.kinetics import Step

    temp = float(sysdef["conditions"]["temperature_K"])
    cutoff = (summary.get("steering") or {}).get("cutoff_kcal")
    n_final = len(summary.get("species") or [])
    mir = (summary.get("mirror_images") or {}).get("species_mirror") or list(range(n_final))
    final_sel = [x["selectivity"] for x in final["targets"]]
    complete = all(final["recall"]["found"].values()) and all(final["recall"]["connected"].values())
    kin = sorted(snap_dir.glob("kinetics_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    parts = sorted(snap_dir.glob("summary_partial_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    part_t = [int(p.stem.split("_")[-1]) / 1e9 for p in parts]
    reach_cache: dict = {}

    def reach_at(t):
        i = int(np.searchsorted(part_t, t, side="right")) - 1
        if i < 0:
            return set()
        if i not in reach_cache:
            try:
                ps = json.loads(parts[i].read_text())
                sp, st = ps.get("species") or [], ps.get("steps") or []
                e0 = float(ps.get("seed_energy") if ps.get("seed_energy") is not None else sp[0]["energy"])
                ks = [Step(s["a"], s["b"], float(s["ts_energy"]), s.get("label", ""), {}) for s in st]
                _, opened = kinetic_reach(ks, [float(s["energy"]) for s in sp], e0, cutoff, None)
                reach_cache[i] = reachable(len(sp), [st[m] for m in opened])
            except (ValueError, KeyError, IndexError, OSError):
                reach_cache[i] = set()
        return reach_cache[i]

    series = []
    for fp in kin:
        t = int(fp.stem.split("_")[-1]) / 1e9
        try:
            data = json.loads(fp.read_text())["data"]
        except (ValueError, KeyError, OSError):
            continue
        sp = data.get("species") or []
        n_t = len(sp)
        pop = {s["index"]: float(s.get("final", 0.0)) for s in sp}
        live = {s["index"] for s in sp if max(float(s.get("max_concentration", 0.0)), float(s.get("flux", 0.0))) > 1e-30}

        def amount(k):
            j = mir[k] if k < len(mir) else k
            twin_virtual = j != k and not (0 <= j < min(n_t, n_final))
            return pop.get(k, 0.0) * (2.0 if twin_virtual else 1.0)

        chans = channel_species(sysdef, [s.get("smiles") or "" for s in sp], species_file)
        reach = reach_at(t)
        found = {c: bool(ks) for c, ks in chans.items()}
        conn = {c: any(k in reach or k in live for k in ks) for c, ks in chans.items()}
        sels = [v[0] for v in target_values(sysdef, chans, amount)]
        ok = all(found.values()) and all(conn.values()) and all(
            _within(s, f, temp) for s, f in zip(sels, final_sel))
        series.append((t - t0, sels, found, conn, ok))
    if not complete:
        t_settle = None
    else:
        t_settle = t_end - t0
        for t, *_, ok in reversed(series):
            if not ok:
                break
            t_settle = t
    # the course, compactly: only snapshots where something changed
    course, last = [], None
    for t, sels, found, conn, ok in series:
        key = (tuple(None if s is None else round(s, 4) for s in sels), tuple(found.values()), tuple(conn.values()), ok)
        if key != last:
            course.append({"t": round(t, 1), "sel": sels, "found": found, "connected": conn, "ok": ok})
            last = key
    return {"settle_s": t_settle, "quick": t_settle is not None and t_settle <= QUICK_S,
            "n_kinetics_snapshots": len(series), "n_partial_snapshots": len(parts), "course": course}


# ----------------------------------------------------------------------------------------------- audit
def _structure(fp):
    from qcdata import Structure

    return Structure.open(str(fp))


def _frames(fp):
    from qcdata import Structure

    return Structure.open_multi(str(fp))


def audit(sysdef: dict, summary: dict, paths: Paths, rec: dict, result: dict) -> dict:
    """The invariant audit of one run: {check: [flags]} plus a few counts (see module docstring)."""
    from mepd.compare import same_molecule
    from mepd.discovery.mirror import _angstrom, mirrored, superimposable
    from mepd.discovery.nanoreactor_generator import stereo_key
    from mepd.nodes.node import StructureNode

    sp = summary.get("species") or []
    steps = summary.get("steps") or []
    n = len(sp)
    flags: dict = {k: [] for k in ("duplicate_species", "duplicate_ts", "mirror_ts", "found_not_connected",
                                   "closed_with_barrier", "budget_overrun", "negative_barrier", "irc_mismatch")}
    info: dict = {}
    structs = {}
    for k, s in enumerate(sp):
        fp = paths(s.get("file"))
        try:
            structs[k] = _structure(fp) if fp and Path(fp).is_file() else None
        except Exception:
            structs[k] = None
    # 1. duplicate species: same stereo key, energies within 0.1 kcal/mol
    keys = {k: (stereo_key(st) if st is not None else None) for k, st in structs.items()}
    same_key = 0
    for i in range(n):
        for j in range(i + 1, n):
            if keys[i] and keys[i] == keys[j]:
                same_key += 1
                de = abs(sp[i]["energy"] - sp[j]["energy"]) * HARTREE_KCAL
                if de < 0.1:
                    flags["duplicate_species"].append({"species": [i, j], "key": keys[i], "dE_kcal": round(de, 4)})
    info["same_stereo_key_pairs"] = same_key
    # TS structures
    ts = {}
    for m, st in enumerate(steps):
        fp = paths((st.get("files") or {}).get("ts"))
        try:
            ts[m] = _structure(fp) if fp and Path(fp).is_file() else None
        except Exception:
            ts[m] = None
    syms = lambda s: [str(x) for x in s.symbols]
    # 2. duplicate TS steps: same pair, |dE| < 0.1 kcal/mol, superimposable (not merely mirror images)
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            a, b = steps[i], steps[j]
            if {a["a"], a["b"]} != {b["a"], b["b"]} or ts[i] is None or ts[j] is None:
                continue
            de = abs(a["ts_energy"] - b["ts_energy"]) * HARTREE_KCAL
            if de < 0.1 and superimposable(syms(ts[i]), _angstrom(ts[i]), _angstrom(ts[j])):
                flags["duplicate_ts"].append({"steps": [a["label"], b["label"]], "pair": [a["a"], a["b"]],
                                              "dE_kcal": round(de, 4)})
    # 3. mirror-image TS pairs: free-energy barrier differences
    mir = (summary.get("mirror_images") or {}).get("species_mirror") or list(range(n))
    m_of = lambda k: mir[k] if k < len(mir) and 0 <= mir[k] < n else -1
    diffs = []

    def bar_from(st, k):
        kb = st.get("kinetics_barrier_kcal")
        if kb is None:
            return None
        return kb[0] if st["a"] == k else kb[1] if st["b"] == k else None

    for i in range(len(steps)):          # (a) twins between the same pair: TS_j = mirror image of TS_i
        for j in range(i + 1, len(steps)):
            a, b = steps[i], steps[j]
            if {a["a"], a["b"]} != {b["a"], b["b"]} or ts[i] is None or ts[j] is None:
                continue
            if superimposable(syms(ts[i]), _angstrom(ts[i]), _angstrom(ts[j])):
                continue
            if not superimposable(syms(ts[i]), _angstrom(ts[i]), _angstrom(mirrored(ts[j]))):
                continue
            ga, gb = bar_from(a, a["a"]), bar_from(b, a["a"])
            de = (a["ts_energy"] - b["ts_energy"]) * HARTREE_KCAL
            diffs.append({"kind": "twins", "steps": [a["label"], b["label"]], "pairs": [[a["a"], a["b"]]] * 2,
                          "dG_kcal": None if ga is None or gb is None else round(ga - gb, 3), "dE_kcal": round(de, 3)})
    best = {}                            # (b) enantiomeric channels: lowest barrier per pair vs its mirror pair
    for m, st in enumerate(steps):
        if st.get("kinetics_barrier_kcal") is None:
            continue
        key = (st["a"], st["b"]) if st["a"] < st["b"] else (st["b"], st["a"])
        g = bar_from(st, key[0])
        if key not in best or g < best[key][0]:
            best[key] = (g, m)
    done = set()
    for (a, b), (g, m) in best.items():
        ma, mb = m_of(a), m_of(b)
        if ma < 0 or mb < 0:
            continue
        mkey = (ma, mb) if ma < mb else (mb, ma)
        if mkey == (a, b) or mkey not in best or frozenset([(a, b), mkey]) in done:
            continue
        done.add(frozenset([(a, b), mkey]))
        _, m2 = best[mkey]
        ref2 = ma                                # the mirror image of species a: the same reference reflected
        g2 = bar_from(steps[m2], ref2)
        diffs.append({"kind": "enantiomeric", "steps": [steps[m]["label"], steps[m2]["label"]],
                      "pairs": [[a, b], list(mkey)], "dG_kcal": None if g2 is None else round(g - g2, 3),
                      "dE_kcal": round((steps[m]["ts_energy"] - steps[m2]["ts_energy"]) * HARTREE_KCAL
                                       - (sp[a]["energy"] - sp[ref2]["energy"]) * HARTREE_KCAL, 3)})
    flags["mirror_ts"] = [d for d in diffs if d["dG_kcal"] is not None and abs(d["dG_kcal"]) > 0.2]
    gs = [abs(d["dG_kcal"]) for d in diffs if d["dG_kcal"] is not None]
    info["mirror_ts_pairs"] = diffs
    info["mirror_ts_spread_kcal"] = {"n": len(gs), "max": max(gs) if gs else None,
                                     "median": float(np.median(gs)) if gs else None}
    # 4. channel products found but not connected
    for c in rec["found"]:
        if rec["found"][c] and not rec["connected"][c]:
            flags["found_not_connected"].append({"channel": c, "species": rec["species_of"][c],
                                                 "linked": rec["linked"][c]})
    reach_any = reachable(n, steps)
    info["species_unreachable"] = sorted(set(range(n)) - reach_any)
    # 5. barriers reported for closed steps (closed = out of reach, recomputed with kinetic_reach)
    try:
        opened = set(open_steps(summary))
    except Exception as exc:
        opened, info["open_steps_error"] = None, f"{type(exc).__name__}: {exc}"
    for m, st in enumerate(steps):
        kb = st.get("kinetics_barrier_kcal")
        if kb is not None and max(kb) > 300:
            flags["closed_with_barrier"].append({"step": st["label"], "kinetics_barrier_kcal": kb,
                                                 "why": "barrier > 300 kcal/mol (TS + 1 Hartree)"})
        elif opened is not None and m not in opened and kb is not None:
            flags["closed_with_barrier"].append({"step": st["label"], "kinetics_barrier_kcal": kb,
                                                 "why": "out of reach (kinetic_reach) but a barrier is reported"})
    if opened is not None:
        info["open_without_barrier"] = [st["label"] for m, st in enumerate(steps)
                                        if m in opened and st.get("kinetics_barrier_kcal") is None]
    # 6. time budget overrun
    budget = result.get("budget_s")
    wall = result.get("wall_s")
    if budget and wall and wall > 1.02 * float(budget):
        flags["budget_overrun"].append({"wall_s": round(wall, 1), "budget_s": budget,
                                        "over_pct": round(100 * (wall / float(budget) - 1), 1)})
    # 7. negative barriers
    for st in steps:
        for what in ("barrier_kcal", "kinetics_barrier_kcal"):
            v = st.get(what)
            if v is not None and min(v) < 0:
                flags["negative_barrier"].append({"step": st["label"], what: v})
    info["warnings"] = summary.get("warnings") or []
    # 8. IRC ends that are not their step's species (bond graphs, stereo off: IRC ends are unminimized)
    nodes = {k: StructureNode(structure=s) for k, s in structs.items() if s is not None}
    checked = 0
    for st in steps:
        fp = paths((st.get("files") or {}).get("irc"))
        if not fp or not Path(fp).is_file() or st["a"] not in nodes or st["b"] not in nodes:
            continue
        try:
            fr = _frames(fp)
            e0, e1 = StructureNode(structure=fr[0]), StructureNode(structure=fr[-1])
            A, B = nodes[st["a"]], nodes[st["b"]]
            fwd = same_molecule(e0, A, stereo=False) and same_molecule(e1, B, stereo=False)
            rev = same_molecule(e0, B, stereo=False) and same_molecule(e1, A, stereo=False)
        except Exception as exc:
            info.setdefault("irc_errors", []).append(f"{st['label']}: {type(exc).__name__}: {exc}")
            continue
        checked += 1
        if not (fwd or rev):
            flags["irc_mismatch"].append({"step": st["label"], "pair": [st["a"], st["b"]],
                                          "smiles": [sp[st["a"]]["smiles"], sp[st["b"]]["smiles"]]})
    info["irc_checked"] = checked
    return {"flags": flags, "n_flags": {k: len(v) for k, v in flags.items()}, "info": info}


# ----------------------------------------------------------------------------------------------- one run
def score_run(sysdef: dict, run_dir: Path, result: dict | None = None, snap_dir: Path | None = None,
              meta: dict | None = None, do_audit: bool = True) -> dict:
    """Metrics (and audit) of one finished run folder (`run_dir` holds summary.json)."""
    run_dir = Path(run_dir)
    result = result or {}
    s = json.loads((run_dir / "summary.json").read_text())
    paths = Paths(s, run_dir)
    temp = float(sysdef["conditions"]["temperature_K"])
    sp = s.get("species") or []
    species_file = lambda k: paths(sp[k].get("file")) if k < len(sp) else None
    chans = channel_species(sysdef, [x.get("smiles") or "" for x in sp], species_file)
    steps = s.get("steps") or []
    reach_open = reachable(len(sp), steps, lambda st: st.get("kinetics_barrier_kcal") is not None)
    reach_any = reachable(len(sp), steps)
    amounts = folded_amounts(s)
    amount = (lambda k: amounts.get(k, (0.0, 0.0))[0]) if amounts else (lambda k: 0.0)
    rec = {"found": {c: bool(ks) for c, ks in chans.items()},
           "connected": {c: any(k in reach_open for k in ks) for c, ks in chans.items()},
           "linked": {c: any(k in reach_any for k in ks) for c, ks in chans.items()},
           "species_of": chans}
    targets = []
    for t, (sel, a, b) in zip(sysdef["targets"], target_values(sysdef, chans, amount)):
        lo, hi = t["range"]
        targets.append({"name": _target_name(t), "selectivity": sel, "num_amount": a, "den_amount": b,
                        "range": [lo, hi], "error_kcal": error_kcal(sel, lo, hi, temp) if amounts else None})
    errs = [x["error_kcal"] for x in targets]
    worst = None if any(e is None for e in errs) or not errs else max(errs, key=abs)
    kin = (s.get("rounds") or [{}])[-1]
    out = {"species": len(sp), "steps": len(steps), "recall": rec,
           "full_recall": all(rec["found"].values()) and all(rec["connected"].values()),
           "targets": targets, "worst_error_kcal": worst,
           "worst_abs_error_kcal": None if worst is None else abs(worst),
           "conversion": (1.0 - amounts.get(0, (0.0, 0.0))[0]) if amounts else None,
           "stopped": kin.get("stopped"), "kinetics_T": (s.get("steering") or {}).get("temperature_K"),
           "cutoff_kcal": (s.get("steering") or {}).get("cutoff_kcal"),
           "wall_s": result.get("wall_s"), "cpu_s": result.get("cpu_s"),
           "cpu_h": None if result.get("cpu_s") is None else result["cpu_s"] / 3600.0,
           "exit": result.get("exit"), "killed": result.get("killed"), "budget_s": result.get("budget_s")}
    if meta:
        out["afh"] = {k: meta.get(k) for k in ("seconds", "cpu_seconds", "cores", "max_task_memory_bytes", "host")}
    if snap_dir is not None and Path(snap_dir).is_dir() and result.get("start"):
        out["settle"] = settle(sysdef, run_dir, Path(snap_dir), s, out, float(result["start"]),
                               float(result.get("end") or result["start"] + (result.get("wall_s") or 0)), species_file)
    else:
        out["settle"] = {"settle_s": None, "quick": False, "note": "no snapshots"}
    if do_audit:
        out["audit"] = audit(sysdef, s, paths, rec, result)
    return out


def score_task(task: dict, sysdef: dict, dst: Path) -> dict:
    """Score one unpacked cluster task folder (bench.py score): dst/run, dst/snapshots, dst/result.json."""
    dst = Path(dst)
    result = json.loads((dst / "result.json").read_text()) if (dst / "result.json").is_file() else {}
    meta = json.loads((dst / "_meta.json").read_text()) if (dst / "_meta.json").is_file() else {}
    base = {"id": task["id"], "system": task["system"], "config": task["config"], "replicate": task["replicate"],
            "seeds": task.get("seeds"), "control_of": task.get("control_of"), "code_commit": result.get("code_commit"),
            "harness_commit": result.get("harness_commit")}
    if not (dst / "run" / "summary.json").is_file():
        why = result.get("killed") or (f"exit {result.get('exit')}" if result else "no result.json")
        return {**base, "ok": False, "stopped": f"no summary.json ({why})", "full_recall": False,
                "worst_error_kcal": None, "wall_s": result.get("wall_s"), "cpu_s": result.get("cpu_s"),
                "cpu_h": None if result.get("cpu_s") is None else result["cpu_s"] / 3600.0,
                "settle": {"settle_s": None, "quick": False}, "budget_s": result.get("budget_s")}
    o = score_run(sysdef, dst / "run", result, dst / "snapshots", meta)
    return {**base, "ok": True, **o}


# ----------------------------------------------------------------------------------------------- aggregate
def _q(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if xs else None


def _stats(vals: list) -> dict:
    """median, min, max, IQR over replicates; None values dropped (counted), +-inf kept (median of
    sorted values; spread over the finite ones)."""
    vs = [v for v in vals if v is not None]
    fin = [v for v in vs if math.isfinite(v)]
    med = float(np.median(sorted(vs))) if vs else None
    if vs and len(vs) % 2 == 0:
        a, b = sorted(vs)[len(vs) // 2 - 1: len(vs) // 2 + 1]
        med = a if a == b else (math.inf if math.isinf(b) and b > 0 else (a + b) / 2 if math.isfinite(a + b) else b)
    return {"n": len(vs), "missing": len(vals) - len(vs), "median": med,
            "min": min(fin) if fin else None, "max": max(fin) if fin else None,
            "iqr": (_q(fin, 75) - _q(fin, 25)) if len(fin) >= 2 else None, "n_inf": len(vs) - len(fin)}


def aggregate(scores: list, suite: dict) -> dict:
    """Per (system, configuration): medians and spreads over the replicates (controls excluded)."""
    groups: dict = {}
    for o in scores:
        if o.get("control_of"):
            continue
        groups.setdefault((o["system"], o["config"]), []).append(o)
    out = {}
    for (system, config), runs in sorted(groups.items()):
        targets = {}
        for i, t in enumerate(suite[system]["targets"]):
            name = _target_name(t)
            errs = [(o["targets"][i]["error_kcal"] if o.get("ok") else None) for o in runs]
            sels = [(o["targets"][i]["selectivity"] if o.get("ok") else None) for o in runs]
            targets[name] = {"error_kcal": _stats(errs), "selectivity": _stats(sels), "range": t["range"]}
        chans = list(suite[system]["channels"])
        rec = {c: {"found": sum(bool(o.get("ok") and o["recall"]["found"][c]) for o in runs) / len(runs),
                   "connected": sum(bool(o.get("ok") and o["recall"]["connected"][c]) for o in runs) / len(runs)}
               for c in chans}
        settles = [o["settle"]["settle_s"] if o.get("settle", {}).get("settle_s") is not None else math.inf
                   for o in runs]
        worst = [o.get("worst_abs_error_kcal") if o.get("ok") and o.get("worst_abs_error_kcal") is not None
                 else math.inf for o in runs]
        flags = {}
        for o in runs:
            for k, v in ((o.get("audit") or {}).get("n_flags") or {}).items():
                flags[k] = flags.get(k, 0) + v
        out[f"{system}/{config}"] = {
            "system": system, "config": config, "held_out": bool(suite[system].get("held_out")), "n": len(runs),
            "replicates": sorted(o["replicate"] for o in runs),
            "full_recall_fraction": sum(bool(o.get("full_recall")) for o in runs) / len(runs),
            "recall": rec, "targets": targets, "worst_abs_error_kcal": _stats(worst),
            "settle_s": _stats(settles),
            "settled_30min_fraction": sum(s <= QUICK_S for s in settles) / len(runs),
            "wall_s": _stats([o.get("wall_s") for o in runs]), "cpu_h": _stats([o.get("cpu_h") for o in runs]),
            "conversion": _stats([o.get("conversion") for o in runs]),
            "species": _stats([o.get("species") for o in runs]), "steps": _stats([o.get("steps") for o in runs]),
            "stopped": sorted(str(o.get("stopped")) for o in runs), "audit_flags": flags}
    return out


def _f(x, fmt=".2f"):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return format(x, fmt)


def table(agg: dict, scores: list) -> str:
    """Markdown: one row per (system, configuration), then one per run."""
    lines = ["| system / config | n | full recall | recall found / connected per channel | selectivity median [min, max] "
             "(expt) | worst abs err kcal median [min, max] | settle s median [min, max] | settled <= 30 min | "
             "wall s median | CPU-h median | stopped | audit flags |", "|" + "---|" * 12]
    for key, g in agg.items():
        rec = ", ".join(f"{c} {v['found']:.0%}/{v['connected']:.0%}" for c, v in g["recall"].items())
        sel = "; ".join(f"{n}: {_f(t['selectivity']['median'], '.3g')} [{_f(t['selectivity']['min'], '.3g')}, "
                        f"{_f(t['selectivity']['max'], '.3g')}] ({_f(t['range'][0], '.3g')}-{_f(t['range'][1], '.3g')})"
                        for n, t in g["targets"].items())
        w, st = g["worst_abs_error_kcal"], g["settle_s"]
        stopped = ", ".join(f"{k} x{g['stopped'].count(k)}" for k in sorted(set(g["stopped"])))
        flags = ", ".join(f"{k} {v}" for k, v in g["audit_flags"].items() if v) or "none"
        lines.append(f"| {key}{' (held out)' if g['held_out'] else ''} | {g['n']} | {g['full_recall_fraction']:.0%} | "
                     f"{rec} | {sel} | {_f(w['median'])} [{_f(w['min'])}, {_f(w['max'])}]{' (+%d inf)' % w['n_inf'] if w['n_inf'] else ''} | "
                     f"{_f(st['median'], '.0f')} [{_f(st['min'], '.0f')}, {_f(st['max'], '.0f')}]{' (+%d never)' % st['n_inf'] if st['n_inf'] else ''} | "
                     f"{g['settled_30min_fraction']:.0%} | {_f(g['wall_s']['median'], '.0f')} | "
                     f"{_f(g['cpu_h']['median'], '.2f')} | {stopped} | {flags} |")
    lines += ["", "| task | seeds | species | steps | found | connected | selectivity | err kcal | conversion | settle s | "
                  "wall s | CPU-h | stopped | flags |", "|" + "---|" * 14]
    for o in sorted(scores, key=lambda o: o["id"]):
        if not o.get("ok"):
            lines.append(f"| {o['id']} | {o.get('seeds')} | - | - | - | - | - | - | - | - | {_f(o.get('wall_s'), '.0f')} | "
                         f"{_f(o.get('cpu_h'))} | {o.get('stopped')} | - |")
            continue
        yn = lambda d: ",".join(f"{k}:{'Y' if v else 'N'}" for k, v in d.items())
        sels = "; ".join(_f(t["selectivity"], ".3g") for t in o["targets"])
        errs = "; ".join(_f(t["error_kcal"], "+.2f") for t in o["targets"])
        nf = (o.get("audit") or {}).get("n_flags") or {}
        lines.append(f"| {o['id']} | {o['seeds']['reactor_seed'] if o.get('seeds') else '-'} | {o['species']} | {o['steps']} | "
                     f"{yn(o['recall']['found'])} | {yn(o['recall']['connected'])} | {sels} | {errs} | "
                     f"{_f(o.get('conversion'), '.3g')} | {_f(o['settle'].get('settle_s'), '.0f')} | "
                     f"{_f(o.get('wall_s'), '.0f')} | {_f(o.get('cpu_h'))} | {o.get('stopped')} | "
                     f"{', '.join(f'{k} {v}' for k, v in nf.items() if v) or '-'} |")
    return "\n".join(lines) + "\n"


def main(argv: list) -> int:
    if not argv or argv[0] not in ("score", "systems"):
        print(__doc__)
        return 2
    if argv[0] == "systems":
        print(json.dumps(load_systems(), indent=1, default=str))
        print("source:", SUITE_SOURCE)
        return 0
    system, run = argv[1], Path(argv[2])
    run_dir = run / "run" if (run / "run" / "summary.json").is_file() else run
    result = json.loads((run / "result.json").read_text()) if (run / "result.json").is_file() else {}
    o = score_run(load_systems()[system], run_dir, result, run / "snapshots" if (run / "snapshots").is_dir() else None)
    print(json.dumps(o, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
