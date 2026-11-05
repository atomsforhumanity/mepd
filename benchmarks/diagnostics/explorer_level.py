"""The explorer's own free energies and kinetics, for a hand-collected set of species and TSs.

Tracing's "ceiling" (BRIEF.md): the selectivity the pipeline gets when it is handed every observed channel through
targeted searches, at the explorer's own levels. This module applies exactly the explorer's code to such a set:

- free energies: mepd.discovery.free_energy.FreeEnergies (qRRHO on the run engine's Hessians, species of several
  molecules apart at the standard state, a TS's spectators apart/bound);
- steering single points: mepd.discovery.energy_refine._single_point on FreeEnergies.parts (molecules apart), the
  shift E_high - E_low relative to the seed, as network_expansion._free_energy_view applies it;
- mirror images: mepd.discovery.mirror.MirrorFinder + Mirrors.extend;
- kinetics: mepd.discovery.kinetics.simulate (Eyring, first-order network from the seed).

Every per-structure number (E_low, thermal correction, E_high) is cached in a JSON file keyed by the geometry, so a
rerun or another script reuses them. Nothing here changes explorer code.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

HARTREE = 627.5094740631
BOHR = 1.8897259886
R_KCAL = 1.987204259e-3


def read_frames(path) -> list:
    """Every frame of an xyz file as (symbols, coordinates in Angstrom)."""
    lines = Path(path).read_text().splitlines()
    out, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[0])
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        out.append(([r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])))
        i += n + 2
    return out


def structure(symbols, xyz_A, charge: int = 0, multiplicity: int = 1):
    from qcdata import Structure

    return Structure(symbols=list(symbols), geometry=np.asarray(xyz_A, dtype=float) * BOHR, charge=int(charge),
                     multiplicity=int(multiplicity))


def load(path, frame: int = 0, charge: int = 0, multiplicity: int = 1):
    sym, xyz = read_frames(path)[frame]
    return structure(sym, xyz, charge, multiplicity)


def angstrom(st) -> np.ndarray:
    return np.asarray(st.geometry, dtype=float) / BOHR


def xyz_text(st, comment: str = "") -> str:
    X = angstrom(st)
    return f"{len(X)}\n{comment}\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in
                                             zip(map(str, st.symbols), X))


def geo_key(st, extra=()) -> str:
    xyz = np.round(np.asarray(st.geometry, dtype=float), 4) + 0.0
    blob = repr((list(map(str, st.symbols)), xyz.tolist(), int(st.charge), int(st.multiplicity)) + tuple(extra))
    return hashlib.sha1(blob.encode()).hexdigest()[:24]


class Cache:
    """{key: {...}} in one JSON file, written after every change."""

    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key):
        return self.d.get(key)

    def put(self, key, value):
        self.d[key] = value
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=0))
        tmp.replace(self.path)


class ExplorerLevel:
    """The explorer's engine (profile), steering engine (steer profile) and free energies at (T, standard state)."""

    def __init__(self, profile, steer_profile: Optional[str], temperature: float, standard_state: str,
                 cache: str, workers: int = 1):
        from mepd.inputs import RunInputs

        from mepd.discovery.free_energy import FreeEnergies

        self.engine = RunInputs.open(profile).engine
        self.steer = RunInputs.open(steer_profile).engine if steer_profile else None
        self.T, self.ss = float(temperature), standard_state
        self.free = FreeEnergies(self.engine, self.T, standard_state, workers)
        self.cache = Cache(cache)
        self.workers = workers

    # -- per structure -------------------------------------------------------------------------------------------
    def parts(self, st, ts: bool, ends=None):
        return self.free.parts(st, ts, ends)

    def correction(self, st, ts: bool, ends=None) -> float:
        """Thermal correction (kcal/mol) as the explorer computes it (molecules apart, a TS's spectators...)."""
        key = "corr:" + self.free._key(st, ts, ends) + f":{self.T:g}:{self.ss}"
        hit = self.cache.get(key)
        if hit is None:
            v = self.free._correction(st, ts, ends)
            hit = {"value": list(v) if isinstance(v, tuple) else float(v)}
            self.cache.put(key, hit)
        v = hit["value"]
        from mepd.discovery.free_energy import both_channels

        return both_channels(*v, self.T) if isinstance(v, list) else float(v)

    def single_point(self, st, ts: bool, ends=None) -> dict:
        """{e_low, e_high, delta_kcal}: the steering single point, molecules apart as the free energies count them."""
        from mepd.discovery.energy_refine import Item, _fragments, _single_point

        groups = self.free._groups(st, ts, ends)
        key = "sp:" + geo_key(st, (ts,) + ((groups,) if groups and len(groups) > 1 else ()))
        hit = self.cache.get(key)
        if hit is None:
            parts = _fragments(st, self.parts(st, ts, ends))
            hit = _single_point(Item("ts" if ts else "species", 0, st, parts), self.engine, self.steer)
            self.cache.put(key, hit)
        return hit

    def energy(self, st) -> float:
        """E_low (Hartree) of the whole structure (the explorer's node energy)."""
        key = "e:" + geo_key(st)
        hit = self.cache.get(key)
        if hit is None:
            from mepd.discovery.energy_refine import _energy

            hit = {"value": _energy(st, self.engine)}
            self.cache.put(key, hit)
        return hit["value"]


@dataclass
class Species:
    label: str
    st: object                      # the structure the explorer would keep (its lowest-energy conformer)
    product: str = ""               # benchmark product label ("" for the seed / others)


@dataclass
class TS:
    label: str
    st: object
    a: int                          # species index of one end (the seed: 0)
    b: int
    ends: Optional[tuple] = None    # the step's two ends in the TS's atom order (IRC first/last frames)
    channel: str = ""


@dataclass
class Network:
    species: list
    steps: list
    notes: list = field(default_factory=list)


def evaluate(level: ExplorerLevel, net: Network, time_s: float, steer: bool = True, mirrors: bool = True) -> dict:
    """The explorer's kinetics on `net`: free energies (+ steering single points), mirror images, simulate.
    Returns per-step free barriers from the seed (kcal/mol), kinetics, and per-product final amounts."""
    from mepd.discovery.kinetics import Step, simulate
    from mepd.discovery.mirror import MirrorFinder
    from mepd.nodes.node import StructureNode

    E = [level.energy(s.st) for s in net.species]
    corr = [level.correction(s.st, False) for s in net.species]
    ts_E = [level.energy(t.st) for t in net.steps]
    ts_corr = [level.correction(t.st, True, t.ends) for t in net.steps]
    if steer and level.steer is not None:
        d_sp = [level.single_point(s.st, False)["delta_kcal"] for s in net.species]
        d_ts = [level.single_point(t.st, True, t.ends)["delta_kcal"] for t in net.steps]
    else:
        d_sp, d_ts = [0.0] * len(E), [0.0] * len(ts_E)
    g = [e + (c + d - d_sp[0]) / HARTREE for e, c, d in zip(E, corr, d_sp)]
    steps = [Step(t.a, t.b, te + (c + d - d_sp[0]) / HARTREE, t.label, {}) for t, te, c, d in
             zip(net.steps, ts_E, ts_corr, d_ts)]
    rec = {"temperature": level.T, "time_s": time_s, "standard_state": level.ss, "steered": bool(steer and level.steer),
           "species": [], "steps": []}
    for k, s in enumerate(net.species):
        rec["species"].append({"index": k, "label": s.label, "product": s.product,
                               "dE_kcal": (E[k] - E[0]) * HARTREE, "corr_kcal": corr[k], "steer_delta_kcal": d_sp[k],
                               "dG_kcal": (g[k] - g[0]) * HARTREE})
    for m, t in enumerate(net.steps):
        rec["steps"].append({"index": m, "label": t.label, "channel": t.channel, "a": t.a, "b": t.b,
                             "dE_kcal": (ts_E[m] - E[0]) * HARTREE, "corr_kcal": ts_corr[m],
                             "steer_delta_kcal": d_ts[m], "dG_kcal": (steps[m].ts_energy - g[0]) * HARTREE})
    n = len(net.species)
    if mirrors:
        finder = MirrorFinder()
        nodes = [StructureNode(structure=s.st) for s in net.species]

        def parts(st):
            return [p for p, _, _ in level.free.parts(st, False)]

        mir = finder(nodes, steps, [t.st for t in net.steps], parts)
        n_all, all_steps, all_g = mir.extend(steps, g, level.T)
        rec["mirrors"] = mir.record()
    else:
        n_all, all_steps, all_g = n, steps, g
    k = simulate(n_all, all_steps, all_g, temperature=level.T, time_s=time_s)
    parent = list(range(n)) + ([v["mirror_of"] for v in rec["mirrors"]["virtual_species"]] if mirrors else [])
    final = {}
    for i, c in enumerate(k.final_concentration):
        lab = net.species[parent[i]].product or net.species[parent[i]].label
        final[lab] = final.get(lab, 0.0) + float(c)
    rec["final"] = final
    rec["final_concentration"] = [float(c) for c in k.final_concentration]
    # the same from the rate constants alone (parallel irreversible channels out of the seed)
    rt = R_KCAL * level.T
    kch = {}
    for st in all_steps:
        if st.a == 0 or st.b == 0:
            other = st.b if st.a == 0 else st.a
            lab = net.species[parent[other]].product or net.species[parent[other]].label
            kch[lab] = kch.get(lab, 0.0) + math.exp(-(st.ts_energy - all_g[0]) * HARTREE / rt)
    rec["rate_sums"] = kch
    rec["notes"] = list(dict.fromkeys(level.free.notes))
    return rec


def ratio(rec: dict, num: str, den: str, use: str = "final") -> float:
    d = rec[use] if use != "final" else rec["final"]
    a, b = d.get(num, 0.0), d.get(den, 0.0)
    return a / b if b > 0 else float("inf")


def kcal_ratio(r: float, T: float) -> float:
    return R_KCAL * T * math.log(r)
