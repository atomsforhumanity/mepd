"""Analyze › Kinetics: the workspace's network as a microkinetic model
(mepd.microkinetics) -- every reaction (nanoreactor or composed) and every
plain edge whose TS search gave a barrier, among species with energies at
one level of theory -- and the analyses the page shows."""
from __future__ import annotations

import json
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from mepd import microkinetics as mk
from mepd.web.workspace import is_species

HARTREE_KCAL = 627.509474


def _level(job: Optional[dict]) -> Optional[str]:
    return ((job or {}).get("level") or {}).get("key")


def edge_ts(edge: dict, jobs: dict, level_key: Optional[str],
            include_unverified: bool) -> tuple[Optional[float], bool, Optional[str], Optional[str]]:
    """(TS energy in Hartree, verified?, job id, why not) of an edge's lowest
    TS at `level_key`: the IRC-verified ones, else (if allowed) path maxima.
    The TS's own energy, never a barrier added to a species' energy: a
    barrier is measured from the structure its search started from (an IRC
    end, a conformer), which need not be the species' stored structure.
    TSs are compared by energy, not by barrier, for the same reason."""
    ver, unver, other_level, no_energy = [], [], False, False

    def consider(e_ts, level, verified, jid):
        nonlocal other_level, no_energy
        if level != level_key:
            other_level = True
        elif e_ts is None:
            no_energy = True
        else:
            (ver if verified else unver).append((e_ts, jid))

    for j in jobs.values():
        if edge["id"] not in j["targets"]["edges"] or j["status"] != "done":
            continue
        sm = j.get("summary") or {}
        if sm.get("barrier_kcal") is None:
            continue
        consider(sm.get("ts_energy_hartree"), _level(j), sm.get("barrier_verified") is not False, j["id"])
    o = edge.get("origin") or {}
    if o.get("barrier_kcal") is not None:
        t = o.get("qmmm_ts") or {}
        if t:
            consider(t.get("energy"), t.get("level"), True, t.get("job"))
        else:
            # (an expansion's step whose IRC is not checked yet is unverified)
            consider(o.get("ts_energy_hartree"), _level(jobs.get(o.get("job"))), o.get("barrier_verified") is not False,
                     o.get("job"))
    for pool, verified in ((ver, True), (unver, False)):
        if pool and (verified or include_unverified):
            e, jid = min(pool, key=lambda x: x[0])
            return e, verified, jid, None
    why = ("no barrier yet" if not (other_level or no_energy or unver) else
           "only unverified barriers" if unver else
           "TS at another level of theory" if other_level and not no_energy else
           "TS energy not known")
    return None, False, None, why


def build(snap: dict, jobs: dict, level_key: Optional[str], *, include_unverified: bool = False,
          full: Optional[dict] = None):
    """(Network, species ids, step info, excluded) for the workspace (or
    one network of it: `snap` as mepd.web.variants.network_view gives it,
    `full` the whole workspace, for a variant's fast estimates)."""
    st = snap["structures"]

    def estimate(edge):
        """A network variant's fast estimate, until its re-optimized TS is in."""
        if not edge.get("variant"):
            return None
        from mepd.web.variants import estimate_ts

        return estimate_ts(edge, full or snap, jobs, level_key)

    def usable(rec):
        return rec and rec.get("energy") is not None and (level_key is None or (rec.get("level") or {}).get("key") == level_key)

    ids = [sid for sid, r in st.items() if is_species(r) and usable(r)]
    index = {sid: k for k, sid in enumerate(ids)}
    names = [st[s]["name"] for s in ids]
    g = [st[s]["energy"] * HARTREE_KCAL for s in ids]
    geom = list(ids)          # the structure whose geometry each model species stands for
    steps, info, excluded = [], [], []

    # A complex is its molecules side by side, not a species of its own: a
    # reaction runs from its separated reactants to its separated products,
    # with its TS energy (relative to the separated molecules, which is the
    # pre-equilibrium rate K_assoc * k through the complex). So "A + B" at
    # 1 M is 1 M of A and 1 M of B, and what a reaction makes is counted in
    # its products' own amounts.
    for r in (snap.get("reactions") or {}).values():
        label = r.get("label") or "reaction"
        edge = snap["edges"].get(r.get("edge") or "")
        if edge is None:
            excluded.append({"label": label, "reason": "no TS endpoints (" + (r.get("complex_reason") or "none") + ")",
                             "reaction": r["id"]})
            continue
        e_ts, verified, jid, why = edge_ts(edge, jobs, level_key, include_unverified)
        est = None
        if e_ts is None:
            est = estimate(edge)
            if est is None:
                excluded.append({"label": label, "reason": "fast estimate not in yet" if edge.get("variant")
                                 and why == "no barrier yet" else why, "reaction": r["id"]})
                continue
            e_ts, verified, jid = est, False, None
        missing = [s for s in dict.fromkeys(r["reactants"] + r["products"]) if s not in index]
        if missing:
            excluded.append({"label": label, "reaction": r["id"],
                             "reason": "energies missing at this level: " + ", ".join(st.get(s, {}).get("name", s) for s in missing),
                             # molecules a minimization would give an energy (Kinetics offers to compute them)
                             "missing": [s for s in missing if is_species(st.get(s))]})
            continue
        steps.append(mk.Step([index[s] for s in r["reactants"]], [index[s] for s in r["products"]],
                             e_ts * HARTREE_KCAL, label, info={"ts": (jid, e_ts) if jid else None}))
        info.append({"kind": "reaction", "reaction": r["id"], "edge": edge["id"], "label": label,
                     "verified": verified, "job": jid, "shuttles": r.get("shuttles") or [],
                     "estimate": est is not None})
    complexes = {sid for sid, rec in st.items() if rec.get("role") == "complex"}
    for e in snap["edges"].values():
        if e["source"] in complexes or e["target"] in complexes:
            continue
        e_ts, verified, jid, why = edge_ts(e, jobs, level_key, include_unverified)
        if e_ts is None:
            if why not in ("no barrier yet", "only unverified barriers"):
                excluded.append({"label": f"{st.get(e['source'], {}).get('name', '?')} -> "
                                          f"{st.get(e['target'], {}).get('name', '?')}", "reason": why, "edge": e["id"]})
            continue
        a, c = e["source"], e["target"]
        label = f"{st.get(a, {}).get('name', '?')} -> {st.get(c, {}).get('name', '?')}"
        if a not in index or c not in index:
            excluded.append({"label": label, "reason": "an end has no energy at this level", "edge": e["id"],
                             "missing": [s for s in (a, c) if s not in index and is_species(st.get(s))]})
            continue
        steps.append(mk.Step([index[a]], [index[c]], e_ts * HARTREE_KCAL, label, info={"ts": (jid, e_ts)}))
        info.append({"kind": "edge", "edge": e["id"], "label": label, "verified": verified, "job": jid,
                     "shuttles": []})
    # The same transformation from several runs (same reactants and products,
    # shuttles included) is one channel: keep its lowest TS, not two copies
    # that would double its rate.
    best: dict = {}
    for k, stp in enumerate(steps):
        a, b = tuple(sorted(stp.reactants)), tuple(sorted(stp.products))
        key = min((a, b), (b, a))
        if key not in best or stp.g_ts < steps[best[key]].g_ts:
            best[key] = k
    keep = sorted(best.values())
    merged = len(steps) - len(keep)
    for k in keep:
        a, b = tuple(sorted(steps[k].reactants)), tuple(sorted(steps[k].products))
        info[k]["copies"] = sum(1 for s2 in steps if min((tuple(sorted(s2.reactants)), tuple(sorted(s2.products))),
                                                          (tuple(sorted(s2.products)), tuple(sorted(s2.reactants))))
                                == min((a, b), (b, a)))
    steps, info = [steps[k] for k in keep], [info[k] for k in keep]
    net = mk.Network(names, np.array(g), steps)
    net.geom = geom
    return net, ids, info, excluded


def ts_geometry(job_or_jobs, e_ts: float, tol: float = 2e-3):
    """The TS structure in the output of a job (or of any of several jobs)
    whose energy is closest to `e_ts` (Hartree, within `tol`): (symbols,
    coordinates in Angstrom), or None."""
    jobs_ = job_or_jobs if isinstance(job_or_jobs, list) else [job_or_jobs]
    files = []
    for job in jobs_:
        out = Path((job or {}).get("output_dir") or "")
        if out.is_dir():
            files += list(out.rglob("ts*.xyz"))
    best = None
    for f in files:
        if "irc" in f.name:
            continue
        en = f.with_suffix(".energies")
        try:
            e = float(en.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if abs(e - e_ts) <= tol and (best is None or abs(e - e_ts) < best[0]):
            best = (abs(e - e_ts), f)
    if best is None:
        return None
    lines = best[1].read_text().splitlines()
    n = int(lines[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def free_energies(ws, jobs: dict, net: mk.Network, *, workers: int = 2) -> tuple[Callable, dict]:
    """G(T) for every species and TS of `net` (built by build()):
    E + quasi-RRHO correction from GFN2-xTB single-point Hessians, cached
    per geometry under the workspace's thermo/ folder. Returns
    (energies_at(T) -> (g, g_ts), report)."""
    from mepd import thermo

    solvent, model = None, "alpb"
    try:
        solv = tomllib.loads(ws.read_profile(ws.level_profile)).get("solvation") if ws.level_profile else None
    except Exception:
        solv = None
    if solv and solv.get("solvent"):
        solvent, model = str(solv["solvent"]).lower(), str(solv.get("model", "alpb")).lower()
    cache = Path(ws.root) / "thermo"
    snap = ws.snapshot()["structures"]

    def species_task(k):
        sid = net.geom[k]
        rec = snap[sid]
        s = ws.load_structure(sid)
        xyz = np.asarray(s.geometry).reshape(-1, 3) * 0.529177210903
        return list(s.symbols), xyz, int(rec["charge"]), int(rec["multiplicity"])

    tasks, approx = {}, []
    for k in range(len(net.names)):
        tasks[("sp", k)] = species_task(k)
    for j, stp in enumerate(net.steps):
        ref = (stp.info or {}).get("ts")
        if not ref:
            continue
        job = jobs.get(ref[0])
        # A TS search's summary carries the best TS of the "Sample more
        # paths" runs that extend it, whose files are in their own folders.
        family = [job] + [j for j in jobs.values() if job and j.get("extends") == job.get("id")]
        if job and job.get("extends") in jobs:
            family.append(jobs[job["extends"]])
        found = ts_geometry([j for j in family if j], ref[1])
        if found is None:
            approx.append(stp.label)
            continue
        tasks[("ts", j)] = (*found, int(job.get("charge", 0)), int(job.get("multiplicity", 1)))

    def run(item):
        key, (symbols, xyz, q, mult) = item
        return key, thermo.cached_frequencies(cache, symbols, xyz, charge=q, multiplicity=mult,
                                              solvent=solvent, model=model)

    new = sum(not (cache / f"{thermo.cache_key(sym, xyz, q, m, solvent, model)}.json").exists()
              for sym, xyz, q, m in tasks.values())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        freqs = dict(pool.map(run, tasks.items()))

    def corr(key, T):
        symbols, xyz, _q, mult = tasks[key]
        fr = freqs[key]
        return thermo.qrrho(fr["freqs"], symbols, xyz, T, rotational_number=fr["rotational_number"],
                            linear=fr["linear"], multiplicity=mult, ts=key[0] == "ts")["g_corr"]

    def energies_at(T):
        g = np.array([net.g[k] + corr(("sp", k), T) for k in range(len(net.names))])
        g_ts = []
        for j, stp in enumerate(net.steps):
            if ("ts", j) in tasks:
                g_ts.append(stp.g_ts + corr(("ts", j), T))
            else:
                # No TS geometry: take the thermal correction of the
                # separated molecules it starts from.
                g_ts.append(stp.g_ts + sum(corr(("sp", i), T) for i in stp.reactants))
        return g, np.array(g_ts)

    n_imag = {key: sum(1 for x in fr["freqs"] if x < -20) for key, fr in freqs.items()}
    report = {"method": "quasi-RRHO, GFN2-xTB single-point Hessians" + (f" ({model.upper()}, {solvent})" if solvent else ""),
              "structures": len(tasks), "computed": new, "ts_without_geometry": approx,
              "extra_imaginary": [net.names[k[1]] if k[0] == "sp" else net.steps[k[1]].label
                                  for k, n in n_imag.items() if n > (1 if k[0] == "ts" else 0)],
              # No negative mode: the reaction coordinate stays in as a vibration (mepd#12).
              "ts_without_imaginary": [net.steps[k[1]].label for k, fr in freqs.items()
                                       if k[0] == "ts" and not any(x < 0 for x in fr["freqs"])]}
    return energies_at, report


def unchecked_running(snap: dict, jobs: dict, excluded: list) -> int:
    """Steps left out as unverified that belong to a running expansion: their IRC check is still to come."""
    n = 0
    for x in excluded:
        if x.get("reason") != "only unverified barriers":
            continue
        o = ((snap["edges"].get(x.get("edge") or "") or {}).get("origin")) or {}
        if (jobs.get(o.get("job") or "") or {}).get("status") in ("running", "queued"):
            n += 1
    return n


def analyze(snap: dict, jobs: dict, level_key: Optional[str], *, initial: dict, held: list, temperature: float,
            time_s: float, target: Optional[str], include_unverified: bool = False, control: bool = True,
            sweep: Optional[list] = None, control_what: str = "amount", free_energy: bool = False,
            ws=None, network: Optional[str] = None) -> dict:
    """`network`: a variant's id (mepd.web.variants), else the parent network."""
    from mepd.web.variants import network_view

    full = snap
    snap = network_view(snap, network)
    net, ids, info, excluded = build(snap, jobs, level_key, include_unverified=include_unverified, full=full)
    warnings = [] if free_energy and ws is not None else [
        "Electronic energies stand in for free energies: absolute rates (bimolecular ones especially, "
        "which lack the entropy cost of meeting) are orders of magnitude; read ratios, controlling steps "
        "and trends."]
    n_est = sum(1 for i in info if i.get("estimate"))
    if n_est:
        warnings.append(f"{n_est} step{'s' if n_est != 1 else ''} use{'' if n_est != 1 else 's'} a fast estimate "
                        "(est.): the parent's barrier shifted by the edit, until the re-optimized TS is in.")
    # Molecules whose missing energy keeps reactions out (one minimization each brings them in).
    need = list(dict.fromkeys(s for x in excluded for s in x.get("missing") or []))
    missing_energy = [{"id": s, "name": full["structures"].get(s, {}).get("name", s) if full else s,
                       "reactions": sum(1 for x in excluded if s in (x.get("missing") or []))} for s in need]
    if not net.steps:
        return {"error": ("No reaction in this workspace has a barrier yet: run TS searches first"
                          + (" (or include unverified barriers)" if not include_unverified else "") + "."
                          if not missing_energy else
                          "No reaction can enter the model yet: the molecules below have no energy at this level."),
                "excluded": excluded, "missing_energy": missing_energy,
                "start_missing": [s for s in initial if s not in ids],
                "unchecked_running": unchecked_running(snap, jobs, excluded)}
    # Only species a step touches, or the user put in.
    used = sorted({i for s in net.steps for i in s.reactants + s.products}
                  | {ids.index(s) for s in initial if s in ids})
    keep = {old: new for new, old in enumerate(used)}
    steps = [mk.Step([keep[i] for i in s.reactants], [keep[i] for i in s.products], s.g_ts, s.label, info=s.info)
             for s in net.steps]
    geom = [net.geom[i] for i in used]
    net = mk.Network([net.names[i] for i in used], net.g[used], steps)
    net.geom = geom
    energies_at, fe_report = None, None
    if free_energy and ws is not None:
        energies_at, fe_report = free_energies(ws, jobs, net)
        g_T, g_ts_T = energies_at(temperature)
        net = mk.Network(net.names, g_T, [mk.Step(s.reactants, s.products, float(x), s.label, info=s.info)
                                          for s, x in zip(net.steps, g_ts_T)])
        fresh = (f" Frequencies computed for {fe_report['computed']} new structure"
                 f"{'s' if fe_report['computed'] != 1 else ''} (kept for the next runs)." if fe_report["computed"] else "")
        warnings.append(f"Free energies at {temperature:g} K ({fe_report['method']}; 1 M standard state).{fresh} "
                        "One structure per species (no conformer averaging); association entropy uses "
                        "gas-phase formulas, which overstate it in solution.")
        if fe_report["ts_without_geometry"]:
            warnings.append("TS geometry not found, thermal correction taken from the reactant side: "
                            + ", ".join(fe_report["ts_without_geometry"]))
        if fe_report.get("ts_without_imaginary"):
            warnings.append("No imaginary frequency in the GFN2-xTB Hessian of the TS (its soft mode counts as a "
                            "vibration, G(TS) too low by ~0.5-1 kcal/mol): "
                            + ", ".join(fe_report["ts_without_imaginary"]))
        if fe_report["extra_imaginary"]:
            warnings.append("Extra imaginary frequencies (below -20 cm-1) in: "
                            + ", ".join(fe_report["extra_imaginary"]))
    sids = [ids[i] for i in used]
    pos = {sid: k for k, sid in enumerate(sids)}
    c0 = np.zeros(len(sids))
    for sid, conc in (initial or {}).items():
        if sid in pos:
            c0[pos[sid]] = float(conc)
    unknown = [snap["structures"].get(s, {}).get("name", s) for s in initial if s not in pos]
    start_missing = [s for s in initial if s not in pos]   # (the page says it at the top: it is what was asked)
    if unknown:
        warnings.append("Not in the model (no energy at this level, or no step with a barrier): "
                        + ", ".join(unknown))
    held_idx = [pos[s] for s in held or [] if s in pos]
    res = mk.simulate(net, c0, temperature, time_s, held=held_idx)
    for (bf, br), s in zip(net.barriers(), net.steps):
        if bf < -0.1 or br < -0.1:
            warnings.append(f"{s.label}: its TS lies below one side ({bf:.1f} / {br:.1f} kcal/mol); it counts as "
                            "barrierless that way.")
    out = {
        "species": [{"id": sid, "name": net.names[k], "g": float(net.g[k]), "c0": float(c0[k]), "final": float(res.c[-1, k]),
                     "max": float(res.c[:, k].max()), "held": k in held_idx,
                     "mass": _mass(full["structures"].get(sid, {}).get("formula") if full else None)}
                    for k, sid in enumerate(sids)],
        "times": res.times.tolist(),
        "series": {sid: res.c[:, k].tolist() for k, sid in enumerate(sids)},
        "steps": [{**i, "barrier_fwd": bf, "barrier_rev": br, "net_rate": float(nr), "extent": float(ex)}
                  for i, (bf, br), nr, ex in zip(info, net.barriers(), res.net_rate, res.extent)],
        "steady": res.steady, "temperature": temperature, "time_s": time_s, "excluded": excluded,
        "missing_energy": missing_energy, "start_missing": start_missing,
        # steps of a running expansion whose IRC check is still to come (left out unless asked for)
        "unchecked_running": unchecked_running(snap, jobs, excluded),
        "warnings": warnings, "mode": "constant feed" if held_idx else "batch",
        "energies": "free" if fe_report else "electronic", "free_energy": fe_report,
    }
    out["coverage"] = coverage(full or snap, jobs, out["species"], temperature, time_s, set(initial or {}))
    if target in pos:
        t = pos[target]
        out["target"] = {"id": target, "name": net.names[t], "final": float(res.c[-1, t]),
                         "rate": mk.formation_rate(net, res, t),
                         "yield_of": float(c0.sum()) or None}
        if control:
            what = control_what if control_what in ("amount", "rate") else "amount"
            drc = mk.degree_of_control(net, c0, temperature, time_s, t, what=what, held=held_idx)
            out["control"] = {"what": what, "steps": drc["steps"], "species": drc["species"]}
        if sweep:
            out["sweep"] = mk.sweep(net, c0, [float(x) for x in sweep], time_s, t, held=held_idx,
                                    energies_at=energies_at)
    elif sweep:
        out["sweep"] = mk.sweep(net, c0, [float(x) for x in sweep], time_s, None, held=held_idx,
                                energies_at=energies_at)
    return out


def _canonical(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    try:
        from rdkit import Chem

        m = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(m) if m is not None else smiles
    except Exception:
        return smiles


def _mass(formula: Optional[str]) -> Optional[float]:
    """Molar mass (g/mol) of a formula like "C2H4O2" (a complex's: all its molecules), or None."""
    if not formula:
        return None
    try:
        from ase.formula import Formula
        from ase.data import atomic_masses, atomic_numbers

        return float(sum(atomic_masses[atomic_numbers[el]] * n for el, n in Formula(formula).count().items()))
    except Exception:
        return None


# Kinetics far from the conditions a network was explored at: what the exploration searched past, and what
# these conditions cross. An exploration steered by kinetics expands a species only when its own kinetics put
# material there, and (steer "access") only past barriers below its cutoff; hotter or longer kinetics cross
# higher barriers, out of species it never expanded, into steps it never looked for.
EXPLORE_OPS = ("graph-enumeration",)
BEYOND_MARGIN_KCAL = 2.0   # crossing this much above what was searched past counts as beyond it


def _exploration(job: dict) -> Optional[dict]:
    """An expansion's conditions (summary.json's steering) and the workspace species it expanded."""
    out = Path(job.get("output_dir") or "")
    summary = None
    for name in ("summary.json", "summary_partial.json"):
        try:
            summary = json.loads((out / name).read_text())
            break
        except (OSError, ValueError):
            continue
    nodes = job.get("live_nodes") or {}
    steering = (summary or {}).get("steering") or {}
    p = job.get("params") or {}
    T = steering.get("temperature_K") or p.get("cutoff_temperature") or p.get("temperature")
    t = steering.get("time_s") or p.get("cutoff_time") or p.get("time_s")
    sources = {i for r in (summary or {}).get("rounds") or [] for i in r.get("sources") or []}
    expanded = {nodes[str(i)] for i in sources if str(i) in nodes}
    expanded |= {nodes["0"]} if "0" in nodes else set(((job.get("targets") or {}).get("structures") or [])[:1])
    mode = steering.get("mode") or p.get("steer")
    if not T or not t or mode not in ("flux", "access"):
        return None   # not steered by kinetics: no conditions to compare with
    from mepd.network_aware import eyring_cutoff_kcal

    return {"job": job["id"], "title": job.get("title"), "status": job.get("status"), "mode": mode,
            "temperature": float(T), "time_s": float(t), "cutoff_kcal": steering.get("cutoff_kcal"),
            "reach_kcal": eyring_cutoff_kcal(float(T), float(t)),
            "threshold": float(steering.get("flux_threshold") or p.get("flux_threshold") or 0.01),
            "expanded": sorted(expanded), "nodes": set(nodes.values())}


def coverage(snap: dict, jobs: dict, species: list, temperature: float, time_s: float, start: set) -> Optional[dict]:
    """How far these kinetics are from what explored the network: the explorations behind its species (their
    conditions; the barrier their kinetics crossed in their time, `reach_kcal`, and the access cutoff they
    searched up to), the barrier these conditions cross (`reach_kcal`), whether that is beyond what was searched
    (`beyond`), and the `frontier`: species that hold material here (at their peak, at least the explorations'
    threshold of the start) but that no exploration expanded, so what they do next was never searched, with
    their share of the starting mass at the end. None when no kinetics-steered exploration made this network."""
    from mepd.network_aware import eyring_cutoff_kcal

    ids = {s["id"] for s in species}
    found = [e for j in jobs.values() if j.get("op") in EXPLORE_OPS and j.get("status") in ("done", "running", "cancelled",
                                                                                            "interrupted", "failed")
             for e in [_exploration(j)] if e is not None and e["nodes"] & ids]
    if not found:
        return None
    reach = eyring_cutoff_kcal(float(temperature), float(time_s))
    searched = max((e["cutoff_kcal"] if e["cutoff_kcal"] is not None else e["reach_kcal"]) for e in found)
    expanded = {sid for e in found for sid in e["expanded"]}
    # The explorer expands states: a mixture's ("ketene + water") molecules were expanded with it.
    structures = snap.get("structures") or {}
    expanded_smiles = {_canonical(x) for sid in expanded for x in (structures.get(sid, {}).get("smiles") or "").split(".") if x}
    expanded |= {s["id"] for s in species if _canonical(structures.get(s["id"], {}).get("smiles")) in expanded_smiles - {None}}
    threshold = min(e["threshold"] for e in found)
    total0 = sum(s["c0"] * (s.get("mass") or 0) for s in species) or None
    total_c0 = sum(s["c0"] for s in species) or 1.0
    frontier = []
    for s in species:
        if s["id"] in expanded or s["id"] in start or s["max"] < threshold * total_c0:
            continue
        share = (s["final"] * s["mass"] / total0) if total0 and s.get("mass") else None
        frontier.append({"id": s["id"], "name": s["name"], "share": share, "peak": s["max"] / total_c0})
    frontier.sort(key=lambda f: -(f["share"] if f["share"] is not None else f["peak"]))
    return {"explorations": [{k: v for k, v in e.items() if k not in ("expanded", "nodes")} for e in found],
            "reach_kcal": reach, "searched_kcal": searched, "beyond": reach > searched + BEYOND_MARGIN_KCAL,
            "frontier": frontier, "frontier_share": sum(f["share"] or 0 for f in frontier), "threshold": threshold}
