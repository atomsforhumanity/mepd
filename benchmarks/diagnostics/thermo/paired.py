"""The recorded IRC-verified TSs under "counted as optimized", and the subsystem TS each whole-box TS with a
molecule that takes no part would get (network_expansion._queue_subsystem_ts), tried offline.

    python benchmarks/diagnostics/thermo/paired.py --runs ROOT --profiles DIR --spread SPREAD.json \\
        [--moved VERIFIED.json] [--bound MOVED.json] --workers 8 --out DIR

ROOT/<run>/run/summary.json and its files (unpacked runs); DIR the runs' profiles; SPREAD.json spread.py's
groups. For every step with files.verified true and a recorded thermal correction, from the run's own files:
  - what the TS is a stationary point of (largest force <= 0.002 eV/Angstrom): the whole box, else the
    subsystem its ends file gives. The recorded files do not say what they were optimized on, so this stands
    for the record a new run's file carries (free_energy.optimized_on_comment);
  - its thermal correction counted as that;
  - a whole-box TS whose two IRC ends leave a molecule as it was (network_expansion.subsystem): what the
    queue's rotamer job would do with the TS itself as the guess -- the TS optimizer on the subsystem, started
    from the whole-box geometry restricted to it (one imaginary mode below -10 cm^-1 or it fails), the reacting
    core kept (CORE_TOLERANCE_ANG), then an IRC with minimized ends, accepted when the two ends have the bonds
    of the whole-box IRC's two ends restricted to the subsystem. Then its energy as the network keeps it
    (subsystem + env), its correction counted apart, and the seconds each stage took.
Heights are free energies above the seed at the explorer's level (electronic energy + thermal correction, the
recorded seed correction), against the run's own cutoff. No product ratio or score is computed or read.
`--moved`/`--bound`: verified.py's and moved.py's outputs, to say which of the TSs the saddle step moved, and of
those it changed by more than 0.05 kcal/mol, are such whole-box TSs.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BOHR = 1.8897259886
EV_A = 27.211386245988 * BOHR
H = 627.5094740631


def rmsd(a, b) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(b.T @ a)
    if np.linalg.det(u @ vt) < 0:
        s[-1] = -s[-1]
    return float(np.sqrt(max(0.0, (a ** 2).sum() + (b ** 2).sum() - 2 * s.sum()) / len(a)))


def one(job):
    folder, profile, T, ss, step, files_a, files_b, seed = job
    import mepd
    from qcdata import Structure

    import mepd.discovery.free_energy as fe
    from mepd.cli_conformers import CORE_TOLERANCE_ANG
    from mepd.discovery.network_expansion import graph_edges, subsystem
    from mepd.nodes.node import StructureNode

    e = float(step["ts_energy"])
    out = {"run": Path(folder).name, "label": step["label"], "e": e, "corr": step["corr"], "open": step["open"],
           "window": step["window"]}
    try:
        fp = Path(folder) / step["files"]["ts"]
        st = Structure.open(str(fp))
        irc = step["files"].get("irc")
        species = tuple(Structure.open(str(Path(folder) / f)) for f in (files_a, files_b))
        ends = fe.step_ends({"ts": str(fp), "irc": str(Path(folder) / irc) if irc else None}, species)
        engine = mepd.engine_from_profile(profile)
        sym = [str(s) for s in st.symbols]
        xyz = np.asarray(st.geometry, dtype=float).reshape(-1, 3)
        n = len(sym)

        def force(atoms):
            nd = StructureNode(structure=Structure(symbols=[sym[k] for k in atoms], geometry=xyz[atoms],
                                                   charge=st.charge, multiplicity=st.multiplicity))
            g = np.asarray(engine.compute_gradients([nd])[0], dtype=float).reshape(-1, 3)
            return float(np.linalg.norm(g, axis=1).max()) * EV_A

        free0 = fe.FreeEnergies(None, T, ss)
        out["counted_before"] = [len(p[0]) for p in free0.parts(st, True, ends)]
        on = None
        if force(list(range(n))) <= 2e-3:
            on = list(range(n))
        else:
            ef = fp.with_name(f"{fp.stem}_ends.xyz")
            if ef.exists():
                fr = Structure.open_multi(str(ef))
                e2 = [np.asarray(k.geometry, dtype=float).reshape(-1, 3) / BOHR for k in (fr[0], fr[-1])]
                cut = fe.ts_groups(sym, e2[0], e2)
                if cut and len(cut) > 1 and force(cut[0]) <= 2e-3:
                    on = list(cut[0])
        out["stationary_on"] = None if on is None else "whole" if len(on) == n else on
        said = st if on is None else st.model_copy(update={"extras": {"xyz_comments": [
            fe.optimized_on_comment(None if len(on) == n else on)]}})
        free = fe.FreeEnergies(engine, T, ss)
        out["molecules"] = [len(p[0]) for p in free.parts(said, True, ends)]
        out["counted"] = free._correction(said, True, ends, e)
        out["counted_notes"] = list(free.notes)
        out["height_recorded"] = (e - seed["e"]) * H + step["corr"] - seed["corr"]
        out["height_counted"] = (e - seed["e"]) * H + out["counted"] - seed["corr"]
        bare = ends[0].model_copy(update={"extras": {}})
        if on is not None and len(on) == n and len(free0.parts(bare, True, ends)) > 1:
            p = out["pairing"] = {}
            t0 = time.time()
            ref = StructureNode(structure=st)
            ref._cached_energy = e
            cut = subsystem(*(StructureNode(structure=x) for x in ends), engine, ref=ref)
            p["subsystem"], p["left_out"] = len(cut.atoms), n - len(cut.atoms)
            start = cut.sub(ref)
            try:
                ts = mepd.find_ts(start, engine, maxiter=300, continue_unconverged=0)
            except Exception as exc:
                p["rejected"] = f"TS optimization: {type(exc).__name__}"
                p["seconds_ts"] = time.time() - t0
                return out
            e_start = float(mepd.energies([start], engine)[0])
            p["rmsd_A"] = rmsd(np.asarray(start.structure.geometry).reshape(-1, 3),
                               np.asarray(ts.structure.geometry).reshape(-1, 3)) / BOHR
            p["dE_kcal"] = (float(ts.energy) - e_start) * H
            p["env_kcal"] = cut.env * H
            whole = cut.whole(ts)
            core = sorted(graph_edges(StructureNode(structure=ends[0])) ^ graph_edges(StructureNode(structure=ends[1])))
            x1 = np.asarray(whole.structure.geometry, dtype=float).reshape(-1, 3) / BOHR
            x0 = xyz / BOHR
            p["core_moved_A"] = max((abs(np.linalg.norm(x1[i] - x1[j]) - np.linalg.norm(x0[i] - x0[j]))
                                     for i, j in core), default=0.0)
            p["seconds_ts"] = time.time() - t0
            if p["core_moved_A"] > CORE_TOLERANCE_ANG:
                p["rejected"] = "reacting core moved"
                return out
            t1 = time.time()
            e_sub = float(ts.energy) + cut.env
            sub_said = whole.structure.model_copy(update={"extras": {"xyz_comments": [fe.optimized_on_comment(cut.atoms)]}})
            p["energy"] = e_sub
            p["molecules"] = [len(q[0]) for q in free.parts(sub_said, True, ends)]
            sub_free = fe.FreeEnergies(engine, T, ss)
            p["correction"] = sub_free._correction(sub_said, True, ends, e_sub)
            p["correction_notes"] = list(sub_free.notes)
            p["height"] = (e_sub - seed["e"]) * H + p["correction"] - seed["corr"]
            p["seconds_correction"] = time.time() - t1
            t2 = time.time()
            try:
                path = mepd.irc(ts, engine, end_keywords={"check_frequencies": False}).path
                got = [graph_edges(path.nodes[0]), graph_edges(path.nodes[-1])]
                want = [graph_edges(cut.sub(StructureNode(structure=x))) for x in ends]
                p["ends_clash"] = bool(any(cut.clashes(cut.whole(nd, ts)) for nd in (path.nodes[0], path.nodes[-1])))
                if not (got == want or got == want[::-1]):
                    p["rejected"] = "IRC ends are not the step's"
            except Exception as exc:
                p["rejected"] = f"IRC: {type(exc).__name__}"
            p["seconds_irc"] = time.time() - t2
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--spread", required=True)
    ap.add_argument("--moved")
    ap.add_argument("--bound")
    ap.add_argument("--only", help="run folders whose name starts with this")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import multiprocessing as mp

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs, cutoff = [], {}
    for fp in sorted(Path(a.runs).glob("*/run/summary.json")):
        folder = fp.parent.parent
        if a.only and not folder.name.startswith(a.only):
            continue
        s = json.loads(fp.read_text())
        st = s["steering"]
        cutoff[folder.name] = st.get("cutoff_kcal")
        sp = {x["index"]: x for x in s["species"]}
        seed = {"e": float(s["seed_energy"]), "corr": sp[0]["energy_terms"]["thermal_correction_kcal"]}
        profile = str(Path(a.profiles) / Path(s["inputs"]).name)
        for step in s["steps"]:
            c = (step.get("ts_energy_terms") or {}).get("thermal_correction_kcal")
            if (step.get("files") or {}).get("verified") is True and c is not None:
                jobs.append((str(folder), profile, st["temperature_K"], st["standard_state"],
                             {"label": step["label"], "ts_energy": step["ts_energy"], "corr": c, "files": step["files"],
                              "open": step.get("closed") is None and step.get("kinetics_barrier_kcal") is not None,
                              "window": bool(step.get("deciding_window"))},
                             sp[step["a"]]["file"], sp[step["b"]]["file"], seed))
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(one, jobs, chunksize=2)
    by = {(r["run"], r["label"]): r for r in rows}
    ok = [r for r in rows if "error" not in r]
    system = lambda r: r["run"].split("-lin2")[0]      # noqa: E731
    lines = [f"{len(rows)} IRC-verified TSs with a recorded thermal correction; {len(rows) - len(ok)} failed here.", ""]
    lines.append(f"- stationary on: the whole box {sum(1 for r in ok if r['stationary_on'] == 'whole')}, a subsystem "
                 f"{sum(1 for r in ok if isinstance(r['stationary_on'], list))}, neither "
                 f"{sum(1 for r in ok if r['stationary_on'] is None)}; counted otherwise before: "
                 f"{sum(1 for r in ok if r['molecules'] != r['counted_before'])}")
    refused = [r for r in ok if any("another structure" in n for n in r.get("counted_notes", []))]
    lines.append(f"- saddle step refused (the Hessian at the geometry given): {len(refused)}: " + "; ".join(
        f"{system(r)} {r['label']} " + next(n for n in r["counted_notes"] if "another structure" in n).split("(")[1].split(")")[0]
        for r in refused))
    pop = [r for r in ok if "pairing" in r]
    acc = [r for r in pop if "rejected" not in r["pairing"]]
    lines += ["", "## Whole-box TSs whose IRC ends leave a molecule as it was", "",
              "| system | verified TSs | such whole-box TSs | subsystem TS accepted | rejected |", "|---|---|---|---|---|"]
    for sy in sorted({system(r) for r in ok}):
        k = [r for r in pop if system(r) == sy]
        lines.append(f"| {sy} | {sum(1 for r in ok if system(r) == sy)} | {len(k)} | "
                     f"{sum(1 for r in k if 'rejected' not in r['pairing'])} | "
                     f"{sum(1 for r in k if 'rejected' in r['pairing'])} |")
    lines.append(f"| all | {len(ok)} | {len(pop)} | {len(acc)} | {len(pop) - len(acc)} |")
    why = Counter(r["pairing"]["rejected"] for r in pop if "rejected" in r["pairing"])
    lines += ["", "Rejected: " + ("; ".join(f"{k}: {v}" for k, v in why.most_common()) or "none") + "."]
    if acc:
        rm = np.array([r["pairing"]["rmsd_A"] for r in acc])
        de = np.array([r["pairing"]["dE_kcal"] for r in acc])
        lines += [f"Accepted: RMSD of the reacting part from the whole-box TS, A: median {np.median(rm):.4f}, max "
                  f"{rm.max():.4f}; energy change of the subsystem, kcal/mol: median {np.median(de):.4f}, lowest "
                  f"{de.min():.4f}; IRC ends put back within a spectator's room (Subsystem.clashes): "
                  f"{sum(1 for r in acc if r['pairing'].get('ends_clash'))}.",
                  f"Seconds per such TS (mean): TS optimization {np.mean([r['pairing']['seconds_ts'] for r in pop]):.2f}, "
                  f"correction {np.mean([r['pairing']['seconds_correction'] for r in acc]):.2f}, IRC "
                  f"{np.mean([r['pairing']['seconds_irc'] for r in acc]):.2f}."]
    lines += ["", "## Heights against the run's cutoff (free energy above the seed, explorer level, kcal/mol)", "",
              "| system | such TSs open in the record | open as recorded height | open counted whole | open through "
              "its subsystem TS | through either | through neither |", "|---|---|---|---|---|---|---|"]
    for sy in sorted({system(r) for r in pop}) + ["all"]:
        k = [r for r in pop if r["open"] and (sy == "all" or system(r) == sy)]

        def under(r, key):
            if key == "sub":
                q = r["pairing"]
                return "rejected" not in q and q.get("height") is not None and q["height"] <= cutoff[r["run"]]
            return r[key] <= cutoff[r["run"]]
        w = [under(r, "height_counted") for r in k]
        sb = [under(r, "sub") for r in k]
        lines.append(f"| {sy} | {len(k)} | {sum(under(r, 'height_recorded') for r in k)} | {sum(w)} | {sum(sb)} | "
                     f"{sum(x or y for x, y in zip(w, sb))} | {sum(not (x or y) for x, y in zip(w, sb))} |")
    if pop:
        up = np.array([r["height_counted"] - r["height_recorded"] for r in pop])
        dn = np.array([r["pairing"]["height"] - r["height_recorded"] for r in acc]) if acc else np.array([0.0])
        lines += ["", f"Counted whole against the recorded height: {up.min():+.2f} to {up.max():+.2f} (median "
                  f"{np.median(up):+.2f}); the subsystem TS against the recorded height: {dn.min():+.2f} to "
                  f"{dn.max():+.2f} (median {np.median(dn):+.2f})."]
    # groups: one TS found again, counted as optimized
    lines += ["", "## Spread of one TS found again (verified members of two or more runs)", "",
              "| group | recorded spread | counted as optimized: molecules, members, spread |", "|---|---|---|"]
    groups, n_groups, n_big = [], 0, 0
    for g in json.loads(Path(a.spread).read_text())["TSs"]:
        ms = [by[(m["run"], m["label"])] for m in g["members"] if m.get("verified") is True
              and (m["run"], m["label"]) in by and "error" not in by[(m["run"], m["label"])]]
        if len({m["run"] for m in ms}) < 2:
            continue
        n_groups += 1
        rec = max(m["corr"] for m in ms) - min(m["corr"] for m in ms)
        parts = defaultdict(list)
        for m in ms:
            parts["+".join(map(str, m["molecules"]))].append(m)
        new = {k: max(m["counted"] for m in v) - min(m["counted"] for m in v) for k, v in parts.items()}
        big = max(new.values()) >= 0.05
        n_big += big
        if rec >= 0.05 or big:
            groups.append({"system": g["system"], "what": g["what"], "energy": g["energy"], "recorded": rec,
                           "counted": {k: [len(parts[k]), new[k]] for k in parts}})
            lines.append(f"| {g['system']} {g['what']} {g['energy']:.6f} | {rec:.3f} | " +
                         "; ".join(f"{k}: {len(parts[k])} members, {new[k]:.3f}" for k in sorted(parts)) + " |")
    lines += ["", f"{n_groups} groups; with a spread of 0.05 or more among members counted alike: {n_big}."]
    sub_groups = defaultdict(list)
    for r in acc:
        sub_groups[(system(r), round(r["pairing"]["energy"], 5))].append(r)
    multi = {k: v for k, v in sub_groups.items() if len({r["run"] for r in v}) >= 2}
    sp = {k: max(r["pairing"]["correction"] for r in v) - min(r["pairing"]["correction"] for r in v)
          for k, v in multi.items()}
    lines += ["", f"The subsystem TSs themselves, grouped by system and energy (1e-5 Eh): {len(sub_groups)} groups, "
              f"{len(multi)} with members of two or more runs; spread of their corrections >= 0.05 in "
              f"{sum(1 for x in sp.values() if x >= 0.05)}" +
              (f", largest {max(sp.values()):.3f}" if sp else "") + "."]
    for k, x in sorted(sp.items(), key=lambda kv: -kv[1])[:10]:
        if x >= 0.05:
            lines.append(f"- {k[0]} {k[1]:.5f}: {x:.3f} over {len(multi[k])} "
                         f"({', '.join(r['run'][-2:] + ':' + r['label'] for r in multi[k][:6])})")
    extra = {}
    if a.moved:
        mv = [(r["run"], r["label"]) for r in json.loads(Path(a.moved).read_text())["rows"] if r.get("applied") and
              (r["saddle"]["rmsd_A"] > 0.01 or abs(r["saddle"]["dE_Eh"]) > 1e-5)]
        bd = [(r["run"], r["label"]) for r in json.loads(Path(a.bound).read_text()) if abs(r.get("dE_kcal", 0)) > 0.05] \
            if a.bound else []
        lines += ["", "## The TSs the saddle step moved"]
        for name, keys in (("moved (> 0.01 A or 1e-5 Eh)", mv), ("changed by more than 0.05 kcal/mol", bd)):
            rs = [by[k] for k in keys if k in by and "error" not in by[k]]
            p_ = [r for r in rs if "pairing" in r]
            a_ = [r for r in p_ if "rejected" not in r["pairing"]]
            extra[name] = {"n": len(keys), "whole_box_with_unchanged_molecule": [f"{r['run']}:{r['label']}" for r in p_],
                           "subsystem_ts_accepted": [f"{r['run']}:{r['label']}" for r in a_],
                           "left": [f"{k[0]}:{k[1]}" for k in keys if k not in {(r["run"], r["label"]) for r in a_}]}
            lines.append(f"- {name}: {len(keys)}; whole-box TSs with a molecule that takes no part: {len(p_)}; of those "
                         f"with an accepted subsystem TS: {len(a_)}; left: {len(keys) - len(a_)} "
                         f"({', '.join(extra[name]['left'][:12])})")
    (out / "paired.json").write_text(json.dumps({"rows": rows, "groups": groups, "cutoff": cutoff, "saddle": extra},
                                                indent=1))
    (out / "paired.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
