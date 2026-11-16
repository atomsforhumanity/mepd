"""Recorded TSs found by several runs: their thermal corrections recomputed, as given and converged first.

    python benchmarks/diagnostics/thermo/recompute.py --spread DIR/spread.json --min-spread 0.01 [--window] \\
        --workers 8 --out DIR

The engine of each TS is its run's own profile (the file its summary.json names, in the run set's inputs/ folder).

Takes the groups of spread.py (one TS found again: same step, same electronic energy) whose recorded spread is at
least --min-spread, and with --window also every group with a member in a deciding window. For each member the
run's own files are read from its run.tgz (TS, its IRC or ends file, its step's two species) and the explorer's
own code (mepd.discovery.free_energy.FreeEnergies._correction, the step's ends found as the run finds them)
gives:
  given      the correction with the Hessian at the geometry in the file (FreeEnergies._saddle switched off):
             what the run recorded, if the record is reproduced;
  converged  the TS converged to HESSIAN_FMAX before its Hessian, counted as its step's ends make it look (a
             file without the record of what it was optimized on: every recorded run's);
  counted    the same, counted as what it was optimized on. The recorded files do not say what that was, so it
             is measured here: the atoms the TS is a stationary point of (largest force <= 0.002 eV/Angstrom)
             -- the subsystem its ends file gives (network_expansion.subsystem's reading of the two ends), else
             the whole box -- and written into the structure as a new run's file would carry it
             (free_energy.optimized_on_comment). Neither stationary: no record, counted as before;
and, of the molecule that carries the TS's reaction mode: how many molecules the TS counts as, its largest atomic
force and its three lowest frequencies at the geometry in the file. Also what the TS is a stationary point of
(largest atomic force of the whole box, and of the atoms the run recorded as reacting, computed alone), and why
its step's ends gave the molecules they gave (bonds of both ends missing at the TS, TS bonds in neither end,
"bonds" made or broken between the recorded reacting atoms and the others). Nothing is searched or built.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BOHR = 1.8897259886


def extract(members, work: Path):
    by_run = defaultdict(list)
    for m in members:
        by_run[m["dir"]].append(m)
    for d, ms in by_run.items():
        dest = work / Path(d).name
        dest.mkdir(parents=True, exist_ok=True)
        names = set()
        for m in ms:
            names |= {f"run/ts/{m['label']}.xyz", f"run/ts/{m['label']}_irc.xyz", f"run/ts/{m['label']}_ends.xyz",
                      f"run/species/species_{m['a']}.xyz", f"run/species/species_{m['b']}.xyz"}
        listed = set(subprocess.run(["tar", "tzf", f"{d}/run.tgz"], capture_output=True, text=True).stdout.split("\n"))
        subprocess.run(["tar", "xzf", f"{d}/run.tgz", "-C", str(dest), *sorted(names & listed)], check=True)
        for m in ms:
            m["root"] = str(dest)


def one(args):
    m, profile, T, ss = args
    import mepd
    from qcdata import Structure

    import mepd.discovery.free_energy as fe
    from mepd.nodes.node import StructureNode

    out = {k: m.get(k) for k in ("run", "label", "e", "corr", "open", "window", "verified")}
    try:
        root = Path(m["root"])
        fp = root / f"run/ts/{m['label']}.xyz"
        st = Structure.open(str(fp))
        species = tuple(Structure.open(str(root / f"run/species/species_{k}.xyz")) for k in (m["a"], m["b"]))
        ends = fe.step_ends({"ts": str(fp), "irc": str(root / f"run/ts/{m['label']}_irc.xyz")}, species)
        engine = mepd.engine_from_profile(profile)
        free = fe.FreeEnergies(engine, T, ss)
        parts = free.parts(st, True, ends)
        out["molecules"] = [len(p[0]) for p in parts]
        sym = [str(s) for s in st.symbols]
        xyz = np.asarray(st.geometry, dtype=float).reshape(-1, 3)
        p0 = parts[0]
        node = StructureNode(structure=Structure(symbols=[sym[k] for k in p0[0]], geometry=xyz[p0[0]], charge=p0[1],
                                                 multiplicity=p0[2]))
        g = np.asarray(engine.compute_gradients([node])[0], dtype=float).reshape(-1, 3)
        out["fmax_eV_A"] = float(np.linalg.norm(g, axis=1).max()) * fe.EV_PER_HARTREE * BOHR
        out["lowest_cm"] = sorted(mepd.frequencies(node, engine).wavenumbers_cm.tolist())[:3]
        def fmax(atoms):
            nd = StructureNode(structure=Structure(symbols=[sym[k] for k in atoms], geometry=xyz[atoms],
                                                   charge=st.charge, multiplicity=st.multiplicity))
            gg = np.asarray(engine.compute_gradients([nd])[0], dtype=float).reshape(-1, 3)
            return float(np.linalg.norm(gg, axis=1).max()) * fe.EV_PER_HARTREE * BOHR

        try:
            out["fmax_whole_eV_A"] = fmax(list(range(len(sym))))
        except Exception:
            out["fmax_whole_eV_A"] = None
        rec = m.get("reacting_atoms")
        out["recorded_reacting"] = len(rec) if rec else None
        if rec and len(rec) < len(sym) and int(st.charge) == 0 and int(st.multiplicity) == 1:
            try:
                out["fmax_recorded_reacting_eV_A"] = fmax(list(rec))
            except Exception:
                out["fmax_recorded_reacting_eV_A"] = None
        ex = [np.asarray(e.geometry, dtype=float).reshape(-1, 3) / BOHR for e in ends]
        b0, b1 = (fe.bonds(sym, e) for e in ex)
        bts = fe.bonds(sym, xyz / BOHR)
        inside = set(rec) if rec else set(range(len(sym)))
        out["perception"] = {"ends_from": "irc" if (root / f"run/ts/{m['label']}_irc.xyz").exists() else
                             "ends file" if (root / f"run/ts/{m['label']}_ends.xyz").exists() else "species",
                             "both_ends_missing_at_ts": len((b0 & b1) - bts), "ts_in_neither_end": len(bts - (b0 | b1)),
                             "changed_across_recorded_spectators": sum(1 for i, j in b0 ^ b1
                                                                       if (i in inside) != (j in inside)),
                             "ts_contacts_across_recorded_spectators": sum(1 for i, j in bts
                                                                           if (i in inside) != (j in inside))}
        on, ef = None, root / f"run/ts/{m['label']}_ends.xyz"
        if ef.exists():
            fr = Structure.open_multi(str(ef))
            e2 = [np.asarray(k.geometry, dtype=float).reshape(-1, 3) / BOHR for k in (fr[0], fr[-1])]
            cut = fe.ts_groups(sym, e2[0], e2)
            if cut and len(cut) > 1 and fmax(cut[0]) <= 2e-3:
                on = list(cut[0])
        if on is None and (out["fmax_whole_eV_A"] or 9) <= 2e-3:
            on = list(range(len(sym)))
        out["stationary_on"] = None if on is None else "whole" if len(on) == len(sym) else on
        saddle = fe.FreeEnergies._saddle
        fe.FreeEnergies._saddle = lambda self, node: (node, 0.0)
        t = time.time()
        out["given"] = free._correction(st, True, ends, m["e"])
        out["given_s"] = time.time() - t
        fe.FreeEnergies._saddle = saddle
        said = st if on is None else st.model_copy(update={"extras": {"xyz_comments": [
            fe.optimized_on_comment(None if len(on) == len(sym) else on)]}})
        out["counted_molecules"] = [len(p[0]) for p in fe.FreeEnergies(None, T, ss).parts(said, True, ends)]
        for name, which in (("converged", st), ("counted", said)):
            free = fe.FreeEnergies(mepd.engine_from_profile(profile), T, ss)
            t = time.time()
            out[name] = free._correction(which, True, ends, m["e"])
            out[name + "_s"] = time.time() - t
            out[name + "_notes"] = list(free.notes)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spread", required=True)
    ap.add_argument("--min-spread", type=float, default=0.01)
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import multiprocessing as mp

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = [g for g in json.loads(Path(a.spread).read_text())["TSs"] if g["n_runs"] >= 2 and
              (g["spread_kcal"] >= a.min_spread or (a.window and g["in_window"]))]
    members = [m for g in groups for m in g["members"]]
    extract(members, out / "files")
    jobs, runs = [], {}
    for g in groups:
        for m in g["members"]:
            if m["dir"] not in runs:
                summary = json.loads((Path(m["dir"]) / "summary.json").read_text())
                runs[m["dir"]] = (str(Path(m["dir"]).parents[1] / "inputs" / Path(summary["inputs"]).name),
                                  summary["steering"]["standard_state"])
            jobs.append((m, runs[m["dir"]][0], g["temperature"], runs[m["dir"]][1]))
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    k = 0
    for g in groups:
        g["members"] = rows[k:k + len(g["members"])]
        k += len(g["members"])
        ok = [r for r in g["members"] if "error" not in r]
        g["errors"] = len(g["members"]) - len(ok)
        for name in ("given", "converged", "counted"):
            v = [r[name] for r in ok]
            g[f"spread_{name}"] = (max(v) - min(v)) if v else None
        g["record_reproduced"] = max((abs(r["given"] - r["corr"]) for r in ok), default=None)
        g["molecule_counts"] = sorted({tuple(r["molecules"]) for r in ok})
        g["moved_by_converging"] = max((abs(r["converged"] - r["given"]) for r in ok), default=None)
        off = [r for r in ok if r["fmax_eV_A"] > 2e-3]
        g["class"] = ("no member computed" if not ok else
                      "counted as different molecules" if len(g["molecule_counts"]) > 1 else
                      "Hessian off a stationary point" if off else "converged TS")
    (out / "recompute.json").write_text(json.dumps({"min_spread": a.min_spread, "window": a.window,
                                                    "groups": groups}, indent=1))
    lines = ["| system | what | E | n | open | in window | class | recorded spread | given | converged | counted | "
             "record reproduced to | molecules | lowest mode cm-1 (range) | fmax eV/A (max) | "
             "moved by converging (max) |", "|" + "---|" * 16]

    def f(x):
        return "-" if x is None else f"{x:.3f}"
    for g in sorted(groups, key=lambda g: -g["spread_kcal"]):
        ok = [r for r in g["members"] if "error" not in r]
        low = [r["lowest_cm"][1] for r in ok if len(r["lowest_cm"]) > 1]
        lines.append(f"| {g['system']} | {g['what']} | {g['energy']:.6f} | {g['n']}"
                     f"{' (%d failed)' % g['errors'] if g['errors'] else ''} | {g['open_in']} | {g['in_window']} | "
                     f"{g['class']} | {g['spread_kcal']:.3f} | "
                     f"{f(g['spread_given'])} | {f(g['spread_converged'])} | {f(g['spread_counted'])} | "
                     f"{f(g['record_reproduced'])} | {' / '.join('+'.join(map(str, c)) for c in g['molecule_counts'])} | "
                     f"{(f'{min(low):.0f}..{max(low):.0f}') if low else '-'} | "
                     f"{max((r['fmax_eV_A'] for r in ok), default=0):.5f} | {f(g['moved_by_converging'])} |")
    head = [f"{len(groups)} groups, {len(rows)} TSs; errors: {sum(1 for r in rows if 'error' in r)}.", ""]
    for cls in sorted({g["class"] for g in groups}):
        gs = [g for g in groups if g["class"] == cls]
        for name in ("given", "converged", "counted"):
            v = [g[f"spread_{name}"] for g in gs if g[f"spread_{name}"] is not None]
            head.append(f"- {cls}: {len(gs)} groups (open in a run: {sum(1 for g in gs if g['open_in'])}, in a deciding "
                        f"window: {sum(1 for g in gs if g['in_window'])}); spread {name}: max {max(v, default=0):.3f}, "
                        f">= 0.05 in {sum(1 for x in v if x >= 0.05)}, >= 0.01 in {sum(1 for x in v if x >= 0.01)}")
    ok = [r for r in rows if "error" not in r]
    head.append(f"- seconds per TS (mean): given {np.mean([r['given_s'] for r in ok]):.2f}, converged "
                f"{np.mean([r['converged_s'] for r in ok]):.2f}, counted {np.mean([r['counted_s'] for r in ok]):.2f}")
    head.append(f"- stationary on: a subsystem {sum(1 for r in ok if isinstance(r['stationary_on'], list))}, the "
                f"whole box {sum(1 for r in ok if r['stationary_on'] == 'whole')}, neither "
                f"{sum(1 for r in ok if r['stationary_on'] is None)}; counted otherwise than that before: "
                f"{sum(1 for r in ok if r['counted_molecules'] != r['molecules'])}")
    cls = {id(r): g["class"] for g in groups for r in g["members"]}
    what = {id(r): (g["system"], g["what"]) for g in groups for r in g["members"]}
    moved = sorted((r for r in ok if abs(r["counted"] - r["converged"]) > 0.05),
                   key=lambda r: -abs(r["counted"] - r["converged"]))
    mlines = [f"{len(moved)} of {len(ok)} TSs move by more than 0.05 kcal/mol when counted as what they are a "
              "stationary point of (counted - converged); by the class of their group: " +
              ", ".join(f"{c}: {sum(1 for r in moved if cls[id(r)] == c)}" for c in sorted(set(cls.values()))) + ".",
              "", "| system | step | run | TS | class | counted before | counted now | converged | counted | change | "
              "open | in window |", "|" + "---|" * 12]
    for r in moved:
        mlines.append(f"| {what[id(r)][0]} | {what[id(r)][1]} | {r['run']} | {r['label']} | {cls[id(r)]} | "
                      f"{'+'.join(map(str, r['molecules']))} | {'+'.join(map(str, r['counted_molecules']))} | "
                      f"{r['converged']:.3f} | {r['counted']:.3f} | {r['counted'] - r['converged']:+.3f} | "
                      f"{'yes' if r['open'] else ''} | {'yes' if r['window'] else ''} |")
    (out / "moved.md").write_text("\n".join(mlines) + "\n")
    notes = defaultdict(int)
    for r in ok:
        for n in r["converged_notes"]:
            notes[n] += 1
    head += [f"- note ({k} TSs): {n}" for n, k in notes.items()] + [""]
    (out / "recompute.md").write_text("\n".join(head + lines) + "\n")
    print("\n".join(head + lines))


if __name__ == "__main__":
    main()
