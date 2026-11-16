"""Tables from place_verify.py (record.json per step) and place_solvent.py (<label>.json per step) folders.

    python benchmarks/diagnostics/addmol/place_report.py verify DIR [DIR ...]
    python benchmarks/diagnostics/addmol/place_report.py solvent DIR [DIR ...]
"""
import json
import sys
from pathlib import Path


def f(x, n=1):
    return "-" if x is None else f"{x:+.{n}f}"


def verify(dirs):
    print("| system | step | estimate: barrier shift (range over repeats) | verified shift | difference | same sign | "
          "estimate: reaction shift | verified | verification status | verified structures | est. s | verify s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    n = right = within = total = 0
    th_rows = []
    for d in dirs:
        for fp in sorted(Path(d).glob("*/record.json")):
            r = json.loads(fp.read_text())
            e, w = r["estimate"], r.get("with_c") or {}
            eb, vb = e["barrier_shift"]["electronic"], r.get("verified_barrier_shift_kcal")
            er, vr = e["reaction_shift"]["electronic"], r.get("verified_reaction_shift_kcal")
            vs = r.get("verified_structures") or {}
            what = []
            if any(not v["molecule_bonds_kept"] for v in vs.values()):
                what.append("molecule's bonds changed")
            if any(v["step_bonds"] == "other" for v in vs.values()):
                what.append("step's bonds other than either end's")
            ok = w.get("status") == "ok" and w.get("barrier_verified")
            total += 1
            diff = None
            if ok and eb is not None and vb is not None and r["parent"].get("status") == "ok":
                n += 1
                diff = eb - vb
                same = (eb > 0) == (vb > 0) or (abs(eb) < 0.5 and abs(vb) < 0.5)
                right += same
                within += abs(diff) <= 2.0
            print(f"| {Path(d).name} | {r['label']} | {f(eb)} ({f(e['barrier_shift']['spread'])[1:]}) | {f(vb)} | {f(diff)} | "
                  f"{'-' if diff is None else ('yes' if same else 'no')} | {f(er)} | {f(vr)} | "
                  f"{e['status']} / parent {r['parent'].get('status')} / with C {w.get('status')} | {'; '.join(what) or 'as the ends'} | "
                  f"{r['estimate_s']:.0f} | {r.get('with_c_s', 0):.0f} |")
            th = r.get("thermo") or {}
            for fr in ("r", "ts", "p"):
                if f"binding_g_corr_{fr}" in th and e["binding"].get(fr, {}).get("g_corr_kcal") is not None:
                    th_rows.append((Path(d).name, r["label"], fr, e["binding"][fr]["electronic"], e["binding"][fr]["g_corr_kcal"],
                                    th[f"binding_g_corr_{fr}"], w.get("status")))
    print(f"\nverified with both ends as the estimate's: {n} of {total}; same sign {right}; within 2 kcal/mol {within}\n")
    print("| system | step | frame | electronic binding (estimate) | G - E of binding, estimate | G - E, full quasi-RRHO of the relaxed complex | difference |")
    print("|---|---|---|---|---|---|---|")
    ds = []
    for s, lab, fr, b, g, full, st in th_rows:
        ds.append(g - full)
        print(f"| {s} | {lab} | {fr} | {f(b)} | {f(g)} | {f(full)} | {f(g - full)} |")
    if ds:
        print(f"\nG - E estimate minus full: mean {sum(ds) / len(ds):+.2f}, mean absolute {sum(abs(x) for x in ds) / len(ds):.2f}, "
              f"largest {max(ds, key=abs):+.2f} kcal/mol over {len(ds)} frames")


def solvent(dirs):
    print("| system | step | binding r / ts / p: gas | gas poses, solvated single points | solvated throughout | "
          "barrier shift: gas | gas + sp | solvated (range) | reaction shift: gas | gas + sp | solvated | status gas / solv | s gas+sp | s solvated |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        for fp in sorted(Path(d).glob("*.json")):
            if fp.name == "run.json":
                continue
            r = json.loads(fp.read_text())
            other = Path(str(d) + "-gasarm") / fp.name      # the gas arm rerun (place_solvent.py --arms gas)
            if other.exists():
                r.update({k: v for k, v in json.loads(other.read_text()).items() if k.startswith("gas_search")})
            g, s = r["gas_search"], r["solvated_search"]
            fin = g.get("final") or {}

            def b(x):
                return " / ".join(f((x.get("binding") or {}).get(k, {}).get("electronic")) for k in ("r", "ts", "p"))

            def sh(x, key):
                return f((x.get(key) or {}).get("electronic"))
            print(f"| {Path(d).name} | {r['label']} | {b(g)} | {b(fin)} | {b(s)} | {sh(g, 'barrier_shift')} | {sh(fin, 'barrier_shift')} | "
                  f"{sh(s, 'barrier_shift')} ({f(s['barrier_shift']['spread'])[1:]}) | {sh(g, 'reaction_shift')} | "
                  f"{sh(fin, 'reaction_shift')} | {sh(s, 'reaction_shift')} | {g['status']} / {s['status']} | "
                  f"{r['gas_search_s']:.0f} | {r['solvated_search_s']:.0f} |")
        rj = Path(d) / "run.json"
        if rj.exists():
            print("\nforms:", json.dumps(json.loads(rj.read_text())["forms"]))


if __name__ == "__main__":
    {"verify": verify, "solvent": solvent}[sys.argv[1]](sys.argv[2:])
