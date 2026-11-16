"""What one `mepd channels` output holds, read for the "add a molecule" testbed.

    python benchmarks/diagnostics/addmol/read_channels.py CHANNELS_OUT [--added-element N] [--out REPORT.json]

From stats.json, pair_mechanisms.json and the channel folders (channels/, alternate-channels/,
offtarget-exit-channels/): the mechanisms proposed (with their labels, "explored:" and "[catalytic: ...]"
included) and how many path searches each got; per channel its TS energy, its barrier from the lowest
reactant-side structure found (sampled start conformers and the reactant-side IRC ends of every channel),
the bonds that differ between its IRC ends, and the bonds the TS geometry holds that neither end does or
lacks that both have (atoms passed through another molecule and returned). Bonds are read from distances
(covalent radii sum x 1.25). --added-element: the molecule that holds this element in the reactant end is
the added one; each channel says whether its atoms are in the changed bonds (end to end) or only in the
TS's (a relay that gives the molecule back). Reads files only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

H = 627.509474
RCOV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "F": 0.57, "Cl": 1.02, "Br": 1.20, "I": 1.39}


def frames(fp: Path):
    lines = fp.read_text().splitlines()
    out, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        sym = [l.split()[0] for l in lines[i + 2: i + 2 + n]]
        xyz = np.array([[float(v) for v in l.split()[1:4]] for l in lines[i + 2: i + 2 + n]])
        out.append((sym, xyz))
        i += n + 2
    return out


def bonds(sym, xyz, scale=1.25):
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    return {(i, j) for i in range(len(sym)) for j in range(i + 1, len(sym))
            if d[i, j] < scale * (RCOV.get(sym[i], 0.8) + RCOV.get(sym[j], 0.8))}


def component(n, bs, start):
    adj = {i: set() for i in range(n)}
    for i, j in bs:
        adj[i].add(j); adj[j].add(i)
    seen, todo = {start}, [start]
    while todo:
        u = todo.pop()
        for v in adj[u] - seen:
            seen.add(v); todo.append(v)
    return seen


def energies(fp: Path):
    return [float(x) for x in fp.read_text().split()] if fp.is_file() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--added-element", default="N")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = a.out_dir
    stats = json.loads((d / "stats.json").read_text()) if (d / "stats.json").is_file() else {}
    pm = json.loads((d / "pair_mechanisms.json").read_text()) if (d / "pair_mechanisms.json").is_file() else []
    rep = {"out_dir": str(d), "total_seconds": stats.get("total_seconds"), "workers": stats.get("workers"),
           "parallel_workers": stats.get("parallel_workers"), "n_pairs_possible": stats.get("n_pairs_possible"),
           "conformers": {k: (stats.get("conformers") or {}).get(k, {}).get("n_final") for k in ("start", "end")},
           "seconds": {k: stats.get(k) for k in ("pair_atom_mapping_seconds", "msmep_seconds", "ts_discovery_seconds")},
           "search_budget_reached": stats.get("search_budget_reached"),
           "mechanisms": stats.get("path_searches_per_mechanism"), "n_path_searches": stats.get("n_path_searches"),
           "pair_mechanisms": pm}
    lows = energies(d / "conformers" / "start.energies")
    chans, routes = [], []
    for kind in ("channels", "alternate-channels", "offtarget-exit-channels"):
        tops = sorted((d / kind).glob("*")) if (d / kind).is_dir() else []
        dirs = []
        for c in tops:   # a multistep route: one folder per step, with path.txt naming its species
            steps_ = sorted(c.glob("step_*"))
            if steps_:
                routes.append({"kind": kind, "name": c.name,
                               "path": [l.strip() for l in (c / "path.txt").read_text().splitlines()] if (c / "path.txt").is_file() else None})
            dirs += steps_ or [c]
        for c in dirs:
            if not (c / "ts.xyz").is_file() or not (c / "irc.xyz").is_file():
                chans.append({"kind": kind, "name": c.name, "files": sorted(p.name for p in c.iterdir())[:8]})
                continue
            ts = frames(c / "ts.xyz")[0]
            irc = frames(c / "irc.xyz")
            e_ts = energies(c / "ts.energies")
            e_irc = energies(c / "irc.energies")
            sym = ts[0]
            b0, b1, bt = bonds(*irc[0]), bonds(*irc[-1]), bonds(*ts, scale=1.35)
            k = next((i for i, s in enumerate(sym) if s == a.added_element), None)
            # the added molecule: the component of that element in whichever IRC end keeps it smallest
            added = min((component(len(sym), b, k) for b in (b0, b1)), key=len) if k is not None else set()
            changed = sorted(b0 ^ b1)
            only_ts = sorted((bt - (b0 | b1)) | ((b0 & b1) - bt))
            name = lambda ij: f"{sym[ij[0]]}{ij[0]}-{sym[ij[1]]}{ij[1]}"
            touches = lambda bs: sorted({name(b) for b in bs if (b[0] in added) != (b[1] in added)
                                         or (b[0] in added and b[1] in added)})
            if e_irc:
                lows.append(e_irc[0]); lows.append(e_irc[-1])
            chans.append({"kind": kind, "name": c.name if not c.name.startswith("step_") else f"{c.parent.name}/{c.name}",
                          "members": (c / "members.txt").read_text().split() if (c / "members.txt").is_file() else None,
                          "ts_energy": min(e_ts) if e_ts else None, "irc_end_energies": [e_irc[0], e_irc[-1]] if e_irc else None,
                          "broken": [name(b) for b in sorted(b0 - b1)], "formed": [name(b) for b in sorted(b1 - b0)],
                          "ts_only_bond_differences": [name(b) for b in only_ts],
                          "added_molecule_atoms": sorted(added),
                          "added_in_end_to_end_changes": touches(changed),
                          "added_in_ts_only_changes": touches(only_ts)})
    start_low = min(energies(d / "conformers" / "start.energies") or [float("nan")])
    for c in chans:
        if c.get("ts_energy") is not None:
            c["barrier_from_lowest_start_conformer_kcal"] = (c["ts_energy"] - start_low) * H
            c["barrier_from_own_irc_lower_end_side_kcal"] = [(c["ts_energy"] - e) * H for e in c["irc_end_energies"]]
    rep["lowest_start_conformer_hartree"] = start_low
    rep["channels"] = chans
    rep["routes"] = routes
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("channels", "pair_mechanisms")}))
    seen = {}
    for m in pm:
        seen.setdefault(m["mechanism"], []).append(m["pair"])
    for m, pairs in seen.items():
        print(f"  mechanism: {m}  ({len(pairs)} search(es))")
    for r in routes:
        print(f"  {r['kind']}/{r['name']}: {' '.join(r['path'] or [])}")
    for c in chans:
        if "ts_energy" not in c:
            print("  ", c); continue
        print(f"  {c['kind']}/{c['name']} n={len(c['members'] or [])} barrier {c['barrier_from_lowest_start_conformer_kcal']:.1f} "
              f"(from own IRC ends {c['barrier_from_own_irc_lower_end_side_kcal'][0]:.1f} / {c['barrier_from_own_irc_lower_end_side_kcal'][1]:.1f}) "
              f"broken {c['broken']} formed {c['formed']} TS-only {c['ts_only_bond_differences']} "
              f"added: ends {c['added_in_end_to_end_changes']} ts {c['added_in_ts_only_changes']}")


if __name__ == "__main__":
    main()
