"""When a `mepd channels --explore-mechanisms` run had its channels through an added molecule.

    PYTHONPATH=<team41/relay code> python benchmarks/diagnostics/addmol/relay_times.py OUT PROF_DIR SMILES_OF_C \
        [--ts-listing FILE] [--json REPORT.json]

OUT: a channels output (any code version); PROF_DIR: profile_channels.py's folder (start.json: the run's start);
SMILES_OF_C: the added molecule. Every TS under OUT/ts with an IRC is read the way searches.json reports it
(`channel_search._SearchTable._step`: IRC ends against the start and the end, bonds that differ, molecules
touched and returned), so runs made before searches.json existed are read the same way. Its time is its IRC
file's modification time minus the run's start (--ts-listing: `ls -l --time-style=+%s ts` saved in the pod,
for outputs unpacked elsewhere). Printed: per mechanism in queue order, its searches, verified TSs that join
start and end through C, their lowest barrier and when the first and the last came; then the time to the
first verified channel through C, to the last TS of the top-ranked relays through C (the mechanisms through C
with the fewest C-H bond changes: a proton passed over C's O-H or N-H), and the run's total.
Reads files only.
"""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("prof", type=Path)
ap.add_argument("molecule")
ap.add_argument("--ts-listing", type=Path)
ap.add_argument("--json", type=Path)
a = ap.parse_args()

from mepd import channel_search as cs
from mepd.atom_mapping import atoms_of_molecule
from mepd.chain import Chain
from mepd.inputs import ChainInputs

out = a.out
t0 = json.loads((a.prof / "start.json").read_text())["t0"]
t1 = json.loads((a.prof / "end.json").read_text())["t1"] if (a.prof / "end.json").is_file() else None
stats = json.loads((out / "stats.json").read_text())
start = list(Chain.from_xyz(out / "conformers" / "start.xyz", ChainInputs(), charge=0, spinmult=1))
end = list(Chain.from_xyz(out / "conformers" / "end.xyz", ChainInputs(), charge=0, spinmult=1))
c_atoms = atoms_of_molecule(start[0].structure, a.molecule)
symbols = [str(s) for s in start[0].structure.symbols]
mtimes = {}
if a.ts_listing:
    for line in a.ts_listing.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[-1].endswith("_irc.xyz"):
            mtimes[parts[-1]] = float(parts[-2])

table = cs._SearchTable.__new__(cs._SearchTable)
table.output, table.start, table.end, table.q, table.m = out, start[0], end[0], 0, 1
table.floor = min(float(n.energy) for n in start)
rows = [r for r in json.loads((out / "pair_mechanisms.json").read_text()) if not r.get("not_in_this_run")]


def c_h_changes(key: str) -> int:
    names = re.sub(r"^explored: ", "", key).split(" [")[0]
    bonds = [b for side in names.split("|") for b in side.strip().split(" ", 1)[1].split(",") if b and b != "none"]
    return sum(1 for b in bonds if sorted(re.sub(r"\d+", "", x) for x in b.split("-")) == ["C", "H"])


mechanisms: dict = {}
for r in rows:
    m = mechanisms.setdefault(r["mechanism"], {"mechanism": r["mechanism"], "searches": 0, "searched": 0, "ts": []})
    m["searches"] += 1
    pair_dir = out / "pairs" / r["pair"]
    m["searched"] += (pair_dir / "tree" / "adj_matrix.txt").exists()
    for fp in sorted((out / "ts").glob(f"ts_{r['pair']}_leaf_*_irc.xyz")):
        label = fp.name[: -len("_irc.xyz")]
        step = table._step(label, r["mechanism"])
        step["t"] = round((mtimes.get(fp.name) or fp.stat().st_mtime) - t0, 1)
        step["through_c"] = bool(step.get("joins_start_and_end") and any(
            mol["returned"] and set(mol["atoms"]) & c_atoms for mol in step.get("molecules") or []))
        # the bonds on C that change: all to an H from N or O?
        on_c = [b for b in (step.get("bonds_broken") or []) + (step.get("bonds_formed") or []) if set(b) & c_atoms]
        step["c_bonds"] = ["-".join(f"{symbols[i]}{i}" for i in b) for b in on_c]
        step["heteroatom_h"] = bool(on_c) and all(
            sorted(symbols[i] for i in b)[0] == "H" and sorted(symbols[i] for i in b)[1] in ("N", "O") for b in on_c)
        m["ts"].append(step)

through = [k for k in mechanisms if k.startswith("explored:") and "[catalytic:" in k
           and any(t["through_c"] for t in mechanisms[k]["ts"]) or (k.startswith("explored:") and "C5H9NO2" in k)]
fewest = min((c_h_changes(k) for k in through), default=None)
top = [k for k in through if c_h_changes(k) == fewest]
print(f"{out}: {len(rows)} searches of {len(mechanisms)} mechanisms; total {stats.get('total_seconds')} s"
      f" (wall from start {round(t1 - t0) if t1 else '?'} s); search budget reached: {stats.get('search_budget_reached')}")
report = {"out": str(out), "total_seconds": stats.get("total_seconds"), "mechanisms": []}
for n, (k, m) in enumerate(mechanisms.items(), start=1):
    good = [t for t in m["ts"] if t["through_c"]]
    line = {"queue_place": n, "mechanism": k, "top_ranked_relay_through_c": k in top, "searches": m["searches"],
            "searched": m["searched"], "ts_with_irc": len(m["ts"]),
            "verified_channels_through_c": len(good),
            "of_them_heteroatom_h": sum(t["heteroatom_h"] for t in good),
            "of_them_the_searched_mechanism": sum(bool(t.get("is_searched_mechanism")) for t in good),
            "lowest_barrier_through_c_kcal": min((t["barrier_kcal"] for t in good), default=None),
            "lowest_barrier_start_to_end_kcal": min((t["barrier_kcal"] for t in m["ts"] if t.get("joins_start_and_end")), default=None),
            "first_through_c_s": min((t["t"] for t in good), default=None),
            "last_ts_s": max((t["t"] for t in m["ts"]), default=None)}
    report["mechanisms"].append(line)
    print(f"{n:3d}. {'*' if k in top else ' '} {k[:92]:92s} searched {m['searched']}/{m['searches']}  TS {len(m['ts']):2d}  "
          f"through C {len(good)} (X-H {line['of_them_heteroatom_h']}, as searched {line['of_them_the_searched_mechanism']})  "
          f"lowest {line['lowest_barrier_through_c_kcal']} / any {line['lowest_barrier_start_to_end_kcal']}  "
          f"first {line['first_through_c_s']} s, last TS {line['last_ts_s']} s")
all_ts = [t for m in mechanisms.values() for t in m["ts"]]
relay = sorted((t for t in all_ts if t["through_c"]), key=lambda t: t["t"])
hetero = [t for t in relay if t["heteroatom_h"]]
report.update(
    first_verified_channel_through_c_s=relay[0]["t"] if relay else None,
    first_verified_heteroatom_h_relay_s=hetero[0]["t"] if hetero else None,
    lowest_barrier_through_c_kcal=min((t["barrier_kcal"] for t in relay), default=None),
    lowest_barrier_heteroatom_h_kcal=min((t["barrier_kcal"] for t in hetero), default=None),
    top_ranked_relays=len(top),
    top_ranked_relays_all_searched=all(mechanisms[k]["searched"] == mechanisms[k]["searches"] for k in top),
    last_ts_of_top_ranked_relays_s=max((t["t"] for k in top for t in mechanisms[k]["ts"]), default=None),
    distinct_barriers_through_c_kcal=sorted({round(t["barrier_kcal"], 1) for t in relay}),
    distinct_c_bond_sets=sorted({" ".join(sorted(re.sub(r"\d+", "", b) for b in t["c_bonds"])) for t in relay}))
print(json.dumps({k: v for k, v in report.items() if k != "mechanisms"}, indent=1))
if a.json:
    a.json.write_text(json.dumps(report, indent=1))
