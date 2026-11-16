"""The recorded whole-box TSs that got a subsystem TS offline (paired.py): is a molecule that takes no part in
contact with the reacting atoms, and how far apart are the two copies?

    python benchmarks/diagnostics/thermo/contact.py PAIRED.json RUNS_ROOT OUT_DIR

Per accepted pair of PAIRED.json, from the run's own TS file and IRC: the reacting atoms (network_expansion's
reading of the two IRC ends), the other molecules, and molecules._neighbours (a spectator atom within CONTACT x
the covalent radii of a reacting atom: what makes the molecule kinetics read the whole-box TS as A + S -> B + S).
The gap is the whole-box TS's free-energy height (counted as one molecule) minus its subsystem TS's (the
bystander apart), kcal/mol at the run's temperature. Geometry and recorded numbers only; no kinetics is run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(paired, root, out_dir):
    from qcdata import Structure

    import mepd.discovery.free_energy as fe
    from mepd.discovery import molecules as M

    rows = [r for r in json.loads(Path(paired).read_text())["rows"]
            if "pairing" in r and "rejected" not in r["pairing"]]
    summaries, out = {}, []
    for r in rows:
        folder = Path(root) / r["run"]
        if r["run"] not in summaries:
            s = json.loads((folder / "run/summary.json").read_text())
            summaries[r["run"]] = {st["label"]: st for st in s["steps"]}
        st = summaries[r["run"]][r["label"]]
        ts = Structure.open(str(folder / st["files"]["ts"]))
        ends = fe.step_ends({"ts": str(folder / st["files"]["ts"]), "irc": str(folder / st["files"]["irc"])}, ())
        parts = fe.FreeEnergies(None, 298.15).parts(ends[0].model_copy(update={"extras": {}}), True, ends)
        react, spectators = list(parts[0][0]), [list(p[0]) for p in parts[1:]]
        near = M._neighbours(ts, react, spectators)
        out.append({"run": r["run"], "label": r["label"], "open": r["open"], "contact": bool(near),
                    "spectators": len(spectators), "in_contact": len(near),
                    "gap_kcal": r["height_counted"] - r["pairing"]["height"]})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "contact.json").write_text(json.dumps(out, indent=1))
    lines = [f"{len(out)} pairs (whole-box TS, its subsystem TS). CONTACT = {M.CONTACT} x the covalent radii.", "",
             "| whole-box TS | pairs | open in the record | gap: min | 25 % | median | 75 % | max |", "|" + "---|" * 8]
    for name, sel in (("no spectator in contact (one TS with its `_sub`: counted through it)", False),
                      ("a spectator in contact (A + S -> B + S: its own reaction)", True)):
        g = np.array([x["gap_kcal"] for x in out if x["contact"] is sel])
        q = np.percentile(g, [0, 25, 50, 75, 100]) if len(g) else [float("nan")] * 5
        lines.append(f"| {name} | {len(g)} | {sum(1 for x in out if x['contact'] is sel and x['open'])} | " +
                     " | ".join(f"{v:.2f}" for v in q) + " |")
    by = {}
    for x in out:
        k = (x["run"].split("-lin2")[0], x["contact"])
        by[k] = by.get(k, 0) + 1
    lines += ["", "By system (no contact / contact): " + "; ".join(
        f"{sy} {by.get((sy, False), 0)} / {by.get((sy, True), 0)}" for sy in sorted({k[0] for k in by}))]
    (out_dir / "contact.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
