"""The candidate file of the team41 core demonstration, second round: L-proline for the aldol query, with the
two relay steps the testbed's channel searches found entered by their TS geometries.

    demo_candidates_ts.py QUERY ENOL_CHANNELS_OUT CC_CHANNELS_OUT OUT [AMOUNT]

QUERY: the query file of the acetone + acetaldehyde run (mepd addmol query on the run folder).
*_CHANNELS_OUT: the output folder of a recorded `mepd channels --explore-mechanisms` run with proline on both
sides of a parent step (acetone -> enol; enol + acetaldehyde -> aldol), with the report.json beside it. From
each, the lowest one-step channel in which the added molecule's bonds change between the two ends: its ts.xyz
and the first and last frame of its irc.xyz.

These geometries are inputs taken from those records, not results of mepd addmol. `mepd addmol test` puts
each TS on the parent's scale as a whole (mepd.discovery.parent.ts_step): no barrier is read from the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BOHR = 0.529177210903


def lowest_with_molecule(report: dict) -> dict:
    took_part = [c for c in report["channels"] if c.get("kind") == "channels" and c.get("added_in_end_to_end_changes")
                 and c.get("barrier_from_lowest_start_conformer_kcal") is not None]
    return min(took_part, key=lambda c: c["barrier_from_lowest_start_conformer_kcal"])


def main(query_fp, enol_out, cc_out, out_fp, amount="0.1"):
    import numpy as np
    from qcdata import Structure

    from mepd import addmol
    from mepd.discovery import molecules as M

    q = json.load(open(query_fp))
    one = lambda text: addmol.match(q["species"], text)[0]   # noqa: E731
    acetone, enol = one("CC(C)=O"), one("C=C(C)O")

    def entry(out_dir, name, label, starts_with, tag):
        out_dir = Path(out_dir)
        ch = lowest_with_molecule(json.load(open(out_dir.parent / "report.json")))
        folder = out_dir / "channels" / ch["name"]
        ts = Structure.open(str(folder / "ts.xyz"))
        irc = Structure.open_multi(str(folder / "irc.xyz"))
        ends = [irc[0], irc[-1]]
        if starts_with not in {key for _a, key in M.molecule_parts(ends[0])}:
            ends.reverse()                       # the reactant end first
        ang = lambda s: (np.asarray(s.geometry, dtype=float).reshape(-1, 3) * BOHR).tolist()   # noqa: E731
        made = sorted(key for _a, key in M.molecule_parts(ends[1]))
        print(name, ch["name"], "report's electronic barrier from the complex (not used):",
              round(ch["barrier_from_lowest_start_conformer_kcal"], 2), "product end:", made)
        return {"id": name, "label": label, "status": "est", "layer": "given",
                "level": "g-xTB (axtb) TS and IRC ends of a channel search; free energy by mepd addmol test",
                "source": f"team41 testbed, channels/{tag}/out/channels/{ch['name']} (ts.xyz, irc.xyz)",
                "ts": {"atoms": [str(x) for x in ts.symbols], "coordinates": ang(ts), "charge": int(ts.charge),
                       "multiplicity": int(ts.multiplicity), "ends": [ang(ends[0]), ang(ends[1])]}}

    steps = [entry(enol_out, "relay-enol", "acetone -> enol, a proton relayed through the added molecule",
                   acetone, "ald-enol-pro-explore20"),
             entry(cc_out, "relay-cc", "enol + acetaldehyde -> aldol, protons relayed through the added molecule",
                   enol, "ald-cc-pro-explore20")]
    json.dump([{"name": "L-proline", "smiles": "OC(=O)[C@@H]1CCCN1", "amount": float(amount),
                "estimates": {"new_steps": steps}}], open(out_fp, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
