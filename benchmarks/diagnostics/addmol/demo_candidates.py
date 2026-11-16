"""The candidate file of the team41 core demonstration: L-proline for the aldol query, with the relay
barriers the testbed measured entered as `est` steps.

    demo_candidates.py QUERY ENOL_REPORT CC_REPORT OUT [AMOUNT]

QUERY: the query file of the adopted acetone + acetaldehyde network (mepd addmol query).
ENOL_REPORT, CC_REPORT: report.json of two recorded `mepd channels --explore-mechanisms` runs with proline on
both sides of a parent step (acetone -> enol; enol + acetaldehyde -> aldol). From each, the lowest channel in
which the added molecule's bonds change between the two ends (it takes part and is returned) is read:
`barrier_from_lowest_start_conformer_kcal`, an electronic g-xTB barrier from the lowest conformer of the
complex of all the molecules of that run.

These numbers are inputs taken from those records, not results of mepd addmol. They are entered as steps
from the complex of the reacting molecule with proline (acetone.added -> enol.added; enol.added + acetaldehyde
-> aldol.added): the other molecule that stood beside in the record is not in the reference, and the barrier
has no free-energy term. Which aldol enantiomer the record's C-C step makes is not read: the step is entered
for both.
"""
from __future__ import annotations

import json
import sys


def lowest_with_molecule(report: dict) -> dict:
    # kind "channels": a one-step channel between the two ends asked for (not a multistep route, not an exit
    # to other products)
    took_part = [c for c in report["channels"] if c.get("kind") == "channels" and c.get("added_in_end_to_end_changes")
                 and c.get("barrier_from_lowest_start_conformer_kcal") is not None]
    return min(took_part, key=lambda c: c["barrier_from_lowest_start_conformer_kcal"])


def main(query_fp, enol_fp, cc_fp, out_fp, amount="0.1"):
    from mepd import addmol

    q = json.load(open(query_fp))
    one = lambda text: addmol.match(q["species"], text)[0]   # noqa: E731
    acetone, enol, ald = one("CC(C)=O"), one("C=C(C)O"), one("CC=O")
    aldols = q["target"]["species"]
    c = addmol.ADDED_ID
    cx = lambda x: addmol.complex_id(x, c)                    # noqa: E731

    def parent(reactants, products):
        """The parent step between these molecules (either way round; a molecule on both sides aside)."""
        for st in q["steps"]:
            r, p = set(st["reactants"]) - set(st["shuttles"]), set(st["products"]) - set(st["shuttles"])
            if {frozenset(r), frozenset(p)} == {frozenset(reactants), frozenset(products)}:
                return st["id"]
        return None

    level = "g-xTB (axtb), FSM-TS + IRC, electronic, from the lowest start conformer of the complex"
    steps = []
    ch = lowest_with_molecule(json.load(open(enol_fp)))
    steps.append({"id": "relay-enol", "label": "acetone -> enol, a proton relayed through the added molecule",
                  "reactants": [cx(acetone)], "products": [cx(enol)], "parent_step": parent([acetone], [enol]),
                  "barrier": {"electronic": ch["barrier_from_lowest_start_conformer_kcal"]}, "status": "est",
                  "layer": "given", "level": level,
                  "source": f"team41 testbed, channels/ald-enol-pro-explore20 report.json, {ch['name']} "
                            f"(bonds of the added molecule that change: {', '.join(ch['added_in_end_to_end_changes'])})"})
    ch = lowest_with_molecule(json.load(open(cc_fp)))
    for k, aldol in enumerate(aldols):
        steps.append({"id": f"relay-cc-{k}", "label": "enol + acetaldehyde -> aldol, protons relayed through the added molecule",
                      "reactants": [cx(enol), ald], "products": [cx(aldol)], "parent_step": parent([enol, ald], [aldol]),
                      "barrier": {"electronic": ch["barrier_from_lowest_start_conformer_kcal"]}, "status": "est",
                      "layer": "given", "level": level,
                      "source": f"team41 testbed, channels/ald-cc-pro-explore20 report.json, {ch['name']} "
                                f"(bonds of the added molecule that change: {', '.join(ch['added_in_end_to_end_changes'])})"})
    cands = [{"name": "L-proline", "smiles": "OC(=O)[C@@H]1CCCN1", "amount": float(amount),
              "estimates": {"new_steps": steps}},
             {"name": "L-proline-bystander-only", "smiles": "OC(=O)[C@@H]1CCCN1", "amount": float(amount)}]
    json.dump(cands, open(out_fp, "w"), indent=1)
    for s in steps:
        print(s["id"], s["barrier"]["electronic"], s["parent_step"], s["source"])


if __name__ == "__main__":
    main(*sys.argv[1:])
