"""Synthetic result files for the Add a molecule page of mepd web: states the demonstration files lack
(a strong catalyst, an inhibitor, a consumed reagent, a molecule that does nothing, a failed layer, kinetics that
cannot be built).

    python synthetic_results.py QUERY.json OUTDIR      # a query of the acetone + acetaldehyde demonstration

Every number that goes in is invented here to exercise the page; the kinetics on top of them are mepd.addmol's
own. They say nothing about chemistry: every candidate is called "synthetic: ..." and every estimate's
`source` says so.
"""
import json
import sys
from pathlib import Path

import numpy as np

from mepd import addmol

SOURCE = "synthetic (invented to exercise the page; not a calculation)"
SMILES = {"ACETONE": "CC(C)=O", "ACETALDEHYDE": "CC=O", "ENOL": "C=C(C)O", "VINYL": "C=CO", "ALDOL": "CC(=O)C[C@@H](C)O"}


def find(q, a, b):
    """The id of a step with frames between the species sets `a` and `b`, molecules on both sides apart; None."""
    for st in q["steps"]:
        left = sorted(x for x in st["reactants"] if x not in st.get("shuttles", []))
        right = sorted(x for x in st["products"] if x not in st.get("shuttles", []))
        if st.get("geometry") and sorted([left, right]) == sorted([sorted(a), sorted(b)]):
            return st["id"]
    return None


def e(free, spread, electronic=None):
    return {"electronic": free - 10.0 if electronic is None else electronic, "free": free, "spread": spread}


def tag(entries, status="est", layer="given", level="synthetic"):
    return [{**x, "status": x.get("status", status), "layer": layer, "level": level, "source": SOURCE} for x in entries]


def beside(q, step, shift=4.0):
    """The step's frames with a water-sized placeholder put `shift` Angstrom off the centroid (for the 3D view)."""
    g = next(st for st in q["steps"] if st["id"] == step)["geometry"]
    mol = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    out = {}
    for key, short in (("reactant", "r"), ("ts", "ts"), ("product", "p")):
        xyz = np.array(g[key])
        far = xyz.mean(axis=0) + np.array([0.0, 0.0, np.abs(xyz - xyz.mean(axis=0)).max() + shift])
        out[short] = np.vstack([xyz, mol + far]).tolist()
    return {"complexes": out, "molecule_symbols": ["O", "H", "H"]}


def main(query: Path, out: Path) -> None:
    q = json.loads(query.read_text())
    ids = {k: addmol.match(q["species"], v)[0] for k, v in SMILES.items()}
    ACETONE, ACETALDEHYDE, ENOL, ALDOL = ids["ACETONE"], ids["ACETALDEHYDE"], ids["ENOL"], ids["ALDOL"]
    ENOLISATION = find(q, [ACETONE], [ENOL])
    CC = find(q, [ALDOL], [ENOL, ACETALDEHYDE])
    BOTH_ENOLS = find(q, [ACETONE, ACETALDEHYDE], [ENOL, ids["VINYL"]])
    assert ENOLISATION and CC and BOTH_ENOLS, "this query is not the aldol demonstration's"
    out.mkdir(parents=True, exist_ok=True)
    none = {"species": [], "steps": [], "new_species": [], "new_steps": []}
    done = [{"name": "given", "status": "done", "what": "synthetic estimates", "entries": 0}]
    cases = {}
    cases["synthetic: strong catalyst"] = ("CC(=O)O", 0.1, {
        **none,
        "species": tag([{"species": ACETONE, "binding": e(-2.0, 0.4), "status": "verified"},
                        {"species": ENOL, "binding": e(-2.5, 0.5), "status": "verified"},
                        {"species": ALDOL, "binding": e(-1.0, 0.6)}]),
        "steps": tag([{"step": BOTH_ENOLS, "binding": {"r": e(-2.0, 0.4), "ts": e(-9.0, 1.1), "p": e(-2.5, 0.5)},
                       **beside(q, BOTH_ENOLS)}], layer="bystander"),
        "new_steps": tag([
            {"id": "syn-relay-enol", "label": "acetone -> enol, a proton relayed through the added molecule",
             "reactants": [f"{ACETONE}.added"], "products": [f"{ENOL}.added"], "parent_step": ENOLISATION,
             "barrier": e(24.0, 0.8), "spread": 0.8, "status": "verified"},
            {"id": "syn-relay-cc", "label": "enol + acetaldehyde -> aldol beside the added molecule",
             "reactants": [f"{ENOL}.added", ACETALDEHYDE], "products": [f"{ALDOL}.added"], "parent_step": CC,
             "barrier": e(16.0, 2.1), "spread": 2.1}])}, "done", done, [])
    cases["synthetic: inhibitor"] = ("CN(C)C", 2.0, {
        **none, "species": tag([{"species": ACETONE, "binding": e(-9.0, 1.2)},
                                {"species": ACETALDEHYDE, "binding": e(-8.0, 1.5)}], layer="bystander")},
        "done", done, [])
    cases["synthetic: consumed reagent"] = ("NO", 0.5, {
        **none,
        "new_species": tag([{"id": "syn-adduct", "name": "acetaldehyde adduct of the added molecule",
                             "formed_from": [ACETALDEHYDE, "added"], "released": [], "energy": e(-14.0, 1.0)}]),
        "new_steps": tag([{"id": "syn-addition", "label": "acetaldehyde + added molecule -> adduct",
                           "reactants": [ACETALDEHYDE, "added"], "products": ["syn-adduct"],
                           "barrier": e(17.0, 1.4), "spread": 1.4}])}, "done", done, [])
    cases["synthetic: no effect"] = ("CCCC", 0.1, {
        **none, "species": tag([{"species": ACETONE, "binding": e(6.0, 0.5, -1.0)}], layer="bystander"),
        "steps": tag([{"step": ENOLISATION, "binding": {"r": e(6.0, 0.5, -1.0), "ts": e(7.0, 0.6, -0.5),
                                                        "p": e(6.5, 0.5, -0.8)}}], layer="bystander")},
        "done", [{"name": "bystander", "status": "done", "done": 2, "total": 2, "level": "synthetic", "wall_s": 20.0,
                  "what": "synthetic estimates"}], [])
    cases["synthetic: failed layer"] = ("C1CC1", 0.1, none, "failed",
                                        [{"name": "bystander", "status": "failed", "done": 0, "total": 115,
                                          "level": q["level"]["key"], "wall_s": 0.4,
                                          "what": "the molecule beside each step's ends and TS, the step's atoms held"}],
                                        [{"layer": "bystander", "why": "the candidate could not be built: "
                                          "RuntimeError: (synthetic) the geometry optimization did not converge"}])
    cases["synthetic: kinetics not built"] = ("CO", 0.1, {
        **none, "new_steps": tag([{"id": "syn-broken", "label": "a step that names a species nobody defined",
                                   "reactants": ["syn-nowhere"], "products": [ALDOL], "barrier": e(10.0, 1.0)}])},
        "done", done, [])
    for name, (smiles, amount, est, status, layers, failed) in cases.items():
        doc = addmol.result(q, {"name": name, "smiles": smiles, "amount": amount}, est, status=status,
                            layers=[{**lay, "entries": sum(len(v) for v in est.values())} if lay["name"] == "given"
                                    else lay for lay in layers], failed=list(failed))
        doc["synthetic"] = SOURCE
        fp = out / ("".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:80] + ".json")
        addmol.write_json(fp, doc)
        t = (doc.get("readout") or {}).get("target") or {}
        print(fp.name, doc["status"], t.get("factor"), ((doc.get("readout") or {}).get("added") or {}).get("fate"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
