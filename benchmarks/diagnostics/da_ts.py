"""Diels-Alder TS guesses from an adduct: stretch the two new sigma bonds, then refine with collect_ts.py.

    python da_ts.py OUTDIR NAME ADDUCT_SMILES REACTANT_SMILES [REACTANT_SMILES] [--seeds N] [--mirror]

The adduct (stereo SMILES) is embedded with RDKit (N seeds, MMFF), the two bonds whose removal splits it into the two
reactants' formulas are found, and the dienophile part is moved away along the line between the two bonds' midpoints
until both are 2.25 A (a synchronous guess). Each guess is written as OUTDIR/NAME_<k>.xyz (and its mirror image with
--mirror); collect_ts.py then optimizes it at g-xTB with an IRC. No reaction template enters the explorer: this is a
diagnostic for the ceiling only.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np


def formula(atoms) -> str:
    from collections import Counter

    c = Counter(atoms)
    return "".join(f"{k}{c[k]}" for k in sorted(c))


def main(argv):
    seeds = 4
    mirror = "--mirror" in argv
    argv = [a for a in argv if a != "--mirror"]
    if "--seeds" in argv:
        i = argv.index("--seeds")
        seeds = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    out, name, adduct, *reactants = argv
    from rdkit import Chem
    from rdkit.Chem import AllChem

    want = sorted(formula([a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(r)).GetAtoms()]) for r in reactants)
    if len(reactants) == 1:          # intramolecular: not handled
        raise SystemExit("two reactants needed")
    m = Chem.AddHs(Chem.MolFromSmiles(adduct))
    cids = AllChem.EmbedMultipleConfs(m, numConfs=seeds, randomSeed=7)
    AllChem.MMFFOptimizeMoleculeConfs(m)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    # the new sigma bonds of a [4+2] adduct join an allylic ring carbon (next to the new ring C=C) to a ring carbon that
    # is neither allylic nor in that C=C; a formula split alone also matched other bond pairs (meta isoprene adduct)
    dbl = {a for b in m.GetBonds() if b.IsInRing() and b.GetBondType() == Chem.BondType.DOUBLE
           for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
    allyl = {n.GetIdx() for a in dbl for n in m.GetAtomWithIdx(a).GetNeighbors()
             if n.IsInRing() and n.GetIdx() not in dbl and n.GetSymbol() == "C"}
    ring_cc = [b for b in m.GetBonds() if b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
               and b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C"
               and len({b.GetBeginAtomIdx(), b.GetEndAtomIdx()} & allyl) == 1
               and not ({b.GetBeginAtomIdx(), b.GetEndAtomIdx()} & dbl)]
    pair = None
    for b1, b2 in itertools.combinations(ring_cc, 2):
        if {b1.GetBeginAtomIdx(), b1.GetEndAtomIdx()} & {b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()}:
            continue
        em = Chem.RWMol(m)
        for b in (b1, b2):
            em.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        frags = Chem.GetMolFrags(em, asMols=False)
        if len(frags) == 2 and sorted(formula([syms[i] for i in f]) for f in frags) == want:
            pair = (b1, b2, frags)
            break
    if pair is None:
        raise SystemExit("no pair of new bonds found")
    b1, b2, frags = pair
    # the dienophile: the fragment holding fewer of the ring atoms (2 of the 6-ring) -- move the smaller-ring part
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in (b1, b2)]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for k, cid in enumerate(cids):
        X = m.GetConformer(cid).GetPositions()
        fa = set(frags[0])
        ends_a = [i if i in fa else j for i, j in bonds]
        ends_b = [j if i in fa else i for i, j in bonds]
        mov = np.array(sorted(frags[1]))
        v = X[ends_b].mean(0) - X[ends_a].mean(0)
        v /= np.linalg.norm(v)
        for _ in range(400):
            d = [np.linalg.norm(X[a] - X[b]) for a, b in zip(ends_a, ends_b)]
            if min(d) >= 2.25:
                break
            X[mov] += 0.01 * v
        for tag, Y in (("", X), ("_mirror", X * np.array([-1.0, 1.0, 1.0]))):
            if tag and not mirror:
                continue
            (out / f"{name}_{k}{tag}.xyz").write_text(
                f"{len(syms)}\n{name} DA TS guess from {adduct} conf {k}{tag}\n"
                + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(syms, Y)))
    print(f"{len(cids)} guesses, new bonds {bonds}")


if __name__ == "__main__":
    main(sys.argv[1:])
