"""Seeds for the suite systems that had none (run once; output committed beside this file).

    /home/user/astera/mepd/.venv/bin/python benchmarks/cluster/suite/make_seeds.py

- acetic_acid.xyz: RDKit ETKDG (seed 7) + MMFF, the syn conformer (H-O-C=O dihedral 0, the lower one).
- cp_mbl_start_complex.xyz: the F0007 start complex with methyl acrylate replaced by
  alpha-methylene-gamma-butyrolactone (RDKit + MMFF): the lactone's CH2=C-C(=O) carbons are superposed
  (Kabsch) on the acrylate's CH2=CH-C(=O) carbons, then turned about its C=C axis to the largest closest
  contact with Cp (the ring is bulkier than the ester), so both Diels-Alder systems start from the same
  encounter geometry of the C=C. Cyclopentadiene is F0007's (atoms 0-10).
- isoprene_ma_start_complex.xyz, furan_maleimide_start_complex.xyz: a generic encounter, no reactive atoms
  aligned (so neither regio- nor endo/exo-chemistry is preset): each molecule's lowest of 20 RDKit ETKDG + MMFF
  conformers, the second molecule turned by a random rotation (numpy seed 11) and its centroid put 5.0 A from
  the first's along a random direction, redrawn until the closest contact is >= 2.5 A.
- cb_diester.xyz: dimethyl cis-cyclobut-3-ene-1,2-dicarboxylate, lowest of 20 RDKit ETKDG + MMFF conformers.
The explorer minimizes every seed itself before it starts.
"""
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

HERE = Path(__file__).resolve().parent
F0007 = HERE.parent.parent / "f0007_start_complex.xyz"


def embed(smiles: str, seed: int = 7) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def write(fp: Path, symbols, xyz, comment: str) -> None:
    lines = [str(len(symbols)), comment] + [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, xyz)]
    fp.write_text("\n".join(lines) + "\n")


def kabsch(P, Q):
    """R, t with R @ p + t ~ q (rows of P onto rows of Q)."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cq - R @ cp


def acetic_acid() -> None:
    m = embed("CC(=O)O")
    conf = m.GetConformer()
    # atoms: C0 C1 O2(=) O3(H) ; the hydroxyl H
    h = next(n.GetIdx() for n in m.GetAtomWithIdx(3).GetNeighbors() if n.GetSymbol() == "H")
    rdMolTransforms.SetDihedralDeg(conf, 2, 1, 3, h, 0.0)
    mp = AllChem.MMFFGetMoleculeProperties(m)
    ff = AllChem.MMFFGetMoleculeForceField(m, mp)
    ff.MMFFAddTorsionConstraint(2, 1, 3, h, False, -1.0, 1.0, 100.0)
    ff.Minimize(maxIts=2000)
    xyz = m.GetConformer().GetPositions()
    write(HERE / "acetic_acid.xyz", [a.GetSymbol() for a in m.GetAtoms()], xyz,
          "acetic acid, syn (RDKit ETKDG seed 7 + MMFF; make_seeds.py)")


def cp_mbl() -> None:
    lines = F0007.read_text().split("\n")
    n = int(lines[0])
    sym = [l.split()[0] for l in lines[2:2 + n]]
    X = np.array([[float(v) for v in l.split()[1:4]] for l in lines[2:2 + n]])
    cp_sym, cp_xyz = sym[:11], X[:11]
    acr = X[[11, 12, 13]]                      # acrylate CH2= , =CH-, C(=O)
    m = embed("C=C1CCOC1=O")
    lx = m.GetConformer().GetPositions()
    # lactone atoms: 0 CH2=, 1 ring C=, 5 C(=O)
    R, t = kabsch(lx[[0, 1, 5]], acr)
    lx = lx @ R.T + t
    # the ring is bulkier than the acrylate's ester: turn it about the C=C axis (C=C kept where the
    # acrylate's is) to the angle with the largest closest contact to Cp (F0007's own is 2.55 A)
    axis = (lx[1] - lx[0]) / np.linalg.norm(lx[1] - lx[0])
    contact = lambda Y: np.min(np.linalg.norm(cp_xyz[:, None, :] - Y[None, :, :], axis=2))

    def turned(deg):
        a = np.radians(deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        Rot = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        return (lx - lx[0]) @ Rot.T + lx[0]

    lx = max((turned(d) for d in range(0, 360, 5)), key=contact)
    dmin = contact(lx)
    assert dmin > 2.4, f"clash: {dmin:.2f} A"
    write(HERE / "cp_mbl_start_complex.xyz", cp_sym + [a.GetSymbol() for a in m.GetAtoms()],
          np.vstack([cp_xyz, lx]),
          f"cyclopentadiene (F0007 complex) + alpha-methylene-gamma-butyrolactone on the acrylate's C=C-C(=O); "
          f"closest contact {dmin:.2f} A (make_seeds.py)")


def lowest(smiles: str, n: int = 20):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = AllChem.EmbedMultipleConfs(m, numConfs=n, randomSeed=7)
    res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
    best = min(range(len(ids)), key=lambda i: res[i][1])
    return [a.GetSymbol() for a in m.GetAtoms()], m.GetConformer(ids[best]).GetPositions()


def encounter(smiles_a: str, smiles_b: str, name: str, comment: str) -> None:
    sa, xa = lowest(smiles_a)
    sb, xb = lowest(smiles_b)
    rng = np.random.default_rng(11)
    xa = xa - xa.mean(0)
    for _ in range(1000):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        xb2 = (xb - xb.mean(0)) @ R.T + 5.0 * d
        dmin = np.min(np.linalg.norm(xa[:, None, :] - xb2[None, :, :], axis=2))
        if dmin >= 2.5:
            break
    write(HERE / name, sa + sb, np.vstack([xa, xb2]), f"{comment}; centroids 5.0 A, closest contact {dmin:.2f} A "
                                                      "(make_seeds.py)")


def cb_diester() -> None:
    sym, x = lowest("COC(=O)[C@H]1C=C[C@H]1C(=O)OC")
    write(HERE / "cb_diester.xyz", sym, x, "dimethyl cis-cyclobut-3-ene-1,2-dicarboxylate (RDKit + MMFF; make_seeds.py)")


if __name__ == "__main__":
    import sys

    todo = sys.argv[1:] or ["acetic_acid", "cp_mbl", "isoprene_ma", "furan_maleimide", "cb_diester"]
    if "acetic_acid" in todo:
        acetic_acid()
    if "cp_mbl" in todo:
        cp_mbl()
    if "isoprene_ma" in todo:
        encounter("C=CC(=C)C", "C=CC(=O)OC", "isoprene_ma_start_complex.xyz", "isoprene + methyl acrylate")
    if "furan_maleimide" in todo:
        encounter("c1ccoc1", "O=C1C=CC(=O)N1", "furan_maleimide_start_complex.xyz", "furan + maleimide")
    if "cb_diester" in todo:
        cb_diester()
