"""A 2D drawing of a structure straight from its geometry: atoms joined where
they are close enough to be bonded, no bond orders, no charges. For species
no Lewis structure fits (ylides, hypervalent atoms, metals, odd radicals:
"unusual bonding"), which have no SMILES to draw from; what it shows is the
molecular graph the geometry has."""
from __future__ import annotations

from typing import Optional


def depict_xyz_svg(xyz: str, width: int = 220, height: int = 160, charge: int = 0) -> Optional[str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDetermineBonds
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromXYZBlock(_with_count(xyz))
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        rdDetermineBonds.DetermineConnectivity(mol, charge=charge)
        # Hydrogens on carbon go (as in a skeletal drawing); the rest stay, so an OH, NH or PH reads.
        keep = Chem.RWMol(mol)
        for atom in sorted(mol.GetAtoms(), key=lambda a: -a.GetIdx()):
            if atom.GetSymbol() == "H" and atom.GetDegree() == 1 and atom.GetNeighbors()[0].GetSymbol() == "C":
                keep.RemoveAtom(atom.GetIdx())
        mol = keep.GetMol()
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)
        mol.UpdatePropertyCache(strict=False)
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2
        opts.padding = 0.12
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    except Exception:
        return None


def _with_count(xyz: str) -> str:
    """An xyz block as RDKit reads it (count line, comment line, atoms)."""
    lines = [ln for ln in xyz.strip().splitlines()]
    if lines and lines[0].strip().isdigit():
        n = int(lines[0])
        body = lines[2:2 + n]
    else:
        body = [ln for ln in lines if len(ln.split()) >= 4]
    return f"{len(body)}\n\n" + "\n".join(body) + "\n"


def formula(xyz: str) -> str:
    from collections import Counter

    lines = _with_count(xyz).splitlines()[2:]
    c = Counter(ln.split()[0] for ln in lines if ln.strip())
    order = ([k for k in ("C", "H") if k in c] if "C" in c else []) + sorted(k for k in c if k not in ("C", "H") or "C" not in c)
    return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in order)
