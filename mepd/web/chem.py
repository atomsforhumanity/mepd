"""Structure parsing/description helpers for the web workspace.

Everything here is cheap (no QM): SMILES embedding, xyz parsing, SMILES
perception from a geometry, and 2D depictions for library cards and graph
nodes.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from qcdata import Structure

_XYZ_FIRST_LINE = re.compile(r"^\s*\d+\s*$")


def looks_like_xyz(text: str) -> bool:
    lines = text.strip().splitlines()
    return bool(lines) and bool(_XYZ_FIRST_LINE.match(lines[0]))


def structures_from_xyz_text(
    text: str, charge: Optional[int] = None, multiplicity: Optional[int] = None
) -> list[Structure]:
    """Every frame of a (possibly multi-frame) xyz text. A charge/multiplicity
    given here overrides whatever the xyz comment line carries."""
    text = text.strip() + "\n"
    try:
        frames = Structure.from_xyz_multi(text)
    except Exception:
        frames = [Structure.from_xyz(text)]
    out = []
    for s in frames:
        updates = {}
        if charge is not None:
            updates["charge"] = charge
        if multiplicity is not None:
            updates["multiplicity"] = multiplicity
        out.append(s.model_copy(update=updates) if updates else s)
    return out


def structure_from_smiles(
    smiles: str, charge: Optional[int] = None, multiplicity: Optional[int] = None
) -> Structure:
    # Same embedding path (rdkit, then openbabel for multi-fragment SMILES)
    # the CLI uses for a SMILES --start/--end, so what the user sees here is
    # what a job started with that SMILES would see.
    from mepd.cli import _load_structure_from_smiles_or_xyz

    return _load_structure_from_smiles_or_xyz(smiles.strip(), charge, multiplicity)


# Small molecules, radicals and ions by their everyday names ("O" is water,
# "[OH]" the hydroxyl radical), so a structure a calculation finds is not
# named by its SMILES. Any SMILES spelling; read canonically (charge and
# radicals are part of the SMILES, so OH, OH- and OH+ are different keys).
_COMMON_NAMES = {
    # hydrogen, oxygen
    "[H]": "H atom", "[H][H]": "H2", "[HH]": "H2", "[H+]": "proton", "[H-]": "hydride",
    "O": "water", "[OH]": "hydroxyl", "[OH-]": "hydroxide", "[OH3+]": "hydronium", "[O]": "O atom",
    "O=O": "O2", "[O][O]": "O2", "OO": "H2O2", "[O]O": "hydroperoxyl", "[O-][O]": "superoxide",
    "[O-]O": "hydroperoxide", "[O-2]": "oxide", "[O-]": "oxyl anion",
    # one carbon
    "C": "methane", "[CH3]": "methyl", "[CH2]": "methylene", "[CH]": "methylidyne", "[CH3+]": "methyl cation",
    "[CH3-]": "methyl anion", "CO": "methanol", "C[O]": "methoxy", "[CH2]O": "hydroxymethyl",
    "C[O-]": "methoxide", "C[OH2+]": "protonated methanol", "C=O": "formaldehyde", "[CH]=O": "formyl",
    "C#[O+]": "formyl cation", "C=[OH+]": "protonated formaldehyde", "[CH]O": "hydroxymethylene",
    "[C-]#[O+]": "CO", "O=C=O": "CO2", "O=CO": "formic acid", "O=C[O-]": "formate", "[O]C=O": "formyloxyl",
    "O=[C]O": "hydroxycarbonyl", "OC(O)=O": "carbonic acid", "O=C([O-])O": "bicarbonate",
    "O=C([O-])[O-]": "carbonate", "OCO": "methanediol",
    # nitrogen
    "N": "ammonia", "[NH2]": "amino radical", "[NH4+]": "ammonium", "[NH2-]": "amide anion", "[NH]": "imidogen",
    "N#N": "N2", "NN": "hydrazine", "C#N": "HCN", "[C-]#N": "cyanide", "[C]#N": "cyano radical",
    "[C-]#[NH+]": "HNC", "CN": "methylamine", "C=N": "methanimine", "NC=O": "formamide", "NC(N)=O": "urea",
    "O=NO": "nitrous acid", "[N]=O": "NO", "[O]N=O": "NO2", "O=[N+]([O-])O": "nitric acid",
    "N=O": "nitroxyl", "[O-][N+]#N": "N2O",
    # two and three carbons
    "CC": "ethane", "C[CH2]": "ethyl", "C=C": "ethene", "C#C": "ethyne", "C=[CH]": "vinyl",
    "CCO": "ethanol", "CC=O": "acetaldehyde", "C[C]=O": "acetyl", "C=CO": "vinyl alcohol",
    "CC(=O)O": "acetic acid", "CC(=O)[O-]": "acetate", "O=CC=O": "glyoxal", "OCC=O": "glycolaldehyde",
    "OCCO": "ethylene glycol", "COC": "dimethyl ether", "C1CO1": "ethylene oxide", "C=C=O": "ketene",
    "CC(C)=O": "acetone", "CCC": "propane", "C=CC": "propene", "CC(C)O": "isopropanol", "CCCO": "1-propanol",
    "CC#N": "acetonitrile", "COC=O": "methyl formate", "O=COCO": "hydroxymethyl formate",
    # amino acids and small biomolecules
    "NCC(=O)O": "glycine", "[NH3+]CC(=O)[O-]": "glycine zwitterion", "CC(N)C(=O)O": "alanine",
    "C[C@H](N)C(=O)O": "L-alanine", "OCC(O)CO": "glycerol", "O=C(O)C(=O)O": "oxalic acid",
    "CC(=O)C(=O)O": "pyruvic acid",
    # sulfur, halogens, phosphorus, solvents
    "S": "H2S", "[SH]": "sulfanyl", "O=S=O": "SO2", "O=S(=O)(O)O": "sulfuric acid", "CS(C)=O": "DMSO",
    "F": "HF", "Cl": "HCl", "Br": "HBr", "I": "HI", "[F-]": "fluoride", "[Cl-]": "chloride", "[Br-]": "bromide",
    "[I-]": "iodide", "[Cl]": "Cl atom", "[Br]": "Br atom", "ClCl": "Cl2", "BrBr": "Br2",
    "O=P(O)(O)O": "phosphoric acid", "ClCCl": "dichloromethane", "ClC(Cl)Cl": "chloroform", "CCl": "chloromethane",
    "C1CCOC1": "THF", "c1ccccc1": "benzene", "Cc1ccccc1": "toluene", "c1ccncc1": "pyridine", "Oc1ccccc1": "phenol",
}


def _canonical(smiles: str) -> Optional[str]:
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():   # scoped: smiles_problem reads RDKit's messages
        mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


@lru_cache(maxsize=1)
def _common_names() -> dict:
    return {_canonical(k): v for k, v in _COMMON_NAMES.items()}


def common_name(smiles: Optional[str]) -> Optional[str]:
    """A small molecule's everyday name ("Cl" -> "HCl", "[OH]" -> "hydroxyl"),
    or that of the molecule it is a protonation state of ("[NH3+]CC(=O)O" ->
    "protonated glycine"), or None."""
    if not smiles:
        return None
    try:
        key = _canonical(smiles)
        if not key:
            return None
        return _common_names().get(key) or _protonation_name(key)
    except Exception:
        return None


def _protonation_name(smiles: str) -> Optional[str]:
    """An ion or zwitterion of a named neutral molecule, by that name:
    "protonated X", "deprotonated X" or "X zwitterion" (one proton off or on,
    or charges that cancel). Radicals and anything else: None."""
    from rdkit import Chem, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            return None
        if not any(a.GetFormalCharge() for a in mol.GetAtoms()):
            return None
        charge = Chem.GetFormalCharge(mol)
        neutral = rdMolStandardize.Uncharger().uncharge(Chem.Mol(mol))
    if Chem.GetFormalCharge(neutral) != 0 or any(a.GetFormalCharge() for a in neutral.GetAtoms()):
        return None
    name = _common_names().get(Chem.MolToSmiles(neutral))
    if not name:
        return None
    return {1: f"protonated {name}", -1: f"deprotonated {name}", 0: f"{name} zwitterion"}.get(charge)


def smiles_problem(smiles: str) -> Optional[str]:
    """Why RDKit cannot read `smiles`, in one line (e.g. "syntax error
    around position 5", "unclosed ring"), or None when it can."""
    import contextlib
    import io
    import re

    from rdkit import Chem, rdBase

    buf = io.StringIO()
    rdBase.LogToPythonStderr()
    with contextlib.redirect_stderr(buf):
        mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return None
    lines = [re.sub(r"^\[[\d:]+\]\s*", "", ln).replace("SMILES Parse Error: ", "").strip()
             for ln in buf.getvalue().splitlines() if ln.strip()]
    if not lines:
        return "not a valid SMILES"
    reason = re.sub(r"\s+(while parsing|for input):.*$", "", lines[0])
    if m := re.match(r"Explicit valence for atom # (\d+) (\w+), (\d+), is greater than permitted", reason):
        reason = f"atom {int(m.group(1)) + 1} ({m.group(2)}) has {m.group(3)} bonds, more than it can"
    elif reason.startswith("Can't kekulize"):
        reason = "an aromatic ring that cannot be drawn with alternating double bonds (check the lowercase atoms)"
    where = next((m.group(1) for ln in lines if (m := re.search(r"around position (\d+)", ln))), None)
    return f"{reason} around position {where}" if where else reason


def _atom_smiles(structure: Structure) -> str:
    """A lone atom or monatomic ion, with no hydrogens added ("[H]", not
    "[HH]", which is H2; "[Cl-]", "[O]")."""
    q = int(getattr(structure, "charge", 0) or 0)
    sign = "" if q == 0 else ("+" if q > 0 else "-") + (str(abs(q)) if abs(q) > 1 else "")
    return f"[{structure.symbols[0]}{sign}]"


def perceive_smiles(structure: Structure) -> Optional[str]:
    if len(structure.symbols) == 1:   # RDKit gives a lone H an implicit H ("[HH]": H2)
        return _atom_smiles(structure)
    import qcinf

    for backend in ("rdkit", "openbabel"):
        try:
            return qcinf.structure_to_smiles(structure, backend=backend)
        except Exception:
            continue
    return None


def canonical_key(smiles: str) -> str:
    """Order-independent identity of a (possibly multi-fragment) SMILES, so
    'C=C.O.N' and 'C=C.N.O' compare equal."""
    try:
        from rdkit import Chem

        frags = []
        for part in smiles.split("."):
            mol = Chem.MolFromSmiles(part)
            frags.append(Chem.MolToSmiles(mol) if mol is not None else part)
        return ".".join(sorted(frags))
    except Exception:
        return ".".join(sorted(smiles.split(".")))


def formula(structure: Structure) -> str:
    try:
        return structure.formula
    except Exception:
        counts: dict[str, int] = {}
        for sym in structure.symbols:
            counts[sym] = counts.get(sym, 0) + 1
        return "".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(counts.items()))


@lru_cache(maxsize=512)
def depict_svg(smiles: str, width: int = 220, height: int = 160) -> Optional[str]:
    """2D depiction of a SMILES (transparent background, theme-neutral
    colours so it reads on light and dark cards)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        # the explorer joins forms it counts as one species ("A|B": the cis and trans of a carbene); draw the first
        smiles = str(smiles).split("|")[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2
        opts.padding = 0.12
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    except Exception:
        return None


def _molecules(smiles: str) -> list[str]:
    """A species' molecules, each canonical (a SMILES the explorer joined with "|" read by its first form)."""
    return [_canonical(m) or m for m in str(smiles).split("|")[0].split(".") if m]


_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def molecule_formula(mol) -> str:
    """A molecule's formula as chemists write it: Hill order (CH₄, H₂O, H₂O₂), but a hydride of boron or
    the nitrogen group with its element first (NH₃, PH₃, N₂H₄)."""
    from collections import Counter

    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    text = CalcMolFormula(mol)
    heavy = Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != "H")
    if len(heavy) == 1 and next(iter(heavy)) in ("B", "N", "P", "As", "Si") and text.startswith("H"):
        m = re.match(r"(H\d*)([A-Z][a-z]?\d*)(.*)", text)
        if m:
            text = m.group(2) + m.group(1) + m.group(3)
    return text.translate(_SUBSCRIPT)


def changed_svg(mols: list, counts: list[int], unchanged: list[tuple[str, int]], width: int, height: int,
                draw=None) -> Optional[str]:
    """Distinct molecules side by side, each with its count when there are several ("3×"), and a muted
    line under them naming the rest ("+ unchanged: 3 CH₄, 2 NH₃, H₂O"). `draw(drawer, mols)` draws them
    (default RDKit's DrawMolecules, prepared as depict_svg's)."""
    from rdkit.Chem.Draw import rdMolDraw2D

    rest = ", ".join(f"{n} {f}" if n > 1 else f for f, n in unchanged)
    line = f"+ unchanged: {rest}" if rest else ""
    cols = min(len(mols), 3)
    rows = -(-len(mols) // cols)
    strip = max(12, height // 9) if line else 0
    pw, ph = width // cols, (height - strip) // rows
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height, pw, ph)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.padding = 0.12
    if draw is None:
        drawer.DrawMolecules([rdMolDraw2D.PrepareMolForDrawing(m) for m in mols])
    else:
        draw(drawer, mols)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    text = [(k % cols * pw + 2, k // cols * ph + max(10, ph // 9), "start", "#000000", max(9, ph // 9), f"{n}×")
            for k, n in enumerate(counts) if n > 1]   # (RDKit's legend font has no ×)
    if line:
        text.append((width / 2, height - strip / 3, "middle", "#8a8f98", min(strip - 2, 1.8 * width / len(line)), line))
    return svg.replace("</svg>", "".join(
        f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="{a}" font-family="sans-serif" font-size="{size:.1f}" '
        f'fill="{fill}">{t.replace("&", "&amp;").replace("<", "&lt;")}</text>\n' for x, y, a, fill, size, t in text)
        + "</svg>")


@lru_cache(maxsize=512)
def depict_changed_svg(smiles: str, start: str, width: int = 220, height: int = 160) -> Optional[str]:
    """A species of several molecules drawn by what differs from the run's start state `start` (the
    multiset difference of their molecules): each changed molecule once, with its count when repeated, and
    a muted line for the unchanged rest. A single molecule, the start itself, or a species that keeps
    every molecule of the start, as depict_svg."""
    from collections import Counter

    try:
        from rdkit import Chem

        have, had = Counter(_molecules(smiles)), Counter(_molecules(start))
        changed, same = have - had, have & had
        if sum(have.values()) < 2 or not changed:
            return depict_svg(smiles, width, height)
        mols = []
        for m in changed:
            mol = Chem.MolFromSmiles(m)
            if mol is None:
                mol = Chem.MolFromSmiles(m, sanitize=False)
                if mol is None:
                    return depict_svg(smiles, width, height)
                mol.UpdatePropertyCache(strict=False)
            mols.append(mol)
        rest = Counter()
        for m, n in same.items():
            mol = Chem.MolFromSmiles(m) or Chem.MolFromSmiles(m, sanitize=False)
            rest[molecule_formula(mol) if mol is not None else m] += n
        return changed_svg(mols, list(changed.values()), sorted(rest.items(), key=lambda x: (-x[1], x[0])),
                           width, height)
    except Exception:
        return depict_svg(smiles, width, height)
