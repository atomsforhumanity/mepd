"""Making structures: from xyz files, multi-frame xyz files and SMILES, and
writing them back out.

Everything returns `StructureNode`s, mepd's structure type (coordinates in
Bohr, a molecular graph, cached energy). xyz files are read and written in
Angstrom; qcdata's comment line carries charge and multiplicity, and
`charge=`/`multiplicity=` override what a file says.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from qcdata import Structure

from mepd.nodes.node import StructureNode

__all__ = ["load", "read_frames", "from_smiles", "write_xyz", "as_node"]


def _override(structure: Structure, charge: int | None, multiplicity: int | None) -> Structure:
    updates = {k: v for k, v in (("charge", charge), ("multiplicity", multiplicity)) if v is not None}
    return structure.model_copy(update=updates) if updates else structure


def as_node(structure: Structure | StructureNode) -> StructureNode:
    """A `StructureNode` for a qcdata `Structure` (a node is returned as is)."""
    if isinstance(structure, StructureNode):
        return structure
    return StructureNode(structure=structure)


def load_structure(path: str | Path, charge: int | None = None, multiplicity: int | None = None) -> Structure:
    """The first frame of an xyz file as a qcdata `Structure`."""
    try:
        structure = Structure.open(str(path))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read '{path}': {type(exc).__name__}: {exc}") from exc
    return _override(structure, charge, multiplicity)


def load(path: str | Path, *, charge: int | None = None, multiplicity: int | None = None) -> StructureNode:
    """Read a structure from an xyz file (Angstrom; the first frame of a
    multi-frame file). Raises FileNotFoundError or ValueError.

    >>> reactant = mepd.load("reactant.xyz", charge=0, multiplicity=1)
    """
    return as_node(load_structure(path, charge, multiplicity))


def read_frames(path: str | Path, *, charge: int | None = None,
                multiplicity: int | None = None) -> list[StructureNode]:
    """Every frame of a multi-frame xyz file (a path, an IRC, a trajectory)."""
    try:
        structures = Structure.open_multi(str(path))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read '{path}': {type(exc).__name__}: {exc}") from exc
    return [as_node(_override(s, charge, multiplicity)) for s in structures]


# RDKit's embedding seed for SMILES: an unseeded embedding depends on how much
# of RDKit's random stream the process used before, so the same SMILES gave a
# different starting geometry (and conformer pool) from run to run.
SMILES_EMBED_SEED = 0xF00D


def _rdkit_seeded(smiles: str, **struct_kwargs) -> Structure:
    """qcinf's RDKit SMILES embedding (ETKDG, then MMFF94s), seeded."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from qcref.constants import ANGSTROM_TO_BOHR

    smiles = smiles.strip()
    if "." in smiles:
        raise ValueError("several molecules: RDKit embeds one at a time")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot read {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv2()
    params.randomSeed = SMILES_EMBED_SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"RDKit could not embed {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")
    return Structure(
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
        geometry=mol.GetConformer().GetPositions() * ANGSTROM_TO_BOHR,
        charge=Chem.GetFormalCharge(mol),
        identifiers={"canonical_smiles": canonical, "smiles": smiles, "canonical_smiles_program": "rdkit"},
        connectivity=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in mol.GetBonds()],
        **struct_kwargs,
    )


def smiles_to_structure(smiles: str, charge: int | None = None, multiplicity: int | None = None) -> Structure:
    """3D structure for a SMILES string, as a qcdata `Structure`.

    Embeds with RDKit, falling back to Open Babel; a SMILES of several molecules
    ("C=C.O") is embedded as one. If that fails, each molecule is embedded on
    its own and the molecules packed together (`mepd.complexes.packed`), atoms
    in the order the SMILES lists them. Raises ValueError if nothing works.
    """
    import qcinf

    errors: list[str] = []

    def embed(smi: str, **kwargs):
        for backend in ("rdkit", "openbabel"):
            try:
                if backend == "rdkit":
                    return _rdkit_seeded(smi, **kwargs)
                return qcinf.smiles_to_structure(smi, backend=backend, **kwargs)
            except Exception as exc:
                errors.append(f"{backend}: {type(exc).__name__}: {exc}")
        return None

    # qcinf computes the charge itself; only the multiplicity is passed through.
    structure = embed(smiles, **({"multiplicity": multiplicity} if multiplicity is not None else {}))
    pieces = [p for p in smiles.split(".") if p]
    if structure is None and len(pieces) > 1:
        from mepd.complexes import packed

        parts = [embed(p) for p in pieces]
        structure = packed(parts) if all(parts) else None
    if structure is None:
        raise ValueError(f"'{smiles}' is not a SMILES string RDKit or Open Babel can embed.\n" + "\n".join(errors))
    return _override(structure, charge, multiplicity)


def from_smiles(smiles: str, *, charge: int | None = None, multiplicity: int | None = None) -> StructureNode:
    """A 3D structure embedded from SMILES (see `smiles_to_structure`).

    >>> water_dimer = mepd.from_smiles("O.O")
    """
    return as_node(smiles_to_structure(smiles, charge, multiplicity))


def write_xyz(path: str | Path, structures: StructureNode | Structure | Iterable[StructureNode | Structure],
              comments: Sequence[str] | None = None) -> Path:
    """Write one structure, or several as frames of one file (Angstrom).

    Each frame's comment line is qcdata's (charge, multiplicity), so `load`
    and `read_frames` get them back; `comments`, if given, are appended to it.
    """
    if isinstance(structures, (StructureNode, Structure)):
        structures = [structures]
    frames = [s.structure if isinstance(s, StructureNode) else s for s in structures]
    if comments is not None and len(comments) != len(frames):
        raise ValueError(f"{len(comments)} comments for {len(frames)} frames.")
    text = []
    for i, frame in enumerate(frames):
        lines = frame.to_xyz().splitlines()
        if comments is not None and comments[i]:
            lines[1] = f"{lines[1]} {comments[i]}".strip()
        text.append("\n".join(lines) + "\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(text))
    return path
