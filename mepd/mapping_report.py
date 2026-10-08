"""What the start/end atom mapping did, for people to see: which mappings
SLAPMapper and its relabelings offered, how each scored, which bonds each
breaks and forms, and a short interpolation from the start to the end under
each. Written by `mepd run` / `mepd channels` as <output>/atom_mapping.json
and drawn on the web Results page (mepd.web.results.atom_mapping_view).

Atoms are numbered from 1 by the start's order; a mapping renumbers the
end's atoms so its atom k is the start's atom k.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

MAX_CANDIDATES = 10        # the chosen one, the current numbering, then the best-scoring others
N_FRAMES = 9


def _edges(structure) -> set[tuple[int, int]]:
    from mepd.nodes.node import StructureNode

    node = StructureNode(structure=structure)
    return {tuple(sorted((int(u), int(v)))) for u, v in node.graph.edges()} if node.graph is not None else set()


def bond_changes(start, end_in_start_order) -> tuple[list, list]:
    """(broken, formed) bonds, as sorted start-atom index pairs, when the
    start becomes `end_in_start_order` (atom k of each is the same atom)."""
    a, b = _edges(start), _edges(end_in_start_order)
    return sorted(a - b), sorted(b - a)


def _xyz(symbols, coords_bohr, comment: str = "") -> str:
    ang = np.asarray(coords_bohr, dtype=float) / 1.8897261246
    return f"{len(symbols)}\n{comment}\n" + "".join(f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, (x, y, z) in zip(symbols, ang))


def _aligned_linear(start, end) -> list[str]:
    """A straight-line path after turning the end onto the start (Kabsch):
    enough to see which atom goes where when no interpolation was built."""
    x0, x1 = np.asarray(start.geometry, dtype=float), np.asarray(end.geometry, dtype=float)
    c0, c1 = x0.mean(axis=0), x1.mean(axis=0)
    u, _, vt = np.linalg.svd((x1 - c1).T @ (x0 - c0))
    d = np.sign(np.linalg.det(u @ vt))
    x1 = (x1 - c1) @ (u @ np.diag([1.0, 1.0, d]) @ vt) + c0
    return [_xyz(start.symbols, x0 + (x1 - x0) * t, f"t={t:.2f}") for t in np.linspace(0.0, 1.0, N_FRAMES)]


def _chain_frames(chain) -> Optional[list[str]]:
    try:
        nodes = list(chain.nodes)
        if len(nodes) < 2:
            return None
        keep = sorted({round(i * (len(nodes) - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)})
        return [_xyz(nodes[i].structure.symbols, nodes[i].structure.geometry, f"image {i + 1}") for i in keep]
    except Exception:
        return None


def _kind(candidate) -> str:
    if candidate.label == "identity":
        return "current"
    if "snap" in candidate.label:
        return "snap"
    am = candidate.atom_map
    if getattr(am, "explored", False):
        return "explored"
    if getattr(am, "relabeling", False):
        return "symmetric"
    return "slapmapper"


def build_report(start, end_given, candidates, *, metric: str, scores: Optional[dict] = None,
                 chosen: Optional[str] = None, chains: Optional[dict] = None, quantity: str = "",
                 notes: Optional[list] = None, slapmapper_identity: Optional[bool] = None,
                 summary: str = "") -> dict:
    """The JSON-able record. `candidates`: mepd.atom_mapping_selection.
    MappingCandidate (each with its end in the start's order)."""
    scores = dict(scores or {})
    chains = dict(chains or {})
    order_seen = {c.label: i for i, c in enumerate(candidates)}
    ranked = sorted(candidates, key=lambda c: (c.label != chosen, c.label != "identity",
                                              scores.get(c.label, float("inf")), order_seen[c.label]))
    out = []
    for c in ranked[:MAX_CANDIDATES]:
        broken, formed = bond_changes(start, c.end_structure)
        frames = _chain_frames(chains[c.label]) if c.label in chains else None
        mapping = None
        if c.atom_map is not None:
            mapping = [int(c.atom_map.mapping.get(i, i)) for i in range(len(start.symbols))]
        out.append({
            "label": c.label, "kind": _kind(c), "chosen": c.label == chosen,
            "score": None if c.label not in scores else float(scores[c.label]),
            "broken": [list(map(int, b)) for b in broken], "formed": [list(map(int, f)) for f in formed],
            "end_index_of_start_atom": mapping,
            "end_xyz": _xyz(c.end_structure.symbols, c.end_structure.geometry),
            "frames": frames or _aligned_linear(start, c.end_structure),
            "interpolation": ("geodesic interpolation" if frames
                              else "straight line between the aligned ends, to show which atom goes where"),
        })
    # Where each differs from the one used: start atoms whose partner in the
    # end is another atom (same element; e.g. which of a methyl's H moves).
    used = next((c for c in ranked if c.label == chosen), None)
    if used is not None:
        ref = np.asarray(used.end_structure.geometry, dtype=float)
        for entry, c in zip(out, ranked):
            g = np.asarray(c.end_structure.geometry, dtype=float)
            entry["differs_at"] = [] if c is used else \
                [int(k) for k in np.where(np.linalg.norm(g - ref, axis=1) > 1e-3)[0]]
    return {
        "metric": metric, "quantity": quantity, "chosen": chosen, "summary": summary,
        "slapmapper_identity": slapmapper_identity, "n_candidates": len(candidates),
        "notes": list(notes or []), "start_xyz": _xyz(start.symbols, start.geometry),
        "symbols": list(start.symbols), "candidates": out,
    }


def write_report(output: Optional[Path], report: dict) -> None:
    if output is None:
        return
    try:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "atom_mapping.json").write_text(json.dumps(report))
    except Exception:
        pass


# ------------------------------------------------------------------ drawing

def _mol_from_xyz(xyz: str, charge: int = 0):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    mol = Chem.MolFromXYZBlock(xyz)
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
    except Exception:
        mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineConnectivity(mol)
    return mol


def mapping_svg(xyz: str, keep_h: set, highlight_bonds: list, colour: tuple, charge: int = 0,
                width: int = 300, height: int = 220) -> Optional[str]:
    """A 2D drawing of one side, each atom labelled by its number (from 1,
    the start's order); hydrogens hidden unless in `keep_h`; the bonds in
    `highlight_bonds` (index pairs) highlighted in `colour` (drawn dashed
    when they are not bonds of this side, e.g. a bond that forms)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = _mol_from_xyz(xyz, charge)
        if mol is None:
            return None
        rw = Chem.RWMol(mol)
        for a in rw.GetAtoms():
            a.SetIntProp("num", a.GetIdx() + 1)
        drop = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "H" and a.GetIdx() not in keep_h
                and a.GetDegree() == 1]
        for i in sorted(drop, reverse=True):
            heavy = rw.GetAtomWithIdx(i).GetNeighbors()[0]
            heavy.SetNumExplicitHs(heavy.GetNumExplicitHs() + 1)
            heavy.SetNoImplicit(True)
            rw.RemoveAtom(i)
        index = {a.GetIntProp("num") - 1: a.GetIdx() for a in rw.GetAtoms()}
        hl_bonds, hl_atoms = [], set()
        for i, j in highlight_bonds:
            if i in index and j in index:
                hl_atoms.update((index[i], index[j]))
                b = rw.GetBondBetweenAtoms(index[i], index[j])
                if b is not None:
                    hl_bonds.append(b.GetIdx())
        draw = rw.GetMol()
        for a in draw.GetAtoms():
            a.SetProp("atomNote", str(a.GetIntProp("num")))
        try:
            Chem.SanitizeMol(draw, Chem.SanitizeFlags.SANITIZE_NONE)
        except Exception:
            pass
        rdDepictor.Compute2DCoords(draw)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2
        opts.padding = 0.12
        opts.annotationFontScale = 0.75
        opts.highlightBondWidthMultiplier = 14
        atom_cols = {k: colour for k in hl_atoms}
        bond_cols = {k: colour for k in hl_bonds}
        drawer.DrawMolecule(draw, highlightAtoms=list(hl_atoms), highlightAtomColors=atom_cols,
                            highlightBonds=hl_bonds, highlightBondColors=bond_cols)
        drawer.FinishDrawing()
        return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    except Exception:
        return None


BROKEN = (0.86, 0.33, 0.29)    # on the start
FORMED = (0.25, 0.62, 0.38)    # on the end


def candidate_drawings(report: dict, candidate: dict, charge: int = 0) -> dict:
    """{start, end}: SVGs of the start (bonds that break highlighted) and of
    the end in the start's numbering (bonds that form highlighted)."""
    changed = {i for pair in candidate["broken"] + candidate["formed"] for i in pair}
    keep_h = {i for i in changed if report["symbols"][i] == "H"}
    return {"start": mapping_svg(report["start_xyz"], keep_h, candidate["broken"], BROKEN, charge),
            "end": mapping_svg(candidate["end_xyz"], keep_h, candidate["formed"], FORMED, charge)}
