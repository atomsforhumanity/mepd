"""Does a network expansion's step connect the two species it is reported
between? Its IRC's two ends, perceived, must be those species (either way
round). A TS search for one pair can slide into an unrelated saddle (e.g. a
methyl rotation of a product, far below both species): the run reports the
step at once and drops it later, so the web checks before an edge counts it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _frames(text: str) -> list[str]:
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        head = lines[i].strip()
        if not head.isdigit():
            i += 1
            continue
        n = int(head)
        out.append("\n".join(lines[i:i + n + 2]))
        i += n + 2
    return out


def _graph_key(smiles: Optional[str]) -> Optional[str]:
    """A species' identity as the explorer tells species apart: its molecular graph
    (atoms, hydrogens included, and which are bonded), whatever charges, radicals or
    bond orders a Lewis structure puts on it ("[OH].[H]" and "[H+].[OH-]" are one).
    An IRC end is not minimized: its stereo, and its Lewis reading, can go either way."""
    if not smiles:
        return None
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
            mol.UpdatePropertyCache(strict=False)
        mol = Chem.RWMol(Chem.AddHs(mol))
        for a in mol.GetAtoms():
            a.SetFormalCharge(0)
            a.SetNumRadicalElectrons(0)
            a.SetIsAromatic(False)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(True)
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for b in mol.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
            b.SetStereo(Chem.BondStereo.STEREONONE)
        # Canonical over the whole set of fragments; hydrogens stay atoms of their own (re-reading
        # the SMILES would fold them back and make acetaldehyde and vinyl alcohol one).
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def same_species(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    """Same molecular graph; None when either cannot be read."""
    ka, kb = _graph_key(a), _graph_key(b)
    if ka is None or kb is None:
        return None
    return ka == kb


def irc_path(job: dict, label: str) -> Path:
    return Path(job.get("output_dir") or "") / "ts" / f"{label}_irc.xyz"


def check(job: dict, label: str, smiles_a: Optional[str], smiles_b: Optional[str]) -> Optional[dict]:
    """{"ok", "ends": [first, last] SMILES, "reversed": runs b -> a} for step
    `label` of `job`, or None when it cannot be told yet (no IRC file, or a
    species without a SMILES)."""
    fp = irc_path(job, label)
    if not label or not fp.exists() or not smiles_a or not smiles_b:
        return None
    try:
        frames = _frames(fp.read_text())
    except OSError:
        return None
    if len(frames) < 2:
        return None
    from mepd.web import chem

    ends = []
    for fr in (frames[0], frames[-1]):
        try:
            (s,) = chem.structures_from_xyz_text(fr, job.get("charge"), job.get("multiplicity"))
            ends.append(chem.perceive_smiles(s))
        except Exception:
            ends.append(None)
    fwd = [same_species(ends[0], smiles_a), same_species(ends[1], smiles_b)]
    rev = [same_species(ends[0], smiles_b), same_species(ends[1], smiles_a)]
    if None in fwd + rev and not (all(fwd) or all(rev)):
        return None
    return {"ok": bool(all(fwd) or all(rev)), "ends": ends, "reversed": bool(all(rev) and not all(fwd))}


def path_frames(job: dict, label: str) -> Optional[tuple[list[str], list[Optional[float]]]]:
    """The step's IRC as frames (xyz text) and energies (Hartree), or None."""
    fp = irc_path(job, label)
    if not fp.exists():
        return None
    frames = _frames(fp.read_text())
    en_fp = fp.with_suffix(".energies")
    energies: list[Optional[float]] = [None] * len(frames)
    if en_fp.exists():
        vals = [float(x) for x in en_fp.read_text().split()]
        if len(vals) == len(frames):
            energies = vals
    return frames, energies


def step_events(job_dir: Path) -> list[dict]:
    fp = Path(job_dir) / "live" / "events.jsonl"
    out = []
    if not fp.exists():
        return out
    for line in fp.read_text().splitlines():
        if '"step"' not in line[:30]:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "step":
            out.append(ev)
    return out
