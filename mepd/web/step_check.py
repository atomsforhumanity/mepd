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


def seed_energy(job: dict, node: Optional[str]) -> Optional[float]:
    """The run's own optimized start (Hartree), when `node` is the job's start: what the finished run
    writes to species/species_0.xyz, read while it runs (or after it stopped or failed) from its
    summary_partial.json (summary.json once it is there). None for any other node."""
    seeds = (job.get("targets") or {}).get("structures") or []
    if node is None or node not in {(job.get("live_nodes") or {}).get("0"), *seeds[:1]}:
        return None
    out = Path(job.get("output_dir") or "")
    for name in ("summary.json", "summary_partial.json"):
        try:
            e = json.loads((out / name).read_text()).get("seed_energy")
        except (OSError, ValueError, AttributeError):
            continue
        if e is not None:
            return float(e)
    return None


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


class _NoEnergies:
    """An engine for `subsystem` that only picks atoms: the energies it asks for (its `env`) are no view's."""

    region = None

    def __init__(self, restraints=None):
        self.restraints = restraints

    def compute_energies(self, nodes):
        return [0.0] * len(nodes)


_SUBSYSTEMS: dict = {}


def step_files(job: dict, label: str) -> dict:
    """A network expansion's step `label`: its files as the run's summary lists them (a channel's TS and
    IRC put back among the spectators), else ts/<label>.xyz and ts/<label>_irc.xyz."""
    out = Path(job.get("output_dir") or "")
    for name in ("summary.json", "summary_partial.json"):
        try:
            steps = json.loads((out / name).read_text()).get("steps") or []
        except (OSError, ValueError):
            continue
        for st in steps:
            if st.get("label") == label and (st.get("files") or {}).get("ts"):
                return st["files"]
    return {"ts": str(out / "ts" / f"{label}.xyz"), "irc": str(irc_path(job, label))}


def subsystem(job: dict, label: str) -> Optional[dict]:
    """{"atoms", "charge", "multiplicity"}: the molecules step `label` of a network expansion ran its path
    search, TS and IRC on -- the explorer's own cut (network_expansion.subsystem) of the step's two ends
    (free_energy.step_ends: its IRC's first and last frames, else the ends its TS search started between).
    None when it ran on every atom: every molecule takes part, or a QM/MM system, or a profile's restraints."""
    if job.get("qmmm"):
        return None
    files = step_files(job, label)
    profile = Path(job.get("output_dir") or "").parent / "inputs" / "profile.toml"
    paths = [Path(p) for p in (files.get("irc"), files.get("ts")) if p] + [profile]
    key = tuple((str(p), p.stat().st_mtime if p.exists() else None) for p in paths)
    if key in _SUBSYSTEMS:
        return _SUBSYSTEMS[key]
    cut = None
    try:
        from mepd.discovery.free_energy import step_ends
        from mepd.discovery.network_expansion import subsystem as explorer_subsystem
        from mepd.nodes.node import StructureNode

        restraints = None
        if profile.exists():
            import tomli

            restraints = tomli.loads(profile.read_text()).get("restraints")
        ends = step_ends(files, ())
        if len(ends) == 2:
            q = {k: job[k] for k in ("charge", "multiplicity") if job.get(k) is not None}
            a, b = (StructureNode(structure=e.model_copy(update=q)) for e in ends)
            found = explorer_subsystem(a, b, _NoEnergies(restraints))
            if found is not None:
                cut = {"atoms": [int(i) for i in found.atoms], "charge": found.charge,
                       "multiplicity": found.multiplicity}
    except Exception:
        cut = None
    _SUBSYSTEMS[key] = cut
    return cut


def subset_xyz(xyz: str, atoms: list) -> str:
    """An xyz frame with only `atoms` (0-based), in their order; the comment line kept."""
    lines = xyz.splitlines()
    n = int(lines[0].split()[0])
    body = lines[2:2 + n]
    return "\n".join([str(len(atoms)), lines[1] if len(lines) > 1 else ""] + [body[i] for i in atoms]) + "\n"
