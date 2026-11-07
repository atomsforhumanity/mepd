"""Earlier expansions as known (`mepd discovery expand --known <folder>`, repeatable): what an earlier
`discovery expand` at the same level of theory found and computed is taken as it is, not computed again.

`read_known` reads each folder (summary.json, provenance.json, its species files, its steering single points
in steer_single_points/ and the Hessian corrections kept in expand_state.pkl) and says why a folder is not
used: another level of theory (provenance.json's `level_key`), no summary.json, other atoms, or this run's own
output folder. The explorer (network_expansion.expand_network) maps a used folder's atoms onto its seed's and
takes its species, steps, single points and Hessians (`KnownRun`).
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class KnownRun:
    """One earlier expansion, in its own atom order and species indices."""
    folder: Path
    summary: dict
    species: list                    # qcdata Structures, one per species
    free_cache: dict = field(default_factory=dict)   # FreeEnergies._cache of its last saved state
    why_no_hessians: str = ""
    charge_mult: Optional[tuple] = None   # (charge, multiplicity) it ran at, when its records say

    @property
    def steer_dir(self) -> Path:
        return self.folder / "steer_single_points"

    def path(self, recorded) -> Optional[Path]:
        """A file its summary names (relative to the directory that run started in, or absolute), here."""
        if not recorded:
            return None
        p = Path(str(recorded))
        if p.is_absolute() and p.exists():
            return p
        parts = p.parts
        for k in range(len(parts)):
            q = self.folder.joinpath(*parts[k:])
            if q.exists():
                return q
        return None

    def sampled_pairs(self) -> set:
        """The species pairs (frozensets) its channel sampling ran on."""
        return {frozenset(map(int, p)) for r in self.summary.get("rounds") or [] for p in r.get("channels_sampled") or []}

    def expanded(self) -> set:
        """The species its generator was run on."""
        return {int(k) for r in self.summary.get("rounds") or [] for k in r.get("sources") or []}

    def proposed_pairs(self) -> set:
        """The pairs its generator proposed and it connected (a step's asked pair that a proposal made)."""
        made = {frozenset((int(r["source"]), int(r["target"]))) for r in self.summary.get("reactions") or []
                if r.get("target") is not None and r.get("target") != r.get("source")}
        asked = {frozenset(map(int, st["files"]["pair"])) for st in self.summary.get("steps") or []
                 if (st.get("files") or {}).get("pair")}
        return made & asked

    def parent(self, j: int) -> int:
        """The species species `j` was found from: its proposal's source, else a lower species it has a step
        with, else the seed."""
        for r in self.summary.get("reactions") or []:
            if r.get("outcome") == "new_species" and r.get("target") == j:
                return int(r["source"])
        lower = [st["b"] if st["a"] == j else st["a"] for st in self.summary.get("steps") or []
                 if j in (st["a"], st["b"]) and min(st["a"], st["b"]) < j]
        return int(min(lower)) if lower else 0


def summary_checksum(folder: Path) -> Optional[str]:
    fp = Path(folder) / "summary.json"
    return hashlib.sha256(fp.read_bytes()).hexdigest() if fp.is_file() else None


def fingerprint(folders) -> list:
    """What a resumed run must have been given: each --known folder and its summary.json's checksum."""
    return [[str(Path(f).resolve()), summary_checksum(Path(f))] for f in folders]


def read_known(folders, level_key: Optional[str], output: Optional[Path] = None,
               say: Callable[[str], None] = lambda m: None) -> tuple[list, list]:
    """(the KnownRuns to use, [{"folder", "why"} of the folders skipped]). A folder is used only at this run's
    level of theory: provenance.json's `level_key` must be `level_key`."""
    runs, skipped = [], []

    def skip(folder, why):
        skipped.append({"folder": str(folder), "why": why})
        say(f"Known run {folder} not used: {why}.")

    for folder in folders:
        folder = Path(folder)
        if output is not None and folder.resolve() == Path(output).resolve():
            skip(folder, "it is this run's own output folder")
            continue
        if not (folder / "summary.json").is_file():
            skip(folder, "it has no summary.json (not a finished discovery expand)")
            continue
        try:
            prov = json.loads((folder / "provenance.json").read_text())
        except (OSError, ValueError):
            skip(folder, "it has no provenance.json, so its level of theory is not known")
            continue
        if not prov.get("level_key") or prov.get("level_key") != level_key:
            skip(folder, f"another level of theory (level_key {prov.get('level_key')}, this run's {level_key})")
            continue
        summary = json.loads((folder / "summary.json").read_text())
        from qcdata import Structure

        try:
            species = [Structure.open(str(folder / "species" / f"species_{int(s['index'])}.xyz"))
                       for s in summary["species"]]
        except Exception as exc:
            skip(folder, f"its species files could not be read ({type(exc).__name__}: {exc})")
            continue
        run = KnownRun(folder, summary, species)
        run.charge_mult = _charge_mult(run, prov.get("argv") or [])
        state = folder / "expand_state.pkl"
        try:
            with open(state, "rb") as fh:
                run.free_cache = dict(pickle.load(fh).get("free_cache") or {})
        except Exception as exc:
            run.why_no_hessians = f"its saved state ({state.name}) could not be read: {type(exc).__name__}"
        runs.append(run)
    return runs, skipped


def _charge_mult(run: KnownRun, argv: list) -> Optional[tuple]:
    """The charge and multiplicity a run used: its --charge / --multiplicity, else those its TS files record
    (species files don't), else None."""
    from mepd.provenance import _flag_value

    q, m = _flag_value(argv, "--charge", "--charge"), _flag_value(argv, "--multiplicity", "--multiplicity")
    if q is not None and m is not None:
        return int(q), int(m)
    from qcdata import Structure

    for st in run.summary.get("steps") or []:
        fp = run.path((st.get("files") or {}).get("ts"))
        if fp is not None and "qcdata_charge" in fp.read_text().split("\n", 2)[1]:
            s = Structure.open(str(fp))
            return int(q) if q is not None else int(s.charge), int(m) if m is not None else int(s.multiplicity)
    return None


def permuted_xyz(src: Path, dst: Path, order: list) -> None:
    """`src`'s frames with atom i of each frame being `order[i]` of the original, every line kept as written
    (the geometry to the digit: the cache keys of the run that wrote it stay those of its atoms)."""
    lines = Path(src).read_text().splitlines()
    out, k = [], 0
    while k < len(lines):
        if not lines[k].strip():
            k += 1
            continue
        n = int(lines[k].split()[0])
        atoms = lines[k + 2:k + 2 + n]
        out += [lines[k], lines[k + 1]] + [atoms[i] for i in order]
        k += 2 + n
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("\n".join(out) + "\n")


def reordered(structure, order: list):
    """`structure` with atom i being its atom `order[i]`."""
    return structure.model_copy(update={"symbols": [structure.symbols[i] for i in order],
                                        "geometry": np.asarray(structure.geometry, dtype=float)[order]})


def atom_order(seed, structure) -> Optional[list]:
    """The order that puts `structure`'s atoms onto `seed`'s (both StructureNodes of one species): identity when
    its atoms and bonds already correspond, else mepd's atom mapping (SLAPMapper; atom_mapping), kept only if
    it changes no bond. None: no such order."""
    from mepd.atom_mapping import _bond_changes_under, suggest_atom_mapping
    from mepd.discovery.network_expansion import graph_edges

    a, b = seed.structure, structure.structure
    if sorted(map(str, a.symbols)) != sorted(map(str, b.symbols)) or (int(a.charge), int(a.multiplicity)) != (
            int(b.charge), int(b.multiplicity)):
        return None
    if list(map(str, a.symbols)) == list(map(str, b.symbols)) and graph_edges(seed) == graph_edges(structure):
        return list(range(len(a.symbols)))
    try:
        m = suggest_atom_mapping(a, b)
    except Exception:
        return None
    if m is None or _bond_changes_under(a, b, m.mapping) != 0:
        return None
    return m.as_order()
