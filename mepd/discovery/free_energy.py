"""Free energies for the explorer's kinetics.

G = E + thermal correction (quasi-RRHO, mepd.thermo.qrrho) at the
kinetics temperature, the frequencies from the run engine's own Hessian at
the geometry the engine optimized -- the level the geometries were made at.
(A Hessian from another method at these geometries is not at a stationary
point of that method: GFN2's biased Hessian at a g-xTB TS turns the reaction
mode into a real ~400 cm^-1 vibration, which then counts as one more
vibration. On formic acid at 1500 K that alone turned the measured
dehydration > decarboxylation order around.)

A species of several molecules (the explorer's products, e.g. CO + H2O)
counts as those molecules apart, each at the standard state: G = the sum of
each molecule's electronic energy (at its geometry in the complex) and
thermal correction -- not the complex's energy, whose binding would count on
top of the separated molecules' entropy. That is what makes a decomposition
product a sink: its translational and rotational entropy. Molecules get a
charge only when the whole is neutral and closed-shell (all neutral, spin
from their electron count); a charged or open-shell complex counts as one
molecule (said in the result).

A TS counts the same way: molecules that take no part in its step
(spectators) count apart, at the standard state, as they do in the step's
species; only the reacting complex stays whole and loses its reaction mode.
Which atoms react comes from the step's two ends in the TS's atom order (its
IRC's): the connected pieces of the union of the ends' bond graphs that
contain a bond made or broken (ts_groups). Not from the TS geometry alone: a
partly formed bond is long, and a distance split could cut the reacting
complex in two. Without ends, or with ends whose bonds don't bracket the
TS's, the TS counts as one molecule.

The rotational symmetry number and linearity come from xtb's analysis of the
geometry (mepd.thermo.frequencies), which is cheap at these sizes.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

HARTREE_TO_KCAL = 627.5094740631
BOHR_PER_ANGSTROM = 1.8897259886
def _electrons(symbol: str) -> int:
    from rdkit import Chem

    return Chem.GetPeriodicTable().GetAtomicNumber(symbol)


def bonds(symbols, xyz_A, scale: float = 1.25) -> set[tuple[int, int]]:
    """(i, j), i > j, of the atom pairs closer than `scale` x the sum of their covalent radii."""
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    rc = np.array([table.GetRcovalent(s) for s in symbols])
    xyz = np.asarray(xyz_A, dtype=float).reshape(-1, 3)
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    i, j = np.nonzero(np.tril(d < scale * (rc[:, None] + rc[None]), -1))
    return set(zip(i.tolist(), j.tolist()))


def _components(n: int, edges) -> list[list[int]]:
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values())


def molecules(symbols, xyz_A, scale: float = 1.25) -> list[list[int]]:
    return _components(len(symbols), bonds(symbols, xyz_A, scale))


def ts_groups(symbols, ts_xyz_A, ends_xyz_A) -> Optional[list[list[int]]]:
    """A TS's molecules as the free energies count them: the reacting complex
    first (every connected piece of the union of the two ends' bond graphs
    that holds a bond made or broken, together), then each spectator. None
    when nothing reacts, or when the ends' bonds don't bracket the TS's (a
    bond in both ends missing at the TS, or a TS bond in neither end): ends
    not in the TS's atom order."""
    b0, b1 = (bonds(symbols, e) for e in ends_xyz_A)
    bts = bonds(symbols, ts_xyz_A)
    if not (b0 & b1 <= bts <= b0 | b1):
        return None
    changed = {i for pair in b0 ^ b1 for i in pair}
    comps = _components(len(symbols), b0 | b1)
    react = sorted(i for c in comps if changed & set(c) for i in c)
    if not react:
        return None
    return [react] + [c for c in comps if not changed & set(c)]


def fragment_charges(symbols, xyz_A, charge: int, multiplicity: int, groups: list[list[int]],
                     tolerance: float = 0.35) -> Optional[list[int]]:
    """Integer charge of each group of atoms: its summed GFN2-xTB Mulliken
    charges (ALPB water) rounded, or None when a sum is further than
    `tolerance` from an integer or the integers don't add up to `charge`."""
    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from mepd.thermo import _xtb

    work = Path(tempfile.mkdtemp(prefix="mepd_fragq_"))
    try:
        lines = [str(len(symbols)), ""] + [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in
                                            zip(symbols, np.asarray(xyz_A).reshape(-1, 3))]
        (work / "in.xyz").write_text("\n".join(lines) + "\n")
        run = subprocess.run([_xtb(), "in.xyz", "--sp", "--chrg", str(int(charge)), "--uhf",
                              str(int(multiplicity) - 1), "--alpb", "water"], cwd=work, capture_output=True,
                             text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        if run.returncode != 0 or not (work / "charges").exists():
            raise RuntimeError(f"xtb failed (exit {run.returncode})")
        q = np.array([float(v) for v in (work / "charges").read_text().split()])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    sums = [float(q[g].sum()) for g in groups]
    ints = [int(round(v)) for v in sums]
    if any(abs(v - i) > tolerance for v, i in zip(sums, ints)) or sum(ints) != int(charge):
        return None
    return ints


_ENDS: dict = {}


def step_ends(files: dict, species_ends: tuple) -> tuple:
    """A step's two ends in its TS's atom order (Structures), for a TS's
    spectators (ts_groups): its IRC's first and last frames (files["irc"]),
    else those of the path its TS job started from (<TS file stem>_ends.xyz),
    else `species_ends` -- the step's two species, which are in the TS's
    atom order only when the TS was searched between them (ts_groups tells
    when they aren't)."""
    from pathlib import Path

    from qcdata import Structure

    ts = files.get("ts")
    for fp in (files.get("irc"), Path(ts).with_name(f"{Path(ts).stem}_ends.xyz") if ts else None):
        if fp and Path(fp).exists():
            key = (str(fp), Path(fp).stat().st_mtime)
            if key not in _ENDS:
                frames = Structure.open_multi(str(fp))
                _ENDS[key] = (frames[0], frames[-1])
            return _ENDS[key]
    return tuple(species_ends)


class FreeEnergies:
    def __init__(self, engine, temperature: float, standard_state: str = "1M", workers: int = 1):
        self.engine, self.T, self.standard_state = engine, float(temperature), standard_state
        self.workers = max(1, int(workers))
        self._cache: dict[str, float] = {}
        self.notes: list[str] = []

    @staticmethod
    def _groups(structure, ts: bool, ends) -> Optional[list[list[int]]]:
        """A TS's groups from its step's two ends (Structures in its atom order; see ts_groups), else None."""
        if not ts or ends is None:
            return None
        return ts_groups([str(s) for s in structure.symbols], np.asarray(structure.geometry) / BOHR_PER_ANGSTROM,
                         [np.asarray(e.geometry) / BOHR_PER_ANGSTROM for e in ends])

    def _key(self, structure, ts: bool, ends=None) -> str:
        xyz = np.round(np.asarray(structure.geometry, dtype=float), 4) + 0.0
        groups = self._groups(structure, ts, ends)     # a TS with spectators apart; one without: as before
        blob = repr((list(map(str, structure.symbols)), xyz.tolist(), int(structure.charge),
                     int(structure.multiplicity), ts) + ((groups,) if groups and len(groups) > 1 else ()))
        return hashlib.sha1(blob.encode()).hexdigest()

    def _one(self, symbols, xyz_A, charge, mult, ts) -> float:
        """Thermal correction (kcal/mol) of one molecule."""
        import mepd
        from qcdata import Structure

        from mepd.nodes.node import StructureNode
        from mepd.thermo import frequencies as xtb_analysis, qrrho

        if len(symbols) == 1:
            return qrrho([], symbols, xyz_A, self.T, multiplicity=mult, standard_state=self.standard_state)["g_corr"]
        shape = xtb_analysis(symbols, xyz_A, charge=charge, multiplicity=mult)
        node = StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(xyz_A) * BOHR_PER_ANGSTROM,
                                                 charge=int(charge), multiplicity=int(mult)))
        w = sorted(mepd.frequencies(node, self.engine).wavenumbers_cm.tolist(), key=abs)
        if len(w) == 3 * len(symbols):          # with the translations and rotations: drop them
            w = w[(5 if shape["linear"] else 6):]
        return qrrho(sorted(w), symbols, xyz_A, self.T, rotational_number=shape["rotational_number"],
                     linear=shape["linear"], multiplicity=mult, ts=ts,
                     standard_state=self.standard_state)["g_corr"]

    def parts(self, structure, ts: bool, ends=None) -> list[tuple[list[int], int, int]]:
        """(atoms, charge, multiplicity) of each molecule the species counts
        as. A TS: its reacting complex first, then its spectators, found from
        its step's two `ends` (see ts_groups); one entry without them. In a charged or open-shell complex each
        molecule's charge is its share of GFN2-xTB Mulliken charges (ALPB
        water, where ion pairs stay ions), rounded; a complex whose shares
        aren't near-integers, or whose spins don't add up, counts as one
        molecule (said in the notes)."""
        symbols = [str(s) for s in structure.symbols]
        xyz = np.asarray(structure.geometry, dtype=float) / BOHR_PER_ANGSTROM
        charge, mult = int(structure.charge), int(structure.multiplicity)
        whole = [(list(range(len(symbols))), charge, mult)]
        if ts:
            groups = self._groups(structure, ts, ends)
            if groups is None and ends is not None:
                self.notes.append("a TS whose step's ends don't match its bonds counted as one molecule, "
                                  "spectators included")
            groups = groups or [list(range(len(symbols)))]
        else:
            groups = molecules(symbols, xyz)
        if len(groups) == 1:
            return whole
        if charge == 0 and mult == 1:
            out = [(g, 0, 1 + sum(_electrons(symbols[k]) for k in g) % 2) for g in groups]
            if all(m == 1 for _, _, m in out):
                return out
        try:
            q = fragment_charges(symbols, xyz, charge, mult, groups)
        except Exception as exc:
            self.notes.append(f"charges of a complex's molecules failed ({type(exc).__name__}); counted as one molecule")
            return whole
        if q is None:
            self.notes.append(f"a charged or open-shell species of {len(groups)} molecules counted as one molecule "
                              "(its molecules' charges are not clear)")
            return whole
        out = []
        for g, qi in zip(groups, q):
            unpaired = (sum(_electrons(symbols[k]) for k in g) - qi) % 2
            out.append((g, qi, 1 + unpaired))
        if sum(m - 1 for _, _, m in out) != mult - 1:
            self.notes.append(f"an open-shell species of {len(groups)} molecules counted as one molecule "
                              "(its molecules' spins don't add up to its multiplicity)")
            return whole
        return out

    def _correction(self, structure, ts: bool, ends=None) -> float:
        symbols = [str(s) for s in structure.symbols]
        xyz = np.asarray(structure.geometry, dtype=float) / BOHR_PER_ANGSTROM
        parts = self.parts(structure, ts, ends)
        if len(parts) == 1:
            return self._one(symbols, xyz, parts[0][1], parts[0][2], ts)
        import mepd
        from qcdata import Structure

        from mepd.nodes.node import StructureNode

        total, frags = 0.0, []
        for n, (p, qi, mi) in enumerate(parts):
            sy = [symbols[k] for k in p]
            total += self._one(sy, xyz[p], qi, mi, ts and n == 0)    # a TS's reacting complex comes first
            frags.append(StructureNode(structure=Structure(symbols=sy, geometry=xyz[p] * BOHR_PER_ANGSTROM,
                                                           charge=qi, multiplicity=mi)))
        # The molecules apart: their electronic energies too, not the complex's (whose binding would
        # otherwise count on top of the separated molecules' entropy). Each at its geometry in the complex.
        whole = StructureNode(structure=structure)
        apart = sum(mepd.energies(frags, self.engine)) - mepd.energies([whole], self.engine)[0]
        return total + apart * HARTREE_TO_KCAL

    def corrections(self, structures: list, ts_flags: list[bool], ends: Optional[list] = None
                    ) -> list[Optional[float]]:
        """Thermal corrections (kcal/mol) for `structures`, each computed once
        (cached by geometry); None where the Hessian failed. `ends`: per
        structure, a TS's step's two ends (see parts) or None."""
        from mepd.cli_common import _fork_map

        ends = ends if ends is not None else [None] * len(structures)
        keys = [self._key(s, t, e) for s, t, e in zip(structures, ts_flags, ends)]
        todo = {}
        for k, s, t, e in zip(keys, structures, ts_flags, ends):
            if k not in self._cache and k not in todo:
                todo[k] = (s, t, e)

        def run(item):
            s, t, e = item
            try:
                return self._correction(s, t, e)
            except Exception as exc:   # reported by the caller: the species keeps its electronic energy
                return f"{type(exc).__name__}: {exc}"

        items = list(todo.items())
        for (k, _), val in zip(items, _fork_map(run, [v for _, v in items], self.workers)):
            self._cache[k] = val
        out = []
        for k in keys:
            v = self._cache[k]
            if isinstance(v, str):
                self.notes.append(f"thermal correction failed ({v}); electronic energy used")
                out.append(None)
            else:
                out.append(float(v))
        return out
