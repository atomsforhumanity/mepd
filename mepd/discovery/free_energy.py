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

The rotational symmetry number and linearity come from xtb's analysis of the
geometry (mepd.thermo.frequencies), which is cheap at these sizes.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

HARTREE_TO_KCAL = 627.5094740631
BOHR_PER_ANGSTROM = 1.8897259886
_EL = {"H": 1, "He": 2, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12, "Si": 14, "P": 15,
       "S": 16, "Cl": 17, "K": 19, "Br": 35, "I": 53}


def molecules(symbols, xyz_A, scale: float = 1.25) -> list[list[int]]:
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    rc = [table.GetRcovalent(s) for s in symbols]
    n = len(symbols)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i):
            if np.linalg.norm(xyz_A[i] - xyz_A[j]) < scale * (rc[i] + rc[j]):
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values())


class FreeEnergies:
    def __init__(self, engine, temperature: float, standard_state: str = "1M", workers: int = 1):
        self.engine, self.T, self.standard_state = engine, float(temperature), standard_state
        self.workers = max(1, int(workers))
        self._cache: dict[str, float] = {}
        self.notes: list[str] = []

    @staticmethod
    def _key(structure, ts: bool) -> str:
        xyz = np.round(np.asarray(structure.geometry, dtype=float), 4) + 0.0
        blob = repr((list(map(str, structure.symbols)), xyz.tolist(), int(structure.charge),
                     int(structure.multiplicity), ts))
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

    def _correction(self, structure, ts: bool) -> float:
        symbols = [str(s) for s in structure.symbols]
        xyz = np.asarray(structure.geometry, dtype=float) / BOHR_PER_ANGSTROM
        charge, mult = int(structure.charge), int(structure.multiplicity)
        parts = [list(range(len(symbols)))] if ts else molecules(symbols, xyz)
        if len(parts) > 1 and (charge != 0 or mult != 1):
            self.notes.append(f"a charged or open-shell species of {len(parts)} molecules counted as one molecule")
            parts = [list(range(len(symbols)))]
        if len(parts) == 1:
            return self._one(symbols, xyz, charge, mult, ts)
        import mepd
        from qcdata import Structure

        from mepd.nodes.node import StructureNode

        total, frags = 0.0, []
        for p in parts:
            sy = [symbols[k] for k in p]
            nel = sum(_EL[s] for s in sy)
            total += self._one(sy, xyz[p], 0, 1 + nel % 2, False)
            frags.append(StructureNode(structure=Structure(symbols=sy, geometry=xyz[p] * BOHR_PER_ANGSTROM, charge=0,
                                                           multiplicity=1 + nel % 2)))
        # The molecules apart: their electronic energies too, not the complex's (whose binding would
        # otherwise count on top of the separated molecules' entropy). Each at its geometry in the complex.
        whole = StructureNode(structure=structure)
        apart = sum(mepd.energies(frags, self.engine)) - mepd.energies([whole], self.engine)[0]
        return total + apart * HARTREE_TO_KCAL

    def corrections(self, structures: list, ts_flags: list[bool]) -> list[Optional[float]]:
        """Thermal corrections (kcal/mol) for `structures`, each computed once
        (cached by geometry); None where the Hessian failed."""
        from mepd.cli_common import _fork_map

        keys = [self._key(s, t) for s, t in zip(structures, ts_flags)]
        todo = {}
        for k, s, t in zip(keys, structures, ts_flags):
            if k not in self._cache and k not in todo:
                todo[k] = (s, t)

        def run(item):
            s, t = item
            try:
                return self._correction(s, t)
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
