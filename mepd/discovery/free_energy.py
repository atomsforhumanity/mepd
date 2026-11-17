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
A TS counts as what it was optimized as. Its file says which atoms that was
(the comment's `optimized_on=`, written by every TS search of the explorer:
optimized_on_comment): the whole box, which then counts as one molecule, or
the subsystem of its step (network_expansion.subsystem), which counts alone
with each molecule of the rest apart. Nothing is read from distances between
the two: a subsystem's TS is put back among spectators that stay where the
reference had them, so its file is no stationary point of the box and an
atom of it can sit within bonding distance of a spectator (isoprene + methyl
acrylate: O-C 1.76 Angstrom, forces of 12-19 eV/Angstrom on the box).

A TS file without that record (written before it existed, or by another
tool) counts as before, and its step's record says so: which atoms react
comes from the step's two ends in the TS's atom order (its IRC's): the
connected pieces of the union of the ends' bond graphs that contain a bond
made or broken (ts_groups). Without ends, or with ends whose bonds don't
bracket the TS's, the TS counts as one molecule. That reading gave one TS
different molecules in different runs (a bond of both IRC ends stretched
past 1.25 x the covalent radii at the TS; a spectator in contact): corrections
10-31 kcal/mol apart.

A TS with spectators was optimized on its reacting complex alone (the
explorer's subsystem, network_expansion.subsystem), so it counts apart only:
G = its reacting complex's and each spectator's electronic energy and
thermal correction, relative to the TS's own electronic energy (`energy`,
which the kinetics add the correction to). It is not also counted bound (the
whole complex as one molecule): that needed a Hessian of the whole box at a
geometry that is no saddle of it.

A TS's Hessian is taken at the saddle, not at the geometry its search stopped at. The searches stop when the
largest atomic force is below 0.001 eV/Angstrom (TSInputs.fmax, which every TS search of the explorer runs to).
Along a mode of ~30 cm^-1 that leaves the geometry ~0.03 Angstrom from the saddle, and where such a mode is
anharmonic its frequency there is another: one acetic acid dehydration TS, found by three runs with energies
within 2e-7 Hartree, had its soft mode at 31, 32 and 7i cm^-1 and so corrections 1.5 kcal/mol apart at 1500 K. The
TS is therefore first converged to HESSIAN_FMAX with the engine's own TS optimizer (the three corrections then
agree to 0.01), and what that changes its electronic energy by counts in the correction.
  * Only a structure a TS optimization produced: one whose file says what it was optimized on (optimized_on),
    and of it the molecule that was optimized. Not a path's highest image kept in place of a TS, which can be a
    minimized end of its piece (11 of 2258 peak files of twelve runs are) -- a minimum passes any force test,
    and the TS optimizer walks it up to the nearest saddle (ethanol: a rotor's, 0.66 Angstrom and +3.7 kcal/mol
    away). Not a file without the record, a fragment at its geometry in a complex, another level's geometry.
  * Only if what it converges to is still that TS: its energy within SAME_SADDLE_KCAL of the given structure's
    (the correction is added to the energy the TS came with, which is also what its IRC, its steering single
    point and its identity were made at) and exactly one negative mode. Otherwise the Hessian is taken at the
    geometry given and the notes say why. In 5078 IRC-verified TSs of twelve runs 7 change by more: one ends on
    the saddle of another motion (602i -> 284i cm^-1, 4.7 kcal/mol lower), six had been left 0.06-0.5 kcal/mol
    above their saddle by a search that stopped on a flat mode.
  * Only where the engine's Hessian is analytic (g-xTB through axtb, the explorer's): there the optimizer has
    the exact curvature and a step costs a fraction of the Hessian; with another engine nothing changes.

The rotational symmetry number and linearity come from xtb's analysis of the
geometry (mepd.thermo.frequencies, a GFN2-xTB run), which is cheap at these
sizes; where GFN2-xTB's SCF fails at that geometry, from a GFN-FF run's.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np

HARTREE_TO_KCAL = 627.5094740631
R_KCAL = 1.987204259e-3   # kcal/mol/K


BOHR_PER_ANGSTROM = 1.8897259886
EV_PER_HARTREE = 27.211386245988
#: A TS's Hessian is taken where its largest atomic force is below this (eV/Angstrom); see the module docstring.
#: g-xTB's gradient noise at the engine's SCF accuracy is ~1e-5 eV/Angstrom (axtb_backend.ACCURACY).
HESSIAN_FMAX = 1e-4
#: The largest energy change (kcal/mol) converging a TS further may make for its Hessian to be taken there.
SAME_SADDLE_KCAL = 0.05


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


OPTIMIZED_ON = "optimized_on="


def optimized_on_comment(atoms=None) -> str:
    """The word a TS file's comment line carries to say what the TS was optimized on: `atoms` (indices in
    the file's order) of a subsystem, or the whole box (None)."""
    return OPTIMIZED_ON + ("whole" if atoms is None else ",".join(str(int(i)) for i in sorted(atoms)))


def unrecorded(structure):
    """`structure` without a record of what it was optimized on: what a file's writer starts from. A structure
    made from a TS file's (an IRC frame, its minimized end -- a species --, an image interpolated from that)
    carries that file's comment along, and the record belongs to the TS's own file only."""
    extras = dict(getattr(structure, "extras", None) or {})
    lines = [" ".join(w for w in str(line).split() if not w.startswith(OPTIMIZED_ON))
             for line in extras.get("xyz_comments") or []]
    if lines == list(extras.get("xyz_comments") or []):
        return structure
    return structure.model_copy(update={"extras": {**extras, "xyz_comments": [x for x in lines if x]}})


def optimized_on(structure) -> Optional[list[int]]:
    """The atoms a TS read from its file was optimized on (every atom: the whole box), or None when the file
    does not say. The last record of the comment counts: a structure made from another file's (a rotamer
    guess, a TS optimized from one) carries that file's comment in front of what its own writer adds."""
    n = len(structure.symbols)
    value = None
    for line in (getattr(structure, "extras", None) or {}).get("xyz_comments") or []:
        for word in str(line).split():
            if word.startswith(OPTIMIZED_ON):
                value = word[len(OPTIMIZED_ON):]
    if value is None:
        return None
    if value == "whole":
        return list(range(n))
    try:
        atoms = sorted({int(i) for i in value.split(",")})
    except ValueError:
        return None
    return atoms if atoms and 0 <= atoms[0] and atoms[-1] < n else None


def counted_as(structure, ends=None, record: bool = True) -> tuple[Optional[list[list[int]]], str]:
    """(a TS's molecules as the free energies count them -- what it was optimized on first, then each molecule
    of the rest -- or None: one molecule; where that comes from: "optimized_on" (the TS file's record),
    "ends" (no record, or `record` False: ts_groups of its step's ends) or "none")."""
    symbols = [str(s) for s in structure.symbols]
    xyz = np.asarray(structure.geometry, dtype=float).reshape(-1, 3) / BOHR_PER_ANGSTROM
    atoms = optimized_on(structure) if record else None
    if atoms is not None:
        rest = [i for i in range(len(symbols)) if i not in set(atoms)]
        spectators = [[rest[k] for k in g] for g in molecules([symbols[i] for i in rest], xyz[rest])] if rest else []
        return [atoms] + spectators, "optimized_on"
    if ends is None:
        return None, "none"
    return ts_groups(symbols, xyz, [np.asarray(e.geometry) / BOHR_PER_ANGSTROM for e in ends]), "ends"


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


class Correction(float):
    """A TS's thermal correction (kcal/mol) with what it counted: `counted` = {"molecules": the atom count
    of each molecule counted, "from": where they come from (counted_as), "saddle": what the saddle step did
    ("used", "not needed: ...", "not run: ...", "refused (...): ...", "failed (...): ..."), and
    "one_molecule_because" where the molecules could not be counted apart}. A float everywhere else; saved and
    sent between processes with it."""

    def __new__(cls, value, counted=None):
        self = super().__new__(cls, value)
        self.counted = counted
        return self


def counted_record(files: dict, species_ends: tuple, cached=None) -> Optional[dict]:
    """For a step's record: what its TS's correction counted, as saved with the correction (`cached`, the
    cache's value: Correction.counted). Nothing is computed here. A correction saved without it (an older
    state's, a known run's) gets the molecules its file or its step's ends name now, marked
    "not_saved_with_the_correction": the correction may have counted one molecule where their charges were not
    clear. None without a TS file."""
    from pathlib import Path

    from qcdata import Structure

    if getattr(cached, "counted", None):
        return dict(cached.counted)
    ts = files.get("ts")
    if not ts or not Path(ts).exists():
        return None
    structure = Structure.open(str(ts))
    groups, source = counted_as(structure, step_ends(files, species_ends) if optimized_on(structure) is None
                                else None)
    return {"molecules": [len(g) for g in groups] if groups else [len(structure.symbols)], "from": source,
            "not_saved_with_the_correction": True}


class FreeEnergies:
    def __init__(self, engine, temperature: float, standard_state: str = "1M", workers: int = 1,
                 by_record: bool = True):
        """`by_record` False: a TS's molecules from its step's ends, whatever its file says it was optimized on
        -- for a run without free energies, which uses this class only to tell a TS's spectators (its steering
        single points) and must do so as it did before the record existed."""
        self.by_record = bool(by_record)
        self.engine, self.T, self.standard_state = engine, float(temperature), standard_state
        self.workers = max(1, int(workers))
        self._cache: dict[str, float] = {}
        self.notes: list[str] = []
        # A folder to keep each molecule's frequencies in, by geometry (None: nothing kept): the same structures at
        # another temperature then need no Hessian (mepd.discovery.parent). Not used for a TS converged here (_saddle).
        self.frequency_cache = None

    def _groups(self, structure, ts: bool, ends) -> Optional[list[list[int]]]:
        """A TS's groups (counted_as: its file's record, else its step's two ends), else None."""
        return counted_as(structure, ends, self.by_record)[0] if ts else None

    def _key(self, structure, ts: bool, ends=None) -> str:
        xyz = np.round(np.asarray(structure.geometry, dtype=float), 4) + 0.0
        groups = self._groups(structure, ts, ends)     # a TS with spectators apart; one without: as before
        # (a TS whose file says what it was optimized on is converged before its Hessian, _saddle: another number
        # than one saved for the same geometry before that record existed -- an older state's, a known run's)
        said = ts and self.by_record and optimized_on(structure) is not None
        blob = repr((list(map(str, structure.symbols)), xyz.tolist(), int(structure.charge),
                     int(structure.multiplicity), ts) + ((groups,) if groups and len(groups) > 1 else ())
                    + (("at its saddle",) if said else ()))
        return hashlib.sha1(blob.encode()).hexdigest()

    def _saddle(self, node):
        """(a converged TS of the engine, converged to HESSIAN_FMAX; its energy change in kcal/mol), else
        (`node`, 0.0): see the module docstring. For a structure a TS optimization produced only (_correction
        decides that from its file's record)."""
        import mepd
        from mepd.inputs import TSInputs

        self._saddle_said = "not run: the engine's Hessian is not analytic"
        if not getattr(self.engine, "analytic_hessian", False):
            return node, 0.0
        try:
            g = np.asarray(self.engine.compute_gradients([node])[0], dtype=float).reshape(-1, 3)
            fmax = float(np.linalg.norm(g, axis=1).max()) * EV_PER_HARTREE * BOHR_PER_ANGSTROM
            if not HESSIAN_FMAX < fmax <= 2 * TSInputs.fmax:
                self._saddle_said = ("not needed: forces already below it" if fmax <= HESSIAN_FMAX
                                     else "not run: forces above a converged TS's")
                return node, 0.0
            tight = mepd.find_ts(node, self.engine, fmax=HESSIAN_FMAX, maxiter=200, continue_unconverged=0,
                                 check_frequencies=False)
            e0, e1 = mepd.energies([node, tight], self.engine)
            # (its Hessian, once: kept on `tight`, which _one reads)
            tight.frequencies = mepd.frequencies(tight, self.engine)
            negative = int((np.asarray(tight.frequencies.wavenumbers_cm) < 0).sum())
        except Exception as exc:   # the Hessian at the geometry given, as before
            self._saddle_said = f"failed ({type(exc).__name__}): the geometry given"
            self.notes.append(f"a TS could not be converged further before its Hessian ({type(exc).__name__})")
            return node, 0.0
        moved = (float(e1) - float(e0)) * HARTREE_TO_KCAL
        if abs(moved) > SAME_SADDLE_KCAL or negative != 1:
            why = f"{moved:+.2f} kcal/mol" if abs(moved) > SAME_SADDLE_KCAL else f"{negative} negative modes"
            self._saddle_said = f"refused ({why}): the geometry given"
            self.notes.append(f"a TS converged further is another structure ({why}); its Hessian is at the "
                              "geometry given")
            return node, 0.0
        self._saddle_said = "used"
        return tight, moved

    def _one(self, symbols, xyz_A, charge, mult, ts, saddle: bool = False) -> float:
        """Thermal correction (kcal/mol) of one molecule. `saddle`: a TS that a TS optimization produced on
        these atoms (_saddle)."""
        import mepd
        from qcdata import Structure

        from mepd.nodes.node import StructureNode
        from mepd.thermo import frequencies as xtb_analysis, qrrho

        if len(symbols) == 1:
            return qrrho([], symbols, xyz_A, self.T, multiplicity=mult, standard_state=self.standard_state)["g_corr"]
        kept = None
        if self.frequency_cache is not None and not (ts and saddle):
            import json
            from pathlib import Path

            blob = repr((list(symbols), (np.round(np.asarray(xyz_A, dtype=float), 5) + 0.0).tolist(), int(charge), int(mult)))
            kept = Path(self.frequency_cache) / f"{hashlib.sha1(blob.encode()).hexdigest()}.json"
            try:
                hit = json.loads(kept.read_text())
                return qrrho(hit["w"], symbols, xyz_A, self.T, rotational_number=hit["rotational_number"],
                             linear=hit["linear"], multiplicity=mult, ts=ts, standard_state=self.standard_state)["g_corr"]
            except (OSError, ValueError, KeyError):
                pass
        try:
            shape = xtb_analysis(symbols, xyz_A, charge=charge, multiplicity=mult)
        except RuntimeError:
            # GFN2-xTB's SCF can fail at a geometry (a TS): the same analysis of the geometry in a GFN-FF run (no
            # SCF; --bhess relaxes each a little, so the two agreed on 983 of 985 structures, not all)
            shape = xtb_analysis(symbols, xyz_A, charge=charge, multiplicity=mult, method_args=["--gfnff"])
        node = StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(xyz_A) * BOHR_PER_ANGSTROM,
                                                 charge=int(charge), multiplicity=int(mult)))
        moved = 0.0
        if ts and saddle:
            node, moved = self._saddle(node)
            xyz_A = np.asarray(node.structure.geometry, dtype=float).reshape(-1, 3) / BOHR_PER_ANGSTROM
        freqs = getattr(node, "frequencies", None)      # (_saddle's, of the structure it converged)
        if freqs is None:
            freqs = mepd.frequencies(node, self.engine)
        w = sorted(freqs.wavenumbers_cm.tolist(), key=abs)
        if len(w) == 3 * len(symbols):          # with the translations and rotations: drop them
            w = w[(5 if shape["linear"] else 6):]
        if kept is not None:
            kept.parent.mkdir(parents=True, exist_ok=True)
            tmp = kept.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"w": sorted(w), "rotational_number": shape["rotational_number"],
                                       "linear": bool(shape["linear"])}))
            os.replace(tmp, kept)
        return moved + qrrho(sorted(w), symbols, xyz_A, self.T, rotational_number=shape["rotational_number"],
                             linear=shape["linear"], multiplicity=mult, ts=ts,
                             standard_state=self.standard_state)["g_corr"]

    def parts(self, structure, ts: bool, ends=None) -> list[tuple[list[int], int, int]]:
        """(atoms, charge, multiplicity) of each molecule the species counts
        as. A TS: what it was optimized on first, then its spectators (counted_as:
        its file's record, else its step's two `ends`); one entry without either. In a charged or open-shell complex each
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
            if groups is None and ends is not None and not (self.by_record and optimized_on(structure) is not None):
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

    def _correction(self, structure, ts: bool, ends=None, energy: Optional[float] = None):
        """Thermal correction (kcal/mol), relative to the structure's electronic energy: `energy` (Hartree), or
        computed when not given."""
        symbols = [str(s) for s in structure.symbols]
        xyz = np.asarray(structure.geometry, dtype=float) / BOHR_PER_ANGSTROM
        said = len(self.notes)
        parts = self.parts(structure, ts, ends)
        # (what a TS optimization produced: the first molecule of a TS whose file says what it was optimized on)
        on = optimized_on(structure) if ts and self.by_record else None
        made = on is not None and sorted(parts[0][0]) == on
        self._saddle_said = None

        def result(value):
            """The correction with what it counted (a TS's; see Correction)."""
            if not ts:
                return float(value)
            groups, source = counted_as(structure, ends, self.by_record)
            counted = {"molecules": [len(p[0]) for p in parts], "from": source,
                       "saddle": self._saddle_said or "not run: not what a TS optimization produced"}
            if groups and len(groups) > 1 and len(parts) == 1:
                counted["one_molecule_because"] = (self.notes[said] if len(self.notes) > said
                                                   else "its molecules are not clear")
            return Correction(value, counted)

        if len(parts) == 1:
            return result(self._one(symbols, xyz, parts[0][1], parts[0][2], ts, *((True,) if made else ())))
        import mepd
        from qcdata import Structure

        from mepd.nodes.node import StructureNode

        total, frags = 0.0, []
        for n, (p, qi, mi) in enumerate(parts):
            sy = [symbols[k] for k in p]
            # a TS's reacting complex comes first
            total += self._one(sy, xyz[p], qi, mi, ts and n == 0, *((True,) if made and n == 0 else ()))
            frags.append(StructureNode(structure=Structure(symbols=sy, geometry=xyz[p] * BOHR_PER_ANGSTROM,
                                                           charge=qi, multiplicity=mi)))
        # The molecules apart: their electronic energies too, not the complex's (whose binding would
        # otherwise count on top of the separated molecules' entropy). Each at its geometry in the complex.
        if energy is None:
            energy = mepd.energies([StructureNode(structure=structure)], self.engine)[0]
        apart = sum(mepd.energies(frags, self.engine)) - float(energy)
        return result(total + apart * HARTREE_TO_KCAL)

    def corrections(self, structures: list, ts_flags: list[bool], ends: Optional[list] = None,
                    energies: Optional[list] = None) -> list[Optional[float]]:
        """Thermal corrections (kcal/mol) for `structures`, each computed once
        (cached by geometry); None where the Hessian failed. `ends`: per
        structure, a TS's step's two ends (see parts) or None. `energies`: per
        structure, its electronic energy (Hartree) or None (see _correction)."""
        from mepd.cli_common import _fork_map

        ends = ends if ends is not None else [None] * len(structures)
        energies = energies if energies is not None else [None] * len(structures)
        keys = [self._key(s, t, e) for s, t, e in zip(structures, ts_flags, ends)]
        todo = {}
        for k, s, t, e, en in zip(keys, structures, ts_flags, ends, energies):
            if k not in self._cache and k not in todo:
                todo[k] = (s, t, e, en)

        def run(item):
            s, t, e, en = item
            try:
                return self._correction(s, t, e, en)
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
                # (a TS saved by an earlier version as apart and bound: the lower, as it counted then)
                out.append(float(min(v) if isinstance(v, tuple) else v))
        return out
