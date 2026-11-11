"""Molecule-level kinetics of an explorer network.

The explorer finds its steps between whole boxes (every species an arrangement of the seed's atoms). Here they are
read as reactions between molecules, with mass action at real concentrations: a box's molecules are its connected
pieces (free_energy.molecules), each keyed by its stereo SMILES (the explorer's own identity, canonicalised so a
mirror image's key can be made); a step a -> b becomes the reaction of the molecules whose bonds change (with every
molecule of either end those atoms are in, and a molecule whose key alone changed), the others being spectators.

  * In a neutral box every molecule is a species, a radical too (keyed by its bond graph: molecule_parts). What
    can't be counted as separate molecules is one species of its own, a complex kept together as in the box model
    (complex_key): the whole of a charged box of several molecules (which one holds the charge is not known
    without a calculation), a molecule whose key can't be made. A complex reacts first order where it is one, at
    its box's free energy, without its molecules' dilution (a limitation for ion pairs), and carries its
    molecules' keys (a channel is credited from it only when it holds the channel's whole product set).
  * A step with spectators whose TS was computed on the whole box (no `<ts>_ends.xyz`: its subsystem's TS landed in
    a spectator's room, network_expansion.Subsystem.clashes) keeps the spectators it touches only when an IRC
    checked it (files["verified"] not False): they are then on both sides of the reaction (a neighbour taking
    part); unchecked, it is left out (the rule of mixture.settle's fix: a TS that fails without its neighbours keeps
    them only if it connects).
  * A step whose TS search failed, kept at its path's highest image (files["ts_kind"] "path_peak"), is left out:
    that image is no saddle (a doubled formic acid run's one, 20 kcal/mol up at g-xTB for HCOOH-assisted
    dehydration, lies 2.5 kcal/mol above the two molecules apart at wB97X-D3BJ/def2-TZVP).
  * Molecule free energies: each molecule at its lowest over the boxes holding it (lowest: every box at or above
    the sum of its molecules', a linear programme). A reaction's TS on this scale is the box TS with the
    spectators taken out.
  * One reaction found in several boxes is one TS when its TS energies are within DEDUP_KCAL and the reacting atoms'
    sorted distances agree (mixture._same_distances); mirror-image twins are never merged with each other.

Free energies stay at their standard state (1 M, or 1 atm); concentrations are in its units. Kinetics:
mepd.microkinetics (mass action, any order). The expansion criteria are Bensberg & Reiher's (Isr. J. Chem. 63,
e202200123 (2023)), both relative to the conversion X (the fraction of the seed's molecules consumed), so they work
at low conversion: a molecule's own reactions are probed once its integrated net flux (sum over its reactions of
|nu| x the integral of |forward - backward|) reaches tau x X; an encounter of two molecules once c_max(m) c_max(n)
(fractions of the starting amount) reaches tau x X; both with an absolute floor FLOOR.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

H_KCAL = 627.509474
DEDUP_KCAL = 0.5
FIT_SLACK_KCAL = 1.0
FLOOR = 1e-6
CONTACT = 1.5            # x the sum of covalent radii: a spectator this close to a reacting atom in a whole-box TS
CLOSED_HARTREE = 0.5     # a step's TS this far above both ends is a closed one (TS + 1 Hartree)
MAX_BOX_ATOMS = 50       # the default seed box (two copies of each molecule) up to this many atoms


def canonical(key: str, mirror: bool = False) -> str:
    """RDKit's canonical isomeric SMILES of `key` (of its mirror image: every tetrahedral centre inverted)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(key)
    if m is None:
        return key.replace("@@", "\0").replace("@", "@@").replace("\0", "@") if mirror else key
    if mirror:
        for at in m.GetAtoms():
            tag = at.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                at.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                at.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(m)


def _xyz(structure):
    from qcref.constants import BOHR_TO_ANGSTROM

    return [str(s) for s in structure.symbols], np.asarray(structure.geometry, dtype=float) * BOHR_TO_ANGSTROM


COMPLEX_CHARGED = "charged group"
COMPLEX_KEY = "key failed"


def complex_key(reason: str, keys: Sequence[str]) -> str:
    """The key of a complex: several molecules kept together as one species (their charges and spins unknown
    without a calculation), or one whose key can't be made; `keys`: its molecules' own keys."""
    return "{" + reason + "}" + ".".join(sorted(keys))


def complex_info(key: str) -> Optional[tuple[str, list]]:
    """(reason, its molecules' keys) of a complex's key; None for a molecule's."""
    if not key.startswith("{") or "}" not in key:
        return None
    reason, rest = key[1:].split("}", 1)
    return reason, rest.split(".") if rest else []


def _graph_key(symbols, atoms, bonds) -> str:
    """A key for a molecule no SMILES can be made of: its formula and a hash of its bond graph."""
    import hashlib
    from collections import Counter

    atoms = sorted(atoms)
    nb = {i: sorted(symbols[j] for e in bonds if i in e for j in e if j != i) for i in atoms}
    lab = {i: symbols[i] + "".join(nb[i]) for i in atoms}
    for _ in range(3):
        lab = {i: lab[i] + "(" + ",".join(sorted(lab[j] for e in bonds if i in e for j in e if j != i)) + ")"
               for i in atoms}
    formula = "".join(f"{s}{n}" for s, n in sorted(Counter(symbols[i] for i in atoms).items()))
    return formula + "#" + hashlib.sha1("|".join(sorted(lab.values())).encode()).hexdigest()[:10]


RADICAL = "radical:"


def radical_key(symbols, atoms, bonds) -> str:
    """A neutral open-shell molecule's key: its bond graph alone (RADICAL + _graph_key), as no bond orders can be
    read from its geometry reliably (formyloxy came out [O][CH][O] at every C-O length tried, never [O]C=O)."""
    return RADICAL + _graph_key(symbols, atoms, bonds)


def smiles_graph_key(smiles: str) -> Optional[str]:
    """The graph key (_graph_key) of a SMILES' molecule, hydrogens added: what a geometry's radical_key is for it
    (a channel's radical product matched without bond orders); None if it can't be read."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = {(max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
             for b in m.GetBonds()}
    return _graph_key(symbols, range(len(symbols)), bonds)


def molecule_parts(structure) -> list[tuple[frozenset, str]]:
    """[(atoms, key)] of a box's species. In a neutral box each molecule is one: a closed-shell one keyed by its
    stereo SMILES, an open-shell one (a radical; odd electron count) by its bond graph (radical_key). The box's
    free energy is fitted over boxes, not computed per molecule, so a radical's own spin state never enters, and
    each radical is diluted and recombines second order as any molecule does. What can't be counted as separate
    molecules stays together as one species, a complex (complex_key), as the box model keeps it: the whole of a
    charged box of several molecules (which one holds the charge is not known without a calculation, and an ion
    pair doesn't separate freely in the gas phase), and a molecule whose key can't be made (reason COMPLEX_KEY)."""
    from mepd.discovery.free_energy import _electrons, bonds, molecules
    from mepd.discovery.mixture import _sub, molecule_key

    symbols, xyz = _xyz(structure)
    groups = molecules(symbols, xyz)
    charge, mult = int(structure.charge), int(structure.multiplicity)
    b = bonds(symbols, xyz)
    odd = lambda g: sum(_electrons(symbols[i]) for i in g) % 2 == 1

    def key_of(g, q, m):
        if q == 0 and odd(g):
            return radical_key(symbols, g, b), True
        try:
            k = molecule_key(_sub(structure, g, q, m))
        except Exception:
            k = None
        if k:
            return canonical(k), True
        return _graph_key(symbols, g, b), False

    if len(groups) == 1:
        k, ok = key_of(groups[0], charge, mult)
        return [(frozenset(groups[0]), k if ok else complex_key(COMPLEX_KEY, [k]))]
    if charge != 0:
        keys = [key_of(g, 0, 2 if odd(g) else 1)[0] for g in groups]
        return [(frozenset(i for g in groups for i in g), complex_key(COMPLEX_CHARGED, keys))]
    out, failed = [], []
    for g in groups:
        k, ok = key_of(g, 0, 2 if odd(g) else 1)
        (out.append((frozenset(g), k)) if ok else failed.append((frozenset(g), complex_key(COMPLEX_KEY, [k]))))
    return out + failed


def bonds_of(structure) -> set:
    from mepd.discovery.free_energy import bonds

    symbols, xyz = _xyz(structure)
    return bonds(symbols, xyz)


def split(pa, pb, ba: set, bb: set) -> tuple:
    """A step's reaction from its ends' molecules (`pa`, `pb`: molecule_parts) and bonds (`ba`, `bb`):
    ("ok", reactant keys, product keys, reacting atoms, spectator atom sets) or (why not,)."""
    changed = set()
    for i, j in ba ^ bb:
        changed |= {i, j}
    same = {atoms: key for atoms, key in pb}
    for atoms, key in pa:
        if atoms in same and same[atoms] != key:
            changed |= set(atoms)
    if not changed:
        return ("no_reaction",)
    react, grow = set(changed), True
    while grow:
        grow = False
        for atoms, _ in list(pa) + list(pb):
            if atoms & react and not atoms <= react:
                react |= atoms
                grow = True
    r = sorted(key for atoms, key in pa if atoms <= react)
    p = sorted(key for atoms, key in pb if atoms <= react)
    spa = sorted((key, atoms) for atoms, key in pa if not atoms <= react)
    spb = sorted((key, atoms) for atoms, key in pb if not atoms <= react)
    if [k for k, _ in spa] != [k for k, _ in spb]:
        return ("spectator_mismatch",)
    if r == p:
        return ("no_reaction",)
    return ("ok", r, p, sorted(react), [atoms for _, atoms in spa])


@dataclass
class MoleculeNetwork:
    names: list                    # molecule keys
    g: np.ndarray                  # kcal/mol, relative to the seed box's molecules
    reactions: list                # {"r": [idx], "p": [idx], "g_ts": kcal, "step": original step, "twin": bool}
    seed: list                     # the seed box's molecules (indices, with repeats)
    counts: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)

    def network(self, absorbing: Sequence[int] = ()):
        from mepd.microkinetics import Network, Step

        g = np.array(self.g, dtype=float)
        g[list(absorbing)] -= H_KCAL
        return Network(list(self.names), g, [Step(list(x["r"]), list(x["p"]), float(x["g_ts"]), str(x["step"]))
                                             for x in self.reactions])

    def c0(self, concentration: float = 1.0, copies: int = 1) -> np.ndarray:
        """Starting amounts: each of the seed box's molecules at `concentration` per copy of the box's
        composition (`copies`: how many times the box repeats the given molecules)."""
        c = np.zeros(len(self.names))
        for i in self.seed:
            c[i] += float(concentration) / max(1, int(copies))
        return c

    def record(self) -> dict:
        return {"names": list(self.names), "g_kcal": [float(x) for x in self.g], "seed": list(self.seed),
                "reactions": [{"r": list(x["r"]), "p": list(x["p"]), "g_ts_kcal": float(x["g_ts"]),
                               "step": int(x["step"]), "twin": bool(x["twin"]),
                               **({"with_neighbours": True} if x.get("with_neighbours") else {})}
                              for x in self.reactions],
                "counts": dict(self.counts), "fit": dict(self.fit)}

    @classmethod
    def from_record(cls, rec: dict) -> "MoleculeNetwork":
        return cls(list(rec["names"]), np.array(rec["g_kcal"], dtype=float),
                   [{"r": x["r"], "p": x["p"], "g_ts": x["g_ts_kcal"], "step": x["step"], "twin": x["twin"]}
                    for x in rec["reactions"]], list(rec["seed"]), dict(rec.get("counts") or {}),
                   dict(rec.get("fit") or {}))


def build(n_species: int, parts: Callable, bonds: Callable, steps: Sequence, energies: Sequence[float],
          virtual_of: Sequence[int] = (), twin_of: Sequence[int] = (), found_twins: Sequence = (),
          ts_info: Optional[Callable] = None) -> MoleculeNetwork:
    """The molecule network of a box network: `n_species` boxes found (parts(k), bonds(k): their molecules and
    bonds), then the virtual mirror images (`virtual_of`: the found box each mirrors); `steps` (kinetics.Step,
    TS Hartree; the found steps first, then the twin of step twin_of[t] for each t), `energies` (Hartree, every
    box); `found_twins`: pairs of found steps that are each other's mirror image. `ts_info(m)` for a found step:
    (TS Structure or None, whether its TS has an ends file, whether an IRC checked it)."""
    n_orig = len(steps) - len(twin_of)
    orig_of = list(range(n_orig)) + list(twin_of)
    n_all = n_species + len(virtual_of)

    def box(k):
        return k if k < n_species else virtual_of[k - n_species]

    cache: dict = {}

    def parts_of(k):
        if k not in cache:
            p = parts(box(k))
            cache[k] = p if (p is None or k < n_species) else [(a, canonical(key, mirror=True)) for a, key in p]
        return cache[k]

    counts: dict = {}

    def count(what):
        counts[what] = counts.get(what, 0) + 1

    found = []
    info_cache: dict = {}
    for j, st in enumerate(steps):
        e_a, e_b = float(energies[st.a]), float(energies[st.b])
        if (st.files or {}).get("closed") or float(st.ts_energy) - max(e_a, e_b) > CLOSED_HARTREE:
            continue
        count("open_box_steps")
        if (steps[orig_of[j]].files or {}).get("ts_kind") == "path_peak":
            # its TS search failed (kept at its path's highest image): no saddle, no barrier to give a rate
            count("left_out_ts_search_failed")
            continue
        s = split(parts_of(st.a), parts_of(st.b), bonds(box(st.a)), bonds(box(st.b)))
        if s[0] != "ok":
            count("left_out_" + s[0])
            continue
        _, r, p, react, spectators = s
        m = orig_of[j]
        x = {"j": j, "step": m, "r": list(r), "p": list(p), "atoms": react, "a": st.a, "b": st.b,
             "g_ts_h": float(st.ts_energy), "twin": j >= n_orig}
        if spectators and ts_info is not None:
            if m not in info_cache:
                info_cache[m] = ts_info(m)
            ts, has_ends, verified = info_cache[m][:3]
            if not has_ends:
                if not verified:
                    count("left_out_whole_box_ts_unchecked")
                    continue
                near = _neighbours(ts, react, spectators) if ts is not None else []
                if near:
                    keys = dict((a, key) for a, key in parts_of(st.a))
                    x["r"] = sorted(x["r"] + [keys[a] for a in near])
                    x["p"] = sorted(x["p"] + [keys[a] for a in near])
                    x["atoms"] = sorted(set(react) | {i for a in near for i in a})
                    x["with_neighbours"] = True
        found.append(x)
    # molecule free energies
    used = sorted({x["a"] for x in found} | {x["b"] for x in found} | {0})
    used = [k for k in used if parts_of(k) is not None]
    names = sorted({key for k in used for _, key in parts_of(k)})
    col = {key: c for c, key in enumerate(names)}
    A = np.zeros((len(used), len(names)))
    y = np.array([float(energies[k]) * H_KCAL for k in used])
    for row, k in enumerate(used):
        for _, key in parts_of(k):
            A[row, col[key]] += 1
    y0 = y[used.index(0)] if 0 in used else 0.0
    G = lowest(A, y - y0)
    resid = (y - y0) - A @ G
    fit_box = dict(zip(used, A @ G))
    for x in found:
        x["g_ts"] = x["g_ts_h"] * H_KCAL - y0 - (fit_box[x["a"]] - sum(G[col[k]] for k in x["r"]))
    # one TS per reaction found in several boxes
    twins = {frozenset(map(int, p)) for p in found_twins}
    dist: dict = {}

    def dists(x):
        if x["step"] not in dist:
            if x["step"] not in info_cache:
                info_cache[x["step"]] = ts_info(x["step"]) if ts_info is not None else (None, False, False)
            ts = info_cache[x["step"]][0]
            if ts is None:
                dist[x["step"]] = None
            else:
                from mepd.discovery.mixture import _distances, _sub

                dist[x["step"]] = _distances(_sub(ts, x["atoms"], 0, 1))
        return dist[x["step"]]

    from mepd.discovery.mixture import _same_distances

    groups: dict = {}
    kept, kept_steps = [], set()
    for x in sorted((x for x in found if not x["twin"]), key=lambda x: x["g_ts"]):
        fwd, rev = (tuple(x["r"]), tuple(x["p"])), (tuple(x["p"]), tuple(x["r"]))
        key = min(fwd, rev)
        dup = False
        for y_ in groups.get(key, []):
            if frozenset((x["step"], y_["step"])) in twins or abs(x["g_ts"] - y_["g_ts"]) >= DEDUP_KCAL:
                continue
            dx, dy = dists(x), dists(y_)
            if dx is not None and dy is not None and _same_distances(dx, dy):
                dup = True
                break
        if dup:
            count("merged_duplicates")
            continue
        groups.setdefault(key, []).append(x)
        kept.append(x)
        kept_steps.add(x["step"])
    for x in found:
        if x["twin"]:
            if x["step"] in kept_steps:
                kept.append(x)
            else:
                count("merged_duplicates")
    counts["reactions"] = len(kept)
    reactions = [{"r": [col[k] for k in x["r"]], "p": [col[k] for k in x["p"]], "g_ts": float(x["g_ts"]),
                  "step": x["step"], "twin": x["twin"], **({"with_neighbours": True} if x.get("with_neighbours")
                                                          else {})} for x in kept]
    seed = [col[key] for _, key in (parts_of(0) or []) if key in col]
    counts["complexes"] = sum(1 for key in names if complex_info(key) is not None)
    return MoleculeNetwork(names, np.asarray(G, dtype=float), reactions, seed, counts,
                           {"boxes": len(used), "above_their_molecules": int((resid > FIT_SLACK_KCAL).sum()),
                            "resid_max_kcal": float(np.abs(resid).max()) if len(resid) else 0.0})


def _neighbours(ts, react, spectators) -> list:
    """The spectator molecules (atom sets) with an atom within CONTACT x covalent radii of a reacting atom in `ts`."""
    from rdkit import Chem

    symbols, xyz = _xyz(ts)
    table = Chem.GetPeriodicTable()
    rc = np.array([table.GetRcovalent(s) for s in symbols])
    out = []
    for atoms in spectators:
        a = sorted(atoms)
        d = np.linalg.norm(xyz[react][:, None] - xyz[a][None], axis=-1)
        if (d < CONTACT * (rc[react][:, None] + rc[a][None])).any():
            out.append(atoms)
    return out


@dataclass
class MoleculeKinetics:
    final: np.ndarray
    c_max: np.ndarray
    net_flux: np.ndarray           # sum over its reactions of |nu| x integral |forward - backward| dt
    conversion: float              # fraction of the seed's molecules consumed
    total0: float                  # the starting amount, summed


def simulate(net: MoleculeNetwork, c0, temperature: float, time_s: float,
             absorbing: Sequence[int] = ()) -> MoleculeKinetics:
    from mepd.microkinetics import _rates, simulate as mk_simulate

    c0 = np.asarray(c0, dtype=float)
    total0 = float(c0.sum())
    if not net.reactions:
        return MoleculeKinetics(c0.copy(), c0.copy(), np.zeros(len(c0)), 0.0, total0)
    mk = net.network(absorbing)
    res = mk_simulate(mk, c0, temperature, time_s)
    kf, kr = mk.rate_constants(temperature)
    r = np.array([np.subtract(*_rates(c, kf, kr, mk.steps)) for c in res.c])     # [time, step]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    # |the integrated net rate| (the solver's own extent: exact) plus what the net rate's sign changes add, read off
    # the output times (zero when a step runs one way throughout)
    integral = np.abs(res.extent) + np.maximum(trapz(np.abs(r), res.times, axis=0)
                                               - np.abs(trapz(r, res.times, axis=0)), 0.0)
    gross = np.zeros((len(net.names), len(mk.steps)))
    for j, st in enumerate(mk.steps):
        for i in st.reactants:
            gross[i, j] += 1.0
        for i in st.products:
            gross[i, j] += 1.0
    shuttle = np.abs(mk.nu) < 0.5          # on both sides: not consumed, no flux through it
    flux = ((gross * ~shuttle) @ integral)
    final = np.asarray(res.c)[-1]
    seed = sorted(set(net.seed))
    start = float(sum(c0[i] for i in seed))
    conv = 0.0 if start <= 0 else max(0.0, 1.0 - float(sum(final[i] for i in seed)) / start)
    return MoleculeKinetics(final, np.asarray(res.c).max(axis=0), flux, conv, total0)


def expansion(net: MoleculeNetwork, kin: MoleculeKinetics, tau: float, floor: Optional[float] = None) -> dict:
    """What to probe next: {"molecules": {i: net flux fraction}, "pairs": {(i, j): c_max product}} (fractions
    of the starting amount), each at least tau x conversion and the floor; the seed's molecules always."""
    floor = FLOOR if floor is None else float(floor)
    total = max(kin.total0, 1e-300)
    x = max(kin.conversion, floor)
    seed = set(net.seed)
    f = kin.net_flux / total
    c = kin.c_max / total
    uni = {i: float(f[i]) for i in range(len(net.names)) if i in seed or f[i] >= max(tau * x, floor)}
    pairs = {}
    for i in range(len(net.names)):
        for j in range(i, len(net.names)):
            v = float(c[i] * c[j])
            if v >= max(tau * x, floor * floor) and (c[i] > 0 and c[j] > 0):
                pairs[(i, j)] = v
    return {"molecules": uni, "pairs": pairs, "conversion": float(kin.conversion)}


def populated(net: MoleculeNetwork, kin: MoleculeKinetics, tau: float, floor: Optional[float] = None) -> set:
    """Molecules holding material: c_max or net flux at least tau x conversion (and the floor), the seed's."""
    floor = FLOOR if floor is None else float(floor)
    total = max(kin.total0, 1e-300)
    x = max(kin.conversion, floor)
    lim = max(tau * x, floor)
    return set(net.seed) | {i for i in range(len(net.names))
                            if kin.c_max[i] / total >= lim or kin.net_flux[i] / total >= lim}


def answer_with(net: MoleculeNetwork, c0, temperature: float, time_s: float, moves: Optional[dict] = None) -> dict:
    """The answer with free energies moved (`moves`: {("molecule", i) or ("reaction", j): kcal/mol}), for an
    uncertainty analysis (uncertainty.effects takes any `answer_with(moves) -> answer`): {molecule key: RT ln of
    its share of the converted material, at least 1 %} for every molecule not in the seed."""
    moves = moves or {}
    g = np.array(net.g, dtype=float)
    reactions = [dict(x) for x in net.reactions]
    for (kind, i), v in moves.items():
        if kind == "molecule":
            g[int(i)] += float(v)
        elif kind == "reaction":
            reactions[int(i)]["g_ts"] = float(reactions[int(i)]["g_ts"]) + float(v)
    moved = MoleculeNetwork(net.names, g, reactions, net.seed)
    kin = simulate(moved, c0, temperature, time_s)
    seed = set(net.seed)
    made = {i: float(kin.final[i]) for i in range(len(net.names)) if i not in seed}
    total = sum(made.values())
    rt = 1.987204258640832e-3 * float(temperature)
    return {net.names[i]: rt * math.log(max(v / total if total > 0 else 0.0, 0.01)) for i, v in made.items()}


def lowest(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each molecule's free energy at its lowest: the largest G (summed over the boxes) with every box at or above
    the sum of its molecules', A G <= y (A: molecules per box, y: box free energies), a linear programme; then,
    among its optima (many when the boxes don't pin every molecule down), the one whose boxes lie closest to
    their own free energies (least squares on y - A G, strictly convex in A G): so the fitted boxes, and every
    reaction free energy, don't depend on the order of the boxes or molecules. G itself is fixed up to the
    directions A d = 0, which no reaction sees; the smallest such G is returned."""
    from scipy.optimize import linprog, minimize

    n = A.shape[1]
    if A.size == 0:
        return np.zeros(n)
    w = A.sum(axis=0)
    res = linprog(-w, A_ub=A, b_ub=y, bounds=[(None, None)] * n, method="highs")
    if res.status != 0:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    best = float(w @ res.x)
    # the optimal face: A G <= y, w.G >= best; the point of it with y - A G smallest (A G unique), G minimal-norm
    tol = 1e-9 * max(1.0, abs(best))
    cons = [{"type": "ineq", "fun": lambda g: y - A @ g, "jac": lambda g: -A},
            {"type": "ineq", "fun": lambda g: np.array([w @ g - best + tol]), "jac": lambda g: w[None, :]}]
    f = lambda g: float(np.sum((y - A @ g) ** 2) + 1e-9 * g @ g)
    jac = lambda g: -2.0 * A.T @ (y - A @ g) + 2e-9 * g
    sol = minimize(f, res.x, jac=jac, constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    # (SLSQP may report its line search stalling at the optimum: what counts is that the point is on the face)
    ok = np.all(A @ sol.x <= y + 1e-6) and w @ sol.x >= best - 1e-6
    g = sol.x if ok and f(sol.x) <= f(res.x) + 1e-12 else res.x
    return np.asarray(g, dtype=float)


def box_index(parts: Sequence) -> dict:
    """{molecule key: [(box, how many)]} of the boxes' molecules (`parts`: molecule_parts per box), for
    boxes_holding: built once per kinetics refresh."""
    from collections import Counter

    out: dict = {}
    for k, p in enumerate(parts):
        for key, n in Counter(key for _, key in (p or [])).items():
            out.setdefault(key, []).append((k, n))
    return out


def boxes_holding(parts: Sequence, molecules: Sequence[int], names: Sequence[str], index: Optional[dict] = None) -> list:
    """Indices of the boxes (`parts`: molecule_parts per box) that hold every molecule of `molecules` (indices into
    `names`, repeats meaning copies), first found first; `index`: box_index(parts), else built here."""
    from collections import Counter

    index = box_index(parts) if index is None else index
    want = Counter(names[i] for i in molecules)
    sets = [{k for k, n in index.get(key, []) if n >= m} for key, m in want.items()]
    return sorted(set.intersection(*sets)) if sets else []


def one_copy_why(structure, max_atoms: int = MAX_BOX_ATOMS) -> Optional[str]:
    """Why a system can't start as two copies (seed_box), or None."""
    if 2 * len(structure.symbols) > int(max_atoms):
        return f"two copies would hold {2 * len(structure.symbols)} atoms, more than {int(max_atoms)}"
    if int(structure.charge) != 0:
        return "it is charged (two copies would need a charge state of their own)"
    if int(structure.multiplicity) != 1:
        return "it is open-shell (two copies would need a spin state of their own)"
    return None


def seed_box(structure, exact: bool = False, max_atoms: int = MAX_BOX_ATOMS):
    """The box the explorer starts from: two copies of the given molecules side by side
    (mixture.place: the given arrangement twice, closest atoms 4.5 A apart), unless `exact`, or the two copies
    would hold more than `max_atoms` atoms, or the given system is charged or open-shell (two copies would need
    a charge and spin state of their own). Returns (structure, copies)."""
    if exact or one_copy_why(structure, max_atoms):
        return structure, 1
    from mepd.discovery.mixture import place

    return place([structure, structure]), 2
