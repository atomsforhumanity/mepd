"""Reaction network expansion: propose products of a structure by graph
rules instead of Hessian sampling, optimize them, and repeat from the new
species.

The built-in generator ("bond-rules") applies every combination of up to
`n_break` bond breaks and `n_form` bond formations to the molecular graph
(formations only between atoms within `form_distance` of each other), keeps
graphs whose atoms stay within normal coordination ranges and that have a
valid Lewis structure for the total charge and spin (RDKit's
DetermineBondOrders), and builds each product's 3D guess from the reactant
geometry by a restrained relaxation onto the new bonds -- so every product
keeps the reactant's atom order and can go straight into a path search.

Methods (see REFERENCES). mepd reimplements published methods; it uses no
code from ZStruct, YARP or Chemoton:
  * the break/form enumeration over the bond graph is based on ZStruct
    (Zimmerman 2013) and on YARP's n-break/n-form ("b2f2") enumeration
    (Zhao & Savoie 2021);
  * the Lewis-structure filter is RDKit's DetermineBondOrders, an
    implementation of xyz2mol (Kim & Kim 2015);
  * flux steering (steer="flux") follows the concentration-flux criterion of
    Bensberg & Reiher (2023) and the rate-based model enlargement of Susnow
    et al. (1997); see mepd.discovery.kinetics;
  * the restrained relaxation that builds each product guess is mepd's own
    heuristic (springs on the product bonds, repulsion elsewhere, a weak
    tether to the source geometry), not a published method;
  * the live view's animations are geodesic interpolations (Zhu, Thompson &
    Martinez 2019).

Generators are registered in mepd/discovery/generators.py ("bond-rules" is
the built-in one); any other plugs in by import path: a callable
`generate(structure, **options)` returning product Structures in the same
atom order (or xyz paths). `products_file` imports products an external
tool (autodE, Chemoton, ...) already wrote, as a multi-frame xyz in the
reactant's atom order.
"""
from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR, HARTREE_TO_KCAL_PER_MOL

from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import _connectivity_matches

OnEvent = Optional[Callable[[str, dict], None]]

# What each stage is, and where it comes from (written into summary.json and
# shown by the CLI and the web UI).
REFERENCES = {
    "enumeration": {
        "method": "break up to n bonds and form up to m bonds on the molecular graph: mepd's reimplementation of "
                  "the enumeration of ZStruct and of YARP's b2f2 scheme (based on their published methods; no code "
                  "from either is used)",
        "cite": ["P. M. Zimmerman, J. Comput. Chem. 34, 1385-1392 (2013), doi:10.1002/jcc.23271",
                 "Q. Zhao, B. M. Savoie, Nat. Comput. Sci. 1, 479-490 (2021), doi:10.1038/s43588-021-00101-3"],
    },
    "lewis_filter": {
        "method": "bond orders and formal charges from connectivity: RDKit DetermineBondOrders (xyz2mol)",
        "cite": ["Y. Kim, W. Y. Kim, Bull. Korean Chem. Soc. 36, 1769-1777 (2015), doi:10.1002/bkcs.10334"],
    },
    "guess_geometry": {
        "method": "mepd heuristic: restrained relaxation of the source geometry onto the product bonds (not a published method)",
        "cite": [],
    },
    "kinetics": {
        "method": "flux steering: Eyring rates from each verified TS (electronic barriers, lowest conformer of each "
                  "species), first-order microkinetics from the seed, expand species whose concentration flux "
                  "reaches the threshold",
        "cite": ["concentration-flux criterion: M. Bensberg, M. Reiher, Isr. J. Chem. 63, "
                 "e202200123 (2023), doi:10.1002/ijch.202200123",
                 "rate-based model enlargement (RMG): R. G. Susnow, A. M. Dean, W. H. Green, P. Peczak, "
                 "L. J. Broadbelt, J. Phys. Chem. A 101, 3731-3740 (1997), doi:10.1021/jp9637690"],
    },
    "animation": {
        "method": "geodesic interpolation between source and product (live view only)",
        "cite": ["X. Zhu, K. C. Thompson, T. J. Martinez, J. Chem. Phys. 150, 164103 (2019), doi:10.1063/1.5090303"],
    },
}
Edge = tuple[int, int]

# (min, max) number of bonded neighbours an atom may have in a proposed
# product (neutral atoms; a charged system widens these by one).
COORDINATION = {
    "H": (1, 1), "B": (2, 4), "C": (1, 4), "N": (1, 4), "O": (1, 2), "F": (0, 1),
    "Si": (2, 4), "P": (1, 5), "S": (1, 4), "Cl": (0, 1), "Se": (1, 2), "Br": (0, 1), "I": (0, 1),
}


def _emit(on_event: OnEvent, event: str, **payload) -> None:
    if on_event is not None:
        on_event(event, payload)


def _edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def graph_edges(node: StructureNode) -> set[Edge]:
    return {_edge(int(i), int(j)) for i, j in node.graph.edges}


@dataclass
class Proposal:
    """One proposed product of `source`: the bonds it breaks and forms."""
    source: int
    broken: tuple[Edge, ...]
    formed: tuple[Edge, ...]
    smiles: str
    structure: object = None  # guess geometry (qcdata Structure), reactant atom order
    frames: Optional[list] = field(default=None, repr=False)  # live view: source-to-guess animation

    @property
    def label(self) -> str:
        parts = [f"-{i}-{j}" for i, j in self.broken] + [f"+{i}-{j}" for i, j in self.formed]
        return " ".join(parts) or "external"


# Metals (and the charges they usually carry as ions, most common first). A
# metal is kept out of the Lewis-structure check -- xyz2mol only handles
# main-group bonding -- as an ion of one of these charges whose bonds to
# ligands are coordination (dative) bonds; the rest must still have a Lewis
# structure with the remaining charge.
METAL_CHARGES = {
    "Li": (1,), "Na": (1,), "K": (1,), "Rb": (1,), "Cs": (1,),
    "Be": (2,), "Mg": (2,), "Ca": (2,), "Sr": (2,), "Ba": (2,),
    "Al": (3,), "Ga": (3,), "In": (3,), "Tl": (1, 3), "Sn": (2, 4), "Pb": (2, 4),
    "Sc": (3,), "Ti": (4, 3, 2), "V": (3, 2, 4), "Cr": (3, 2), "Mn": (2, 3), "Fe": (2, 3), "Co": (2, 3),
    "Ni": (2, 0), "Cu": (1, 2), "Zn": (2,), "Y": (3,), "Zr": (4,), "Mo": (0, 2, 3), "Ru": (2, 3),
    "Rh": (1, 3), "Pd": (0, 2), "Ag": (1,), "Cd": (2,), "Hf": (4,), "W": (0, 6), "Re": (1, 3), "Os": (2,),
    "Ir": (1, 3), "Pt": (0, 2), "Au": (1, 3), "Hg": (2, 1), "La": (3,), "Ce": (3, 4),
}


def _lewis_mol(symbols, edges, charge, multiplicity, *, allow_radicals, allow_zwitterions):
    """The graph with bond orders and formal charges (xyz2mol) if it has a
    Lewis structure with the given total charge and spin; None otherwise."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.RWMol()
    for s in symbols:
        atom = Chem.Atom(s)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i, j in edges:
        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol.AddConformer(Chem.Conformer(len(symbols)))
    mol = mol.GetMol()
    if not symbols:
        return mol if charge == 0 else None
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=int(charge), allowChargedFragments=True)
    except Exception:
        return None
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != charge:
        return None
    n_charged = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
    if n_charged > abs(charge) + (2 if allow_zwitterions else 0):
        return None
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    if not allow_radicals and radicals != multiplicity - 1:
        return None
    return mol


def _display_smiles(structure) -> Optional[str]:
    """A SMILES to draw `structure` by, when its Lewis SMILES could not be
    assigned (hypervalent atoms, unusual bonding): connectivity-perceived,
    as the web library does (RDKit, then Open Babel). None if neither can."""
    import qcinf

    for backend in ("rdkit", "openbabel"):
        try:
            return qcinf.structure_to_smiles(structure, backend=backend)
        except Exception:
            continue
    return None


def _smiles(mol) -> str:
    from rdkit import Chem

    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))
    except Exception:
        return Chem.MolToSmiles(mol)


_DONORS = {"N", "O", "F", "P", "S", "Cl", "Se", "Br", "I", "As", "Te"}
_TRANSITION = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Mo", "Ru", "Rh", "Pd", "Ag",
               "Cd", "Hf", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}


def _can_coordinate(atom, metal: str) -> bool:
    """A ligand atom that can bind a metal: a lone-pair donor (heteroatom or
    anion), or -- for transition metals -- a carbon of a pi bond."""
    if atom.GetFormalCharge() < 0:
        return True
    if atom.GetSymbol() in _DONORS:
        return True
    if metal in _TRANSITION and atom.GetSymbol() == "C":
        return any(b.GetBondType() in (b.GetBondType().DOUBLE, b.GetBondType().TRIPLE, b.GetBondType().AROMATIC)
                   for b in atom.GetBonds())
    return False


def lewis_smiles(symbols: Sequence[str], edges: Iterable[Edge], charge: int = 0, multiplicity: int = 1,
                 *, allow_radicals: bool = False, allow_zwitterions: bool = False) -> Optional[str]:
    """Canonical SMILES of the graph if it has a Lewis structure with the
    given total charge (and, unless `allow_radicals`, exactly
    `multiplicity - 1` unpaired electrons); None otherwise. Metal atoms (see
    METAL_CHARGES) are ions bound to their ligands by dative bonds, so a
    metal's binding site is part of the species (free Mg2+ and Mg2+ on a
    carbonyl O are different species)."""
    import itertools

    from rdkit import Chem

    symbols = list(symbols)
    edges = [(int(i), int(j)) for i, j in edges]
    metals = [k for k, s in enumerate(symbols) if s in METAL_CHARGES]
    if not metals:
        mol = _lewis_mol(symbols, edges, charge, multiplicity,
                         allow_radicals=allow_radicals, allow_zwitterions=allow_zwitterions)
        return _smiles(mol) if mol is not None else None
    organic = [k for k in range(len(symbols)) if k not in metals]
    where = {k: n for n, k in enumerate(organic)}
    org_edges = [(where[i], where[j]) for i, j in edges if i in where and j in where]
    ligand_bonds = [(i, j) if i in where else (j, i) for i, j in edges if (i in where) != (j in where)]
    if any(i in metals and j in metals for i, j in edges):
        return None   # metal-metal bonds: beyond this simple ionic picture
    # Coordination does not count against a ligand atom's valence here: it
    # donates a lone pair. Try the metals' usual charges, most common first.
    for charges in itertools.islice(itertools.product(*(METAL_CHARGES[symbols[k]] for k in metals)), 64):
        mol = _lewis_mol([symbols[k] for k in organic], org_edges, charge - sum(charges), multiplicity,
                         allow_radicals=allow_radicals, allow_zwitterions=True if allow_zwitterions else False)
        if mol is None:
            continue
        if not all(_can_coordinate(mol.GetAtomWithIdx(where[lig]), symbols[metal]) for lig, metal in ligand_bonds):
            return None   # e.g. a metal "bound" to a C-H hydrogen or an sp3 carbon: not a coordination bond
        rw = Chem.RWMol(mol)
        index = {}
        for k, q in zip(metals, charges):
            atom = Chem.Atom(symbols[k])
            atom.SetFormalCharge(int(q))
            atom.SetNoImplicit(True)
            index[k] = rw.AddAtom(atom)
        for lig, metal in ligand_bonds:
            rw.AddBond(where[lig], index[metal], Chem.BondType.DATIVE)
        try:
            return _smiles(rw.GetMol())
        except Exception:
            continue
    return None


def _coordination_ok(symbols, degree, charged: bool) -> bool:
    for s, d in zip(symbols, degree):
        lo, hi = COORDINATION.get(s, (0, 6))
        if d < lo - charged or d > hi + charged:
            return False
    return True


def enumerate_bond_changes(
    symbols: Sequence[str], coords_angstrom: np.ndarray, edges: set[Edge], *,
    charge: int = 0, multiplicity: int = 1, n_break: int = 2, n_form: int = 2,
    form_distance: float = 4.0, max_products: int = 50, max_combinations: int = 500_000,
    allow_radicals: bool = False, allow_zwitterions: bool = False, source: int = 0,
    exclude_smiles: Iterable[str] = (),
) -> tuple[list[Proposal], dict]:
    """Every distinct product species reachable by <= n_break breaks and
    <= n_form formations, fewest changes first (then the closest formed
    atoms), at most `max_products`. Also returns counts for reporting."""
    n = len(symbols)
    coords = np.asarray(coords_angstrom, dtype=float)
    dist = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    bonds = sorted(edges)
    formable = sorted(_edge(i, j) for i in range(n) for j in range(i + 1, n)
                      if (i, j) not in edges and dist[i, j] <= form_distance)
    degree0 = np.zeros(n, dtype=int)
    for i, j in bonds:
        degree0[i] += 1
        degree0[j] += 1
    seen = set(exclude_smiles)
    reactant = lewis_smiles(symbols, bonds, charge, multiplicity, allow_radicals=True, allow_zwitterions=True)
    if reactant:
        seen.add(reactant)
    found: list[tuple[tuple, Proposal]] = []
    stats = {"combinations": 0, "lewis_checked": 0, "clipped": False, "formable_pairs": len(formable)}
    for nb, nf in sorted(((b, f) for b in range(n_break + 1) for f in range(n_form + 1) if b + f),
                         key=lambda bf: (sum(bf), bf)):
        for broken in itertools.combinations(bonds, nb):
            deg_b = degree0.copy()
            for i, j in broken:
                deg_b[i] -= 1
                deg_b[j] -= 1
            for formed in itertools.combinations(formable, nf):
                stats["combinations"] += 1
                if stats["combinations"] > max_combinations:
                    stats["clipped"] = True
                    break
                deg = deg_b.copy()
                for i, j in formed:
                    deg[i] += 1
                    deg[j] += 1
                if not _coordination_ok(symbols, deg, charge != 0):
                    continue
                product = (set(bonds) - set(broken)) | set(formed)
                stats["lewis_checked"] += 1
                smi = lewis_smiles(symbols, product, charge, multiplicity,
                                   allow_radicals=allow_radicals, allow_zwitterions=allow_zwitterions)
                if smi is None or smi in seen:
                    continue
                seen.add(smi)
                key = (nb + nf, sum(dist[i, j] for i, j in formed))
                found.append((key, Proposal(source, tuple(broken), tuple(formed), smi)))
            if stats["clipped"]:
                break
        if stats["clipped"]:
            break
    found.sort(key=lambda kp: kp[0])
    stats["distinct_products"] = len(found)
    return [p for _, p in found[:max_products]], stats


def embed_product(symbols: Sequence[str], coords_angstrom: np.ndarray, product_edges: set[Edge],
                  *, tether: float = 0.02, attempts: int = 6, env: Optional[np.ndarray] = None,
                  env_atoms: Optional[int] = None, env_distance: float = 1.8) -> np.ndarray:
    """A guess geometry (Angstrom) for the product graph, starting from the
    reactant's: bonded pairs pulled to covalent length, 1-3 pairs and
    non-bonded pairs pushed out of contact, a weak tether to the start so
    the rest of the molecule stays put. `env`: fixed atoms (a QM/MM
    environment, Angstrom) the first `env_atoms` atoms are kept
    `env_distance` away from."""
    from rdkit import Chem
    from scipy.optimize import minimize

    table = Chem.GetPeriodicTable()
    x0 = np.asarray(coords_angstrom, dtype=float)
    n = len(symbols)
    rcov = np.array([table.GetRcovalent(s) for s in symbols])
    iu, ju = np.triu_indices(n, 1)
    rsum = rcov[iu] + rcov[ju]
    adj = np.zeros((n, n), dtype=bool)
    for i, j in product_edges:
        adj[i, j] = adj[j, i] = True
    one_three = (adj.astype(int) @ adj.astype(int)) > 0
    bonded = adj[iu, ju]
    target = np.where(bonded, rsum, np.where(one_three[iu, ju], 1.5 * rsum, np.minimum(2.2 * rsum, 3.0)))

    def fg(flat):
        x = flat.reshape(n, 3)
        d = x[iu] - x[ju]
        r = np.linalg.norm(d, axis=1) + 1e-12
        dev = r - target
        active = bonded | (dev < 0)  # bonds are springs; everything else only repels
        e = np.sum(dev[active] ** 2) + tether * np.sum((x - x0) ** 2)
        coef = np.where(active, 2 * dev / r, 0.0)[:, None] * d
        g = 2 * tether * (x - x0)
        np.add.at(g, iu, coef)
        np.add.at(g, ju, -coef)
        if env is not None and len(env):
            m = n if env_atoms is None else int(env_atoms)
            de = x[:m, None, :] - env[None, :, :]
            re = np.linalg.norm(de, axis=2) + 1e-12
            short = np.minimum(re - env_distance, 0.0)
            e += np.sum(short ** 2)
            g[:m] += np.sum((2 * short / re)[:, :, None] * de, axis=1)
        return e, g.ravel()

    def relax(start):
        res = minimize(fg, start.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        return res.x.reshape(n, 3), float(res.fun)

    # Restarts from randomly kicked starts escape symmetric traps (e.g. an H
    # moving along a linear molecule's axis has to go around the atom in
    # between, not through it); the first whose bonds are the product's wins.
    rng = np.random.default_rng(0)
    best = None
    for attempt in range(attempts):
        x, fun = relax(x0 + (0.0 if attempt == 0 else 0.3 * attempt) * rng.standard_normal(x0.shape))
        if _perceived_edges(symbols, x) == set(product_edges):
            return x
        if best is None or fun < best[1]:
            best = (x, fun)
    return best[0]


def build_partners(sources: list, target, run_inputs) -> tuple[list, list]:
    """For each node in `sources`, `target` built in its frame: every atom
    where it is in that node, the target's bonds pulled to length
    (`embed_product`), then minimized with the run's engine. A partner whose
    bonds or stereochemistry come out different from `target`'s is dropped
    with its source. `target` (same atom order as the sources) also gives
    the charge and spin. Returns (kept sources, their partners)."""
    from mepd.nodes.nodehelpers import _is_connectivity_identical

    want = {_edge(int(u), int(v)) for u, v in target.graph.edges()}
    kept, guesses = [], []
    for node in sources:
        try:
            xyz = embed_product(list(node.structure.symbols),
                                np.asarray(node.structure.geometry) / ANGSTROM_TO_BOHR, want)
        except Exception:
            continue
        kept.append(node)
        guesses.append(StructureNode(structure=target.structure.model_copy(
            update={"geometry": np.asarray(xyz) * ANGSTROM_TO_BOHR})))
    keywords = {"coordsys": "cart", "maxit": 500, **(getattr(run_inputs, "geometry_optimizer_kwds", None) or {})}
    batch = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch):
        try:
            trajectories = batch(guesses, keywords=keywords)
        except TypeError:
            trajectories = batch(guesses)
    else:
        trajectories = [run_inputs.engine.compute_geometry_optimization(g, keywords=keywords) for g in guesses]
    pairs = []
    for src, traj in zip(kept, trajectories):
        if not traj:
            continue
        built = StructureNode(structure=traj[-1].structure)
        if {_edge(int(u), int(v)) for u, v in built.graph.edges()} == want and _is_connectivity_identical(
                built, target, verbose=False, collect_comparison=False):
            pairs.append((src, built))
    return [s for s, _ in pairs], [b for _, b in pairs]


def _perceived_edges(symbols, coords_angstrom) -> set[Edge]:
    """Bonds as mepd perceives them (openbabel, as StructureNode.graph)."""
    from qcdata import Structure

    node = StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(coords_angstrom) * ANGSTROM_TO_BOHR))
    return graph_edges(node)


def _structure_with(structure, coords_angstrom):
    return structure.model_copy(update={"geometry": np.asarray(coords_angstrom) * ANGSTROM_TO_BOHR})


# neighbours of an atom with no multiple bond: a bond between two such atoms has three staggered rotamers
_SATURATED = {"C": 4, "Si": 4, "N": 3, "P": 3, "O": 2, "S": 2}


def ts_rotamer_guesses(ts, ends) -> tuple[list, list[Edge]]:
    """TS guesses for the other rotamers of transition state `ts` (a Structure): the same reacting core, the rest
    turned about one or two of its rotatable bonds. `ends`: the TS's two IRC ends (Structures in its atom order);
    the bonds that differ between them are the core, and a bond is rotatable when both ends have it, it is in no
    ring of the two ends' bonds together (so the core's forming ring stays shut and its distances stay as they
    are), and the side it turns is not a symmetric top (CH3, CF3, NH2). Each bond turns by 180 degrees (120 and
    240 when both its atoms are saturated); each pair of bonds by those first turns together.
    Returns ([(guess Structure, what)], core pairs)."""
    symbols = [str(s) for s in ts.symbols]
    xyz = np.asarray(ts.geometry, dtype=float) / ANGSTROM_TO_BOHR
    e1, e2 = (graph_edges(StructureNode(structure=e)) for e in ends)
    n = len(symbols)
    adj = [set() for _ in range(n)]
    for i, j in e1 | e2:
        adj[i].add(j)
        adj[j].add(i)

    def side(u, v) -> set:
        """The atoms reached from v without crossing the bond u-v."""
        seen, todo = {v}, [v]
        while todo:
            x = todo.pop()
            for y in adj[x] - seen:
                if not (x == v and y == u):
                    seen.add(y)
                    todo.append(y)
        return seen

    bonds = []
    for u, v in sorted(e1 & e2):
        s = side(u, v)
        if u in s:
            continue
        moving, pivot, anchor = (s, v, u) if len(s) <= n - len(s) else (set(range(n)) - s, u, v)
        rest = moving - {pivot}
        if not rest or (len(rest) >= 2 and all(len(adj[a]) == 1 for a in rest) and len({symbols[a] for a in rest}) == 1):
            continue
        saturated = all(len(adj[a]) == _SATURATED.get(symbols[a], -1) for a in (u, v))
        bonds.append((anchor, pivot, sorted(moving), (120.0, 240.0) if saturated else (180.0,)))

    def turn(x, bond, degrees):
        anchor, pivot, moving, _ = bond
        axis = (x[pivot] - x[anchor]) / np.linalg.norm(x[pivot] - x[anchor])
        t, p = np.radians(degrees), x[moving] - x[pivot]
        out = x.copy()
        out[moving] = (p * np.cos(t) + np.cross(axis, p) * np.sin(t) + np.outer(p @ axis, axis) * (1.0 - np.cos(t))
                       + x[pivot])
        return out

    out = [(turn(xyz, b, d), f"{b[0]}-{b[1]} {d:g}") for b in bonds for d in b[3]]
    out += [(turn(turn(xyz, b1, b1[3][0]), b2, b2[3][0]), f"{b1[0]}-{b1[1]} {b1[3][0]:g}, {b2[0]}-{b2[1]} {b2[3][0]:g}")
            for k, b1 in enumerate(bonds) for b2 in bonds[k + 1:]]
    return [(_structure_with(ts, x), what) for x, what in out], sorted(e1 ^ e2)


class _View:
    """What the bond rules see of a structure: the whole molecule, or, for a
    QM/MM system (mepd.qmmm), its QM model -- the QM atoms capped with their
    link hydrogens, at the QM charge and spin. Link hydrogens never take
    part in a bond change; product guesses are built inside the fixed
    environment and put back into the full system."""

    def __init__(self, region=None):
        self.region = region

    def symbols(self, node) -> list:
        return list(self.region.model_symbols) if self.region else list(node.symbols)

    def charge_mult(self, node) -> tuple[int, int]:
        if self.region:
            return int(self.region.qm_charge), int(self.region.qm_multiplicity)
        return int(node.structure.charge), int(node.structure.multiplicity)

    def coords(self, node) -> np.ndarray:
        x = np.asarray(node.coords, dtype=float) / ANGSTROM_TO_BOHR
        return self.region.model_coords(x) if self.region else x

    def edges(self, node) -> set:
        if self.region:
            return _perceived_edges(self.symbols(node), self.coords(node))
        return graph_edges(node)

    def keep(self, props: list) -> list:
        if not self.region:
            return props
        nqm = len(self.region.qm_atoms)
        return [p for p in props if all(i < nqm and j < nqm for i, j in (*p.broken, *p.formed))]

    def environment(self, node) -> Optional[np.ndarray]:
        """Environment atoms near the QM region (Angstrom), without the
        atoms of cut bonds."""
        if not self.region:
            return None
        x = np.asarray(node.coords, dtype=float) / ANGSTROM_TO_BOHR
        skip = set(self.region.qm_atoms) | {m for _, m in self.region.links}
        mm = np.array([i for i in range(len(x)) if i not in skip], dtype=int)
        if not len(mm):
            return None
        d = np.linalg.norm(x[mm][:, None] - x[self.region.qm_atoms][None], axis=2).min(axis=1)
        return x[mm[d < 6.0]]

    def full_structure(self, node, guess) -> object:
        if not self.region:
            return _structure_with(node.structure, guess)
        x = np.asarray(node.coords, dtype=float) / ANGSTROM_TO_BOHR
        x[self.region.qm_atoms] = np.asarray(guess)[:len(self.region.qm_atoms)]
        return _structure_with(node.structure, x)



def nanoreactor_anchors(region) -> list[int]:
    """Atoms of a QM/MM system's QM model (`region.model_structure`) the
    nanoreactor holds in place relative to each other: the link H, which
    stand where the protein's backbone (or the rest of a molecule) was, and
    the QM atoms they cap (CB of a side chain). Held, the side chains of the
    QM region stay arranged as the protein keeps them instead of flying off
    in the hot MD."""
    nqm = len(region.qm_atoms)
    return [nqm + k for k in range(len(region.links))] + sorted({region.qm_atoms.index(q) for q, _ in region.links})


def _open_structure(fp):
    from qcdata import Structure

    return Structure.open(str(fp))


def _region_proposals(view: "_View", node, source: int, items: Iterable) -> tuple[list[Proposal], dict]:
    """Products a whole-molecule generator made of a QM/MM system's QM model
    (QM atoms + link H, `region.model_structure`), put back into the full
    system: aligned onto the QM atoms where they are, relaxed briefly
    against the nearby environment (the generator optimized them in vacuum,
    so they can overlap it), the environment left as it is. A product whose
    link hydrogen stopped being a cap (lost its bond to its QM atom, or
    bonded to something else) breaks the QM/MM boundary and is dropped.
    SMILES and bond changes are those of the QM model. Returns (proposals,
    {"kept", "boundary"})."""
    from qcdata import Structure

    from mepd.rigid_alignment import kabsch_align

    region = view.region
    nqm = len(region.qm_atoms)
    symbols = view.symbols(node)
    x0 = view.coords(node)                     # the QM model as it sits in the system (Angstrom)
    edges0 = view.edges(node)
    caps = [(region.qm_atoms.index(q), nqm + k) for k, (q, _) in enumerate(region.links)]
    env = view.environment(node)
    charge, mult = view.charge_mult(node)
    out, n_boundary = [], 0
    for k, item in enumerate(items):
        if not isinstance(item, Structure):
            item = Structure.open(str(item))
        if list(item.symbols) != symbols:
            raise ValueError(f"a product of the QM region lists {len(item.symbols)} atoms, not the QM model's "
                             f"{len(symbols)} in its order (QM atoms, then link H)")
        y = kabsch_align(np.asarray(item.geometry, dtype=float) / ANGSTROM_TO_BOHR, x0)
        edges = _perceived_edges(symbols, y)
        if any((min(q, h), max(q, h)) not in edges or sum(h in e for e in edges) != 1 for q, h in caps):
            n_boundary += 1
            continue
        guess = embed_product(symbols, y, edges, env=env, env_atoms=nqm)
        out.append(Proposal(source, tuple(sorted(edges0 - edges)), tuple(sorted(edges - edges0)),
                            # No Lewis structure for it (e.g. a radical the simple
                            # picture can't place): named after it is optimized,
                            # like an imported product.
                            lewis_smiles(symbols, edges, charge, mult) or f"external-{k}",
                            structure=view.full_structure(node, guess)))
    return out, {"kept": len(out), "boundary": n_boundary}


def _qmmm_region(engine, node):
    from mepd.qmmm import QMMMRegion

    region = getattr(engine, "region", None)
    return region if isinstance(region, QMMMRegion) and len(node.symbols) == region.natoms else None


def _import_generator(path: str) -> Callable:
    module_name, _, attr = str(path).replace(":", ".").rpartition(".")
    if not module_name:
        from mepd.discovery.generators import get_generator

        get_generator(path)   # raises, naming the built-in generators
    return getattr(importlib.import_module(module_name), attr)


def external_proposals(structures: Iterable, source_structure, source: int) -> list[Proposal]:
    """Proposals from product structures made elsewhere; they must list the
    same elements in the same order as the source."""
    from qcdata import Structure

    out = []
    for k, s in enumerate(structures):
        if not isinstance(s, Structure):
            s = Structure.open(str(s))
        if list(s.symbols) != list(source_structure.symbols):
            raise ValueError(f"Product {k} does not list the source's atoms in the same order "
                             f"({len(s.symbols)} vs {len(source_structure.symbols)} atoms); products must "
                             "be atom-mapped to the reactant.")
        s = s.model_copy(update={"charge": source_structure.charge, "multiplicity": source_structure.multiplicity})
        out.append(Proposal(source, (), (), f"external-{k}", structure=s))
    return out


@dataclass
class Species:
    node: StructureNode
    smiles: str
    round: int
    rel_energy_kcal: float = 0.0
    validation: Optional[dict] = None
    loose: bool = False   # minimized only to the proposals' loose threshold so far


@dataclass
class ProposedEdge:
    source: int
    proposal: Proposal
    outcome: str  # "new_species", "known_species", "reverted", "not_minimum", "failed"
    target: Optional[int] = None  # species index it optimized to
    intended: Optional[bool] = None  # landed on the proposed connectivity
    error: str = ""


@dataclass
class ExpansionResult:
    species: List[Species] = field(default_factory=list)
    edges: List[ProposedEdge] = field(default_factory=list)
    rounds: List[dict] = field(default_factory=list)
    rejected: List[StructureNode] = field(default_factory=list)  # not minima after the Hessian check
    steps: list = field(default_factory=list)  # flux steering: verified elementary steps (kinetics.Step)
    kinetics: object = None  # flux steering: the latest kinetics.KineticsResult
    channels_log: list = field(default_factory=list)  # each sampled step's TSs before and after channel sampling

    def connections(self) -> list[tuple[int, int]]:
        """Distinct (source, product) species pairs to connect by path search."""
        pairs = []
        for e in self.edges:
            if e.target is not None and e.target != e.source:
                pair = (min(e.source, e.target), max(e.source, e.target))
                if pair not in pairs:
                    pairs.append(pair)
        return pairs


# Engines whose minimizer takes a force threshold (`fmax`, eV/Angstrom).
_FMAX_ENGINES = ("GXTBCalculator", "ASEEngine", "ModifiedEngine")


def _takes_fmax(engine) -> bool:
    return any(c.__name__ in _FMAX_ENGINES for c in type(engine).__mro__)


def _optimize(engine, nodes: list[StructureNode], maxiter: int, on_event: OnEvent,
              on_start: Callable[[int], None] = lambda k: None, fmax: Optional[float] = None) -> list:
    """Optimized node (or an exception) per input, isolating failures.
    `on_start(k)` is called as optimization k (0-based) starts. `fmax`: stop
    at this largest force (eV/Angstrom) instead of the engine's own threshold."""
    keywords = {"coordsys": "cart", "maxiter": int(maxiter)}
    if fmax:
        keywords["fmax"] = float(fmax)
    batch = getattr(engine, "compute_geometry_optimizations", None)
    if callable(batch) and len(nodes) > 1:
        for k in range(len(nodes)):
            on_start(k)
        try:
            trajs = batch(nodes, keywords=keywords)
            if len(trajs) == len(nodes):
                _emit(on_event, "candidate_done", index=len(nodes), total=len(nodes))
                return [t[-1] if t else ValueError("empty trajectory") for t in trajs]
        except Exception:
            pass
    out = []
    for k, node in enumerate(nodes, start=1):
        on_start(k - 1)
        try:
            try:
                traj = engine.compute_geometry_optimization(node, keywords=keywords)
            except TypeError:
                traj = engine.compute_geometry_optimization(node)
            out.append(traj[-1] if traj else ValueError("empty trajectory"))
        except Exception as exc:
            out.append(exc)
        _emit(on_event, "candidate_done", index=k, total=len(nodes))
    return out


_OUTCOME_LABEL = {"new_species": "new species", "known_species": "known species", "reverted": "back to source",
                  "not_minimum": "not a minimum", "failed": "failed"}


def _morph_frames(job) -> list[str]:
    """xyz frames of a geodesic interpolation (Zhu, Thompson & Martinez,
    J. Chem. Phys. 150, 164103 (2019)) from structure a to b, moved as one
    rigid body so the first frame sits on `reference` (Bohr). A plain
    function of plain data, so it runs in a worker process."""
    from qcdata import Structure

    from mepd.chainhelpers import run_geodesic

    symbols, charge, mult, xa, xb, reference, nimages = job
    sub = StructureNode._global_graph_atoms
    if sub is not None and len(symbols) == sub[1]:
        # A QM/MM system: geodesic over the QM atoms, the environment moved
        # linearly (a geodesic over thousands of atoms is far too slow).
        qm = list(sub[0])
        sa, sb = (StructureNode(structure=Structure(symbols=[symbols[i] for i in qm], geometry=np.asarray(x)[qm]),
                                has_molecular_graph=False) for x in (xa, xb))
        try:
            qm_frames = [np.asarray(n.coords) for n in run_geodesic([sa, sb], nimages=nimages, align=False)]
        except Exception:
            qm_frames = [np.asarray(xa)[qm] + t * (np.asarray(xb)[qm] - np.asarray(xa)[qm])
                         for t in np.linspace(0, 1, nimages)]
        coords = []
        for k, q in enumerate(qm_frames):
            t = k / max(1, len(qm_frames) - 1)
            c = np.asarray(xa, dtype=float) + t * (np.asarray(xb, dtype=float) - np.asarray(xa, dtype=float))
            c[qm] = q
            coords.append(c)
        a = StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(xa), charge=charge,
                                              multiplicity=mult), has_molecular_graph=False)
        nodes = [a] * len(coords)
    else:
        a, b = (StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(x), charge=charge,
                                                  multiplicity=mult)) for x in (xa, xb))
        try:
            nodes = list(run_geodesic([a, b], nimages=nimages))
        except Exception:
            nodes = [a, b]
        coords = [np.asarray(n.coords, dtype=float) for n in nodes]
    shift = coords[0].mean(axis=0)
    u, _, vt = np.linalg.svd((coords[0] - shift).T @ reference)
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return [n.structure.model_copy(update={"geometry": (c - shift) @ rot}).to_xyz() for n, c in zip(nodes, coords)]


class _LiveReactions:
    """Live view (web UI): one stream per proposed reaction, animating the
    source turning into the product by geodesic interpolation -- into the
    proposed guess while it waits and optimizes, then into what it
    optimized to. Costs nothing when nobody is watching."""

    def __init__(self, seed: StructureNode, seed_energy: float, nimages: int = 16, workers: int = 1):
        from mepd import progress

        self._progress = progress
        self.enabled = progress._stream_path("probe") is not None
        self.e0 = seed_energy
        self.nimages = nimages
        self.workers = max(1, int(workers))
        self._streams: dict[int, str] = {}
        self._frames_of: dict[int, list] = {}   # proposal -> seed-to-guess frames
        self._pool = None
        self._pending: list = []
        # Every animation starts from the seed in one fixed pose (centred,
        # principal axes along x, y, z), so they all begin in the same place.
        x = np.asarray(seed.coords, dtype=float)
        x = x - x.mean(axis=0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        if np.linalg.det(vt) < 0:
            vt[-1] *= -1
        self.reference = x @ vt.T

    def job(self, a: StructureNode, b: StructureNode):
        st = a.structure
        return (list(st.symbols), int(st.charge), int(st.multiplicity), np.asarray(a.coords, dtype=float),
                np.asarray(b.coords, dtype=float), self.reference, self.nimages)

    def _submit(self, job, done: Callable[[list], None]) -> None:
        """Compute frames off the main loop (forked workers when that is safe,
        else threads) and hand them to `done` when ready."""
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

            from mepd.cli_common import _unsafe_to_fork

            if self.workers > 1 and _unsafe_to_fork() is None:
                import multiprocessing

                self._pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("fork"))
            else:
                self._pool = ThreadPoolExecutor(1)
        fut = self._pool.submit(_morph_frames, job)
        fut.add_done_callback(lambda f: done(f.result() if f.exception() is None else []))
        self._pending.append(fut)

    def close(self) -> None:
        """Wait for the last animations, so every stream ends finished."""
        if self._pool is not None:
            for fut in self._pending:
                try:
                    fut.result()
                except Exception:
                    pass
            self._pool.shutdown(wait=True)
            self._pool = None

    def _write(self, p: Proposal, species, frames: list, *, status: str, outcome=None,
               product_kcal=None, product_smiles=None, note: str = "") -> None:
        stream = self._streams.setdefault(id(p), f"rxn{len(self._streams) + 1:04d}")
        src = species[p.source]
        sym = list(src.node.symbols)
        bonds = [f"−{sym[i]}{i}–{sym[j]}{j}" for i, j in p.broken] + [f"+{sym[i]}{i}–{sym[j]}{j}" for i, j in p.formed]
        # An external generator's proposal is named "external-<k>", not a SMILES:
        # never draw by that name (the live view showed broken images).
        fallback = None if p.smiles.startswith("external-") else (p.smiles or None)
        self._progress.write_morph(
            stream, frames, label=f"#{int(stream[3:])} from species {p.source}",
            caption=(" ".join(bonds) or "from the external generator") + note,
            status=status, finished=status in ("done", "failed"), outcome=outcome,
            energies_kcal=(src.rel_energy_kcal, product_kcal),
            reactant_smiles=src.smiles or self._drawable(src.node.structure),
            product_smiles=product_smiles or fallback,
            extra={"source": p.source, "bonds": " ".join(bonds)})

    def _event(self, edge: ProposedEdge, species) -> None:
        """A new species (or a new link between two known ones) as soon as
        it is accepted, so a viewer can add it to its network right away."""
        src = species[edge.source]
        sym = list(src.node.symbols)
        p = edge.proposal
        caption = " ".join([f"−{sym[i]}{i}–{sym[j]}{j}" for i, j in p.broken]
                           + [f"+{sym[i]}{i}–{sym[j]}{j}" for i, j in p.formed])
        if edge.outcome == "new_species":
            s = species[edge.target]
            self._progress.append_live_event({
                "event": "species", "index": edge.target, "parent": edge.source, "smiles": s.smiles,
                "energy_hartree": float(s.node.energy), "rel_energy_kcal": s.rel_energy_kcal, "round": s.round,
                "validation": s.validation, "xyz": s.node.structure.to_xyz(), "caption": caption})
        elif edge.outcome == "known_species" and edge.target is not None and edge.target != edge.source:
            self._progress.append_live_event({
                "event": "reaction", "source": edge.source, "target": edge.target, "caption": caption})

    def propose(self, p: Proposal, species, frames: Optional[list] = None, status: str = "queued") -> None:
        if not self.enabled:
            return
        if frames is not None:
            self._frames_of[id(p)] = frames
        frames = self._frames_of.get(id(p))
        if frames is None:  # not built with the guess (external generator): make it now
            frames = _morph_frames(self.job(species[p.source].node, StructureNode(structure=p.structure)))
            self._frames_of[id(p)] = frames
        self._write(p, species, frames, status=status)

    def prepare(self, proposals: list, species) -> None:
        """Animations of proposals made without one (an external generator's), all at once on the
        animation workers -- built one after another in the main loop they held up every round
        (50 collision products of a 23-atom system: ~13 minutes before the first optimization)."""
        if not self.enabled:
            return
        todo = [p for p in proposals if id(p) not in self._frames_of and p.frames is None]
        if not todo:
            return
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        from mepd.cli_common import _unsafe_to_fork

        jobs = [self.job(species[p.source].node, StructureNode(structure=p.structure)) for p in todo]
        width = max(1, min(self.workers, len(jobs)))
        if width > 1 and _unsafe_to_fork() is None:
            import multiprocessing

            pool = ProcessPoolExecutor(width, mp_context=multiprocessing.get_context("fork"))
        else:
            pool = ThreadPoolExecutor(width)
        with pool:
            for p, frames in zip(todo, pool.map(_morph_frames, jobs)):
                self._frames_of[id(p)] = frames

    def optimizing(self, p: Proposal, species) -> None:
        self.propose(p, species, status="running")

    def _drawable(self, structure) -> Optional[str]:
        """`_display_smiles`, remembered per geometry (the seed recurs in every record)."""
        cache = self.__dict__.setdefault("_smiles_cache", {})
        key = id(structure)
        if key not in cache:
            cache[key] = (structure, _display_smiles(structure))
        return cache[key][1]

    def finish(self, edge: ProposedEdge, species, product: Optional[StructureNode] = None) -> None:
        if not self.enabled:
            return
        self._event(edge, species)
        p = edge.proposal
        node = product if product is not None else (
            species[edge.target].node if edge.target is not None else StructureNode(structure=p.structure))
        try:
            kcal = (float(node.energy) - self.e0) * HARTREE_TO_KCAL_PER_MOL
        except Exception:
            kcal = None
        note = " · relaxed to a different product than proposed" if edge.intended is False else ""
        if edge.error:
            note += f" · {edge.error}"
        write = lambda frames: self._write(  # noqa: E731
            p, species, frames or self._frames_of.get(id(p)) or [], status="failed" if edge.outcome == "failed" else "done",
            outcome=_OUTCOME_LABEL.get(edge.outcome, edge.outcome), product_kcal=kcal,
            product_smiles=(species[edge.target].smiles if edge.target is not None else None)
            or self._drawable(node.structure), note=note)
        self._submit(self.job(species[p.source].node, node), write)


def _build_guesses(props: list[Proposal], node: StructureNode, coords: np.ndarray, live, workers: int,
                   view: Optional[_View] = None) -> None:
    """Each proposal's 3D guess (and, for a watching live view, its
    seed-to-guess animation), across `workers` forked processes."""
    from mepd.cli_common import _fork_map

    view = view or _View()
    symbols = view.symbols(node)
    edges = view.edges(node)
    env = view.environment(node)
    nfeel = len(view.region.qm_atoms) if view.region else None

    def build(k):
        p = props[k]
        guess = embed_product(symbols, coords, (edges - set(p.broken)) | set(p.formed), env=env, env_atoms=nfeel)
        frames = None
        if live.enabled:
            structure = view.full_structure(node, guess)
            frames = _morph_frames(live.job(node, StructureNode(structure=structure, has_molecular_graph=False)))
        return guess, frames

    for p, (guess, frames) in zip(props, _fork_map(build, list(range(len(props))), workers)):
        p.structure = view.full_structure(node, guess)
        p.frames = frames


def _generator_cutoff(generator, options: dict, access_cutoff_kcal, species) -> dict:
    """The reactor generator's options for expanding `species`: what is left of the access cutoff above it
    (its rel_energy_kcal above the seed), so products that can't be reached within the cutoff don't hold up
    its convergence."""
    if str(generator).strip().lower() != "reactor" or access_cutoff_kcal is None:
        return options
    return {**options, "cutoff_kcal": float(access_cutoff_kcal) - float(species.rel_energy_kcal)}


def _propose_structures_at_once(frontier, result, generator, generator_options, products_file, view,
                                max_products, workers, access_cutoff_kcal=None) -> dict:
    """A structures generator's products for every frontier species, run concurrently.

    One species at a time leaves CPUs idle: a nanoreactor species runs only its
    `n_seeds` trajectories. Here as many species run at once as `workers` allows
    (each with its seeds), so a round's proposals take about one species' time.
    Returns {species index: items}; species it doesn't cover are proposed in the loop.
    """
    from mepd.discovery.generators import get_generator, is_named

    if products_file is not None or view.region is not None or len(frontier) < 2 or not is_named(generator):
        return {}
    gen = get_generator(generator)
    if gen.kind != "structures":
        return {}
    gen.check()
    opts = dict(generator_options or {})
    per_species = max(1, int(opts.get("n_seeds", 1) or 1))
    if "parallel" in opts:
        per_species = max(1, min(per_species, int(opts["parallel"] or per_species)))
    width = max(1, min(len(frontier), int(workers) // per_species))
    if width < 2:
        return {}
    from concurrent.futures import ThreadPoolExecutor

    if "parallel" in opts or "parallel" in gen.options:
        # each species gets its share of the workers: `width` species each with all of them (a generator's
        # default) oversubscribed the CPUs `width` times over (9 x 48 processes on 48 cores)
        opts["parallel"] = max(1, int(workers) // width)

    def one(src):
        try:
            return gen.propose(result.species[src].node.structure, max_products=max_products,
                               options=_generator_cutoff(generator, dict(opts), access_cutoff_kcal,
                                                         result.species[src]))
        except Exception:
            return None   # the loop retries it on its own and reports the error there

    with ThreadPoolExecutor(max_workers=width) as pool:
        got = dict(zip(frontier, pool.map(one, frontier)))
    return {k: v for k, v in got.items() if v is not None}


_STATE_VERSION = 1


_STEER_ENGINES: tuple = ()


def _steer_init(engine, steer_engine) -> None:
    """The spawned process a GPU steering level runs in gets both engines once, so the model is loaded once."""
    global _STEER_ENGINES
    _STEER_ENGINES = (engine, steer_engine)


def _steer_point(item) -> "dict | str":
    """One steering single point (energy_refine._single_point) in that
    process; a failure comes back as its message."""
    from mepd.discovery.energy_refine import _single_point

    try:
        return _single_point(item, *_STEER_ENGINES)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _state_fingerprint(engine, steer_engine, steer, kin, access_cutoff_kcal, generator, generator_options,
                       products_file) -> dict:
    """The settings a saved exploration was made with that decide its network: a resume must use the same.
    (How many rounds, species, workers or processes per generator are not among them.)"""
    import re

    from mepd.discovery.energy_refine import _level

    def level(eng):   # how many processes or threads an engine uses doesn't change the network
        return re.sub(r"\('n_(parallel|threads)', '[^']*'\)(, )?", "", _level(eng)) if eng is not None else None

    return {"engine": level(engine), "steer_engine": level(steer_engine),
            "steer": steer, "kinetics": {k: kin[k] for k in sorted(kin)}, "access_cutoff_kcal": access_cutoff_kcal,
            "generator": str(generator), "products_file": products_file,
            "generator_options": {k: str(v) for k, v in sorted((generator_options or {}).items()) if k != "parallel"}}


def _write_state(fp: Path, state: dict) -> None:
    """Pickle `state` to `fp` atomically: a run killed mid-write keeps the previous state."""
    import pickle

    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_name(fp.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"version": _STATE_VERSION, **state}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(fp)


def _load_state(fp: Optional[Path], fingerprint: dict) -> dict:
    import pickle

    if fp is None or not Path(fp).is_file():
        raise ValueError(f"Nothing to resume: no saved exploration at {fp}.")
    with open(fp, "rb") as fh:
        state = pickle.load(fh)
    if state.get("version") != _STATE_VERSION:
        raise ValueError(f"{fp} was saved by another version of mepd (state version {state.get('version')}, "
                         f"this one reads {_STATE_VERSION}); it can't be resumed.")
    differ = [k for k in fingerprint if state["fingerprint"].get(k) != fingerprint[k]]
    if differ:
        raise ValueError("The saved exploration was made with other settings; resume it with the ones it was "
                         "saved with. Differs: " + "; ".join(
                             f"{k}: saved {state['fingerprint'].get(k)!r}, now {fingerprint[k]!r}" for k in differ))
    return state


def expand_network(
    seed: StructureNode, engine, *, rounds: int = 1, energy_window_kcal: float = 60.0,
    generator: str = "bond-rules", generator_options: Optional[dict] = None,
    products_file: Optional[str] = None, maxiter: int = 500, n_break: int = 2, n_form: int = 2,
    form_distance: float = 4.0, max_products: int = 50, allow_radicals: bool = False,
    allow_zwitterions: bool = False, max_species: int = 200, validate_minima: Optional[dict] = None,
    workers: int = 1, minimize_seed: bool = True, steer: str = "window", connect: Optional[Callable] = None,
    kinetics: Optional[dict] = None, on_event: OnEvent = None, target: Optional[StructureNode] = None,
    access_cutoff_kcal: Optional[float] = None, reroute_attempts: int = 3, max_expand_per_round: int = 10,
    checkpoint: Optional[Callable] = None, refine: Optional[Callable] = None,
    verify: Optional[Callable] = None, proposal_fmax: Optional[float] = 0.05,
    steer_engine=None, steer_save_dir: Optional[Path] = None,
    state_file: Optional[Path] = None, resume: bool = False, schedule: str = "rounds",
    time_budget_s: Optional[float] = None,
) -> ExpansionResult:
    """Breadth-first network expansion from `seed`. Each round proposes
    products of every species found in the previous round (within
    `energy_window_kcal` of the seed), optimizes them, and adds the ones
    that are new species. Species identity is connectivity + stereo
    (conformers of one species are merged). A proposal whose SMILES is an
    already-known species becomes an edge to it without being optimized
    again. `validate_minima` ({"frequency_cutoff", "rescue_displacement"})
    Hessian-checks every new species before it is accepted.

    `steer` decides which species the next round expands:
      * "window": every new species within `energy_window_kcal` of the seed;
      * "flux": after each round, `connect(pairs, species_nodes)` finds a
        verified TS for every new reaction (returning, per elementary step,
        {"start", "end", "ts", "label", "files"}; IRC ends that are no known
        species join the network as intermediates), microkinetics runs on
        every verified step (`kinetics`: temperature, time_s, threshold;
        see mepd.discovery.kinetics), and only species whose concentration
        flux reaches the threshold are expanded. The network stops growing
        when none does.
      * "access": like "flux", every new reaction gets a verified TS, but a
        species is expanded when the network-wide accessibility (the best
        route over every verified step found by any search; see
        mepd.network_aware.accessibility) reaches it climbing at most
        `access_cutoff_kcal` out of a well AND the microkinetics on the
        network so far (from the seed, at `kinetics` temperature and time)
        puts material into it: its peak concentration or its concentration
        flux reaches `kinetics["threshold"]` (Bensberg & Reiher's two
        criteria). At most `max_expand_per_round` are expanded per round,
        the most populated first. Proposed products lying more than the
        cutoff above the seed are not connected (no route can reach them
        within it). A hotter or longer reaction (a larger cutoff, faster
        rates) explores more.

    `verify` (with steer="access"): steps `connect` returns with
    files["verified"] = False (a TS optimized without an IRC) enter the
    kinetics at once; after each round, those touching a populated species
    get an IRC (`verify([(step index, ts file)])` -> [(index, start, end) or
    None ends]), are moved to the species the IRC actually connects, or
    dropped if it fails, and the kinetics run again.

    `target`: a product to reach. The seed -> target pair is connected first
    (a two-ended recursive search; its intermediates join the network), and
    expansion then proceeds from every species within reach. With
    steer="access", a target the network can't yet reach within the cutoff
    (the route found climbs too high) is rerouted: each round, up to
    `reroute_attempts` of the most accessible species not yet tried get a
    two-ended search towards the target, and species on the target's side
    (connected to it by steps whose TS lies within floor + cutoff, and
    themselves within it) are expanded too, so routes around the blocking
    step can meet from both ends.

    `proposal_fmax` (eV/Angstrom): proposals are minimized only to this
    largest force -- enough to tell which species a proposal is and where its
    energy lies against the cutoff, and a floppy collision complex that would
    take hundreds of tight steps no longer holds up its round. Species whose
    free energies the kinetics need are minimized tightly (the engine's own
    threshold) first; IRC ends always are. None or 0: tight from the start.
    Engines without a force threshold (_takes_fmax) always minimize their own way.

    `steer_engine` (steer="flux"/"access"): a higher level of theory (e.g.
    DFT) whose single points steer: every species and TS the kinetics use
    gets G = G_explorer + (E_high - E_low) at the explorer's geometry (as
    mepd.discovery.energy_refine "single-point"), so which species are
    expanded and which steps get channel sampling follow the higher level's
    flux. Each geometry is computed once (kept in `steer_save_dir` when
    given, so a rerun picks them up). Which species lie within the barrier
    cutoff, and the live ranking of path-search pieces, stay at the
    explorer's level.

    `state_file`: the whole exploration (network, what was connected and
    searched, Hessian and steering caches, where in a round it is) is saved
    there at every stage of a round, and every couple of minutes while pairs
    are being connected. `resume`: carry on from it -- a run cut off by a time
    limit loses only the path searches of the pairs it had not finished, and
    `rounds` can be raised to grow a finished network further. The settings
    that decide the network (engines, steering, kinetics, cutoff, generator)
    must be the ones it was saved with.

    `schedule="async"` (steer="access", a connector with its own jobs: the
    HorizontalConnector) drops the rounds: a species' proposals are a job in
    the connector's queue, started the moment the live kinetics populate it
    within reach, and the pairs to its products join the same queue as they
    come in. IRC checks, channel sampling, the pieces it leaves at its depth
    limit and steering single points are jobs of the same queue too, each
    queued the moment the live kinetics (steered by the single points back so
    far) ask for it: no stage waits for another. No distance from the seed
    caps the growth (`rounds` is not used): the queue ends when the kinetics
    ask for nothing more ("converged"), or when `time_budget_s`, a stop
    request or `max_species` runs out -- then no new job starts and the
    running ones are cancelled (one record in `rounds`, its "stopped" saying which)."""
    if schedule not in ("rounds", "async"):
        raise ValueError(f"schedule must be 'rounds' or 'async', got {schedule!r}.")
    if schedule == "async":
        from mepd.discovery.generators import is_named as is_named_generator

        why = ("steer='access'" if steer != "access" else
               "a connector that runs the explorer's jobs (HorizontalConnector)" if not hasattr(connect, "register")
               else "no target (a two-ended start runs in rounds)" if target is not None
               else "a named generator (no products file)" if products_file is not None or not is_named_generator(generator)
               else None)
        if why:
            raise ValueError(f"schedule='async' needs {why}.")
    if steer not in ("window", "flux", "access"):
        raise ValueError(f"steer must be 'window', 'flux' or 'access', got {steer!r}.")
    if steer in ("flux", "access") and connect is None:
        raise ValueError(f"steer={steer!r} needs a `connect` callback that finds each reaction's TS.")
    if steer == "access" and access_cutoff_kcal is None:
        raise ValueError("steer='access' needs access_cutoff_kcal.")
    kin = {"temperature": 298.15, "time_s": 3600.0, "threshold": 0.01, **(kinetics or {})}
    # kinetics on free energies (kin["free_energy"]): thermal corrections from the engine's own Hessians
    free = None
    if kin.get("free_energy"):
        from mepd.discovery.free_energy import FreeEnergies

        free = FreeEnergies(engine, kin["temperature"], kin.get("standard_state", "1M"), workers)
    if getattr(connect, "ices", False):
        # each TS job computes its TS's thermal correction in its worker (free-energy gate, below)
        connect.free = free
    # higher-level single points that steer the kinetics (steer_engine): E_high - E_low per geometry (Hartree)
    steer_delta: dict = {}
    steer_saved = None
    steer_pool = [None]   # the spawned process a GPU steering level runs in
    if steer_engine is not None:
        from mepd.discovery.energy_refine import _level, _Saved

        steer_saved = _Saved(steer_save_dir, f"{_level(engine)} -> {_level(steer_engine)}")
    loose_fmax = float(proposal_fmax) if proposal_fmax and _takes_fmax(engine) else None
    from mepd.discovery.generators import get_generator, is_named
    from mepd.discovery.generators import propose as propose_products

    if products_file is None and is_named(generator):
        get_generator(generator).check()   # fail before any energy is computed
    # Every barrier and relative energy is measured from minima at this level,
    # so the seed must be one too: a SMILES embedding or a structure from
    # another level is not (and would give too-low, even negative, barriers).
    saved = None
    fingerprint = _state_fingerprint(engine, steer_engine, steer, kin, access_cutoff_kcal, generator,
                                     generator_options, products_file) if state_file is not None else None
    if fingerprint is not None and schedule != "rounds":
        fingerprint["schedule"] = schedule
    if resume:
        saved = _load_state(state_file, fingerprint)
        seed = saved["result"].species[0].node.copy()
        minimize_seed = False
    seed = seed.copy()
    if minimize_seed:
        opt = _optimize(engine, [StructureNode(structure=seed.structure)], maxiter, None)[0]
        if isinstance(opt, Exception):
            raise RuntimeError(f"Could not minimize the seed at this level of theory: {type(opt).__name__}: {opt}")
        if validate_minima is not None:
            from mepd.elementarystep import validate_minimum_with_rescue

            opt, record = validate_minimum_with_rescue(
                opt, engine, frequency_cutoff=float(validate_minima.get("frequency_cutoff", -20.0)),
                rescue_displacement=float(validate_minima.get("rescue_displacement", 0.3)), label="seed")
            if not record["is_minimum"]:
                raise RuntimeError(f"The seed is not a minimum at this level of theory: {record.get('validation')}")
        seed = opt
    elif seed._cached_energy is None:
        engine.compute_energies([seed])
    e0 = float(seed.energy)
    view = _View(_qmmm_region(engine, seed))
    symbols = view.symbols(seed)
    charge, mult = view.charge_mult(seed)
    result = ExpansionResult(species=[Species(seed, lewis_smiles(symbols, view.edges(seed), charge, mult,
                                                                 allow_radicals=True, allow_zwitterions=True) or "", 0)])
    if saved is not None:
        result = saved["result"]
    live = _LiveReactions(seed, e0, workers=workers)

    def _adopt_if_lower(k: int, node, loose: bool = False) -> None:
        """Keep the lowest conformer found as species k's reference."""
        if float(node.energy) < float(result.species[k].node.energy):
            result.species[k].node = node.copy()
            result.species[k].rel_energy_kcal = (float(node.energy) - e0) * HARTREE_TO_KCAL_PER_MOL
            result.species[k].loose = loose

    def _tighten(indices) -> None:
        """Minimize the loosely minimized species among `indices` to the
        engine's own threshold (their Hessians need a true minimum). One that
        changes connectivity on the way keeps its loose geometry (said)."""
        todo = [k for k in indices if result.species[k].loose]
        if not todo:
            return
        _phase("tighten", species=len(todo))
        opts = _optimize(engine, [StructureNode(structure=result.species[k].node.structure) for k in todo],
                         maxiter, None)
        for k, opt in zip(todo, opts):
            sp = result.species[k]
            sp.loose = False
            if isinstance(opt, Exception) or not _connectivity_matches(opt, sp.node):
                why = f"{type(opt).__name__}: {opt}" if isinstance(opt, Exception) else "its connectivity changed"
                _emit(on_event, "warning", message=f"species {k}: tight minimization not used ({why}); "
                                                   "its loosely minimized geometry is kept")
                continue
            sp.node = opt.copy()
            sp.rel_energy_kcal = (float(opt.energy) - e0) * HARTREE_TO_KCAL_PER_MOL

    def _classify(p, guess, opt, record) -> ProposedEdge:
        intended = _connectivity_matches(opt, guess) if p.broken or p.formed else None
        match = next((k for k, s in enumerate(result.species) if _connectivity_matches(opt, s.node)), None)
        if match is not None:
            _adopt_if_lower(match, opt, loose=loose_fmax is not None)
            return ProposedEdge(p.source, p, "reverted" if match == p.source else "known_species", match, intended)
        if len(result.species) >= max_species:
            return ProposedEdge(p.source, p, "failed", error="max_species reached")
        rel = (float(opt.energy) - e0) * HARTREE_TO_KCAL_PER_MOL
        smi = lewis_smiles(symbols, view.edges(opt), charge, mult, allow_radicals=True, allow_zwitterions=True) or ""
        result.species.append(Species(opt.copy(), smi, rnd, rel, record, loose=loose_fmax is not None))
        idx = len(result.species) - 1
        _emit(on_event, "species_found", index=idx, smiles=smi, rel_energy_kcal=rel, round=rnd)
        return ProposedEdge(p.source, p, "new_species", idx, intended)

    premin: dict = {}   # id(IRC end) -> its minimum, from _minimize_ends

    def _minimize_ends(nodes) -> None:
        """Minimize IRC ends all at once (the engine's parallelism), for _species_for to use."""
        todo = list({id(n): n for n in nodes if n is not None and id(n) not in premin}.values())
        if len(todo) > 1:
            for n, opt in zip(todo, _optimize(engine, [StructureNode(structure=n.structure) for n in todo],
                                              maxiter, None)):
                premin[id(n)] = (n, opt)

    def _species_for(node) -> tuple[int, bool]:
        """The species an IRC end is. The end is minimized first (an IRC stops
        short of the minimum). If it is a known species in a lower conformer
        than the one on record, that conformer becomes the species' reference
        (barriers are measured from the lowest conformer found); otherwise
        it joins the network as an intermediate."""
        hit = premin.pop(id(node), None)
        opt = hit[1] if hit is not None and hit[0] is node else \
            _optimize(engine, [StructureNode(structure=node.structure)], maxiter, None)[0]
        if isinstance(opt, Exception):
            return -1, False
        node = opt
        match = next((k for k, s in enumerate(result.species) if _connectivity_matches(node, s.node)), None)
        if match is not None:
            _adopt_if_lower(match, node)
            return match, False
        if len(result.species) >= max_species:
            return -1, False
        rel = (float(node.energy) - e0) * HARTREE_TO_KCAL_PER_MOL
        smi = lewis_smiles(symbols, view.edges(node), charge, mult, allow_radicals=True, allow_zwitterions=True) or ""
        result.species.append(Species(node.copy(), smi, rnd, rel))
        _emit(on_event, "species_found", index=len(result.species) - 1, smiles=smi, rel_energy_kcal=rel, round=rnd)
        return len(result.species) - 1, True

    def _new_pairs(round_edges: list[ProposedEdge], max_rel_kcal: Optional[float] = None) -> list:
        pairs = []
        for e in round_edges:
            if e.target is not None and e.target != e.source:
                if max_rel_kcal is not None and result.species[e.target].rel_energy_kcal > max_rel_kcal:
                    continue   # no route reaches a product this high within the cutoff
                pair = (e.source, e.target)
                if pair not in connected and pair[::-1] not in connected and pair not in pairs:
                    pairs.append(pair)
        return pairs

    def _connect(pairs: list, search: bool = False) -> None:
        """Verified steps (TS + IRC between two minima) for `pairs`; IRC ends
        that are no known species join the network as intermediates.
        `search`: a path search for every pair, no single-TS shortcut."""
        connected.update(pairs)
        _emit(on_event, "connecting", round=rnd, total=len(pairs))
        # pieces of earlier searches iced until material reaches them go into every call (the queue gates them)
        if not pairs and not getattr(connect, "iced", None):
            return
        # Saved with the state: a run cut off here resumes with the pairs not done yet (_resume_connect).
        pos["inflight"] = {"pairs": [list(p) for p in pairs], "search": bool(search), "done": [],
                           "stage": pos.get("current")}
        _save(force=True)
        if getattr(connect, "end_keywords", False) is None:
            connect.end_keywords = {"coordsys": "cart", "maxiter": int(maxiter)}   # as _optimize's
        if getattr(connect, "live", False):
            # each TS joins the network as it is found; the kinetics of what is found so far decide
            # which pieces of the path searches run (_piece_relevant)
            def found_one(rec):
                _add_found(rec)
                _save()

            def pair_done(pair):
                pos["inflight"]["done"].append(list(pair))
                _save(force=True)

            extra = {"on_pair_done": pair_done} if getattr(connect, "reports_pairs", False) else {}
            if getattr(connect, "ices", False) and free is not None:
                extra["free_cache"] = free._cache
            connect(pairs, [s.node for s in result.species], **({"search": True} if search else {}),
                    on_found=found_one, relevant=_piece_relevant, **extra)
        else:
            for found in connect(pairs, [s.node for s in result.species], **({"search": True} if search else {})):
                _add_found(found)
        pos["inflight"] = None
        if getattr(connect, "ices", False) and result.rounds:
            result.rounds[-1]["iced_pieces"] = len(connect.iced)
        _save(force=True)

    # label -> (a, b, highest path image, its file) of a path-search piece whose TS still awaits its IRC
    peak_fallback: dict = {}
    # species pairs whose TS failed its IRC check: channel sampling for them even before material gets there
    sample_more: set = set()
    # species pairs a generator proposed: each gets channel sampling once its source is populated, whatever
    # its single TS found
    proposed_pairs: set = set()

    def _add_found(found) -> None:
        """One TS found: its ends as species (an IRC end no known species is a new intermediate), the step."""
        from mepd.discovery.kinetics import Step

        if "pair" in found:   # a TS between two known species (fast connect): no ends to identify
            (a, new_a), (b, new_b) = (int(found["pair"][0]), False), (int(found["pair"][1]), False)
        else:
            for node, opt in zip((found["start"], found["end"]), found.get("ends") or ()):
                premin[id(node)] = (node, opt)     # minimized by the job that found the TS
            _minimize_ends([found["start"], found["end"]])
            (a, new_a), (b, new_b) = _species_for(found["start"]), _species_for(found["end"])
        for idx, new, parent in ((a, new_a, b), (b, new_b, a)):
            if new and live.enabled:  # an intermediate the IRC found: tell the live view
                live._event(ProposedEdge(parent, Proposal(parent, (), (), result.species[idx].smiles),
                                         "new_species", idx, None), result.species)
        if a == b or a < 0 or b < 0:
            return
        if found.get("peak") is not None and found.get("label"):
            peak_fallback[found["label"]] = (a, b, found["peak"], found["files"]["peak"])
        ts_e = float(found["ts"].energy)
        # Two path searches can find the same TS; counting it twice would
        # double that channel's rate. Distinct TSs between the same pair
        # stay (parallel channels).
        if any({st.a, st.b} == {a, b} and _same_ts(st, ts_e, found["ts"].structure) for st in result.steps):
            return
        # A path's highest image (no TS found) only bounds a barrier: a pair keeps one, and only while it
        # lies under every other step of the pair (a lower step makes the bound say nothing).
        same = [n for n, st in enumerate(result.steps) if {st.a, st.b} == {a, b}]
        if (found.get("files") or {}).get("ts_kind") == "path_peak":
            if any(result.steps[n].ts_energy <= ts_e for n in same):
                return
            drop = {n for n in same if result.steps[n].files.get("ts_kind") == "path_peak"}
        else:
            drop = {n for n in same if result.steps[n].files.get("ts_kind") == "path_peak"
                    and result.steps[n].ts_energy >= ts_e}
        if drop:
            result.steps[:] = [st for n, st in enumerate(result.steps) if n not in drop]
        result.steps.append(Step(a, b, ts_e, found.get("label", ""), dict(found.get("files") or {})))
        if live.enabled:   # the web graph fills that edge in (barrier, TS, IRC) as it happens
            from mepd.discovery.kinetics import negative_barrier_warning, raw_barriers

            # The raw barriers -- a negative one is shown and flagged, never hidden; the finished
            # run's summary replaces these if a species' energy drops later.
            fwd, rev = raw_barriers([result.steps[-1]], [float(s.node.energy) for s in result.species])[0]
            live._progress.append_live_event({
                "event": "step", "a": a, "b": b, "label": found.get("label", ""), "ts_energy": ts_e,
                "barrier_kcal": [fwd, rev],
                "warning": negative_barrier_warning(found.get("label", ""), a, b, fwd, rev)})

    def _same_ts(st, ts_e: float, ts_structure) -> bool:
        """Whether step `st`'s TS is this one (`ts_e`, Hartree; its Structure): the same energy (0.1 kcal/mol)
        and geometry (superimposable by a proper rotation and a permutation of identical atoms, so a mirror
        image -- the twin TS -- or another conformer's TS is another channel); energy alone when a geometry
        is missing or undecided."""
        from mepd.discovery.mirror import _angstrom, superimposable

        if abs(st.ts_energy - ts_e) * HARTREE_TO_KCAL_PER_MOL >= 0.1:
            return False
        other = _ts_structure(st)
        if other is None or ts_structure is None or list(map(str, other.symbols)) != list(map(str, ts_structure.symbols)):
            return True
        return superimposable(list(map(str, other.symbols)), _angstrom(other), _angstrom(ts_structure)) is not False

    live_kin = {"n_steps": -1, "hot": set(), "version": 0}
    live_steer = [None]   # schedule async: queues a steering single point (the live kinetics are steered too)
    known_end: dict = {}
    # every structure counts with its mirror image (mepd.discovery.mirror): the twin of a chiral TS
    mirror_finder = None
    if kin.get("mirror_images", True):
        from mepd.discovery.mirror import MirrorFinder

        mirror_finder = MirrorFinder()
    mirror_said: set = set()
    split_cache: dict = {}

    def _mirror_of():
        """Each species' mirror image in the network (-1: not found), or None without mirror images."""
        return None if mirror_finder is None else mirror_finder.species_mirrors([s.node for s in result.species])

    def _simulate(steps, energies, free_basis: bool = False):
        """The kinetics of the network found so far (`steps`/`energies`: one
        energy view of result.steps/result.species), every structure counted
        with its mirror image; `free_basis`: the species count as the
        molecules the free energies split them into."""
        from mepd.discovery.kinetics import simulate

        n = len(result.species)
        if mirror_finder is None or not steps:
            return simulate(n, steps, energies, temperature=kin["temperature"], time_s=kin["time_s"])

        def parts(structure):
            from mepd.discovery.mirror import _key

            key = _key(structure)
            if key not in split_cache:
                split_cache[key] = [g for g, _, _ in free.parts(structure, False)]
            return split_cache[key]

        mirrors = mirror_finder([s.node for s in result.species], result.steps,
                                [_ts_structure(st) for st in result.steps],
                                parts if free_basis and free is not None else None)
        for msg in mirrors.undecided:
            if msg not in mirror_said:
                mirror_said.add(msg)
                _emit(on_event, "warning", message=f"mirror images: {msg}")
        n_all, all_steps, all_energies = mirrors.extend(steps, energies, kin["temperature"])
        k = simulate(n_all, all_steps, all_energies, temperature=kin["temperature"], time_s=kin["time_s"])
        k.n_found, k.mirrors = n, mirrors.record()
        return k

    def _live_hot(final: bool = False) -> set:
        """The species populated (max concentration or flux >= threshold) in the kinetics of the network
        found so far (free energies at the explorer's level when the kinetics use them -- each TS's thermal
        correction comes from the job that found it -- else electronic; recomputed whenever a step was
        added). `live_kin["load"]`: each one's larger of the two, to rank them."""
        import time

        # The kinetics of 400 species take ~5 s (mirror images included), and the connector waits for this
        # answer before it starts its next job: refreshed when steps were added, but at most once in three
        # times as long as the last refresh took (always for the final say), so the workers keep running.
        stale = (len(result.steps), live_kin["version"]) != live_kin["n_steps"]
        if stale and (final or time.monotonic() - live_kin.get("at", 0.0) >= 3.0 * live_kin.get("took", 0.0)):
            t0 = time.monotonic()
            energies = [float(s.node.energy) for s in result.species]
            g, steps, basis = (_free_energy_view(energies, live_steer[0]) if live_steer[0] is not None
                               else _explorer_view(energies))
            k = _simulate(steps, g, basis.startswith("free"))
            live_kin["k"] = k
            tau = float(kin["threshold"])
            live_kin["hot"] = {0} | {i for i in range(len(result.species))
                                     if k.max_concentration[i] >= tau or k.flux[i] >= tau}
            live_kin["load"] = {i: max(k.max_concentration[i], k.flux[i]) for i in range(len(result.species))}
            _show_kinetics(k, basis)
            live_kin["n_steps"] = (len(result.steps), live_kin["version"])
            live_kin["at"], live_kin["took"] = time.monotonic(), time.monotonic() - t0
        return live_kin["hot"]

    def _show_kinetics(k, basis: str) -> None:
        """The live kinetics as the web's live view shows them: each species' population, which ones are
        populated (they decide what runs next) and expanded, and what energies they rest on."""
        from mepd.progress import write_live_state

        tau = float(kin["threshold"])
        write_live_state("kinetics", "kinetics", {
            "basis": basis, "threshold": tau, "steps": len(result.steps),
            "steered": sum(v is not None for v in steer_delta.values()),
            "species": [{"index": i, "smiles": sp.smiles, "rel_kcal": sp.rel_energy_kcal,
                         "max_concentration": float(k.max_concentration[i]), "flux": float(k.flux[i]),
                         "final": float(k.final_concentration[i]), "expanded": i in expanded,
                         "populated": i == 0 or k.max_concentration[i] >= tau or k.flux[i] >= tau}
                        for i, sp in enumerate(result.species)]})

    def _piece_relevant(chain, final: bool = False):
        """Run a waiting piece of a path search only if material reaches one of its ends: a known species
        the live kinetics populate (_live_hot)."""
        hot = _live_hot(final)
        for node in (chain[0], chain[-1]):
            # (the species it is, or None and how many species it was compared with): asked for every waiting
            # piece each time a job ends, an end that is no species yet is compared only with the new ones
            held, idx, checked = known_end.get(id(node), (None, None, 0))
            if held is not node:   # (kept alive with its entry: an id is only unique while its object lives)
                idx, checked = None, 0
            if idx is None and checked < len(result.species):
                idx = next((i for i in range(checked, len(result.species))
                            if _connectivity_matches(node, result.species[i].node)), None)
                known_end[id(node)] = (node, idx, len(result.species))
            if idx is not None and idx in hot:
                return True
        return None

    def _kinetics_record(pairs: list):
        from mepd.discovery.kinetics import negative_barrier_warnings

        energies = [float(s.node.energy) for s in result.species]
        k_energies, k_steps, basis = _free_energy_view(energies)
        energies = [float(s.node.energy) for s in result.species]   # tightened species
        kinetic = _simulate(k_steps, k_energies, basis.startswith("free"))
        # a step closed out of reach has no barrier of its own (its TS + 1 Hartree only stops its rate)
        kinetic.barriers_kcal = [None if m < len(k_steps) and k_steps[m].files.get("closed") else b
                                 for m, b in enumerate(kinetic.barriers_kcal)]
        result.kinetics = kinetic
        warnings = negative_barrier_warnings(result.steps, energies)
        for w in warnings:
            _emit(on_event, "warning", message=w)
        result.rounds[-1]["kinetics"] = {
            "connected_pairs": [list(p) for p in pairs], "verified_steps": len(result.steps),
            "flux": kinetic.flux, "max_concentration": kinetic.max_concentration,
            "final_concentration": kinetic.final_concentration, "warnings": warnings, "energies": basis,
            **({"mirror_images": kinetic.mirrors} if kinetic.mirrors is not None else {})}
        return kinetic

    def _ts_structure(st):
        fp = (st.files or {}).get("ts")
        if not fp or not Path(fp).exists():
            return None
        ref = result.species[0].node.structure
        return _open_structure(fp).model_copy(update={"charge": ref.charge, "multiplicity": ref.multiplicity})

    def _ts_ends(st):
        """Step `st`'s two ends in its TS's atom order (free_energy.step_ends)."""
        from mepd.discovery.free_energy import step_ends

        return step_ends(st.files or {}, (result.species[st.a].node.structure, result.species[st.b].node.structure))

    def _steer_shifts(structs: list, queue=None) -> list:
        """E_high - E_low (Hartree) of each (structure, is_ts, a TS's step's ends or None), the molecules of a
        species -- a TS's spectators -- apart when the free energies count them so; None where a single point
        failed. `queue(key, item)`: the missing single points are handed to it (jobs of the async queue) and are
        None until they come back."""
        import hashlib

        from mepd.cli_common import _fork_map
        from mepd.discovery.energy_refine import Item, _fragments, _single_point
        from mepd.discovery.free_energy import FreeEnergies

        splitter = free or FreeEnergies(engine, kin["temperature"], kin.get("standard_state", "1M"))
        keys = []
        todo: dict = {}
        for s, is_ts, ends in structs:
            # a TS with spectators apart is another item than the TS whole (keys saved before kept theirs)
            groups = splitter._groups(s, is_ts, ends) if free is not None else None
            xyz = np.round(np.asarray(s.geometry, dtype=float), 4) + 0.0
            key = hashlib.sha1(repr((list(map(str, s.symbols)), xyz.tolist(), int(s.charge), int(s.multiplicity),
                                     is_ts) + ((groups,) if groups and len(groups) > 1 else ()))
                               .encode()).hexdigest()[:24]
            keys.append(key)
            if key in steer_delta or key in todo:
                continue
            done = steer_saved.get("single_points", key)
            if done is not None:
                steer_delta[key] = done["delta_kcal"] / HARTREE_TO_KCAL_PER_MOL
                continue
            parts = [s] if free is None else _fragments(s, splitter.parts(s, is_ts, ends))
            todo[key] = Item("ts" if is_ts else "species", 0, s, parts)
        if todo and queue is not None:
            for key, item in todo.items():
                queue(key, item)
            return [steer_delta.get(k) for k in keys]
        if todo:
            _emit(on_event, "steer_single_points", round=rnd, total=len(todo))
            at_once = max(1, int(workers) // max(1, int(getattr(steer_engine, "threads", 1) or 1)))

            def one(item):
                try:
                    return _single_point(item, engine, steer_engine)
                except Exception as exc:
                    return f"{type(exc).__name__}: {exc}"

            if getattr(steer_engine, "holds_torch_model", False):
                # a GPU model: loaded once in a spawned process of its own, so this one never initialises CUDA
                # (it could no longer fork its workers) and no worker loads a copy of the model
                if steer_pool[0] is None:
                    import multiprocessing
                    from concurrent.futures import ProcessPoolExecutor

                    steer_pool[0] = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                                        initializer=_steer_init, initargs=(engine, steer_engine))
                done = list(steer_pool[0].map(_steer_point, todo.values()))
            else:
                done = _fork_map(one, list(todo.values()), at_once)
            for key, r in zip(todo, done):
                if isinstance(r, str):
                    _emit(on_event, "warning", message=f"steering single point failed ({r}): explorer energy used")
                    steer_delta[key] = None
                    continue
                steer_saved.put("single_points", key, r)
                steer_delta[key] = r["delta_kcal"] / HARTREE_TO_KCAL_PER_MOL
        return [steer_delta[k] for k in keys]

    def _free_energy_view(energies, queue=None):
        """(species energies, steps, basis) for the kinetics: the explorer's
        (_explorer_view), moved by the steering level's single points when
        there is one (`queue`: see _steer_shifts)."""
        from mepd.discovery.kinetics import Step

        g, steps, basis = _explorer_view(energies)
        if steer_engine is None or not result.steps:
            return g, steps, basis
        from mepd.discovery.energy_refine import kinetic_reach
        from mepd.discovery.free_energy import channels_shift

        reach, open_steps = kinetic_reach(result.steps, [float(s.node.energy) for s in result.species], e0,
                                          access_cutoff_kcal, _mirror_of())
        sp_idx = sorted(reach)
        ts = {m: _ts_structure(result.steps[m]) for m in open_steps}
        open_steps = [m for m in open_steps if ts[m] is not None]
        ends = {m: _ts_ends(result.steps[m]) for m in open_steps}
        # a TS with spectators is counted apart and bound, the lower kept (free_energy.both_channels): each its own shift
        channels = {m: c for m in open_steps
                    if free is not None and (c := free.ts_channels(ts[m], ends[m])) is not None}
        bound = list(channels)
        d = _steer_shifts([(result.species[k].node.structure, False, None) for k in sp_idx]
                          + [(ts[m], True, ends[m]) for m in open_steps] + [(ts[m], True, None) for m in bound], queue)
        d_sp, d_ts = dict(zip(sp_idx, d[:len(sp_idx)])), dict(zip(open_steps, d[len(sp_idx):]))
        d_bound = dict(zip(bound, d[len(sp_idx) + len(open_steps):]))
        if d_sp.get(0) is None and queue is not None:
            return g, steps, basis     # the seed's single point is still running
        if d_sp.get(0) is None:
            _emit(on_event, "warning", message="the seed's steering single point failed: kinetics at the "
                                               "explorer's level this round")
            return g, steps, basis
        # relative to the seed, on the explorer's scale (the rest of the network keeps the explorer's energies)
        g = [e + (d_sp[k] - d_sp[0] if d_sp.get(k) is not None else 0.0) for k, e in enumerate(g)]
        for m, (ca, cb) in channels.items():
            if d_ts.get(m) is not None and d_bound.get(m) is not None:
                # both channels at the steering level: what moves their combination
                d_ts[m] = channels_shift(ca, cb, d_ts[m] * HARTREE_TO_KCAL_PER_MOL, d_bound[m] * HARTREE_TO_KCAL_PER_MOL,
                                         kin["temperature"]) / HARTREE_TO_KCAL_PER_MOL
            else:
                d_ts[m] = None
        steps = [Step(st.a, st.b, st.ts_energy + (d_ts[m] - d_sp[0] if d_ts.get(m) is not None else 0.0),
                      st.label, st.files) for m, st in enumerate(steps)]
        n = sum(x is not None for x in d)
        return g, steps, f"{basis}; {type(steer_engine).__name__} single points on {n} of {len(d)} structures"

    def _explorer_view(energies):
        """(species energies, steps, basis) for the kinetics: free energies when
        asked for and every correction worked, else the electronic ones (said)."""
        from mepd.discovery.kinetics import Step

        if free is None or not result.steps:
            return energies, result.steps, "electronic"
        # Only the part of the network some route reaches within the cutoff gets Hessians: the rest carries
        # no flux. Its steps are closed (TS + 1 Hartree), its species keep their electronic energy.
        from mepd.discovery.energy_refine import kinetic_reach

        reach, open_steps = kinetic_reach(result.steps, energies, e0, access_cutoff_kcal, _mirror_of())
        _tighten(sorted(reach))   # their Hessians need true minima
        energies = [float(s.node.energy) for s in result.species]
        ts_structs = []
        ref = result.species[0].node.structure
        for st in (result.steps[m] for m in open_steps):
            fp = (st.files or {}).get("ts")
            if not fp or not Path(fp).exists():
                _emit(on_event, "warning", message=f"step {st.label} has no TS geometry: kinetics on electronic "
                                                   "energies this round")
                return energies, result.steps, "electronic"
            ts = _open_structure(fp)
            ts_structs.append(ts.model_copy(update={"charge": ref.charge, "multiplicity": ref.multiplicity}))
        sp_idx = sorted(reach)
        corr = free.corrections([result.species[k].node.structure for k in sp_idx] + ts_structs,
                                [False] * len(sp_idx) + [True] * len(ts_structs),
                                [None] * len(sp_idx) + [_ts_ends(result.steps[m]) for m in open_steps])
        for note in dict.fromkeys(free.notes):
            _emit(on_event, "warning", message=f"free energies: {note}")
        free.notes.clear()
        if any(c is None for c in corr):
            return energies, result.steps, "electronic"
        sp_corr = dict(zip(sp_idx, corr[:len(sp_idx)]))
        ts_corr = dict(zip(open_steps, corr[len(sp_idx):]))
        g = [e + sp_corr.get(k, 0.0) / HARTREE_TO_KCAL_PER_MOL for k, e in enumerate(energies)]
        # a closed step: TS + 1 Hartree, no rate -- marked, so no one reads that as its barrier
        steps = [Step(st.a, st.b, st.ts_energy + ts_corr[m] / HARTREE_TO_KCAL_PER_MOL, st.label, st.files)
                 if m in ts_corr else Step(st.a, st.b, st.ts_energy + 1.0, st.label, {**st.files, "closed": True})
                 for m, st in enumerate(result.steps)]
        closed = len(result.steps) - len(open_steps)
        return g, steps, (f"free energy ({kin['temperature']:g} K, {kin.get('standard_state', '1M')})"
                          + (f"; {closed} step(s) out of reach closed" if closed else ""))

    def _steer_by_flux(round_edges: list[ProposedEdge]) -> list[int]:
        from mepd.discovery.kinetics import select_for_expansion

        ctx = pos["ctx"]
        ctx.setdefault("done", [])
        if "pairs" not in ctx:
            ctx["pairs"] = _new_pairs(round_edges)
        pairs = ctx["pairs"]
        if _stage("connected"):
            _connect(pairs)
            _stage_done("connected")
        _search_deferred([])   # iced channel-search pieces the round's new steps now send material to
        kinetic = _kinetics_record(pairs)
        picks = select_for_expansion(kinetic, expanded, kin["threshold"])
        _emit(on_event, "kinetics", round=rnd, flux=kinetic.flux, picks=picks)
        return picks

    def _reroute_attempts(acc) -> list:
        """Two-ended searches towards the target from the most accessible
        species not yet tried -- none if a route already reaches it."""
        if t_idx is None or reroute_attempts <= 0 or (t_idx in acc and acc[t_idx].span <= access_cutoff_kcal):
            return []
        within = [k for k, st in sorted(acc.items(), key=lambda kv: kv[1].span)
                  if st.span <= access_cutoff_kcal and k != t_idx]
        attempts = [(k, t_idx) for k in within if (k, t_idx) not in connected and (t_idx, k) not in connected]
        return attempts[:reroute_attempts]

    def _reroute(acc, attempts=None) -> tuple:
        """Target not reached within the cutoff: the target-side species to
        expand, and (`attempts` None) two-ended searches towards it run now.
        A round passes the attempts it already ran inside its own connect
        (one queue, no extra phase waiting on its slowest search). Returns
        (acc after any new searches, extra frontier); records what blocked
        the route."""
        from mepd.network_aware import accessibility, rel_kcal

        info = {"reached": t_idx in acc and acc[t_idx].span <= access_cutoff_kcal}
        if t_idx in acc:
            info["route"] = acc[t_idx].as_dict()
        if info["reached"] or reroute_attempts <= 0:
            result.rounds[-1]["target_route"] = info
            return acc, []
        if attempts is None:
            attempts = _reroute_attempts(acc)
            if attempts:
                _connect(attempts)
                acc = accessibility(result.steps, [float(sp.node.energy) for sp in result.species], e0)
        energies = [float(sp.node.energy) for sp in result.species]
        rel = [rel_kcal(e, e0) for e in energies]
        # the target's side: reachable from it over steps that could lie on a route within the cutoff
        side, todo = {t_idx}, [t_idx]
        while todo:
            u = todo.pop()
            for st in result.steps:
                for a, b in ((st.a, st.b), (st.b, st.a)):
                    if a == u and b not in side and rel_kcal(st.ts_energy, e0) <= access_cutoff_kcal \
                            and rel[b] <= access_cutoff_kcal:
                        side.add(b)
                        todo.append(b)
        blocking = [{"a": st.a, "b": st.b, "ts_rel_kcal": round(rel_kcal(st.ts_energy, e0), 2)}
                    for st in result.steps
                    if ((st.a in acc and acc[st.a].span <= access_cutoff_kcal) != (st.b in acc and acc[st.b].span <= access_cutoff_kcal))]
        info.update(reached=t_idx in acc and acc[t_idx].span <= access_cutoff_kcal,
                    tried_to_target=[list(p) for p in attempts], target_side=sorted(side), blocking_steps=blocking)
        if t_idx in acc:
            info["route"] = acc[t_idx].as_dict()
        result.rounds[-1]["target_route"] = info
        _emit(on_event, "reroute", round=rnd, **{k: v for k, v in info.items() if k != "blocking_steps"})
        extra = [k for k in side if k not in expanded and rel[k] <= access_cutoff_kcal] if not info["reached"] else []
        return acc, extra

    def _verify_needed(kinetic) -> int:
        """IRC-check the unverified steps that touch a populated species."""
        if verify is None:
            return 0
        tau = float(kin["threshold"])
        hot = {k for k in range(min(len(kinetic.flux), len(result.species)))
               if kinetic.max_concentration[k] >= tau or kinetic.flux[k] >= tau}
        items = [(n, st.files.get("ts")) for n, st in enumerate(result.steps)
                 if st.files.get("verified") is False and st.files.get("ts") and (st.a in hot or st.b in hot)]
        if not items:
            return 0
        checked = list(verify(items))
        _minimize_ends([n for _, start, end in checked for n in (start, end)])
        stepped = [(result.steps[idx], start, end) for idx, start, end in checked]
        for st, start, end in stepped:
            _irc_checked(st, start, end)
        return len(items)

    def _irc_checked(st, start, end) -> None:
        """Step `st`'s IRC ends (None: its IRC failed): the step under its real ends, or dropped."""
        asked = tuple(st.files.get("pair") or ())
        piece = (st.a, st.b)
        dropped = start is None
        duplicate = False
        if dropped:
            missed = asked
        else:
            a, new_a = _species_for(start)
            b, new_b = _species_for(end)
            # A species no proposal made, found as this IRC's end: a product of the species the search was asked
            # from, with no route to it but this step (F0007: an adduct as the end of a TS between two other
            # species, 48-81 kcal/mol up, in 11 of the 16 baseline cases of a channel found but not connected)
            if asked:
                for k, new in ((a, new_a), (b, new_b)):
                    if new and k not in asked and result.species[k].rel_energy_kcal <= (
                            access_cutoff_kcal if access_cutoff_kcal is not None else float("inf")):
                        irc_products.append((int(asked[0]), k))
            if a < 0 or b < 0 or a == b:
                dropped = True
            else:
                st.a, st.b = a, b
                st.files["verified"] = True
                # its real ends may be those of a step with this very TS (found from another pair): one channel
                duplicate = any(o is not st and {o.a, o.b} == {a, b} and _same_ts(o, st.ts_energy, _ts_structure(st))
                                for o in result.steps)
                irc_fp = Path(st.files["ts"]).with_name(f"{Path(st.files['ts']).stem}_irc.xyz") \
                    if st.files.get("ts") else None
                if irc_fp is not None and irc_fp.exists() and not st.files.get("irc"):
                    st.files["irc"] = str(irc_fp)    # the path through the TS, for the web's step view
            missed = asked if asked and set(asked) != {a, b} else ()
        # The single TS optimization found a TS of some other step (kept,
        # under its real ends) or none: the pair asked for still needs its
        # path search -- the steps between them, intermediates included.
        if len(missed) == 2 and tuple(missed) not in searched:
            retry.append(tuple(missed))
        if dropped or duplicate:
            result.steps[:] = [x for x in result.steps if x is not st]
        # A path-search piece whose TS turned out to belong to another motion (both IRC ends one
        # species, or other species) keeps its path's highest image, as when no TS is found at all.
        fb = peak_fallback.pop(st.label, None)
        if fb is not None and (dropped or {st.a, st.b} != set(piece)):
            # ... and the pair gets all its paths sampled (conformers, mechanisms): the real TS between
            # these two ends is what the kinetics need, not the bound
            sample_more.add(frozenset(piece))
            fa, fb_, peak, fp = fb
            _add_found({"pair": (fa, fb_), "ts": peak, "label": f"{Path(fp).stem}",
                        "files": {"ts": fp, "verified": True, "ts_kind": "path_peak"}})

    def _verify_and_search(kinetic) -> int:
        """IRC-check the steps the kinetics need; pairs a single TS didn't
        connect get a full path search (its steps checked the same way)."""
        checked = _verify_needed(kinetic)
        while retry:
            # each search's first piece runs; its deeper pieces only once the kinetics put material at an end
            todo = list(dict.fromkeys(retry))
            retry.clear()
            searched.update(todo)
            _connect(todo, search=True)
            checked += _verify_needed(_kinetics_record(todo))
        return checked

    refined: set = set()

    def _refine_todo(kinetic) -> list:
        """(pair key, a, b) of every step the kinetics need sampled now (see `_refine_steps`)."""
        if refine is None or not (result.steps or proposed_pairs):
            return []
        from mepd.network_aware import accessibility

        energies = [float(sp.node.energy) for sp in result.species]
        if access_cutoff_kcal is not None:
            acc = accessibility(result.steps, energies, e0)
            reach = {k for k, st in acc.items() if st.span <= access_cutoff_kcal} | {0}
        else:
            reach = set(range(len(result.species)))
        tau = float(kin["threshold"])
        hot = {0} | {k for k in range(min(len(kinetic.flux), len(result.species)))
                     if kinetic.max_concentration[k] >= tau or kinetic.flux[k] >= tau}
        todo = []
        for st in result.steps:
            key = frozenset((st.a, st.b))
            if key in refined or key in {k for k, _, _ in todo}:
                continue
            open_ = access_cutoff_kcal is None or (st.ts_energy - e0) * HARTREE_TO_KCAL_PER_MOL <= access_cutoff_kcal
            if st.a in reach and st.b in reach and open_ and (st.a in hot or st.b in hot or key in sample_more):
                todo.append((key, st.a, st.b))
        # a proposed product of a populated species, whatever its single TS found: that one guess can land on
        # another mechanism and leave the direct one unsearched (formic acid: the carbene's TS instead of the
        # four-centre dehydration's)
        for key in sorted(proposed_pairs - refined - {k for k, _, _ in todo}, key=sorted):
            a, b = sorted(key)
            if a in hot or b in hot:
                todo.append((key, a, b))
        return todo

    def _prefetch_steps(kinetic) -> None:
        """Start sampling the steps these kinetics would send to `refine` (`refine.prefetch`, if it has
        one) while the round's verification runs: its tail leaves most cores idle, and `refine` then finds
        those steps done -- or samples them again if their species changed meanwhile."""
        prefetch = getattr(refine, "prefetch", None)
        todo = _refine_todo(kinetic) if prefetch is not None else []
        if todo:
            prefetch([(result.species[a].node, result.species[b].node, f"{a}_{b}") for _, a, b in todo])

    def _refine_steps(kinetic) -> int:
        """Channel sampling (`refine`, e.g. `mepd channels`: conformers and mechanisms of both ends, a TS and
        IRC per distinct path) for every step the kinetics need -- open within the cutoff and touching a
        populated species, or between a pair whose TS failed its IRC check -- once per species pair. Its
        direct channels replace the step's single TS; multistep routes and off-target exits join the network
        with whatever new species their IRCs end in; lower end conformers become the species' references.
        Returns the number of pairs sampled."""
        todo = _refine_todo(kinetic)
        if not todo:
            if getattr(refine, "prefetch", None) is not None:
                refine([])   # ends a background sampling (`_prefetch_steps`) of steps no longer needed
            return 0
        _emit(on_event, "refining", round=rnd, total=len(todo))
        outs = refine([(result.species[a].node, result.species[b].node, f"{a}_{b}") for _, a, b in todo])
        _minimize_ends([n for out in outs if out for n in out.get("conformers", [])]
                       + [rec[k] for out in outs if out for rec in out.get("direct", []) + out.get("other", [])
                          if "pair" not in rec for k in ("start", "end")])
        for (key, a, b), out in zip(todo, outs):
            _channels_done(key, a, b, out)
        result.rounds[-1].setdefault("channels_sampled", []).extend([a, b] for _, a, b in todo)
        _search_deferred([d for out in outs if out for d in out.get("deferred", [])])
        return len(todo)

    def _channels_done(key, a, b, out) -> None:
        """Step (a, b)'s channel sampling is back (`out`; None: it failed and the step keeps its single TS)."""
        refined.add(key)
        if out:
            for node in out.get("conformers", []):
                k, _ = _species_for(node)          # the lowest conformer of each end becomes the reference
            before = [st for st in result.steps if frozenset((st.a, st.b)) == key]
            if out.get("direct"):
                # sampled TSs of this step supersede the single one (counting both would double its rate)
                result.steps[:] = [st for st in result.steps if frozenset((st.a, st.b)) != key]
            for rec in out.get("direct", []) + out.get("other", []):
                _add_found(rec)
            after = [st for st in result.steps if frozenset((st.a, st.b)) == key]
            # what sampling changed: the step's TSs before and after, so how far a first TS overestimates its
            # barrier can be measured (the cutoff decides which steps get sampled from the first TS alone)
            result.channels_log.append({
                "a": a, "b": b, "round": rnd, "direct": len(out.get("direct", [])),
                "before": [{"label": st.label, "ts_energy": st.ts_energy, "kind": st.files.get("ts_kind", "ts"),
                            "ts": st.files.get("ts")} for st in before],
                "after": [{"label": st.label, "ts_energy": st.ts_energy} for st in after]})

    def _search_deferred(pieces: list) -> None:
        """Pieces the channel searches left unsplit at their depth limit ({"start", "end"}): searched by the
        connector, as pairs of their own, only where the kinetics now put material at one of their ends --
        the test the connector's own waiting pieces get (_piece_relevant)."""
        if connect is None:
            return
        # pieces earlier calls left out are asked again: a step found since may send material their way
        pieces = deferred_iced + list(pieces)
        if not pieces:
            return
        hot = [p for p in pieces if _piece_relevant((p["start"], p["end"]), final=True)]
        deferred_iced[:] = [p for p in pieces if not any(p is h for h in hot)]
        _minimize_ends([p[k] for p in hot for k in ("start", "end")])
        pairs = []
        for p in hot:
            a, _ = _species_for(p["start"])
            b, _ = _species_for(p["end"])
            if min(a, b) < 0 or a == b or (a, b) in connected or (b, a) in connected or (a, b) in pairs:
                continue
            pairs.append((a, b))
        result.rounds[-1].setdefault("deferred", []).append({"pieces": len(pieces), "hot": len(hot),
                                                             "iced": len(deferred_iced),
                                                             "searched": [list(p) for p in pairs]})
        _connect(pairs)

    def _steer_by_access(round_edges: list[ProposedEdge]) -> list[int]:
        from mepd.network_aware import accessibility, log_event

        ctx, done = pos["ctx"], pos["ctx"].setdefault("done", [])
        if "pairs" not in ctx:
            pairs = _new_pairs(round_edges, max_rel_kcal=access_cutoff_kcal)
            ctx["skipped"] = len(_new_pairs(round_edges)) - len(pairs)
            # Searches towards a target no route reaches yet go in the same queue
            # as the round's pairs, from the species reachable before this round.
            attempts = []
            if t_idx is not None:
                attempts = [p for p in _reroute_attempts(
                    accessibility(result.steps, [float(s.node.energy) for s in result.species], e0)) if p not in pairs]
            ctx["pairs"], ctx["attempts"] = pairs, attempts
        pairs, attempts, skipped = ctx["pairs"], ctx["attempts"], ctx["skipped"]
        proposed_pairs.update(map(frozenset, pairs))
        if _stage("connected"):
            _connect(pairs + attempts)
            _stage_done("connected")
        _phase("connected", round=rnd, pairs=len(pairs) + len(attempts))
        if verify is not None:
            if _stage("verified"):
                kinetic = _kinetics_record(pairs)
                if "channels" not in ctx.get("done", []):
                    _prefetch_steps(kinetic)
                ctx["irc_verified"] = _verify_and_search(kinetic)
                _stage_done("verified")
            result.rounds[-1]["irc_verified"] = ctx["irc_verified"]
            _phase("verified", round=rnd, ircs=ctx["irc_verified"])
        if _stage("channels"):
            ctx["channels"] = _refine_steps(_kinetics_record(pairs))
            _stage_done("channels")
        _phase("channels", round=rnd, steps=ctx["channels"])
        _search_deferred([])   # iced channel-search pieces the round's new steps now send material to
        acc = accessibility(result.steps, [float(s.node.energy) for s in result.species], e0)
        extra = []
        if t_idx is not None:
            acc, extra = _reroute(acc, attempts)
        kinetic = _kinetics_record(pairs + attempts)
        tau = float(kin["threshold"])
        within = [k for k, st in acc.items() if st.span <= access_cutoff_kcal and k not in expanded]
        populated = [k for k in within if k < len(kinetic.flux)
                     and (kinetic.max_concentration[k] >= tau or kinetic.flux[k] >= tau)]
        populated.sort(key=lambda k: (-kinetic.max_concentration[k], -kinetic.flux[k]))
        picks = populated[:max(0, int(max_expand_per_round))]
        picks += [k for k in extra if k not in picks][:max(0, int(reroute_attempts))]
        result.rounds[-1]["access_gate"] = {"within_reach_unexpanded": len(within), "populated": len(populated),
                                            "capped_at": int(max_expand_per_round)}
        result.rounds[-1]["access"] = {
            "cutoff_kcal": access_cutoff_kcal, "pairs_not_connected_above_cutoff": skipped,
            "within_reach": {k: st.as_dict() for k, st in acc.items() if st.span <= access_cutoff_kcal},
            "out_of_reach": {k: st.as_dict() for k, st in acc.items() if st.span > access_cutoff_kcal}}
        log_event({"event": "access_round", "round": rnd, "species": len(result.species), "steps": len(result.steps),
                   "connected": len(pairs), "not_connected_above_cutoff": skipped, "expand_next": picks})
        _emit(on_event, "access", round=rnd, picks=picks, within=len(result.rounds[-1]["access"]["within_reach"]))
        return picks

    def _stage(name) -> bool:
        """True when this round's stage `name` is still to run (a resumed round skips the ones done)."""
        if name in pos["ctx"].get("done", []):
            return False
        pos["current"] = name
        return True

    def _stage_done(name) -> None:
        pos["ctx"].setdefault("done", []).append(name)
        pos["current"] = None
        _save(force=True)

    def _resume_connect() -> None:
        """The connect call a resumed run was cut off in: the pairs it had not finished."""
        fl = pos.get("inflight")
        if not fl:
            return
        finished = {tuple(p) for p in fl["done"]}
        todo = [tuple(p) for p in fl["pairs"] if tuple(p) not in finished]
        pos["current"], pos["inflight"] = fl.get("stage"), None
        _emit(on_event, "resumed_connect", pairs=len(todo), finished=len(finished))
        _connect(todo, search=fl["search"])
        if fl.get("stage") == "connected":
            _stage_done("connected")    # the round's own connect call: nothing else of it to run

    def _phase(name, **kw):
        from mepd.network_aware import log_event

        log_event({"event": "phase", "phase": name, **kw})

    connected: set[tuple[int, int]] = set()
    deferred_iced: list = []                # channel-search pieces no material reached yet (_search_deferred)
    retry: list = []                        # pairs whose single TS turned out to belong elsewhere
    irc_products: list = []                 # (asked source, species first found as an IRC end): connected (async)
    searched: set[tuple[int, int]] = set()  # pairs that already had their full path search
    expanded: set[int] = set()
    frontier = [0]
    t_idx = None
    # Where the run is, saved with the state: the round, its stage, what the round decided so far
    # ("ctx"), the stages done, and a connect call in flight.
    pos: dict = {"round": 1, "stage": "propose", "ctx": {}, "inflight": None, "current": None}
    last_save = [0.0]

    def _save(force: bool = False) -> None:
        """The state to `state_file` -- at once with `force`, else at most every two minutes."""
        import time

        if state_file is None or (not force and time.monotonic() - last_save[0] < 120.0):
            return
        last_save[0] = time.monotonic()
        _write_state(state_file, {
            "fingerprint": fingerprint,
            "result": result, "pos": pos, "connected": connected, "retry": retry, "searched": searched,
            "expanded": expanded, "frontier": frontier, "t_idx": t_idx, "refined": refined,
            "sample_more": sample_more, "peak_fallback": peak_fallback, "steer_delta": steer_delta,
            "proposed_pairs": proposed_pairs,
            "free_cache": dict(free._cache) if free is not None else {},
            "deferred_iced": deferred_iced, "pending": sorted(pending) if schedule == "async" else [],
            "queue": connect.queue_state() if hasattr(connect, "queue_state") else None})
        if schedule == "async" and checkpoint is not None:
            checkpoint(result)   # no rounds to wait for: the network so far (steps, their TS and IRC files) for viewers

    if saved is not None:
        connected |= saved["connected"]
        retry[:] = saved["retry"]
        searched |= saved["searched"]
        expanded |= saved["expanded"]
        frontier, t_idx = list(saved["frontier"]), saved["t_idx"]
        refined |= saved["refined"]
        sample_more |= saved["sample_more"]
        proposed_pairs |= saved.get("proposed_pairs", set())
        peak_fallback.update(saved["peak_fallback"])
        steer_delta.update(saved["steer_delta"])
        if free is not None:
            free._cache.update(saved["free_cache"])
        deferred_iced[:] = saved.get("deferred_iced", [])
        if hasattr(connect, "restore"):
            connect.restore(saved.get("queue"))
        pos.update(saved["pos"])
        _emit(on_event, "resumed", round=pos["round"], stage=pos["stage"], species=len(result.species),
              steps=len(result.steps))
    if target is not None and saved is None:
        # Round 0: the two-ended search seed -> target; its intermediates and
        # verified steps start the network.
        rnd = 0
        tnode = target.copy()
        if minimize_seed:
            opt = _optimize(engine, [StructureNode(structure=tnode.structure)], maxiter, None)[0]
            if isinstance(opt, Exception):
                raise RuntimeError(f"Could not minimize the target: {type(opt).__name__}: {opt}")
            tnode = opt
        elif tnode._cached_energy is None:
            engine.compute_energies([tnode])
        t_idx = next((k for k, sp in enumerate(result.species) if _connectivity_matches(tnode, sp.node)), None)
        if t_idx is None:
            rel = (float(tnode.energy) - e0) * HARTREE_TO_KCAL_PER_MOL
            smi = lewis_smiles(symbols, view.edges(tnode), charge, mult, allow_radicals=True,
                               allow_zwitterions=True) or ""
            result.species.append(Species(tnode.copy(), smi, 0, rel))
            t_idx = len(result.species) - 1
            _emit(on_event, "species_found", index=t_idx, smiles=smi, rel_energy_kcal=rel, round=0)
        result.rounds.append({"round": 0, "sources": [0], "target": t_idx, "proposed": 0, "new_species": [t_idx],
                              "expanded_next": [], "generator_stats": []})
        if t_idx != 0 and connect is not None:
            _connect([(0, t_idx)])
            if steer == "access":
                from mepd.network_aware import accessibility

                acc = accessibility(result.steps, [float(sp.node.energy) for sp in result.species], e0)
                acc, extra = _reroute(acc)   # the route found climbs too high: look for others at once
                # the seed first; the rest of the route only once the kinetics say material gets there
                frontier = [0] + [k for k in extra if k != 0][:max(0, int(reroute_attempts))]
                result.rounds[-1]["expanded_next"] = frontier
    if schedule == "async":
        if view.region is not None:
            raise ValueError("schedule='async' doesn't run QM/MM systems yet; use rounds.")
        gen_obj = get_generator(generator)
        job_opts = dict(generator_options or {})
        if gen_obj.kind == "structures":
            width = int(job_opts.get("parallel") or job_opts.get("n_seeds") or max(1, workers // 2))
        else:
            width = 4
        width = max(1, min(int(workers), width))
        if int(job_opts.get("n_seeds") or 0) > 0:
            # It never runs more processes than trajectories per batch: reserving more workers for it
            # (`parallel` follows --workers) left the rest idle and counted busy (4 MD runs, 10/10 busy).
            width = min(width, int(job_opts["n_seeds"]))
        if gen_obj.kind == "structures" and ("parallel" in job_opts or "parallel" in gen_obj.options):
            job_opts["parallel"] = width   # its share of the workers, not all of them

        def _propose_job(src, structure, known_smiles, gen_opts):
            """In a worker: the generator's proposals for species `src`, minimized (loosely, as in rounds),
            with their Hessian checks when asked for. Proposals of species known when it was queued come
            back unminimized ("known"). `gen_opts`, a structures generator's options for `src`, are made when
            it is queued: the worker's copy of the network is the one from when its pool was forked."""
            node = StructureNode(structure=structure)
            if gen_obj.kind == "structures":
                gen_obj.check()
                props = external_proposals(gen_obj.propose(structure, max_products=max_products,
                                                           options=gen_opts), structure, src)
            else:
                coords = view.coords(node)
                props, _ = propose_products(
                    generator, symbols, coords, view.edges(node), options=generator_options, charge=charge,
                    multiplicity=mult, n_break=n_break, n_form=n_form, form_distance=form_distance,
                    max_products=max_products, allow_radicals=allow_radicals, allow_zwitterions=allow_zwitterions,
                    source=src)
                props = view.keep(props)
                _build_guesses(props, node, coords, SimpleNamespace(enabled=False), width, view)
            todo = [k for k, p in enumerate(props) if p.smiles not in known_smiles]
            eng = connect.job_inputs.engine
            opts = _optimize(eng, [StructureNode(structure=props[k].structure) for k in todo], maxiter, None,
                             fmax=loose_fmax)
            out = ["known"] * len(props)
            for k, opt in zip(todo, opts):
                record = None
                if not isinstance(opt, Exception) and validate_minima is not None:
                    from mepd.elementarystep import validate_minimum_with_rescue

                    opt, record = validate_minimum_with_rescue(
                        opt, eng, frequency_cutoff=float(validate_minima.get("frequency_cutoff", -20.0)),
                        rescue_displacement=float(validate_minima.get("rescue_displacement", 0.3)),
                        label=f"proposal {props[k].label}")
                out[k] = (opt, record)
            return props, out

        # The seed's proposals are the queue's first job; a later species' wait until the steps found so far have
        # their IRC checks, rotamers and channel sampling (F0007, 32 workers: the reactor for the most populated
        # adduct took all 32 workers for the last 6-13 minutes of a 30-minute budget in 9 of 10 runs, while the
        # TS searches of the endo and exo conformer pairs waited)
        connect.register("propose", _propose_job, slots=width, rank=3)

        class _SpeciesNodes:
            """The species' nodes as the queue indexes them, growing as species are found."""
            def __getitem__(self, i):
                return result.species[i].node

            def __len__(self):
                return len(result.species)

        pending: set = set(map(tuple, (saved or {}).get("pending", ())))   # pairs queued, not finished
        expanding: set = set()                                             # proposals running

        def _expandable(final: bool = False) -> list:
            """Species to propose from now: populated in the live kinetics, within reach of the cutoff, and not
            expanded yet -- the most populated first. No distance from the seed caps it: the run ends when the
            kinetics ask for nothing more, or a budget runs out (`_out_of_budget`)."""
            from mepd.network_aware import accessibility

            hot = _live_hot(final)
            acc = accessibility(result.steps, [float(sp.node.energy) for sp in result.species], e0)
            load = live_kin.get("load", {})
            picks = [k for k in hot if k not in expanded and k not in expanding and k < len(result.species)
                     and (k == 0 or (k in acc and acc[k].span <= access_cutoff_kcal))]
            return sorted(picks, key=lambda k: -load.get(k, 1.0))

        def _proposed(src, out) -> None:
            """A species' proposals are back: species and edges as in a round; pairs to new products join
            the running queue at once."""
            expanding.discard(src)
            expanded.add(src)
            rec = result.rounds[-1]
            if out is None:     # not counted as expanded: said in the round (the why: the queue's job event)
                rec.setdefault("proposals_failed", []).append(src)
                _emit(on_event, "warning", message=f"proposals for species {src} failed")
                return
            rec["sources"].append(src)
            props, res = out
            rec["proposed"] += len(props)
            known = {sp.smiles: k for k, sp in enumerate(result.species) if sp.smiles}
            edges = []
            for p, r in zip(props, res):
                if r == "known":
                    if p.smiles in known:
                        edges.append(ProposedEdge(p.source, p, "known_species", known[p.smiles], True))
                    continue
                opt, record = r
                if isinstance(opt, Exception):
                    edge = ProposedEdge(p.source, p, "failed", error=f"{type(opt).__name__}: {opt}")
                elif record is not None and not record["is_minimum"]:
                    edge = ProposedEdge(p.source, p, "not_minimum", error=str(record.get("validation")))
                    result.rejected.append(opt.copy())
                else:
                    edge = _classify(p, StructureNode(structure=p.structure), opt, record)
                edges.append(edge)
            for edge in edges:
                result.edges.append(edge)
                live.finish(edge, result.species)
                if edge.outcome == "new_species":
                    rec["new_species"].append(edge.target)
            pairs = _new_pairs(edges, max_rel_kcal=access_cutoff_kcal)
            proposed_pairs.update(map(frozenset, pairs))
            connected.update(pairs)
            pending.update(pairs)
            _phase("proposed", round=rnd, source=src, proposals=len(props), pairs=len(pairs))
            if pairs:
                connect.add_pairs(pairs)
            _save()
            _follow_up()

        def _expand_now(final: bool = False) -> bool:
            picks = _expandable(final)
            for k in picks:
                expanding.add(k)
                known_smiles = frozenset(sp.smiles for sp in result.species if sp.smiles)
                connect.submit("propose", (k, result.species[k].node.structure, known_smiles,
                                           _generator_cutoff(generator, dict(job_opts), access_cutoff_kcal,
                                                             result.species[k])),
                               lambda out, k=k: _proposed(k, out), label=f"species {k}")
            if picks:
                _emit(on_event, "expanding", round=rnd, species=picks)
            return bool(picks)

        # Everything the rounds do between their connect calls -- IRC checks, channel sampling, steering
        # single points -- runs here as jobs of the same queue, each queued the moment the live kinetics ask
        # for it (`_follow_up`, after every result): no stage waits for another to finish. Channel sampling
        # is its conformer pairs, each connected by the queue like any pair (a fast TS, a path search only
        # if that fails).
        def _minimized(nodes) -> list:
            return _optimize(connect.job_inputs.engine, [StructureNode(structure=n.structure) for n in nodes],
                             maxiter, None)

        def _irc_job(ts_file):
            """In a worker: a step's IRC, its two ends minimized."""
            _, start, end = verify([(0, ts_file)])[0]
            return None if start is None else (start, end, _minimized([start, end]))

        def _channels_job(a, b, label):
            """In a worker: a step's conformers and atom-mapped conformer pairs (`refine.pairs`)."""
            return refine.pairs((a, b, label))

        def _rotamer_job(guess, fp, core, ends):
            """In a worker: a TS optimized from one rotamer guess of a step's TS (`ts_rotamer_guesses`), written
            to `fp` with its parent's IRC ends beside it (`<stem>_ends.xyz`, what tells its spectators); None
            unless every distance of the reacting `core` stayed within CORE_TOLERANCE_ANG of the guess's (the
            parent's): it went to another TS. With free energies, also its (cache key, thermal correction), as the
            queue's own TS jobs compute it."""
            import mepd
            from mepd.cli_conformers import CORE_TOLERANCE_ANG

            ts = mepd.find_ts(StructureNode(structure=guess), connect.job_inputs.engine,
                              maxiter=int(getattr(connect, "fast_ts_steps", 300)), continue_unconverged=0)
            x0, x1 = (np.asarray(s.geometry, dtype=float) / ANGSTROM_TO_BOHR for s in (guess, ts.structure))
            if any(abs(np.linalg.norm(x1[i] - x1[j]) - np.linalg.norm(x0[i] - x0[j])) > CORE_TOLERANCE_ANG
                   for i, j in core):
                return None
            fp = Path(fp)
            mepd.write_xyz(fp, ts, comments=[f"E={ts.energy:.10f}"])
            mepd.write_xyz(fp.with_name(f"{fp.stem}_ends.xyz"), [StructureNode(structure=e) for e in ends])
            corr = None
            if free is not None:
                try:
                    st = _open_structure(fp).model_copy(update={"charge": guess.charge,
                                                                "multiplicity": guess.multiplicity})
                    corr = (free._key(st, True, ends), free._correction(st, True, ends))
                except Exception:
                    corr = None    # the kinetics compute it, or keep the electronic energy (said)
            return ts, str(fp), corr

        def _steer_job(item):
            try:
                return _single_point(item, engine, steer_engine)
            except Exception as exc:
                return f"{type(exc).__name__}: {exc}"

        if verify is not None:
            connect.register("irc", _irc_job, rank=1)
        if refine is not None:
            if not hasattr(refine, "pairs"):
                raise ValueError("schedule='async' needs a `refine` with `pairs` (its conformer pairs).")
            connect.register("channels", _channels_job, slots=int(getattr(refine, "width", 1)))
            if verify is not None:
                connect.register("rotamer", _rotamer_job, rank=1)
        if steer_engine is not None and not getattr(steer_engine, "holds_torch_model", False):
            from mepd.discovery.energy_refine import _single_point

            connect.register("steer", _steer_job, slots=int(getattr(steer_engine, "threads", 1) or 1), rank=0)
            live_steer[0] = lambda key, item: _queue_steer(key, item)
        queued_irc: set = set()       # TS files whose IRC is queued or done
        queued_channels: set = set()  # species pairs whose sampling is queued or done
        queued_steer: set = set()     # steering single points queued
        queued_rotamers: set = set()  # TS files whose rotamers are queued or done

        lowest_ts: dict = {"steps": -1, "pairs": {}}   # species pair -> its lowest TS energy, at that many steps

        def _height(key) -> float:
            """kcal/mol above the seed of the lowest TS found between a species pair so far -- or, with none yet
            (its single TS failed or belongs to another step), of the higher of the two species, which no TS
            between them can lie below -- as the queue asks it each time it orders its waiting jobs: the channels
            of the lowest steps are sampled first. F0007's endo and exo steps, the two the selectivity rests on,
            waited 4-12 minutes behind the conformer pairs of 35-42 steps that carry no flux, all queued with the
            seed's proposals before any of their TSs was found; an exo adduct whose single TS's IRC went
            elsewhere waited 12 minutes more when a pair without a TS came last."""
            if lowest_ts["steps"] != len(result.steps):
                pairs: dict = {}
                for st in result.steps:
                    k = frozenset((st.a, st.b))
                    pairs[k] = min(pairs.get(k, st.ts_energy), st.ts_energy)
                lowest_ts.update(steps=len(result.steps), pairs=pairs)
            e = lowest_ts["pairs"].get(key)
            if e is None:
                e = max(float(result.species[k].node.energy) for k in key)
            return (e - e0) * HARTREE_TO_KCAL_PER_MOL

        def _queue_rotamers(st) -> int:
            """The rotamers of step `st`'s TS (`ts_rotamer_guesses`, about its rotatable bonds), each a TS search
            of the queue's. F0007: the single TS of each adduct is the conformer its proposal happened to be in
            (endo s-trans, exo with the ester E: 1.5 and 7 kcal/mol over the lowest); turning the acrylate and
            ester bonds of those two TSs gives the endo s-cis, exo s-cis and exo s-trans ones."""
            ts = _ts_structure(st)
            if ts is None:
                return 0
            ends = _ts_ends(st)
            guesses, core = ts_rotamer_guesses(ts, ends)
            stem = Path(st.files["ts"]).stem
            for k, (guess, what) in enumerate(guesses):
                connect.submit("rotamer", (guess, str(Path(st.files["ts"]).with_name(f"{stem}_rot{k}.xyz")), core,
                                           ends),
                               lambda out, a=st.a, b=st.b: _rotamer_back(a, b, stem, out),
                               label=f"{st.a}→{st.b} · {stem} turned {what}",
                               priority=(st.ts_energy - e0) * HARTREE_TO_KCAL_PER_MOL)
            return len(guesses)

        def _rotamer_back(a, b, parent, out) -> None:
            """A rotamer of TS `parent` is back: one more TS between its ends (unless the same TS is there
            already), IRC-checked like any; an IRC that fails drops it, and asks for no path search."""
            if out is not None:
                ts, fp, corr = out
                if corr is not None and free is not None:
                    free._cache[corr[0]] = corr[1]
                _add_found({"pair": (a, b), "ts": ts, "label": Path(fp).stem,
                            "files": {"ts": fp, "verified": False, "rotamer_of": parent}})
            _save()
            _follow_up()

        def _queue_steer(key, item) -> None:
            if key in queued_steer:
                return
            queued_steer.add(key)

            def back(r):
                if r is None or isinstance(r, str):
                    _emit(on_event, "warning", message=f"steering single point failed ({r}): explorer energy used")
                    steer_delta[key] = None
                else:
                    steer_saved.put("single_points", key, r)
                    steer_delta[key] = r["delta_kcal"] / HARTREE_TO_KCAL_PER_MOL
                live_kin["version"] += 1
                _follow_up()

            connect.submit("steer", (item,), back, label=f"{item.kind} {key[:8]}")

        def _irc_back(st, out) -> None:
            if any(x is st for x in result.steps):     # (channel sampling may have replaced it meanwhile)
                start = end = None
                if out is not None:
                    start, end, opts = out
                    premin[id(start)], premin[id(end)] = (start, opts[0]), (end, opts[1])
                _irc_checked(st, start, end)
                result.rounds[-1]["irc_verified"] = result.rounds[-1].get("irc_verified", 0) + 1
            _save()
            _follow_up()

        def _channels_back(key, a, b, out, priority=0.0) -> None:
            """A step's conformer pairs are back: the lowest conformer of each end becomes its species'
            reference if lower, and every pair joins the queue (`ch_<a>_<b>_<k>`) at the step's `priority`."""
            refined.add(key)
            rec = result.rounds[-1]
            rec.setdefault("channels_sampled", []).append([a, b])
            rec["channels"] = len(rec["channels_sampled"])
            if out:
                for pool in (out["start"], out["end"]):
                    with_e = [n for n in pool if getattr(n, "_cached_energy", None) is not None]
                    if with_e:
                        low = min(with_e, key=lambda n: float(n.energy))
                        premin[id(low)] = (low, low)       # minimized by the job
                        _species_for(low)
                for k, (x, y) in enumerate(out["pairs"]):
                    connect.add_channel((a, b), x, y, f"ch_{a}_{b}_{k}", priority=priority)
                rec["channel_pairs"] = rec.get("channel_pairs", 0) + len(out["pairs"])
            _save()
            _follow_up()

        def _follow_up(final: bool = False) -> bool:
            """Queue whatever the live kinetics now ask for; True if anything was queued. Nothing once a budget
            ran out: the queue then cancels the jobs running and ends."""
            if _out_of_budget():
                return False
            hot = _live_hot(final)          # (queues the steering single points they lack)
            n = 0
            if verify is not None:          # IRC checks of the steps that touch a populated species
                for st in list(result.steps):
                    fp = st.files.get("ts")
                    if st.files.get("verified") is False and fp and fp not in queued_irc and (st.a in hot or st.b in hot):
                        queued_irc.add(fp)
                        connect.submit("irc", (fp,), lambda out, st=st: _irc_back(st, out),
                                       label=f"{st.a}→{st.b} · {Path(fp).stem}")
                        n += 1
            todo = [p for p in dict.fromkeys(retry) if p not in searched]   # pairs an IRC check sent back
            retry.clear()
            ends = [p for p in dict.fromkeys(irc_products) if p not in connected and p[::-1] not in connected]
            irc_products.clear()
            if ends:                                # each new IRC-end species from the species it was asked from
                connected.update(ends)
                pending.update(ends)
                connect.add_pairs(ends)
                n += len(ends)
            if todo:
                searched.update(todo)
                connected.update(todo)
                pending.update(todo)
                connect.add_pairs(todo, search=True)
                n += len(todo)
            if refine is not None and live_kin.get("k") is not None:       # channel sampling, lowest steps first
                for key, a, b in _refine_todo(live_kin["k"]):
                    if key not in queued_channels:
                        queued_channels.add(key)
                        h = lambda key=key: _height(key)
                        connect.submit("channels", (result.species[a].node, result.species[b].node, f"{a}_{b}"),
                                       lambda out, key=key, a=a, b=b, h=h: _channels_back(key, a, b, out, h),
                                       label=f"{a}→{b}", priority=h)
                        n += 1
            if refine is not None and verify is not None:   # the rotamers of every IRC-checked TS the kinetics use
                for st in list(result.steps):
                    fp = st.files.get("ts")
                    if (fp and st.files.get("verified") is True and st.files.get("irc") and fp not in queued_rotamers
                            and "rotamer_of" not in st.files and st.files.get("ts_kind") != "path_peak"
                            and (st.a in hot or st.b in hot)
                            and (st.ts_energy - e0) * HARTREE_TO_KCAL_PER_MOL <= access_cutoff_kcal):
                        queued_rotamers.add(fp)
                        n += _queue_rotamers(st)
            return _expand_now(final) or n > 0

        def _found_async(found) -> None:
            _add_found(found)
            _save()
            _follow_up()

        def _pair_done_async(pair) -> None:
            pending.discard(tuple(pair))
            _save()

        import time as _time

        from mepd import search_control

        deadline = _time.monotonic() + float(time_budget_s) if time_budget_s else None
        stopped = [None]

        def _out_of_budget() -> bool:
            """Out of time (`time_budget_s`), asked to stop (the web's Stop & analyze), or at `max_species`."""
            if stopped[0] is None:
                if deadline is not None and _time.monotonic() > deadline:
                    stopped[0] = "time budget"
                elif search_control.stop_requested():
                    stopped[0] = "stop requested"
                elif len(result.species) >= max_species:
                    stopped[0] = "max species"
            return stopped[0] is not None

        connect.stop_when = _out_of_budget
        if connect.end_keywords is None:
            connect.end_keywords = {"coordsys": "cart", "maxiter": int(maxiter)}
        cycle = int(pos["round"]) if saved is not None else 1
        rnd = cycle
        result.rounds.append({"round": cycle, "schedule": "async", "sources": [], "proposed": 0,
                              "new_species": [], "expanded_next": [], "generator_stats": []})
        pos.update(round=cycle, stage="async", ctx={}, current=None, inflight=None)
        _phase("cycle_start", round=cycle)
        connect(sorted(pending), _SpeciesNodes(), on_found=_found_async, relevant=_piece_relevant,
                on_pair_done=_pair_done_async, free_cache=free._cache if free is not None else None,
                start=_follow_up, on_drain=lambda: _follow_up(final=True))
        result.rounds[-1]["iced_pieces"] = len(connect.iced)
        # why it ended: "converged" (the kinetics asked for nothing more), or the budget that ran out
        result.rounds[-1]["stopped"] = stopped[0] or "converged"
        live_steer[0] = None    # the queue is closed: missing steering single points are computed here, not queued
        _emit(on_event, "phase", round=cycle, phase=f"ended: {result.rounds[-1]['stopped']}")
        _kinetics_record([])
        _save(force=True)
        if checkpoint is not None:
            checkpoint(result)
        first, in_round = rounds + 1, False     # no rounds below
    else:
        first = int(pos["round"]) if saved is not None else 1
        in_round = saved is not None and pos["stage"] != "propose"   # cut off after its proposals
        if saved is not None:
            rnd = first
            _resume_connect()
    for rnd in range(first, rounds + 1):
        if in_round and rnd == first:
            n_edges_before, new_frontier = pos["ctx"]["n_edges_before"], pos["ctx"]["new_frontier"]
        else:
            if not frontier:
                break
            expanded.update(frontier)
            _phase("round_start", round=rnd, frontier=len(frontier))
            n_edges_before = len(result.edges)
            proposals, stats_all = [], []
            prefetched = _propose_structures_at_once(frontier, result, generator, generator_options, products_file,
                                                     view, max_products, workers, access_cutoff_kcal)
            for src in frontier:
                node = result.species[src].node
                coords = view.coords(node)
                gen = get_generator(generator) if products_file is None and is_named(generator) else None
                if view.region is not None and (gen is None or gen.kind == "structures"):
                    # QM/MM: a whole-molecule generator (or an imported one) works on
                    # the QM model -- QM atoms capped with link H, at the QM charge and
                    # spin -- and its products are put back into the environment.
                    model = view.region.model_structure(node.structure)
                    opts = dict(generator_options or {})
                    embedded = bool(opts.pop("embedded", False)) and gen is not None and gen.name == "nanoreactor"
                    if embedded:
                        # The nanoreactor's MD with QM/MM forces, the environment present
                        # all along: its products are whole systems.
                        from mepd.discovery.qmmm_reactor import qmmm_reactor_products

                        items, reactor_counts = qmmm_reactor_products(
                            node.structure, engine, view.region, max_products=max_products, on_event=on_event,
                            **{k: v for k, v in opts.items() if k != "md_method"})
                        for key, what in (("mm_changed", "changed a bond in the MM region"),
                                          ("boundary", "stretched a QM/MM cut bond")):
                            if reactor_counts[key]:
                                _emit(on_event, "warning", message=f"{reactor_counts[key]} QM/MM nanoreactor event(s) "
                                                                   f"{what}; not used as products")
                    elif gen is not None:
                        gen.check()
                        if gen.name == "nanoreactor" and "anchors" not in opts and view.region.links:
                            opts["anchors"] = ",".join(str(a) for a in nanoreactor_anchors(view.region))
                        items = gen.propose(model, max_products=max_products, options=opts)
                    elif products_file is not None:
                        from mepd.qcdata_structure_helpers import read_multiple_structure_from_file
                        items = read_multiple_structure_from_file(products_file, charge, mult) if src == 0 else []
                    else:
                        items = _import_generator(generator)(model, **dict(generator_options or {}))
                    # Products of the whole system (e.g. a products file made for it) go
                    # in as they are; products of the QM model are put back.
                    items = [s if not isinstance(s, (str, Path)) else _open_structure(s) for s in items]
                    whole = [s for s in items if len(s.symbols) == len(node.symbols)]
                    props = external_proposals(whole, node.structure, src) if whole else []
                    region_props, counts = _region_proposals(
                        view, node, src, [s for s in items if len(s.symbols) != len(node.symbols)])
                    props += region_props
                    stats = {(gen.name if gen is not None else "external"): len(props),
                             **({"broke_qmmm_boundary": counts["boundary"]} if counts["boundary"] else {})}
                    if counts["boundary"]:
                        _emit(on_event, "warning", message=f"{counts['boundary']} product(s) of the QM region broke a "
                                                           "QM/MM boundary bond (a link H stopped being a cap); dropped")
                elif gen is not None and gen.kind == "structures":   # e.g. crest-msreact: product geometries
                    if src in prefetched:
                        items = prefetched[src]
                    else:
                        gen.check()
                        items = gen.propose(node.structure, max_products=max_products,
                                            options=_generator_cutoff(generator, dict(generator_options or {}),
                                                                      access_cutoff_kcal,
                                                                      result.species[src]))
                    props = external_proposals(items, node.structure, src)
                    stats = {gen.name: len(props)}
                elif gen is not None:
                    props, stats = propose_products(
                        generator, symbols, coords, view.edges(node), options=generator_options, charge=charge,
                        multiplicity=mult, n_break=n_break, n_form=n_form, form_distance=form_distance,
                        max_products=max_products, allow_radicals=allow_radicals, allow_zwitterions=allow_zwitterions,
                        source=src,
                    )
                    props = view.keep(props)
                    _build_guesses(props, node, coords, live, workers, view)
                else:   # imported products, or a generator given by import path
                    if products_file is not None:  # an external tool's products of the seed
                        from mepd.qcdata_structure_helpers import read_multiple_structure_from_file
                        items = read_multiple_structure_from_file(products_file, charge, mult) if src == 0 else []
                    else:
                        items = _import_generator(generator)(node.structure, **dict(generator_options or {}))
                    props = external_proposals(items, node.structure, src)
                    stats = {"external": len(props)}
                stats_all.append({"source": src, **stats})
                proposals.extend(props)
            known = {s.smiles: k for k, s in enumerate(result.species) if s.smiles}
            for p in [p for p in proposals if p.smiles in known]:
                edge = ProposedEdge(p.source, p, "known_species", known[p.smiles], True)
                result.edges.append(edge)
                live.finish(edge, result.species)
            proposals = [p for p in proposals if p.smiles not in known]
            _phase("proposed", round=rnd, proposals=len(proposals))
            live.prepare(proposals, result.species)
            _phase("animations", round=rnd)
            for p in proposals:
                live.propose(p, result.species, frames=p.frames)
            _emit(on_event, "proposed", round=rnd, total=len(proposals))
            guesses = [StructureNode(structure=p.structure) for p in proposals]
            optimized = _optimize(engine, guesses, maxiter, on_event,
                                  on_start=lambda k: live.optimizing(proposals[k], result.species), fmax=loose_fmax)
            _phase("optimized", round=rnd, proposals=len(guesses))
            new_frontier = []
            for p, guess, opt in zip(proposals, guesses, optimized):
                edge, record = None, None
                if isinstance(opt, Exception):
                    edge = ProposedEdge(p.source, p, "failed", error=f"{type(opt).__name__}: {opt}")
                elif validate_minima is not None:
                    from mepd.elementarystep import validate_minimum_with_rescue

                    opt, record = validate_minimum_with_rescue(
                        opt, engine, frequency_cutoff=float(validate_minima.get("frequency_cutoff", -20.0)),
                        rescue_displacement=float(validate_minima.get("rescue_displacement", 0.3)),
                        label=f"proposal {p.label}")
                    if not record["is_minimum"]:
                        edge = ProposedEdge(p.source, p, "not_minimum", error=str(record.get("validation")))
                        result.rejected.append(opt.copy())
                if edge is None:
                    edge = _classify(p, guess, opt, record)
                result.edges.append(edge)
                live.finish(edge, result.species, product=None if isinstance(opt, Exception) else opt)
                if edge.outcome == "new_species" and result.species[edge.target].rel_energy_kcal <= energy_window_kcal:
                    new_frontier.append(edge.target)
            found = [e.target for e in result.edges if e.outcome == "new_species" and result.species[e.target].round == rnd]
            result.rounds.append({"round": rnd, "sources": frontier, "proposed": len(proposals),
                                  "new_species": found, "expanded_next": new_frontier, "generator_stats": stats_all})
            pos.update(round=rnd, stage="steer", current=None,
                       ctx={"n_edges_before": n_edges_before, "new_frontier": new_frontier, "done": []})
            _save(force=True)
        if steer == "access":
            new_frontier = _steer_by_access(result.edges[n_edges_before:])
            result.rounds[-1]["expanded_next"] = new_frontier
            result.rounds[-1]["new_species"] = [k for k, s in enumerate(result.species) if s.round == rnd and k]
        elif steer == "flux":
            new_frontier = _steer_by_flux(result.edges[n_edges_before:])
            result.rounds[-1]["expanded_next"] = new_frontier
            result.rounds[-1]["new_species"] = [k for k, s in enumerate(result.species) if s.round == rnd and k]
        frontier = new_frontier
        pos.update(round=rnd + 1, stage="propose", ctx={}, current=None)
        _save(force=True)
        if checkpoint is not None:
            checkpoint(result)   # a run cut off later keeps the rounds it finished
    # an async run that ran out of time, was stopped or reached max_species ends with what it has: these would run
    # on unbudgeted (F0007, 200 species at 1459 s of 1800: an hour more of whole channel searches)
    out_of_time = schedule == "async" and result.rounds and \
        result.rounds[-1].get("stopped") in ("time budget", "stop requested", "max species")
    if steer == "access" and t_idx is not None and connect is not None and result.rounds and not out_of_time:
        # Searches towards the target that the last round's network opened up
        # would have joined the next round's queue: there is none, so run them now.
        from mepd.network_aware import accessibility

        rnd = result.rounds[-1]["round"]
        acc = accessibility(result.steps, [float(s.node.energy) for s in result.species], e0)
        attempts = _reroute_attempts(acc)
        if attempts:
            _connect(attempts)
            acc = accessibility(result.steps, [float(s.node.energy) for s in result.species], e0)
            _reroute(acc, attempts)
            _kinetics_record(attempts)
            result.rounds[-1]["final_reroute"] = [list(p) for p in attempts]
    if steer == "access" and verify is not None and result.rounds and not out_of_time:
        # the final prediction rests on IRC-checked steps only where it matters
        rnd = result.rounds[-1]["round"]
        if _verify_and_search(result.kinetics if result.kinetics is not None else _kinetics_record([])):
            _kinetics_record([])
            result.rounds[-1]["final_irc_check"] = True
    if steer == "access" and refine is not None and result.rounds and result.steps and not out_of_time:
        # steps the last round made relevant get their channel sampling too
        rnd = result.rounds[-1]["round"]
        if _refine_steps(result.kinetics if result.kinetics is not None else _kinetics_record([])):
            _kinetics_record([])
    _save(force=True)
    live.close()
    if steer_pool[0] is not None:
        steer_pool[0].shutdown()
    return result
