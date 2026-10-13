"""Network variants: edit one species (e.g. swap hydrogens of an epoxide for
methyls) and carry the edit through every reaction of a network, so that all
its stationary points can be recomputed with the edit in place.

Nothing in a network records which atom of a species is which atom of a
complex, a TS or a reaction's other end, so the edit is carried by graph
matching, one hop at a time:

1. `derive_edit` turns a Design edit (atoms tagged with their index in the
   original molecule) into an `Edit` on that molecule: element/charge swaps
   on kept atoms, hydrogens removed, and new atoms (groups, hydrogens) with
   coordinates in the molecule's own frame. A hydrogen replaced by a group
   keeps its index (it becomes the group's first atom), as in
   `mepd.substituents.attach`.
2. In each reaction, the edited species is matched onto every fragment of
   the reaction's end geometry it occurs in, under every graph isomorphism
   (its automorphisms included): a symmetric molecule (ethylene oxide) gives
   several placements, which can be different reactions once substituted
   (ring opening at the substituted or the unsubstituted carbon).
   Placements that would give another stereoisomer of the edited species
   are dropped; the rest are reduced to one per distinct substituted
   reaction (`reaction_key`; mirror images count as one).
3. `apply_edit` builds the substituted reactant end, TS and product end in
   the reaction's shared atom order: new atoms are placed by fitting the
   anchor atom's neighbourhood (the same atoms in every frame), a replaced
   hydrogen's group along that frame's own X-H bond.
4. The edited atoms come out of the reaction in product fragments, which are
   variants of the reaction's product species; those enter the reactions
   they take part in (`propagate`, a breadth-first closure over the
   network, in both directions).

A reaction in which an edited hydrogen is transferred (or a removed one
takes part) has no analogue in the variant and is `blocked`. Reactions with
no edited species are shared by the parent and the variant network.

Everything here is geometry and graphs (Angstrom), no engine calls: the
fast estimate and the re-optimization run per reaction in `mepd variant`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

ADJ_SCALE = 1.25        # bonded: closer than 1.25 x the sum of covalent radii (as mepd.substituents)
CLASH = 1.0             # Å: a new atom closer than this to a kept one is re-spun about its bond
MANY_PLACEMENTS = 4     # the preview warns when one parent reaction gives more variants than this


class VariantError(ValueError):
    """An edit that cannot be carried through the network (with why)."""


# ------------------------------------------------------------------ graphs

def _rcov(symbol: str) -> float:
    from rdkit import Chem

    return Chem.GetPeriodicTable().GetRcovalent(symbol)


def adjacency(symbols, coords, scale: float = ADJ_SCALE) -> np.ndarray:
    """Bonded atom pairs from distances (coordinates in Angstrom)."""
    c = np.asarray(coords, dtype=float)
    r = np.array([_rcov(s) for s in symbols])
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    adj = d < scale * (r[:, None] + r[None])
    np.fill_diagonal(adj, False)
    return adj


def components(adj: np.ndarray) -> list[list[int]]:
    """Connected fragments (sorted atom lists), in order of their first atom."""
    n = len(adj)
    seen, out = set(), []
    for i in range(n):
        if i in seen:
            continue
        comp, todo = {i}, [i]
        while todo:
            a = todo.pop()
            for b in np.nonzero(adj[a])[0]:
                b = int(b)
                if b not in comp:
                    comp.add(b)
                    todo.append(b)
        seen |= comp
        out.append(sorted(comp))
    return out


def _raw_mol(symbols, edges: Iterable[tuple], kinds: Optional[dict] = None):
    """An unsanitized RDKit graph: atoms by element, bonds all single unless
    `kinds` labels them (used to mark formed/broken bonds)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = Chem.RWMol()
    for s in symbols:
        a = Chem.Atom(s)
        a.SetNoImplicit(True)
        m.AddAtom(a)
    for i, j in edges:
        m.AddBond(int(i), int(j), (kinds or {}).get((min(i, j), max(i, j)), Chem.BondType.SINGLE))
    m = m.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return m


def _edges(adj: np.ndarray, atoms: Optional[list] = None) -> list[tuple]:
    idx = range(len(adj)) if atoms is None else atoms
    s = set(idx)
    return [(int(i), int(j)) for i in idx for j in np.nonzero(adj[i])[0] if j > i and int(j) in s]


def _ranks(mol) -> list[int]:
    from rdkit import Chem

    return list(Chem.CanonicalRankAtoms(mol, breakTies=False))


def _signs(symbols, coords, adj, ranks) -> list[tuple]:
    """Tetrahedral configurations: (rank, +1/-1) for every 4-connected atom
    whose neighbours are all different in the graph, from the 3D geometry.
    Double-bond E/Z needs bond orders and is not read (it does not change
    under substitution of hydrogens on sp3 atoms, the common case)."""
    c = np.asarray(coords, dtype=float)
    out = []
    for a in range(len(symbols)):
        nb = [int(b) for b in np.nonzero(adj[a])[0]]
        if len(nb) != 4 or len({ranks[b] for b in nb}) != 4:
            continue
        nb.sort(key=lambda b: ranks[b], reverse=True)
        v = [c[b] - c[a] for b in nb[:3]]
        det = float(np.dot(v[0], np.cross(v[1], v[2])))
        if abs(det) > 1e-3:
            out.append((ranks[a], 1 if det > 0 else -1))
    return out


def molecule_key(symbols, coords, adj, charge: int = 0, *, mirror: bool = True) -> tuple:
    """Identity of a molecule (or complex) from its connectivity, its
    tetrahedral stereo and its charge; with `mirror`, a molecule and its
    mirror image share the key (an achiral network treats them alike)."""
    from rdkit import Chem

    mol = _raw_mol(symbols, _edges(adj))
    ranks = _ranks(mol)
    smi = Chem.MolToSmiles(mol)
    s = _signs(symbols, coords, adj, ranks)
    keys = [(smi, tuple(sorted(s)), int(charge))]
    if mirror:
        keys.append((smi, tuple(sorted((r, -v) for r, v in s)), int(charge)))
    return min(keys)


def smiles_from_graph(symbols, coords, adj, charge: int = 0) -> Optional[str]:
    """SMILES of a molecule from its known bonds and total charge (bond
    orders assigned to fit), with tetrahedral stereo from its 3D geometry:
    a built fragment's distances may be off (an S where an O was), its
    connectivity is not."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    try:
        mol = _raw_mol(symbols, _edges(adj))
        rw = Chem.RWMol(mol)
        for a in rw.GetAtoms():
            a.SetNoImplicit(False)
        mol = rw.GetMol()
        conf = Chem.Conformer(len(symbols))
        for i, (x, y, z) in enumerate(np.asarray(coords, dtype=float)):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineBondOrders(mol, charge=int(charge))
        mol = _fewer_charges(mol)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def _fewer_charges(mol):
    """Bond orders assigned from connectivity can come out charge-separated
    ([O-]C=[S+] for O=C-S): move each such pair back, X+=Y-Z- -> X-Y=Z,
    while the result is still a valid molecule."""
    from rdkit import Chem

    single, double = Chem.BondType.SINGLE, Chem.BondType.DOUBLE
    for _ in range(len(mol.GetAtoms())):
        changed = False
        for bond in mol.GetBonds():
            if bond.GetBondType() != double:
                continue
            for x, y in ((bond.GetBeginAtom(), bond.GetEndAtom()), (bond.GetEndAtom(), bond.GetBeginAtom())):
                if x.GetFormalCharge() != 1:
                    continue
                for z in y.GetNeighbors():
                    b2 = mol.GetBondBetweenAtoms(y.GetIdx(), z.GetIdx())
                    if z.GetIdx() == x.GetIdx() or z.GetFormalCharge() != -1 or b2.GetBondType() != single:
                        continue
                    trial = Chem.RWMol(mol)
                    trial.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).SetBondType(single)
                    trial.GetBondBetweenAtoms(y.GetIdx(), z.GetIdx()).SetBondType(double)
                    trial.GetAtomWithIdx(x.GetIdx()).SetFormalCharge(0)
                    trial.GetAtomWithIdx(z.GetIdx()).SetFormalCharge(0)
                    try:
                        Chem.SanitizeMol(trial)
                    except Exception:
                        continue
                    mol, changed = trial.GetMol(), True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            return mol
    return mol


def reaction_key(symbols, r_coords, p_coords, adj_r, adj_p, charge: int = 0) -> tuple:
    """Identity of a reaction: the reactant graph with the bonds it forms
    and breaks marked, and the configurations at both ends (mirror images
    are one reaction)."""
    from rdkit import Chem

    union = adj_r | adj_p
    kinds = {}
    for i, j in _edges(union):
        kinds[(i, j)] = (Chem.BondType.SINGLE if adj_r[i, j] and adj_p[i, j] else
                         Chem.BondType.DOUBLE if adj_r[i, j] else Chem.BondType.TRIPLE)
    mol = _raw_mol(symbols, _edges(union), kinds)
    ranks = _ranks(mol)
    smi = Chem.MolToSmiles(mol)
    sr = _signs(symbols, r_coords, adj_r, ranks)
    sp = _signs(symbols, p_coords, adj_p, ranks)
    a = (smi, tuple(sorted(sr)), tuple(sorted(sp)), int(charge))
    b = (smi, tuple(sorted((r, -v) for r, v in sr)), tuple(sorted((r, -v) for r, v in sp)), int(charge))
    return min(a, b)


def isomorphisms(sym_a, adj_a, sym_b, adj_b, atoms_b: Optional[list] = None):
    """Every graph isomorphism from molecule a onto molecule b (or onto the
    fragment `atoms_b` of b), as lists: a's atom i -> b's atom."""
    import networkx as nx
    from networkx.algorithms import isomorphism as iso

    atoms_b = list(range(len(sym_b))) if atoms_b is None else list(atoms_b)
    if len(atoms_b) != len(sym_a):
        return
    ga, gb = nx.Graph(), nx.Graph()
    ga.add_nodes_from((i, {"el": s}) for i, s in enumerate(sym_a))
    ga.add_edges_from(_edges(adj_a))
    gb.add_nodes_from((i, {"el": sym_b[i]}) for i in atoms_b)
    gb.add_edges_from(_edges(adj_b, atoms_b))
    gm = iso.GraphMatcher(ga, gb, node_match=lambda x, y: x["el"] == y["el"])
    for m in gm.isomorphisms_iter():
        yield [m[i] for i in range(len(sym_a))]


# ------------------------------------------------------------------ edits

@dataclass
class Edit:
    """An edit of one molecule, in that molecule's atom indices.

    Atoms 0..n-1 are the molecule's (`symbols`, `coords`, `adj`: the source
    geometry the edit was made on); new atoms are n.. (`new_symbols`,
    `new_coords` in the source's frame, `new_charges`). `bonds` join each
    new atom to a source atom or another new atom; each connected group of
    new atoms hangs off exactly one source atom (its anchor)."""
    symbols: list
    coords: np.ndarray
    adj: np.ndarray
    charge: int = 0                                    # the source molecule's total charge
    swaps: dict = field(default_factory=dict)          # atom -> (element, formal charge change)
    removed: list = field(default_factory=list)        # hydrogens deleted
    replaced: dict = field(default_factory=dict)       # hydrogen -> new atom that takes its index
    new_symbols: list = field(default_factory=list)
    new_coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    new_charges: list = field(default_factory=list)
    bonds: list = field(default_factory=list)         # (i, j), indices in the combined numbering

    @property
    def n(self) -> int:
        return len(self.symbols)

    @property
    def charge_delta(self) -> int:
        return int(sum(dq for _, dq in self.swaps.values()) + sum(self.new_charges))

    def touched(self) -> set:
        """Source atoms the edit changes or hangs new atoms on."""
        out = set(self.swaps) | set(self.removed) | set(self.replaced)
        for i, j in self.bonds:
            for a in (i, j):
                if a < self.n:
                    out.add(a)
        return out

    def is_empty(self) -> bool:
        return not (self.swaps or self.removed or self.new_symbols)

    def groups(self) -> list[tuple]:
        """(anchor, new-atom indices (combined numbering), replaced H or None)."""
        n, m = self.n, len(self.new_symbols)
        adj = {k: set() for k in range(n, n + m)}
        anchor_of = {}
        for i, j in self.bonds:
            if i >= n and j >= n:
                adj[i].add(j)
                adj[j].add(i)
            else:
                new, old = (i, j) if i >= n else (j, i)
                anchor_of.setdefault(new, []).append(old)
        seen, out = set(), []
        root_h = {v: h for h, v in self.replaced.items()}
        for k in range(n, n + m):
            if k in seen:
                continue
            comp, todo = {k}, [k]
            while todo:
                a = todo.pop()
                for b in adj[a]:
                    if b not in comp:
                        comp.add(b)
                        todo.append(b)
            seen |= comp
            anchors = {a for c in comp for a in anchor_of.get(c, [])}
            h = next((root_h[c] for c in comp if c in root_h), None)
            if h is not None:
                anchors = {int(np.nonzero(self.adj[h])[0][0])} if self.adj[h].any() else anchors
            if len(anchors) != 1:
                raise VariantError("a new group must hang off exactly one atom of the original molecule")
            out.append((anchors.pop(), sorted(comp), h))
        return out

    def describe(self) -> list[str]:
        """Plain-language lines, e.g. 'H5 on C1 -> CH3'."""
        lines = []
        for a, (el, dq) in sorted(self.swaps.items()):
            lines.append(f"{self.symbols[a]}{a + 1} -> {el}{'' if not dq else f' ({dq:+d})'}")
        for anchor, comp, h in self.groups():
            formula = _formula([self.new_symbols[k - self.n] for k in comp])
            what = f"H{h + 1} on {self.symbols[anchor]}{anchor + 1}" if h is not None else \
                f"{self.symbols[anchor]}{anchor + 1}"
            lines.append(f"{what} -> {formula}" if h is not None else f"{what} + {formula}")
        for h in sorted(self.removed):
            nb = np.nonzero(self.adj[h])[0]
            on = f" on {self.symbols[int(nb[0])]}{int(nb[0]) + 1}" if len(nb) else ""
            lines.append(f"H{h + 1}{on} removed")
        return lines

    def to_json(self) -> dict:
        return {"symbols": list(self.symbols), "coords": np.asarray(self.coords).tolist(),
                "adj": _edges(self.adj), "charge": self.charge,
                "swaps": {str(k): list(v) for k, v in self.swaps.items()},
                "removed": list(self.removed), "replaced": {str(k): v for k, v in self.replaced.items()},
                "new_symbols": list(self.new_symbols), "new_coords": np.asarray(self.new_coords).tolist(),
                "new_charges": list(self.new_charges), "bonds": [list(b) for b in self.bonds]}

    @classmethod
    def from_json(cls, d: dict) -> "Edit":
        n = len(d["symbols"])
        adj = np.zeros((n, n), dtype=bool)
        for i, j in d["adj"]:
            adj[i, j] = adj[j, i] = True
        return cls(symbols=list(d["symbols"]), coords=np.array(d["coords"], dtype=float).reshape(-1, 3), adj=adj,
                   charge=int(d.get("charge", 0)),
                   swaps={int(k): (v[0], int(v[1])) for k, v in d["swaps"].items()},
                   removed=[int(x) for x in d["removed"]], replaced={int(k): int(v) for k, v in d["replaced"].items()},
                   new_symbols=list(d["new_symbols"]),
                   new_coords=np.array(d["new_coords"], dtype=float).reshape(-1, 3),
                   new_charges=[int(x) for x in d["new_charges"]], bonds=[tuple(b) for b in d["bonds"]])


def _formula(symbols) -> str:
    from collections import Counter

    c = Counter(symbols)
    order = (["C", "H"] if "C" in c else []) + sorted(k for k in c if k not in ("C", "H") or "C" not in c)
    return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in dict.fromkeys(order) if k in c)


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with dst ~ src @ R.T + t (proper rotation)."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cd - cs @ r.T


def _rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from mepd.substituents import _rotation as rot

    return rot(a / np.linalg.norm(a), b / np.linalg.norm(b))


def _axis_rotation(axis: np.ndarray, ang: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * kx @ kx


def derive_edit(parent_mol, design_mol, *, tag: str = "molAtomMapNumber") -> Edit:
    """The edit that turns `parent_mol` into `design_mol` (RDKit molecules
    with explicit hydrogens and 3D coordinates). Every atom of the design
    that comes from the parent carries the parent's index + 1 as `tag`
    (Design writes it as the atom-map number); untagged atoms are new.

    Allowed: changing the element or charge of kept atoms (with hydrogens
    re-added or removed on those atoms), and new groups or hydrogens on
    kept atoms, a group usually taking a hydrogen's place. Refused (with
    the reason): removing heavy atoms, making or breaking bonds between the
    parent's atoms, a new group bonded to two parent atoms, removing a
    hydrogen from an atom that is otherwise unchanged, and an edit that
    leaves an unpaired electron."""
    from scipy.optimize import linear_sum_assignment

    n = parent_mol.GetNumAtoms()
    psym = [a.GetSymbol() for a in parent_mol.GetAtoms()]
    pxyz = np.array(parent_mol.GetConformer().GetPositions(), dtype=float)
    padj = np.zeros((n, n), dtype=bool)
    for b in parent_mol.GetBonds():
        padj[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = padj[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = True
    pcharge = [a.GetFormalCharge() for a in parent_mol.GetAtoms()]

    orig = {}                       # design index -> parent index
    for a in design_mol.GetAtoms():
        if a.HasProp(tag) and int(a.GetProp(tag)) > 0:
            p = int(a.GetProp(tag)) - 1
            if not 0 <= p < n or p in orig.values():
                raise VariantError("the design's atom tags do not match the original molecule: load it again")
            orig[a.GetIdx()] = p
    kept = set(orig.values())
    lost_heavy = [i for i in range(n) if i not in kept and psym[i] != "H"]
    if lost_heavy:
        raise VariantError("atoms of the original molecule were deleted (" +
                           ", ".join(f"{psym[i]}{i + 1}" for i in lost_heavy) +
                           "): only element/charge changes and new groups can be carried through a network")
    dxyz = np.array(design_mol.GetConformer().GetPositions(), dtype=float)
    # The design in the parent's frame (Clean may have moved everything).
    tagged = sorted(orig)
    r, t = _kabsch(dxyz[tagged], pxyz[[orig[i] for i in tagged]])
    dxyz = dxyz @ r.T + t

    d_of = {p: d for d, p in orig.items()}
    for b in design_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in orig and j in orig and not padj[orig[i], orig[j]]:
            raise VariantError(f"a new bond {psym[orig[i]]}{orig[i] + 1}-{psym[orig[j]]}{orig[j] + 1} between atoms of "
                               "the original molecule: rings and connectivity cannot be changed across a network")
    for i, j in _edges(padj):
        if i in kept and j in kept and design_mol.GetBondBetweenAtoms(d_of[i], d_of[j]) is None:
            raise VariantError(f"the bond {psym[i]}{i + 1}-{psym[j]}{j + 1} was removed: rings and connectivity "
                               "cannot be changed across a network")

    swaps = {}
    replaced_by_single = {}         # tagged H whose element changed: a one-atom group in its place
    for d, p in orig.items():
        a = design_mol.GetAtomWithIdx(d)
        if a.GetSymbol() != psym[p] or a.GetFormalCharge() != pcharge[p]:
            if psym[p] == "H":
                replaced_by_single[p] = d
            else:
                swaps[p] = (a.GetSymbol(), a.GetFormalCharge() - pcharge[p])

    new_d = [a.GetIdx() for a in design_mol.GetAtoms() if a.GetIdx() not in orig]
    new_d += sorted(replaced_by_single.values())
    # Groups of new atoms and the parent atom each hangs off.
    new_set = set(new_d)
    comp_of, comps = {}, []
    for k in new_d:
        if k in comp_of:
            continue
        comp, todo = {k}, [k]
        while todo:
            a = todo.pop()
            for nb in design_mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j in new_set and j not in comp:
                    comp.add(j)
                    todo.append(j)
        for a in comp:
            comp_of[a] = len(comps)
        comps.append(sorted(comp))
    groups = []                    # (anchor parent index, design atoms, root design atom)
    for comp in comps:
        if len(comp) == 1 and comp[0] in replaced_by_single.values():
            p = orig[comp[0]]
            nb = [int(x) for x in np.nonzero(padj[p])[0]]
            if len(nb) != 1:
                raise VariantError(f"H{p + 1} is not bonded to one atom")
            groups.append((nb[0], comp, comp[0], p))
            continue
        links = [(a, nb.GetIdx()) for a in comp for nb in design_mol.GetAtomWithIdx(a).GetNeighbors()
                 if nb.GetIdx() in orig]
        if len(links) != 1:
            raise VariantError("a new group is bonded to " + ("no atom" if not links else "two atoms") +
                               " of the original molecule: only substituents on one atom can be carried through a network")
        groups.append((orig[links[0][1]], comp, links[0][0], None))

    # Hydrogens of the parent that are gone, per anchor, and the new groups
    # that take their places (paired by bond direction).
    removed = [i for i in range(n) if psym[i] == "H" and i not in kept]
    replaced, deleted, added = {}, [], []          # parent H -> group; H deleted; groups with no H to replace
    by_anchor = {}
    for h in removed:
        nb = [int(x) for x in np.nonzero(padj[h])[0]]
        if len(nb) != 1:
            raise VariantError(f"H{h + 1} is not bonded to one atom")
        by_anchor.setdefault(nb[0], {"h": [], "g": []})["h"].append(h)
    for gi, (anchor, comp, root, h_fixed) in enumerate(groups):
        if h_fixed is not None:
            replaced[h_fixed] = gi
            continue
        by_anchor.setdefault(anchor, {"h": [], "g": []})["g"].append(gi)
    for anchor, slot in by_anchor.items():
        hs, gs = slot["h"], slot["g"]
        if hs and gs:
            u = np.array([(pxyz[h] - pxyz[anchor]) / np.linalg.norm(pxyz[h] - pxyz[anchor]) for h in hs])
            v = np.array([(dxyz[groups[g][2]] - pxyz[anchor]) /
                          (np.linalg.norm(dxyz[groups[g][2]] - pxyz[anchor]) + 1e-9) for g in gs])
            rows, cols = linear_sum_assignment(-(u @ v.T))
            paired_h, paired_g = set(), set()
            for i, j in zip(rows, cols):
                replaced[hs[i]] = gs[j]
                paired_h.add(hs[i])
                paired_g.add(gs[j])
            deleted += [h for h in hs if h not in paired_h]
            added += [g for g in gs if g not in paired_g]
        else:
            deleted += hs
            added += gs
    # A hydrogen "replaced" by a hydrogen (Design re-adds the hydrogens of
    # an atom it touched) is no change: keep the parent's.
    unchanged = {h for h, g in replaced.items()
                 if len(groups[g][1]) == 1 and design_mol.GetAtomWithIdx(groups[g][2]).GetSymbol() == "H"
                 and groups[g][3] is None}
    for h in unchanged:
        del replaced[h]
    skip = {g for g in range(len(groups)) if g not in replaced.values() and g not in added}
    for h in deleted:
        anchor = int(np.nonzero(padj[h])[0][0])
        if anchor not in swaps:
            raise VariantError(f"H{h + 1} on {psym[anchor]}{anchor + 1} was removed without changing that atom: "
                               "that changes the molecule's charge or leaves a radical (use Charge or Element)")
    for g in added:
        anchor, comp = groups[g][0], groups[g][1]
        lone_h = len(comp) == 1 and design_mol.GetAtomWithIdx(comp[0]).GetSymbol() == "H"
        if lone_h and anchor not in swaps:
            raise VariantError(f"a hydrogen was added to {psym[anchor]}{anchor + 1} without changing that atom")

    # The new atoms, numbered after the parent's.
    new_symbols, new_coords, new_charges, idx = [], [], [], {}
    for g in range(len(groups)):
        if g in skip:
            continue
        for d in groups[g][1]:
            idx[d] = n + len(new_symbols)
            a = design_mol.GetAtomWithIdx(d)
            new_symbols.append(a.GetSymbol())
            new_coords.append(dxyz[d])
            new_charges.append(a.GetFormalCharge())
    bonds = []
    for b in design_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx or j in idx:
            if (i in idx or i in orig) and (j in idx or j in orig):
                bonds.append((idx[i] if i in idx else orig[i], idx[j] if j in idx else orig[j]))
    # A tagged H turned into another element keeps its parent index as its anchor link.
    for h, d in replaced_by_single.items():
        anchor = int(np.nonzero(padj[h])[0][0])
        if (anchor, idx[d]) not in bonds and (idx[d], anchor) not in bonds:
            bonds.append((anchor, idx[d]))
    edit = Edit(symbols=psym, coords=pxyz, adj=padj, charge=int(sum(pcharge)), swaps=swaps,
                removed=sorted(deleted), replaced={h: idx[groups[g][2]] for h, g in replaced.items()},
                new_symbols=new_symbols, new_coords=np.array(new_coords).reshape(-1, 3), new_charges=new_charges,
                bonds=bonds)
    # Electron count parity: an odd change leaves an unpaired electron.
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    dz = sum(pt.GetAtomicNumber(el) - pt.GetAtomicNumber(psym[a]) for a, (el, _) in swaps.items())
    dz += sum(pt.GetAtomicNumber(s) for s in new_symbols) - sum(1 for _ in edit.removed) - len(edit.replaced)
    if (dz - edit.charge_delta) % 2:
        raise VariantError("the edit leaves an unpaired electron (an odd number of electrons added or removed)")
    if edit.is_empty():
        raise VariantError("the design is the same molecule as the original: nothing to carry through the network")
    return edit


# ------------------------------------------------------------------ applying an edit to a frame

@dataclass
class Built:
    """A frame with an edit applied."""
    symbols: list
    coords: np.ndarray
    free: list            # new atoms and the atoms a group replaced (to relax)
    swapped: list         # atoms whose element/charge changed
    keep: list            # old frame index for each kept old atom (new index = position)
    clash: float          # closest new-to-kept distance (Å)
    new_charges: dict = field(default_factory=dict)   # built index -> formal charge of a new atom


def _local_points(edit: Edit, anchor: int, frame_adj: Optional[np.ndarray], sigma: list) -> list[int]:
    """Source atoms that fix the anchor's local frame: itself and its
    neighbours that are still bonded to it in this frame."""
    nbs = [int(b) for b in np.nonzero(edit.adj[anchor])[0]]
    if frame_adj is not None:
        nbs = [b for b in nbs if frame_adj[sigma[anchor], sigma[b]]]
    return [anchor] + nbs


def apply_edit(edits: list[tuple], symbols, coords, frame_adj: Optional[np.ndarray] = None) -> Built:
    """Apply edits [(Edit, sigma)] to one frame (`sigma[i]` = the frame
    atom for the edit's source atom i; edits on different fragments).
    Kept atoms keep their order (removed hydrogens dropped), a replaced
    hydrogen's index becomes its group's first atom, other new atoms are
    appended. `frame_adj`: which bonds exist in this frame (the anchor's
    neighbourhood is fitted only on atoms still bonded to it)."""
    c = np.array(coords, dtype=float)
    sym = list(symbols)
    extra_sym, extra_xyz, extra_q = [], [], []
    replaced_at = {}           # frame index -> new coords/symbol
    removed, swapped, free = set(), [], []
    placed_groups = []
    for edit, sigma in edits:
        n = edit.n
        for a, (el, _) in edit.swaps.items():
            sym[sigma[a]] = el
            swapped.append(sigma[a])
        for h in edit.removed:
            removed.add(sigma[h])
        for anchor, comp, h in edit.groups():
            pts = _local_points(edit, anchor, frame_adj, sigma)
            src = edit.coords[pts]
            dst = c[[sigma[p] for p in pts]]
            xyz = edit.new_coords[[k - n for k in comp]]
            if len(pts) >= 3:
                rot, t = _kabsch(src, dst)
                xyz = xyz @ rot.T + t
            else:
                xyz = xyz - edit.coords[anchor] + c[sigma[anchor]]
            if h is not None:
                # Along this frame's own X-H bond, at the design's bond length.
                root = edit.replaced[h]
                ri = comp.index(root)
                a_xyz = c[sigma[anchor]]
                cur = xyz[ri] - a_xyz
                want = c[sigma[h]] - a_xyz
                rot = _rotation(cur, want)
                length = float(np.linalg.norm(edit.new_coords[root - n] - edit.coords[anchor]))
                xyz = (xyz - a_xyz) @ rot.T + a_xyz
                xyz += want / np.linalg.norm(want) * length - (xyz[ri] - a_xyz)
            placed_groups.append((edit, sigma, anchor, comp, h, xyz))
    # Kept atoms, then new atoms; respin groups that land on kept atoms.
    kept_idx = [i for i in range(len(sym)) if i not in removed]
    gone = removed | {sigma[h] for _, sigma, _, _, h, _ in placed_groups if h is not None}
    others = np.array([c[i] for i in kept_idx if i not in gone]) if len(kept_idx) else np.zeros((0, 3))
    worst = 9.9
    for gi, (edit, sigma, anchor, comp, h, xyz) in enumerate(placed_groups):
        a_xyz = c[sigma[anchor]]
        mask = np.ones(len(others), dtype=bool)
        anchor_pos = [k for k, i in enumerate([i for i in kept_idx if i not in gone]) if i == sigma[anchor]]
        mask[anchor_pos] = False
        rest = others[mask]
        movable = [k for k in range(len(comp)) if h is None or comp[k] != edit.replaced[h]]

        def dmin(x):
            if not len(rest) or not movable:
                return 9.9
            return float(np.min(np.linalg.norm(x[movable][:, None] - rest[None], axis=-1)))

        best = dmin(xyz)
        if best < CLASH and h is not None:
            axis = c[sigma[h]] - a_xyz
            for k in range(1, 24):
                trial = (xyz - a_xyz) @ _axis_rotation(axis, 2 * np.pi * k / 24).T + a_xyz
                d = dmin(trial)
                if d > best:
                    best, xyz = d, trial
            placed_groups[gi] = (edit, sigma, anchor, comp, h, xyz)
        worst = min(worst, best)
    for edit, sigma, anchor, comp, h, xyz in placed_groups:
        for k, atom in enumerate(comp):
            el = edit.new_symbols[atom - edit.n]
            q = edit.new_charges[atom - edit.n]
            if h is not None and atom == edit.replaced[h]:
                replaced_at[sigma[h]] = (el, xyz[k], q)
            else:
                extra_sym.append(el)
                extra_xyz.append(xyz[k])
                extra_q.append(q)
    out_sym, out_xyz, keep, new_q = [], [], [], {}
    for i in kept_idx:
        if i in replaced_at:
            el, x, q = replaced_at[i]
            out_sym.append(el)
            out_xyz.append(x)
            free.append(len(out_sym) - 1)
            new_q[len(out_sym) - 1] = q
        else:
            out_sym.append(sym[i])
            out_xyz.append(c[i])
        keep.append(i)
    new_of = {old: k for k, old in enumerate(keep)}
    for el, x, q in zip(extra_sym, extra_xyz, extra_q):
        out_sym.append(el)
        out_xyz.append(x)
        free.append(len(out_sym) - 1)
        new_q[len(out_sym) - 1] = q
    return Built(symbols=out_sym, coords=np.array(out_xyz), free=sorted(free),
                 swapped=sorted(new_of[i] for i in swapped), keep=keep, clash=worst, new_charges=new_q)


def built_adjacency(edits: list[tuple], frame_adj: np.ndarray, built: Built) -> np.ndarray:
    """The substituted frame's bonds: the frame's own (minus removed atoms)
    plus each edit's new bonds, without re-reading them from distances."""
    n = len(built.symbols)
    adj = np.zeros((n, n), dtype=bool)
    new_of = {old: k for k, old in enumerate(built.keep)}
    for i, j in _edges(frame_adj):
        if i in new_of and j in new_of:
            adj[new_of[i], new_of[j]] = adj[new_of[j], new_of[i]] = True
    # New atoms, in the order apply_edit appended them.
    cursor = len(built.keep)
    for edit, sigma in edits:
        where = {}
        for anchor, comp, h in edit.groups():
            for atom in comp:
                if h is not None and atom == edit.replaced[h]:
                    where[atom] = new_of[sigma[h]]
        order = []
        for anchor, comp, h in edit.groups():
            for atom in comp:
                if atom not in where:
                    order.append(atom)
        for atom in order:
            where[atom] = cursor
            cursor += 1
        for i, j in edit.bonds:
            a = where[i] if i >= edit.n else new_of[sigma[i]]
            b = where[j] if j >= edit.n else new_of[sigma[j]]
            adj[a, b] = adj[b, a] = True
    return adj


# ------------------------------------------------------------------ network propagation

@dataclass
class Species:
    """A parent species: a reference geometry (Angstrom) and its charge."""
    id: str
    symbols: list
    coords: np.ndarray
    charge: int = 0
    multiplicity: int = 1
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.adj = adjacency(self.symbols, self.coords)
        self.key = molecule_key(self.symbols, self.coords, self.adj, self.charge)
        self.graph_key = self.key[0]


@dataclass
class Reaction:
    """A parent reaction: its species (with repeats) and its geometries in
    one shared atom order: reactant end, TS (None if no TS was found yet)
    and product end, each (symbols, coords in Angstrom)."""
    id: str
    reactants: list
    products: list
    r: Optional[tuple] = None
    ts: Optional[tuple] = None
    p: Optional[tuple] = None
    charge: int = 0
    multiplicity: int = 1
    label: str = ""
    note: str = ""          # e.g. "remapped" when the end's order was repaired


@dataclass
class VariantSpecies:
    key: tuple
    parent: str                 # parent species id
    edit: Edit                  # on a geometry of the parent species
    symbols: list               # the variant molecule (for adding it to the graph)
    coords: np.ndarray
    charge: int
    multiplicity: int
    via: str = ""               # variant reaction it first came out of ("" for the edited species)
    label: str = ""
    adj: Optional[np.ndarray] = None   # its bonds (the built geometry's distances may be off)

    @property
    def smiles(self) -> Optional[str]:
        adj = self.adj if self.adj is not None else adjacency(self.symbols, self.coords)
        return smiles_from_graph(self.symbols, self.coords, adj, self.charge)


@dataclass
class VariantReaction:
    key: tuple
    parent: str                 # parent reaction id
    status: str                 # ok | blocked | no_ts
    why: str = ""
    symbols: list = field(default_factory=list)
    r: Optional[np.ndarray] = None
    ts: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None
    free: dict = field(default_factory=dict)       # frame -> atoms to relax in the fast estimate
    reacting: list = field(default_factory=list)   # atoms whose bonds change
    clash: float = 9.9
    charge: int = 0
    multiplicity: int = 1
    reactants: list = field(default_factory=list)  # species ids: parent ids or variant labels ("v:<n>")
    products: list = field(default_factory=list)
    placement: str = ""
    entry: str = ""             # "r" or "p": the side the edit came in from
    bonds_r: list = field(default_factory=list)    # the substituted ends' bonds (not re-read from distances)
    bonds_p: list = field(default_factory=list)

    def spec(self, parent: "Reaction", parent_energies: Optional[list] = None) -> dict:
        """The input of `mepd variant` for this reaction (Angstrom)."""
        par = {"symbols": list(parent.r[0]), "charge": parent.charge,
               "r": np.asarray(parent.r[1]).tolist(), "p": np.asarray(parent.p[1]).tolist(),
               "ts": np.asarray(parent.ts[1]).tolist() if parent.ts is not None else None,
               "energies": parent_energies}
        return {"symbols": list(self.symbols), "charge": self.charge, "multiplicity": self.multiplicity,
                "r": np.asarray(self.r).tolist(), "p": np.asarray(self.p).tolist(),
                "ts": np.asarray(self.ts).tolist() if self.ts is not None else None,
                "bonds_r": [list(map(int, b)) for b in self.bonds_r],
                "bonds_p": [list(map(int, b)) for b in self.bonds_p],
                "free": {k: [int(i) for i in v] for k, v in self.free.items()},
                "reacting": [int(i) for i in self.reacting], "parent": par}


@dataclass
class Plan:
    base: str
    edit: Edit
    species: dict = field(default_factory=dict)            # label "v:<n>" -> VariantSpecies
    reactions: list = field(default_factory=list)          # VariantReaction
    shared: list = field(default_factory=list)             # parent reaction ids untouched
    replaced: list = field(default_factory=list)           # parent species ids with variants
    problems: dict = field(default_factory=dict)           # parent reaction id -> why it was not carried
    rejected: dict = field(default_factory=dict)           # parent reaction id -> why a placement was dropped
    warnings: list = field(default_factory=list)

    def variants_of(self, sid: str) -> list[str]:
        return [k for k, v in self.species.items() if v.parent == sid]


def _reach(edit: Edit, sigma, rx: Reaction, x_in, adj_in, adj_out) -> float:
    """Distance (Å, in the TS when there is one) from the atoms an edit
    placement changes to the nearest atom whose bonds change."""
    center = np.nonzero((adj_in != adj_out).any(axis=1))[0]
    ts = _frame(rx, "ts")
    x = ts[1] if ts is not None and len(ts[1]) == len(x_in) else x_in
    at = [sigma[a] for a in edit.touched()]
    if not len(center) or not at:
        return 0.0
    return float(np.min(np.linalg.norm(x[at][:, None, :] - x[center][None, :, :], axis=-1)))


def _frame(rx: Reaction, which: str):
    f = getattr(rx, which)
    if f is None:
        return None
    return list(f[0]), np.asarray(f[1], dtype=float)


def _assign_fragments(frags, sym, adj, coords, sids, species: dict) -> Optional[list]:
    """Which fragment of an end is which species: [(sid, atoms)], or None
    when the end does not hold exactly the reaction's species."""
    want = list(sids)
    out = []
    for atoms in frags:
        fsym = [sym[i] for i in atoms]
        sub = adj[np.ix_(atoms, atoms)]
        key = molecule_key(fsym, coords[atoms], sub)[0]
        hit = next((s for s in want if species[s].graph_key == key), None)
        if hit is None:
            return None
        want.remove(hit)
        out.append((hit, atoms))
    return out if not want else None


def _fragment_edit(frag_atoms: list, sym, coords, adj, built: Built, built_adj: np.ndarray) -> Optional[Edit]:
    """The edit, on the unsubstituted fragment `frag_atoms` of a frame,
    that gives the substituted fragment: read off the built frame."""
    new_of = {old: k for k, old in enumerate(built.keep)}
    local = {a: k for k, a in enumerate(frag_atoms)}
    n = len(frag_atoms)
    fsym = [sym[a] for a in frag_atoms]
    fxyz = coords[frag_atoms]
    fadj = adj[np.ix_(frag_atoms, frag_atoms)]
    # Atoms of the built frame that belong to the substituted fragment.
    members = {new_of[a] for a in frag_atoms if a in new_of}
    grow = list(members)
    while grow:
        a = grow.pop()
        for b in np.nonzero(built_adj[a])[0]:
            b = int(b)
            if b not in members and b >= len(built.keep):
                members.add(b)
                grow.append(b)
    old_of = {k: a for a, k in new_of.items()}
    swaps, removed, replaced = {}, [], {}
    new_symbols, new_coords, new_charges, bonds, idx = [], [], [], [], {}
    for a in frag_atoms:
        if a not in new_of:
            removed.append(local[a])
            continue
        k = new_of[a]
        if built.symbols[k] != sym[a]:
            if sym[a] == "H":
                idx[k] = n + len(new_symbols)
                replaced[local[a]] = idx[k]
                new_symbols.append(built.symbols[k])
                new_coords.append(built.coords[k])
                new_charges.append(built.new_charges.get(k, 0))
            else:
                swaps[local[a]] = (built.symbols[k], 0)
        elif k in built.swapped:
            swaps[local[a]] = (built.symbols[k], 0)
    for k in sorted(m for m in members if m >= len(built.keep)):
        idx[k] = n + len(new_symbols)
        new_symbols.append(built.symbols[k])
        new_coords.append(built.coords[k])
        new_charges.append(built.new_charges.get(k, 0))
    for i, j in _edges(built_adj, sorted(members)):
        if i in idx or j in idx:
            a = idx[i] if i in idx else local[old_of[i]]
            b = idx[j] if j in idx else local[old_of[j]]
            bonds.append((a, b))
    e = Edit(symbols=fsym, coords=fxyz, adj=fadj, swaps=swaps, removed=sorted(removed), replaced=replaced,
             new_symbols=new_symbols, new_coords=np.array(new_coords).reshape(-1, 3),
             new_charges=new_charges, bonds=bonds)
    return None if e.is_empty() else e


def _charge_delta_by_atom(edits: list[tuple]) -> dict:
    """Frame atom -> formal charge change carried by it (swaps), plus new
    atoms' charges assigned to their anchor's frame atom."""
    out = {}
    for edit, sigma in edits:
        for a, (_, dq) in edit.swaps.items():
            out[sigma[a]] = out.get(sigma[a], 0) + dq
        for anchor, comp, h in edit.groups():
            q = sum(edit.new_charges[k - edit.n] for k in comp)
            if q:
                out[sigma[anchor]] = out.get(sigma[anchor], 0) + q
    return out


def propagate(species: dict, reactions: list, base: str, edit: Edit, *, max_rounds: int = 50) -> Plan:
    """Carry `edit` (on a geometry of species `base`) through `reactions`,
    in both directions, until no new variant species appear."""
    if base not in species:
        raise VariantError("the edited species is not in the network")
    plan = Plan(base=base, edit=edit)
    ident = list(range(edit.n))
    if Species("_", edit.symbols, edit.coords, edit.charge).graph_key != species[base].graph_key:
        raise VariantError("the design's original molecule is not the species it is applied to")
    built = apply_edit([(edit, ident)], edit.symbols, edit.coords, edit.adj)
    badj = built_adjacency([(edit, ident)], edit.adj, built)
    q = species[base].charge + edit.charge_delta
    key0 = molecule_key(built.symbols, built.coords, badj, q)
    plan.species["v:0"] = VariantSpecies(key=key0, parent=base, edit=edit, symbols=built.symbols,
                                         coords=built.coords, charge=q,
                                         multiplicity=species[base].multiplicity, label="v:0", adj=badj)
    by_key = {key0: "v:0"}
    seen_rx: dict = {}
    done_combos: set = set()

    for _ in range(max_rounds):
        grew = False
        for rx in reactions:
            for side in ("r", "p"):
                sids = rx.reactants if side == "r" else rx.products
                other = rx.products if side == "r" else rx.reactants
                involved = [s for s in sids if plan.variants_of(s)]
                if not involved:
                    continue
                f_in, f_out = _frame(rx, side), _frame(rx, "p" if side == "r" else "r")
                if f_in is None or f_out is None:
                    plan.problems[rx.id] = "no geometry for this reaction's ends"
                    continue
                sym, x_in = f_in
                _, x_out = f_out
                if len(sym) != len(f_out[0]) or list(sym) != list(f_out[0]):
                    plan.problems[rx.id] = "its two ends do not have the same atoms in the same order"
                    continue
                adj_in, adj_out = adjacency(sym, x_in), adjacency(sym, x_out)
                frags_in = _assign_fragments(components(adj_in), sym, adj_in, x_in, sids, species)
                frags_out = _assign_fragments(components(adj_out), sym, adj_out, x_out, other, species)
                if frags_in is None or frags_out is None:
                    plan.problems[rx.id] = ("its geometries do not hold the reaction's species (the ends were read "
                                            "as other molecules)")
                    continue
                slots = [(sid, atoms) for sid, atoms in frags_in if plan.variants_of(sid)]
                choices = [plan.variants_of(sid) for sid, _ in slots]
                for combo in itertools.product(*choices):
                    ck = (rx.id, side, combo)
                    if ck in done_combos:
                        continue
                    done_combos.add(ck)
                    grew |= _carry(plan, rx, side, sym, x_in, x_out, adj_in, adj_out, slots, combo,
                                   frags_out, species, by_key, seen_rx)
        if not grew:
            break
    else:
        plan.warnings.append(f"stopped after {max_rounds} rounds: the variant network may be incomplete")
    _consistent(plan)
    replaced = {v.parent for v in plan.species.values()}
    for rx in reactions:
        if not set(rx.reactants + rx.products) & replaced:
            plan.shared.append(rx.id)
        elif not any(vr.parent == rx.id for vr in plan.reactions):
            plan.problems.setdefault(rx.id, plan.rejected.get(rx.id) or "no placement of the edit fits this reaction")
    plan.replaced = sorted(replaced)
    counts: dict = {}
    for vr in plan.reactions:
        counts[vr.parent] = counts.get(vr.parent, 0) + (vr.status != "blocked")
    for rid, k in counts.items():
        if k > MANY_PLACEMENTS:
            plan.warnings.append(f"{rid} gives {k} variant reactions (distinct placements of the edit)")
    for rid in list(plan.problems):
        if any(vr.parent == rid for vr in plan.reactions):
            del plan.problems[rid]
    return plan


def _carry(plan: Plan, rx: Reaction, side: str, sym, x_in, x_out, adj_in, adj_out, slots, combo,
           frags_out, species, by_key, seen_rx) -> bool:
    """All placements of the variants `combo` (one per slot) in reaction
    `rx` entered from `side`; adds new variant reactions and species.
    True if anything new appeared."""
    per_slot = []
    for (sid, atoms), vlabel in zip(slots, combo):
        v = plan.species[vlabel]
        fsym = [sym[i] for i in atoms]
        options, images = [], set()
        for sigma_local in isomorphisms(v.edit.symbols, v.edit.adj, sym, adj_in, atoms):
            sigma = list(sigma_local)
            img = _edit_image(v.edit, sigma)
            if img in images:
                continue
            images.add(img)
            b = apply_edit([(v.edit, sigma)], sym, x_in, adj_in)
            badj = built_adjacency([(v.edit, sigma)], adj_in, b)
            frag = _built_fragment(b, badj, atoms)
            k = molecule_key([b.symbols[i] for i in frag], b.coords[frag], badj[np.ix_(frag, frag)], v.charge)
            if k == v.key:
                options.append(sigma)
        if not options:
            plan.problems.setdefault(rx.id, f"no placement of the edit in {species[sid].name or sid} gives the same "
                                            "stereoisomer")
            return False
        del fsym
        # Placements that give the same reaction (below, the first one is
        # kept) differ only in where the edit sits, e.g. which methyl of a
        # spectator NMe4+ gets an OH: the one nearest the reacting atoms
        # first, where a group can take part (an H-bond to the epoxide).
        options.sort(key=lambda s: _reach(v.edit, s, rx, x_in, adj_in, adj_out))
        per_slot.append([(v.edit, s) for s in options])
    grew = False
    for pick in itertools.product(*per_slot):
        edits = list(pick)
        ts_frame = _frame(rx, "ts")
        b_in = apply_edit(edits, sym, x_in, adj_in)
        b_out = apply_edit(edits, sym, x_out, adj_out)
        adj_b_in = built_adjacency(edits, adj_in, b_in)
        adj_b_out = built_adjacency(edits, adj_out, b_out)
        dq = sum(_charge_delta_by_atom(edits).values())
        charge = rx.charge + dq
        if side == "r":
            r_b, p_b, adj_r, adj_p = b_in, b_out, adj_b_in, adj_b_out
        else:
            r_b, p_b, adj_r, adj_p = b_out, b_in, adj_b_out, adj_b_in
        key = reaction_key(r_b.symbols, r_b.coords, p_b.coords, adj_r, adj_p, charge)
        if (rx.id, key) in seen_rx:
            continue
        seen_rx[(rx.id, key)] = True
        grew = True
        # Blocked: an edited hydrogen (removed or replaced) changes its bonds.
        changed = {i for i in range(len(sym)) if (adj_in[i] != adj_out[i]).any()}
        hit = sorted({s[h] for e, s in edits for h in list(e.removed) + list(e.replaced)} & changed)
        reacting = sorted(r_b.keep.index(i) for i in changed if i in r_b.keep)
        vr = VariantReaction(key=key, parent=rx.id, status="ok", symbols=r_b.symbols, r=r_b.coords, p=p_b.coords,
                             reacting=reacting, clash=min(b_in.clash, b_out.clash), charge=charge,
                             multiplicity=rx.multiplicity, entry=side,
                             placement="; ".join(_placement_text(e, s, sym) for e, s in edits))
        # Relaxed in the fast estimate: new atoms, and swapped ones (an S
        # where an O was), except in the TS where those react.
        vr.free = {"r": sorted(set(r_b.free) | set(r_b.swapped)), "p": sorted(set(p_b.free) | set(p_b.swapped))}
        vr.bonds_r, vr.bonds_p = _edges(adj_r), _edges(adj_p)
        if hit:
            vr.status = "blocked"
            vr.why = "an edited hydrogen takes part in this step (" + ", ".join(f"H{h + 1}" for h in hit) + ")"
        if ts_frame is not None and vr.status == "ok":
            ts_sym, x_ts = ts_frame
            kept_bonds = adj_in & adj_out
            b_ts = apply_edit(edits, sym, x_ts, kept_bonds)
            vr.ts = b_ts.coords
            vr.free["ts"] = sorted(set(b_ts.free) | (set(b_ts.swapped) - set(reacting)))
            vr.clash = min(vr.clash, b_ts.clash)
        elif vr.status == "ok":
            vr.status = "no_ts"
            vr.why = ((rx.note + ": ") if "longer route" in rx.note else "the parent has no TS yet: ") + \
                "a path search is needed"
        # Species on both sides: unchanged parents, or variants (new ones registered).
        in_labels = list(combo)
        in_parents = [sid for sid, _ in slots]
        side_in = (rx.reactants if side == "r" else rx.products)
        rest_in = list(side_in)
        for s in in_parents:
            rest_in.remove(s)
        out_labels = []
        if vr.status != "blocked":
            b_out_adj = adj_b_out
            q_atoms = _charge_delta_by_atom(edits)
            outs = []
            for sid, atoms in frags_out:
                frag = _built_fragment(b_out, b_out_adj, atoms)
                fsym = [b_out.symbols[i] for i in frag]
                untouched = (fsym == [sym[a] for a in atoms] and len(frag) == len(atoms)
                             and not (set(frag) & set(b_out.free)) and not (set(frag) & set(b_out.swapped)))
                outs.append((sid, atoms, frag, fsym, untouched))
            # A species the variant replaces never comes out as itself: that
            # placement moved the edit onto other molecules (e.g. a symmetric
            # carbonate run backwards hands the edited O to CO2).
            bare = [sid for sid, _, _, _, untouched in outs if untouched and plan.variants_of(sid)]
            if bare:
                plan.rejected[rx.id] = ("the edit would leave " + ", ".join(sorted({species[s].name or s for s in bare}))
                                        + " unchanged")
                continue
            for sid, atoms, frag, fsym, untouched in outs:
                if untouched:
                    out_labels.append(sid)
                    continue
                fx = b_out.coords[frag]
                fadj = b_out_adj[np.ix_(frag, frag)]
                fq = species[sid].charge + sum(q_atoms.get(a, 0) for a in atoms)
                k = molecule_key(fsym, fx, fadj, fq)
                label = by_key.get(k)
                if label is None:
                    fe = _fragment_edit(atoms, sym, x_out, adj_out, b_out, b_out_adj)
                    if fe is None:
                        out_labels.append(sid)
                        continue
                    fe.charge = species[sid].charge
                    _carry_charges(fe, edits, atoms)
                    label = f"v:{len(plan.species)}"
                    while label in plan.species:
                        label = f"v:{int(label[2:]) + 1}"
                    by_key[k] = label
                    plan.species[label] = VariantSpecies(key=k, parent=sid, edit=fe, symbols=fsym, coords=fx, charge=fq,
                                                         multiplicity=species[sid].multiplicity,
                                                         via=f"{rx.id}#{len(plan.reactions)}", label=label,
                                                         adj=fadj)
                out_labels.append(label)
        if side == "r":
            vr.reactants, vr.products = in_labels + rest_in, out_labels
        else:
            vr.reactants, vr.products = out_labels, in_labels + rest_in
        plan.reactions.append(vr)
    return grew


def _consistent(plan: Plan) -> None:
    """Keep the variant network that descends from the edited species, in
    which no species it replaces appears as itself: drop reactions with a
    bare replaced species (on either side; one can become replaced after
    the reaction was carried), then species no kept reaction reaches from
    the edited one, until nothing changes."""
    while True:
        replaced = {v.parent for v in plan.species.values()}
        keep = [vr for vr in plan.reactions
                if not any(x in replaced for x in vr.reactants + vr.products if not x.startswith("v:"))]
        reach, grow = {"v:0"}, True
        while grow:
            grow = False
            for vr in keep:
                if vr.status == "blocked":
                    continue
                ids = {x for x in vr.reactants + vr.products if x.startswith("v:")}
                if ids & reach and not ids <= reach:
                    reach |= ids
                    grow = True
        keep = [vr for vr in keep if {x for x in vr.reactants + vr.products if x.startswith("v:")} <= reach]
        species = {k: v for k, v in plan.species.items() if k in reach}
        if len(keep) == len(plan.reactions) and len(species) == len(plan.species):
            return
        for vr in plan.reactions:
            if vr not in keep:
                plan.rejected.setdefault(vr.parent, "the edit would leave a species it replaces unchanged")
        plan.reactions, plan.species = keep, species


def _carry_charges(fe: Edit, edits: list[tuple], atoms: list) -> None:
    """Formal-charge changes of swapped atoms, carried onto the fragment's
    edit (by frame atom)."""
    local = {a: k for k, a in enumerate(atoms)}
    for edit, sigma in edits:
        for a, (el, dq) in edit.swaps.items():
            f = sigma[a]
            if f in local and local[f] in fe.swaps:
                fe.swaps[local[f]] = (fe.swaps[local[f]][0], dq)


def _built_fragment(b: Built, badj: np.ndarray, atoms: list) -> list[int]:
    """Indices in the built frame of the substituted fragment grown from
    the frame's atoms `atoms` (kept ones plus new atoms bonded to them)."""
    new_of = {old: k for k, old in enumerate(b.keep)}
    out = {new_of[a] for a in atoms if a in new_of}
    grow = list(out)
    while grow:
        a = grow.pop()
        for c in np.nonzero(badj[a])[0]:
            c = int(c)
            if c not in out and c >= len(b.keep):
                out.add(c)
                grow.append(c)
    return sorted(out)


def _edit_image(edit: Edit, sigma: list) -> tuple:
    """Where an edit lands in a frame (for de-duplicating placements that
    differ only in atoms the edit does not touch)."""
    return (tuple(sorted((sigma[a], v) for a, v in edit.swaps.items())),
            tuple(sorted(sigma[h] for h in edit.removed)),
            tuple(sorted((sigma[h], v) for h, v in edit.replaced.items())),
            tuple(sorted((sigma[anchor], h is not None and sigma[h]) for anchor, _, h in edit.groups())))


def _placement_text(edit: Edit, sigma: list, sym) -> str:
    parts = []
    for anchor, comp, h in edit.groups():
        formula = _formula([edit.new_symbols[k - edit.n] for k in comp])
        parts.append(f"{formula} on {sym[sigma[anchor]]}{sigma[anchor] + 1}")
    for a, (el, _) in edit.swaps.items():
        parts.append(f"{sym[sigma[a]]}{sigma[a] + 1}->{el}")
    return ", ".join(parts)
