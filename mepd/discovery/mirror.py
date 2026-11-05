"""Mirror images in the explorer's kinetics.

Every structure has a mirror image (x -> -x) of exactly the same energy. A
transition state that cannot be superimposed on its own mirror image (a
chiral TS) therefore has a twin, and the reaction runs through both. The
explorer finds one twin, or both, by chance: in Cp + methyl acrylate it
found both endo twins and one exo, and counting what was found tilted the
kinetics towards endo. So nothing is collapsed and nothing is left to
chance: the kinetics count every structure together with its mirror image.

- A step whose TS superimposes on its mirror image (an achiral TS) has no
  twin.
- A step whose twin is already in the network (a step between the mirror
  images of its ends whose TS superimposes on its mirrored TS) is counted
  once each, as found.
- Otherwise the twin is added: a step at the same energy between the mirror
  images of the ends. When both ends are their own mirror images (achiral
  molecules) that is a second channel between the same pair: the step
  counts twice. When an end is chiral the twin leads to its enantiomer,
  which joins the kinetics as a species of its own (same energy) if the
  explorer did not find it.
- A species that is its own mirror molecule but whose structure (each
  molecule the free energies count apart) is chiral -- a chiral conformer
  of an achiral molecule -- stands for two mirror-image minima: its free
  energy gets -RT ln 2 per chiral molecule. Without that, a chiral TS out of
  a chiral conformer would count twice where the two conformers each have
  their own TS (a factor of 2/2 = 1).

These are the optical-isomer terms of the reaction-path degeneracy
(Fernandez-Ramos, Ellingson, Meana-Paneda, Marques & Truhlar, Theor. Chem.
Acc. 118, 813 (2007)); the rotational symmetry numbers are already in the
free energies (mepd.discovery.free_energy).

Which species is a structure's mirror image: the network's own identity
test (mepd.compare.same_molecule: bond graph + stereo SMILES). Whether two
geometries superimpose: over the automorphisms of the heavy-atom bond
graph (partial bonds included, so it works for TSs too), a proper rotation
onto the heavy atoms, the hydrogens assigned to their nearest counterparts,
all-atom RMSD within `tol_angstrom`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from mepd.discovery.kinetics import Step

BOHR_PER_ANGSTROM = 1.8897259886
HARTREE_TO_KCAL = 627.5094740631
TOL_ANGSTROM = 0.15
MAX_AUTOMORPHISMS = 2000


def mirrored(structure):
    """The mirror image of a qcdata Structure (x -> -x)."""
    xyz = np.asarray(structure.geometry, dtype=float).copy()
    xyz[:, 0] *= -1.0
    return structure.model_copy(update={"geometry": xyz})


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """RMSD of P onto Q after the best proper rotation; (rmsd, R, centroid P, centroid Q)."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = (P - cp) @ R.T - (Q - cq)
    return float(np.sqrt((diff ** 2).sum() / len(P))), R, cp, cq


def _heavy_graph(symbols: Sequence[str], X: np.ndarray, scale: float = 1.3, hydrogens: bool = False):
    """Heavy-atom graph (bonds and partial bonds: within `scale` x the sum of
    covalent radii), each heavy atom labeled with its element and its count
    of nearest hydrogens. `hydrogens`: every atom a node (for molecules with
    too few heavy atoms to fix an orientation)."""
    import networkx as nx
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    heavy = [i for i, s in enumerate(symbols) if hydrogens or s != "H"]
    hyd = [i for i, s in enumerate(symbols) if not hydrogens and s == "H"]
    n_h = {i: 0 for i in heavy}
    if heavy:
        for h in hyd:
            n_h[min(heavy, key=lambda i: float(np.linalg.norm(X[i] - X[h])))] += 1
    G = nx.Graph()
    for i in heavy:
        G.add_node(i, label=(symbols[i], n_h[i]))
    rc = {i: table.GetRcovalent(symbols[i]) for i in heavy}
    for a, i in enumerate(heavy):
        for j in heavy[:a]:
            if np.linalg.norm(X[i] - X[j]) < scale * (rc[i] + rc[j]):
                G.add_edge(i, j)
    return G, heavy, hyd


def superimposable(symbols: Sequence[str], X: np.ndarray, Y: np.ndarray, *,
                   tol_angstrom: float = TOL_ANGSTROM) -> Optional[bool]:
    """Whether geometry Y (Angstrom, the same atoms as X) superimposes on X
    by a proper rotation and a permutation of equivalent atoms. None when
    the heavy-atom graph has more than MAX_AUTOMORPHISMS automorphisms and no
    match was found among the first ones (undecided)."""
    from networkx.algorithms.isomorphism import GraphMatcher
    from scipy.optimize import linear_sum_assignment

    symbols = [str(s) for s in symbols]
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    G, heavy, hyd = _heavy_graph(symbols, X)
    if len(heavy) < 3:   # one or two heavy atoms fix no orientation: permute the hydrogens too
        G, heavy, hyd = _heavy_graph(symbols, X, hydrogens=True)
    matcher = GraphMatcher(G, G, node_match=lambda a, b: a["label"] == b["label"])
    for k, mapping in enumerate(matcher.isomorphisms_iter()):
        if k >= MAX_AUTOMORPHISMS:
            return None
        src = [mapping[i] for i in heavy]     # Y's atom that plays X's atom i
        rmsd, R, cp, cq = _kabsch_rmsd(Y[src], X[heavy])
        if rmsd >= tol_angstrom:
            continue
        if not hyd:
            return True
        Yh = (Y[hyd] - cp) @ R.T + cq
        cost = np.linalg.norm(Yh[:, None, :] - X[hyd][None, :, :], axis=2)
        r, c = linear_sum_assignment(cost)
        total = rmsd ** 2 * len(heavy) + float((cost[r, c] ** 2).sum())
        if math.sqrt(total / len(symbols)) < tol_angstrom:
            return True
    return False


def _angstrom(structure) -> np.ndarray:
    return np.asarray(structure.geometry, dtype=float) / BOHR_PER_ANGSTROM


def is_chiral(structure, *, tol_angstrom: float = TOL_ANGSTROM) -> Optional[bool]:
    """Whether a structure's geometry cannot be superimposed on its mirror image."""
    X = _angstrom(structure)
    Y = X.copy()
    Y[:, 0] *= -1.0
    same = superimposable(list(map(str, structure.symbols)), X, Y, tol_angstrom=tol_angstrom)
    return None if same is None else not same


def _key(structure) -> str:
    xyz = np.round(np.asarray(structure.geometry, dtype=float), 5) + 0.0
    return hashlib.sha1(repr((list(map(str, structure.symbols)), xyz.tolist(), int(structure.charge),
                              int(structure.multiplicity))).encode()).hexdigest()[:24]


@dataclass
class Mirrors:
    """How the mirror images extend a network (geometry only: the same for
    every set of energies)."""
    n_species: int
    virtual_of: list = field(default_factory=list)      # added species n_species + v mirrors species virtual_of[v]
    species_mirror: list = field(default_factory=list)  # each species' mirror image (an index, maybe virtual)
    chiral_conformers: dict = field(default_factory=dict)  # species -> its chiral molecules (each -RT ln 2)
    twins: list = field(default_factory=list)          # (step index, mirror a, mirror b): added steps
    found_twins: list = field(default_factory=list)    # (step, step): twins both in the network
    undecided: list = field(default_factory=list)      # notes: structures it could not decide

    def extend(self, steps: Sequence[Step], energies: Sequence[float], temperature: float
               ) -> tuple[int, list, list]:
        """(number of species, steps, energies) with the mirror images: added
        species and steps carry the energies of the ones they mirror; a chiral
        conformer of an achiral molecule gets -RT ln 2 per chiral molecule."""
        rt = 8.314462618e-3 / 4.184 * temperature / HARTREE_TO_KCAL      # Hartree
        g = [float(e) for e in energies]
        # two species the explorer found that are each other's mirror image have one energy: the lower
        # (the other's lowest conformer, reflected, is a conformer of each)
        for k, j in enumerate(self.species_mirror[:self.n_species]):
            if k < j < self.n_species:
                g[k] = g[j] = min(g[k], g[j])
        for k, n in self.chiral_conformers.items():
            g[k] -= n * rt * math.log(2.0)
        g += [g[k] for k in self.virtual_of]
        out = list(steps)
        for m, o in self.found_twins:   # the same for twin TSs both found (on one energy scale)
            if abs(out[m].ts_energy - out[o].ts_energy) * HARTREE_TO_KCAL <= 1.0:
                low = min(out[m].ts_energy, out[o].ts_energy)
                out[m] = Step(out[m].a, out[m].b, low, out[m].label, out[m].files)
                out[o] = Step(out[o].a, out[o].b, low, out[o].label, out[o].files)
        for m, ma, mb in self.twins:
            st = steps[m]
            out.append(Step(ma, mb, st.ts_energy, f"{st.label} (mirror image)",
                            {"mirror_of": st.label, **({"ts": st.files["ts"]} if st.files.get("ts") else {})}))
        return self.n_species + len(self.virtual_of), out, g

    def record(self) -> dict:
        return {"virtual_species": [{"index": self.n_species + v, "mirror_of": k}
                                    for v, k in enumerate(self.virtual_of)],
                "species_mirror": list(self.species_mirror),
                "chiral_conformers": {str(k): n for k, n in self.chiral_conformers.items()},
                "added_twins": [{"step": m, "a": ma, "b": mb} for m, ma, mb in self.twins],
                "found_twins": [list(p) for p in self.found_twins],
                "undecided": list(self.undecided)}


class MirrorFinder:
    """Builds `Mirrors` for a network, remembering what it decided per
    geometry (the explorer asks again every time its network grows)."""

    def __init__(self, *, tol_angstrom: float = TOL_ANGSTROM):
        self.tol = tol_angstrom
        self._chiral: dict = {}
        self._mirror_key: dict = {}
        self._hash: dict = {}
        self._same: dict = {}

    def _is_chiral(self, structure, what: str, notes: list) -> bool:
        key = _key(structure)
        if key not in self._chiral:
            self._chiral[key] = is_chiral(structure, tol_angstrom=self.tol)
        c = self._chiral[key]
        if c is None:
            notes.append(f"{what}: too many symmetry-equivalent atom orderings to decide its mirror image; "
                         "counted as achiral")
            return False
        return c

    def _graph_hash(self, node) -> str:
        """A hash of the heavy-atom bond graph: equal for the same molecule (a
        cheap first test before the stereo SMILES)."""
        import networkx as nx

        key = _key(node.structure)
        if key not in self._hash:
            self._hash[key] = nx.weisfeiler_lehman_graph_hash(node.graph.remove_Hs(), node_attr="element")
        return self._hash[key]

    def _compare(self, mirror_node, node) -> str:
        """'same', 'undetermined' or 'different' (mepd.compare.compare_structures), remembered."""
        from mepd.compare import compare_structures

        key = (_key(mirror_node.structure), _key(node.structure))
        if key not in self._same:
            c = compare_structures(mirror_node, node)
            self._same[key] = "same" if c.same else "undetermined" if c.undetermined else "different"
        return self._same[key]

    def _same_species(self, mirror_node, species_nodes, k: int) -> Optional[int]:
        """The species `mirror_node` (the mirror image of species k) is: k
        itself first, then any other; None when none is."""
        # undetermined stereo: the explorer could not tell them apart either
        if self._compare(mirror_node, species_nodes[k]) != "different":
            return k
        h = self._graph_hash(mirror_node)
        for j, node in enumerate(species_nodes):
            if j != k and self._graph_hash(node) == h and self._compare(mirror_node, node) == "same":
                return j
        return None

    def species_mirrors(self, species_nodes: Sequence) -> list:
        """Each species' mirror image among `species_nodes` (itself for an
        achiral molecule); -1 when the explorer did not find it."""
        from mepd.nodes.node import StructureNode

        out = []
        for k, node in enumerate(species_nodes):
            key = _key(node.structure)
            if key not in self._mirror_key:
                self._mirror_key[key] = StructureNode(structure=mirrored(node.structure))
            j = self._same_species(self._mirror_key[key], species_nodes, k)
            out.append(-1 if j is None else j)
        return out

    def __call__(self, species_nodes: Sequence, steps: Sequence[Step], ts_structures: Sequence,
                 parts: Optional[Callable] = None) -> Mirrors:
        """`species_nodes`: StructureNodes of the species; `ts_structures`:
        each step's TS (or path peak) Structure, None if it has none;
        `parts(structure)`: the atom groups the free energies count apart
        (None: the whole structure)."""
        n = len(species_nodes)
        out = Mirrors(n)
        notes = out.undecided
        mirror = list(range(n))
        for k, (node, j) in enumerate(zip(species_nodes, self.species_mirrors(species_nodes))):
            if j < 0:
                mirror[k] = n + len(out.virtual_of)
                out.virtual_of.append(k)
            else:
                mirror[k] = j
                if j == k:
                    st = node.structure
                    groups = parts(st) if parts is not None else [list(range(len(st.symbols)))]
                    count = 0
                    for g in groups:
                        sub = st.model_copy(update={"symbols": [st.symbols[i] for i in g],
                                                    "geometry": np.asarray(st.geometry)[g]})
                        count += self._is_chiral(sub, f"species {k}", notes)
                    if count:
                        out.chiral_conformers[k] = count
        out.species_mirror = mirror + list(out.virtual_of)   # an added species' mirror image is the one it mirrors

        def mirror_of(i):
            return out.species_mirror[i]

        taken = set()
        for m, (st, ts) in enumerate(zip(steps, ts_structures)):
            if ts is None or st.a == st.b:
                continue
            if not self._is_chiral(ts, f"TS of step {st.label}", notes):
                continue
            ma, mb = mirror_of(st.a), mirror_of(st.b)
            twin = None
            X = _angstrom(mirrored(ts))
            for o, (st2, ts2) in enumerate(zip(steps, ts_structures)):
                if o == m or ts2 is None or {st2.a, st2.b} != {ma, mb}:
                    continue
                if abs(st2.ts_energy - st.ts_energy) * HARTREE_TO_KCAL > 1.0:
                    continue
                if superimposable(list(map(str, ts.symbols)), _angstrom(ts2), X, tol_angstrom=self.tol):
                    twin = o
                    break
            if twin is not None:
                if (twin, m) not in taken:
                    out.found_twins.append((m, twin))
                    taken.add((m, twin))
                continue
            out.twins.append((m, ma, mb))
        return out
