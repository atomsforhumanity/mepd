"""Is this the same molecule? Does this IRC connect these two structures?

`same_molecule` is mepd's one answer to "same species": the bond graph
(heavy atoms, ignoring atom order) and, by default, stereochemistry, ignoring
conformation. `connects` applies it to both ends of an IRC (or any two
structures), in either direction.

Metal-ligand contacts. By default every bond counts, metal-ligand contacts
included, so a ligand that binds, comes off, or binds through another atom
makes another species. With `metal_ligand_contacts=False` they are left out:
two structures are the same species when their covalent bonds and
stereochemistry are, whatever the coordination at the metal, and
`Comparison.metal_ligand_changed` records whether the contacts differ. This
follows the treatment of dative metal-ligand contacts as zero-order bonds,
kept apart from covalent bond changes, in YARP's organometallic extension
(Z. Li, H.-H. Hsu, B. M. Savoie, "Finding the Roads Less Calculated in
Organometallics with Automated Transition State Searches", ChemRxiv (2026),
doi:10.26434/chemrxiv.15006374/v1; preprint), where a step that changes only
the coordination is a conformational reorganization rather than an
elementary step.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from mepd.nodes.node import StructureNode

__all__ = ["same_molecule", "compare_structures", "Comparison", "connects", "Match"]


@dataclass(frozen=True)
class Comparison:
    """How two structures compare, and why (`mepd.compare_structures`).

    `graph`: their heavy-atom bond graphs are isomorphic. `stereo`: True or
    False when the stereo SMILES (built from the full bond graph, hydrogens
    included, with bond orders and stereochemistry from 3D) could be compared;
    None when that could not be decided ("can't tell"): a SMILES failed to
    build, bond orders could not be assigned for one of them (then E/Z
    stereochemistry is invisible), the system has 100+ atoms, or the small-ring
    trans-alkene count could not be perceived. `reason` says which.
    """
    graph: bool
    stereo: bool | None
    smiles: tuple[str | None, str | None] = (None, None)
    bond_orders: tuple[bool, bool] = (False, False)
    reason: str = ""
    #: which method decided `stereo`: "rdkit" (bond orders from RDKit), or
    #: "openbabel" (its own bond orders and stereo SMILES; used only when RDKit
    #: could not decide, and only to say "same")
    method: str = "rdkit"
    #: with `metal_ligand_contacts=False`: whether the metal-ligand contacts
    #: differ although the covalent bonds match (None: not checked, or no metal)
    metal_ligand_changed: bool | None = None

    @property
    def same(self) -> bool:
        """Same molecule, stereochemistry included; "can't tell" is not the same."""
        return self.graph and self.stereo is True

    @property
    def undetermined(self) -> bool:
        """The graphs match but the stereochemistry could not be compared."""
        return self.graph and self.stereo is None


def _openbabel_stereo_smiles(node: StructureNode, metal_ligand_contacts: bool = True) -> str:
    """OpenBabel's canonical isomeric SMILES of `node`, from its own bond-order
    and stereo perception on the 3D geometry (independent of RDKit's); without
    `metal_ligand_contacts`, of the structure with its metal atoms taken out."""
    from openbabel import pybel

    from mepd.nodes.nodehelpers import _METAL_SYMBOLS

    xyz = node.structure.to_xyz()
    if not metal_ligand_contacts:
        lines = xyz.splitlines()
        atoms = [ln for ln in lines[2:] if ln.split() and ln.split()[0] not in _METAL_SYMBOLS]
        xyz = "\n".join([str(len(atoms)), lines[1], *atoms]) + "\n"
    return pybel.readstring("xyz", xyz).write("can").split()[0]


def _second_opinion(c: Comparison, a: StructureNode, b: StructureNode, metal_ligand_contacts: bool = True) -> Comparison:
    """RDKit could not decide the stereochemistry: ask OpenBabel. Its "same" is
    accepted; its "different" is not taken as a mismatch (for the unusual
    valences RDKit rejects, OpenBabel's bond orders differ between geometries
    of one molecule too), so the comparison stays undecided."""
    try:
        ob_smiles = _openbabel_stereo_smiles if metal_ligand_contacts else (
            lambda n: _openbabel_stereo_smiles(n, metal_ligand_contacts=False))
        oa, ob = ob_smiles(a), ob_smiles(b)
    except Exception as exc:
        return Comparison(c.graph, None, c.smiles, c.bond_orders, f"{c.reason}; OpenBabel failed: {type(exc).__name__}")
    if oa == ob:
        return Comparison(c.graph, True, (oa, ob), c.bond_orders, f"{c.reason}; OpenBabel stereo SMILES agree",
                          method="openbabel")
    return Comparison(c.graph, None, c.smiles, c.bond_orders,
                      f"{c.reason}; OpenBabel stereo SMILES differ ({oa} vs {ob}), not decisive")


def compare_structures(a: StructureNode, b: StructureNode, *, metal_ligand_contacts: bool = True) -> Comparison:
    """Compare two structures: bond graphs (heavy atoms, atom order ignored),
    then stereochemistry (stereo SMILES from 3D; hydrogens' positions in the
    graph included, so a tautomer differs; plus the count of trans small-ring
    alkenes, which SMILES cannot express). Conformation is ignored. A
    comparison that cannot be decided says so (`stereo is None`) instead of
    passing. When RDKit cannot decide, OpenBabel's stereo SMILES give a second
    opinion, which can only confirm "same" (`Comparison.method`). With
    `metal_ligand_contacts=False`, bonds to metal atoms are left out of both
    the graphs and the stereochemistry (see the module docstring).

    >>> c = mepd.compare_structures(irc_end, reactant); c.same, c.stereo, c.smiles
    """
    c = _compare_rdkit(a, b, metal_ligand_contacts)
    if c.graph and c.stereo is None and max(len(a.coords), len(b.coords)) < 100:
        c = _second_opinion(c, a, b, metal_ligand_contacts)
    if not metal_ligand_contacts and c.graph and _has_metal(a):
        c = replace(c, metal_ligand_changed=not a.graph.remove_Hs().is_bond_isomorphic_to(b.graph.remove_Hs()))
    return c


def _has_metal(node: StructureNode) -> bool:
    from mepd.nodes.nodehelpers import _METAL_SYMBOLS

    return any(s in _METAL_SYMBOLS for s in node.structure.symbols)


def _covalent(graph):
    """`graph` without its bonds to metal atoms (the metal atoms stay, unbonded)."""
    from mepd.nodes.nodehelpers import _METAL_SYMBOLS

    metal = {i for i, d in graph.nodes(data=True) if d.get("element") in _METAL_SYMBOLS}
    if not metal:
        return graph
    g = graph.copy()
    g.remove_edges_from([(u, v) for u, v in graph.edges() if u in metal or v in metal])
    g.set_neighbors()
    return g


def _compare_rdkit(a: StructureNode, b: StructureNode, metal_ligand_contacts: bool = True) -> Comparison:
    """`compare_structures` with RDKit only (bond orders from RDKit's DetermineBondOrders)."""
    from mepd.nodes.nodehelpers import _n_trans_small_ring_alkenes, _stereo_smiles

    if getattr(a, "graph", None) is None or getattr(b, "graph", None) is None:
        return Comparison(graph=False, stereo=None, reason="a structure has no bond graph")
    ga, gb = (n.graph if metal_ligand_contacts else _covalent(n.graph) for n in (a, b))
    # the keyword only when it is off, so the default calls are exactly as before
    smiles = _stereo_smiles if metal_ligand_contacts else (
        lambda n, bond_orders=True: _stereo_smiles(n, bond_orders, metal_ligand_contacts=False))
    if not ga.remove_Hs().is_bond_isomorphic_to(gb.remove_Hs()):
        return Comparison(graph=False, stereo=None, reason="heavy-atom bond graphs differ" if metal_ligand_contacts
                          else "heavy-atom covalent bond graphs differ (metal-ligand contacts left out)")
    if max(len(a.coords), len(b.coords)) >= 100:
        return Comparison(graph=True, stereo=None, reason="100+ atoms: stereo SMILES are not built")
    try:
        (sa, oa), (sb, ob) = (smiles(n) for n in (a, b))
    except Exception as exc:
        return Comparison(graph=True, stereo=None, reason=f"stereo SMILES failed: {type(exc).__name__}: {exc}"[:300])
    if not (oa and ob):
        which = " and ".join(n for n, ok in (("first", oa), ("second", ob)) if not ok)
        # Without bond orders, E/Z is invisible: compare what can be seen, but
        # a "same" there is not a decision.
        try:
            ta, tb = (smiles(n, False)[0] for n in (a, b))
        except Exception:
            ta = tb = None
        if ta is not None and ta != tb:
            return Comparison(graph=True, stereo=False, smiles=(ta, tb), bond_orders=(oa, ob),
                              reason="different without bond orders")
        return Comparison(graph=True, stereo=None, smiles=(sa, sb), bond_orders=(oa, ob),
                          reason=f"bond orders could not be assigned for the {which} structure: E/Z not compared")
    if sa != sb:
        return Comparison(graph=True, stereo=False, smiles=(sa, sb), bond_orders=(oa, ob),
                          reason="stereo SMILES differ")
    na, nb = _n_trans_small_ring_alkenes(a), _n_trans_small_ring_alkenes(b)
    if na is None or nb is None:
        return Comparison(graph=True, stereo=None, smiles=(sa, sb), bond_orders=(oa, ob),
                          reason="small-ring trans-alkene count could not be perceived")
    if na != nb:
        return Comparison(graph=True, stereo=False, smiles=(sa, sb), bond_orders=(oa, ob),
                          reason=f"small-ring trans alkenes differ ({na} vs {nb})")
    return Comparison(graph=True, stereo=True, smiles=(sa, sb), bond_orders=(oa, ob), reason="same")


def same_molecule(a: StructureNode, b: StructureNode, *, stereo: bool = True,
                  metal_ligand_contacts: bool = True) -> bool:
    """True when `a` and `b` are the same molecule (or set of molecules):
    isomorphic bond graphs without hydrogens, atom order ignored, and with
    `stereo` the same stereochemistry too (`mepd.compare_structures`). A stereochemistry
    that cannot be compared (see `Comparison.stereo`) is not the same.
    Conformation is ignored. Structures without a graph never match. With
    `metal_ligand_contacts=False`, bonds to metal atoms are not compared.

    >>> mepd.same_molecule(irc_end, reactant)
    """
    c = compare_structures(a, b, metal_ligand_contacts=metal_ligand_contacts)
    return c.graph if not stereo else c.same


@dataclass(frozen=True)
class Match:
    """Which of two expected structures the ends of a path reproduce.

    `start`/`end`: whether an end is the same molecule as the expected start
    or end. `reversed`: the path runs end -> start (its first frame matched
    the expected end). `both` means the path connects the two.
    """
    start: bool
    end: bool
    reversed: bool = False
    #: the `Comparison` behind each side (start side, end side), in the chosen
    #: direction; a record of why, not part of equality
    comparisons: tuple = field(default=(), compare=False)

    @property
    def undetermined(self) -> tuple[bool, bool]:
        """Per side (start, end): the graphs match but the stereochemistry
        could not be compared, so the side does not count as matched."""
        return tuple(bool(c.undetermined) for c in self.comparisons) if self.comparisons else (False, False)

    @property
    def both(self) -> bool:
        return self.start and self.end

    @property
    def any(self) -> bool:
        return self.start or self.end

    @property
    def none(self) -> bool:
        return not self.any

    def __bool__(self) -> bool:
        return self.both


def _ends(path) -> tuple[StructureNode, StructureNode]:
    ends = getattr(path, "ends", None)       # mepd.IRCResult
    if ends is not None:
        return ends
    nodes = getattr(path, "nodes", path)     # a Chain, or a sequence of structures
    nodes = list(nodes)
    if len(nodes) < 2:
        raise ValueError("A path needs at least two structures to have two ends.")
    return nodes[0], nodes[-1]


def connects(path, start: StructureNode, end: StructureNode, *, stereo: bool = True,
             metal_ligand_contacts: bool = True) -> Match:
    """Which of `start` and `end` the two ends of `path` are (`same_molecule`).

    `path` is an `IRCResult`, a `Chain`, or any sequence of structures (its
    first and last are its ends). Both directions are tried; the one that
    matches more sides wins, the forward one on a tie. `bool(match)` is
    True only when both sides match.

    >>> hit = mepd.connects(mepd.irc(ts, engine), reactant, product)
    >>> hit.both, hit.start, hit.end, hit.reversed
    """
    a, b = _ends(path)

    def side(c: Comparison) -> bool:
        return c.same if stereo else c.graph

    cmp = lambda x, y: compare_structures(x, y, metal_ligand_contacts=metal_ligand_contacts)  # noqa: E731
    fwd = (cmp(a, start), cmp(b, end))
    if all(side(c) for c in fwd):
        return Match(True, True, False, fwd)
    rev = (cmp(b, start), cmp(a, end))
    # More matched sides wins; then more sides whose graphs match (an undecided
    # stereochemistry must not flip the direction); forward on a tie.
    score = lambda cs: (sum(side(c) for c in cs), sum(c.graph for c in cs))  # noqa: E731
    if score(rev) > score(fwd):
        return Match(side(rev[0]), side(rev[1]), True, rev)
    return Match(side(fwd[0]), side(fwd[1]), False, fwd)
