"""Is this the same molecule? Does this IRC connect these two structures?

`same_molecule` is mepd's one answer to "same species": the bond graph
(heavy atoms, ignoring atom order) and, by default, stereochemistry, ignoring
conformation. `connects` applies it to both ends of an IRC (or any two
structures), in either direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

    @property
    def same(self) -> bool:
        """Same molecule, stereochemistry included; "can't tell" is not the same."""
        return self.graph and self.stereo is True

    @property
    def undetermined(self) -> bool:
        """The graphs match but the stereochemistry could not be compared."""
        return self.graph and self.stereo is None


def _openbabel_stereo_smiles(node: StructureNode) -> str:
    """OpenBabel's canonical isomeric SMILES of `node`, from its own bond-order
    and stereo perception on the 3D geometry (independent of RDKit's)."""
    from openbabel import pybel

    return pybel.readstring("xyz", node.structure.to_xyz()).write("can").split()[0]


def _second_opinion(c: Comparison, a: StructureNode, b: StructureNode) -> Comparison:
    """RDKit could not decide the stereochemistry: ask OpenBabel. Its "same" is
    accepted; its "different" is not taken as a mismatch (for the unusual
    valences RDKit rejects, OpenBabel's bond orders differ between geometries
    of one molecule too), so the comparison stays undecided."""
    try:
        oa, ob = _openbabel_stereo_smiles(a), _openbabel_stereo_smiles(b)
    except Exception as exc:
        return Comparison(c.graph, None, c.smiles, c.bond_orders, f"{c.reason}; OpenBabel failed: {type(exc).__name__}")
    if oa == ob:
        return Comparison(c.graph, True, (oa, ob), c.bond_orders, f"{c.reason}; OpenBabel stereo SMILES agree",
                          method="openbabel")
    return Comparison(c.graph, None, c.smiles, c.bond_orders,
                      f"{c.reason}; OpenBabel stereo SMILES differ ({oa} vs {ob}), not decisive")


def compare_structures(a: StructureNode, b: StructureNode) -> Comparison:
    """Compare two structures: bond graphs (heavy atoms, atom order ignored),
    then stereochemistry (stereo SMILES from 3D; hydrogens' positions in the
    graph included, so a tautomer differs; plus the count of trans small-ring
    alkenes, which SMILES cannot express). Conformation is ignored. A
    comparison that cannot be decided says so (`stereo is None`) instead of
    passing. When RDKit cannot decide, OpenBabel's stereo SMILES give a second
    opinion, which can only confirm "same" (`Comparison.method`).

    >>> c = mepd.compare_structures(irc_end, reactant); c.same, c.stereo, c.smiles
    """
    c = _compare_rdkit(a, b)
    if c.graph and c.stereo is None and max(len(a.coords), len(b.coords)) < 100:
        return _second_opinion(c, a, b)
    return c


def _compare_rdkit(a: StructureNode, b: StructureNode) -> Comparison:
    """`compare_structures` with RDKit only (bond orders from RDKit's DetermineBondOrders)."""
    from mepd.nodes.nodehelpers import _n_trans_small_ring_alkenes, _stereo_smiles

    if getattr(a, "graph", None) is None or getattr(b, "graph", None) is None:
        return Comparison(graph=False, stereo=None, reason="a structure has no bond graph")
    if not a.graph.remove_Hs().is_bond_isomorphic_to(b.graph.remove_Hs()):
        return Comparison(graph=False, stereo=None, reason="heavy-atom bond graphs differ")
    if max(len(a.coords), len(b.coords)) >= 100:
        return Comparison(graph=True, stereo=None, reason="100+ atoms: stereo SMILES are not built")
    try:
        (sa, oa), (sb, ob) = _stereo_smiles(a), _stereo_smiles(b)
    except Exception as exc:
        return Comparison(graph=True, stereo=None, reason=f"stereo SMILES failed: {type(exc).__name__}: {exc}"[:300])
    if not (oa and ob):
        which = " and ".join(n for n, ok in (("first", oa), ("second", ob)) if not ok)
        # Without bond orders, E/Z is invisible: compare what can be seen, but
        # a "same" there is not a decision.
        try:
            ta, tb = _stereo_smiles(a, False)[0], _stereo_smiles(b, False)[0]
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


def same_molecule(a: StructureNode, b: StructureNode, *, stereo: bool = True) -> bool:
    """True when `a` and `b` are the same molecule (or set of molecules):
    isomorphic bond graphs without hydrogens, atom order ignored, and with
    `stereo` the same stereochemistry too (`mepd.compare_structures`). A stereochemistry
    that cannot be compared (see `Comparison.stereo`) is not the same.
    Conformation is ignored. Structures without a graph never match.

    >>> mepd.same_molecule(irc_end, reactant)
    """
    c = compare_structures(a, b)
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


def connects(path, start: StructureNode, end: StructureNode, *, stereo: bool = True) -> Match:
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

    fwd = (compare_structures(a, start), compare_structures(b, end))
    if all(side(c) for c in fwd):
        return Match(True, True, False, fwd)
    rev = (compare_structures(b, start), compare_structures(a, end))
    # More matched sides wins; then more sides whose graphs match (an undecided
    # stereochemistry must not flip the direction); forward on a tie.
    score = lambda cs: (sum(side(c) for c in cs), sum(c.graph for c in cs))  # noqa: E731
    if score(rev) > score(fwd):
        return Match(side(rev[0]), side(rev[1]), True, rev)
    return Match(side(fwd[0]), side(fwd[1]), False, fwd)
