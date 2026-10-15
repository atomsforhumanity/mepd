"""Is this the same molecule? Does this IRC connect these two structures?

`same_molecule` is mepd's one answer to "same species": the bond graph
(heavy atoms, ignoring atom order) and, by default, stereochemistry, ignoring
conformation. `connects` applies it to both ends of an IRC (or any two
structures), in either direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mepd.nodes.node import StructureNode

__all__ = ["same_molecule", "connects", "Match"]


def same_molecule(a: StructureNode, b: StructureNode, *, stereo: bool = True) -> bool:
    """True when `a` and `b` are the same molecule (or set of molecules):
    isomorphic bond graphs without hydrogens, atom order ignored, and with
    `stereo` the same stereochemistry too (stereo SMILES from 3D, plus the
    count of trans small-ring alkenes, which stereo SMILES cannot see).
    Conformation is ignored. Structures without a graph never match.

    >>> mepd.same_molecule(irc_end, reactant)
    """
    from mepd.nodes.nodehelpers import _is_connectivity_identical, _n_trans_small_ring_alkenes

    if getattr(a, "graph", None) is None or getattr(b, "graph", None) is None:
        return False
    if not _is_connectivity_identical(a, b, verbose=False, collect_comparison=False,
                                      disregard_stereochem=not stereo):
        return False
    if not stereo:
        return True
    na, nb = _n_trans_small_ring_alkenes(a), _n_trans_small_ring_alkenes(b)
    return na is None or nb is None or na == nb


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
    fwd = (same_molecule(a, start, stereo=stereo), same_molecule(b, end, stereo=stereo))
    if all(fwd):
        return Match(True, True, False)
    rev = (same_molecule(b, start, stereo=stereo), same_molecule(a, end, stereo=stereo))
    if sum(rev) > sum(fwd):
        return Match(rev[0], rev[1], True)
    return Match(fwd[0], fwd[1], False)
