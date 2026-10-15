"""Calculations on structures with an engine: energies, minimization,
transition-state search, IRC, geodesic interpolation and barriers.

Every function takes the structures and an engine (`mepd.engine(...)`), returns
structures or a result object, and raises instead of returning None:
`mepd.UnsupportedOperation` when the engine cannot do it, `mepd.ConvergenceError`
when the calculation ran but did not reach a usable result. Nothing is printed
and no files are written; `mepd.write_xyz` writes results out.

Engines implement these through optional methods (see `mepd.engines.Engine`):
`compute_geometry_optimization(node, keywords=None) -> trajectory`,
`compute_geometry_optimizations(nodes, keywords=None) -> trajectories` (batched),
`compute_transition_state(node, keywords=None) -> node` and
`compute_irc_chain(ts_node, keywords=None) -> Chain`. An engine without an IRC
of its own gets geomeTRIC's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, overload

import numpy as np

from mepd.errors import ConvergenceError, UnsupportedOperation
from mepd.nodes.node import StructureNode
from mepd.units import HARTREE_TO_KCAL

__all__ = ["energies", "optimize", "find_ts", "irc", "IRCResult", "interpolate", "barrier_kcal"]


def _method(engine, name: str, what: str):
    fn = getattr(engine, name, None)
    if not callable(fn):
        raise UnsupportedOperation(f"{type(engine).__name__} cannot do {what}.")
    return fn


def _kw(keywords: dict) -> dict:
    """Engine keywords, passed only when there are some: an engine method may
    take just the structure."""
    return {"keywords": dict(keywords)} if keywords else {}


def energies(structures: StructureNode | Sequence[StructureNode], engine) -> np.ndarray:
    """Energies in Hartree (one per structure; a single structure gives a
    one-element array). They are also cached on each structure (`node.energy`)."""
    nodes = [structures] if isinstance(structures, StructureNode) else list(structures)
    return np.asarray(engine.compute_energies(nodes), dtype=float)


@overload
def optimize(structures: StructureNode, engine, **keywords: Any) -> StructureNode: ...
@overload
def optimize(structures: Sequence[StructureNode], engine, **keywords: Any) -> list[StructureNode]: ...


def optimize(structures, engine, **keywords):
    """Minimize one structure, or several (batched when the engine can).
    Returns the minimized structure(s), in the same order.

    `keywords` go to the engine's optimizer (e.g. `maxiter=200`). Raises
    ConvergenceError if an optimization does not converge.

    >>> reactant, product = mepd.optimize([reactant, product], engine)
    """
    single = isinstance(structures, StructureNode)
    nodes = [structures] if single else list(structures)
    batch = getattr(engine, "compute_geometry_optimizations", None)
    if callable(batch) and len(nodes) > 1:
        trajectories = batch(nodes, **_kw(keywords))
    else:
        one = _method(engine, "compute_geometry_optimization", "geometry optimization")
        trajectories = [one(n, **_kw(keywords)) for n in nodes]
    out = []
    for i, trajectory in enumerate(trajectories):
        if not trajectory:
            raise ConvergenceError(f"The optimization of structure {i} returned no structures.")
        out.append(trajectory[-1])
    return out[0] if single else out


def find_ts(guess: StructureNode, engine, **keywords: Any) -> StructureNode:
    """Optimize a transition-state guess to a first-order saddle point.

    `keywords` go to the engine's TS optimizer (for the ASE-based engines, Sella:
    e.g. `fmax`, `maxiter`). Raises ConvergenceError when it ends on no usable
    structure.

    >>> ts = mepd.find_ts(path.get_ts_node(), engine)
    """
    fn = _method(engine, "compute_transition_state", "transition-state optimization")
    try:
        ts = fn(node=guess, **_kw(keywords))
    except (ConvergenceError, UnsupportedOperation):
        raise
    except Exception as exc:
        raise ConvergenceError(f"Transition-state optimization failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(ts, StructureNode):
        raise ConvergenceError(
            f"Transition-state optimization ended on no usable structure ({type(ts).__name__}).")
    return ts


@dataclass
class IRCResult:
    """An intrinsic reaction coordinate: `path` (a Chain from one minimum
    through the TS to the other); `ends` its two end structures."""
    path: Any

    @property
    def ends(self) -> tuple[StructureNode, StructureNode]:
        return self.path.nodes[0], self.path.nodes[-1]

    @property
    def nodes(self) -> list[StructureNode]:
        return list(self.path.nodes)


def irc(ts: StructureNode, engine, **keywords: Any) -> IRCResult:
    """Follow the IRC down both sides of a transition state.

    Uses the engine's own IRC (Sella for the ASE-based engines, including
    g-xTB), else geomeTRIC's. `keywords` go to it. Raises ConvergenceError if
    the IRC fails. `mepd.connects(result, reactant, product)` checks where it
    leads.
    """
    fn = getattr(engine, "compute_irc_chain", None)
    try:
        if callable(fn):
            chain = fn(ts, **_kw(keywords))
        else:
            from mepd.irc_geometric import compute_irc_chain_with_geometric

            chain = compute_irc_chain_with_geometric(engine, ts, keywords=dict(keywords) if keywords else None)
    except (ConvergenceError, UnsupportedOperation):
        raise
    except Exception as exc:
        raise ConvergenceError(f"IRC failed: {type(exc).__name__}: {exc}") from exc
    if chain is None or len(chain.nodes) < 2:
        raise ConvergenceError("IRC returned no path.")
    return IRCResult(path=chain)


def interpolate(start: StructureNode, end: StructureNode, *, nimages: int = 15, **options: Any):
    """Geodesic interpolation (Zhu, Thompson and Martinez, 2019) between two
    structures in the same atom order: a Chain of `nimages` structures that
    avoids atom collisions. `options` go to the geodesic smoother (e.g.
    `friction`, `nudge`, `random_seed`)."""
    from mepd.chainhelpers import run_geodesic

    if list(start.symbols) != list(end.symbols):
        raise ValueError("start and end must list the same atoms in the same order (map them first).")
    return run_geodesic([start, end], nimages=nimages, **options)


def barrier_kcal(ts: StructureNode, reactants: StructureNode | Sequence[StructureNode], engine=None) -> float:
    """Barrier in kcal/mol: the TS energy minus the lowest energy among
    `reactants` (e.g. all reactant conformers found, so the barrier is measured
    from the lowest one). Energies not yet computed are computed with `engine`.
    The TS and the reactants must hold the same atoms."""
    pool = [reactants] if isinstance(reactants, StructureNode) else list(reactants)
    if not pool:
        raise ValueError("barrier_kcal needs at least one reactant structure.")
    nodes = [ts, *pool]
    if any(sorted(n.symbols) != sorted(ts.symbols) for n in pool):
        raise ValueError("The TS and the reactants must hold the same atoms.")
    missing = [n for n in nodes if getattr(n, "_cached_energy", None) is None]
    if missing:
        if engine is None:
            raise ValueError("Some energies are not computed yet; pass an engine.")
        energies(missing, engine)
    return float((ts.energy - min(n.energy for n in pool)) * HARTREE_TO_KCAL)
