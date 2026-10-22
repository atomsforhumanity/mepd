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

import inspect
from dataclasses import dataclass
from typing import Any, Sequence, overload

import numpy as np

from mepd.errors import ConvergenceError, UnsupportedOperation
from mepd.nodes.node import StructureNode
from mepd.units import HARTREE_TO_KCAL

__all__ = ["energies", "optimize", "minimize", "frequencies", "Frequencies", "find_ts", "irc", "IRCResult",
           "interpolate", "barrier_kcal", "IMAGINARY_CUTOFF_CM"]


def _method(engine, name: str, what: str):
    fn = getattr(engine, name, None)
    if not callable(fn):
        raise UnsupportedOperation(f"{type(engine).__name__} cannot do {what}.")
    return fn


def _kw(keywords: dict) -> dict:
    """Engine keywords, passed only when there are some: an engine method may
    take just the structure."""
    return {"keywords": dict(keywords)} if keywords else {}


def _takes_keywords(fn) -> bool:
    """Whether an engine method takes a `keywords` argument (a user's engine may not)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "keywords" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


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


#: A vibrational mode counts as imaginary when its frequency is below
#: -IMAGINARY_CUTOFF_CM cm^-1. Smaller negative values are numerical noise of a
#: finite-difference Hessian along very soft modes (torsions, intermolecular).
IMAGINARY_CUTOFF_CM = 10.0


@dataclass
class Frequencies:
    """Harmonic vibrational frequencies in cm^-1, translations and rotations
    removed, ascending; imaginary modes are negative. `modes`: the matching
    Cartesian displacement vectors, one (n_atoms, 3) array per frequency (None
    when the engine gave none)."""
    wavenumbers_cm: np.ndarray
    modes: np.ndarray | None = None

    def imaginary(self, cutoff_cm: float = IMAGINARY_CUTOFF_CM) -> np.ndarray:
        """The imaginary modes: those below -`cutoff_cm`."""
        return self.wavenumbers_cm[self.wavenumbers_cm < -abs(cutoff_cm)]

    def n_imaginary(self, cutoff_cm: float = IMAGINARY_CUTOFF_CM) -> int:
        return int(len(self.imaginary(cutoff_cm)))

    def lowest(self, n: int = 3) -> list[float]:
        return [float(x) for x in self.wavenumbers_cm[:n]]


def frequencies(structure: StructureNode, engine) -> Frequencies:
    """Harmonic frequencies of `structure` from the engine's Hessian (g-xTB's
    own vibrational analysis for the g-xTB engine; finite differences of
    gradients otherwise). They are also kept on the structure (`.frequencies`).

    >>> mepd.frequencies(ts, engine).n_imaginary()
    1
    """
    node = structure.copy()
    if not (callable(getattr(engine, "_compute_hessian_result", None)) or callable(getattr(engine, "compute_hessian", None))):
        raise UnsupportedOperation(f"{type(engine).__name__} cannot compute a Hessian, so frequencies cannot be "
                                   "checked (find_ts(..., check_frequencies=False) skips the check).")
    try:
        own = getattr(engine, "_compute_hessian_result", None)
        if callable(own):
            result = own(node)
        else:
            from mepd.engines.engine import build_hessian_result_from_matrix

            result = build_hessian_result_from_matrix(node=node, hessian=engine.compute_hessian(node))
        w = np.asarray(result.results.freqs_wavenumber, dtype=float)
        modes = getattr(result.results, "normal_modes_cartesian", None)
    except UnsupportedOperation:
        raise
    except Exception as exc:
        raise ConvergenceError(f"Frequency calculation failed: {type(exc).__name__}: {exc}") from exc
    order = np.argsort(w)
    modes = np.asarray(modes, dtype=float)[order] if modes is not None and len(modes) == len(w) else None
    freqs = Frequencies(wavenumbers_cm=w[order], modes=modes)
    structure.frequencies = freqs
    return freqs


def minimize(structure: StructureNode, engine, *, check_frequencies: bool = True, escape_saddles: int = 3,
             displacement_angstrom: float = 0.1, imaginary_cutoff_cm: float = IMAGINARY_CUTOFF_CM,
             **keywords: Any) -> StructureNode:
    """Minimize `structure` to a true minimum.

    The minimization must converge (`keywords` go to the engine's optimizer,
    e.g. `fmax` in eV/Angstrom). With `check_frequencies` (the default) its
    frequencies are then checked: an optimizer can also stop on a saddle point,
    where the forces vanish too. A structure with an imaginary mode is displaced
    along the lowest one by `displacement_angstrom` (largest atomic shift), both
    ways, re-minimized, and the lower result kept; up to `escape_saddles` times.
    Raises ConvergenceError when a minimization does not converge, and
    WrongStationaryPointError when it is still a saddle after that. The result
    carries `.frequencies` and `.saddle_escapes` (how many displacements it took).
    Escaping a saddle can lead to another species: compare the result with what
    you expected (`mepd.same_molecule`).

    >>> end = mepd.minimize(irc.ends[1], engine, fmax=0.001)
    """
    from mepd.errors import ElectronicStructureError, WrongStationaryPointError
    from mepd.units import ANGSTROM_TO_BOHR

    current = optimize(structure, engine, **keywords)
    current.saddle_escapes = 0
    if not check_frequencies:
        return current
    for attempt in range(escape_saddles + 1):
        freqs = frequencies(current, engine)
        if freqs.n_imaginary(imaginary_cutoff_cm) == 0:
            current.saddle_escapes = attempt
            return current
        if attempt == escape_saddles or freqs.modes is None:
            break
        mode = np.asarray(freqs.modes[0], dtype=float).reshape(-1, 3)
        step = mode / np.linalg.norm(mode, axis=1).max() * displacement_angstrom * ANGSTROM_TO_BOHR
        trials = []
        for sign in (1.0, -1.0):
            try:
                trials.append(optimize(current.update_coords(np.asarray(current.coords) + sign * step), engine,
                                       **keywords))
            except (ConvergenceError, ElectronicStructureError):   # this direction failed; try the other
                continue
        if not trials:
            break
        energies(trials, engine)
        current = min(trials, key=lambda n: n.energy)
    raise WrongStationaryPointError(
        f"Minimization ended on a saddle point ({freqs.n_imaginary(imaginary_cutoff_cm)} imaginary modes below "
        f"-{imaginary_cutoff_cm:g} cm^-1; lowest {', '.join(f'{x:.0f}' for x in freqs.lowest())} cm^-1), "
        f"also after {escape_saddles} displacement(s) along the imaginary mode.", structure=current,
        frequencies=freqs)


def find_ts(guess: StructureNode, engine, *, check_frequencies: bool = True,
            imaginary_cutoff_cm: float = IMAGINARY_CUTOFF_CM, continue_unconverged: int = 1,
            **keywords: Any) -> StructureNode:
    """Optimize a transition-state guess to a first-order saddle point.

    `keywords` go to the engine's TS optimizer (for the ASE-based engines, Sella:
    e.g. `fmax` in eV/Angstrom, `maxiter`; the g-xTB engine also takes
    `exact_hessian` and `hessian_every`). `fmax` and `maxiter` default to
    `mepd ts`'s (`TSInputs`: 0.001 eV/Angstrom, 1000 steps), not the engine's
    looser optimizer default, at which soft modes read as extra imaginary ones. Raises ConvergenceError when the
    optimizer does not converge, and (`check_frequencies`, the default)
    WrongStationaryPointError when the converged structure does not have exactly
    one imaginary mode below -`imaginary_cutoff_cm` cm^-1. The checked
    frequencies are on the returned TS (`ts.frequencies`). An optimization that
    runs out of steps is continued from where it stopped (a fresh start of the
    optimizer, not of the search) up to `continue_unconverged` more times.

    >>> ts = mepd.find_ts(path.get_ts_node(), engine)
    """
    from mepd.errors import WrongStationaryPointError

    fn = _method(engine, "compute_transition_state", "transition-state optimization")
    if _takes_keywords(fn):
        from mepd.inputs import TSInputs

        keywords = {"fmax": TSInputs.fmax, "maxiter": TSInputs.maxiter, **keywords}
    current = guess
    for round_ in range(max(0, int(continue_unconverged)) + 1):
        try:
            ts = fn(node=current, **_kw(keywords))
            break
        except (ConvergenceError, UnsupportedOperation) as exc:
            last = getattr(exc, "structure", None)
            if round_ == continue_unconverged or last is None or isinstance(exc, UnsupportedOperation):
                raise
            current = last   # out of steps: go on from where it stopped
        except Exception as exc:
            raise ConvergenceError(f"Transition-state optimization failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(ts, StructureNode):
        raise ConvergenceError(
            f"Transition-state optimization ended on no usable structure ({type(ts).__name__}).")
    if check_frequencies:
        freqs = frequencies(ts, engine)
        n = freqs.n_imaginary(imaginary_cutoff_cm)
        if n != 1:
            raise WrongStationaryPointError(
                f"The converged TS has {n} imaginary modes below -{imaginary_cutoff_cm:g} cm^-1, not 1 "
                f"(lowest: {', '.join(f'{x:.0f}' for x in freqs.lowest())} cm^-1).", structure=ts, frequencies=freqs)
    return ts


@dataclass
class IRCResult:
    """An intrinsic reaction coordinate: `path` (a Chain from one minimum
    through the TS to the other); `ends` its two end structures. With
    minimized ends (the default of `mepd.irc`), `path` runs from the first
    minimum, through the IRC, to the second, and `irc_path` is the IRC alone."""
    path: Any
    irc_path: Any = None

    @property
    def ends(self) -> tuple[StructureNode, StructureNode]:
        return self.path.nodes[0], self.path.nodes[-1]

    @property
    def nodes(self) -> list[StructureNode]:
        return list(self.path.nodes)


def irc(ts: StructureNode, engine, *, minimize_ends: bool = True, end_keywords: dict | None = None,
        **keywords: Any) -> IRCResult:
    """Follow the IRC down both sides of a transition state.

    Uses the engine's own IRC (Sella for the ASE-based engines, including
    g-xTB), else geomeTRIC's. `keywords` go to it. With `minimize_ends` (the
    default) each end is then minimized to a true minimum (`mepd.minimize`,
    frequency-checked; `end_keywords` go to it): an IRC stops short of the
    minimum, about 2 kcal/mol above it typically and sometimes on the way to
    another species. Raises ConvergenceError if the IRC fails or an end cannot
    be minimized. `mepd.connects(result, reactant, product)` checks where it
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
    if not minimize_ends:
        return IRCResult(path=chain, irc_path=chain)
    ends = []
    for side, end in (("first", chain.nodes[0]), ("last", chain.nodes[-1])):
        try:
            ends.append(minimize(end, engine, **(end_keywords or {})))
        except ConvergenceError as exc:
            raise ConvergenceError(f"IRC done, but its {side} end could not be minimized: {exc}") from exc
    from mepd.chain import Chain

    path = Chain.model_validate({"nodes": [ends[0], *chain.nodes, ends[1]], "parameters": chain.parameters})
    return IRCResult(path=path, irc_path=chain)


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
