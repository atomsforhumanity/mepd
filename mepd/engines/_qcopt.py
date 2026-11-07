"""Minimizations, transition-state searches and IRCs with qcopt, AFH's optimizer.

qcopt (github.com/atomsforhumanity/qcopt) takes its gradients and Hessians from the mepd
engine here: each step is one `engine.compute_gradients` call on a one-node list, and
exact Hessians come from `engine.compute_hessian`, so the energies are exactly the
engine's (its backend, SCF retries and settings included).

On 100 Reaction-QM reactions at g-xTB 2.0.1 (axtb; qcopt's benchmarks, findings artifact
https://claude.ai/artifact/QZwzfWYDatwNXmXp3onsjp):

- TS searches from the dataset's guesses: 100/100 first-order saddles, median 12
  gradients, against Sella's 16 (and 28.4 vs. 15.9 on average), the same saddle in 96.
- Minimizations of the 200 reactants and products: 196 true minima, median 33
  gradients, against Sella's 143 with its default (no exact Hessian), 5x less wall time.
- IRCs: the same minima as Sella's IRC in 100/100, with 375 gradients including both
  end minimizations against Sella's 2,407.

Keywords use mepd's conventions (`fmax` in eV/Angstrom, `maxiter`); any other keyword
is qcopt's own (`qcopt.OptimizationKeywords`, `qcopt.IRCKeywords`), and an unknown one
raises.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qcref import constants as c

from mepd.errors import ElectronicStructureError, GeometryOptimizationNotConvergedError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache

# eV/Angstrom -> Hartree/bohr
_FORCE = float(c.EV_TO_HARTREE) / float(c.ANGSTROM_TO_BOHR)
# Sella IRC options with no qcopt counterpart: `keep_going` steps past Sella's own
# corrector failures (qcopt has no inner loop to fail); the others tune that corrector.
_SELLA_IRC_IGNORED = ("keep_going",)
# qcopt (>= 0.2) fails an IRC whose branches stop before their minima: out of steps, or
# stalled or overshooting with the gradient above its end criterion. mepd minimizes the
# ends afterwards, so such a path is used; a failed calculation, or a branch that never
# went downhill, still raises.
_IRC_UNFINISHED = ("max_steps", "energy rose", "stalled")
log = logging.getLogger(__name__)
# geomeTRIC's coordinate choice, which mepd's CLI adds to every minimization ("cart");
# Sella ignores it too. qcopt's own is `coordinates` (TRIC by default).
_GEOMETRIC_IGNORED = ("coordsys",)
_SELLA_ONLY = ("eta", "gamma", "optimizer_kwds", "optimizer_kwargs", "order", "diag_every_n")


def _qcopt():
    try:
        import qcopt
    except ImportError as exc:
        raise ElectronicStructureError(
            msg="optimizer \"qcopt\" needs the qcopt package (`pip install qcopt`; "
                "github.com/atomsforhumanity/qcopt).") from exc
    return qcopt


def _pop_steps(kwds: dict[str, Any], default: int) -> int:
    return int(kwds.pop("maxiter", kwds.pop("maxit", kwds.pop("steps", default))))


def _reject_sella_options(kwds: dict[str, Any], what: str) -> None:
    sella_only = sorted(k for k in kwds if k in _SELLA_ONLY)
    if sella_only:
        raise ValueError(f"{', '.join(sella_only)}: Sella options; the qcopt {what} does not take them "
                         "(qcopt's own keywords are in qcopt.OptimizationKeywords / qcopt.IRCKeywords).")


def _evaluator(engine, node: StructureNode, *, hessian: bool, sink=None):
    """qcopt's view of the engine: energy and gradient (and the Hessian) at bohr positions."""
    qcopt = _qcopt()
    energies: list[float] = []
    frames: list[str] = []

    def point(x: np.ndarray) -> tuple[float, np.ndarray]:
        here = node.update_coords(np.asarray(x, dtype=float).reshape(-1, 3))
        gradient = np.asarray(engine.compute_gradients([here])[0], dtype=float)
        energy = float(here.energy)
        if sink is not None:
            try:
                energies.append(energy)
                frames.append(here.structure.to_xyz())
                sink(energies, frames)
            except Exception:
                pass  # the live view must never break the optimization
        return energy, gradient

    def exact_hessian(x: np.ndarray) -> np.ndarray:
        here = node.update_coords(np.asarray(x, dtype=float).reshape(-1, 3))
        return np.asarray(engine.compute_hessian(here), dtype=float)

    return qcopt.FunctionEvaluator(point, exact_hessian if hessian else None, name=type(engine).__name__)


def _nodes(frames, node: StructureNode) -> list[StructureNode]:
    """The frames as StructureNodes holding their energies and gradients."""
    nodes, results = [], []
    for frame in frames:
        if not frame.success or frame.results.energy is None or frame.results.gradient is None:
            continue
        nodes.append(node.update_coords(np.asarray(frame.structure.geometry, dtype=float)))
        res = FakeQCIOResults.model_validate({"energy": float(frame.results.energy),
                                              "gradient": np.asarray(frame.results.gradient, dtype=float)})
        results.append(FakeQCIOOutput.model_validate({"results": res}))
    update_node_cache(node_list=nodes, results=results)
    return nodes


def _finish(out, node: StructureNode, what: str, maxiter: int, fmax: float) -> list[StructureNode]:
    """The trajectory of a converged run; raise if it failed or did not converge."""
    nodes = _nodes(out.results.frames, node)
    if out.success:
        if not nodes:
            raise ElectronicStructureError(msg=f"qcopt {what} returned no structures.")
        return nodes
    reason = (out.traceback or "").strip()
    if reason.startswith("not converged") and nodes:
        largest = float(np.linalg.norm(np.asarray(nodes[-1].gradient, dtype=float), axis=1).max()) / _FORCE
        err = GeometryOptimizationNotConvergedError(
            msg=(f"qcopt {what} did not converge in {maxiter} steps: largest force {largest:.2e} "
                 f"eV/Angstrom, target {fmax:.2e}. Refusing to return an unconverged structure."))
        # where it stopped, so a caller can continue from there instead of from the start
        err.structure = nodes[-1]
        err.steps = int(maxiter)
        raise err
    raise ElectronicStructureError(msg=f"qcopt {what} failed: {reason.splitlines()[-1] if reason else 'no reason'}",
                                   obj=reason)


def minimize(engine, node: StructureNode, *, fmax: float, maxiter: int, hessian_every: int | None = None,
             **keywords: Any) -> list[StructureNode]:
    """Minimize `node` to a largest atomic force below `fmax` (eV/Angstrom).

    The starting Hessian is Lindh's model: on the 200 benchmark minimizations an exact
    g-xTB Hessian (25-40 gradients' worth of time) did not pay for itself.
    `hessian_every` > 0 starts from the engine's exact Hessian and renews it every
    that many steps instead. Returns the trajectory (accepted steps); raises
    GeometryOptimizationNotConvergedError (with `.structure`) when out of steps.
    """
    from mepd import progress as _progress

    qcopt = _qcopt()
    keywords = {k: v for k, v in keywords.items() if k not in _GEOMETRIC_IGNORED}
    _reject_sella_options(keywords, "minimizer")
    exact = bool(hessian_every)
    if exact:
        keywords = {"initial_hessian": "exact", "hessian_every": int(hessian_every), **keywords}
    evaluator = _evaluator(engine, node, hessian=exact, sink=_progress.minimization_sink())
    try:
        out = qcopt.minimize(node.structure, {"program": type(engine).__name__}, hessian=exact or None,
                             evaluator=evaluator, maxiter=int(maxiter),
                             convergence={"gmax": float(fmax) * _FORCE}, **keywords)
    except ValueError as exc:   # qcopt's keyword validation
        raise ValueError(f"qcopt minimizer: {exc}") from exc
    return _finish(out, node, "minimization", maxiter, fmax)


def transition_state(engine, node: StructureNode, *, fmax: float, maxiter: int, exact_hessian: bool = True,
                     hessian_every: int | None = None, **keywords: Any) -> list[StructureNode]:
    """Optimize a transition-state guess to a first-order saddle point (largest atomic
    force below `fmax`, eV/Angstrom).

    `exact_hessian` starts from the engine's Hessian, recomputed when the updated one
    loses its negative mode and every `hessian_every` steps (0/None: only then);
    without it the starting Hessian comes from finite differences of gradients.
    Returns the trajectory; its last structure is the TS.
    """
    qcopt = _qcopt()
    keywords = {k: v for k, v in keywords.items() if k not in _GEOMETRIC_IGNORED}
    _reject_sella_options(keywords, "TS search")
    if hessian_every:
        keywords = {"hessian_every": int(hessian_every), **keywords}
    exact = bool(exact_hessian) and hasattr(engine, "compute_hessian")
    evaluator = _evaluator(engine, node, hessian=exact)
    try:
        out = qcopt.find_ts(node.structure, {"program": type(engine).__name__}, hessian=exact or None,
                            evaluator=evaluator, maxiter=int(maxiter),
                            convergence={"gmax": float(fmax) * _FORCE}, **keywords)
    except ValueError as exc:
        raise ValueError(f"qcopt TS search: {exc}") from exc
    return _finish(out, node, "TS search", maxiter, fmax)


def irc(engine, ts_node: StructureNode, keywords: dict[str, Any] | None = None):
    """Follow the IRC down both sides of `ts_node`; a Chain from one end to the other.

    mepd's keywords: `fmax` (eV/Angstrom; a branch ends below it, default qcopt's
    5e-4 Hartree/bohr = 0.026 eV/Angstrom), `maxiter`/`steps` (per branch) and `dx`
    (Sella's step, Angstrom amu^1/2; becomes qcopt's `step_size` in bohr amu^1/2).
    The ends are not minimized here (`mepd.irc` does that).
    """
    from mepd.chain import Chain

    qcopt = _qcopt()
    kwds = dict(keywords or {})
    for key in _SELLA_IRC_IGNORED + _GEOMETRIC_IGNORED:
        kwds.pop(key, None)
    _reject_sella_options(kwds, "IRC")
    if "fmax" in kwds:
        kwds["gmax"] = float(kwds.pop("fmax")) * _FORCE
    if any(k in kwds for k in ("maxiter", "maxit", "steps")):
        kwds["max_steps"] = _pop_steps(kwds, 300)
    if "dx" in kwds:
        kwds["step_size"] = float(kwds.pop("dx")) * float(c.ANGSTROM_TO_BOHR)
    exact = hasattr(engine, "compute_hessian")
    evaluator = _evaluator(engine, ts_node, hessian=exact)
    try:
        out = qcopt.irc(ts_node.structure, {"program": type(engine).__name__}, hessian=exact or None,
                        evaluator=evaluator, **kwds)
    except ValueError as exc:
        raise ValueError(f"qcopt IRC: {exc}") from exc
    if not out.success:
        reason = (out.traceback or "").strip()
        ends = (out.results.extras or {}).get("branch_end") or {}
        unfinished_only = (
            bool(ends) and out.results.ts_index is not None and "downhill" not in reason
            and all((r or "").startswith("converged") or r in _IRC_UNFINISHED for r in ends.values()))
        if not unfinished_only:
            raise ElectronicStructureError(msg=f"qcopt IRC failed: {reason.splitlines()[-1] if reason else ''}",
                                           obj=reason)
        log.warning("qcopt IRC did not reach both minima (%s); using its path, whose ends mepd minimizes",
                    reason.splitlines()[-1])
    frames = out.results.frames
    k = int(out.results.ts_index)
    before, after = _nodes(frames[:k], ts_node), _nodes(frames[k + 1:], ts_node)
    if not before and not after:
        raise ElectronicStructureError(msg="qcopt IRC produced no structures off the transition state.")
    return Chain.model_validate({"nodes": [*before, ts_node.copy(), *after]})
