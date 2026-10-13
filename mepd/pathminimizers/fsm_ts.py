"""FSM-TS: the freezing string method with a TS-based elementary-step check,
and GSM (or NEB) when that check finds the TS does not connect the pair.

For one pair (reactant R, product P), as MSMEP hands it over:

1. FSM (FreezingNEB) grows a path; its own elementary-step check is off.
2. The path's highest image is optimized to a TS.
3. Pseudo-IRC: one Hessian at the TS, a push +/- along its imaginary mode,
   and a minimization of each pushed geometry.
4. The two minima decide, by molecular graph:
   - R and P: an elementary step (a leaf; the TS is kept on the result);
   - one end and a new species, or two new species: not elementary; MSMEP
     splits at the new species (R -> X -> P, or R -> X -> Y -> P);
5. ... and the TS is discarded and the pair is searched again with the
   fallback method -- GSM by default, or NEB; either with its own
   elementary-step checks, which may split it -- when the minima are
   both R or both P (the TS belongs to another motion, e.g. a conformer
   change), when the TS optimization fails, or when what it found is not a
   TS (no imaginary frequency, or more than one: a higher-order saddle) --
   and also when a minimum is a species of a pair MSMEP split above this
   one: that TS leads back up the tree, and splitting there would only
   revisit that pair (the same pair finds the same TS again).

FSM is quick to give a path whose highest point optimizes to a real TS;
the pseudo-IRC costs one Hessian and two minimizations, far less than an
IRC, and answers the same question MSMEP asks: does this one TS join R
and P?

GSM is the default fallback: on 20 KAIST reactions (playgroud
results/pathbench) it found the lowest channel more often than NEB (17 vs 14
within 1 kcal/mol of the best method; 1 vs 2 with none). Without a GSM
binary the fallback is NEB.
"""
from __future__ import annotations

import copy
import logging
import os
import shutil
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np

from mepd.chain import Chain
from mepd.elementarystep import (IS_ELEM_STEP, ElemStepResults, _comparison_preparer, _prepare_node_for_comparison,
                                 check_if_elem_step, elem_step_check_kwargs)
from mepd.engines.engine import Engine
from mepd.errors import NoneConvergedException
from mepd.helper_functions import RMSD
from mepd.nodes.node import StructureNode
from mepd.optimizers.optimizer import Optimizer
from mepd.pathminimizers.pathminimizer import PathMinimizer
from mepd.progress import print_persistent

logger = logging.getLogger(__name__)

# Keys the fallback (GSM or NEB) shares with the FSM-TS run (the rest are
# that method's own defaults, then `gsm_inputs` / `neb_inputs` overrides).
_SHARED_WITH_FALLBACK = ("skip_identical_graphs", "disregard_stereochem", "do_elem_step_checks",
                    "validate_minima_with_hessian", "hessian_minimum_frequency_cutoff",
                    "hessian_minima_rescue_displacement")


def fsm_ts_defaults() -> dict:
    """FSM's own settings plus the TS / pseudo-IRC / fallback knobs. The
    string is only a TS guess here (the TS optimization and pseudo-IRC do the
    rest), so it is coarse: 3 beads and 3 minimization steps per growth."""
    from mepd.inputs import fneb_defaults

    return {
        **fneb_defaults(),
        "min_images": 3,                      # beads: the path is grown in steps of d0 / (3 + 1)
        "max_min_iter": 3,                    # minimization steps per growth step
        "ts_maxiter": 500,                    # TS optimization steps (engines that take it)
        "pseudo_irc_amplitude": 0.3,          # bohr: how far the most-moved atom is pushed off the TS
        "pseudo_irc_maxiter": 500,            # each minimization
        "pseudo_irc_hessian": "auto",         # auto | engine | gradients (auto: gradients for g-xTB)
        "pseudo_irc_imaginary_cutoff": 20.0,  # cm^-1: below -this is an imaginary mode
        "fallback": "GSM",                    # GSM | NEB | none (none: keep the FSM path, FSM's usual check)
        "gsm_inputs": {},                     # overrides for a GSM fallback (GSM's path_min_inputs keys)
        "neb_inputs": {},                     # overrides for an NEB fallback (NEBInputs keys)
    }


@dataclass
class FSMTS(PathMinimizer):
    initial_chain: Chain
    engine: Engine
    optimizer: Optimizer
    parameters: SimpleNamespace = None
    gi_inputs: SimpleNamespace = None
    neb_optimizer: Optional[Optimizer] = None   # an NEB fallback's (a fresh one); default: a copy of `optimizer`
    chain_trajectory: list = field(default_factory=list)

    def __post_init__(self):
        from mepd.inputs import RunInputs

        if self.parameters is None or self.gi_inputs is None:
            ri = RunInputs(path_min_method="FSM-TS")
            self.parameters = self.parameters or ri.path_min_inputs
            self.gi_inputs = self.gi_inputs or ri.gi_inputs
        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0
        self.optimized = None
        self.path_method_used = "FSM"
        self.fallback_reason: Optional[str] = None
        self.ts_node: Optional[StructureNode] = None   # the optimized TS of an elementary FSM leaf
        self.pseudo_irc: Optional[dict] = None         # what the pseudo-IRC found (for the tree log)

    def _p(self, key: str):
        return getattr(self.parameters, key, fsm_ts_defaults().get(key))

    def _log(self, *parts, level: str = "info"):
        """A line kept in the run's log (stdout, and progress.log in the web
        app), not the transient status line: each FSM-TS step is part of the
        record of how a pair was decided."""
        if getattr(self.parameters, "verbosity", 1) >= 1:
            print_persistent(("FSM-TS: " if level != "warning" else "FSM-TS ⚠ ") + " ".join(str(p) for p in parts))

    # ------------------------------------------------------------------ run
    def optimize_chain(self) -> ElemStepResults:
        from mepd.pathminimizers.fneb import FreezingNEB

        fsm = FreezingNEB(initial_chain=self.initial_chain, engine=self.engine, optimizer=self.optimizer,
                          parameters=SimpleNamespace(**{**vars(self.parameters), "do_elem_step_checks": False}),
                          gi_inputs=self.gi_inputs)
        try:
            fsm.optimize_chain()
        except Exception as exc:   # FSM itself failed: NEB from the same interpolation
            self.chain_trajectory = list(fsm.chain_trajectory)
            return self._fallback(f"FSM failed ({type(exc).__name__}: {exc})"[:300])
        self.chain_trajectory = list(fsm.chain_trajectory)
        self.optimized = fsm.optimized or (self.chain_trajectory[-1] if self.chain_trajectory else None)
        self.grad_calls_made += fsm.grad_calls_made
        self.geom_grad_calls_made += fsm.geom_grad_calls_made
        chain = self.chain_trajectory[-1] if self.chain_trajectory else None
        # No interior image: FSM found no barrier to split on (as FNEB treats it).
        if not self._p("do_elem_step_checks") or chain is None or len(chain) <= 2:
            self._log("no TS check: " + ("do_elem_step_checks is off" if not self._p("do_elem_step_checks")
                                          else "FSM found no barrier between the ends") + "; kept as one step")
            return IS_ELEM_STEP
        return self._ts_and_pseudo_irc(chain)

    # ------------------------------------------------------------ TS + check
    def _ts_and_pseudo_irc(self, chain: Chain) -> ElemStepResults:
        guess = chain.get_ts_node()
        peak = (chain.energies.max() - chain.energies[0]) * 627.509474
        self._log(f"FSM path done ({len(chain)} images, peak {peak:.1f} kcal/mol); optimizing the TS from its "
                  f"highest image (step 2 of 4)")
        try:
            try:
                ts = self.engine.compute_transition_state(node=guess, keywords={"maxiter": int(self._p("ts_maxiter"))})
            except TypeError:
                ts = self.engine.compute_transition_state(node=guess)
        except Exception as exc:
            return self._fallback(f"TS optimization failed ({type(exc).__name__}: {exc})"[:300], chain)
        if not isinstance(ts, StructureNode):   # e.g. qccompute hands back its ProgramOutput
            return self._fallback("TS optimization did not converge", chain)
        ts = StructureNode(structure=ts.structure)
        try:
            self.engine.compute_energies([ts])
            modes = self._modes(ts)
        except Exception as exc:
            return self._fallback(f"Hessian at the TS failed ({type(exc).__name__}: {exc})"[:300], chain)
        cutoff = -abs(float(self._p("pseudo_irc_imaginary_cutoff")))
        n_imag = int(sum(1 for f in modes.freqs if f < cutoff))
        info = {"ts_energy": float(ts.energy), "imaginary_freq": float(modes.freqs[0]), "n_imaginary": n_imag}
        self._log(f"TS converged (E = {float(ts.energy):.6f} Eh); Hessian: {n_imag} imaginary mode(s), lowest "
                  f"{modes.freqs[0]:.0f} cm^-1 (step 3 of 4)")
        if n_imag == 0:
            self.pseudo_irc = info
            return self._fallback(f"the TS optimization ended at a minimum (lowest frequency "
                                  f"{modes.freqs[0]:.0f} cm^-1)", chain)
        if n_imag > 1:   # a higher-order saddle, not a TS (a proper TS has exactly one imaginary mode)
            self.pseudo_irc = info
            imag = ", ".join(f"{f:.0f}" for f in modes.freqs[:n_imag])
            return self._fallback(f"the TS optimization ended at a saddle of order {n_imag} "
                                  f"({imag} cm^-1), not a TS", chain)

        from mepd.discovery.vri import _optimize_nodes, push_along_mode

        pushed = push_along_mode(ts, modes.lowest_mode_cart, float(self._p("pseudo_irc_amplitude")))
        self._log(f"pseudo-IRC: pushed {self._p('pseudo_irc_amplitude')} bohr both ways along that mode; "
                  f"minimizing both sides")
        ends = _optimize_nodes(self.engine, pushed, {"coordsys": "cart", "maxiter": int(self._p("pseudo_irc_maxiter"))})
        if any(e is None for e in ends):
            self.pseudo_irc = info
            return self._fallback("a pseudo-IRC minimization failed", chain)
        ends = [StructureNode(structure=e.structure) for e in ends]
        try:
            self.engine.compute_energies(ends)
        except Exception:
            pass
        verdict, labels = self._classify(chain[0], chain[-1], ends, self._ancestor_species())
        self.pseudo_irc = {**info, "ends": labels, "verdict": verdict}
        self._log(f"pseudo-IRC ends: {labels[0]} / {labels[1]} -> "
                  + {"elementary": "an elementary step (step 4 of 4: no split)",
                     "split": "not elementary: splitting at the new species (step 4 of 4)",
                     "not connected": "this TS does not join the pair"}[verdict])
        if verdict == "elementary":
            self.ts_node = ts
            self._end_on_ts(chain, ts)
            r_side, p_side = (ends if labels[0] == "reactant" else ends[::-1])
            return ElemStepResults(is_elem_step=True, is_concave=True, splitting_criterion=None,
                                   minimization_results=[r_side, p_side], number_grad_calls=0)
        if verdict == "split":
            new = self._order_new(chain[0], chain[-1], [e for e, lab in zip(ends, labels) if lab == "new"])
            return ElemStepResults(is_elem_step=False, is_concave=True, splitting_criterion="maxima",
                                   minimization_results=new, number_grad_calls=0, new_structures=list(new))
        if "earlier" in labels:
            return self._fallback("pseudo-IRC: the TS leads back to a species this branch came from (an end of a "
                                  "pair split above), so it does not help this pair", chain)
        side = "reactant" if labels[0] == "reactant" else "product"
        return self._fallback(f"pseudo-IRC: both sides of the TS minimize to the {side}, so this TS does not "
                              f"join the pair", chain)

    def _end_on_ts(self, chain: Chain, ts: StructureNode) -> None:
        """The result's last chain becomes [reactant, converged TS, product]
        (the FSM path stays in the history before it). Everything that takes
        a path's highest image as its TS -- `mepd run --use-tsopt`, channels,
        the web results -- then starts from the TS the pseudo-IRC verified."""
        try:
            self.engine.compute_gradients([ts])   # as the path's other images have
        except Exception:
            pass
        final = Chain.model_validate({"nodes": [chain[0].copy(), ts, chain[-1].copy()],
                                      "parameters": chain.parameters})
        self.chain_trajectory.append(final)
        self.optimized = final

    def _modes(self, ts: StructureNode):
        """Projected normal modes at the TS (rigid-body motion removed)."""
        from mepd.discovery import vri

        how = str(self._p("pseudo_irc_hessian") or "auto").lower()
        engine = self.engine
        if how == "gradients" or (how == "auto" and type(engine).__name__ == "GXTBCalculator"):
            # g-xTB's own Hessian disagrees with its gradients (see vri.GradientHessianEngine).
            engine = vri.GradientHessianEngine(engine)
        return vri.stationary_point_modes(ts, engine)

    # -------------------------------------------------------------- verdict
    def _same(self, a, b) -> bool:
        """Same species (molecular graph); geometry when graphs are off."""
        from mepd.nodes.nodehelpers import _is_connectivity_identical, is_identical

        prepare = _comparison_preparer(self.engine)
        a, b = _prepare_node_for_comparison(a, prepare), _prepare_node_for_comparison(b, prepare)
        if getattr(a, "graph", None) is not None and getattr(b, "graph", None) is not None:
            return bool(_is_connectivity_identical(a, b, verbose=False, collect_comparison=False,
                                                   disregard_stereochem=bool(self._p("disregard_stereochem"))))
        params = self.initial_chain.parameters
        return bool(is_identical(a, b, fragment_rmsd_cutoff=params.node_rms_thre, kcal_mol_cutoff=params.node_ene_thre,
                                 verbose=False, collect_comparison=False))

    def _ancestor_species(self) -> list:
        """The ends of the pairs MSMEP split above this one (set by MSMEP on the chain)."""
        pairs = getattr(self.initial_chain, "_split_ancestors", None) or []
        return [StructureNode(structure=n.structure) for pair in pairs for n in pair]

    def _classify(self, r, p, ends, earlier: Optional[list] = None) -> tuple[str, list[str]]:
        """('elementary' | 'split' | 'not connected', [label of each end]).
        A minimum that is neither end but a species of a pair split above
        (`earlier`) is labelled "earlier": the TS leads back up the tree, so
        splitting there would only revisit that pair (FSM-TS is
        deterministic: the same pair finds the same TS again)."""
        r = StructureNode(structure=r.structure)
        p = StructureNode(structure=p.structure)
        labels = []
        for e in ends:
            labels.append("reactant" if self._same(e, r) else "product" if self._same(e, p)
                          else "earlier" if any(self._same(e, a) for a in earlier or []) else "new")
        if sorted(labels) == ["product", "reactant"]:
            return "elementary", labels
        if "earlier" in labels or (labels[0] == labels[1] and labels[0] != "new"):
            return "not connected", labels
        if labels == ["new", "new"] and self._same(ends[0], ends[1]):
            labels = ["new", "new (same species)"]   # both sides one new species: split there once
        return "split", labels

    @staticmethod
    def _order_new(r, p, new: list) -> list:
        """New minima in path order: the one nearer the reactant first."""
        if len(new) < 2:
            return new
        a, b = new
        d = lambda x, y: RMSD(np.asarray(x.coords), np.asarray(y.coords))[0]
        return [a, b] if d(a, r) + d(b, p) <= d(b, r) + d(a, p) else [b, a]

    # ------------------------------------------------------------- fallback
    def _fallback_method(self) -> Optional[str]:
        """"GSM", "NEB" or None (no fallback). `neb_fallback = false`, the
        older switch, still turns it off; GSM without a binary is NEB."""
        if self._p("neb_fallback") is False:
            return None
        method = str(self._p("fallback") or "none").upper()
        if method in ("NONE", "FALSE", "OFF", ""):
            return None
        if method not in ("GSM", "NEB"):
            raise ValueError(f"FSM-TS fallback must be GSM, NEB or none, not {self._p('fallback')!r}")
        if method == "GSM" and not self._gsm_executable():
            self._log("no GSM binary (set gsm_inputs.executable or $GSM_EXECUTABLE); the fallback is NEB",
                      level="warning")
            return "NEB"
        return method

    def _gsm_executable(self) -> Optional[str]:
        exe = (dict(self._p("gsm_inputs") or {}).get("executable") or os.getenv("GSM_EXECUTABLE") or "gsm")
        return shutil.which(str(exe))

    def _fallback(self, reason: str, fsm_chain: Optional[Chain] = None) -> ElemStepResults:
        """Discard the TS; search the pair with GSM or NEB (or, with the
        fallback off, keep the FSM path with FSM's usual elementary-step check)."""
        method = self._fallback_method()
        self._log(f"{reason}; " + (f"searching this pair again with {method}" if method else "keeping the FSM path"),
                  level="warning")
        self.fallback_reason = reason
        if method is None:
            if fsm_chain is None:
                raise NoneConvergedException(trajectory=self.chain_trajectory, msg=reason, obj=self)
            short = Chain.model_validate({"nodes": [fsm_chain[0], fsm_chain.get_ts_node(), fsm_chain[-1]],
                                          "parameters": fsm_chain.parameters})
            return check_if_elem_step(short, engine=self.engine, **elem_step_check_kwargs(self.parameters))
        self.ts_node = None
        if method == "GSM":
            try:
                return self._run_fallback("GSM", self._gsm())
            except NoneConvergedException:
                raise
            except Exception as exc:   # the binary failed: NEB still gets the pair
                self._log(f"GSM failed ({type(exc).__name__}: {exc})"[:300] + "; searching with NEB", level="warning")
                self.fallback_reason = f"{reason}; GSM failed ({type(exc).__name__})"
        return self._run_fallback("NEB", self._neb())

    def _shared(self) -> dict:
        return {k: getattr(self.parameters, k) for k in _SHARED_WITH_FALLBACK if hasattr(self.parameters, k)}

    def _gsm(self):
        from mepd.inputs import RunInputs
        from mepd.pathminimizers.gsm import GSM

        own = vars(RunInputs(path_min_method="GSM").path_min_inputs)
        params = SimpleNamespace(**{**own, **self._shared(), **dict(self._p("gsm_inputs") or {})})
        return GSM(initial_chain=self.initial_chain, engine=self.engine, parameters=params, gi_inputs=self.gi_inputs)

    def _neb(self):
        from mepd.inputs import NEBInputs
        from mepd.neb import NEB

        params = SimpleNamespace(**{**NEBInputs().__dict__, **self._shared(), **dict(self._p("neb_inputs") or {})})
        return NEB(initial_chain=self.initial_chain, parameters=params,
                   optimizer=self.neb_optimizer or copy.deepcopy(self.optimizer), engine=self.engine)

    def _run_fallback(self, method: str, minimizer) -> ElemStepResults:
        self.path_method_used = method
        try:
            result = minimizer.optimize_chain()
        finally:   # MSMEP reads the trajectory, also when the fallback did not converge
            self.chain_trajectory = list(getattr(minimizer, "chain_trajectory", None) or []) or self.chain_trajectory
            self.optimized = getattr(minimizer, "optimized", None) or (self.chain_trajectory[-1] if self.chain_trajectory else None)
            self.grad_calls_made += getattr(minimizer, "grad_calls_made", 0)
            self.geom_grad_calls_made += getattr(minimizer, "geom_grad_calls_made", 0)
        return result
