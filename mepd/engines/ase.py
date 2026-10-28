import copy
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Hartree
from numpy.typing import NDArray
from qcref.constants import ANGSTROM_TO_BOHR
from mepd.qcdata_structure_helpers import (
    structure_to_ase_atoms,
    ase_atoms_to_structure,
)

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.errors import (
    EnergiesNotComputedError,
    GradientsNotComputedError,
    ElectronicStructureError,
    GeometryOptimizationNotConvergedError,
)
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache

from ase.optimize.optimize import Optimizer
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.bfgs import BFGS
from ase.optimize.fire import FIRE
from ase.optimize.mdmin import MDMin
try:
    from mepd.engines._sella import sella as _sella

    _sella_module = _sella()          # without torch on a machine with no GPU
    SellaOptimizer, SellaIRC = _sella_module.Sella, _sella_module.IRC
except Exception:
    SellaOptimizer = None
    SellaIRC = None



from ase.io import Trajectory

from pathlib import Path


def _rigid_basis(positions: NDArray) -> NDArray:
    """Orthonormal columns spanning the rigid translations and rotations of
    `positions` (n_atoms x 3): 6 columns, 5 for a linear molecule."""
    centered = positions - positions.mean(axis=0)
    vecs = [np.tile(np.eye(3)[k], len(positions)) for k in range(3)]
    vecs += [np.cross(np.eye(3)[k], centered).ravel() for k in range(3)]
    u, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return u[:, s > 1e-6 * s.max()]


def _sella_irc_class(base):
    """Sella's IRC, ending at the minimum. Sella stops a direction only once
    its Hessian's lowest eigenvalue is positive, but that Hessian keeps
    translations and rotations (IRC runs with proj_rot=False), whose
    near-zero eigenvalues noise can push below zero: at the minimum it then
    keeps stepping along a rotation, a rigid spin of the whole system, until
    `steps` runs out. Here the curvature is judged on vibrations only. With
    atoms fixed rotations are real motions, so Sella's own test stands."""

    class _IRC(base):
        def converged(self, forces=None):
            if self.first or self.atoms.constraints:
                return super().converged(forces)
            if self.pes.H.evals is None or not self.pes.converged(self.fmax)[0]:
                return False
            basis = _rigid_basis(self.atoms.get_positions())
            u, _, _ = np.linalg.svd(basis, full_matrices=True)
            vib = u[:, basis.shape[1]:]
            hessian = np.asarray(self.pes.H.asarray(), dtype=float)
            return bool(np.linalg.eigvalsh(vib.T @ hessian @ vib)[0] > 0)

    return _IRC


def _align_frames(frames: list[NDArray], gradients: list[NDArray], reference: NDArray):
    """Strip rigid motion from an IRC branch: each frame is rotated and
    translated onto the one before it (the first onto `reference`), its
    gradient rotated with it. Returns the new (frames, gradients)."""
    prev = np.asarray(reference, dtype=float)
    out_frames, out_grads = [], []
    for coords, grad in zip(frames, gradients):
        coords = np.asarray(coords, dtype=float)
        mu, mu_prev = coords.mean(axis=0), prev.mean(axis=0)
        u, _, vt = np.linalg.svd((coords - mu).T @ (prev - mu_prev))
        rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        prev = (coords - mu) @ rot + mu_prev
        out_frames.append(prev)
        out_grads.append(np.asarray(grad, dtype=float) @ rot)
    return out_frames, out_grads
import tempfile

#: Largest atomic force (eV/Angstrom) at which minimizations and TS
#: optimizations count as converged unless told otherwise. 0.001: at the
#: former 0.01, g-xTB TSs could carry a spurious second imaginary mode and
#: lie up to 34 kcal/mol off the saddle they belong to (KAIST 500 benchmark,
#: 2026-10-19; CHANGELOG).
DEFAULT_FMAX = 0.001

AVAIL_OPTS = {
    "LBFGS": LBFGS,
    "BFGS": BFGS,
    "FIRE": FIRE,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
}


# Per-thread calculator copies (see `ASEEngine._calculator_in_use`), keyed by
# id(engine) with a weak reference to guard against id reuse.
_THREAD_CALCULATORS = threading.local()
_ENGINE_LOCKS: dict = {}  # id(engine) -> (weakref to engine, RLock); engines are unhashable dataclasses
_ENGINE_LOCKS_GUARD = threading.Lock()


def _engine_lock(engine) -> threading.RLock:
    with _ENGINE_LOCKS_GUARD:
        entry = _ENGINE_LOCKS.get(id(engine))
        if entry is None or entry[0]() is not engine:
            for key in [k for k, (ref, _) in _ENGINE_LOCKS.items() if ref() is None]:
                del _ENGINE_LOCKS[key]
            entry = _ENGINE_LOCKS[id(engine)] = (weakref.ref(engine), threading.RLock())
        return entry[1]


@dataclass
class ASEEngine(Engine):
    """
    !!! Warning:
    ASE uses the following standard units:
        - energy (eV)
        - positions (Angstroms)

        Neb-dynamics uses Hartree and Bohr for coordinates.
        Appropriate conversions must me made.
    """

    calculator: Calculator
    ase_optimizer: Optimizer = None
    geometry_optimizer: str = "LBFGSLineSearch"
    transition_state_optimizer: str = "SELLA"
    # Atoms held in place (ase FixAtoms) in optimizations, TS searches and IRCs.
    fixed_atoms: list | None = None
    # The IRC's two branches at once, each in a thread with its own copy of
    # the calculator (the same path, in about half the time). For calculators
    # that are cheap to copy and run outside Python (g-xTB's subprocess);
    # not for an in-process model, which a copy would duplicate in memory.
    irc_branches_at_once: bool = False

    def __post_init__(self):
        if self.ase_optimizer is None:
            assert (
                self.geometry_optimizer is not None and self.geometry_optimizer in AVAIL_OPTS.keys()
            ), f"Must input either an ase optimizer or a string name for an\
             available optimizer: {AVAIL_OPTS.keys()}"

            self.ase_optimizer = AVAIL_OPTS[self.geometry_optimizer]

    def _constrain(self, atoms: Atoms) -> Atoms:
        if self.fixed_atoms:
            from ase.constraints import FixAtoms

            atoms.set_constraint(FixAtoms(indices=[i for i in self.fixed_atoms if i < len(atoms)]))
        return atoms

    @contextmanager
    def _calculator_in_use(self):
        """The calculator for this thread. An ASE calculator keeps the last
        structure and its results, so two threads using one at once can read
        each other's energies/forces. Worker threads (mepd runs engine calls
        in thread pools) each get their own deep copy; the main thread uses
        the original. A calculator that cannot be copied is shared under a
        lock instead (correct, but serialized). Subclasses whose `calculator`
        is a property building a fresh calculator per access (FAIRChemEngine)
        are used as is."""
        if isinstance(getattr(type(self), "calculator", None), property):
            yield self.calculator
            return
        base = self.calculator
        calc = base
        if threading.current_thread() is not threading.main_thread():
            cache = _THREAD_CALCULATORS.__dict__.setdefault("by_engine", {})
            entry = cache.get(id(self))
            if entry is None or entry[0]() is not self or entry[1] is not base:
                try:
                    own = copy.deepcopy(base)
                    if hasattr(own, "reset"):
                        own.reset()
                except Exception:
                    own = None
                entry = cache[id(self)] = (weakref.ref(self), base, own)
            if entry[2] is not None:
                yield entry[2]
                return
        with _engine_lock(self):
            yield calc

    def _extract_optimizer_run_kwargs(
        self,
        keywords: dict[str, Any] | None,
        *,
        default_steps: int = 500,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        kwds = dict(keywords or {})
        fmax = float(kwds.pop("fmax", DEFAULT_FMAX))
        maxiter = kwds.pop("maxiter", kwds.pop("maxit", kwds.pop("steps", default_steps)))
        run_kwds = {"fmax": fmax, "steps": int(maxiter)}

        optimizer_kwds = kwds.pop("optimizer_kwds", kwds.pop("optimizer_kwargs", {}))
        if optimizer_kwds is None:
            optimizer_kwds = {}
        elif not isinstance(optimizer_kwds, dict):
            raise ValueError("`optimizer_kwds` must be a dictionary if provided.")
        return run_kwds, dict(optimizer_kwds)

    def _run_ase_optimization(
        self,
        node: StructureNode,
        optimizer_cls: type[Any],
        keywords: dict[str, Any] | None = None,
    ) -> list[StructureNode]:
        run_kwds, optimizer_kwds = self._extract_optimizer_run_kwargs(keywords)

        atoms = self._constrain(structure_to_ase_atoms(node.structure))
        tmp = tempfile.NamedTemporaryFile(suffix=".traj", mode="w+", delete=False)
        with self._calculator_in_use() as calc:
            atoms.calc = calc
            optimizer = optimizer_cls(
                atoms=atoms,
                logfile=None,
                trajectory=tmp.name,
                **optimizer_kwds,
            )  # ASE does geometry updates in-place.
            from mepd import progress as _progress

            sink = _progress.minimization_sink()
            if sink is not None:
                live_energies: list[float] = []
                live_frames: list[str] = []

                def _report_step() -> None:
                    try:
                        live_energies.append(atoms.get_potential_energy() / Hartree)
                        live_frames.append(ase_atoms_to_structure(
                            atoms=atoms, charge=node.structure.charge,
                            multiplicity=node.structure.multiplicity).to_xyz())
                        sink(live_energies, live_frames)
                    except Exception:
                        pass  # the live view must never break the optimization

                optimizer.attach(_report_step, interval=1)
            try:
                converged = optimizer.run(**run_kwds)
            except Exception as exc:
                # Keep the cause: "Electronic structure failed." alone hid why
                # (e.g. an xtb solvent run that stopped).
                raise ElectronicStructureError(msg=f"Electronic structure failed: {exc}", obj=exc)
            # ASE returns the last frame whether or not it converged; a run that
            # used up its steps is not a minimum (or a TS) and must not pass as one.
            if not converged:
                largest = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
                Path(tmp.name).unlink(missing_ok=True)
                err = GeometryOptimizationNotConvergedError(
                    msg=(f"{optimizer_cls.__name__} did not converge in {run_kwds['steps']} steps: largest "
                         f"force {largest:.2e} eV/Angstrom, target {run_kwds['fmax']:.2e}. Refusing to "
                         "return an unconverged structure."))
                # where it stopped, so a caller can continue from there instead of from the start
                err.structure = StructureNode(structure=ase_atoms_to_structure(
                    atoms=atoms, charge=node.structure.charge, multiplicity=node.structure.multiplicity))
                err.steps = int(run_kwds["steps"])
                raise err

        charge = node.structure.charge
        multiplicity = node.structure.multiplicity

        aT = Trajectory(tmp.name)
        traj_list = []
        for i, _ in enumerate(aT):
            traj_list.append(
                ase_atoms_to_structure(
                    atoms=aT[i],
                    charge=charge,
                    multiplicity=multiplicity,
                )
            )

        energies = [obj.get_potential_energy() / Hartree for obj in aT]
        # ASE forces are -dE/dx in eV/Angstrom; convert to +dE/dx in Hartree/Bohr.
        gradients = [(-1 * obj.get_forces() / ANGSTROM_TO_BOHR) / Hartree for obj in aT]
        all_results = []
        for ene, grad in zip(energies, gradients):
            res = FakeQCIOResults.model_validate({"energy": ene, "gradient": grad})
            out = FakeQCIOOutput.model_validate({"results": res})
            all_results.append(out)
        Path(tmp.name).unlink()
        node_list = [StructureNode(structure=struct) for struct in traj_list]
        update_node_cache(node_list=node_list, results=all_results)
        return node_list

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        try:
            grads = np.array([node.gradient for node in chain])
        except GradientsNotComputedError:
            node_list = self._run_calc(chain=chain, calctype="gradient")
            grads = np.array([node.gradient for node in node_list])
        return grads

    def compute_energies(self, chain: Chain) -> NDArray:
        try:
            enes = np.array([node.energy for node in chain])
        except EnergiesNotComputedError:
            node_list = self._run_calc(chain=chain, calctype="energy")
            enes = np.array([node.energy for node in node_list])

        return enes

    def _run_calc(
        self, calctype: str, chain: Union[Chain, List]
    ) -> List[StructureNode]:
        if isinstance(chain, Chain):
            assert isinstance(
                chain.nodes[0], StructureNode
            ), "input Chain has nodes incompatible with QCComputeEngine."
            node_list = chain.nodes
        elif isinstance(chain, list):
            assert isinstance(
                chain[0], StructureNode
            ), "input list has nodes incompatible with QCComputeEngine."
            node_list = chain
        else:
            raise ValueError(
                f"Input needs to be a Chain or a List. You input a: {type(chain)}"
            )

        # now create program inputs for each geometry that is not frozen or already computes
        inds_frozen = [
            i for i, node in enumerate(node_list) if node._cached_energy is not None
        ]
        all_ase_atoms = [structure_to_ase_atoms(
            node.structure) for node in node_list]
        non_frozen_ase_atoms = [
            atoms for i, atoms in enumerate(all_ase_atoms) if i not in inds_frozen
        ]
        non_frozen_results = [
            self.compute_func(atoms=atoms) for atoms in non_frozen_ase_atoms
        ]

        # merge the results
        all_results = []
        for i, node in enumerate(node_list):
            if i in inds_frozen:
                all_results.append(node_list[i]._cached_result)
            else:
                all_results.append(non_frozen_results.pop(0))
        update_node_cache(node_list=node_list, results=all_results)

        return node_list

    def compute_func(self, atoms: Atoms):
        try:
            with self._calculator_in_use() as calc:
                ene_ev = calc.get_potential_energy(atoms=atoms)  # eV
                # ASE outputs the negative gradient
                grad_ev_ang = calc.get_forces(atoms=atoms) * (-1)  # eV / Angstroms
            ene = ene_ev / Hartree  # Hartree
            grad = (grad_ev_ang / ANGSTROM_TO_BOHR) / Hartree  # Hartree / Bohr
            res = FakeQCIOResults.model_validate(
                {"energy": ene, "gradient": grad})
            return FakeQCIOOutput.model_validate({"results": res})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in ASE calculation: {e}")
            raise ElectronicStructureError(msg=f"Electronic structure failed: {e}", obj=e)

    def _compute_gradient_from_atoms(self, atoms: Atoms) -> NDArray:
        """Return dE/dx in Hartree/Bohr for an ASE Atoms object."""
        with self._calculator_in_use() as calc:
            grad_ev_ang = calc.get_forces(atoms=atoms) * (-1.0)
        return (grad_ev_ang / ANGSTROM_TO_BOHR) / Hartree

    def compute_hessian(
        self,
        node: StructureNode,
        step_size: float | None = None,
    ) -> NDArray:
        """
        Compute Hessian via central differences of gradients.

        This is significantly faster than the default Engine fallback because
        it scales as O(3N) gradient evaluations rather than O((3N)^2) energy
        evaluations.
        """
        if not isinstance(node, StructureNode):
            return super().compute_hessian(node=node, step_size=step_size)

        h = float(step_size if step_size is not None else self.finite_difference_hessian_step_size)
        if h <= 0:
            raise ValueError("finite-difference Hessian step size must be positive.")

        coords_bohr = np.asarray(node.coords, dtype=float)
        refshape = coords_bohr.shape
        x0 = coords_bohr.reshape(-1)
        ndof = x0.size
        if ndof == 0:
            raise ValueError("Cannot compute Hessian for a node with zero coordinates.")

        atoms_template = structure_to_ase_atoms(node.structure)
        hessian = np.zeros((ndof, ndof), dtype=float)
        for i in range(ndof):
            disp = np.zeros(ndof, dtype=float)
            disp[i] = h

            atoms_plus = atoms_template.copy()
            atoms_plus.positions = (x0 + disp).reshape(refshape) / ANGSTROM_TO_BOHR
            grad_plus = self._compute_gradient_from_atoms(atoms_plus).reshape(-1)

            atoms_minus = atoms_template.copy()
            atoms_minus.positions = (x0 - disp).reshape(refshape) / ANGSTROM_TO_BOHR
            grad_minus = self._compute_gradient_from_atoms(atoms_minus).reshape(-1)

            hessian[:, i] = (grad_plus - grad_minus) / (2.0 * h)

        # Enforce symmetry to damp numerical finite-difference noise.
        return 0.5 * (hessian + hessian.T)

    def compute_geometry_optimization(self, node: StructureNode, keywords={}) -> list[StructureNode]:
        """
        Computes a geometry optimization using ASE calculation and optimizer
        """
        return self._run_ase_optimization(
            node=node,
            optimizer_cls=self.ase_optimizer,
            keywords=keywords,
        )

    def compute_transition_state(
        self,
        node: StructureNode,
        keywords: dict | None = None,
    ) -> StructureNode:
        if self.transition_state_optimizer.upper() != "SELLA":
            raise ElectronicStructureError(
                msg=(
                    "ASE transition-state optimization now only supports `SELLA`."
                )
            )
        if SellaOptimizer is None:
            raise ElectronicStructureError(
                msg=(
                    "ASE transition-state optimization requires the `sella` package, "
                    "but it is unavailable in this environment. Install `sella` to use "
                    "`ASEEngine.compute_transition_state`."
                )
            )

        ts_keywords = dict(keywords or {})
        ts_optimizer_kwds = dict(ts_keywords.pop("optimizer_kwds", ts_keywords.pop("optimizer_kwargs", {})) or {})
        ts_optimizer_kwds.setdefault("order", 1)
        ts_keywords["optimizer_kwds"] = ts_optimizer_kwds

        trajectory = self._run_ase_optimization(
            node=node,
            optimizer_cls=SellaOptimizer,
            keywords=ts_keywords,
        )
        if len(trajectory) == 0:
            raise ElectronicStructureError(
                msg="ASE transition-state optimization completed but no trajectory was produced."
            )
        return trajectory[-1]

    def compute_irc_chain(
        self,
        ts_node: StructureNode,
        keywords: dict | None = None,
    ) -> Chain:
        if SellaIRC is None:
            raise ElectronicStructureError(
                msg=(
                    "ASE IRC computation requires the `sella` package, but it is unavailable "
                    "in this environment. Install `sella` to use IRC with `ASEEngine`."
                )
            )

        ts_keywords = dict(keywords or {})
        # 0.01 eV/Angstrom: at 0.1 an IRC can stop on a shoulder well above the minimum, and minimizing from
        # there can slide into another isomer that a bond-graph comparison does not see
        fmax = float(ts_keywords.pop("fmax", 0.01))
        steps = int(ts_keywords.pop("maxiter", ts_keywords.pop("maxit", ts_keywords.pop("steps", 3000))))

        optimizer_kwds = ts_keywords.pop("optimizer_kwds", ts_keywords.pop("optimizer_kwargs", {}))
        if optimizer_kwds is None:
            optimizer_kwds = {}
        elif not isinstance(optimizer_kwds, dict):
            raise ValueError("`optimizer_kwds` must be a dictionary if provided.")
        optimizer_kwds = dict(optimizer_kwds)

        for key in ("dx", "eta", "gamma", "keep_going"):
            if key in ts_keywords and key not in optimizer_kwds:
                optimizer_kwds[key] = ts_keywords.pop(key)

        if ts_keywords:
            # Forward any remaining keyword customizations to the Sella IRC constructor.
            optimizer_kwds.update(ts_keywords)

        self.compute_energies([ts_node])

        def _run_direction(direction: str) -> list[StructureNode]:
            atoms = self._constrain(structure_to_ase_atoms(ts_node.structure))
            tmp = tempfile.NamedTemporaryFile(suffix=".traj", mode="w+", delete=False)
            with self._calculator_in_use() as calc:
                atoms.calc = calc
                # Sella's inner loop can fail to converge on the first steps
                # off the saddle (seen on every g-xTB IRC tried): it then
                # stops the whole IRC. keep_going steps on past it, and the
                # IRC reaches its minima; tried once, from the TS again.
                tries = [optimizer_kwds]
                if not optimizer_kwds.get("keep_going"):
                    tries.append({**optimizer_kwds, "keep_going": True})
                for k, kwds in enumerate(tries):
                    if k:
                        atoms.set_positions(structure_to_ase_atoms(ts_node.structure).get_positions())
                        Path(tmp.name).write_bytes(b"")
                    optimizer = _sella_irc_class(SellaIRC)(
                        atoms=atoms,
                        logfile=None,
                        trajectory=tmp.name,
                        **kwds,
                    )
                    try:
                        optimizer.run(fmax=fmax, steps=steps, direction=direction)
                        break
                    except Exception as exc:
                        if k + 1 < len(tries) and type(exc).__name__ == "IRCInnerLoopConvergenceFailure":
                            continue
                        raise ElectronicStructureError(msg=f"ASE IRC computation failed ({type(exc).__name__}).",
                                                       obj=exc)

            charge = ts_node.structure.charge
            multiplicity = ts_node.structure.multiplicity
            aT = [frame for frame in Trajectory(tmp.name)]
            energies = [obj.get_potential_energy() / Hartree for obj in aT]
            gradients = [(-1 * obj.get_forces() / ANGSTROM_TO_BOHR) / Hartree for obj in aT]
            if not atoms.constraints:
                # Sella steps in Cartesians with rotations left in, so the path also drifts rigidly; drop that.
                positions, gradients = _align_frames(
                    [obj.get_positions() for obj in aT], gradients, structure_to_ase_atoms(ts_node.structure).get_positions())
                for obj, pos in zip(aT, positions):
                    obj.positions = pos
            traj_nodes = [
                StructureNode(
                    structure=ase_atoms_to_structure(
                        atoms=obj,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                )
                for obj in aT
            ]
            all_results = []
            for ene, grad in zip(energies, gradients):
                res = FakeQCIOResults.model_validate({"energy": ene, "gradient": grad})
                out = FakeQCIOOutput.model_validate({"results": res})
                all_results.append(out)
            Path(tmp.name).unlink()
            update_node_cache(node_list=traj_nodes, results=all_results)
            return traj_nodes

        if self.irc_branches_at_once:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="irc") as pool:
                reverse, forward = pool.submit(_run_direction, "reverse"), pool.submit(_run_direction, "forward")
                reverse_nodes, forward_nodes = reverse.result(), forward.result()
        else:
            reverse_nodes = _run_direction("reverse")
            forward_nodes = _run_direction("forward")

        if len(reverse_nodes) == 0 and len(forward_nodes) == 0:
            raise ElectronicStructureError(
                msg="ASE IRC completed but no trajectory points were produced."
            )

        def _strip_initial_ts(nodes: list[StructureNode]) -> list[StructureNode]:
            if len(nodes) == 0:
                return nodes
            if np.allclose(np.asarray(nodes[0].coords), np.asarray(ts_node.coords), atol=1e-6):
                return nodes[1:]
            return nodes

        left_branch = list(reversed(_strip_initial_ts(reverse_nodes)))
        right_branch = _strip_initial_ts(forward_nodes)
        if len(left_branch) == 0 and len(right_branch) == 0:
            raise ElectronicStructureError(
                msg=(
                    "ASE IRC completed but produced no non-TS trajectory points. "
                    "Check that the input is a first-order saddle and increase IRC steps if needed."
                )
            )
        irc_nodes = left_branch + [ts_node.copy()] + right_branch
        return Chain.model_validate({"nodes": irc_nodes})
