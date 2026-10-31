"""FAIR-Chem machine-learned potentials (UMA and other fairchem checkpoints)
as a level of theory.

Energies and gradients for a whole chain are evaluated in batched forward
passes on the GPU instead of one image at a time; geometry optimizations,
transition-state searches and IRCs go through ASE exactly as for
`ASEEngine`. The model is loaded lazily and shared between copies of the
engine, and never pickled, so GSM's engine server and the parallel MSMEP
runner each reuse or reload it instead of serialising GPU tensors.

Requires `fairchem-core` (and a Hugging Face login with access to the
gated `facebook/UMA` weights for the named UMA models).
"""
from __future__ import annotations

import threading
from typing import List, Union

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
from numpy.typing import NDArray
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.chain import Chain
from mepd.engines.ase import AVAIL_OPTS, ASEEngine
from mepd.errors import ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache
from mepd.qcdata_structure_helpers import structure_to_ase_atoms

_EV_PER_ANG_TO_HARTREE_PER_BOHR = 1.0 / (Hartree * ANGSTROM_TO_BOHR)
_LOAD_LOCK = threading.RLock()


def _inference_settings(name):
    """fairchem InferenceSettings for mepd's `inference_settings` name (see
    FAIRChemEngine). "default": uncompiled and unmerged, without the
    activation checkpointing of fairchem's "batch" (which recomputes the
    network for the forces: about 20% slower per call, for memory mepd does
    not need); its energies match fairchem's default to 1e-7 Eh."""
    if not isinstance(name, str):
        return name
    if name == "default":
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

        return InferenceSettings(tf32=False, activation_checkpointing=False, merge_mole=False, compile=False,
                                 external_graph_gen=False, internal_graph_gen_version=2)
    return "default" if name == FAIRChemEngine.COMPILED else name


def _linear_time_mole() -> None:
    """Make UMA's mixture-of-linear-experts layers cost linear time in the batch size.

    Without fairchem's compiled extension, `MOLE.forward` slices its input once per structure, and the
    backward pass (the forces) of each slice writes a zero tensor the size of the whole batch: a batch of
    256 24-atom structures took 5.4 s, 21 ms per structure, as much as one at a time. `torch.split` (whose
    backward is one concatenation) gives the same numbers (to 2e-6 eV) in 0.63 s."""
    import torch
    from fairchem.core.models.uma.nn import mole

    if getattr(mole.MOLE.forward, "_mepd_linear", False):
        return
    sliced = mole.MOLE.forward

    def forward(self, x):
        g = self.global_mole_tensors
        sizes = g.mole_sizes.tolist()
        if g.ac_start_idx != 0 or sum(sizes) != x.shape[0]:   # activation-checkpointed chunks
            return sliced(self, x)
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum("eoi, be->boi", self.weights, g.expert_mixing_coefficients)
        return torch.cat([mole.linear_with_folded_batch(part, weights[n], bias=self.bias)
                          for n, part in enumerate(torch.split(x, sizes))], dim=0)

    forward._mepd_linear = True
    mole.MOLE.forward = forward


class FAIRChemEngine(ASEEngine):
    """A fairchem MLIP. `model` is a pretrained model name (e.g. "uma-s-1p2p1",
    "uma-m-1p1"); `checkpoint`, if given, is a local checkpoint file used
    instead. `task` selects the UMA head ("omol" for molecules, which reads
    charge and spin multiplicity from each structure). `device` defaults to
    CUDA when available. `batch_size` caps how many structures go through
    one forward pass.

    `inference_settings`: "default" is mepd's, the model as it is (no
    torch.compile, experts not merged): ready in seconds, any molecules in
    one batch. "compile" is fairchem's own default (merged experts and
    torch.compile): about 2.5x faster per call once compiled, but the
    compile costs 20-40 s in every process that loads the model and again
    for each new batch size, and it holds for one molecule only -- worth it
    for a long MD of one system, not for path searches. "turbo" (compiled,
    TF32), "batch" and "traineval" are fairchem's presets."""

    # mepd's names for fairchem InferenceSettings; anything else goes to fairchem as is.
    COMPILED = "compile"
    # a torch model (CUDA once loaded): callers that fork workers run it in a process of its own
    holds_torch_model = True

    def __init__(
        self,
        model: str = "uma-s-1p2p1",
        task: str = "omol",
        device: str | None = None,
        checkpoint: str | None = None,
        inference_settings: str = "default",
        batch_size: int = 32,
        geometry_optimizer: str = "LBFGSLineSearch",
        transition_state_optimizer: str = "SELLA",
        ase_optimizer=None,
    ):
        self.model = model
        self.task = task
        self.device = device
        self.checkpoint = checkpoint
        self.inference_settings = inference_settings
        self.batch_size = int(batch_size)
        self.geometry_optimizer = geometry_optimizer
        self.transition_state_optimizer = transition_state_optimizer
        self.ase_optimizer = ase_optimizer if ase_optimizer is not None else AVAIL_OPTS[geometry_optimizer]
        self._predictor = None
        # Address of a model server (`serve`): this engine and every copy of it (forked workers included)
        # then send structures there instead of loading the model themselves.
        self.server: str | None = None

    def __repr__(self) -> str:
        source = f"checkpoint={self.checkpoint!r}" if self.checkpoint else f"model={self.model!r}"
        return f"FAIRChemEngine({source}, task={self.task!r}, device={self.device!r})"

    def __eq__(self, other) -> bool:
        return self is other

    __hash__ = object.__hash__

    # --- model loading -------------------------------------------------
    @property
    def predictor(self):
        with _LOAD_LOCK:   # threads that start together load the model once
            if self._predictor is None:
                self._predictor = self._locked(self._load_predictor())
        return self._predictor

    @staticmethod
    def _locked(predictor):
        """`predictor` predicting one batch at a time: UMA keeps a batch's expert mixing in module-level
        state, so two threads predicting at once (path-search branches run in threads) mix up each
        other's batches ("index 1 is out of bounds for dimension 0 with size 1")."""
        lock, predict = threading.Lock(), predictor.predict

        def locked_predict(*args, **kwargs):
            with lock:
                return predict(*args, **kwargs)

        predictor.predict = locked_predict
        return predictor

    def _resolved_device(self) -> str:
        if self.device:
            return str(self.device)
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_predictor(self):
        try:
            from fairchem.core import pretrained_mlip
        except ImportError as exc:
            raise ImportError(
                "engine_name='fairchem' needs fairchem-core (pip install fairchem-core); "
                "the UMA models also need a Hugging Face login with access to facebook/UMA."
            ) from exc
        try:
            _linear_time_mole()
        except Exception:   # another fairchem version's layout: its own (slower) batching stays
            pass
        device = self._resolved_device()
        settings = _inference_settings(self.inference_settings)
        if self.checkpoint:
            return pretrained_mlip.load_predict_unit(self.checkpoint, inference_settings=settings, device=device)
        try:
            return pretrained_mlip.get_predict_unit(self.model, inference_settings=settings, device=device)
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            if "Gated" in type(exc).__name__ or any(code in text for code in ("401", "403", "restricted")):
                from mepd.engines.mlip import MODELS, UMA_ACCESS_HELP

                spec = MODELS.get(self.model)
                repo = spec.options.get("repo", "facebook/UMA") if spec else "facebook/UMA"
                raise PermissionError(
                    f"Can't download {self.model!r}: {text.splitlines()[0]}\n" + UMA_ACCESS_HELP.format(repo=repo)
                ) from exc
            raise

    @property
    def calculator(self):
        """A fresh ASE calculator on the shared model: ASE calculators keep
        per-structure results, so optimizations running concurrently must
        not share one."""
        if self.server:
            return _EngineCalculator(self)
        from fairchem.core import FAIRChemCalculator

        return FAIRChemCalculator(self.predictor, task_name=self.task)

    @calculator.setter
    def calculator(self, value) -> None:  # ASEEngine's dataclass __init__ is never used here
        raise AttributeError("FAIRChemEngine builds its calculator from `model`/`checkpoint`.")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_predictor"] = None
        state.pop("_server_process", None)
        return state

    def __deepcopy__(self, memo):
        # The model is read-only at inference time: share it rather than load
        # another copy onto the GPU (not copy.copy, which goes through
        # __getstate__ and would drop it).
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        clone.__dict__.update(self.__dict__)
        return clone

    # --- batched energies and gradients ---------------------------------
    def _batch_predict(self, atoms_chunk) -> tuple[np.ndarray, np.ndarray]:
        """One forward pass over `atoms_chunk`: energies (eV, one per
        structure) and forces (eV/Angstrom, all atoms concatenated in order)."""
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

        calc = self.calculator
        for atoms in atoms_chunk:
            calc.predictor.validate_atoms_data(atoms, calc.task_name)
        pred = calc.predictor.predict(atomicdata_list_to_batch([calc.a2g(atoms) for atoms in atoms_chunk]))
        return (pred["energy"].detach().cpu().numpy().reshape(-1),
                pred["forces"].detach().cpu().numpy().reshape(-1, 3))

    def _predict(self, atoms_list) -> tuple[list[float], list[np.ndarray]]:
        """Energies (Hartree) and gradients (Hartree/Bohr) for `atoms_list`,
        in batches of `batch_size` (on the model server, if there is one)."""
        if self.server:
            return _ask_server(self.server, atoms_list)
        energies, gradients = [], []
        size = max(1, self.batch_size)
        for start in range(0, len(atoms_list), size):
            chunk = atoms_list[start:start + size]
            e, f = self._batch_predict(chunk)
            offset = 0
            for i, atoms in enumerate(chunk):
                n = len(atoms)
                energies.append(float(e[i]) / Hartree)
                gradients.append(-f[offset:offset + n] * _EV_PER_ANG_TO_HARTREE_PER_BOHR)
                offset += n
        return energies, gradients

    def _run_calc(self, calctype: str, chain: Union[Chain, List]) -> List[StructureNode]:
        node_list = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [i for i, node in enumerate(node_list) if node._cached_energy is None]
        results = [node_list[i]._cached_result if i not in todo else None for i in range(len(node_list))]
        if todo:
            try:
                energies, gradients = self._predict([structure_to_ase_atoms(node_list[i].structure) for i in todo])
            except ImportError:
                raise
            except Exception as exc:
                raise ElectronicStructureError(msg=f"FAIR-Chem prediction failed: {exc}", obj=exc) from exc
            for i, e, g in zip(todo, energies, gradients):
                res = FakeQCIOResults.model_validate({"energy": e, "gradient": g})
                results[i] = FakeQCIOOutput.model_validate({"results": res})
        update_node_cache(node_list=node_list, results=results)
        return node_list

    def compute_geometry_optimizations(self, nodes: list[StructureNode], keywords=None,
                                       progress_callback=None) -> list[list[StructureNode]]:
        """Minimize `nodes` together: each by its own BFGS (ASE's, the same steps), all of them evaluated in
        one batched pass per iteration -- one structure at a time pays the GPU's per-call latency once per
        step for each. Per node, a one-frame trajectory (the minimum, with energy and gradient), or [] if it
        did not converge within the steps. `keywords`: `fmax` (eV/Angstrom) and `maxiter`, as the engine's
        own optimizations."""
        run_kwds, _ = self._extract_optimizer_run_kwargs(keywords)
        fmax, steps = run_kwds["fmax"], run_kwds["steps"]
        if getattr(self, "fixed_atoms", None):
            return [self.compute_geometry_optimization(n, keywords) for n in nodes]
        atoms = [structure_to_ase_atoms(n.structure) for n in nodes]
        x = [a.positions.copy().reshape(-1) for a in atoms]
        hess = [np.eye(len(xi)) * 70.0 for xi in x]          # ASE BFGS: alpha 70 eV/A^2, maxstep 0.2 A
        x0, f0 = [None] * len(x), [None] * len(x)
        out: list = [[] for _ in nodes]
        active = list(range(len(nodes)))
        for _ in range(steps + 1):
            for k in active:
                atoms[k].positions = x[k].reshape(-1, 3)
            energies, gradients = self._predict([atoms[k] for k in active])
            still = []
            for k, e, g in zip(active, energies, gradients):
                f = (-np.asarray(g) / _EV_PER_ANG_TO_HARTREE_PER_BOHR).reshape(-1)
                if np.linalg.norm(f.reshape(-1, 3), axis=1).max() < fmax:
                    node = StructureNode(structure=nodes[k].structure.model_copy(
                        update={"geometry": x[k].reshape(-1, 3) * ANGSTROM_TO_BOHR}))
                    res = FakeQCIOResults.model_validate({"energy": e, "gradient": g})
                    update_node_cache(node_list=[node], results=[FakeQCIOOutput.model_validate({"results": res})])
                    out[k] = [node]
                    if progress_callback is not None:
                        progress_callback(sum(1 for t in out if t), len(nodes))
                    continue
                if x0[k] is not None:
                    dr, df = x[k] - x0[k], f - f0[k]
                    a, dg = dr @ df, hess[k] @ dr
                    b = dr @ dg
                    if abs(a) > 1e-12 and abs(b) > 1e-12:
                        hess[k] -= np.outer(df, df) / a + np.outer(dg, dg) / b
                omega, vecs = np.linalg.eigh(hess[k])
                step = vecs @ (f @ vecs / np.abs(omega))
                longest = np.linalg.norm(step.reshape(-1, 3), axis=1).max()
                if longest > 0.2:
                    step *= 0.2 / longest
                x0[k], f0[k] = x[k].copy(), f
                x[k] = x[k] + step
                still.append(k)
            active = still
            if not active:
                break
        return out

    def compute_func(self, atoms):
        energies, gradients = self._predict([atoms])
        res = FakeQCIOResults.model_validate({"energy": energies[0], "gradient": gradients[0]})
        return FakeQCIOOutput.model_validate({"results": res})

    def _compute_gradient_from_atoms(self, atoms) -> NDArray:
        return self._predict([atoms])[1][0]

    def compute_hessian(self, node: StructureNode, step_size: float | None = None) -> NDArray:
        """Central differences of the model's own gradients, with all 6N
        displaced structures evaluated in batched passes."""
        if not isinstance(node, StructureNode):
            return super().compute_hessian(node=node, step_size=step_size)
        h = float(step_size if step_size is not None else self.finite_difference_hessian_step_size)
        x0 = np.asarray(node.coords, dtype=float)
        shape, flat = x0.shape, x0.reshape(-1)
        base = structure_to_ase_atoms(node.structure)
        displaced = []
        for i in range(flat.size):
            for sign in (1.0, -1.0):
                x = flat.copy()
                x[i] += sign * h
                atoms = base.copy()
                atoms.positions = x.reshape(shape) / ANGSTROM_TO_BOHR
                displaced.append(atoms)
        _, grads = self._predict(displaced)
        hessian = np.array([(grads[2 * i] - grads[2 * i + 1]).reshape(-1) / (2.0 * h) for i in range(flat.size)])
        return 0.5 * (hessian + hessian.T)


# --- a model server ----------------------------------------------------------
# One process holds the model on the GPU; the explorer's processes (its forked workers included) send it their
# structures. CUDA then never starts in a process that forks, and requests that arrive together, from any
# number of workers, go through the GPU in one batch.

_SERVER_KEYS = ("model", "task", "device", "checkpoint", "inference_settings", "batch_size")
_CLIENTS = threading.local()


def serve(engine: "FAIRChemEngine", folder=None) -> "FAIRChemEngine":
    """Start a model server for `engine`'s model and point `engine` at it (returns it). The server is a
    spawned process that exits with this one; its socket lives in `folder` (a new temporary folder)."""
    import multiprocessing
    import tempfile
    from pathlib import Path

    if engine.server:
        return engine
    folder = Path(folder) if folder else Path(tempfile.mkdtemp(prefix="mepd-uma-"))
    folder.mkdir(parents=True, exist_ok=True)
    address = str(folder / "model.sock")
    ctx = multiprocessing.get_context("spawn")
    ours, theirs = ctx.Pipe()
    proc = ctx.Process(target=_serve, args=({k: getattr(engine, k) for k in _SERVER_KEYS}, address, theirs),
                       daemon=True, name="mepd-model-server")
    proc.start()
    theirs.close()
    if not ours.poll(600):
        proc.kill()
        raise RuntimeError("The model server did not start within 10 minutes.")
    status = ours.recv()
    if status != "ready":
        raise RuntimeError(f"The model server failed to start: {status}")
    engine.server = address
    engine._server_process = proc
    return engine


def _serve(settings: dict, address: str, ready) -> None:
    import os
    import queue
    import sys
    from multiprocessing.connection import Listener

    parent = os.getppid()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
        except Exception:
            pass
    try:
        engine = FAIRChemEngine(**{k: v for k, v in settings.items() if k != "batch_size"},
                                batch_size=max(256, int(settings.get("batch_size") or 0)))
        engine._predict(_warm_up_atoms())
        listener = Listener(address, family="AF_UNIX")
    except Exception as exc:
        ready.send(f"{type(exc).__name__}: {exc}")
        return
    requests: queue.Queue = queue.Queue()
    merged: dict = {}   # (composition, charge, spin) -> an engine with that system's experts merged

    def handle(conn):
        while True:
            try:
                req = conn.recv()
            except (EOFError, OSError):
                return
            done, slot = threading.Event(), {}
            requests.put((req, slot, done))
            done.wait()
            try:
                conn.send(slot["out"])
            except OSError:
                return

    def accept():
        while True:
            try:
                conn = listener.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    ready.send("ready")
    ready.close()
    while True:
        try:
            batch = [requests.get(timeout=5.0)]
        except queue.Empty:
            if os.getppid() != parent:
                return
            continue
        n = len(batch[0][0])
        while n < engine.batch_size:   # whatever else has arrived goes in the same pass
            try:
                item = requests.get_nowait()
            except queue.Empty:
                break
            batch.append(item)
            n += len(item[0])
        atoms = [_atoms(rec) for req, _, _ in batch for rec in req]
        try:
            energies, gradients = _predict_grouped(engine, merged, atoms)
        except Exception as exc:
            for _, slot, done in batch:
                slot["out"] = ("error", f"{type(exc).__name__}: {exc}")
                done.set()
            continue
        k = 0
        for req, slot, done in batch:
            slot["out"] = ("ok", energies[k:k + len(req)], gradients[k:k + len(req)])
            k += len(req)
            done.set()


MERGED_SYSTEMS = 4   # compositions the server keeps a merged model for


def _system_key(atoms) -> tuple:
    numbers, counts = np.unique(atoms.numbers, return_counts=True)
    return (tuple(numbers.tolist()), tuple(counts.tolist()), int(atoms.info.get("charge", 0)),
            int(atoms.info.get("spin", 1)))


def _predict_grouped(engine, merged: dict, atoms):
    """`engine._predict(atoms)`, each composition (with charge and spin) that keeps coming back on a model
    whose experts are merged for it: UMA's mixture of experts then reduces to one plain network, 2.5x the
    structures per second in a batch (1.25 against 3.1 ms per 24-atom structure), the same numbers. Others
    (e.g. one molecule of a complex, for its free energy) go to `engine`, which takes any mixture."""
    groups: dict = {}
    for i, a in enumerate(atoms):
        groups.setdefault(_system_key(a), []).append(i)
    energies, gradients = [None] * len(atoms), [None] * len(atoms)
    for key, idx in groups.items():
        target = merged.get(key)
        if target is None and len(merged) < MERGED_SYSTEMS and engine.inference_settings == "default" \
                and engine.task == "omol":
            try:
                from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

                target = FAIRChemEngine(
                    model=engine.model, task=engine.task, device=engine.device, checkpoint=engine.checkpoint,
                    batch_size=engine.batch_size,
                    inference_settings=InferenceSettings(
                        tf32=False, activation_checkpointing=False, merge_mole=True, compile=False,
                        external_graph_gen=False, internal_graph_gen_version=2))
                _predict_fitting(target, [atoms[idx[0]]])
            except Exception:
                target = None
            merged[key] = target or engine
        e, g = _predict_fitting(merged.get(key) or engine, [atoms[i] for i in idx])
        for i, ei, gi in zip(idx, e, g):
            energies[i], gradients[i] = ei, gi
    return energies, gradients


def _predict_fitting(engine, atoms):
    """`engine._predict(atoms)`, in halves (and so on) where a batch doesn't fit in GPU memory."""
    import torch

    try:
        return engine._predict(atoms)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(atoms) == 1:
            raise
    half = len(atoms) // 2
    e1, g1 = _predict_fitting(engine, atoms[:half])
    e2, g2 = _predict_fitting(engine, atoms[half:])
    return e1 + e2, g1 + g2


def _warm_up_atoms():
    from ase import Atoms

    water = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    water.info.update(charge=0, spin=1)
    return [water]


def _record(atoms) -> tuple:
    return (atoms.numbers.copy(), atoms.positions.copy(), int(atoms.info.get("charge", 0)),
            int(atoms.info.get("spin", 1)))


def _atoms(rec):
    from ase import Atoms

    numbers, positions, charge, spin = rec
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.info.update(charge=charge, spin=spin)
    return atoms


def _ask_server(address: str, atoms_list) -> tuple[list[float], list[np.ndarray]]:
    """Energies (Hartree) and gradients (Hartree/Bohr) from the model server: one connection per process
    and thread."""
    import os
    from multiprocessing.connection import Client

    if not atoms_list:
        return [], []
    conn = getattr(_CLIENTS, "conn", None)
    if conn is None or _CLIENTS.pid != os.getpid() or _CLIENTS.address != address:
        conn = _CLIENTS.conn = Client(address, family="AF_UNIX")
        _CLIENTS.pid, _CLIENTS.address = os.getpid(), address
    conn.send([_record(a) for a in atoms_list])
    out = conn.recv()
    if out[0] != "ok":
        raise RuntimeError(f"model server: {out[1]}")
    return list(out[1]), list(out[2])


class _EngineCalculator(Calculator):
    """An ASE calculator on an engine's `_predict` (the model server's), for optimizations and TS searches."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energies, gradients = self.engine._predict([self.atoms])
        self.results = {"energy": energies[0] * Hartree,
                        "forces": -np.asarray(gradients[0]) / _EV_PER_ANG_TO_HARTREE_PER_BOHR}
