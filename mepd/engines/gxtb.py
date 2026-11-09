from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Any, Callable, List, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
from numpy.typing import NDArray

from mepd.chain import Chain
from qcref.constants import ANGSTROM_TO_BOHR
from mepd.engines.engine import (
    Engine,
    FiniteDifferenceHessianOutput,
    FiniteDifferenceHessianResults,
    build_hessian_result_from_matrix,
)
from mepd.errors import (
    ElectronicStructureError,
    EnergiesNotComputedError,
    GeometryOptimizationNotConvergedError,
    GradientsNotComputedError,
)
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache
from mepd.programs import standalone_gxtb
from mepd.qcdata_structure_helpers import ase_atoms_to_structure
from mepd.cpus import available_cpus


_TOTAL_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+Eh")
_OPT_FAILED_RE = re.compile(r"FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN (\d+) ITERATIONS", re.IGNORECASE)
_OPT_CONVERGED_RE = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED AFTER (\d+) ITERATIONS", re.IGNORECASE)


def _check_gxtb_optimization_converged(stdout: str, *, maxiter: int | None) -> None:
    """Raise if g-xTB's own output reports the optimization did not converge.

    g-xTB happily returns its last frame (via xtbopt.xyz/xtbopt.log) even when
    it simply ran out of optimization cycles -- there is no other signal in
    the returned trajectory that distinguishes "already at a minimum" from
    "gave up mid-optimization". Only stdout says which one actually happened.
    """
    failed = _OPT_FAILED_RE.search(stdout)
    if failed is None:
        return
    iterations = failed.group(1)
    budget_note = f" (iteration budget: {int(maxiter)})" if maxiter is not None else ""
    raise GeometryOptimizationNotConvergedError(
        msg=(
            f"g-xTB geometry optimization did not converge in {iterations} iterations"
            f"{budget_note}. Refusing to treat an unconverged geometry as a minimum."
        ),
        obj=stdout,
    )



_XTBOPT_ENERGY_RE = re.compile(r"energy:\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")


def _read_xtbopt_progress(fp: Path, natoms: int) -> tuple[list[float], list[str]]:
    """Energies and xyz frames of every step xtb has written to its
    optimization log so far (see `progress.read_xyz_steps`)."""
    from mepd.progress import read_xyz_steps

    return read_xyz_steps(fp, natoms, _XTBOPT_ENERGY_RE)


class _GXTBASEResultsCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        engine: "GXTBCalculator",
        *,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, list(properties), system_changes)
        if self.atoms is None:
            raise ElectronicStructureError(
                msg="ASE did not provide atoms for g-xTB calculation."
            )

        charge = int(self.atoms.info.get("charge", self.charge))
        multiplicity = int(self.atoms.info.get("spin", self.multiplicity))
        structure = ase_atoms_to_structure(
            atoms=self.atoms,
            charge=charge,
            multiplicity=multiplicity,
        )
        # Energies only: no molecular graph (perceiving one every ASE step is
        # wasted time, and slow for a large QM/MM system).
        node = StructureNode(structure=structure, has_molecular_graph=False)

        energy_hartree = float(self.engine.compute_energies([node])[0])
        gradient_hartree_bohr = np.asarray(
            self.engine.compute_gradients([node])[0], dtype=float
        )
        self.results["energy"] = energy_hartree * Hartree
        # ASE forces are -dE/dx in eV/Angstrom.
        self.results["forces"] = -gradient_hartree_bohr * Hartree * ANGSTROM_TO_BOHR


def restore_rigid_body_hessian(hessian: NDArray, coords, gradient) -> NDArray:
    """Replace the rigid-body part of a Cartesian Hessian with its exact value.

    Because the energy is invariant to rigid motion, the Hessian's action on
    the six rigid-body directions follows from the gradient alone: H t = 0
    for a translation, and H (a x r) = a x g (atom by atom) for a rotation
    about axis a. Programs that project translations/rotations out of the
    Hessian (g-xTB does) set those to zero, which is right only where g = 0;
    at non-stationary points (along an IRC) that error shifts soft
    frequencies by tens of cm^-1. With Q an orthonormal basis of the rigid
    directions and P = 1 - QQ^T, this returns
        P H P + (HQ) Q^T + Q (HQ)^T - Q (Q^T H Q) Q^T,
    keeping the input's internal block and the exact rigid-body couplings.
    """
    H = np.asarray(hessian, dtype=float)
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    G = np.asarray(gradient, dtype=float).reshape(-1, 3)
    n = X.size
    r = X - X.mean(axis=0)
    V, HV = [], []
    for a in np.eye(3):
        V.append(np.tile(a, (len(X), 1)).reshape(-1))
        HV.append(np.zeros(n))
    for a in np.eye(3):
        V.append(np.cross(a, r).reshape(-1))
        HV.append(np.cross(a, G).reshape(-1))
    V, HV = np.array(V).T, np.array(HV).T
    _, sv, Wt = np.linalg.svd(V, full_matrices=False)
    keep = sv > 1e-8 * sv.max()
    M = Wt[keep].T / sv[keep]
    Q, HQ = V @ M, HV @ M
    P = np.eye(n) - Q @ Q.T
    QHQ = 0.5 * (Q.T @ HQ + (Q.T @ HQ).T)
    out = P @ H @ P + HQ @ Q.T + Q @ HQ.T - Q @ QHQ @ Q.T
    return 0.5 * (out + out.T)


_SETPRIV = shutil.which("setpriv")


log = logging.getLogger(__name__)

# Retries for an SCF that does not converge (tried in order): Fermi smearing
# at a raised electronic temperature damps the charge oscillation.
# xtb prints this for GFN0/1/2-xTB runs (also with --silent), never for g-xTB
_GFN_HAMILTONIAN = re.compile(r":\s*Hamiltonian\s+(GFN\d?-xTB)")
SCF_FALLBACKS = (("--etemp", "1000", "--iterations", "1000"), ("--etemp", "3000", "--iterations", "1000"))


def _scf_not_converged(output: str) -> bool:
    return bool(re.search(r"SCF not converged|did not converge|cannot be satisfied", output))


_TBLITE = threading.local()
_TBLITE_SOLVENTS = {"--alpb": "alpb-solvation", "--gbsa": "gbsa-solvation"}


def _tblite_options(extra_args: list) -> dict | None:
    """The tblite settings for an xtb command line's options, or None when it has one tblite isn't given
    here (anything but --gfn 1/2, --alpb/--gbsa <solvent>, --etemp, --acc)."""
    opts = {"method": "GFN2-xTB", "solvation": None, "etemp": 300.0, "acc": 1.0}
    args = [str(a) for a in extra_args]
    k = 0
    while k < len(args):
        a = args[k]
        value = args[k + 1] if k + 1 < len(args) else None
        if a == "--gfn" and value in ("1", "2"):
            opts["method"] = f"GFN{value}-xTB"
        elif a in _TBLITE_SOLVENTS and value and not value.startswith("-"):
            opts["solvation"] = (_TBLITE_SOLVENTS[a], value)
        elif a == "--etemp" and value:
            opts["etemp"] = float(value)
        elif a == "--acc" and value:
            opts["acc"] = float(value)
        else:
            return None
        k += 2
    return opts


def _tblite_point(node, extra_args) -> tuple[float, np.ndarray] | None:
    """(energy in Hartree, gradient in Hartree/Bohr) for `node` from tblite, restarting from the stored
    wavefunction of the nearest of this thread's recent structures with the same atoms; None when tblite
    isn't installed, doesn't take these options or fails (the caller then runs xtb).

    A failed SCF runs all its cycles (~0.85 s at 24 atoms, xtb itself then converges in ~35 ms), and the
    next steps of an optimization sit right next to it: so a failure is retried once from a fresh guess,
    and structures within 1 Bohr RMS of one that failed both ways skip tblite (Suzuki run 34: one TS
    optimization spent 12 min in ~800 failed attempts while 95 cores idled)."""
    opts = _tblite_options(extra_args)
    if opts is None:
        return None
    try:
        from tblite.interface import Calculator
    except ImportError:
        return None
    folder = os.environ.get("MEPD_CALL_LOG")
    t0 = time.time()
    numbers = np.array([int(z) for z in node.structure.atomic_numbers])
    pos = np.asarray(node.structure.geometry, dtype=float)   # Bohr
    charge, uhf = int(node.structure.charge), max(0, int(node.structure.multiplicity) - 1)
    key = (tuple(numbers.tolist()), charge, uhf, opts["method"], opts["solvation"], opts["etemp"], opts["acc"])
    cache = _TBLITE.__dict__.setdefault("cache", {})
    if os.getpid() != _TBLITE.__dict__.get("pid"):          # a forked child: its parent's objects aren't its own
        cache.clear()
        _TBLITE.pid = os.getpid()
    entry = cache.get(key)
    if entry is not None and any(float(np.sqrt(((p - pos) ** 2).sum(axis=1).mean())) <= 1.0 for p in entry[2]):
        return None
    try:
        if entry is None:
            calc = Calculator(opts["method"], numbers, pos, charge=float(charge), uhf=uhf)
            calc.set("verbosity", 0)
            calc.set("temperature", opts["etemp"] * 3.166808578545117e-06)
            calc.set("accuracy", opts["acc"])
            if opts["solvation"]:
                calc.add(*opts["solvation"])
            entry = cache[key] = (calc, [], [])
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        calc, recent, failed = entry
        calc.update(pos)
    except Exception:
        return None
    guess = None
    if recent:
        dist, guess = min(((float(np.sqrt(((p - pos) ** 2).sum(axis=1).mean())), r) for p, r in recent),
                          key=lambda t: t[0])
        if dist > 1.0:   # Bohr RMS: too far for its wavefunction to help
            guess = None
    res = None
    for start in ([guess, None] if guess is not None else [None]):
        try:
            # copy: tblite otherwise overwrites the guess in place, leaving the stored structure with another one's
            res = calc.singlepoint(start, copy=True) if start is not None else calc.singlepoint()
            energy, gradient = float(res.get("energy")), np.array(res.get("gradient"))
            break
        except Exception:
            res = None
    if res is None:
        failed.append(pos.copy())
        if len(failed) > 32:
            failed.pop(0)
        _log_tblite(folder, t0, False)
        return None
    recent.append((pos.copy(), res))
    if len(recent) > 32:
        recent.pop(0)
    _log_tblite(folder, t0, True)
    return energy, gradient


def _log_tblite(folder, t0: float, ok: bool) -> None:
    if not folder:
        return
    try:
        import json

        with open(Path(folder) / f"{os.getpid()}.jsonl", "a") as fh:
            fh.write(json.dumps({"t": round(t0, 3), "wall": round(time.time() - t0, 4), "cpu": 0, "ok": ok,
                                 "flags": ["--tblite", "--grad"], "site": _call_site()}) + "\n")
    except OSError:
        pass


def _call_site() -> str:
    """The mepd functions (outside the engines) that led to this call, innermost first."""
    import traceback

    names = []
    for frame in reversed(traceback.extract_stack(limit=40)[:-3]):
        f = frame.filename.replace(os.sep, "/")
        if "/mepd/" in f and "/mepd/engines/" not in f:
            names.append(frame.name)
            if len(names) == 4:
                break
    return "<".join(names)


def _run_gxtb_process(cmd: list[str], *, cwd: Path, env: dict, timeout_s: float,
                      watch: Callable[[], None] | None = None) -> subprocess.CompletedProcess[str]:
    """`_run_gxtb_process_` with, when MEPD_CALL_LOG names a folder, one JSON line per run in
    <folder>/<pid>.jsonl: start time, wall and CPU seconds, the run's flags and where in mepd it came from
    (for finding what a long run spends its time on; axtb_backend logs its calls the same way)"""
    folder = os.environ.get("MEPD_CALL_LOG")
    if not folder:
        return _run_gxtb_process_(cmd, cwd=cwd, env=env, timeout_s=timeout_s, watch=watch)
    import json
    import resource

    t0, c0 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    ok = False
    try:
        done = _run_gxtb_process_(cmd, cwd=cwd, env=env, timeout_s=timeout_s, watch=watch)
        ok = done.returncode == 0
        return done
    finally:
        c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        rec = {"t": round(t0, 3), "wall": round(time.time() - t0, 4),
               "cpu": round(c1.ru_utime + c1.ru_stime - c0.ru_utime - c0.ru_stime, 4), "ok": ok,
               "flags": [a for a in cmd[1:] if a.startswith("-")], "site": _call_site()}
        try:
            with open(Path(folder) / f"{os.getpid()}.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass


def _run_gxtb_process_(cmd: list[str], *, cwd: Path, env: dict, timeout_s: float,
                       watch: Callable[[], None] | None = None) -> subprocess.CompletedProcess[str]:
    """Run one g-xTB command and return its CompletedProcess.

    * On Linux the process gets a parent-death signal (`setpriv --pdeathsig
      KILL`): it is killed when the process that started it dies, so a job
      that is killed or stopped early never leaves g-xTB running on its own.
      (`preexec_fn` could do the same, but is unsafe with threads -- and
      this engine runs inside threaded code, e.g. GSM's engine server.)
      Only an executable that resolves is wrapped, so a missing one still
      fails with FileNotFoundError (and a stand-in `subprocess.run`, as in
      tests, sees the plain command).
    * `timeout_s` > 0 bounds the wall time: the process is killed and an
      ElectronicStructureError raised.
    * `watch`, if given, is called about twice a second while it runs.
    """
    limit = float(timeout_s or 0) or None
    full = [_SETPRIV, "--pdeathsig", "KILL", "--", *cmd] if _SETPRIV and shutil.which(cmd[0]) else list(cmd)

    def timed_out(out: str = "") -> ElectronicStructureError:
        return ElectronicStructureError(
            msg=f"g-xTB did not finish within {limit:.0f} s (gxtb_engine_kwds.timeout_s); killed it.", obj=out)

    try:
        if watch is None:
            kwds = {"timeout": limit} if limit is not None else {}
            try:
                done = subprocess.run(full, cwd=cwd, env=env, text=True, capture_output=True, check=False, **kwds)
            except subprocess.TimeoutExpired as exc:   # run() has already killed it
                raise timed_out(str(exc.stdout or "") + str(exc.stderr or "")) from None
            return subprocess.CompletedProcess(cmd, done.returncode, done.stdout, done.stderr)
        proc = subprocess.Popen(full, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise ElectronicStructureError(
            msg=(
                f"g-xTB executable `{cmd[0]}` was not found. "
                "Set `GXTB_EXECUTABLE` or pass `executable` to GXTBCalculator."
            )
        ) from exc
    deadline = None if limit is None else time.monotonic() + limit
    while True:
        wait = 0.5 if deadline is None else max(0.0, min(0.5, deadline - time.monotonic()))
        try:
            stdout, stderr = proc.communicate(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            if deadline is not None and time.monotonic() >= deadline:
                proc.kill()
                stdout, stderr = proc.communicate()
                raise timed_out((stdout or "") + (stderr or "")) from None
            try:
                watch()
            except Exception:
                pass
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


# xtb's implicit-solvent flags (and their short forms).
_SOLVATION_FLAGS = frozenset({"--alpb", "--gbsa", "-g", "--cosmo", "--cpcmx", "--gbe"})


def _split_args(args) -> list[str]:
    """One argv entry per word: xtb reads "--alpb water" given as one
    argument as an unknown flag and runs in the gas phase without a word
    (mepd#11), so entries with spaces are split as a shell would."""
    out = []
    for arg in [args] if isinstance(args, str) else args:
        arg = str(arg)
        out.extend(shlex.split(arg) if any(c.isspace() for c in arg) else [arg])
    return out


@dataclass
class GXTBCalculator(Engine):
    """Direct local g-xTB engine using the xtb executable with the g-xTB flag."""

    executable: str | Path | None = None
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    keep_workdirs: bool = False
    add_gxtb_flag: bool = True
    n_threads: int = 1
    # SCC accuracy (xtb --acc) for the numerical Hessian. At the default (1.0)
    # the displaced gradients are converged loosely enough that Hessian
    # elements are off by ~0.01-0.03 Eh/bohr^2 (xtb itself warns "Hessian
    # element ... is not symmetric"); 0.001 matches central differences of
    # g-xTB gradients to ~1e-4 for ~20% more time.
    hessian_acc: float = 0.001
    # Minimizations: "qcopt" (AFH's optimizer, mepd.engines._qcopt, to
    # `min_fmax`; the default since 2026-11-06: on 100 Reaction-QM reactions
    # inside mepd, TS searches and IRCs with half of Sella's gradients and 69
    # against 50 IRCs connecting; see CHANGELOG), "sella" (Sella, to
    # `min_fmax`; the default from 2026-10-19: 100% converged and 99.4% true
    # minima on 1000 IRC ends; its minimizer uses g-xTB's Hessian only with
    # `hessian_every`), "geometric", "lbfgs", or "xtb" (xtb's ANCopt, whose
    # "normal" threshold is ~0.05 eV/Angstrom).
    minimizer: str | None = None          # None: DEFAULT_MINIMIZER
    min_fmax: float = 0.001
    DEFAULT_MINIMIZER: ClassVar[str] = "qcopt"
    # TS searches and IRCs: "sella" (through ASE) or "qcopt" (mepd.engines._qcopt).
    ts_optimizer: str | None = None       # None: DEFAULT_TS_OPTIMIZER
    DEFAULT_TS_OPTIMIZER: ClassVar[str] = "qcopt"
    # Independent calculations (a chain's images, a batch of geometry
    # optimizations) run as up to this many concurrent g-xTB processes.
    # 0 = as many as the machine has cores for at n_threads each.
    n_parallel: int = 0
    # Wall-time limit per g-xTB call (seconds; 0 = none). A calculation that
    # hangs (e.g. an SCF that never settles) fails with an error instead of
    # blocking its path search forever.
    timeout_s: float = 3600.0
    # GFN-xTB gradients (not g-xTB) through the tblite library in this process instead of an xtb run each:
    # no process start or files, and each SCF starts from the wavefunction of the nearest structure this
    # thread computed (a path's image one step earlier): ~12 against ~45 ms for 24 atoms, the same energies
    # to ~3e-6 Eh. Hessians and xtb's own optimizations still run xtb. Runs whose options tblite doesn't
    # take (or that fail there) run xtb as before.
    in_process: bool = False

    # "xtb": run the g-xTB program for every calculation. "axtb": g-xTB 2.0.1 from axtb, AFH's Rust
    # reimplementation, in this process (mepd.engines.axtb_backend): the program's energies, no
    # process start per call, analytic Hessians. It takes only --etemp/--acc/--iterations in
    # extra_args (SCF accuracy 0.01 unless --acc is given: axtb_backend.ACCURACY), no minimizer =
    # "xtb", and timeout_s does not apply. With add_gxtb_flag false (GFN2-xTB, as the program runs
    # it without --gxtb) it computes GFN2-xTB, in xtb 6.7.1's conventions unless axtb_conventions
    # names others (axtb_backend.GFN2_CONVENTIONS).
    # None (the default): "axtb", unless the settings need the program -- minimizer = "xtb", an
    # extra_args option axtb does not take, an explicit `executable`, or no axtb installed -- then
    # "xtb". A profile that names a backend keeps it.
    backend: str | None = None
    # axtb's conventions: None/"axtb" (its default) or "gxtb" (reproduces the program to ~1e-12 Eh).
    axtb_conventions: str | None = None
    # g-xTB in implicit solvent through axtb's composite, E_gxtb + [E_GFN2+model - E_GFN2], in one
    # in-process call with analytic Hessians: {"solvent": <xtb name>, "model": "alpb" | "gbsa"}.
    # Set by mepd.solvation.solvate_engine from a profile's [solvation] table (backend "axtb" only);
    # the program route is mepd.solvation.SolvatedEngine (three xtb runs per gradient). The GFN2
    # halves follow axtb_conventions: "gxtb" reproduces those three runs, rung for rung.
    axtb_solvation: dict | None = None

    def __post_init__(self) -> None:
        if self._ts_optimizer() not in ("sella", "qcopt"):
            raise ValueError(f"ts_optimizer must be \"sella\" or \"qcopt\", not {self.ts_optimizer!r}.")
        if self.backend is None:
            self.backend = self._default_backend()
        if self.backend not in ("xtb", "axtb"):
            raise ValueError(f"backend must be \"xtb\" or \"axtb\", not {self.backend!r}.")
        if self.backend == "axtb":
            from mepd.engines import axtb_backend

            if not self.add_gxtb_flag and self.axtb_solvation:
                raise ValueError("axtb_solvation is g-xTB's solvation composite; with add_gxtb_flag = false "
                                 "(GFN2-xTB) a profile's [solvation] table uses the xtb program instead.")
            if (self.minimizer or self.DEFAULT_MINIMIZER) == "xtb":
                raise ValueError("minimizer = \"xtb\" runs the g-xTB program's own optimizer; with backend = "
                                 "\"axtb\" use qcopt, sella, geometric or lbfgs.")
            self._axtb_settings = axtb_backend.settings_from_args(_split_args(self.extra_args))
        if self.axtb_solvation:
            from mepd.engines import axtb_backend

            if self.backend != "axtb":
                raise ValueError("axtb_solvation is g-xTB in solvent through axtb (backend = \"axtb\"); with the "
                                 "program, a profile's [solvation] table wraps the engine instead.")
            if not axtb_backend.supports("composite"):
                raise ValueError("this axtb build has no g-xTB solvation composite (axtb >= 0.1.0 has); a profile's "
                                 "[solvation] table then uses the xtb program for the GFN2-xTB correction")
            sv = dict(self.axtb_solvation)
            self.axtb_solvation = {"solvent": str(sv["solvent"]), "model": str(sv.get("model", "alpb")).lower()}
            axtb_backend.solvation_settings(self.axtb_solvation["solvent"], self.axtb_solvation["model"])
        # Nothing configured: $GXTB_EXECUTABLE, else `gxtb`, which (if not on
        # PATH) _resolve_executable looks up -- or fetches -- on first use.
        self._default_executable = self.executable is None and not os.getenv("GXTB_EXECUTABLE")
        if self.executable is None:
            self.executable = os.getenv("GXTB_EXECUTABLE") or "gxtb"
        self.executable = str(self.executable)
        self.extra_args = _split_args(self.extra_args)
        if self.add_gxtb_flag or standalone_gxtb(self.executable):
            solvation = [a for a in self.extra_args if a.split("=")[0] in _SOLVATION_FLAGS]
            if solvation:
                raise ValueError(
                    f"g-xTB has no implicit-solvent parameters ({' '.join(solvation)} in extra_args): with "
                    "--alpb/--gbsa xtb stops, and --cosmo/--cpcmx/--gbe give gas-phase or implausible energies. "
                    "Use a profile [solvation] table (method = \"correction\": GFN2-xTB solvation on top of "
                    "g-xTB) instead.")
        self.env = {str(k): str(v) for k, v in dict(self.env or {}).items()}

    def _default_backend(self) -> str:
        """axtb when it computes what these settings ask for, else the g-xTB program."""
        if self.executable is not None or (self.minimizer or self.DEFAULT_MINIMIZER) == "xtb":
            return "xtb"
        from mepd.engines import axtb_backend

        if axtb_backend.version() is None:
            return "xtb"
        try:
            axtb_backend.settings_from_args(_split_args(self.extra_args))
        except ValueError:
            return "xtb"
        return "axtb"

    def _ts_optimizer(self) -> str:
        return str(self.ts_optimizer or self.DEFAULT_TS_OPTIMIZER).lower()

    def _resolve_executable(self) -> None:
        """No executable configured and `gxtb` not on PATH: an installed g-xTB
        build, or (first use) the official release, fetched by mepd.programs.
        An executable given explicitly (even "gxtb") is used as given."""
        if getattr(self, "_default_executable", False) and self.executable == "gxtb" and shutil.which("gxtb") is None:
            from mepd.programs import gxtb_executable

            try:
                self.executable = gxtb_executable() or self.executable
            except RuntimeError as exc:
                raise ElectronicStructureError(msg=str(exc)) from exc

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        try:
            grads = np.array([node.gradient for node in chain])
        except GradientsNotComputedError:
            node_list = self._run_calc(chain=chain)
            grads = np.array([node.gradient for node in node_list])

        return grads

    def compute_energies(self, chain: Union[Chain, List]) -> NDArray:
        try:
            enes = np.array([node.energy for node in chain])
        except EnergiesNotComputedError:
            node_list = self._run_calc(chain=chain)
            enes = np.array([node.energy for node in node_list])

        return enes

    def _run_calc(self, chain: Union[Chain, List]) -> list[StructureNode]:
        node_list = self._coerce_nodes(chain)
        inds_cached = [
            i
            for i, node in enumerate(node_list)
            if node._cached_energy is not None and node._cached_gradient is not None
        ]
        results: list[FakeQCIOOutput | None] = [None] * len(node_list)
        for i in inds_cached:
            results[i] = node_list[i]._cached_result

        todo = [i for i in range(len(node_list)) if i not in inds_cached]
        for i, res in zip(todo, self._map(self._compute_node, [node_list[i] for i in todo])):
            results[i] = res

        update_node_cache(node_list=node_list, results=results)
        return node_list

    @staticmethod
    def _coerce_nodes(chain: Union[Chain, List]) -> list[StructureNode]:
        if isinstance(chain, Chain):
            node_list = chain.nodes
        elif isinstance(chain, list):
            node_list = chain
        else:
            raise ValueError(f"Input needs to be a Chain or a List. You input a: {type(chain)}")

        if not node_list:
            return []
        if not isinstance(node_list[0], StructureNode):
            raise AssertionError(
                f"input nodes are incompatible with GXTBCalculator: {node_list[0]}"
            )
        return node_list

    def _axtb_kwds(self) -> dict:
        if not self.add_gxtb_flag:      # GFN2-xTB, as the program runs without --gxtb
            from mepd.engines.axtb_backend import GFN2_CONVENTIONS

            return {"threads": max(1, int(self.n_threads)), "conventions": self.axtb_conventions or GFN2_CONVENTIONS,
                    "settings": {"guess": "fresh", **getattr(self, "_axtb_settings", {})}, "solvation": None,
                    "method": "gfn2"}
        return {"threads": max(1, int(self.n_threads)), "conventions": self.axtb_conventions,
                "settings": getattr(self, "_axtb_settings", {}), "solvation": self.axtb_solvation}

    @property
    def analytic_hessian(self) -> bool:
        """compute_hessian is analytic (axtb), not finite differences of gradients."""
        return self.backend == "axtb"

    def _axtb_compute_node(self, node: StructureNode) -> FakeQCIOOutput:
        from mepd.engines import axtb_backend

        molden = bool(os.environ.get("MEPD_ORBITALS_DIR", "").strip())
        out = axtb_backend.point(node, molden=molden, **self._axtb_kwds())
        if molden and out.get("molden"):
            from mepd.orbitals import save_run_orbitals

            with tempfile.TemporaryDirectory(prefix="axtb-") as tmp:
                fp = Path(tmp) / "molden.input"
                fp.write_text(out["molden"])
                save_run_orbitals(fp, node.structure.to_xyz(), "axtb",
                                  "g-xTB" if self.add_gxtb_flag else "GFN2-xTB")   # axtb's Molden conventions
        res = FakeQCIOResults.model_validate({"energy": out["energy"], "gradient": out["gradient"]})
        return FakeQCIOOutput.model_validate({"results": res})

    def _compute_node(self, node: StructureNode) -> FakeQCIOOutput:
        if self.backend == "axtb":
            return self._axtb_compute_node(node)
        if self.in_process and not self.add_gxtb_flag and Path(self.executable).name != "gxtb":
            out = _tblite_point(node, self.extra_args)
            if out is not None:
                energy, gradient = out
                res = FakeQCIOResults.model_validate({"energy": energy, "gradient": gradient})
                return FakeQCIOOutput.model_validate({"results": res})
        with tempfile.TemporaryDirectory(prefix="gxtb-") as tmp:
            workdir = Path(tmp)
            xyz_path = workdir / "structure.xyz"
            xyz_path.write_text(node.structure.to_xyz())
            completed = self._run_gxtb(
                xyz_path=xyz_path,
                charge=int(node.structure.charge),
                multiplicity=int(node.structure.multiplicity),
                cwd=workdir,
                optimize=False,
            )
            try:
                energy = self._parse_energy(workdir=workdir, stdout=completed.stdout)
                gradient = self._parse_gradient(workdir / "gradient", natoms=len(node.symbols))
            except Exception as exc:
                raise ElectronicStructureError(
                    msg="Failed to parse g-xTB output.", obj=completed.stdout + completed.stderr
                ) from exc

            if self.keep_workdirs:
                persistent = Path.cwd() / "gxtb-workdirs"
                persistent.mkdir(exist_ok=True)
                shutil.copytree(workdir, persistent / workdir.name, dirs_exist_ok=True)

        res = FakeQCIOResults.model_validate({"energy": energy, "gradient": gradient})
        return FakeQCIOOutput.model_validate({"results": res})

    def _run_gxtb(
        self,
        *,
        xyz_path: Path,
        charge: int,
        multiplicity: int,
        cwd: Path,
        optimize: bool,
        watch: Callable[[], None] | None = None,
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self._resolve_executable()
        cmd = [
            self.executable,
            str(xyz_path.name),
            "--silent",
            "--chrg",
            str(charge),
        ]
        cmd.append("--opt" if optimize else "--grad")
        uhf = max(0, int(multiplicity) - 1)
        if uhf:
            cmd.extend(["--uhf", str(uhf)])
        if self.add_gxtb_flag and not standalone_gxtb(self.executable):
            cmd.append("--gxtb")
        cmd.extend(self.extra_args if extra_args is None else extra_args)
        # A job's viewer can show this SCF's orbitals (mepd.orbitals): write them as molden.
        orbitals_kind = self._orbitals_kind(cmd)
        if orbitals_kind:
            cmd.append("--molden")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(self.n_threads))
        env.update(self.env)
        # `watch` (a live viewer reading the optimizer's step log) is called
        # about twice a second while it runs.
        completed = _run_gxtb_process(cmd, cwd=cwd, env=env, timeout_s=self.timeout_s, watch=watch)
        # An SCF that never settles (oscillating, e.g. on a stretched path
        # image): retry with Fermi smearing, which damps it. One geometry's
        # retry costs little; failing it would end the whole path search.
        # Only g-xTB runs that set no electronic temperature themselves: GFN-xTB
        # runs through this class (add_gxtb_flag off, e.g. solvation's gas /
        # solvated pairs) choose theirs, which must match within a pair.
        own_retries = SCF_FALLBACKS if self.add_gxtb_flag and "--etemp" not in cmd else ()
        for retry in own_retries:
            if completed.returncode == 0 or not _scf_not_converged(completed.stdout + completed.stderr):
                break
            (cwd / "xtbrestart").unlink(missing_ok=True)   # the failed run's charges
            completed = _run_gxtb_process(cmd + list(retry), cwd=cwd, env=env, timeout_s=self.timeout_s, watch=watch)
            if completed.returncode == 0:
                log.warning("g-xTB SCF did not converge for %s; converged with %s", xyz_path.name, " ".join(retry))
        if completed.returncode == 0:
            self._check_ran_gxtb(completed)
        if completed.returncode != 0:
            tried = "" if not (own_retries and _scf_not_converged(completed.stdout + completed.stderr)) else \
                f" (SCF not converged, also with {'; '.join(' '.join(r) for r in SCF_FALLBACKS)})"
            raise ElectronicStructureError(
                msg=f"g-xTB calculation failed with exit code {completed.returncode}{tried}.",
                obj=completed.stdout + completed.stderr,
            )
        if orbitals_kind:
            from mepd.orbitals import save_run_orbitals

            final = cwd / "xtbopt.xyz" if optimize and (cwd / "xtbopt.xyz").is_file() else xyz_path
            save_run_orbitals(cwd / "molden.input", final.read_text(), orbitals_kind,
                              "g-xTB" if orbitals_kind == "gxtb" else "GFN2-xTB")
        return completed

    def _orbitals_kind(self, cmd: list) -> str | None:
        """"gxtb" / "gfn2" when this run's orbitals are to be kept ($MEPD_ORBITALS_DIR set) and
        mepd.orbitals reads that program's molden files; else None (e.g. GFN1-xTB)."""
        if not os.environ.get("MEPD_ORBITALS_DIR", "").strip():
            return None
        if self.add_gxtb_flag or standalone_gxtb(self.executable):
            return "gxtb"
        gfn = cmd[cmd.index("--gfn") + 1] if "--gfn" in cmd and cmd.index("--gfn") + 1 < len(cmd) else "2"
        return "gfn2" if str(gfn) == "2" else None

    def compute_hessian(
        self,
        node: StructureNode,
        step_size: float | None = None,
    ) -> NDArray:
        if self.backend == "axtb":
            # Analytic and unprojected: its rigid-body part is already exact.
            from mepd.engines import axtb_backend

            return np.asarray(axtb_backend.hessian(node, **self._axtb_kwds())["hessian"], dtype=float)
        with tempfile.TemporaryDirectory(prefix="gxtb-hess-") as tmp:
            workdir = Path(tmp)
            xyz_path = workdir / "structure.xyz"
            xyz_path.write_text(node.structure.to_xyz())
            completed = self._run_gxtb_hessian(
                xyz_path=xyz_path,
                charge=int(node.structure.charge),
                multiplicity=int(node.structure.multiplicity),
                cwd=workdir,
            )
            try:
                hessian = self._parse_hessian(workdir / "hessian", natoms=len(node.symbols))
            except Exception as exc:
                raise ElectronicStructureError(
                    msg="Failed to parse g-xTB Hessian output.",
                    obj=completed.stdout + completed.stderr,
                ) from exc
            # g-xTB projects rigid translations/rotations out of its Hessian,
            # which is only valid at stationary points; restore the exact
            # rigid-body part from the gradient (see restore_rigid_body_hessian).
            gradient = node._cached_gradient
            if gradient is None:
                gradient = self.compute_gradients([node.copy()])[0]
            hessian = restore_rigid_body_hessian(hessian, node.coords, gradient)

            if self.keep_workdirs:
                persistent = Path.cwd() / "gxtb-workdirs"
                persistent.mkdir(exist_ok=True)
                shutil.copytree(workdir, persistent / workdir.name, dirs_exist_ok=True)

        return hessian

    def _compute_hessian_result(self, node: StructureNode, **kwargs):
        """Prefer g-xTB's own projected vibrational analysis (parsed from its
        g98.out output -- real frequencies with translation/rotation already
        removed by g-xTB itself, plus real normal-mode displacement vectors)
        over reimplementing mass-weighting/projection ourselves. Falls back
        to the generic build_hessian_result_from_matrix (mass-weight the raw
        Hessian, discard the lowest-magnitude modes as trans/rot) only if
        g98.out isn't available or fails to parse.

        With backend "axtb": axtb's analytic Hessian, with translations and rotations projected out
        in mass-weighted coordinates here (mepd.discovery.vri.projected_frequencies; 3N-6 modes,
        frequencies as the program's to ~0.2 cm^-1). Not axtb's own analysis: it starts rayon's
        global thread pool, and a forked child (mepd forks workers) then deadlocks in it.
        """
        natoms = len(node.symbols)
        if self.backend == "axtb":
            from mepd.discovery.vri import node_masses, projected_frequencies
            from mepd.engines import axtb_backend

            hessian = np.asarray(axtb_backend.hessian(node, **self._axtb_kwds())["hessian"], dtype=float)
            modes = projected_frequencies(hessian, np.asarray(node.coords, dtype=float).reshape(natoms, 3),
                                          node_masses(node))
            return FiniteDifferenceHessianOutput(
                input_data=SimpleNamespace(structure=node.structure),
                results=FiniteDifferenceHessianResults(
                    hessian=hessian,
                    normal_modes_cartesian=[np.asarray(m, dtype=float).reshape(natoms, 3) for m in modes.modes_cart],
                    freqs_wavenumber=[float(f) for f in modes.freqs],
                ),
                success=True,
            )
        with tempfile.TemporaryDirectory(prefix="gxtb-hess-") as tmp:
            workdir = Path(tmp)
            xyz_path = workdir / "structure.xyz"
            xyz_path.write_text(node.structure.to_xyz())
            completed = self._run_gxtb_hessian(
                xyz_path=xyz_path,
                charge=int(node.structure.charge),
                multiplicity=int(node.structure.multiplicity),
                cwd=workdir,
            )
            try:
                hessian = self._parse_hessian(workdir / "hessian", natoms=natoms)
            except Exception as exc:
                raise ElectronicStructureError(
                    msg="Failed to parse g-xTB Hessian output.",
                    obj=completed.stdout + completed.stderr,
                ) from exc

            g98_path = workdir / "g98.out"
            freqs = modes = None
            if g98_path.exists():
                try:
                    freqs, modes = self._parse_g98_frequencies_and_modes(
                        g98_path.read_text(), natoms=natoms
                    )
                except Exception:
                    freqs = modes = None

            if self.keep_workdirs:
                persistent = Path.cwd() / "gxtb-workdirs"
                persistent.mkdir(exist_ok=True)
                shutil.copytree(workdir, persistent / workdir.name, dirs_exist_ok=True)

        if freqs and modes and len(freqs) == len(modes):
            return FiniteDifferenceHessianOutput(
                input_data=SimpleNamespace(structure=node.structure),
                results=FiniteDifferenceHessianResults(
                    hessian=hessian,
                    normal_modes_cartesian=modes,
                    freqs_wavenumber=freqs,
                ),
                success=True,
            )

        # Fallback: g-xTB's own frequency analysis wasn't available/parseable.
        return build_hessian_result_from_matrix(node=node, hessian=hessian)

    def _run_gxtb_hessian(
        self,
        *,
        xyz_path: Path,
        charge: int,
        multiplicity: int,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        self._resolve_executable()
        cmd = [
            self.executable,
            str(xyz_path.name),
            "--silent",
            "--chrg",
            str(charge),
            "--hess",
            "--acc",
            str(self.hessian_acc),
        ]
        uhf = max(0, int(multiplicity) - 1)
        if uhf:
            cmd.extend(["--uhf", str(uhf)])
        if self.add_gxtb_flag and not standalone_gxtb(self.executable):
            cmd.append("--gxtb")
        cmd.extend(self.extra_args)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(self.n_threads))
        env.update(self.env)
        completed = _run_gxtb_process(cmd, cwd=cwd, env=env, timeout_s=self.timeout_s)
        if completed.returncode != 0:
            raise ElectronicStructureError(
                msg=f"g-xTB Hessian calculation failed with exit code {completed.returncode}.",
                obj=completed.stdout + completed.stderr,
            )
        self._check_ran_gxtb(completed)
        return completed

    def _check_ran_gxtb(self, completed: subprocess.CompletedProcess[str]) -> None:
        """A run meant as g-xTB whose xtb reports a GFN Hamiltonian ran the wrong
        method (e.g. an xtb build not given `--gxtb`, mepd#4): stop, don't return it."""
        if not self.add_gxtb_flag:
            return
        found = _GFN_HAMILTONIAN.search(completed.stdout)
        if found:
            raise ElectronicStructureError(
                msg=f"{self.executable} ran {found.group(1)}, not g-xTB: give it the g-xTB build of xtb, "
                    "or set add_gxtb_flag = false for GFN-xTB.",
                obj=completed.stdout + completed.stderr,
            )

    def compute_geometry_optimization(
        self,
        node: StructureNode,
        keywords: dict[str, Any] | None = None,
    ) -> list[StructureNode]:
        """Minimize to a largest atomic force of `fmax` (eV/Angstrom; default
        `min_fmax`) with `optimizer` (default `minimizer`): "qcopt" (AFH's
        optimizer: TRIC coordinates, Lindh's model Hessian, or g-xTB's exact
        one renewed every `hessian_every` steps), "sella" (Sella's minimizer;
        g-xTB's exact Hessian only with `hessian_every`), "geometric"
        (geomeTRIC, TRIC coordinates) or "lbfgs"
        (LBFGSLineSearch). With `minimizer="xtb"` and no `fmax`/`optimizer`,
        g-xTB's own optimizer (ANCopt, "normal" thresholds). Either way an
        unconverged run raises."""
        kwds = dict(keywords or {})
        if "fmax" in kwds or "optimizer" in kwds or (self.minimizer or self.DEFAULT_MINIMIZER) != "xtb":
            return self._minimize_to_fmax(node, kwds)
        extra_args = list(self.extra_args)
        maxiter = kwds.pop("maxiter", kwds.pop("maxit", kwds.pop("steps", None)))
        if maxiter is not None:
            extra_args.extend(["--cycles", str(int(maxiter))])

        with tempfile.TemporaryDirectory(prefix="gxtb-opt-") as tmp:
            workdir = Path(tmp)
            xyz_path = workdir / "structure.xyz"
            xyz_path.write_text(node.structure.to_xyz())
            from mepd import progress as _progress

            sink = _progress.minimization_sink()
            watch = None
            if sink is not None:
                natoms = len(node.symbols)

                def watch(final: bool = False) -> None:
                    energies, frames = _read_xtbopt_progress(workdir / "xtbopt.log", natoms)
                    if energies:
                        sink(energies, frames, final=final)

            completed = self._run_gxtb(
                xyz_path=xyz_path,
                charge=int(node.structure.charge),
                multiplicity=int(node.structure.multiplicity),
                cwd=workdir,
                optimize=True,
                watch=watch,
                extra_args=extra_args,
            )
            if watch is not None:
                watch(final=True)  # the steps written after the last poll
            _check_gxtb_optimization_converged(completed.stdout, maxiter=maxiter)
            try:
                opt_nodes = self._parse_optimization_trajectory(
                    node=node,
                    fp=workdir / "xtbopt.log",
                )
                if not opt_nodes:
                    opt_nodes = [
                        self._parse_optimized_node(node=node, fp=workdir / "xtbopt.xyz")
                    ]
            except Exception as exc:
                raise ElectronicStructureError(
                    msg="Failed to parse g-xTB optimization output.",
                    obj=completed.stdout + completed.stderr,
                ) from exc

            final_result = self._compute_node(opt_nodes[-1])
            opt_nodes[-1]._cached_result = final_result
            opt_nodes[-1]._cached_energy = final_result.results.energy
            opt_nodes[-1]._cached_gradient = final_result.results.gradient

            if self.keep_workdirs:
                persistent = Path.cwd() / "gxtb-workdirs"
                persistent.mkdir(exist_ok=True)
                shutil.copytree(workdir, persistent / workdir.name, dirs_exist_ok=True)
        return opt_nodes

    def compute_geometry_optimizations(
        self,
        nodes: list[StructureNode],
        keywords: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[StructureNode]]:
        """Batch geometry optimizations, one g-xTB subprocess per node, up to
        `n_parallel` at a time.

        `progress_callback`, if given, is called as `progress_callback(completed,
        total)` after each node finishes, so per-candidate progress is
        available live, unlike a true remote batch backend.
        """
        lock = threading.Lock()
        done = [0]

        def _optimize(node):
            traj = self.compute_geometry_optimization(node=node, keywords=keywords)
            with lock:
                done[0] += 1
                if progress_callback is not None:
                    progress_callback(done[0], len(nodes))
            return traj

        kwds = dict(keywords or {})
        python_driven = ("fmax" in kwds or "optimizer" in kwds
                         or (self.minimizer or self.DEFAULT_MINIMIZER) != "xtb")
        width = min(len(nodes), int(self.n_parallel) or max(1, available_cpus() // max(1, int(self.n_threads))))
        if python_driven and width > 1 and progress_callback is None:
            # Optimizers driven from Python (Sella, LBFGS) as threads of one process share its interpreter
            # lock: 450 proposal minimizations on 48 threads kept 3-6 g-xTB runs going. Forked processes
            # each have their own (xtb's own optimizer runs in its process either way).
            from mepd.cli_common import _fork_map

            return _fork_map(lambda node: self.compute_geometry_optimization(node=node, keywords=keywords),
                             list(nodes), width)
        return self._map(_optimize, list(nodes))

    def _minimize_to_fmax(self, node: StructureNode, kwds: dict[str, Any]) -> list[StructureNode]:
        fmax = float(kwds.pop("fmax", self.min_fmax))
        steps = int(kwds.pop("maxiter", kwds.pop("maxit", kwds.pop("steps", 3000))))
        optimizer = str(kwds.pop("optimizer", self.minimizer or self.DEFAULT_MINIMIZER)).lower()
        if optimizer == "xtb":
            raise ValueError("optimizer='xtb' has no force threshold: leave out `fmax` to use xtb's own optimizer.")
        every = kwds.pop("hessian_every", None)
        if optimizer == "qcopt":
            from mepd.engines import _qcopt

            return _qcopt.minimize(self, node, fmax=fmax, maxiter=steps, hessian_every=every, **kwds)
        if optimizer == "geometric":
            from mepd.irc_geometric import minimize_with_geometric

            return minimize_with_geometric(self, node, fmax=fmax, maxiter=steps)
        ase_engine = self._as_ase_engine_for_node(node)
        if optimizer == "sella":
            from mepd.engines._sella import sella

            Sella = sella().Sella

            optimizer_kwds = {"order": 0, "hessian_function": self._ase_hessian_function(node)}
            if every:
                optimizer_kwds["diag_every_n"] = int(every)
            return ase_engine._run_ase_optimization(
                node=node, optimizer_cls=Sella, keywords={"fmax": fmax, "steps": steps, "optimizer_kwds": optimizer_kwds})
        if optimizer != "lbfgs":
            raise ValueError(f"Unknown optimizer '{optimizer}': qcopt, lbfgs, sella or geometric.")
        return ase_engine.compute_geometry_optimization(node, {"fmax": fmax, "steps": steps})

    def _map(self, fn, items: list) -> list:
        """`[fn(x) for x in items]`, with up to `n_parallel` running at once."""
        width = int(self.n_parallel) or max(1, available_cpus() // max(1, int(self.n_threads)))
        width = min(width, len(items))
        if width <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=width) as pool:
            return list(pool.map(fn, items))

    def _as_ase_engine_for_node(self, node: StructureNode):
        from mepd.engines.ase import ASEEngine

        calculator = _GXTBASEResultsCalculator(
            self,
            charge=int(node.structure.charge),
            multiplicity=int(node.structure.multiplicity),
        )
        return ASEEngine(calculator=calculator, irc_branches_at_once=True)

    def compute_transition_state(
        self,
        node: StructureNode,
        keywords: dict[str, Any] | None = None,
    ) -> StructureNode:
        """Optimize a TS guess with `ts_optimizer` ("sella" through ASE, or
        "qcopt") using local g-xTB energies/gradients.

        Either starts from g-xTB's own Hessian instead of estimating the
        curvature from gradients, and recomputes it every `hessian_every` steps
        (default: only at the start, and for qcopt when the updated Hessian
        loses its negative mode; 1 = every step). `exact_hessian=False` restores
        Sella's gradient-based estimate (qcopt: finite differences of
        gradients)."""
        kwds = dict(keywords or {})
        exact = kwds.pop("exact_hessian", True)
        every = kwds.pop("hessian_every", None)
        if self._ts_optimizer() == "qcopt":
            from mepd.engines import _qcopt
            from mepd.engines.ase import DEFAULT_FMAX

            fmax = float(kwds.pop("fmax", DEFAULT_FMAX))
            steps = int(kwds.pop("maxiter", kwds.pop("maxit", kwds.pop("steps", 500))))
            return _qcopt.transition_state(self, node, fmax=fmax, maxiter=steps, exact_hessian=exact,
                                           hessian_every=every, **kwds)[-1]
        if exact:
            optimizer_kwds = dict(kwds.pop("optimizer_kwds", kwds.pop("optimizer_kwargs", {})) or {})
            optimizer_kwds.setdefault("hessian_function", self._ase_hessian_function(node))
            if every:
                optimizer_kwds.setdefault("diag_every_n", int(every))
            kwds["optimizer_kwds"] = optimizer_kwds
        return self._as_ase_engine_for_node(node).compute_transition_state(
            node=node,
            keywords=kwds,
        )

    def _ase_hessian_function(self, node: StructureNode):
        """Sella's `hessian_function`: g-xTB's Hessian at an ASE geometry, in eV/Angstrom^2."""
        from ase.units import Bohr, Hartree

        from mepd.qcdata_structure_helpers import ase_atoms_to_structure

        charge, multiplicity = int(node.structure.charge), int(node.structure.multiplicity)

        def hessian(atoms) -> NDArray:
            here = StructureNode(structure=ase_atoms_to_structure(atoms=atoms, charge=charge, multiplicity=multiplicity))
            return np.asarray(self.compute_hessian(here), dtype=float) * Hartree / Bohr**2

        return hessian

    def compute_irc_chain(
        self,
        ts_node: StructureNode,
        keywords: dict[str, Any] | None = None,
    ) -> Chain:
        """Compute an IRC with `ts_optimizer` ("sella" through ASE, or "qcopt"
        from g-xTB's Hessian at the TS) using local g-xTB energies/gradients."""
        if self._ts_optimizer() == "qcopt":
            from mepd.engines import _qcopt

            return _qcopt.irc(self, ts_node, keywords)
        return self._as_ase_engine_for_node(ts_node).compute_irc_chain(
            ts_node=ts_node,
            keywords=keywords,
        )

    @staticmethod
    def _parse_energy(*, workdir: Path, stdout: str) -> float:
        if "CPCM-X post-SCF" in stdout:
            # With --cpcmx the `energy` file holds the conductor-like (COSMO,
            # epsilon = infinity) SCF energy the CPCM-X evaluation starts
            # from: the same for every solvent. The CPCM-X total is printed.
            matches = _TOTAL_ENERGY_RE.findall(stdout)
            if matches:
                return float(matches[-1])
        energy_fp = workdir / "energy"
        if energy_fp.exists():
            for line in energy_fp.read_text().splitlines():
                fields = line.split()
                if len(fields) >= 2 and fields[0].isdigit():
                    return float(fields[1])

        matches = _TOTAL_ENERGY_RE.findall(stdout)
        if matches:
            return float(matches[-1])
        raise ValueError("Could not find total energy in g-xTB output.")

    @staticmethod
    def _parse_gradient(fp: Path, natoms: int) -> NDArray:
        if not fp.exists():
            raise FileNotFoundError(f"g-xTB gradient file not found: {fp}")
        rows = []
        for raw_line in fp.read_text().splitlines():
            fields = raw_line.split()
            if len(fields) != 3:
                continue
            try:
                rows.append([float(value.replace("D", "E")) for value in fields])
            except ValueError:
                continue
        if len(rows) < natoms:
            raise ValueError(f"Expected at least {natoms} gradient rows, found {len(rows)}.")
        return np.asarray(rows[-natoms:], dtype=float)

    @staticmethod
    def _parse_hessian(fp: Path, natoms: int) -> NDArray:
        if not fp.exists():
            raise FileNotFoundError(f"g-xTB Hessian file not found: {fp}")
        ndof = int(natoms) * 3
        values: list[float] = []
        in_block = False
        for raw_line in fp.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("$hessian"):
                in_block = True
                continue
            if lower.startswith("$end"):
                break
            if not in_block and stripped.startswith("$"):
                continue
            if not in_block:
                continue
            for field in stripped.split():
                try:
                    values.append(float(field.replace("D", "E")))
                except ValueError:
                    continue
        expected = ndof * ndof
        if len(values) < expected:
            raise ValueError(
                f"Expected {expected} Hessian values for {natoms} atoms, found {len(values)}."
            )
        hessian = np.asarray(values[:expected], dtype=float).reshape((ndof, ndof))
        return 0.5 * (hessian + hessian.T)

    @staticmethod
    def _parse_g98_frequencies_and_modes(
        text: str, natoms: int
    ) -> tuple[list[float], list[NDArray]]:
        """Parse g-xTB's Gaussian-98-format frequency output (g98.out).

        Unlike the raw `hessian` file, this already reports g-xTB's own
        *projected* vibrational frequencies (translation/rotation removed)
        plus real per-atom normal-mode displacement vectors -- modes are
        printed in blocks of up to 3, each block starting with a
        "Frequencies --" line followed by per-atom XYZ displacement triplets.
        """
        lines = text.splitlines()
        freqs: list[float] = []
        mode_rows: list[list[list[float]]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("Frequencies --"):
                block_freqs = [float(v) for v in line.split("--", 1)[1].split()]
                n_in_block = len(block_freqs)
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("Atom"):
                    j += 1
                j += 1  # skip the "Atom AN X Y Z ..." header itself
                block_rows = [[] for _ in range(n_in_block)]
                for a in range(natoms):
                    values = [float(v) for v in lines[j + a].split()[2:]]
                    for m in range(n_in_block):
                        block_rows[m].append(values[3 * m: 3 * m + 3])
                freqs.extend(block_freqs)
                mode_rows.extend(block_rows)
                i = j + natoms
            else:
                i += 1

        modes = [np.asarray(rows, dtype=float) for rows in mode_rows]
        return freqs, modes

    @staticmethod
    def _parse_optimized_node(node: StructureNode, fp: Path) -> StructureNode:
        nodes = GXTBCalculator._parse_optimization_trajectory(node=node, fp=fp)
        if not nodes:
            raise ValueError(f"No optimized geometry found in {fp}.")
        return nodes[-1]

    @staticmethod
    def _parse_optimization_trajectory(node: StructureNode, fp: Path) -> list[StructureNode]:
        if not fp.exists():
            raise FileNotFoundError(f"g-xTB optimization file not found: {fp}")
        lines = fp.read_text().splitlines()
        nodes: list[StructureNode] = []
        i = 0
        natoms_expected = len(node.symbols)
        while i < len(lines):
            try:
                natoms = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
            if natoms != natoms_expected or i + natoms + 1 >= len(lines):
                i += 1
                continue
            comment = lines[i + 1]
            coords_angstrom = []
            symbols = []
            for raw_line in lines[i + 2 : i + 2 + natoms]:
                fields = raw_line.split()
                if len(fields) < 4:
                    coords_angstrom = []
                    break
                symbols.append(fields[0])
                coords_angstrom.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if len(coords_angstrom) == natoms:
                new_node = node.update_coords(np.asarray(coords_angstrom, dtype=float) * ANGSTROM_TO_BOHR)
                energy_match = re.search(r"energy:\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", comment)
                if energy_match:
                    energy = float(energy_match.group(1))
                    result = FakeQCIOOutput.model_validate(
                        {"results": FakeQCIOResults.model_validate({"energy": energy, "gradient": np.zeros_like(new_node.coords)})}
                    )
                    new_node._cached_result = result
                    new_node._cached_energy = energy
                    new_node._cached_gradient = np.zeros_like(new_node.coords)
                nodes.append(new_node)
            i += natoms + 2
        return nodes
