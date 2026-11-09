"""g-xTB through axtb, in this process.

axtb (github.com/atomsforhumanity/axtb) is AFH's Rust reimplementation of g-xTB 2.0.1. Called here,
an energy and gradient needs no g-xTB program start, input files or output parsing, and Hessians are
analytic (the program differentiates gradients numerically, at `hessian_acc`). axtb reproduces the
program's numbers: with conventions="gxtb" total energies to <= 2e-12 Eh, with its default
conventions ("axtb": CODATA 2022 units, integer electron count, smooth cutoffs) within ~1e-9 Eh on
small molecules (axtb's README and bench/ report).

`GXTBCalculator(backend="axtb")` routes its energies, gradients and Hessians here; minimizations, TS
optimizations and IRCs are the same Python drivers (Sella, geomeTRIC) either way.

Also here, each in one in-process call (mepd#40, mepd#41):
* g-xTB in implicit solvent, the composite E_gxtb + [E_GFN2+ALPB/GBSA - E_GFN2] (mepd.solvation),
  with analytic Hessians (`solvation=`; the program route runs three xtb processes per gradient
  and differentiates the composite numerically);
* the GFN2-xTB solvation pair alone, the correction for any other engine (`solvation_pair`);
* GFN-FF, the QM/MM environment's default force field (`gfnff_point`, `gfnff_hessian`): no xtb
  process or topology file per call, and none of xtb's memory ceiling (xtb allocates
  N_AB^2 x N_H for hydrogen bonds: 1700 waters need 37 GiB there, 0.5 GiB here).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

METHOD = "gxtb-v2.0.1"
# The program's retry ladder for an SCF that does not converge (gxtb.SCF_FALLBACKS): Fermi
# smearing at 1000 K, then 3000 K, with up to 1000 iterations.
FALLBACKS = ({"etemp": 1000.0, "max_iter": 1000}, {"etemp": 3000.0, "max_iter": 1000})
# xtb command-line options axtb takes (as Calculator keywords); anything else raises.
_OPTIONS = {"--etemp": ("etemp", float), "--acc": ("accuracy", float), "--iterations": ("max_iter", int)}
# SCF accuracy (xtb --acc) unless extra_args set one. At the program's default (1.0) the SCF stops loosely
# enough that its gradients can be off by ~1e-4 Eh/bohr, five times the minimizations' 1e-3 eV/Angstrom
# (1.9e-5 Eh/bohr), on structures far from equilibrium: qcopt then never reached it on 35 of 31k IRC ends
# (methyl formate, scoreboard sb-t11). At 0.01 the error is ~1e-6 for ~4 more SCF iterations (9 -> 13);
# qcopt's benchmarks are at 0.01 too.
ACCURACY = 0.01

_LOCAL = threading.local()


def settings_from_args(extra_args) -> dict[str, Any]:
    """axtb Calculator keywords for the xtb options in `extra_args` that axtb takes (--etemp K,
    --acc X, --iterations N); the accuracy is ACCURACY unless --acc sets it. Any other option raises
    ValueError: axtb would not apply it."""
    out: dict[str, Any] = {"accuracy": ACCURACY}
    args = [str(a) for a in extra_args]
    k = 0
    while k < len(args):
        name = args[k]
        if name in _OPTIONS and k + 1 < len(args):
            key, conv = _OPTIONS[name]
            out[key] = conv(args[k + 1])
            k += 2
            continue
        raise ValueError(f"backend = \"axtb\" does not take the xtb option {name!r} (extra_args); it takes "
                         f"{', '.join(_OPTIONS)}.")
    return out


def version() -> str | None:
    try:
        import axtb
    except ImportError:
        return None
    return str(axtb.__version__)


_FEATURES: dict = {}


def supports(feature: str) -> bool:
    """Whether the installed axtb can do `feature`: "composite" (g-xTB in implicit solvent,
    `solvation_correction`; also taken as the sign of the 0.1.0 release mepd was checked with, for
    the GFN2 solvation pair) or "gfnff" (GFN-FF with xtb 6.7.1's conventions). Older builds (the
    image's former pin 59cc1a4 had neither) make mepd use the xtb program for those. Probed once,
    on H2."""
    if feature not in ("composite", "gfnff"):
        raise ValueError(f"unknown axtb feature {feature!r}")
    if feature not in _FEATURES:
        ok = False
        if version() is not None:
            import axtb

            h2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
            try:
                if feature == "composite":
                    axtb.Calculator(METHOD, [1, 1], h2, alpb="water", solvation_correction="gfn2")
                else:
                    axtb.Calculator("gfnff", [1, 1], h2, conventions=GFNFF_CONVENTIONS, topology_positions=h2)
                ok = True
            except Exception:          # an unknown keyword, method or conventions: an older build
                ok = False
        _FEATURES[feature] = ok
    return _FEATURES[feature]


def _import():
    try:
        import axtb
    except ImportError as exc:
        from mepd.errors import ElectronicStructureError

        raise ElectronicStructureError(
            msg="backend = \"axtb\" needs the axtb Python package (`pip install axtb`, or mepd's gxtb "
                "extra).") from exc
    return axtb


def _thread_cache() -> dict:
    """This thread's calculators (an axtb Calculator serves one call at a time: a second thread
    using it gets "Already borrowed"). A forked child starts with none of its parent's."""
    cache = _LOCAL.__dict__.setdefault("cache", {})
    if _LOCAL.__dict__.get("pid") != os.getpid():
        cache.clear()
        _LOCAL.pid = os.getpid()
    return cache


def _remember(cache: dict, key, calc, limit: int = 16) -> None:
    cache[key] = calc
    while len(cache) > limit:
        cache.pop(next(iter(cache)))


def _calculator(numbers: tuple, positions: np.ndarray, charge: int, uhf: int | None, threads: int,
                conventions: str | None, settings: dict):
    """This thread's axtb Calculator for one molecule and settings (made once, then reused: g-xTB
    starts every SCF from the program's own guess, so a reused one gives the same numbers as a new
    one). A forked child makes its own."""
    cache = _thread_cache()
    key = (numbers, charge, uhf, threads, conventions, tuple(sorted(settings.items())))
    calc = cache.get(key)
    if calc is None:
        axtb = _import()
        kwds = dict(settings)
        if conventions:
            kwds["conventions"] = conventions
        calc = axtb.Calculator(METHOD, list(numbers), positions, charge=float(charge), uhf=uhf,
                               threads=int(threads), **kwds)
        _remember(cache, key, calc)
    return calc


def _molecule(node) -> tuple[tuple, np.ndarray, int, int | None]:
    numbers = tuple(int(z) for z in node.structure.atomic_numbers)
    positions = np.ascontiguousarray(node.structure.geometry, dtype=float)     # bohr
    multiplicity = int(node.structure.multiplicity)
    # As the program is run: --uhf only for open shells.
    uhf = multiplicity - 1 if multiplicity > 1 else None
    return numbers, positions, int(node.structure.charge), uhf


def _scf_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return "scf" in text or "converge" in text


def _log(kind: str, t0: float, c0: float, ok: bool, extra: dict | None = None) -> None:
    """With $MEPD_CALL_LOG set (a folder), one JSON line per call in <folder>/<pid>.jsonl, as the
    program runs log theirs (gxtb._run_gxtb_process)."""
    folder = os.environ.get("MEPD_CALL_LOG")
    if not folder:
        return
    rec = {"t": round(t0, 3), "wall": round(time.time() - t0, 5), "cpu": round(time.thread_time() - c0, 5),
           "ok": ok, "flags": ["--axtb", kind], **(extra or {})}
    try:
        with open(Path(folder) / f"{os.getpid()}.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def solvation_settings(solvent: str, model: str) -> dict:
    """Calculator keywords for g-xTB in `solvent` through the GFN2-xTB composite (`model` "alpb" or
    "gbsa"; solvent by xtb's name, mepd.solvation.SOLVENTS' keys)."""
    model = str(model).lower()
    if model not in ("alpb", "gbsa"):
        raise ValueError(f"axtb's solvation composite takes ALPB or GBSA, not {model!r}")
    return {model: str(solvent), "solvation_correction": "gfn2"}


def _warn_rungs(res: dict, what: str) -> dict | None:
    """The retry rungs a composite's parts converged on ({part: rung}), logged; None if none."""
    rungs = {name: part.get("retry") for name, part in (res.get("parts") or {}).items() if part.get("retry")}
    if rungs:
        import logging

        logging.getLogger(__name__).warning(
            "axtb %s: an SCF converged only on a retry rung (%s); this structure's energy is from a smeared "
            "or longer SCF", what, ", ".join(f"{k}: {v}" for k, v in rungs.items()))
    return rungs or None


def point(node, *, threads: int = 1, conventions: str | None = None, settings: dict | None = None,
          molden: bool = False, solvation: dict | None = None) -> dict:
    """Energy (Eh) and gradient (Eh/bohr, (N, 3)) of `node`. An SCF that does not converge is
    retried with the program's ladder (FALLBACKS), unless the settings fix an electronic
    temperature; each retry is logged. With `molden`, the result's "molden" is the orbitals' file.

    `solvation` ({"solvent", "model"}): g-xTB in that implicit solvent, the composite
    E_gxtb + [E_GFN2+model - E_GFN2] in one calculation (mepd#40). axtb runs the retry ladders of
    both parts itself (g-xTB's FALLBACKS; the GFN2 pair's solvation.SolvationCorrection.SCC_FALLBACKS,
    both halves alike), stops on a first geometry whose dG_solv is zero, and its conventions follow
    `conventions` ("gxtb": the three programs mepd ran before, rung for rung)."""
    from mepd.errors import ElectronicStructureError

    settings = dict(settings or {})
    if solvation:
        settings.update(solvation_settings(solvation["solvent"], solvation.get("model", "alpb")))
    numbers, positions, charge, uhf = _molecule(node)
    ladder = [settings]
    if "etemp" not in settings and "solvation_correction" not in settings:
        ladder += [{**settings, **fb} for fb in FALLBACKS]
    t0, c0 = time.time(), time.thread_time()
    errors = []
    for k, st in enumerate(ladder):
        try:
            calc = _calculator(numbers, positions, charge, uhf, threads, conventions, st)
            res = calc.singlepoint(positions, gradient=True, molden=True if molden else None)
        except Exception as exc:          # axtb raises on SCF failure (it never returns an unconverged result)
            errors.append(f"{st or 'defaults'}: {exc}")
            if not _scf_failure(exc):
                break
            continue
        if k:
            import logging

            logging.getLogger(__name__).warning(
                "axtb g-xTB SCF did not converge; converged with etemp %s K, %s iterations", st["etemp"],
                st["max_iter"])
        rungs = _warn_rungs(res, "g-xTB in solvent") if solvation else None
        _log("--grad", t0, c0, True, {"retries": k} if k else ({"rungs": rungs} if rungs else None))
        out = {"energy": float(res["energy"]), "gradient": np.asarray(res["gradient"], dtype=float)}
        if solvation:
            out["components"] = dict(res.get("components") or {})
            out["rungs"] = rungs
        if molden:
            out["molden"] = res.get("molden")
        return out
    _log("--grad", t0, c0, False)
    raise ElectronicStructureError(msg="axtb g-xTB calculation failed: " + " | ".join(errors))


def hessian(node, *, threads: int = 1, conventions: str | None = None, settings: dict | None = None,
            solvation: dict | None = None) -> dict:
    """axtb's analytic g-xTB Hessian of `node` ("hessian", (3N, 3N) Eh/bohr^2, not projected), with
    the energy and gradient. No SCF retry ladder, as for the program's --hess. With `solvation`
    (see `point`): the composite's, H_gxtb + H_GFN2+model - H_GFN2, every part analytic.

    Without axtb's vibrational analysis (vibrations=False): that multiplies matrices on rayon's
    global thread pool (axtb 59cc1a4, vibrations::modes_axtb), which a forked child inherits
    without its threads and waits on forever."""
    from mepd.errors import ElectronicStructureError

    settings = dict(settings or {})
    if solvation:
        settings.update(solvation_settings(solvation["solvent"], solvation.get("model", "alpb")))
    numbers, positions, charge, uhf = _molecule(node)
    t0, c0 = time.time(), time.thread_time()
    try:
        calc = _calculator(numbers, positions, charge, uhf, threads, conventions, settings)
        res = calc.hessian(positions, vibrations=False)
    except Exception as exc:
        _log("--hess", t0, c0, False)
        raise ElectronicStructureError(msg=f"axtb g-xTB Hessian failed: {exc}") from exc
    if solvation:
        _warn_rungs({"parts": res.get("hessian_parts")}, "g-xTB in solvent, Hessian")
    _log("--hess", t0, c0, True)
    return res


# ------------------------------------------------------------ the GFN2-xTB solvation pair
def _gfn2_calculator(numbers: tuple, positions: np.ndarray, charge: int, uhf: int | None, threads: int,
                     conventions: str | None, settings: dict):
    """This thread's GFN2-xTB Calculator for one molecule and settings (cached as `_calculator`)."""
    cache = _thread_cache()
    key = ("gfn2", numbers, charge, uhf, threads, conventions, tuple(sorted(settings.items())))
    calc = cache.get(key)
    if calc is None:
        axtb = _import()
        kwds = dict(settings)
        if conventions:
            kwds["conventions"] = conventions
        calc = axtb.Calculator("gfn2", list(numbers), positions, charge=float(charge), uhf=uhf,
                               threads=int(threads), **kwds)
        _remember(cache, key, calc)
    return calc


def solvation_pair(node, *, solvent: str, model: str = "alpb", ladder=({},), threads: int = 1,
                   conventions: str | None = None, accuracy: float = ACCURACY) -> tuple[float, np.ndarray, int]:
    """(dG_solv in Eh, its gradient in Eh/bohr, the ladder rung it converged on) of `node`: GFN2-xTB
    with `model` in `solvent` minus GFN2-xTB in the gas phase, both halves with the same settings.
    `ladder`: Calculator settings tried in turn while an SCF does not converge (both halves rerun).
    Raises mepd's ElectronicStructureError when every rung fails or axtb refuses the input."""
    from mepd.errors import ElectronicStructureError

    numbers, positions, charge, uhf = _molecule(node)
    solv = {str(model).lower(): str(solvent)}
    t0, c0 = time.time(), time.thread_time()
    errors = []
    for k, rung in enumerate(ladder):
        st = {"accuracy": float(accuracy), **dict(rung)}
        try:
            gas = _gfn2_calculator(numbers, positions, charge, uhf, threads, conventions, st)
            sol = _gfn2_calculator(numbers, positions, charge, uhf, threads, conventions, {**st, **solv})
            r_gas = gas.singlepoint(positions, gradient=True)
            r_sol = sol.singlepoint(positions, gradient=True)
        except Exception as exc:
            errors.append(f"{rung or 'defaults'}: {exc}")
            if not _scf_failure(exc):
                break
            continue
        _log("--gfn2-pair", t0, c0, True, {"retries": k} if k else None)
        return (float(r_sol["energy"]) - float(r_gas["energy"]),
                np.asarray(r_sol["gradient"], dtype=float) - np.asarray(r_gas["gradient"], dtype=float), k)
    _log("--gfn2-pair", t0, c0, False)
    raise ElectronicStructureError(msg="axtb GFN2-xTB solvation pair failed: " + " | ".join(errors))


# ------------------------------------------------------------ GFN-FF
#: axtb's conventions for GFN-FF unless told otherwise: xtb 6.7.1's force field, the one mepd ran
#: through the xtb program (the official 6.7.1 release in the image, and the g-xTB 2.0.1 build).
#: "axtb" / "xtb" are xtb 6.8.0's (lanthanides and actinides reparametrized, topology changes).
GFNFF_CONVENTIONS = "xtb-6.7.1"


def gfnff_memory_gb(natoms: int) -> float:
    """Peak memory of one axtb GFN-FF energy and gradient (measured, axtb 0.1.0: 10 MiB at 300
    atoms, 189 MiB at 3000, 460 MiB at 5100, 1.5 GiB at 9900)."""
    return 0.02 + 1.6e-8 * natoms ** 2


def _gfnff_key(numbers, charge: int, reference: np.ndarray | None, conventions: str, threads: int):
    import hashlib

    h = hashlib.sha1(np.asarray(numbers, dtype=np.int64).tobytes())
    h.update(str(int(charge)).encode())
    if reference is not None:
        h.update(np.ascontiguousarray(reference, dtype=float).tobytes())
    return ("gfnff", h.hexdigest(), conventions, int(threads))


# GFN-FF calculators by topology, shared by every thread of this process: each caller borrows an
# idle one (an axtb Calculator serves one call at a time) and puts it back, so a topology is set up
# once per concurrent caller, also from the short-lived threads of a pool made per batch. GFN-FF
# has no SCF, so which calculator a geometry gets changes nothing in its numbers.
_GFNFF_POOL: dict = {}
_GFNFF_GUARD = threading.Lock()
_GFNFF_PID = [os.getpid()]
_GFNFF_TOPOLOGIES = 8          # topologies kept (least recently used dropped)


@contextlib.contextmanager
def _gfnff_calculator(numbers, positions, charge: int, reference, conventions: str | None, threads: int):
    """A GFN-FF Calculator with the topology of `reference` (bohr), borrowed for one call; with no
    reference, a new one with the topology of `positions` (as xtb without a gfnff_topo)."""
    conventions = conventions or GFNFF_CONVENTIONS
    if reference is None:
        axtb = _import()
        yield axtb.Calculator("gfnff", [int(z) for z in numbers], positions, charge=float(charge),
                              threads=int(threads), conventions=conventions)
        return
    key = _gfnff_key(numbers, charge, reference, conventions, threads)
    with _GFNFF_GUARD:
        if _GFNFF_PID[0] != os.getpid():         # a forked child: none of the parent's
            _GFNFF_POOL.clear()
            _GFNFF_PID[0] = os.getpid()
        idle = _GFNFF_POOL.pop(key, [])
        calc = idle.pop() if idle else None
        _GFNFF_POOL[key] = idle                  # most recently used last
    if calc is None:
        axtb = _import()
        calc = axtb.Calculator("gfnff", [int(z) for z in numbers], np.ascontiguousarray(reference, dtype=float),
                               charge=float(charge), threads=int(threads), conventions=conventions)
    try:
        yield calc
    finally:
        with _GFNFF_GUARD:
            if _GFNFF_PID[0] == os.getpid():
                _GFNFF_POOL.setdefault(key, []).append(calc)
                while len(_GFNFF_POOL) > _GFNFF_TOPOLOGIES:
                    _GFNFF_POOL.pop(next(iter(_GFNFF_POOL)))


def gfnff_point(numbers, positions, *, charge: int = 0, reference=None, conventions: str | None = None,
                threads: int = 1) -> tuple[float, np.ndarray]:
    """GFN-FF energy (Eh) and gradient (Eh/bohr, (N, 3)) at `positions` (bohr), with the topology of
    `reference` (bohr; None: of `positions`, as xtb run without a gfnff_topo)."""
    from mepd.errors import ElectronicStructureError

    positions = np.ascontiguousarray(positions, dtype=float).reshape(-1, 3)
    t0, c0 = time.time(), time.thread_time()
    try:
        with _gfnff_calculator(numbers, positions, charge, reference, conventions, threads) as calc:
            res = calc.singlepoint(positions, gradient=True)
    except Exception as exc:
        _log("--gfnff", t0, c0, False)
        raise ElectronicStructureError(msg=f"axtb GFN-FF failed: {exc}") from exc
    _log("--gfnff", t0, c0, True)
    return float(res["energy"]), np.asarray(res["gradient"], dtype=float).reshape(-1, 3)


def gfnff_hessian(numbers, positions, *, charge: int = 0, reference=None, conventions: str | None = None,
                  threads: int = 1) -> np.ndarray:
    """axtb's analytic GFN-FF Hessian ((3N, 3N) Eh/bohr^2, not projected) at `positions`, with the
    topology of `reference` (as `gfnff_point`). Memory: the matrix, 72 N^2 bytes, and axtb's
    work (3.1 GiB at 5100 atoms)."""
    from mepd.errors import ElectronicStructureError

    positions = np.ascontiguousarray(positions, dtype=float).reshape(-1, 3)
    t0, c0 = time.time(), time.thread_time()
    try:
        with _gfnff_calculator(numbers, positions, charge, reference, conventions, threads) as calc:
            res = calc.hessian(positions, vibrations=False)
    except Exception as exc:
        _log("--gfnff --hess", t0, c0, False)
        raise ElectronicStructureError(msg=f"axtb GFN-FF Hessian failed: {exc}") from exc
    _log("--gfnff --hess", t0, c0, True)
    return np.asarray(res["hessian"], dtype=float)
