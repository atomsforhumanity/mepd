"""g-xTB through axtb, in this process.

axtb (github.com/atomsforhumanity/axtb) is AFH's Rust reimplementation of g-xTB 2.0.1. Called here,
an energy and gradient needs no g-xTB program start, input files or output parsing, and Hessians are
analytic (the program differentiates gradients numerically, at `hessian_acc`). axtb reproduces the
program's numbers: with conventions="gxtb" total energies to <= 2e-12 Eh, with its default
conventions ("axtb": CODATA 2022 units, integer electron count, smooth cutoffs) within ~1e-9 Eh on
small molecules (axtb's README and bench/ report).

`GXTBCalculator(backend="axtb")` routes its energies, gradients and Hessians here; minimizations, TS
optimizations and IRCs are the same Python drivers (Sella, geomeTRIC) either way.
"""

from __future__ import annotations

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

_LOCAL = threading.local()


def settings_from_args(extra_args) -> dict[str, Any]:
    """axtb Calculator keywords for the xtb options in `extra_args` that axtb takes (--etemp K,
    --acc X, --iterations N). Any other option raises ValueError: axtb would not apply it."""
    out: dict[str, Any] = {}
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


def _import():
    try:
        import axtb
    except ImportError as exc:
        from mepd.errors import ElectronicStructureError

        raise ElectronicStructureError(
            msg="backend = \"axtb\" needs the axtb Python package (`pip install axtb`, or mepd's gxtb "
                "extra).") from exc
    return axtb


def _calculator(numbers: tuple, positions: np.ndarray, charge: int, uhf: int | None, threads: int,
                conventions: str | None, settings: dict):
    """This thread's axtb Calculator for one molecule and settings (made once, then reused: g-xTB
    starts every SCF from the program's own guess, so a reused one gives the same numbers as a new
    one). A forked child makes its own."""
    cache = _LOCAL.__dict__.setdefault("cache", {})
    if _LOCAL.__dict__.get("pid") != os.getpid():
        cache.clear()
        _LOCAL.pid = os.getpid()
    key = (numbers, charge, uhf, threads, conventions, tuple(sorted(settings.items())))
    calc = cache.get(key)
    if calc is None:
        axtb = _import()
        kwds = dict(settings)
        if conventions:
            kwds["conventions"] = conventions
        calc = axtb.Calculator(METHOD, list(numbers), positions, charge=float(charge), uhf=uhf,
                               threads=int(threads), **kwds)
        cache[key] = calc
        if len(cache) > 16:
            cache.pop(next(iter(cache)))
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


def point(node, *, threads: int = 1, conventions: str | None = None, settings: dict | None = None,
          molden: bool = False) -> dict:
    """Energy (Eh) and gradient (Eh/bohr, (N, 3)) of `node`. An SCF that does not converge is
    retried with the program's ladder (FALLBACKS), unless the settings fix an electronic
    temperature; each retry is logged. With `molden`, the result's "molden" is the orbitals' file."""
    from mepd.errors import ElectronicStructureError

    settings = dict(settings or {})
    numbers, positions, charge, uhf = _molecule(node)
    ladder = [settings]
    if "etemp" not in settings:
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
        _log("--grad", t0, c0, True, {"retries": k} if k else None)
        out = {"energy": float(res["energy"]), "gradient": np.asarray(res["gradient"], dtype=float)}
        if molden:
            out["molden"] = res.get("molden")
        return out
    _log("--grad", t0, c0, False)
    raise ElectronicStructureError(msg="axtb g-xTB calculation failed: " + " | ".join(errors))


def hessian(node, *, threads: int = 1, conventions: str | None = None, settings: dict | None = None) -> dict:
    """axtb's analytic g-xTB Hessian of `node` ("hessian", (3N, 3N) Eh/bohr^2, not projected), with
    the energy and gradient. No SCF retry ladder, as for the program's --hess.

    Without axtb's vibrational analysis (vibrations=False): that multiplies matrices on rayon's
    global thread pool (axtb 59cc1a4, vibrations::modes_axtb), which a forked child inherits
    without its threads and waits on forever."""
    from mepd.errors import ElectronicStructureError

    numbers, positions, charge, uhf = _molecule(node)
    t0, c0 = time.time(), time.thread_time()
    try:
        calc = _calculator(numbers, positions, charge, uhf, threads, conventions, dict(settings or {}))
        res = calc.hessian(positions, vibrations=False)
    except Exception as exc:
        _log("--hess", t0, c0, False)
        raise ElectronicStructureError(msg=f"axtb g-xTB Hessian failed: {exc}") from exc
    _log("--hess", t0, c0, True)
    return res
