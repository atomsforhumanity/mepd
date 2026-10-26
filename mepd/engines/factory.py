"""Getting an engine: by name, or from a profile file; and thread control."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = ["engine", "engine_from_profile", "set_threads"]

# engine name -> the RunInputs field holding its options
_OPTION_FIELDS = {
    "gxtb": "gxtb_engine_kwds",
    "xtb": "xtb_engine_kwds",
    "ase": "ase_engine_kwds",
    "mlip": "mlip_engine_kwds",
    "fairchem": "fairchem_engine_kwds",
    "psi4": "psi4_engine_kwds",
    "orca": "orca_engine_kwds",
    "terachem": "terachem_engine_kwds",
}


def engine(name: str = "gxtb", *, program: str | None = None, solvation: dict | None = None, **options: Any):
    """An engine by name, with its options as keyword arguments.

    `name`: "gxtb" (local g-xTB), "xtb" (GFN2/GFN1-xTB; `method=`), "mlip"
    (`model="aimnet2-rxn"` etc.; `mepd models` lists them), "fairchem", "ase"
    (`calculator="package.module:Class"`, `calculator_kwds={...}`), "psi4", "orca",
    "terachem", or "qccompute"/"chemcloud" (`program="psi4"` plus the program's
    keywords as options). `solvation`, e.g. {"solvent": "water"}, adds implicit
    solvent. These are the `engine_name` and `*_engine_kwds` of a profile file.

    >>> eng = mepd.engine("gxtb")
    >>> eng = mepd.engine("mlip", model="aimnet2-rxn")
    """
    from mepd.inputs import RunInputs

    kwds: dict[str, Any] = {"engine_name": name}
    if program is not None:
        kwds["program"] = program
    if solvation is not None:
        kwds["solvation"] = solvation
    if options:
        if name in _OPTION_FIELDS:
            kwds[_OPTION_FIELDS[name]] = options
        elif name in ("qccompute", "chemcloud"):
            kwds["program_kwds"] = options
        else:
            raise ValueError(f"Unknown engine '{name}'; known: {', '.join([*_OPTION_FIELDS, 'qccompute', 'chemcloud'])}.")
    return RunInputs(**kwds).engine


def engine_from_profile(path: str | Path):
    """The engine a profile file (.toml, as `mepd ... --inputs` takes) describes."""
    from mepd.inputs import RunInputs

    return RunInputs.open(path).engine


def set_threads(n: int) -> None:
    """Threads for numerical libraries in this process and in the programs it
    starts (OMP/MKL/OpenBLAS). Set it before the first calculation. An engine's
    own thread option (e.g. g-xTB's `n_threads`) still applies to that engine."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(int(n))
