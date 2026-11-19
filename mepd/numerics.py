"""Numerics epochs: what a result was computed with, per method (Jan's decision 3, 2026-11-17).

A result (a path search, a TS, a ledgered answer) may be reused only under the same numerics. mepd's numbers
change when a commit changes a method, not on every commit, so each method has an epoch here and a committed
checkout is identified by the epochs alone: results stay reusable across commits that change only speed, docs
or other methods. Uncommitted edits to mepd/ add "+<commit 12>.dirty.<diff sha256 8>", so a dirty checkout
never matches anything else.

**The rule.** A commit that changes the numbers a method produces (energies, paths, which TS or minimum is found,
a frequency check) bumps that method's epoch below, with a line in its history, and says so in CHANGELOG under a
**Numerics:** note. A change that leaves every number as it was (speed, logging, docs) bumps nothing.

    numerics_epoch("fsm_ts")   # 'energies=2,fneb=1,frequencies=2,gsm=1,...,axtb=0.1.0,gsm.build=...,qcopt=0.2.0'
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

# method -> epoch. History (newest last); bump on any change of numbers.
EPOCHS: dict[str, int] = {
    # 2: ce07475, an engine no longer returns an energy or gradient another level computed.
    "energies": 2,
    # 2: 44480c3, translations and rotations projected out instead of dropping the six smallest modes.
    "frequencies": 2,
    "interpolation": 1,
    "minimize": 1,
    "ts_search": 1,
    "irc": 1,
    # 2: 9eda03d, tangent without rigid motion; axtb's SCF failures caught.
    "neb": 2,
    "fneb": 1,
    "gsm": 1,
    # MLP-GI's geodesic path (mepd.mlp_geodesic): on the profile's engine, or on an ML potential (numerics_epoch's
    # mlp_model) whose model and package version are added to the string.
    "mlpgi": 1,
    "fsm_ts": 1,
    "msmep": 1,
}

# A method's numbers also depend on the methods it uses (its fallbacks included).
DEPENDS: dict[str, tuple[str, ...]] = {
    "minimize": ("energies", "frequencies"),            # saddle escapes follow the frequency check's modes
    "ts_search": ("energies", "frequencies"),
    "irc": ("energies", "minimize", "frequencies"),
    "neb": ("energies", "interpolation", "minimize"),   # the elementary-step check minimizes
    "fneb": ("energies", "interpolation"),
    "gsm": ("energies", "interpolation"),
    "mlpgi": ("energies", "interpolation", "minimize"), # the engine's surface or the final path on it; step check minimizes
    "fsm_ts": ("fneb", "ts_search", "minimize", "frequencies", "gsm", "neb"),   # GSM, then NEB, as fallbacks
    "msmep": ("neb", "fneb", "fsm_ts", "gsm", "minimize", "frequencies"),
}

# Programs whose versions set numbers mepd reports, by the method that calls them (until the stack's store
# identities record them per calculation).
_PROGRAMS: dict[str, tuple[str, ...]] = {
    "energies": ("axtb",),
    "minimize": ("qcopt",),
    "ts_search": ("qcopt",),
    "irc": ("qcopt",),
    "mlpgi": ("ase",),                                  # ASE's FIRE drives the path's optimization stages
}

# ML potential family (mepd.engines.mlip) -> the distribution whose version sets its numbers.
_MLP_PACKAGES: dict[str, str] = {
    "aimnet2": "aimnet",
    "ani": "torchani",
    "fairchem": "fairchem-core",
    "mace": "mace-torch",
    "orb": "orb-models",
}


def methods_of(*methods: str) -> list[str]:
    """The methods given and every method they use, sorted. Unknown names raise KeyError."""
    seen: set[str] = set()
    todo = list(methods)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        if m not in EPOCHS:
            raise KeyError(f"no numerics epoch for {m!r} (known: {', '.join(sorted(EPOCHS))})")
        seen.add(m)
        todo.extend(DEPENDS.get(m, ()))
    return sorted(seen)


def _version(package: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return "none"


def _file_hash(path) -> str:
    """sha256[:12] of a file; "none" when there is none."""
    if not path or not Path(path).is_file():
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def gsm_build(executable: Optional[str] = None) -> str:
    """The molecularGSM binary's identity: sha256[:12] of the file `executable` (or $GSM_EXECUTABLE, or `gsm` on
    PATH) resolves to; "none" when there is none."""
    name = executable or os.getenv("GSM_EXECUTABLE") or "gsm"
    return _file_hash(name if os.path.sep in str(name) else shutil.which(str(name)))


def mlp_identity(model: str, checkpoint: Optional[str] = None) -> list[str]:
    """The ML potential MLP-GI optimizes on: its name, its family's package version (mepd.engines.mlip.MODELS),
    and sha256[:12] of a local checkpoint file when one is given."""
    from mepd.engines.mlip import MODELS

    spec = MODELS.get(str(model))
    package = _MLP_PACKAGES.get(spec.family) if spec else None
    parts = [f"mlp={model}", f"mlp.package={package}=={_version(package)}" if package else "mlp.package=none"]
    if checkpoint:
        parts.append(f"mlp.checkpoint={_file_hash(Path(checkpoint).expanduser())}")
    return parts


def dirty_suffix(code: Optional[dict] = None) -> str:
    """"+<commit 12>.dirty.<diff 8>" when mepd/ has uncommitted edits, else "" (mepd.provenance.code_state)."""
    if code is None:
        from mepd.provenance import code_state

        code = code_state()
    if code.get("dirty") and code.get("commit"):
        return f"+{str(code['commit'])[:12]}.dirty.{str(code.get('diff_sha256', ''))[:8]}"
    return ""


def numerics_epoch(*methods: str, gsm_executable: Optional[str] = None, mlp_model: Optional[str] = None,
                   mlp_checkpoint: Optional[str] = None, code: Optional[dict] = None) -> str:
    """The numerics a result of these methods was computed with: the epoch of each method used (with its
    dependencies), the versions of the programs they call, the GSM build when GSM is among them, the ML potential
    when MLP-GI runs on one (`mlp_model`, `mlp_checkpoint`: the profile's path_min_inputs), and the dirty suffix.
    Equal strings mean reusable results; any bump, version or build change, or uncommitted edit makes them
    differ."""
    used = methods_of(*methods)
    parts = [f"{m}={EPOCHS[m]}" for m in used]
    programs = sorted({p for m in used for p in _PROGRAMS.get(m, ())})
    parts += [f"{p}={_version(p)}" for p in programs]
    if "gsm" in used:
        parts.append(f"gsm.build={gsm_build(gsm_executable)}")
    if "mlpgi" in used and mlp_model:
        parts += mlp_identity(mlp_model, mlp_checkpoint)
    return ",".join(parts) + dirty_suffix(code)


def epochs_of(methods: Iterable[str]) -> dict[str, int]:
    """{method: epoch} for the methods given and those they use."""
    return {m: EPOCHS[m] for m in methods_of(*methods)}
