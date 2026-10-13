"""Run provenance: which code, settings and binaries produced an output.

Every `mepd` command given `--output` leaves `provenance.json` there (or
`<file>.provenance.json` when the output is a file): the command line, the
profile it ran with (text and level key), mepd's version and git state, the
external binaries it could have called with their hashes, library versions,
host, timing and exit code. Results that later become a published finding
(see the mepd-findings repo) are then traceable without reconstructing any
of this by hand. Set MEPD_NO_PROVENANCE=1 to turn it off.

Study scripts outside the CLI can call `run_manifest` / `write_manifest`
directly.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

# Binaries a profile can route work to when it doesn't name one explicitly.
DEFAULT_BINARIES = ("xtb", "gxtb-xtb", "crest", "terachem", "gsm")
PACKAGES = ("mepd", "numpy", "scipy", "rdkit", "ase", "sella", "geodesic-interpolate",
            "qcop", "qcio", "qcdata", "openmm", "torch")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def code_state() -> dict:
    """mepd's version, and its git commit/dirty state when run from a checkout."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        out = {"version": version("mepd")}
    except PackageNotFoundError:
        out = {"version": "unknown"}
    root = Path(__file__).resolve().parents[1]
    head = _git(["rev-parse", "HEAD"], root) if (root / ".git").exists() else None
    if head:
        diff = _git(["diff", "HEAD", "--", "mepd"], root) or ""
        out.update(commit=head, dirty=bool(diff), source=str(root))
        if diff:
            out["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
    return out


def package_versions(names: Sequence[str] = PACKAGES) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            continue
    return out


def _profile_binaries(profile: dict) -> list[str]:
    """Executable paths a profile names (e.g. gxtb_engine_kwds.executable, path_min_inputs.executable)."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and k in ("executable", "binary", "crest_executable", "program_path"):
                    found.append(v)
                else:
                    walk(v)

    walk(profile)
    return found


def binary_info(name_or_path: str) -> dict:
    path = shutil.which(name_or_path) or os.path.expanduser(name_or_path)
    p = Path(path)
    if not p.is_file():
        return {"name": name_or_path, "found": False}
    real = p.resolve()
    try:
        digest = _sha256(real)
    except OSError:
        digest = None
    return {"name": name_or_path, "path": str(p), "resolved": str(real), "sha256": digest}


def run_manifest(argv: Sequence[str], inputs: Optional[Path] = None, extra: Optional[dict] = None) -> dict:
    profile_text, profile = None, {}
    if inputs and Path(inputs).is_file():
        profile_text = Path(inputs).read_text()
        try:
            import tomli

            profile = tomli.loads(profile_text)
        except Exception:
            profile = {}
    try:
        from mepd.web.workspace import level_key

        lkey = level_key(profile_text)
    except Exception:
        lkey = None
    names = list(dict.fromkeys([*_profile_binaries(profile), *DEFAULT_BINARIES]))
    return {
        "schema": "mepd-provenance/1",
        "argv": list(argv),
        "cwd": os.getcwd(),
        "profile_path": str(Path(inputs).resolve()) if inputs else None,
        "profile_text": profile_text,
        "level_key": lkey,
        "mepd": code_state(),
        "packages": package_versions(),
        "binaries": [b for b in (binary_info(n) for n in names) if b.get("found", True)],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
        "env": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "XTBHOME") if k in os.environ},
        **(extra or {}),
    }


def write_manifest(output: Path, manifest: dict) -> Path:
    output = Path(output)
    dest = output / "provenance.json" if output.is_dir() else output.with_name(output.name + ".provenance.json")
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, default=str))
    os.replace(tmp, dest)
    return dest


def _flag_value(argv: Sequence[str], long: str, short: str) -> Optional[str]:
    for i, a in enumerate(argv):
        if a in (long, short) and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(long + "="):
            return a.split("=", 1)[1]
    return None


def record_cli_run(argv: Sequence[str], started: _dt.datetime, returncode: int) -> Optional[Path]:
    """Called once per `mepd` process; never raises."""
    if os.environ.get("MEPD_NO_PROVENANCE"):
        return None
    try:
        out = _flag_value(argv, "--output", "-o")
        if not out or not Path(out).exists():
            return None
        inputs = _flag_value(argv, "--inputs", "-i")
        finished = _dt.datetime.now(_dt.timezone.utc)
        manifest = run_manifest(["mepd", *argv], Path(inputs) if inputs else None, {
            "started": started.isoformat(timespec="seconds"),
            "finished": finished.isoformat(timespec="seconds"),
            "wall_s": round((finished - started).total_seconds(), 1),
            "returncode": returncode,
        })
        return write_manifest(Path(out), manifest)
    except Exception:
        return None
