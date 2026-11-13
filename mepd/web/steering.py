"""Built-in steering profiles: higher levels of theory a network expansion can steer with without a saved
workspace profile. linus (AFH's in-process DFT, atomsforhumanity/linus) is the default where it is installed
(the web image); elsewhere there is none and the default stays the run's own level."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

# The name the form shows and a job's params keep (steer_profile).
LINUS = "linus · ωB97X-D3BJ/def2-TZVP (built in)"

# The explorer team's gated steering settings (sb-t22: 8 of 8 systems, accuracy and recall equal to ORCA
# RIJCOSX, 5-8x faster per point) except the grid: linus's own linus:2 instead of orca:2, whose ORCA-measured
# data linus keeps out of its releases. On the isoprene + methyl acrylate box the two grids differ by
# 0.004 kcal/mol; on 42 benchmark species and TSs both give TS-TS energies within 0.11 kcal/mol of ORCA.
# Memory 1.5: a worker peaks near 1.4 GB at 25 atoms. linus calls it memory_gib since its settings rework
# (2026-11-12) and memory_gb before; the profile names it as the installed linus does (linus_toml).
_LINUS_TEMPLATE = """\
# Built in (mepd.web.steering): wB97X-D3BJ/def2-TZVP single points with linus, in process through qccompute.
# RI-JK (def2-universal-jkfit), linus's own DefGrid2-sized grid, 1 thread, 1.5 GiB per point.
engine_name = "qccompute"
program = "linus"
[program_kwds]
model = {method = "wb97x-d3bj", basis = "def2-tzvp"}
keywords = {jk = "df", grid = "linus:2", threads = 1, MEMORY = 1.5}
"""


def linus_available() -> bool:
    return importlib.util.find_spec("linus") is not None and importlib.util.find_spec("qccompute") is not None


def _linus_memory_key() -> str:
    """The installed linus's name for its memory budget: memory_gib, or memory_gb before 2026-11-12."""
    try:
        from linus._qc import KEYWORDS
    except Exception:
        return "memory_gib"
    return "memory_gb" if "memory_gb" in KEYWORDS and "memory_gib" not in KEYWORDS else "memory_gib"


def linus_toml() -> str:
    """The built-in linus steering profile, in the installed linus's keywords."""
    return _LINUS_TEMPLATE.replace("MEMORY = ", f"{_linus_memory_key()} = ")


def builtin_profiles() -> dict[str, str]:
    """{name: profile TOML} of the built-in steering profiles this install can run."""
    return {LINUS: linus_toml()} if linus_available() else {}


def default_steer_profile() -> str:
    """The expansion form's default steering profile: linus where installed, else none ("")."""
    return LINUS if linus_available() else ""


def builtin_profile_engine(name: str = LINUS):
    """The engine a built-in profile gives, loaded as `--steer-energies` loads it (the image's self-check)."""
    from mepd.cli_common import _open_run_inputs

    text = {LINUS: linus_toml()}[name]
    with tempfile.TemporaryDirectory(prefix="mepd-steer-") as tmp:
        path = Path(tmp) / "steer.toml"
        path.write_text(text)
        return _open_run_inputs(path).engine
