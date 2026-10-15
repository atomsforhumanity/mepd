"""mepd -- reaction path discovery.

Note the Tcl/Tk guard below: it has to run before any `mepd.*` submodule
(and so before `openbabel.pybel`) is imported, which is why it sits at the
very top of the package's `__init__`.
"""

import os
import sys

# macOS fork safety. `mepd.cli_common._fork_map` runs pairs/TS-opts/mechanism
# searches in *forked* children (fork, not spawn, so the closure's structures,
# engine and RunInputs reach each child for free instead of being pickled).
#
# Importing `tkinter` registers Tcl's macOS notifier `pthread_atfork` handlers.
# After a fork-without-exec, the child's copy of Tcl's `notifierInitLock` is
# recorded as owned by a thread that does not exist there, so the next time
# that child forks -- which is the first thing it does, to launch the engine's
# subprocess -- `AtForkPrepare` tries to take the lock and the kernel SIGKILLs
# the child ("BUG IN CLIENT OF LIBPLATFORM: os_unfair_lock is corrupt"). No
# Python-level exception is raised, so the whole run dies with an opaque
# `BrokenProcessPool` a few milliseconds into the first parallel stage.
#
# Nothing in mepd draws with Tk: the import arrives transitively through
# `openbabel.pybel`, which wants it only for `Molecule.draw()` and already
# falls back to `tk = None` when the import fails. Blocking it keeps Tcl's
# atfork handlers unregistered and the children alive. `None` in `sys.modules`
# is the documented way to make an import fail; set MEPD_ALLOW_TKINTER=1 to
# keep Tk (and run the parallel stages with `--workers 1`).
if (
    sys.platform == "darwin"
    and "_tkinter" not in sys.modules
    and not os.environ.get("MEPD_ALLOW_TKINTER")
):
    for _tk_module in ("_tkinter", "tkinter"):
        sys.modules.setdefault(_tk_module, None)
    del _tk_module

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mepd.chain import Chain
    from mepd.inputs import (
        ChainInputs,
        GIInputs,
        NEBInputs,
        NetworkInputs,
        PathMinInputs,
        RunInputs,
    )
    from mepd.neb import NEB
    from mepd.nodes.node import Node, StructureNode, XYNode
    from mepd.msmep import MSMEP
    from mepd.TreeNode import TreeNode
    from mepd.pot import Pot
    from mepd.NetworkBuilder import NetworkBuilder
    from mepd.compare import Match, connects, same_molecule
    from mepd.compute import IRCResult, barrier_kcal, energies, find_ts, interpolate, irc, optimize
    from mepd.engines.factory import engine, engine_from_profile, set_threads
    from mepd.errors import ConvergenceError, UnsupportedOperation
    from mepd.structures import from_smiles, load, read_frames, write_xyz

_CORE_EXPORTS = {
    "Node": ("mepd.nodes.node", "Node"),
    "StructureNode": ("mepd.nodes.node", "StructureNode"),
    "XYNode": ("mepd.nodes.node", "XYNode"),
    "Chain": ("mepd.chain", "Chain"),
    "NEB": ("mepd.neb", "NEB"),
    "PathMinInputs": ("mepd.inputs", "PathMinInputs"),
    "NEBInputs": ("mepd.inputs", "NEBInputs"),
    "ChainInputs": ("mepd.inputs", "ChainInputs"),
    "GIInputs": ("mepd.inputs", "GIInputs"),
    "NetworkInputs": ("mepd.inputs", "NetworkInputs"),
    "RunInputs": ("mepd.inputs", "RunInputs"),
    "MSMEP": ("mepd.msmep", "MSMEP"),
    "Pot": ("mepd.pot", "Pot"),
}

# The public API: flat functions for building structures, getting engines and
# running calculations. See docs/python-api.md.
_API_EXPORTS = {
    # structures (Bohr inside; xyz files are Angstrom)
    "load": ("mepd.structures", "load"),
    "read_frames": ("mepd.structures", "read_frames"),
    "from_smiles": ("mepd.structures", "from_smiles"),
    "write_xyz": ("mepd.structures", "write_xyz"),
    # engines
    "engine": ("mepd.engines.factory", "engine"),
    "engine_from_profile": ("mepd.engines.factory", "engine_from_profile"),
    "set_threads": ("mepd.engines.factory", "set_threads"),
    # calculations
    "energies": ("mepd.compute", "energies"),
    "optimize": ("mepd.compute", "optimize"),
    "find_ts": ("mepd.compute", "find_ts"),
    "irc": ("mepd.compute", "irc"),
    "IRCResult": ("mepd.compute", "IRCResult"),
    "interpolate": ("mepd.compute", "interpolate"),
    "barrier_kcal": ("mepd.compute", "barrier_kcal"),
    # comparing structures
    "same_molecule": ("mepd.compare", "same_molecule"),
    "connects": ("mepd.compare", "connects"),
    "Match": ("mepd.compare", "Match"),
    # errors
    "UnsupportedOperation": ("mepd.errors", "UnsupportedOperation"),
    "ConvergenceError": ("mepd.errors", "ConvergenceError"),
}
_CORE_EXPORTS.update(_API_EXPORTS)
# `TreeNode` and `NetworkBuilder` (the classes) are intentionally not
# re-exported here: their modules are named `mepd.TreeNode`/
# `mepd.NetworkBuilder` too, and Python's import machinery registers each
# submodule as an attribute on the `mepd` package as soon as anything imports
# it -- which permanently shadows a `__getattr__`-based lazy export of the
# class under the same name. Use `from mepd.TreeNode import TreeNode` /
# `from mepd.NetworkBuilder import NetworkBuilder` instead.

__all__ = sorted([*_CORE_EXPORTS, "engines", "units"])


def __getattr__(name: str):
    if name == "units":
        import mepd.units as units_module
        globals()["units"] = units_module
        return units_module
    if name == "engines":
        import mepd.engines as engines_module
        globals()["engines"] = engines_module
        return engines_module
    if name in _CORE_EXPORTS:
        module_name, attr_name = _CORE_EXPORTS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
