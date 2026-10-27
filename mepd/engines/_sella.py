"""Import Sella without torch on machines with no NVIDIA GPU.

Sella imports torch when it is first imported (sella/_gpu.py), only to route large
eigendecompositions to a CUDA GPU when one is present. Loading torch costs about
400 MiB of memory per process, which on a CPU-only machine buys nothing: a cluster
job that runs thousands of g-xTB searches at once pays it in every task. So when no
GPU can be visible to this process, torch is hidden while Sella is imported; Sella
then takes its CPU path, exactly as it does on a machine without torch. torch stays
importable afterwards for anything else that needs it (e.g. ML potentials).
"""
from __future__ import annotations

import glob
import os
import sys


def _gpu_visible() -> bool:
    if os.environ.get("SELLA_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1"):
        return False
    # NVIDIA device nodes (Linux), or the WSL GPU bridge
    return bool(glob.glob("/dev/nvidia[0-9]*")) or os.path.exists("/dev/dxg")


def sella():
    """The `sella` module, imported without torch when no GPU is visible."""
    if "sella" in sys.modules or "torch" in sys.modules or _gpu_visible():
        import sella as module

        return module
    sys.modules["torch"] = None          # `import torch` raises ImportError, which Sella handles
    try:
        import sella as module
    finally:
        if sys.modules.get("torch", 0) is None:
            del sys.modules["torch"]
    return module
