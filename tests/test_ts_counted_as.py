"""A fast TS found on disk after a crash comes back with its energy."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qcdata import Structure

from mepd.nodes.node import StructureNode

BOHR = 1.8897259886
# formic acid (O0 C1 O2 H3 H4) next to a spectator water (O5 H6 H7), Angstrom
SYM = ["O", "C", "O", "H", "H", "O", "H", "H"]
ACID = np.array([[0.74, -0.62, -0.08], [-0.41, 0.06, -0.16], [-0.56, 1.17, 0.27], [1.40, -0.07, 0.37],
                 [-1.17, -0.55, -0.67]])
WATER = np.array([[6.0, 0, 0], [6.96, 0, 0], [5.76, 0.93, 0]])
START = np.vstack([ACID, WATER])


def _st(xyz, comment=None):
    extras = {"xyz_comments": comment.split()} if comment else {}
    return Structure(symbols=SYM, geometry=np.asarray(xyz) * BOHR, charge=0, multiplicity=1, extras=extras)


def test_a_fast_ts_found_on_disk_after_a_crash_comes_back_with_its_energy(tmp_path):
    """A run killed between writing a fast TS and saving its state finds the file again on --resume: the queue
    then takes it from the file. Without its energy the main process stopped with EnergiesNotComputedError
    (acetic acid, staged, killed at 200 s: the resume ended in 2 s, on develop 95e094d too)."""
    from mepd.discovery.horizontal import HorizontalConnector

    ts = StructureNode(structure=_st(START))
    ts._cached_energy = -1.5
    queue = SimpleNamespace(ts_dir=tmp_path, free=None)
    queue._ts_correction = lambda *a, **k: None
    HorizontalConnector._write_ts(queue, ts, "fast_0_1")
    back, label, corr = HorizontalConnector._fast_ts(queue, 0, 1, ts, ts)
    assert label == "fast_0_1" and back.energy == -1.5
