"""QCComputeEngine with compute_program "qccompute" (local): energies are energy calculations, not gradients."""
import numpy as np
import pytest
from qcdata import Structure

from mepd.engines.qccompute import QCComputeEngine
from mepd.nodes.node import StructureNode
from mepd.program_args import ProgramArgs

WATER = Structure(symbols=["O", "H", "H"], geometry=[[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])


def test_local_energies_run_energy_calculations(monkeypatch):
    eng = QCComputeEngine(program_args=ProgramArgs(model={"method": "hf", "basis": "sto-3g"}), program="linus",
                          compute_program="qccompute")
    seen = []

    def fake(inp, **kwargs):
        from types import SimpleNamespace
        seen.append(str(inp.calctype))
        return SimpleNamespace(success=True, input_data=inp, results=SimpleNamespace(energy=-1.5, gradient=None))

    monkeypatch.setattr(eng, "compute_func", fake)
    nodes = [StructureNode(structure=WATER), StructureNode(structure=WATER)]
    assert np.allclose(eng.compute_energies(nodes), [-1.5, -1.5])
    assert seen == ["energy", "energy"] or all(s.endswith("energy") for s in seen)
    eng.compute_energies(nodes)   # cached: nothing recomputed
    assert len(seen) == 2


def test_linus_single_point_through_the_engine():
    pytest.importorskip("linus")
    pytest.importorskip("qccompute")
    eng = QCComputeEngine(program_args=ProgramArgs(model={"method": "hf", "basis": "sto-3g"}, keywords={"threads": 1}),
                          program="linus", compute_program="qccompute")
    e = eng.compute_energies([StructureNode(structure=WATER)])[0]
    assert -75.0 < e < -74.9
