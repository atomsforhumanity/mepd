"""The ORCA engine (mepd.engines.orca): what reaches ORCA's input, and how its results come back.
ORCA itself is not needed: its process and qccodec's decoding are stood in for."""
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

from mepd.engines import orca as orca_mod
from mepd.errors import ConvergenceError
from mepd.nodes.node import StructureNode


@pytest.fixture
def orca_dir(tmp_path, monkeypatch):
    d = tmp_path / "orca"
    d.mkdir()
    for name in ("orca", "orca_scfgrad"):
        (d / name).write_text("#!/bin/sh\n")
        (d / name).chmod(0o755)
    monkeypatch.setenv("MEPD_ORCA", str(d / "orca"))
    return d


def _water(x=0.0):
    return Structure(symbols=["O", "H", "H"], geometry=np.array([[x, 0, 0], [x, 0, 1.8], [x, 1.7, -0.5]]))


class _FakeOrca:
    """Records each input; `converged` decides what the 'run' printed."""

    def __init__(self, monkeypatch, converged=True, n_steps=3):
        self.inputs, self.converged, self.n_steps = [], converged, n_steps
        monkeypatch.setattr(orca_mod.subprocess, "run", self.run)
        import qccodec
        monkeypatch.setattr(qccodec, "decode", self.decode)

    def run(self, cmd, cwd, **kw):
        self.inputs.append((cwd / "orca.inp").read_text())
        out = ("THE OPTIMIZATION HAS CONVERGED\n" if self.converged else "") + "****ORCA TERMINATED NORMALLY****\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    def decode(self, program, calctype, *, input_data, **kw):
        s = input_data.structure
        traj = [SimpleNamespace(input_data=SimpleNamespace(structure=s.model_copy(update={"geometry": s.geometry + 0.01 * k})),
                                results=SimpleNamespace(energy=-76.0 - 0.001 * k, gradient=np.zeros((3, 3))))
                for k in range(self.n_steps)]
        n = 3 * len(s.symbols)
        return SimpleNamespace(trajectory=traj, energy=-76.0, gradient=np.ones((3, 3)), hessian=np.eye(n))


def test_an_orca_on_the_path_without_orcas_programs_is_not_orca(tmp_path, monkeypatch):
    reader = tmp_path / "bin" / "orca"        # GNOME's screen reader is also called orca
    reader.parent.mkdir()
    reader.write_text("#!/usr/bin/python3\n")
    reader.chmod(0o755)
    monkeypatch.delenv("MEPD_ORCA", raising=False)
    monkeypatch.setenv("PATH", str(reader.parent))
    monkeypatch.setattr(orca_mod.Path, "is_file", lambda self: str(self) == str(reader))
    assert orca_mod.orca_executable() is None


def test_the_ts_search_is_orcas_own_from_an_analytic_hessian(orca_dir, monkeypatch):
    fake = _FakeOrca(monkeypatch)
    eng = orca_mod.OrcaEngine(threads=16, maxcore=2500)
    ts = eng.compute_transition_state(StructureNode(structure=_water()), keywords={"maxiter": 50})
    inp = fake.inputs[0].lower()
    assert "%pal nprocs 16 end" in inp and "%maxcore 2500" in inp
    assert "! wb97x-d3bj def2/j rijcosx defgrid2 tightscf tightopt def2-tzvp" in inp
    assert "! optts" in inp and "calc_hess" in inp and "recalc_hess" in inp and "maxiter" in inp
    assert ts.energy == pytest.approx(-76.002)      # the last step


def test_a_largest_force_criterion_becomes_orcas(orca_dir, monkeypatch):
    fake = _FakeOrca(monkeypatch)
    nodes = orca_mod.OrcaEngine().compute_geometry_optimization(StructureNode(structure=_water()), {"fmax": 0.0514})
    assert "! opt" in fake.inputs[0].lower() and "tolmaxg" in fake.inputs[0].lower()
    tol = float(re.search(r"tolmaxg\s+(\S+)", fake.inputs[0]).group(1))
    assert tol == pytest.approx(0.0514 / 51.422, rel=1e-3)      # eV/A -> Eh/bohr
    assert len(nodes) == 3 and nodes[-1].energy == pytest.approx(-76.002)
    # enough steps by default for a loose pair of molecules (an IRC end)
    assert int(re.search(r"maxiter\s+(\d+)", fake.inputs[0], re.I).group(1)) == 300


def test_an_unconverged_search_raises_with_where_it_stopped(orca_dir, monkeypatch):
    _FakeOrca(monkeypatch, converged=False)
    with pytest.raises(ConvergenceError) as err:
        orca_mod.OrcaEngine().compute_transition_state(StructureNode(structure=_water()))
    assert err.value.structure is not None


def test_the_hessian_is_orcas_and_kept_per_geometry(orca_dir, monkeypatch):
    fake = _FakeOrca(monkeypatch)
    eng = orca_mod.OrcaEngine(method="B3LYP/G", basis="def2-SVP")
    node = StructureNode(structure=_water(3.0))
    h1, h2 = eng.compute_hessian(node), eng.compute_hessian(node.copy())
    assert np.allclose(h1, np.eye(9)) and np.allclose(h1, h2)
    assert len(fake.inputs) == 1 and "! freq" in fake.inputs[0].lower()


def test_a_solvent_uses_orcas_cpcm(orca_dir, monkeypatch):
    from mepd.solvation import SolvatedEngine, solvate_engine

    fake = _FakeOrca(monkeypatch)
    eng = solvate_engine(orca_mod.OrcaEngine(), {"solvent": "chcl3"})
    assert not isinstance(eng, SolvatedEngine)       # no xTB correction on top
    eng.compute_energies([StructureNode(structure=_water(5.0))])
    assert "cpcm(chloroform)" in fake.inputs[0].lower()


def test_a_profile_names_the_orca_engine(orca_dir):
    from mepd.inputs import RunInputs

    eng = RunInputs(engine_name="orca", orca_engine_kwds={"method": "r2SCAN-3c", "basis": "", "keywords": ""}).engine
    assert isinstance(eng, orca_mod.OrcaEngine) and eng.method == "r2SCAN-3c"
