"""g-xTB's SCF can fail to converge on a stretched path image; the engine
retries that one calculation with Fermi smearing (a raised electronic
temperature) before failing the path search."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from qcdata import Structure

import mepd.engines.gxtb as gxtb
from mepd.errors import ElectronicStructureError
from mepd.nodes.node import StructureNode

_FAILED = "-1- tblite_calculator_singlepoint: SCF not converged in 250 cycles\nabnormal termination of xtb\n"


def _water():
    return Structure(symbols=["O", "H", "H"],
                     geometry=np.array([[0, 0, 0], [1.43, 0, 0.95], [-1.43, 0, 0.95]], dtype=float),
                     charge=0, multiplicity=1)


def _fake(monkeypatch, converges_with):
    calls = []

    def run(cmd, *, cwd, env, timeout_s, watch=None):
        calls.append(cmd)
        if converges_with is None or converges_with not in cmd:
            return subprocess.CompletedProcess(cmd, 128, _FAILED, "")
        (cwd / "energy").write_text("$energy\n     1   -76.0   -76.0   -76.0\n$end\n")
        (cwd / "gradient").write_text("   1.0E-03   0.0E+00   2.0E-03\n" * 3)
        return subprocess.CompletedProcess(cmd, 0, "normal termination of xtb", "")

    monkeypatch.setattr(gxtb, "_run_gxtb_process", run)
    return calls


def test_an_unconverged_scf_is_retried_with_smearing(monkeypatch):
    calls = _fake(monkeypatch, converges_with="1000")
    node = StructureNode(structure=_water())
    gxtb.GXTBCalculator(executable="gxtb").compute_gradients([node])
    assert node.energy == pytest.approx(-76.0)
    assert len(calls) == 2 and calls[1][-4:] == ["--etemp", "1000", "--iterations", "1000"]


def test_it_fails_with_the_retries_named_when_none_converges(monkeypatch):
    calls = _fake(monkeypatch, converges_with=None)
    with pytest.raises(ElectronicStructureError, match="SCF not converged, also with --etemp 1000"):
        gxtb.GXTBCalculator(executable="gxtb").compute_gradients([StructureNode(structure=_water())])
    assert len(calls) == 1 + len(gxtb.SCF_FALLBACKS)


def test_no_retry_for_runs_that_choose_their_own_settings(monkeypatch):
    """GFN-xTB runs through this class (no --gxtb flag, e.g. solvation's gas /
    solvated pairs) and runs that already set an electronic temperature are
    left alone: their callers retry them, matched within a pair."""
    calls = _fake(monkeypatch, converges_with="1000")
    with pytest.raises(ElectronicStructureError):
        gxtb.GXTBCalculator(executable="xtb", add_gxtb_flag=False,
                            extra_args=["--gfn", "2"]).compute_gradients([StructureNode(structure=_water())])
    with pytest.raises(ElectronicStructureError):
        gxtb.GXTBCalculator(executable="gxtb", extra_args=["--etemp", "500"]).compute_gradients(
            [StructureNode(structure=_water())])
    assert len(calls) == 2


def test_tblite_failures_retry_fresh_then_skip_the_neighbourhood(monkeypatch, tmp_path):
    """A failed in-process SCF is retried once without the neighbour's wavefunction; when that fails too,
    nearby structures go straight to xtb instead of paying for another full set of SCF cycles."""
    import json

    import tblite.interface

    calls = []

    class Calc:
        def __init__(self, *a, **k): pass
        def set(self, *a): pass
        def add(self, *a): pass
        def update(self, pos): self.pos = pos

        def singlepoint(self, guess=None, copy=False):
            calls.append(guess is not None)
            if self.pos[0, 0] > 0.5:                     # the "hard" region
                raise RuntimeError("SCF not converged in 250 cycles")
            return {"energy": -5.0, "gradient": np.zeros((3, 3))}

    monkeypatch.setattr(tblite.interface, "Calculator", Calc)
    monkeypatch.setenv("MEPD_CALL_LOG", str(tmp_path))
    gxtb._TBLITE.__dict__.clear()

    def node(x):
        s = _water()
        geom = s.geometry.copy()
        geom[0, 0] = x
        return StructureNode(structure=Structure(symbols=s.symbols, geometry=geom, charge=0, multiplicity=1))

    assert gxtb._tblite_point(node(0.0), []) is not None and calls == [False]
    assert gxtb._tblite_point(node(0.6), []) is None and calls[1:] == [True, False]   # guess, then fresh
    assert gxtb._tblite_point(node(0.7), []) is None and len(calls) == 3              # next step: skipped
    assert gxtb._tblite_point(node(-3.0), []) is not None                             # far enough away again
    log = [json.loads(l) for f in tmp_path.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert [r["ok"] for r in log] == [True, False, True]
