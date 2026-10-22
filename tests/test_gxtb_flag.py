"""--gxtb is decided from the program, not its file name (mepd#4): the g-xTB 2.0.1
release is an xtb 6.7.1 build that runs GFN2-xTB unless given --gxtb, so a `gxtb`
symlink to it must still get the flag. A run meant as g-xTB that reports a GFN
Hamiltonian is an error, not a result."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from qcdata import Structure

import mepd.engines.gxtb as gxtb
from mepd import programs
from mepd.errors import ElectronicStructureError
from mepd.nodes.node import StructureNode


def _script(path: Path, version_line: str) -> str:
    path.write_text(f"#!/bin/sh\necho '{version_line}'\n")
    path.chmod(0o755)
    return str(path)


def _water():
    return Structure(symbols=["O", "H", "H"],
                     geometry=np.array([[0, 0, 0.22], [0, 1.43, -0.88], [0, -1.43, -0.88]], dtype=float),
                     charge=0, multiplicity=1)


def test_standalone_gxtb_is_told_from_its_version_output(tmp_path):
    programs._is_xtb_build.cache_clear()
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    xtb_build = _script(tmp_path / "a" / "gxtb", "   * xtb version 6.7.1 (26dd68d)")
    standalone = _script(tmp_path / "b" / "gxtb", "g-xTB standalone 1.0")
    assert programs.standalone_gxtb(xtb_build) is False
    assert programs.standalone_gxtb(standalone) is True
    assert programs.standalone_gxtb(str(tmp_path / "missing" / "gxtb")) is True       # unknown: as before
    assert programs.standalone_gxtb(_script(tmp_path / "xtb", "anything")) is False   # other names: never probed


def test_a_gxtb_symlink_to_the_xtb_build_gets_the_flag(tmp_path, monkeypatch):
    programs._is_xtb_build.cache_clear()
    exe = _script(tmp_path / "gxtb", "   * xtb version 6.7.1")
    calls = []

    def run(cmd, *, cwd, env, timeout_s, watch=None):
        calls.append(cmd)
        (cwd / "energy").write_text("$energy\n     1   -76.4   -76.4   -76.4\n$end\n")
        (cwd / "gradient").write_text("   1.0E-03   0.0E+00   2.0E-03\n" * 3)
        return subprocess.CompletedProcess(cmd, 0, "normal termination of xtb", "")

    monkeypatch.setattr(gxtb, "_run_gxtb_process", run)
    gxtb.GXTBCalculator(executable=exe).compute_gradients([StructureNode(structure=_water())])
    assert "--gxtb" in calls[0]


def test_a_gfn_hamiltonian_in_a_gxtb_run_is_an_error(monkeypatch):
    out = "          :  Hamiltonian                  GFN2-xTB          :\nnormal termination of xtb\n"

    def run(cmd, *, cwd, env, timeout_s, watch=None):
        (cwd / "energy").write_text("$energy\n     1   -5.07   -5.07   -5.07\n$end\n")
        (cwd / "gradient").write_text("   1.0E-03   0.0E+00   2.0E-03\n" * 3)
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(gxtb, "_run_gxtb_process", run)
    with pytest.raises(ElectronicStructureError, match="ran GFN2-xTB, not g-xTB"):
        gxtb.GXTBCalculator(executable="xtb").compute_gradients([StructureNode(structure=_water())])
    node = StructureNode(structure=_water())       # GFN-xTB asked for: fine
    gxtb.GXTBCalculator(executable="xtb", add_gxtb_flag=False).compute_gradients([node])
    assert node.energy == pytest.approx(-5.07)


_BUILD = Path.home() / ".local" / "opt" / "gxtb-2.0.1" / "bin" / "xtb"


@pytest.mark.skipif(not os.access(_BUILD, os.X_OK), reason="needs the local g-xTB 2.0.1 build")
def test_real_gxtb_symlink_runs_gxtb(tmp_path):
    """The issue's reproduction: a `gxtb` symlink to the 2.0.1 build ran GFN2 (-5.07 Eh)."""
    programs._is_xtb_build.cache_clear()
    link = tmp_path / "gxtb"
    link.symlink_to(_BUILD)
    node = StructureNode(structure=_water())
    gxtb.GXTBCalculator(executable=str(link)).compute_gradients([node])
    assert node.energy == pytest.approx(-76.4, abs=0.1)
