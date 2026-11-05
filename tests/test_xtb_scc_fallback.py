"""GFN1/GFN2 through the xtb program: an SCC that does not converge is
retried with Fermi smearing (as the g-xTB engine does), and errors name the
method that ran."""
import stat

import numpy as np
import pytest

from mepd.engines.gfnff import GFNFFEngine, XTBEngine
from mepd.errors import ElectronicStructureError

_FAKE = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
open("calls", "a").write(" ".join(args) + "\\n")
if ("--etemp" not in args) or "{always_fail}" == "1":
    print("-1- scf: Self consistent charge iterator did not converge")
    print("abnormal termination of xtb")
    sys.exit(1)
open("in.engrad", "w").write("2\\n-1.5\\n" + "0.0\\n" * 6)
"""


def _fake(tmp_path, always_fail=False):
    exe = tmp_path / "xtb"
    exe.write_text(_FAKE.replace("{always_fail}", "1" if always_fail else "0"))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def _run(engine, tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    out = engine._run(wd, ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), 0)
    return out, (wd / "calls").read_text().splitlines()


def test_scc_failure_is_retried_with_fermi_smearing(tmp_path):
    (energy, grad), calls = _run(XTBEngine(executable=_fake(tmp_path), n_parallel=1), tmp_path)
    assert energy == -1.5 and grad.shape == (2, 3)
    assert len(calls) == 2 and "--etemp 1000" in calls[1] and "--etemp" not in calls[0]


def test_scc_failure_after_all_retries_names_the_method(tmp_path):
    with pytest.raises(ElectronicStructureError, match=r"^GFN2-xTB \(xtb\) failed"):
        _run(XTBEngine(executable=_fake(tmp_path, always_fail=True), n_parallel=1), tmp_path)
    calls = (tmp_path / "wd" / "calls").read_text().splitlines()
    assert len(calls) == 3 and "--etemp 3000" in calls[2]


def test_gfnff_is_not_retried(tmp_path):
    with pytest.raises(ElectronicStructureError, match=r"^GFN-FF \(xtb\) failed"):
        _run(GFNFFEngine(executable=_fake(tmp_path, always_fail=True), n_parallel=1), tmp_path)
    assert len((tmp_path / "wd" / "calls").read_text().splitlines()) == 1


# ---- one surface per path (#24): a rung that worked is held, and the batch is recomputed at it

_FAKE_BY_GEOMETRY = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
open("{log}", "a").write(" ".join(args) + "\\n")
z = float(open("in.xyz").read().split()[-1])
kelvin = float(args[args.index("--etemp") + 1]) if "--etemp" in args else 300.0
if z > 2.0 and kelvin < 1000:   # a stretched image: SCC fails without smearing
    print("-1- scf: Self consistent charge iterator did not converge")
    sys.exit(1)
open("in.engrad", "w").write("2\\n%.6f\\n" % (-1.0 - kelvin / 1e6) + "0.0\\n" * 6)
"""


def test_a_rung_that_worked_is_held_and_the_whole_batch_shares_it(tmp_path):
    from qcdata import Structure

    from mepd.nodes.node import StructureNode

    exe = tmp_path / "xtb"
    exe.write_text(_FAKE_BY_GEOMETRY.replace("{log}", str(tmp_path / "calls")))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    engine = XTBEngine(executable=str(exe), n_parallel=1)
    nodes = [StructureNode(structure=Structure(symbols=["H", "H"], charge=0, multiplicity=1,
                                               geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z * 1.8897]])))
             for z in (0.74, 1.5, 2.5)]   # Å; the last image's SCC needs smearing
    energies = engine.compute_energies(nodes)
    assert np.allclose(energies, -1.0 - 1000 / 1e6)   # all three at 1000 K, none at 300 K
    assert engine._scc_floor == 1000.0
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 3 + 1 + 3   # three runs, one retry, the batch again at 1000 K
    later = StructureNode(structure=nodes[0].structure)
    engine.compute_energies([later])                  # and every calculation after
    assert "--etemp 1000.0" in (tmp_path / "calls").read_text().splitlines()[-1]
