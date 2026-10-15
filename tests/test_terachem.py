"""TeraChem as a local engine (mepd/engines/terachem.py) and as the runner of
TeraChem's own QM/MM: checked against a stand-in `terachem` that reads the
same input files and prints TeraChem's output format (a toy model with exact
gradients), so units, signs and the point-charge file are tested without a
GPU. `MEPD_TERACHEM_LIVE=1` runs the same checks on the real TeraChem."""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from qcref.constants import ANGSTROM_TO_BOHR

FAKE = textwrap.dedent('''\
    import re, sys
    from pathlib import Path
    import numpy as np
    A = 1.8897261246257702
    kw = {}
    for line in Path(sys.argv[1]).read_text().splitlines():
        p = line.split(None, 1)
        if len(p) == 2:
            kw[p[0].lower()] = p[1].strip()
    if kw.get("method") == "explode":
        print("ERROR: SCF did not converge"); sys.exit(1)
    Path("log").open("a").write(kw.get("guess", "-") + "\\n")
    rows = Path(kw["coordinates"]).read_text().splitlines()[2:]
    sym = [r.split()[0] for r in rows if r.strip()]
    x = np.array([[float(v) for v in r.split()[1:4]] for r in rows if r.strip()]) * A
    q = np.array([0.4 if s == "H" else -0.8 for s in sym])
    e, g = 0.0, np.zeros_like(x)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d = x[i] - x[j]; r = np.linalg.norm(d)
            e += 0.3 * (r - 1.9) ** 2; f = 0.6 * (r - 1.9) * d / r
            g[i] += f; g[j] -= f
    pc, gpc = None, None
    if "pointcharges" in kw:
        assert kw.get("pointcharges_self_interaction") == "false"
        prow = [r.split() for r in Path(kw["pointcharges"]).read_text().splitlines()[2:] if r.strip()]
        pq = np.array([float(r[0]) for r in prow]); px = np.array([[float(v) for v in r[1:4]] for r in prow]) * A
        gpc = np.zeros_like(px)
        for i in range(len(x)):
            for j in range(len(px)):
                d = x[i] - px[j]; r = np.linalg.norm(d)
                e += q[i] * pq[j] / r
                f = -q[i] * pq[j] * d / r ** 3
                g[i] += f; gpc[j] -= f
    Path("scr").mkdir(exist_ok=True)
    (Path("scr") / ("c0" if kw["spinmult"] == "1" else "ca0")).write_bytes(b"orbitals")
    if kw["spinmult"] != "1":
        (Path("scr") / "cb0").write_bytes(b"orbitals")
    Path("scr/charge_mull.xls").write_text("".join(f"{i+1} {s} {c:.4f}\\n" for i, (s, c) in enumerate(zip(sym, q))))
    print("|       *                 TeraChem v1.9-2024.09-dev               *")
    print(f"FINAL ENERGY: {e - 76.0:.12f} a.u.")
    print("Gradient units are Hartree/Bohr")
    print("---------------------------------------------------")
    print("         dE/dX            dE/dY            dE/dZ")
    for v in g:
        print("  %.10f  %.10f  %.10f" % tuple(v))
    if gpc is not None:
        print("------- MM / Point charge part -------")
        for v in gpc:
            print("  %.10f  %.10f  %.10f" % tuple(v))
    print("---------------------------------------------------")
    print("Net gradient:   0.0   0.0   0.0")
''')

WATER = (["O", "H", "H"], np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]]) * ANGSTROM_TO_BOHR)
CHARGES = [(-0.834, np.array([2.8, 0.3, 0.1]) * ANGSTROM_TO_BOHR), (0.417, np.array([3.4, -0.5, 0.3]) * ANGSTROM_TO_BOHR)]


@pytest.fixture
def fake_terachem(tmp_path, monkeypatch):
    script = tmp_path / "fake_tc.py"
    script.write_text(FAKE)
    exe = tmp_path / "terachem"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    exe.chmod(0o755)
    monkeypatch.setenv("MEPD_TERACHEM", str(exe))
    return exe


def _engine(**kw):
    from mepd.engines.terachem import TeraChemEngine

    if os.environ.get("MEPD_TERACHEM_LIVE"):
        kw.setdefault("basis", "6-31gs")
    return TeraChemEngine(**kw)


def _fd_checks(eng, h=1e-3, tol=1e-5):
    sym, x0 = WATER
    e0, g, gpc = eng.energy_gradient(sym, x0, point_charges=CHARGES)
    assert g.shape == (3, 3) and gpc.shape == (2, 3)
    for a, k in ((0, 2), (1, 1)):                    # QM atoms
        es = []
        for s in (1, -1):
            y = x0.copy()
            y[a, k] += s * h
            es.append(eng.energy_gradient(sym, y, point_charges=CHARGES)[0])
        assert abs((es[0] - es[1]) / (2 * h) - g[a, k]) < tol
    for j, k in ((0, 0), (1, 2)):                    # point charges
        es = []
        for s in (1, -1):
            pcs = [(q, p.copy()) for q, p in CHARGES]
            pcs[j][1][k] += s * h
            es.append(eng.energy_gradient(sym, x0, point_charges=pcs)[0])
        assert abs((es[0] - es[1]) / (2 * h) - gpc[j, k]) < tol
    return e0


def test_point_charge_gradients_match_the_energy(fake_terachem):
    eng = _engine()
    _fd_checks(eng)
    assert np.allclose(eng.last_charges, [-0.8, 0.4, 0.4])


def test_previous_orbitals_are_the_next_guess(fake_terachem, tmp_path):
    eng = _engine(workdir=str(tmp_path / "runs"), keep_runs=True)
    sym, x = WATER
    eng.energy_gradient(sym, x)
    eng.energy_gradient(sym, x * 1.01)
    eng.energy_gradient(sym, x, multiplicity=3, charge=1)
    eng.energy_gradient(sym, x, multiplicity=3, charge=1)
    runs = sorted((tmp_path / "runs").iterdir(), key=lambda p: p.stat().st_mtime_ns)
    guesses = [(r / "log").read_text().strip() for r in runs]
    assert guesses == ["-", "c0", "-", "ca0 cb0"]
    assert "method ub3lyp" in (runs[2] / "tc.in").read_text()


def test_failure_says_why(fake_terachem):
    from mepd.errors import ElectronicStructureError

    with pytest.raises(ElectronicStructureError, match="SCF did not converge"):
        _engine(method="explode").energy_gradient(*WATER)


def test_profile_wiring(fake_terachem, tmp_path, monkeypatch):
    """engine_name = "terachem" builds the engine; it is a point-charge QM
    level; the license file in ~/.config/terachem is found."""
    import mepd.engines.terachem as tc
    from mepd.inputs import RunInputs
    from mepd.qmmm import embedding_problem
    from mepd.web.workspace import effective_level, level_label

    lic = tmp_path / "license.key"
    lic.write_text("key x\nserver y\n")
    monkeypatch.setattr(tc, "LICENSE_FILE", lic)
    monkeypatch.delenv("TERACHEM_LICENSE_PATH", raising=False)
    ri = RunInputs(engine_name="terachem", terachem_engine_kwds={"method": "wpbeh", "basis": "def2-svp"})
    assert os.environ["TERACHEM_LICENSE_PATH"] == str(lic)
    assert isinstance(ri.engine, tc.TeraChemEngine) and ri.engine.method == "wpbeh"
    assert embedding_problem("electrostatic", "terachem") is None
    text = 'engine_name = "terachem"\n[terachem_engine_kwds]\nmethod = "wpbeh"\nbasis = "def2-svp"\ntimeout = 60\n'
    assert effective_level(text) == {"engine": "terachem", "method": "wpbeh", "basis": "def2-svp"}
    assert level_label(text) == "terachem wpbeh/def2-svp"


def test_electrostatic_qmmm_with_terachem_has_exact_gradients(fake_terachem):
    """TeraChem as the QM level of mepd's electrostatic QM/MM (TIP3P water in
    OpenMM): QM/MM gradients on QM and water atoms are exact."""
    pytest.importorskip("openmm")
    from mepd.engines.qmmm import QMMMEngine
    from mepd.nodes.node import StructureNode
    from mepd.qmmm import QMMMRegion
    from tests.test_qmmm import _cluster

    s = _cluster()
    r = QMMMRegion.build(s, "0-5", active_radius=None, mm="tip3p")
    eng = QMMMEngine(base=_engine(), region=r)
    node = lambda x: StructureNode(structure=s.model_copy(update={"geometry": x}), has_molecular_graph=False)
    x0 = np.asarray(s.geometry, dtype=float)
    out = eng.evaluate([node(x0)])[0]
    assert out["qm_charges"] is not None and len(out["qm_charges"]) == 6
    h = 1e-3
    for a, c in [(1, 0), (6, 1), (7, 2)]:     # a QM atom, a water O, a water H
        ep, em = (eng.evaluate([node(x0 + sign * h * np.eye(len(x0) * 3)[3 * a + c].reshape(-1, 3))])[0]["energy"]
                  for sign in (1, -1))
        assert abs((ep - em) / (2 * h) - out["gradient"][a, c]) < 2e-5


def test_native_terachem_qmmm_runs_locally(tmp_path, monkeypatch):
    """[qmmm] mm = "terachem" with engine_name = "terachem": TeraChem's own
    QM/MM run by the local command, its QM and MM gradient blocks placed."""
    from mepd.engines.terachem_qmmm import TeraChemQMMMEngine

    exe = tmp_path / "terachem"
    exe.write_text("#!/bin/sh\n" + textwrap.dedent('''\
        test -f ref.prmtop && test -f ref.rst7 && grep -q "qmindices qmindices.dat" tc.in || exit 3
        echo "FINAL ENERGY: -152.5 a.u."
        echo "         dE/dX            dE/dY            dE/dZ"
        echo "  0.1 0.2 0.3"
        echo "  0.4 0.5 0.6"
        echo "------- MM / Point charge part -------"
        echo "  0.7 0.8 0.9"
        echo "---------------------------------------------------"
        '''))
    exe.chmod(0o755)
    eng = TeraChemQMMMEngine(tcin="method b3lyp\nbasis 6-31gs\n", prmtop="%FLAG TITLE\n", qm_atoms=[0, 2],
                             compute_program="local", command=str(exe))
    out = eng._submit([eng._input(np.zeros((3, 3)))])[0]
    e, g = eng.energy_gradient_from_stdout(out.stdout, 3)
    assert e == -152.5 and np.allclose(g, [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [0.4, 0.5, 0.6]])


@pytest.mark.skipif(not os.environ.get("MEPD_TERACHEM_LIVE"), reason="set MEPD_TERACHEM_LIVE=1 to run TeraChem")
def test_live_terachem_point_charges(monkeypatch):
    """The real TeraChem: point-charge file read in Å, its gradient on the
    charges and the energy (with the QM-charge interaction) consistent."""
    monkeypatch.delenv("MEPD_TERACHEM", raising=False)
    e0 = _fd_checks(_engine(keywords={"convthre": "1e-7", "threall": "1e-14"}), h=2e-3, tol=2e-5)
    eng = _engine()
    e_gas = eng.energy_gradient(*WATER)[0]
    assert abs(e0 - e_gas) > 1e-4           # the charges change the energy


def test_a_build_without_code_for_this_gpu_is_named():
    from mepd.engines.terachem import _failure

    text = ("|       *  Supported architecture SMs: 52 61 70 86 89             *\n"
            "| Connecting to license server '1.2.3.4' port '8877'...\n"
            "|   Device 0: NVIDIA GeForce RTX 5090 Laptop GPU, 24022MB, CC 12.0 -- CPU THREAD 0\n"
            "Nuclear repulsion energy (QM atoms): 9.205226885397 a.u.\n")
    assert "SM 52, 61, 70, 86, 89 only; NVIDIA GeForce RTX 5090 Laptop GPU is SM 12.0" in _failure(text)
    assert "SM 8.9" not in _failure(text.replace("CC 12.0", "CC 8.9"))
