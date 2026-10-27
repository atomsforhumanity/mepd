"""Pieces of the QM/MM nanoreactor in a protein: protonation states set by
hand (a buried catalytic acid), the QM side-chain charge read from the state
a residue is in, xtb's SCC fallbacks and fixed electronic temperature, and
the live events of a QM/MM reactor found among its QM atoms only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mepd.errors import ElectronicStructureError


def _topology(residues):
    """An OpenMM Topology of (chain, name, id, atom names) residues."""
    app = pytest.importorskip("openmm.app")
    top = app.Topology()
    chains = {}
    for ch, name, rid, atoms in residues:
        c = chains.setdefault(ch, top.addChain(ch))
        r = top.addResidue(name, c, id=str(rid))
        for a in atoms:
            top.addAtom(a, app.Element.getBySymbol(a[0]), r)
    return top


ASP = ["N", "H", "CA", "HA", "CB", "HB2", "HB3", "CG", "OD1", "OD2", "C", "O"]


def test_protonation_states_become_modeller_variants():
    from mepd.qmmm_protein import _variants, parse_protonation

    top = _topology([("A", "ASP", 40, ASP), ("A", "ASP", 103, ASP), ("B", "ASP", 103, ASP)])
    states = parse_protonation("A:ASP103=ash, B:ASP103=ASP")
    variants, done = _variants(top, states)
    assert variants == [None, "ASH", "ASP"]
    assert done == {"A:ASP103": "ASH", "B:ASP103": "ASP"}
    with pytest.raises(ValueError, match="state of GLU"):
        _variants(top, {"A:ASP40": "GLH"})
    with pytest.raises(ValueError, match="no residue A:ASP7"):
        _variants(top, {"A:ASP7": "ASH"})
    with pytest.raises(ValueError, match="RESIDUE=STATE"):
        parse_protonation("A:ASP103")


def test_side_chain_charge_follows_the_hydrogens():
    from mepd.qmmm_protein import side_chain_charge

    top = _topology([("A", "ASP", 40, ASP), ("A", "ASP", 103, ASP + ["HD2"]),
                     ("A", "HIS", 48, ["CB", "CG", "ND1", "HD1", "CE1", "NE2", "HE2", "CD2"]),
                     ("A", "HIS", 49, ["CB", "CG", "ND1", "HD1", "CE1", "NE2", "CD2"]),
                     ("A", "TYR", 16, ["CB", "CG", "OH", "HH"])])
    assert [side_chain_charge(r) for r in top.residues()] == [-1, 0, 1, 0, 0]


def _fake_xtb(monkeypatch, converges):
    """xtb runs recorded as their argv; `converges(argv)` decides each."""
    from types import SimpleNamespace

    from mepd.engines import gfnff

    calls = []

    def run(argv, **kw):
        calls.append(list(argv))
        if converges(argv):
            return SimpleNamespace(returncode=0, stdout="| TOTAL ENERGY  -1.0 Eh |\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="scf: Self consistent charge iterator did not converge\n",
                               stderr="")

    monkeypatch.setattr(gfnff.subprocess, "run", run)
    return calls


def test_xtb_retries_a_failed_scc_and_holds_a_fixed_electronic_temperature(monkeypatch, tmp_path):
    from mepd.engines import gfnff

    calls = _fake_xtb(monkeypatch, lambda argv: "--etemp" in argv)
    eng = gfnff.XTBEngine(method="gfn2", executable="xtb")
    assert eng._run(tmp_path, ["H"], np.zeros((1, 3)), 0, grad=False, uhf=1)[0] == -1.0
    assert len(calls) == 2 and calls[1][-4:] == ["--etemp", "1000", "--iterations", "1000"]
    calls.clear()
    eng._run(tmp_path, ["H"], np.zeros((1, 3)), 0, grad=False, uhf=1, etemp=3000.0)
    assert len(calls) == 1 and calls[0][-2:] == ["--etemp", "3000.0"]   # one surface: no other etemp


def test_xtb_gives_up_after_the_last_fallback(monkeypatch, tmp_path):
    from mepd.engines import gfnff

    _fake_xtb(monkeypatch, lambda argv: False)
    with pytest.raises(ElectronicStructureError):
        gfnff.XTBEngine(method="gfn2", executable="xtb")._run(tmp_path, ["H"], np.zeros((1, 3)), 0, grad=False)


def test_hot_scc_is_set_on_the_qm_level_only_inside():
    from mepd.discovery.qmmm_reactor import _hot_scc, _xtb_qm_engine
    from mepd.engines.gfnff import XTBEngine

    class Wrapper:
        def __init__(self, base):
            self.base = base

    level = XTBEngine(method="gfn2", executable="xtb")
    xtb = _xtb_qm_engine(Wrapper(Wrapper(level)))
    assert xtb is level
    with _hot_scc(xtb, 3000.0):
        assert level.electronic_temperature == 3000.0
    assert level.electronic_temperature is None
    assert _xtb_qm_engine(Wrapper(object())) is None     # no xtb level: nothing to set
    with _hot_scc(None, 3000.0):
        pass


def _segment(fp: Path, symbols, frames):
    fp.write_text("".join(f"{len(symbols)}\n energy: 0.0\n" + "".join(
        f"{s} {x:.5f} {y:.5f} {z:.5f}\n" for s, (x, y, z) in zip(symbols, f)) for f in frames))


def test_live_events_of_a_qmmm_reactor_are_among_its_qm_atoms(tmp_path):
    """Two argon 'environment' atoms first, then H2 + H: an H atom changes
    partner. The event is found on the QM atoms, its indices in the whole
    system's numbering."""
    from mepd.discovery.nanoreactor_generator import write_live_events

    symbols = ["Ar", "Ar", "H", "H", "H"]
    before = np.array([[-8, 0, 0], [8, 0, 0], [0, 0, 0], [0.74, 0, 0], [3.0, 0, 0]], float)
    after = np.array([[-8, 0, 0], [8, 0, 0], [-1.5, 0, 0], [1.5, 0, 0], [2.24, 0, 0]], float)
    md = tmp_path / "run_00" / "md"
    md.mkdir(parents=True)
    _segment(md / "segment_000.xyz", symbols, [before] * 100 + [after] * 100)
    write_live_events(tmp_path / "run_00", total_charge=0, dt_fs=2.0, atoms=[2, 3, 4])
    data = json.loads((tmp_path / "run_00" / "live_events.json").read_text())
    assert data["events"], data
    for ev in data["events"]:
        assert set(ev["atoms"]) <= {2, 3, 4}
        assert all(set(c["atoms"]) <= {2, 3, 4} for c in ev["bond_changes"])
