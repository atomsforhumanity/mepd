"""The public API (docs/python-api.md): flat functions on `mepd`."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import mepd
from mepd.structures import as_node
from mepd.units import ANGSTROM_TO_BOHR, HARTREE_TO_KCAL

HCN = np.array([[0.0, 0.0, -1.065], [0.0, 0.0, 0.0], [0.0, 0.0, 1.156]])
HNC = np.array([[0.0, 0.0, 2.165], [0.0, 0.0, 0.0], [0.0, 0.0, 1.170]])
ETHANAL = np.array([[-1.27, -0.10, 0.00], [0.20, 0.10, 0.00], [0.80, 1.16, 0.00], [-1.80, 0.85, 0.00],
                    [-1.53, -0.67, 0.89], [-1.53, -0.67, -0.89], [0.78, -0.85, 0.00]])
VINYL_ALCOHOL = np.array([[-1.23, -0.15, 0.00], [0.10, 0.05, 0.00], [0.75, 1.24, 0.00], [1.70, 1.10, 0.00],
                          [-1.65, -1.15, 0.00], [-1.90, 0.70, 0.00], [0.75, -0.82, 0.00]])
BENT = np.array([[1.10, 0.0, 0.55], [0.0, 0.0, 0.0], [0.0, 0.0, 1.17]])   # H halfway round: a TS guess


def _node(xyz, symbols=("H", "C", "N")):
    from qcdata import Structure

    return as_node(Structure(symbols=list(symbols), geometry=np.asarray(xyz) * ANGSTROM_TO_BOHR,
                                             charge=0, multiplicity=1))


def test_public_names_are_exported_and_documented():
    for name in ("load", "read_frames", "from_smiles", "write_xyz", "engine", "engine_from_profile", "set_threads",
                 "energies", "optimize", "find_ts", "irc", "interpolate", "barrier_kcal", "same_molecule",
                 "connects", "Match", "IRCResult", "UnsupportedOperation", "ConvergenceError", "units",
                 "channels", "read_channels", "ChannelsOptions", "ChannelsResult", "Channel", "Step",
                 "ChannelSearchError"):
        assert name in mepd.__all__
        obj = getattr(mepd, name)
        assert obj.__doc__, name
    importlib.import_module("mepd.irc_geometric")   # importing the module must not shadow the function
    assert callable(mepd.irc)


def test_write_then_read_keeps_geometry_charge_and_multiplicity(tmp_path):
    from qcdata import Structure

    anion = as_node(Structure(symbols=["O", "H"], geometry=[[0, 0, 0], [0, 0, 1.83]],
                                              charge=-1, multiplicity=1))
    fp = mepd.write_xyz(tmp_path / "sub" / "two.xyz", [anion, anion], comments=["first", "second"])
    frames = mepd.read_frames(fp)
    assert len(frames) == 2 and "second" in fp.read_text().splitlines()[5]
    one = mepd.load(fp)
    assert one.structure.charge == -1
    np.testing.assert_allclose(one.coords, anion.coords, atol=1e-10)
    assert mepd.load(fp, charge=0, multiplicity=3).structure.multiplicity == 3
    with pytest.raises(FileNotFoundError):
        mepd.load(tmp_path / "missing.xyz")


def test_from_smiles_embeds_single_and_several_molecules():
    assert len(mepd.from_smiles("O").symbols) == 3
    dimer = mepd.from_smiles("O.O")
    assert len(dimer.symbols) == 6
    assert mepd.from_smiles("[OH-]").structure.charge == -1
    with pytest.raises(ValueError):
        mepd.from_smiles("this is not smiles")


def test_same_molecule_ignores_atom_order_and_conformation_but_not_stereo():
    r = mepd.from_smiles("F[C@H](Cl)Br")
    s = mepd.from_smiles("F[C@@H](Cl)Br")
    assert mepd.same_molecule(r, r)
    assert not mepd.same_molecule(r, s)
    assert mepd.same_molecule(r, s, stereo=False)
    assert not mepd.same_molecule(r, mepd.from_smiles("FC(Cl)(Cl)Br"), stereo=False)
    # atom order does not matter
    from qcdata import Structure

    order = [2, 0, 3, 1, 4]
    shuffled = as_node(Structure(
        symbols=[r.symbols[i] for i in order], geometry=np.asarray(r.coords)[order], charge=0, multiplicity=1))
    assert mepd.same_molecule(r, shuffled)


def test_connects_tries_both_directions_and_reports_each_side():
    hcn, hnc = _node(HCN), _node(HNC)
    assert mepd.connects([hcn, hnc], hcn, hnc) == mepd.Match(True, True, False)
    rev = mepd.connects([hnc, hcn], hcn, hnc)
    assert rev.both and rev.reversed and bool(rev)
    one = mepd.connects([hcn, hcn], hcn, hnc)
    assert one.start and not one.end and not one and one.any
    assert mepd.connects([hnc, hnc], hcn, hcn).none


def test_unsupported_operation_is_explicit():
    class EnergyOnly:
        def compute_energies(self, nodes):
            return np.zeros(len(nodes))

    with pytest.raises(mepd.UnsupportedOperation):
        mepd.find_ts(_node(BENT), EnergyOnly())
    with pytest.raises(mepd.UnsupportedOperation):
        mepd.optimize(_node(HCN), EnergyOnly())
    assert isinstance(mepd.UnsupportedOperation("x"), NotImplementedError)


def test_barrier_uses_the_lowest_reactant_and_needs_energies():
    ts, a, b = _node(BENT), _node(HCN), _node(HCN)
    for n, e in ((ts, -1.0), (a, -1.05), (b, -1.06)):
        n._cached_energy = e
    assert mepd.barrier_kcal(ts, [a, b]) == pytest.approx(0.06 * HARTREE_TO_KCAL)
    with pytest.raises(ValueError):
        mepd.barrier_kcal(_node(BENT), [a])          # TS energy missing, no engine
    with pytest.raises(ValueError):
        mepd.barrier_kcal(ts, [mepd.from_smiles("O")])  # different atoms


def test_interpolate_needs_mapped_endpoints():
    path = mepd.interpolate(_node(HCN), _node(HNC), nimages=5)
    assert len(path.nodes) == 5
    with pytest.raises(ValueError):
        mepd.interpolate(_node(HCN), _node(HCN, symbols=("C", "H", "N")), nimages=5)


def test_engine_by_name_matches_the_profile_route(tmp_path):
    from mepd.engines.gxtb import GXTBCalculator

    eng = mepd.engine("gxtb", n_threads=1)
    assert isinstance(eng, GXTBCalculator) and eng.n_threads == 1
    profile = tmp_path / "p.toml"
    profile.write_text('engine_name = "gxtb"\n[gxtb_engine_kwds]\nn_threads = 1\n')
    assert type(mepd.engine_from_profile(profile)) is type(eng)
    with pytest.raises(ValueError):
        mepd.engine("nonsense", foo=1)


@pytest.fixture(scope="module")
def gxtb():
    eng = mepd.engine("gxtb", n_threads=1)
    try:
        mepd.energies(_node(HCN), eng)
    except Exception as exc:  # no g-xTB binary here
        pytest.skip(f"g-xTB unavailable: {exc}")
    return eng


def test_ts_irc_and_connects_end_to_end(gxtb):
    """HCN -> HNC with g-xTB: minimize both ends, find the TS from a bent guess,
    follow its IRC, and check it connects the two (in either direction)."""
    hcn, hnc = mepd.optimize([_node(HCN), _node(HNC)], gxtb)
    ts = mepd.find_ts(_node(BENT), gxtb)
    path = mepd.irc(ts, gxtb)
    assert isinstance(path, mepd.IRCResult) and len(path.nodes) > 2
    assert mepd.connects(path, hcn, hnc).both
    barrier = mepd.barrier_kcal(ts, hcn, gxtb)
    assert 20 < barrier < 80


def test_the_docs_quickstart_runs(gxtb, tmp_path):
    """docs/python-api.md's first example, on ethanal -> vinyl alcohol (a linear
    molecule like HCN is a poor test: its bending direction is a coin toss, so
    the last digit of a unit constant can send the geodesic guess elsewhere)."""
    eng = gxtb
    sym = ("C", "C", "O", "H", "H", "H", "H")
    mepd.write_xyz(tmp_path / "reactant.xyz", _node(ETHANAL, sym))
    mepd.write_xyz(tmp_path / "product.xyz", _node(VINYL_ALCOHOL, sym))
    reactant = mepd.load(tmp_path / "reactant.xyz")
    product = mepd.load(tmp_path / "product.xyz")
    reactant, product = mepd.optimize([reactant, product], eng)

    path = mepd.interpolate(reactant, product, nimages=12)
    mepd.energies(path.nodes, eng)
    ts = mepd.find_ts(path.get_ts_node(), eng)
    irc = mepd.irc(ts, eng)
    assert mepd.connects(irc, reactant, product)
    assert 50 < mepd.barrier_kcal(ts, reactant, eng) < 100
    assert len(mepd.read_frames(mepd.write_xyz(tmp_path / "irc.xyz", irc.nodes))) == len(irc.nodes)


def test_channel_options_match_the_command_and_reject_bad_settings():
    import inspect

    from mepd.cli_channels import channels as command

    cli = {k: p.default.default for k, p in inspect.signature(command).parameters.items()}
    opts = mepd.ChannelsOptions()
    for name in mepd.ChannelsOptions.__dataclass_fields__:
        assert getattr(opts, name) == cli[name], name
    with pytest.raises(ValueError, match="--n-conformers"):
        mepd.ChannelsOptions(n_conformers=-1).validate()
    with pytest.raises(TypeError):
        mepd.ChannelsOptions(no_such_option=1)


def test_channels_finds_the_hcn_isomerization_and_reads_back(gxtb, tmp_path, capsys, monkeypatch):
    """A whole channel search from Python on HCN -> HNC (one conformer each)."""
    import logging

    from mepd.cli_common import PROGRESS_LOGGERS, _EchoHandler

    hcn, hnc = mepd.optimize([_node(HCN), _node(HNC)], gxtb)
    # importing mepd.cli (as other tests do) makes the commands' progress print; measure the library alone
    for name in PROGRESS_LOGGERS:
        log = logging.getLogger(name)
        monkeypatch.setattr(log, "handlers", [h for h in log.handlers if not isinstance(h, _EchoHandler)])
    capsys.readouterr()
    result = mepd.channels(hcn, hnc, output=tmp_path / "run", engine=gxtb, n_conformers=1, atom_mapping=False,
                           workers=1, parallel=False, validate_minima_with_hessian=False)
    printed = capsys.readouterr().out
    # the search's own progress is logged, not printed (path-search step lines still print)
    assert "Sampling conformers" not in printed and "channel(s) found for the requested pair" not in printed
    assert isinstance(result, mepd.ChannelsResult) and result.routes
    best = result.lowest()
    assert mepd.connects(best.steps[0].irc, hcn, hnc).both
    assert 20 < mepd.barrier_kcal(best.ts, result.start_conformers) < 80
    again = mepd.read_channels(tmp_path / "run")
    assert [c.folder for c in again.routes] == [c.folder for c in result.routes]
