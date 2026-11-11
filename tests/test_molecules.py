"""Molecule-level kinetics of an explorer network (mepd.discovery.molecules), the explorer's default seed box of two
copies, the reactor's product cap counting molecule-level reactions, and the scoreboard's reading of it."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from qcdata import Structure

from mepd.discovery import molecules as M
from mepd.discovery.kinetics import Step

A2B = 1.8897259886


def _box(*molecules, shift=6.0):
    """A Structure of the molecules (each [(symbol, x, y, z)], Angstrom) side by side, `shift` A apart in x."""
    sy, xyz = [], []
    for k, mol in enumerate(molecules):
        for s, x, y, z in mol:
            sy.append(s)
            xyz.append([x + k * shift, y, z])
    return Structure(symbols=sy, geometry=np.array(xyz) * A2B, charge=0, multiplicity=1)


H2 = [("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)]
H_H_APART = [("H", 0.0, 0.0, 0.0), ("H", 2.5, 0.0, 0.0)]          # the same two atoms, no bond: 2 H atoms
N2 = [("N", 0.0, 0.0, 0.0), ("N", 1.10, 0.0, 0.0)]
HCN = [("H", -1.06, 0.0, 0.0), ("C", 0.0, 0.0, 0.0), ("N", 1.15, 0.0, 0.0)]
HNC = [("H", 2.17, 0.0, 0.0), ("C", 0.0, 0.0, 0.0), ("N", 1.17, 0.0, 0.0)]     # the same atoms, H on N


def test_a_step_is_the_reaction_of_the_molecules_it_changes_the_rest_look_on():
    a, b = _box(HCN, N2), _box(HNC, N2)
    s = M.split(M.molecule_parts(a), M.molecule_parts(b), M.bonds_of(a), M.bonds_of(b))
    assert s[0] == "ok" and s[1] != s[2] and len(s[1]) == len(s[2]) == 1
    assert s[3] == [0, 1, 2] and [sorted(x) for x in s[4]] == [[3, 4]]        # N2 a spectator
    # nothing changes: no reaction
    assert M.split(M.molecule_parts(a), M.molecule_parts(a), M.bonds_of(a), M.bonds_of(a)) == ("no_reaction",)


def test_a_radical_pair_box_is_no_set_of_molecules():
    # two H atoms beside N2: three molecules, two of them open-shell, in a singlet box: charges and spins unknown
    box = _box(H_H_APART, N2)
    assert M.molecule_parts(box) is None
    a = _box(H2, N2)
    assert M.split(M.molecule_parts(a), M.molecule_parts(box), M.bonds_of(a), M.bonds_of(box)) == ("open_shell_group",)


def _parts_net(boxes):
    """parts/bonds callables over boxes given as [(atoms, key)] lists, bonds made up per key (a key change = a
    reaction of that molecule)."""
    parts = lambda k: [(frozenset(a), key) for a, key in boxes[k]] if boxes[k] is not None else None
    bonds = lambda k: {(max(a), min(a)) for a, key in boxes[k] if key.endswith("=")} if boxes[k] else set()
    return parts, bonds


H = 627.509474


def test_spectators_leave_the_step_and_one_reaction_found_in_two_boxes_is_one_ts():
    # boxes: 0 = A + S, 1 = B + S (A -> B, S looks on), 2 = A + S', 3 = B + S' (the same reaction, S another one)
    boxes = [[((0, 1), "C"), ((2,), "O")], [((0, 1), "C="), ((2,), "O")],
             [((0, 1), "C"), ((2,), "N")], [((0, 1), "C="), ((2,), "N")]]
    parts, bonds = _parts_net(boxes)
    e = {"C": 0.0, "C=": 2.0, "O": -1.0, "N": -3.0}
    g = [sum(e[k] for _, k in b) / H for b in boxes]
    steps = [Step(0, 1, (e["C"] + e["O"] + 10.0) / H, "s0", {}), Step(2, 3, (e["C"] + e["N"] + 10.2) / H, "s1", {})]
    net = M.build(4, parts, bonds, steps, g, ts_info=lambda m: (None, True, True))
    # both are A -> B with S out: 10 and 10.2 kcal/mol over A; no TS geometry to tell them apart: kept as two
    assert net.names == ["C", "C=", "N", "O"]
    assert len(net.reactions) == 2 and all(x["r"] == [0] and x["p"] == [1] for x in net.reactions)
    assert sorted(x["g_ts"] - net.g[0] for x in net.reactions) == pytest.approx([10.0, 10.2])
    assert net.g[1] - net.g[0] == pytest.approx(2.0)


def test_twins_are_never_merged_and_a_box_above_its_molecules_leaves_the_fit():
    boxes = [[((0, 1), "C")], [((0, 1), "C=")], [((0, 1), "C=")]]     # box 2: the product again, 30 kcal/mol up
    parts, bonds = _parts_net(boxes)
    g = [0.0, 2.0 / H, 32.0 / H]
    steps = [Step(0, 1, 10.0 / H, "s0", {}), Step(0, 2, 40.0 / H, "s1", {}), Step(0, 1, 10.0 / H, "s0 twin", {})]
    net = M.build(3, parts, bonds, steps, g, twin_of=[0], ts_info=lambda m: (None, True, True))
    assert net.g[1] - net.g[0] == pytest.approx(2.0)          # the product at its lowest
    assert net.fit["left_out_boxes"] == 1
    # the twin of step 0 is kept beside it (two TSs); step 1, measured from its own box's energy, is 40 kcal/mol up
    assert sum(1 for x in net.reactions if x["step"] == 0) == 2 and any(x["twin"] for x in net.reactions)


def test_a_whole_box_ts_keeps_the_neighbour_it_touches_only_when_checked():
    boxes = [[((0, 1), "C"), ((2,), "O")], [((0, 1), "C="), ((2,), "O")]]
    parts, bonds = _parts_net(boxes)
    g = [0.0, 2.0 / H]
    ts = Structure(symbols=["C", "C", "O"], geometry=np.array([[0, 0, 0], [1.5, 0, 0], [2.6, 0, 0]]) * A2B)
    unchecked = M.build(2, parts, bonds, [Step(0, 1, 10.0 / H, "s", {})], g, ts_info=lambda m: (ts, False, False))
    assert unchecked.reactions == [] and unchecked.counts["left_out_whole_box_ts_unchecked"] == 1
    checked = M.build(2, parts, bonds, [Step(0, 1, 10.0 / H, "s", {})], g, ts_info=lambda m: (ts, False, True))
    (x,) = checked.reactions
    o = checked.names.index("O")
    assert o in x["r"] and o in x["p"] and x["with_neighbours"]          # the water-like neighbour on both sides


def _dimer_net(g_ts=15.0):
    """2 A <-> A2 (A2 2 kcal/mol below two A), and A <-> B (unimolecular, high)."""
    return M.MoleculeNetwork(["A", "A2", "B"], np.array([0.0, -2.0, 5.0]),
                             [{"r": [0, 0], "p": [1], "g_ts": g_ts, "step": 0, "twin": False},
                              {"r": [0], "p": [2], "g_ts": 30.0, "step": 1, "twin": False}], [0])


def test_a_dimerization_runs_at_the_concentration_given():
    net = _dimer_net()
    T, t = 298.15, 1e-3
    fast = M.simulate(net, net.c0(1.0), T, t)
    slow = M.simulate(net, net.c0(0.01), T, t)
    # second order: at 1/100 the concentration, the dimer's share of A converted is far smaller
    assert fast.final[1] / 1.0 > 10 * slow.final[1] / 0.01
    assert net.c0(1.0, copies=2)[0] == pytest.approx(0.5)


def test_the_criteria_are_relative_to_conversion_and_count_net_flux():
    net = _dimer_net(g_ts=24.0)      # slow: low conversion in the time
    kin = M.simulate(net, net.c0(1.0), 298.15, 10.0)
    assert 0 < kin.conversion < 0.05
    want = M.expansion(net, kin, 0.01)
    assert 1 in want["molecules"]                       # the dimer: all the converted material went through it
    assert 2 not in want["molecules"]                   # B: nothing
    assert (0, 1) in want["pairs"] and (0, 0) in want["pairs"]
    # a fast equilibrium carries no net flux once it sits there
    eq = M.MoleculeNetwork(["A", "B"], np.array([0.0, 3.0]),
                           [{"r": [0], "p": [1], "g_ts": 5.0, "step": 0, "twin": False}], [0])
    k = M.simulate(eq, eq.c0(1.0), 298.15, 3600.0)
    # what flowed in net (to the output grid's resolution of the early overshoot), not the traffic: ~1e12 times more
    assert k.final[1] <= k.net_flux[1] < 1.1 * k.final[1]


def test_answer_with_moves_the_answer():
    # A -> B and A -> C, 1 kcal/mol apart: B:C = exp(1/RT); lowering C's TS by 2 turns it round
    net = M.MoleculeNetwork(["A", "B", "C"], np.array([0.0, -15.0, -15.0]),
                            [{"r": [0], "p": [1], "g_ts": 20.0, "step": 0, "twin": False},
                             {"r": [0], "p": [2], "g_ts": 21.0, "step": 1, "twin": False}], [0])
    c0, rt = net.c0(1.0), 1.987204258640832e-3 * 298.15
    base = M.answer_with(net, c0, 298.15, 1e6)
    assert set(base) == {"B", "C"} and base["B"] - base["C"] == pytest.approx(1.0, abs=1e-3)
    moved = M.answer_with(net, c0, 298.15, 1e6, {("reaction", 1): -2.0})
    assert moved["C"] - moved["B"] == pytest.approx(1.0, abs=1e-3) and rt > 0


def test_the_seed_box_is_two_copies_unless_exact_too_big_or_charged():
    s = _box(N2)
    two, n = M.seed_box(s)
    assert n == 2 and len(two.symbols) == 4 and len(M.molecule_parts(two)) == 2
    assert M.seed_box(s, exact=True) == (s, 1)
    big = Structure(symbols=["H"] * 26, geometry=np.arange(78, dtype=float).reshape(26, 3) * 3)
    assert M.seed_box(big)[1] == 1
    assert M.seed_box(s.model_copy(update={"charge": 1, "multiplicity": 2}))[1] == 1


def test_the_reactor_cap_counts_new_molecule_level_reactions():
    from mepd.discovery.reactor_generator import _new_reaction

    seed = _box(HCN, N2, HCN)
    one = _box(HNC, N2, HCN)          # the first HCN turns into HNC
    other = _box(HCN, N2, HNC)        # the second one: the same reaction beside other spectators
    new = _new_reaction(seed, [])
    assert new(one) and not new(other)                     # one slot for the two
    assert new.seen == [M.reaction_key(seed, one)]
    assert not _new_reaction(seed, [M.reaction_key(seed, one)])(other)   # a reaction the network has: none


def test_discovery_expand_starts_from_two_copies_unless_exact(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import mepd.cli_common as cli_common
    import mepd.discovery.network_expansion as ne
    from mepd.cli import app
    from mepd.inputs import RunInputs
    from tests.test_network_expansion import _ShiftEngine, _filed_connector

    real = ne.expand_network
    seen = {}

    def expand(seed, *a, **kw):
        seen.setdefault("atoms", []).append(len(seed.structure.symbols))
        seen.setdefault("kinetics", []).append(dict(kw["kinetics"]))
        return real(seed, *a, **{**kw, "connect": _filed_connector(5.0, tmp_path), "verify": None, "refine": None})

    def run_inputs(_path):
        ri = RunInputs()
        ri.engine = _ShiftEngine()
        return ri

    monkeypatch.setattr(cli_common, "_open_run_inputs", run_inputs)
    monkeypatch.setattr(ne, "expand_network", expand)
    for extra, atoms, copies in (([], 6, 2), (["--exact"], 3, 1)):
        out = tmp_path / f"out{copies}"
        res = CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "--steer", "access",
                                       "--barrier-cutoff", "30", "--concentration", "0.5", "-o", str(out), *extra])
        assert res.exit_code == 0, res.output
        summary = json.loads((out / "summary.json").read_text())
        assert seen["atoms"][-1] == atoms and summary["steering"]["copies"] == copies
        assert seen["kinetics"][-1]["concentration"] == 0.5 and seen["kinetics"][-1]["copies"] == copies
        mol = summary["rounds"][-1]["kinetics"]["molecules"]
        assert mol["copies"] == copies and sum(mol["c0"]) == pytest.approx(0.5)     # HCN at 0.5 M, however many copies


def test_the_scoreboard_scores_the_molecules_first_passage():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import score

    net = M.MoleculeNetwork([M.canonical("C=O"), M.canonical("O"), M.canonical("[C-]#[O+]"), M.canonical("OC=O"),
                             M.canonical("O=C=O"), M.canonical("[H][H]")], np.zeros(6) + [0, -5, -5, 0, -5, -5],
                            [{"r": [3], "p": [1, 2], "g_ts": 20.0, "step": 0, "twin": False},
                             {"r": [3], "p": [4, 5], "g_ts": 21.0, "step": 1, "twin": False}], [3])
    rec = {**net.record(), "c0": list(net.c0(0.005)), "temperature_K": 500.0, "time_s": 1e-3}
    sysdef = {"channels": {"dehydration": {"products": ["[C-]#[O+]", "O"]},
                           "decarboxylation": {"products": ["O=C=O", "[H][H]"]}}}
    out = score.molecule_channels(sysdef, rec)
    assert out["found"] == {"dehydration": True, "decarboxylation": True} and all(out["connected"].values())
    ratio = out["amounts"]["dehydration"] / out["amounts"]["decarboxylation"]
    assert ratio == pytest.approx(math.exp(1.0 / (1.987204258640832e-3 * 500.0)), rel=1e-3)
