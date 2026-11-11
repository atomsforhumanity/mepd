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


def test_in_a_neutral_box_each_radical_is_a_species_of_its_own():
    # two H atoms beside N2 in a singlet box: three species, the H atoms radicals keyed by their bond graph
    box = _box(H_H_APART, N2)
    keys = sorted(k for _, k in M.molecule_parts(box))
    assert keys.count(M.RADICAL + M.smiles_graph_key("[H]")) == 2 and M.canonical("N#N") in keys
    a = _box(H2, N2)
    s = M.split(M.molecule_parts(a), M.molecule_parts(box), M.bonds_of(a), M.bonds_of(box))
    assert s[0] == "ok" and s[1] == [M.canonical("[H][H]")] and s[2] == [M.RADICAL + M.smiles_graph_key("[H]")] * 2


def test_b2_a_homolysis_beside_radicals_is_first_order():
    # {H2, H, H} -> {H, H, H, H}: H2 -> 2 H, the radicals already there look on
    a, b = _box(H2, H_H_APART, shift=8), _box(H_H_APART, H_H_APART, shift=8)
    s = M.split(M.molecule_parts(a), M.molecule_parts(b), M.bonds_of(a), M.bonds_of(b))
    h = M.RADICAL + M.smiles_graph_key("[H]")
    assert s[0] == "ok" and s[1] == [M.canonical("[H][H]")] and s[2] == [h, h]


def _ch3_ocho(r1=1.25, r2=1.25, ang=125.0):
    t = np.radians(ang / 2)
    ch3 = [("C", 0, 0, 0), ("H", 1.08, 0, 0), ("H", -0.54, 0.935, 0), ("H", -0.54, -0.935, 0)]
    ocho = [("C", 0, 0, 0), ("H", -1.10, 0, 0), ("O", r1 * np.cos(t), r1 * np.sin(t), 0),
            ("O", r2 * np.cos(t), -r2 * np.sin(t), 0)]
    return _box(ch3, ocho, shift=7.0)


def test_b1_a_radical_channel_is_diluted_as_a_molecular_one():
    # M -> CH3 + OCHO (two radicals, from a real geometry) against M -> C + D with the same free energies: the same
    # ratio at any concentration (a complex would run first order, undiluted: 6:1 at 1 M, 226:1 at 0.005)
    pair = sorted(k for _, k in M.molecule_parts(_ch3_ocho()))
    assert len(pair) == 2 and all(k.startswith(M.RADICAL) for k in pair)
    net = M.MoleculeNetwork(["M", *pair, "C", "D"], np.array([0.0, 5.0, 5.0, 5.0, 5.0]),
                            [{"r": [0], "p": [1, 2], "g_ts": 40.0, "step": 0, "twin": False},
                             {"r": [0], "p": [3, 4], "g_ts": 40.0, "step": 1, "twin": False}], [0])
    for c in (1.0, 0.005):
        k = M.simulate(net, net.c0(c), 1500.0, 2e-3)
        assert k.final[1] / k.final[3] == pytest.approx(1.0, rel=1e-6)


def test_a_radical_s_key_does_not_depend_on_its_geometry():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    keys = {k for r1, r2 in ((1.25, 1.25), (1.20, 1.33), (1.19, 1.36)) for _, k in M.molecule_parts(_ch3_ocho(r1, r2))}
    m = Chem.AddHs(Chem.MolFromSmiles("[O]C=O"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    x = m.GetConformer().GetPositions()
    s = Structure(symbols=[a.GetSymbol() for a in m.GetAtoms()], geometry=x * A2B, charge=0, multiplicity=2)
    ((_, embedded),) = M.molecule_parts(s)
    assert keys == {M.RADICAL + M.smiles_graph_key("[CH3]"), M.RADICAL + M.smiles_graph_key("[O]C=O")}
    assert embedded == M.RADICAL + M.smiles_graph_key("[O]C=O")


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
    assert net.fit["above_their_molecules"] == 1
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


def test_discovery_expand_starts_from_one_copy_unless_two_are_asked(tmp_path, monkeypatch):
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
    for i, (extra, atoms, copies) in enumerate((([], 3, 1), (["--copies", "2"], 6, 2),
                                                (["--copies", "2", "--exact"], 3, 1))):
        out = tmp_path / f"out{i}"
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


def test_a_step_whose_ts_search_failed_is_left_out():
    boxes = [[((0, 1), "C")], [((0, 1), "C=")]]
    parts, bonds = _parts_net(boxes)
    steps = [Step(0, 1, 10.0 / H, "peak", {"ts_kind": "path_peak"}), Step(0, 1, 12.0 / H, "ts", {})]
    net = M.build(2, parts, bonds, steps, [0.0, 2.0 / H], ts_info=lambda m: (None, True, True))
    assert [x["step"] for x in net.reactions] == [1] and net.counts["left_out_ts_search_failed"] == 1


def test_the_fit_takes_each_molecule_at_its_lowest():
    # the reviewer's toy: one molecule in six boxes (conformers), five at 0 and one at -3 kcal/mol: G = -3, not -0.5
    A = np.ones((6, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -3.0])
    assert M.lowest(A, y)[0] == pytest.approx(-3.0)
    # unique and order-independent: boxes {A,B}=0, {A2,B}=5, {A2,B2}=3, {A,B2}=2 in every order of boxes and keys
    import itertools

    A4 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=float)
    y4 = np.array([0.0, 5.0, 3.0, 2.0])
    fits = set()
    for rp in itertools.permutations(range(4)):
        for cp in itertools.permutations(range(4)):
            g = M.lowest(A4[list(rp)][:, list(cp)], y4[list(rp)])
            G = np.zeros(4)
            G[list(cp)] = g
            fits.add(tuple(np.round(A4 @ G, 6) + 0.0))
    assert len(fits) == 1
    # two molecules: each at its lowest, every box at or above its molecules' sum
    A = np.array([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    y = np.array([0.0, 1.0, 0.5, -2.0])
    G = M.lowest(A, y)
    assert G[0] == pytest.approx(-2.0) and np.all(A @ G <= y + 1e-9)


def _ions():
    """[M+H]+-like: H3O+ (one molecule, charged), and a box of H2O + H+ ... as two boxes of one atom set: O, H, H,
    H; plus a neutral spectator-free seed of OH- + H2O for scenario B."""
    h3o = [("O", 0.0, 0.0, 0.0), ("H", 0.98, 0.0, 0.0), ("H", -0.33, 0.92, 0.0), ("H", -0.33, -0.46, 0.80)]
    h2o_h = [("O", 0.0, 0.0, 0.0), ("H", 0.96, 0.0, 0.0), ("H", -0.24, 0.93, 0.0), ("H", -2.5, -2.5, 2.5)]
    return h3o, h2o_h


def test_scenario_a_a_charged_seed_keeps_its_fragmentation():
    # a charged one-molecule seed (H3O+) and its fragmentation (H2O + H+): the products' box is charged and of two
    # molecules, so one complex; the step A -> B + C at 10 kcal/mol stays, A -> D at 25 doesn't win
    h3o, h2o_h = _ions()
    a = _box(h3o).model_copy(update={"charge": 1})
    b = _box(h2o_h).model_copy(update={"charge": 1})
    pa, pb = M.molecule_parts(a), M.molecule_parts(b)
    assert len(pa) == 1 and not M.complex_info(pa[0][1])
    ((_, kb),) = pb
    assert M.complex_info(kb)[0] == M.COMPLEX_CHARGED and len(M.complex_info(kb)[1]) == 2
    boxes = {0: a, 1: b}
    d = [("O", 0.0, 0.0, 0.0), ("H", 1.5, 0.0, 0.0), ("H", -0.33, 0.92, 0.0), ("H", -0.33, -0.46, 0.80)]
    boxes[2] = _box(d).model_copy(update={"charge": 1})          # "D": another arrangement of A's atoms
    steps = [Step(0, 1, 10.0 / H, "frag", {}), Step(0, 2, 25.0 / H, "other", {})]
    net = M.build(3, lambda k: M.molecule_parts(boxes[k]), lambda k: M.bonds_of(boxes[k]), steps,
                  [0.0, -5.0 / H, -5.0 / H], ts_info=lambda m: (None, True, True))
    assert net.counts["complexes"] >= 1 and len(net.reactions) >= 1
    frag = next(x for x in net.reactions if x["step"] == 0)
    assert len(frag["r"]) == 1 and len(frag["p"]) == 1                 # first order: A -> the complex
    kin = M.simulate(net, net.c0(1.0), 298.15, 1.0)
    assert kin.final[frag["p"][0]] > 0.9                              # the 10 kcal/mol fragmentation wins


def test_scenario_b_a_charged_seed_of_several_molecules_is_one_complex_with_its_amount():
    # OH- + H2O (charged, two molecules): the seed is one complex, at the concentration given
    oh = [("O", 0.0, 0.0, 0.0), ("H", 0.97, 0.0, 0.0)]
    h2o = [("O", 0.0, 0.0, 0.0), ("H", 0.96, 0.0, 0.0), ("H", -0.24, 0.93, 0.0)]
    seed = _box(oh, h2o).model_copy(update={"charge": -1})
    parts = M.molecule_parts(seed)
    assert len(parts) == 1 and M.complex_info(parts[0][1])[0] == M.COMPLEX_CHARGED
    net = M.build(1, lambda k: parts, lambda k: M.bonds_of(seed), [], [0.0])
    assert net.seed == [0] and net.c0(0.5)[0] == pytest.approx(0.5)


def test_a_molecule_without_a_key_is_a_complex_of_its_own_with_its_reason(monkeypatch):
    import mepd.discovery.mixture as mixture

    monkeypatch.setattr(mixture, "molecule_key", lambda s: (_ for _ in ()).throw(ValueError("no key")))
    ((_, key),) = M.molecule_parts(_box(N2))
    assert M.complex_info(key)[0] == M.COMPLEX_KEY and M.complex_info(key)[1][0] == "[N][N]"


def test_the_scoreboard_finds_a_radical_channel_from_real_geometries_and_never_from_part_of_one():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import score

    ch3, ocho = sorted(k for _, k in M.molecule_parts(_ch3_ocho(1.20, 1.33)))
    partial = M.complex_key(M.COMPLEX_CHARGED, [ch3, M.canonical("O=C=O")])     # CH3 beside CO2: no channel
    net = M.MoleculeNetwork([M.canonical("COC=O"), M.canonical("CO"), M.canonical("[C-]#[O+]"), ch3, ocho, partial],
                            np.array([0.0, -5.0, -5.0, -2.5, -2.5, -5.0]),
                            [{"r": [0], "p": [1, 2], "g_ts": 20.0, "step": 0, "twin": False},
                             {"r": [0], "p": [3, 4], "g_ts": 21.0, "step": 1, "twin": False},
                             {"r": [0], "p": [5], "g_ts": 19.0, "step": 2, "twin": False}], [0])
    rec = {**net.record(), "c0": list(net.c0(1.0)), "temperature_K": 500.0, "time_s": 1e-3}
    sysdef = {"channels": {"methanol_co": {"products": ["CO", "[C-]#[O+]"]},
                           "methyl_formyloxy": {"products": ["[CH3]", "[O]C=O"]}}}
    out = score.molecule_channels(sysdef, rec)
    assert out["found"] == {"methanol_co": True, "methyl_formyloxy": True} and all(out["connected"].values())
    ratio = out["amounts"]["methanol_co"] / out["amounts"]["methyl_formyloxy"]
    assert ratio == pytest.approx(math.exp(1.0 / (1.987204258640832e-3 * 500.0)), rel=1e-3)   # the partial: none

def test_qmmm_or_restrained_systems_keep_the_box_kinetics():
    from mepd.discovery.network_expansion import expand_network
    from tests.test_network_expansion import _AsyncConnector, _GraphEngine, _node

    class Restrained(_GraphEngine):
        restraints = [object()]

    events = []
    res = expand_network(_node("C#N"), Restrained(), connect=_AsyncConnector(), steer="access",
                         access_cutoff_kcal=30.0, max_products=5, schedule="async", allow_zwitterions=True,
                         kinetics={"threshold": 1e-6}, on_event=lambda e, p: events.append((e, p)))
    assert res.kinetics_model == "box (QM/MM or restraints)"
    assert "molecules" not in res.rounds[-1]["kinetics"]
    assert any(e == "warning" and "kinetics: box (QM/MM or restraints)" in p["message"] for e, p in events)


def test_the_options_say_what_they_do(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import mepd.cli_common as cli_common
    import mepd.discovery.network_expansion as ne
    from mepd.cli import app
    from tests.test_network_expansion import _filed_connector

    real = ne.expand_network
    monkeypatch.setattr(cli_common, "_open_run_inputs", lambda _p: _ri())
    monkeypatch.setattr(ne, "expand_network", lambda *a, **kw: real(*a, **{**kw, "connect": _filed_connector(5.0, tmp_path),
                                                                          "verify": None, "refine": None}))
    run = lambda *extra: CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions",
                                                  "--barrier-cutoff", "30", "-o", str(tmp_path / str(len(extra))),
                                                  *extra])
    bad = run("--steer", "access", "--concentration", "0")
    assert bad.exit_code != 0 and "--concentration must be a number above 0" in bad.output
    nan = run("--steer", "access", "--concentration", "nan")
    assert nan.exit_code != 0 and "--concentration must be a number above 0" in nan.output
    ignored = run("--steer", "window", "--copies", "2", "--concentration", "0.5")
    assert "--copies, --concentration only act with --steer access" in ignored.output
    charged = CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "--charge", "1",
                                       "--multiplicity", "2", "--steer", "access", "--copies", "2",
                                       "--barrier-cutoff", "30", "-o", str(tmp_path / "c")])
    assert "--copies 2: one copy, as it is charged" in charged.output
    restrained = _ri()
    restrained.engine.restraints = [object()]
    monkeypatch.setattr(cli_common, "_open_run_inputs", lambda _p: restrained)
    held = CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "--steer", "access",
                                    "--copies", "2", "--barrier-cutoff", "30", "-o", str(tmp_path / "r")])
    assert "--copies 2: one copy, as its engine is QM/MM or restrained" in held.output
    monkeypatch.setattr(cli_common, "_open_run_inputs", lambda _p: _ri())
    ok = run("--steer", "access", "--concentration", "0.5")
    assert ok.exit_code == 0, ok.output
    assert "Answer (molecule-level kinetics" in ok.output
    assert json.loads((tmp_path / "4" / "summary.json").read_text())["steering"]["kinetics"] == "molecules"


def _ri():
    from mepd.inputs import RunInputs
    from tests.test_network_expansion import _ShiftEngine

    ri = RunInputs()
    ri.engine = _ShiftEngine()
    return ri


def _embedded(smiles, seed=3):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m) if AllChem.MMFFHasAllMoleculeParams(m) else AllChem.UFFOptimizeMolecule(m)
    x = m.GetConformer().GetPositions()
    unpaired = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    return Structure(symbols=[a.GetSymbol() for a in m.GetAtoms()], geometry=x * A2B, charge=0,
                     multiplicity=1 + unpaired)


def test_a_radical_s_key_carries_its_stereo():
    # E- and Z-crotyl, from real geometries: two species; each the same key from another embedding
    e1, e2 = (M.molecule_parts(_embedded("C/C=C/[CH2]", s))[0][1] for s in (3, 11))
    z1 = M.molecule_parts(_embedded("C/C=C\\[CH2]"))[0][1]
    assert e1 == e2 and e1 != z1 and M.radical_graph(e1) == M.radical_graph(z1)
    # a chiral radical's two enantiomers
    r = M.molecule_parts(_embedded("C[C@H](Cl)[CH2]"))[0][1]
    s = M.molecule_parts(_embedded("C[C@@H](Cl)[CH2]"))[0][1]
    assert r != s and M.radical_graph(r) == M.radical_graph(s)


def test_graph_keys_are_exact():
    assert M.smiles_graph_key("C1CCC2CCCCC2C1") != M.smiles_graph_key("C1CCC(C1)C1CCCC1")      # decalin, bicyclopentyl
    assert M.smiles_graph_key("CC(C)CCC") != M.smiles_graph_key("CCC(C)CC")                     # a methyl moved along


def test_the_fit_is_exact_fast_and_says_when_it_falls_back(monkeypatch):
    import time

    import scipy.optimize

    rng = np.random.default_rng(1)
    A = np.zeros((1200, 600))
    for r in range(1200):
        A[r, rng.choice(600, size=rng.integers(1, 4), replace=False)] = 1
    y = A @ rng.normal(0, 20, 600) + rng.normal(0, 2, 1200)
    M.lowest(A[:10], y[:10])                                   # (imports)
    t0 = time.monotonic()
    g = M.lowest(A, y)
    assert time.monotonic() - t0 < 2.0                         # (< 1 s on the reference machine; slack for load)
    s = y - A @ g
    assert s.min() > -1e-8
    # its point: the LP's total slack, and no feasible point of that face closer to the boxes (a random direction
    # along the face never lowers |s|^2)
    perm = rng.permutation(1200)
    g2 = M.lowest(A[perm], y[perm])
    assert np.allclose(A @ g, A @ g2, atol=1e-6)
    # forced to fail: the LP vertex, said
    def broken(*a, **k):
        raise RuntimeError("iteration limit")

    monkeypatch.setattr(scipy.optimize, "nnls", broken)
    info = {}
    A4 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=float)
    M.lowest(A4, np.array([0.0, 5.0, 3.0, 2.0]), info)
    assert info.get("fallback") is True
    boxes = [[((0, 1), "A"), ((2,), "B")], [((0, 1), "A="), ((2,), "B")], [((0, 1), "A="), ((2,), "B=")],
             [((0, 1), "A"), ((2,), "B=")]]
    parts, bonds = _parts_net(boxes)
    steps = [Step(a, b, 9.0 / H, f"s{a}{b}", {}) for a, b in ((0, 1), (0, 3), (1, 2), (3, 2))]
    net = M.build(4, parts, bonds, steps, [0.0, 5.0 / H, 3.0 / H, 2.0 / H], ts_info=lambda m: (None, True, True))
    assert "fallback" in net.fit and "fallback" in net.record()["fit"]
