import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qcref.constants import ANGSTROM_TO_BOHR
from typer.testing import CliRunner

import mepd.cli_common as cli_common
from mepd.cli import app
from mepd.cli_common import _load_structure_from_smiles_or_xyz
from mepd.discovery.network_expansion import (
    _perceived_edges, embed_product, enumerate_bond_changes, expand_network, graph_edges, lewis_smiles,
)
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode


def _node(smiles):
    return StructureNode(structure=_load_structure_from_smiles_or_xyz(smiles, None, None))


def _ang(node):
    return np.asarray(node.coords) / ANGSTROM_TO_BOHR


class _GraphEngine:
    """Optimization returns the input geometry; the energy is minus the
    number of bonds (mHa), so the fake surface still orders species."""

    def __init__(self):
        self.optimized = 0

    def _energize(self, node):
        node = node.copy()
        node._cached_energy = -1e-3 * len(graph_edges(node))
        return node

    def compute_energies(self, nodes):
        for n in nodes:
            n._cached_energy = self._energize(n)._cached_energy
        return [n._cached_energy for n in nodes]

    def compute_geometry_optimization(self, node, keywords=None):
        self.optimized += 1
        return [self._energize(node)]


def test_lewis_filter_keeps_closed_shell_products_only_by_default():
    assert lewis_smiles(["C", "H", "H", "H", "H"], [(0, 1), (0, 2), (0, 3), (0, 4)]) == "C"
    carbene = ["C", "H", "H"], [(0, 1), (0, 2)]
    assert lewis_smiles(*carbene) is None
    assert lewis_smiles(*carbene, allow_radicals=True) is not None
    hnc = ["C", "N", "H"], [(0, 1), (1, 2)]
    assert lewis_smiles(*hnc) is None  # C-≡N+-H needs separated charges
    assert lewis_smiles(*hnc, allow_zwitterions=True) == "[C-]#[NH+]"


def test_enumeration_finds_ethanol_decompositions_fewest_changes_first():
    n = _node("CCO")
    props, stats = enumerate_bond_changes(list(n.symbols), _ang(n), graph_edges(n))
    smiles = [p.smiles for p in props]
    assert {"C=C.O", "CC=O.[H][H]", "COC"} <= set(smiles)
    assert "CCO" not in smiles and len(set(smiles)) == len(smiles)
    changes = [len(p.broken) + len(p.formed) for p in props]
    assert changes == sorted(changes)
    assert not stats["clipped"]
    assert len(enumerate_bond_changes(list(n.symbols), _ang(n), graph_edges(n), max_products=2)[0]) == 2
    # Nothing to form within 0.5 A, and bare bond breaks leave radicals.
    assert enumerate_bond_changes(list(n.symbols), _ang(n), graph_edges(n), form_distance=0.5)[0] == []


@pytest.mark.parametrize("smiles, product", [("CCO", "C=C.O"), ("C#N", "[C-]#[NH+]")])
def test_embedded_guess_has_the_proposed_bonds(smiles, product):
    n = _node(smiles)
    props, _ = enumerate_bond_changes(list(n.symbols), _ang(n), graph_edges(n), allow_zwitterions=True)
    p = next(p for p in props if p.smiles == product)
    bonds = (graph_edges(n) - set(p.broken)) | set(p.formed)
    assert _perceived_edges(n.symbols, embed_product(list(n.symbols), _ang(n), bonds)) == bonds


def test_expansion_rounds_add_species_and_link_known_ones_without_reoptimizing():
    eng = _GraphEngine()
    result = expand_network(_node("C#N"), eng, rounds=2, allow_zwitterions=True)
    assert [s.smiles for s in result.species] == ["C#N", "[C-]#[NH+]"]
    assert [(e.source, e.target, e.outcome) for e in result.edges] == [(0, 1, "new_species"), (1, 0, "known_species")]
    assert eng.optimized == 2  # the seed and HNC; the back-reaction to HCN is not optimized again
    assert result.connections() == [(0, 1)]


def test_energy_window_stops_expansion_of_high_species():
    result = expand_network(_node("C#N"), _GraphEngine(), rounds=2, allow_zwitterions=True, energy_window_kcal=-1.0)
    assert len(result.rounds) == 1 and result.rounds[0]["expanded_next"] == []


def test_products_from_another_tool_by_file_must_be_atom_mapped(tmp_path):
    seed = _node("CCO")
    props, _ = enumerate_bond_changes(list(seed.symbols), _ang(seed), graph_edges(seed))
    p = next(p for p in props if p.smiles == "C=C.O")
    bonds = (graph_edges(seed) - set(p.broken)) | set(p.formed)
    xyz = embed_product(list(seed.symbols), _ang(seed), bonds)
    fp = tmp_path / "products.xyz"
    fp.write_text(f"{len(xyz)}\n\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(seed.symbols, xyz)))
    result = expand_network(seed, _GraphEngine(), products_file=str(fp))
    assert [s.smiles for s in result.species] == ["CCO", "C=C.O"]
    bad = tmp_path / "bad.xyz"
    bad.write_text("3\n\nO 0 0 0\nH 0 0 1\nH 0 1 0\n")
    with pytest.raises(ValueError, match="atom-mapped"):
        expand_network(seed, _GraphEngine(), products_file=str(bad))


def test_custom_generator_by_import_path(monkeypatch):
    seen = {}

    def generate(structure, scale=1.0):
        seen["scale"] = scale
        geom = np.array(structure.geometry) * scale
        return [structure.model_copy(update={"geometry": geom})]

    monkeypatch.setitem(sys.modules, "my_generators", SimpleNamespace(propose=generate))
    result = expand_network(_node("CCO"), _GraphEngine(), generator="my_generators:propose",
                            generator_options={"scale": 1.0})
    assert seen == {"scale": 1.0}
    assert [e.outcome for e in result.edges] == ["reverted"]  # the "product" is ethanol again


def test_cli_writes_species_and_reactions(tmp_path, monkeypatch):
    def run_inputs(_path):
        ri = RunInputs()
        ri.engine = _GraphEngine()
        return ri

    monkeypatch.setattr(cli_common, "_open_run_inputs", run_inputs)
    out = tmp_path / "out"
    res = CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "--rounds", "2", "--steer", "window", "-o", str(out)])
    assert res.exit_code == 0, res.output
    summary = json.loads((out / "summary.json").read_text())
    assert [s["smiles"] for s in summary["species"]] == ["C#N", "[C-]#[NH+]"]
    assert summary["outcomes"] == {"new_species": 1, "known_species": 1}
    assert summary["connections"] == [[0, 1]]
    assert (out / "species.xyz").exists() and (out / "species" / "species_1.xyz").exists()


def test_web_operation_runs_the_cli_and_reads_its_output(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from mepd.web.operations import get_operation
    from mepd.web.results import collect, detect_operation

    op = get_operation("graph-enumeration")
    assert op.available and op.cli_problem() is None

    def run_inputs(_path):
        ri = RunInputs()
        ri.engine = _GraphEngine()
        return ri

    monkeypatch.setattr(cli_common, "_open_run_inputs", run_inputs)
    out = tmp_path / "out"
    CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "-o", str(out)])
    assert detect_operation(out) == "graph-enumeration"
    result = collect({"op": "graph-enumeration", "output_dir": str(out), "external": True})
    assert result["headline"].startswith("2 species")
    assert "[C-]#[NH+]" in result["groups"][0]["entries"][1]["note"]


def test_live_view_animates_each_reaction_from_source_to_product(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path))
    expand_network(_node("C#N"), _GraphEngine(), rounds=2, allow_zwitterions=True)
    streams = {fp.stem: json.loads(fp.read_text()) for fp in sorted(tmp_path.glob("rxn*.json"))}
    assert list(streams) == ["rxn0001", "rxn0002"]
    forward, back = streams["rxn0001"], streams["rxn0002"]
    assert forward["kind"] == "morph" and forward["finished"] and forward["outcome"] == "new species"
    assert back["outcome"] == "known species"  # linked without re-optimizing, still animated
    frames = forward["geometry"]["frames"]
    assert len(frames) > 2  # a geodesic interpolation, not just the two ends
    first, last = (np.array([[float(v) for v in line.split()[1:]] for line in f.strip().splitlines()[2:]])
                   for f in (frames[0], frames[-1]))
    h_near = lambda x: np.argmin(np.linalg.norm(x[:2] - x[2], axis=1))  # H next to C (0) or N (1)
    assert (h_near(first), h_near(last)) == (0, 1)
    assert forward["plot"]["reactant_smiles"] == "C#N" and forward["plot"]["product_smiles"] == "[C-]#[NH+]"


def test_parallel_guesses_match_serial_and_live_animations_start_at_the_seed(tmp_path, monkeypatch):
    seed = _node("CCO")
    serial = expand_network(seed, _GraphEngine(), max_products=6, workers=1)
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path))
    parallel = expand_network(seed, _GraphEngine(), max_products=6, workers=3)
    assert [s.smiles for s in parallel.species] == [s.smiles for s in serial.species]
    for a, b in zip(serial.species, parallel.species):
        assert np.allclose(a.node.coords, b.node.coords)
    firsts = []
    for fp in sorted(tmp_path.glob("rxn*.json")):
        data = json.loads(fp.read_text())
        assert data["finished"] and len(data["geometry"]["frames"]) > 2
        firsts.append(data["geometry"]["frames"][0])
    assert len(firsts) == 6 and len(set(firsts)) == 1  # every animation begins from the same seed pose


def test_summary_names_the_methods_and_references(tmp_path, monkeypatch):
    def run_inputs(_path):
        ri = RunInputs()
        ri.engine = _GraphEngine()
        return ri

    monkeypatch.setattr(cli_common, "_open_run_inputs", run_inputs)
    out = tmp_path / "out"
    CliRunner().invoke(app, ["discovery", "expand", "CCO", "--max-products", "2", "--workers", "1", "-o", str(out)])
    methods = json.loads((out / "summary.json").read_text())["methods"]
    assert "doi:10.1002/jcc.23271" in " ".join(methods["enumeration"]["cite"])       # ZStruct
    assert "doi:10.1002/bkcs.10334" in " ".join(methods["lewis_filter"]["cite"])     # xyz2mol
    assert "not a published method" in methods["guess_geometry"]["method"]


def test_flux_steering_expands_only_species_the_kinetics_reach():
    """HCN -> HNC: a low TS lets flux into HNC (so it is expanded); a high
    one does not (so the network stops after round 1)."""
    calls = []

    def connector(ts_kcal):
        def connect(pairs, nodes):
            calls.append(list(pairs))
            out = []
            for i, j in pairs:
                ts = nodes[i].copy()
                ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + ts_kcal / 627.509
                out.append({"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"{i}-{j}"})
            return out
        return connect

    low = expand_network(_node("C#N"), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="flux",
                         connect=connector(5.0))
    assert [r["sources"] for r in low.rounds][:2] == [[0], [1]]
    assert low.kinetics.flux[1] > 0.01 and len(low.steps) == 1
    assert calls[0] == [(0, 1)] and calls[1:] == []  # round 2's back-reaction is already connected

    calls.clear()
    high = expand_network(_node("C#N"), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="flux",
                          connect=connector(60.0))
    assert len(high.rounds) == 1 and high.rounds[0]["expanded_next"] == []
    assert high.kinetics.flux[1] < 1e-6


def test_flux_steering_needs_a_connector():
    with pytest.raises(ValueError, match="connect"):
        expand_network(_node("C#N"), _GraphEngine(), steer="flux")


def test_the_same_ts_found_by_two_searches_counts_once():
    def connect(pairs, nodes):
        ts = nodes[0].copy()
        ts._cached_energy = nodes[0].energy + 0.05
        twin = ts.copy()
        twin._cached_energy = ts._cached_energy + 1e-6  # same TS, found again
        other = ts.copy()
        other._cached_energy = ts._cached_energy + 0.01  # a second, higher channel
        return [{"start": nodes[0], "end": nodes[1], "ts": t, "label": str(k)} for k, t in enumerate((ts, twin, other))]

    res = expand_network(_node("C#N"), _GraphEngine(), rounds=2, allow_zwitterions=True, steer="flux", connect=connect)
    assert [s.label for s in res.steps] == ["0", "2"]


class _ShiftEngine(_GraphEngine):
    """Energies also fall as atom 0 moves along +x (so a translated copy of a
    species is a lower 'conformer'); minimizing a structure lowers it by a
    further 1 mHa, so unminimized input sits above its own minimum."""

    def _energize(self, node, relaxed=False):
        node = super()._energize(node)
        node._cached_energy += -1e-4 * float(np.asarray(node.coords)[0, 0]) - (1e-3 if relaxed else 0.0)
        return node

    def compute_energies(self, nodes):
        for n in nodes:
            n._cached_energy = self._energize(n)._cached_energy
        return [n._cached_energy for n in nodes]

    def compute_geometry_optimization(self, node, keywords=None):
        self.optimized += 1
        return [self._energize(node, relaxed=True)]


def test_the_seed_is_minimized_before_anything_is_measured_from_it():
    seed = _node("C#N")
    raw = _ShiftEngine().compute_energies([seed.copy()])[0]
    res = expand_network(seed, _ShiftEngine(), allow_zwitterions=True)
    assert res.species[0].node.energy == pytest.approx(raw - 1e-3)
    unminimized = expand_network(seed, _ShiftEngine(), allow_zwitterions=True, minimize_seed=False)
    assert unminimized.species[0].node.energy == pytest.approx(raw)


def test_a_lower_conformer_reached_by_an_irc_becomes_the_reference():
    def connect(pairs, nodes):
        out = []
        for i, j in pairs:
            low = nodes[i].copy()
            low.structure = low.structure.model_copy(
                update={"geometry": np.asarray(low.coords) + np.array([10.0, 0.0, 0.0])})
            low._cached_energy = None
            ts = nodes[j].copy()
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + 0.02
            out.append({"start": low, "end": nodes[j], "ts": ts, "label": f"{i}-{j}"})
        return out

    before = _ShiftEngine().compute_energies([_node("C#N")])[0]
    res = expand_network(_node("C#N"), _ShiftEngine(), rounds=2, allow_zwitterions=True, steer="flux",
                         connect=connect)
    seed = res.species[0]
    assert seed.rel_energy_kcal < -0.5          # the shifted (lower) seed conformer is now the reference
    assert float(seed.node.energy) < before - 1e-3
    energies = [float(s.node.energy) for s in res.species]
    assert all(st.ts_energy >= max(energies[st.a], energies[st.b]) for st in res.steps)  # no negative barrier


def _connector(ts_kcal, calls=None):
    def connect(pairs, nodes):
        if calls is not None:
            calls.append(list(pairs))
        out = []
        for i, j in pairs:
            ts = nodes[i].copy()
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + ts_kcal / 627.509
            out.append({"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"{i}-{j}"})
        return out
    return connect


def test_access_steering_expands_species_within_reach_of_the_cutoff():
    """HCN -> HNC: with a TS 5 kcal/mol over the higher end, HNC is within a 30 kcal/mol cutoff and is
    expanded next; with a 60 kcal/mol TS it is out of reach and the network stops after round 1."""
    low = expand_network(_node("C#N"), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="access",
                         connect=_connector(5.0), access_cutoff_kcal=30.0)
    assert [r["sources"] for r in low.rounds][:2] == [[0], [1]]
    assert 1 in low.rounds[0]["access"]["within_reach"]
    high = expand_network(_node("C#N"), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="access",
                          connect=_connector(60.0), access_cutoff_kcal=30.0)
    assert len(high.rounds) == 1 and high.rounds[0]["expanded_next"] == []
    assert 1 in high.rounds[0]["access"]["out_of_reach"]


def test_access_steering_does_not_connect_products_above_the_cutoff():
    """A product lying more than the cutoff above the seed can't be reached within it: no path search.
    (_GraphEngine: one bond fewer = 0.63 kcal/mol higher.)"""
    calls = []
    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="access", connect=_connector(5.0, calls),
                         access_cutoff_kcal=0.1, max_products=20)
    assert any(s.rel_energy_kcal > 0.1 for s in res.species)          # a product above the cutoff was found ...
    assert calls == [] and res.rounds[0]["access"]["pairs_not_connected_above_cutoff"] == 1   # ... and not searched


def test_a_target_is_searched_first_and_its_route_seeds_the_frontier():
    calls = []
    res = expand_network(_node("C#N"), _GraphEngine(), rounds=1, allow_zwitterions=True, steer="access",
                         connect=_connector(5.0, calls), access_cutoff_kcal=30.0, target=_node("[C-]#[NH+]"))
    assert calls[0] == [(0, 1)]                      # round 0: seed -> target
    # the seed is expanded first; route species follow once the kinetics put material into them
    assert res.rounds[0]["round"] == 0 and res.rounds[0]["expanded_next"] == [0]


def test_access_steering_needs_a_cutoff():
    with pytest.raises(ValueError, match="access_cutoff_kcal"):
        expand_network(_node("C#N"), _GraphEngine(), steer="access", connect=_connector(5.0))


def test_a_target_behind_a_high_step_is_rerouted_through_reachable_species():
    """Methanol -> formaldehyde + H2: the direct search finds a TS far above the cutoff; the
    explorer then searches from other species within reach towards the target and finds a low route."""
    calls = []

    def connect(pairs, nodes):
        calls.append(list(pairs))
        out = []
        for i, j in pairs:
            ts = nodes[i].copy()
            high = {i, j} == {0, 1}                                      # the direct route
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + (60.0 if high else 5.0) / 627.509
            out.append({"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"{i}-{j}"})
        return out

    res = expand_network(_node("CO"), _GraphEngine(), rounds=2, steer="access", connect=connect,
                         access_cutoff_kcal=30.0, target=_node("C=O.[H][H]"), max_products=20)
    assert res.rounds[0]["target"] == 1                                 # the target's index stays readable
    assert res.rounds[0]["target_route"]["reached"] is False
    assert res.rounds[0]["target_route"]["blocking_steps"]
    reached = [r.get("target_route", {}).get("reached") for r in res.rounds]
    assert True in reached                                               # a route within the cutoff was found
    tried = [p for r in res.rounds for p in r.get("target_route", {}).get("tried_to_target", [])]
    assert all(j == 1 for _, j in tried)     # extra searches (if any were needed) all head for the target
    # one connect call per round after round 0: searches towards the target join the round's queue
    assert len(calls) - len([r for r in res.rounds if r["round"] >= 1]) <= 1 + bool(res.rounds[-1].get("final_reroute"))

    # one round only: the search the round opened up (2 -> target) still runs, after it
    calls.clear()
    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="access", connect=connect,
                         access_cutoff_kcal=30.0, target=_node("C=O.[H][H]"), max_products=20)
    assert res.rounds[-1]["target_route"]["reached"] is True   # in the round, or by the final pass


def test_access_steering_expands_only_populated_species_and_caps_each_round():
    """Within reach is not enough: the kinetics must put material into a species, and at most
    max_expand_per_round are expanded (the most populated first)."""
    res = expand_network(_node("CO"), _GraphEngine(), rounds=2, steer="access", connect=_connector(5.0),
                         access_cutoff_kcal=30.0, max_products=20, max_expand_per_round=1,
                         kinetics={"temperature": 300.0, "time_s": 3600.0, "threshold": 0.01})
    gate = res.rounds[0]["access_gate"]
    assert len(res.rounds[0]["expanded_next"]) <= 1
    assert gate["populated"] <= gate["within_reach_unexpanded"]


def _unverified_connector(ts_kcal):
    def connect(pairs, nodes):
        out = []
        for i, j in pairs:
            ts = nodes[i].copy()
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + ts_kcal / 627.509
            out.append({"start": nodes[i], "end": nodes[j], "pair": (i, j), "ts": ts, "label": f"fast_{i}_{j}",
                        "files": {"ts": f"fast_{i}_{j}.xyz", "verified": False}})
        return out
    return connect


def test_fast_steps_are_irc_checked_only_where_the_kinetics_need_them():
    holder = {}

    def verify(items):
        holder.setdefault("calls", []).append([i for i, _ in items])
        nodes = holder["res_nodes"]()
        return [(i, nodes[holder["steps"]()[i].a], nodes[holder["steps"]()[i].b]) for i, _ in items]

    res_ref = {}
    holder["res_nodes"] = lambda: [s.node for s in res_ref["r"].species]
    holder["steps"] = lambda: res_ref["r"].steps

    class Lazy(list):
        pass

    # run once to get the object, then check verification marks
    import mepd.discovery.network_expansion as ne
    orig = ne.ExpansionResult

    class Capture(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            res_ref["r"] = self

    ne.ExpansionResult = Capture
    try:
        res = expand_network(_node("C#N"), _GraphEngine(), rounds=2, allow_zwitterions=True, steer="access",
                             connect=_unverified_connector(5.0), verify=verify, access_cutoff_kcal=30.0)
    finally:
        ne.ExpansionResult = orig
    assert holder.get("calls")                                 # the populated step was checked
    assert all(st.files.get("verified") for st in res.steps)   # and is now IRC-verified


def test_a_fast_step_whose_irc_fails_is_dropped():
    res = expand_network(_node("C#N"), _GraphEngine(), rounds=1, allow_zwitterions=True, steer="access",
                         connect=_unverified_connector(5.0), verify=lambda items: [(i, None, None) for i, _ in items],
                         access_cutoff_kcal=30.0)
    assert res.steps == []


def test_structure_generator_runs_frontier_species_at_once(monkeypatch):
    # A nanoreactor-like generator: every frontier species is proposed concurrently,
    # `workers // n_seeds` at a time, instead of one after another.
    import threading
    import time
    from types import SimpleNamespace

    from mepd.discovery import generators
    from mepd.discovery.network_expansion import _propose_structures_at_once

    live, peak, lock = [0], [0], threading.Lock()

    def slow(structure, *, max_products, options):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.2)
        with lock:
            live[0] -= 1
        return [structure]

    monkeypatch.setitem(generators.GENERATORS, "slow-md",
                        generators.Generator("slow-md", "slow", "test", slow, kind="structures"))
    result = SimpleNamespace(species={k: SimpleNamespace(node=SimpleNamespace(structure=f"s{k}")) for k in range(6)})
    view = SimpleNamespace(region=None)
    got = _propose_structures_at_once(list(range(6)), result, "slow-md", {"n_seeds": 2}, None, view, 5, workers=6)
    assert got == {k: [f"s{k}"] for k in range(6)}
    assert peak[0] == 3   # 6 workers / 2 seeds per species
    # too few workers for two species at once: left to the loop
    assert _propose_structures_at_once([0, 1], result, "slow-md", {"n_seeds": 8}, None, view, 5, workers=8) == {}


def test_checkpoint_gets_the_network_after_every_round():
    seen = []
    res = expand_network(_node("CO"), _GraphEngine(), rounds=2, steer="access", connect=_connector(5.0),
                         access_cutoff_kcal=30.0, max_products=20,
                         checkpoint=lambda r: seen.append((len(r.rounds), len(r.species))))
    assert [n for n, _ in seen] == [r["round"] for r in res.rounds if r["round"] >= 1]


def test_a_pair_whose_single_ts_belongs_elsewhere_gets_a_path_search():
    """Formic acid -> CO + H2O: the single TS optimization found the carbene -> CO + H2O TS. The step is
    kept under its IRC ends, and the pair asked for still gets its full path search."""
    calls = []

    def connect(pairs, nodes, search=False):
        calls.append((list(pairs), search))
        out = []
        for i, j in pairs:
            ts = nodes[i].copy()
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + 5.0 / 627.509
            out.append({"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"{i}-{j}-{search}",
                        **({} if search else {"pair": (i, j)}),
                        "files": {"verified": False, "ts": "x.xyz", **({} if search else {"pair": [i, j]})}})
        return out

    def verify(items):   # every single-TS step's IRC lands on (target, target): no step for the pair asked
        for idx, _ in items:
            yield idx, None, None

    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="access", connect=connect, verify=verify,
                         access_cutoff_kcal=30.0, max_products=20)
    fast = [p for ps, s in calls if not s for p in ps]
    searched = [p for ps, s in calls if s for p in ps]
    assert fast and sorted(searched) == sorted(set(fast))   # each pair the single TS missed: one path search


def test_relevant_steps_get_channel_sampling_that_replaces_the_single_ts_and_adds_exits():
    """refine (mepd channels in the CLI) runs once per relevant step; its direct channel replaces the single
    TS (no double counting), its off-target exit adds a species and a step."""
    asked = []

    def refine(items):
        out = []
        for a, b, label in items:
            asked.append(label)
            ts = a.copy()
            ts._cached_energy = max(a.energy, b.energy) + 2.0 / 627.509     # a lower TS than the single one
            exit_end = _node("C=O.[H][H]")
            ets = a.copy()
            ets._cached_energy = a.energy + 3.0 / 627.509
            out.append({"direct": [{"start": a, "end": b, "ts": ts, "label": f"ch_{label}",
                                    "files": {"verified": True}}],
                        "other": [{"start": a, "end": exit_end, "ts": ets, "label": f"exit_{label}",
                                   "files": {"verified": True}}],
                        "conformers": []})
        return out

    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="access", connect=_connector(5.0),
                         access_cutoff_kcal=30.0, max_products=20, refine=refine)
    assert asked                                                          # sampled
    sampled = {frozenset(p) for r in res.rounds for p in r.get("channels_sampled", [])}
    for key in sampled:
        labels = [st.label for st in res.steps if frozenset((st.a, st.b)) == key]
        assert labels and all(l.startswith(("ch_", "exit_")) for l in labels)   # single TS replaced
    assert any(st.label.startswith("exit_") for st in res.steps)           # off-target exit joined the network
    assert len(asked) == len(set(asked))                                   # once per pair


def test_proposals_are_minimized_loosely_and_species_the_free_energies_need_tightly(monkeypatch):
    """Proposals stop at the loose threshold; before the kinetics take Hessians, the species within reach are
    minimized to the engine's own threshold. Species out of reach stay loose."""
    import mepd.discovery.network_expansion as ne
    from mepd.discovery.free_energy import FreeEnergies

    class Engine(_ShiftEngine):
        def __init__(self):
            super().__init__()
            self.fmax = []

        def compute_geometry_optimization(self, node, keywords=None):
            self.fmax.append((keywords or {}).get("fmax"))
            return super().compute_geometry_optimization(node, keywords)

    monkeypatch.setattr(ne, "_takes_fmax", lambda engine: True)
    monkeypatch.setattr(FreeEnergies, "corrections", lambda self, structures, ts: [0.0] * len(structures))
    engine = Engine()
    res = expand_network(_node("C#N"), engine, rounds=1, allow_zwitterions=True, steer="access",
                         connect=_connector(5.0), access_cutoff_kcal=30.0, kinetics={"free_energy": True})
    assert 0.05 in engine.fmax and None in engine.fmax           # loose proposals, tight seed and tightening
    reached = {k for st in res.steps for k in (st.a, st.b)}
    assert reached and not any(res.species[k].loose for k in reached)
    tight = expand_network(_node("C#N"), Engine(), rounds=1, allow_zwitterions=True, proposal_fmax=0)
    assert not any(s.loose for s in tight.species)


def test_a_path_piece_whose_ts_belongs_to_another_motion_keeps_its_path_peak(refine=None):
    """A barrierless association: the TS search from the piece's highest image lands on a rotation saddle, whose
    IRC ends are one species. The IRC check drops that TS; the piece keeps its highest image instead (an upper
    bound), as when the TS search fails outright -- otherwise the network loses the step."""
    def connect(pairs, nodes, search=False):
        out = []
        for i, j in pairs:
            ts = nodes[i].copy()
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + 5.0 / 627.509
            if not search:   # the single TS optimization: its IRC fails, so the pair gets a path search
                out.append({"start": nodes[i], "end": nodes[j], "pair": (i, j), "ts": ts, "label": f"fast_{i}_{j}",
                            "files": {"verified": False, "ts": "x.xyz", "pair": [i, j]}})
                continue
            peak = nodes[j].copy()
            peak._cached_energy = max(nodes[i].energy, nodes[j].energy) + 3.0 / 627.509
            out.append({"start": nodes[i], "end": nodes[j], "ts": ts, "peak": peak, "label": f"pair_{i}_{j}_leaf_0",
                        "files": {"verified": False, "ts": "y.xyz", "peak": f"ts/pair_{i}_{j}_leaf_0_peak.xyz"}})
        return out

    holder = {}

    def verify(items):
        for idx, _ in items:
            st = holder["r"].steps[idx]
            if st.files.get("pair"):
                yield idx, None, None
            else:                       # the rotation saddle: both IRC ends are the piece's start
                start = holder["r"].species[st.a].node
                yield idx, start.copy(), start.copy()

    import mepd.discovery.network_expansion as ne
    orig = ne.ExpansionResult

    class Capture(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder["r"] = self

    ne.ExpansionResult = Capture
    try:
        res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="access", connect=connect, verify=verify,
                             access_cutoff_kcal=30.0, max_products=20, refine=refine)
    finally:
        ne.ExpansionResult = orig
    if refine is not None:
        return res
    assert res.steps and all(st.files.get("ts_kind") == "path_peak" for st in res.steps)
    for st in res.steps:
        top = max(res.species[st.a].node.energy, res.species[st.b].node.energy)
        assert abs((st.ts_energy - top) * 627.509 - 3.0) < 1e-6
        assert st.files["ts"].endswith("_peak.xyz") and st.files["verified"] is True


def test_a_pair_whose_ts_failed_its_irc_check_gets_its_paths_sampled():
    """... and that pair gets channel sampling (all conformer pairs and mechanisms): its direct channel, the
    real TS between the two ends, replaces the path-peak bound."""
    asked = []

    def refine(items):
        out = []
        for a, b, label in items:
            asked.append(label)
            ts = a.copy()
            ts._cached_energy = max(a.energy, b.energy) + 1.0 / 627.509
            out.append({"direct": [{"start": a, "end": b, "ts": ts, "label": f"ch_{label}",
                                    "files": {"verified": True}}], "other": [], "conformers": []})
        return out

    res = test_a_path_piece_whose_ts_belongs_to_another_motion_keeps_its_path_peak(refine=refine)
    assert asked and res.steps
    assert all(st.label.startswith("ch_") for st in res.steps)   # every bound replaced by a sampled TS
    assert len(asked) == len(set(asked))


def test_explorer_channel_searches_sample_every_pair_of_every_mechanism(monkeypatch, tmp_path):
    import mepd
    from mepd.discovery.cli_expand import _channels_refiner

    seen = {}

    def fake(a, b, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(mepd, "channels", fake)
    run_inputs = SimpleNamespace(engine=SimpleNamespace())
    _channels_refiner(run_inputs, tmp_path, 1, {})([(_node("CO"), _node("C=O.[H][H]"), "0_1")])
    assert seen["pairs_per_mechanism"] == 0
    _channels_refiner(run_inputs, tmp_path, 1, {"pairs_per_mechanism": 3})([(_node("CO"), _node("C=O.[H][H]"), "0_1")])
    assert seen["pairs_per_mechanism"] == 3
