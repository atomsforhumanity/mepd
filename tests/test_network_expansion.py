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
    _perceived_edges, _structure_with, embed_product, enumerate_bond_changes, expand_network, graph_edges,
    lewis_smiles, ts_rotamer_guesses,
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


def test_a_ts_found_again_counts_once_but_its_mirror_image_and_another_conformer_stay(tmp_path):
    """One TS, by energy and geometry: the same TS under another numbering of its identical atoms counts once;
    its mirror image (the twin TS) and another conformer's TS at the same energy are other channels."""
    import mepd

    rng = np.random.default_rng(0)

    def connect(pairs, nodes):
        base = nodes[0].copy()
        x = np.asarray(base.structure.geometry) + rng.normal(scale=0.4, size=(len(base.structure.symbols), 3))
        h = [k for k, s in enumerate(base.structure.symbols) if s == "H"]
        swapped = x.copy()
        swapped[[h[0], h[1]]] = x[[h[1], h[0]]]
        conformer = x.copy()
        conformer[h[-1]] += [0.0, 1.5, 0.0]       # its OH (or one H) elsewhere: > 0.15 A RMSD
        out = []
        for k, g in enumerate((x, swapped, conformer, x * [-1.0, 1.0, 1.0])):
            ts = StructureNode(structure=base.structure.model_copy(update={"geometry": g}))
            ts._cached_energy = nodes[0].energy + 0.05 + 1e-7 * k
            fp = tmp_path / f"{k}.xyz"
            if not fp.exists():
                mepd.write_xyz(fp, ts)
            out.append({"start": nodes[0], "end": nodes[1], "ts": ts, "label": str(k), "files": {"ts": str(fp)}})
        return out

    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="flux", connect=connect)
    assert [s.label for s in res.steps] == ["0", "2", "3"]


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


def test_a_fast_step_whose_irc_lands_on_a_step_with_the_same_ts_counts_once(tmp_path):
    """Fast TSs asked for different pairs can be one TS: once their IRCs give them the same ends, one channel."""
    import mepd

    seen = {}

    def connect(pairs, nodes, search=False):
        if search:          # the pairs the fast TSs missed: no other TS
            return []
        seen["nodes"] = nodes
        seen.setdefault("pairs", []).extend(pairs)
        fp = tmp_path / "ts.xyz"
        out = []
        for i, j in pairs:
            ts = nodes[0].copy()
            ts._cached_energy = nodes[0].energy + 5.0 / 627.509
            if not fp.exists():
                mepd.write_xyz(fp, ts)
            out.append({"start": nodes[i], "end": nodes[j], "pair": (i, j), "ts": ts, "label": f"fast_{i}_{j}",
                        "files": {"ts": str(fp), "verified": False, "pair": [i, j]}})
        return out

    def verify(items):     # every TS's IRC: the seed <-> species 1
        return [(idx, seen["nodes"][0].copy(), seen["nodes"][1].copy()) for idx, _ in items]

    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, allow_zwitterions=True, steer="access",
                         connect=connect, verify=verify, access_cutoff_kcal=30.0)
    assert len(seen["pairs"]) > 1                              # several pairs asked, so several fast TSs
    assert [(st.a, st.b) for st in res.steps] == [(0, 1)]


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
    # the log keeps the TS each sampled step had before sampling, and what replaced it
    assert {frozenset((r["a"], r["b"])) for r in res.channels_log} == sampled
    for r in res.channels_log:
        assert r["before"] and not any(b["label"].startswith("ch_") for b in r["before"])
        assert any(x["label"].startswith("ch_") for x in r["after"])


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
    monkeypatch.setattr(FreeEnergies, "corrections", lambda self, structures, ts, ends=None, energies=None: [0.0] * len(structures))
    engine = Engine()
    res = expand_network(_node("C#N"), engine, rounds=1, allow_zwitterions=True, steer="access",
                         connect=_connector(5.0), access_cutoff_kcal=30.0, kinetics={"free_energy": True})
    assert 0.05 in engine.fmax and None in engine.fmax           # loose proposals, tight seed and tightening
    reached = {k for st in res.steps for k in (st.a, st.b)}
    assert reached and not any(res.species[k].loose for k in reached)
    tight = expand_network(_node("C#N"), Engine(), rounds=1, allow_zwitterions=True, proposal_fmax=0)
    assert not any(s.loose for s in tight.species)


def test_a_step_out_of_reach_has_no_kinetics_barrier(monkeypatch):
    """A step the free-energy kinetics close (its TS above the cutoff) runs at TS + 1 Hartree, which only stops
    its rate: its kinetics barrier is None, not ~630 kcal/mol."""
    from mepd.discovery.free_energy import FreeEnergies

    monkeypatch.setattr(FreeEnergies, "corrections", lambda self, structures, ts, ends=None, energies=None: [0.0] * len(structures))
    res = expand_network(_node("C#N"), _ShiftEngine(), rounds=1, allow_zwitterions=True, steer="access",
                         connect=_connector(5.0), access_cutoff_kcal=3.0, kinetics={"free_energy": True})
    assert res.steps and res.kinetics.barriers_kcal[:len(res.steps)] == [None] * len(res.steps)
    res = expand_network(_node("C#N"), _ShiftEngine(), rounds=1, allow_zwitterions=True, steer="access",
                         connect=_connector(5.0), access_cutoff_kcal=30.0, kinetics={"free_energy": True})
    assert all(b is not None and b[0] < 10.0 for b in res.kinetics.barriers_kcal[:len(res.steps)])


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


def test_explorer_channel_searches_sample_the_best_pairs_of_every_mechanism(monkeypatch, tmp_path):
    import mepd
    from mepd.discovery.cli_expand import _channels_refiner

    seen = {}

    def fake(a, b, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(mepd, "channels", fake)
    run_inputs = SimpleNamespace(engine=SimpleNamespace(), path_min_inputs=SimpleNamespace())
    monkeypatch.setattr(mepd, "channels", lambda a, b, profile=None, **kw: (seen.update(kw, profile=profile), fake(a, b)))
    _channels_refiner(run_inputs, tmp_path, 1, {})([(_node("CO"), _node("C=O.[H][H]"), "0_1")])
    assert seen["pairs_per_mechanism"] == 3
    pmi = seen["profile"].path_min_inputs      # each mechanism's path only; a split's pieces: deferred
    assert pmi.recursive_split_max_depth == 0 and pmi.defer_split_pieces is True
    _channels_refiner(run_inputs, tmp_path, 1, {"pairs_per_mechanism": 0})([(_node("CO"), _node("C=O.[H][H]"), "0_1")])
    assert seen["pairs_per_mechanism"] == 0


def test_prefetched_channel_steps_are_sampled_once_and_again_if_their_species_changed(monkeypatch, tmp_path):
    """`refine.prefetch` samples steps in the background while the explorer verifies; `refine` then reads
    them, waits for one still running, samples the rest itself -- each step once -- and samples a step
    again when its species' structures are not the ones it was sampled for."""
    import os
    import time

    import mepd
    from mepd.discovery.cli_expand import _channels_refiner

    calls = tmp_path / "calls"
    calls.mkdir()

    def fake(a, b, output, **kw):
        label = output.name
        (calls / f"{label}-{os.getpid()}-{time.time()}").touch()
        time.sleep(1.0 if label == "pair_0_1" else 0.2)
        return SimpleNamespace(direct=[], multistep=[], offtarget=[], start_conformers=[a], end_conformers=[b])

    monkeypatch.setattr(mepd, "channels", fake)
    run_inputs = SimpleNamespace(engine=SimpleNamespace())
    a, b, c = _node("CO"), _node("C=O.[H][H]"), _node("CCO")
    refine = _channels_refiner(run_inputs, tmp_path, 2, {}, prefetch_steps=True)
    refine.prefetch([(a, b, "0_1"), (b, c, "1_2")])
    time.sleep(0.5)   # the background has started on step 0_1
    outs = refine([(a, b, "0_1"), (b, c, "1_2"), (a, c, "0_2")])
    assert all(out is not None for out in outs)
    sampled = sorted(f.name.split("-")[0] for f in calls.iterdir())
    assert sampled == ["pair_0_1", "pair_0_2", "pair_1_2"]
    assert not list(tmp_path.glob("channels/*/explorer_prefetching"))

    moved = a.update_coords(np.asarray(a.coords) + 0.05)
    refine([(moved, b, "0_1"), (b, c, "1_2")])
    sampled = sorted(f.name.split("-")[0] for f in calls.iterdir())
    assert sampled == ["pair_0_1", "pair_0_1", "pair_0_2", "pair_1_2"]
    assert (tmp_path / "channels" / "pair_0_1.stale-0").is_dir()


class _SteerEngine(_GraphEngine):
    """A 'higher level': like _GraphEngine, but structures pushed far along
    +x (the TSs of _ts_file_connector) lie 55 kcal/mol lower."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def compute_energies(self, nodes):
        self.calls += len(nodes)
        out = super().compute_energies(nodes)
        for n in nodes:
            if float(np.asarray(n.coords)[0, 0]) > 40.0:
                n._cached_energy -= 55.0 / 627.509
        return [n._cached_energy for n in nodes]


def _ts_file_connector(tmp_path, ts_kcal):
    def connect(pairs, nodes):
        out = []
        for i, j in pairs:
            ts = nodes[i].copy()
            ts.structure = ts.structure.model_copy(update={"geometry": np.asarray(ts.coords) + [50.0, 0.0, 0.0]})
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + ts_kcal / 627.509
            fp = tmp_path / f"ts_{i}_{j}.xyz"
            ts.structure.save(fp)
            out.append({"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"{i}-{j}", "files": {"ts": str(fp)}})
        return out
    return connect


def test_a_higher_levels_single_points_steer_the_flux(tmp_path):
    """HCN -> HNC with a TS 60 kcal/mol up at the explorer's level (no flux, no expansion) but 5 at the
    steering level: HNC gets the flux and is expanded; each single point is kept and reused."""
    alone = expand_network(_node("C#N"), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="flux",
                           connect=_ts_file_connector(tmp_path, 60.0))
    assert alone.rounds[0]["expanded_next"] == []
    seed = _node("C#N")
    high = _SteerEngine()
    res = expand_network(seed.copy(), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="flux",
                         connect=_ts_file_connector(tmp_path, 60.0), steer_engine=high,
                         steer_save_dir=tmp_path / "steer")
    assert [r["sources"] for r in res.rounds][:2] == [[0], [1]]
    assert res.kinetics.flux[1] > 0.01
    assert "single points" in res.rounds[0]["kinetics"]["energies"]
    assert len(list((tmp_path / "steer" / "single_points").glob("*.json"))) == 3   # 2 species + 1 TS
    again = _SteerEngine()
    expand_network(seed.copy(), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="flux",
                   connect=_ts_file_connector(tmp_path, 60.0), steer_engine=again, steer_save_dir=tmp_path / "steer")
    assert again.calls == 0


class _GPUSteerEngine(_SteerEngine):
    """The same steering level, as a torch model would declare itself."""
    holds_torch_model = True


def test_a_gpu_steering_level_runs_in_a_process_of_its_own(tmp_path):
    """Its single points are computed in one spawned process (this one never loads the model, so it can
    still fork its workers), with the same result as in-process."""
    seed = _node("C#N")
    high = _GPUSteerEngine()
    res = expand_network(seed.copy(), _GraphEngine(), rounds=3, allow_zwitterions=True, steer="flux",
                         connect=_ts_file_connector(tmp_path, 60.0), steer_engine=high,
                         steer_save_dir=tmp_path / "steer")
    assert high.calls == 0                                   # nothing computed here
    assert [r["sources"] for r in res.rounds][:2] == [[0], [1]] and res.kinetics.flux[1] > 0.01
    assert len(list((tmp_path / "steer" / "single_points").glob("*.json"))) == 3


class _Killed(BaseException):
    """A time limit's SIGTERM, as far as the explorer can tell: nothing of it catches it."""


class _LiveConnector:
    """A live connector (on_found, on_pair_done) whose TSs lie 5 kcal/mol over the higher end; `kill_after`
    pairs into the run it is 'killed'."""
    live = True
    reports_pairs = True

    def __init__(self, kill_after=None):
        self.kill_after, self.pairs, self.calls = kill_after, [], []

    def __call__(self, pairs, nodes, search=False, on_found=None, relevant=None, on_pair_done=None):
        self.calls.append(list(pairs))
        for i, j in pairs:
            if self.kill_after is not None and len(self.pairs) >= self.kill_after:
                raise _Killed()
            self.pairs.append((i, j))
            ts = nodes[i].copy()
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + 5.0 / 627.509
            on_found({"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"{i}-{j}"})
            on_pair_done((i, j))
        return []


def _expand_resumable(seed, connect, state, refine=None, resume=False, **kw):
    return expand_network(seed.copy(), _GraphEngine(), rounds=2, steer="access", connect=connect,
                          access_cutoff_kcal=60.0, max_products=20, refine=refine, state_file=state, resume=resume,
                          kinetics={"threshold": 1e-6}, **kw)


def _network(res):
    return sorted(s.smiles for s in res.species), sorted((st.a, st.b, round(st.ts_energy, 8)) for st in res.steps)


def test_a_run_cut_off_while_connecting_resumes_with_the_pairs_it_had_not_finished(tmp_path):
    seed = _node("CCO")
    whole = _LiveConnector()
    ref = _expand_resumable(seed, whole, tmp_path / "ref.pkl")
    first_round = len(whole.calls[0])
    assert len(whole.pairs) > first_round + 2      # round 2 connects pairs too
    cut = _LiveConnector(kill_after=first_round + 2)
    with pytest.raises(_Killed):
        _expand_resumable(seed, cut, tmp_path / "state.pkl")
    rest = _LiveConnector()
    res = _expand_resumable(seed, rest, tmp_path / "state.pkl", resume=True)
    assert not set(cut.pairs) & set(rest.pairs)              # nothing finished is connected again
    assert sorted(cut.pairs + rest.pairs) == sorted(whole.pairs)
    assert _network(res) == _network(ref)
    assert [r["expanded_next"] for r in res.rounds] == [r["expanded_next"] for r in ref.rounds]


def test_a_run_cut_off_in_channel_sampling_resumes_there(tmp_path):
    seed = _node("CCO")
    sampled = []

    def refine(items, kill=None):
        out = []
        for a, b, label in items:
            if kill is not None and len(sampled) >= kill:
                raise _Killed()
            sampled.append(label)
            out.append({"direct": [], "other": [], "conformers": []})
        return out

    _expand_resumable(seed, _LiveConnector(), tmp_path / "ref.pkl", refine=refine)
    n_ref, sampled[:] = len(sampled), []
    assert n_ref > 1
    with pytest.raises(_Killed):
        _expand_resumable(seed, _LiveConnector(), tmp_path / "state.pkl", refine=lambda it: refine(it, kill=n_ref - 1))
    again = _LiveConnector()
    _expand_resumable(seed, again, tmp_path / "state.pkl", refine=refine, resume=True)
    assert len(sampled) >= n_ref                 # the round's channel sampling ran again (each step: its record)
    assert len(again.calls) < 2                  # round 1's connect did not (only round 2's, if any)


def test_resuming_with_other_settings_is_refused(tmp_path):
    seed = _node("CCO")
    _expand_resumable(seed, _LiveConnector(), tmp_path / "state.pkl")
    with pytest.raises(ValueError, match="other settings"):
        expand_network(seed.copy(), _GraphEngine(), rounds=2, steer="access", connect=_LiveConnector(),
                       access_cutoff_kcal=20.0, max_products=20, state_file=tmp_path / "state.pkl", resume=True,
                       kinetics={"threshold": 1e-6})
    with pytest.raises(ValueError, match="Nothing to resume"):
        _expand_resumable(seed, _LiveConnector(), tmp_path / "none.pkl", resume=True)


def test_a_gpu_levels_label_names_its_model_without_loading_it():
    """Saved single points are keyed by the level's label; building it must not load the model (CUDA in the
    explorer's own process would stop it forking its workers), and two models must not share a label."""
    from mepd.discovery.energy_refine import _level
    from mepd.engines.fairchem import FAIRChemEngine

    small, medium = FAIRChemEngine(model="uma-s-1p2"), FAIRChemEngine(model="uma-m-1p1")
    assert "uma-s-1p2" in _level(small) and _level(small) != _level(medium)
    assert small._predictor is None and medium._predictor is None


def test_pieces_left_at_the_channel_depth_limit_are_searched_only_where_material_reaches_them():
    """A piece a step's channel searches left unsplit at their depth limit becomes a connector pair of its
    own when the kinetics put material at one of its ends (here: the seed); one between two species nobody
    reaches is not searched."""
    hot_end, cold = _node("[CH3].[OH]"), (_node("[CH2]O.[H]"), _node("[CH]O.[H][H]"))

    def refine(items):
        return [{"direct": [], "other": [], "conformers": [],
                 "deferred": [{"start": a, "end": hot_end}, {"start": cold[0], "end": cold[1]}]}
                for a, b, label in items]

    connect = _LiveConnector()
    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="access", connect=connect,
                         access_cutoff_kcal=60.0, max_products=20, refine=refine, kinetics={"threshold": 1e-6})
    smiles = [s.smiles for s in res.species]
    deferred = [d for r in res.rounds for d in r.get("deferred", [])]
    assert deferred and all(d["hot"] < d["pieces"] for d in deferred)
    searched = {tuple(p) for d in deferred for p in d["searched"]}
    assert searched == {(0, len(smiles) - 1)}     # the hot piece's far end joined the network as a species
    assert set(map(tuple, connect.calls[-1])) <= searched | {p[::-1] for p in searched}


class _IcingConnector(_LiveConnector):
    """A live connector that ices one piece on its first call and reports what it was handed later."""
    ices = True

    def __init__(self, kill_after=None):
        super().__init__(kill_after)
        self.iced, self.restored, self.free = [], None, None

    def queue_state(self):
        return {"iced": list(self.iced)}

    def restore(self, state):
        self.restored = state
        self.iced = list((state or {}).get("iced", []))

    def __call__(self, pairs, nodes, search=False, on_found=None, relevant=None, on_pair_done=None,
                 free_cache=None):
        out = super().__call__(pairs, nodes, search, on_found, relevant, on_pair_done)
        if not self.iced and len(self.calls) == 1:
            self.iced.append(((0, 1), ("a piece", 1, 1)))
        return out


def test_iced_pieces_go_into_every_connect_call_and_survive_a_resume(tmp_path):
    seed = _node("CCO")
    first = _IcingConnector()
    res = _expand_resumable(seed, first, tmp_path / "state.pkl")
    assert len(first.calls) >= 2
    assert any(r.get("iced_pieces") == 1 for r in res.rounds)
    calls_before = len(first.calls)
    import pickle

    state = pickle.load(open(tmp_path / "state.pkl", "rb"))
    assert state["queue"] == {"iced": [((0, 1), ("a piece", 1, 1))]}
    again = _IcingConnector()
    expand_network(seed.copy(), _GraphEngine(), rounds=3, steer="access", connect=again, access_cutoff_kcal=60.0,
                   max_products=20, state_file=tmp_path / "state.pkl", resume=True, kinetics={"threshold": 1e-6})
    assert again.restored == {"iced": [((0, 1), ("a piece", 1, 1))]}
    assert calls_before >= 2


def test_channel_pieces_nothing_reaches_are_kept_and_asked_again_next_round():
    hot_end, cold = _node("[CH3].[OH]"), (_node("[CH2]O.[H]"), _node("[CH]O.[H][H]"))
    rounds_seen = []

    def refine(items):
        rounds_seen.append(len(items))
        return [{"direct": [], "other": [], "conformers": [],
                 "deferred": [{"start": cold[0], "end": cold[1]}] if len(rounds_seen) == 1 else []}
                for a, b, label in items]

    res = expand_network(_node("CO"), _GraphEngine(), rounds=2, steer="access", connect=_LiveConnector(),
                         access_cutoff_kcal=60.0, max_products=20, refine=refine, kinetics={"threshold": 1e-6})
    deferred = [(r["round"], d) for r in res.rounds for d in r.get("deferred", [])]
    assert deferred and all(d["iced"] >= 1 for _, d in deferred)    # the cold piece is kept, not dropped
    assert len({rnd for rnd, _ in deferred}) == 2                   # and asked again in the next round


class _AsyncConnector(_IcingConnector):
    """The HorizontalConnector's own-job interface, run in order in this process: registered jobs, pairs
    added while a call runs, and the drain hook."""

    def __init__(self):
        super().__init__()
        self._jobs, self.end_keywords, self.queue, self.channel_labels = {}, None, [], []
        self.job_inputs = SimpleNamespace(engine=_GraphEngine())
        self.ranks, self.priorities = {}, []     # what the explorer asked of the real queue's order
        self.at_submit = []                      # (kind, label, priority when queued)

    def register(self, kind, fn, slots=1, rank=2):
        self._jobs[kind] = fn
        self.ranks[kind] = rank

    def submit(self, kind, args, on_done, label="", priority=0.0):
        self.priorities.append((kind, label, priority))
        self.at_submit.append((kind, label, priority() if callable(priority) else priority))
        self.queue.append(("own", kind, args, on_done))

    def add_pairs(self, pairs, search=False):
        self.queue.extend(("pair", p) for p in pairs)

    def add_channel(self, pair, a, b, label, priority=0.0):
        self.channel_labels.append(label)
        self.priorities.append(("channel pair", label, priority))
        self.queue.append(("pair", tuple(pair)))

    def __call__(self, pairs, nodes, search=False, on_found=None, relevant=None, on_pair_done=None,
                 free_cache=None, start=None, on_drain=None):
        self.calls.append(list(pairs))
        self.queue.extend(("pair", p) for p in pairs)
        if start is not None:
            start()
        while True:
            while self.queue:
                job = self.queue.pop(0)
                if job[0] == "own":
                    try:
                        out = self._own(job[1], job[2])
                    except Exception:          # as in the real queue: a failed job comes back as None
                        out = None
                    job[3](out)
                    continue
                i, j = job[1]
                self.pairs.append((i, j))
                on_found(self._found(i, j, nodes))
                on_pair_done((i, j))
            if not (on_drain is not None and on_drain()):
                break
        return []

    def _found(self, i, j, nodes):
        ts = nodes[i].copy()
        ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + 5.0 / 627.509
        return {"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"{i}-{j}"}

    def _own(self, kind, args):
        return self._jobs[kind](*args)


class _ForkedAsyncConnector(_AsyncConnector):
    """As _AsyncConnector, but its own jobs run in one worker process forked at the first job, as the real
    queue's pool workers are: whatever a job reads of the explorer's state is its copy from that moment."""

    def __init__(self):
        super().__init__()
        self._pipe = None

    def _own(self, kind, args):
        import multiprocessing as mp
        import os

        if self._pipe is None:
            here, there = mp.Pipe()
            pid = os.fork()
            if pid == 0:                          # the worker: run jobs until told to stop, never return
                try:
                    here.close()
                    while (msg := there.recv()) is not None:
                        try:
                            there.send(("ok", self._jobs[msg[0]](*msg[1])))
                        except Exception as exc:   # sent back: the test fails with it
                            there.send(("error", f"{type(exc).__name__}: {exc}"))
                finally:
                    os._exit(0)
            there.close()
            self._pipe, self._pid = here, pid
        self._pipe.send((kind, args))
        if not self._pipe.poll(self.timeout_s):     # a hung worker fails the test instead of hanging the suite
            self.close(kill=True)
            raise RuntimeError(f"the {kind} job got no answer from the worker in {self.timeout_s:g} s")
        status, out = self._pipe.recv()
        if status == "error":
            raise RuntimeError(f"the {kind} job failed in the worker: {out}")
        return out

    timeout_s = 300.0

    def close(self, kill=False):
        import os
        import signal

        if self._pipe is not None:
            if kill:
                os.kill(self._pid, signal.SIGKILL)
            else:
                self._pipe.send(None)
            os.waitpid(self._pid, 0)
            self._pipe = None


def _hcn_reactor(calls):
    """A stand-in "reactor" (a structures generator) for HCN <-> HNC: the H moved to the other heavy atom.
    Records the options each call got."""

    def propose(structure, *, max_products, options):
        calls.append(dict(options))
        sy, xyz = list(structure.symbols), np.asarray(structure.geometry, dtype=float).copy()
        h, c, n = sy.index("H"), sy.index("C"), sy.index("N")
        near, far = (c, n) if np.linalg.norm(xyz[h] - xyz[c]) < np.linalg.norm(xyz[h] - xyz[n]) else (n, c)
        axis = (xyz[far] - xyz[near]) / np.linalg.norm(xyz[far] - xyz[near])
        xyz[h] = xyz[far] + axis * np.linalg.norm(xyz[h] - xyz[near])
        return [structure.model_copy(update={"geometry": xyz})]

    return propose


def test_async_reactor_expands_species_found_after_the_job_workers_started(monkeypatch):
    """The reactor's options for a species (what is left of the access cutoff above it) are made when its
    proposals are queued, in the main process: a pool worker's copy of the network is the one from when it
    was forked (only the seed), so reading the species there failed for every species but the seed
    (IndexError, 07c1cd9) and the network never grew past the seed's own proposals."""
    from mepd.discovery import generators

    calls = []
    monkeypatch.setitem(generators.GENERATORS, "reactor",
                        generators.Generator("reactor", "Reactor", "test", _hcn_reactor(calls), kind="structures"))
    kw = dict(steer="access", access_cutoff_kcal=30.0, max_products=5, kinetics={"threshold": 1e-6},
              schedule="async", generator="reactor", generator_options={"n_seeds": 1}, allow_zwitterions=True)
    forked = _ForkedAsyncConnector()
    try:
        res = expand_network(_node("C#N"), _GraphEngine(), connect=forked, **kw)
    finally:
        forked.close()
    assert {0, 1} <= set(res.rounds[0]["sources"])       # HNC, found after the worker forked, was expanded
    assert "proposals_failed" not in res.rounds[0]
    # in one process: each species' call got what is left of the cutoff above that species
    res = expand_network(_node("C#N"), _GraphEngine(), connect=_AsyncConnector(), **kw)
    assert [c["cutoff_kcal"] for c in calls] == pytest.approx([30.0 - s.rel_energy_kcal for s in res.species[:2]])


def test_async_a_failed_proposal_job_is_said_and_not_counted_as_an_expansion(monkeypatch):
    """A species whose proposals failed is listed under the round's proposals_failed, not its sources: the
    summary must not read it as expanded (07c1cd9's IndexError hid behind exactly that)."""
    from mepd.discovery import generators

    hcn, calls = _hcn_reactor([]), [0]

    def propose(structure, *, max_products, options):
        calls[0] += 1
        if calls[0] > 1:
            raise RuntimeError("the generator failed")
        return hcn(structure, max_products=max_products, options=options)

    monkeypatch.setitem(generators.GENERATORS, "reactor",
                        generators.Generator("reactor", "Reactor", "test", propose, kind="structures"))
    res = expand_network(_node("C#N"), _GraphEngine(), connect=_AsyncConnector(), steer="access",
                         access_cutoff_kcal=30.0, max_products=5, kinetics={"threshold": 1e-6}, schedule="async",
                         generator="reactor", generator_options={"n_seeds": 1}, allow_zwitterions=True)
    assert res.rounds[0]["sources"] == [0] and res.rounds[0]["proposals_failed"] == [1]


def test_async_schedule_expands_species_as_the_kinetics_reach_them_in_one_queue():
    connect = _AsyncConnector()
    res = expand_network(_node("CCO"), _GraphEngine(), rounds=2, steer="access", connect=connect,
                         access_cutoff_kcal=60.0, max_products=20, kinetics={"threshold": 1e-6},
                         schedule="async")
    first = res.rounds[0]
    assert first["schedule"] == "async"
    assert 0 in first["sources"] and len(first["sources"]) > 1   # products expanded inside the same queue
    assert connect.calls[0] == []                                 # nothing was waiting for a round to end
    assert first["stopped"] == "converged"                        # it ended because nothing more was asked for
    # no distance from the seed caps it (rounds=2 is not used): species past one step are expanded too
    from collections import deque

    nbrs = {}
    for st in res.steps:
        nbrs.setdefault(st.a, set()).add(st.b)
        nbrs.setdefault(st.b, set()).add(st.a)
    dist, todo = {0: 0}, deque([0])
    while todo:
        u = todo.popleft()
        for v in nbrs.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                todo.append(v)
    expanded = {s for r in res.rounds for s in r["sources"]}
    assert max(dist[s] for s in expanded) >= 2
    assert len(set(connect.pairs)) == len(connect.pairs)          # each reaction connected once


def test_async_schedule_stops_on_a_budget_not_a_depth():
    connect = _AsyncConnector()
    res = expand_network(_node("CCO"), _GraphEngine(), steer="access", connect=connect, access_cutoff_kcal=60.0,
                         max_products=20, kinetics={"threshold": 1e-6}, schedule="async", max_species=4)
    assert res.rounds[0]["stopped"] == "max species"
    assert len(res.species) <= 4 + 20       # one batch of proposals may land after the cap is reached


def test_async_schedule_out_of_time_ends_with_what_it_has(monkeypatch):
    """Stopped (time budget, or Stop & analyze), an async run doesn't go on to sample the channels of the
    steps it found after the queue (unbudgeted: 48 min past a 1 h budget in j_4e10ef7a)."""
    from mepd import search_control

    calls = []

    def stop_requested():   # once the seed's proposals are in
        calls.append(1)
        return len(calls) > 1

    monkeypatch.setattr(search_control, "stop_requested", stop_requested)
    asked = []

    def refine(items):
        asked.extend(items)
        return [{"direct": [], "other": [], "conformers": []} for _ in items]

    refine.pairs, refine.width = (lambda item: {"start": [], "end": [], "pairs": []}), 3
    res = expand_network(_node("CCO"), _GraphEngine(), steer="access", connect=_AsyncConnector(),
                         access_cutoff_kcal=60.0, max_products=20, kinetics={"threshold": 1e-6}, schedule="async",
                         refine=refine)
    assert res.rounds[0]["stopped"] == "stop requested" and res.steps
    assert asked == []


def test_async_schedule_needs_access_steering_and_a_job_queue():
    with pytest.raises(ValueError, match="HorizontalConnector"):
        expand_network(_node("CO"), _GraphEngine(), steer="access", connect=_LiveConnector(),
                       access_cutoff_kcal=60.0, schedule="async")
    with pytest.raises(ValueError, match="steer='access'"):
        expand_network(_node("CO"), _GraphEngine(), steer="flux", connect=_AsyncConnector(), schedule="async")


def test_every_proposed_pair_gets_channel_sampling_whatever_its_single_ts_found():
    """Formic acid -> CO + H2O: the single TS landed on the carbene's step. The pair's channels (conformers,
    mechanisms) are sampled all the same, which is where the direct four-centre TS is found."""
    asked = []

    def connect(pairs, nodes, search=False):
        out = []
        for i, j in pairs:
            ts = nodes[i].copy()
            ts._cached_energy = max(nodes[i].energy, nodes[j].energy) + 5.0 / 627.509
            out.append({"start": nodes[i], "end": nodes[j], "ts": ts, "label": f"{i}-{j}-{search}", "pair": (i, j),
                        "files": {"verified": False, "ts": "x.xyz", "pair": [i, j]}})
        return out

    def verify(items):   # every single-TS step's IRC lands elsewhere: no step for the pair asked
        for idx, _ in items:
            yield idx, None, None

    def refine(items):
        asked.extend(tuple(sorted(map(int, label.split("_")))) for _, _, label in items)
        return [{"direct": [], "other": [], "conformers": []} for _ in items]

    res = expand_network(_node("CO"), _GraphEngine(), rounds=1, steer="access", connect=connect, verify=verify,
                         access_cutoff_kcal=30.0, max_products=20, refine=refine)
    proposed = {tuple(sorted(p)) for e in res.edges if e.target is not None and e.target != e.source
                for p in [(e.source, e.target)]}
    assert proposed and proposed <= set(asked)


def test_channel_pieces_handed_back_unminimized_are_read_as_deferred(tmp_path):
    from mepd.discovery.cli_expand import _deferred_pieces

    tree = tmp_path / "pairs" / "pair_0" / "tree"
    tree.mkdir(parents=True)
    frame = lambda x: f"2\n\nH {x} 0 0\nH {x + 0.74} 0 0\n"
    for k in range(2):
        (tree / f"node_0_deferred_{k}.xyz").write_text(frame(k) + frame(k + 5))
    pieces = _deferred_pieces(tmp_path)
    assert [round(float(p["start"].structure.geometry[0][0] * 0.529177), 2) for p in pieces] == [0.0, 1.0]


def test_async_schedule_samples_channels_inside_the_one_queue_without_a_stage_barrier():
    """Channel sampling hands back conformer pairs; each is connected by the same queue (a fast TS first),
    not by a path search of its own, and nothing waits for a stage to end."""
    connect = _AsyncConnector()
    sampled = []

    def refine(items):
        raise AssertionError("the async queue never runs a whole channel search")

    def pairs(item):
        a, b, label = item
        sampled.append(label)
        assert len(connect.calls) == 1                                     # while the queue runs
        return {"start": [a], "end": [b], "pairs": [(a.copy(), b.copy()), (a.copy(), b.copy())]}

    refine.pairs, refine.width = pairs, 3
    res = expand_network(_node("CCO"), _GraphEngine(), rounds=2, steer="access", connect=connect,
                         access_cutoff_kcal=60.0, max_products=20, kinetics={"threshold": 1e-6},
                         schedule="async", refine=refine)
    assert sampled and len(connect.calls) == 1 and len(res.rounds) == 1   # one queue, no cycles after it
    assert res.rounds[0]["channels"] == len(sampled) == len(set(sampled))  # each species pair sampled once
    assert len(connect.channel_labels) == 2 * len(sampled) == len(set(connect.channel_labels))


def _butadiene_acrylate():
    """Butadiene + methyl acrylate (the seed's atom order) and its Diels-Alder adduct (bonds 0-5 and 3-4 formed)."""
    reactant = _node("C=CC=C.C=CC(=O)OC")
    symbols = [str(s) for s in reactant.structure.symbols]
    edges = graph_edges(reactant) | {(0, 5), (3, 4)}
    adduct = StructureNode(structure=_structure_with(reactant.structure, embed_product(symbols, _ang(reactant), edges)))
    assert graph_edges(adduct) == edges
    return reactant, adduct


def test_ts_rotamers_turn_the_bonds_outside_the_reacting_core():
    """A TS's rotamers: its rotatable bonds outside the reacting core (the bonds its two IRC ends don't share)
    turned one at a time and two together. A Diels-Alder TS of methyl acrylate: the acrylate C-C(=O) and the ester
    C(=O)-O bonds turn (s-cis/s-trans, Z/E); bonds of the forming ring and the methyl (a symmetric top) don't, and
    every core distance and bond stays as it was."""
    reactant, adduct = _butadiene_acrylate()
    guesses, core = ts_rotamer_guesses(adduct.structure, (reactant.structure, adduct.structure))
    assert core == [(0, 5), (3, 4)]
    assert [what for _, what in guesses] == ["5-6 180", "6-8 180", "5-6 180, 6-8 180"]
    x0 = _ang(adduct)
    for guess, what in guesses:
        x = _ang(StructureNode(structure=guess))
        for i, j in core + sorted(graph_edges(adduct)):      # the core's distances and every bond's length kept
            assert np.linalg.norm(x[i] - x[j]) == pytest.approx(np.linalg.norm(x0[i] - x0[j]), abs=1e-9)
        moved = set(np.where(np.abs(x - x0).sum(axis=1) > 1e-6)[0])
        assert moved and moved <= {7, 8, 9, 19, 20, 21}      # the ester's atoms only (6 is on the first axis)


class _TSFileAsyncConnector(_ForkedAsyncConnector):
    """As _ForkedAsyncConnector, with each pair's TS written to `ts_dir` as the real queue's fast TSs are
    (fast_<i>_<j>.xyz, unverified, the pair it was asked for)."""

    def __init__(self, ts_dir):
        super().__init__()
        self.ts_dir = ts_dir

    def _found(self, i, j, nodes):
        import mepd

        rec = super()._found(i, j, nodes)
        fp = self.ts_dir / f"fast_{i}_{j}.xyz"
        mepd.write_xyz(fp, rec["ts"], comments=[f"E={rec['ts'].energy:.10f}"])
        rec.update(pair=(i, j), label=fp.stem, files={"ts": str(fp), "verified": False, "pair": [i, j]})
        return rec


def test_async_rotamers_of_an_irc_checked_ts_join_its_step_from_a_forked_worker(monkeypatch, tmp_path):
    """F0007: each adduct's single TS is the conformer its proposal happened to be in (endo s-trans, exo with the
    ester E). Once a TS the kinetics use has its IRC, its rotamers are TS searches of the queue (ahead of channel
    sampling and later proposals), run in its forked workers: each that keeps the reacting core is one more TS of
    the step, IRC-checked like any; a rotamer's own rotamers are not searched."""
    import mepd
    from mepd.discovery import generators

    reactant, adduct = _butadiene_acrylate()

    def diels_alder(structure, *, max_products, options):      # the seed's one proposal: the adduct
        return [adduct.structure] if len(graph_edges(StructureNode(structure=structure))) == 20 else []

    def find_ts(node, engine, **kw):                           # the guess is a TS; each rotamer its own energy
        ts = StructureNode(structure=node.structure)
        ts._cached_energy = -0.012 + 1e-3 * (float(np.abs(np.asarray(node.structure.geometry)).sum()) % 1.0)
        return ts

    def verify(items):                                         # every TS connects the seed and the adduct
        for idx, fp in items:
            mepd.write_xyz(Path(fp).with_name(f"{Path(fp).stem}_irc.xyz"), [reactant, adduct])
        return [(idx, reactant.copy(), adduct.copy()) for idx, _ in items]

    def refine(items):
        raise AssertionError("the async queue never runs a whole channel search")

    refine.pairs, refine.width = (lambda item: {"start": [], "end": [], "pairs": []}), 3
    monkeypatch.setitem(generators.GENERATORS, "reactor",
                        generators.Generator("reactor", "Reactor", "test", diels_alder, kind="structures"))
    monkeypatch.setattr(mepd, "find_ts", find_ts)
    from pathlib import Path

    forked = _TSFileAsyncConnector(tmp_path)
    try:
        res = expand_network(reactant.copy(), _GraphEngine(), connect=forked, verify=verify, refine=refine,
                             steer="access", access_cutoff_kcal=30.0, max_products=5, kinetics={"threshold": 1e-6},
                             schedule="async", generator="reactor", generator_options={"n_seeds": 1})
    finally:
        forked.close()
    rotamers = [st for st in res.steps if st.files.get("rotamer_of")]
    assert {(st.a, st.b) for st in res.steps} == {(0, 1)}
    assert len(rotamers) == 3 and {st.files["rotamer_of"] for st in rotamers} == {"fast_0_1"}
    assert all(st.files["verified"] is True and Path(st.files["ts"]).is_file() for st in rotamers)
    jobs = [(label, p) for kind, label, p in forked.priorities if kind == "rotamer"]
    assert len(jobs) == 3                                      # none for the rotamers' own TSs
    parent = next(st for st in res.steps if not st.files.get("rotamer_of"))
    assert all(p == pytest.approx((parent.ts_energy - res.species[0].node.energy) * 627.509474) for _, p in jobs)
    assert forked.ranks["rotamer"] < forked.ranks["channels"] < forked.ranks["propose"]


def test_async_channel_sampling_is_queued_lowest_step_first():
    """Each step's conformer-pair job, and the TS searches of its pairs, take as their priority in the queue the
    height above the seed of the lowest TS found between its species so far, or, before there is one (the seed's
    channel jobs are queued with its proposals; a single TS can fail or belong to another step), of the higher of
    the two species, which no TS between them can lie below. Asked whenever the queue orders its jobs, so the
    steps the flux runs through are sampled first (F0007: the endo and exo steps waited 4-12 minutes behind the
    conformer pairs of 35-42 steps that carry no flux; an exo pair whose single TS went elsewhere, 12 more)."""
    connect = _AsyncConnector()

    def refine(items):
        raise AssertionError("the async queue never runs a whole channel search")

    refine.pairs = lambda item: {"start": [item[0]], "end": [item[1]], "pairs": [(item[0].copy(), item[1].copy())]}
    refine.width = 3
    res = expand_network(_node("CCO"), _GraphEngine(), steer="access", connect=connect, access_cutoff_kcal=60.0,
                         max_products=20, kinetics={"threshold": 1e-6}, schedule="async", refine=refine)
    e0 = res.species[0].node.energy
    height = {}
    for st in res.steps:
        key = tuple(sorted((st.a, st.b)))
        height[key] = min(height.get(key, np.inf), (st.ts_energy - e0) * 627.509474)
    pair_of = lambda kind, label: tuple(sorted(map(int, label.split("→") if kind == "channels" else label.split("_")[1:3])))
    asked = [(kind, label, p) for kind, label, p in connect.priorities if kind in ("channels", "channel pair")]
    assert asked and len({p() for _, _, p in asked}) > 1
    for kind, label, p in asked:
        assert p() == pytest.approx(height[pair_of(kind, label)])
    floor = lambda a, b: (max(res.species[a].node.energy, res.species[b].node.energy) - e0) * 627.509474
    queued = [(label, p) for kind, label, p in connect.at_submit if kind == "channels"]
    assert queued and all(p == pytest.approx(floor(*pair_of("channels", label))) for label, p in queued)

def test_async_a_species_first_found_as_an_irc_end_is_connected_from_the_species_asked(monkeypatch, tmp_path):
    """The seed's single TS towards HNC turns out, by its IRC, to join HNC and H + CN, a species no proposal made.
    That species is then a pair of its own from the seed (whose reaction the search was asked for): otherwise its
    only route is the step that found it (F0007: adducts found only as the ends of TSs between two other species,
    48-81 kcal/mol up, were 11 of the baseline's 16 channels found but not connected)."""
    import mepd
    from mepd.discovery import generators

    calls = []
    monkeypatch.setitem(generators.GENERATORS, "reactor",
                        generators.Generator("reactor", "Reactor", "test", _hcn_reactor(calls), kind="structures"))
    seed = _node("C#N")
    hnc = StructureNode(structure=_hcn_reactor([])(seed.structure, max_products=1, options={})[0])
    x = _ang(seed)
    x[2] = x[1] + (x[1] - x[0]) / np.linalg.norm(x[1] - x[0]) * 3.0       # the H 3 A past the N: H + CN apart
    torn = StructureNode(structure=_structure_with(seed.structure, x))

    def verify(items):
        out = []
        for idx, fp in items:
            if Path(fp).stem == "fast_0_1":
                mepd.write_xyz(Path(fp).with_name("fast_0_1_irc.xyz"), [hnc, torn])
                out.append((idx, hnc.copy(), torn.copy()))
            else:
                out.append((idx, None, None))
        return out

    from pathlib import Path

    forked = _TSFileAsyncConnector(tmp_path)
    try:
        res = expand_network(seed.copy(), _GraphEngine(), connect=forked, verify=verify, steer="access",
                             access_cutoff_kcal=30.0, max_products=5, kinetics={"threshold": 1e-6},
                             schedule="async", generator="reactor", generator_options={"n_seeds": 1},
                             allow_zwitterions=True)
    finally:
        forked.close()
    k = next(i for i, sp in enumerate(res.species) if graph_edges(sp.node) == graph_edges(torn))
    assert not any(e.target == k and e.source != k for e in res.edges)    # no proposal made it
    assert (0, k) in forked.pairs                              # connected from the seed all the same


class _DrainCheckingConnector(_AsyncConnector):
    """_AsyncConnector whose TSs are written to files (the steering level needs their geometry), and which,
    like the real queue, takes the drain hook at its word: jobs it queued while saying it added nothing are
    left over (the real queue tried to ice them as pieces)."""

    def __init__(self, tmp_path):
        super().__init__()
        self.tmp_path, self.left_over = tmp_path, []

    def _found(self, i, j, nodes):
        rec = super()._found(i, j, nodes)
        fp = self.tmp_path / f"ts_{i}_{j}.xyz"
        rec["ts"].structure.save(fp)
        return {**rec, "files": {"ts": str(fp)}}

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        self.left_over += [job[1] for job in self.queue if job[0] == "own"]
        return out


def test_async_the_steering_single_points_the_drains_kinetics_ask_for_run(monkeypatch, tmp_path):
    """sb-t14 hcooh r4 (develop 544cb14) crashed, "TypeError: 'NoneType' object is not iterable": the queue's
    last job added a step, its kinetics refresh was held back (at most once in three times as long as the last
    took), the drain's refresh queued that step's TS single point at the steering level, and the drain hook
    said it had queued nothing -- so the queue iced the single point's job as if it were a piece. Here every
    refresh but the first and the drains' is held back (the kinetics 'take' 100 s on a clock that only they move)."""
    import time

    import mepd.discovery.kinetics as kinetics

    clock = [0.0]
    simulate = kinetics.simulate

    def slow_simulate(*a, **k):
        clock[0] += 100.0
        return simulate(*a, **k)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(kinetics, "simulate", slow_simulate)
    connect = _DrainCheckingConnector(tmp_path)
    res = expand_network(_node("C=O"), _GraphEngine(), steer="access", connect=connect, access_cutoff_kcal=60.0,
                         max_products=20, kinetics={"threshold": 1e-6}, schedule="async", allow_zwitterions=True,
                         steer_engine=_SteerEngine(), steer_save_dir=tmp_path / "steer")
    assert res.rounds[0]["stopped"] == "converged"
    assert any(kind == "steer" for kind, _, _ in connect.at_submit)
    assert connect.left_over == []


def test_async_a_species_first_found_as_a_rotamers_irc_end_is_connected_from_the_species_asked(monkeypatch, tmp_path):
    """Isoprene + methyl acrylate r3 (sb-t14, both arms; para:meta 2.8e8, 15 kcal/mol off): no proposal made the
    meta adduct, and it was found only as the IRC end of a rotamer of the seed's hetero-Diels-Alder TS (the [3,3]
    shift of that adduct to meta). A rotamer's step carried no asked pair, so the rule above passed it by: the seed
    was never paired with meta, whose only route stayed the stepwise one. Now it is, as for any TS's IRC end; a
    rotamer whose IRC goes elsewhere still asks for no path search of its step's pair (its TS is the parent's)."""
    import mepd
    from mepd.discovery import generators

    reactant, adduct = _butadiene_acrylate()
    x = _ang(adduct)
    h = next(i for i, s in enumerate(adduct.structure.symbols) if str(s) == "H")
    x[h] = x[h] + 3.0                                          # an H torn off: a species no proposal makes
    torn = StructureNode(structure=_structure_with(adduct.structure, x))

    def diels_alder(structure, *, max_products, options):
        return [adduct.structure] if len(graph_edges(StructureNode(structure=structure))) == 20 else []

    def find_ts(node, engine, **kw):
        ts = StructureNode(structure=node.structure)
        ts._cached_energy = -0.012 + 1e-3 * (float(np.abs(np.asarray(node.structure.geometry)).sum()) % 1.0)
        return ts

    def verify(items):                                         # the first rotamer's IRC joins the adduct and `torn`
        out = []
        for idx, fp in items:
            a, b = (adduct, torn) if Path(fp).stem == "fast_0_1_rot0" else (reactant, adduct)
            mepd.write_xyz(Path(fp).with_name(f"{Path(fp).stem}_irc.xyz"), [a, b])
            out.append((idx, a.copy(), b.copy()))
        return out

    def refine(items):
        raise AssertionError("the async queue never runs a whole channel search")

    refine.pairs, refine.width = (lambda item: {"start": [], "end": [], "pairs": []}), 3
    monkeypatch.setitem(generators.GENERATORS, "reactor",
                        generators.Generator("reactor", "Reactor", "test", diels_alder, kind="structures"))
    monkeypatch.setattr(mepd, "find_ts", find_ts)
    from pathlib import Path

    forked = _TSFileAsyncConnector(tmp_path)
    searched = []
    add_pairs = forked.add_pairs
    forked.add_pairs = lambda pairs, search=False: (searched.extend(pairs) if search else None, add_pairs(pairs))
    try:
        res = expand_network(reactant.copy(), _GraphEngine(), connect=forked, verify=verify, refine=refine,
                             steer="access", access_cutoff_kcal=30.0, max_products=5, kinetics={"threshold": 1e-6},
                             schedule="async", generator="reactor", generator_options={"n_seeds": 1},
                             allow_radicals=True)
    finally:
        forked.close()
    k = next(i for i, sp in enumerate(res.species) if graph_edges(sp.node) == graph_edges(torn))
    assert any(st.files.get("rotamer_of") and {st.a, st.b} == {1, k} for st in res.steps)
    assert not any(e.target == k and e.source != k for e in res.edges)    # no proposal made it
    assert (0, k) in forked.pairs                              # connected from the seed all the same
    assert (0, 1) not in searched                              # and no path search of the seed-adduct pair


def test_async_schedule_at_max_species_ends_with_what_it_has():
    """An async run stopped by max_species, like one out of time, doesn't go on to sample channels or IRC-check
    steps after its queue (unbudgeted, outside the queue: F0007 reached 200 species at 1459 s of a 1800 s budget,
    then ran whole channel searches for 40 minutes and failed queueing a steering single point into the closed
    queue)."""
    asked = []

    def refine(items):
        asked.extend(items)
        return [{"direct": [], "other": [], "conformers": []} for _ in items]

    refine.pairs, refine.width = (lambda item: {"start": [], "end": [], "pairs": []}), 3
    res = expand_network(_node("CCO"), _GraphEngine(), steer="access", connect=_AsyncConnector(),
                         access_cutoff_kcal=60.0, max_products=20, kinetics={"threshold": 1e-6}, schedule="async",
                         refine=refine, max_species=4)
    assert res.rounds[0]["stopped"] == "max species" and res.steps
    assert asked == []


def test_the_cli_caps_an_expansion_at_400_species_by_default():
    import inspect

    from mepd.discovery.cli_expand import expand

    assert inspect.signature(expand).parameters["max_species"].default.default == 400



class _UnverifiedAsyncConnector(_AsyncConnector):
    """As _AsyncConnector, with TS files: the seed's TSs come back IRC-checked, except that the one asked for its
    second product joins its first and second products instead, unchecked (as the real queue's single TSs are),
    and the one asked for its third joins its second and third, checked (reached only through the unchecked one)."""

    def __init__(self, tmp_path):
        super().__init__()
        self.tmp_path = tmp_path

    def _found(self, i, j, nodes):
        out = super()._found(i, j, nodes)
        a, b, checked = {(0, 2): (1, 2, False), (0, 3): (2, 3, True)}.get((i, j), (i, j, True))
        fp = self.tmp_path / f"ts_{a}_{b}.xyz"
        out["ts"]._cached_energy = max(nodes[a].energy, nodes[b].energy) + 5.0 / 627.509
        out["ts"].structure.save(fp)
        return {**out, "pair": (a, b), "label": f"{a}-{b}", "files": {"ts": str(fp), "verified": checked,
                                                                     "pair": [i, j]}}


def _no_irc(items):
    raise AssertionError("no IRC check expected: the unchecked step touches no populated species")


def _run_unchecked(tmp_path, **kw):
    return expand_network(_node("CCO"), _GraphEngine(), steer="access", connect=_UnverifiedAsyncConnector(tmp_path),
                          verify=_no_irc, access_cutoff_kcal=60.0, max_products=20, kinetics={"threshold": 2.0},
                          schedule="async", **kw)


def test_out_of_time_the_final_kinetics_leave_out_the_steps_no_irc_checked(monkeypatch, tmp_path):
    """An async run that ran out of time skips the end-of-run IRC checks: its final kinetics rest on IRC-checked steps
    only (the F0007 smoke run put all its product through one unchecked step, a conformer change's saddle 4.4 kcal/mol
    below the seed). The unchecked step stays in the network, marked as left out of the final kinetics, and why."""
    from mepd import search_control
    from mepd.discovery.network_expansion import UNVERIFIED_LEFT_OUT

    calls = []
    monkeypatch.setattr(search_control, "stop_requested", lambda: calls.append(1) or len(calls) > 2)
    res = _run_unchecked(tmp_path)
    assert res.rounds[0]["stopped"] == "stop requested"
    m = next(m for m, st in enumerate(res.steps) if st.files.get("pair") == [0, 2])   # the step stays
    assert res.steps[m].files["verified"] is False
    assert res.steps[m].files["left_out_of_final_kinetics"] == UNVERIFIED_LEFT_OUT
    assert res.kinetics.barriers_kcal[m] is None                                        # no rate through it
    beyond = next(n for n, st in enumerate(res.steps) if st.files.get("pair") == [0, 3])   # 2 -> 3, checked
    assert res.kinetics.barriers_kcal[beyond] is None                  # only the unchecked step reached it: closed
    assert "left_out_of_final_kinetics" not in res.steps[beyond].files
    checked = [n for n, st in enumerate(res.steps) if st.files.get("verified") is True and n != beyond]
    assert checked and all(res.kinetics.barriers_kcal[n] is not None for n in checked)
    assert not any("left_out_of_final_kinetics" in res.steps[n].files for n in checked)
    assert "1 step(s) no IRC checked left out, 1 only they reached closed" in res.rounds[0]["kinetics"]["energies"]
    # every step the final kinetics keep open is reached from the seed through open steps (what a rebuild of the
    # kinetics from the summary's barriers needs, benchmarks/score.py)
    opened = [st for st, b in zip(res.steps, res.kinetics.barriers_kcal) if b is not None]
    reach = {0}
    for _ in opened:
        reach |= {x for st in opened if {st.a, st.b} & reach for x in (st.a, st.b)}
    assert all(st.a in reach and st.b in reach for st in opened)


def test_in_budget_the_final_kinetics_are_as_before(monkeypatch, tmp_path):
    """A run that ends within its budget (its end-of-run IRC checks run) is unchanged: the same network and final
    kinetics as with the out-of-time rule switched off, and no step marked left out."""
    from mepd.discovery import network_expansion as ne

    def run(sub):   # by SMILES, not index: the async workers' finishing order (so the numbering) varies with load
        res = _run_unchecked(tmp_path / sub)
        smi = [s.smiles for s in res.species]
        steps = sorted((tuple(sorted((smi[st.a], smi[st.b]))), st.ts_energy, b is None, b or 0.0)
                       for st, b in zip(res.steps, res.kinetics.barriers_kcal))
        conc = dict(zip(smi, res.kinetics.final_concentration))   # equal to rounding: the solve's order differs too
        return res, (sorted((s.smiles, s.node.energy) for s in res.species), steps,
                     res.rounds[0]["kinetics"]["energies"]), conc

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    res, with_rule, conc_with = run("a")
    monkeypatch.setattr(ne, "OUT_OF_TIME", ())
    _, without, conc_without = run("b")
    assert res.rounds[0]["stopped"] == "converged" and res.steps
    assert not any("left_out_of_final_kinetics" in st.files for st in res.steps)
    assert with_rule == without
    assert conc_with == pytest.approx(conc_without, rel=1e-9, abs=1e-15)
