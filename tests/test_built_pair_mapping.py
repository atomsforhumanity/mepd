"""Atom mapping with a built endpoint (--pair-from start/end): every
candidate's partner is built and minimized, and the pairs are compared, not
the given product (mepd.built_pair_mapping). Also: SLAPMapper's single
suggestion pairs equivalent atoms correctly (acetone -> its enol is one H
moving, not two)."""

from __future__ import annotations

import pytest

pytest.importorskip("slapmapper")
pytest.importorskip("rdkit")


def _embedded(smiles: str, seed: int):
    from qcconst.constants import ANGSTROM_TO_BOHR
    from qcdata import Structure
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return Structure(symbols=[a.GetSymbol() for a in m.GetAtoms()],
                     geometry=m.GetConformer().GetPositions() * ANGSTROM_TO_BOHR, charge=0, multiplicity=1)


def _acetone_and_enol():
    # Same element order (C C C O H...): the enol's numbering moves two H
    # (one between the carbons, one onto O) where one suffices.
    return _embedded("CC(C)=O", 1), _embedded("C=C(C)O", 7)


def test_slapmapper_suggestion_moves_one_h_not_two():
    from mepd.atom_mapping import _bond_changes_under, check_atom_mapping

    start, end = _acetone_and_enol()
    n = len(start.symbols)
    assert _bond_changes_under(start, end, {i: i for i in range(n)}) == 4
    m = check_atom_mapping(start, end)
    assert not m.is_identity and _bond_changes_under(start, end, m.mapping) == 2


def _no_optimizer(monkeypatch):
    """Minimization returns the embedded guess."""
    import mepd.built_pair_mapping as bp

    monkeypatch.setattr(bp, "_minimize", lambda guesses, run_inputs: list(guesses))


def _rank_by_bond_changes(monkeypatch, seen: list):
    """The path comparison stubbed out: fewest bond changes from the source
    wins (records the compared picks)."""
    import mepd.atom_mapping_selection as sel

    def rank(picks, fallback, start_structure, run_inputs, use_path=True):
        seen.extend(picks)
        score = {c.label: sel._bond_changes(start_structure, c.end_structure) for c in picks}
        best = min(picks, key=lambda c: score[c.label])
        return 0.0, best, "stub", score, "bond changes", []

    monkeypatch.setattr(sel, "_rank_picks", rank)


def test_one_build_per_mechanism_no_symmetric_relabelings(monkeypatch):
    """Built from acetone: one product per mechanism, not one per methyl H.
    The candidates' relabelings of symmetric atoms are not built, and none
    are added; the pair comes back in the start's order with one H moved."""
    from mepd.atom_mapping import check_atom_mapping, expand_mapping_fully
    from mepd.atom_mapping_selection import _bond_changes
    from mepd.built_pair_mapping import build_and_rank
    from mepd.inputs import RunInputs

    _no_optimizer(monkeypatch)
    seen: list = []
    _rank_by_bond_changes(monkeypatch, seen)
    start, end = _acetone_and_enol()
    m = check_atom_mapping(start, end)
    relabelings = expand_mapping_fully(m, start, end, max_variants=50)
    assert len(relabelings) > 1
    out = build_and_rank(start, end, relabelings, "start", RunInputs(engine_name="gxtb"), echo=lambda s: None)
    assert out is not None and out.start is start
    assert list(out.end.symbols) == list(start.symbols) and _bond_changes(out.start, out.end) == 2
    # the given numbering (2 H moving) and SLAPMapper's one-H transfer: two mechanisms, one build each
    assert sorted(_bond_changes(start, c.end_structure) for c in seen) == [2, 4]


def test_failed_build_falls_back_to_an_equivalent_site_and_says_so(monkeypatch):
    """When the as-given site's build fails, the same bonds at an equivalent
    site (another methyl H) are built instead, still one per mechanism, and
    the log says the fallback happened."""
    import mepd.built_pair_mapping as bp
    from mepd.atom_mapping import check_atom_mapping, realign_end_to_start
    from mepd.atom_mapping_selection import _bond_changes
    from mepd.inputs import RunInputs

    start, end = _acetone_and_enol()
    m = check_atom_mapping(start, end)
    given = bp._edges(realign_end_to_start(m, end))
    monkeypatch.setattr(bp, "_minimize", lambda guesses, ri: [None if bp._edges(g) == given else g for g in guesses])
    _rank_by_bond_changes(monkeypatch, [])
    lines: list = []
    out = bp.build_and_rank(start, end, [m], "start", RunInputs(engine_name="gxtb"), echo=lines.append)
    assert out is not None and " sym" in out.label and _bond_changes(out.start, out.end) == 2
    assert bp._edges(out.end) != given
    assert any("SYMMETRY-SWAP FALLBACK" in ln and "equivalent site instead" in ln for ln in lines)
    assert any("via the symmetry-swap fallback" in ln for ln in lines)


def test_from_the_product_the_pair_comes_back_in_start_order(monkeypatch):
    from mepd.atom_mapping import check_atom_mapping
    from mepd.atom_mapping_selection import _bond_changes
    from mepd.built_pair_mapping import _edges, build_and_rank
    from mepd.inputs import RunInputs

    _no_optimizer(monkeypatch)
    _rank_by_bond_changes(monkeypatch, [])
    start, end = _acetone_and_enol()
    out = build_and_rank(start, end, [check_atom_mapping(start, end)], "end", RunInputs(engine_name="gxtb"),
                         echo=lambda s: None)
    assert out is not None
    assert _bond_changes(out.start, out.end) == 2
    assert _edges(out.start) == _edges(start)     # the built reactant has the reactant's bonds, in its order


def test_partners_whose_bonds_change_are_dropped_with_a_warning(monkeypatch):
    """A built partner that minimizes back to something else is dropped; if
    every fewest-change one is lost, the log says so."""
    import mepd.built_pair_mapping as bp
    from mepd.atom_mapping import check_atom_mapping
    from mepd.inputs import RunInputs

    start, end = _acetone_and_enol()

    def minimize(guesses, run_inputs):     # every one-H product fails to minimize
        return [None if len(bp._edges(start) ^ bp._edges(g)) == 2 else g for g in guesses]

    monkeypatch.setattr(bp, "_minimize", minimize)
    _rank_by_bond_changes(monkeypatch, [])
    lines: list = []
    out = bp.build_and_rank(start, end, [check_atom_mapping(start, end)], "start", RunInputs(engine_name="gxtb"),
                            echo=lines.append)
    assert out is not None and out.n_kept < out.n_bond_sets
    assert any("fewest bond changes (2)" in ln for ln in lines)


@pytest.mark.parametrize("src", ["start", "end"])
def test_channels_builds_one_pair_per_conformer_and_mapping(monkeypatch, src):
    """channels --pair-from with --atom-mapping: every conformer builds its
    partner once per mechanism of the candidate mappings; the pairs come back
    in the start's atom order."""
    import mepd.built_pair_mapping as bp
    from mepd.atom_mapping_selection import _bond_changes
    from mepd.cli_channels import _built_pairs_by_mechanism
    from mepd.inputs import RunInputs
    from mepd.nodes.node import StructureNode

    _no_optimizer(monkeypatch)
    _rank_by_bond_changes(monkeypatch, [])
    monkeypatch.setattr(bp, "_path_score", lambda a, b, ri: (float(_bond_changes(a, b)), None, None))
    start, end = _acetone_and_enol()
    conformers = [StructureNode(structure=start if src == "start" else end)] * 2
    rows, n_maps = _built_pairs_by_mechanism(conformers, src, StructureNode(structure=start),
                                             StructureNode(structure=end), RunInputs(engine_name="gxtb"))
    assert n_maps >= 1
    per_conformer = {k: sorted(_bond_changes(r.start, r.end) for kk, r in rows if kk == k) for k in (0, 1)}
    assert per_conformer[0] == per_conformer[1] and per_conformer[0][0] == 2    # the one-H transfer
    assert len(per_conformer[0]) == len({r.key for k, r in rows if k == 0})    # one per mechanism
    assert all(list(r.start.symbols) == list(start.symbols) for _, r in rows)


def test_built_pairs_are_ranked_together_above_one_floor():
    """Past --atom-mapping-candidates, every built pair from both sides is
    ranked by its xtb peak (absolute: one shared floor), not by the barrier
    above its own start, and only the lowest are searched."""
    from mepd.built_pair_mapping import MechanismPair
    from mepd.cli_channels import _choose_built_pairs

    def pair(peak, floor):
        return MechanismPair(key="break C1-H1 | form H1-O1", score=peak, start=None, end=None, n_variants=1,
                             floor=floor)

    rows = [("start", 0, pair(-100.0, -130.0)),    # 30 above its own start, 30 above the floor
            ("start", 1, pair(-105.0, -110.0)),    # 5 above its own (strained) start, 25 above the floor
            ("end", 0, pair(-120.0, -125.0))]      # 10 above the floor
    ranked, chosen, lines = _choose_built_pairs(rows, 2)
    assert [(s, k) for s, k, _ in ranked] == [("end", 0), ("start", 1), ("start", 0)] and chosen == [0, 1]
    assert lines[0].split()[:3] == ["built_0", "searched", "10.0"] and lines[2].split()[1] == "skipped"
    # a pick further down the list is searched in place of the lowest
    _, chosen, lines = _choose_built_pairs(rows, 2, picks={"built_2"})
    assert chosen == [2] and lines[2].split()[1] == "searched" and lines[0].split()[1] == "skipped"


def test_built_pair_cap_can_apply_per_mechanism():
    """--built-cap-per-mechanism: the lowest --atom-mapping-candidates of
    each mechanism are searched, not the lowest of all together."""
    from mepd.built_pair_mapping import MechanismPair
    from mepd.cli_channels import _choose_built_pairs

    def pair(key, peak):
        return MechanismPair(key=key, score=peak, start=None, end=None, n_variants=1, floor=-130.0)

    a, b = "break C1-H1 | form H1-O1", "break C1-C2 | form C2-O1"
    rows = [("start", k, pair(a, -120.0 + k)) for k in range(3)] + [("end", k, pair(b, -100.0 + k)) for k in range(2)]
    _, chosen, _ = _choose_built_pairs(rows, 2)
    assert chosen == [0, 1]                       # both of mechanism a
    _, chosen, lines = _choose_built_pairs(rows, 2, per_mechanism=True)
    assert chosen == [0, 1, 3, 4] and lines[2].split()[1] == "skipped"


def test_built_pair_list_round_trips(tmp_path):
    """The ranked list a run saves is what --built-pairs reads back: same
    order, structures, scores, and which entries were searched."""
    from mepd.built_pair_mapping import MechanismPair
    from mepd.cli_channels import BUILT_PAIRS_JSON, _load_built_pairs, _save_built_pairs
    import json

    start, end = _acetone_and_enol()
    ranked = [("start", 0, MechanismPair(key="k1", score=-100.0, start=start, end=end, n_variants=1, floor=-110.0,
                                         path=[start, end], path_energies=[-110.0, -101.0])),
              ("end", 3, MechanismPair(key="k2", score=-95.0, start=start, end=end, n_variants=1, floor=-120.0))]
    _save_built_pairs(tmp_path, ranked, {"built_0"}, "snap-gi-xtb")
    entries = json.loads((tmp_path / BUILT_PAIRS_JSON).read_text())["entries"]
    assert [e["barrier_kcal"] for e in entries] == [20.0, 25.0]
    back, searched = _load_built_pairs(tmp_path, 0, 1)
    assert searched == {"built_0"}
    assert [(s, k, r.key, r.score) for s, k, r in back] == [(s, k, r.key, r.score) for s, k, r in ranked]
    assert abs(back[1][2].end.geometry - end.geometry).max() < 1e-5
    assert [e["has_path"] for e in entries] == [True, False]
    path = json.loads((tmp_path / "paths" / "built_0.json").read_text())
    assert len(path["frames"]) == 2 and path["energies_kcal"] == [-110.0, -101.0]


@pytest.mark.parametrize("metric", ["endpoint-rmsd", "geodesic-distance", "snap-gi-xtb"])
def test_built_pairs_are_scored_by_the_atom_mapping_metric(monkeypatch, metric):
    """channels' built pairs follow --atom-mapping-metric; snap-gi-xtb (the
    default) is the xtb peak, falling back to the GI length without xtb."""
    import mepd.atom_mapping_selection as sel
    from mepd.built_pair_mapping import _path_score
    from mepd.inputs import RunInputs

    monkeypatch.setattr(sel, "_xtb_engine", lambda: None)
    start, end = _acetone_and_enol()
    ri = RunInputs(gi_inputs={"nimages": 6})
    ri.atom_mapping_inputs.metric = metric
    got, floor, path = _path_score(start, end, ri)
    assert floor is None and (path is None) == (metric != "snap-gi-xtb")   # only snap-gi-xtb interpolates here
    if metric == "endpoint-rmsd":
        assert got == pytest.approx(sel._aligned_rmsd(start, end))
    else:   # geodesic-distance, and snap-gi-xtb without xtb: the GI length
        cand = sel.MappingCandidate(label="pair", end_structure=end, atom_map=None)
        assert got == pytest.approx(float(sel._interpolate(cand, start, ri)[1].length))
