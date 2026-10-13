"""Network variants (mepd/network_variants.py): an edit of one species
carried through every reaction of a network by graph matching."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from mepd import network_variants as NV  # noqa: E402
from mepd.web import design as D  # noqa: E402


def _species(sid, smiles, charge=0, seed=3):
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        pass
    return NV.Species(sid, [a.GetSymbol() for a in m.GetAtoms()], m.GetConformer().GetPositions(), charge,
                      name=smiles)


def _reaction(rid, text, reactants, products, charge=0, ts=True):
    from mepd.reaction_smiles import reaction_pair

    pair = reaction_pair(text)
    sym = [a.GetSymbol() for a in pair.reactant.GetAtoms()]
    r = pair.reactant.GetConformer().GetPositions()
    p = pair.product.GetConformer().GetPositions()
    return NV.Reaction(rid, reactants, products, r=(sym, r), ts=(sym, (r + p) / 2) if ts else None, p=(sym, p),
                       charge=charge)


@pytest.fixture(scope="module")
def network():
    species = {s.id: s for s in [
        _species("EO", "C1CO1"), _species("CO2", "O=C=O"), _species("Br", "[Br-]", -1),
        _species("ALK", "[O-]CCBr", -1), _species("BRC", "O=C([O-])OCCBr", -1), _species("EC", "O=C1OCCO1")]}
    reactions = [
        _reaction("R0", "[CH2:1]1[CH2:2][O:3]1.[O:4]=[C:5]=[O:6]>>[CH2:1]1[CH2:2][O:3][C:5](=[O:6])[O:4]1",
                  ["EO", "CO2"], ["EC"]),
        _reaction("R1", "[Br-:7].[CH2:1]1[CH2:2][O:3]1>>[Br:7][CH2:1][CH2:2][O-:3]", ["Br", "EO"], ["ALK"], -1),
        _reaction("R2", "[Br:7][CH2:1][CH2:2][O-:3].[O:4]=[C:5]=[O:6]>>[Br:7][CH2:1][CH2:2][O:3][C:5](=[O:6])[O-:4]",
                  ["ALK", "CO2"], ["BRC"], -1),
        _reaction("R3", "[Br:7][CH2:1][CH2:2][O:3][C:5](=[O:6])[O-:4]>>[Br-:7].[CH2:1]1[CH2:2][O:3][C:5](=[O:6])[O:4]1",
                  ["BRC"], ["Br", "EC"], -1),
    ]
    return species, reactions


def _tagged(species: NV.Species):
    """The species as Design loads it from the graph: a molblock whose atoms
    carry their index + 1 as atom-map numbers."""
    xyz = f"{len(species.symbols)}\n\n" + "\n".join(
        f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(species.symbols, species.coords))
    mol, _ = D.mol_from_xyz(xyz, species.charge)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return Chem.MolToMolBlock(mol)


def _edit(species: NV.Species, *ops):
    parent = _tagged(species)
    mb = parent
    for op in ops:
        mb = D.edit(mb, op)["molblock"]
    return NV.derive_edit(D.read(parent), D.read(mb))


def _hs(species, on):
    return [i for i, s in enumerate(species.symbols) if s == "H" and species.adj[i, on]]


def _carbons(species):
    return [i for i, s in enumerate(species.symbols) if s == "C"]


# ------------------------------------------------------------------ derive_edit

def test_methyl_for_hydrogen_keeps_the_hydrogens_index(network):
    eo = network[0]["EO"]
    c = _carbons(eo)[0]
    h = _hs(eo, c)[0]
    e = _edit(eo, {"op": "group", "atom": h, "group": "methyl"})
    assert e.replaced and list(e.replaced) == [h]
    assert e.new_symbols.count("C") == 1 and e.new_symbols.count("H") == 3
    assert not e.removed and not e.swaps
    assert e.charge_delta == 0
    assert e.describe() == [f"H{h + 1} on C{c + 1} -> CH3"]


def test_element_swap_with_hydrogen_added(network):
    eo = network[0]["EO"]
    o = eo.symbols.index("O")
    e = _edit(eo, {"op": "element", "atom": o, "element": "N"})
    assert e.swaps == {o: ("N", 0)}
    assert e.new_symbols == ["H"] and not e.replaced


def test_element_swap_without_hydrogen_change(network):
    eo = network[0]["EO"]
    o = eo.symbols.index("O")
    e = _edit(eo, {"op": "element", "atom": o, "element": "S"})
    assert e.swaps == {o: ("S", 0)} and not e.new_symbols and not e.removed


def test_hydrogens_redone_by_design_are_not_an_edit(network):
    """Design re-adds every hydrogen of an atom it touches (new, untagged
    atoms in the same places): that is no change."""
    eo = network[0]["EO"]
    c = _carbons(eo)[0]
    h = _hs(eo, c)[0]
    e = _edit(eo, {"op": "group", "atom": h, "group": "methyl"}, {"op": "hydrogens"})
    assert list(e.replaced) == [h] and not e.removed


def test_refusals(network):
    eo = network[0]["EO"]
    c0, c1 = _carbons(eo)
    o = eo.symbols.index("O")
    with pytest.raises(NV.VariantError, match="deleted"):
        _edit(eo, {"op": "delete", "atom": o})
    with pytest.raises(NV.VariantError, match="bond"):
        _edit(eo, {"op": "bond", "a": c0, "b": o, "order": 0})
    with pytest.raises(NV.VariantError, match="same molecule"):
        _edit(eo, {"op": "hydrogens"})


# ------------------------------------------------------------------ propagation

def test_methyl_epoxide_through_the_network(network):
    species, reactions = network
    eo = species["EO"]
    c = _carbons(eo)[0]
    e = _edit(eo, {"op": "group", "atom": _hs(eo, c)[0], "group": "methyl"})
    plan = NV.propagate(species, reactions, "EO", e)
    assert not plan.problems, plan.problems
    by = {}
    for vr in plan.reactions:
        by.setdefault(vr.parent, []).append(vr)
    # Each step splits by which carbon carries the methyl (the attacked one
    # or not); mirror images are one reaction.
    assert {k: len(v) for k, v in by.items()} == {"R0": 2, "R1": 2, "R2": 2, "R3": 2}
    assert all(vr.status == "ok" for vr in plan.reactions)
    parents = sorted(v.parent for v in plan.species.values())
    # Propylene oxide, two bromoalkoxides, two bromocarbonates, ONE propylene carbonate.
    assert parents == ["ALK", "ALK", "BRC", "BRC", "EC", "EO"]
    assert plan.replaced == ["ALK", "BRC", "EC", "EO"]
    assert plan.shared == []
    # Every variant reaction's frames: same atoms, the edit in all three.
    for vr in plan.reactions:
        assert len(vr.symbols) == len(vr.r) == len(vr.p) == len(vr.ts)
        assert vr.symbols.count("C") == {"R0": 4, "R1": 3, "R2": 4, "R3": 4}[vr.parent]
        assert vr.clash > 0.9
    # Both bromoalkoxides go on to the same propylene carbonate.
    pc = [k for k, v in plan.species.items() if v.parent == "EC"]
    assert len(pc) == 1
    assert sum(pc[0] in vr.products for vr in plan.reactions if vr.parent == "R3") == 2


def test_variant_species_are_real_molecules(network):
    species, reactions = network
    eo = species["EO"]
    c = _carbons(eo)[0]
    e = _edit(eo, {"op": "group", "atom": _hs(eo, c)[0], "group": "methyl"})
    plan = NV.propagate(species, reactions, "EO", e)
    from rdkit.Chem import rdDetermineBonds

    seen = set()
    for v in plan.species.values():
        xyz = f"{len(v.symbols)}\n\n" + "\n".join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip(v.symbols, v.coords))
        m = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineConnectivity(m)
        seen.add(len(Chem.GetMolFrags(m)))
    assert seen == {1}


def test_tetramethyl_gives_one_variant_per_step(network):
    species, reactions = network
    eo = species["EO"]
    ops = []
    mb = _tagged(eo)
    for h in [i for i, s in enumerate(eo.symbols) if s == "H"]:
        # Design indices shift as hydrogens go: always edit the first tagged H.
        mol = D.read(mb)
        idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H" and a.GetAtomMapNum() == h + 1)
        mb = D.edit(mb, {"op": "group", "atom": idx, "group": "methyl"})["molblock"]
    del ops
    e = NV.derive_edit(D.read(_tagged(eo)), D.read(mb))
    assert len(e.replaced) == 4
    plan = NV.propagate(species, reactions, "EO", e)
    counts = {}
    for vr in plan.reactions:
        counts[vr.parent] = counts.get(vr.parent, 0) + 1
    assert counts == {"R0": 1, "R1": 1, "R2": 1, "R3": 1}


def test_cis_dimethyl_never_becomes_trans(network):
    species, reactions = network
    eo = species["EO"]
    c0, c1 = _carbons(eo)
    pos = eo.coords
    o = eo.symbols.index("O")
    # One hydrogen on each carbon, on the same face of the ring.
    normal = np.cross(pos[c1] - pos[c0], pos[o] - pos[c0])
    h0 = max(_hs(eo, c0), key=lambda h: float(np.dot(pos[h] - pos[c0], normal)))
    h1 = max(_hs(eo, c1), key=lambda h: float(np.dot(pos[h] - pos[c1], normal)))
    mb = _tagged(eo)
    for h in (h0, h1):
        mol = D.read(mb)
        idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == h + 1)
        mb = D.edit(mb, {"op": "group", "atom": idx, "group": "methyl"})["molblock"]
    e = NV.derive_edit(D.read(_tagged(eo)), D.read(mb))
    plan = NV.propagate(species, reactions, "EO", e)
    base = plan.species["v:0"]
    for vr in plan.reactions:
        if "v:0" not in vr.reactants and "v:0" not in vr.products:
            continue
        # The epoxide fragment in each built frame has the cis configuration.
        side = vr.r if "v:0" in vr.reactants else vr.p
        adj = NV.adjacency(vr.symbols, side)
        frag = next(f for f in NV.components(adj) if len(f) == len(base.symbols)
                    and sorted(vr.symbols[i] for i in f) == sorted(base.symbols))
        k = NV.molecule_key([vr.symbols[i] for i in frag], side[frag], adj[np.ix_(frag, frag)], base.charge)
        assert k == base.key


def test_transferred_hydrogen_blocks_the_step():
    species = {s.id: s for s in [_species("ALD", "CC=O"), _species("ENOL", "C=CO")]}
    rx = _reaction("T", "[CH3:1][CH:2]=[O:3]>>[CH2:1]=[CH:2][OH:3]", ["ALD"], ["ENOL"])
    ald = species["ALD"]
    c_methyl = next(i for i in _carbons(ald) if len(_hs(ald, i)) == 3)
    e = _edit(ald, {"op": "group", "atom": _hs(ald, c_methyl)[0], "group": "fluoro"})
    plan = NV.propagate(species, [rx], "ALD", e)
    statuses = sorted(vr.status for vr in plan.reactions)
    assert "blocked" in statuses and "ok" in statuses


def test_untouched_reactions_are_shared(network):
    species, reactions = network
    extra = {**species, "W": _species("W", "O"), "W2": _species("W2", "O.O", seed=5)}
    # A reaction with no epoxide in it (water dimer as a stand-in) stays shared.
    co2 = species["CO2"]
    eo = species["EO"]
    e = _edit(eo, {"op": "group", "atom": _hs(eo, _carbons(eo)[0])[0], "group": "methyl"})
    plan = NV.propagate(extra, reactions + [NV.Reaction("RW", ["CO2"], ["CO2"], r=(co2.symbols, co2.coords),
                                                       p=(co2.symbols, co2.coords))], "EO", e)
    assert plan.shared == ["RW"]


def test_edit_round_trips_through_json(network):
    eo = network[0]["EO"]
    e = _edit(eo, {"op": "group", "atom": _hs(eo, _carbons(eo)[0])[0], "group": "trifluoromethyl"})
    import json

    back = NV.Edit.from_json(json.loads(json.dumps(e.to_json())))
    assert back.replaced == e.replaced and back.new_symbols == e.new_symbols
    assert np.allclose(back.new_coords, e.new_coords) and (back.adj == e.adj).all()


def test_heavy_atom_swap_through_the_network(network):
    """O -> S at the atom that reacts: thiirane opens to a thiolate, and on."""
    species, reactions = network
    eo = species["EO"]
    e = _edit(eo, {"op": "element", "atom": eo.symbols.index("O"), "element": "S"})
    plan = NV.propagate(species, reactions, "EO", e)
    assert not plan.problems, plan.problems
    by = {}
    for vr in plan.reactions:
        by.setdefault(vr.parent, []).append(vr)
    # Symmetric thiirane: one variant per step on the way in. The ring
    # closure also runs back: bromide opens the oxathiolanone at C-O as
    # well, and that thiocarboxylate closes through S or through O.
    assert {k: len(v) for k, v in by.items() if k != "R3"} == {"R0": 1, "R1": 1, "R2": 1}
    smiles = {v.smiles for v in plan.species.values()}
    assert {"C1CS1", "[S-]CCBr", "O=C1OCCS1", "S=C1OCCO1"} <= smiles
    alk = [v for v in plan.species.values() if v.parent == "ALK"]
    assert len(alk) == 1 and alk[0].charge == -1
    # Nothing that would hand the edit to CO2 (and leave the epoxide or the alkoxide as they were).
    assert not any(v.parent == "CO2" for v in plan.species.values())
    for vr in plan.reactions:
        assert vr.symbols.count("S") == 1
        assert not set(vr.reactants + vr.products) & set(plan.replaced)


def test_hydrogen_turned_into_fluorine_with_the_element_tool(network):
    eo = network[0]["EO"]
    h = _hs(eo, _carbons(eo)[0])[0]
    e = _edit(eo, {"op": "element", "atom": h, "element": "F"})
    assert list(e.replaced) == [h] and e.new_symbols == ["F"]
    plan = NV.propagate(*network, "EO", e)
    assert all(vr.symbols.count("F") == 1 for vr in plan.reactions)


def test_group_on_a_spectator_goes_nearest_the_reaction():
    # NMe4+ only watches the ring opening: of its twelve equivalent H, the OH
    # takes one on the methyl nearest the reacting atoms (where it can act).
    species = {s.id: s for s in [_species("EO", "C1CO1"), _species("Br", "[Br-]", -1),
                                 _species("N", "C[N+](C)(C)C", 1), _species("ALK", "[O-]CCBr", -1)]}
    rx = _reaction("R1", "[Br-:7].[CH3:8][N+:9]([CH3:10])([CH3:11])[CH3:12].[CH2:1]1[CH2:2][O:3]1>>"
                   "[Br:7][CH2:1][CH2:2][O-:3].[CH3:8][N+:9]([CH3:10])([CH3:11])[CH3:12]",
                   ["Br", "N", "EO"], ["ALK", "N"])
    n = species["N"]
    e = _edit(n, {"op": "group", "atom": _hs(n, _carbons(n)[0])[0], "group": "hydroxyl"})
    plan = NV.propagate(species, [rx], "N", e)
    (vr,) = plan.reactions
    sym, ts = rx.ts
    center = [i for i in range(len(sym)) if sym[i] in ("Br", "O") or (sym[i] == "C" and any(
        sym[j] == "O" and np.linalg.norm(rx.r[1][i] - rx.r[1][j]) < 1.6 for j in range(len(sym))))]
    methyls = [i for i in range(len(sym)) if sym[i] == "C" and i not in center]
    dist = {i: min(np.linalg.norm(ts[i] - ts[j]) for j in center) for i in methyls}
    # The OH oxygen takes the replaced hydrogen's index.
    (o_new,) = [i for i in range(len(sym)) if vr.symbols[i] == "O" and sym[i] == "H"]
    anchor = min(methyls, key=lambda i: np.linalg.norm(vr.ts[i] - vr.ts[o_new]))
    assert dist[anchor] == min(dist.values())


def test_halide_swap_on_a_bare_ion(network):
    # Br- loads into Design with its charge (it used to come in as a neutral
    # Br atom, and the swap was refused as an unpaired electron).
    br = network[0]["Br"]
    e = _edit(br, {"op": "element", "atom": 0, "element": "I"}, {"op": "charge", "atom": 0, "delta": -1})
    assert e.swaps == {0: ("I", 0)} and e.charge_delta == 0
    plan = NV.propagate(network[0], network[1], "Br", e)
    assert {v.smiles for v in plan.species.values()} >= {"[I-]", "[O-]CCI", "O=C([O-])OCCI"}
