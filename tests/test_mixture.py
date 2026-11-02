import json

import numpy as np
import pytest

pytest.importorskip("rdkit")

from mepd.discovery import mixture  # noqa: E402
from mepd.structures import smiles_to_structure  # noqa: E402

BOHR = 1.8897259886


def _ordered_like(structure, pattern):
    """`structure`'s atoms reordered to the element sequence `pattern`."""
    pools = {}
    for i, a in enumerate(structure.symbols):
        pools.setdefault(str(a), []).append(i)
    order = [pools[a].pop(0) for a in pattern]
    xyz = np.asarray(structure.geometry)[order]
    return structure.model_copy(update={"symbols": [str(structure.symbols[i]) for i in order], "geometry": xyz})


def _join(parts):
    from qcdata import Structure

    return Structure(symbols=[str(a) for p in parts for a in p.symbols],
                     geometry=np.vstack([np.asarray(p.geometry) for p in parts]), charge=0, multiplicity=1)


def _seed(tmp_path, ts_water_offset_angstrom):
    """A finished explorer run: ethylene + butadiene + water -> cyclohexene + water, the water 8 A from the
    ring in the product and `ts_water_offset_angstrom` in the TS."""
    a = mixture.place([smiles_to_structure(s, 0, 1) for s in ("C=C", "C=CC=C", "O")])
    n = 16
    reactive = a.model_copy(update={"symbols": list(a.symbols)[:n], "geometry": np.asarray(a.geometry)[:n]})
    ring = _ordered_like(smiles_to_structure("C1=CCCCC1", 0, 1), [str(x) for x in reactive.symbols])
    ring = ring.model_copy(update={"geometry": np.asarray(ring.geometry) - np.asarray(ring.geometry).mean(0)})
    water = a.model_copy(update={"symbols": list(a.symbols)[n:], "geometry": np.asarray(a.geometry)[n:]})

    def near(offset):
        w = np.asarray(water.geometry)
        return _join([ring, water.model_copy(update={
            "geometry": w - w[0] + np.asarray(ring.geometry)[0] + np.array([offset, 0, 0]) * BOHR})])

    b, ts = near(8.0), near(ts_water_offset_angstrom)
    d = tmp_path / "seedA"
    (d / "run" / "species").mkdir(parents=True)
    (d / "run" / "ts").mkdir()
    (d / "run" / "species" / "s0.xyz").write_text(a.to_xyz())
    (d / "run" / "species" / "s1.xyz").write_text(b.to_xyz())
    (d / "run" / "ts" / "t.xyz").write_text(ts.to_xyz())
    (d / "run" / "summary.json").write_text(json.dumps({
        "structure": "seed.xyz",
        "species": [{"file": "run/species/s0.xyz"}, {"file": "run/species/s1.xyz"}],
        "steps": [{"a": 0, "b": 1, "label": "fast_0_1", "files": {"ts": "run/ts/t.xyz", "verified": True}},
                  {"a": 0, "b": 1, "label": "unverified", "files": {"ts": "run/ts/t.xyz", "verified": False}}]}))
    return d


def _parts(structure, ts):
    from mepd.discovery.free_energy import molecules

    xyz = np.asarray(structure.geometry) / BOHR
    return [(g, 0, 1) for g in molecules([str(x) for x in structure.symbols], xyz)]


def test_compositions_count_copies_of_one_molecule():
    assert mixture.compositions(2, 2) == [(0,), (1,), (0, 0), (0, 1), (1, 1)]
    assert len(mixture.compositions(2, 3)) == 9


def test_place_keeps_molecules_apart():
    from mepd.discovery.free_energy import molecules

    s = mixture.place([smiles_to_structure(x, 0, 1) for x in ("C1=CC=CC1", "C1=CC=CC1", "C=CC=C")])
    assert len(molecules([str(x) for x in s.symbols], np.asarray(s.geometry) / BOHR)) == 3


def test_a_distant_molecule_is_a_spectator_and_leaves_the_step(tmp_path):
    mols, steps = mixture.read_seed(_seed(tmp_path, 8.0), _parts)
    assert len(steps) == 1                          # the unverified step is not used
    st = steps[0]
    water = mixture.molecule_key(smiles_to_structure("O", 0, 1))
    assert water not in st.reactants + st.products and st.shuttles == []
    assert len(st.reactants) == 2 and len(st.products) == 1
    assert len(st.ts.symbols) == 16                 # the TS without the water


def test_a_molecule_in_contact_in_the_ts_stays_on_both_sides(tmp_path):
    _, steps = mixture.read_seed(_seed(tmp_path, 1.2), _parts)
    water = mixture.molecule_key(smiles_to_structure("O", 0, 1))
    assert steps[0].shuttles == [water] and len(steps[0].ts.symbols) == 19


def _merged(steps, mols, chiral_ts):
    return mixture.merge(mols, steps, temperature=443.0,
                         energies=lambda ss: np.array([-0.01 * len(s.symbols) - 0.05 * (len(s.symbols) == 16) for s in ss]),
                         corrections=lambda ss, flags: [0.0] * len(ss),
                         chiral=lambda s: chiral_ts and len(s.symbols) == 16,
                         mirror_key=mixture.molecule_key)


def test_a_ts_found_twice_counts_once_and_a_chiral_ts_twice(tmp_path):
    mols, steps = mixture.read_seed(_seed(tmp_path, 8.0), _parts)
    assert len(_merged(steps + steps, mols, False).steps) == 1
    backwards = [mixture.Found(f.products, f.reactants, f.ts, f.source) for f in steps]
    assert len(_merged(steps + backwards, mols, False).steps) == 1      # found once each way: one TS
    assert len(_merged(steps, mols, True).steps) == 2       # achiral ends: its mirror twin is a second channel


def test_merged_kinetics_run_from_the_feed(tmp_path):
    mols, steps = mixture.read_seed(_seed(tmp_path, 8.0), _parts)
    m = _merged(steps, mols, False)
    k = {mixture.molecule_key(smiles_to_structure(s, 0, 1)): 1.0 for s in ("C=C", "C=CC=C")}
    net, res = mixture.kinetics(m, k, 443.0, 1.0)
    ring = m.names.index(mixture.molecule_key(smiles_to_structure("C1=CCCCC1", 0, 1)))
    assert res.c[-1, ring] > 0.5                    # TS at the products' level: fast and downhill


# Formic acid dimer -> CO2 + H2 + formic acid (g-xTB, a 2-generation-0 seed): the second molecule relays
# protons through partial O...H bonds of 1.34 and 1.45 A, so it is part of the TS, not a spectator.
_DIMER = """ O          1.602360982336        0.915327304129       -0.267549701261
 C          1.878339057080       -0.311918179099        0.092976242589
 O          1.049158062541       -1.165998450029        0.343093396689
 H          0.616541854598        1.067337801563       -0.312951185432
 H          2.964589111330       -0.492861510908        0.146934012320
 O         -1.602345218086       -0.915309606274        0.267598181034
 C         -1.878347995189        0.311913769230       -0.092971108561
 O         -1.049175874111        1.165980682440       -0.343185307111
 H         -0.616521611559       -1.067323281996        0.312928183706
 H         -2.964598368940        0.492851470943       -0.146872713970
"""
_RELAY_TS = """ O          1.465996457332        0.789566930516       -0.109715400355
 C          1.737696603014       -0.455758832031       -0.187534843549
 O          0.921946455521       -1.387103662184       -0.146489166189
 H          0.457769902595        1.053922939026        0.000147406185
 H          2.810317612680       -0.682177563632       -0.301389256324
 O         -2.997442828582        0.445677363344        0.308509041273
 C         -1.824041713500        0.563252317992        0.200567211342
 O         -0.945592378332        1.412079779802        0.152902256482
 H         -0.372886278879       -1.064805545257       -0.008814901602
 H         -1.253763831850       -0.674653727577        0.091817652736
"""
_CO2_H2_ACID = """ O         -0.306853203006        0.805939504203        1.915369017099
 C         -0.862383045020        0.725516405232        0.708261827802
 O         -0.373995087373        0.140759382392       -0.222307452300
 H          0.537496916590        0.323559198596        1.912776104464
 H         -1.822479362142        1.265231206017        0.684498387425
 O          2.080554328184       -0.748234774688        1.330127805169
 C          2.240883583375       -1.187893364407        0.272337565244
 O          2.413762822704       -1.629894932812       -0.770816847020
 H         -1.725920431521        0.116830314470       -2.630309772502
 H         -2.181066521793        0.188187060996       -3.199936635380
"""


def test_a_proton_relay_partner_is_not_a_spectator(tmp_path):
    d = tmp_path / "dimer"
    (d / "run").mkdir(parents=True)
    for name, xyz in (("a", _DIMER), ("b", _CO2_H2_ACID), ("ts", _RELAY_TS)):
        (d / "run" / f"{name}.xyz").write_text("10\n\n" + xyz)
    (d / "run" / "summary.json").write_text(json.dumps({
        "species": [{"file": "run/a.xyz"}, {"file": "run/b.xyz"}],
        "steps": [{"a": 0, "b": 1, "label": "relay", "files": {"ts": "run/ts.xyz", "verified": True}}]}))
    _, steps = mixture.read_seed(d, _parts)
    acid = mixture.molecule_key(smiles_to_structure("OC=O", 0, 1))
    assert steps[0].reactants == [acid] and steps[0].shuttles == [acid] and len(steps[0].ts.symbols) == 10
