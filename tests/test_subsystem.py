"""The explorer's subsystem cut (network_expansion.subsystem): a reaction between two whole states -- a box
of molecules from the reactor, a seed of several molecules -- runs its path search, TS, IRC and their
Hessians on the molecules whose bonds change (and those that relay an atom), never on the whole box; the
other molecules (spectators) stay where they are and count apart."""
from pathlib import Path

import numpy as np
import pytest
from qcdata import Structure

from mepd.discovery.network_expansion import subsystem
from mepd.nodes.node import StructureNode

A2B = 1.8897259886

# formic acid (Z), Angstrom: O0 C1 O2(=) H3(on O0) H4(on C1)
HCOOH = [("O", 0.74, -0.62, -0.08), ("C", -0.41, 0.06, -0.16), ("O", -0.56, 1.17, 0.27), ("H", 1.40, -0.07, 0.37),
         ("H", -1.17, -0.55, -0.67)]
# ... -> CO + H2O: O0 H3 H4 the water, C1 O2 the CO
CO_H2O = [("O", 3.0, 0, 0), ("C", -0.41, 0.06, -0.16), ("O", -0.56, 1.17, 0.27), ("H", 3.96, 0, 0), ("H", 2.76, 0.93, 0)]
WATER = [("O", 6.0, 0, 0), ("H", 6.96, 0, 0), ("H", 5.76, 0.93, 0)]     # O5 H6 H7
N2 = [("N", -6.0, 0, 0), ("N", -7.1, 0, 0)]


def _node(*atoms, energy=None):
    atoms = [a for group in atoms for a in group]
    n = StructureNode(structure=Structure(symbols=[a[0] for a in atoms],
                                          geometry=np.array([a[1:] for a in atoms], dtype=float) * A2B))
    n._cached_energy = energy
    return n


class _Engine:
    """-1 Hartree per atom: env is then minus the spectators' atom count."""
    def compute_energies(self, nodes):
        for n in nodes:
            n._cached_energy = -1.0 * len(n.symbols)
        return np.array([n._cached_energy for n in nodes])


def test_a_spectator_is_left_out_and_put_back_where_it_was():
    start, end = _node(HCOOH, WATER, energy=-20.0), _node(CO_H2O, WATER)
    cut = subsystem(start, end, _Engine())
    assert cut.atoms == [0, 1, 2, 3, 4] and (cut.charge, cut.multiplicity) == (0, 1)
    assert cut.env == pytest.approx(-20.0 + 5.0)          # E(whole start) - E(its subsystem)
    sub = cut.sub(end)
    assert list(sub.symbols) == ["O", "C", "O", "H", "H"]
    sub._cached_energy = -4.0
    back = cut.whole(sub)
    assert len(back.symbols) == 8 and back.energy == pytest.approx(-4.0 - 15.0)
    assert np.allclose(back.coords[5:], start.coords[5:])                 # the spectator where it was
    # by its centre of mass alone: the subsystem's own shape and orientation, not fitted onto the start's
    x = back.coords[:5] - back.coords[:5].mean(axis=0)
    assert np.allclose(x, sub.coords - sub.coords.mean(axis=0))
    # with a frame (the structure a computation started from, in its coordinates): that frame's motion
    turned = cut.sub(start).coords @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]) + 3.0
    frame = StructureNode(structure=cut.sub(start).structure.model_copy(update={"geometry": turned}))
    assert np.allclose(cut.whole(frame, frame).coords, start.coords, atol=1e-8)


def test_a_molecule_that_relays_an_atom_is_in_the_subsystem():
    # formic acid hands H3 to the water next to it, the water hands H6 to O2: a proton relay; N2 looks on
    near_water = [("O", 2.2, 0.5, 0.2), ("H", 2.1, 1.3, 0.6), ("H", 2.9, 0.6, -0.4)]
    end_fa = [HCOOH[0], HCOOH[1], HCOOH[2], ("H", 2.0, -0.2, 0.5), HCOOH[4]]
    end_w = [near_water[0], ("H", -0.3, 2.05, 0.4), near_water[2]]
    start, end = _node(HCOOH, near_water, N2, energy=-30.0), _node(end_fa, end_w, N2)
    cut = subsystem(start, end, _Engine())
    assert cut.atoms == list(range(8))                    # the relaying water too; N2 left out
    assert cut.env == pytest.approx(-30.0 + 8.0)


def test_when_every_molecule_reacts_there_is_no_cut():
    eng = _Engine()
    assert subsystem(_node(HCOOH, energy=-5.0), _node(CO_H2O), eng) is None          # one molecule
    near_water = [("O", 2.2, 0.5, 0.2), ("H", 2.1, 1.3, 0.6), ("H", 2.9, 0.6, -0.4)]
    end_fa = [HCOOH[0], HCOOH[1], HCOOH[2], ("H", 2.0, -0.2, 0.5), HCOOH[4]]
    end_w = [near_water[0], ("H", -0.3, 2.05, 0.4), near_water[2]]
    assert subsystem(_node(HCOOH, near_water), _node(end_fa, end_w), eng) is None     # both molecules react
    assert subsystem(_node(HCOOH, WATER), _node(HCOOH, WATER), eng) is None           # nothing reacts
    # a QM/MM system or restraints: their engines number the whole system's atoms
    start, end = _node(HCOOH, WATER, energy=-20.0), _node(CO_H2O, WATER)
    assert subsystem(start, end, type("QMMM", (_Engine,), {"region": object()})()) is None
    assert subsystem(start, end, type("Restrained", (_Engine,), {"restraints": [object()]})()) is None


# --- the explorer end to end (g-xTB) -------------------------------------------------------------------

def _xyz(text):
    rows = [line.split() for line in text.strip().splitlines()]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:]] for r in rows])


# minimized at g-xTB (formic acid; formic acid + water; formic acid + N2), and each one's product guess
SEEDS = {
    "HCOOH": ("""
O  0.77059444606960192 -0.58347432773608876 -0.01882328596190915
C  -0.42218658456243574 0.02105549268606284 0.05261578879500380
O  -0.58823096419112730 1.19881511685449316 -0.09237069788753778
H  1.45003319872529657 0.08359157089578892 -0.20280671150325191
H  -1.21021009604133156 -0.71998785270025267 0.26138490655769353""", """
O  0.73324950479135897 -0.57461552772633973 -1.02511804695028430
C  -0.42242253367557109 0.25744025819204858 -1.06282405020566584
O  -0.59535488765522149 1.23809729455678075 -0.05208996108593204
H  0.42866775053249473 -0.31761476581351455 1.05059983065619544
H  -0.14413983399305683 -0.60330725920897310 1.08943222758568536"""),          # -> CO2 + H2
    "HCOOH + H2O": ("""
O  1.73845708934511967 0.97086819269934388 -0.32897839185765343
C  0.88095613081677604 1.43279632479136221 -1.26500328969949472
O  1.03504566614532800 2.46862812602497073 -1.84358338459011883
O  0.16355075345792905 -1.43872454094614333 0.12658780360423821
H  2.45939313168880158 1.61147976186575415 -0.22885025871298342
H  0.05331695388193557 0.72284618674957568 -1.39700152437858161
H  0.39206528605298357 -2.12677411181472253 0.75312127041647170
H  0.85036224125667237 -0.76761403321751609 0.18141502636911996""", """
O  2.22185497690197753 0.66114127301941217 -0.48262563728221447
C  0.97772479421153657 0.24635575578897317 -1.04524411055246835
O  0.49483739220954986 1.56853253705168783 -0.76467813606926516
O  0.60196467919041374 -0.96899123955721367 -0.34263224834364620
H  2.55483382516723223 1.53491081143186303 -0.21849212357986753
H  0.03990706102787781 0.42808090502416773 -1.53219377480409369
H  0.47362175735400952 -1.67646542173454716 0.32693694394146411
H  0.20840276658294568 1.07994128512828191 0.05663633784108534"""),          # -> HC(OH)3: both react
    "HCOOH + N2": ("""
O  0.22197436706055268 0.15934896000065932 -1.09911899854887030
C  -0.03152137951563336 -1.15093710407859295 -1.08234090310440334
O  0.57908658171250516 -1.94544539090228197 -0.42275959482376269
N  3.06058425484112906 -0.41398345751394855 1.41381666801542583
N  2.48787489294572151 0.28328496262865555 0.80410804977370243
H  0.96534565427734653 0.34818053267234311 -0.50094910758030442
H  -0.87264185894511137 -1.38980790361862616 -1.75466366664721773""", """
O  0.91648937136875408 0.00260966187356222 -1.82421017585646061
C  0.91599823401731351 -1.41131669054901843 -1.65023702717551002
O  0.57017397052009666 -1.96037493330263013 -0.38804409519590721
N  3.15201435091904658 -0.53082590817815456 1.50903043879588639
N  2.41564541093462104 0.40510176127335101 0.74208166481005833
H  -0.56782194459874824 -0.15601137023033596 -0.32779614972175153
H  -0.99179688078458683 -0.45854192169856084 -0.70273220857176311"""),         # -> CO2 + H2, N2 looks on
}


def _expand(name, tmp_path, monkeypatch):
    """One explorer run on `name`'s seed and product (the queue, IRC checks, free energies), recording the
    atoms of every structure a TS search, IRC and Hessian was given."""
    pytest.importorskip("axtb")
    import mepd
    import mepd.compute
    from mepd.discovery.cli_expand import _irc_verifier
    from mepd.discovery.horizontal import HorizontalConnector
    from mepd.discovery.network_expansion import expand_network
    from mepd.inputs import RunInputs

    (sym, x), (_, p) = (_xyz(t) for t in SEEDS[name])
    seed = StructureNode(structure=Structure(symbols=sym, geometry=x * A2B))
    mepd.write_xyz(tmp_path / "products.xyz", StructureNode(structure=Structure(symbols=sym, geometry=p * A2B)))
    seen = {"find_ts": [], "irc": [], "hessian": []}
    for name_, key in (("find_ts", "find_ts"), ("irc", "irc"), ("frequencies", "hessian")):
        real = getattr(mepd.compute, name_)

        def spy(node, *a, _real=real, _key=key, **k):
            seen[_key].append(len(node.symbols))
            return _real(node, *a, **k)

        monkeypatch.setattr(mepd.compute, name_, spy)
        monkeypatch.setattr(mepd, name_, spy, raising=False)
    run_inputs = RunInputs()
    res = expand_network(seed, run_inputs.engine, products_file=str(tmp_path / "products.xyz"), steer="access",
                         access_cutoff_kcal=150.0, connect=HorizontalConnector(run_inputs, tmp_path, 1, say=lambda *a: None),
                         verify=_irc_verifier(run_inputs, tmp_path, 1), kinetics={"free_energy": True})
    return res, seen, run_inputs.engine


def test_a_ts_next_to_a_spectator_is_searched_on_the_reacting_molecule_alone(tmp_path, monkeypatch):
    import mepd

    res, seen, eng = _expand("HCOOH + N2", tmp_path, monkeypatch)
    (st,) = res.steps
    assert st.files["verified"] is True and {st.a, st.b} == {0, 1}
    # the TS search, its IRC and every Hessian on formic acid (5 atoms) or a molecule apart -- never all 7
    assert seen["find_ts"] and set(seen["find_ts"]) == {5} and set(seen["irc"]) == {5}
    assert 7 not in seen["hessian"]
    # the TS is kept whole, N2 where it is in the seed; its energy: formic acid's TS + what N2 adds to the seed
    ts = mepd.load(st.files["ts"])
    seed = res.species[0].node
    assert len(ts.symbols) == 7 and np.allclose(ts.coords[3:5], seed.coords[3:5])
    sub = [0, 1, 2, 5, 6]
    alone = [StructureNode(structure=Structure(symbols=[str(n.symbols[k]) for k in sub], geometry=n.coords[sub]))
             for n in (ts, seed)]
    e_ts, e_seed = eng.compute_energies(alone)
    assert st.ts_energy == pytest.approx(e_ts + float(seed.energy) - e_seed, abs=1e-6)
    # the IRC drawn whole: N2 in every frame where it is at the TS
    assert all(np.allclose(f.coords[3:5], ts.coords[3:5]) for f in mepd.read_frames(st.files["irc"]))


# What the explorer gave on develop 6863884 (before the cut) for seeds whose every molecule reacts (exact on
# the machine that recorded them): each species' energy, and each step's TS energy (Hartree) and free-energy barriers (kcal/mol, both ways).
BEFORE = {
    "HCOOH": ([-189.79135496848835, -189.7996495501047],
              [(-189.6618711929768, 90.90132971348241, 74.7977931340884)]),
    "HCOOH + H2O": ([-266.2449784210655, -266.22788530758714],
                    [(-266.15766640170943, 52.060339567851365, 40.47973297130469)]),
}


def _fresh(name, tmp):
    """In a process of its own (spawned): what `_expand` gives -- g-xTB's last digits depend on what the
    process computed before (on develop too: formic acid's barrier moves by 3e-6 kcal/mol after another run)."""
    res, seen, _ = _expand(name, Path(tmp), pytest.MonkeyPatch())
    return ([float(s.node.energy) for s in res.species],
            [(st.ts_energy, *res.kinetics.barriers_kcal[m]) for m, st in enumerate(res.steps)],
            len(res.species[0].node.symbols), seen)


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_when_every_molecule_reacts_the_explorer_gives_what_it_gave_before(name, tmp_path):
    pytest.importorskip("axtb")
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        species, steps, n, seen = pool.submit(_fresh, name, str(tmp_path)).result()
    assert set(seen["find_ts"]) == {n} and set(seen["irc"]) == {n}       # all of it, as before
    # the same to within floating point: another CPU and BLAS (the image's) move the last digits (~1e-10 Eh)
    want_species, want_steps = BEFORE[name]
    assert len(steps) == len(want_steps)
    assert species == pytest.approx(want_species, rel=0, abs=1e-7)                      # Hartree
    # TS energies (Hartree), then the free-energy barriers (kcal/mol), where a TS optimization and Hessian amplify
    # that noise (5.5e-4 kcal/mol on the image)
    assert [s[0] for s in steps] == pytest.approx([s[0] for s in want_steps], rel=0, abs=1e-6)
    assert [x for s in steps for x in s[1:]] == pytest.approx([x for s in want_steps for x in s[1:]], rel=0, abs=1e-2)
