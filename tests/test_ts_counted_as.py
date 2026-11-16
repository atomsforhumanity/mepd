"""A TS counts as what it was optimized on (its file's `optimized_on=` record), not as the bond lengths of its
step's ends make it look: free_energy.counted_as."""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

import mepd
from mepd.discovery.free_energy import (FreeEnergies, counted_as, counted_record, optimized_on,
                                        optimized_on_comment)
from mepd.nodes.node import StructureNode

BOHR = 1.8897259886
# formic acid (O0 C1 O2 H3 H4) next to a spectator water (O5 H6 H7), Angstrom
SYM = ["O", "C", "O", "H", "H", "O", "H", "H"]
ACID = np.array([[0.74, -0.62, -0.08], [-0.41, 0.06, -0.16], [-0.56, 1.17, 0.27], [1.40, -0.07, 0.37],
                 [-1.17, -0.55, -0.67]])
WATER = np.array([[6.0, 0, 0], [6.96, 0, 0], [5.76, 0.93, 0]])
START = np.vstack([ACID, WATER])
# both ends keep C1-O0; at the "TS" it is stretched past 1.25 x the covalent radii (a bond of both ends missing)
STRETCHED = START.copy()
STRETCHED[[0, 3]] += [1.0, -0.6, 0.0]
END = START.copy()
END[4] = START[0] + [0.5, -0.8, 0.0]          # H4 from C1 to O0: bonds C1-H4 broken, O0-H4 made


def _st(xyz, comment=None):
    extras = {"xyz_comments": comment.split()} if comment else {}
    return Structure(symbols=SYM, geometry=np.asarray(xyz) * BOHR, charge=0, multiplicity=1, extras=extras)


def test_the_record_goes_through_a_ts_file(tmp_path):
    node = StructureNode(structure=_st(START))
    for atoms, back in ((None, list(range(8))), ([4, 0, 1, 2, 3], [0, 1, 2, 3, 4])):
        mepd.write_xyz(tmp_path / "ts.xyz", node, comments=[f"E=-1.5 {optimized_on_comment(atoms)}"])
        assert optimized_on(Structure.open(str(tmp_path / "ts.xyz"))) == back
    assert optimized_on(_st(START)) is None                              # a file from before the record
    assert optimized_on(_st(START, "E=-1.5 optimized_on=0,1,99")) is None  # not this file's atoms: no record
    assert optimized_on(_st(START, "E=-1.5 optimized_on=x")) is None


def test_a_ts_counts_as_what_it_was_optimized_on_whatever_its_bonds_look_like():
    ends = (_st(START), _st(END))
    # read from the ends' bonds (no record), the stretched TS is one molecule, spectator included: as before
    assert counted_as(_st(STRETCHED), ends) == (None, "ends")
    assert counted_as(_st(START), ends) == ([[0, 1, 2, 3, 4], [5, 6, 7]], "ends")
    # its record: the subsystem alone, the water apart ...
    sub = optimized_on_comment(range(5))
    assert counted_as(_st(STRETCHED, sub), ends) == ([[0, 1, 2, 3, 4], [5, 6, 7]], "optimized_on")
    # ... also where the subsystem was put back onto the spectator (O0 1.2 Angstrom from the water's O)
    clash = STRETCHED.copy()
    clash[5:] += clash[0] - clash[5] + [1.2, 0.0, 0.0]
    assert counted_as(_st(clash, sub), ends) == ([[0, 1, 2, 3, 4], [5, 6, 7]], "optimized_on")
    # optimized as the whole box: one molecule, though its ends would set the water apart
    assert counted_as(_st(START, optimized_on_comment()), ends) == ([list(range(8))], "optimized_on")
    assert counted_as(_st(START), None) == (None, "none")


def test_the_free_energies_parts_keys_and_notes_follow_the_record():
    ends = (_st(START), _st(END))
    free = FreeEnergies(None, 1500.0)
    assert [p[0] for p in free.parts(_st(STRETCHED), True, ends)] == [list(range(8))]
    assert free.notes == ["a TS whose step's ends don't match its bonds counted as one molecule, spectators included"]
    free = FreeEnergies(None, 1500.0)
    cut = _st(STRETCHED, optimized_on_comment(range(5)))
    assert [p[0] for p in free.parts(cut, True, ends)] == [[0, 1, 2, 3, 4], [5, 6, 7]]
    whole = _st(START, optimized_on_comment())
    assert [p[0] for p in free.parts(whole, True, ends)] == [list(range(8))]
    assert not free.notes
    # a correction saved for this geometry before its file said what it was optimized on (no saddle step then)
    # is not this one's
    assert free._key(whole, True, ends) != free._key(_st(START), True, ends)
    # one geometry counted two ways is two cache entries
    assert free._key(cut, True, ends) != free._key(_st(STRETCHED), True, ends)
    # a species (no TS) is never read from a record
    assert [p[0] for p in free.parts(whole, False)] == [[0, 1, 2, 3, 4], [5, 6, 7]]


def test_the_queue_s_ts_files_say_what_they_were_optimized_on(tmp_path):
    from mepd.discovery.horizontal import HorizontalConnector

    ts = StructureNode(structure=_st(START))
    ts._cached_energy = -1.5
    queue = SimpleNamespace(ts_dir=tmp_path)
    HorizontalConnector._write_ts(queue, ts, "whole")
    HorizontalConnector._write_ts(queue, ts, "cut", (ts, ts), SimpleNamespace(atoms=[0, 1, 2, 3, 4]))
    assert optimized_on(Structure.open(str(tmp_path / "whole.xyz"))) == list(range(8))
    assert optimized_on(Structure.open(str(tmp_path / "cut.xyz"))) == [0, 1, 2, 3, 4]
    assert (tmp_path / "cut_ends.xyz").exists() and not (tmp_path / "whole_ends.xyz").exists()
    HorizontalConnector._write_ts(queue, ts, "peak", (ts, ts), optimized=False)   # a path's highest image: no record
    assert optimized_on(Structure.open(str(tmp_path / "peak.xyz"))) is None
    from mepd.discovery.network_expansion import _xyz_energy

    assert _xyz_energy(Structure.open(str(tmp_path / "cut.xyz"))) == -1.5


def test_a_known_run_s_ts_keeps_its_record_under_another_atom_order(tmp_path):
    from mepd.discovery.known import permuted_xyz

    mepd.write_xyz(tmp_path / "ts.xyz", StructureNode(structure=_st(START)),
                   comments=[f"E=-1.5 {optimized_on_comment(range(5))}"])
    order = [5, 6, 7, 0, 1, 2, 3, 4]              # the water first
    permuted_xyz(tmp_path / "ts.xyz", tmp_path / "moved.xyz", order)
    moved = Structure.open(str(tmp_path / "moved.xyz"))
    assert list(moved.symbols) == [SYM[i] for i in order]
    assert optimized_on(moved) == [3, 4, 5, 6, 7]
    mepd.write_xyz(tmp_path / "w.xyz", StructureNode(structure=_st(START)), comments=[f"E=-1.5 {optimized_on_comment()}"])
    permuted_xyz(tmp_path / "w.xyz", tmp_path / "w2.xyz", order)
    assert optimized_on(Structure.open(str(tmp_path / "w2.xyz"))) == list(range(8))


def test_a_step_s_record_says_what_its_ts_counts_as_and_where_that_comes_from(tmp_path):
    ends = (_st(START), _st(END))
    node = StructureNode(structure=_st(STRETCHED))
    mepd.write_xyz(tmp_path / "old.xyz", node, comments=["E=-1.5"])
    mepd.write_xyz(tmp_path / "new.xyz", node, comments=[f"E=-1.5 {optimized_on_comment(range(5))}"])
    # what the correction counted is saved with it (Correction.counted) and the record is that, uncomputed
    from mepd.discovery.free_energy import Correction

    counted = {"molecules": [8], "from": "optimized_on", "saddle": "used", "one_molecule_because": "x"}
    kept = pickle.loads(pickle.dumps(Correction(-101.5, counted)))            # a state file, a worker's answer
    assert kept == -101.5 and kept + 1.0 == -100.5 and kept.counted == counted
    assert counted_record({"ts": str(tmp_path / "new.xyz")}, ends, kept) == counted
    # a correction saved without it (an older state, a known run): the molecules the file names now, marked
    late = {"not_saved_with_the_correction": True}
    assert counted_record({"ts": str(tmp_path / "old.xyz")}, ends) == {"molecules": [8], "from": "ends", **late}
    assert counted_record({"ts": str(tmp_path / "new.xyz")}, ends, -101.5) == {"molecules": [5, 3],
                                                                             "from": "optimized_on", **late}
    assert counted_record({"ts": str(tmp_path / "gone.xyz")}, ends) is None


def test_a_subsystem_is_found_from_the_two_ends_alone():
    """network_expansion.subsystem reads its ends' bonds: a frame that carries another search's record (a TS
    file's) does not decide it."""
    from mepd.discovery.network_expansion import subsystem

    class Engine:
        def compute_energies(self, nodes):
            return [0.0] * len(nodes)

    a = StructureNode(structure=_st(START, optimized_on_comment()))
    a._cached_energy = 0.0
    cut = subsystem(a, StructureNode(structure=_st(END)), Engine())
    assert cut is not None and cut.atoms == [0, 1, 2, 3, 4]


def test_a_molecule_that_hands_an_atom_on_is_not_one_that_takes_no_part():
    """What a whole-box TS's subsystem TS leaves out (network_expansion.subsystem's reading of the two IRC ends):
    a molecule with the same atoms and the same bonds at both ends. A water that takes the acid's H3 and gives
    its own H6 to the carbonyl O has the same formula at both ends but not the same atoms: it reacts."""
    bystander = [p[0] for p in FreeEnergies(None, 298.15).parts(_st(START), True, (_st(START), _st(END)))]
    assert bystander == [[0, 1, 2, 3, 4], [5, 6, 7]]
    relay = START.copy()
    relay[3] = START[5] + [-0.6, -0.75, 0.0]        # H3: off O0, onto the water's O5
    relay[6] = START[2] + [0.6, 0.75, 0.0]          # H6: off O5, onto the carbonyl O2
    from mepd.discovery.free_energy import bonds

    b0, b1 = bonds(SYM, START), bonds(SYM, relay)
    assert b0 ^ b1 == {(3, 0), (5, 3), (6, 5), (6, 2)}
    assert [p[0] for p in FreeEnergies(None, 298.15).parts(_st(START), True, (_st(START), _st(relay)))] == [list(range(8))]


def test_a_fast_ts_found_on_disk_after_a_crash_comes_back_with_its_energy(tmp_path):
    """A run killed between writing a fast TS and saving its state finds the file again on --resume: the queue
    then takes it from the file. Without its energy the main process stopped with EnergiesNotComputedError
    (acetic acid, staged, killed at 200 s: the resume ended in 2 s, on develop 95e094d too)."""
    from mepd.discovery.horizontal import HorizontalConnector

    ts = StructureNode(structure=_st(START))
    ts._cached_energy = -1.5
    queue = SimpleNamespace(ts_dir=tmp_path, free=None)
    queue._ts_correction = lambda *a, **k: None
    HorizontalConnector._write_ts(queue, ts, "fast_0_1")
    back, label, corr = HorizontalConnector._fast_ts(queue, 0, 1, ts, ts)
    assert label == "fast_0_1" and back.energy == -1.5


def test_a_file_written_from_another_file_s_structure_reads_its_own_record(tmp_path):
    """A structure read from a TS file keeps that file's comment (extras), and mepd.write_xyz writes it in front
    of the new one: a rotamer guess made from a whole-box TS and optimized on a subsystem read as whole."""
    parent = Structure(symbols=SYM, geometry=START * BOHR, charge=0, multiplicity=1,
                       extras={"xyz_comments": ["E=-5.0", optimized_on_comment()]})
    node = StructureNode(structure=parent)
    mepd.write_xyz(tmp_path / "rot.xyz", node, comments=[f"E=-1.0 {optimized_on_comment(range(5))}"])
    line = (tmp_path / "rot.xyz").read_text().splitlines()[1]
    assert line.index("optimized_on=whole") < line.index("optimized_on=0,1,2,3,4")
    assert optimized_on(Structure.open(str(tmp_path / "rot.xyz"))) == [0, 1, 2, 3, 4]


def test_without_free_energies_a_ts_s_molecules_are_read_as_before_the_record():
    """A run on electronic energies uses FreeEnergies only to tell a TS's spectators (its steering single points,
    their cache keys): by_record False reads them from the step's ends whatever the file says, so such a run is
    what it was before the files said what they were optimized on."""
    ends = (_st(START), _st(END))
    old = FreeEnergies(None, 1500.0, by_record=False)
    for record in (optimized_on_comment(), optimized_on_comment(range(5))):
        assert [p[0] for p in old.parts(_st(START, record), True, ends)] == [[0, 1, 2, 3, 4], [5, 6, 7]]
        assert [p[0] for p in old.parts(_st(STRETCHED, record), True, ends)] == [list(range(8))]
        assert old._key(_st(START, record), True, ends) == old._key(_st(START), True, ends)
    assert len(old.notes) == 2          # the stretched one: as before, said


def test_the_record_says_one_molecule_where_a_charged_ts_could_not_be_counted_apart(monkeypatch):
    """HO- + H2O proton transfer beside a far water, charge -1, optimized on its 5 atoms. When the molecules'
    charges are not clear the correction counts the whole box as one molecule (FreeEnergies.parts): what it
    counted is saved with it and the step's record says that, not the split its file names; the saddle step does
    not run on the box. Nothing is computed when the record is made."""
    import mepd.discovery.free_energy as fe

    sym = ["O", "H", "H", "O", "H", "O", "H", "H"]
    r = np.array([[0, 0, 0], [0.96, 0, 0], [-0.3, 0.9, 0], [2.6, 0, 0], [2.9, 0.9, 0],
                  [0, 0, 8], [0.96, 0, 8], [-0.3, 0.9, 8]], float)
    p, ts = r.copy(), r.copy()
    p[1], ts[1] = [1.64, 0, 0], [1.3, 0, 0]

    def anion(x, comment=None):
        return Structure(symbols=sym, geometry=x * BOHR, charge=-1, multiplicity=1,
                         extras={"xyz_comments": comment.split()} if comment else {})

    ends = (anion(r), anion(p))
    said = anion(ts, f"E=-1.0 {optimized_on_comment(range(5))}")
    seen = []
    free = FreeEnergies(None, 298.15)
    monkeypatch.setattr(free, "_one", lambda *a: seen.append(a) or 0.0)
    monkeypatch.setattr("mepd.energies", lambda nodes, engine: [0.0] * len(nodes))
    monkeypatch.setattr(fe, "fragment_charges", lambda *a, **k: [-1, 0])
    apart = free._correction(said, True, ends, -1.0)
    assert apart.counted["molecules"] == [5, 3] and "one_molecule_because" not in apart.counted
    assert seen[0][4:] == (True, True)                                  # the 5 atoms: with the saddle step
    seen.clear()
    monkeypatch.setattr(fe, "fragment_charges", lambda *a, **k: None)    # not clear
    whole = free._correction(said, True, ends, -1.0)
    assert whole.counted["molecules"] == [8] and whole.counted["from"] == "optimized_on"
    assert "charges are not clear" in whole.counted["one_molecule_because"]
    assert len(seen) == 1 and len(seen[0][0]) == 8 and len(seen[0]) == 5       # the box, and no saddle step on it
    monkeypatch.setattr(fe, "fragment_charges", lambda *a, **k: 1 / 0)         # the record computes nothing
    assert counted_record({"ts": "nowhere.xyz"}, ends, whole) == whole.counted


def test_a_path_peak_made_from_a_ts_s_irc_end_is_written_without_that_ts_s_record(tmp_path):
    """The comment of a TS file travels with its structure through the IRC frames to the minimized end, a
    species' node. A path's highest image that is such a node (or interpolated from one) said `optimized_on`
    in its file, whatever its writer wrote, and got the saddle step. Every writer of a TS file starts from the
    structure without a record."""
    from mepd.discovery.free_energy import unrecorded
    from mepd.discovery.horizontal import HorizontalConnector

    end = _st(START, f"E=-99.0 {optimized_on_comment()}")                # an IRC end of a whole-box TS
    assert optimized_on(unrecorded(end)) is None and unrecorded(_st(START)).extras == _st(START).extras
    peak = StructureNode(structure=end)
    peak._cached_energy = -1.5
    queue = SimpleNamespace(ts_dir=tmp_path)
    HorizontalConnector._write_ts(queue, peak, "peak", None, None, optimized=False)
    assert optimized_on(Structure.open(str(tmp_path / "peak.xyz"))) is None
    HorizontalConnector._write_ts(queue, peak, "ts", None, SimpleNamespace(atoms=[0, 1, 2, 3, 4]))
    assert (tmp_path / "ts.xyz").read_text().splitlines()[1].count("optimized_on=") == 1
    assert optimized_on(Structure.open(str(tmp_path / "ts.xyz"))) == [0, 1, 2, 3, 4]


def test_a_subsystem_ts_is_not_the_mirror_image_of_a_whole_box_ts():
    """A chiral whole-box TS P, its subsystem TS 0.01 Angstrom from it, and the other enantiomer's two: the mirror
    image of P is P', of P_sub is P'_sub. Read from energy and geometry alone P pairs with whichever comes
    first; on free energies the two count as different molecules and their files say which each is."""
    from mepd.discovery.kinetics import Step
    from mepd.discovery.mirror import MirrorFinder, mirrored

    sym = ["C", "H", "F", "Cl", "Br"]

    def st(x, comment=None):
        return Structure(symbols=sym, geometry=np.asarray(x, float) * BOHR, charge=0, multiplicity=1,
                         extras={"xyz_comments": [comment]} if comment else {})

    a = st([[0, 0, 0], [1.09, 0, 0], [-0.6, 1.2, 0], [-0.8, -1.5, 0], [3.5, 2.0, 0]])     # planar: achiral
    b = st([[0, 0, 0], [1.09, 0, 0], [-0.6, 1.2, 0], [3.0, -1.5, 0], [-0.9, -1.7, 0]])
    whole, sub = optimized_on_comment(), optimized_on_comment(range(4))
    P = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.8, -0.8, 0.8], [1.0, -1.0, -1.0], [-1.3, 1.3, -1.3]])
    S = P + np.random.default_rng(0).normal(0, 0.01, (5, 3))
    M = np.array([-1.0, 1.0, 1.0])
    ts = [st(P, whole), st(S * M, sub), st(S, sub), st(P * M, whole)]     # P, P'_sub, P_sub, P'
    assert np.allclose(np.asarray(mirrored(ts[0]).geometry), np.asarray(ts[3].geometry))
    steps = [Step(0, 1, -1.0, label, {}) for label in ("P", "P'_sub", "P_sub", "P'")]
    nodes = [StructureNode(structure=a), StructureNode(structure=b)]
    twins = {frozenset(t) for t in MirrorFinder(by_record=True)(nodes, steps, ts).found_twins}
    assert twins == {frozenset((0, 3)), frozenset((1, 2))}
    plain = {frozenset(t) for t in MirrorFinder()(nodes, steps, ts).found_twins}      # as before: the first that fits
    assert frozenset((0, 1)) in plain
