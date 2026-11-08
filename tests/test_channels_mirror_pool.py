"""Channel pairs when only one end is achiral (isoprene + methyl acrylate -> a chiral adduct): the achiral end's
pool gets every conformer's mirror image, since (R*, P) mirrors (R, P*), a pair with the enantiomer no pool
holds -- without it one approach is never searched (isoprene-free/NOTES.md, diag19). The mirror half of the
pairs gets its own quota per mechanism: its endpoint RMSDs run higher."""
import json
from types import SimpleNamespace

import numpy as np
from qcdata import Structure

from mepd import cli_channels
from mepd._qcinf_compat import snap_rmsd
from mepd.chain import Chain
from mepd.conformers import is_own_mirror_molecule, mirror_image, mirror_images_missing
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode

from test_channels_face_pairs import _chfcl_br
from test_cli_channels import _call_channels, _install_fake_gxtb_with_coordinate_dependent_energy, \
    _run_inputs_for_test


def _node(structure, energy=None):
    node = StructureNode(structure=structure)
    node._cached_energy = energy
    return node


def test_mirror_images_missing_closes_a_pool_once():
    chiral_conformer = _node(_chfcl_br(3.2, 0.0), -1.0)   # planar CHFCl, Br above: its mirror is another conformer
    added = mirror_images_missing([chiral_conformer], 0.1)
    assert len(added) == 1 and added[0].energy == -1.0
    assert snap_rmsd(added[0].structure, mirror_image(chiral_conformer.structure)) < 1e-6
    assert mirror_images_missing([chiral_conformer, *added], 0.1) == []   # closed: nothing more
    # a conformer that is its own mirror image (all atoms in one plane) adds nothing
    flat = Structure(symbols=["C", "H", "F", "Cl", "Br"], charge=0, multiplicity=1,
                     geometry=np.array([[0, 0, 0], [2.06, 0, 0], [-1.3, 2.2, 0], [-1.7, -2.9, 0], [6, 0, 0]], float))
    assert mirror_images_missing([_node(flat)], 0.1) == []


def test_chirality_of_an_end_is_its_molecule():
    assert not is_own_mirror_molecule(_node(_chfcl_br(1.95, 0.35)))      # CHFClBr: a stereocentre
    assert is_own_mirror_molecule(_node(_chfcl_br(3.2, 0.0)))            # CHFCl + Br: no stereocentre


def test_mirror_half_gets_its_own_quota(tmp_path, monkeypatch):
    product = _chfcl_br(1.95, 0.35)
    structures = [_node(_chfcl_br(3.2, 0.0)), _node(_chfcl_br(3.3, 0.0)),   # 0, 1: sampled
                  _node(mirror_image(_chfcl_br(3.2, 0.0))),                  # 2: an added mirror image
                  _node(product)]                                             # 3

    def choices(start, end, metric, run_inputs, max_variants_per_mechanism=None):
        score = 3.0 if start is structures[2].structure else 1.0             # the mirror half scores worse
        return [SimpleNamespace(key="form C-Br", score=score, n_variants=1,
                                winner=SimpleNamespace(end_structure=end, atom_map=None))]

    monkeypatch.setattr("mepd.atom_mapping_selection.select_per_mechanism", choices)
    run_inputs = SimpleNamespace(atom_mapping_inputs=SimpleNamespace(metric="endpoint-rmsd", n_candidates=4))
    pairs = [(0, 3), (1, 3), (2, 3)]
    _, shared, _ = cli_channels._expand_pairs_by_mechanism(structures, pairs, True, run_inputs,
                                                           pairs_per_mechanism=1, workers=1)
    assert len(shared) == 1 and shared[0][0] != 2                           # one cut over all: the mirror half lost
    _, split, _ = cli_channels._expand_pairs_by_mechanism(structures, pairs, True, run_inputs,
                                                          pairs_per_mechanism=1, workers=1, mirrors=frozenset({2}))
    assert len(split) == 2 and {i for i, _ in split} & {0, 1} and (2, 3) in split


def test_achiral_start_chiral_end_closes_the_start_pool(tmp_path, monkeypatch):
    """1-butanol -> (R)-2-butanol: every reactant conformer's mirror image is in the reactant pool (and so in the
    pairs), none was merged, and the product pool is as sampled."""
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)
    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)
    out = tmp_path / "out"
    _call_channels(start="CCCCO", end="CC[C@@H](C)O", inputs=inputs_fp, n_conformers=0, n_embed=0, max_pairs=0,
                   conformers_only=True, output=out)
    stats = json.loads((out / "stats.json").read_text())
    start, end = stats["conformers"]["start"], stats["conformers"]["end"]
    assert start["n_mirror_images_added"] > 0 and start["n_mirror_images_merged"] == 0
    assert "n_mirror_images_added" not in end
    pool = Chain.from_xyz(out / "conformers" / "start.xyz", ChainInputs()).nodes
    assert start["n_final"] == len(pool) == start["n_kept"] + start["n_mirror_images_added"]
    assert mirror_images_missing(list(pool), 0.1) == []
    assert stats["n_pairs"] == len(pool) * end["n_final"]


def test_both_ends_achiral_as_before(tmp_path, monkeypatch):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)
    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)
    out = tmp_path / "out"
    _call_channels(start="CCCCO", end="CCOCC", inputs=inputs_fp, n_conformers=0, n_embed=0, max_pairs=0,
                   conformers_only=True, output=out)
    stats = json.loads((out / "stats.json").read_text())
    assert all("n_mirror_images_added" not in stats["conformers"][s] for s in ("start", "end"))
