"""Channel pairs per mechanism: a start that holds a bond-forming partner on the other side of an atom the
product makes a stereocentre goes after the right-face starts, even when its endpoint RMSD is lower
(isoprene + methyl acrylate's R para adduct: wrong-face pairs gave its concerted TS in 1 of 29 searches)."""
import json
from types import SimpleNamespace

import numpy as np
from qcdata import Structure

from mepd import cli_channels
from mepd.channel_search import _wrong_face
from mepd.nodes.node import StructureNode

ANG = 1 / 0.529177210903


def _chfcl_br(z_br: float, pyramid: float) -> Structure:
    """CHFCl with Br on the z axis at `z_br` (Angstrom); H, F, Cl tilted away from Br by `pyramid`."""
    sub = {"H": (1.09, 0.0), "F": (1.36, 120.0), "Cl": (1.77, 240.0)}
    xyz = [(0.0, 0.0, 0.0)]
    for d, ang in sub.values():
        t = np.radians(ang)
        xyz.append((d * np.cos(t), d * np.sin(t), -np.sign(z_br) * pyramid))
    xyz.append((0.0, 0.0, z_br))
    return Structure(symbols=["C", "H", "F", "Cl", "Br"], geometry=np.array(xyz) * ANG, charge=0, multiplicity=1)


def test_wrong_face_is_geometry():
    product = _chfcl_br(1.95, 0.35)                    # Br bonded on +z: a stereocentre at C
    assert not _wrong_face(_chfcl_br(3.2, 0.0), product)   # Br waits on +z: the right face
    assert _wrong_face(_chfcl_br(-3.2, 0.0), product)      # Br waits on -z: the other face
    # no stereocentre formed (H, H, F, Br): nothing to turn over
    achiral = product.model_copy(update={"symbols": ["C", "H", "F", "H", "Br"]})
    flat = _chfcl_br(-3.2, 0.0).model_copy(update={"symbols": ["C", "H", "F", "H", "Br"]})
    assert not _wrong_face(flat, achiral)


def test_right_face_start_kept_over_lower_rmsd(tmp_path, monkeypatch):
    product = _chfcl_br(1.95, 0.35)
    structures = [StructureNode(structure=_chfcl_br(-3.2, 0.0)),    # 0: wrong face, best RMSD
                  StructureNode(structure=_chfcl_br(3.2, 0.0)),     # 1: right face
                  StructureNode(structure=product)]

    def choices(start, end, metric, run_inputs, max_variants_per_mechanism=None):
        score = 1.0 if start is structures[0].structure else 2.0
        return [SimpleNamespace(key="form C-Br", score=score, n_variants=1,
                                winner=SimpleNamespace(end_structure=end, atom_map=None))]

    monkeypatch.setattr("mepd.atom_mapping_selection.select_per_mechanism", choices)
    run_inputs = SimpleNamespace(atom_mapping_inputs=SimpleNamespace(metric="endpoint-rmsd", n_candidates=4))
    _, cands, _ = cli_channels._expand_pairs_by_mechanism(structures, [(0, 2), (1, 2)], True, run_inputs,
                                                           pairs_per_mechanism=1, workers=1, output=tmp_path)
    assert cands == [(1, 2)]
    table = json.loads((tmp_path / "pair_mechanisms.json").read_text())
    assert [r["start_conformer"] for r in table] == [1]
