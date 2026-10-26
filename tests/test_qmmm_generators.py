"""A whole-molecule product generator (e.g. CREST msreact) on a QM/MM system
works on its QM model (QM atoms capped with link H), and each product is put
back into the full system; a product that breaks a QM/MM boundary bond (its
link H stops being a cap) is dropped."""

from __future__ import annotations

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.nodes.node import StructureNode
from mepd.qmmm import QMMMRegion

from test_qmmm import QM_PROPANOL, _propanol  # noqa: E402  (tests/ is on sys.path)


def test_products_of_the_qm_model_go_back_into_the_system():
    from mepd.discovery.network_expansion import _region_proposals, _View

    s = _propanol()
    region = QMMMRegion.build(s, QM_PROPANOL, active_radius=None)
    view = _View(region)
    node = StructureNode(structure=s, has_molecular_graph=False)
    model = region.model_structure(s)
    nqm = len(region.qm_atoms)
    x = np.asarray(model.geometry, dtype=float)

    # A product that keeps the cap, turned and shifted (the generator's frame).
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    kept = model.model_copy(update={"geometry": x @ turn.T + 3.0})
    # One whose link H moved onto the oxygen: the cap is gone.
    o = view.symbols(node).index("O")
    y = x.copy()
    y[nqm] = x[o] + np.array([0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR])
    broke = model.model_copy(update={"geometry": y})

    props, counts = _region_proposals(view, node, 0, [kept, broke])
    assert counts == {"kept": 1, "boundary": 1}
    (p,) = props
    full = np.asarray(p.structure.geometry, dtype=float)
    assert full.shape == np.asarray(s.geometry).shape
    mm = [i for i in range(len(s.symbols)) if i not in region.qm_atoms]
    assert np.allclose(full[mm], np.asarray(s.geometry)[mm])          # the environment stays where it was
    moved = np.linalg.norm(full[region.qm_atoms] - np.asarray(s.geometry)[region.qm_atoms], axis=1)
    assert moved.max() < 0.5 * ANGSTROM_TO_BOHR                       # aligned back onto the QM atoms
    assert p.broken == () and p.formed == ()                          # the same molecule: no bond changed
    assert p.smiles


def test_a_product_with_no_lewis_structure_is_named_after_optimizing(monkeypatch):
    """lewis_smiles can find no Lewis structure (None): the proposal is then
    labelled like an imported product and named once optimized, instead of
    failing the run (\"'NoneType' object has no attribute 'startswith'\")."""
    import mepd.discovery.network_expansion as ne

    s = _propanol()
    region = QMMMRegion.build(s, QM_PROPANOL, active_radius=None)
    view = ne._View(region)
    node = StructureNode(structure=s, has_molecular_graph=False)
    monkeypatch.setattr(ne, "lewis_smiles", lambda *a, **k: None)
    (p,), _ = ne._region_proposals(view, node, 0, [region.model_structure(s)])
    assert p.smiles == "external-0"


def test_the_nanoreactor_anchors_a_qm_region_at_its_link_h_and_the_atoms_they_cap():
    from mepd.discovery.network_expansion import nanoreactor_anchors

    s = _propanol()
    region = QMMMRegion.build(s, QM_PROPANOL, active_radius=None)
    model = region.model_structure(s)
    nqm = len(region.qm_atoms)
    anchors = nanoreactor_anchors(region)
    assert anchors[:len(region.links)] == list(range(nqm, nqm + len(region.links)))
    assert all(model.symbols[a] == "H" for a in anchors[:len(region.links)])
    assert anchors[len(region.links):] == sorted(region.qm_atoms.index(q) for q, _ in region.links)
