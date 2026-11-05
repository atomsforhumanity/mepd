"""Channels on a QM/MM step (#27): conformers are the QM region's in its
environment (MD of the QM atoms with QM/MM forces, then minimized), and the
rest of the channel search runs on the QM/MM system as is (no atom mapping,
no mirror-image merge)."""
import json

import numpy as np
import pytest
from qcdata import Structure

from mepd.discovery.qmmm_reactor import qm_bond_key
from mepd.nodes.node import StructureNode

from tests.test_qmmm import PROPANOL, QM_PROPANOL, _xtb


@pytest.fixture
def qmmm_inputs():
    from mepd.inputs import RunInputs

    _xtb()
    ri = RunInputs(engine_name="xtb", qmmm={"reference": PROPANOL, "qm_atoms": QM_PROPANOL, "active_radius": None},
                   xtb_engine_kwds={"n_parallel": 1})
    yield ri
    StructureNode.set_global_graph_atoms(None)


def test_qm_region_conformers_keep_the_species_and_start_with_the_input(qmmm_inputs):
    from mepd.qmmm_conformers import qm_rmsd, qmmm_region, sample_qm_region_conformers

    region = qmmm_region(qmmm_inputs)
    assert region is not None and region.qm_atoms == QM_PROPANOL
    start = StructureNode(structure=Structure.from_xyz(PROPANOL))
    stats = {}
    confs = sample_qm_region_conformers(start, qmmm_inputs, n_conformers=4, temperature=800.0, time_ps=0.05,
                                        frames=4, equilibrate_ps=0.01, stats=stats)
    assert confs[0] is start and 1 <= len(confs) <= 4
    assert stats["n_generated"] == 4 and stats["n_kept"] == len(confs)
    key = qm_bond_key(start.structure, region)
    for i, c in enumerate(confs[1:], 1):
        assert qm_bond_key(c.structure, region) == key                    # the same species
        assert all(qm_rmsd(c, k, region) >= 0.1 for k in confs[:i])     # distinct


def test_channels_on_a_qmmm_profile_samples_the_qm_region_and_skips_atom_mapping(qmmm_inputs, tmp_path, monkeypatch):
    from mepd import qmmm_conformers
    from mepd.channel_search import ChannelsOptions, run_channels

    calls = {}

    def fake_sampler(node, run_inputs, **kw):
        calls.setdefault("n", 0)
        calls["n"] += 1
        calls["kw"] = kw
        return [node]

    monkeypatch.setattr(qmmm_conformers, "sample_qm_region_conformers", fake_sampler)
    monkeypatch.setattr("mepd.channel_search._check_endpoint_atom_mapping",
                        lambda *a, **k: pytest.fail("atom mapping on a QM/MM system"))
    s = Structure.from_xyz(PROPANOL)
    opts = ChannelsOptions(conformers_only=True, n_conformers=3, qmmm_temperature=350.0, qmmm_md_ps=0.5)
    run_channels(StructureNode(structure=s), StructureNode(structure=s), qmmm_inputs, opts, tmp_path)
    assert calls["n"] == 2 and calls["kw"]["temperature"] == 350.0 and calls["kw"]["time_ps"] == 0.5
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_pairs"] == 1
    with pytest.raises(Exception, match="pair-from both"):
        run_channels(StructureNode(structure=s), StructureNode(structure=s), qmmm_inputs,
                     ChannelsOptions(conformers_only=True, pair_from="start"), tmp_path / "b")
