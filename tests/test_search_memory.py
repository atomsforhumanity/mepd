"""A search holds only what it uses: Sella comes without torch on a machine with
no GPU, and GSM's live-string history keeps no molecular graphs (one stopped
search's string gets them back before it is analyzed), and reading a finished
tree back for its TS guesses loads only each search's first and last string."""

import subprocess
import sys

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.pathminimizers.gsm import _graphless_copy, _with_graphs

WATER = "3\n\nO 0.000 0.000 0.000\nH 0.758 0.000 0.504\nH -0.758 0.000 0.504\n"


def test_sella_without_a_gpu_does_not_load_torch_and_torch_stays_importable():
    code = (
        "import sys\n"
        "from mepd.engines._sella import sella\n"
        "s = sella()\n"
        "assert s.Sella and s.IRC\n"
        "assert 'torch' not in sys.modules, 'torch was loaded'\n"
        "try:\n"
        "    import torch\n"
        "except ModuleNotFoundError:\n"
        "    pass                    # torch not installed here: nothing to check\n"
        "else:\n"
        "    assert torch.zeros(2).sum().item() == 0.0\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env={**__import__("os").environ, "SELLA_DISABLE_GPU": "1"},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]


def test_live_snapshots_have_no_graph_and_a_stopped_string_gets_them_back():
    reactant = StructureNode(structure=Structure.from_xyz(WATER))
    assert reactant.graph is not None
    coords = np.asarray(reactant.coords) + 0.05
    snap = _graphless_copy(reactant, coords)
    assert snap.graph is None and not snap.has_molecular_graph
    np.testing.assert_allclose(snap.coords, coords)
    np.testing.assert_allclose(reactant.coords, Structure.from_xyz(WATER).geometry)   # reactant untouched
    snap._cached_energy, snap._cached_gradient = -76.1, np.full((3, 3), 0.02)

    chain = Chain.model_validate({"nodes": [reactant, snap, reactant], "parameters": ChainInputs()})
    full = _with_graphs(chain, reactant)
    assert all(n.graph is not None and n.has_molecular_graph for n in full.nodes)
    assert full.nodes[0] is reactant
    np.testing.assert_allclose(full.nodes[1].coords, coords)
    assert full.nodes[1].energy == -76.1
    np.testing.assert_allclose(full.nodes[1].gradient, np.full((3, 3), 0.02))


def _water_string(shift):
    reactant = StructureNode(structure=Structure.from_xyz(WATER))
    nodes = []
    for i, e in enumerate([-76.0, -75.6 - shift, -76.05]):
        node = reactant.update_coords(np.asarray(reactant.coords) + 0.01 * i + shift)
        node._cached_energy = e
        node._cached_gradient = np.zeros((3, 3))
        nodes.append(node)
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def test_tree_read_for_ts_guesses_loads_only_first_and_last_string(tmp_path):
    from mepd.cli_common import _pair_ts_guess_tasks
    from mepd.inputs import NEBInputs
    from mepd.neb import NEB
    from mepd.optimizers.vpo import VelocityProjectedOptimizer
    from mepd.TreeNode import TreeNode

    history = [_water_string(0.001 * k) for k in range(30)]
    neb = NEB(initial_chain=history[0], optimizer=VelocityProjectedOptimizer(), parameters=NEBInputs(),
              engine=None, optimized=history[-1], chain_trajectory=history)
    tree_dir = tmp_path / "pair_0_1" / "tree"
    tree_dir.parent.mkdir(parents=True)
    TreeNode(data=neb, children=[], index=0).write_to_disk(tree_dir)

    full = TreeNode.read_from_disk(tree_dir, chain_parameters=ChainInputs())
    ends = TreeNode.read_from_disk(tree_dir, chain_parameters=ChainInputs(), ends_only=True)
    assert len(full.data.chain_trajectory) == 30 and len(ends.data.chain_trajectory) == 2
    for a, b in ((full.data.initial_chain, ends.data.initial_chain), (full.data.optimized, ends.data.optimized)):
        assert np.array_equal(a.coordinates, b.coordinates) and np.allclose(a.energies, b.energies)

    (label, guess), = _pair_ts_guess_tasks(tmp_path / "pair_0_1", 0, 1)
    want = full.data.chain_trajectory[-1].get_ts_node()
    assert label == "ts_pair_0_1_leaf_0"
    assert np.array_equal(guess.coords, want.coords) and guess.energy == want.energy
