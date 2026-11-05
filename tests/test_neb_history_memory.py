"""A long NEB keeps every step's string to write it out; only the last few keep their images' bond graphs
(~2/3 of a step's memory: a deep search's history took a worker past 14 GB)."""

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.neb import HISTORY_GRAPH_STEPS, _drop_graphs
from mepd.nodes.node import StructureNode

WATER = """3

O 0.0 0.0 0.0
H 0.0 0.757 0.587
H 0.0 -0.757 0.587
"""


def test_old_steps_drop_graphs_without_touching_a_shared_image():
    node = StructureNode(structure=Structure.from_xyz(WATER))
    other = node.update_coords(np.asarray(node.coords) + 0.01)
    old = Chain.model_validate({"nodes": [node, other], "parameters": ChainInputs()})
    new = Chain.model_validate({"nodes": [node, other.update_coords(other.coords + 0.01)],
                                "parameters": ChainInputs()})   # the frozen start is the same object
    _drop_graphs(old)
    assert all(n.graph is None for n in old.nodes)
    assert new.nodes[0] is node and node.graph is not None and new.nodes[1].graph is not None
    np.testing.assert_array_equal(old.nodes[1].coords, other.coords)
    assert HISTORY_GRAPH_STEPS >= 6   # rollback after an engine error reaches 5 steps back
