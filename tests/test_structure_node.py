import numpy as np
from qcdata import Structure

from mepd.nodes.node import StructureNode


def test_update_coords_rebuilds_molecular_graph():
    node = StructureNode(
        structure=Structure(
            symbols=["H", "H"],
            geometry=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
            charge=0,
            multiplicity=1,
        )
    )

    updated = node.update_coords(np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=float))

    assert node.graph.number_of_edges() == 1
    assert updated.graph.number_of_edges() == 0


def _h2(x=1.4):
    return StructureNode(structure=Structure(
        symbols=["H", "H"], geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]], dtype=float),
        charge=0, multiplicity=1))


def test_graph_is_built_on_first_read_and_matches_eager_perception(monkeypatch):
    import mepd.nodes.node as node_module
    from mepd.qcdata_structure_helpers import structure_to_molecule

    built = []
    monkeypatch.setattr(node_module, "structure_to_molecule",
                        lambda s: built.append(1) or structure_to_molecule(s))
    node = _h2()
    moved = [node.update_coords(np.array([[0.0, 0.0, 0.0], [1.4 + 0.01 * i, 0.0, 0.0]])) for i in range(5)]
    assert built == []                                   # no node was asked for its graph
    graph = moved[2].graph
    assert len(built) == 1 and moved[2].graph is graph   # built once, then kept
    assert sorted(graph.edges) == sorted(structure_to_molecule(moved[2].structure).edges)
    assert moved[2].copy().graph is graph                # a copy keeps the built graph


def test_deferred_graph_survives_copy_pickle_and_serialization():
    import pickle

    node = _h2().update_coords(np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
    assert node.copy().graph.number_of_edges() == 0
    assert pickle.loads(pickle.dumps(node)).graph.number_of_edges() == 0
    data = _h2().update_coords(np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])).to_serializable()
    assert StructureNode.from_serializable(data).graph.number_of_edges() == 1


def test_graphless_nodes_and_qmmm_subsets_are_unchanged():
    node = _h2()
    node.has_molecular_graph = False
    node.graph = None
    assert node.update_coords(node.coords).graph is None
    three = StructureNode(structure=Structure(
        symbols=["H", "H", "He"], geometry=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        charge=0, multiplicity=1))
    with StructureNode.graph_atoms_in_this_thread([0, 1], 3):
        sub = three.update_coords(three.coords)
        assert "graph" in vars(sub)                       # a subset graph is built at once, as before
    assert sub.graph_atom_indices_source == "qmmm_qm_atoms" and sub.graph.number_of_nodes() == 2
