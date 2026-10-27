"""A TS search that slides off the reaction is said so (#25): a QM/MM TS whose
imaginary mode hardly overlaps the path tangent, or any TS whose IRC ends are
one species, gets a clear message and <label>_check_failed.txt."""
from types import SimpleNamespace

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.qmmm_path import reaction_mode_problems

A2B = 1.8897259886


def _node(h1=(0.958, 0.0, 0.0)):
    xyz = np.array([[0.0, 0.0, 0.0], h1, [-0.24, 0.93, 0.0]]) * A2B
    return StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=xyz, charge=0, multiplicity=1))


class _Micro:
    outer = [0, 1, 2]

    def __init__(self, w, mode):
        v = np.asarray(mode, dtype=float).ravel()
        self.w, self.V = np.asarray(w, dtype=float), np.column_stack([v / np.linalg.norm(v)] + [np.zeros(9)] * 8)

    def qm_modes(self, structure, x, echo=None):
        return self.w, self.V


def _engine(micro):
    return SimpleNamespace(_micro=lambda: micro)


def test_a_mode_along_the_tangent_passes_and_one_across_it_is_flagged():
    tangent = np.zeros((3, 3))
    tangent[1, 0] = 1.0                                   # H1 leaves along x
    along = tangent.copy()
    across = np.zeros((3, 3))
    across[2, 2] = 1.0                                    # H2 turns out of plane
    ts = _node()
    assert reaction_mode_problems(ts, _engine(_Micro([-0.5] + [1.0] * 8, along)), tangent) == []
    (msg,) = reaction_mode_problems(ts, _engine(_Micro([-0.5] + [1.0] * 8, across)), tangent)
    assert "overlaps the path tangent only 0.00" in msg and "atoms 2" in msg
    (msg,) = reaction_mode_problems(ts, _engine(_Micro([0.1] * 9, along)), tangent)
    assert "no negative curvature" in msg
    # Nothing to compare: no microiterations, or no tangent.
    assert reaction_mode_problems(ts, _engine(None), tangent) == []
    assert reaction_mode_problems(ts, _engine(_Micro([0.1] * 9, along)), None) == []


def test_irc_ends_of_one_species_are_reported(monkeypatch, tmp_path):
    from mepd.cli_common import _optimize_ts_and_irc

    ts = _node(h1=(1.3, 0.0, 0.0))
    for name, far_end, flagged in (("same", _node(h1=(0.97, 0.0, 0.0)), True),
                                   ("differ", _node(h1=(2.8, 0.0, 0.0)), False)):
        path = Chain.model_validate({"nodes": [_node(), ts, far_end], "parameters": ChainInputs()})
        for n, e in zip(path.nodes, (-76.4, -76.3, -76.4)):
            n._cached_energy = e
        monkeypatch.setattr("mepd.compute.find_ts", lambda guess, engine, **kw: ts)
        monkeypatch.setattr("mepd.compute.irc", lambda ts, engine, **kw: SimpleNamespace(path=path))
        engine = SimpleNamespace(compute_transition_state=lambda **kw: ts)
        out = _optimize_ts_and_irc(ts, SimpleNamespace(engine=engine, ts_inputs=None), tmp_path / name,
                                   run_irc=True, label="ts_leaf_1")
        check = tmp_path / name / "ts_leaf_1_check_failed.txt"
        assert bool(out.problems) is flagged and check.exists() is flagged
        if flagged:
            assert "same species" in check.read_text()
