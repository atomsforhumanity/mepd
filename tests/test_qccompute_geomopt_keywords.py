from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure
from qcdata.models.inputs import FileInput

from mepd.engines.qccompute import QCComputeEngine
from mepd.program_args import ProgramArgs
from mepd.nodes.node import StructureNode


def _program_of(inp_obj):
    """Read the target program back off an input model (qcdata >=0.19)."""
    first = inp_obj[0] if isinstance(inp_obj, list) else inp_obj
    return first.program


def _node_at_x(x: float) -> StructureNode:
    struct = Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]], dtype=float),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=struct)


def test_terachem_single_geomopt_preserves_program_keywords():
    engine = QCComputeEngine(
        program="terachem",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 7, "precision": "mixed"},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    assert captured["program"] == "terachem"
    kw = captured["input"].keywords
    assert kw["threads"] == 7
    assert kw["precision"] == "mixed"
    assert kw["purify"] == "no"
    assert kw["new_minimizer"] == "yes"


def test_terachem_batch_geomopt_preserves_program_keywords():
    engine = QCComputeEngine(
        program="terachem",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 3, "gpus": 1},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["inputs"] = inp_obj
        return [SimpleNamespace(), SimpleNamespace()]

    engine.compute_func = _fake_compute_func
    _ = engine.compute_geometry_optimizations([_node_at_x(1.0), _node_at_x(1.2)])

    assert captured["program"] == "terachem"
    assert isinstance(captured["inputs"], list)
    assert len(captured["inputs"]) == 2
    for prog_input in captured["inputs"]:
        kw = prog_input.keywords
        assert kw["threads"] == 3
        assert kw["gpus"] == 1
        assert kw["purify"] == "no"
        assert kw["new_minimizer"] == "yes"


def test_terachem_single_geomopt_with_frozen_atoms_uses_constraints_file_input():
    engine = QCComputeEngine(
        program="terachem",
        compute_program="chemcloud",
        frozen_atom_indices=[0],
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 7},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["input"] = inp_obj
        captured["collect_files"] = kwargs.get("collect_files")
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0), keywords={"maxiter": 10})

    assert captured["program"] == "terachem"
    assert captured["collect_files"] is True
    assert isinstance(captured["input"], FileInput)
    tcin = captured["input"].files["tc.in"]
    assert "run minimize" in tcin
    assert "threads 7" in tcin
    assert "maxit 10" in tcin
    # new_minimizer (mepd's default) reads $constraint_freeze, not $constraints (mepd#3)
    assert "new_minimizer yes" in tcin
    assert "$constraints" not in tcin
    assert tcin.rstrip().endswith("end\n\n$constraint_freeze\nxyz 1\n$end")
    assert "frozen_atom_indices" not in tcin


def test_terachem_batch_geomopt_with_frozen_atoms_uses_constraints_file_inputs():
    engine = QCComputeEngine(
        program="terachem",
        compute_program="chemcloud",
        frozen_atom_indices=[1],
        program_args=ProgramArgs(
            model={"method": "b3lyp", "basis": "6-31g*"},
            keywords={},
        ),
    )
    captured = {}

    class _Output:
        files = {
            "optim.xyz": (
                "2\n"
                "frame0\n"
                "H 0.000000 0.000000 0.000000\n"
                "H 1.000000 0.000000 0.000000\n"
            )
        }

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["inputs"] = inp_obj
        captured["collect_files"] = kwargs.get("collect_files")
        return [_Output(), _Output()]

    engine.compute_func = _fake_compute_func
    _ = engine.compute_geometry_optimizations([_node_at_x(1.0), _node_at_x(1.2)])

    assert captured["program"] == "terachem"
    assert captured["collect_files"] is True
    assert isinstance(captured["inputs"], list)
    assert len(captured["inputs"]) == 2
    for inp in captured["inputs"]:
        assert isinstance(inp, FileInput)
        tcin = inp.files["tc.in"]
        assert "$constraints" not in tcin
        assert "$constraint_freeze\nxyz 2\n$end" in tcin


def test_qccompute_prepare_node_for_comparison_uses_non_frozen_atoms():
    engine = QCComputeEngine(frozen_atom_indices=[1])
    node = _node_at_x(1.0)

    prepared = engine.prepare_node_for_comparison(node)

    assert prepared.comparison_atom_indices == [0]
    assert prepared.graph_atom_indices_source == "qccompute_non_frozen_atoms"
    assert prepared.graph_subset_atom_count == 1
    assert prepared.graph_total_atom_count == 2
    assert prepared.disable_smiles is True


def test_non_terachem_geomopt_default_keywords_use_coordsys():
    engine = QCComputeEngine(
        program="xtb",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    assert captured["program"] == engine.geometry_optimizer
    kw = captured["input"].keywords
    assert kw["coordsys"] == "cart"
    assert kw["maxit"] == 500
    assert kw["convergence_set"] == "GAU_TIGHT"


def test_non_terachem_geomopt_uses_engine_geometry_optimizer_keywords():
    engine = QCComputeEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={
            "coordsys": "tric",
            "maxit": 120,
            "convergence_set": "GAU_VERYTIGHT",
            "convergence_energy": 1e-6,
        },
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    kw = captured["input"].keywords
    assert kw["coordsys"] == "tric"
    assert kw["maxit"] == 120
    assert kw["convergence_set"] == "GAU_VERYTIGHT"
    assert kw["convergence_energy"] == pytest.approx(1e-6)


def test_non_terachem_geomopt_call_keywords_override_engine_defaults():
    engine = QCComputeEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={"coordsys": "tric", "maxit": 120},
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0), keywords={"maxit": 25})

    kw = captured["input"].keywords
    assert kw["coordsys"] == "tric"
    assert kw["maxit"] == 25


def test_non_terachem_geomopt_partial_engine_keywords_keep_defaults():
    engine = QCComputeEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={"maxit": 80},
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    kw = captured["input"].keywords
    assert kw["coordsys"] == "cart"
    assert kw["maxit"] == 80
    assert kw["convergence_set"] == "GAU_TIGHT"


@pytest.mark.parametrize("compute_program, expected", [("qccompute", "crest"), ("chemcloud", "bigchem")])
def test_hessian_uses_bigchem_only_on_chemcloud(compute_program, expected, monkeypatch):
    """A local engine must compute its own Hessian, not ask for a ChemCloud
    login (which blocked `--validate-minima-with-hessian` on a prompt)."""
    engine = QCComputeEngine(program="crest", compute_program="qccompute",
                             program_args=ProgramArgs(model={"method": "gfn2"}))
    engine.compute_program = compute_program
    seen = []
    monkeypatch.setattr(engine, "compute_func", lambda inp, **kw: seen.append(_program_of(inp)) or SimpleNamespace())
    monkeypatch.setattr(engine, "_chemcloud_compute_with_retries",
                        lambda prog, inp, **kw: seen.append(prog) or SimpleNamespace(results=None))
    engine._compute_hessian_result(_node_at_x(1.4))
    assert seen[0] == expected


def test_terachem_frozen_atoms_follow_the_optimizer():
    """$constraint_freeze/xyz for new_minimizer, $constraints/atom for the old optimizer."""
    engine = QCComputeEngine(program="terachem", compute_program="chemcloud",
                             program_args=ProgramArgs(model={"method": "hf", "basis": "sto-3g"}, keywords={}))
    build = lambda tc: engine._build_terachem_geomopt_file_input(
        _node_at_x(1.0), tc_keywords=tc, optimizer_keywords={}, frozen_atom_indices=[2, 0, 0, -1]).files["tc.in"]
    new = build({"new_minimizer": "yes"})
    assert "$constraint_freeze\nxyz 1,3\n$end" in new and "$constraints" not in new
    assert new.index("\nend\n") < new.index("$constraint_freeze")
    old = build({"new_minimizer": False})
    assert "$constraints\natom 1\natom 3\n$end" in old and "$constraint_freeze" not in old
    assert "$constraint" not in engine._build_terachem_geomopt_file_input(
        _node_at_x(1.0), tc_keywords={"new_minimizer": "yes"}, optimizer_keywords={},
        frozen_atom_indices=[]).files["tc.in"]


def _captured_ts_keywords(keywords):
    engine = QCComputeEngine(program="xtb", compute_program="qccompute",
                             program_args=ProgramArgs(model={"method": "gfn2"}, keywords={}))
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["input"] = inp_obj
        return SimpleNamespace(success=False)

    engine.compute_func = _fake_compute_func
    engine.compute_transition_state(_node_at_x(1.0), keywords=keywords)
    return captured["input"].keywords


def test_ts_keywords_reach_geometric():
    """compute_transition_state passes its keywords on (mepd#5): FSM-TS's maxiter and fmax."""
    kw = _captured_ts_keywords({"maxiter": 37, "fmax": 0.001})
    assert kw["maxiter"] == 37 and "fmax" not in kw
    # 0.001 eV/Å as a gradient in Hartree/bohr
    assert kw["convergence_gmax"] == pytest.approx(0.001 * 0.529177210903 / 27.211386245988)
    assert kw["convergence_gmax"] == pytest.approx(1.9447e-5, rel=1e-4)


def test_ts_keywords_defaults_and_pass_through():
    assert _captured_ts_keywords(None) == {"maxiter": 1000}
    assert _captured_ts_keywords({"steps": 5, "convergence_set": "GAU_TIGHT"}) == {
        "maxiter": 5, "convergence_set": "GAU_TIGHT"}
    given = {"fmax": 0.05, "convergence_gmax": 1e-6}
    assert _captured_ts_keywords(given)["convergence_gmax"] == 1e-6      # an explicit geomeTRIC value wins
    assert given == {"fmax": 0.05, "convergence_gmax": 1e-6}             # the caller's dict is not changed
