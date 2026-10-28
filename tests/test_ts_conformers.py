"""Conformers of a transition state (#15): CREST samples around the reacting
core, held at the TS distances, and mepd checks what comes back."""
from __future__ import annotations

import json

import numpy as np
import pytest
import typer
from qcdata.models.structure import Structure
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.conformers import (ConformerInputs, CrestInputs, constraint_file, generate_conformers,
                             ts_reactive_core)
from mepd.nodes.node import StructureNode
from tests.test_conformers import _butane_node, _frames_xyz


def test_reactive_core_is_the_pairs_the_imaginary_mode_stretches():
    # A proton halfway between two waters' oxygens: O0-H1...O2, each O with
    # its own H. The mode moves the proton from O0 to O2.
    ang = np.array([[0.0, 0, 0], [1.2, 0, 0], [2.4, 0, 0], [-0.3, 0.9, 0], [2.7, 0.9, 0]])
    s = Structure(symbols=["O", "H", "O", "H", "H"], geometry=ang * ANGSTROM_TO_BOHR, charge=1, multiplicity=1)
    mode = np.zeros((5, 3))
    mode[1] = [1.0, 0, 0]
    mode[[0, 3]] = [-0.05, 0, 0]
    mode[[2, 4]] = [-0.05, 0, 0]
    assert ts_reactive_core(s, mode) == [(0, 1), (1, 2)]


def test_constraint_file_is_one_based_and_may_leave_the_force_constant_out():
    assert constraint_file([(0, 12), (3, 11)], 1.0) == (
        "$constrain\n force constant=1\n distance: 1, 13, auto\n distance: 4, 12, auto\n$end\n")
    assert "force constant" not in constraint_file([(0, 1)], None)


def _fake_crest(tmp_path, frames_text: str, fail_on_force_constant: bool = False):
    """Writes `frames_text` as the ensemble; keeps the argv and constrain.inp
    it got. Optionally fails as some CREST 3.0.2 builds do on a force constant."""
    (tmp_path / "ensemble.xyz").write_text(frames_text)
    fail = ('if grep -q "force constant" constrain.inp; then echo "Error allocating 2188448768777856 bytes '
            'parse_xtbinput.f90" >&2; exit 1; fi\n') if fail_on_force_constant else ""
    script = tmp_path / "crest"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" > "{tmp_path}/argv.txt"\n'
        f'cp constrain.inp "{tmp_path}/constrain.inp" 2>/dev/null\n'
        + fail +
        f'cp "{tmp_path}/ensemble.xyz" crest_conformers.xyz\n')
    script.chmod(0o755)
    return script


def _pentane_frames():
    """Pentane (C1..C5 = atoms 0..4): the start, a copy moved as a whole
    (a conformer that held C1-C5), a conformer with another C1-C5 distance,
    and the start with an H pulled off (a bond broke)."""
    start = _butane_node(smiles="CCCCC")
    x = np.asarray(start.structure.geometry)
    moved = StructureNode(structure=start.structure.model_copy(update={"geometry": x + 0.7}))
    folded = x.copy()
    folded[4] += (folded[4] - folded[0]) * 0.3
    torn = x.copy()
    torn[5] += (torn[5] - torn[0]) * 3.0
    as_node = lambda g: StructureNode(structure=start.structure.model_copy(update={"geometry": g}))  # noqa: E731
    return start, _frames_xyz([moved, as_node(folded), as_node(torn)], energies=[-10.0, -10.5, -11.0])


def test_constrained_crest_holds_the_core_and_drops_what_moved_or_broke(tmp_path):
    start, frames = _pentane_frames()
    fake = _fake_crest(tmp_path, frames)
    stats: dict = {}
    confs = generate_conformers(start, ConformerInputs(
        backend="crest", crest=CrestInputs(executable=str(fake), constrain_pairs=((0, 4),))), stats)

    argv = (tmp_path / "argv.txt").read_text().split()
    assert argv[argv.index("--cinp") + 1] == "constrain.inp" and "--noreftopo" in argv
    # CREST takes a word after --notopo as atoms to exclude and checks topology again.
    assert argv[argv.index("--notopo") + 1].startswith("-")
    assert (tmp_path / "constrain.inp").read_text() == constraint_file([(0, 4)], 1.0)
    assert stats["n_dropped_constraint_moved"] == 1 and stats["n_dropped_bonds_changed"] == 1
    assert stats["crest_force_constant"] == 1.0 and stats["crest_constrained_pairs"] == [[0, 4]]
    assert stats["n_in_window"] == 1    # the copy that held C1-C5 (a duplicate of the start, so not kept)
    assert len(confs) == 1


def test_unconstrained_crest_keeps_its_topology_checks(tmp_path):
    start, frames = _pentane_frames()
    fake = _fake_crest(tmp_path, frames)
    generate_conformers(start, ConformerInputs(backend="crest", crest=CrestInputs(executable=str(fake))))
    argv = (tmp_path / "argv.txt").read_text().split()
    assert not {"--cinp", "--noreftopo", "--notopo"} & set(argv)


def test_a_build_that_fails_on_the_force_constant_runs_with_the_default(tmp_path):
    start, frames = _pentane_frames()
    fake = _fake_crest(tmp_path, frames, fail_on_force_constant=True)
    stats: dict = {}
    generate_conformers(start, ConformerInputs(
        backend="crest", crest=CrestInputs(executable=str(fake), constrain_pairs=((0, 4),))), stats)
    assert "force constant" not in (tmp_path / "constrain.inp").read_text()
    assert stats["crest_force_constant"] == 0.05


def test_constrain_pairs_outside_the_structure_are_refused(tmp_path):
    from mepd.conformers import CrestError

    start, frames = _pentane_frames()
    with pytest.raises(CrestError, match="0-based"):
        generate_conformers(start, ConformerInputs(backend="crest", crest=CrestInputs(
            executable=str(_fake_crest(tmp_path, frames)), constrain_pairs=((0, 99),))))


def test_cli_ts_samples_with_crest_around_the_given_core(tmp_path, monkeypatch):
    from mepd.cli_conformers import conformers, parse_pairs

    assert parse_pairs("0-12, 3:11") == [(0, 12), (3, 11)]
    start, frames = _pentane_frames()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_crest(bin_dir, frames)
    monkeypatch.setenv("PATH", f"{bin_dir}:" + __import__("os").environ["PATH"])
    xyz = tmp_path / "ts.xyz"
    xyz.write_text(start.structure.to_xyz())
    common = dict(inputs=None, charge=None, multiplicity=None, n_conformers=20, rmsd_cutoff=0.1,
                  crest_method="--gfn2", crest_ewin=6.0, minimize=False, validate_minima_with_hessian=False)
    with pytest.raises(typer.BadParameter, match="--ts"):
        conformers(str(xyz), backend=None, ts=False, freeze="0-4", output=tmp_path / "a", **common)
    with pytest.raises(typer.BadParameter, match="CREST"):
        conformers(str(xyz), backend="rdkit", ts=True, freeze="0-4", output=tmp_path / "b", **common)
    conformers(str(xyz), backend=None, ts=True, freeze="0-4", output=tmp_path / "out", **common)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["backend"] == "crest" and summary["ts"] and summary["frozen_pairs"] == [[0, 4]]
    assert "--notopo" in (bin_dir / "argv.txt").read_text()


def test_web_ts_conformers_run_with_crest_and_join_the_ts(tmp_path, monkeypatch):
    from mepd.web import chem, operations
    from mepd.web.app import attach_conformers
    from mepd.web.operations import ConformersParams, JobContext, WorkspaceError, _build_conformers
    from mepd.web.results import collect_conformers
    from mepd.web.workspace import Workspace

    start, _ = _pentane_frames()
    ws = Workspace(tmp_path / "ws")
    s = chem.structures_from_xyz_text(start.structure.to_xyz(), 0, 1)[0]
    rec = ws.add_structure(s, origin={"kind": "xyz"}, role="ts", merge=False)
    ctx = JobContext(ws, tmp_path / "job", tmp_path / "job" / "output", [ws.structure_view(rec["id"])], None)
    with pytest.raises(WorkspaceError, match="CREST"):
        _build_conformers(ctx, ConformersParams(backend="rdkit"))
    monkeypatch.setattr(operations.shutil, "which", lambda name: None)
    with pytest.raises(WorkspaceError, match="CREST isn't installed"):
        _build_conformers(ctx, ConformersParams())        # automatic: CREST for a TS
    monkeypatch.setattr(operations.shutil, "which", lambda name: "/usr/bin/" + name)
    assert "--ts" in _build_conformers(ctx, ConformersParams())
    assert "--ts" in _build_conformers(ctx, ConformersParams(backend="crest"))

    # A finished `mepd conformers --ts --minimize`: two TS conformers, one
    # optimized onto another TS (its core moved).
    out = tmp_path / "job" / "output"
    (out / "optimized").mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"backend": "crest", "ts": True, "frozen_pairs": [[0, 4]],
                                                  "stats": {"n_generated": 3}}))
    x = np.asarray(start.structure.geometry)
    rows = []
    for k, (shift, e, same) in enumerate([(0.0, -20.0, True), (0.3, -20.5, False)]):
        node = start.structure.model_copy(update={"geometry": x + shift})
        (out / "optimized" / f"ts_{k}.xyz").write_text(node.to_xyz().replace("\n\n", f"\n{e}\n", 1))
        np.savetxt(out / "optimized" / f"ts_{k}.energies", [e])
        rows.append({"label": f"ts_{k}", "status": "ok", "energy_hartree": e, "core_change_ang": 0.0 if same else 0.5,
                     "same_core": same})
    (out / "optimized" / "summary.json").write_text(json.dumps({"frozen_pairs": [[0, 4]], "ts": rows}))
    result = collect_conformers(out, 0, 1)
    (group,) = result["groups"]
    assert group["kind"] == "ts" and len(group["entries"]) == 2
    job = {"id": "j1", "op": "conformers", "targets": {"structures": [rec["id"]], "edges": []},
           "params": {"minimize": True, "backend": "crest"}, "level": {"key": "k1", "label": "gxtb"},
           "charge": 0, "multiplicity": 1}
    assert attach_conformers(ws, job, result) == 1   # the one whose core moved is not this TS's


def test_crest_samples_in_the_profiles_implicit_solvent(tmp_path):
    from mepd.cli_conformers import _crest_solvent_args

    toml = tmp_path / "p.toml"
    toml.write_text('engine_name = "gxtb"\n[solvation]\nsolvent = "thf"\nmodel = "alpb"\n')
    assert _crest_solvent_args(toml) == ("--alpb", "thf")
    toml.write_text('engine_name = "gxtb"\n[solvation]\nsolvent = "thf"\nmodel = "gbsa"\n')
    assert _crest_solvent_args(toml) == ("--gbsa", "thf")
    toml.write_text('engine_name = "gxtb"\n[solvation]\nsolvent = "thf"\nmodel = "cpcmx"\n')
    with pytest.raises(typer.BadParameter, match="no gradient"):   # no such profile (CPCM-X: single points only)
        _crest_solvent_args(toml)
    assert _crest_solvent_args(None) == ()


def test_ts_conformers_that_are_mirror_images_are_marked(tmp_path, monkeypatch):
    """Enantiomeric TS conformers (same energy, RMSD large until reflected):
    marked as each other's mirror image, kept as distinct saddle points."""
    from types import SimpleNamespace

    from mepd import cli_conformers

    start = _butane_node(smiles="CC(F)CCl")           # a chiral, non-planar conformer
    x = np.asarray(start.structure.geometry)
    as_node = lambda g: StructureNode(structure=start.structure.model_copy(update={"geometry": g}))  # noqa: E731
    found = [as_node(x), as_node(x + 0.5), as_node(x * [1, 1, -1])]   # itself, moved, its mirror image
    for n in found:
        n._cached_energy = -10.0
    monkeypatch.setattr("mepd.cli_common._open_run_inputs", lambda inputs: SimpleNamespace())
    monkeypatch.setattr("mepd.cli_common._optimize_ts_and_irc",
                        lambda node, ri, out, run_irc, label: SimpleNamespace(ts_node=node))
    cli_conformers._optimize_ts_conformers(found, [(0, 1)], None, 0.1, tmp_path)
    rows = json.loads((tmp_path / "summary.json").read_text())["ts"]
    assert rows[1]["same_ts_as"] == "ts_0"
    assert rows[2].get("mirror_of") == "ts_0" and "same_ts_as" not in rows[2]
