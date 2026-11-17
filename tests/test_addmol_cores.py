"""`mepd addmol test --cores N`: the scan lanes and the estimate and verification pool share at most N CPUs."""
import os

import pytest

from mepd import cli_addmol


def test_the_cpu_set_is_bounded_by_cores():
    every = cli_addmol._cpu_set()
    assert every == sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else every
    assert cli_addmol._cpu_set(1) == every[:1]
    assert cli_addmol._cpu_set(10 ** 6) == every          # (more than there are: all of them)


def test_the_option_is_there():
    from typer.testing import CliRunner

    from mepd.cli import app

    out = CliRunner().invoke(app, ["addmol", "test", "--help"], terminal_width=200)
    assert out.exit_code == 0 and "--cores" in out.output
    bad = CliRunner().invoke(app, ["addmol", "test", __file__, "--cores", "0", "-o", "x"])
    assert bad.exit_code != 0


def test_result_files_are_one_per_candidate_and_never_the_record(tmp_path):
    import json

    long = "x" * 90
    f = tmp_path / "c.json"
    f.write_text(json.dumps([{"name": "provenance", "smiles": "O", "amount": 1},
                             {"name": long + "a", "smiles": "N", "amount": 1}, {"name": long + "b", "smiles": "C", "amount": 1}]))
    out = cli_addmol._candidates([], f, None)
    files = [c["_file"] for c in out]
    assert "provenance.json" not in files and len(set(files)) == 3
    # two SMILES that read alike once made a file name
    two = cli_addmol._candidates(["C/C=C/C", "C/C=C\\\\C"], None, 1.0)
    assert len({c["_file"] for c in two}) == 2
