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
