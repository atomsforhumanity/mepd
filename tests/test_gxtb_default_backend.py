"""The g-xTB engine's backend when a profile names none: axtb, unless the settings need the program."""
import pytest

from mepd.engines import axtb_backend
from mepd.engines.gxtb import GXTBCalculator

needs_axtb = pytest.mark.skipif(axtb_backend.version() is None, reason="axtb not installed")


@needs_axtb
def test_unset_backend_is_axtb_with_its_default_accuracy():
    eng = GXTBCalculator()
    assert eng.backend == "axtb"
    assert eng._axtb_settings["accuracy"] == axtb_backend.ACCURACY


@needs_axtb
def test_unset_backend_takes_the_xtb_options_axtb_takes():
    assert GXTBCalculator(extra_args="--etemp 1000 --acc 0.1").backend == "axtb"


@pytest.mark.parametrize("kw", [
    {"add_gxtb_flag": False},                 # GFN-xTB: axtb computes g-xTB only
    {"minimizer": "xtb"},                     # the program's own optimizer
    {"extra_args": "--gfn 2"},                # an option axtb would not apply
    {"executable": "/opt/gxtb/bin/gxtb"},     # a program named explicitly
])
def test_unset_backend_falls_back_to_the_program(kw):
    assert GXTBCalculator(**kw).backend == "xtb"


def test_a_named_backend_is_kept():
    assert GXTBCalculator(backend="xtb").backend == "xtb"
    with pytest.raises(ValueError):
        GXTBCalculator(backend="axtb", add_gxtb_flag=False)


def test_unset_backend_without_axtb_is_the_program(monkeypatch):
    monkeypatch.setattr(axtb_backend, "version", lambda: None)
    assert GXTBCalculator().backend == "xtb"
