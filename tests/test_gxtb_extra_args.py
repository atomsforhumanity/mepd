"""g-xTB extra_args: one argument per word, and no implicit solvent (mepd#11)."""
import pytest

from mepd.conformers import CrestInputs, _build_crest_argv
from mepd.engines.gxtb import GXTBCalculator


def test_an_entry_with_spaces_becomes_separate_arguments():
    eng = GXTBCalculator(executable="xtb", add_gxtb_flag=False, extra_args=["--alpb water", "--etemp", 500])
    assert eng.extra_args == ["--alpb", "water", "--etemp", "500"]
    assert GXTBCalculator(executable="xtb", add_gxtb_flag=False, extra_args="--gbsa  acetone").extra_args == \
        ["--gbsa", "acetone"]


@pytest.mark.parametrize("args", [["--alpb water"], ["--alpb", "water"], ["--gbsa=thf"], ["--cosmo", "water"],
                                  ["--cpcmx", "water"], ["-g", "water"], ["--gbe", "water"]])
def test_gxtb_refuses_solvent_flags(args):
    with pytest.raises(ValueError, match="no implicit-solvent"):
        GXTBCalculator(executable="xtb", extra_args=args)


def test_gfn2_through_the_same_engine_keeps_its_solvent():
    assert GXTBCalculator(executable="xtb", add_gxtb_flag=False, extra_args=["--alpb", "water"]).extra_args == \
        ["--alpb", "water"]


def test_crest_extra_args_are_split_too():
    argv = _build_crest_argv(CrestInputs(extra_args=("--alpb water", "--quick")), "crest", "in.xyz", 0, 1)
    assert argv[-3:] == ["--alpb", "water", "--quick"]
