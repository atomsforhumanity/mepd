"""The expansion's and the mixture's Nanoreactor card: one MD choice whose forces (piston, collide) are switched
on and off, sent as `--generator nanoreactor --generator-option forces=...`; the older Reactor MD and Collision MD
cards open as it; the Reactor view gets the forces and the pulls."""
import json

import pytest
from pydantic import ValidationError

from mepd.web.operations import OPERATIONS, requirement_met
from mepd.web.workspace import WorkspaceError


class Ctx:
    output_dir = "/out"
    structures = [{"id": "s_a"}]
    jobs = {}

    def snapshot_structure(self, rec, name):
        return f"/in/{name}.xyz"

    def common_flags(self):
        return ["--charge", "0", "--multiplicity", "1"]

    def level(self):
        return {"key": "k"}


def _options(argv):
    return [argv[i + 1] for i, w in enumerate(argv) if w == "--generator-option"]


def _expand(**params):
    op = OPERATIONS["graph-enumeration"]
    return op.build(Ctx(), op.parse_params({"generator": "md", **params}))


def test_the_nanoreactor_always_names_its_forces_and_only_their_settings():
    argv = _expand()
    assert argv[argv.index("--generator") + 1] == "nanoreactor"
    assert _options(argv) == ["forces=piston,collide", "n_seeds=0", "max_batches=4", "time_ps=5", "compress=0.7"]
    # the CLI's order, whatever the order they were switched on in
    assert _options(_expand(md_forces=["collide", "piston"]))[0] == "forces=piston,collide"
    assert _options(_expand(md_forces=["piston"], md_temperature=2500))[:2] == ["forces=piston", "temperature=2500"]
    # collisions alone: their own loop, none of the piston's settings
    assert _options(_expand(md_forces=["collide"])) == ["forces=collide"]
    assert _options(_expand(md_forces=["collide"], md_collisions=12)) == ["forces=collide", "n_collisions=12"]
    with pytest.raises(ValidationError):
        _expand(md_forces=[])
    # QM/MM forces: the piston only
    assert _options(_expand(md_forces=["piston"], md_embedded=True))[-1] == "embedded=true"
    with pytest.raises(WorkspaceError):
        _expand(md_embedded=True)


def test_older_cards_still_run_as_before_and_open_as_the_nanoreactor():
    op = OPERATIONS["graph-enumeration"]
    argv = op.build(Ctx(), op.parse_params({"generator": "reactor"}))
    assert argv[argv.index("--generator") + 1] == "reactor" and "forces" not in " ".join(argv)
    cards = op.params_model.model_json_schema()["properties"]["generator"]["cards"]
    assert cards["md"]["title"] == "Nanoreactor" and not cards["md"].get("hidden")
    assert cards["reactor"]["hidden"] and cards["reactor"]["moved_to"] == {"generator": "md", "md_forces": ["piston", "collide"]}
    assert cards["collide"]["moved_to"] == {"generator": "md", "md_forces": ["collide"]}
    assert cards["nanoreactor"]["hidden"] and "moved_to" not in cards["nanoreactor"]   # the xtb program's: not the same runs
    props = op.params_model.model_json_schema()["properties"]
    assert props["md_forces"]["minItems"] == 1 and props["md_forces"]["labels"]["piston"].startswith("Piston")
    assert props["md_embedded"]["qmmm_only"]


def test_a_field_can_require_several_things_and_a_list():
    p = OPERATIONS["graph-enumeration"].parse_params({"generator": "md", "md_forces": ["collide", "piston"]})
    assert requirement_met(p, "generator=md;md_forces=piston|collide,piston")
    assert not requirement_met(p, "generator=md;md_forces=collide")
    assert not requirement_met(p, "generator=bond-rules;md_forces=collide,piston")
    assert requirement_met(p, "md_forces")


def test_the_mixture_grows_by_the_nanoreactor_with_collisions_by_default():
    op = OPERATIONS["network-mixture"]
    ctx = Ctx()
    ctx.structures = [{"id": "s_a"}, {"id": "s_b"}]
    argv = op.build(ctx, op.parse_params({}))
    grow = next(w for w in argv if w.startswith("--expand-args="))
    assert grow.startswith("--expand-args=--generator nanoreactor --generator-option forces=collide")
    argv = op.build(ctx, op.parse_params({"md_forces": ["piston", "collide"], "md_temperature": 1800}))
    grow = next(w for w in argv if w.startswith("--expand-args="))
    assert "forces=piston,collide --generator-option temperature=1800" in grow
    argv = op.build(ctx, op.parse_params({"grow_by": "bond-rules"}))
    assert next(w for w in argv if w.startswith("--expand-args=")).startswith("--expand-args=--generator bond-rules")


def test_the_reactor_view_has_the_forces_and_pulls_of_the_run(tmp_path):
    from mepd.web.nanoreactor import reactor_view

    md = tmp_path / "md"
    md.mkdir()
    (md / "segment_000.xyz").write_text("".join(f"2\nt={k}\nH 0 0 {0.1 * k}\nH 0 0 {2 - 0.1 * k}\n" for k in range(5)))
    pull = {"t0_fs": 0.0, "t1_fs": 6.0, "atoms": [0, 1], "molecules": None, "push": None}
    (md / "schedule.json").write_text(json.dumps({"dump_fs": 2.0, "time_ps": 0.01, "seed": 0,
                                                  "segments": [[0.01, 5.0]], "forces": ["collide"], "pulls": [pull]}))
    d = reactor_view(tmp_path)
    assert d["forces"] == ["collide"] and d["pulls"] == [pull] and d["n_frames"] == 5
    (md / "schedule.json").write_text(json.dumps({"dump_fs": 2.0, "time_ps": 0.01, "segments": [[0.01, 5.0]]}))
    assert reactor_view(tmp_path)["pulls"] == [] and reactor_view(tmp_path)["forces"] is None


def test_the_nanoreactor_tool_is_merged_into_the_expansions_grow():
    """The standalone Nanoreactor is no longer its own tool: the expansion grows by the Nanoreactor by default
    (several molecules: the Reacting mixture). The operation stays for older jobs and the Reactor's Stop & analyze."""
    nano = OPERATIONS["nanoreactor"]
    assert nano.hidden and nano.describe()["hidden"]
    assert OPERATIONS["graph-enumeration"].parse_params({}).generator == "md"
    assert OPERATIONS["network-mixture"].parse_params({}).grow_by == "md"
