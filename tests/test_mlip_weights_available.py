"""Gated models (UMA, eSEN) count as usable once their weights are on the
machine -- e.g. baked into the container image -- in the demo too."""
from __future__ import annotations

import importlib.util

import pytest

from mepd.engines.mlip import weights_available

pytestmark = pytest.mark.skipif(importlib.util.find_spec("fairchem") is None, reason="needs fairchem-core")

UMA = "uma-s-1p2p1"
PROFILE = f'engine_name = "mlip"\n[mlip_engine_kwds]\nmodel = "{UMA}"\n'


def _bake(cache):
    """fairchem's Hugging Face cache layout, as the image's `uma` stage leaves it."""
    repo = cache / "models--facebook--UMA"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")
    ckpt = repo / "snapshots" / "abc123" / "checkpoints" / f"{UMA}.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")


def test_open_models_are_available_and_gated_ones_only_once_their_weights_are_here(tmp_path, monkeypatch):
    monkeypatch.setenv("FAIRCHEM_CACHE_DIR", str(tmp_path))
    assert weights_available("aimnet2")
    assert not weights_available(UMA)
    assert not weights_available("no-such-model")
    _bake(tmp_path)
    assert weights_available(UMA)
    assert not weights_available("uma-s-1p1")                    # gated, and not the one baked in


def test_the_demo_runs_a_gated_model_whose_weights_are_baked_in(tmp_path, monkeypatch):
    from mepd.web.demo import DemoPolicy
    from mepd.web.workspace import WorkspaceError

    monkeypatch.setenv("FAIRCHEM_CACHE_DIR", str(tmp_path))
    with pytest.raises(WorkspaceError, match="runs only these models"):
        DemoPolicy().check_profile(PROFILE)
    _bake(tmp_path)
    DemoPolicy().check_profile(PROFILE)                          # no error
