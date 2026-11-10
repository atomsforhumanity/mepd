"""Local defaults that the container image does not set for us: one BLAS thread per process,
and `mepd optimize`'s step budget."""

import os
import subprocess
import sys

from typer.testing import CliRunner

_BLAS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _env_after_import(**env_extra) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k not in _BLAS}
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", "import os, mepd; print(*(os.environ[k] for k in %r))" % (_BLAS,)],
        capture_output=True, text=True, env=env, timeout=120, check=True,
    )
    return out.stdout.split()


def test_importing_mepd_caps_blas_at_one_thread():
    assert _env_after_import() == ["1", "1"]


def test_an_exported_blas_thread_count_is_kept():
    assert _env_after_import(OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="2") == ["4", "2"]


class _Engine:
    def __init__(self):
        self.keywords = []

    def compute_geometry_optimization(self, node, keywords=None):
        self.keywords.append(dict(keywords or {}))
        node._cached_energy = -1.0
        return [node]


def _optimize_keywords(tmp_path, monkeypatch, profile: str) -> dict:
    import mepd.cli as cli

    engine = _Engine()
    opened = cli.RunInputs.open.__func__

    def open_with_stub(cls, path):
        ri = opened(cls, path)
        ri.engine = engine
        return ri

    monkeypatch.setattr(cli.RunInputs, "open", classmethod(open_with_stub))
    (tmp_path / "w.xyz").write_text("3\n\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    (tmp_path / "p.toml").write_text(profile)
    res = CliRunner().invoke(cli.app, ["optimize", str(tmp_path / "w.xyz"), "--inputs", str(tmp_path / "p.toml"),
                                       "--output", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
    (kw,) = engine.keywords
    return kw


def test_optimize_allows_3000_steps(tmp_path, monkeypatch):
    assert _optimize_keywords(tmp_path, monkeypatch, 'engine_name = "gxtb"\n')["maxit"] == 3000


def test_optimize_keeps_the_profiles_step_limit(tmp_path, monkeypatch):
    profile = 'engine_name = "gxtb"\n\n[geometry_optimizer_kwds]\nmaxit = 40\n'
    assert _optimize_keywords(tmp_path, monkeypatch, profile)["maxit"] == 40
