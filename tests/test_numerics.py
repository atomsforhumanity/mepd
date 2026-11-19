"""mepd.numerics: per-method numerics epochs (Jan's decision 3)."""
from __future__ import annotations

import os
import stat

import pytest

import mepd.numerics as N

CLEAN = {"commit": "a" * 40, "dirty": False}
DIRTY = {"commit": "b" * 40, "dirty": True, "diff_sha256": "c" * 64}


def test_a_method_brings_the_methods_it_uses():
    used = N.methods_of("fsm_ts")
    for m in ("fsm_ts", "fneb", "ts_search", "minimize", "frequencies", "energies", "gsm", "neb", "interpolation"):
        assert m in used
    assert "msmep" not in used
    assert N.methods_of("frequencies") == ["frequencies"]


def test_unknown_methods_are_refused():
    with pytest.raises(KeyError, match="no numerics epoch"):
        N.numerics_epoch("no_such_method", code=CLEAN)


def test_a_clean_checkout_is_named_by_epochs_not_commits(monkeypatch):
    a = N.numerics_epoch("ts_search", code=CLEAN)
    b = N.numerics_epoch("ts_search", code={"commit": "d" * 40, "dirty": False})
    assert a == b                                   # another commit, same numbers: reusable
    assert "ts_search=" in a and "energies=" in a and "qcopt=" in a and "axtb=" in a
    assert "+" not in a


def test_a_bump_or_a_dirty_checkout_changes_it(monkeypatch):
    before = N.numerics_epoch("neb", code=CLEAN)
    monkeypatch.setitem(N.EPOCHS, "energies", N.EPOCHS["energies"] + 1)
    assert N.numerics_epoch("neb", code=CLEAN) != before       # a dependency's bump reaches its users
    dirty = N.numerics_epoch("neb", code=DIRTY)
    assert dirty.endswith("+" + "b" * 12 + ".dirty." + "c" * 8)


def test_gsm_build_is_the_binary_hash(tmp_path, monkeypatch):
    exe = tmp_path / "gsm"
    exe.write_bytes(b"#!/bin/sh\necho gsm v1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    v1 = N.gsm_build(str(exe))
    assert len(v1) == 12
    exe.write_bytes(b"#!/bin/sh\necho gsm v2\n")
    assert N.gsm_build(str(exe)) != v1
    assert N.gsm_build(str(tmp_path / "missing")) == "none"
    monkeypatch.setenv("GSM_EXECUTABLE", str(exe))
    assert f"gsm.build={N.gsm_build()}" in N.numerics_epoch("gsm", code=CLEAN)
    assert "gsm.build" not in N.numerics_epoch("ts_search", code=CLEAN)


def test_every_dependency_has_an_epoch():
    for m, deps in N.DEPENDS.items():
        assert m in N.EPOCHS
        for d in deps:
            assert d in N.EPOCHS


def test_an_untracked_module_makes_the_checkout_dirty(tmp_path):
    """git diff ignores new files: a new uncommitted module in mepd/ must still count as uncommitted edits."""
    import hashlib
    import subprocess

    from mepd.provenance import _checkout_state

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True,
                       env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                            "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_COUNT": "0"})

    (tmp_path / "mepd").mkdir()
    (tmp_path / "mepd" / "a.py").write_text("x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert _checkout_state(tmp_path)["dirty"] is False
    (tmp_path / "mepd" / "new_method.py").write_text("y = 2\n")
    state = _checkout_state(tmp_path)
    assert state["dirty"] is True and state["untracked"] == ["mepd/new_method.py"]
    assert N.dirty_suffix(state).startswith("+" + state["commit"][:12] + ".dirty.")
    (tmp_path / "mepd" / "new_method.py").unlink()
    (tmp_path / "mepd" / "a.py").write_text("x = 2\n")             # a tracked edit hashes as before: the diff alone
    state = _checkout_state(tmp_path)
    diff = subprocess.run(["git", "diff", "HEAD", "--", "mepd"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert state["dirty"] is True and state["diff_sha256"] == hashlib.sha256(diff.strip().encode()).hexdigest()
