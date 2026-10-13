"""The `grad.py` GSM runs per node gradient: the compiled client (when a C
compiler is there) and the Python one talk to the engine server the same
way -- send the node id, succeed on "OK", fail with the server's message
otherwise."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading

import pytest

from mepd.pathminimizers import gsm as G


def _serve(workdir, replies):
    """A stand-in engine server on workdir/engine.sock: answers each
    connection with the next reply and records what it was sent."""
    seen = []
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(workdir / "engine.sock"))
    srv.listen(4)

    def loop():
        for reply in replies:
            conn, _ = srv.accept()
            seen.append(conn.recv(4096).decode())
            conn.sendall(reply)
            conn.close()
        srv.close()

    threading.Thread(target=loop, daemon=True).start()
    return seen


def _clients(tmp_path, monkeypatch):
    out = [("python", G._python_grad_client_source())]
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    G._grad_client_cache.clear()
    compiled = G._compiled_grad_client()
    if compiled is not None:
        out.append(("compiled", compiled))
    return out


def test_clients_send_the_node_and_report_server_errors(tmp_path, monkeypatch):
    for name, client in _clients(tmp_path, monkeypatch):
        work = tmp_path / name
        work.mkdir()
        grad = work / "grad.py"
        if isinstance(client, str):
            grad.write_text(client)
        else:
            shutil.copyfile(client, grad)
        grad.chmod(0o755)
        seen = _serve(work, [b"OK", b"ERROR\nboom"])
        ok = subprocess.run("./grad.py 0000.07 1 0", shell=True, cwd=work, capture_output=True, text=True)
        bad = subprocess.run("./grad.py 0000.08 1 0", shell=True, cwd=work, capture_output=True, text=True)
        assert ok.returncode == 0, (name, ok.stderr)
        assert bad.returncode != 0 and "boom" in bad.stderr, name
        assert seen == ["0000.07", "0000.08"], name


def test_without_a_compiler_the_python_client_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("CC", raising=False)
    G._grad_client_cache.clear()
    try:
        assert G._compiled_grad_client() is None
    finally:
        G._grad_client_cache.clear()
    monkeypatch.setenv(G._GRAD_CLIENT_ENV, "python")
    assert G._compiled_grad_client() is None
    assert G._python_grad_client_source().startswith("#!")


@pytest.mark.skipif(not shutil.which("cc") and not shutil.which("gcc"), reason="no C compiler")
def test_the_compiled_client_is_built_once(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    G._grad_client_cache.clear()
    first = G._compiled_grad_client()
    G._grad_client_cache.clear()
    mtime = first.stat().st_mtime_ns
    assert G._compiled_grad_client() == first and first.stat().st_mtime_ns == mtime
    assert os.access(first, os.X_OK)
    G._grad_client_cache.clear()
