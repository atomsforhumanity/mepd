import datetime as dt
import json

from mepd import provenance

NOW = dt.datetime(2026, 10, 13, tzinfo=dt.timezone.utc)


def test_manifest_records_profile_and_code(tmp_path):
    prof = tmp_path / "p.toml"
    prof.write_text('engine_name = "gxtb"\n[gxtb_engine_kwds]\nexecutable = "/no/such/gxtb"\n')
    m = provenance.run_manifest(["mepd", "run"], prof)
    assert m["profile_text"].startswith('engine_name = "gxtb"')
    assert m["level_key"] and len(m["level_key"]) == 10
    assert m["mepd"]["version"]
    assert "numpy" in m["packages"]
    # a missing binary named by the profile is left out, not recorded as found
    assert all(b["name"] != "/no/such/gxtb" for b in m["binaries"])


def test_record_cli_run_writes_into_output_dir(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    dest = provenance.record_cli_run(["conformers", "CCO", "--output", str(out)], NOW, 0)
    assert dest == out / "provenance.json"
    m = json.loads(dest.read_text())
    assert m["argv"] == ["mepd", "conformers", "CCO", "--output", str(out)]
    assert m["returncode"] == 0 and m["started"].startswith("2026-10-13")


def test_file_output_gets_sidecar(tmp_path):
    out = tmp_path / "network.json"
    out.write_text("{}")
    dest = provenance.record_cli_run(["discovery", "x", "-o", str(out)], NOW, 1)
    assert dest == tmp_path / "network.json.provenance.json"
    assert json.loads(dest.read_text())["returncode"] == 1


def test_no_output_or_disabled(tmp_path, monkeypatch):
    assert provenance.record_cli_run(["run", "--output", str(tmp_path / "missing")], NOW, 0) is None
    assert provenance.record_cli_run(["--help"], NOW, 0) is None
    monkeypatch.setenv("MEPD_NO_PROVENANCE", "1")
    assert provenance.record_cli_run(["run", "--output", str(tmp_path)], NOW, 0) is None


def test_never_raises(tmp_path):
    assert provenance.record_cli_run(["run", "--output", str(tmp_path), "--inputs", "/dev/null/x"], NOW, 0)
