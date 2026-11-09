"""The network expansion's built-in steering profile: linus wB97X-D3BJ/def2-TZVP, the default where linus is
installed (the web image), absent elsewhere; written as the job's steer.toml and loaded as --steer-energies is."""
import pytest

from mepd.web import steering
from mepd.web.operations import OPERATIONS
from mepd.web.workspace import WorkspaceError


class Ctx:
    output_dir = "/out"
    structures = [{"id": "s_a"}]
    jobs = {}

    def __init__(self, tmp_path):
        self.job_dir = tmp_path

    def snapshot_structure(self, rec, name):
        return f"/in/{name}.xyz"

    def common_flags(self):
        return ["--charge", "0", "--multiplicity", "1"]

    def level(self):
        return {"key": "k"}

    class ws:
        @staticmethod
        def read_profile(name):
            return {"my dft": 'engine_name = "qccompute"\nprogram = "psi4"\n'}[name]


def _expand(tmp_path, **params):
    op = OPERATIONS["graph-enumeration"]
    return op.build(Ctx(tmp_path), op.parse_params(params))


def test_linus_is_offered_and_the_default_only_where_it_is_installed(monkeypatch):
    installed = steering.linus_available()
    prop = OPERATIONS["graph-enumeration"].params_model.model_json_schema()["properties"]["steer_profile"]
    assert prop["choices_from"] == "profiles"
    assert prop["default"] == (steering.LINUS if installed else "")
    assert prop["builtins"] == ([steering.LINUS] if installed else [])
    monkeypatch.setattr(steering, "linus_available", lambda: False)
    assert steering.builtin_profiles() == {} and steering.default_steer_profile() == ""
    monkeypatch.setattr(steering, "linus_available", lambda: True)
    assert list(steering.builtin_profiles()) == [steering.LINUS] and steering.default_steer_profile() == steering.LINUS


def test_the_builtin_profile_is_the_job_steer_toml(tmp_path, monkeypatch):
    monkeypatch.setattr(steering, "linus_available", lambda: True)
    argv = _expand(tmp_path, steer="access", steer_profile=steering.LINUS)
    toml = argv[argv.index("--steer-energies") + 1]
    assert toml == str(tmp_path / "inputs" / "steer.toml")
    text = (tmp_path / "inputs" / "steer.toml").read_text()
    assert text == steering.LINUS_TOML and 'grid = "linus:2"' in text and "orca" not in text.split("[program_kwds]")[1]
    # a saved profile still works, and none steers with nothing
    argv = _expand(tmp_path, steer="access", steer_profile="my dft")
    assert "psi4" in (tmp_path / "inputs" / "steer.toml").read_text()
    assert "--steer-energies" not in _expand(tmp_path, steer="access", steer_profile="")
    # a job that asked for linus where it is not installed says so
    monkeypatch.setattr(steering, "linus_available", lambda: False)
    with pytest.raises(WorkspaceError, match="linus is not installed"):
        _expand(tmp_path, steer="access", steer_profile=steering.LINUS)


def test_the_builtin_profile_loads_as_steer_energies_does():
    eng = steering.builtin_profile_engine()
    assert type(eng).__name__ == "QCComputeEngine" and eng.program == "linus"
    kw = eng.program_args.keywords
    assert kw["jk"] == "df" and kw["grid"] == "linus:2" and kw["threads"] == 1 and kw["memory_gb"] == 1.5
    assert eng.program_args.model.method == "wb97x-d3bj" and eng.program_args.model.basis == "def2-tzvp"


@pytest.mark.skipif(not steering.linus_available(), reason="needs linus")
def test_linus_gives_a_water_energy_at_the_builtin_level():
    from qcdata import Structure

    from mepd.nodes.node import StructureNode

    xyz = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
    e = steering.builtin_profile_engine().compute_energies([StructureNode(structure=Structure.from_xyz(xyz))])[0]
    assert -76.5 < e < -76.3


def test_steering_is_in_the_main_form_and_its_profile_on_the_profiles_page(tmp_path, monkeypatch):
    """Not under Advanced: shown whenever the expansion steers by access. The built-in profile is listed for the
    Profiles page (read-only, Save as... copies it) through /api/state."""
    prop = OPERATIONS["graph-enumeration"].params_model.model_json_schema()["properties"]["steer_profile"]
    assert not prop["advanced"] and prop["requires"] == "steer=access"
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    monkeypatch.setattr(steering, "linus_available", lambda: True)
    with TestClient(create_app(tmp_path / "ws")) as c:
        assert c.get("/api/state").json()["builtin_profiles"] == {steering.LINUS: steering.LINUS_TOML}
    monkeypatch.setattr(steering, "linus_available", lambda: False)
    with TestClient(create_app(tmp_path / "ws2")) as c:
        assert c.get("/api/state").json()["builtin_profiles"] == {}
