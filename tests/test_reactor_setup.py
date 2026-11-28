"""The Nanoreactor setup (#/reactor/new): each species read before anything is added (name, atoms, charge,
spin), the box's spin carried into the unattended Nanoreactor job and the interactive reactor, and a spin the
electrons cannot have refused."""
import pytest
from fastapi.testclient import TestClient

from mepd.web import chem
from mepd.web.app import create_app


@pytest.fixture(autouse=True)
def _isolated_recent_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture
def client(tmp_path):
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        yield c


def test_species_are_read_by_smiles_or_everyday_name():
    ald = chem.species_info("CC=O")
    assert (ald["name"], ald["atoms"], ald["charge"], ald["electrons"], ald["multiplicity"]) == ("acetaldehyde", 7, 0, 24, 1)
    oh = chem.species_info("[OH-]")
    assert (oh["atoms"], oh["charge"], oh["electrons"], oh["multiplicity"]) == (2, -1, 10, 1)
    assert chem.species_info("[OH]")["multiplicity"] == 2               # a radical: its unpaired electron
    water = chem.species_info("water")
    assert (water["input"], water["smiles"], water["atoms"]) == ("water", "O", 3)
    # O2 as written (O=O) has no radical in its SMILES; its ground state is a triplet, and the page says why.
    for typed in ("O=O", "oxygen", "[O][O]"):
        o2 = chem.species_info(typed)
        assert o2["multiplicity"] == 3 and o2["smiles"] in ("O=O", "[O][O]")
    assert "triplet" in chem.species_info("O=O")["spin_note"]
    assert chem.species_info("C1CC")["error"] == "unclosed ring"


def test_the_box_spin_adds_up_the_species_or_is_refused():
    assert chem.box_multiplicity(30, [1, 1, 1]) == 1
    assert chem.box_multiplicity(30, [1, 3]) == 3           # O2 triplet in water
    assert chem.box_multiplicity(32, [3, 3]) == 5           # two O2
    assert chem.box_multiplicity(31, [1, 2]) == 2
    assert chem.box_multiplicity(32, [3, 3], 1) == 1        # asked for: the lowest
    with pytest.raises(ValueError, match="even count takes an odd"):
        chem.box_multiplicity(30, [1], 2)
    with pytest.raises(ValueError, match="odd count takes an even"):
        chem.box_multiplicity(31, [2], 3)


def test_the_preview_endpoint_reads_every_row(client):
    rows = client.post("/api/species/preview", json={"smiles": ["2", "CC=O", "hydroxide", "C1CC"]}).json()
    assert [r.get("name") for r in rows[1:3]] == ["acetaldehyde", "hydroxide"]
    assert rows[2]["charge"] == -1 and "error" in rows[3]
    assert client.get("/api/state").json()["workspace"]["structures"] == {}   # nothing added


def test_the_nanoreactor_job_runs_at_the_box_spin(client):
    (ald,) = client.post("/api/structures", json={"text": "CC=O", "optimize": False}).json()
    (o2,) = client.post("/api/structures", json={"text": "O=O", "multiplicity": 3, "optimize": False}).json()
    assert o2["multiplicity"] == 3

    def argv(params):
        r = client.post("/api/jobs", json={"op": "nanoreactor", "structures": [ald["id"], o2["id"]],
                                           "params": {"copies": "2, 1", **params}, "dry_run": True})
        assert r.status_code == 200, r.text
        return r.json()[0]["argv"]

    a = argv({})                                   # from the species: the O2's triplet
    assert a[a.index("--multiplicity") + 1] == "3" and a[a.index("--charge") + 1] == "0"
    a = argv({"multiplicity": 1})                  # the lowest, asked for
    assert a[a.index("--multiplicity") + 1] == "1"
    r = client.post("/api/jobs", json={"op": "nanoreactor", "structures": [ald["id"], o2["id"]],
                                       "params": {"copies": "2, 1", "multiplicity": 2}, "dry_run": True})
    assert r.status_code == 400 and "cannot hold 64 electrons" in r.json()["detail"]


def test_the_interactive_reactor_runs_at_the_box_spin(client):
    pytest.importorskip("tblite.interface")
    (w,) = client.post("/api/structures", json={"text": "O", "optimize": False}).json()
    (o2,) = client.post("/api/structures", json={"text": "O=O", "multiplicity": 3, "optimize": False}).json()
    box = client.post("/api/sandbox", json={"counts": {w["id"]: 2, o2["id"]: 1}, "temperature": 300}).json()
    try:
        assert box["multiplicity"] == 3 and box["charge"] == 0
    finally:
        client.delete(f"/api/sandbox/{box['id']}")
    r = client.post("/api/sandbox", json={"counts": {w["id"]: 2, o2["id"]: 1}, "multiplicity": 2})
    assert r.status_code == 400 and "even count takes an odd" in r.json()["detail"]
