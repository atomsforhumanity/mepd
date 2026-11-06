"""The live network view's data (mepd.web.live_network): explorer kinetics
refreshes become compact frames, corrections compact events, and the end state
is never a trailing unsteered refresh."""
import json

from mepd.web import live_network as ln


def _kin(t, basis="free energy (353 K); single points on 3 of 3 structures", final_b=0.3):
    return {
        "basis": basis, "t_wall": 1000.0 + t, "t_start": 1000.0,
        "species": [
            {"id": "0", "smiles": "C=C.C=C", "G": 0.0, "max_concentration": 1.0, "final": 0.1, "populated": True, "energy_source": "dft"},
            {"id": "1", "smiles": "C1CCC1", "G": -20.0, "max_concentration": 0.6, "final": 0.6, "populated": True, "energy_source": "dft"},
            {"id": "2", "smiles": "C=CC=C", "G": -5.0, "max_concentration": final_b, "final": final_b, "populated": True, "energy_source": "gxtb"},
        ],
        "mirror_species": [{"id": "m1", "mirror_of": "1", "max_concentration": 0.1, "final": 0.1}],
        "step_list": [
            {"id": "a", "from": "0", "to": "1", "dG_fwd": 25.0, "dG_rev": 45.0, "energy_source": "dft", "irc_ok": True, "state": "open", "flux": 0.7},
            {"id": "b", "from": "0", "to": "2", "dG_fwd": 28.0, "dG_rev": 33.0, "energy_source": "gxtb", "irc_ok": None, "state": "open", "flux": 0.3},
            {"id": "c", "from": "1", "to": "2", "dG_fwd": 60.0, "dG_rev": 75.0, "energy_source": "gxtb", "state": "closed", "flux": None},
        ],
    }


def test_a_refresh_becomes_a_compact_frame():
    f = ln.compact(_kin(12.0))
    assert f["t"] == 12.0 and f["steered"]
    ids = [s[0] for s in f["species"]]
    assert ids[0] == "0"                                   # the seed first
    one = next(s for s in f["species"] if s[0] == "1")
    assert one[2] == 0.7                                   # its mirror image's material folded in
    assert {s[0] for s in f["steps"]} == {"a", "b"}        # only open steps carry
    assert f["gone"] == [["c", "1", "2", "closed"]]
    assert f["counts"]["closed"] == 1 and f["counts"]["open"] == 2
    assert ln.compact({"species": [{"id": "0"}]}) is None   # not the live-network schema


def test_a_job_records_its_refreshes_and_ends_on_a_steered_one(tmp_path):
    for t in (0.0, 30.0):
        assert ln.record(tmp_path, _kin(t))
    ln.record(tmp_path, _kin(60.0, basis="free energy (353 K)"))       # unsteered tail
    (tmp_path / "live" / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "species", "index": 1},
        {"event": "change", "kind": "barrier_moved", "t_wall": 1031.0, "ids": ["a"], "old": {"dG_fwd": 27.0},
         "new": {"dG_fwd": 25.0, "energy_source": "dft"}, "why": "its TS's steering single point came back"},
    ]) + "\n")
    tl = ln.from_job(tmp_path)
    assert len(tl["frames"]) == 3 and tl["final"] == 1
    assert tl["events"] == [[31.0, "barrier_moved", ["a"], {"from": 27.0, "to": 25.0, "src": "dft",
                                                           "why": "its TS's steering single point came back"}]]
    assert tl["smiles"]["1"] == "C1CCC1"
