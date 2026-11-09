"""A network expansion of a box of several molecules, as the web draws it: species by the molecules that
differ from the start, each step on the molecules its path search ran on (the explorer's subsystem), a
propose job's phase, and a step whose TS lies below one of its ends drawn as no barrier."""
import json

from mepd.web import chem, live_network, step_check
from mepd.web.depict_graph import depict_xyz_svg
from mepd.web.results import _expansion_steps
from tests.test_expansion_steps_web import _ws

BOX = "C.C.C.N.N.O.OO.[H][H].[H][H]"
START = "C.C.C.C.N.N.N.O.O"


def test_a_box_species_is_drawn_by_the_molecules_that_differ_from_the_start():
    svg = chem.depict_changed_svg(BOX, START, 220, 160)
    assert "+ unchanged: 3 CH₄, 2 NH₃, H₂O" in svg      # the rest, in a muted line
    assert "2×" in svg and svg.count("<text") == 2      # H2O2 once, H2 once with its count
    # the start itself, a species of one molecule: drawn as before
    assert chem.depict_changed_svg(START, START, 220, 160) == chem.depict_svg(START, 220, 160)
    assert chem.depict_changed_svg("CCO", START, 220, 160) == chem.depict_svg("CCO", 220, 160)


def _xyz(smiles):
    return chem.structure_from_smiles(smiles).to_xyz()


def test_a_box_species_without_a_smiles_is_drawn_by_its_changed_molecules_too():
    svg = depict_xyz_svg(_xyz("O.O.OO"), start=_xyz("O.O.O.O"))
    assert "+ unchanged: 2 H₂O" in svg
    assert "unchanged" not in depict_xyz_svg(_xyz("O.O.OO"))     # no start: the whole box, as before


# Formic acid -> CO2 + H2 next to a water that only watches (atoms 5-7).
_A = """C 0 0 0
O 1.2 0 0
O -0.7 1.1 0
H -0.6 -0.9 0
H -0.2 1.9 0"""
_B = """C 0 0 0
O 1.16 0 0
O -1.16 0 0
H 0 -3 0
H 0.74 -3 0"""
_W = """O 10 0 0
H 10.96 0 0
H 9.76 0.93 0"""


def _frame(mol, e):
    return f"8\nE={e}\n{mol}\n{_W}\n"


def test_a_steps_views_show_the_molecules_its_path_search_ran_on(tmp_path):
    out = tmp_path / "out"
    (out / "ts").mkdir(parents=True)
    (out / "ts" / "s1_irc.xyz").write_text(_frame(_A, -1.0) + _frame(_A, -0.9) + _frame(_B, -1.01))
    (out / "ts" / "s1.xyz").write_text(_frame(_A, -0.9))
    job = {"output_dir": str(out), "charge": 0, "multiplicity": 1}
    cut = step_check.subsystem(job, "s1")
    assert cut == {"atoms": [0, 1, 2, 3, 4], "charge": 0, "multiplicity": 1}   # the explorer's cut: no water
    assert step_check.subset_xyz(_frame(_B, -1.01), cut["atoms"]).splitlines()[0] == "5"
    # the run's TS and IRC entries carry it (the views cut their frames; importing keeps them whole)
    summary = {"species": [], "steps": [{"a": 0, "b": 1, "label": "s1", "barrier_kcal": [10.0, 11.0],
                                         "files": {"ts": str(out / "ts" / "s1.xyz"), "irc": str(out / "ts" / "s1_irc.xyz")}}]}
    ts, irc = _expansion_steps(out, summary, 0, 1)
    assert ts[0]["subsystem"] == irc[0]["subsystem"] == cut
    assert ts[0]["frames"][0]["xyz"].splitlines()[0].strip() == "8"
    # every molecule takes part: no cut, every atom shown
    (out / "ts" / "s2_irc.xyz").write_text(f"5\n\n{_A}\n" + f"5\n\n{_B}\n")
    assert step_check.subsystem(job, "s2") is None
    # a QM/MM system's engines number the whole system: no cut
    assert step_check.subsystem({**job, "qmmm": "q"}, "s1") is None


def test_a_propose_jobs_phase_is_read_for_the_live_view(tmp_path):
    out = tmp_path / "out"
    (tmp_path / "job.json").write_text(json.dumps({"output_dir": str(out)}))
    for k, done in ((0, True), (1, True), (2, False)):
        (out / "reactor" / f"run_{k:02d}" / "md").mkdir(parents=True)
        if done:
            (out / "reactor" / f"run_{k:02d}" / "md" / "done").write_text("")
    live = tmp_path / "live"
    live.mkdir()

    def phase(name, **d):
        (live / "propose_3.json").write_text(json.dumps({"kind": "phase", "data": {"phase": name, **d}}))

    phase("generate", runs_from=1)          # run_00 was an earlier job's
    assert live_network.propose_phase(tmp_path, 3) == "MD: 1/2 trajectories done"
    (live / "queue.json").write_text(json.dumps({"data": {"running": [{"kind": "propose", "what": "species 3 · batch 2"}]}}))
    assert live_network.activity(tmp_path)["queue"]["running"][0]["phase"] == "MD: 1/2 trajectories done"
    phase("minimize", total=37)
    assert live_network.propose_phase(tmp_path, 3) == "minimizing 37 products"
    phase("hessian", done=5, total=37)
    assert live_network.propose_phase(tmp_path, 3) == "Hessian checks 5/37"


def test_a_step_whose_ts_lies_below_one_of_its_ends_has_no_barrier(tmp_path):
    # the live network: drawn unchecked, without its barriers, marked; one above both ends as before
    fine = ["a", "0", "1", 12.0, 3.0, "g", 1, 0.5]
    below = ["b", "0", "2", -4.0, 9.0, "g", None, 0.0]
    tl = live_network.timeline([{"steps": [fine, below], "steered": True, "smiles": {}}], [], None, source={})
    assert tl["frames"][0]["steps"] == [fine, ["b", "0", "2", None, None, "g", -1, 0.0, 1]]
    # the final network: its edge carries no barrier (no kinetics, no profile) and says why
    ws, jobs, (s0, s1, s2) = _ws(tmp_path)
    job = {"id": "j", "live_nodes": {"0": s0, "1": s1, "2": s2}}
    known = ws.snapshot()["structures"]
    step = {"event": "step", "a": 0, "b": 1, "label": "low", "ts_energy": -77.0, "barrier_kcal": [5.0, -2.0]}
    assert jobs._adopt_step(job, step, job["live_nodes"], known)
    (edge,) = ws.snapshot()["edges"].values()
    o = edge["origin"]
    assert o["ts_below_ends"] and o.get("barrier_kcal") is None and o["barrier_verified"] is False
    assert o["headline"] == "TS below its ends: not this reaction, unchecked"
    # a step with a real barrier replaces it; another TS below its ends never replaces that one
    assert jobs._adopt_step(job, {**step, "label": "real", "barrier_kcal": [15.0, 8.0]}, job["live_nodes"], known)
    assert not jobs._adopt_step(job, step, job["live_nodes"], known)
    assert ws.snapshot()["edges"][edge["id"]]["origin"]["barrier_kcal"] == 15.0
    # the run's results: its TS without a barrier or an energy, and no IRC profile
    out = tmp_path / "out"
    (out / "ts").mkdir(parents=True)
    (out / "ts" / "low.xyz").write_text(_frame(_A, -0.9))
    (out / "ts" / "low_irc.xyz").write_text(_frame(_A, -1.0) + _frame(_A, -0.9) + _frame(_B, -1.01))
    summary = {"species": [], "steps": [{"a": 0, "b": 1, "label": "low", "barrier_kcal": [5.0, -2.0],
                                         "files": {"ts": str(out / "ts" / "low.xyz"), "irc": str(out / "ts" / "low_irc.xyz")}}]}
    ts, irc = _expansion_steps(out, summary, 0, 1)
    assert irc == [] and ts[0]["barrier_kcal"] is None and ts[0]["frames"][0]["energy_kcal"] is None
    assert ts[0]["note"].startswith("TS below its ends")
