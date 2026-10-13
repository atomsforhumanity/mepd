"""Network variants in a workspace (mepd/web/variants.py): the parent
network read from TS jobs, the variant's records, which network kinetics
sees, fast estimates as TS energies, and the follow-up jobs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("fastapi")

from rdkit import Chem  # noqa: E402

from mepd.web import design as D  # noqa: E402
from mepd.web import kinetics as K  # noqa: E402
from mepd.web import results as R  # noqa: E402
from mepd.web import variants as V  # noqa: E402
from mepd.web.workspace import Workspace  # noqa: E402

LEVEL = {"profile": "default", "key": "lk", "label": "test level"}
BOHR = 0.529177210903


def _xyz(symbols, coords, comment="") -> str:
    return f"{len(symbols)}\n{comment}\n" + "\n".join(
        f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, coords)) + "\n"


def _structure(symbols, coords, charge=0):
    from qcdata import Structure

    return Structure(symbols=list(symbols), geometry=np.asarray(coords) / BOHR, charge=charge, multiplicity=1)


def _species(ws, smiles, energy, charge=0):
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=3)
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    rec = ws.add_or_merge(_structure(sym, m.GetConformer().GetPositions(), charge), origin={"kind": "test"},
                          energy=energy, level=LEVEL, optimized=True)["rec"]
    return rec["id"]


def _reaction(ws, jobs, text, reactants, products, e_ts, charge=0):
    """A reaction between two complexes, with a finished TS job whose IRC
    frames (start, TS, end) are in the complexes' atom order."""
    from mepd.reaction_smiles import reaction_pair

    pair = reaction_pair(text)
    sym = [a.GetSymbol() for a in pair.reactant.GetAtoms()]
    r, p = pair.reactant.GetConformer().GetPositions(), pair.product.GetConformer().GetPositions()
    a = ws.add_complex(_structure(sym, r, charge), reactants, origin={"kind": "test"})
    b = ws.add_complex(_structure(sym, p, charge), products, origin={"kind": "test"})
    edge = ws.add_edge(a["id"], b["id"], origin={"kind": "test"}, reaction=False)
    rx = ws.put_reaction(reactants=reactants, products=products, origin={"kind": "test", "job": None,
                                                                         "index": edge["id"]},
                         label=text.split(">>")[0][:20], complexes=[a["id"], b["id"]], edge=edge["id"])
    jid = f"j_{len(jobs)}"
    out = ws.jobs_dir / jid / "output"
    out.mkdir(parents=True)
    frames = [{"xyz": _xyz(sym, x), "energy_hartree": e, "energy_kcal": 0.0, "path_length": 0.0}
              for x, e in ((r, e_ts - 0.02), ((r + p) / 2, e_ts), (p, e_ts - 0.03))]
    job = {"id": jid, "op": "ts", "status": "done", "finished": 1.0, "targets": {"structures": [a["id"], b["id"]],
           "edges": [edge["id"]]}, "level": LEVEL, "output_dir": str(out), "charge": charge, "multiplicity": 1,
           "summary": {"barrier_kcal": 12.5, "ts_energy_hartree": e_ts, "barrier_verified": True}}
    result = {"headline": "", "barrier_kcal": 12.5, "summary": [], "warnings": [],
              "groups": [{"title": "Direct", "kind": "channel",
                          "entries": [{"id": "channel_0", "label": "Channel 0", "note": "", "barrier_kcal": 12.5,
                                       "ts_index": 1, "frames": frames}]}],
              "_finished": 1.0, "_version": R.RESULT_VERSION}
    (ws.jobs_dir / jid / "result.json").write_text(json.dumps(result))
    jobs[jid] = job
    return rx["id"], edge["id"]


@pytest.fixture
def network(tmp_path):
    ws = Workspace(tmp_path / "ws")
    jobs = {}
    sp = {"EO": _species(ws, "C1CO1", -153.0), "Br": _species(ws, "[Br-]", -2574.0, -1),
          "ALK": _species(ws, "[O-]CCBr", -2727.05, -1), "CO2": _species(ws, "O=C=O", -188.0),
          "BRC": _species(ws, "O=C([O-])OCCBr", -2915.08, -1)}
    r1 = _reaction(ws, jobs, "[Br-:7].[CH2:1]1[CH2:2][O:3]1>>[Br:7][CH2:1][CH2:2][O-:3]",
                   [sp["Br"], sp["EO"]], [sp["ALK"]], -2727.0, -1)
    r2 = _reaction(ws, jobs, "[Br:7][CH2:1][CH2:2][O-:3].[O:4]=[C:5]=[O:6]>>[Br:7][CH2:1][CH2:2][O:3][C:5](=[O:6])[O-:4]",
                   [sp["ALK"], sp["CO2"]], [sp["BRC"]], -2915.04, -1)
    return ws, jobs, sp, r1, r2


def _design(ws, sid, *ops):
    """The species loaded into Design (atoms tagged), then edited."""
    xyz = ws.structure_path(sid).read_text()
    info, warnings = D.from_xyz(xyz, ws.structure(sid)["charge"])
    tagged = D.tag_atoms(info["molblock"], sanitized=not warnings)["molblock"]
    mb = tagged
    for op in ops:
        mb = D.edit(mb, op)["molblock"]
    return {"molblock": mb, "source": {"kind": "graph", "structure": sid, "parent_molblock": tagged}}


def _methyl(ws, sid):
    """Methyl for the species' first hydrogen (Design keeps the file's atom order on load)."""
    from qcdata import Structure

    h = list(Structure.open(str(ws.structure_path(sid))).symbols).index("H")
    return {"op": "group", "atom": h, "group": "methyl"}


def test_tagged_design_shows_plain_smiles(network):
    ws, jobs, sp, *_ = network
    d = _design(ws, sp["EO"], _methyl(ws, sp["EO"]))
    info = D.describe(D.read(d["molblock"]))
    assert ":" not in info["smiles"]          # no atom-map numbers in what Design shows
    assert any(a.GetAtomMapNum() for a in D.read(d["molblock"]).GetAtoms())


def test_preview_reads_the_parent_ts_frames(network):
    ws, jobs, sp, (r1, e1), (r2, e2) = network
    d = _design(ws, sp["EO"], _methyl(ws, sp["EO"]))
    p = V.preview(ws, jobs, d, LEVEL["key"])
    rows = {r["parent"]: r for r in p["reactions"]}
    assert set(rows) == {r1, r2}
    # Methyl on the attacked carbon or on the other one: two of each.
    assert [len(rows[r]["variants"]) for r in (r1, r2)] == [2, 2]
    assert all(v["status"] == "ok" for r in rows.values() for v in r["variants"])
    assert p["jobs"] == 4 and p["new_species"] == 5


def test_create_tags_records_and_views(network):
    ws, jobs, sp, (r1, e1), (r2, e2) = network
    d = _design(ws, sp["EO"], _methyl(ws, sp["EO"]))
    made = V.create(ws, jobs, d, LEVEL["key"], name="Me")
    var = made["variant"]
    assert len(made["edges"]) == 4 and len(made["species"]) == 5
    snap = ws.snapshot()
    assert snap["variants"][var["id"]]["name"] == "Me"
    assert set(var["replaced"]) == {sp["EO"], sp["ALK"], sp["BRC"]}
    for eid in made["edges"]:
        e = snap["edges"][eid]
        assert e["variant"] == var["id"] and Path(e["origin"]["spec"]).exists()
        spec = json.loads(Path(e["origin"]["spec"]).read_text())
        assert len(spec["symbols"]) == len(spec["r"]) == len(spec["ts"]) == len(spec["p"])
        assert spec["parent"]["energies"] is not None
    # A variant reaction's complexes hold the variant species.
    vrx = [r for r in snap["reactions"].values() if r.get("variant") == var["id"]]
    assert len(vrx) == 4
    for r in vrx:
        cx = snap["structures"][r["complexes"][0]]
        assert sorted(cx["members"]) == sorted(r["reactants"])
    parent = V.network_view(snap, None)
    assert not any(r.get("variant") for r in parent["structures"].values())
    assert set(parent["reactions"]) == {r1, r2}
    mine = V.network_view(snap, var["id"])
    assert sp["EO"] not in mine["structures"] and sp["CO2"] in mine["structures"]
    assert r1 not in mine["reactions"] and len(mine["reactions"]) == 4
    # Update: nothing new to add.
    again = V.create(ws, jobs, {**d, "molblock": var["design_molblock"]}, LEVEL["key"], vid=var["id"])
    assert again["edges"] == [] and again["species"] == []


def test_kinetics_uses_fast_estimates_until_reoptimized(network):
    ws, jobs, sp, (r1, e1), (r2, e2) = network
    d = _design(ws, sp["EO"], _methyl(ws, sp["EO"]))
    made = V.create(ws, jobs, d, LEVEL["key"], name="Me")
    vid = made["variant"]["id"]
    # Energies for the variant species (as their optimize jobs would give).
    for sid in made["species"]:
        rec = ws.structure(sid)
        rec["energy"], rec["level"] = -100.0 - len(rec["formula"]), LEVEL
    ws._save()
    eid = made["edges"][0]
    ws.patch_edge_origin(eid, estimate={"status": "ok", "barrier_shift": 2.0})
    snap = ws.snapshot()
    net, ids, info, excluded = K.build(V.network_view(snap, vid), jobs, LEVEL["key"], full=snap)
    est = [i for i in info if i.get("estimate")]
    assert len(est) == 1 and est[0]["edge"] == eid
    # Its TS: the variant's reactants + the parent's TS height above its own + the shift.
    vr = next(r for r in snap["reactions"].values() if r.get("edge") == eid)
    parent = snap["reactions"][vr["parent"]]
    pe = jobs[next(j for j in jobs if snap["reactions"][vr["parent"]]["edge"] in jobs[j]["targets"]["edges"])]
    above = pe["summary"]["ts_energy_hartree"] - sum(snap["structures"][s]["energy"] for s in parent["reactants"])
    want = sum(snap["structures"][s]["energy"] for s in vr["reactants"]) + above + 2.0 / 627.509474
    k = next(k for k, i in enumerate(info) if i.get("estimate"))
    assert net.steps[k].g_ts == pytest.approx(want * K.HARTREE_KCAL)
    # The parent network is untouched by the variant.
    pnet, pids, pinfo, _ = K.build(V.network_view(snap, None), jobs, LEVEL["key"], full=snap)
    assert {i["reaction"] for i in pinfo} == {r1, r2}
    # Compare: the parent's mixture carried over.
    initial, held, target, match, notes = V.compare_inputs(snap, vid, {sp["EO"]: 1.0, sp["CO2"]: 1.0}, [], sp["BRC"])
    assert sp["CO2"] in initial and sp["EO"] not in initial
    assert len(match[sp["BRC"]]) == 2 and target in match[sp["BRC"]]


class _Manager:
    """The bits of JobManager after_job uses: submissions are recorded."""

    def __init__(self, ws, jobs):
        self.ws, self.jobs, self.calls = ws, jobs, []

    def submit(self, op, **kw):
        self.calls.append((op, kw))
        return [{"id": f"j_new{len(self.calls)}"}]


def test_follow_up_jobs(network, tmp_path):
    ws, jobs, sp, *_ = network
    d = _design(ws, sp["EO"], _methyl(ws, sp["EO"]))
    made = V.create(ws, jobs, d, LEVEL["key"], name="Me")
    m = _Manager(ws, jobs)
    # A reaction whose parent has no TS: its ends are minimized first (then a path search), after the estimates.
    ws.patch_edge_origin(made["edges"][1], status="no_ts")
    started = V.start_jobs(m, made, "default")
    assert [c[0] for c in m.calls] == ["variant-fast"] * 3 + ["variant-reopt"] and len(started) == 4
    assert m.calls[-1][1]["edge_ids"] == [made["edges"][1]]
    eid = made["edges"][0]
    fast = {"id": "j_f", "op": "variant-fast", "status": "done", "targets": {"structures": [], "edges": [eid]},
            "profile": "default", "output_dir": str(tmp_path / "f")}
    V.after_job(m, fast, {"variant": {"fast": {"status": "ok", "barrier_shift": 1.5, "reaction_shift": 0.2}}})
    assert ws.edge(eid)["origin"]["estimate"]["barrier_shift"] == 1.5
    assert m.calls[-1][0] == "variant-reopt"
    reopt = {**fast, "id": "j_r", "op": "variant-reopt", "output_dir": str(tmp_path / "r")}
    V.after_job(m, reopt, {"variant": {"reopt": {"status": "other_minima", "barrier_verified": False}}})
    # The parent's own search (a "ts" job here) on the variant's ends.
    assert m.calls[-1][0] == "ts" and m.calls[-1][1]["edge_ids"] == [eid]
    assert "fallback" in ws.edge(eid)["origin"]


def test_variant_jobs_emit_only_real_cli_flags(network, tmp_path):
    import click
    import typer

    from mepd.cli import app
    from mepd.web.operations import JobContext, VariantParams, get_operation

    ws, jobs, sp, *_ = network
    (ws.profiles_dir / "default.toml").write_text('engine_name = "gxtb"\n')
    made = V.create(ws, jobs, _design(ws, sp["EO"], _methyl(ws, sp["EO"])), LEVEL["key"], name="Me")
    eid = made["edges"][0]
    e = ws.edge(eid)
    cmd = typer.main.get_command(app).get_command(click.Context(typer.main.get_command(app)), "variant")
    known = {o for p in cmd.params for o in p.opts}
    for key in ("variant-fast", "variant-reopt"):
        recs = [ws.structure(e["source"]), ws.structure(e["target"])]
        ctx = JobContext(ws, tmp_path / key, tmp_path / key / "output", recs, "default", edge_ids=[eid], jobs=jobs)
        argv = get_operation(key).build(ctx, VariantParams())
        assert argv[0] == "variant" and Path(argv[1]).exists()
        unknown = [w for w in argv if w.startswith("--") and w not in known]
        assert not unknown, (key, unknown)


def _reoptimized(ws, jobs, made):
    """Each variant edge gets a finished re-optimization whose IRC frames
    (from its spec) are in the edge's order, as `mepd variant` writes them."""
    for eid in made["edges"]:
        spec = json.loads(Path(ws.edge(eid)["origin"]["spec"]).read_text())
        sym = spec["symbols"]
        e_ts = -3000.0 + len(jobs) * 1e-3
        jid = f"j_{len(jobs)}"
        (ws.jobs_dir / jid).mkdir(parents=True)
        frames = [{"xyz": _xyz(sym, np.asarray(spec[k])), "energy_hartree": e, "energy_kcal": 0.0, "path_length": 0.0}
                  for k, e in (("r", e_ts - 0.02), ("ts", e_ts), ("p", e_ts - 0.03))]
        jobs[jid] = {"id": jid, "op": "variant-reopt", "status": "done", "finished": 1.0, "level": LEVEL,
                     "targets": {"structures": [], "edges": [eid]}, "output_dir": str(ws.jobs_dir / jid / "output"),
                     "charge": spec["charge"], "multiplicity": 1,
                     "summary": {"barrier_kcal": 12.0, "ts_energy_hartree": e_ts, "barrier_verified": True}}
        (ws.jobs_dir / jid / "result.json").write_text(json.dumps({
            "headline": "", "barrier_kcal": 12.0, "summary": [], "warnings": [], "_finished": 1.0,
            "_version": R.RESULT_VERSION, "groups": [{"title": "Re-optimized", "kind": "variant", "entries": [
                {"id": "variant_irc", "label": "IRC", "note": "", "barrier_kcal": 12.0, "ts_index": 1,
                 "frames": frames}]}]}))


def test_a_variant_built_on_another(network):
    # Methyl on the epoxide, then iodide for bromide in that variant: the
    # second edit is carried through the first variant's reactions (their
    # re-optimized TSs), and compared with the first variant.
    ws, jobs, sp, (r1, e1), (r2, e2) = network
    me = V.create(ws, jobs, _design(ws, sp["EO"], _methyl(ws, sp["EO"])), LEVEL["key"], name="Me")
    a = me["variant"]["id"]
    _reoptimized(ws, jobs, me)
    d = _design(ws, sp["Br"], {"op": "element", "atom": 0, "element": "I"}, {"op": "charge", "atom": 0, "delta": -1})
    d["source"]["network"] = a
    p = V.preview(ws, jobs, d, LEVEL["key"])
    assert p["on"] == a and p["on_name"] == "Me"
    a_rx = {r["rid"] for rs in me["variant"]["reactions"].values() for r in rs}
    rows = {r["parent"] for r in p["reactions"]}
    assert rows == a_rx                            # the first variant's reactions, not the parent's
    assert all(v["status"] == "ok" for r in p["reactions"] for v in r["variants"])
    made = V.create(ws, jobs, d, LEVEL["key"], name="Me+I")
    b = made["variant"]["id"]
    snap = ws.snapshot()
    assert snap["variants"][b]["on"] == a and V.chain(snap, b) == [b, a]
    view = V.network_view(snap, b)
    assert sp["EO"] not in view["structures"] and sp["Br"] not in view["structures"]
    assert not any(r.get("variant") == a for r in view["reactions"].values())
    assert {r["variant"] for r in view["reactions"].values()} == {b}
    assert not set(V.network_view(snap, a)["structures"]) & set(made["species"])
    # The parent's mixture, through both edits.
    initial, held, target, match, notes = V.compare_inputs(snap, b, {sp["EO"]: 1.0, sp["Br"]: 0.1}, [], sp["BRC"])
    def flat(sid):
        m = Chem.MolFromSmiles(snap["structures"][sid]["smiles"])
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m)

    assert {flat(s): c for s, c in initial.items()} == {"CC1CO1": 1.0, "[I-]": 0.1}
    assert flat(target) in ("CC(CI)OC(=O)[O-]", "CC(I)COC(=O)[O-]")
    # Its re-optimization fails: the search of the reaction it all started from is repeated.
    root = jobs[next(j for j in jobs if e1 in jobs[j]["targets"]["edges"])]
    root.update(op="channels", params={"pairs_per_mechanism": 3})
    m = _Manager(ws, jobs)
    on_r1 = [e for e in made["edges"]
             if ws.edge(ws.edge(e)["origin"]["parent_edge"])["origin"].get("parent_edge") == e1]
    assert on_r1
    reopt = {"id": "j_r", "op": "variant-reopt", "status": "done", "targets": {"structures": [], "edges": on_r1[:1]},
             "profile": "default", "output_dir": "/nonexistent"}
    V.after_job(m, reopt, {"variant": {"reopt": {"status": "other_minima", "barrier_verified": False}}})
    assert m.calls[-1][0] == "channels" and m.calls[-1][1]["params"] == {"pairs_per_mechanism": 3}
    # Iodide on its own: the same molecule as Me+I's iodide merges into that record; both views have it,
    # and deleting Me+I leaves it to the other variant.
    solo = V.create(ws, jobs, _design(ws, sp["Br"], {"op": "element", "atom": 0, "element": "I"},
                                      {"op": "charge", "atom": 0, "delta": -1}), LEVEL["key"], name="I")["variant"]["id"]
    snap = ws.snapshot()
    iodide = next(s for s, r in snap["structures"].items() if r.get("smiles") == "[I-]")
    assert snap["structures"][iodide]["variant"] == b
    assert iodide in V.network_view(snap, b)["structures"] and iodide in V.network_view(snap, solo)["structures"]
    V.delete(ws, b)
    snap = ws.snapshot()
    assert snap["structures"][iodide]["variant"] == solo and iodide in V.network_view(snap, solo)["structures"]
    V.delete(ws, solo)
    assert iodide not in ws.snapshot()["structures"]
    b = V.create(ws, jobs, d, LEVEL["key"], name="Me+I")["variant"]["id"]
    # The first variant can't go while the second is built on it.
    with pytest.raises(Exception, match="built on"):
        V.delete(ws, a)
    V.delete(ws, b)
    V.delete(ws, a)
