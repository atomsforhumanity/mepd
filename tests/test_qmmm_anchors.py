"""The Nanoreactor on a QM/MM system's QM model holds its anchors (the link H and the QM atoms they cap)
at their distances to each other in every form of it -- xtb's piston MD, the reactor and collide -- and
never pulls them. Without anchors (any run that isn't QM/MM) nothing changes."""
import json

import numpy as np
import pytest
from qcdata import Structure

from mepd.discovery.langevin import held_distances
from mepd.discovery.nanoreactor import _ANCHOR_FORCE
from mepd.discovery.nanoreactor_generator import parse_anchors
from mepd.discovery.reactor_generator import collision_pair

from test_reactor_generator import SYM, XYZ  # noqa: E402  (tests/ is on sys.path)

BOHR = 0.529177210903


def test_the_anchor_restraint_is_xtbs_constrain_with_an_exact_gradient():
    # xtb 6.7.1: H2 at 0.74 A held at 0.84 A ($constrain force constant=0.5, atoms: 1,2 against a reference)
    # adds 0.017855296300 Eh: E = k (r - r0)^2
    ref = np.array([[0, 0, 0], [0, 0, 0.84], [5, 0, 0]]) / BOHR
    x = np.array([[0, 0, 0], [0, 0, 0.74], [5, 0, 0]]) / BOHR
    e, _ = held_distances([0, 1], ref, _ANCHOR_FORCE)(x)
    assert e == pytest.approx(0.017855296300, rel=1e-5)   # (xtb read the 0.74 A as 0.7400001 A)
    # every pair of the anchors, the others untouched; the gradient by finite differences
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(6, 3)) * 2.0
    x = ref + rng.normal(size=ref.shape) * 0.2
    hold = held_distances([4, 0, 2, 0], ref, _ANCHOR_FORCE)
    e, g = hold(x)
    fd = np.zeros_like(x)
    for a in range(len(x)):
        for c in range(3):
            xp, xm = x.copy(), x.copy()
            xp[a, c] += 1e-6
            xm[a, c] -= 1e-6
            fd[a, c] = (hold(xp)[0] - hold(xm)[0]) / 2e-6
    assert np.allclose(g, fd, atol=1e-7)
    assert not g[[1, 3, 5]].any() and np.allclose(g.sum(axis=0), 0.0)
    assert hold(ref)[0] == 0.0
    assert held_distances([], ref, _ANCHOR_FORCE) is None and held_distances([3], ref, _ANCHOR_FORCE) is None
    assert parse_anchors("") == [] and parse_anchors("5,0") == [5, 0]


def test_collide_never_picks_an_anchor():
    xyz = np.vstack([XYZ, XYZ + [6.0, 0, 0]])
    held = [0, 3, 6]
    rng = np.random.default_rng(1)
    picks = [collision_pair(SYM * 2, xyz, rng, anchors=held) for _ in range(300)]
    assert any(len(p) == 4 for p in picks) and any(len(p) == 2 for p in picks)
    assert all(not set(held) & {a for part in p for a in (part if isinstance(part, tuple) else (part,))}
               for p in picks)   # neither pulled nor pushed with its molecule
    # no anchors: the same draws as before
    r1, r2 = np.random.default_rng(2), np.random.default_rng(2)
    assert [collision_pair(SYM * 2, xyz, r1) for _ in range(50)] == \
           [collision_pair(SYM * 2, xyz, r2, anchors=()) for _ in range(50)]


def _collide_jobs(monkeypatch, **kw):
    import mepd.discovery.collide_generator as cg

    got = []
    monkeypatch.setattr(cg, "_run_in_processes", lambda args, width: got.extend(args) or [None] * len(args))
    cg.collide_products(Structure(symbols=SYM, geometry=XYZ / BOHR, charge=0, multiplicity=1), seed=5, parallel=1,
                        **kw)
    return got


# formic acid, seed 5: the collisions (pair, seed, bond pushed) collide ran before anchors (develop 6863884)
COLLIDE_BEFORE = [((2, 0), 5, None), ((4, 3), 6, None), ((4, 0), 7, (1, 0)), ((4, 2), 8, None), ((4, 0), 9, (4, 1)),
                  ((3, 2), 10, None), ((3, 1), 11, (3, 0)), ((4, 0), 12, None), ((2, 0), 13, (2, 1)),
                  ((3, 1), 14, None), ((4, 3), 15, (4, 1)), ((4, 2), 16, (2, 1)), ((4, 3), 17, (1, 0)),
                  ((4, 3), 18, (3, 0)), ((3, 1), 19, (1, 0)), ((3, 2), 20, (1, 0)), ((3, 2), 21, (2, 1)),
                  ((4, 2), 22, (4, 1)), ((3, 2), 23, (3, 0)), ((2, 0), 24, (1, 0))]


def test_collide_skips_moves_on_anchors_and_without_them_runs_what_it_ran(monkeypatch):
    for kw in ({}, {"anchors": ""}):
        jobs = _collide_jobs(monkeypatch, **kw)
        assert [(a[4], a[9], a[10]) for a in jobs] == COLLIDE_BEFORE
        assert all(a[14] == [] for a in jobs)
    jobs = _collide_jobs(monkeypatch, anchors="3,1")
    assert jobs and all(not {1, 3} & {*a[4], *(a[10] or ())} for a in jobs)
    assert all(a[14] == [3, 1] for a in jobs)


def test_the_reactor_without_anchors_pulls_what_it_pulled(tmp_path):
    """A fixed-seed reactor trajectory pulls the pairs it pulled before anchors (develop 6863884), at the same
    times; with anchors="" it is the same trajectory, frame for frame."""
    pytest.importorskip("axtb")
    from mepd.discovery.reactor_generator import _trajectory

    kw = dict(temperature=1500.0, time_ps=0.4, compress=0.7, period_ps=0.2, duty=0.5, pull_fs=50.0, gap_fs=20.0,
              k=100.0, intra_hops=2, seed=3, md_method="gfn2")
    plain = _trajectory(SYM, XYZ, 0, 1, workdir=str(tmp_path / "plain"), **kw)
    none = _trajectory(SYM, XYZ, 0, 1, anchors=parse_anchors(""), **kw)
    assert np.array_equal(plain, none)
    (fp,) = (tmp_path / "plain").glob("run_*/md/schedule.json")
    pulls = [(p["t0_fs"], p["atoms"], p["molecules"]) for p in json.loads(fp.read_text())["pulls"]]
    assert pulls == [(41.0, [3, 2], None), (93.0, [4, 3], None), (152.0, [2, 3], [[0, 1, 2], [3, 4]]),
                     (227.5, [2, 0], None), (297.0, [2, 0], None), (348.0, [3, 1], [[3], [0, 1, 2]])]


def _diol_in_water():
    """Butane-1,4-diol and a water 4 A from it; the QM region: both HO-CH2 ends and the water (two cuts,
    C1-C2 and C4-C3), so the QM model is two capped methanols and a water. The system, its region, its QM
    model and the model's coordinates (Angstrom)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from mepd.qmmm import QMMMRegion

    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCO.O"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    x = mol.GetConformer().GetPositions()
    sy = [a.GetSymbol() for a in mol.GetAtoms()]
    nb = lambda i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()]
    water = [6] + nb(6)
    x[water] += x[:6].mean(axis=0) - x[6] + [0.0, 0.0, 4.0]
    qm = sorted({0, 1, 4, 5, 6} | {h for i in (0, 1, 4, 5, 6) for h in nb(i) if sy[h] == "H"})
    s = Structure(symbols=sy, geometry=x / BOHR)
    region = QMMMRegion.build(s, qm, active_radius=None)
    model = region.model_structure(s)
    return s, region, model, np.asarray(model.geometry) * BOHR


def _drift(frames, anchors) -> float:
    """The largest change of any distance between two anchors from the first frame (Angstrom)."""
    a = np.asarray(anchors)
    i, j = np.triu_indices(len(a), 1)
    d = lambda f: np.linalg.norm(np.asarray(f)[a[i]] - np.asarray(f)[a[j]], axis=1)
    return max(float(np.abs(d(f) - d(frames[0])).max()) for f in frames)


def test_the_expansion_hands_its_anchors_to_every_form_of_the_nanoreactor(monkeypatch):
    from mepd.discovery import generators
    from mepd.discovery.network_expansion import expand_network, nanoreactor_anchors
    from mepd.nodes.node import StructureNode

    from test_network_expansion import _GraphEngine

    system, region, model, _ = _diol_in_water()
    engine = _GraphEngine()
    engine.region = region
    want = ",".join(str(a) for a in nanoreactor_anchors(region))
    assert len(region.links) == 2
    for name, expect in (("reactor", want), ("collide", want), ("nanoreactor", want), ("crest-msreact", None)):
        seen = []
        monkeypatch.setitem(generators.GENERATORS, name, generators.Generator(
            name, name, "test", lambda s, *, max_products, options: seen.append((len(s.symbols), dict(options))) or [],
            kind="structures"))
        expand_network(StructureNode(structure=system, has_molecular_graph=False), engine, rounds=1, generator=name,
                       generator_options={"seed": 1})
        ((natoms, opts),) = seen
        assert natoms == len(model.symbols)                  # the QM model
        assert opts.get("anchors") == expect


def test_the_reactor_holds_a_qm_models_anchors_at_2200_k(tmp_path):
    """Two capped methanols and a water (the QM model of two residues' ends and a substrate) in the reactor at
    2200 K, piston and pulls: every distance between anchors stays within 0.2 A of the start, where without
    anchors the two methanols drift Angstroms apart; no pull takes an anchor."""
    pytest.importorskip("axtb")
    from mepd.discovery.network_expansion import nanoreactor_anchors
    from mepd.discovery.reactor_generator import _trajectory

    _, region, model, xyz = _diol_in_water()
    anchors = nanoreactor_anchors(region)
    kw = dict(temperature=2200.0, time_ps=1.0, compress=0.7, period_ps=0.5, duty=0.75, pull_fs=100.0, gap_fs=50.0,
              k=100.0, intra_hops=2, seed=0)
    held = _trajectory(list(model.symbols), xyz, 0, 1, anchors=anchors, workdir=str(tmp_path), **kw)
    free = _trajectory(list(model.symbols), xyz, 0, 1, **kw)
    assert _drift(held, anchors) < 0.2
    assert _drift(free, anchors) > 1.0
    (fp,) = tmp_path.glob("run_*/md/schedule.json")
    pulls = json.loads(fp.read_text())["pulls"]
    assert pulls and all(not set(anchors) & {*p["atoms"], *(a for m in p["molecules"] or [] for a in m)}
                         for p in pulls)


def test_collide_and_the_reactors_quench_hold_the_anchors():
    """Collide pushes the water onto a capped methanol (turning the water, not the anchored methanol) and
    quenches; the reactor's quench of a snapshot whose second methanol moved 0.5 A puts it back."""
    pytest.importorskip("axtb")
    from mepd.discovery.collide_generator import _collide
    from mepd.discovery.network_expansion import nanoreactor_anchors
    from mepd.discovery.reactor_generator import _quench

    _, region, model, xyz = _diol_in_water()
    anchors = nanoreactor_anchors(region)
    sy = list(model.symbols)
    o_water = next(i for i, s in enumerate(sy) if s == "O" and min(
        np.linalg.norm(xyz[a] - xyz[i]) for a in anchors) > 3.0)
    end = _collide(sy, xyz, 0, 1, (o_water, 0), 1000.0, 200.0, 100.0, 100.0, 0, anchors=anchors)
    assert not isinstance(end, str)
    assert _drift([xyz, end], anchors) < 0.2
    # the second methanol (its capped carbon's molecule) moved off by 0.5 A, then quenched
    from mepd.discovery.free_energy import molecules

    second = next(m for m in molecules(sy, xyz, scale=1.3) if anchors[-1] in m)
    moved = xyz.copy()
    moved[second] += [0.5, 0.0, 0.0]
    hold = held_distances(anchors, xyz / BOHR, _ANCHOR_FORCE)
    back, _ = _quench(sy, moved, 0, 1, restraint=hold)
    left, _ = _quench(sy, moved, 0, 1)
    assert _drift([xyz, back], anchors) < 0.1 < _drift([xyz, left], anchors)
