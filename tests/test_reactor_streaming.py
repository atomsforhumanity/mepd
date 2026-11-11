"""The async explorer takes a batched generator's (the reactor's) products after every batch: they are
minimized and connected while the species' next batch runs, until the batches converge or run out. The seed's
batches are one job, their products pooled (the generator's one-call form)."""
import numpy as np
import pytest

from mepd.discovery.network_expansion import expand_network
from tests.test_network_expansion import _AsyncConnector, _ForkedAsyncConnector, _GraphEngine, _hcn_reactor, _node

XYZ = np.array([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [2.2, 0.0, 0.0]])


def _h_on(structure):
    """The heavy atom the H sits on: "C" (HCN) or "N" (HNC)."""
    sy, xyz = list(structure.symbols), np.asarray(structure.geometry, dtype=float)
    h = sy.index("H")
    return min(("C", "N"), key=lambda e: np.linalg.norm(xyz[h] - xyz[sy.index(e)]))


def _batched(log, connector, converge_after=3):
    """A batched stand-in reactor for HCN <-> HNC: batch 1 moves the H (the isomer), later batches find
    nothing new; it converges once its count (carried in the state) reaches `converge_after`."""
    hcn = _hcn_reactor([])

    def batch(structure, *, max_products, options, state=None):
        st = {"batch": 0, "proposed": [], "seen": 0, **(state or {})}   # (a first state may carry known_reactions)
        b = st["batch"]
        on = _h_on(structure)
        log.append({"species": "HCN" if on == "C" else "HNC", "batch": b, "seen": st["seen"],
                    "pairs_so_far": list(connector.pairs) if connector is not None else None})
        products = hcn(structure, max_products=max_products, options=options) if b == 0 else []
        st.update(batch=b + 1, seen=st["seen"] + 1, proposed=st["proposed"] + ["x"] * len(products))
        return products, st, st["seen"] >= converge_after or st["batch"] >= int(options.get("max_batches", 10))

    def whole(structure, *, max_products, options):   # the one-call form: rounds, and the async seed
        log.append({"species": "HCN" if _h_on(structure) == "C" else "HNC", "whole": True})
        return hcn(structure, max_products=max_products, options=options)

    return whole, batch


def _node_species():
    """Which of HCN / HNC the seed _node("C#N") is (the H on C)."""
    return "HCN"


def _kw():
    return dict(steer="access", access_cutoff_kcal=30.0, max_products=5, kinetics={"threshold": 1e-6},
                schedule="async", generator="reactor", generator_options={"n_seeds": 1}, allow_zwitterions=True)


def test_each_batch_is_connected_before_the_next_runs_and_batches_run_until_converged(monkeypatch):
    from mepd.discovery import generators

    log, connect = [], _AsyncConnector()
    whole, batch = _batched(log, connect)
    monkeypatch.setitem(generators.GENERATORS, "reactor",
                        generators.Generator("reactor", "Reactor", "test", whole, kind="structures",
                                             propose_batch=batch))
    res = expand_network(_node("C#N"), _GraphEngine(), connect=connect, **_kw())
    later = [e for e in log if e["species"] != _node_species() and "batch" in e]
    # (2) a later species (HNC, the seed's product) got batches until its carried count converged: 1, 2, 3 --
    # and (3) the count carried over
    assert [e["batch"] for e in later] == [0, 1, 2] and [e["seen"] for e in later] == [0, 1, 2]
    # (1) its first batch's product (HCN, the seed) was connected before its second batch ran
    assert (0, 1) in later[1]["pairs_so_far"] or (1, 0) in later[1]["pairs_so_far"]
    assert res.species[1].smiles and len(res.species) == 2
    # counted expanded once, after its last batch
    assert res.rounds[0]["sources"].count(1) == 1 and 0 in res.rounds[0]["sources"]
    assert "proposals_failed" not in res.rounds[0]
    labels = [lbl for kind, lbl, _ in connect.at_submit if kind == "propose"]
    assert "species 1 · batch 1" in labels and "species 1 · batch 3" in labels


def test_the_seed_s_reactor_batches_are_one_job_their_products_pooled(monkeypatch):
    """Isoprene + methyl acrylate (scoreboard sb-iso vs sb-dev6): streamed, the seed's first batch alone filled the
    product cap with 12-18 of the seed's own products (all four batches found 42-57) and its next batch waited
    behind the TS searches, so the Diels-Alder adducts were proposed in 3 of 10 runs instead of 9. The seed's
    proposals are one job again: the generator's one-call form (its batches pooled), no batch label; later
    species still go batch by batch."""
    from mepd.discovery import generators

    log, connect = [], _AsyncConnector()
    whole, batch = _batched(log, connect)
    monkeypatch.setitem(generators.GENERATORS, "reactor",
                        generators.Generator("reactor", "Reactor", "test", whole, kind="structures",
                                             propose_batch=batch))
    res = expand_network(_node("C#N"), _GraphEngine(), connect=connect, **_kw())
    seed = [e for e in log if e["species"] == _node_species()]
    assert seed == [{"species": "HCN", "whole": True}]                 # one call, the one-call form
    labels = [lbl for kind, lbl, _ in connect.at_submit if kind == "propose"]
    assert labels.count("species 0") == 1 and not any(lbl.startswith("species 0 ·") for lbl in labels)
    assert any(lbl.startswith("species 1 · batch") for lbl in labels)
    assert res.rounds[0]["sources"].count(0) == 1


def test_a_batch_runs_in_a_forked_worker_and_its_state_comes_back(monkeypatch):
    from mepd.discovery import generators

    log = []
    whole, batch = _batched(log, None, converge_after=2)
    monkeypatch.setitem(generators.GENERATORS, "reactor",
                        generators.Generator("reactor", "Reactor", "test", whole, kind="structures",
                                             propose_batch=batch))
    forked = _ForkedAsyncConnector()
    try:
        res = expand_network(_node("C#N"), _GraphEngine(), connect=forked, **_kw())
    finally:
        forked.close()
    assert {0, 1} <= set(res.rounds[0]["sources"]) and "proposals_failed" not in res.rounds[0]


def test_the_reactor_in_batches_keeps_its_convergence_rule_and_proposes_each_product_once(monkeypatch):
    """reactor_batch: the batches, their seeds and the stopping rule of reactor_products, with the counts
    carried in the state; each product goes out with the batch that first made it, once, up to the cap."""
    import mepd.discovery.reactor_generator as rg
    from qcdata import Structure

    seeds, calls = [], [0]

    def fake(jobs, width):   # A every batch (enough to converge at the 3rd), and one new product per batch
        calls[0] += 1
        seeds.extend(j[5]["seed"] for j in jobs)
        new = f"new{calls[0]}"
        return [{"states": [(1, "A", XYZ.tolist()), (2, new, (XYZ + 0.01 * calls[0]).tolist())],
                 "keys": [["A", True, 5.0]] * (10 if calls[0] < 3 else 60) + [[new, True, 5.0]],
                 "events": 11, "torn": 0}]

    monkeypatch.setattr(rg, "_run_in_processes", fake)
    s = Structure(symbols=["H", "C", "N"], geometry=XYZ / 0.529177210903)
    whole = rg.reactor_products(s, n_seeds=1, parallel=1, max_batches=5)
    n_whole, seeds_whole = calls[0], list(seeds)

    calls[0], seeds[:] = 0, []
    state, done, got = None, False, []
    while not done:
        out, state, done = rg.reactor_batch(s, state, n_seeds=1, parallel=1, max_batches=5)
        got.append(len(out))
    assert calls[0] == n_whole == 3 and seeds == seeds_whole          # same batches, same seeds, same stop
    assert got == [2, 1, 1] and sum(got) == len(whole)                  # A once, each new product once
    assert state["first"]["A"] == 80 and state["batch"] == 3            # the counts carried over
    # the cap on products counts every batch's
    calls[0], state, done, got = 0, None, False, []
    while not done:
        out, state, done = rg.reactor_batch(s, state, n_seeds=1, parallel=1, max_batches=5, max_products=2)
        got.append(len(out))
    assert got == [2, 0, 0]


def test_the_explorer_hands_md_method_to_every_reactor_batch(monkeypatch, capsys):
    """md_method (axtb's GFN2-xTB by default) reaches each batch's trajectory jobs, from the explorer's
    generator options through Generator.propose_batch and reactor_batch; each batch's log line names it."""
    pytest.importorskip("axtb")
    import mepd.discovery.reactor_generator as rg

    methods = []

    def fake(jobs, width):
        methods.extend((j[5]["md_method"], j[5]["md_server"]) for j in jobs)
        return [{"states": [], "keys": [], "events": 0, "torn": 0} for _ in jobs]

    monkeypatch.setattr(rg, "_run_in_processes", fake)
    for opts, want in (({}, "gfn2"), ({"md_method": "gxtb-v2.0.1"}, "gxtb-v2.0.1")):
        methods.clear()
        kw = _kw()
        kw["generator_options"] = {"n_seeds": 1, "max_batches": 2, **opts}
        expand_network(_node("C#N"), _GraphEngine(), connect=_AsyncConnector(), **kw)
        assert len(methods) == 2 and set(methods) == {(want, None)}     # both batches, no UMA server
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("reactor: batch")]
        assert [ln.split(":")[1] for ln in lines] == [f" batch {b} (MD" for b in (1, 2)]
        assert all(f"(MD: {want} on axtb)" in ln for ln in lines)
