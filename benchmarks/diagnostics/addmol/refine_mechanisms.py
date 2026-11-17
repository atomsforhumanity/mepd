"""The mechanisms the explorer's `refine` would sample for the verified steps of stored runs (mapping only).

    PYTHONPATH=<code> python benchmarks/diagnostics/addmol/refine_mechanisms.py OUT.json RUN_OUTPUT_DIR ... [--workers 8]
    python benchmarks/diagnostics/addmol/refine_mechanisms.py --compare OLD.json NEW.json

RUN_OUTPUT_DIR: a `discovery expand` task's output (summary.json + run.tgz). For every step whose TS has a verified
IRC, the two IRC ends are given to `atom_mapping.suggest_mechanism_candidates(explore=2)` (what `refine` asks
`mepd channels` for: `explore_mechanisms: 2`); nothing is searched. Per reaction the JSON holds the keys in the
order the function lists them and, for each explored mapping, whether it is SLAPMapper's first mapping or an
earlier explored one through symmetry-equivalent atoms of the start or the end (computed here, the
same under every code version). --compare: how many reactions get another set of mechanisms and why, and how
many get the same set in another queue order (old code queued by the keys' alphabetical order, new code in the
listed order).
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path


def relabels(sym, adj0, adj1, a, b):
    """One mechanism through equivalent atoms: the start's bonds and the bonds mapping `a` gives the end, as one
    two-layer graph on the start's atoms, are isomorphic to mapping `b`'s."""
    import networkx as nx

    def layers(order):
        g = nx.Graph()
        g.add_nodes_from((i, {"el": sym[i]}) for i in adj0)
        inv = {int(v): i for i, v in enumerate(order)}
        after = {(min(inv[j], inv[k]), max(inv[j], inv[k])) for j in adj1 for k in adj1[j]}
        before = {(i, k) for i in adj0 for k in adj0[i] if i < k}
        g.add_edges_from((i, k, {"kind": (pair in before, pair in after)}) for pair in before | after for i, k in [pair])
        return g

    return nx.is_isomorphic(layers(a), layers(b), node_match=lambda x, y: x["el"] == y["el"],
                            edge_match=lambda x, y: x["kind"] == y["kind"])


def one(task):
    name, label, fp = task
    from mepd.atom_mapping import suggest_mechanism_candidates
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs
    from mepd.qcdata_structure_helpers import structure_to_molecule

    try:
        irc = Chain.from_xyz(fp, ChainInputs(), charge=0, spinmult=1)
        a, b = irc[0].structure, irc[-1].structure
        groups = suggest_mechanism_candidates(a, b, explore=2)
        g = structure_to_molecule(a)
        adj0 = {i: {int(k) for k in g.neighbors(i)} for i in range(len(a.symbols))}
        g1 = structure_to_molecule(b)
        adj1 = {i: {int(k) for k in g1.neighbors(i)} for i in range(len(a.symbols))}
        sym = [str(x) for x in a.symbols]
        keys = list(groups)
        orders = [tuple(v[0].as_order()) for v in groups.values()]
        plain = [o for k, o in zip(keys, orders) if not k.startswith("explored:")]
        flags = []
        for n, (k, o) in enumerate(zip(keys, orders)):
            if not k.startswith("explored:"):
                flags.append(None)
            elif any(relabels(sym, adj0, adj1, p, o) for p in plain):
                flags.append("slapmapper's own, renamed")
            elif any(relabels(sym, adj0, adj1, orders[m], o) for m in range(n) if keys[m].startswith("explored:")):
                flags.append("an earlier explored one, renamed")
            else:
                flags.append("")
        return f"{name}/{label}", {"atoms": len(a.symbols), "keys": keys, "flags": flags}
    except Exception as exc:
        return f"{name}/{label}", {"error": f"{type(exc).__name__}: {exc}"}


def collect(out: Path, runs: list, workers: int) -> None:
    import multiprocessing

    tasks = []
    tmp = Path(tempfile.mkdtemp(prefix="refine_mech_"))
    for run in runs:
        run = Path(run)
        steps = [s for s in json.loads((run / "summary.json").read_text())["steps"]
                 if s["files"].get("verified") and s["files"].get("irc")]
        wanted = {s["files"]["irc"]: s["label"] for s in steps}
        with tarfile.open(run / "run.tgz") as tar:
            for m in tar:
                key = next((w for w in wanted if m.name.endswith(w.split("run/", 1)[1])), None)
                if key and m.isfile():
                    fp = tmp / f"{run.name}__{wanted[key]}_irc.xyz"
                    fp.write_bytes(tar.extractfile(m).read())
                    tasks.append((run.name, wanted[key], fp))
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = dict(pool.map(one, sorted(set(tasks)), chunksize=4))
    import mepd

    out.write_text(json.dumps({"code": mepd.__file__, "reactions": res}, indent=1))
    print(len(res), "reactions;", sum("error" in v for v in res.values()), "failed")


def compare(old_fp: Path, new_fp: Path) -> None:
    old, new = (json.loads(Path(f).read_text())["reactions"] for f in (old_fp, new_fp))
    by_run: dict = {}
    examples: dict = {}
    for name in sorted(old):
        o, n = old[name], new.get(name, {})
        c = by_run.setdefault(name.split("/")[0], {"reactions": 0, "failed": 0, "same list and order": 0,
                                                    "same mechanisms, other queue order": 0,
                                                    "other mechanisms: slapmapper's own was listed again": 0,
                                                    "other mechanisms: an explored one was listed twice": 0,
                                                    "other mechanisms: a place was lost to a copy dropped afterwards (fewer than 2 explored listed)": 0,
                                                    "other mechanisms: other": 0})
        c["reactions"] += 1
        if "error" in o or "error" in n:
            c["failed"] += 1
            continue
        if set(o["keys"]) == set(n["keys"]):
            cause = "same list and order" if sorted(o["keys"]) == n["keys"] else "same mechanisms, other queue order"
        elif "slapmapper's own, renamed" in o["flags"]:
            cause = "other mechanisms: slapmapper's own was listed again"
        elif "an earlier explored one, renamed" in o["flags"]:
            cause = "other mechanisms: an explored one was listed twice"
        elif sum(k.startswith("explored:") for k in o["keys"]) < sum(k.startswith("explored:") for k in n["keys"]):
            cause = "other mechanisms: a place was lost to a copy dropped afterwards (fewer than 2 explored listed)"
        else:
            cause = "other mechanisms: other"
        c[cause] += 1
        examples.setdefault(cause, [])
        if len(examples[cause]) < 2:
            examples[cause].append({"reaction": name, "atoms": o["atoms"], "old (as queued: alphabetical)": sorted(o["keys"]),
                                    "new (as queued)": n["keys"]})
    print(json.dumps({"by run": by_run, "examples": examples}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
        collect(Path(args[0]), [a for a in args[1:] if a != str(workers)], workers)
