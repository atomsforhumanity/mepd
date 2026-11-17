"""MINMAX path search (`mepd channels --method minmax`; provisional name).

One path minimization per elementary step, on the most promising guess for
it, instead of a search per conformer pair:

1. sample conformers of both ends (N and M, minimized at the profile's level);
2. enumerate the candidate mechanisms by atom mapping (SLAPMapper's, plus
   `minmax_explore` explored ones);
3. build the product in the frame of every reactant conformer, and
4. the reactant in the frame of every product conformer, once per mechanism
   (mepd.built_pair_mapping: partners whose bonds or stereochemistry change
   are dropped);
5. score all (N + M) x X built pairs by the GFN2-xTB energy of the highest
   point of their geodesic interpolation (absolute, so pairs built from
   different conformers compare on one scale);
6. path-minimize only the pair whose peak is lowest: the minimum over pairs
   of the maximum along the path, hence MINMAX.

A path that is not an elementary step is split as MSMEP splits it, and each
piece's two minima go through 1-6 again (new conformers, mechanisms, built
pairs and scores) before that piece is searched, until every piece is
elementary. Then every leaf gets a TS optimization and IRC and the TSs are
classified, as for Reaction channels, so the output reads like a channels run.

The search is one MSMEP tree (pairs/pair_minmax/tree): MSMEP's own depth,
cycle and stop limits apply; `_MinmaxMSMEP._piece_chain` swaps each split
piece for its best-scoring built pair. What was sampled, built, scored and
chosen at every node is in minmax/n<index>/ (pools, built_pairs/) and
minmax.json.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Optional

from mepd.chain import Chain
from mepd.msmep import MSMEP

logger = logging.getLogger("mepd.minmax")

STREAM = "pair_minmax"
LINEAGE_ATTRS = ("_same_pair_split_lineage", "_split_ancestors", "_na_route")


def _log(message: str = "") -> None:
    from mepd.channel_search import _log as log

    log(message)


def _species_key(node) -> str:
    """The same for two geometries only when their atoms are bonded alike
    (atom indices included) and their stereochemistry matches: a conformer
    pool sampled for one is a pool for the other, in the same numbering."""
    edges = sorted(tuple(sorted((int(u), int(v)))) for u, v in node.graph.edges())
    try:
        from mepd.NetworkBuilder import _stereochemical_smiles_key

        stereo = _stereochemical_smiles_key(node) or ""
    except Exception:
        stereo = ""
    return json.dumps([edges, stereo])


def _same_species(a, b) -> bool:
    from mepd.nodes.nodehelpers import _is_connectivity_identical

    try:
        return bool(_is_connectivity_identical(a, b, verbose=False))
    except Exception:
        return False


def _label(node) -> str:
    from mepd.channel_search import _species_label

    try:
        return _species_label(node)
    except Exception:
        return "?"


class MinmaxSearch:
    """The outer procedure (steps 1-6) for one pair of minima, and the
    record of every node it chose a pair for."""

    def __init__(self, run_inputs, options, output: Path, conformer_inputs, workers: int):
        self.run_inputs = run_inputs
        self.options = options
        self.output = Path(output)
        self.conformer_inputs = conformer_inputs
        self.workers = max(1, int(workers))
        self.pools: dict[str, list] = {}       # _species_key -> minimized pool
        self.records: dict[int, dict] = {}     # tree node index -> what was chosen for it
        self.root_pools: Optional[tuple] = None

    # -- step 1
    def pool(self, node, label: str, folder: Path, stats: dict) -> list:
        """`node`'s conformers, minimized and deduplicated as channels does
        (one pool per species and numbering, shared by every node of the run)."""
        from mepd.channel_search import _minimize_conformer_pool
        from mepd.conformers import (_subselect_conformers, generate_conformers,
                                     merge_degenerate_complex_conformers, merge_mirror_images)
        from mepd.inputs import ChainInputs

        key = _species_key(node)
        if key in self.pools:
            stats["reused"] = True
            stats["n_final"] = len(self.pools[key])
            _log(f"  {label}: reusing the {len(self.pools[key])} conformer(s) sampled for this species earlier.")
            return self.pools[key]
        opts = self.options
        try:
            confs = generate_conformers(node, self.conformer_inputs, stats)
        except Exception as exc:
            _log(f"  {label}: conformer sampling failed ({type(exc).__name__}: {exc}); using the geometry as is.")
            confs = [node]
        stats["n_sampled"] = len(confs)
        if opts.minimize_ends:
            confs = _minimize_conformer_pool(confs, label, self.run_inputs) or [node]
            confs = _subselect_conformers(confs, n_max=None, rmsd_cutoff=opts.rmsd_cutoff)
            confs = merge_degenerate_complex_conformers(confs, opts.rmsd_cutoff, opts.complex_energy_tol)
        # Built pairs are not crossed between pools, so a mirror twin only repeats its twin's pairs mirrored.
        confs = merge_mirror_images(confs, opts.rmsd_cutoff)
        stats["n_final"] = len(confs)
        folder.mkdir(parents=True, exist_ok=True)
        Chain.model_validate({"nodes": confs, "parameters": ChainInputs()}).write_to_disk(folder / f"{label}_pool.xyz")
        _log(f"  {label}: {len(confs)} conformer(s).")
        self.pools[key] = confs
        return confs

    # -- steps 2-6
    def choose(self, a, b, index: int, depth: int):
        """The best-scoring built pair between minima `a` and `b` (same atom
        order) as (start, end) StructureNodes, or (a, b) when none builds."""
        from mepd import search_control
        from mepd.channel_search import _built_pairs_by_mechanism, _choose_built_pairs, _save_built_pairs
        from mepd.nodes.node import StructureNode

        t0 = time.perf_counter()
        folder = self.output / "minmax" / f"n{index}"
        rec: dict = {"index": index, "depth": depth, "start": _label(a), "end": _label(b), "conformers": {}}
        self.records[index] = rec
        _log(f"MINMAX node {index} (depth {depth}): {rec['start']} -> {rec['end']}")
        pools = {}
        for side, node in (("start", a), ("end", b)):
            pools[side] = self.pool(node, side, folder / "conformers", rec["conformers"].setdefault(side, {}))
        if index == 0:
            self.root_pools = (pools["start"], pools["end"])
        rows, n_maps = [], 0
        for side in ("start", "end"):
            if search_control.stop_requested():
                break
            _log(f"  building the {'product' if side == 'start' else 'reactant'} in the frame of each of the "
                 f"{len(pools[side])} {'reactant' if side == 'start' else 'product'} conformer(s), per mechanism...")
            found, m = _built_pairs_by_mechanism(pools[side], side, a, b, self.run_inputs,
                                                 workers=min(self.workers, len(pools[side])))
            rows += [(side, k, r) for k, r in found]
            n_maps = max(n_maps, m)
        rec["n_mappings"] = n_maps
        rec["n_built_pairs"] = len(rows)
        rec["seconds_choosing"] = round(time.perf_counter() - t0, 3)
        if not rows:
            rec["chosen"] = None
            rec["note"] = "no partner built with its bonds and stereochemistry intact: searched as given"
            _log("  WARNING: no partner could be built for any conformer and mechanism; searching these two "
                 "minima as they are.")
            return a, b
        ranked, chosen, lines = _choose_built_pairs(rows, 1)
        n_mech = len({r.key for _, _, r in ranked})
        n_src = len(pools["start"]) + len(pools["end"])
        _log(f"  {len(ranked)} built pair(s) of at most {n_src} conformer(s) x {n_maps} mapping(s) = "
             f"{n_src * n_maps}, {n_mech} mechanism(s), ranked by the GFN2-xTB peak of their interpolation; "
             "the lowest is searched:")
        for ln in lines[:12]:
            _log(ln)
        if len(lines) > 12:
            _log(f"  ... {len(lines) - 12} more in {folder / 'built_pairs'}")
        _save_built_pairs(folder / "built_pairs", ranked, {"built_0"}, "snap-gi-xtb")
        src, k, best = ranked[0]
        floors = [r.floor for _, _, r in ranked if r.floor is not None]
        rec["n_mechanisms"] = n_mech
        rec["chosen"] = {
            "label": "built_0", "built_from": src, "conformer": k, "mechanism": best.key,
            "peak_kcal": best.score,
            "peak_above_lowest_start_kcal": (best.score - min(floors)) if floors and best.floor is not None else None,
            "runner_up_peak_kcal": ranked[1][2].score if len(ranked) > 1 else None,
        }
        return StructureNode(structure=best.start), StructureNode(structure=best.end)

    def chain(self, a, b) -> Chain:
        """The first path between `a` and `b`, built as channels builds a pair's."""
        from mepd.interpolation import initial_chain

        ri = self.run_inputs
        seed = Chain.model_validate({"nodes": [a, b], "parameters": copy.deepcopy(ri.chain_inputs)})
        return initial_chain(seed, ri.chain_inputs, ri.gi_inputs)

    def piece_chain(self, chain_frag: Chain, depth: int, index: int) -> Chain:
        """What a split piece is searched from: its best-scoring built pair,
        carrying the piece's lineage (MSMEP's cycle and same-pair limits)."""
        from mepd import search_control
        from mepd.nodes.node import StructureNode

        fresh = lambda n: StructureNode(structure=n.structure)   # MSMEP may have dropped the nodes' graphs
        a, b = fresh(chain_frag[0]), fresh(chain_frag[-1])
        if search_control.search_over(self.run_inputs.path_min_inputs) is not None:
            return chain_frag                       # MSMEP keeps it as it is
        if _same_species(a, b):
            self.records[index] = {"index": index, "depth": depth, "start": _label(a), "end": _label(b),
                                   "chosen": None, "note": "a conformer change: left to MSMEP as split"}
            return chain_frag
        for x, y in ((fresh(x), fresh(y)) for x, y in getattr(chain_frag, "_split_ancestors", None) or []):
            if (_same_species(a, x) and _same_species(b, y)) or (_same_species(a, y) and _same_species(b, x)):
                self.records[index] = {"index": index, "depth": depth, "start": _label(a), "end": _label(b),
                                       "chosen": None, "note": "these two species are already being split higher "
                                       "up this branch (a cycle): searched as split, no new pair"}
                _log(f"MINMAX node {index}: {_label(a)} -> {_label(b)} repeats a pair above it; searched as split.")
                return chain_frag
        start, end = self.choose(a, b, index, depth)
        new = self.chain(start, end)
        for attr in LINEAGE_ATTRS:
            if getattr(chain_frag, attr, None) is not None:
                setattr(new, attr, copy.deepcopy(getattr(chain_frag, attr)))
        return new


class _MinmaxMSMEP(MSMEP):
    def __init__(self, inputs, search: MinmaxSearch):
        super().__init__(inputs=inputs)
        self._minmax = search

    def _piece_chain(self, chain_frag: Chain, tree_depth: int, tree_node_index: int) -> Chain:
        return self._minmax.piece_chain(chain_frag, tree_depth, tree_node_index)


def _verdicts(tree) -> dict:
    """{tree node index: "split" / "elementary" / a leaf status}."""
    out = {}

    def walk(node):
        if node.children:
            out[node.index] = "split"
        elif node.data is not None:
            out[node.index] = "elementary"
        else:
            out[node.index] = getattr(node, "leaf_status", None) or "no path"
        for c in node.children:
            walk(c)

    walk(tree)
    return out


def run_minmax(start_node, end_node, run_inputs, options, output: Path, conformer_inputs, stats: dict,
               write_stats) -> Optional[dict]:
    """The MINMAX search between `start_node` and `end_node`, written to
    `output` as a channels run is (conformers/, pairs/, ts/, channels/, ...,
    stats.json), plus minmax.json and minmax/. With `options.conformers_only`,
    stops after the first pair choice and returns {"start", "end", "pairs"}."""
    from mepd import progress, search_control
    from mepd.channel_search import ChannelSearchError, _discover_channels
    from mepd.cli_common import _completed_tree_dirs
    from mepd.cpus import available_cpus
    from mepd.inputs import ChainInputs

    output = Path(output)
    cpu = available_cpus()
    # 0 (auto): conformers are built from in parallel, up to this many at once.
    workers = int(options.workers) or min(cpu, max(1, int(options.n_conformers or 8)))
    stats["method"] = "minmax"
    stats["workers"] = workers
    search = MinmaxSearch(run_inputs, options, output, conformer_inputs, workers)
    # Builds run `workers` conformers at once: each gets its share of the machine for its batch of minimizations.
    engine = run_inputs.engine
    auto_parallel = getattr(engine, "n_parallel", None) == 0
    if auto_parallel:
        engine.n_parallel = max(1, cpu // workers)

    t0 = time.perf_counter()
    start, end = search.choose(start_node, end_node, 0, 0)
    stats["minmax_root_choice_seconds"] = round(time.perf_counter() - t0, 3)

    # The root's pools, where the web and `mepd visualize` look for a channels run's conformers.
    conf_dir = output / "conformers"
    conf_dir.mkdir(parents=True, exist_ok=True)
    for name, pool in zip(("start", "end"), search.root_pools or ([start_node], [end_node])):
        Chain.model_validate({"nodes": pool, "parameters": ChainInputs()}).write_to_disk(conf_dir / f"{name}.xyz")
    stats["conformers"] = {side: dict(v) for side, v in search.records[0]["conformers"].items()}

    def write_record():
        (output / "minmax.json").write_text(json.dumps(
            {"nodes": [search.records[i] for i in sorted(search.records)]}, indent=2, default=str) + "\n")

    write_record()
    write_stats()
    if options.conformers_only:
        _log("--conformers-only: stopping after the first pair choice.")
        return {"start": search.root_pools[0], "end": search.root_pools[1], "pairs": [(start, end)]}

    if auto_parallel:
        engine.n_parallel = cpu                     # one path search at a time
    stats["engine_parallel"] = getattr(engine, "n_parallel", None)
    max_path_searches = int(options.max_path_searches)
    if max_path_searches > 0:
        search_control.start_path_search_count(run_inputs.path_min_inputs, output / "path_searches.count",
                                               max_path_searches)
        stats["max_path_searches"] = max_path_searches
    budget = float(options.search_budget) if isinstance(options.search_budget, (int, float)) else 0.0
    if budget > 0:
        setattr(run_inputs.path_min_inputs, "recursive_split_deadline", time.time() + budget)
        stats["search_budget_seconds"] = budget

    tree_dir = output / "pairs" / STREAM / "tree"
    tree_dir.mkdir(parents=True, exist_ok=True)
    setattr(run_inputs.path_min_inputs, "attempted_pairs_payload", [])
    t0 = time.perf_counter()
    progress.set_live_stream(STREAM)
    status = "failed"
    try:
        _log(f"MINMAX: searching node 0 ({run_inputs.path_min_method}); a split is re-sampled per piece.")
        history = _MinmaxMSMEP(run_inputs, search).run_recursive_minimize(search.chain(start, end))
        history.write_to_disk(tree_dir)
        status = "done"
    except Exception as exc:
        raise ChannelSearchError(f"The MINMAX path search failed ({type(exc).__name__}: {exc}).") from exc
    finally:
        progress.end_live_stream(status)
        for i, verdict in (_verdicts(history).items() if status == "done" else []):
            search.records.setdefault(i, {"index": i, "chosen": None, "note": "searched as MSMEP split it"})
            search.records[i]["verdict"] = verdict
        write_record()
    stats["msmep_seconds"] = round(time.perf_counter() - t0, 3)
    stats["n_nodes"] = len(search.records)
    stats["n_pairs_chosen"] = sum(1 for r in search.records.values() if r.get("chosen"))
    if max_path_searches > 0:
        stats["path_searches"] = search_control.path_searches(run_inputs.path_min_inputs)
    if search_control.stop_requested():
        stats["stopped_on_request"] = True
    write_stats()

    if not _completed_tree_dirs(output / "pairs"):
        raise ChannelSearchError("The MINMAX path search left no path; nothing to classify.")
    q, m = int(start_node.structure.charge), int(start_node.structure.multiplicity)
    t0 = time.perf_counter()
    warnings = _discover_channels(output, start_node, end_node, run_inputs, q, m, workers=workers)
    if warnings:
        stats["endpoint_mismatch_warnings"] = warnings
    stats["ts_discovery_seconds"] = round(time.perf_counter() - t0, 3)
    write_stats()
    return None
