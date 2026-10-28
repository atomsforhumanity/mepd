"""Channel search between two structures: conformer sampling, per-pair
atom mapping, MSMEP path search, TS optimization + IRC on every leaf, and
classification into channels (one step from start to end), alternate
channels (several steps through intermediates) and off-target exits.

The public entry points are `mepd.channels(start, end, ...)`, which runs a
search and returns a `ChannelsResult`, and `mepd.read_channels(folder)`,
which reads the result of an earlier search (the `mepd channels` command
writes the same folders). A search works in its output folder: it writes
conformers, pair trees and TS/IRC results as it goes, and a rerun into the
same folder resumes. Progress is logged on the "mepd.channel_search" logger
(`logging.basicConfig(level=logging.INFO)` shows it).
"""
from __future__ import annotations

import collections
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mepd.atom_mapping_selection import METRICS as _ATOM_MAPPING_METRICS
from mepd.atom_mapping_metrics import OFFERED as _OFFERED_METRICS
from mepd.chain import Chain
from mepd.nodes.nodehelpers import _is_connectivity_identical
from mepd.cli_common import (
    TsIrcResult,
    _check_endpoint_atom_mapping,
    _collect_ts_guess_tasks,
    _completed_tree_dirs,
    _connectivity_matches,
    _fork_map,
    _geometry_optimizer_keywords,
    _minimize_endpoints,
    _optimize_ts_and_irc,
    PairFollowUp,
    _pair_ts_guess_tasks,
    _run_msmep_pairs,
)
from mepd.errors import InputsError
from mepd.inputs import NetworkInputs, RunInputs

logger = logging.getLogger("mepd.channel_search")


def _log(message: str = "") -> None:
    logger.info(message)


def _warn(message: str) -> None:
    logger.warning(message)


class ChannelSearchError(RuntimeError):
    """A channel search that could not run to the end (no usable conformers,
    no pair search completed)."""

def _describe_mechanism(key: str) -> str:
    """'break C27-C29,C28-O20 | form C27-C28' -> 'break C27-C29 and C28-O20, form C27-C28';
    symmetry-equivalent bonds (the same name) are counted: 'form 2 x C10-C12'."""
    parts = []
    for side in key.split("|"):
        verb, _, bonds = side.strip().partition(" ")
        counts: dict[str, int] = {}
        for b in bonds.split(","):
            if b and b != "none":
                counts[b] = counts.get(b, 0) + 1
        names = [f"{k} x {b}" if k > 1 else b for b, k in counts.items()]
        if names:
            parts.append(f"{verb} " + (" and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]))
    return ", ".join(parts) or "no bond changes"


def _expand_pairs_by_mechanism(
    structures: list,
    candidates: list[tuple[int, int]],
    atom_mapping: bool,
    run_inputs: "RunInputs",
    *,
    pairs_per_mechanism: int = 0,
    workers: int = 1,
    output: Optional[Path] = None,
) -> tuple[list, list[tuple[int, int]], dict]:
    """--atom-mapping for conformer pairs: turn every (reactant, product)
    conformer pair into one path search PER MECHANISM.

    SLAPMapper's equal-minimal-cost mappings can describe different
    chemistry -- for a Claisen, the [3,3] shift and a [1,3] shift tie,
    because bond orders aren't scored -- so a pair is not reduced to one
    mapping. Within each mechanism, the symmetry-equivalent relabelings
    (which of a CH2's hydrogens goes where) are all scored against THIS
    pair's geometry with --atom-mapping-metric and the best is kept, since
    a bad assignment forces e.g. a CH2 rotation onto the path; across
    mechanisms, nothing is thrown away (`select_per_mechanism`).

    `pairs_per_mechanism` > 0 then keeps, for each mechanism, only the pairs
    it scores best on, instead of every pair x every mechanism.

    Each kept (pair, mechanism) gets its own product entry appended to
    `structures` (the conformer pools on disk stay as sampled), and the
    table of what each path search is -- reactant/product conformer,
    mechanism, score -- goes to <output>/pair_mechanisms.json.

    Returns `(structures, candidates, summary)`; a no-op when `atom_mapping`
    is off."""
    if not atom_mapping or not candidates:
        return structures, candidates, {}

    from mepd.atom_mapping_selection import near_lowest, rmsd_window, select_per_mechanism
    from mepd.nodes.node import StructureNode

    metric = run_inputs.atom_mapping_inputs.metric
    from mepd.inputs import MAPPING_CANDIDATE_LIMIT as budget
    # Filtered GI path: every pair by endpoint RMSD first (microseconds each),
    # GI paths only for the pairs near each mechanism's lowest RMSD. Snap +
    # GI + xtb: every pair by snap's pick first, GI + xtb only for the pairs kept.
    filtered = metric == "rmsd-geodesic"
    first = "endpoint-rmsd" if filtered else "snap" if metric == "snap-gi-xtb" else metric
    _log(
        f"Finding the mechanisms of the {len(candidates)} pair(s): every way the reactant's atoms can "
        f"become the product's, grouped by which bonds break and form (per mechanism, the atom "
        f"numbering that fits best by {first} is used)..."
    )

    def _one(pair, metric=first, only_keys=None):
        i, j = pair
        try:
            choices = select_per_mechanism(
                structures[i].structure, structures[j].structure, metric, run_inputs,
                max_variants_per_mechanism=budget, **({"only_keys": only_keys} if only_keys is not None else {}),
            )
        except Exception as exc:
            return pair, None, f"{type(exc).__name__}: {exc}"
        return pair, [(c.key, c.score, c.n_variants, c.winner.end_structure, c.winner.atom_map is None)
                      for c in choices], None

    rows = []  # one per (pair, mechanism)
    n_failed = 0
    for (i, j), choices, error in _fork_map(_one, list(candidates), workers):
        if not choices:
            if error:
                _warn(f"  pair ({i}, {j}): mechanism enumeration failed ({error}); keeping it as is.")
                n_failed += 1
            rows.append({"i": i, "j": j, "key": "unmapped", "score": None,
                         "n_variants": 0, "structure": None})
            continue
        for key, score, n_variants, end_structure, is_identity in choices:
            rows.append({"i": i, "j": j, "key": key, "score": score, "n_variants": n_variants,
                         "structure": None if is_identity else end_structure})

    keys = sorted({r["key"] for r in rows})
    if filtered:
        rows = _gi_near_lowest_rmsd(rows, keys, _one, workers, rmsd_window(run_inputs),
                                    max(1, pairs_per_mechanism), near_lowest)
    if pairs_per_mechanism > 0:
        kept = []
        for key in keys:
            mine = [r for r in rows if r["key"] == key]
            mine.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))
            kept += mine[:pairs_per_mechanism]
        rows = kept
    if metric == "snap-gi-xtb":
        todo = [r for r in rows if r["key"] != "unmapped"]
        _log(f"Choosing each kept pair's numbering among snap's picks by GI + xtb ({len(todo)} pair(s))...")
        for r, (_, choices, _) in zip(todo, _fork_map(lambda r: _one((r["i"], r["j"]), "snap-gi-xtb", {r["key"]}),
                                                     todo, workers)):
            for key, score, _, end_structure, is_identity in choices or []:
                if key == r["key"]:
                    r["structure"] = None if is_identity else end_structure

    # A rerun into the same folder (e.g. sampling more pairs per mechanism)
    # must give every (reactant, product, mechanism) its earlier label: a
    # finished pair is found -- and skipped -- by its folder name, and a label
    # re-used for a different pair would attach its results to the wrong one.
    # Earlier rows keep their product slots; new rows go after them.
    prior = {}
    if output is not None and (output / "pair_mechanisms.json").exists():
        import json as _json

        try:
            for row in _json.loads((output / "pair_mechanisms.json").read_text()):
                pj = int(str(row["pair"]).rsplit("_", 1)[1])
                prior[(int(row["start_conformer"]), int(row["end_structure"]), row["mechanism"])] = pj
        except Exception as exc:
            _warn(f"--atom-mapping: could not read the earlier pair table ({exc}); numbering afresh.")
            prior = {}
    new_structures = list(structures)
    slots = sorted(pj for pj in prior.values() if pj >= len(structures))
    if slots:
        # Hold the earlier product slots (a placeholder until filled below).
        new_structures += [structures[0]] * (slots[-1] - len(structures) + 1)
    new_candidates = []
    for r in rows:
        earlier = prior.get((r["i"], r["j"], r["key"]))
        if r["structure"] is None:
            r["pair_j"] = r["j"]
        elif earlier is not None and earlier >= len(structures):
            new_structures[earlier] = StructureNode(structure=r["structure"])
            r["pair_j"] = earlier
        else:
            new_structures.append(StructureNode(structure=r["structure"]))
            r["pair_j"] = len(new_structures) - 1
        if (r["i"], r["pair_j"]) not in new_candidates:
            # Mechanisms whose chosen numbering is the input's own share one search: run twice, the
            # copies would also write the same pair folder at once.
            new_candidates.append((r["i"], r["pair_j"]))
    if prior:
        reused = sum(1 for r in rows if (r["i"], r["j"], r["key"]) in prior)
        _log(f"--atom-mapping: {reused} path search(es) keep their earlier labels; "
                   f"{len(rows) - reused} are new.")

    counts = {key: sum(1 for r in rows if r["key"] == key) for key in keys}
    _log(f"{len(keys)} mechanism(s) (atoms are named by element and symmetry class, not by number):")
    for k, key in enumerate(keys, start=1):
        _log(f"  {k}. {_describe_mechanism(key)}: {counts[key]} path search(es)")
    _warn(
        f"Path searches: {len(new_candidates)} of the {len(candidates)} pair(s)"
        + (f" -- the {pairs_per_mechanism} best-matching per mechanism (--pairs-per-mechanism)"
           if pairs_per_mechanism > 0 else "")
        + (f"; mapping failed for {n_failed} pair(s)." if n_failed else ".")
    )

    if output is not None:
        import json

        table = [
            {"pair": f"pair_{r['i']}_{r['pair_j']}", "start_conformer": r["i"],
             "end_structure": r["j"], "mechanism": r["key"], "score": r["score"],
             "n_symmetry_variants": r["n_variants"], **({"endpoint_rmsd": r["rmsd"]} if "rmsd" in r else {})}
            for r in rows
        ]
        # Earlier rows this run did not keep (e.g. a lower cap) stay listed, so their labels stay reserved.
        kept_keys = {(r["i"], r["j"], r["key"]) for r in rows}
        for (i, j, key), pj in prior.items():
            if (i, j, key) not in kept_keys:
                table.append({"pair": f"pair_{i}_{pj}", "start_conformer": i, "end_structure": j,
                              "mechanism": key, "score": None, "n_symmetry_variants": 0, "not_in_this_run": True})
        (output / "pair_mechanisms.json").write_text(json.dumps(table, indent=2) + "\n")

    summary = {"n_mechanisms": len(keys), "path_searches_per_mechanism": counts}
    return new_structures, new_candidates, summary


def _gi_near_lowest_rmsd(rows: list[dict], keys: list, score_pair, workers: int, window: float, keep: int,
                         near_lowest) -> list[dict]:
    """--atom-mapping-metric rmsd-geodesic, second stage: per mechanism, the
    (pair, mechanism) rows within `window` standard deviations of the lowest
    endpoint RMSD (at least `keep` of them) get GI paths -- which choose
    their symmetry variant and rank them -- and the rest are dropped.
    On the KAIST direct-only sweep a 1-sigma window held 6% of the pairs and
    56% of what GI paths over every pair would have picked; 2 sigma, 21% and
    82%; 3 sigma, 51% and 96% (see docs/channels_candidates.md)."""
    unmapped = [r for r in rows if r["score"] is None]
    chosen: dict = {}
    n_scored = 0
    for key in keys:
        mine = [r for r in rows if r["key"] == key and r["score"] is not None]
        n_scored += len(mine)
        for k in near_lowest([r["score"] for r in mine], window, keep=keep):
            chosen.setdefault((mine[k]["i"], mine[k]["j"]), {})[key] = mine[k]
    n_chosen = sum(len(v) for v in chosen.values())
    _log(f"Filtered GI path: {n_chosen} of {n_scored} (pair, mechanism) combination(s) are within "
               f"{window:g} standard deviation(s) of their mechanism's lowest endpoint RMSD; computing GI "
               f"paths for those (--atom-mapping-rmsd-window to widen).")
    out = list(unmapped)
    tasks = [(pair, set(by_key)) for pair, by_key in chosen.items()]
    results = _fork_map(lambda t: score_pair(t[0], "rmsd-geodesic", t[1]), tasks, workers)
    for (pair, by_key), (_, choices, error) in zip(chosen.items(), results):
        got = {key: (score, nv, st, is_id) for key, score, nv, st, is_id in (choices or [])}
        for key, row in by_key.items():
            if key in got:
                score, nv, st, is_id = got[key]
                out.append({**row, "rmsd": row["score"], "score": score, "n_variants": nv,
                            "structure": None if is_id else st})
            else:
                # The GI path could not be computed: keep the pair, ranked last.
                out.append({**row, "rmsd": row["score"], "score": None})
                if error:
                    _warn(f"  pair {pair}: GI path failed ({error}); ranked last.")
    return out


def _load_conformer_pool(fp: Path, endpoint, label: str, charge: int, multiplicity: int) -> list:
    """A saved final conformer pool (--start-pool / --end-pool) for
    `endpoint`, with its energies. Only conformers whose bonds match the
    endpoint's atom for atom are kept: a pool saved when this molecule's atoms
    were numbered differently (e.g. it was the other end of another pair, so
    the atom-mapping check reordered it) would pair the wrong atoms."""
    from mepd.inputs import ChainInputs

    if not fp.exists():
        raise InputsError(f"--{label}-pool {fp} does not exist.")
    pool = list(Chain.from_xyz(fp, ChainInputs(), charge=charge, spinmult=multiplicity).nodes)

    def bonds(node):
        return {frozenset(e) for e in node.graph.edges()}

    want_symbols, want_bonds = list(endpoint.symbols), bonds(endpoint)
    kept = [n for n in pool if list(n.symbols) == want_symbols and bonds(n) == want_bonds]
    if len(kept) < len(pool):
        _log(f"  {len(pool) - len(kept)} of {len(pool)} saved {label} conformer(s) are numbered or bonded "
                   f"differently from this run's {label} endpoint; not used.")
    return kept


def _built_pairs_by_mechanism(sources: list, src: str, start_node, end_node, run_inputs: RunInputs, *,
                              workers: int = 1) -> tuple[list, int]:
    """--pair-from with --atom-mapping: each conformer in `sources` (the
    `src` side) builds the other endpoint once per candidate mapping
    (mepd.built_pair_mapping.best_per_mechanism: one build per mechanism,
    equivalent sites only if the given one fails), minimized, dropped if its
    bonds or stereochemistry change. Returns ([(conformer index,
    MechanismPair)], M), M the number of candidate mappings; the pairs are in
    the start's atom order."""
    from mepd.atom_mapping import check_atom_mapping
    from mepd.built_pair_mapping import best_per_mechanism
    from mepd.cli_common import _candidate_atom_maps

    pair_from = "start" if src == "start" else "end"
    atom_map = check_atom_mapping(start_node.structure, end_node.structure)
    if atom_map is None:
        return [], 0
    maps = _candidate_atom_maps(start_node.structure, end_node.structure, atom_map, run_inputs)
    _log(f"  per conformer: the {'product' if src == 'start' else 'reactant'} of each of the candidate "
               "mappings built and minimized...")

    def one(node):
        a, b = (node.structure, end_node.structure) if src == "start" else (start_node.structure, node.structure)
        fallback: list = []    # only the symmetry-swap fallback's lines are shown
        echo = lambda ln: fallback.append(ln.strip()) if "SYMMETRY-SWAP FALLBACK" in ln else None
        try:
            return best_per_mechanism(a, b, maps, pair_from, run_inputs, echo=echo), None, fallback
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}", fallback

    rows = []
    for k, (found, error, fallback) in enumerate(_fork_map(one, list(sources), workers)):
        for ln in fallback:
            _log(f"  conformer {k}: {ln}")
        if error:
            _log(f"  conformer {k}: {error}")
        rows += [(k, r) for r in found]
    return rows, len(maps)


BUILT_PAIRS_JSON = "built_pairs.json"


def _save_built_pairs(folder: Path, ranked: list, searched: set, metric: str) -> None:
    """The ranked built pairs ([(side, conformer index, MechanismPair)]) as
    <folder>/start.xyz and end.xyz (entry k is frame k, in the start's atom
    order) and built_pairs.json: one entry per pair, labeled built_<rank>,
    with whether it has been searched. The interpolation each was scored by
    goes to paths/<label>.json (xyz frames, xtb energies in kcal/mol); one
    saved by an earlier run is kept."""
    from mepd.inputs import ChainInputs
    from mepd.nodes.node import StructureNode

    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(("start", "end")):
        Chain.model_validate({"nodes": [StructureNode(structure=(r.start, r.end)[i]) for _, _, r in ranked],
                              "parameters": ChainInputs()}).write_to_disk(folder / f"{name}.xyz")
    floors = [r.floor for _, _, r in ranked if r.floor is not None]
    floor = min(floors) if floors else None
    (folder / "paths").mkdir(exist_ok=True)
    for n, (_, _, r) in enumerate(ranked):
        if r.path:
            (folder / "paths" / f"built_{n}.json").write_text(json.dumps(
                {"frames": [s.to_xyz() for s in r.path], "energies_kcal": r.path_energies}))
    entries = [{"label": f"built_{n}", "rank": n, "built_from": src, "conformer": k, "mechanism": r.key,
                "mechanism_text": _describe_mechanism(r.key), "score": r.score,
                "barrier_kcal": (r.score - floor) if floor is not None and r.floor is not None else None,
                "n_variants": r.n_variants, "floor": r.floor, "searched": f"built_{n}" in searched,
                "has_path": (folder / "paths" / f"built_{n}.json").exists()}
               for n, (src, k, r) in enumerate(ranked)]
    (folder / BUILT_PAIRS_JSON).write_text(json.dumps(
        {"metric": metric, "floor_kcal": floor, "entries": entries}, indent=2) + "\n")


def _load_built_pairs(folder: Path, charge: int, multiplicity: int) -> tuple[list, set]:
    """What `_save_built_pairs` wrote: (ranked, labels searched already)."""
    from mepd.built_pair_mapping import MechanismPair
    from mepd.inputs import ChainInputs

    fp = folder / BUILT_PAIRS_JSON
    if not fp.exists():
        raise InputsError(f"--built-pairs {folder} has no {BUILT_PAIRS_JSON}.")
    entries = json.loads(fp.read_text())["entries"]
    starts, ends = (list(Chain.from_xyz(folder / f"{name}.xyz", ChainInputs(), charge=charge,
                                        spinmult=multiplicity).nodes) for name in ("start", "end"))
    if not len(entries) == len(starts) == len(ends):
        raise InputsError(f"--built-pairs {folder}: {BUILT_PAIRS_JSON} and its xyz files disagree.")
    ranked = [(e["built_from"], e["conformer"],
               MechanismPair(key=e["mechanism"], score=e["score"], start=a.structure, end=b.structure,
                             n_variants=e["n_variants"], floor=e["floor"]))
              for e, a, b in zip(entries, starts, ends)]
    return ranked, {e["label"] for e in entries if e["searched"]}


def _choose_built_pairs(rows: list, max_searches: int, picks: Optional[set] = None,
                        done: frozenset = frozenset(), per_mechanism: bool = False) -> tuple[list, list, list]:
    """All built pairs, both sides and every mechanism together ([(src,
    conformer index, MechanismPair)]), ranked by score (snap-gi-xtb: the xtb
    peak, so above one shared floor, the lowest start among them). Searched:
    the entries labeled in `picks` (built_<rank>), else the lowest
    `max_searches` (`per_mechanism`: the lowest `max_searches` of each
    mechanism); `done`, labels searched in an earlier run. Returns
    (ranked, ranks searched, report lines)."""
    ranked = sorted(rows, key=lambda t: t[2].score)
    if picks is not None:
        chosen = [n for n in range(len(ranked)) if f"built_{n}" in picks]
    elif per_mechanism:
        seen: collections.Counter = collections.Counter()
        chosen = []
        for n, (_, _, r) in enumerate(ranked):
            seen[r.key] += 1
            if seen[r.key] <= max_searches:
                chosen.append(n)
    else:
        chosen = list(range(min(max_searches, len(ranked))))
    floors = [r.floor for _, _, r in rows if r.floor is not None]
    floor = min(floors) if floors else None
    lines = []
    for n, (src, k, r) in enumerate(ranked):
        value = (f"{r.score - floor:6.1f} kcal/mol" if floor is not None and r.floor is not None
                 else f"{r.score:.4g}")
        built = (f"product built from reactant conformer {k}" if src == "start"
                 else f"reactant built from product conformer {k}")
        status = "searched" if n in chosen else "done    " if f"built_{n}" in done else "skipped "
        lines.append(f"  built_{n:<3} {status}  {value}  {built}: "
                     f"{_describe_mechanism(r.key)}")
    return ranked, chosen, lines


def _minimize_conformer_pool(nodes: list, label: str, run_inputs: RunInputs) -> list:
    """Optimize every node in a conformer pool with the QM engine, dropping
    (with a warning) any conformer that fails to converge or produces an
    empty trajectory rather than failing the whole batch -- unlike
    `_minimize_endpoints` (which hard-stops on a single non-convergent
    endpoint), one bad conformer out of many shouldn't sink the run.
    """
    from mepd.errors import GeometryOptimizationNotConvergedError

    if not nodes:
        return nodes

    keywords = _geometry_optimizer_keywords(run_inputs)
    optimized: list = []

    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch_optimizer):
        try:
            try:
                trajectories = batch_optimizer(nodes, keywords=keywords)
            except TypeError:
                trajectories = batch_optimizer(nodes)
        except Exception as exc:
            _warn(
                f"{label} conformer batch minimization failed "
                f"({type(exc).__name__}: {exc}); keeping input geometries."
            )
            return nodes
        for i, trajectory in enumerate(trajectories):
            if trajectory:
                optimized.append(trajectory[-1])
            else:
                _log(f"{label} conformer {i} optimization returned an empty trajectory; dropping it.")
        return optimized

    single_optimizer = getattr(run_inputs.engine, "compute_geometry_optimization", None)
    if not callable(single_optimizer):
        _log(
            f"Engine {type(run_inputs.engine).__name__} does not support geometry "
            f"optimization; keeping input geometries for {label}."
        )
        return nodes

    for i, node in enumerate(nodes):
        try:
            trajectory = single_optimizer(node, keywords=keywords)
            if trajectory:
                optimized.append(trajectory[-1])
            else:
                _log(f"{label} conformer {i} optimization returned an empty trajectory; dropping it.")
        except GeometryOptimizationNotConvergedError as exc:
            _warn(f"{label} conformer {i} did not converge ({exc}); dropping it.")
        except Exception as exc:
            _warn(f"{label} conformer {i} minimization failed ({type(exc).__name__}: {exc}); dropping it.")
    return optimized


def _register_route_class(known: list, node) -> int:
    """Greedily assign `node` to an existing connectivity class in `known`
    (a list of representative nodes so far), registering a new class if none
    match. Mirrors `NetworkBuilder._get_ind_td`'s registration pattern."""
    for i, rep in enumerate(known):
        if _connectivity_matches(node, rep):
            return i
    known.append(node)
    return len(known) - 1


_TS_FINGERPRINT_TOL_BOHR = 0.1


def _same_ts(a, b, rmsd_cutoff: float, kcal_mol_cutoff: float) -> bool:
    """Whether two optimized TSs are the same saddle point.

    Unlike `is_identical` (index-wise Kabsch fit, plus a perceived-graph
    check), this must see through atom relabeling: every conformer pair gets
    its own --atom-mapping reindexing, so the same TS reached from two pairs
    routinely comes back with equivalent atoms (e.g. a methylene's two H)
    swapped, and mirror-image pairs return its mirror image. So: energies
    within `kcal_mol_cutoff`, then snap-RMSD (permutation-aware) to b or its
    mirror image under `rmsd_cutoff`. snap-RMSD perceives connectivity from
    geometry, which is ambiguous at a saddle point's half-formed bonds (it
    can fail, or align badly: an SN2 TS against its own copy came back at
    5.2 bohr), so the sorted interatomic-distance list, itself invariant to
    permutation, rotation and reflection, is checked first: matching
    distances are the same TS whatever snap-RMSD says."""
    import numpy as np

    from mepd._qcinf_compat import snap_rmsd
    from mepd.conformers import mirror_image

    try:
        if abs(a.energy - b.energy) * 627.5 >= kcal_mol_cutoff:
            return False
    except Exception:
        pass

    def _fingerprint(node):
        g = np.asarray(node.structure.geometry, dtype=float)
        d = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)
        return np.sort(d[np.triu_indices(len(g), 1)])

    if sorted(a.structure.symbols) == sorted(b.structure.symbols) and \
            float(np.abs(_fingerprint(a) - _fingerprint(b)).max()) < _TS_FINGERPRINT_TOL_BOHR:
        return True
    try:
        return min(
            snap_rmsd(a.structure, b.structure),
            snap_rmsd(a.structure, mirror_image(b.structure)),
        ) < rmsd_cutoff
    except Exception:
        return False


def _cluster_by_ts_identity(candidates: list[tuple], run_inputs: RunInputs) -> list[list[tuple]]:
    """Greedily cluster `(ts_node, irc_chain, label)` candidates into classes
    of the same TS (`_same_ts`, with the cutoffs `NetworkBuilder` uses for
    "are these the same structure": node_rms_thre bohr, node_ene_thre
    kcal/mol)."""
    clusters: list[list[tuple]] = []
    for candidate in candidates:
        ts_node = candidate[0]
        for cluster in clusters:
            if _same_ts(
                ts_node, cluster[0][0],
                rmsd_cutoff=run_inputs.chain_inputs.node_rms_thre,
                kcal_mol_cutoff=run_inputs.chain_inputs.node_ene_thre,
            ):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    return clusters


def _load_ts_and_irc_from_disk(
    ts_dir: Path, label: str, charge: int, multiplicity: int
) -> Optional["TsIrcResult"]:
    """Resume support for `channels`: if a prior run already wrote this
    label's TS (and IRC) to disk, reload them instead of skip-and-forget --
    unlike `ts`'s simpler "already done" skip, `channels` needs the actual
    IRC endpoints to classify this result, not just a done/not-done flag."""
    from mepd.inputs import ChainInputs

    ts_path = ts_dir / f"{label}.xyz"
    if not ts_path.exists():
        return None
    ts_chain = Chain.from_xyz(ts_path, ChainInputs(), charge=charge, spinmult=multiplicity)
    ts_node = ts_chain[0]

    irc_path = ts_dir / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
    irc_chain = None
    if irc_path.exists():
        irc_chain = Chain.from_xyz(irc_path, ChainInputs(), charge=charge, spinmult=multiplicity)
    return TsIrcResult(ts_node=ts_node, irc_chain=irc_chain)


def _write_classified_group(
    candidates: list[tuple],
    out_dir: Path,
    prefix: str,
    index: int,
    extra_info: Optional[str] = None,
) -> None:
    from mepd.inputs import ChainInputs

    group_dir = out_dir / f"{prefix}_{index}"
    group_dir.mkdir(parents=True, exist_ok=True)

    def _ts_energy(candidate):
        try:
            return candidate[0].energy
        except Exception:
            return float("inf")

    ts_node, irc_chain, _ = min(candidates, key=_ts_energy)
    Chain.model_validate(
        {"nodes": [ts_node], "parameters": ChainInputs()}
    ).write_to_disk(group_dir / "ts.xyz")
    irc_chain.write_to_disk(group_dir / "irc.xyz")

    lines = [extra_info] if extra_info else []
    lines.extend(label for _, _, label in candidates)
    (group_dir / "members.txt").write_text("\n".join(lines) + "\n")


_MAX_ALTERNATE_CHANNEL_STEPS = 6
_MAX_ALTERNATE_CHANNELS = 200


def _species_label(node) -> str:
    """Best-effort SMILES for a discovered minimum, to name it in the
    human-readable `path.txt`/`members.txt` sidecars.

    Prefers whatever identifiers the structure already carries, and
    otherwise derives one from the perceived bond graph: a structure
    reloaded from a bare xyz -- which is exactly what resuming a run gives
    you -- has no identifiers at all, and a route written out as
    "? -> ? -> ?" tells you nothing about the chemistry it found."""
    from mepd.NetworkBuilder import _stereochemical_smiles_key

    try:
        key = _stereochemical_smiles_key(node)
        if key:
            return key
    except Exception:
        pass
    try:
        smiles = getattr(getattr(node, "graph", None), "smiles", "")
        if smiles:
            return str(smiles)
    except Exception:
        pass
    return "?"


def _cluster_ts_energy(cluster: list) -> float:
    """Energy of a TS class, for ranking distinct classes of the same step."""
    try:
        return cluster[0][0].energy
    except Exception:
        return float("inf")


def _write_multistep_group(
    out_dir: Path,
    prefix: str,
    index: int,
    species_path: list,
    step_clusters: list,
    species_labels: list,
) -> None:
    """Write one multistep route -- an *alternate channel* -- as a `path.txt`
    naming the species it walks through plus one `step_<n>/` folder per
    elementary step, each holding that leg's TS/IRC in exactly the layout a
    single-step group uses.

    `step_clusters[n]` is every distinct TS class found for leg n. The
    lowest-barrier one defines the route and goes in `step_<n>/` itself; a
    leg found with genuinely different TSs keeps the rest alongside it in
    `step_<n>/alternate_ts_<m>/` rather than dropping them -- they are real
    mechanisms for that step, just not the cheapest one."""
    group_dir = out_dir / f"{prefix}_{index}"
    group_dir.mkdir(parents=True, exist_ok=True)

    arrow = "\n  -> ".join(species_labels[c] for c in species_path)
    lines = [
        f"{len(step_clusters)} step(s), {len(species_path) - 2} intermediate(s)",
        f"  {arrow}",
    ]
    for n, clusters in enumerate(step_clusters):
        ranked = sorted(clusters, key=_cluster_ts_energy)
        extra_info = (
            f"connects: {species_labels[species_path[n]]} "
            f"<-> {species_labels[species_path[n + 1]]}"
        )
        _write_classified_group(ranked[0], group_dir, "step", n, extra_info=extra_info)
        if len(ranked) > 1:
            lines.append(
                f"  step_{n}: {len(ranked) - 1} further distinct TS class(es) "
                f"for this step in step_{n}/alternate_ts_*/"
            )
        for m, alternate in enumerate(ranked[1:]):
            _write_classified_group(
                alternate, group_dir / f"step_{n}", "alternate_ts", m,
                extra_info=extra_info,
            )
    (group_dir / "path.txt").write_text("\n".join(lines) + "\n")


def _ts_opt_failed_path(ts_dir: Path, label: str) -> Path:
    return ts_dir / f"{label}_tsopt_failed.txt"


def _ts_opt_failed(ts_dir: Path, label: str) -> bool:
    return _ts_opt_failed_path(ts_dir, label).exists()


def _ts_and_irc_once(node, label: str, ts_dir: Path, run_inputs: RunInputs, charge: int, multiplicity: int):
    """A leaf's TS and IRC: from disk if done before, else computed (and
    written). A TS optimization that failed is recorded, so a rerun into the
    same folder (Sample more paths) does not spend the same time failing
    again; None then."""
    result = _load_ts_and_irc_from_disk(ts_dir, label, charge, multiplicity)
    if result is not None:
        return result
    if _ts_opt_failed(ts_dir, label):
        return None
    result = _optimize_ts_and_irc(node, run_inputs, ts_dir, run_irc=True, label=label)
    if result is None:
        ts_dir.mkdir(parents=True, exist_ok=True)
        _ts_opt_failed_path(ts_dir, label).write_text("transition-state optimization failed\n")
    return result


def _discover_channels(
    output: Path,
    start_node,
    end_node,
    run_inputs: RunInputs,
    charge: Optional[int],
    multiplicity: Optional[int],
    workers: int = 1,
) -> None:
    """TS-opt + IRC every leaf across every completed pair tree in `output`,
    then classify by walking the reaction graph those IRCs span.

    Every converged leaf contributes one elementary step: an edge between
    the connectivity classes of the two minima its IRC relaxes into. Leaves
    whose TS-opt or IRC failed to converge, or whose IRC collapsed into the
    same minimum on both sides, are dropped. Seeding that graph's species
    list with the requested --start/--end pair means the edges then sort
    into three buckets:

    * `channels/` -- a single elementary step straight from --start to
      --end.
    * `alternate-channels/` -- a *multistep* route from --start to --end: a
      simple path through one or more discovered intermediates, every leg
      of it an IRC-verified step. One folder per route, `step_<n>/` per leg.
    * `offtarget-exit-channels/` -- a step off some discovered minimum for
      which no onward sequence of steps ever reaches --end. A TS to an
      intermediate we never found a way forward from is an exit off the
      reaction path, not a route to the requested product.

    Within a bucket, candidates are further deduplicated into distinct TS
    classes -- this is what turns e.g. 4 conformer-pair runs into "2
    channels" when 2 of those runs converged to essentially the same TS.
    """
    import itertools

    import networkx as nx

    tree_charge = charge if charge is not None else 0
    tree_multiplicity = multiplicity if multiplicity is not None else 1

    tasks = _collect_ts_guess_tasks(output, charge, multiplicity)
    if not tasks:
        _log("No TS guesses found across completed pairs; no channels to classify.")
        return

    ts_dir = output / "ts"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Species classes, seeded so that --start is class 0 and --end is class 1;
    # every IRC endpoint is then registered against the same list, which is
    # what lets a multi-leg route be recognized as reaching the real product.
    species: list = []
    start_cls = _register_route_class(species, start_node)
    end_cls = _register_route_class(species, end_node)
    degenerate_pair = start_cls == end_cls
    if degenerate_pair:
        _warn(
            "WARNING: --start and --end have identical connectivity; no "
            "channel can be defined for this pair, so every discovered step "
            "is reported as an off-target exit channel."
        )

    edge_candidates: dict = {}
    n_failed = 0

    # TS-opt + IRC every leaf not already on disk (most were done in the
    # pair workers as each search finished), `workers` at a time. Each leaf
    # writes only its own <label>*.xyz, and the classification loop below
    # reads everything back from disk, so it is identical either way.
    pending = [
        (label, node) for label, node in tasks
        if _load_ts_and_irc_from_disk(ts_dir, label, tree_charge, tree_multiplicity) is None
        and not _ts_opt_failed(ts_dir, label)
    ]
    if workers > 1 and len(pending) > 1:
        _log(
            f"Optimizing {len(pending)} TS guess(es) across "
            f"{min(workers, len(pending))} worker process(es)."
        )

        def _opt(task):
            label, node = task
            return _ts_and_irc_once(node, label, ts_dir, run_inputs, tree_charge, tree_multiplicity)

        _fork_map(_opt, pending, workers)

    for label, guess_node in tasks:
        result = _ts_and_irc_once(guess_node, label, ts_dir, run_inputs, tree_charge, tree_multiplicity)
        if result is None or result.irc_chain is None or len(result.irc_chain) < 2:
            n_failed += 1
            continue

        irc_first, irc_last = result.irc_chain[0], result.irc_chain[-1]
        i = _register_route_class(species, irc_first)
        j = _register_route_class(species, irc_last)
        if i == j:
            # IRC collapsed to the same minimum on both sides -- not a
            # genuine two-minima step.
            n_failed += 1
            continue
        edge_candidates.setdefault(frozenset((i, j)), []).append(
            (result.ts_node, result.irc_chain, label)
        )

    # One graph edge per pair of species; each edge carries the distinct TS
    # classes found for that step.
    edge_steps: dict = {
        key: _cluster_by_ts_identity(cands, run_inputs)
        for key, cands in edge_candidates.items()
    }
    species_labels = [_species_label(node) for node in species]
    # A step that reaches a stereo variant of --start/--end (same bonds,
    # different stereochemistry) is not what was queried: say so, rather
    # than leave it to be read as an off-target exit or a missing channel.
    mismatches = []
    for key in edge_steps:
        for k in key:
            if k in (start_cls, end_cls):
                continue
            for q, name in ((start_node, "--start"), (end_node, "--end")):
                if _is_connectivity_identical(species[k], q, verbose=False, collect_comparison=False,
                                              disregard_stereochem=True):
                    mismatches.append(f"a step reaches {species_labels[k]}, a stereo variant of {name} "
                                      f"({_species_label(q)}), not {name} itself")
    mismatch_warnings = list(dict.fromkeys(mismatches))
    for msg in mismatch_warnings:
        _warn(f"WARNING: result does not match the queried endpoints: {msg}.")

    graph = nx.Graph()
    graph.add_nodes_from(range(len(species)))
    graph.add_edges_from(tuple(sorted(key)) for key in edge_steps)

    # Every simple --start -> --end path; a 1-edge path is a direct channel,
    # anything longer is an alternate (multistep) channel.
    routes: list = []
    if not degenerate_pair and graph.has_node(start_cls) and graph.has_node(end_cls):
        routes = list(
            itertools.islice(
                nx.all_simple_paths(
                    graph, start_cls, end_cls, cutoff=_MAX_ALTERNATE_CHANNEL_STEPS
                ),
                _MAX_ALTERNATE_CHANNELS,
            )
        )
    routes.sort(key=len)

    on_route_edges: set = set()
    multistep_routes: list = []
    for path in routes:
        keys = [frozenset((path[n], path[n + 1])) for n in range(len(path) - 1)]
        on_route_edges.update(keys)
        if len(keys) > 1:
            multistep_routes.append((path, keys))

    direct_key = frozenset((start_cls, end_cls))
    channel_clusters = [] if degenerate_pair else edge_steps.get(direct_key, [])
    for k, cluster in enumerate(channel_clusters):
        _write_classified_group(cluster, output / "channels", "channel", k)

    alt_dir = output / "alternate-channels"
    for k, (path, keys) in enumerate(multistep_routes):
        _write_multistep_group(
            alt_dir, "alternate_channel", k, path,
            [edge_steps[key] for key in keys], species_labels,
        )

    offtarget_dir = output / "offtarget-exit-channels"
    n_offtarget = 0
    for key in edge_steps:
        if key in on_route_edges:
            continue
        i, j = sorted(key)
        for cluster in edge_steps[key]:
            _write_classified_group(
                cluster, offtarget_dir, "offtarget_exit_channel", n_offtarget,
                extra_info=f"connects: {species_labels[i]} <-> {species_labels[j]}",
            )
            n_offtarget += 1

    # Off-target results: under --direct-only every leg that ran was meant
    # to join --start and --end, so each step that doesn't is a mismatch;
    # in any mode, a run that found nothing but off-target steps is one.
    offtarget_keys = [key for key in edge_steps if key not in on_route_edges]
    if offtarget_keys and (getattr(run_inputs.path_min_inputs, "direct_only", False)
                           or not (channel_clusters or multistep_routes)):
        for key in offtarget_keys:
            i, j = sorted(key)
            mismatch_warnings.append(
                f"an IRC-verified step connects {species_labels[i]} <-> {species_labels[j]}, "
                f"not the queried {species_labels[start_cls]} <-> {species_labels[end_cls]}")
            _warn(f"WARNING: result does not match the queried endpoints: {mismatch_warnings[-1]}.")

    n_contributing = sum(len(c) for c in channel_clusters)
    _log(
        f"{len(channel_clusters)} channel(s) found for the requested pair "
        f"({n_contributing} contributing run(s)), "
        f"{len(multistep_routes)} alternate channel(s), "
        f"{n_offtarget} off-target exit channel(s), {n_failed} failed."
    )
    return mismatch_warnings


@dataclass
class ChannelsOptions:
    """Settings of a channel search. Each field is the `mepd channels` option
    of the same name (underscores for dashes, e.g. `n_conformers` is
    `--n-conformers`), with the same default; `mepd channels --help`
    describes each one."""
    method: str = 'conformers'
    atom_mapping: bool = True
    debug_dump: bool = False
    atom_mapping_candidates: int = 3
    explore_mechanisms: int = 0
    relax_mechanisms: Optional[int] = None
    relax_margin: Optional[float] = None
    atom_mapping_metric: str = 'snap-gi-xtb'
    atom_mapping_rmsd_window: Optional[float] = None
    atom_mapping_gi_variants: Optional[int] = None
    atom_mapping_veto_margin: float = 0.0
    rebuild_splits_from: str = 'none'
    atom_mapping_recheck_splits: bool = False
    backend: str = 'rdkit'
    crest_method: str = '--gfn2'
    crest_threads: int = 1
    crest_ewin: float = 6.0
    crest_timeout: float = 3600.0
    crest_nci: bool = True
    complex_energy_tol: float = 0.5
    n_conformers: int = 50
    n_embed: int = 0
    rdkit_torsion_prefs: str = 'both'
    rdkit_ewin: Optional[float] = None
    pair_from: str = 'both'
    built_pairs: Optional[Path] = None
    built_cap_per_mechanism: bool = False
    search_built: Optional[str] = None
    rmsd_cutoff: float = 0.1
    random_seed: int = 0
    start_pool: Optional[Path] = None
    end_pool: Optional[Path] = None
    minimize_ends: bool = True
    max_pairs: int = 0
    conformers_only: bool = False
    parallel: bool = True
    parallel_workers: Optional[int] = None
    pairs_per_mechanism: int = 3
    workers: int = 0
    search_budget: float = 0.0
    direct_only: Optional[bool] = None
    validate_minima_with_hessian: bool = True
    hessian_minimum_frequency_cutoff: float = -20.0
    hessian_minima_rescue_displacement: float = 0.3

    @property
    def sampled_sides(self) -> tuple[str, ...]:
        return {"both": ("start", "end"), "start+end": ("start", "end"),
                "start": ("start",), "end": ("end",)}[self.pair_from]

    def validate(self) -> None:
        """Raise InputsError (saying what to change) for settings that cannot work."""
        if self.method != "conformers":
            raise InputsError(f"Unknown --method '{self.method}'. Known: 'conformers'.")
        if self.backend not in ("rdkit", "crest"):
            raise InputsError(f"Unknown --backend '{self.backend}'. Known: 'rdkit', 'crest'.")
        if self.n_conformers < 0:
            raise InputsError("--n-conformers must be a non-negative integer (0 = no cap).")
        if self.n_embed < 0:
            raise InputsError("--n-embed must be a non-negative integer (0 = auto).")
        sampled_sides = {"both": ("start", "end"), "start+end": ("start", "end"),
                         "start": ("start",), "end": ("end",)}.get(self.pair_from)
        if sampled_sides is None:
            raise InputsError("--pair-from must be both, start, end or start+end.")
        if self.rmsd_cutoff <= 0:
            raise InputsError("--rmsd-cutoff must be a positive number.")
        if self.max_pairs < 0:
            raise InputsError("--max-pairs must be a non-negative integer (0 = no cap).")
        if self.rdkit_ewin is not None and self.rdkit_ewin <= 0:
            raise InputsError("--rdkit-ewin must be positive.")
        if self.atom_mapping_metric not in _ATOM_MAPPING_METRICS:
            raise InputsError(
                f"--atom-mapping-metric must be one of {', '.join(_OFFERED_METRICS)}."
            )
        if self.atom_mapping_rmsd_window is not None and self.atom_mapping_rmsd_window < 0:
            raise InputsError("--atom-mapping-rmsd-window must be 0 or more (standard deviations).")
        if self.atom_mapping_candidates <= 0:
            raise InputsError("--atom-mapping-candidates must be a positive integer.")
        if self.search_built and self.built_pairs is None:
            raise InputsError("--search-built picks entries of a saved list: give it --built-pairs.")
        if self.built_pairs is not None and (self.pair_from == "both" or not self.atom_mapping):
            raise InputsError("--built-pairs is the list of a --pair-from start/end/start+end run with "
                                     "--atom-mapping: give those too.")
        if self.crest_threads <= 0:
            raise InputsError("--crest-threads must be a positive integer.")
        if self.workers < 0:
            raise InputsError("--workers must be non-negative (0 = auto).")
        if self.pairs_per_mechanism < 0:
            raise InputsError("--pairs-per-mechanism must be non-negative (0 = all pairs).")
        if self.crest_timeout <= 0:
            raise InputsError("--crest-timeout must be positive.")
        if self.rdkit_torsion_prefs not in ("both", "etkdg", "none"):
            raise InputsError("--rdkit-torsion-prefs must be 'both', 'etkdg' or 'none'.")
        if self.complex_energy_tol < 0:
            raise InputsError("--complex-energy-tol must be non-negative (0 disables).")
        if self.rebuild_splits_from not in ("none", "start", "end"):
            raise InputsError("--rebuild-splits-from must be none, start or end.")
        if self.search_budget < 0:
            raise InputsError("--search-budget must be >= 0.")


def apply_options(run_inputs: RunInputs, options: ChannelsOptions) -> None:
    """Carry the search settings that live in the profile (path search, atom
    mapping) into `run_inputs`."""
    if isinstance(options.direct_only, bool):
        run_inputs.path_min_inputs.direct_only = options.direct_only
    run_inputs.path_min_inputs.validate_minima_with_hessian = options.validate_minima_with_hessian
    run_inputs.path_min_inputs.hessian_minimum_frequency_cutoff = options.hessian_minimum_frequency_cutoff
    run_inputs.path_min_inputs.hessian_minima_rescue_displacement = options.hessian_minima_rescue_displacement
    run_inputs.atom_mapping_inputs.n_candidates = options.atom_mapping_candidates
    run_inputs.atom_mapping_inputs.metric = options.atom_mapping_metric
    if options.atom_mapping_rmsd_window is not None:
        run_inputs.atom_mapping_inputs.rmsd_window = options.atom_mapping_rmsd_window
    if isinstance(options.atom_mapping_gi_variants, int):
        run_inputs.atom_mapping_inputs.gi_variant_cap = max(0, options.atom_mapping_gi_variants)
    run_inputs.atom_mapping_inputs.veto_margin = options.atom_mapping_veto_margin
    run_inputs.atom_mapping_inputs.recheck_on_split = options.atom_mapping_recheck_splits
    run_inputs.atom_mapping_inputs.explore_mechanisms = max(0, options.explore_mechanisms) if isinstance(options.explore_mechanisms, int) else 0
    if isinstance(options.relax_mechanisms, int):
        run_inputs.atom_mapping_inputs.relax_top = max(0, options.relax_mechanisms)
    if isinstance(options.relax_margin, (int, float)):
        run_inputs.atom_mapping_inputs.relax_margin = max(0.0, float(options.relax_margin))
    run_inputs.atom_mapping_inputs.rebuild_on_split = None if options.rebuild_splits_from == "none" else options.rebuild_splits_from


def run_channels(
    start_node,
    end_node,
    run_inputs: RunInputs,
    options: ChannelsOptions,
    output: Path,
    *,
    endpoints_are_guesses: bool = False,
) -> None:
    """Run a channel search between `start_node` and `end_node` (same atoms;
    their charge and multiplicity are the search's) with `run_inputs`, in the
    folder `output`. `endpoints_are_guesses`: they were embedded from SMILES,
    so (with `minimize_ends`) they are minimized before anything else.
    Raises InputsError for settings that cannot work and ChannelSearchError
    when the search cannot run to the end. `read_channels(output)` reads what
    it found."""
    options.validate()
    output = Path(output)
    sampled_sides = options.sampled_sides
    method = options.method
    atom_mapping = options.atom_mapping
    debug_dump = options.debug_dump
    backend = options.backend
    crest_method = options.crest_method
    crest_threads = options.crest_threads
    crest_ewin = options.crest_ewin
    crest_timeout = options.crest_timeout
    crest_nci = options.crest_nci
    complex_energy_tol = options.complex_energy_tol
    n_conformers = options.n_conformers
    n_embed = options.n_embed
    rdkit_torsion_prefs = options.rdkit_torsion_prefs
    rdkit_ewin = options.rdkit_ewin
    pair_from = options.pair_from
    built_pairs = options.built_pairs
    built_cap_per_mechanism = options.built_cap_per_mechanism
    search_built = options.search_built
    rmsd_cutoff = options.rmsd_cutoff
    random_seed = options.random_seed
    start_pool = options.start_pool
    end_pool = options.end_pool
    minimize_ends = options.minimize_ends
    max_pairs = options.max_pairs
    conformers_only = options.conformers_only
    parallel = options.parallel
    parallel_workers = options.parallel_workers
    pairs_per_mechanism = options.pairs_per_mechanism
    workers = options.workers
    search_budget = options.search_budget

    import os
    import time

    from mepd.conformers import (
        ConformerInputs, CrestInputs, _subselect_conformers,
        merge_degenerate_complex_conformers, merge_mirror_images,
    )
    from mepd.sampling import generate_seed_pairs
    from mepd.nodes.node import StructureNode
    from mepd.NetworkBuilder import NetworkBuilder

    apply_options(run_inputs, options)


    if minimize_ends and endpoints_are_guesses:
        _log(
            "An endpoint was given as SMILES (an RDKit/openbabel-embedded guess, "
            "not a real minimum); minimizing endpoints at the input level of "
            "theory before checking the --start/--end atom mapping and "
            "sampling conformers. Pass --no-minimize-ends to skip."
        )
        start_node, end_node = _minimize_endpoints(start_node, end_node, run_inputs)
        if debug_dump:
            from mepd.inputs import ChainInputs

            debug_dir = Path(output) / "smiles_minimization_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            Chain.model_validate(
                {"nodes": [start_node], "parameters": ChainInputs()}
            ).write_to_disk(debug_dir / "start_minimized.xyz")
            Chain.model_validate(
                {"nodes": [end_node], "parameters": ChainInputs()}
            ).write_to_disk(debug_dir / "end_minimized.xyz")
            _log(f"  --debug-dump: wrote minimized SMILES endpoints to {debug_dir}")

    # The --atom-mapping sanity check compares geodesic-interpolation path
    # energies between the unpermuted and permuted --end -- run it AFTER
    # minimization (rather than on the raw SMILES embedding) so that
    # comparison reflects real minima, not embedding artifacts/strain that
    # can otherwise dominate the energy delta the veto decision is based on.
    # With an endpoint built from the other (--pair-from), the mapping is
    # chosen per conformer by the built pairs (_built_pairs_by_mechanism).
    if not (atom_mapping and pair_from != "both"):
        realigned_end_structure = _check_endpoint_atom_mapping(
            start_node.structure, end_node.structure, atom_mapping, run_inputs,
            debug_dump=debug_dump, output=output,
        )
        if realigned_end_structure is not end_node.structure:
            end_node = StructureNode(structure=realigned_end_structure)

    conformer_inputs = ConformerInputs(
        backend=backend,
        n_conformers=n_conformers or None,
        n_embed=n_embed or None,
        rdkit_ewin_kcal=rdkit_ewin,
        rdkit_torsion_prefs=rdkit_torsion_prefs,
        rmsd_cutoff=rmsd_cutoff,
        random_seed=random_seed,
        crest=CrestInputs(
            method=crest_method,
            threads=crest_threads,
            ewin_kcal=crest_ewin,
            timeout_s=crest_timeout,
            nci_for_complexes=crest_nci,
        ),
    )

    # Per-stage yield and wall time, rewritten after every stage so a run
    # that dies (or a --conformers-only run) still leaves what it measured.
    run_started = time.perf_counter()
    search_budget = float(search_budget) if isinstance(search_budget, (int, float)) else 0.0
    search_deadline = time.time() + search_budget if search_budget > 0 else None
    stats: dict = {"backend": backend, "workers": workers, "conformers": {}}
    output.mkdir(parents=True, exist_ok=True)

    def _write_stats() -> None:
        stats["total_seconds"] = round(time.perf_counter() - run_started, 3)
        (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # Saved final pools of an earlier run (same sampler settings and level of
    # theory; the caller decides): used as is, not re-sampled or re-minimized.
    endpoints = {"start": start_node, "end": end_node}
    reused: dict = {}
    for label, fp in (("start", start_pool), ("end", end_pool)):
        if not isinstance(fp, (str, Path)):   # unset (None, or typer's default when called directly)
            continue
        node = endpoints[label]
        pool = _load_conformer_pool(Path(fp), node, label, node.structure.charge, node.structure.multiplicity)
        if pool:
            reused[label] = pool
            stats["conformers"][label] = {"reused_from": str(fp), "n_reused": len(pool)}
            _log(f"Using {len(pool)} saved {label} conformer(s) from {fp} (not re-sampled or re-minimized).")
        else:
            _log(f"No usable saved {label} conformers in {fp}; sampling them instead.")
    if len(reused) == 2:
        start_confs, end_confs = reused["start"], reused["end"]
    else:
        _log(f"Sampling conformers ({conformer_inputs.backend})...")
        if not reused:
            start_confs, end_confs = generate_seed_pairs(
                method, start_node, end_node, conformer_inputs=conformer_inputs,
                stats=stats["conformers"],
            )
        else:
            from mepd.conformers import generate_conformers

            (label,) = {"start", "end"} - set(reused)
            sampled = generate_conformers(endpoints[label], conformer_inputs,
                                          stats["conformers"].setdefault(label, {}))
            start_confs = reused.get("start", sampled)
            end_confs = reused.get("end", sampled)
        _log(f"  {len(start_confs)} reactant and {len(end_confs)} product conformer(s).")
        for label, name in (("start", "reactant"), ("end", "product")):
            skipped = stats["conformers"].get(label, {}).get("rdkit_skipped")
            if skipped and label not in reused:
                _warn(f"  WARNING: {name} conformers not sampled, only the input geometry kept ({skipped}).")

    pools = {"start": start_confs, "end": end_confs}
    for label in ("start", "end"):
        side = stats["conformers"].setdefault(label, {})
        if minimize_ends and label not in reused and label in sampled_sides:
            side_name = "reactant" if label == "start" else "product"
            _log(f"Minimizing the {len(pools[label])} {side_name} conformer(s) at the profile's "
                       "level of theory...")
            t0 = time.perf_counter()
            pools[label] = _minimize_conformer_pool(pools[label], label, run_inputs)
            side["minimize_seconds"] = round(time.perf_counter() - t0, 3)
            side["n_minimized"] = len(pools[label])
            # Distinct starting geometries routinely relax into the same QM
            # minimum; every duplicate left in would multiply the pair count
            # with runs that repeat each other, so dedup again at the level
            # the paths will actually be computed at.
            pools[label] = _subselect_conformers(pools[label], n_max=None, rmsd_cutoff=rmsd_cutoff)
            dropped = side["n_minimized"] - len(pools[label])
            if dropped:
                _log(
                    f"  Dropped {dropped}: relaxed into the same minimum as a kept one "
                    f"(RMSD < {rmsd_cutoff} bohr)."
                )
            # A loosely bound complex's conformers differ mostly in how far
            # apart the molecules sit; compare molecule by molecule instead.
            before = len(pools[label])
            pools[label] = merge_degenerate_complex_conformers(
                pools[label], rmsd_cutoff, complex_energy_tol,
            )
            side["n_complex_degenerate_merged"] = before - len(pools[label])
            if side["n_complex_degenerate_merged"]:
                _log(
                    f"  Dropped {side['n_complex_degenerate_merged']}: the same molecules as a kept one, "
                    f"only placed differently (each molecule matches, energy within "
                    f"{complex_energy_tol} kcal/mol)."
                )
            _log(f"  Kept {len(pools[label])}.")

    # Each side's final pool before mirror images are merged (a merge drops
    # conformers from one side only, depending on the other side): what a
    # later run with the same settings can take as its --start/--end-pool.
    from mepd.inputs import ChainInputs as _ChainInputs

    (output / "conformers").mkdir(parents=True, exist_ok=True)
    for label in ("start", "end"):
        if pools[label] and label in sampled_sides:   # a built side is never a sampled pool
            Chain.model_validate({"nodes": pools[label], "parameters": _ChainInputs()}).write_to_disk(
                output / "conformers" / f"{label}_pool.xyz")

    # Mirror-image conformers of an achiral molecule give mirror-image paths,
    # so pairs built from them repeat each other -- but they can only be
    # merged in ONE of the two pools. With {R, R*} x {P, P*}, (R, P) mirrors
    # (R*, P*) while (R, P*) mirrors (R*, P): two genuinely different
    # combinations, and merging both pools would keep only (R, P) and lose
    # the other. Merging one pool drops exactly the redundant pairs; pick
    # whichever leaves fewer.
    merged = {label: merge_mirror_images(pools[label], rmsd_cutoff) for label in pools}
    n_if = {
        "start": len(merged["start"]) * len(pools["end"]),
        "end": len(pools["start"]) * len(merged["end"]),
    }
    mirror_side = min(n_if, key=n_if.get)
    n_mirrors = len(pools[mirror_side]) - len(merged[mirror_side])
    if n_mirrors:
        _log(
            f"Merged {n_mirrors} {'reactant' if mirror_side == 'start' else 'product'} conformer(s) that are "
            f"mirror images of kept ones (the molecule is achiral, so their paths would be mirror images too): "
            f"{len(pools['start']) * len(pools['end'])} -> {n_if[mirror_side]} pairs."
        )
        pools[mirror_side] = merged[mirror_side]
    for label in pools:
        stats["conformers"][label]["n_mirror_images_merged"] = n_mirrors if label == mirror_side else 0
        stats["conformers"][label]["n_final"] = len(pools[label])
    built_chosen = None   # --pair-from with --atom-mapping: which of the built pairs (all in the pools) to search
    if pair_from != "both":
        from mepd.discovery.network_expansion import build_partners

        t0 = time.perf_counter()
        sampled_pools = dict(pools)
        pairs, n_built, rows, n_maps = [], {}, [], 0
        searched_before: set = set()
        if built_pairs is not None:
            rows, searched_before = _load_built_pairs(Path(built_pairs), start_node.structure.charge,
                                                      start_node.structure.multiplicity)
            _log(f"--built-pairs: {len(rows)} ranked built pair(s) from {built_pairs}, not built again "
                       f"({len(searched_before)} searched already).")
        for src in (sampled_sides if built_pairs is None else ()):
            other = "end" if src == "start" else "start"
            _log(f"--pair-from {pair_from}: building a {'product' if other == 'end' else 'reactant'} in the "
                       f"frame of each of the {len(pools[src])} {'reactant' if src == 'start' else 'product'} "
                       "conformer(s)...")
            if atom_mapping:
                found, m = _built_pairs_by_mechanism(pools[src], src, start_node, end_node, run_inputs,
                                                     workers=max(1, workers))
                rows += [(src, k, r) for k, r in found]
                n_maps, n_built[src] = max(n_maps, m), len(found)
                _log(f"  {len(found)} built.")
                continue
            kept, built = build_partners(pools[src], end_node if other == "end" else start_node, run_inputs)
            pairs += list(zip(kept, built)) if src == "start" else list(zip(built, kept))
            n_built[src] = len(built)
            _log(f"  {len(built)} built (each paired with the conformer it was built from).")
        stats["pair_from"] = {"mode": pair_from, "n_built_from": n_built,
                              "seconds": round(time.perf_counter() - t0, 3)}
        if rows:
            # At most (N_reactant + N_product) x M built pairs; past
            # --atom-mapping-candidates, the lowest by xtb peak are searched.
            max_searches = run_inputs.atom_mapping_inputs.n_candidates
            picks = {x.strip() for x in search_built.split(",") if x.strip()} if search_built else None
            ranked, built_chosen, lines = _choose_built_pairs(rows, max_searches, picks, frozenset(searched_before),
                                                            per_mechanism=built_cap_per_mechanism)
            unknown = sorted((picks or set()) - {f"built_{n}" for n in range(len(ranked))})
            if unknown:
                raise InputsError(f"--search-built: {', '.join(unknown)} not in {built_pairs} "
                                         f"(built_0 to built_{len(ranked) - 1}).")
            metric = run_inputs.atom_mapping_inputs.metric
            how = ("the xtb peak along their interpolation, above the lowest start among them"
                   if metric == "snap-gi-xtb" else metric)
            n_mech = len({r.key for _, _, r in ranked})
            each = " of each mechanism" if built_cap_per_mechanism else ""
            over = (len(ranked) > len(built_chosen))
            if picks is not None:
                head = f"--search-built: searching {len(built_chosen)} chosen of the {len(ranked)} built pairs:"
            elif built_pairs is not None:
                head = (f"--pair-from {pair_from}: {len(ranked)} built pair(s) (--built-pairs), ranked by {how}; "
                        f"the lowest {max_searches}{each} searched (--atom-mapping-candidates):")
            else:
                n_sources = sum(len(pools[s]) for s in sampled_sides)
                head = (f"--pair-from {pair_from}: {len(ranked)} built pair(s), of at most {n_sources} conformer(s) x "
                        f"{n_maps} mapping(s) = {n_sources * n_maps}, {n_mech} mechanism(s); "
                        + (f"more than --atom-mapping-candidates ({max_searches}){each}, so all are ranked by "
                           f"{how} and the lowest {max_searches}{each} searched:" if over
                           else "all searched:"))
            _log(head)
            for ln in lines:
                _log(ln)
            _save_built_pairs(output / "built_pairs", ranked, searched_before | (
                set() if conformers_only else {f"built_{n}" for n in built_chosen}), metric)
            # Every built pair stays in the pools, in rank order, so a pair's
            # label is the same in every run of this list; only the chosen
            # ones are searched.
            pairs = [(StructureNode(structure=r.start), StructureNode(structure=r.end)) for _, _, r in ranked]
            stats["pair_from"].update(n_built_pairs=len(ranked), n_mappings=n_maps, max_searches=max_searches,
                                      n_searched=len(built_chosen), n_mechanisms=n_mech,
                                      cap_per_mechanism=built_cap_per_mechanism, searched=[f"built_{n}" for n in built_chosen])
        if pairs:
            pools = {"start": [a for a, _ in pairs], "end": [b for _, b in pairs]}
        else:
            # Nothing built with the right bonds and stereochemistry: search
            # the sampled conformers as they are, as --pair-from both would.
            _log(f"--pair-from {pair_from}: no partner could be built; pairing the sampled conformers "
                       "instead (as --pair-from both).")
            pools, pair_from = sampled_pools, "both"
            stats["pair_from"]["fallback"] = "both"
            for label in ("start", "end"):   # a side that was only to be built was never minimized
                if minimize_ends and label not in reused and label not in sampled_sides:
                    pools[label] = _minimize_conformer_pool(pools[label], label, run_inputs)
    start_confs, end_confs = pools["start"], pools["end"]
    _write_stats()

    if not start_confs or not end_confs:
        raise ChannelSearchError("No usable conformers for one or both endpoints; nothing to run.")

    # Persisted as soon as the pools are final -- before pairing and
    # mapping, which can take a long time for big pools, so an interrupted
    # run still keeps its (expensive) conformers -- and so `mepd visualize
    # <output>` can show every conformer, not just those in searched pairs.
    from mepd.inputs import ChainInputs

    conformers_dir = output / "conformers"
    conformers_dir.mkdir(parents=True, exist_ok=True)
    Chain.model_validate(
        {"nodes": start_confs, "parameters": ChainInputs()}
    ).write_to_disk(conformers_dir / "start.xyz")
    Chain.model_validate(
        {"nodes": end_confs, "parameters": ChainInputs()}
    ).write_to_disk(conformers_dir / "end.xyz")

    structures = start_confs + end_confs
    n_start = len(start_confs)
    candidates = [
        (i, n_start + j) for i in range(n_start) for j in range(len(end_confs))
    ] if pair_from == "both" else [(i, n_start + i) for i in (built_chosen if built_chosen is not None
                                                               else range(n_start))]
    stats["n_pairs_possible"] = len(candidates)

    if max_pairs and len(candidates) > max_pairs:
        _log(
            f"Pairs: {n_start} reactant x {len(end_confs)} product conformers = {len(candidates)}; "
            f"keeping the first {max_pairs} (capping at --max-pairs={max_pairs})."
        )
        candidates = candidates[:max_pairs]
    stats["n_pairs"] = len(candidates)

    # workers=0 / parallel_workers=None ("auto"): resolved fresh at each
    # stage from how many genuinely independent things there are to run --
    # a fixed number picked once up front would be wrong at BOTH ends: too
    # small for mapping (hundreds of independent (pair, mechanism) scores,
    # cheap and embarrassingly parallel under --atom-mapping-metric
    # endpoint-rmsd) and too large for path search (--pairs-per-mechanism
    # usually leaves only a handful of pairs, so naively handing all of them
    # `cpu_count` workers wastes the rest of the machine that --parallel
    # could otherwise be using per pair).
    cpu_count = os.cpu_count() or 1
    workers_auto = workers == 0
    parallel_workers_auto = parallel_workers is None
    if workers_auto:
        workers = max(1, min(len(candidates), cpu_count))

    # The endpoint-level --atom-mapping check above ran on the two input
    # structures, before any conformer existed; which mechanism a pair can
    # follow, and how its equivalent atoms are best labeled, depend on that
    # pair's own geometry. Map every pair, one path search per mechanism.
    t0 = time.perf_counter()
    # --pair-from: each pair was built from one atom correspondence already.
    structures, candidates, mechanism_summary = _expand_pairs_by_mechanism(
        structures, candidates, atom_mapping and pair_from == "both", run_inputs,
        pairs_per_mechanism=pairs_per_mechanism, workers=workers, output=output,
    )
    stats["pair_atom_mapping_seconds"] = round(time.perf_counter() - t0, 3)
    stats.update(mechanism_summary)
    stats["n_path_searches"] = len(candidates)
    _write_stats()

    if workers_auto:
        workers = max(1, min(len(candidates), cpu_count))
    if parallel and parallel_workers_auto:
        parallel_workers = max(1, cpu_count // workers)
    stats["workers"] = workers
    stats["parallel_workers"] = parallel_workers if parallel else None
    # Engines that can run a chain's images concurrently (n_parallel = 0 is
    # "auto") get what is left of the machine per branch.
    if getattr(run_inputs.engine, "n_parallel", None) == 0:
        run_inputs.engine.n_parallel = max(
            1, cpu_count // (workers * (parallel_workers if parallel else 1))
        )
    stats["engine_parallel"] = getattr(run_inputs.engine, "n_parallel", None)

    if conformers_only:
        _log(
            f"--conformers-only: {len(start_confs)} x {len(end_confs)} conformers, "
            f"{len(candidates)} pair(s); stopping before path search."
        )
        return

    pairs_dir = output / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if search_deadline is not None:
        # Read by MSMEP (no further splits) and by _run_msmep_pairs (no new
        # pairs); forked pair workers inherit it with run_inputs.
        setattr(run_inputs.path_min_inputs, "recursive_split_deadline", search_deadline)
        stats["search_budget_seconds"] = search_budget
    # Each pair's TS optimizations and IRCs start as soon as its search is
    # done, in processes of their own (one serial chain of gradients each):
    # alongside the searches still running, a few at a time; once none are
    # left (all done, or Stop & analyze), on as many processes as the
    # searches were given (pairs x branches x images), not just --workers.
    q, m = int(start_node.structure.charge), int(start_node.structure.multiplicity)
    ts_dir = output / "ts"
    search_procs = workers * (parallel_workers if parallel else 1) * max(1, int(stats.get("engine_parallel") or 1))
    analysis_width = max(workers, min(search_procs, cpu_count))
    follow_up = PairFollowUp(
        tasks=lambda pair_dir: [t for t in _pair_ts_guess_tasks(pair_dir, q, m)
                                if _load_ts_and_irc_from_disk(ts_dir, t[0], q, m) is None
                                and not _ts_opt_failed(ts_dir, t[0])],
        run=lambda task: _ts_and_irc_once(task[1], task[0], ts_dir, run_inputs, q, m) and None,
        during=workers, after=analysis_width,
    )
    from mepd.network_aware import resolve_cutoff

    if resolve_cutoff(run_inputs.path_min_inputs) is not None \
            and getattr(run_inputs.path_min_inputs, "network_floor_hartree", None) is None:
        # Network-aware pruning: one floor for every pair, the lowest reactant
        # conformer (barrier floor rule), not each pair's own start.
        energies = []
        for node in structures[:n_start]:
            try:
                e = node.energy
            except Exception:
                e = None
            if e is None:
                run_inputs.engine.compute_energies([node])
                e = node.energy
            energies.append(float(e))
        run_inputs.path_min_inputs.network_floor_hartree = min(energies)
        stats["network_floor_hartree"] = min(energies)
    _run_msmep_pairs(
        structures, candidates, pairs_dir, run_inputs,
        parallel=parallel, parallel_workers=parallel_workers, workers=workers,
        follow_up=follow_up,
    )
    stats["msmep_seconds"] = round(time.perf_counter() - t0, 3)
    if search_deadline is not None:
        stats["search_budget_reached"] = time.time() > search_deadline
    from mepd import search_control

    if search_control.stop_requested():   # Stop & analyze: what follows runs on the paths found so far
        stats["stopped_on_request"] = True
    if getattr(run_inputs.path_min_inputs, "direct_only", False):
        # Legs through other species that were not run; pairs left with
        # nothing to search, and pairs with some legs dropped.
        from mepd.cli_common import direct_only_counts

        stats["direct_only"] = direct_only_counts(pairs_dir)
    _write_stats()

    tree_dirs = _completed_tree_dirs(pairs_dir)
    if not tree_dirs:
        if (stats.get("direct_only") or {}).get("pairs_not_characterized"):
            # Expected with --direct-only, not a failure: every search needed an intermediate.
            _log("No channel: every pair's legs run between species other than the start and end, and "
                       "--direct-only does not run those. Rerun without it to characterize them.")
            return
        raise ChannelSearchError("No pairs completed successfully; nothing to build a network from.")

    # network.json is a by-product: TS optimization and classification read
    # the pair trees themselves, so a network that can't be built (e.g. no
    # elementary leaf left once --direct-only dropped legs) is reported,
    # not fatal.
    builder = NetworkBuilder(data_dir=output, network_inputs=NetworkInputs())
    try:
        pot = builder.create_rxn_network_from_paths(tree_dirs)
    except Exception as exc:
        _warn(f"WARNING: network construction failed ({type(exc).__name__}: {exc}); "
                   "continuing with TS optimization and classification.")
        stats["network_error"] = f"{type(exc).__name__}: {exc}"
    else:
        network_path = output / "network.json"
        pot.write_to_disk(network_path)
        _log(
            f"Wrote network to {network_path} "
            f"({pot.number_of_nodes} nodes, {pot.graph.number_of_edges()} edges)"
        )

    t0 = time.perf_counter()
    # What is left (pairs searched before this run, a follow-up that failed): as wide.
    mismatch_warnings = _discover_channels(
        output, start_node, end_node, run_inputs, q, m, workers=analysis_width,
    )
    if mismatch_warnings:
        stats["endpoint_mismatch_warnings"] = mismatch_warnings
    stats["ts_discovery_seconds"] = round(time.perf_counter() - t0, 3)
    _write_stats()


# --- results -----------------------------------------------------------------


@dataclass
class Step:
    """One elementary step: its transition state and IRC. `alternatives`:
    other distinct TSs found for the same step (higher in energy)."""
    ts: object
    irc: object
    folder: Path
    connects: str = ""
    alternatives: list = field(default_factory=list)


@dataclass
class Channel:
    """A way from start to end found by a channel search.

    `kind`: "direct" (one step, `channels/`), "multistep" (several steps
    through intermediates, `alternate-channels/`) or "offtarget" (a step off
    the route that never reaches the end, `offtarget-exit-channels/`).
    `ts` is the highest-energy TS along it; `species` names the minima it
    passes (multistep routes)."""
    kind: str
    steps: list
    folder: Path
    species: list = field(default_factory=list)

    @property
    def ts(self):
        def energy(step):
            try:
                return step.ts.energy
            except Exception:
                return float("-inf")
        return max(self.steps, key=energy).ts


@dataclass
class ChannelsResult:
    """What a channel search found, read from its output folder.

    `direct`, `multistep`, `offtarget`: lists of `Channel`. `start_conformers`
    and `end_conformers`: the endpoint pools searched (with energies), e.g.
    for `mepd.barrier_kcal(channel.ts, result.start_conformers)`.
    `stats`: the run's stats.json."""
    folder: Path
    direct: list
    multistep: list
    offtarget: list
    start_conformers: list
    end_conformers: list
    stats: dict

    @property
    def routes(self) -> list:
        """Every way from start to end: direct channels, then multistep ones."""
        return self.direct + self.multistep

    def lowest(self):
        """The route whose highest TS is lowest in energy, or None."""
        def energy(channel):
            try:
                return channel.ts.energy
            except Exception:
                return float("inf")
        return min(self.routes, key=energy, default=None)


def _read_xyz_chain(fp: Path, charge: int, multiplicity: int):
    from mepd.inputs import ChainInputs

    return Chain.from_xyz(fp, ChainInputs(), charge=charge, spinmult=multiplicity)


def _read_step(folder: Path, charge: int, multiplicity: int) -> Step:
    from mepd.compute import IRCResult

    ts = _read_xyz_chain(folder / "ts.xyz", charge, multiplicity)[0]
    irc_fp = folder / "irc.xyz"
    irc = IRCResult(path=_read_xyz_chain(irc_fp, charge, multiplicity)) if irc_fp.exists() else None
    members = (folder / "members.txt").read_text().splitlines() if (folder / "members.txt").exists() else []
    connects = members[0].removeprefix("connects: ") if members and members[0].startswith("connects: ") else ""
    alternatives = [_read_step(d, charge, multiplicity)
                    for d in sorted(folder.glob("alternate_ts_*"), key=lambda d: int(d.name.rsplit("_", 1)[1]))]
    return Step(ts=ts, irc=irc, folder=folder, connects=connects, alternatives=alternatives)


def _numbered(folder: Path, prefix: str) -> list[Path]:
    return sorted((d for d in folder.glob(f"{prefix}_*") if d.is_dir() and d.name.rsplit("_", 1)[1].isdigit()),
                  key=lambda d: int(d.name.rsplit("_", 1)[1]))


def read_channels(folder, *, charge: Optional[int] = None, multiplicity: Optional[int] = None) -> ChannelsResult:
    """Read what a channel search (`mepd.channels` or `mepd channels`) wrote
    into `folder`. Charge and multiplicity default to the start conformers'
    (written by the search); give them for an older folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"No channel-search folder at {folder}.")
    conf = folder / "conformers"
    if charge is None or multiplicity is None:
        from mepd.structures import load_structure

        first = next((fp for fp in (conf / "start.xyz", conf / "start_pool.xyz") if fp.exists()), None)
        s = load_structure(first) if first else None
        charge = charge if charge is not None else (int(s.charge) if s else 0)
        multiplicity = multiplicity if multiplicity is not None else (int(s.multiplicity) if s else 1)

    def pool(name):
        fp = conf / f"{name}.xyz"
        return list(_read_xyz_chain(fp, charge, multiplicity).nodes) if fp.exists() else []

    direct = [Channel("direct", [_read_step(d, charge, multiplicity)], d)
              for d in _numbered(folder / "channels", "channel")]
    multistep = []
    for d in _numbered(folder / "alternate-channels", "alternate_channel"):
        lines = (d / "path.txt").read_text().splitlines() if (d / "path.txt").exists() else []
        species = [ln.strip().removeprefix("-> ").strip() for ln in lines[1:]
                   if ln.startswith("  ") and not ln.strip().startswith("step_")]
        multistep.append(Channel("multistep", [_read_step(s, charge, multiplicity) for s in _numbered(d, "step")],
                                 d, species=species))
    offtarget = [Channel("offtarget", [_read_step(d, charge, multiplicity)], d)
                 for d in _numbered(folder / "offtarget-exit-channels", "offtarget_exit_channel")]
    stats_fp = folder / "stats.json"
    return ChannelsResult(folder=folder, direct=direct, multistep=multistep, offtarget=offtarget,
                          start_conformers=pool("start"), end_conformers=pool("end"),
                          stats=json.loads(stats_fp.read_text()) if stats_fp.exists() else {})


def channels(start, end, *, output, profile=None, engine=None, **options) -> ChannelsResult:
    """Search for every way from `start` to `end`: conformers of both,
    paths between pairs of them, a TS and IRC for every path's peak, sorted
    into direct channels, multistep routes and off-target exits.

    `start`, `end`: structures in the same atom order (or several molecules
    each). `output`: the folder the search works in (a rerun resumes it).
    `profile`: a profile .toml or `RunInputs` (path search settings, engine);
    `engine` overrides its engine. `options`: `ChannelsOptions` fields, e.g.
    `n_conformers=5, search_budget=1800, direct_only=True`.

    >>> result = mepd.channels(r, p, output="runs/claisen", profile="gsm.toml", n_conformers=5)
    >>> best = result.lowest()
    >>> mepd.barrier_kcal(best.ts, result.start_conformers)
    """
    run_inputs = profile if isinstance(profile, RunInputs) else (RunInputs.open(profile) if profile else RunInputs())
    if engine is not None:
        run_inputs.engine = engine
    opts = ChannelsOptions(**options)
    run_channels(start, end, run_inputs, opts, Path(output))
    return read_channels(output, charge=int(start.structure.charge), multiplicity=int(start.structure.multiplicity))
