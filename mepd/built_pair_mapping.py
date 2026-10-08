"""Atom mapping when one endpoint is built from the other (--pair-from).

With both endpoints as given, a candidate mapping is judged by the path to
the given product (mepd.atom_mapping_selection). When the product is built
in the reactant's frame instead (or the reactant in the product's), that
product is not the path search's endpoint: each mapping gives its own built
partner. So here every candidate mapping is turned into the bonds it implies
on the source's atoms, a partner with those bonds is embedded in the
source's frame and minimized (all of them, as one batch), partners whose
bonds or stereochemistry change are dropped, and the rest are ranked by the
path from the source to each minimized partner -- the pair that is then
searched. One partner is built per mechanism, from its first candidate:
relabelings of symmetric atoms are not built (each conformer of the source
places the equivalent atoms differently already). Snap is not used: the
built partner is already in the source's frame.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR
from qcdata import Structure


@dataclass
class BuiltPair:
    start: Structure       # in the start's atom order
    end: Structure         # in the start's atom order
    label: str             # the winning candidate
    scores: dict           # {label: the quantity compared}
    quantity: str          # what `scores` hold
    notes: list            # lines explaining the choice
    n_bond_sets: int       # distinct bond sets built
    n_kept: int            # of them, minimized with bonds and stereochemistry intact


def _edges(structure: Structure) -> set:
    from mepd.nodes.node import StructureNode

    return {tuple(sorted((int(u), int(v)))) for u, v in StructureNode(structure=structure).graph.edges()}


def _in_end_order(atom_map, start: Structure) -> Structure:
    """`start` renumbered to the end's atom order under `atom_map`
    (start atom i -> end atom mapping[i])."""
    from mepd.atom_mapping import reorder_structure

    inverse = [0] * len(atom_map.mapping)
    for i, j in atom_map.mapping.items():
        inverse[j] = i
    return reorder_structure(start, inverse)


def _minimize(guesses: list[Structure], run_inputs) -> list[Optional[Structure]]:
    """Each guess minimized with the run's engine (one batch when the
    engine can); None where an optimization failed."""
    from mepd.nodes.node import StructureNode

    nodes = [StructureNode(structure=g) for g in guesses]
    keywords = {"coordsys": "cart", "maxit": 500, **(getattr(run_inputs, "geometry_optimizer_kwds", None) or {})}
    batch = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    try:
        if callable(batch):
            try:
                trajectories = batch(nodes, keywords=keywords)
            except TypeError:
                trajectories = batch(nodes)
        else:
            raise AttributeError
    except Exception:
        trajectories = []
        for n in nodes:
            try:
                trajectories.append(run_inputs.engine.compute_geometry_optimization(n, keywords=keywords))
            except Exception:
                trajectories.append(None)
    return [traj[-1].structure if traj else None for traj in trajectories]


def _build(start: Structure, end: Structure, atom_maps: list, pair_from: str, run_inputs, echo) -> tuple:
    """Every mechanism's partner built and minimized: (jobs, kept). jobs:
    (label, mapping, source, target, bonds) per mechanism; kept:
    (label, mapping, source, built) for those whose bonds and
    stereochemistry survived minimization."""
    from mepd.atom_mapping import AtomMapping, _automorphism_generators, mechanism_key, realign_end_to_start
    from mepd.discovery.network_expansion import embed_product
    from mepd.nodes.node import StructureNode
    from mepd.nodes.nodehelpers import _is_connectivity_identical

    n = len(start.symbols)
    maps = [("identity", AtomMapping(mapping={i: i for i in range(n)}, cost=0, n_alternatives=1))] \
        if list(start.symbols) == list(end.symbols) else []
    maps += [(f"mapping_{k}", m) for k, m in enumerate(m for m in atom_maps if not m.is_identity)]
    from mepd.inputs import MAPPING_CANDIDATE_LIMIT as cap

    # One build per mechanism, from its first candidate as given: no
    # relabelings of the source's symmetric atoms (which of a methyl's H
    # moves) among the candidates, and swaps only as a fallback when the
    # given site fails to build (below). Each conformer is
    # its own source, so where the equivalent atoms sit to react already
    # varies across the pool. Built from the start, a mapping's bonds are the
    # end's, renumbered onto the start's atoms; from the end, the start's
    # bonds renumbered onto the end's atoms.
    seen, jobs = set(), []
    for label, m in maps:
        if len(jobs) >= cap:
            break
        source, target = (start, realign_end_to_start(m, end)) if pair_from == "start" else (end, _in_end_order(m, start))
        key = mechanism_key(source, target) if pair_from == "start" else mechanism_key(target, source)
        if key not in seen:
            seen.add(key)
            jobs.append((label, m, source, target, frozenset(_edges(target))))
    built_name = "product" if pair_from == "start" else "reactant"
    echo(f"Atom mapping: building the {built_name} of each of {len(jobs)} mechanism(s) "
         f"({len(maps)} candidate mapping(s){', capped' if len(jobs) >= cap else ''}) in its frame, minimizing them, then comparing "
         f"the paths to them...")

    def build(batch):
        """Each job's kept (label, mapping, source, built), or None."""
        guesses, embedded = [], []
        for k, (_, _, source, target, want) in enumerate(batch):
            try:
                xyz = embed_product(list(source.symbols),
                                    np.asarray(source.geometry).reshape(-1, 3) / ANGSTROM_TO_BOHR, set(want))
            except Exception:
                continue
            guesses.append(target.model_copy(update={"geometry": np.asarray(xyz) * ANGSTROM_TO_BOHR}))
            embedded.append(k)
        out = [None] * len(batch)
        for k, s in zip(embedded, _minimize(guesses, run_inputs) if guesses else []):
            label, m, source, target, want = batch[k]
            if s is None or frozenset(_edges(s)) != want:
                continue
            if not _is_connectivity_identical(StructureNode(structure=s), StructureNode(structure=target),
                                              verbose=False, collect_comparison=False):
                continue   # stereochemistry changed
            out[k] = (label, m, source, s)
        return out

    results = build(jobs)
    failed = [k for k, r in enumerate(results) if r is None]
    if failed:
        # Fallback, only for a mechanism whose as-given build failed: the
        # same bonds at the source's equivalent sites (its bond set's orbit
        # under the source's symmetry, walked from nauty's generators), the
        # first that builds kept in its place. Still one build per mechanism.
        gens = _automorphism_generators(start if pair_from == "start" else end)
        swaps = [(k, job) for k in failed for job in _equivalent_sites(jobs[k], gens, pair_from, start, end, cap)]
        swapped = build([job for _, job in swaps]) if swaps else []
        tried = {k: sum(1 for i, _ in swaps if i == k) for k in failed}
        for (k, _), r in zip(swaps, swapped):
            if r is not None and results[k] is None:
                results[k] = r
        for k in failed:
            what = (f"built at an equivalent site instead ({results[k][0]})" if results[k] is not None
                    else "no equivalent site built either" if tried[k] else "it has no equivalent sites")
            echo(f"  SYMMETRY-SWAP FALLBACK: {jobs[k][0]}'s {built_name} did not keep its bonds and "
                 f"stereochemistry as given; tried {tried[k]} equivalent site(s): {what}.")
    kept = [r for r in results if r is not None]
    echo(f"  {len(kept)} of {len(jobs)} kept their bonds and stereochemistry on minimization"
         + (f" ({sum(1 for k in failed if results[k] is not None)} via the symmetry-swap fallback)." if failed else "."))
    return jobs, kept


def _equivalent_sites(job: tuple, gens: list, pair_from: str, start: Structure, end: Structure, cap: int) -> list:
    """`job`'s bonds moved to the source's equivalent sites: one job per
    other bond set in its orbit under the group `gens` generate, labeled
    "<label> sym<k>", at most `cap`."""
    from mepd.atom_mapping import realign_end_to_start

    label, m0, _, _, want0 = job
    n = len(start.symbols)

    def relabeled(h):
        """`m0` with the source's atom i relabeled h[i]: the partner's bonds
        move from atoms (i, j) to (h[i], h[j])."""
        if pair_from == "start":
            inverse = [0] * n
            for i, j in enumerate(h):
                inverse[j] = i
            mapping = {i: m0.mapping[inverse[i]] for i in range(n)}
        else:
            mapping = {i: h[m0.mapping[i]] for i in range(n)}
        return dataclasses.replace(m0, mapping=mapping, relabeling=True)

    seen, queue, out = {want0}, [(want0, list(range(n)))], []
    while queue and len(out) < cap:
        want, h = queue.pop(0)
        if want != want0:
            m = relabeled(h)
            source, target = (start, realign_end_to_start(m, end)) if pair_from == "start" else (end, _in_end_order(m, start))
            out.append((f"{label} sym{len(out) + 1}", m, source, target, want))
        for g in gens:
            moved = frozenset(tuple(sorted((g[u], g[v]))) for u, v in want)
            if moved not in seen:
                seen.add(moved)
                queue.append((moved, [g[x] for x in h]))
    return out


def _rank(kept: list, start: Structure, pair_from: str, run_inputs) -> tuple:
    """`kept` (from `_build`) ranked as paths from the source (shared by
    all) to each built partner: (best label, scores, quantity, notes)."""
    from mepd.atom_mapping_selection import MappingCandidate, _rank_picks, score_candidate

    picks = [MappingCandidate(label=label, end_structure=s, atom_map=None if label == "identity" else m)
             for label, m, _, s in kept]
    source = kept[0][2] if pair_from == "end" else start
    metric = run_inputs.atom_mapping_inputs.metric
    if metric in ("snap", "snap-gi-xtb"):
        _, best, _, scores, quantity, notes = _rank_picks(picks, picks, source, run_inputs,
                                                          use_path=metric == "snap-gi-xtb")
    else:
        scores = {c.label: score_candidate(c, metric, source, run_inputs)[0] for c in picks}
        best, quantity, notes = min(picks, key=lambda c: scores[c.label]), metric, []
    if pair_from == "end" and quantity:
        quantity += ", from the product"
    return best.label, scores, quantity, notes


def _pair(pair_from: str, start: Structure, end: Structure, m, built: Structure) -> tuple:
    """(start, end) of a built pair, in the start's atom order."""
    from mepd.atom_mapping import realign_end_to_start

    if pair_from == "start":
        return start, built
    # the built start and the end are both in the end's order
    return realign_end_to_start(m, built), realign_end_to_start(m, end)


def build_and_rank(start: Structure, end: Structure, atom_maps: list, pair_from: str, run_inputs,
                   echo: Callable[[str], None] = print) -> Optional[BuiltPair]:
    """`pair_from` "start": the end is built in the start's frame (the start
    is kept); "end": the start in the end's frame. `atom_maps`: the
    candidate mappings (start atom i -> end atom mapping[i]); the current
    numbering competes too when the elements are in the same order. Returns
    the chosen pair in the start's atom order, or None if no partner kept
    its bonds and stereochemistry."""
    jobs, kept = _build(start, end, atom_maps, pair_from, run_inputs, echo)
    if not kept:
        return None
    best, scores, quantity, notes = _rank(kept, start, pair_from, run_inputs)
    label, m, _, built = next(k for k in kept if k[0] == best)
    changes = {job[0]: len(_edges(job[2]) ^ job[4]) for job in jobs}
    fewest = min(changes.values())
    n_changes = changes[label.split(" sym")[0]]    # a fallback's equivalent site changes as many bonds
    if n_changes > fewest:
        echo(f"WARNING: no {'product' if pair_from == 'start' else 'reactant'} with the fewest bond changes "
             f"({fewest}) survived minimization with its bonds and stereochemistry; the chosen one has "
             f"{n_changes}.")
    pair_start, pair_end = _pair(pair_from, start, end, m, built)
    return BuiltPair(start=pair_start, end=pair_end, label=label, scores=scores, quantity=quantity, notes=notes,
                     n_bond_sets=len(jobs), n_kept=len(kept))


@dataclass
class MechanismPair:
    key: str               # mechanism_key: which bonds break and form, by element and symmetry class
    score: float           # by --atom-mapping-metric (_path_score); lower is better
    start: Structure       # in the start's atom order
    end: Structure
    n_variants: int        # built variants of this mechanism compared
    floor: Optional[float] = None   # snap-gi-xtb: the xtb energy (kcal/mol) of the pair's start, else None
    path: Optional[list] = None            # the interpolation scored (Structures), if one was
    path_energies: Optional[list] = None   # its xtb energies (kcal/mol, absolute), if computed


def best_per_mechanism(start: Structure, end: Structure, atom_maps: list, pair_from: str, run_inputs,
                       echo: Callable[[str], None] = print) -> list[MechanismPair]:
    """Like `build_and_rank`, but one pair per mechanism instead of one in
    all: the built partners grouped by which bonds break and form.
    For channels, where every mechanism gets its own path searches."""
    from mepd.atom_mapping import mechanism_key

    _, kept = _build(start, end, atom_maps, pair_from, run_inputs, echo)
    groups: dict = {}
    for k in kept:
        label, m, _, built = k
        a, b = _pair(pair_from, start, end, m, built)
        groups.setdefault(mechanism_key(a, b), []).append(k)
    out = []
    for key, members in groups.items():
        best, _, _, _ = _rank(members, start, pair_from, run_inputs) if len(members) > 1 else (members[0][0],) * 4
        label, m, _, built = next(k for k in members if k[0] == best)
        a, b = _pair(pair_from, start, end, m, built)
        score, floor, path = _path_score(a, b, run_inputs)
        out.append(MechanismPair(key=key, score=score, start=a, end=b, n_variants=len(members), floor=floor,
                                 path=path[0] if path else None, path_energies=path[1] if path else None))
    return out


def _path_score(start: Structure, end: Structure, run_inputs) -> tuple[float, Optional[float], Optional[tuple]]:
    """(score, floor, path): one number to compare built pairs across conformers
    and mechanisms, lower better, by --atom-mapping-metric. snap-gi-xtb
    (default): the xtb energy (kcal/mol) of the peak along the interpolated
    path, absolute, so pairs compare against one shared floor; floor is the
    xtb energy of the path's start. Without xtb, the path's length. snap and
    endpoint-rmsd, the aligned endpoint RMSD; any other metric, its
    `score_candidate` score. floor None but with xtb; score inf if it can't
    be had. path: (the interpolation's Structures, their xtb energies in
    kcal/mol or None), None if no interpolation was made."""
    import math

    from mepd.atom_mapping_selection import (KCAL_PER_HARTREE, MappingCandidate, _interpolate, _xtb_engine,
                                             score_candidate)

    metric = run_inputs.atom_mapping_inputs.metric
    candidate = MappingCandidate(label="pair", end_structure=end, atom_map=None)
    try:
        if metric != "snap-gi-xtb":
            return float(score_candidate(candidate, "endpoint-rmsd" if metric == "snap" else metric, start,
                                         run_inputs)[0]), None, None
        chain, smoother = _interpolate(candidate, start, run_inputs)
    except Exception:
        return math.inf, None, None
    frames = [node.structure for node in chain]
    engine = _xtb_engine()
    if engine is not None:
        try:
            e = [float(x) for x in engine.compute_energies(chain)]
            if e and all(math.isfinite(x) for x in e):
                kcal = [x * KCAL_PER_HARTREE for x in e]
                return max(kcal), kcal[0], (frames, kcal)
        except Exception:
            pass
    return float(smoother.length), None, (frames, None)
