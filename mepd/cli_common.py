"""Shared infrastructure for `mepd/cli.py`'s commands: loading/echoing
`RunInputs`, endpoint loading (xyz/SMILES), endpoint minimization,
best-of-N atom-mapping selection at the top-level --start/--end pair,
forked-process parallelism (`_fork_map`), running a batch of MSMEP pair
searches to completion, TS+IRC optimization, and collecting TS-guess
tasks from a tree/network. Used by `run`, `ts`, `network-splits` (in
`mepd/cli.py`) and by `channels` (in `mepd/cli_channels.py`) -- split out
so those command modules don't have to import from each other.
"""
from __future__ import annotations

import copy
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import typer
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import RunInputs
from mepd.nodes.nodehelpers import _connectivity_matches, _n_trans_small_ring_alkenes  # noqa: F401



# Progress of the reusable steps here (endpoint minimization, atom mapping,
# pair searches, TS + IRC) is logged, not printed, so library callers (e.g.
# mepd.channels) stay quiet unless they configure logging; the command line
# prints the "mepd" log on stdout (print_log_on_stdout, called by mepd.cli).
logger = logging.getLogger("mepd.cli_common")


def _say(message: str = "") -> None:
    logger.info(message)


def _warn(message: str) -> None:
    """Something went wrong but the run goes on: shown even without logging set up."""
    logger.warning(message)


class _EchoHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        typer.echo(record.getMessage())


# The loggers whose progress messages the commands print (other mepd loggers,
# e.g. the geodesic smoother's, keep their own levels and stay quiet).
PROGRESS_LOGGERS = ("mepd.cli_common", "mepd.channel_search")


def print_log_on_stdout() -> None:
    """Print the progress loggers' messages on stdout, one plain line each,
    as the commands always have. Idempotent."""
    for name in PROGRESS_LOGGERS:
        log = logging.getLogger(name)
        if any(isinstance(h, _EchoHandler) for h in log.handlers):
            continue
        log.addHandler(_EchoHandler())
        log.setLevel(logging.INFO)
        log.propagate = False

def _format_run_inputs_value(value) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        return ", ".join(f"{k}={_format_run_inputs_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return str(list(value)) if value else "[]"
    if isinstance(value, bool):
        return "[green]True[/green]" if value else "[red]False[/red]"
    if value == "":
        return "[dim]--[/dim]"
    return str(value)


def _dataclass_field_values(obj) -> dict:
    """All active fields of a dataclass instance (engine/optimizer), with
    their currently-resolved values -- including ones left at their class
    default and never mentioned in the TOML. Private/internal fields (leading
    underscore) are excluded."""
    import dataclasses

    if obj is None or not dataclasses.is_dataclass(obj):
        return {}
    return {
        f.name: getattr(obj, f.name, None)
        for f in dataclasses.fields(obj)
        if not f.name.startswith("_")
    }


def _echo_run_inputs_summary(run_inputs: RunInputs) -> None:
    """Print the settings actually in effect for this run (after any
    --inputs TOML has been loaded and any CLI-flag overrides applied)."""
    try:
        config = run_inputs.to_dict()
    except Exception as exc:
        typer.echo(f"(could not render RunInputs summary: {exc})", err=True)
        return

    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold cyan]RunInputs[/bold cyan]")

    # The raw *_kwds dicts only reflect what the TOML happened to set --
    # replaced below with the fully resolved engine/optimizer objects, which
    # include every active field even when left at its default.
    for key in ("optimizer_kwds", "ase_engine_kwds", "gxtb_engine_kwds"):
        config.pop(key, None)

    top_level, sections = {}, {}
    for key, value in config.items():
        if isinstance(value, dict):
            sections[key] = value
        else:
            top_level[key] = value

    def _new_table(title: str | None = None) -> Table:
        table = Table(title=title, box=box.ROUNDED, show_header=True, title_style="bold")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value")
        return table

    table = _new_table()
    for key, value in top_level.items():
        table.add_row(key, _format_run_inputs_value(value))
    console.print(table)

    for key, value in sections.items():
        table = _new_table(title=key)
        if not value:
            table.add_row("[dim](empty)[/dim]", "")
        else:
            for sub_key, sub_value in value.items():
                table.add_row(sub_key, _format_run_inputs_value(sub_value))
        console.print(table)

    for label, obj in (("engine", run_inputs.engine), ("optimizer", run_inputs.optimizer)):
        fields = _dataclass_field_values(obj)
        table = _new_table(title=f"{label} ({type(obj).__name__})")
        if not fields:
            table.add_row("[dim](no fields)[/dim]", "")
        else:
            for sub_key, sub_value in fields.items():
                table.add_row(sub_key, _format_run_inputs_value(sub_value))
        console.print(table)


def _load_endpoint(fp: Path, charge: Optional[int], multiplicity: Optional[int]) -> Structure:
    from mepd.structures import load_structure

    try:
        return load_structure(fp, charge, multiplicity)
    except Exception as exc:
        raise typer.BadParameter(f"Could not load '{fp}': {type(exc).__name__}: {exc}")


def _open_run_inputs(inputs: Optional[Path]) -> RunInputs:
    if inputs is None:
        return RunInputs()
    try:
        return RunInputs.open(inputs)
    except Exception as exc:
        raise typer.BadParameter(f"Could not load '{inputs}': {type(exc).__name__}: {exc}")


def _check_endpoint_options(start: Optional[str], end: Optional[str], reaction: Optional[str]) -> None:
    """Either --start and --end, or --reaction -- never both, never neither."""
    if reaction is not None and (start is not None or end is not None):
        raise typer.BadParameter("Give either --reaction or --start/--end, not both.")
    if reaction is None and (start is None or end is None):
        raise typer.BadParameter(
            "Give both --start and --end (xyz files or SMILES), or one --reaction "
            "'reactants>>products' reaction SMILES."
        )


def _reaction_endpoints(
    reaction: str, charge: Optional[int], multiplicity: Optional[int]
) -> tuple[Structure, Structure]:
    """Both endpoints of a --reaction SMILES, embedded with one shared atom
    order (see mepd/reaction_smiles.py)."""
    from mepd.reaction_smiles import ReactionSmilesError, reaction_structures

    try:
        start_structure, end_structure, pair = reaction_structures(reaction, charge, multiplicity)
    except (ReactionSmilesError, ValueError) as exc:
        raise typer.BadParameter(f"--reaction {reaction!r}: {exc}")
    if pair.source == "given":
        _say("--reaction: atom mapping from the map numbers in the reaction SMILES "
                   "(hydrogens by smallest endpoint RMSD).")
    else:
        _say(f"--reaction: atom mapping by SLAPMapper ({pair.n_candidates} candidate(s), chosen by smallest "
                   f"endpoint RMSD, {pair.rmsd:.2f} A): {pair.mapped_smiles}")
    for note in pair.notes:
        _say(f"  {note}")
    return start_structure, end_structure


def _load_structure_from_smiles_or_xyz(
    value: str, charge: Optional[int], multiplicity: Optional[int]
) -> Structure:
    """Load a Structure from an xyz file path, or -- if `value` isn't an
    existing file -- embed it in 3D from a SMILES string.

    The SMILES route is `mepd.structures.smiles_to_structure`.
    """
    path = Path(value)
    if path.exists():
        return _load_endpoint(path, charge, multiplicity)

    from mepd.structures import smiles_to_structure

    try:
        return smiles_to_structure(value, charge, multiplicity)
    except ValueError as exc:
        hint = ""
        if path.suffix or "/" in value or "\\" in value:
            hint = (
                f" (looks like a file path -- check for a typo; '{value}' does "
                "not exist)"
            )
        raise typer.BadParameter(
            f"'{value}' is neither an existing xyz file nor a valid SMILES string{hint}.\n{exc}"
        )


def _geometry_optimizer_keywords(run_inputs: RunInputs, *, default_maxiter: int = 500) -> dict:
    keywords = {"coordsys": "cart", "maxit": int(default_maxiter)}
    keywords.update(dict(getattr(run_inputs, "geometry_optimizer_kwds", {}) or {}))
    return keywords


def _refuse_unconverged_endpoint(label: str, exc: Exception, run_inputs: RunInputs) -> None:
    """Hard-stop: an endpoint failed to actually converge during --minimize-ends.

    Unlike other minimization failures (engine doesn't support optimization,
    a transient crash), a reported non-convergence means we'd otherwise hand
    NEB an endpoint that isn't really a minimum -- silently proceeding would
    make the whole run meaningless. Refuse instead of guessing.
    """
    current_maxiter = _geometry_optimizer_keywords(run_inputs).get("maxit")
    _warn(
        f"{label.capitalize()} endpoint minimization did not converge: {exc}\n"
        "Refusing to proceed with an un-minimized endpoint.\n"
        "Options:\n"
        "  1. Provide an already-minimized structure for this endpoint instead of using --minimize-ends.\n"
        "  2. Increase the optimizer's iteration budget by setting maxit (or maxiter) under "
        f"[geometry_optimizer_kwds] in your RunInputs TOML (current: {current_maxiter})."
    )
    raise typer.Exit(code=1)


def _minimize_endpoints(start_node, end_node, run_inputs: RunInputs):
    """Optimize the start/end endpoint geometries before building the initial chain.

    Mirrors the source neb-dynamics CLI's `--minimize-ends` behavior: prefer a
    batched call if the engine supports it, fall back to per-node calls. Most
    failures (engine doesn't support optimization, a transient crash) keep the
    original input geometry with a warning rather than aborting the run -- but
    a reported non-convergence (GeometryOptimizationNotConvergedError) is a
    hard stop, since silently continuing would run NEB on a non-minimum.
    """
    from mepd.errors import GeometryOptimizationNotConvergedError

    _say("Minimizing input endpoints...")
    keywords = _geometry_optimizer_keywords(run_inputs)
    endpoints = [start_node, end_node]
    labels = ("start", "end")

    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch_optimizer):
        try:
            try:
                trajectories = batch_optimizer(endpoints, keywords=keywords)
            except TypeError:
                trajectories = batch_optimizer(endpoints)
        except GeometryOptimizationNotConvergedError as exc:
            _refuse_unconverged_endpoint("an", exc, run_inputs)
        except Exception as exc:
            _warn(f"Endpoint batch minimization failed ({type(exc).__name__}: {exc}); keeping input geometries.")
        else:
            if not isinstance(trajectories, (list, tuple)) or len(trajectories) < 2:
                _say("Endpoint batch minimization returned an unexpected result; keeping input geometries.")
            else:
                for i, label in enumerate(labels):
                    trajectory = trajectories[i]
                    if trajectory:
                        endpoints[i] = trajectory[-1]
                    else:
                        _say(f"{label.capitalize()} endpoint optimization returned an empty trajectory; keeping input geometry.")
        return endpoints[0], endpoints[1]

    single_optimizer = getattr(run_inputs.engine, "compute_geometry_optimization", None)
    if not callable(single_optimizer):
        _say(f"Engine {type(run_inputs.engine).__name__} does not support geometry optimization; keeping input geometries.")
        return start_node, end_node

    for i, label in enumerate(labels):
        _say(f"Minimizing {label} endpoint...")
        try:
            trajectory = single_optimizer(endpoints[i], keywords=keywords)
            if trajectory:
                endpoints[i] = trajectory[-1]
            else:
                _say(f"{label.capitalize()} endpoint optimization returned an empty trajectory; keeping input geometry.")
        except GeometryOptimizationNotConvergedError as exc:
            _refuse_unconverged_endpoint(label, exc, run_inputs)
        except Exception as exc:
            _warn(f"{label.capitalize()} endpoint minimization failed ({type(exc).__name__}: {exc}); keeping input geometry.")

    return endpoints[0], endpoints[1]


def _dump_atom_mapping_candidates(
    candidates: list,
    metric: str,
    start_structure: Structure,
    run_inputs: "RunInputs",
    debug_dump_dir: Path,
):
    """`--debug-dump` variant of candidate scoring: computes ALL THREE
    metrics (`mepd.atom_mapping_selection.METRICS`) for every candidate --
    not just the selected one -- and writes each candidate's
    geodesic-interpolated path (with energies) plus a `scores.txt` summary
    table under `debug_dump_dir`, so a real run's --debug-dump doubles as
    comparative data across metrics. Returns a `SelectionResult` using
    `metric` for the actual decision, same as `select_best_candidate`."""
    from mepd.atom_mapping_selection import METRICS, SelectionResult, score_candidate_all_metrics

    debug_dump_dir.mkdir(parents=True, exist_ok=True)
    all_scores: dict[str, dict[str, float]] = {}
    chains: dict[str, Chain] = {}
    for candidate in candidates:
        scores, chain = score_candidate_all_metrics(candidate, start_structure, run_inputs)
        all_scores[candidate.label] = scores
        chains[candidate.label] = chain
        chain.write_to_disk(debug_dump_dir / f"{candidate.label}.xyz")

    identity = next((c for c in candidates if c.label == "identity"), None)
    best = min(candidates, key=lambda c: all_scores[c.label][metric])
    veto_margin = run_inputs.atom_mapping_inputs.veto_margin
    if (
        identity and best is not identity
        and all_scores[identity.label][metric] - all_scores[best.label][metric] <= veto_margin
    ):
        best = identity

    lines = ["candidate\t" + "\t".join(METRICS)]
    for candidate in candidates:
        row = [candidate.label] + [f"{all_scores[candidate.label][m]:.6f}" for m in METRICS]
        marker = "  <- selected" if candidate.label == best.label else ""
        lines.append("\t".join(row) + marker)
    scores_path = debug_dump_dir / "scores.txt"
    scores_path.write_text("\n".join(lines) + "\n")
    _say(f"  --debug-dump: wrote all-candidate, all-metric comparison to {scores_path}")

    return SelectionResult(
        winner=best,
        scores={label: scores[metric] for label, scores in all_scores.items()},
        chains=chains,
    )


def _snap_relabels(candidate, start_structure: Structure) -> bool:
    """Would snap renumber any symmetric atoms of this candidate?"""
    from mepd.atom_mapping_selection import _snap_picks

    return any(p.label != "identity" for p in _snap_picks(candidate, start_structure))


def _candidate_atom_maps(start_structure: Structure, end_structure: Structure, atom_map, run_inputs) -> list:
    """The candidate mappings the --atom-mapping check compares: SLAPMapper's
    (the budget filled with relabelings of symmetric atoms), or, when the
    bonds already correspond (`atom_map` is the identity), the current
    numbering's relabelings; plus --explore-mechanisms' mechanisms."""
    from mepd.atom_mapping import expand_mapping_fully, suggest_atom_mapping_candidates
    from mepd.inputs import MAPPING_CANDIDATE_LIMIT as n_candidates

    # Bonds that already correspond still leave the symmetric atoms (a
    # methyl's hydrogens) to be matched: the current numbering competes
    # with its symmetry relabelings.
    atom_maps = (expand_mapping_fully(atom_map, start_structure, end_structure, max_variants=n_candidates)
                 if atom_map.is_identity else
                 suggest_atom_mapping_candidates(start_structure, end_structure, max_candidates=n_candidates))
    seeds = atom_maps[:20]   # where --explore-mechanisms starts (as before: what it finds depends on them)
    if run_inputs.atom_mapping_inputs.metric in ("snap", "snap-gi-xtb"):
        # Snap places the symmetric atoms itself, from any one numbering of a
        # mechanism: started from each relabeling it gave the same picks as
        # from the current numbering alone (200 starts, 3 picks), so they go.
        atom_maps = [m for m in atom_maps if not getattr(m, "relabeling", False)
                     and not (atom_map.is_identity and not m.is_identity)]
        if atom_map.is_identity and not any(m.is_identity for m in atom_maps):
            atom_maps.insert(0, atom_map)
    n_explore = int(getattr(run_inputs.atom_mapping_inputs, "explore_mechanisms", 0) or 0)
    if n_explore > 0:
        # Mechanisms beyond SLAPMapper's minimal-edit ones compete too:
        # relays and exchanges through other molecules (a catalyst, solvent).
        from mepd.atom_mapping import catalytic_participants, explore_mechanisms, mechanism_key, realign_end_to_start
        explored = explore_mechanisms(start_structure, end_structure, seeds, max_new=n_explore)
        for m in explored:
            aligned = realign_end_to_start(m, end_structure)
            cats = catalytic_participants(start_structure, aligned)
            _say(f"Atom mapping: explored mechanism ({int(m.cost)} bond changes) "
                       f"{mechanism_key(start_structure, aligned)}"
                       + (f"  [catalytic: {', '.join(cats)}]" if cats else ""))
        atom_maps = list(atom_maps) + explored
    return atom_maps


def _map_and_build_pair(start_structure: Structure, end_structure: Structure, pair_from: str, run_inputs):
    """--pair-from start/end with --atom-mapping: the built endpoint depends
    on the mapping, so each candidate mapping's partner is built, minimized
    and its path compared (mepd.built_pair_mapping), rather than the mapping
    being chosen against the given geometry first. Returns the chosen pair
    (start, end) in the start's atom order, or None when the mapping cannot
    be checked (the caller then builds from the current numbering)."""
    if len(start_structure.symbols) != len(end_structure.symbols) or getattr(run_inputs, "qmmm", None):
        return None
    from mepd.atom_mapping import HAS_SLAPMAPPER, check_atom_mapping

    if not HAS_SLAPMAPPER:
        return None
    try:
        atom_map = check_atom_mapping(start_structure, end_structure)
        if atom_map is None:
            return None
        atom_maps = _candidate_atom_maps(start_structure, end_structure, atom_map, run_inputs)
    except Exception as exc:
        _warn(f"Could not enumerate candidate atom mappings ({type(exc).__name__}: {exc}); "
                   "building from the current numbering.")
        return None
    from mepd.built_pair_mapping import build_and_rank

    result = build_and_rank(start_structure, end_structure, atom_maps, pair_from, run_inputs, echo=typer.echo)
    if result is None:
        return None
    scores = result.scores
    if len(scores) > 1:
        _say(f"  compared by {result.quantity}:")
        ranked = sorted(scores, key=scores.get)
        for label in ranked[:5]:
            mark = "  <- chosen" if label == result.label else ""
            name = "current numbering" if label == "identity" else label
            _say(f"    {name:<22}{scores[label]:.4f}{mark}")
        if len(ranked) > 5:
            _say(f"    ({len(ranked) - 5} more, up to {scores[ranked[-1]]:.4f})")
    for line in result.notes:
        _say(line)
    _say("Keeping the current numbering." if result.label == "identity" else
               f"Renumbering the end's atoms ({result.label}) so they match the start's.")
    return result.start, result.end


def _check_endpoint_atom_mapping(
    start_structure: Structure,
    end_structure: Structure,
    atom_mapping: bool,
    run_inputs: "RunInputs",
    *,
    debug_dump: bool = False,
    output: Optional[Path] = None,
) -> Structure:
    """Sanity-check --start/--end's atom correspondence via SLAPMapper's
    Weisfeiler-Lehman-like/sequential-LAP atom-to-atom mapping (Koda,
    ChemRxiv 2025), warning if it disagrees with the identity mapping
    implied by the two Structures sharing the same atom indexing.

    Returns `end_structure` unchanged, unless `atom_mapping` is set and a
    disagreeing mapping was found -- in which case up to
    `run_inputs.atom_mapping_inputs.n_candidates` of SLAPMapper's candidate
    mappings are each scored (alongside "don't reindex") via
    `mepd.atom_mapping_selection.select_best_candidate`, and the
    best-scoring option's `end_structure` is returned.

    When `debug_dump` is set (and output is given), every candidate is
    additionally scored under all three selection metrics and dumped to
    `<output>/realign_debug/` -- see `_dump_atom_mapping_candidates`.
    """
    if len(start_structure.symbols) != len(end_structure.symbols):
        _say(
            "Note: skipping the --start/--end atom-mapping sanity check "
            "(--start and --end have different atom counts)."
        )
        return end_structure
    if getattr(run_inputs, "qmmm", None):
        _say("Note: QM/MM system: --start and --end keep the system's atom order (no atom mapping).")
        return end_structure

    from mepd.atom_mapping import HAS_SLAPMAPPER

    if not HAS_SLAPMAPPER:
        _say(
            "Note: skipping the --start/--end atom-mapping sanity check "
            "('slapmapper' not installed; `pip install mepd[aam]` to enable it)."
        )
        return end_structure

    from mepd.atom_mapping import check_atom_mapping

    try:
        atom_map = check_atom_mapping(start_structure, end_structure)
    except Exception as exc:
        _warn(
            f"Atom-mapping check between --start and --end failed "
            f"({type(exc).__name__}: {exc}); skipping."
        )
        return end_structure

    if atom_map is None:
        _warn(
            "Note: SLAPMapper could not find any --start/--end atom mapping "
            "(e.g. mismatched element composition); skipping the reordering check."
        )
        return end_structure

    if atom_map.is_identity and not atom_mapping:
        _say(
            "Atom mapping: the start's and end's atoms are already numbered the same way "
            "(checked with SLAPMapper), so nothing is renumbered."
        )
        _report_mapping(output, start_structure, end_structure, None, metric="", chosen="identity",
                        slapmapper_identity=True,
                        summary="SLAPMapper found the end already numbered like the start; nothing was renumbered.")
        return end_structure

    if not atom_mapping:
        return end_structure

    from mepd.atom_mapping_selection import build_candidates, select_best_candidate
    from mepd.inputs import MAPPING_CANDIDATE_LIMIT as n_candidates
    try:
        atom_maps = _candidate_atom_maps(start_structure, end_structure, atom_map, run_inputs)
    except Exception as exc:
        _warn(
            f"Could not enumerate --atom-mapping candidate mappings "
            f"({type(exc).__name__}: {exc}); keeping --end's original atom ordering."
        )
        return end_structure

    candidates = build_candidates(start_structure, end_structure, atom_maps)
    metric = run_inputs.atom_mapping_inputs.metric
    snap = metric in ("snap", "snap-gi-xtb")
    if [c.label for c in candidates] in ([], ["identity"]) and not (
            snap and candidates and _snap_relabels(candidates[0], start_structure)):
        # Nothing but the current numbering (no symmetric atoms to relabel).
        if atom_map.is_identity:
            _say("Atom mapping: the start's and end's atoms are already numbered the same way "
                       "(checked with SLAPMapper), so nothing is renumbered.")
        _report_mapping(output, start_structure, end_structure, candidates, metric=metric, chosen="identity",
                        slapmapper_identity=atom_map.is_identity,
                        summary="SLAPMapper found the end already numbered like the start, and no other numbering "
                                "(no symmetric atoms to relabel); nothing was renumbered.")
        return end_structure

    n_other = sum(c.label != "identity" for c in candidates)
    n_exp = sum(c.label != "identity" and getattr(c.atom_map, "explored", False) for c in candidates)
    if atom_map.is_identity and snap:
        _say(
            "Atom mapping: the start's and end's bonds already correspond (checked with SLAPMapper); comparing "
            "snap's placements of the symmetric atoms for the current numbering"
            + (f" and {n_exp} explored mechanism(s)" if n_exp else "") + f", by {metric}...")
    elif atom_map.is_identity:
        _say(
            f"Atom mapping: the start's and end's bonds already correspond (checked with SLAPMapper); comparing "
            f"the current numbering with {n_other - n_exp} relabeling(s) of its symmetric atoms"
            + (f" and {n_exp} explored mechanism(s)" if n_exp else "") + f", by {metric}...")
    else:
        # What SLAPMapper found, and what was added: the budget (n_candidates)
        # is filled with relabelings of symmetric atoms of its best mapping.
        n_sym = sum(c.label != "identity" and getattr(c.atom_map, "relabeling", False) for c in candidates)
        pool = (f"snap's placements of the symmetric atoms for " if snap else "") + \
            f"{n_other - n_sym - n_exp} numbering(s) from SLAPMapper"
        if n_sym:
            pool += f" and {n_sym} relabeling(s) of their symmetric atoms"
        if n_exp:
            pool += f", plus {n_exp} explored mechanism(s)"
        if len(atom_maps) >= n_candidates and not snap:
            pool += f" (capped at {n_candidates})"
        _say(
            f"Atom mapping: the end's atoms may be numbered differently from the start's. Comparing its "
            f"current numbering with {pool}, by {metric} (lower is better)..."
            if n_other < len(candidates) else
            f"Atom mapping: the end's atoms are in a different element order from the start's, so its current "
            f"numbering can't be used. Choosing among {pool}, by {metric} (lower is better)..."
        )

    debug_dump_dir = Path(output) / "realign_debug" if debug_dump and output is not None else None
    try:
        if debug_dump_dir is not None:
            result = _dump_atom_mapping_candidates(
                candidates, metric, start_structure, run_inputs, debug_dump_dir
            )
        else:
            result = select_best_candidate(
                candidates, metric, start_structure, run_inputs,
                veto_margin=run_inputs.atom_mapping_inputs.veto_margin,
            )
    except Exception as exc:
        _warn(
            f"Atom-mapping candidate selection failed ({type(exc).__name__}: {exc}); "
            "keeping --end's original atom ordering."
        )
        return end_structure

    # The best numbering and the current one, not every candidate (there can be hundreds).
    scores = result.scores
    what = result.quantity or metric
    if result.quantity:
        _say(f"  compared by {result.quantity}:")
    for line in getattr(result, "notes", None) or []:
        _say(f"  {line}")
    ranked = sorted(scores, key=scores.get)
    shown = [result.winner.label] + (["identity"] if result.winner.label != "identity" and "identity" in scores else [])
    for label in shown:
        role = "chosen" if label == result.winner.label else "current"
        name = "current numbering" if label == "identity" else label
        _say(f"  {role + ':':<10}{name:<20}{scores[label]:.4f}" + ("" if result.quantity else f"  ({metric})"))
    rest = [k for k in ranked if k not in shown]
    if rest:
        _say(f"  ({len(rest)} other(s), {scores[rest[0]]:.4f} to {scores[rest[-1]]:.4f}"
                   + ("" if result.quantity else f" {what}") + ")")

    if result.winner.label == "identity":
        _say("Keeping the end's current atom numbering: no other fits clearly better.")
    else:
        _say(f"Renumbering the end's atoms ({result.winner.label}) so they match the start's.")
    shown = list(candidates) if any(c.label == result.winner.label for c in candidates) \
        else [result.winner, *candidates]          # (snap's picks are candidates of their own)
    n = len(shown)
    _report_mapping(
        output, start_structure, end_structure, shown, metric=metric, chosen=result.winner.label,
        scores=scores, chains=getattr(result, "chains", None), quantity=result.quantity or "",
        notes=getattr(result, "notes", None), slapmapper_identity=atom_map.is_identity,
        summary=(f"{n} numbering{'s' if n != 1 else ''} of the end compared by {result.quantity or metric}; "
                 + ("the end's current numbering was kept." if result.winner.label == "identity"
                    else "the end was renumbered (the mapping used is marked ★).")))

    return result.winner.end_structure


def _report_mapping(output, start_structure, end_structure, candidates, *, metric: str, chosen: str, **kw) -> None:
    """<output>/atom_mapping.json for the Results page (mepd.mapping_report);
    never fails the run."""
    if output is None:
        return
    try:
        from mepd.atom_mapping_selection import MappingCandidate
        from mepd.mapping_report import build_report, write_report

        if not candidates:
            candidates = [MappingCandidate(label="identity", end_structure=end_structure, atom_map=None)]
        write_report(output, build_report(start_structure, end_structure, candidates, metric=metric,
                                          chosen=chosen, **kw))
    except Exception as exc:
        _warn(f"(Could not record the atom mapping for the Results page: {type(exc).__name__}: {exc})")


def _tree_leaves(tree_dir: Path) -> tuple[set[int], set[int]]:
    """(searched leaves, direct_only-rejected branches) of a tree on disk,
    from adj_matrix.txt and the node files alone (no NEB is loaded)."""
    import numpy as np

    adj = np.atleast_2d(np.loadtxt(tree_dir / "adj_matrix.txt"))
    n = adj.shape[0]
    has_children = {i for i in range(n) if any(adj[i, j] for j in range(i + 1, n))}
    searched = {i for i in range(n) if (tree_dir / f"node_{i}.xyz").exists() and i not in has_children}
    rejected = {i for i in range(n) if (tree_dir / f"node_{i}_rejected.xyz").exists()}
    return searched, rejected


def _uncharacterized_by_direct_only(tree_dir: Path) -> bool:
    """A pair tree in which --direct-only left nothing searched: some legs
    were dropped (whole branches, or pieces of a split) and no leaf with a
    search of its own remains."""
    try:
        searched, rejected = _tree_leaves(tree_dir)
    except Exception:
        return False
    dropped = bool(rejected) or any(tree_dir.glob("node_*_rejected_*.xyz"))
    return dropped and not searched


def direct_only_counts(pairs_dir: Path) -> dict:
    trees = [t for t in pairs_dir.glob("*/tree") if (t / "adj_matrix.txt").exists()]
    return {
        "legs_not_run": len(list(pairs_dir.glob("*/tree/node_*_rejected_*.xyz"))),
        "pairs_not_characterized": sum(1 for t in trees if _uncharacterized_by_direct_only(t)),
        "pairs_partly_characterized": sum(
            1 for t in trees if list(t.glob("node_*_rejected_*.xyz")) and not _uncharacterized_by_direct_only(t)),
    }


def _completed_tree_dirs(completion_dir: Path) -> list[Path]:
    """Pair trees that finished AND have a usable root. A pair whose very
    first NEB failed still writes a tree (adj_matrix.txt, with the root saved
    as node_0_failed.xyz) -- enough to mark it done for resuming, but
    network construction needs node_0.xyz, and one such tree used to sink
    the whole run's classification."""
    if not completion_dir.is_dir():
        return []
    trees = [
        p / "tree" for p in completion_dir.iterdir()
        if (p / "tree" / "adj_matrix.txt").exists()
    ]
    rejected = [t for t in trees if _uncharacterized_by_direct_only(t)]
    if rejected:
        _say(
            f"{len(rejected)} pair(s) had every leg between other species and were not characterized "
            "(--direct-only): " + ", ".join(t.parent.name for t in sorted(rejected))
        )
    trees = [t for t in trees if t not in rejected]
    failed = [t for t in trees if not (t / "node_0.xyz").exists()]
    if failed:
        _warn(
            f"Skipping {len(failed)} pair tree(s) whose root NEB failed: "
            + ", ".join(t.parent.name for t in sorted(failed))
        )
    return sorted(t for t in trees if (t / "node_0.xyz").exists())


_FORK_JOB: dict = {}


def _fork_call(item):
    return _FORK_JOB["fn"](item)


def _fork_probe_child() -> None:
    """Runs in a throwaway forked child: does the one thing every real worker
    does first -- launch a subprocess -- so a parent with broken `atfork`
    handlers kills us here, in milliseconds, instead of mid-run.

    `cwd` is set deliberately: CPython only takes its `posix_spawn` shortcut
    when `cwd is None`, and it is the `fork()` path that runs the handlers, so
    a probe without a `cwd` would not exercise what the engines exercise."""
    import subprocess

    subprocess.run(["/usr/bin/true"], capture_output=True, cwd="/")


def _fork_probe_ok() -> bool:
    """Whether a forked child survives long enough to start a subprocess."""
    import multiprocessing

    try:
        child = multiprocessing.get_context("fork").Process(target=_fork_probe_child)
        child.start()
    except OSError:
        return False
    child.join(60)
    if child.is_alive():  # wedged on an inherited lock rather than killed
        child.kill()
        child.join()
        return False
    return child.exitcode == 0


def _unsafe_to_fork() -> Optional[str]:
    """Why forking children that will launch subprocesses is unsafe here, or
    None when it is fine.

    On macOS a `pthread_atfork` handler that grabs a lock -- Tcl's notifier is
    the one mepd hit, via `import tkinter`; CoreFoundation and the ObjC runtime
    have the same shape -- hands every forked child a lock owned by a thread
    that does not exist there. The child is then SIGKILLed the instant it forks
    again to run the engine, with no Python traceback and nothing but a
    `BrokenProcessPool` to show for it. `mepd/__init__.py` keeps Tk out of the
    interpreter to prevent the known case; the probe then *verifies* the
    result rather than inferring it from which modules happen to be imported,
    since the handlers are registered in C and are not all enumerable from
    Python. One fork and one `/usr/bin/true`, a few times per run.

    On any platform, a process that has initialised CUDA (a GPU model such as
    a FAIR-Chem MLIP is loaded) cannot fork: the children inherit a CUDA
    context they cannot use."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return "CUDA is initialised in this process (a GPU model is loaded) and does not survive fork"
    if sys.platform != "darwin":
        return None
    if sys.modules.get("_tkinter") is not None:
        return "tkinter/Tcl is loaded, and its atfork handlers kill forked children on macOS"
    if not _fork_probe_ok():
        return "a probe child was killed as soon as it launched a subprocess (macOS atfork handlers)"
    return None


def _fork_map(fn, items: list, workers: int) -> list:
    """`[fn(x) for x in items]`, across `workers` forked processes when
    workers > 1. `fn` (and everything it closes over -- structures, the
    engine, RunInputs) reaches the children through fork rather than
    pickling; only each item and its return value are pickled.

    Falls back to running `items` serially -- rather than losing the run --
    both when forking is known to be unsafe up front and when a child dies
    without raising (SIGKILL, a segfaulting engine, the OOM killer), which
    `ProcessPoolExecutor` can only report as `BrokenProcessPool` for the whole
    pool. Every caller's `fn` writes its own files or returns its own value,
    so re-running is wasted time, never a wrong answer."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]

    unsafe = _unsafe_to_fork()
    if unsafe:
        _say(
            f"Running {len(items)} item(s) serially instead of across "
            f"{min(workers, len(items))} worker process(es): {unsafe}."
        )
        return [fn(x) for x in items]

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    _FORK_JOB["fn"] = fn
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(items)),
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            return list(pool.map(_fork_call, items))
    except BrokenProcessPool:
        _warn(
            "A worker process died without reporting an error (killed by the "
            "OS, or a crash inside the engine); retrying the remaining work "
            "serially. Re-run with --workers 1 to skip the failed attempt."
        )
        return [fn(x) for x in items]
    finally:
        _FORK_JOB.pop("fn", None)


def _run_msmep_pairs(
    structures,
    candidates: list,
    pairs_dir: Path,
    run_inputs: RunInputs,
    *,
    parallel: bool,
    parallel_workers: Optional[int],
    workers: int = 1,
    follow_up: Optional["PairFollowUp"] = None,
) -> None:
    """Runs recursive NEB/MSMEP autosplitting for each (i, j) structure-index
    pair in `candidates`, writing each result to <pairs_dir>/pair_<i>_<j>/tree/.

    Resumable with no separate manifest/state file: a pair is skipped if its
    directory already holds a completed tree, so re-running against the same
    `pairs_dir` picks up wherever a prior run left off -- the directory tree
    on disk IS the resume state. Shared by --network-completion (candidates
    seeded from newly-discovered intermediates) and `network-splits`
    (candidates seeded from a user-supplied list of minima).

    `workers` > 1 runs that many pairs at once, each in its own forked
    process; pairs share nothing but read-only inputs and write to their
    own pair_<i>_<j>/ directory.

    `follow_up` (`channels`: each pair's TS optimizations and IRCs) starts
    a pair's follow-up work as soon as its tree is written, in processes of
    its own, so it neither waits for the other searches nor holds up the
    next pair (see PairFollowUp).
    """
    from mepd.msmep import MSMEP
    import mepd.chainhelpers as ch

    todo = []
    for i, j in candidates:
        if (pairs_dir / f"pair_{i}_{j}" / "tree" / "adj_matrix.txt").exists():
            _say(f"Skipping pair ({i}, {j}): already completed.")
        else:
            todo.append((i, j))

    def _one(pair) -> None:
        i, j = pair
        pair_dir = pairs_dir / f"pair_{i}_{j}"
        tree_dir = pair_dir / "tree"
        # `todo` was filtered against the disk before the first attempt, but
        # `_fork_map` may re-run this list serially after a worker died; pairs
        # that did finish in the meantime are on disk and stay there.
        if (tree_dir / "adj_matrix.txt").exists():
            _say(f"Skipping pair ({i}, {j}): already completed.")
            return
        from mepd import search_control

        over = search_control.search_over(run_inputs.path_min_inputs)
        if over is not None:
            _say(f"Skipping pair ({i}, {j}): " + ("stop requested before it started." if over == "stopped"
                                                         else "--search-budget reached before it started."))
            return
        pair_dir.mkdir(parents=True, exist_ok=True)
        _say(f"Starting pair_{i}_{j} ({run_inputs.path_min_method} path search)...")
        # MSMEP's "endpoints already attempted elsewhere" dedup (meant to stop
        # redundant re-splitting of the same species discovered mid-recursion
        # within ONE pair's own tree) stores its cache directly on this shared
        # `run_inputs.path_min_inputs` object -- reset it before each pair so
        # it can't leak across pairs and wrongly skip a later, genuinely
        # different pair that merely resembles an earlier one under
        # `is_identical`'s generous default thresholds (this matters most for
        # `mepd channels`, where many pairs deliberately share the same
        # reactant/product molecular graph across different conformers).
        setattr(run_inputs.path_min_inputs, "attempted_pairs_payload", [])
        from mepd import progress as _progress

        # One live-view stream per pair: pairs run concurrently in separate
        # worker processes and must not overwrite each other's live file.
        _progress.set_live_stream(f"pair_{i}_{j}")
        stream_status = "failed"
        try:
            seed_chain = Chain.model_validate({
                "nodes": [structures[i], structures[j]],
                "parameters": copy.deepcopy(run_inputs.chain_inputs),
            })
            from mepd.interpolation import initial_chain as _initial_chain

            pair_chain = _initial_chain(seed_chain, run_inputs.chain_inputs, run_inputs.gi_inputs)
            msmep = MSMEP(inputs=run_inputs)
            if parallel:
                pair_history = msmep.run_parallel_recursive_minimize(
                    pair_chain, max_workers=parallel_workers
                )
            else:
                pair_history = msmep.run_recursive_minimize(pair_chain)
            pair_history.write_to_disk(tree_dir)
            stream_status = "done"
        except Exception as exc:
            _warn(f"Pair ({i}, {j}) failed ({type(exc).__name__}: {exc}); skipping.")
        finally:
            _progress.end_live_stream(stream_status)

    if workers > 1 and len(todo) > 1:
        _say(f"Running {len(todo)} path search(es), {min(workers, len(todo))} at a time "
                   f"({min(workers, len(todo))} worker process(es)).")
    if follow_up is None or workers <= 1 or not todo or _unsafe_to_fork():
        _fork_map(_one, todo, workers)
        return
    _fork_pairs_with_follow_up(_one, todo, pairs_dir, workers, follow_up)


@dataclass
class PairFollowUp:
    """Work that follows each finished pair search (`_run_msmep_pairs`):
    `tasks(pair_dir)` lists it (in the parent, from what the pair wrote) and
    `run(task)` does one (in a forked process; it writes its own files).
    At most `during` run at once while searches are still running or
    queued, `after` once none are (nothing else is left to use the cores).
    Whatever has not run when the pairs are done is the caller's to finish."""

    tasks: Callable[[Path], list]
    run: Callable[[object], object]
    during: int = 1
    after: int = 1


def _fork_pairs_with_follow_up(one, todo: list, pairs_dir: Path, workers: int, follow_up: PairFollowUp) -> None:
    """`_fork_map(one, todo, workers)`, with each pair's follow-up tasks
    queued into the same process pool as soon as that pair is done (pairs
    first: a free search slot never waits behind follow-up work)."""
    import collections
    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    workers = max(1, workers)
    size = max(workers + max(1, follow_up.during), follow_up.after)
    pairs, extra = collections.deque(todo), collections.deque()
    running: dict = {}
    _FORK_JOB["fn"] = lambda item: one(item[1]) if item[0] == "pair" else follow_up.run(item[1])
    _say(f"Each search's follow-up (TS optimization and IRC) starts as soon as it finishes: up to "
               f"{follow_up.during} worker process(es) beside the searches, {follow_up.after} once they are done.")
    try:
        with ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context("fork")) as pool:
            while pairs or extra or running:
                n_pairs = sum(kind == "pair" for kind, _ in running.values())
                while pairs and n_pairs < workers:
                    pair = pairs.popleft()
                    running[pool.submit(_fork_call, ("pair", pair))] = ("pair", pair)
                    n_pairs += 1
                limit = follow_up.during if (pairs or n_pairs) else follow_up.after
                while extra and len(running) - n_pairs < limit:
                    task = extra.popleft()
                    running[pool.submit(_fork_call, ("task", task))] = ("task", task)
                if not running:
                    break
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in done:
                    kind, item = running.pop(fut)
                    try:
                        fut.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:   # left to the caller's own pass
                        _say(f"{'Pair' if kind == 'pair' else 'Follow-up'} {item!r:.80}: "
                                   f"{type(exc).__name__}: {exc}")
                        continue
                    if kind == "pair":
                        i, j = item
                        try:
                            extra.extend(follow_up.tasks(pairs_dir / f"pair_{i}_{j}"))
                        except Exception as exc:
                            _say(f"After pair ({i}, {j}): {type(exc).__name__}: {exc}")
    except BrokenProcessPool:
        _warn(
            "A worker process died without reporting an error (killed by the "
            "OS, or a crash inside the engine); retrying the remaining work "
            "serially. Re-run with --workers 1 to skip the failed attempt."
        )
        for pair in todo:
            one(pair)
    finally:
        _FORK_JOB.pop("fn", None)


@dataclass
class TsIrcResult:
    ts_node: object
    irc_chain: Optional[Chain] = None


def _optimize_ts_and_irc(
    ts_guess_node,
    run_inputs: RunInputs,
    output: Path,
    *,
    run_irc: bool,
    label: str = "ts",
    direction=None,
) -> Optional["TsIrcResult"]:
    """Optimize a TS-guess node with the engine and, if requested, follow up
    with an IRC -- writes <label>.xyz (and <label>_irc.xyz) into `output`.

    Shared by `ts`, `run --use-tsopt`, and `channels`. Never raises/exits
    itself: returns a `TsIrcResult` (with `irc_chain=None` if `run_irc` was
    False or the IRC itself failed), or None on TS-optimization
    failure/no support, so each caller decides whether that is fatal (a
    standalone `ts` invocation should exit non-zero; a TS opt launched
    automatically after `run` should just warn and let the NEB result
    stand; `channels` needs the IRC chain itself to classify the result).
    """
    from mepd.compute import find_ts, irc
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.inputs import ChainInputs

    engine = run_inputs.engine
    frozen = isinstance(engine, FrozenAtomsEngine)
    if not callable(getattr(engine, "compute_transition_state", None)):
        _say(f"Engine {type(engine).__name__} does not support transition-state optimization.")
        return None
    _say(f"Optimizing transition state ({label})...")
    try:
        # A QM/MM search climbs along the given direction (a path's tangent).
        # QM/MM (frozen atoms): no frequency check, the whole system's Hessian
        # is out of reach.
        ts_kwds = run_inputs.ts_inputs.find_ts_keywords(engine) if getattr(run_inputs, "ts_inputs", None) else {}
        ts_node = find_ts(ts_guess_node, engine, v0=direction, echo=typer.echo,
                          **{**ts_kwds, "check_frequencies": False}) if frozen \
            else find_ts(ts_guess_node, engine, **ts_kwds)
    except Exception as exc:
        cause = exc.__cause__ or exc
        _warn(f"Transition-state optimization failed ({label}): {type(cause).__name__}: {cause}")
        return None

    output.mkdir(parents=True, exist_ok=True)
    ts_path = output / f"{label}.xyz"
    Chain.model_validate({"nodes": [ts_node], "parameters": ChainInputs()}).write_to_disk(ts_path)
    _say(f"Wrote optimized TS structure to {ts_path}")
    if not run_irc:
        return TsIrcResult(ts_node=ts_node)

    _say(f"Computing IRC ({label})...")
    # Beside the TS: why its IRC failed, so a result can say so (rather
    # than "no IRC was run").
    failed_fp = output / ("irc_failed.txt" if label == "ts" else f"{label}_irc_failed.txt")
    try:
        irc_chain = (irc(ts_node, engine, echo=typer.echo, minimize_ends=False) if frozen
                     else irc(ts_node, engine)).path
    except Exception as exc:
        cause = exc.__cause__ or exc
        _warn(f"IRC computation failed ({label}; {type(cause).__name__}: {cause}); TS structure was still written.")
        failed_fp.write_text(f"{type(cause).__name__}: {cause}\n")
        return TsIrcResult(ts_node=ts_node)

    failed_fp.unlink(missing_ok=True)
    irc_path = output / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
    irc_chain.write_to_disk(irc_path)
    _say(f"Wrote IRC path to {irc_path}")
    return TsIrcResult(ts_node=ts_node, irc_chain=irc_chain)


def _ts_guess_tasks_from_tree(tree, label_prefix: str) -> list[tuple[str, object]]:
    """Every leaf's TS-guess node from an MSMEP split-tree, labeled
    `<label_prefix>leaf_<leaf.index>` -- matches the naming `mepd run
    --use-tsopt` already uses for a bare tree (`ts_leaf_<index>`), so the two
    commands' outputs land on the same files in a shared --output directory."""
    from mepd.errors import EnergiesNotComputedError

    tasks = []
    for leaf in tree.ordered_leaves:
        if not leaf.data or not leaf.data.chain_trajectory:
            continue
        label = f"{label_prefix}leaf_{leaf.index}"
        try:
            guess_node = leaf.data.chain_trajectory[-1].get_ts_node()
        except EnergiesNotComputedError:
            # Its search ended without energies (e.g. the engine's SCF failed
            # on it): no TS guess, but the other leaves still get theirs.
            _warn(f"Skipping {label}: its path has no energies (the search on it failed).")
            continue
        tasks.append((label, guess_node))
    return tasks


def _pair_ts_guess_tasks(pair_dir: Path, charge: int, multiplicity: int) -> list[tuple[str, object]]:
    """The TS guesses of one `mepd channels` pair (<pairs>/pair_<i>_<j>/),
    labeled `ts_pair_<i>_<j>_leaf_<index>`; none if its search has not
    finished or found nothing."""
    from mepd.inputs import ChainInputs
    from mepd.TreeNode import TreeNode

    tree_dir = pair_dir / "tree"
    if not (tree_dir / "adj_matrix.txt").exists():
        return []
    if not (tree_dir / "node_0.xyz").exists():
        return []   # root search failed or was not run (--direct-only): already reported
    try:
        pair_tree = TreeNode.read_from_disk(
            tree_dir, chain_parameters=ChainInputs(), charge=charge, multiplicity=multiplicity,
        )
    except Exception as exc:
        _warn(f"Skipping {pair_dir.name}: could not load tree ({type(exc).__name__}: {exc})")
        return []
    return _ts_guess_tasks_from_tree(pair_tree, f"ts_{pair_dir.name}_")


def _ts_guess_tasks_from_network(pot) -> list[tuple[str, object]]:
    """Every edge's TS-guess node(s) from a network.json (`Pot`), labeled
    `ts_edge_<source>_<target>` (or `..._<i>` if an edge holds more than one
    NEB chain)."""
    tasks = []
    for u, v, data in pot.graph.edges(data=True):
        nebs = data.get("list_of_nebs") or []
        for i, chain in enumerate(nebs):
            try:
                guess_node = chain.get_ts_node()
            except Exception:
                continue
            suffix = "" if len(nebs) == 1 else f"_{i}"
            tasks.append((f"ts_edge_{u}_{v}{suffix}", guess_node))
    return tasks


def _collect_ts_guess_tasks(
    input_path: Path, charge: Optional[int], multiplicity: Optional[int]
) -> list[tuple[str, object]]:
    """Scan `input_path` for TS-guess nodes to optimize, auto-detecting its
    shape (mirrors `_load_visualization_object`'s detection, minus the bare
    NEB-history-folder and single-chain-xyz cases, which have no leaf/edge
    structure to pull guesses from):

    - a directory with adj_matrix.txt: an MSMEP split-tree (`mepd run
      --recursive`/`--parallel` output) -- one guess per leaf.
    - a directory with conformers/ and/or pairs/: a `mepd channels` output
      -- one guess per leaf of each completed pair's tree.
    - a .json file: a network.json (`Pot`) -- one guess per edge.
    - anything else: a single TS-guess xyz file, labeled "ts" (the original,
      pre-rework behavior).
    """
    from mepd.inputs import ChainInputs
    from mepd.nodes.node import StructureNode
    from mepd.pot import Pot
    from mepd.TreeNode import TreeNode

    tree_charge = charge if charge is not None else 0
    tree_multiplicity = multiplicity if multiplicity is not None else 1

    if input_path.is_dir():
        if (input_path / "adj_matrix.txt").exists():
            tree = TreeNode.read_from_disk(
                input_path, chain_parameters=ChainInputs(),
                charge=tree_charge, multiplicity=tree_multiplicity,
            )
            return _ts_guess_tasks_from_tree(tree, "ts_")

        pairs_dir = input_path / "pairs"
        if pairs_dir.is_dir() or (input_path / "conformers").is_dir():
            tasks: list[tuple[str, object]] = []
            if pairs_dir.is_dir():
                for pair_dir in sorted(pairs_dir.iterdir()):
                    tasks.extend(_pair_ts_guess_tasks(pair_dir, tree_charge, tree_multiplicity))
            return tasks

        raise typer.BadParameter(
            f"'{input_path}' is a directory but has neither adj_matrix.txt (a "
            "split-tree) nor conformers/ or pairs/ (a `mepd channels` output)."
        )

    if input_path.suffix == ".json":
        pot = Pot.read_from_disk(input_path)
        return _ts_guess_tasks_from_network(pot)

    guess_structure = _load_endpoint(input_path, charge, multiplicity)
    return [("ts", StructureNode(structure=guess_structure))]


