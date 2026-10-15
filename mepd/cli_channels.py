"""`mepd channels`: the command line for a channel search. The search
itself (conformer pools, per-pair atom mapping, MSMEP path search, TS/IRC,
classification into `channels/`, `alternate-channels/` and
`offtarget-exit-channels/`) is `mepd.channel_search`; Python callers use
`mepd.channels(...)` / `mepd.read_channels(...)`.

`channels` (the command function) is imported and registered onto `app` by
`mepd/cli.py` rather than decorated here, so this module never imports
`mepd.cli`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mepd.atom_mapping_metrics import HELP as _MAPPING_HELP, OFFERED as _OFFERED_METRICS
from mepd.channel_search import (  # noqa: F401  (also re-exported for older callers)
    BUILT_PAIRS_JSON,
    Channel,
    ChannelSearchError,
    ChannelsOptions,
    ChannelsResult,
    Step,
    _MAX_ALTERNATE_CHANNELS,
    _MAX_ALTERNATE_CHANNEL_STEPS,
    _TS_FINGERPRINT_TOL_BOHR,
    _built_pairs_by_mechanism,
    _choose_built_pairs,
    _cluster_by_ts_identity,
    _cluster_ts_energy,
    _describe_mechanism,
    _discover_channels,
    _expand_pairs_by_mechanism,
    _gi_near_lowest_rmsd,
    _load_built_pairs,
    _load_conformer_pool,
    _load_ts_and_irc_from_disk,
    _minimize_conformer_pool,
    _numbered,
    _read_step,
    _read_xyz_chain,
    _register_route_class,
    _same_ts,
    _save_built_pairs,
    _species_label,
    _ts_and_irc_once,
    _ts_opt_failed,
    _ts_opt_failed_path,
    _write_classified_group,
    _write_multistep_group,
    apply_options,
    read_channels,
    run_channels,
)
from mepd.cli_common import (
    _check_endpoint_options,
    _echo_run_inputs_summary,
    _load_structure_from_smiles_or_xyz,
    _open_run_inputs,
    _reaction_endpoints,
    print_log_on_stdout,
)
from mepd.errors import InputsError


def channels(
    start: Optional[str] = typer.Option(
        None, "--start", help="Path to the reactant-endpoint xyz file, or a SMILES string (or use --reaction)."),
    end: Optional[str] = typer.Option(
        None, "--end", help="Path to the product-endpoint xyz file, or a SMILES string (or use --reaction)."),
    reaction: Optional[str] = typer.Option(
        None, "--reaction",
        help="Both endpoints as one reaction SMILES, 'reactants>>products' (agents, as in "
        "'reactants>agents>products', are ignored). Atom map numbers ([CH3:1]...) are used when every "
        "heavy atom has one; otherwise SLAPMapper maps the reaction. Instead of --start/--end.",
    ),
    method: str = typer.Option(
        "conformers", "--method",
        help="Seed-generation strategy for perturbing the search to surface "
        "alternate TS channels between --start and --end. Currently only "
        "'conformers' (conformer sampling of each endpoint, see --backend) is "
        "implemented; more methods may be added later.",
    ),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True,
        help="Path to a RunInputs TOML file. Uses built-in defaults if omitted.",
    ),
    charge: Optional[int] = typer.Option(
        None, "--charge", help="Override the molecular charge on both endpoints."
    ),
    multiplicity: Optional[int] = typer.Option(
        None, "--multiplicity", help="Override the spin multiplicity on both endpoints."
    ),
    atom_mapping: bool = typer.Option(
        True, "--atom-mapping/--no-atom-mapping",
        help="Every run checks whether SLAPMapper's Weisfeiler-Lehman-like/"
        "sequential-LAP atom-to-atom mapping (Koda, ChemRxiv 2025) between "
        "--start and --end agrees with their shared input atom ordering, "
        "warning if not. Passing this flag additionally reindexes --end's "
        "atoms (and therefore every product conformer generated from it) to "
        "match the suggested mapping instead of just warning. No-op when "
        "--start/--end are both SMILES or atom counts differ.",
    ),
    debug_dump: bool = typer.Option(
        False, "--debug-dump",
        help="Write extra diagnostic structures to disk for inspection (e.g. via "
        "`mepd visualize`): (1) when --start/--end is SMILES and --minimize-ends "
        "runs its pre-conformer-sampling minimization, the minimized endpoints go "
        "to <output>/smiles_minimization_debug/; (2) when --atom-mapping triggers "
        "its candidate-mapping selection, every candidate's interpolated path "
        "(xyz + energies) and a scores.txt comparing all three selection metrics "
        "per candidate go to <output>/realign_debug/.",
    ),
    atom_mapping_candidates: int = typer.Option(
        3, "--atom-mapping-candidates",
        help="--pair-from with --atom-mapping: the most path searches. Every conformer builds its partner "
        "once per candidate mapping (at most (reactant + product conformers) x mappings built pairs); if "
        "there are more, all are ranked by the xtb peak along their interpolation, above the lowest start "
        "among them, and the lowest are searched. Unused otherwise.",
    ),
    explore_mechanisms: int = typer.Option(
        0, "--explore-mechanisms",
        help="Also consider up to this many mechanisms beyond SLAPMapper's minimal-edit ones: relays and "
        "exchanges through other molecules (catalyst, solvent), found by swapping same-element atoms or "
        "groups with at most +2 bond edits per swap, with no rule about which atoms may move. Each is "
        "scored like SLAPMapper's (snap + GI + xtb); ones in which a molecule takes part and is "
        "regenerated are labelled [catalytic: ...]. 0 = off.",
    ),
    relax_mechanisms: Optional[int] = typer.Option(
        None, "--relax-mechanisms",
        help="snap-gi-xtb: rank mappings by relaxed path energy. The N best by raw xtb peak (plus SLAPMapper's own "
        "minimal mapping, always) have their interpolated paths relaxed except for the bonds that break or form, "
        "and the lowest relaxed barrier wins; a mapping with more bond changes must beat SLAPMapper's by more than "
        "--relax-margin. Default: the profile's atom_mapping_inputs.relax_top (4); 0 = raw peaks only.",
    ),
    relax_margin: Optional[float] = typer.Option(
        None, "--relax-margin",
        help="kcal/mol by which a mapping with more bond changes than SLAPMapper's minimal one must beat its relaxed "
        "barrier (else SLAPMapper's is kept). Default: the profile's atom_mapping_inputs.relax_margin (5).",
    ),
    atom_mapping_metric: str = typer.Option(
        "snap-gi-xtb", "--atom-mapping-metric",
        help="--atom-mapping: " + _MAPPING_HELP + " One of: " + ", ".join(_OFFERED_METRICS) + ".",
    ),
    atom_mapping_rmsd_window: Optional[float] = typer.Option(
        None, "--atom-mapping-rmsd-window",
        help="rmsd-geodesic: compute GI paths for the conformer pairs within this many standard deviations "
        "of each mechanism's lowest endpoint RMSD. Default: the profile's atom_mapping_inputs.rmsd_window "
        "(1). Larger keeps more of what GI paths over every pair would pick, at more cost."),
    atom_mapping_gi_variants: Optional[int] = typer.Option(
        None, "--atom-mapping-gi-variants",
        help="rmsd-geodesic: GI paths for this many lowest-endpoint-RMSD symmetry variants per pair and "
        "mechanism (0 = all). Default: atom_mapping_inputs.gi_variant_cap (20)."),
    atom_mapping_veto_margin: float = typer.Option(
        0.0, "--atom-mapping-veto-margin",
        help="--atom-mapping: a non-identity candidate must beat 'don't "
        "reindex' by more than this (in --atom-mapping-metric's own units) to "
        "be adopted; otherwise the original --end ordering is kept even if "
        "some candidate scored marginally better. Default 0.0 is a pure "
        "best-of-N with no calibrated stability margin yet.",
    ),
    rebuild_splits_from: str = typer.Option(
        "none", "--rebuild-splits-from",
        help="Experimental: at every recursive split, rebuild each new leg's product in the frame of its "
        "start (start), or its start in the frame of its product (end) -- each atom where it is, the "
        "new bonds pulled to length, minimized -- kept only if bonds and stereochemistry survive. "
        "Legs then meet at different conformers of an intermediate. none: legs end where the search found them.",
    ),
    atom_mapping_recheck_splits: bool = typer.Option(
        False, "--atom-mapping-recheck-splits",
        help="Experimental: also re-run the same best-of-N atom-mapping "
        "selection (--atom-mapping-metric etc.) at every new (reactant, "
        "product) pair MSMEP's recursive splitting discovers, not just the "
        "original --start/--end pair. Independent of --atom-mapping. Atom "
        "order is otherwise preserved throughout the recursion, so this only "
        "changes anything for a split whose own reactant/product pair "
        "happens to have SLAPMapper-detectable symmetry.",
    ),
    backend: str = typer.Option(
        "rdkit", "--backend",
        help="Conformer-generation backend: 'rdkit' (ETKDG embedding + MMFF "
        "relaxation; cheap) or 'crest' (CREST iterative metadynamics; needs the "
        "`crest` binary, much more expensive, see --crest-*). Either way the "
        "candidates are deduplicated by snap-RMSD (--rmsd-cutoff), capped at "
        "--n-conformers, and re-minimized with the engine (--minimize-ends).",
    ),
    crest_method: str = typer.Option(
        "--gfn2", "--crest-method",
        help="--backend crest: CREST sampling-level flag, e.g. '--gfn2', "
        "'--gfnff', or '--gfn2//gfnff'.",
    ),
    crest_threads: int = typer.Option(
        1, "--crest-threads", help="--backend crest: CREST's -T thread count.",
    ),
    crest_ewin: float = typer.Option(
        6.0, "--crest-ewin",
        help="--backend crest: CREST's --ewin energy window (kcal/mol) for "
        "retained conformers.",
    ),
    crest_timeout: float = typer.Option(
        3600.0, "--crest-timeout",
        help="--backend crest: wall-clock limit (s) for one CREST call.",
    ),
    crest_nci: bool = typer.Option(
        True, "--crest-nci/--no-crest-nci",
        help="--backend crest: run CREST in NCI mode (an ellipsoidal wall that "
        "keeps a complex together) for endpoints made of more than one molecule.",
    ),
    complex_energy_tol: float = typer.Option(
        0.5, "--complex-energy-tol",
        help="For endpoints made of more than one molecule: after minimization, "
        "drop a conformer when every molecule's own conformation matches an "
        "already-kept one (snap-RMSD < --rmsd-cutoff, molecule by molecule) and "
        "the energies agree within this many kcal/mol. 0 disables.",
    ),
    n_conformers: int = typer.Option(
        50, "--n-conformers",
        help="Maximum number of distinct conformers to keep for EACH endpoint, "
        "after generation and RMSD-based deduplication run to completion "
        "uncapped. If dedup still leaves more than this, the pool is capped by "
        "farthest-point selection (maximize each new pick's distance to the "
        "nearest already-kept conformer) rather than truncation, so a capped "
        "pool still spans the conformational landscape broadly instead of "
        "clustering around the low-energy end. 0 = no cap: keep every distinct "
        "conformer the backend's own parameters let through (CREST's "
        "--crest-ewin; RDKit's --n-embed and --rdkit-ewin) -- some molecules "
        "are combinatorially flexible enough that this is thousands of pairs.",
    ),
    n_embed: int = typer.Option(
        0, "--n-embed",
        help="Number of raw conformer embeddings to attempt per endpoint before "
        "deduplication (rdkit backend only). Should comfortably exceed --n-conformers. "
        "0 = pick from rotatable-bond count (50 / 200 / 300 for <=7 / 8-12 / >12).",
    ),
    rdkit_torsion_prefs: str = typer.Option(
        "both", "--rdkit-torsion-prefs",
        help="--backend rdkit: embed with ETKDG's experimental torsion preferences "
        "('etkdg'), without them ('none'), or both pooled ('both', default). The "
        "preferences alone never sample e.g. s-cis dienes.",
    ),
    rdkit_ewin: Optional[float] = typer.Option(
        None, "--rdkit-ewin",
        help="--backend rdkit: keep only embeddings within this many kcal/mol "
        "(MMFF94) of the lowest -- the counterpart of --crest-ewin. Default: no window.",
    ),
    pair_from: str = typer.Option(
        "both", "--pair-from",
        help="both: sample both endpoints' conformers and pair every reactant conformer with every product "
        "conformer. start: sample only the reactant's conformers and build each one's product in its frame "
        "(each atom where it is, the product's bonds pulled to length, minimized), so the product geometries "
        "are sampled through the reactant's and only what reacts moves; each conformer is searched with its "
        "own product, under the endpoint-level atom mapping. end: the same from the product side. "
        "start+end: both -- every reactant conformer with a product built from it, and every product "
        "conformer with a reactant built from it.",
    ),
    built_pairs: Optional[Path] = typer.Option(
        None, "--built-pairs",
        help="--pair-from with --atom-mapping: the ranked built pairs an earlier run saved "
        "(<its output>/built_pairs), used instead of building them again. With a higher "
        "--atom-mapping-candidates the next ones down its list are searched; pairs searched already are "
        "skipped (their folders are done).",
    ),
    built_cap_per_mechanism: bool = typer.Option(
        False, "--built-cap-per-mechanism",
        help="--pair-from with --atom-mapping: search the lowest --atom-mapping-candidates built pairs of "
        "each mechanism, instead of the lowest of all of them together.",
    ),
    search_built: Optional[str] = typer.Option(
        None, "--search-built",
        help="With --built-pairs: search exactly these entries of its list (comma-separated labels, e.g. "
        "built_7,built_12) instead of the lowest --atom-mapping-candidates.",
    ),
    rmsd_cutoff: float = typer.Option(
        0.1, "--rmsd-cutoff",
        help="Minimum pairwise RMSD (bohr) for two conformers of the same endpoint "
        "to count as distinct.",
    ),
    random_seed: int = typer.Option(0, "--random-seed", help="Random seed for conformer embedding."),
    start_pool: Optional[Path] = typer.Option(
        None, "--start-pool",
        help="Use this multi-frame xyz (with its .energies) as the start endpoint's final conformer pool "
        "instead of sampling one: e.g. conformers/start_pool.xyz of an earlier run with the same "
        "sampler settings. With --minimize-ends the pool must already be minimized at this run's "
        "level of theory; it is used as is (no sampling, no minimization).",
    ),
    end_pool: Optional[Path] = typer.Option(
        None, "--end-pool", help="As --start-pool, for the end endpoint.",
    ),
    minimize_ends: bool = typer.Option(
        True, "--minimize-ends/--no-minimize-ends",
        help="Optimize endpoint geometries with the QM engine. On by default. "
        "This governs two separate minimizations: (1) BEFORE conformer sampling, "
        "if --start/--end was given as SMILES (an RDKit/openbabel-embedded guess, "
        "not a real minimum), that raw embedded structure is minimized at your "
        "input level of theory first, so conformer sampling starts from an actual "
        "minimum rather than an arbitrary embedding; and (2) every generated "
        "conformer is optimized before pairing, since RDKit/MMFF conformers are "
        "not QM minima either. A conformer that fails to converge in step (2) is "
        "dropped rather than failing the whole run; a SMILES endpoint that fails "
        "to converge in step (1) is a hard stop (see `mepd run --minimize-ends`).",
    ),
    max_pairs: int = typer.Option(
        0, "--max-pairs",
        help="Hard cap on the number of reactant x product conformer pairs considered "
        "(before --atom-mapping expands them per mechanism). 0 (default) = no cap: "
        "selection is left to --pairs-per-mechanism.",
    ),
    conformers_only: bool = typer.Option(
        False, "--conformers-only",
        help="Stop after building, minimizing, deduplicating and atom-mapping the "
        "conformer pools (writes conformers/ and stats.json) -- i.e. before any "
        "path search. For sizing a run: how many pairs would it take, and what "
        "did conformer generation cost?",
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel",
        help="Run each pair's recursive autosplitting (MSMEP) with branches "
        "evaluated in parallel, whenever a pair's path actually splits at an "
        "intermediate. On by default: for a single interactive query this is "
        "free extra parallelism on top of --workers, which alone leaves most "
        "cores idle once there are only a handful of pairs left to search.",
    ),
    parallel_workers: Optional[int] = typer.Option(
        None, "--parallel-workers",
        help="Concurrent workers for --parallel, per pair. Default: auto -- "
        "coordinated with --workers so workers x parallel_workers fills the "
        "machine (cpu_count // workers) rather than MSMEP's own context-free "
        "min(4, cpu count) default.",
    ),
    pairs_per_mechanism: int = typer.Option(
        3, "--pairs-per-mechanism",
        help="--atom-mapping: every conformer pair is mapped once per mechanism "
        "SLAPMapper allows (e.g. [3,3] and [1,3] shifts for a Claisen); for each "
        "mechanism, only the K pairs its geodesic score (--atom-mapping-metric) "
        "likes best get a path search. 0 = every pair x every mechanism (slow, "
        "mostly redundant). Sampled pairs only: --pair-from builds are capped by "
        "--atom-mapping-candidates. See docs/channels_candidates.md.",
    ),
    workers: int = typer.Option(
        0, "--workers", "-j",
        help="Run this many conformer pairs' MSMEP searches at once, and "
        "afterwards this many TS optimizations + IRCs at once, each in its own "
        "process. Stacks with --parallel (branches within one pair): together "
        "they're what determines whether a single query actually uses more "
        "than one core. 0 (default) = auto: size to the number of independent "
        "searches at each stage, capped at the machine's core count, so a "
        "single interactive query uses the whole machine without needing to "
        "be told to.",
    ),
    search_budget: float = typer.Option(
        0.0, "--search-budget",
        help="Wall-clock seconds (counted from the start of the command) after "
        "which no new path-search work starts: pairs not yet begun are skipped "
        "and a search that finishes is not split further (it is kept as a "
        "'time_budget' leaf with its own path, so every tree stays continuous). "
        "TS optimization, IRC and channel classification then run on whatever "
        "was found, so a slow reaction still yields its channels instead of "
        "being killed with nothing. Searches already queued at the deadline "
        "still run once. 0 (default) = no budget.",
    ),
    direct_only: Optional[bool] = typer.Option(
        None, "--direct-only/--allow-multistep",
        help="--direct-only: focus compute on the queried pair. When a path search splits, only the pieces "
        "with at least one end at a queried species (the start or end, in any conformer or stereo variant) "
        "are run; a leg between two other species is not. E.g. A->B splitting into A->C, C->D, D->A', A'->B "
        "runs A->C, D->A' and A'->B, not C->D. Default: the profile's path_min_inputs.direct_only, else "
        "every piece is followed.",
    ),
    validate_minima_with_hessian: bool = typer.Option(
        True, "--validate-minima-with-hessian/--no-validate-minima-with-hessian", "-H/-noH",
        help="When a minima-based autosplit is proposed during each pair's MSMEP, "
        "compute Hessians for optimized split candidates and reject candidates "
        "with significant imaginary modes. On by default -- this is a "
        "correctness check, not a convenience.",
    ),
    hessian_minimum_frequency_cutoff: float = typer.Option(
        -20.0, "--hessian-minimum-frequency-cutoff",
        help="Minimum allowed frequency (cm^-1) for --validate-minima-with-hessian.",
    ),
    hessian_minima_rescue_displacement: float = typer.Option(
        0.3, "--hessian-minima-rescue-displacement",
        help="Displacement (bohr) applied along the lowest-frequency mode when "
        "rescuing a Hessian-rejected minimum, for --validate-minima-with-hessian.",
    ),
    output: Path = typer.Option(
        Path("mepd_channels_output"), "--output", "-o",
        help="Directory to write seed pools, per-seed trees, TS-opt+IRC "
        "results, classified channels/alternate-channels/offtarget-exit-"
        "channels, and the completed byproduct network into.",
    ),
) -> None:
    """Multi-channel TS discovery for a fixed --start/--end pair: generate
    alternate seed guesses (--method), run a recursive NEB/MSMEP for every
    seed, optimize a TS and IRC for every resulting leaf, then classify by
    walking the reaction graph those IRCs span.

    A leaf whose IRC reconnects the requested --start/--end pair in one step
    is a channel, and lands in <output>/channels/ (deduplicated into
    distinct TS classes, e.g. 4 conformer-pair runs converging to 2 real
    channels). A *sequence* of steps that gets from --start to --end through
    discovered intermediates is an alternate channel, and lands in
    <output>/alternate-channels/ as one folder per route with a step_<n>/
    per leg. A step that leaves some minimum but from which no onward
    sequence of discovered steps ever reaches --end is an off-target exit
    channel, and lands in <output>/offtarget-exit-channels/. Everything
    completed is also combined into one byproduct network at
    <output>/network.json."""
    args = {k: getattr(v, "default", v) if isinstance(v, typer.models.OptionInfo) else v
            for k, v in dict(locals()).items()}   # called as a plain function: typer defaults
    start, end, reaction = args["start"], args["end"], args["reaction"]
    charge, multiplicity, output = args["charge"], args["multiplicity"], Path(args["output"])
    _check_endpoint_options(start, end, reaction)
    options = ChannelsOptions(**{k: args[k] for k in ChannelsOptions.__dataclass_fields__})
    try:
        options.validate()
    except InputsError as exc:
        raise typer.BadParameter(str(exc))

    from mepd.nodes.node import StructureNode

    run_inputs = _open_run_inputs(args["inputs"])
    apply_options(run_inputs, options)
    _echo_run_inputs_summary(run_inputs)

    if reaction is not None:
        start_structure, end_structure = _reaction_endpoints(reaction, charge, multiplicity)
        guesses = True
    else:
        guesses = not Path(start).exists() or not Path(end).exists()
        start_structure = _load_structure_from_smiles_or_xyz(start, charge, multiplicity)
        end_structure = _load_structure_from_smiles_or_xyz(end, charge, multiplicity)
    start_node = StructureNode(structure=start_structure)
    end_node = StructureNode(structure=end_structure)

    print_log_on_stdout()
    try:
        run_channels(start_node, end_node, run_inputs, options, output, endpoints_are_guesses=guesses)
    except InputsError as exc:
        raise typer.BadParameter(str(exc))
    except ChannelSearchError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)
