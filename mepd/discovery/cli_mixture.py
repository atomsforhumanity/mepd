"""`mepd discovery mixture`: explore every combination of the feed molecules
and merge the networks into one kinetic model (mepd.discovery.mixture)."""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import List, Optional

import typer

from mepd.cpus import available_cpus
from mepd.discovery.cli import discovery_app


@discovery_app.command("mixture")
def mixture(
    feed: List[str] = typer.Option(
        ..., "--feed", help="A feed molecule: SMILES or an xyz file of one molecule. Repeat for each."),
    concentration: List[float] = typer.Option(
        [], "--concentration",
        help="Starting concentration of each --feed, in order, in the standard state's unit (mol/L for 1M, atm "
             "of partial pressure for 1atm). Default 1 each."),
    temperature: float = typer.Option(..., "--temperature", help="Temperature (K) of the experiment."),
    time_s: float = typer.Option(..., "--time", help="Reaction time (s) of the experiment."),
    standard_state: str = typer.Option("1M", "--standard-state", help="'1M' (solution) or '1atm' (gas)."),
    copies: int = typer.Option(
        2, "--copies",
        help="Generation 0 explores every combination of 1 to this many feed molecules, copies of one molecule "
             "included (2: A, B, A+A, A+B, B+B), so reactants that react with themselves compete with the target."),
    generations: int = typer.Option(
        2, "--generations",
        help="Generation g > 0 seeds each populated product molecule next to each feed molecule (1 = feed "
             "combinations only)."),
    threshold: float = typer.Option(
        0.01, "--threshold",
        help="A product is populated when its peak concentration in the merged kinetics reaches this fraction of "
             "the largest feed concentration."),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True, help="RunInputs TOML (level of theory) for every seed and the merge."),
    expand_args: str = typer.Option(
        "", "--expand-args",
        help="Further options for every seed's `mepd discovery expand`, as one string. Unless it says otherwise: "
             "--generator collide, --cutoff-temperature/--temperature = --temperature, --cutoff-time/--time = "
             "--time, --standard-state, --rounds 2."),
    workers: int = typer.Option(max(1, available_cpus()), "--workers", help="Processes for all seeds together."),
    seed_workers: int = typer.Option(
        8, "--seed-workers", help="Workers per seed; --workers / this many seeds run at once."),
    output: Path = typer.Option(Path("mepd_mixture_output"), "--output", "-o", help="Directory to write into."),
) -> None:
    """Explore a reacting mixture: one explorer run per combination of the
    feed molecules (and, in later generations, per populated product next to
    each feed molecule), merged into one network of molecules whose kinetics
    start from the feed's concentrations. Writes mixture.json (molecules with
    their concentrations, reactions with barriers and extents) and seeds/<id>/
    (each explorer run, reusable: a rerun skips finished seeds)."""
    from mepd.cli_common import _load_structure_from_smiles_or_xyz, _open_run_inputs
    from mepd.discovery.mixture import explore_mixture

    if concentration and len(concentration) != len(feed):
        raise typer.BadParameter("give one --concentration per --feed, or none.")
    for name, value in (("--copies", copies), ("--generations", generations), ("--workers", workers),
                        ("--seed-workers", seed_workers)):
        if value <= 0:
            raise typer.BadParameter(f"{name} must be a positive integer.")
    conc = list(concentration) or [1.0] * len(feed)
    molecules = [(f, _load_structure_from_smiles_or_xyz(f, None, None), c) for f, c in zip(feed, conc)]
    given = shlex.split(expand_args)
    flags = {a.split("=")[0] for a in given if a.startswith("--")}
    defaults = {"--generator": "collide", "--cutoff-temperature": temperature, "--temperature": temperature,
                "--cutoff-time": time_s, "--time": time_s, "--standard-state": standard_state, "--rounds": 2}
    args = given + [str(x) for k, v in defaults.items() if k not in flags for x in (k, v)]
    if inputs is not None and "--inputs" not in flags and "-i" not in flags:
        args += ["--inputs", str(Path(inputs).resolve())]
    run_inputs = _open_run_inputs(inputs)
    typer.echo(f"Mixture of {len(feed)} feed molecule(s); seeds explored with: {' '.join(args)}")
    explore_mixture(molecules, output, args, run_inputs.engine, temperature=temperature, time_s=time_s,
                    standard_state=standard_state, copies=copies, generations=generations, threshold=threshold,
                    workers=workers, seed_workers=seed_workers, log=typer.echo)
    typer.echo(f"Written: {output / 'mixture.json'}")
