"""`mepd conformers`: sample the conformers of one molecule with RDKit or CREST
(mepd.conformers), optionally minimizing each at the --inputs level of theory;
or, with --ts, of a transition state around its frozen reacting core."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer


def conformers(
    structure: str = typer.Argument(..., help="The molecule: a path to an xyz file, or a SMILES string."),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", help="RunInputs TOML (level of theory for --minimize). Built-in defaults if omitted."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Override the molecular charge."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="Override the spin multiplicity."),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="'rdkit' (ETKDG + MMFF) or 'crest' (CREST metadynamics). Default ('auto'): rdkit; "
        "crest with --ts."),
    ts: bool = typer.Option(
        False, "--ts", help="The structure is a transition state: hold its reacting core (--freeze) at the TS "
        "distances and sample the rest with CREST. --minimize then TS-optimizes every conformer."),
    freeze: Optional[str] = typer.Option(
        None, "--freeze", help="With --ts: the reacting core, atom pairs (0-based) such as '0-12,3-11'. Default: "
        "the pairs whose distance the TS's imaginary mode (at the --inputs level) changes most -- the forming "
        "and breaking bonds."),
    n_conformers: int = typer.Option(20, "--n-conformers", help="Most distinct conformers to keep (0 = no cap)."),
    rmsd_cutoff: float = typer.Option(
        0.1, "--rmsd-cutoff", help="Conformers closer than this (bohr, symmetry-aware RMSD) count as one."),
    crest_method: str = typer.Option("--gfn2", "--crest-method", help="CREST's level: --gfn2, --gfnff or --gfn2//gfnff."),
    crest_ewin: float = typer.Option(6.0, "--crest-ewin", help="CREST energy window (kcal/mol)."),
    minimize: bool = typer.Option(
        True, "--minimize/--no-minimize",
        help="Minimize every conformer at the --inputs level (as `mepd optimize`), so their energies can be "
        "compared; with --ts, TS-optimize each (as `mepd ts`). Without it, energies are the backend's own (MMFF "
        "or CREST) and are not written as mepd energies."),
    validate_minima_with_hessian: bool = typer.Option(
        False, "--validate-minima-with-hessian/--no-validate-minima-with-hessian", "-H/-noH",
        help="With --minimize: Hessian-check every minimized conformer."),
    output: Path = typer.Option(Path("mepd_conformers_output"), "--output", "-o", help="Directory to write results into."),
) -> None:
    """Conformers of one molecule, from RDKit (ETKDG + MMFF) or CREST.

    Writes conformers.xyz (every kept conformer, most stable first by the
    backend's energy) and summary.json. With --minimize (default), also runs
    `mepd optimize` on them into optimized/ (opt_<i>.xyz + summary.json).
    CREST runs single-threaded (-T 1).

    With --ts the structure is a transition state: CREST samples it with
    the reacting core's distances held (`$constrain`, no topology checks),
    and mepd drops a conformer whose held distances moved or whose other
    bonds changed. --minimize then TS-optimizes each into optimized/
    (ts_<i>.xyz; summary.json with each TS's energy and how far its core
    moved), so you can tell which conformer gives the lowest TS.
    """
    from mepd.chain import Chain
    from mepd.cli_common import _load_structure_from_smiles_or_xyz
    from mepd.conformers import ConformerInputs, CrestInputs, generate_conformers
    from mepd.inputs import ChainInputs
    from mepd.nodes.node import StructureNode

    if backend in (None, "auto"):
        backend = "crest" if ts else "rdkit"
    if backend not in ("rdkit", "crest"):
        raise typer.BadParameter("--backend must be 'rdkit' or 'crest'.")
    if ts and backend != "crest":
        raise typer.BadParameter("--ts samples with CREST (constraints): leave out --backend or use crest.")
    if freeze and not ts:
        raise typer.BadParameter("--freeze is for a transition state: add --ts.")
    seed = StructureNode(structure=_load_structure_from_smiles_or_xyz(structure, charge, multiplicity))
    output.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    core: list = []
    if ts:
        core = parse_pairs(freeze) if freeze else _core_from_mode(seed, inputs)
        n = len(seed.structure.symbols)
        if not core or any(not (0 <= a < n) or i == j for i, j in core for a in (i, j)):
            raise typer.BadParameter(f"--freeze {freeze!r}: pairs of two different atoms, 0-based, of {n}.")
        typer.echo("Reacting core held fixed: " + ", ".join(
            f"{seed.structure.symbols[i]}{i}-{seed.structure.symbols[j]}{j} {_distance(seed, i, j):.3f} A"
            for i, j in core))
    solvent_args = _crest_solvent_args(inputs) if backend == "crest" else ()
    settings = ConformerInputs(backend=backend, n_conformers=n_conformers or None, rmsd_cutoff=rmsd_cutoff,
                               crest=CrestInputs(method=crest_method, ewin_kcal=crest_ewin, threads=1,
                                                 extra_args=solvent_args, constrain_pairs=tuple(core)))
    typer.echo(f"Sampling conformers with {backend}...")
    found = generate_conformers(seed, settings, stats)
    typer.echo(f"Kept {len(found)} conformer(s) ({stats})")
    Chain.model_validate({"nodes": [n.copy() for n in found], "parameters": ChainInputs()}).write_to_disk(
        output / "conformers.xyz")
    summary = {"structure": structure, "backend": backend, "stats": stats, "n_conformers": len(found),
               "minimized": bool(minimize), "conformers_file": str(output / "conformers.xyz")}
    if ts:
        summary["ts"] = True
        summary["frozen_pairs"] = [list(p) for p in core]
    (output / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if minimize and ts:
        _optimize_ts_conformers(found, core, inputs, rmsd_cutoff, output / "optimized")
    elif minimize:
        from mepd.cli import optimize

        typer.echo("Minimizing each conformer at the --inputs level...")
        optimize(structures=[output / "conformers.xyz"], inputs=inputs, charge=seed.structure.charge,
                 multiplicity=seed.structure.multiplicity, validate_minima_with_hessian=validate_minima_with_hessian,
                 hessian_minimum_frequency_cutoff=-20.0, hessian_minima_rescue_displacement=0.3,
                 output=output / "optimized")


# A TS-optimized conformer whose core distances moved more than this (A)
# from the seed TS's likely went to another TS.
CORE_TOLERANCE_ANG = 0.3


def parse_pairs(text: str) -> list[tuple[int, int]]:
    """'0-12,3-11' (or '0:12 3:11') -> [(0, 12), (3, 11)]."""
    pairs = []
    for item in text.replace(";", ",").replace(" ", ",").split(","):
        if not item:
            continue
        parts = item.replace(":", "-").split("-")
        if len(parts) != 2:
            raise typer.BadParameter(f"--freeze: {item!r} is not a pair 'i-j'.")
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def _crest_solvent_args(inputs: Optional[Path]) -> tuple:
    """CREST samples in the --inputs profile's implicit solvent ([solvation]:
    ALPB or GBSA; a profile can't ask for CPCM-X, which has no gradient)."""
    from mepd.cli_common import _open_run_inputs
    from mepd.solvation import get_solvent

    solvation = (_open_run_inputs(inputs).solvation or {}) if inputs is not None else {}
    solvent, model = solvation.get("solvent"), str(solvation.get("model", "alpb")).lower()
    if not solvent:
        return ()
    typer.echo(f"CREST samples in {solvent} ({model.upper()}), as the profile.")
    return tuple(get_solvent(solvent).flag(model))


def _distance(node, i: int, j: int) -> float:
    from qcref.constants import ANGSTROM_TO_BOHR

    x = node.structure.geometry
    return float(sum((x[i][k] - x[j][k]) ** 2 for k in range(3)) ** 0.5 / ANGSTROM_TO_BOHR)


def _core_from_mode(seed, inputs: Optional[Path]) -> list[tuple[int, int]]:
    """The TS's reacting core from its imaginary mode at the --inputs level."""
    from mepd.cli_common import _open_run_inputs
    from mepd.compute import frequencies
    from mepd.conformers import ts_reactive_core

    typer.echo("Finding the reacting core from the TS's imaginary mode (Hessian at the --inputs level)...")
    freqs = frequencies(seed, _open_run_inputs(inputs).engine)
    n = freqs.n_imaginary()
    if n == 0 or freqs.modes is None:
        raise typer.BadParameter(
            f"The structure has no imaginary mode at this level (lowest: {freqs.lowest()} cm^-1), so its "
            "reacting core can't be read from it: give --freeze, or TS-optimize it first (`mepd ts`).")
    if n > 1:
        typer.echo(f"Warning: {n} imaginary modes; the core is read from the lowest ({freqs.lowest(1)[0]:.0f} cm^-1).")
    return ts_reactive_core(seed.structure, freqs.modes[0])


def _optimize_ts_conformers(found, core, inputs: Optional[Path], rmsd_cutoff: float, out: Path) -> None:
    """TS-optimize each conformer (as `mepd ts`) into `out`; summary.json lists
    each TS's energy, how far its core distances moved from the seed TS, and
    which converged onto a TS found already."""
    import numpy as np

    from mepd._qcinf_compat import snap_rmsd
    from mepd.cli_common import _open_run_inputs, _optimize_ts_and_irc

    run_inputs = _open_run_inputs(inputs)
    seed = found[0]
    rows, kept = [], []
    typer.echo(f"TS-optimizing each of the {len(found)} conformer(s) at the --inputs level...")
    for k, node in enumerate(found):
        label = f"ts_{k}"
        res = _optimize_ts_and_irc(node, run_inputs, out, run_irc=False, label=label)
        if res is None:
            rows.append({"label": label, "status": "failed"})
            continue
        t = res.ts_node
        moved = max(abs(_distance(t, i, j) - _distance(seed, i, j)) for i, j in core)
        row = {"label": label, "status": "ok", "energy_hartree": float(t.energy), "core_change_ang": round(moved, 4),
               "same_core": moved <= CORE_TOLERANCE_ANG}
        mirror = t.structure.model_copy(update={"geometry": np.asarray(t.structure.geometry) * [1.0, 1.0, -1.0]})
        for other, onode in kept:
            try:
                if snap_rmsd(t.structure, onode.structure) < rmsd_cutoff:
                    row["same_ts_as"] = other
                    break
                if "mirror_of" not in row and snap_rmsd(mirror, onode.structure) < rmsd_cutoff:
                    # Its enantiomer (same energy): a distinct saddle point, one
                    # more way through the same barrier, so kept as a TS.
                    row["mirror_of"] = other
            except ValueError:
                continue
        else:
            kept.append((label, t))
        rows.append(row)
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        low = min(r["energy_hartree"] for r in ok)
        for r in ok:
            r["relative_kcal"] = round((r["energy_hartree"] - low) * 627.509474, 3)
        best = min([r for r in ok if r["same_core"]] or ok, key=lambda r: r["energy_hartree"])
        seed_row = rows[0] if rows[0]["status"] == "ok" else None
        gain = f", {seed_row['relative_kcal']:.2f} kcal/mol below the input's" if seed_row else ""
        mirrors = sum(1 for r in ok if r.get("mirror_of") and not r.get("same_ts_as"))
        typer.echo(f"{len(ok)} TS(s) optimized ({len(kept)} distinct"
                   + (f", of which {mirrors} the mirror image of another" if mirrors else "")
                   + f"); lowest: {best['label']}{gain}.")
        for r in ok:
            if not r["same_core"]:
                typer.echo(f"Warning: {r['label']}'s core moved {r['core_change_ang']:.2f} A from the input's: "
                           "probably another TS.")
    (out / "summary.json").write_text(json.dumps({"frozen_pairs": [list(p) for p in core], "ts": rows}, indent=1))
