"""`mepd orbitals`: the molecular orbitals along a reaction path, followed from
frame to frame (mepd.orbital_path), at a DFT level computed with linus.

Writes <output>/orbitals.json (energies of the orbitals around the frontier at
every frame computed, the lines that follow each orbital along the path, what
each orbital is, and a few sentences on what happens), <output>/frames.xyz
(the frames computed) and <output>/frames/<k>.molden.gz (their orbitals).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


def orbitals_(
    path: Path = typer.Argument(..., exists=True, help="The path as a multi-frame xyz file (an IRC, a path piece)."),
    charge: int = typer.Option(0, "--charge"),
    multiplicity: int = typer.Option(1, "--multiplicity"),
    functional: str = typer.Option("wb97x-d3bj", "--functional", help="Any functional linus knows."),
    basis: str = typer.Option("def2-svp", "--basis", help="Any basis set linus knows."),
    frames: int = typer.Option(41, "--frames", min=2, help="Frames computed, spread evenly along the path's length "
                                                            "(both ends and the TS always among them)."),
    ts_frame: Optional[int] = typer.Option(None, "--ts-frame", min=1, help="The TS's frame in the file (1-based)."),
    window: int = typer.Option(6, "--window", min=1, help="Orbitals shown each side of the frontier."),
    stability: bool = typer.Option(True, "--stability/--no-stability",
                                   help="A closed shell: also its broken-symmetry (unrestricted) solution at every "
                                        "frame, to find where the closed-shell orbitals fail (diradical character)."),
    threads: int = typer.Option(4, "--threads", min=1),
    output: Path = typer.Option(Path("mepd_orbitals_output"), "--output", "-o"),
):
    from mepd import orbital_path as op

    try:
        import linus  # noqa: F401
    except ImportError:
        typer.echo("mepd orbitals needs linus (pip install afh-linus[qc]); it is in the web image.")
        raise typer.Exit(code=1)
    symbols, coords = op.read_frames(path.read_text())
    typer.echo(f"Orbitals along {path.name}: {len(coords)} frames, {min(frames, len(coords))} computed at "
               f"{functional}/{basis} (linus), charge {charge}, multiplicity {multiplicity}.")
    doc = op.analyze(symbols, coords, charge, multiplicity, functional, basis, output,
                     ts_index=ts_frame - 1 if ts_frame else None, count=frames, threads=threads, window=window,
                     stability=stability,
                     log=lambda m: typer.echo(m))
    ok = sum(f is not None for f in doc["frames"])
    typer.echo(f"{ok} of {len(doc['frames'])} frames computed; {len(doc['lines'])} orbitals followed "
               f"({doc['seconds']:.0f} s). {output / 'orbitals.json'}")
    for s in doc["story"][:6]:
        typer.echo(f"  {s['text']}")
    if not ok:
        raise typer.Exit(code=1)
