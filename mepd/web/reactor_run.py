"""A Nanoreactor run with the Reactor's forces: `mepd reactor-run`, what Launch nanoreactor starts when the setup
page's Collide is on.

`mepd discovery nanoreactor` drives its MD with xtb's own MD (or a profile's engine) and the piston wall only.
The collision pulls live in the Reactor's MD (LangevinMD on axtb, with mepd.discovery.langevin.pull_pairs and
reactor_generator.collision_pair, as the interactive reactor and network growth's reactor runs use them). This
runs that MD on the packed box: relaxed first, the piston swinging the wall, and every so often two molecules
(or two atoms of one) pulled together, as network growth does it (reactor_generator.OPTIONS: pulls of 300 fs,
gaps drawn around 200 fs). It writes the trajectory as it goes in the nanoreactor's layout (md/packed.xyz,
relax.xyz, reactor.xyz, segment_kkk.xyz, schedule.json with the forces and each pull, live_events.json), so the
job's Reactor tab shows it live; a stop file (Stop & analyze) ends the MD early. Then it hands the trajectory
to `mepd discovery nanoreactor --trajectory`, which finds the reactions and refines them at the compute
profile's level, as it does for an interactive reactor's Stop & analyze.

Only explorer functions are called here; mepd/discovery is the explorer team's, unchanged.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

TRAJ_FS = 2.0          # frame spacing the event detection reads (the nanoreactor's --dump)
LIVE_FRAMES = 125      # frames per segment file the live view reads (250 fs)
METHODS = {"gfn2": "gfn2", "gxtb": "gxtb-v2.0.1"}   # what drives this MD (axtb)


def _write_frames(fp: Path, symbols, frames, comment: str = "") -> None:
    text = "".join(f"{len(symbols)}\n{comment}\n" + "".join(f"{e} {x:.5f} {y:.5f} {z:.5f}\n"
                                                           for e, (x, y, z) in zip(symbols, f)) for f in frames)
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(fp)


def reactor_run(
    molecules: list[str] = typer.Argument(..., help="What goes in: SMILES or xyz files, each optionally with a "
                                                    "count ('O*8' or '8*O')."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", exists=True,
                                          help="RunInputs TOML: the level the reactions are refined at."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Total charge (default: the molecules')."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="Total spin multiplicity (default: "
                                               "the molecules' unpaired electrons added up)."),
    temperature: float = typer.Option(2000.0, "--temperature", help="MD temperature (K)."),
    time_ps: float = typer.Option(5.0, "--time", help="Simulated time (ps)."),
    compress: float = typer.Option(0.6, "--compress", help="Piston: narrow radius as a fraction of the wide one."),
    period: float = typer.Option(1.0, "--period", help="Piston period (ps)."),
    duty: float = typer.Option(0.75, "--duty", help="Fraction of each period at the wide radius."),
    forces: str = typer.Option("piston,collide", "--forces", help="piston and/or collide, comma-separated."),
    md_method: str = typer.Option("gfn2", "--md-method", help="What drives the MD: gfn2 or gxtb (axtb)."),
    radius: Optional[float] = typer.Option(None, "--radius", help="Wide wall radius (A). Default: from the atoms."),
    seed: int = typer.Option(0, "--seed", help="Random seed of the packing, the thermostat and the pulls."),
    relax_steps: int = typer.Option(60, "--relax-steps", help="Steepest-descent steps on the packing first."),
    instances: int = typer.Option(3, "--instances", help="Occurrences refined per reaction and species."),
    min_lifetime: float = typer.Option(20.0, "--min-lifetime", help="Bonds shorter-lived than this (fs) are "
                                       "vibrations."),
    merge_window: float = typer.Option(100.0, "--merge-window", help="Bond changes this close (fs) are one "
                                       "reaction."),
    output: Path = typer.Option(..., "--output", "-o", help="Output folder."),
) -> None:
    """A nanoreactor whose MD has the Reactor's forces (Collide as well as the piston), then the nanoreactor's
    analysis and refinement of its trajectory."""
    from mepd import search_control
    from mepd.cli_common import _load_structure_from_smiles_or_xyz
    from mepd.discovery.cli_nanoreactor import _parse_molecule
    from mepd.discovery.langevin import BOHR, KCAL, LangevinMD, pull_pairs
    from mepd.discovery.nanoreactor import _masses, pack_reactor
    from mepd.discovery.nanoreactor_generator import write_live_events
    from mepd.discovery.reactor_generator import OPTIONS as GEN, collision_pair
    from mepd.web.chem import box_multiplicity
    from mepd.web.sandbox import _Z, piston_radius

    active = [f.strip() for f in forces.split(",") if f.strip()]
    if not set(active) <= {"piston", "collide"} or not active:
        raise typer.BadParameter(f"--forces takes piston and/or collide, got {forces!r}")
    if md_method not in METHODS:
        raise typer.BadParameter(f"--md-method takes {' or '.join(METHODS)} here (axtb), got {md_method!r}")
    structures = []
    for item in molecules:
        what, n = _parse_molecule(item)
        s = _load_structure_from_smiles_or_xyz(what, None, None)
        structures += [s] * n
    symbols, coords, wide, _ = pack_reactor(structures, radius, seed=seed)
    wide = float(wide)
    q = int(charge) if charge is not None else sum(int(s.charge) for s in structures)
    electrons = sum(_Z.get(x, 0) for x in symbols) - q
    try:
        mult = box_multiplicity(electrons, [int(s.multiplicity) for s in structures], multiplicity)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    output.mkdir(parents=True, exist_ok=True)
    md_dir = output / "md"
    md_dir.mkdir(exist_ok=True)
    typer.echo(f"Reactor: {len(symbols)} atoms, charge {q}, multiplicity {mult}, wall {wide:.1f} -> "
               f"{wide * compress:.1f} A; forces {', '.join(active)}; MD {md_method} (axtb)")

    piston = ({"period_ps": float(period), "compress": float(compress), "duty": float(duty), "t0": 0.0}
              if "piston" in active and float(compress) < 1 else None)
    n_marks = max(1, int(round(time_ps * 20)))
    schedule = {"dump_fs": TRAJ_FS, "time_ps": float(time_ps), "seed": int(seed), "forces": active, "pulls": [],
                "engine": f"LangevinMD {md_method}",
                "segments": [[0.05, piston_radius(50.0 * i, wide, piston)] for i in range(n_marks)]}

    def save_schedule():
        tmp = md_dir / "schedule.json.tmp"
        tmp.write_text(json.dumps(schedule))
        tmp.replace(md_dir / "schedule.json")

    save_schedule()
    _write_frames(md_dir / "packed.xyz", symbols, [coords], "nanoreactor packed")

    k_au = float(GEN["k"]) / KCAL * BOHR * BOHR
    pull_fs, gap_fs, hops = float(GEN["pull_fs"]), float(GEN["gap_fs"]), int(GEN["intra_hops"])
    pair, masses = [None], _masses(symbols)
    md = LangevinMD(symbols, np.asarray(coords, dtype=float) / BOHR, q, mult, temperature=temperature, wall=wide,
                    seed=seed, method=METHODS[md_method],
                    bias=lambda x: pull_pairs(x, [pair[0]] if pair[0] is not None else [], k_au, masses))
    rng = md.rng

    typer.echo(f"Relaxing the packed reactor ({len(symbols)} atoms)...")
    relaxed = [md.pos * BOHR]
    try:
        for i in range(int(relax_steps)):
            md.relax(1)
            relaxed.append(md.pos * BOHR)
            if (i + 1) % 10 == 0 or i + 1 == relax_steps:
                _write_frames(md_dir / "relax.xyz", symbols, relaxed[1:])
        _write_frames(md_dir / "reactor.xyz", symbols, [md.pos * BOHR], "nanoreactor relaxed")
        md.start()
    except Exception as exc:
        typer.echo(f"The reactor MD failed while relaxing the packing: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    frames, t_fs, written, seg = [md.pos * BOHR], 0.0, 0, 0
    next_frame, pull_end = TRAJ_FS, 0.0
    next_pull = float(rng.exponential(gap_fs)) if "collide" in active and gap_fs > 0 else float("inf")
    total_fs, stopped, failed = float(time_ps) * 1000.0, False, None
    t_start, last_say = time.time(), 0.0

    # The frames since the last segment also go, one by one, to md/xtb.trj (as the nanoreactor's own MD writes
    # its running segment): the live view shows each frame, not one block every LIVE_FRAMES.
    running = [(md_dir / "xtb.trj").open("w")]

    def flush(final: bool = False):
        nonlocal written, seg
        if len(frames) - written >= LIVE_FRAMES or (final and len(frames) > written):
            _write_frames(md_dir / f"segment_{seg:03d}.xyz", symbols, frames[written:])
            written, seg = len(frames), seg + 1
            running[0].close()
            running[0] = (md_dir / "xtb.trj").open("w")
            write_live_events(output, total_charge=q, dt_fs=TRAJ_FS)
        elif len(frames) > written:
            x = frames[-1]
            running[0].write(f"{len(symbols)}\n\n" + "".join(f"{e} {a:.4f} {b:.4f} {c:.4f}\n"
                                                            for e, (a, b, c) in zip(symbols, x)))
            running[0].flush()

    while t_fs < total_fs - 1e-9:
        if pair[0] is not None and t_fs >= pull_end:
            pair[0] = None
            next_pull = t_fs + (float(rng.exponential(gap_fs)) if gap_fs > 0 else 0.0)
        elif pair[0] is None and t_fs >= next_pull:
            pair[0] = collision_pair(symbols, md.pos * BOHR, rng, hops)
            pull_end = t_fs + pull_fs
            if pair[0] is None:           # nothing to pull together: the piston alone from here
                next_pull = float("inf")
            else:
                p = pair[0]
                schedule["pulls"].append({"t0_fs": round(t_fs, 3), "t1_fs": round(pull_end, 3),
                                          "atoms": [int(a) for a in p[:2]],
                                          "molecules": [[int(a) for a in m] for m in p[2:4]] if len(p) > 2 else None,
                                          "push": None})
                save_schedule()
        md.wall = piston_radius(t_fs, wide, piston)
        try:
            if not md.step():             # an SCF that failed: stepped back and cooled down (as the Reactor)
                continue
        except Exception as exc:
            failed = f"{type(exc).__name__}: {exc}"
            break
        t_fs += md.dt
        if t_fs >= next_frame - 1e-9:
            frames.append(md.pos * BOHR)
            next_frame += TRAJ_FS
            flush()
            if search_control.stop_requested():
                stopped = True
                break
            if time.time() - last_say > 10 or t_fs >= total_fs - 1e-9:
                last_say = time.time()
                rate = t_fs / 1000.0 / max(last_say - t_start, 1e-9)
                left = (total_fs - t_fs) / 1000.0 / rate if rate > 0 else 0
                typer.echo(f"MD {t_fs / 1000:.2f} / {time_ps:.2f} ps (wall {md.wall:.1f} A, {md.T:.0f} K)"
                           + (f" · about {left / 60:.0f} min left" if left >= 90 else
                              f" · about {left:.0f} s left" if left >= 1 else ""))
    flush(final=True)
    running[0].close()
    (md_dir / "xtb.trj").unlink(missing_ok=True)
    (md_dir / "done").write_text("")
    if failed:
        typer.echo(f"The reactor MD failed at {t_fs / 1000:.2f} of {time_ps:g} ps: {failed}")
        if t_fs < 100.0:
            raise typer.Exit(code=1)
        typer.echo("Analyzing the trajectory it wrote.")
    elif stopped:
        typer.echo(f"Stop requested: the MD ended at {t_fs / 1000:.2f} of {time_ps:g} ps; analyzing the "
                   "trajectory so far.")
    _write_frames(md_dir / "trajectory.xyz", symbols, frames)
    _write_frames(md_dir / "start.xyz", symbols, frames[:1], f"qcdata_charge={q} qcdata_multiplicity={mult}")

    # The nanoreactor's own analysis and refinement of this trajectory, into the same folder.
    argv = [sys.executable, "-m", "mepd.cli", "discovery", "nanoreactor", str(md_dir / "start.xyz"),
            "--trajectory", str(md_dir / "trajectory.xyz"), "--dump", str(TRAJ_FS), "--charge", str(q),
            "--multiplicity", str(mult), "--instances", str(instances), "--min-lifetime", str(min_lifetime),
            "--merge-window", str(merge_window), "--output", str(output)]
    if inputs is not None:
        argv[6:6] = ["--inputs", str(inputs)]
    typer.echo("Analyzing the trajectory (mepd discovery nanoreactor --trajectory)...")
    sys.stdout.flush()
    raise typer.Exit(code=subprocess.call(argv))
