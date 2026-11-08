"""The interactive Reactor, unattended, as a network-expansion product generator.

What a person does in the Reactor (mepd.web.sandbox) with the piston on and
"keep colliding" ticked: Langevin dynamics on GFN2-xTB (axtb; or g-xTB, `md_method`) inside the
nanoreactor's soft wall, which the piston swings between a wide and a
squeezed radius, while Collide every so often pushes two molecules chosen at
random together for a moment (whole: a capped force along the line between a
random heavy atom of each, spread over each molecule by mass), or pulls two
atoms of one molecule together, and lets go.
Nothing picks the chemistry: the piston and the random pulls supply the energy, and the
kinetics decide afterwards which of the reactions matter.

The trajectory is read like the nanoreactor's (its bond hysteresis, minimum
lifetime and event detection). The structure right after each reaction event
is quenched and is one product: whole, in the input's atom order, one per
stereo SMILES of the quenched structure (canonical: the same reaction in
another of a cluster's identical molecules is one product, and a hot
snapshot that relaxes back is none). Batches of `n_seeds` independent
trajectories run until the seed's own reactions have converged: until the
events out of the seed's state whose product was seen only once
(Good-Turing's estimate of the chance that the seed's next reaction gives a
product not yet seen) are at most `unseen` of them, and there are at least
1/`unseen` such events, or `max_batches` ran. Every state reached is
proposed, those the seed reached most often first.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Optional

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.cpus import available_cpus
from mepd.discovery.langevin import BOHR, KCAL
TRAJ_FS = 2.0                    # frame spacing read by the event detection (the Reactor's and nanoreactor's)

OPTIONS = {"temperature": 2200.0, "time_ps": 5.0, "compress": 0.7, "period_ps": 1.0, "duty": 0.75,
           "pull_fs": 300.0, "gap_fs": 200.0, "k": 100.0, "intra_hops": 2, "n_seeds": 0, "seed": 0,
           "unseen": 0.05, "max_batches": 4, "parallel": 0, "workdir": "", "cutoff_kcal": None,
           "md_method": "gfn2", "forces": "piston,collide"}
FORCES = ("piston", "collide")   # the MD's forces besides the method's and the wall's (nanoreactor forces=...)
LIVE_FRAMES = 125                # frames per segment file a live viewer reads (250 fs)


def collision_pair(symbols, xyz_A, rng, intra_hops: int = 2) -> Optional[tuple]:
    """The Reactor's Collide pick (mepd/web/sandbox.py runs this one too), with `rng`: two
    heavy atoms of two different molecules, any two with no species preferred,
    with the two molecules' atoms, (i, j, A, B): the whole molecules are pushed
    together along i-j (langevin.pull_pairs); or, 30% of the time and always
    with one molecule, two atoms of one molecule at least `intra_hops` bonds
    apart (pulled themselves: a lone molecule has no partner, and without these
    pulls formic acid made nothing in 240 ps at 2200 K). Unlike the Reactor's
    (3 bonds, a heavy atom first, heavy pairs preferred), any such pair, two
    hydrogens too: a person drags whatever looks promising, an unattended run must not
    leave out the four-centre eliminations and H2 losses of a small molecule
    (formic acid: the Reactor's rule leaves one pair, an H shift)."""
    from mepd.discovery.collide_generator import _bonds, _hops

    n = len(symbols)
    bonds = _bonds(symbols, np.asarray(xyz_A))
    hops = _hops(n, bonds)
    mols, seen = [], set()
    for a in range(n):
        if a not in seen:
            m = [b for b in range(n) if np.isfinite(hops[a, b])]
            seen.update(m)
            mols.append(m)
    heavy = lambda i: symbols[i] != "H"
    pool = [(a, b) for m in mols for a in m for b in m if b < a and hops[a, b] >= max(2, int(intra_hops))]
    pick_intra = (lambda: tuple(int(x) for x in pool[rng.integers(len(pool))])) if pool else (lambda: None)
    if len(mols) < 2 or (pool and rng.random() < 0.3):
        return pick_intra()
    pick = lambda xs: xs[rng.integers(len(xs))]
    a = pick(mols)
    b = pick([m for m in mols if m is not a])
    hv = lambda m: [i for i in m if heavy(i)] or m
    return int(pick(hv(a))), int(pick(hv(b))), tuple(a), tuple(b)


def forces_of(forces) -> list:
    """The forces named in `forces` ("piston,collide", a list), in FORCES' order; raises on an unknown one."""
    given = [str(f).strip().lower() for f in (forces.split(",") if isinstance(forces, str) else forces or [])]
    given = [f for f in given if f]
    unknown = sorted(set(given) - set(FORCES))
    if unknown or not given:
        raise ValueError(f"forces are one or more of {', '.join(FORCES)} (got {forces!r}).")
    return [f for f in FORCES if f in given]


def _trajectory(symbols, xyz_A, charge, multiplicity, *, temperature, time_ps, compress, period_ps, duty, pull_fs,
                gap_fs, k, intra_hops, seed, workdir: str = "", md_method: str = "gfn2",
                md_server: Optional[str] = None, forces: str = "piston,collide") -> np.ndarray:
    """One Reactor run: frames (Angstrom) every TRAJ_FS fs. With `workdir`, it is also written as it runs into
    a new <workdir>/run_kk laid out like a nanoreactor job (md/segment_kkk.xyz every LIVE_FRAMES frames,
    md/schedule.json with the forces and each pull, live_events.json): the web UI's Reactor view shows it live.
    `forces`: "piston" swings the wall, "collide" pulls a random pair together every so often (both: the Reactor)."""
    from mepd.discovery.langevin import LangevinMD, pull_pairs
    from mepd.discovery.nanoreactor import _masses
    from mepd.discovery.nanoreactor import auto_radius
    from mepd.web.sandbox import piston_radius

    pos = np.asarray(xyz_A, dtype=float) / BOHR
    pos = pos - pos.mean(axis=0)
    # the wall: as the Reactor started from one geometry (2 A past the farthest atom), at least the nanoreactor's
    wide = max(auto_radius(len(symbols)), float(np.linalg.norm(pos, axis=1).max()) * BOHR + 2.0)
    active = forces_of(forces)
    piston = {"period_ps": float(period_ps), "compress": float(compress), "duty": float(duty), "t0": 0.0} \
        if float(compress) < 1 and "piston" in active else None
    k_au = float(k) / KCAL * BOHR * BOHR
    pair, masses = [None], _masses(symbols)
    md = LangevinMD(symbols, pos, charge, multiplicity, temperature=temperature, wall=wide, seed=seed,
                    method=md_method, server=md_server,
                    bias=lambda x: pull_pairs(x, [pair[0]] if pair[0] is not None else [], k_au, masses))
    rng = md.rng
    md.start()
    frames, t_fs = [md.pos * BOHR], 0.0
    live = _LiveRun(workdir, symbols, int(charge), float(time_ps), wide, piston, seed, active)
    next_frame, pull_end = TRAJ_FS, 0.0
    if "collide" in active:
        next_pull = float(rng.exponential(gap_fs)) if gap_fs > 0 else 0.0
    else:                                   # the piston alone: no pulls
        next_pull = float("inf")
    while t_fs < float(time_ps) * 1000.0 - 1e-9:
        if pair[0] is not None and t_fs >= pull_end:
            pair[0] = None
            next_pull = t_fs + (float(rng.exponential(gap_fs)) if gap_fs > 0 else 0.0)
        elif pair[0] is None and t_fs >= next_pull:
            pair[0] = collision_pair(symbols, md.pos * BOHR, rng, intra_hops)
            pull_end = t_fs + float(pull_fs)
            if pair[0] is None:             # nothing to pull: the piston alone
                next_pull = float("inf")
            else:
                live.pull(t_fs, pull_end, pair[0])
        md.wall = piston_radius(t_fs, wide, piston)
        try:
            if not md.step():               # an SCF that failed: stepped back, cooled down (as the Reactor)
                continue
        except Exception:
            break
        t_fs += md.dt
        if t_fs >= next_frame - 1e-9:
            frames.append(md.pos * BOHR)
            next_frame += TRAJ_FS
            live.add(frames)
    live.add(frames, final=True)
    return np.asarray(frames)


class _LiveRun:
    """A trajectory written for the web's Reactor view as it runs (nothing without a workdir)."""

    def __init__(self, workdir: str, symbols, charge: int, time_ps: float, wide: float, piston, seed: int = 0,
                 forces=("piston", "collide"), **extra):
        from mepd.discovery.nanoreactor_generator import run_dir
        from mepd.web.sandbox import piston_radius

        self.run = run_dir(workdir) if workdir else None
        self.symbols, self.charge, self.written, self.k = list(symbols), charge, 0, 0
        if self.run is not None:   # the wall radius every 50 fs, as the view reads a piston schedule
            n = max(1, int(round(time_ps * 20)))
            self.schedule = {"dump_fs": TRAJ_FS, "time_ps": time_ps, "seed": int(seed),
                             "segments": [[0.05, piston_radius(50.0 * i, wide, piston)] for i in range(n)],
                             "forces": list(forces), "pulls": [], **extra}
            self._schedule()

    def pull(self, t0_fs: float, t1_fs: float, pair, push=None) -> None:
        """A pull from t0_fs to t1_fs (fs): two atoms (i, j), or (i, j, A, B) for molecules A and B pushed together."""
        if self.run is None:
            return
        molecules = [[int(a) for a in m] for m in pair[2:4]] if len(pair) > 2 else None
        self.schedule["pulls"].append({"t0_fs": round(float(t0_fs), 3), "t1_fs": round(float(t1_fs), 3),
                                       "atoms": [int(a) for a in pair[:2]], "molecules": molecules,
                                       "push": [int(a) for a in push] if push is not None else None})
        self._schedule()

    def _schedule(self) -> None:
        fp = self.run / "md" / "schedule.json"
        fp.with_suffix(".tmp").write_text(json.dumps(self.schedule))
        fp.with_suffix(".tmp").replace(fp)

    def add(self, frames, final: bool = False) -> None:
        if self.run is None:
            return
        from mepd.discovery.nanoreactor_generator import write_live_events

        if final:   # (the view stops following a run once this is there)
            self._write(frames)
            (self.run / "md" / "done").write_text("")
            write_live_events(self.run, total_charge=self.charge, dt_fs=TRAJ_FS)
            return
        if len(frames) - self.written >= LIVE_FRAMES:
            self._write(frames)
            write_live_events(self.run, total_charge=self.charge, dt_fs=TRAJ_FS)

    def _write(self, frames) -> None:
        if len(frames) <= self.written:
            return
        block = frames[self.written:]
        text = "".join(f"{len(self.symbols)}\n\n" + "".join(f"{e} {x:.4f} {y:.4f} {z:.4f}\n"
                                                           for e, (x, y, z) in zip(self.symbols, f)) for f in block)
        fp = self.run / "md" / f"segment_{self.k:03d}.xyz"
        fp.with_suffix(".tmp").write_text(text)
        fp.with_suffix(".tmp").replace(fp)
        self.written, self.k = len(frames), self.k + 1


def _quench(symbols, xyz_A, charge, multiplicity, md_method: str = "gfn2",
            md_server: Optional[str] = None) -> tuple[np.ndarray, Optional[float]]:
    """`xyz_A` relaxed as collide's quench (200 steepest-descent steps on the MD's method, no new chemistry),
    Angstrom, and its energy (kcal/mol); the snapshot itself and None if the SCF fails."""
    from mepd.discovery.langevin import LangevinMD

    try:
        md = LangevinMD(symbols, np.asarray(xyz_A, dtype=float) / BOHR, charge, multiplicity, temperature=300.0,
                        method=md_method, server=md_server)
        md.relax(200)
        return md.pos * BOHR, md.forces(md.pos, biased=False)[0] * KCAL
    except Exception:
        return np.asarray(xyz_A, dtype=float), None


def _states(symbols, xyz_A, charge, multiplicity, max_products, options) -> dict:
    """One trajectory's reactor states, [(event frame, key, xyz Angstrom)] earliest first, and every event's
    [key, whether it left the seed's state, its quenched energy above the quenched seed (kcal/mol, at the MD's
    method; None if unknown)], repeats included. The structure after each event is quenched and
    keyed by its stereo SMILES: a hot snapshot's bonds (a third of the events in (HCOOH)4 at 1800 K fall back to
    the seed on quenching) and the atom numbering (the same reaction in another of four identical molecules)
    are no new product. An event whose quenched state is the one before it changed nothing and is not counted."""
    from qcdata import Structure

    from mepd.discovery.nanoreactor import DetectSettings, bond_history, detect_events
    from mepd.discovery.nanoreactor_generator import stereo_key

    frames = _trajectory(symbols, xyz_A, charge, multiplicity, **options)
    detect = DetectSettings()
    hist = bond_history(symbols, frames, detect, TRAJ_FS)
    events = detect_events(len(symbols), hist, detect, TRAJ_FS)
    structure = Structure(symbols=list(symbols), geometry=np.asarray(xyz_A) * ANGSTROM_TO_BOHR,
                          charge=int(charge), multiplicity=int(multiplicity))
    seed = prev = stereo_key(structure)
    method, server = options.get("md_method", "gfn2"), options.get("md_server")
    _, e_seed = _quench(symbols, xyz_A, int(charge), int(multiplicity), method, server)
    bonded0 = {a for p in hist.initial for a in p}
    states, seen, keys, torn = [], {seed}, [], 0
    for ev in sorted(events, key=lambda e: (e.product_frame, e.start)):
        # the hot MD eventually tears a structure into atoms: a state with an atom that lost all its bonds
        # (it had some) is no product guess
        if bonded0 - {a for p in hist.bonds_at(ev.product_frame) for a in p}:
            torn += 1
            continue
        x, e = _quench(symbols, frames[ev.product_frame], int(charge), int(multiplicity), method, server)
        key = stereo_key(structure.model_copy(update={"geometry": x * ANGSTROM_TO_BOHR}))
        if key is None or key == prev:
            continue
        keys.append([key, prev == seed, None if e is None or e_seed is None else round(e - e_seed, 2)])
        prev = key
        if key not in seen:
            seen.add(key)
            states.append((int(ev.product_frame), key, x.tolist()))
    return {"states": states, "keys": keys, "events": len(events), "torn": torn}


def reactor_products(structure, *, max_products: int = 20, temperature: float = 2200.0, time_ps: float = 5.0,
                     compress: float = 0.7, period_ps: float = 1.0, duty: float = 0.75, pull_fs: float = 300.0,
                     gap_fs: float = 200.0, k: float = 100.0, intra_hops: int = 2, n_seeds: int = 0, seed: int = 0,
                     unseen: float = 0.05, max_batches: int = 4, parallel: int = 0, workdir: str = "",
                     cutoff_kcal: Optional[float] = None, md_method: str = "gfn2",
                     forces: str = "piston,collide") -> list:
    """Product structures (input atom order) of batches of `n_seeds` Reactor
    runs of `structure` (default: one per process), `parallel` at a time
    (default: all CPUs), until converged (the module's docstring). With
    `cutoff_kcal` (the explorer's access cutoff left above `structure`), a
    product more than that above `structure` (quenched, GFN2-xTB) is still
    proposed but doesn't count toward convergence: no step to it can be
    within the cutoff, as a barrier is at least the reaction energy.
    (All batches, then the products pooled: `reactor_batch` hands each batch's
    over as it ends.)"""
    width = max(1, int(parallel) or available_cpus())
    n = max(1, int(n_seeds) or width)
    options = _md_options(temperature, time_ps, compress, period_ps, duty, pull_fs, gap_fs, k, intra_hops, workdir,
                          md_method, _md_server(md_method, min(width, n)), forces)
    runs, errors = [], []
    for batch in range(max(1, int(max_batches))):
        got, bad = _run_batch(structure, batch, n, width, max_products, seed, options)
        runs += got
        errors += bad
        first = _first(runs, cutoff_kcal)
        converged = _report_batch(batch, first, {k for r in runs for k, *_ in r["keys"]}, cutoff_kcal, unseen,
                                   md_method)
        if converged:
            break
    # the products the seed reached most often first (fewest steps, most often), then the earliest
    count = Counter(k for r in runs for k, *_ in r["keys"])
    pooled = sorted((-first[k], -count[k], f, s, k, x) for s, r in enumerate(runs) for f, k, x in r["states"])
    out, seen = [], set()
    for *_, key, x in pooled:
        if key in seen:
            continue
        seen.add(key)
        if len(out) < int(max_products):
            out.append(structure.model_copy(update={"geometry": np.asarray(x) * ANGSTROM_TO_BOHR}))
    _report_end(len(out), len(runs), sum(r["events"] for r in runs), sum(r["torn"] for r in runs), time_ps,
                temperature, errors)
    return out


def reactor_batch(structure, state: Optional[dict] = None, *, max_products: int = 20, temperature: float = 2200.0,
                  time_ps: float = 5.0, compress: float = 0.7, period_ps: float = 1.0, duty: float = 0.75,
                  pull_fs: float = 300.0, gap_fs: float = 200.0, k: float = 100.0, intra_hops: int = 2,
                  n_seeds: int = 0, seed: int = 0, unseen: float = 0.05, max_batches: int = 4, parallel: int = 0,
                  workdir: str = "", cutoff_kcal: Optional[float] = None, md_method: str = "gfn2",
                  forces: str = "piston,collide") -> tuple:
    """The next batch of `reactor_products` for a caller that takes each batch's products the moment it
    ends (the explorer's async queue: they are optimized and connected while the next batch runs).
    `state` is what the previous batches left (None: the first); returns (this batch's products not
    proposed before -- the seed's most frequent first, by the counts so far --, the new state, done).
    The batches, their seeds, the convergence rule and the cap of `max_products` in all are
    reactor_products'; only the pooled ordering across batches is not (each batch's products go out
    as it ends, so a later batch's more frequent product can't come before an earlier one's)."""
    st = dict(state or {"batch": 0, "first": {}, "count": {}, "states": [], "proposed": [], "runs": 0,
                        "events": 0, "torn": 0, "errors": []})
    width = max(1, int(parallel) or available_cpus())
    n = max(1, int(n_seeds) or width)
    options = _md_options(temperature, time_ps, compress, period_ps, duty, pull_fs, gap_fs, k, intra_hops, workdir,
                          md_method, _md_server(md_method, min(width, n)), forces)
    batch = int(st["batch"])
    runs, errors = _run_batch(structure, batch, n, width, max_products, seed, options)
    first = Counter(st["first"]) + _first(runs, cutoff_kcal)
    count = Counter(st["count"]) + Counter(k for r in runs for k, *_ in r["keys"])
    states = set(st["states"]) | {k for r in runs for k, *_ in r["keys"]}
    converged = _report_batch(batch, first, states, cutoff_kcal, unseen, md_method)
    done = converged or batch + 1 >= max(1, int(max_batches))
    proposed = list(st["proposed"])
    taken = set(proposed)
    pooled = sorted((-first[k], -count[k], f, s, k, x) for s, r in enumerate(runs) for f, k, x in r["states"])
    out = []
    for *_, key, x in pooled:
        if key in taken or len(proposed) >= int(max_products):
            continue
        taken.add(key)
        proposed.append(key)
        out.append(structure.model_copy(update={"geometry": np.asarray(x) * ANGSTROM_TO_BOHR}))
    st.update(batch=batch + 1, first=dict(first), count=dict(count), states=sorted(states), proposed=proposed,
              runs=st["runs"] + len(runs), events=st["events"] + sum(r["events"] for r in runs),
              torn=st["torn"] + sum(r["torn"] for r in runs), errors=list(st["errors"]) + errors)
    if done:
        _report_end(len(proposed), st["runs"], st["events"], st["torn"], time_ps, temperature, st["errors"])
    return out, st, done


def _md_options(temperature, time_ps, compress, period_ps, duty, pull_fs, gap_fs, k, intra_hops, workdir,
                md_method, md_server, forces="piston,collide") -> dict:
    return dict(temperature=float(temperature), time_ps=float(time_ps), compress=float(compress),
                period_ps=float(period_ps), duty=float(duty), pull_fs=float(pull_fs), gap_fs=float(gap_fs),
                k=float(k), intra_hops=int(intra_hops), workdir=str(workdir or ""), md_method=str(md_method),
                md_server=md_server, forces=",".join(forces_of(forces)))


def _run_batch(structure, batch: int, n: int, width: int, max_products, seed, options) -> tuple:
    """Batch `batch`'s `n` trajectories (seeds seed + batch*n + s): (runs, errors)."""
    symbols = [str(s) for s in structure.symbols]
    xyz = np.asarray(structure.geometry, dtype=float) / ANGSTROM_TO_BOHR
    jobs = [(symbols, xyz.tolist(), int(structure.charge), int(structure.multiplicity), int(max_products),
             {**options, "seed": int(seed) + batch * n + s}) for s in range(n)]
    done = _run_in_processes(jobs, min(width, n))
    return [r for r in done if isinstance(r, dict)], [r for r in done if isinstance(r, str)]


def _first(runs, cutoff_kcal) -> Counter:
    """The products of the seed's own reactions (within the cutoff), counted."""
    return Counter(k for r in runs for k, from_seed, de in r["keys"]
                   if from_seed and (cutoff_kcal is None or de is None or de <= float(cutoff_kcal)))


def _report_batch(batch: int, first: Counter, states: set, cutoff_kcal, unseen, md_method: str = "gfn2") -> bool:
    """Print the batch's line (with the MD's method); whether the seed's reactions have converged."""
    events = sum(first.values())
    f1 = sum(c == 1 for c in first.values()) / events if events else 1.0
    below = "" if cutoff_kcal is None else f"; counted: products at most {float(cutoff_kcal):.0f} kcal/mol up"
    on = "fairchem" if md_method == "uma" else "axtb"
    print(f"reactor: batch {batch + 1} (MD: {md_method} on {on}): {len(first)} distinct products in {events} events "
          f"from the seed's state ({len(states)} states in all{below}); {f1:.0%} of them gave a product "
          f"seen once (converged at {unseen:.0%}).", flush=True)
    # Good-Turing's estimate resolves 1/events: below 1/unseen events, none seen once proves nothing
    return f1 <= float(unseen) and events * float(unseen) >= 1


def _report_end(n_out, n_runs, events, torn, time_ps, temperature, errors) -> None:
    print(f"reactor: {n_out} distinct product(s) from {n_runs} trajectories of {time_ps:g} ps at "
          f"{temperature:g} K ({events} events, {torn} states with a loose atom skipped).", flush=True)
    if errors:
        print(f"reactor: {len(errors)} trajectory(ies) failed, the first: {errors[0]}", flush=True)


def _md_server(md_method: str, width: int) -> Optional[str]:
    """UMA with several trajectories at once: a model server (one per process, kept) that batches their steps."""
    if md_method != "uma" or width <= 1:
        return None
    from mepd.discovery.langevin import uma_server

    return uma_server()


def _run_in_processes(jobs: list, width: int) -> list:
    """`_states(*job)` for each job, in fresh Python processes `width` at a
    time (started, not forked: the explorer calls generators from threads)."""
    import json
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    if width <= 1:
        return [_states(*j) for j in jobs]

    def run(job):
        r = subprocess.run([sys.executable, "-c", "from mepd.discovery.reactor_generator import _worker; _worker()"],
                           input=json.dumps(job), capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1"})
        if r.returncode != 0:
            return f"reactor process failed: {r.stderr.strip()[-300:]}"
        return json.loads(r.stdout.strip().splitlines()[-1])

    with ThreadPoolExecutor(max_workers=width) as pool:
        return list(pool.map(run, jobs))


def _worker() -> None:
    import json
    import sys

    symbols, xyz, q, m, max_products, options = json.loads(sys.stdin.read())
    print(json.dumps(_states(symbols, np.asarray(xyz), q, m, max_products, options)))
