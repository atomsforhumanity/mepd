"""The nanoreactor as a network-expansion product generator.

One structure (for a QM/MM system: its QM region capped with its link
hydrogens, see network_expansion._region_proposals) is put in the
nanoreactor's piston: hot molecular dynamics inside a spherical wall that
periodically squeezes it (mepd.discovery.nanoreactor.run_reactor_md). The
bond graph is followed along the trajectory with the nanoreactor's
hysteresis and minimum lifetime, and the structure as it is right after
each reaction event becomes one product: whole, in the input's atom order,
so it plugs into the expansion like any other product guess (re-optimized
at the run's level, classified, connected by path searches; for a QM/MM
system, put back into its environment first).

Unlike `mepd discovery nanoreactor` (several molecules; each reaction cut
out with only the molecules it needs), this keeps every atom: what changes
is the state of this one structure.

Based on the ab initio nanoreactor (L.-P. Wang et al., Nat. Chem. 6, 1044
(2014), doi:10.1038/nchem.2099), as reimplemented in
mepd.discovery.nanoreactor; the MD is xtb's.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR

OPTIONS = {"temperature": 2500.0, "time_ps": 10.0, "compress": 0.7, "period_ps": 1.0, "md_method": "auto",
           "seed": 0, "n_seeds": 1, "parallel": 0, "workdir": ""}


def _md_method(name: str) -> tuple[str, Optional[str]]:
    from mepd.discovery.nanoreactor import MD_METHODS, pick_md_method

    if name == "auto":
        method, exe = pick_md_method(None)
    else:
        method, exe = name, None
    if method == "level" or method not in MD_METHODS:
        raise RuntimeError("The nanoreactor generator runs its MD with xtb (GFN2/GFN1) or g-xTB: install either "
                           "(conda install -c conda-forge xtb), or set GXTB_EXECUTABLE.")
    return method, exe


def run_dir(workdir: str) -> Optional[Path]:
    """A new `run_kk` folder under `workdir` (one per species expanded), laid
    out like a nanoreactor job's output (md/, live_events.json) so the web
    UI's Reactor view can show it; None = a temporary folder."""
    if not workdir:
        return None
    base = Path(workdir)
    base.mkdir(parents=True, exist_ok=True)
    k = len(list(base.glob("run_*")))
    while True:   # claim a folder atomically: several trajectories may start at once
        d = base / f"run_{k:02d}"
        try:
            d.mkdir()
            break
        except FileExistsError:
            k += 1
    (d / "md").mkdir(parents=True, exist_ok=True)
    return d


def write_live_events(run: Optional[Path], *, total_charge: int, dt_fs: float) -> None:
    """The reactor's events so far, for the Reactor view (never fails the run)."""
    if run is None:
        return
    import json

    from mepd.discovery.nanoreactor import DetectSettings, live_events

    try:
        data = live_events(run / "md", total_charge=total_charge, detect=DetectSettings(), dt_fs=dt_fs)
        tmp = run / "live_events.json.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(run / "live_events.json")
    except Exception:
        pass


def segment_hook(run: Optional[Path], total_charge: int, dt_fs: float, on_event=None):
    """on_event for the MD: after each segment, the events so far."""
    def hook(event, payload):
        if event == "md_segment":
            write_live_events(run, total_charge=total_charge, dt_fs=dt_fs)
        if on_event is not None:
            on_event(event, payload)
    return hook


def reactor_products(structure, *, max_products: int = 20, temperature: float = 2500.0, time_ps: float = 10.0,
                     compress: float = 0.7, period_ps: float = 1.0, md_method: str = "auto", seed: int = 0,
                     n_seeds: int = 1, parallel: int = 0, workdir: str = "", on_event=None) -> list:
    """Product structures (input atom order) of hot piston MD of `structure`:
    the structure right after each reaction event, one per distinct bond
    graph, in the order they appeared, at most `max_products`.

    `n_seeds` > 1 runs that many independent trajectories (random seeds
    `seed`, `seed`+1, ...; `parallel` of them at once, default all) and pools
    their states: one trajectory samples one sequence of events, so several
    short ones cover more of the first-step chemistry than one long one
    (which mostly goes on to fragment). Products are taken earliest-event
    first across trajectories, de-duplicated by bond graph."""
    n_seeds = max(1, int(n_seeds))
    if n_seeds > 1:
        from concurrent.futures import ThreadPoolExecutor

        width = max(1, int(parallel) or n_seeds)
        kw = dict(max_products=max_products, temperature=temperature, time_ps=time_ps, compress=compress,
                  period_ps=period_ps, md_method=md_method, workdir=workdir, on_event=on_event)
        with ThreadPoolExecutor(max_workers=width) as pool:
            runs = list(pool.map(lambda k: _reactor_states(structure, seed=int(seed) + k, **kw), range(n_seeds)))
        # earliest event first across trajectories, one product per bond graph
        pooled = sorted((frame_rank, k, bonds, prod) for k, states in enumerate(runs)
                        for frame_rank, bonds, prod in states)
        out, seen = [], set()
        for _, _, bonds, prod in pooled:
            if bonds in seen:
                continue
            seen.add(bonds)
            if len(out) < int(max_products):
                out.append(prod)
        print(f"nanoreactor: {len(out)} distinct product(s) from {n_seeds} trajectories "
              f"({sum(len(r) for r in runs)} reactor states).", flush=True)
        return out
    return [prod for _, _, prod in _reactor_states(
        structure, max_products=max_products, temperature=temperature, time_ps=time_ps, compress=compress,
        period_ps=period_ps, md_method=md_method, seed=seed, workdir=workdir, on_event=on_event)]


def _reactor_states(structure, *, max_products: int, temperature: float, time_ps: float, compress: float,
                    period_ps: float, md_method: str, seed: int, workdir: str, on_event=None) -> list:
    """One trajectory: [(event rank, bond set, product structure)], earliest first."""
    from mepd.discovery.nanoreactor import (DetectSettings, ReactorSettings, auto_radius, bond_history,
                                            detect_events, read_xyz_frames, run_reactor_md)

    symbols = [str(s) for s in structure.symbols]
    xyz = np.asarray(structure.geometry, dtype=float) / ANGSTROM_TO_BOHR
    xyz = xyz - xyz.mean(axis=0)      # the wall is centred on the origin
    extent = float(np.max(np.linalg.norm(xyz, axis=1))) if len(xyz) else 0.0
    method, exe = _md_method(str(md_method))
    settings = ReactorSettings(temperature=float(temperature), time_ps=float(time_ps), compress=float(compress),
                               period_ps=float(period_ps), method=method, seed=int(seed),
                               radius=max(auto_radius(len(symbols)), extent + 1.5))
    detect = DetectSettings()
    run = run_dir(workdir)
    unstable = []

    def hook(event, payload):
        if event == "warning" and "MD is unstable" in str(payload.get("message", "")):
            unstable.append(payload)
        segment_hook(run, int(structure.charge), settings.dump_fs, on_event)(event, payload)

    with tempfile.TemporaryDirectory(prefix="mepd_reactor_gen_") as tmp:
        d = run / "md" if run is not None else Path(tmp)
        traj = run_reactor_md(symbols, xyz, charge=int(structure.charge), multiplicity=int(structure.multiplicity),
                              settings=settings, workdir=d, executable=exe, on_event=hook)
        _, frames, _ = read_xyz_frames(traj)
    write_live_events(run, total_charge=int(structure.charge), dt_fs=settings.dump_fs)
    if unstable:
        warn(f"xtb stopped {len(unstable)} of {len(piston_segments(settings))} MD segments early (\"MD is unstable\"): "
             f"{settings.temperature:g} K is too hot; the trajectory has gaps. Lower the temperature.")
    hist = bond_history(symbols, frames, detect, settings.dump_fs)
    events = detect_events(len(symbols), hist, detect, settings.dump_fs)
    out, torn, keys = select_states(structure, frames, hist, events, max_products, with_keys=True)
    report(len(events), len(set(keys)), torn, len(out), settings.temperature)
    return out


def warn(message: str) -> None:
    """A warning in the job log (the web UI lists these on the result page)."""
    print(f"WARNING: nanoreactor: {message}", flush=True)


def report(n_events: int, states: int, torn: int, kept: int, temperature: float) -> None:
    if n_events == 0:
        warn(f"no reaction in the MD at {temperature:g} K: run it longer or hotter.")
    elif kept == 0 and torn:
        warn(f"all {torn} reactor states after an event had atoms torn off (loose atoms are no product guess), "
             f"so there are no products: {temperature:g} K is too hot, try 2500-3500 K.")
    elif torn:
        print(f"nanoreactor: {kept} product(s) from {states} reactor state(s); {torn} with loose atoms skipped.",
              flush=True)


def piston_segments(settings) -> list:
    from mepd.discovery.nanoreactor import piston_schedule

    return piston_schedule(settings)


def select_states(structure, frames, hist, events, max_products: int, with_keys: bool = False):
    """The structure after each event, one per distinct bond graph and
    stereochemistry (`stereo_key`), earliest first, skipping states with an
    atom that lost all its bonds. Returns (products, number skipped as torn,
    every other event's (bonds, stereo key), repeats included); with
    `with_keys`, each product is (event frame, its key, structure)."""
    start = (frozenset(hist.initial), stereo_key(structure))
    seen = {start}
    bonded0 = {a for p in hist.initial for a in p}
    out, torn, keys = [], 0, []
    for ev in sorted(events, key=lambda e: (e.product_frame, e.start)):
        bonds = frozenset(hist.bonds_at(ev.product_frame))
        # The hot MD eventually tears a structure into atoms: a state with an
        # atom that lost all its bonds (it had some) is no product guess.
        if bonded0 - {a for p in bonds for a in p}:
            torn += 1
            continue
        prod = structure.model_copy(update={"geometry": frames[ev.product_frame] * ANGSTROM_TO_BOHR})
        key = (bonds, stereo_key(prod))
        if key != start:
            keys.append(key)
        if key in seen:
            continue
        seen.add(key)
        if len(out) < int(max_products):
            out.append((int(ev.product_frame), key, prod) if with_keys else prod)
    return out, torn, keys


def stereo_key(structure) -> Optional[str]:
    """The stereo SMILES of `structure` (mepd.compare's, stereochemistry from
    3D), so two diastereomers with one bond graph (endo and exo) are two
    products; None when it can't be built."""
    from mepd.nodes.node import StructureNode
    from mepd.nodes.nodehelpers import _stereo_smiles

    try:
        return _stereo_smiles(StructureNode(structure=structure))[0]
    except Exception:
        return None
