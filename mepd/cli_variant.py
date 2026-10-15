"""`mepd variant`: recompute one reaction of a network variant
(mepd/network_variants.py) -- the parent reaction's reactant end, TS and
product end with an edit (e.g. methyls for hydrogens) built in.

The spec (JSON, written by the web UI or by hand) holds the substituted
frames in one atom order, in Angstrom:

    {"symbols": [...], "charge": 0, "multiplicity": 1,
     "r": [[x, y, z], ...], "ts": [...] | null, "p": [...],
     "bonds_r": [[i, j], ...], "bonds_p": [...],      # the substituted ends' bonds
     "free": {"r": [...], "ts": [...], "p": [...]},  # new atoms (relaxed in the fast estimate)
     "parent": {"symbols": [...], "r": ..., "ts": ..., "p": ...,
                "energies": [E_r, E_ts, E_p] | null}}  # Hartree, at the same level

* fast: only the new atoms are relaxed, everything else stays at the
  parent's geometries; single points give the barrier and reaction-energy
  shifts against the parent's frames (as `mepd substituents --mode fast`).
  The TS is not a saddle point of the new molecule: a first estimate.
* reoptimize: both ends are minimized (their bonds and configurations must
  survive), the TS is optimized from the substituted TS (the fast one when
  it exists) and its IRC must connect the two substituted ends.

Writes <output>/fast/ or <output>/reopt/ (frames, energies, summary.json).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from qcref.constants import HARTREE_TO_KCAL_PER_MOL as HARTREE_TO_KCAL

ANGSTROM_TO_BOHR = 1 / 0.529177210903


def variant(
    spec: Path = typer.Argument(..., exists=True, help="The variant reaction's spec (JSON; docs/network-variants.md)."),
    mode: str = typer.Option("fast", "--mode", help="fast: relax only the new atoms, single points. "
                             "reoptimize: minimize both ends, optimize the TS, IRC."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="RunInputs TOML: the level of theory."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Override the spec's charge."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="Override the spec's multiplicity."),
    output: Path = typer.Option(..., "--output", "-o", help="Output folder (fast/ or reopt/ inside it)."),
) -> None:
    """Recompute one reaction of a network variant: a fast estimate or a full re-optimization."""
    from mepd.cli_common import _open_run_inputs

    if mode not in ("fast", "reoptimize"):
        raise typer.BadParameter("--mode must be fast or reoptimize")
    d = json.loads(Path(spec).read_text())
    if charge is not None:
        d["charge"] = charge
    if multiplicity is not None:
        d["multiplicity"] = multiplicity
    run_inputs = _open_run_inputs(inputs)
    output = Path(output)
    if mode == "fast":
        summary = _fast(d, run_inputs, output / "fast")
    else:
        summary = _reoptimize(d, run_inputs, output)
    typer.echo(f"{summary['status']}: {summary.get('headline', '')}")
    if summary["status"] in ("failed",):
        raise typer.Exit(code=1)


def _node(symbols, coords_angstrom, charge, multiplicity):
    from qcdata import Structure

    from mepd.nodes.node import StructureNode

    return StructureNode(structure=Structure(symbols=list(symbols),
                                             geometry=np.asarray(coords_angstrom, dtype=float) * ANGSTROM_TO_BOHR,
                                             charge=int(charge), multiplicity=int(multiplicity)))


def _angstrom(node) -> np.ndarray:
    return np.asarray(node.coords, dtype=float) / ANGSTROM_TO_BOHR


def _adj(n: int, bonds) -> np.ndarray:
    a = np.zeros((n, n), dtype=bool)
    for i, j in bonds:
        a[i, j] = a[j, i] = True
    return a


def _write_frames(nodes, energies, fp: Path) -> None:
    from mepd.cli_solvent import _write

    _write(nodes, energies, fp)


def _json(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _save(summary: dict, fp: Path) -> dict:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(summary, indent=1, default=_json))
    return summary


def _same_bonds(symbols, coords, bonds) -> bool:
    from mepd.network_variants import adjacency

    return bool(np.array_equal(adjacency(symbols, coords), _adj(len(symbols), bonds)))


def _configurations(symbols, coords, bonds) -> tuple:
    """Tetrahedral configurations (not mirror-merged): a minimization must
    not invert a centre."""
    from mepd.network_variants import _edges, _ranks, _raw_mol, _signs

    adj = _adj(len(symbols), bonds)
    ranks = _ranks(_raw_mol(symbols, _edges(adj)))
    return tuple(sorted(_signs(symbols, coords, adj, ranks)))


def _fast(d: dict, run_inputs, out: Path) -> dict:
    from mepd import substituents as X
    from mepd.network_variants import adjacency

    engine = run_inputs.engine
    sym, q, mult = d["symbols"], int(d.get("charge", 0)), int(d.get("multiplicity", 1))
    frames = [k for k in ("r", "ts", "p") if d.get(k) is not None]
    built, reacted = {}, []
    for k in frames:
        xyz = np.asarray(d[k], dtype=float)
        free = list(d.get("free", {}).get(k) or [])
        before = adjacency(sym, xyz)
        node = _node(sym, xyz, q, mult)
        built[k] = X.relax_group(engine, node, free) if free else node
        after = adjacency(sym, _angstrom(built[k]))
        # The new atoms must keep their own bonds (and make no new ones),
        # else they reacted with the rest while relaxing.
        if free and not np.array_equal(before[free], after[free]):
            reacted.append({"r": "reactant", "ts": "TS", "p": "product"}[k])
    nodes = [built[k] for k in frames]
    energies = [float(e) for e in engine.compute_energies(nodes)]
    e = dict(zip(frames, energies))
    par = d.get("parent") or {}
    pe = par.get("energies")
    if not pe or any(v is None for v in pe):
        pnodes = [_node(par["symbols"], par[k], int(par.get("charge", 0)), mult) for k in frames]
        pe_list = [float(v) for v in engine.compute_energies(pnodes)]
        pe = dict(zip(frames, pe_list))
    else:
        pe = dict(zip(("r", "ts", "p"), pe))
    _write_frames(nodes, energies, out / "frames.xyz")
    summary = {"mode": "fast", "frames": frames, "energies": e, "parent_energies": pe,
               "status": "reacted" if reacted else "ok"}
    if reacted:
        summary["why"] = f"the new atoms formed or broke bonds while relaxing ({', '.join(reacted)})"
    if "ts" in e and "ts" in pe:
        summary["barrier_shift"] = ((e["ts"] - e["r"]) - (pe["ts"] - pe["r"])) * HARTREE_TO_KCAL
    summary["reaction_shift"] = ((e["p"] - e["r"]) - (pe["p"] - pe["r"])) * HARTREE_TO_KCAL
    if "barrier_shift" in summary:
        summary["headline"] = (f"fast estimate: barrier {summary['barrier_shift']:+.1f} kcal/mol, "
                               f"reaction {summary['reaction_shift']:+.1f} vs the parent")
    else:
        summary["headline"] = f"fast estimate: reaction {summary['reaction_shift']:+.1f} kcal/mol vs the parent"
    return _save(summary, out / "summary.json")


def _reoptimize(d: dict, run_inputs, output: Path) -> dict:
    from mepd.cli_common import _geometry_optimizer_keywords, _optimize_ts_and_irc

    out = output / "reopt"
    out.mkdir(parents=True, exist_ok=True)
    engine = run_inputs.engine
    sym, q, mult = d["symbols"], int(d.get("charge", 0)), int(d.get("multiplicity", 1))
    bonds = {"r": d["bonds_r"], "p": d["bonds_p"]}
    # Start from the fast estimate's frames when there are any (the new
    # atoms already relaxed).
    start = {k: np.asarray(d[k], dtype=float) for k in ("r", "ts", "p") if d.get(k) is not None}
    fast_fp = output / "fast" / "frames.xyz"
    fast_sum = output / "fast" / "summary.json"
    if fast_fp.exists() and fast_sum.exists():
        try:
            fs = json.loads(fast_sum.read_text())
            if fs.get("status") == "ok":
                from qcdata import Structure

                for k, st in zip(fs["frames"], Structure.open_multi(str(fast_fp))):
                    start[k] = np.asarray(st.geometry) / ANGSTROM_TO_BOHR
        except Exception:
            pass
    summary = {"mode": "reoptimize", "status": "ok", "notes": []}

    # 1. The ends: minimized, keeping their bonds and configurations.
    kw = _geometry_optimizer_keywords(run_inputs)
    ends, e_end = {}, {}
    for k in ("r", "p"):
        want = _configurations(sym, start[k], bonds[k])
        try:
            traj = engine.compute_geometry_optimization(_node(sym, start[k], q, mult), keywords=kw)
            node = traj[-1]
        except Exception as exc:
            summary["notes"].append(f"{'reactant' if k == 'r' else 'product'} end did not minimize: "
                                    f"{type(exc).__name__}: {exc}")
            node = _node(sym, start[k], q, mult)
        xyz = _angstrom(node)
        if not _same_bonds(sym, xyz, bonds[k]) or _configurations(sym, xyz, bonds[k]) != want:
            summary["status"] = "ends_changed"
            summary["notes"].append(f"the {'reactant' if k == 'r' else 'product'} end changed its bonds or "
                                    "configuration while minimizing")
        ends[k] = node
        e_end[k] = float(engine.compute_energies([node])[0])
    _write_frames([ends["r"]], [e_end["r"]], out / "start.xyz")
    _write_frames([ends["p"]], [e_end["p"]], out / "end.xyz")
    summary["end_energies"] = e_end

    # 2. The TS from the substituted guess, and its IRC.
    if "ts" not in start:
        summary["status"] = "no_ts"
        summary["headline"] = "ends minimized; the parent has no TS: a path search is needed"
        return _save(summary, out / "summary.json")
    res = _optimize_ts_and_irc(_node(sym, start["ts"], q, mult), run_inputs, out, run_irc=True, label="ts")
    if res is None:
        summary["status"] = "ts_failed"
        summary["headline"] = "the TS optimization failed: a path search is needed"
        return _save(summary, out / "summary.json")
    e_ts = float(engine.compute_energies([res.ts_node])[0])
    summary["ts_energy_hartree"] = e_ts
    irc = res.irc_chain
    if irc is None:
        summary["status"] = "irc_failed"
        summary["headline"] = "TS found, IRC failed: what it connects is unknown"
        summary["barrier_kcal"] = (e_ts - e_end["r"]) * HARTREE_TO_KCAL
        summary["barrier_floor_hartree"] = e_end["r"]
        summary["barrier_verified"] = False
        return _save(summary, out / "summary.json")
    engine.compute_energies(irc.nodes)
    a, b = _angstrom(irc.nodes[0]), _angstrom(irc.nodes[-1])
    forward = _same_bonds(sym, a, bonds["r"]) and _same_bonds(sym, b, bonds["p"])
    backward = _same_bonds(sym, b, bonds["r"]) and _same_bonds(sym, a, bonds["p"])
    if backward and not forward:
        irc = irc.copy()
        irc.nodes.reverse()
        irc.write_to_disk(out / "irc.xyz")
    verified = forward or backward
    e_irc = [float(n.energy) for n in (irc.nodes[0], irc.nodes[-1])]
    floor = min(e_end["r"], e_irc[0]) if verified else e_end["r"]
    pfloor = min(e_end["p"], e_irc[1]) if verified else e_end["p"]
    summary.update(barrier_kcal=(e_ts - floor) * HARTREE_TO_KCAL, barrier_floor_hartree=floor,
                   product_floor_hartree=pfloor, reverse_barrier_kcal=(e_ts - pfloor) * HARTREE_TO_KCAL,
                   reaction_kcal=(pfloor - floor) * HARTREE_TO_KCAL, barrier_verified=verified)
    if not verified:
        summary["status"] = "other_minima"
        summary["headline"] = (f"TS {summary['barrier_kcal']:.1f} kcal/mol, but its IRC does not connect the "
                               "substituted ends: a path search is needed")
    elif summary["status"] == "ok":
        summary["headline"] = f"barrier {summary['barrier_kcal']:.1f} kcal/mol, IRC connects both ends"
    else:
        summary["headline"] = (f"barrier {summary['barrier_kcal']:.1f} kcal/mol (IRC verified), but "
                               + "; ".join(summary["notes"]))
    return _save(summary, out / "summary.json")
