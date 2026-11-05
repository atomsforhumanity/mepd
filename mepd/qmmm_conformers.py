"""Conformers of a QM/MM system's QM region, in its environment: what `mepd
channels` samples on a QM/MM step (RDKit and CREST see whole molecules, not a
species held in a protein or a solvent cage).

1. Langevin MD (BAOAB) of the QM atoms only, with the profile's QM/MM forces,
   the environment held where it is: the QM region turns over its torsions
   and hydrogen bonds in the cage it reacts in.
2. Frames at even intervals after a short equilibration whose QM bond graph
   is still the start's (a frame that reacted is no conformer) are minimized
   with the profile's engine (microiterations: the moving environment
   relaxes around each).
3. Minima that kept the QM bond graph are deduplicated by the RMSD of the QM
   atoms (no fitting: the environment fixes the frame), and the lowest
   `n_conformers` kept, the input first.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR

_ENERGY_JUMP_EH = 1.0   # between MD steps: a wrong SCC solution, not chemistry


def qmmm_region(run_inputs):
    """The QMMMRegion of a QM/MM profile's engine, or None."""
    eng = run_inputs.engine
    for _ in range(5):
        region = eng.__dict__.get("region") if hasattr(eng, "__dict__") else None
        if region is not None:
            return region
        eng = getattr(eng, "base", None)
        if eng is None:
            return None
    return None


def qm_rmsd(a, b, region) -> float:
    """RMSD (bohr) of the QM atoms, unfitted: the environment holds the frame."""
    qm = list(region.qm_atoms)
    d = np.asarray(a.coords, dtype=float)[qm] - np.asarray(b.coords, dtype=float)[qm]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def sample_qm_region_conformers(node, run_inputs, *, n_conformers: int = 10, temperature: float = 400.0,
                                time_ps: float = 1.0, frames: int = 20, step_fs: float = 0.5,
                                equilibrate_ps: float = 0.1, friction: float = 0.01, rmsd_cutoff: float = 0.1,
                                seed: int = 0, stats: Optional[dict] = None, minimize=None, echo=None) -> list:
    """Minimized QM-region conformers of a QM/MM structure, the input first,
    then by energy. `rmsd_cutoff` (bohr, QM atoms) tells two apart.
    `minimize(nodes) -> nodes` minimizes candidates (default: the profile's
    engine, one by one); `stats` gets what each stage let through."""
    from mepd.discovery.nanoreactor import _ACC, _KB, _KIN, _masses
    from mepd.discovery.qmmm_reactor import _xtb_qm_engine, qm_bond_key
    from mepd.nodes.node import StructureNode

    echo = echo or (lambda *_: None)
    stats = stats if stats is not None else {}
    region = qmmm_region(run_inputs)
    if region is None:
        raise ValueError("QM-region conformers need a QM/MM profile ([qmmm])")
    engine = run_inputs.engine
    structure = node.structure
    symbols = [str(s) for s in structure.symbols]
    qm = np.asarray(region.qm_atoms, dtype=int)
    key0 = qm_bond_key(structure, region)
    masses = _masses(symbols)[qm][:, None]
    kt = _KB * float(temperature) / _KIN
    c1 = np.exp(-friction * step_fs)
    c2 = np.sqrt((1 - c1 ** 2) * kt / masses)
    rng = np.random.default_rng(seed)
    xtb = _xtb_qm_engine(engine)
    last = {"energy": None}

    def gradient(pos):
        n = StructureNode(structure=structure.model_copy(update={"geometry": pos}), has_molecular_graph=False)
        g = np.array(engine.compute_gradients([n])[0], dtype=float).reshape(-1, 3)
        if xtb is not None and last["energy"] is not None and abs(float(n.energy) - last["energy"]) > _ENERGY_JUMP_EH:
            xtb.forget_scc_guess()
            n = StructureNode(structure=n.structure, has_molecular_graph=False)
            g = np.array(engine.compute_gradients([n])[0], dtype=float).reshape(-1, 3)
        last["energy"] = float(n.energy)
        return g[qm]

    n_steps = max(1, int(round(time_ps * 1000.0 / step_fs)))
    first = min(n_steps, int(round(equilibrate_ps * 1000.0 / step_fs)))
    takes = set(np.linspace(first, n_steps, max(1, int(frames)), dtype=int).tolist())
    pos = np.asarray(structure.geometry, dtype=float).copy()
    x = pos[qm].copy()
    v = rng.normal(size=x.shape) * np.sqrt(kt / masses)
    g = gradient(pos)
    candidates, reacted = [], 0
    echo(f"QM-region MD: {len(qm)} QM atoms at {temperature:g} K for {time_ps:g} ps ({n_steps} QM/MM gradients), "
         f"the environment held...")
    for step in range(1, n_steps + 1):
        v -= 0.5 * step_fs * _ACC * g / masses
        x += 0.5 * step_fs * v
        v = c1 * v + c2 * rng.normal(size=v.shape)
        x += 0.5 * step_fs * v
        pos[qm] = x
        g = gradient(pos)
        v -= 0.5 * step_fs * _ACC * g / masses
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"the QM-region MD blew up at step {step}")
        if step in takes:
            frame = structure.model_copy(update={"geometry": pos.copy()})
            if qm_bond_key(frame, region) == key0:
                candidates.append(StructureNode(structure=frame))
            else:
                reacted += 1
    stats.update(backend="qmmm-md", temperature=float(temperature), time_ps=float(time_ps),
                 n_generated=len(candidates) + reacted, n_reacted_frames=reacted)
    echo(f"  {len(candidates)} frame(s) to minimize ({reacted} reacted, left out).")
    if minimize is None:
        def minimize(nodes):
            out = []
            for n in nodes:
                try:
                    traj = engine.compute_geometry_optimization(n, keywords={"fmax": 0.02, "maxiter": 500})
                    out.append(traj[-1] if traj else None)
                except Exception:
                    out.append(None)
            return out
    minima = [m for m in minimize(candidates) if m is not None]
    same = [m for m in minima if qm_bond_key(m.structure, region) == key0]
    stats["n_minimized"], stats["n_changed_on_minimizing"] = len(minima), len(minima) - len(same)
    energies = []
    for m in same:
        try:
            energies.append(float(m.energy))
        except Exception:
            energies.append(float("inf"))
    kept = [node]
    for _, m in sorted(zip(energies, same), key=lambda t: t[0]):
        if n_conformers and len(kept) >= n_conformers:
            break
        if all(qm_rmsd(m, k, region) >= rmsd_cutoff for k in kept):
            kept.append(StructureNode(structure=m.structure))
            kept[-1]._cached_energy = m._cached_energy
    stats["n_kept"] = len(kept)
    echo(f"  {len(kept)} distinct QM-region conformer(s) kept (input included).")
    return kept
