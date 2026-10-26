"""Collisions as a network-expansion product generator: the interactive
Reactor's Collide button, run unattended.

Collide does what a person does by hand in the Reactor (mepd.web.sandbox):
pull two atoms toward their moving midpoint with a spring for a moment, then
let go, in Langevin dynamics on GFN2-xTB at a moderate temperature. Here one
structure gets `n_collisions` such pulls, each from its own start, on pairs
picked the way the Reactor picks them (atoms of two molecules; or of one
molecule at least `min_hops` bonds apart -- a ring closing, an H moving
across). The pull supplies the energy a reaction needs where it is needed,
so a unimolecular decomposition happens without heating the whole molecule
to thousands of K (the piston nanoreactor rarely makes those: Chang, Meisner,
Xu, Martinez, J. Phys. Chem. A 127, 9580 (2023)).

After the pull and a free run, the structure is quenched; if its bond graph
differs from the start (and no atom was torn off), it is one product guess,
whole and in the input's atom order. Distinct bond graphs only, in the order
the collisions ran.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR

BOHR = 0.529177210903            # Angstrom
KCAL = 627.509474
AUTO_MAX = 96                    # n_collisions 0: every move up to this many
MAX_PULL = 0.08                  # Hartree/bohr cap on the pull's force, as the Reactor's (~100 kcal/mol/A)

OPTIONS = {"n_collisions": 0, "temperature": 1000.0, "pull_fs": 300.0, "relax_fs": 300.0, "k": 100.0,
           "min_hops": 2, "seed": 0, "parallel": 0}

_Z = {"H": 1, "He": 2, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12, "Si": 14, "P": 15,
      "S": 16, "Cl": 17, "K": 19, "Br": 35, "I": 53}


def _bonds(symbols, xyz_A, scale: float = 1.3) -> frozenset:
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    rc = np.array([table.GetRcovalent(s) for s in symbols])
    d = np.linalg.norm(xyz_A[:, None] - xyz_A[None], axis=-1)
    n = len(symbols)
    return frozenset((i, j) for i in range(n) for j in range(i) if d[i, j] < scale * (rc[i] + rc[j]))


def _hops(n: int, bonds) -> np.ndarray:
    nb = [[] for _ in range(n)]
    for i, j in bonds:
        nb[i].append(j)
        nb[j].append(i)
    out = np.full((n, n), np.inf)
    for s in range(n):
        out[s, s] = 0
        q = [s]
        for u in q:
            for v in nb[u]:
                if out[s, v] == np.inf:
                    out[s, v] = out[s, u] + 1
                    q.append(v)
    return out


def collision_pairs(symbols, xyz_A, min_hops: int = 2) -> list[tuple[int, int]]:
    """Every pair Collide could pull: atoms of two molecules, or of one
    molecule at least `min_hops` bonds apart. Two hydrogens count (H2 loss)."""
    n = len(symbols)
    hops = _hops(n, _bonds(symbols, np.asarray(xyz_A)))
    return [(i, j) for i in range(n) for j in range(i) if hops[i, j] >= max(2, int(min_hops))]


def collision_moves(symbols, xyz_A, min_hops: int = 2) -> list[tuple[int, int, Optional[tuple[int, int]]]]:
    """Every collision: a pair to pull together, alone or with one bond on the
    path between them dragged apart -- a person in the Reactor holding one atom in while
    pulling another away. One move makes the bond the pull forms; with the
    push it can also break the bond that must go at the same time (a
    four-centre elimination such as HCOOH -> CO + H2O)."""
    bonds = _bonds(symbols, np.asarray(xyz_A))
    hops = _hops(len(symbols), bonds)
    out = []
    for i, j in collision_pairs(symbols, xyz_A, min_hops):
        out.append((i, j, None))
        # only a bond on a shortest path between the two: the one that must give way as the new bond forms
        # (four-centre: i-x-j, push x-j or i-x; six-centre: along i-x-y-j)
        on_path = lambda a, b: (hops[i, a] + 1 + hops[b, j] == hops[i, j]) or (hops[i, b] + 1 + hops[a, j] == hops[i, j])
        out += [(i, j, b) for b in sorted(bonds) if np.isfinite(hops[i, j]) and on_path(*b)]
    return out


_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3, "Si": 4}


def cycloaddition_moves(symbols, xyz_A) -> list:
    """Two pulls at once between two molecules, [4+2] style: the ends of a
    conjugated diene (three bonds apart through unsaturated atoms) of one onto
    the two atoms of a pi bond of the other (a Diels-Alder or hetero-Diels-
    Alder). One pull alone makes a single bond that falls back; a concerted
    cycloaddition needs both. Returns (i, j, None, (k, l)): pull i-j and k-l."""
    from mepd.discovery.free_energy import molecules

    xyz_A = np.asarray(xyz_A)
    bonds = _bonds(symbols, xyz_A)
    nb = {a: set() for a in range(len(symbols))}
    for a, b in bonds:
        nb[a].add(b)
        nb[b].add(a)
    unsat = {a for a in nb if symbols[a] in _VALENCE and len(nb[a]) < _VALENCE[symbols[a]]}
    sub = frozenset((a, b) for a, b in bonds if a in unsat and b in unsat)
    hops = _hops(len(symbols), sub)
    mols = molecules(symbols, xyz_A, scale=1.3)
    out = []
    for A in mols:
        for B in mols:
            if A is B:
                continue
            dienes = [(a, b) for a in A for b in A if a < b and a in unsat and b in unsat and hops[a, b] == 3]
            enes = [(a, b) for a in B for b in B if a < b and ((a, b) in sub or (b, a) in sub)]
            for d1, d2 in dienes:
                for e1, e2 in enes:
                    out += [(d1, e1, None, (d2, e2)), (d1, e2, None, (d2, e1))]
    return out


def _tiers(symbols, xyz_A, moves) -> list[list]:
    """Moves in the order Collide tries them: (1) two heavy atoms of two
    molecules -- the Reactor's own choice when there are several molecules
    (bimolecular chemistry: additions, cycloadditions); (2) two heavy atoms of
    one molecule (ring closures, rearrangements); (3) pulls with a hydrogen
    (H transfers, H2 loss); (4) pulls with a bond dragged apart
    (eliminations). Taken in turn, one of each, so a capped n_collisions samples every kind."""
    from mepd.discovery.free_energy import molecules

    mol = {}
    for k, group in enumerate(molecules(symbols, np.asarray(xyz_A), scale=1.3)):
        for a in group:
            mol[a] = k
    heavy = lambda a: symbols[a] != "H"
    tiers = [[], [], [], [], []]
    for m in moves:
        i, j, push = m[:3]
        if len(m) > 3 and m[3] is not None:
            tiers[0].append(m)        # cycloadditions first: twice, each from its own random orientation
        elif push is not None:
            tiers[4].append(m)
        elif heavy(i) and heavy(j):
            tiers[1 if mol[i] != mol[j] else 2].append(m)
        else:
            tiers[3].append(m)
    tiers[0] = tiers[0] * 2
    return tiers


def _turn_partner(symbols, xyz_A, atom, rng) -> np.ndarray:
    """The molecule holding `atom` turned to a random orientation about its own
    centre (a collision comes from any direction), then moved out along the line
    between the centres until no two atoms of the two are closer than 2.2 A."""
    from scipy.spatial.transform import Rotation

    from mepd.discovery.free_energy import molecules

    xyz = np.array(xyz_A, dtype=float)
    mols = molecules(symbols, xyz, scale=1.3)
    B = next(m for m in mols if atom in m)
    rest = [a for a in range(len(symbols)) if a not in B]
    if not rest:
        return xyz
    c = xyz[B].mean(axis=0)
    xyz[B] = (xyz[B] - c) @ Rotation.random(random_state=rng).as_matrix().T + c
    away = c - xyz[rest].mean(axis=0)
    away = away / max(np.linalg.norm(away), 1e-9)
    for _ in range(100):
        if np.linalg.norm(xyz[B][:, None] - xyz[rest][None], axis=-1).min() >= 2.2:
            break
        xyz[B] += 0.2 * away
    return xyz


def _collide(symbols, xyz_A, charge, multiplicity, pair, temperature, pull_fs, relax_fs, k_kcal, seed, push=None,
             pull2=None):
    """One collision: pull `pair` (and `pull2`) together, and drag the bond
    `push` apart until it lets go; between two molecules, from a random
    orientation of the partner. The quenched end geometry (Angstrom), or None
    if the dynamics failed."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from tblite.interface import Calculator

    from mepd.discovery.nanoreactor import _masses, _wall

    Z = np.array([_Z[s] for s in symbols])
    from mepd.discovery.free_energy import molecules

    pulls = [tuple(pair)] + ([tuple(pull2)] if pull2 is not None else [])
    mols = molecules(symbols, np.asarray(xyz_A), scale=1.3)
    if next(m for m in mols if pair[0] in m) is not next(m for m in mols if pair[1] in m):
        xyz_A = _turn_partner(symbols, xyz_A, pair[1], np.random.default_rng(seed + 7919))
    pos = np.asarray(xyz_A, dtype=float) / BOHR
    pos = pos - pos.mean(axis=0)
    masses = _masses(symbols)
    radius = float(np.max(np.linalg.norm(pos, axis=1))) + 4.0 / BOHR
    calc = Calculator("GFN2-xTB", Z, pos, charge=float(charge), uhf=int(multiplicity) - 1)
    calc.set("verbosity", 0)
    calc.set("temperature", 3000.0 * 3.166811563e-6)   # Fermi smearing, as the Reactor: bonds break cleanly
    wall = 20.0 / KCAL * BOHR
    k = float(k_kcal) / KCAL * BOHR * BOHR
    if push is not None:
        from rdkit import Chem

        rc = Chem.GetPeriodicTable()
        let_go = 1.5 * (rc.GetRcovalent(symbols[push[0]]) + rc.GetRcovalent(symbols[push[1]])) / BOHR
    res = [None]

    def forces(x, pulling):
        calc.update(x)
        try:
            r = calc.singlepoint(res[0]) if res[0] is not None else calc.singlepoint()
        except Exception:
            r = calc.singlepoint()
        res[0] = r
        g = np.array(r.get("gradient"))
        g = g + _wall(x, radius, wall)[1]
        if pulling:
            for i, j in pulls:
                mid = 0.5 * (x[i] + x[j])
                for a in (i, j):
                    f = k * (x[a] - mid)
                    norm = np.linalg.norm(f)
                    if norm > MAX_PULL:
                        f *= MAX_PULL / norm
                    g[a] += f
            if push is not None:
                a, b = push
                d = x[a] - x[b]
                r = np.linalg.norm(d)
                if r < let_go:          # drag apart with the capped force until the bond has let go
                    f = MAX_PULL * d / max(r, 1e-9)
                    g[a] -= f
                    g[b] += f
        return g

    kB = 3.166811563e-6 / 1.066551
    acc = 0.93766
    dt, gamma = 0.5, 0.02
    rng = np.random.default_rng(seed)
    vel = rng.normal(size=pos.shape) * np.sqrt(kB * temperature / masses)[:, None]
    c1 = math.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1 * c1) * kB * temperature / masses)[:, None]
    try:
        grad = forces(pos, True)
        n_pull, n_all = int(pull_fs / dt), int((pull_fs + relax_fs) / dt)
        for step in range(n_all):
            pulling = step < n_pull
            vel -= 0.5 * dt * acc * grad / masses[:, None]
            pos = pos + 0.5 * dt * vel
            vel = c1 * vel + c2 * rng.normal(size=vel.shape)
            pos = pos + 0.5 * dt * vel
            grad = forces(pos, pulling)
            vel -= 0.5 * dt * acc * grad / masses[:, None]
            if not np.all(np.isfinite(pos)):
                return None
        for _ in range(200):          # quench: steepest descent, steps capped (no new chemistry)
            step = -grad / masses[:, None] * 20.0
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            pos = pos + np.where(norm > 0.1, step * 0.1 / np.maximum(norm, 1e-12), step)
            grad = forces(pos, False)
    except Exception:
        return None
    return pos * BOHR


def collide_products(structure, *, max_products: int = 20, n_collisions: int = 0, temperature: float = 1000.0,
                     pull_fs: float = 300.0, relax_fs: float = 300.0, k: float = 100.0, min_hops: int = 2,
                     seed: int = 0, parallel: int = 0) -> list:
    """Product structures (input atom order) from `n_collisions` pulls of
    pairs Collide could pick (all of them when there are fewer; a random
    sample otherwise), `parallel` at a time (default: all CPUs)."""
    import os

    symbols = [str(s) for s in structure.symbols]
    xyz = np.asarray(structure.geometry, dtype=float) / ANGSTROM_TO_BOHR
    start = _bonds(symbols, xyz)
    moves = collision_moves(symbols, xyz, min_hops) + cycloaddition_moves(symbols, xyz)
    if not moves:
        print("collide: no pair to pull (one or two atoms).", flush=True)
        return []
    rng = np.random.default_rng(seed)
    # one move from each kind in turn (cycloadditions, between molecules, within one, with an H, pull and
    # push), each kind in random order: a capped n_collisions samples every kind, not just the first
    tiers = [[t[i] for i in rng.permutation(len(t))] for t in _tiers(symbols, xyz, moves)]
    order = [t[r] for r in range(max(map(len, tiers))) for t in tiers if r < len(t)]
    # 0 (default): every move once, at most AUTO_MAX (small molecules: all of them, a fraction of a second each)
    n = int(n_collisions) or min(len(order), AUTO_MAX)
    jobs = [(order[m % len(order)], int(seed) + m) for m in range(max(n, 1))]
    width = max(1, int(parallel) or os.cpu_count() or 1)
    args = [(symbols, xyz, int(structure.charge), int(structure.multiplicity), (mv[0], mv[1]), float(temperature),
             float(pull_fs), float(relax_fs), float(k), s, mv[2], mv[3] if len(mv) > 3 else None) for mv, s in jobs]
    ends = _run_in_processes(args, min(width, len(args)))
    out, seen, failed, torn, same = [], {start}, 0, 0, 0
    for end in ends:
        if end is None:
            failed += 1
            continue
        b = _bonds(symbols, end)
        if b in seen:
            same += b == start
            continue
        if any(not any(i in p for p in b) for i in range(len(symbols))):
            torn += 1           # an atom on its own: no product guess (as the nanoreactor)
            continue
        seen.add(b)
        if len(out) < int(max_products):
            out.append(structure.model_copy(update={"geometry": end * ANGSTROM_TO_BOHR}))
    print(f"collide: {len(out)} distinct product(s) from {len(jobs)} collisions of "
          f"{len(moves)} possible moves ({same} back to the start, {torn} with a lone atom, {failed} failed).", flush=True)
    return out


def _run_in_processes(args: list, width: int) -> list:
    """`_collide(*a)` for each of `args`, in `width` fresh Python processes
    (started, not forked: the explorer calls generators from threads, and a
    fork of a threaded process can deadlock). Each process gets its share as
    JSON on stdin and answers in JSON."""
    import json
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    if width <= 1:
        return [_collide(*a) for a in args]
    chunks = [list(range(k, len(args), width)) for k in range(width)]

    def run(idx):
        payload = [[a[0], np.asarray(a[1]).tolist(), *a[2:4], list(a[4]), *a[5:10],
                    list(a[10]) if a[10] is not None else None, list(a[11]) if a[11] is not None else None]
                   for a in (args[i] for i in idx)]
        r = subprocess.run([sys.executable, "-c", "from mepd.discovery.collide_generator import _worker; _worker()"],
                           input=json.dumps(payload), capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        if r.returncode != 0:
            return [None] * len(idx)
        return [None if e is None else np.asarray(e) for e in json.loads(r.stdout.strip().splitlines()[-1])]

    import os

    out = [None] * len(args)
    with ThreadPoolExecutor(max_workers=width) as pool:
        for idx, ends in zip(chunks, pool.map(run, chunks)):
            for i, e in zip(idx, ends):
                out[i] = e
    return out


def _worker() -> None:
    import json
    import sys

    jobs = json.loads(sys.stdin.read())
    ends = []
    for j in jobs:
        sym, xyz, q, m, pair, T, pf, rf, k, seed, push, pull2 = j
        e = _collide(sym, np.asarray(xyz), q, m, tuple(pair), T, pf, rf, k, seed, tuple(push) if push else None,
                     tuple(pull2) if pull2 else None)
        ends.append(None if e is None else np.asarray(e).tolist())
    print(json.dumps(ends))
