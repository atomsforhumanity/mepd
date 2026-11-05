"""Collisions as a network-expansion product generator: the interactive
Reactor's Collide button, run unattended.

Collide does what a person does by hand in the Reactor (mepd.web.sandbox):
pull two atoms toward their moving midpoint with a spring for a moment, then
let go, in Langevin dynamics on GFN2-xTB at a moderate temperature. Here one
structure gets `n_collisions` such pulls, each from its own start, on pairs
picked the way the Reactor picks them (atoms of two molecules; or of one
molecule at least `min_hops` bonds apart -- a ring closing, an H moving
across). Between two molecules the pull moves the whole molecules (the force
spread over each by mass, aimed so the two atoms meet), from a random
orientation of the partner: what forms is up to the dynamics, a concerted
cycloaddition included. The pull supplies the energy a reaction needs where
it is needed, so a unimolecular decomposition happens without heating the
whole molecule to thousands of K (the piston nanoreactor rarely makes those: Chang, Meisner,
Xu, Martinez, J. Phys. Chem. A 127, 9580 (2023)).

After the pull and a free run, the structure is quenched; if its bond graph
differs from the start (and no atom was torn off), it is one product guess,
whole and in the input's atom order. Distinct bond graphs only, in the order
the collisions ran.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR
from mepd.cpus import available_cpus

BOHR = 0.529177210903            # Angstrom
KCAL = 627.509474
AUTO_MAX = 96                    # n_collisions 0: every move up to this many

OPTIONS = {"n_collisions": 0, "temperature": 1000.0, "pull_fs": 300.0, "relax_fs": 300.0, "k": 100.0,
           "min_hops": 2, "seed": 0, "parallel": 0}

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
    tiers = [[], [], [], []]
    for i, j, push in moves:
        if push is not None:
            tiers[3].append((i, j, push))
        elif heavy(i) and heavy(j):
            tiers[0 if mol[i] != mol[j] else 1].append((i, j, push))
        else:
            tiers[2].append((i, j, push))
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


def _collide(symbols, xyz_A, charge, multiplicity, pair, temperature, pull_fs, relax_fs, k_kcal, seed, push=None):
    """One collision: pull `pair` together, and drag the bond `push` apart
    until it lets go; between two molecules, the whole molecules, from a
    random orientation of the partner. The quenched end geometry (Angstrom),
    or the error if the dynamics failed."""
    from mepd.discovery.free_energy import molecules
    from mepd.discovery.langevin import MAX_PULL, LangevinMD, pull_pairs
    from mepd.discovery.nanoreactor import _masses

    pulls = [tuple(pair)]
    mols = molecules(symbols, np.asarray(xyz_A), scale=1.3)
    A, B = (next(m for m in mols if a in m) for a in pair)
    if A is not B:
        xyz_A = _turn_partner(symbols, xyz_A, pair[1], np.random.default_rng(seed + 7919))
        pulls = [(*pair, A, B)]
    masses = _masses(symbols)
    pos = np.asarray(xyz_A, dtype=float) / BOHR
    pos = pos - pos.mean(axis=0)
    k = float(k_kcal) / KCAL * BOHR * BOHR
    if push is not None:
        from rdkit import Chem

        rc = Chem.GetPeriodicTable()
        let_go = 1.5 * (rc.GetRcovalent(symbols[push[0]]) + rc.GetRcovalent(symbols[push[1]])) / BOHR
    pulling = [True]

    def bias(x):
        if not pulling[0]:
            return np.zeros_like(x)
        g = pull_pairs(x, pulls, k, masses)
        if push is not None:
            a, b = push
            d = x[a] - x[b]
            r = np.linalg.norm(d)
            if r < let_go:          # drag apart with the capped force until the bond has let go
                f = MAX_PULL * d / max(r, 1e-9)
                g[a] -= f
                g[b] += f
        return g

    try:
        md = LangevinMD(symbols, pos, charge, multiplicity, temperature=temperature,
                        wall=(float(np.max(np.linalg.norm(pos, axis=1))) + 4.0 / BOHR) * BOHR, seed=seed, bias=bias)
        md.start()
        n_pull, n_all = int(pull_fs / md.dt), int((pull_fs + relax_fs) / md.dt)
        done = 0
        while done < n_all:
            pulling[0] = done < n_pull
            done += md.step()
        md.relax(200)                 # quench: no new chemistry
    except Exception as exc:     # reported (the first one) by collide_products
        return f"{type(exc).__name__}: {exc}"
    return md.pos * BOHR


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
    moves = collision_moves(symbols, xyz, min_hops)
    if not moves:
        print("collide: no pair to pull (one or two atoms).", flush=True)
        return []
    rng = np.random.default_rng(seed)
    # one move from each kind in turn (between molecules, within one, with an H, pull and push), each kind
    # in random order: a capped n_collisions samples every kind, not just the first
    tiers = [[t[i] for i in rng.permutation(len(t))] for t in _tiers(symbols, xyz, moves)]
    order = [t[r] for r in range(max(map(len, tiers))) for t in tiers if r < len(t)]
    # 0 (default): every move once, at most AUTO_MAX (small molecules: all of them, a fraction of a second each)
    n = int(n_collisions) or min(len(order), AUTO_MAX)
    jobs = [(order[m % len(order)], int(seed) + m) for m in range(max(n, 1))]
    width = max(1, int(parallel) or available_cpus())
    args = [(symbols, xyz, int(structure.charge), int(structure.multiplicity), (mv[0], mv[1]), float(temperature),
             float(pull_fs), float(relax_fs), float(k), s, mv[2]) for mv, s in jobs]
    ends = _run_in_processes(args, min(width, len(args)))
    from mepd.discovery.nanoreactor_generator import stereo_key

    start_key = (start, stereo_key(structure))
    out, seen, failed, torn, same = [], {start_key}, 0, 0, 0
    errors = [e for e in ends if isinstance(e, str)]
    for end in ends:
        if end is None or isinstance(end, str):
            failed += 1
            continue
        b = _bonds(symbols, end)
        if any(not any(i in p for p in b) for i in range(len(symbols))):
            torn += 1           # an atom on its own: no product guess (as the nanoreactor)
            continue
        prod = structure.model_copy(update={"geometry": end * ANGSTROM_TO_BOHR})
        key = (b, stereo_key(prod))     # diastereomers (endo, exo) are two products
        if key in seen:
            same += key == start_key
            continue
        seen.add(key)
        if len(out) < int(max_products):
            out.append(prod)
    print(f"collide: {len(out)} distinct product(s) from {len(jobs)} collisions of "
          f"{len(moves)} possible moves ({same} back to the start, {torn} with a lone atom, {failed} failed).", flush=True)
    if errors:
        print(f"collide: {len(errors)} collision(s) raised an error, the first: {errors[0]}", flush=True)
    return out


def _run_in_processes(args: list, width: int) -> list:
    """`_collide(*a)` for each of `args`, in fresh Python processes, `width`
    at a time (started, not forked: the explorer calls generators from
    threads, and a fork of a threaded process can deadlock). Each process
    gets its collision as JSON on stdin and answers in JSON."""
    import json
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    if width <= 1:
        return [_collide(*a) for a in args]
    # one collision per process, `width` at a time: fixed shares of ~10 left a round's proposals waiting on the
    # process with the slowest collisions while the others' cores idled (a process starts in ~0.1 s, a
    # collision takes ~25 s)
    chunks = [[i] for i in range(len(args))]

    def run(idx):
        payload = [[a[0], np.asarray(a[1]).tolist(), *a[2:4], list(a[4]), *a[5:10],
                    list(a[10]) if a[10] is not None else None]
                   for a in (args[i] for i in idx)]
        r = subprocess.run([sys.executable, "-c", "from mepd.discovery.collide_generator import _worker; _worker()"],
                           input=json.dumps(payload), capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        if r.returncode != 0:
            return [f"collision process failed: {r.stderr.strip()[-300:]}"] * len(idx)
        return [e if e is None or isinstance(e, str) else np.asarray(e)
                for e in json.loads(r.stdout.strip().splitlines()[-1])]

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
        sym, xyz, q, m, pair, T, pf, rf, k, seed, push = j
        e = _collide(sym, np.asarray(xyz), q, m, tuple(pair), T, pf, rf, k, seed, tuple(push) if push else None)
        ends.append(e if e is None or isinstance(e, str) else np.asarray(e).tolist())
    print(json.dumps(ends))
