"""What a molecule beside a reaction step does to it, without taking part: the bystander estimate.

`bystander(frames, molecule, engine)` places a molecule C next to the reactant end, the transition
state and the product end of one step (the three frames in one atom order), relaxes C with the step's
atoms held, and returns the lowest complex of each frame, C's binding energy to each, and from those the
shift of the barrier and of the reaction energy. Docs: docs/bystander.md (what the number is and is not).

Placers are generic: none knows a reaction class, a functional group or a mechanism.
  "random"      random packed orientations (`mepd.complexes.packed`);
  "seeded"      C's most charged atoms put against the step's atoms whose partial charge changes most
                between the ends and the TS and against its most polar atoms, opposite charges facing;
  "screen-ff"   many rigid poses scored by point charges + a 12-6 term, the best relaxed;
  "screen-sp"   the same pool scored by single points of `engine`;
  "mixed"       half the poses seeded, half screened by single points.
With `carry` (default) the best poses of each frame are also relaxed on the other two frames, so the
three frames are compared at the same sites.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR

VERSION = 1
HARTREE_TO_KCAL = 627.509474
PLACERS = ("mixed", "random", "seeded", "screen-ff", "screen-sp")
#: Bondi radii (A. Bondi, J. Phys. Chem. 68, 441 (1964); H from Rowland and Taylor 1996), 2.0 A otherwise.
VDW = {"H": 1.10, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Na": 2.27, "Mg": 1.73, "Al": 1.84,
       "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "K": 2.75, "Br": 1.85, "I": 1.98, "Li": 1.82}
UNMOVED = 1e-3          # A: a relaxation that moved no atom of C further than this did not relax it
CONTACT = 1.3           # C is in contact when one of its atoms is within 1.3 x the vdW radii of the step's
START = 0.85            # placed poses start with the facing atoms at 0.85 x their vdW radii
CLASH = 0.6             # ... and no other pair closer than 0.6 x


def _xyz(node) -> np.ndarray:
    return np.asarray(node.structure.geometry, dtype=float).reshape(-1, 3) / ANGSTROM_TO_BOHR


def _node(symbols, xyz, charge: int, multiplicity: int = 1):
    from qcdata import Structure

    from mepd.nodes.node import StructureNode

    return StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(xyz) * ANGSTROM_TO_BOHR,
                                             charge=int(charge), multiplicity=int(multiplicity)))


def _rvdw(symbols) -> np.ndarray:
    return np.array([VDW.get(s, 2.0) for s in symbols])


def _energy(engine, node) -> float:
    return float(engine.compute_energies([node])[0])


def gfn2_charges(node) -> np.ndarray:
    """GFN2-xTB partial charges (axtb in process): what the seeded placer and the point-charge screen read."""
    from mepd.engines import axtb_backend
    from mepd.engines.axtb_backend import _molecule

    numbers, positions, charge, uhf = _molecule(node)
    return np.asarray(axtb_backend.gfn2_point(numbers, positions, charge=charge, uhf=uhf)["charges"], dtype=float)


def conformers(smiles: str, engine, *, n: int = 5, seed: int = 0, window_kcal: float = 5.0) -> list:
    """Up to `n` conformers of a molecule from RDKit (ETKDG), each minimized with `engine`, those that kept
    their bonds, lowest first, duplicates (same energy to 0.05 kcal/mol) and those above the window dropped."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from mepd.network_variants import adjacency
    from mepd.substituents import relax_group

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, 4 * n), randomSeed=seed + 1))
    if not ids:
        raise ValueError(f"RDKit could not embed {smiles!r}")
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    charge = Chem.GetFormalCharge(mol)
    want = np.asarray(Chem.GetAdjacencyMatrix(mol), dtype=bool)
    out = []
    for cid in ids:
        node = relax_group(engine, _node(symbols, mol.GetConformer(cid).GetPositions(), charge),
                           list(range(len(symbols))), fmax=0.01, steps=500)
        if np.array_equal(adjacency(symbols, _xyz(node)), want):
            out.append((float(node.energy), node))
    if not out:
        raise ValueError(f"no conformer of {smiles!r} kept its bonds at this level (another protonation form?)")
    out.sort(key=lambda t: t[0])
    kept = []
    for e, node in out:
        if (e - out[0][0]) * HARTREE_TO_KCAL > window_kcal:
            break
        if all(abs(e - k[0]) * HARTREE_TO_KCAL > 0.05 for k in kept):
            kept.append((e, node))
    return [node for _, node in kept[:n]]


# ------------------------------------------------------------------ geometry
def _rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b+c*c-d*d]])


def _onto(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The rotation taking unit vector u onto unit vector v."""
    w = np.cross(u, v)
    s, c = np.linalg.norm(w), float(np.dot(u, v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * K @ K


def _spin(axis: np.ndarray, angle: float) -> np.ndarray:
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """R, t with P @ R.T + t ~ Q."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cq - R @ cp


def _ratio(X, rx, C, rc) -> np.ndarray:
    """Distances between the step's atoms and C's, over the sum of their vdW radii."""
    return np.linalg.norm(X[:, None] - C[None], axis=-1) / (rx[:, None] + rc[None])


# ------------------------------------------------------------------- placers
def sites(charges: dict, k: int = 3) -> list[int]:
    """The step's atoms a polar molecule is put against: the `k` whose partial charge differs most between
    the TS and either end, and the `k` with the largest charge in any frame. `charges`: frame -> charges."""
    q = np.array([charges[f] for f in charges])
    polar = np.abs(q).max(axis=0)
    change = (np.abs(q - np.asarray(charges["ts"])[None]).max(axis=0) if "ts" in charges
              else np.abs(q - q[0][None]).max(axis=0))
    out = []
    for score in (change, polar):
        for i in np.argsort(-score)[:k]:
            if int(i) not in out:
                out.append(int(i))
    return out


def _seeded(X, symx, qx, site_atoms, confs, rng, n: int) -> list:
    """`n` poses: for a site and one of C's three most oppositely charged atoms, C turned so that atom faces
    the site from outside (along the direction pointing away from the site's neighbours), spun at random."""
    rx = _rvdw(symx)
    combos = []
    for si, s in enumerate(site_atoms):
        for ci, (csym, cxyz, cq, _e) in enumerate(confs):
            order = np.argsort(cq if qx[s] > 0 else -cq)[:3]          # opposite sign, largest first
            for pi, a in enumerate(order):
                combos.append((si + pi + ci + rng.uniform(0, 0.9), s, ci, int(a)))
    combos.sort(key=lambda t: t[0])
    poses = []
    k = 0
    while len(poses) < n and combos and k < 20 * n:
        _, s, ci, a = combos[k % len(combos)]
        k += 1
        csym, cxyz = confs[ci][:2]
        rc = _rvdw(csym)
        away = X[s] - np.delete(X, s, axis=0)
        d = (away / np.linalg.norm(away, axis=1, keepdims=True) ** 3).sum(axis=0) if len(away) else rng.normal(size=3)
        d = d / (np.linalg.norm(d) + 1e-12) + rng.normal(scale=0.2, size=3)
        d /= np.linalg.norm(d)
        body = cxyz.mean(axis=0) - cxyz[a]
        R0 = _onto(body / np.linalg.norm(body), d) if np.linalg.norm(body) > 1e-6 else _rotation(rng)
        dist = START * (rx[s] + rc[a])
        for attempt in range(24):
            R = _spin(d, rng.uniform(0, 2 * np.pi)) @ R0
            C = (cxyz - cxyz[a]) @ R.T + X[s] + d * dist
            ratio = _ratio(X, rx, C, rc)
            ratio[s, a] = 9.0
            if ratio.min() >= CLASH:
                poses.append((ci, C, f"seeded:site{s}:atom{a}"))
                break
            if attempt % 6 == 5:
                dist += 0.4
    return poses


def _random(X, symx, confs, rng, n: int) -> list:
    """`n` random packed poses (mepd.complexes.packed), the step's atoms put back on the frame."""
    from mepd.complexes import _structure, packed

    poses = []
    for _ in range(n):
        ci = int(rng.integers(len(confs)))
        csym, cxyz = confs[ci][:2]
        both = packed([_structure(symx, X, 0), _structure(csym, cxyz, 0)], seed=int(rng.integers(2 ** 31 - 2)))
        g = np.asarray(both.geometry, dtype=float).reshape(-1, 3) / ANGSTROM_TO_BOHR
        R, t = _kabsch(g[:len(X)], X)
        poses.append((ci, g[len(X):] @ R.T + t, "random"))
    return poses


def _surface(X, symx, confs, rng, m: int) -> list:
    """`m` rigid poses on the step's surface: a random direction and orientation, C moved in until its
    closest atom is at 0.85 x the vdW radii."""
    rx = _rvdw(symx)
    centre = X.mean(axis=0)
    reach = float(np.linalg.norm(X - centre, axis=1).max())
    poses = []
    for _ in range(m):
        ci = int(rng.integers(len(confs)))
        csym, cxyz = confs[ci][:2]
        rc = _rvdw(csym)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        # a random point of the step as the target, so pockets are reached and not only the hull
        target = X[int(rng.integers(len(X)))]
        C0 = (cxyz - cxyz.mean(axis=0)) @ _rotation(rng).T
        far = reach + float(np.linalg.norm(C0, axis=1).max()) + 4.0
        lo, hi = 0.0, 2 * far
        base = target + u * 0.0
        for _ in range(18):                      # bisect on the distance along u from the target
            mid = 0.5 * (lo + hi)
            if _ratio(X, rx, C0 + base + u * mid, rc).min() < START:
                lo = mid
            else:
                hi = mid
        poses.append((ci, C0 + base + u * hi, "surface"))
    return poses


def _ff_score(X, rx, qx, C, rc, qc) -> float:
    """Point charges (kcal/mol) + a 12-6 term (0.1 kcal/mol wells at 0.9 x the vdW radii): the rigid screen."""
    d = np.linalg.norm(X[:, None] - C[None], axis=-1)
    s = (0.9 * (rx[:, None] + rc[None]) / d) ** 6
    return float((332.06 * qx[:, None] * qc[None] / d).sum() + (0.1 * (s * s - 2 * s)).sum())


def _distinct(scored: list, n: int) -> list:
    """The `n` best-scored poses that are not within 1 A (RMS over C's atoms) of a better one."""
    out = []
    for _, pose in sorted(scored, key=lambda t: t[0]):
        if all(p[0] != pose[0] or np.sqrt(((p[1] - pose[1]) ** 2).sum(axis=1).mean()) > 1.0 for p in out):
            out.append(pose)
        if len(out) == n:
            break
    return out


def place(placer: str, engine, symx, X, qx, site_atoms, confs, rng, n: int, *, charge: int = 0,
          multiplicity: int = 1, pool: int = 200) -> list:
    """`n` starting poses [(conformer index, C's coordinates in A, origin)] of C on one frame."""
    if placer == "mixed":       # half seeded, half screened by single points (the odd one seeded)
        k = (n + 1) // 2
        return (_seeded(X, symx, qx, site_atoms, confs, rng, k)
                + place("screen-sp", engine, symx, X, qx, site_atoms, confs, rng, n - k, charge=charge,
                        multiplicity=multiplicity, pool=pool))
    if placer == "random":
        return _random(X, symx, confs, rng, n)
    if placer == "seeded":
        return _seeded(X, symx, qx, site_atoms, confs, rng, n)
    if placer not in ("screen-ff", "screen-sp"):
        raise ValueError(f"placer {placer!r}: one of {PLACERS}")
    cands = _surface(X, symx, confs, rng, pool)
    rx = _rvdw(symx)
    if placer == "screen-ff":
        scored = [(_ff_score(X, rx, qx, C, _rvdw(confs[ci][0]), confs[ci][2]), (ci, C, "screen-ff"))
                  for ci, C, _ in cands]
    else:
        nodes = [_node(list(symx) + list(confs[ci][0]), np.vstack([X, C]), charge, multiplicity) for ci, C, _ in cands]
        scored = []
        for (ci, C, _), node in zip(cands, nodes):
            try:
                scored.append((_energy(engine, node) - confs[ci][3], (ci, C, "screen-sp")))
            except Exception:
                continue
    return _distinct(scored, n)


# ---------------------------------------------------------------- relaxation
def relax_pose(engine, symbols, xyz, free, *, charge: int, multiplicity: int, bonds_x, bonds_c, n_x: int,
               fmax: float = 0.05, steps: int = 300) -> dict:
    """One placed complex relaxed with every atom outside `free` held (mepd.substituents.relax_group), and
    what became of it: energy, geometry, how far C moved, its closest contact, whether C or the step's atoms
    changed their own bonds (`reacted`), and the pairs of one atom each within bonding range (`cross_bonds`:
    a coordinated Lewis acid or a new covalent bond; the distance alone cannot tell them apart)."""
    from mepd.network_variants import adjacency
    from mepd.substituents import relax_group

    xyz = np.asarray(xyz, dtype=float)
    out = relax_group(engine, _node(symbols, xyz, charge, multiplicity), list(free), fmax=fmax, steps=steps)
    new = _xyz(out)
    adj = adjacency(symbols, new)
    ratio = _ratio(new[:n_x], _rvdw(symbols[:n_x]), new[n_x:], _rvdw(symbols[n_x:]))
    return {"energy": float(out.energy) if out.energy is not None else _energy(engine, out), "xyz": new,
            "moved": float(np.abs(new[n_x:] - xyz[n_x:]).max()), "contact": float(ratio.min()),
            "reacted": bool(not np.array_equal(adj[:n_x, :n_x], bonds_x) or not np.array_equal(adj[n_x:, n_x:], bonds_c)),
            "cross_bonds": [[int(i), int(j) + n_x] for i, j in zip(*np.nonzero(adj[:n_x, n_x:]))]}


def _eligible(p: dict) -> bool:
    return not p["reacted"] and p["moved"] > UNMOVED and p["contact"] <= CONTACT


# -------------------------------------------------------------- free energy
def rigid_modes(hessian, symbols, xyz, n_x: int) -> np.ndarray:
    """The six frequencies (cm^-1; negative: imaginary) of C moving as a rigid body against the step's atoms
    as another rigid body: the Hessian (Eh/bohr^2) of the complex projected on the relative translations
    and rotations of the two, mass-weighted."""
    from mepd.thermo import _masses

    m = np.asarray(_masses(symbols), dtype=float)
    r = np.asarray(xyz, dtype=float) * ANGSTROM_TO_BOHR
    sm = np.sqrt(np.repeat(m, 3))

    def rigid(idx) -> np.ndarray:
        vecs = []
        com = (m[idx, None] * r[idx]).sum(axis=0) / m[idx].sum()
        for a in range(3):
            t = np.zeros((len(m), 3))
            t[idx, a] = 1.0
            vecs.append(t.ravel() * sm)
            e = np.zeros(3)
            e[a] = 1.0
            w = np.zeros((len(m), 3))
            w[idx] = np.cross(e, r[idx] - com)
            vecs.append(w.ravel() * sm)
        return np.array(vecs).T

    x, c = np.arange(n_x), np.arange(n_x, len(m))
    def basis(v) -> np.ndarray:
        U, s, _ = np.linalg.svd(v, full_matrices=False)
        return U[:, s > 1e-6 * s.max()]

    whole = basis(rigid(np.arange(len(m))))
    both = np.hstack([rigid(x), rigid(c)])
    B = basis(both - whole @ (whole.T @ both))                  # without the complex's own rigid motions
    H = np.asarray(hessian, dtype=float)
    H = 0.5 * (H + H.T) / np.outer(sm, sm)
    w = np.linalg.eigvalsh(B.T @ H @ B)
    # Eh/(bohr^2 amu) -> cm^-1
    return np.sign(w) * np.sqrt(np.abs(w)) * 5140.4871


def binding_free_correction(engine, symbols, xyz, n_x: int, *, charge: int = 0, multiplicity: int = 1,
                            temperature: float = 298.15) -> dict:
    """G - E of binding at 1 mol/L, estimated without relaxing the complex: C's translation and rotation
    (and the step's) are replaced by the complex's, and by the six rigid-body vibrations of C against the
    step (`rigid_modes`, quasi-RRHO as mepd.thermo.qrrho). The internal vibrations of both are taken as
    unchanged, and rotational symmetry numbers as 1. kcal/mol."""
    from mepd.thermo import qrrho

    xyz = np.asarray(xyz, dtype=float)
    freqs = rigid_modes(engine.compute_hessian(_node(symbols, xyz, charge, multiplicity)), symbols, xyz, n_x)
    g = (qrrho([abs(f) for f in freqs], symbols, xyz, temperature)["g_corr"]
         - qrrho([], symbols[:n_x], xyz[:n_x], temperature)["g_corr"]
         - qrrho([], symbols[n_x:], xyz[n_x:], temperature)["g_corr"])
    return {"g_corr_kcal": float(g), "rigid_modes_cm": [float(f) for f in freqs]}


# ------------------------------------------------------------------ the call
def search(frames: dict, confs: list, engine, *, placer: str, poses: int, seed: int, carry: int = 1,
           relax_substrate: bool = False, reaction_atoms: Optional[Sequence[int]] = None,
           proposed: Optional[dict] = None, charges: Optional[dict] = None, charge: int = 0,
           multiplicity: int = 1, pool: int = 200, fmax: float = 0.05, steps: int = 300,
           on_pose: Optional[Callable] = None, polish: int = 0) -> dict:
    """One independent repeat: frame -> every relaxed pose, in the order relaxed (placed ones, then the ones
    carried from the other frames), each as `relax_pose` returns it plus "conformer", "origin" and "t" (s
    since the repeat began). `frames`: frame -> (symbols, coordinates in A); `confs`: C's conformers as
    (symbols, coordinates, charges or None, energy)."""
    from mepd.network_variants import adjacency

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    symx = list(next(iter(frames.values()))[0])
    n_x = len(symx)
    csym = list(confs[0][0])
    symbols = symx + csym
    held = set(range(n_x)) if not relax_substrate else set(int(i) for i in (reaction_atoms or range(n_x)))
    free = [i for i in range(len(symbols)) if i not in held]
    bonds_c = adjacency(csym, confs[0][1])
    site_atoms = sites(charges) if charges else []
    done = {}
    for f, (_, X) in frames.items():
        X = np.asarray(X, dtype=float)
        bonds_x = adjacency(symx, X)
        start = place(placer, engine, symx, X, charges[f] if charges else None, site_atoms, confs, rng, poses,
                      charge=charge, multiplicity=multiplicity, pool=pool)
        for C in (proposed or {}).get(f, []):
            start.append((0, np.asarray(C, dtype=float), "proposed"))
        done[f] = []
        for ci, C, origin in start:
            try:
                p = relax_pose(engine, symbols, np.vstack([X, C]), free, charge=charge, multiplicity=multiplicity,
                               bonds_x=bonds_x, bonds_c=bonds_c, n_x=n_x, fmax=fmax, steps=steps)
            except Exception as exc:                 # an SCF failure on one pose is one pose lost
                p = {"failed": f"{type(exc).__name__}: {exc}"}
            p.update(conformer=ci, origin=origin, t=time.perf_counter() - t0)
            done[f].append(p)
            if on_pose:
                on_pose(f, p)
    if carry:
        carry_poses(frames, done, engine, symbols, free, bonds_c, n_x, carry, charge=charge,
                    multiplicity=multiplicity, fmax=fmax, steps=steps, t0=t0)
    if polish:
        # The lowest poses relaxed on, five times tighter (a floppy complex stops early at `fmax`, and the
        # deeper basins are reached only then), and the best of them carried once more, as tightly, so
        # that a basin found at one frame is also tried at the other two.
        for f, (_, X) in frames.items():
            bonds_x = adjacency(symx, np.asarray(X, dtype=float))
            low = sorted([p for p in done[f] if "failed" not in p and _eligible(p)], key=lambda p: p["energy"])
            for src in low[:polish]:
                try:
                    p = relax_pose(engine, symbols, src["xyz"], free, charge=charge, multiplicity=multiplicity,
                                   bonds_x=bonds_x, bonds_c=bonds_c, n_x=n_x, fmax=fmax / 5, steps=steps)
                except Exception as exc:
                    p = {"failed": f"{type(exc).__name__}: {exc}"}
                p.update(conformer=src["conformer"], origin="polished:" + src["origin"], t=time.perf_counter() - t0)
                if "failed" not in p:
                    p["moved"] = max(p["moved"], 2 * UNMOVED)
                done[f].append(p)
        carry_poses(frames, done, engine, symbols, free, bonds_c, n_x, 1, charge=charge, multiplicity=multiplicity,
                    fmax=fmax / 5, steps=steps, t0=t0, placed_only=False)
    return done


def carry_poses(frames, done, engine, symbols, free, bonds_c, n_x, carry, *, charge, multiplicity, fmax, steps,
                t0, upto: Optional[int] = None, placed_only: bool = True) -> dict:
    """The `carry` lowest eligible poses of each frame (of its first `upto` placed ones), relaxed on every
    other frame: C's coordinates are kept in the frame they were found in, the other frame's atoms
    superposed on that frame's. Appended to `done` (origin "carried:<frame>")."""
    from mepd.network_variants import adjacency

    best = {}
    for f, ps in done.items():
        placed = [p for p in ps if "failed" not in p and not (placed_only and p["origin"].startswith("carried"))][:upto]
        best[f] = sorted([p for p in placed if _eligible(p)], key=lambda p: p["energy"])[:carry]
    added = {f: [] for f in done}
    for f, (_, X) in frames.items():
        X = np.asarray(X, dtype=float)
        bonds_x = adjacency(symbols[:n_x], X)
        for g, ps in best.items():
            if g == f:
                continue
            for src in ps:
                R, t = _kabsch(src["xyz"][:n_x], X)
                C = src["xyz"][n_x:] @ R.T + t
                try:
                    p = relax_pose(engine, symbols, np.vstack([X, C]), free, charge=charge, multiplicity=multiplicity,
                                   bonds_x=bonds_x, bonds_c=bonds_c, n_x=n_x, fmax=fmax, steps=steps)
                except Exception as exc:
                    p = {"failed": f"{type(exc).__name__}: {exc}"}
                # A carried pose starts where another frame's relaxation ended: unmoved means it is a minimum.
                p.update(conformer=src["conformer"], origin=f"carried:{g}", t=time.perf_counter() - t0)
                if "failed" not in p:
                    p["moved"] = max(p["moved"], 2 * UNMOVED)
                added[f].append(p)
    for f in done:
        done[f].extend(added[f])
    return added


def lowest(poses: list) -> Optional[dict]:
    ok = [p for p in poses if "failed" not in p and _eligible(p)]
    return min(ok, key=lambda p: p["energy"]) if ok else None


def shifts(binding: dict) -> dict:
    """Barrier, reverse-barrier and reaction-energy shifts from C's binding to each frame (kcal/mol)."""
    out = {}
    if "ts" in binding and "r" in binding:
        out["barrier_shift"] = binding["ts"] - binding["r"]
    if "ts" in binding and "p" in binding:
        out["reverse_barrier_shift"] = binding["ts"] - binding["p"]
    if "r" in binding and "p" in binding:
        out["reaction_shift"] = binding["p"] - binding["r"]
    return out


def molecule_from_smiles(smiles: str, engine, *, n: int = 3, seed: int = 0) -> dict:
    """The `molecule` argument of `bystander` from a SMILES: {"symbols", "charge", "multiplicity", "conformers"}
    (`conformers`: RDKit, minimized with `engine`, lowest first)."""
    nodes = conformers(smiles, engine, n=n, seed=seed)
    st = nodes[0].structure
    return {"symbols": list(st.symbols), "charge": int(st.charge), "multiplicity": int(st.multiplicity),
            "conformers": [_xyz(c).tolist() for c in nodes], "smiles": smiles}


def _level(engine) -> str:
    solv = getattr(engine, "axtb_solvation", None) or getattr(engine, "solvation", None) or {}
    name = getattr(engine, "level", None) or type(engine).__name__
    return f"{name}" + (f" in {solv.get('solvent')} ({solv.get('model', 'alpb')})" if isinstance(solv, dict) and solv.get("solvent") else "")


SPREAD_METHOD = ("the range (largest minus smallest) of the value over the independent repeats, each repeat's "
                 "value from its own lowest complex of each frame; None with one repeat")
FREE_METHOD = ("electronic binding + a quasi-RRHO correction at 1 mol/L without relaxing the complex: the "
               "translation and rotation of C and of the step replaced by the complex's and by the six rigid-body "
               "vibrations of C against the held step (the complex's Hessian projected on them); internal "
               "vibrations taken as unchanged, symmetry numbers 1")


def bystander(frames: dict, molecule: dict, engine, *, temperature: float = 298.15, starts: Sequence[dict] = (),
              seed: int = 0, placer: str = "seeded", poses: int = 6, repeats: int = 3, carry: int = 3,
              relax_substrate: bool = False, reaction_atoms: Optional[Sequence[int]] = None, final_engine=None,
              final_keep: int = 3, free_energy: bool = True, charges: Optional[Callable] = gfn2_charges,
              pool: int = 200, fmax: float = 0.05, steps: int = 300, polish: int = 2) -> dict:
    """The bystander estimate of one step with the molecule C beside it (docs/bystander.md).

    frames          one step in one atom order, coordinates in A: {"symbols", "charge", "multiplicity", "r",
                    "ts" (or None: no barrier shift then), "p"}. Its atoms are held where they are.
    molecule        C: {"symbols", "charge", "multiplicity", "conformers": [coordinates, ...]}, each conformer
                    a minimum at `engine`'s level (`molecule_from_smiles`).
    engine          energies and gradients for the search (and a Hessian for the free-energy estimate); where
                    the network has a solvent, the engine carries it.
    temperature     K, for the free-energy estimate.
    starts          further starting poses, each {"frame": "r" | "ts" | "p", "coords": C's atoms in that
                    frame's own coordinates} (a model's or a user's): relaxed beside the generic ones, never
                    instead of them.
    seed            the repeats use seeds seed, seed + 1, ...
    placer, poses   how C is placed (PLACERS) and how many poses are relaxed per frame and repeat; `carry`:
                    how many of each frame's best poses are also relaxed on the other frames (0: none);
                    `polish`: how many of each frame's lowest poses are then relaxed five times tighter,
                    the best of them carried once more (0: none).
    repeats         independent searches: values are from the lowest complex of each frame over all of them,
                    "spread" from each repeat alone.
    relax_substrate also free the step's atoms that change no bond (`reaction_atoms`: those that do; default
                    the atoms of the bonds that differ between the ends).
    final_engine    a second level (e.g. the same method in implicit solvent): single points on the
                    `final_keep` lowest poses of each frame and repeat, reported under "final".

    Returns (kcal/mol; JSON-serialisable):
      "status"   "ok"; "reacted": in some frame every pose, or a pose below the lowest complex, changed C's or
                 the step's own bonds; "unmoved": no pose of some frame moved on relaxing; "failed": nothing
                 could be relaxed, or no pose stayed in contact. "why" says which for every status but "ok".
      "binding"  per frame {"electronic": E(frame.C) - E(frame) - E(C), "free": the same as a free energy at
                 1 mol/L (None when not estimated), "spread", and about the lowest complex: "contact" (closest
                 pair over its vdW radii), "cross_bonds" (pairs of one atom each within bonding range: a
                 coordinated Lewis acid or a new bond), "origin", "conformer", "poses" (counts)}.
      "barrier_shift", "reaction_shift", "reverse_barrier_shift"   {"electronic", "free", "spread"}:
                 binding(ts) - binding(r), binding(p) - binding(r), binding(ts) - binding(p).
      "complexes" per frame the lowest complex's coordinates (A), the step's atoms first, then C's.
      "molecule_energy_hartree", "step_energies_hartree", "free_method", "spread_method", "level",
      "poses" (relaxations run), "gradient_calls", "wall_s", "placer", "repeats", "version", and "final"
      (binding and shifts at `final_engine`'s level) when asked."""
    from mepd.network_variants import adjacency

    t0 = time.perf_counter()
    order = [f for f in ("r", "ts", "p") if frames.get(f) is not None]
    symx, q_x, mult = list(frames["symbols"]), int(frames.get("charge", 0)), int(frames.get("multiplicity", 1))
    csym, q_c = list(molecule["symbols"]), int(molecule.get("charge", 0))
    n_x, charge = len(symx), q_x + q_c
    X = {f: np.asarray(frames[f], dtype=float).reshape(-1, 3) for f in order}
    empty = {"electronic": None, "free": None, "spread": None}
    out = {"version": VERSION, "status": "ok", "why": None, "binding": {}, "barrier_shift": dict(empty),
           "reaction_shift": dict(empty), "reverse_barrier_shift": dict(empty), "complexes": {},
           "molecule_energy_hartree": None, "free_method": None, "spread_method": SPREAD_METHOD,
           "level": _level(engine), "poses": 0, "gradient_calls": None, "wall_s": 0.0, "placer": placer,
           "repeats": repeats, "seed": seed, "symbols": symx + csym, "charge": charge, "multiplicity": mult}
    calls = {"n": 0}
    own_gradients = engine.compute_gradients

    def counted(chain):
        calls["n"] += 1
        return own_gradients(chain)

    def finish():
        if engine.__dict__.get("compute_gradients") is counted:
            del engine.__dict__["compute_gradients"]
        out["gradient_calls"], out["wall_s"] = calls["n"], time.perf_counter() - t0
        return out

    try:
        engine.compute_gradients = counted
    except Exception:
        calls = {"n": None}
    try:
        x_nodes = {f: _node(symx, X[f], q_x, mult) for f in order}
        c_nodes = [_node(csym, c, q_c, int(molecule.get("multiplicity", 1))) for c in molecule["conformers"]]
        e_c = [_energy(engine, c) for c in c_nodes]
        need_q = placer in ("mixed", "seeded", "screen-ff")
        q_frames = {f: charges(x_nodes[f]) for f in order} if need_q else None
        confs = [(csym, _xyz(c), charges(c) if need_q else None, e) for c, e in zip(c_nodes, e_c)]
        if relax_substrate and reaction_atoms is None:
            a, b = adjacency(symx, X[order[0]]), adjacency(symx, X[order[-1]])
            reaction_atoms = sorted(set(int(i) for i in np.nonzero((a != b).any(axis=1))[0]))
        e_x = {}
        for f in order:
            node = x_nodes[f]
            if relax_substrate:
                from mepd.substituents import relax_group

                node = relax_group(engine, node, [i for i in range(n_x) if i not in set(reaction_atoms)],
                                   fmax=fmax, steps=steps)
            e_x[f] = _energy(engine, node)
        proposed = {}
        for s in starts:
            proposed.setdefault(s["frame"], []).append(np.asarray(s["coords"], dtype=float).reshape(-1, 3))
        runs = [search({f: (symx, X[f]) for f in order}, confs, engine, placer=placer, poses=poses, seed=seed + k,
                       carry=carry, relax_substrate=relax_substrate, reaction_atoms=reaction_atoms,
                       proposed=proposed if k == 0 else None, charges=q_frames, charge=charge, multiplicity=mult,
                       pool=pool, fmax=fmax, steps=steps, polish=polish) for k in range(repeats)]
    except Exception as exc:
        out.update(status="failed", why=f"{type(exc).__name__}: {exc}")
        return finish()
    e_c0 = min(e_c)
    out["molecule_energy_hartree"], out["step_energies_hartree"] = e_c0, e_x
    out["poses"] = sum(len(run[f]) for run in runs for f in order)

    def bind(p, f):
        return (p["energy"] - e_x[f] - e_c0) * HARTREE_TO_KCAL

    problems, binding, best_pose = [], {}, {}
    for f in order:
        allp = [p for run in runs for p in run[f]]
        good = [p for p in allp if "failed" not in p]
        best = lowest(allp)
        counts = {"relaxed": len(allp), "failed": len(allp) - len(good), "reacted": sum(p["reacted"] for p in good),
                  "unmoved": sum(p["moved"] <= UNMOVED for p in good),
                  "out_of_contact": sum(p["contact"] > CONTACT for p in good)}
        name = {"r": "reactant end", "ts": "TS", "p": "product end"}[f]
        out["binding"][f] = {**empty, "poses": counts}
        if best is None:
            if good and counts["unmoved"] == len(good):
                problems.append(("unmoved", f"no pose moved on relaxing at the {name}"))
            elif good and counts["reacted"] == len(good):
                problems.append(("reacted", f"every pose changed bonds at the {name}"))
            elif good and counts["out_of_contact"]:
                problems.append(("failed", f"no pose stayed in contact at the {name}"))
            else:
                problems.append(("failed", f"no pose could be relaxed at the {name}: "
                                 + "; ".join(sorted({p['failed'] for p in allp if 'failed' in p}))[:300]))
            continue
        if any(p["reacted"] and p["energy"] < best["energy"] for p in good):
            problems.append(("reacted", f"a pose that changed bonds lies below the lowest complex at the {name}"))
        binding[f], best_pose[f] = bind(best, f), best
        out["complexes"][f] = best["xyz"].tolist()
        out["binding"][f].update(electronic=binding[f], contact=best["contact"], cross_bonds=best["cross_bonds"],
                                 origin=best["origin"], conformer=best["conformer"], energy_hartree=best["energy"])
    per_repeat = []
    for run in runs:
        b = {f: bind(lowest(run[f]), f) for f in order if lowest(run[f]) is not None}
        per_repeat.append({**shifts(b), **b})

    def spread(key):
        v = [r[key] for r in per_repeat if key in r]
        return float(max(v) - min(v)) if len(v) > 1 and len(v) == len(per_repeat) else None

    for f in binding:
        out["binding"][f]["spread"] = spread(f)
    for key, val in shifts(binding).items():
        out[key] = {"electronic": val, "free": None, "spread": spread(key), "repeats": [r.get(key) for r in per_repeat]}
    if problems:
        rank = ["failed", "unmoved", "reacted"]
        problems.sort(key=lambda t: rank.index(t[0]))
        out["status"], out["why"] = problems[0][0], "; ".join(w for _, w in problems)
    if free_energy and callable(getattr(engine, "compute_hessian", None)):
        g = {}
        for f in binding:
            try:
                fe = binding_free_correction(engine, symx + csym, best_pose[f]["xyz"], n_x, charge=charge,
                                             multiplicity=mult, temperature=temperature)
            except Exception as exc:
                out["binding"][f]["free_error"] = f"{type(exc).__name__}: {exc}"
                continue
            g[f] = binding[f] + fe["g_corr_kcal"]
            out["binding"][f].update(free=g[f], **fe)
        for key, val in shifts(g).items():
            out[key]["free"] = val
        out["free_method"] = FREE_METHOD
    if final_engine is not None:
        try:
            out["final"] = _final(final_engine, x_nodes, c_nodes, runs, order, symx + csym, charge, mult, final_keep)
        except Exception as exc:
            out["final"] = {"error": f"{type(exc).__name__}: {exc}"}
    return finish()


def _final(engine, x_nodes, c_nodes, runs, order, symbols, charge, mult, keep) -> dict:
    """Single points at a second level on the `keep` lowest eligible poses of each frame and repeat (the
    lowest of them at that level is the frame's complex there)."""
    # fresh structures: these were computed at the search's level, and a structure keeps its energy
    fresh = lambda n: _node(n.structure.symbols, _xyz(n), n.structure.charge, n.structure.multiplicity)  # noqa: E731
    e_c = min(_energy(engine, fresh(c)) for c in c_nodes)
    e_x = {f: _energy(engine, fresh(x_nodes[f])) for f in order}
    binding, per_repeat, complexes = {}, [], {}
    for run in runs:
        b = {}
        for f in order:
            ok = sorted([p for p in run[f] if "failed" not in p and _eligible(p)], key=lambda p: p["energy"])[:keep]
            if not ok:
                continue
            es = [_energy(engine, _node(symbols, p["xyz"], charge, mult)) for p in ok]
            k = int(np.argmin(es))
            b[f] = (es[k] - e_x[f] - e_c) * HARTREE_TO_KCAL
            if f not in binding or b[f] < binding[f]:
                binding[f], complexes[f] = b[f], ok[k]["xyz"].tolist()
        per_repeat.append({**shifts(b), **b})

    def spread(key):
        v = [r[key] for r in per_repeat if key in r]
        return float(max(v) - min(v)) if len(v) > 1 and len(v) == len(per_repeat) else None

    out = {"level": _level(engine), "binding": {f: {"electronic": v, "spread": spread(f)} for f, v in binding.items()},
           "complexes": complexes}
    for key, val in shifts(binding).items():
        out[key] = {"electronic": val, "spread": spread(key)}
    return out


# ----------------------------------------------------------------- a network
def _one(args):
    key, frames, molecule, make_engine, make_final, kwargs = args
    try:
        res = bystander(frames, molecule, make_engine(), final_engine=make_final() if make_final else None, **kwargs)
    except Exception as exc:
        res = {"status": "failed", "why": f"{type(exc).__name__}: {exc}"}
    return key, res


def bystander_steps(steps: dict, molecule: dict, make_engine: Callable, *, workers: int = 1,
                    make_final: Optional[Callable] = None, on_result: Optional[Callable] = None, **kwargs) -> dict:
    """`bystander` for many steps, `workers` processes at once (one thread each), started in the order given:
    put the steps that matter first. `steps`: key -> frames; `make_engine` (and `make_final`): picklable
    callables that build the engine in each process (e.g. functools.partial(mepd.engine_from_profile, path)).
    `on_result(key, result)` is called as each step finishes. Returns key -> result, in the order given."""
    jobs = [(k, fr, molecule, make_engine, make_final, kwargs) for k, fr in steps.items()]
    out = {}
    if workers <= 1:
        for job in jobs:
            k, res = _one(job)
            out[k] = res
            if on_result:
                on_result(k, res)
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for fut in as_completed([pool.submit(_one, job) for job in jobs]):
            k, res = fut.result()
            out[k] = res
            if on_result:
                on_result(k, res)
    return {k: out[k] for k in steps}
