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
  "screen-sp"   the same pool scored by single points of `engine`.
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
PLACERS = ("random", "seeded", "screen-ff", "screen-sp")
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
    whole, _ = np.linalg.qr(rigid(np.arange(len(m))))
    both = np.hstack([rigid(x), rigid(c)])
    both = both - whole @ (whole.T @ both)                      # without the complex's own rigid motions
    U, s, _ = np.linalg.svd(both, full_matrices=False)
    B = U[:, s > 1e-6 * s.max()][:, :6]
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
           on_pose: Optional[Callable] = None) -> dict:
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
    return done


def carry_poses(frames, done, engine, symbols, free, bonds_c, n_x, carry, *, charge, multiplicity, fmax, steps,
                t0, upto: Optional[int] = None) -> dict:
    """The `carry` lowest eligible poses of each frame (of its first `upto` placed ones), relaxed on every
    other frame: C's coordinates are kept in the frame they were found in, the other frame's atoms
    superposed on that frame's. Appended to `done` (origin "carried:<frame>")."""
    from mepd.network_variants import adjacency

    best = {}
    for f, ps in done.items():
        placed = [p for p in ps if "failed" not in p and not p["origin"].startswith("carried")][:upto]
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
        out["barrier_shift_kcal"] = binding["ts"] - binding["r"]
    if "ts" in binding and "p" in binding:
        out["reverse_barrier_shift_kcal"] = binding["ts"] - binding["p"]
    if "r" in binding and "p" in binding:
        out["reaction_shift_kcal"] = binding["p"] - binding["r"]
    return out


def _spread(values: list) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"values": values, "std": None, "range": None}
    return {"values": values, "std": float(np.std(v)), "range": float(max(v) - min(v))}


def bystander(frames: dict, molecule, engine, *, reaction_atoms: Optional[Sequence[int]] = None,
              placer: str = "seeded", poses: int = 6, repeats: int = 3, seed: int = 0, carry: int = 1,
              relax_substrate: bool = False, final_engine=None, final_keep: int = 3, free_energy: bool = True,
              temperature: float = 298.15, proposed: Optional[dict] = None, charges: Optional[Callable] = gfn2_charges,
              pool: int = 200, fmax: float = 0.05, steps: int = 300) -> dict:
    """The bystander estimate of one step with the molecule C beside it (docs/bystander.md).

    frames          {"r": node, "ts": node, "p": node} in one atom order ("ts" may be missing: no barrier
                    shift then). Their atoms are held where they are.
    molecule        C: a structure, or several conformers of it (one atom order); `conformers(smiles, engine)`.
    engine          energies and gradients for the search (a Hessian for the free-energy estimate).
    reaction_atoms  the step's atoms that change bonds; only used with `relax_substrate`, which frees the
                    step's other atoms too (default: the atoms of the bonds that differ between the ends).
    placer, poses   how C is placed and how many poses are relaxed per frame and repeat; `carry`: how many
                    of each frame's best poses are also relaxed on the other frames (0: none).
    repeats, seed   independent searches (seeds seed, seed+1, ...): the result is from the lowest complex
                    of each frame over all of them, `spread` from each repeat alone.
    final_engine    a second level (e.g. the same method in implicit solvent): single points on the
                    `final_keep` lowest poses of each frame and repeat, reported under "final".
    proposed        {frame: [C's coordinates in A, ...]}: further starting poses (e.g. a model's), relaxed
                    beside the generic ones, never instead of them.

    Returns a dict: "status" ("ok"; "reacted": in some frame every pose, or a pose below the lowest
    complex, changed C's or the step's bonds; "did_not_move": no pose of some frame moved on relaxing;
    "no_contact": none ended in contact; "failed"), "why", per frame under "frames" the lowest complex
    ("xyz" in A, "energy" Eh, "binding_kcal", "binding_free_kcal" at 1 mol/L, "contact", "cross_bonds",
    "origin", "conformer", counts of poses), "barrier_shift_kcal", "reverse_barrier_shift_kcal",
    "reaction_shift_kcal" (electronic, each frame from its lowest complex), the same under "free" and
    "final", "spread" (per quantity: the values of the repeats, their standard deviation and range), and
    "cost" (seconds, relaxations)."""
    from mepd.network_variants import adjacency

    t0 = time.perf_counter()
    confs_in = list(molecule) if isinstance(molecule, (list, tuple)) else [molecule]
    order = [f for f in ("r", "ts", "p") if frames.get(f) is not None]
    any_frame = frames[order[0]]
    symx, q_x, mult = list(any_frame.structure.symbols), int(any_frame.structure.charge), int(any_frame.structure.multiplicity)
    csym, q_c = list(confs_in[0].structure.symbols), int(confs_in[0].structure.charge)
    n_x, charge = len(symx), q_x + q_c
    X = {f: _xyz(frames[f]) for f in order}
    out = {"version": VERSION, "status": "ok", "why": None, "placer": placer, "poses": poses, "repeats": repeats,
           "seed": seed, "carry": carry, "relax_substrate": bool(relax_substrate), "symbols": symx + csym,
           "charge": charge, "multiplicity": mult, "n_step_atoms": n_x, "frames": {}}
    try:
        e_c = [_energy(engine, c) for c in confs_in]
        need_q = placer in ("seeded", "screen-ff")
        q_frames = {f: charges(frames[f]) for f in order} if need_q else None
        confs = [(csym, _xyz(c), charges(c) if need_q else None, e) for c, e in zip(confs_in, e_c)]
        if relax_substrate and reaction_atoms is None:
            a, b = adjacency(symx, X[order[0]]), adjacency(symx, X[order[-1]])
            reaction_atoms = sorted(set(int(i) for i in np.nonzero((a != b).any(axis=1))[0]))
        e_x = {}
        for f in order:
            node = frames[f]
            if relax_substrate:
                from mepd.substituents import relax_group

                node = relax_group(engine, node, [i for i in range(n_x) if i not in set(reaction_atoms)],
                                   fmax=fmax, steps=steps)
            e_x[f] = _energy(engine, node)
        runs = [search({f: (symx, X[f]) for f in order}, confs, engine, placer=placer, poses=poses, seed=seed + k,
                       carry=carry, relax_substrate=relax_substrate, reaction_atoms=reaction_atoms,
                       proposed=proposed if k == 0 else None, charges=q_frames, charge=charge, multiplicity=mult,
                       pool=pool, fmax=fmax, steps=steps) for k in range(repeats)]
    except Exception as exc:
        out.update(status="failed", why=f"{type(exc).__name__}: {exc}")
        return out
    e_c0 = min(e_c)
    out["molecule_energy"], out["step_energies"] = e_c0, e_x

    def bind(p, f):
        return (p["energy"] - e_x[f] - e_c0) * HARTREE_TO_KCAL

    problems, binding, per_repeat = [], {}, []
    for f in order:
        allp = [p for run in runs for p in run[f]]
        good = [p for p in allp if "failed" not in p]
        best = lowest(allp)
        counts = {"relaxed": len(allp), "failed": len(allp) - len(good), "reacted": sum(p["reacted"] for p in good),
                  "unmoved": sum(p["moved"] <= UNMOVED for p in good),
                  "out_of_contact": sum(p["contact"] > CONTACT for p in good)}
        name = {"r": "reactant end", "ts": "TS", "p": "product end"}[f]
        if best is None:
            if good and counts["unmoved"] == len(good):
                problems.append(("did_not_move", f"no pose moved on relaxing at the {name}"))
            elif good and counts["reacted"] == len(good):
                problems.append(("reacted", f"every pose changed bonds at the {name}"))
            elif good and all(p["reacted"] or p["moved"] <= UNMOVED or p["contact"] > CONTACT for p in good) \
                    and counts["out_of_contact"]:
                problems.append(("no_contact", f"no pose stayed in contact at the {name}"))
            else:
                problems.append(("failed", f"no pose could be relaxed at the {name}: "
                                 + "; ".join(sorted({p['failed'] for p in allp if 'failed' in p}))[:300]))
            out["frames"][f] = {"poses": counts}
            continue
        if any(p["reacted"] and p["energy"] < best["energy"] for p in good):
            problems.append(("reacted", f"a pose that changed bonds lies below the lowest complex at the {name}"))
        binding[f] = bind(best, f)
        out["frames"][f] = {"xyz": best["xyz"].tolist(), "energy": best["energy"], "binding_kcal": binding[f],
                            "contact": best["contact"], "cross_bonds": best["cross_bonds"], "origin": best["origin"],
                            "conformer": best["conformer"], "poses": counts}
    for run in runs:
        b = {f: bind(lowest(run[f]), f) for f in order if lowest(run[f]) is not None}
        per_repeat.append({**shifts(b), **{f"binding_{f}_kcal": v for f, v in b.items()}})
    out.update(shifts(binding))
    keys = sorted({k for r in per_repeat for k in r})
    out["spread"] = {k: _spread([r.get(k) for r in per_repeat]) for k in keys}
    if problems:
        rank = ["failed", "did_not_move", "no_contact", "reacted"]
        problems.sort(key=lambda t: rank.index(t[0]))
        out["status"], out["why"] = problems[0][0], "; ".join(w for _, w in problems)
    if free_energy and callable(getattr(engine, "compute_hessian", None)):
        g = {}
        for f in binding:
            try:
                fe = binding_free_correction(engine, symx + csym, np.asarray(out["frames"][f]["xyz"]), n_x,
                                             charge=charge, multiplicity=mult, temperature=temperature)
            except Exception as exc:
                out["frames"][f]["binding_free_error"] = f"{type(exc).__name__}: {exc}"
                continue
            g[f] = binding[f] + fe["g_corr_kcal"]
            out["frames"][f].update(binding_free_kcal=g[f], **fe)
        out["free"] = {"method": "rigid-body modes of C against the held step, quasi-RRHO, 1 mol/L",
                       "temperature": temperature, **shifts(g)}
    if final_engine is not None:
        try:
            out["final"] = _final(final_engine, frames, confs_in, runs, order, symx + csym, charge, mult, final_keep)
        except Exception as exc:
            out["final"] = {"error": f"{type(exc).__name__}: {exc}"}
    out["cost"] = {"wall_s": time.perf_counter() - t0,
                   "relaxations": sum(len(run[f]) for run in runs for f in order)}
    return out


def _final(engine, frames, confs_in, runs, order, symbols, charge, mult, keep) -> dict:
    """Single points at a second level on the `keep` lowest eligible poses of each frame and repeat."""
    e_c = min(_energy(engine, c) for c in confs_in)
    binding, per_repeat, frames_out = {}, [], {}
    for run in runs:
        b = {}
        for f in order:
            ok = sorted([p for p in run[f] if "failed" not in p and _eligible(p)], key=lambda p: p["energy"])[:keep]
            if not ok:
                continue
            e_x = _energy(engine, frames[f])
            es = [_energy(engine, _node(symbols, p["xyz"], charge, mult)) for p in ok]
            k = int(np.argmin(es))
            b[f] = (es[k] - e_x - e_c) * HARTREE_TO_KCAL
            if f not in binding or b[f] < binding[f]:
                binding[f] = b[f]
                frames_out[f] = {"xyz": ok[k]["xyz"].tolist(), "energy": es[k], "binding_kcal": b[f]}
        per_repeat.append({**shifts(b), **{f"binding_{f}_kcal": v for f, v in b.items()}})
    keys = sorted({k for r in per_repeat for k in r})
    return {"frames": frames_out, **shifts(binding),
            "spread": {k: _spread([r.get(k) for r in per_repeat]) for k in keys}}


# ----------------------------------------------------------------- a network
def _one(args):
    key, frames, molecule, make_engine, make_final, kwargs = args
    engine = make_engine()
    try:
        res = bystander(frames, molecule, engine, final_engine=make_final() if make_final else None, **kwargs)
    except Exception as exc:
        res = {"status": "failed", "why": f"{type(exc).__name__}: {exc}"}
    return key, res


def bystander_steps(steps: dict, molecule, make_engine: Callable, *, workers: int = 1,
                    make_final: Optional[Callable] = None, on_result: Optional[Callable] = None, **kwargs) -> dict:
    """`bystander` for many steps, `workers` processes at once, in the order given (put the steps that
    matter first). `steps`: key -> frames; `make_engine` (and `make_final`): picklable callables building
    the engine in each process (e.g. functools.partial(mepd.engine_from_profile, "profile.toml")).
    `on_result(key, result)` is called as each step finishes. Returns key -> result."""
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
        futures = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futures):
            k, res = fut.result()
            out[k] = res
            if on_result:
                on_result(k, res)
    return {k: out[k] for k in steps}
