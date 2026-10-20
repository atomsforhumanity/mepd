"""Orbital levels of a geometry or a path, for the orbital explorer.

Per frame: the whole system's orbitals around its frontier and those of each
molecule computed alone (mepd.fragment_orbitals), as numbers only:
energies, occupations, how much of each whole orbital each fragment orbital
makes up (|<phi|psi>|^2, an MO interaction diagram's lines), and the overlaps
between the fragments' orbitals. Surfaces are built only for the orbitals
asked for (cube).

Along a path each system's orbitals are followed from frame to frame by their
overlap (analytic, the same integrals), not by energy order: a line in the
correlation diagram keeps the orbital's character through a level crossing,
and carries its sign, so its lobes keep their colours from frame to frame.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from mepd import fragment_orbitals as fo
from mepd import orbitals as orb

WINDOW = 5          # orbitals kept each side of the frontier: HOMO-4 .. HOMO, LUMO .. LUMO+4
MATCH = 8           # ... matched against this many each side of the next frame's frontier
FOLLOW = 0.5        # |overlap| with the next frame's orbital below which a line ends
LINK = 0.03         # fragment -> whole contributions listed from this weight
TOUCH = 0.005       # fragment-fragment overlaps listed from this |S|


def _window(side: fo._Side, half: int) -> list[int]:
    lo = max(0, side.homo - half + 1)
    hi = min(len(side.orbs), side.homo + 1 + half)
    return list(range(lo, hi))


def _orbitals(side: fo._Side, idx: list[int]) -> list[dict]:
    return [{"i": int(i), "rel": int(i - side.homo), "energy_ev": round(side.orbs[i]["energy"] * orb.EV, 3),
             "occ": round(float(side.orbs[i]["occ"]), 3)} for i in idx]


def frame_levels(symbols: list, coords_A, whole_molden: str, fragment_moldens: list, frags: list,
                 method: str) -> dict:
    """The numbers of one frame (see the module docstring), from the molden
    files of the whole system and of each fragment (`frags`: {atoms,
    formula, charge, multiplicity} as fragment_orbitals.fragments_of)."""
    whole = fo._Side(whole_molden, method)
    sides = [fo._Side(t, method) for t in fragment_moldens]
    wi = _window(whole, WINDOW)
    out = {"whole": {"orbitals": _orbitals(whole, wi), "homo": int(whole.homo)}, "fragments": [], "overlaps": []}
    fis = []
    for f, s in zip(frags, sides):
        idx = _window(s, WINDOW)
        fis.append(idx)
        out["fragments"].append({**{k: f[k] for k in ("atoms", "formula", "charge", "multiplicity")},
                                 "orbitals": _orbitals(s, idx), "homo": int(s.homo)})
    # The interaction diagram's lines: each whole orbital in each fragment's orbitals.
    for fi, (s, idx) in enumerate(zip(sides, fis)):
        proj = s.C[idx] @ fo.overlap(s.basis, whole.basis) @ whole.C[wi].T     # (frag orbitals, whole orbitals)
        for a, i in enumerate(idx):
            for b, j in enumerate(wi):
                w = float(proj[a, b] ** 2)
                if w >= LINK:
                    out["whole"]["orbitals"][b].setdefault("from", []).append(
                        {"fragment": fi, "i": int(i), "w": round(w, 3)})
    # Fragment-fragment overlaps (signed, as the orbitals come; the explorer reads |S|).
    for f in range(len(sides)):
        for g in range(f + 1, len(sides)):
            S = sides[f].C[fis[f]] @ fo.overlap(sides[f].basis, sides[g].basis) @ sides[g].C[fis[g]].T
            for a, i in enumerate(fis[f]):
                for b, j in enumerate(fis[g]):
                    if abs(S[a, b]) >= TOUCH:
                        out["overlaps"].append({"a": [f, int(i)], "b": [g, int(j)], "s": round(float(S[a, b]), 4)})
    return out


def _centroid(side: fo._Side) -> np.ndarray:
    return np.mean([p for _, p in side.mol["atoms"]], axis=0)


def _moved(side: fo._Side, onto: fo._Side) -> list:
    """`side`'s basis translated so its atoms' centroid sits on `onto`'s: a
    molecule that only moved (approaching its partner) then overlaps its own
    orbitals of the frame before by ~1."""
    shift = _centroid(onto) - _centroid(side)
    return [(c + shift, l, a, cs, sc) for c, l, a, cs, sc in side.basis]


def follow(sides_by_frame: list) -> list:
    """Lines through a path's frames for one system ([_Side per frame], None
    where a frame failed): [{"points": [[frame, orbital index, sign], ...]}].
    Consecutive frames' orbitals are matched by |overlap| (an assignment,
    within MATCH of the frontier; the system's translation taken out); a line goes on while its match overlaps by
    FOLLOW or more, its sign flipped where the next SCF returned -psi."""
    from scipy.optimize import linear_sum_assignment

    lines: list = []
    live: dict = {}            # orbital index in the previous frame -> line
    prev = None
    for k, side in enumerate(sides_by_frame):
        if side is None:
            live, prev = {}, None
            continue
        shown = set(_window(side, WINDOW))
        nxt: dict = {}
        if prev is not None:
            a, b = _window(prev, MATCH), _window(side, MATCH)
            O = prev.C[a] @ fo.overlap(prev.basis, _moved(side, prev)) @ side.C[b].T
            rows, cols = linear_sum_assignment(-np.abs(O))
            for r, c in zip(rows, cols):
                i, j = a[r], b[c]
                if i in live and abs(O[r, c]) >= FOLLOW:
                    line = live[i]
                    sign = line["points"][-1][2] * (1 if O[r, c] >= 0 else -1)
                    line["points"].append([k, int(j), sign])
                    nxt[j] = line
        for j in sorted(shown):
            if j not in nxt:
                line = {"points": [[k, int(j), 1]]}
                lines.append(line)
                nxt[j] = line
        live, prev = nxt, side
    # Lines that only ran outside the window: dropped (they were matched to keep the shown ones honest).
    return [ln for ln in lines if any(p[1] in set(_window(sides_by_frame[p[0]], WINDOW)) for p in ln["points"])]


def run_frame(symbols: list, coords_A, charge: int, multiplicity: int, method: str, engine_kwds: Optional[dict],
              frags: list, whole_molden: Optional[str] = None) -> tuple[str, list]:
    """The molden files of one frame: the whole system's (given, e.g. the
    calculation's own, or a single point) and each fragment's (alone)."""
    xyz = np.asarray(coords_A, dtype=float).reshape(-1, 3)
    engine_json = json.dumps(engine_kwds or {}, sort_keys=True, default=str) if method == "terachem" else "{}"
    if whole_molden is None:
        if method == "terachem":
            whole_molden = orb.run_terachem_molden(list(symbols), xyz, charge, multiplicity, engine_kwds)
        else:
            whole_molden = orb.run_molden(list(symbols), xyz, charge, multiplicity, method)
    moldens = []
    for f in frags:
        syms = tuple(symbols[a] for a in f["atoms"])
        coords = tuple(round(float(x), 5) for x in xyz[f["atoms"]].ravel())
        moldens.append(fo._fragment_molden(syms, coords, int(f["charge"]), int(f["multiplicity"]), method,
                                           engine_json))
    return whole_molden, moldens


def cut(symbols: list, frames_A: list, charge: int) -> list:
    """The molecules a path is cut into: those of an end holding more than
    one (the first, else the last), the same in every frame; [] when both
    ends are one molecule (only the whole system is shown)."""
    for ref in (frames_A[0], frames_A[-1]):
        frags = fo.fragments_of(symbols, ref, charge)
        if len(frags) > 1:
            return frags if len(frags) <= fo.MAX_FRAGMENTS else []
    return []


def analyze_path(symbols: list, frames_A: list, charge: int, multiplicity: int, method: str,
                 engine_kwds: Optional[dict] = None, wholes: Optional[list] = None, workers: int = 3,
                 keep=None) -> dict:
    """Every frame's numbers and each system's lines through the path.
    `wholes`: per frame, the whole system's molden when a calculation kept
    it (else a single point). `keep(k, whole, fragment_moldens)` is called
    with each frame's files (the web store keeps them for cubes)."""
    frags = cut(symbols, frames_A, charge)

    def one(k):
        try:
            whole, moldens = run_frame(symbols, frames_A[k], charge, multiplicity, method, engine_kwds, frags,
                                       (wholes or [None] * len(frames_A))[k])
            if keep is not None:
                keep(k, whole, moldens)
            return whole, moldens
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=1 if method == "terachem" else workers) as pool:
        files = list(pool.map(one, range(len(frames_A))))
    frames, whole_sides, frag_sides = [], [], [[] for _ in frags]
    for k, got in enumerate(files):
        if got is None:
            frames.append(None)
            whole_sides.append(None)
            for lst in frag_sides:
                lst.append(None)
            continue
        whole, moldens = got
        frames.append(frame_levels(symbols, frames_A[k], whole, moldens, frags, method))
        whole_sides.append(fo._Side(whole, method))
        for lst, t in zip(frag_sides, moldens):
            lst.append(fo._Side(t, method))
    return {"method": method, "fragments": [{k: f[k] for k in ("atoms", "formula", "charge", "multiplicity")}
                                            for f in frags],
            "frames": frames, "lines": {"whole": follow(whole_sides),
                                        "fragments": [follow(s) for s in frag_sides]}}


def cube(molden: str, method: str, index: int, symbols: list, coords_A, sign: int = 1,
         max_points: int = orb.MAX_POINTS) -> str:
    """One orbital (index in the sorted alpha orbitals) as a cube file on a
    grid around the whole geometry `coords_A` (a fragment's orbital drawn in
    place), times `sign`."""
    side = fo._Side(molden, method)
    xyz = np.asarray(coords_A, dtype=float).reshape(-1, 3)
    Z = {s: i for i, s in enumerate(orb._SYMBOLS)}
    whole = {"atoms": [(Z.get(s, 0), p) for s, p in zip(symbols, xyz / orb.BOHR)]}
    origin, h, n = orb.grid_for(xyz / orb.BOHR, max_points=max_points)
    axes = [origin[i] + h * np.arange(n[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    (v,) = orb.evaluate(side.mol, [side.orbs[index]["coef"] * (1 if sign >= 0 else -1)], pts,
                        orb.D_NORM[method], orb.NORMALIZED_PRIMS[method])
    return orb.cube_text(f"orbital {index}", whole, origin, h, n, v)


def pair_overlap(molden_a: str, ia: int, molden_b: str, ib: int, method: str) -> float:
    """<psi_a|psi_b> of two orbitals from two molden files (signed)."""
    a, b = fo._Side(molden_a, method), fo._Side(molden_b, method)
    return float(a.C[ia] @ fo.overlap(a.basis, b.basis) @ b.C[ib])
