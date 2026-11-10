"""Fragment orbitals: why two molecules bond, in frontier-orbital terms.

The orbitals of a whole system (a complex, a TS) are already mixtures of its
molecules'. Frontier MO theory reasons with the molecules' own: one gives
from its HOMO into the other's LUMO, and the interaction grows with their
overlap and shrinks with their energy gap (second order: ~ S^2 / dE,
Klopman-Salem). So: cut the geometry into its molecules (bonding perceived
on a reference geometry -- a path's reactant end, so a TS keeps its
reactants), run each one alone where it sits (its own charge and spin), and
take the overlap integrals of their orbitals analytically from the basis
in the molden files. Degenerate orbitals (hydroxide's lone pairs, an
atom's p set) are one energy level: their overlap is taken over the whole
level (sqrt of the sum of squares, the same for any mix of them), and the
mix drawn is the one that overlaps best. Also how much of the whole system's HOMO and LUMO is
each molecule's frontier orbitals (|<phi_frag|psi_whole>|^2), when the
whole system's orbitals are at hand.

References: frontier orbitals, K. Fukui, T. Yonezawa, H. Shingu, J. Chem.
Phys. 20, 722 (1952); S^2/dE, G. Klopman, J. Am. Chem. Soc. 90, 223 (1968)
and L. Salem, ibid. 90, 543 (1968); orbital symmetry, R. B. Woodward, R.
Hoffmann, Angew. Chem. Int. Ed. 8, 781 (1969); the whole system's orbitals in
fragment orbitals, S. Dapprich, G. Frenking, J. Phys. Chem. 99, 9352 (1995);
overlap integrals, S. Obara, A. Saika, J. Chem. Phys. 84, 3963 (1986).

Overlaps are exact for the basis (Obara-Saika over Cartesian Gaussians, the
same conventions as mepd.orbitals.evaluate: D_NORM, NORMALIZED_PRIMS); each
orbital's <psi|psi> = 1 is the check that a program's file was read right.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import numpy as np

from mepd import orbitals as orb

LEVELS = 3            # occupied and empty energy levels on each side in the overlap tables
DEGENERATE_EV = 0.02  # orbitals this close in energy are one level (any mix of them is as good)
FRONTIER_SHARE = 0.5   # the default levels overlap at least this share of the best pair's overlap
MIN_S = 1e-3         # pairs overlapping less than this at every level are listed last
SINGLY = 1.95          # a frontier level holding less than this per orbital is partly empty (a SOMO)
MIN_GAP_EV = 1.0      # ranking by S^2 / gap: gaps below this (or negative, an anion's) count as this
MAX_FRAGMENTS = 6


# ------------------------------------------------------------------ basis and overlaps

def basis(mol: dict, method: str) -> list:
    """The shells of a parsed molden file as (center bohr, l, alphas,
    contraction coefficients for unnormalized primitives, per-component
    scale) -- each basis function is scale * sum_k c_k x^lx y^ly z^lz
    exp(-a_k r^2), in molden's component order (mepd.orbitals._CART)."""
    out = []
    for atom, l, prims in mol["shells"]:
        alphas = np.array([a for a, _ in prims], dtype=float)
        cs = np.array([c for _, c in prims], dtype=float)
        if orb.NORMALIZED_PRIMS[method]:
            cs = cs * np.array([orb._prim_norm(a, l, 0, 0) for a in alphas])
        axis = orb._self_overlap(alphas, cs, l, 0, 0)
        scale = [1.0 / math.sqrt(axis if orb.D_NORM[method] == "axis" else orb._self_overlap(alphas, cs, *c))
                 for c in orb._CART[l]]
        out.append((np.asarray(mol["atoms"][atom][1], dtype=float), l, alphas, cs, np.array(scale)))
    return out


def _axis_table(la: int, lb: int, PA: np.ndarray, PB: np.ndarray, p: np.ndarray) -> np.ndarray:
    """1D Obara-Saika overlaps E[i, j] (over primitive pairs), without the
    s-type prefactor: E[0, 0] = 1."""
    E = np.zeros((la + 1, lb + 1) + p.shape)
    E[0, 0] = 1.0
    half = 0.5 / p
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                v = PA * E[i - 1, j]
                if i > 1:
                    v = v + (i - 1) * half * E[i - 2, j]
                if j > 0:
                    v = v + j * half * E[i - 1, j - 1]
            else:
                v = PB * E[i, j - 1]
                if j > 1:
                    v = v + (j - 1) * half * E[i, j - 2]
            E[i, j] = v
    return E


def overlap(ba: list, bb: list) -> np.ndarray:
    """<chi_a|chi_b> between two bases (see basis)."""
    na = sum(len(orb._CART[l]) for _, l, _, _, _ in ba)
    nb = sum(len(orb._CART[l]) for _, l, _, _, _ in bb)
    S = np.zeros((na, nb))
    ia = 0
    for A, la, aa, ca, sa in ba:
        ib = 0
        for B, lb, ab, cb, sb in bb:
            p = aa[:, None] + ab[None, :]
            mu = aa[:, None] * ab[None, :] / p
            AB2 = float(((A - B) ** 2).sum())
            pre = (math.pi / p) ** 1.5 * np.exp(-mu * AB2) * np.outer(ca, cb)
            if np.abs(pre).max() < 1e-14:
                ib += len(orb._CART[lb])
                continue
            P = (aa[:, None, None] * A + ab[None, :, None] * B) / p[..., None]
            tabs = [_axis_table(la, lb, P[..., k] - A[k], P[..., k] - B[k], p) for k in range(3)]
            for u, (ax, ay, az) in enumerate(orb._CART[la]):
                for v, (bx, by, bz) in enumerate(orb._CART[lb]):
                    val = (pre * tabs[0][ax, bx] * tabs[1][ay, by] * tabs[2][az, bz]).sum()
                    S[ia + u, ib + v] = val * sa[u] * sb[v]
            ib += len(orb._CART[lb])
        ia += len(orb._CART[la])
    return S


# ------------------------------------------------------------------ fragments

def fragments_of(symbols: list, coords_A, charge: int = 0) -> list[dict]:
    """The molecules of a geometry: [{atoms, charge, multiplicity, formula}],
    bonding perceived from distances, each molecule's charge and spin from
    its Lewis structure (summing to `charge`; neutral ones preferred)."""
    from mepd.discovery.nanoreactor import Labeler, components, perceive_bonds

    xyz = np.asarray(coords_A, dtype=float).reshape(-1, 3)
    bonds = perceive_bonds(list(symbols), xyz)
    mols = components(len(symbols), bonds)
    labels = Labeler(list(symbols), max_charge=max(1, abs(int(charge)))).assign(mols, bonds, int(charge))
    return [{"atoms": list(m), "charge": lab.charge, "multiplicity": lab.multiplicity, "formula": lab.formula}
            for m, lab in zip(mols, labels)]


@lru_cache(maxsize=512)
def _fragment_molden(symbols: tuple, coords: tuple, charge: int, multiplicity: int, method: str,
                     engine_json: str) -> str:
    import json

    xyz = np.array(coords).reshape(-1, 3)
    if method == "terachem":
        return orb.run_terachem_molden(list(symbols), xyz, charge, multiplicity, json.loads(engine_json))
    return orb.run_molden(list(symbols), xyz, charge, multiplicity, method)


class _Side:
    """One molecule's (or the whole system's) orbitals: sorted alpha MOs, HOMO
    index, basis, and energy levels (degenerate orbitals grouped)."""

    def __init__(self, text: str, method: str, spin: str = "alpha"):
        self.mol = orb.parse_molden(text, orb.D_NORM[method])
        try:
            self.orbs, self.homo, self.lumo = orb._frontier(self.mol["mos"], spin)
        except orb.OrbitalError:   # no electrons left (a bare cation: Mg2+, Li+, H+; H's beta): empty orbitals only
            alpha = [m for m in self.mol["mos"] if m["spin"].startswith(spin[0])] or self.mol["mos"]
            self.orbs, self.homo, self.lumo = sorted(alpha, key=lambda m: m["energy"]), -1, 0
        self.basis = basis(self.mol, method)
        self.C = np.array([m["coef"] for m in self.orbs])   # (n_mo, n_bf)
        e = np.array([m["energy"] for m in self.orbs]) * orb.EV
        self.occ, self.virt = [], []     # [(name, [orbital indices])], from the frontier outwards
        i = self.homo
        while i >= 0 and len(self.occ) < LEVELS:
            group = [i]
            while i - 1 >= 0 and e[group[0]] - e[i - 1] < DEGENERATE_EV:
                i -= 1
                group.append(i)
            self.occ.append(("homo" if group[0] == self.homo else f"homo-{self.homo - group[0]}", group))
            i -= 1
        i = self.lumo
        while i is not None and i < len(self.orbs) and len(self.virt) < LEVELS:
            group = [i]
            while i + 1 < len(self.orbs) and e[i + 1] - e[group[0]] < DEGENERATE_EV:
                i += 1
                group.append(i)
            self.virt.append(("lumo" if group[0] == self.lumo else f"lumo+{group[0] - self.lumo}", group))
            i += 1
        # A radical's singly occupied level (xtb writes open shells spin-averaged: 1.0, or 1.5 each in a
        # degenerate pair) can give an electron and take one: it is on both sides, as "somo". Two radicals
        # bond by pairing their SOMOs, which a HOMO -> LUMO picture would miss entirely.
        if self.occ and min(self.orbs[k]["occ"] for k in self.occ[0][1]) < SINGLY:
            self.occ[0] = ("somo", self.occ[0][1])
            self.virt.insert(0, ("somo", self.occ[0][1]))
        self.levels = dict(self.occ + self.virt)

    def energy_ev(self, name: str) -> float:
        return self.orbs[self.levels[name][0]]["energy"] * orb.EV


def _grouped(molecules: list, groups: list, symbols: list) -> list:
    """Molecules merged into fragments: `groups` lists their indices, each
    list one fragment, every molecule in exactly one. A group's charge is its
    molecules' sum; it is a singlet with an even number of electrons, else a
    doublet (low spin: two radicals in one group pair up)."""
    flat = sorted(i for g in groups for i in g)
    if flat != list(range(len(molecules))) or any(not g for g in groups):
        raise orb.OrbitalError("the groups must hold every molecule exactly once")
    Z = {s: i for i, s in enumerate(orb._SYMBOLS)}
    out = []
    for g in groups:
        members = [molecules[int(i)] for i in g]
        atoms = sorted(a for m in members for a in m["atoms"])
        q = sum(m["charge"] for m in members)
        electrons = sum(Z.get(symbols[a], 0) for a in atoms) - q
        out.append({"atoms": atoms, "charge": q, "multiplicity": 1 if electrons % 2 == 0 else 2,
                    "formula": " + ".join(m["formula"] for m in members), "members": [int(i) for i in g]})
    return out


def _block(sd: _Side, sa: _Side, S: np.ndarray, occ: list, virt: list) -> np.ndarray:
    return sd.C[occ] @ S @ sa.C[virt].T


def analyze(symbols: list, coords_A, charge: int = 0, multiplicity: int = 1, method: str = "gxtb",
            engine_kwds: Optional[dict] = None, fragments: Optional[list] = None,
            reference_A=None, whole_molden: Optional[str] = None, whole_method: Optional[str] = None,
            cubes: bool = True,
            groups: Optional[list] = None,
            pick: Optional[dict] = None, max_points: int = orb.MAX_POINTS) -> dict:
    """Frontier-orbital interactions between the molecules of one geometry.

    `fragments`: atom index lists, or {atoms, charge[, multiplicity]} (a
    drawn molecule, e.g. Design's: its charge as given, however close it sits
    to the others); else perceived on `reference_A`, or on the geometry
    itself. `groups`: lists of those molecules' indices, each list
    one fragment (several molecules computed together, e.g. a water dimer as
    one side). Returns
      fragments: [{atoms, formula, charge, multiplicity,
                   levels: [{name, energy_ev, degeneracy, occupied}]}],
      pairs: [{donor, acceptor, occ, virt (the level names of the best
               interaction, by S^2 / gap), s, gap_ev, frontier_gap_ev (HOMO ->
               LUMO; pairs are ordered by it),
               table: [{occ, virt, s, gap_ev}] for every level pair}],
      drawn (with `cubes`): {donor, acceptor, occ, virt, s, occ_cube, virt_cube}
        -- `pick` {donor, acceptor, occ, virt}, else the first pair's best;
        within degenerate levels the mixes that overlap best, signed so the
        overlap is positive,
      whole (with `whole_molden`): {homo, lumo: [{fragment, level, weight}]}."""
    import json

    if method not in orb.METHODS:
        raise orb.OrbitalError(f"orbitals come from one of {', '.join(orb.METHODS)}, not {method!r}")
    xyz = np.asarray(coords_A, dtype=float).reshape(-1, 3)
    if fragments:
        frags = [{"atoms": sorted(int(a) for a in (f["atoms"] if isinstance(f, dict) else f))} for f in fragments]
        labelled = {tuple(f["atoms"]): f for f in fragments_of(symbols, xyz, charge)}
        for f, given in zip(frags, fragments):   # a given cut: its charge as given, else from a Lewis structure
            if isinstance(given, dict) and given.get("charge") is not None:
                from mepd.discovery.nanoreactor import _formula

                Z = {s: i for i, s in enumerate(orb._SYMBOLS)}
                q = int(given["charge"])
                electrons = sum(Z.get(symbols[a], 0) for a in f["atoms"]) - q
                f.update(charge=q, formula=_formula([symbols[a] for a in f["atoms"]]),
                         multiplicity=int(given.get("multiplicity") or (1 if electrons % 2 == 0 else 2)))
                continue
            hit = labelled.get(tuple(f["atoms"]))
            if hit is None:
                from mepd.discovery.nanoreactor import Labeler, perceive_bonds

                lab = Labeler(list(symbols)).assign([tuple(f["atoms"])], perceive_bonds(list(symbols), xyz), None)[0]
                hit = {"charge": lab.charge, "multiplicity": lab.multiplicity, "formula": lab.formula}
            f.update(charge=hit["charge"], multiplicity=hit["multiplicity"], formula=hit["formula"])
    else:
        ref = xyz if reference_A is None else np.asarray(reference_A, dtype=float).reshape(-1, 3)
        frags = fragments_of(symbols, ref, charge)
    molecules = [{k: f[k] for k in ("atoms", "formula", "charge", "multiplicity")} for f in frags]
    if groups and not fragments:
        frags = _grouped(frags, groups, symbols)
    covered = sorted(a for f in frags for a in f["atoms"])
    if covered != list(range(len(symbols))):
        raise orb.OrbitalError("the fragments must split the atoms: each atom in exactly one")
    if len(frags) < 2:
        raise orb.OrbitalError("one molecule: no fragments to compare")
    if len(frags) > MAX_FRAGMENTS:
        raise orb.OrbitalError(f"{len(frags)} molecules: fragment orbitals are compared for up to {MAX_FRAGMENTS}")
    engine_json = json.dumps(engine_kwds or {}, sort_keys=True, default=str) if method == "terachem" else "{}"

    def run(f):
        syms = tuple(symbols[a] for a in f["atoms"])
        coords = tuple(round(float(x), 5) for x in xyz[f["atoms"]].ravel())
        return _Side(_fragment_molden(syms, coords, int(f["charge"]), int(f["multiplicity"]), method, engine_json),
                     method)

    workers = 1 if method == "terachem" else min(len(frags), 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sides = list(pool.map(run, frags))

    out_frags = []
    for f, s in zip(frags, sides):
        out_frags.append({**f, "levels": [
            {"name": name, "energy_ev": round(s.energy_ev(name), 3), "degeneracy": len(idx), "occupied": k == 0}
            for k, group in enumerate((s.occ, s.virt)) for name, idx in group]})

    cross: dict = {}

    def S_of(i, j):
        if (i, j) not in cross:
            cross[(i, j)] = overlap(sides[i].basis, sides[j].basis)
        return cross[(i, j)]

    def score(s, gap):
        return s * s / max(gap, MIN_GAP_EV)

    pairs = []
    for d in range(len(frags)):
        for a in range(len(frags)):
            if a == d or not sides[a].virt or not sides[d].occ:   # (a bare cation only takes)
                continue
            sd, sa = sides[d], sides[a]
            table = []
            for on, oi in sd.occ:
                for vn, vi in sa.virt:
                    s = float(np.linalg.norm(_block(sd, sa, S_of(d, a), oi, vi)))
                    table.append({"occ": on, "virt": vn, "s": round(s, 4),
                                  "gap_ev": round(sa.energy_ev(vn) - sd.energy_ev(on), 3)})
            # The default levels: the frontier-most pair that really overlaps -- the lowest empty level, then
            # the highest filled one, with at least FRONTIER_SHARE of the best overlap in the table. (Not by
            # overlap^2 / gap: a bare cation's levels sit far below a molecule's, every gap is negative, and
            # a minimal basis' many-fold p or d sets then win on overlap alone.)
            top = max(t["s"] for t in table)
            vrank = {n: k for k, (n, _) in enumerate(sa.virt)}
            orank = {n: k for k, (n, _) in enumerate(sd.occ)}
            ranked = sorted(table, key=lambda t: (vrank[t["virt"]], orank[t["occ"]]))
            best = next((t for t in ranked if t["s"] > 0 and t["s"] >= FRONTIER_SHARE * top), table[0])
            pairs.append({"donor": d, "acceptor": a, "occ": best["occ"], "virt": best["virt"], "s": best["s"],
                          "gap_ev": best["gap_ev"], "frontier_gap_ev": table[0]["gap_ev"], "table": table})
    # Which way electrons go: the direction with the smaller HOMO -> LUMO gap (frontier control). Not by
    # S^2 / gap over all levels: a minimal basis' high, many-fold levels (an anion's d set 20+ eV up) would
    # outscore the real donation. Pairs that barely touch (|S| < MIN_S at every level) go last.
    pairs.sort(key=lambda p: (p["s"] < MIN_S, p["frontier_gap_ev"], -score(p["s"], p["gap_ev"])))

    # Does the shown geometry split into the same molecules on its own? Not where they have bonded (a
    # path's frames near the other end): the "molecules" are then pieces of one, cut apart.
    from mepd.discovery.nanoreactor import components, perceive_bonds

    here = [tuple(m) for m in components(len(symbols), perceive_bonds(list(symbols), xyz))]
    result = {"method": method, "fragments": out_frags, "pairs": pairs,
              "cut": {"same": sorted(here) == sorted(tuple(m["atoms"]) for m in molecules), "here": len(here),
                      "molecules": molecules, "groups": [list(g) for g in groups] if groups and not fragments else None}}
    if whole_molden is not None:
        try:
            result["whole"] = _composition(_Side(whole_molden, whole_method or method), sides)
        except Exception:
            pass
    if cubes and pairs:
        p = dict(pick or {})
        first = pairs[0]
        d, a = int(p.get("donor", first["donor"])), int(p.get("acceptor", first["acceptor"]))
        mine = next((q for q in pairs if q["donor"] == d and q["acceptor"] == a), first)
        d, a = mine["donor"], mine["acceptor"]
        on = p.get("occ") if p.get("occ") in sides[d].levels else mine["occ"]
        vn = p.get("virt") if p.get("virt") in sides[a].levels else mine["virt"]
        result["drawn"] = _drawn(sides, frags, d, a, on, vn, S_of(d, a), xyz, symbols, method, max_points)
    return result


def _drawn(sides, frags, d, a, on, vn, S, xyz_A, symbols, method, max_points) -> dict:
    """Cube files of the donor's level `on` and the acceptor's `vn`, on one grid
    around the whole system (drawn in place, together): within degenerate
    levels the mixes that overlap best (top singular vectors of the overlap
    block), signed so their overlap is positive."""
    sd, sa = sides[d], sides[a]
    oi, vi = sd.levels[on], sa.levels[vn]
    B = _block(sd, sa, S, oi, vi)
    U, sig, Vt = np.linalg.svd(B)
    u, v = (U[:, 0], Vt[0]) if sig[0] > 1e-8 else (np.eye(len(oi))[0], np.eye(len(vi))[0])
    if u @ B @ v < 0:
        v = -v
    occ_c = u @ sd.C[oi]
    virt_c = v @ sa.C[vi]
    Z = {s: i for i, s in enumerate(orb._SYMBOLS)}
    whole = {"atoms": [(Z.get(s, 0), p) for s, p in zip(symbols, xyz_A / orb.BOHR)]}
    origin, h, n = orb.grid_for(xyz_A / orb.BOHR, max_points=max_points)
    axes = [origin[i] + h * np.arange(n[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    (occ_v,) = orb.evaluate(sd.mol, [occ_c], pts, orb.D_NORM[method], orb.NORMALIZED_PRIMS[method])
    (virt_v,) = orb.evaluate(sa.mol, [virt_c], pts, orb.D_NORM[method], orb.NORMALIZED_PRIMS[method])
    name = lambda f, k: f"{frags[f]['formula']} {k.upper()}"   # noqa: E731
    return {"donor": d, "acceptor": a, "occ": on, "virt": vn, "s": round(float(abs(u @ B @ v)), 4),
            "occ_cube": orb.cube_text(name(d, on), whole, origin, h, n, occ_v),
            "virt_cube": orb.cube_text(name(a, vn), whole, origin, h, n, virt_v)}


def _composition(w: _Side, sides: list) -> dict:
    """The whole system's HOMO and LUMO as the fragments' frontier levels:
    weight sum over a level's orbitals of |<phi|psi>|^2 (the fragments'
    orbitals are not orthogonal to each other, so the weights describe, they
    do not split exactly); "other": the rest of each fragment's orbitals."""
    out = {}
    for k in ("homo", "lumo"):
        if k not in w.levels:
            continue
        i = w.levels[k][0]
        rows = []
        for fi, s in enumerate(sides):
            proj = s.C @ overlap(s.basis, w.basis) @ w.C[i]     # <phi_m|psi> for every orbital m of the fragment
            mine = 0.0
            for name, idx in dict(s.occ + s.virt).items():
                wt = float((proj[idx] ** 2).sum())
                mine += wt
                if wt >= 0.02:
                    rows.append({"fragment": fi, "level": name, "weight": round(wt, 3)})
            rest = float((proj ** 2).sum()) - mine
            if rest >= 0.05:
                rows.append({"fragment": fi, "level": "other", "weight": round(rest, 3)})
        out[k] = sorted(rows, key=lambda r: -r["weight"])
    return out


def profile(frames: list, symbols: list, charge: int, multiplicity: int, method: str, fragments: list,
            engine_kwds: Optional[dict] = None) -> list:
    """Along a path: per frame, each donor -> acceptor pair's table of level
    overlaps (the same fragments in every frame); no cube files. A frame that
    fails gives None."""
    out = []
    for coords in frames:
        try:
            r = analyze(symbols, coords, charge, multiplicity, method, engine_kwds, fragments=fragments, cubes=False)
            out.append([{"donor": p["donor"], "acceptor": p["acceptor"],
                         "table": [[t["occ"], t["virt"], t["s"]] for t in p["table"]]} for p in r["pairs"]])
        except Exception:
            out.append(None)
    return out
