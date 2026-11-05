"""Intrinsic bond orbitals (IBOs): a structure's filled orbitals as its bonds
and lone pairs, and how they move along a reaction path.

The canonical orbitals of a molecule (HOMO, HOMO-1, ...) are spread over the
whole system, and any rotation among the filled ones describes the same
wavefunction. Knizia's intrinsic bond orbitals are the rotation in which each
orbital sits on as few atoms as possible: they come out as the bonds and lone
pairs of the Lewis structure, and as three-centre bonds where there is none
(a transition state's forming and breaking bonds). "On an atom" is measured
with intrinsic atomic orbitals (IAOs): a minimal set of atomic orbitals,
polarized by the molecule, that holds the filled orbitals exactly, so the
picture does not depend on the basis set.

Along a reaction path (an IRC) most IBOs barely change. The few that do are
the curly arrows of the mechanism, read off the wavefunction instead of drawn
by hand, and when each one moves tells whether the steps are concerted and
synchronous (Knizia & Klein 2015). Each orbital is followed from frame to
frame by its overlap with the previous frame's orbitals; its change is how far
its charge distribution over the atoms has moved from the first frame's.

How:
  1. The orbitals of the frame (a molden file, as for mepd.orbitals), turned
     into the orientation shown when the calculation ran it turned.
  2. IAOs (Knizia 2013, eq. 1-2): the minimal reference basis is the
     calculation's own valence shells for xtb (a valence-only minimal basis
     already: the IAOs are then its orthonormalized atomic orbitals,
     polarization shells such as Cl 3d left out), else MINAO.
  3. IBOs: Jacobi rotations maximizing sum_i sum_A (n_A^i)^p, n_A^i the
     orbital's IAO population on atom A (Knizia 2013, eq. 4; p = EXPONENT).
     Core orbitals (all-electron files) stay out; each spin channel (or, in
     xtb's spin-averaged open shells, the singly occupied ones) is localized
     on its own.

References: IAO and IBO, G. Knizia, J. Chem. Theory Comput. 9, 4834 (2013);
electron flow along IRCs, G. Knizia, J. E. M. N. Klein, Angew. Chem. Int. Ed.
54, 5518 (2015), and the orbital change measure and frame-to-frame matching,
J. E. M. N. Klein, G. Knizia, Angew. Chem. Int. Ed. 57, 11913 (2018) and J.
E. M. N. Klein, G. Knizia, H. S. Rzepa, ChemistryOpen 8, 1244 (2019);
orbital centroids as arrows, G. Vidossich, A. Lledos, Dalton Trans. 43,
11145 (2014). The MINAO reference basis (Knizia's minimal basis from the
cc-pVTZ sets) is the one distributed with PySCF (Apache-2.0),
mepd/data/minao.json.
"""
from __future__ import annotations

import gzip
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from mepd import fragment_orbitals as fo
from mepd import orbitals as orb

# The IBO functional's power. Knizia (2013) used 4 and found 2 (Pipek-Mezey with IAO charges) effectively the
# same except in aromatic rings; IboView's default has been 2 since 2015. Along a path through an aromatic TS
# (Diels-Alder, g-xTB IRC) 4 flipped between two localizations from frame to frame (an orbital's moved share
# 0.28 -> 0.79 -> 0.50 -> 0.75 within 10 frames), 2 changed smoothly; benzene's pi orbitals came out the same.
EXPONENT = 2
GRAD_TOL = 1e-10           # localization: stop when sum of the pair gradients^2 falls below this
MAX_SWEEPS = 200
PAIR_OCC = 1.5             # spin-averaged files: an orbital holding at least this is a pair, less a single electron
SITE_SHARE = 0.9           # an orbital's atoms: the fewest that hold this share of it...
SITE_MIN = 0.1             # ...each holding at least this much
ACTIVE = 0.1               # along a path: an orbital whose charge distribution moved this far (RMS over atoms,
#                            in electrons per orbital electron) at some frame is active -- a curly arrow
STATE_VERSION = 2          # bump when the kept results' layout or meaning changes
_DATA = Path(__file__).parent / "data"


# ------------------------------------------------------------------ elements

def _core_pairs(z: int) -> int:
    """Core orbitals (pairs) below the valence shell: those an all-electron
    calculation has and xtb's valence-only basis does not."""
    for top, core in ((2, 0), (10, 1), (18, 5), (30, 9), (36, 14), (48, 18), (54, 23), (80, 34), (86, 39)):
        if z <= top:
            return core
    return 43


def _valence_ls(z: int) -> tuple:
    """The angular momenta of an element's minimal valence shells."""
    if z <= 2 or z in (3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88):
        return (0,)
    if 21 <= z <= 30 or 39 <= z <= 48 or 57 <= z <= 80:
        return (0, 2)
    return (0, 1)


@lru_cache(maxsize=1)
def _minao() -> dict:
    return json.loads((_DATA / "minao.json").read_text())


# ------------------------------------------------------------------ turning a file's orbitals

def _monomials(d: np.ndarray, l: int) -> np.ndarray:
    return np.stack([d[:, 0] ** a * d[:, 1] ** b * d[:, 2] ** c for a, b, c in orb._CART[l]], axis=1)


def _cart_rotation(l: int, R: np.ndarray) -> np.ndarray:
    """T with m_c(d @ R) = sum_c' T[c, c'] m_c'(d) for the Cartesian
    monomials m of degree l (an exact polynomial identity: solved on random
    points)."""
    d = np.random.default_rng(7).normal(size=(4 * len(orb._CART[l]) + 4, 3))
    X, *_ = np.linalg.lstsq(_monomials(d, l), _monomials(d @ R, l), rcond=None)
    return X.T


def turned(mol: dict, method: str, shown_A, align: dict) -> dict:
    """A parsed molden file's orbitals on the geometry shown: `align`
    (mepd.orbitals.find_run_orbitals: (x - from) @ R + to maps the shown
    geometry onto the file's) gives the turn; the atoms move to the shown
    positions and every orbital's coefficients turn with them."""
    R = np.asarray(align["R"], dtype=float)
    bas = fo.basis(mol, method)
    nbf = sum(len(orb._CART[l]) for _, l, _, _, _ in bas)
    M = np.eye(nbf)
    k = 0
    for _, l, _, _, scale in bas:
        n = len(orb._CART[l])
        if l:
            # psi = sum_c C_c scale_c r(.) m_c(d_file), d_file = d_shown @ R
            M[k:k + n, k:k + n] = scale[:, None] * _cart_rotation(l, R) / scale[None, :]
        k += n
    shown = np.asarray(shown_A, dtype=float).reshape(-1, 3) / orb.BOHR
    return {**mol, "atoms": [(z, p) for (z, _), p in zip(mol["atoms"], shown)],
            "mos": [{**m, "coef": np.asarray(m["coef"]) @ M} for m in mol["mos"]]}


# ------------------------------------------------------------------ centroids

def dipoles(b: list) -> np.ndarray:
    """<chi_a| x, y, z |chi_b> (bohr) within one basis (as
    mepd.fragment_orbitals.basis): (3, n, n). x chi_b = (x - B_x) chi_b +
    B_x chi_b, and (x - B_x) raises chi_b's x power by one (the Obara-Saika
    tables taken one order higher)."""
    n = sum(len(orb._CART[l]) for _, l, _, _, _ in b)
    D = np.zeros((3, n, n))
    ia = 0
    for A, la, aa, ca, sa in b:
        ib = 0
        for B, lb, ab, cb, sb in b:
            p = aa[:, None] + ab[None, :]
            mu = aa[:, None] * ab[None, :] / p
            pre = (math.pi / p) ** 1.5 * np.exp(-mu * float(((A - B) ** 2).sum())) * np.outer(ca, cb)
            if np.abs(pre).max() < 1e-14:
                ib += len(orb._CART[lb])
                continue
            P = (aa[:, None, None] * A + ab[None, :, None] * B) / p[..., None]
            tabs = [fo._axis_table(la, lb + 1, P[..., k] - A[k], P[..., k] - B[k], p) for k in range(3)]
            for u, pa in enumerate(orb._CART[la]):
                for v, pb in enumerate(orb._CART[lb]):
                    e = [tabs[k][pa[k], pb[k]] for k in range(3)]
                    for k in range(3):
                        m = tabs[k][pa[k], pb[k] + 1] + B[k] * e[k]
                        D[k, ia + u, ib + v] = (pre * m * e[(k + 1) % 3] * e[(k + 2) % 3]).sum() * sa[u] * sb[v]
            ib += len(orb._CART[lb])
        ia += len(orb._CART[la])
    return D


# ------------------------------------------------------------------ IAOs

def _orth(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """X's columns orthonormalized symmetrically in the metric S."""
    w, V = np.linalg.eigh(X.T @ S @ X)
    return X @ (V / np.sqrt(np.maximum(w, 1e-14))) @ V.T


def iaos(S1: np.ndarray, S12: np.ndarray, S2: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Intrinsic atomic orbitals (Knizia 2013, eq. 1-2) as coefficients in
    the computational basis (S1), one per reference function (S2, cross
    overlap S12), orthonormal, spanning the occupied orbitals C exactly."""
    P12 = np.linalg.solve(S1, S12)
    Ct = _orth(P12 @ np.linalg.solve(S2, S12.T @ C), S1)      # the occupied orbitals, depolarized
    O = C @ (C.T @ S1)
    Ot = Ct @ (Ct.T @ S1)
    one = np.eye(len(S1))
    return _orth((O @ Ot + (one - O) @ (one - Ot)) @ P12, S1)


def _reference(mol: dict, b1: list, S1: np.ndarray, all_electron: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S12, S2, atom of each reference function). Valence-only files: the
    file's own first shell of each minimal valence angular momentum per atom.
    All-electron: MINAO, its d shells as the five real spherical functions."""
    if not all_electron:
        idx, atom_of, k, seen = [], [], 0, set()
        for (atom, l, _), _b in zip(mol["shells"], b1):
            n = len(orb._CART[l])
            if l in _valence_ls(mol["atoms"][atom][0]) and (atom, l) not in seen:
                seen.add((atom, l))
                idx += range(k, k + n)
                atom_of += [atom] * n
            k += n
        idx = np.array(idx)
        return S1[:, idx], S1[np.ix_(idx, idx)], np.array(atom_of)
    data = _minao()
    b2, atom_of, blocks = [], [], []
    for a, (z, xyz) in enumerate(mol["atoms"]):
        sym = orb._SYMBOLS[z] if z < len(orb._SYMBOLS) else str(z)
        if sym not in data:
            raise orb.OrbitalError(f"bond orbitals: no minimal reference basis for {sym}")
        for l, prims in data[sym]:
            alphas = np.array([p[0] for p in prims], dtype=float)
            cs = np.array([p[1] for p in prims], dtype=float) * np.array([orb._prim_norm(x, l, 0, 0) for x in alphas])
            scale = np.array([1.0 / math.sqrt(orb._self_overlap(alphas, cs, *c)) for c in orb._CART[l]])
            b2.append((np.asarray(xyz, dtype=float), l, alphas, cs, scale))
            if l == 2:   # Cartesian xx yy zz xy xz yz -> z2, x2-y2, xy, xz, yz (normalization does not matter here)
                s = 1.0 / scale
                T = np.zeros((6, 5))
                T[:3, 0] = [-s[0], -s[1], 2 * s[2]]
                T[:2, 1] = [s[0], -s[1]]
                T[3, 2] = T[4, 3] = T[5, 4] = 1.0
            elif l > 2:
                raise orb.OrbitalError("bond orbitals: f shells in the reference basis are not supported")
            else:
                T = np.eye(len(orb._CART[l]))
            blocks.append(T)
            atom_of += [a] * T.shape[1]
    nc = sum(t.shape[0] for t in blocks)
    Tsph = np.zeros((nc, sum(t.shape[1] for t in blocks)))
    i = j = 0
    for t in blocks:
        Tsph[i:i + t.shape[0], j:j + t.shape[1]] = t
        i, j = i + t.shape[0], j + t.shape[1]
    S12 = fo.overlap(b1, b2) @ Tsph
    S2 = Tsph.T @ fo.overlap(b2, b2) @ Tsph
    return S12, S2, np.array(atom_of)


# ------------------------------------------------------------------ IBOs

def localize(Q: np.ndarray, atom_of: np.ndarray, p: int = EXPONENT) -> tuple[np.ndarray, int]:
    """Rotate the orthonormal columns of Q (orbitals in IAO coefficients) to
    maximize sum_i sum_A (n_A^i)^p, n_A^i = sum over A's IAOs of Q^2 (Knizia
    2013; the 2x2 rotation angles of his reference code). Returns (U, sweeps):
    the localized orbitals are Q @ U."""
    n = Q.shape[1]
    U = np.eye(n)
    if n < 2:
        return U, 0
    order = np.argsort(atom_of, kind="stable")
    Q = Q[order].copy()
    starts = np.r_[0, np.nonzero(np.diff(atom_of[order]))[0] + 1]
    for sweep in range(1, MAX_SWEEPS + 1):
        grad = 0.0
        for i in range(n):
            qi = Q[:, i]
            for j in range(i):
                qj = Q[:, j]
                Qii = np.add.reduceat(qi * qi, starts)
                Qjj = np.add.reduceat(qj * qj, starts)
                Qij = np.add.reduceat(qi * qj, starts)
                if p == 2:
                    A = float((4 * Qij ** 2 - (Qii - Qjj) ** 2).sum())
                    B = float((4 * Qij * (Qii - Qjj)).sum())
                else:
                    A = float((-Qii ** 4 - Qjj ** 4 + 6 * (Qii ** 2 + Qjj ** 2) * Qij ** 2
                               + Qii ** 3 * Qjj + Qii * Qjj ** 3).sum())
                    B = float((4 * Qij * (Qii ** 3 - Qjj ** 3)).sum())
                grad += B * B
                phi = 0.25 * math.atan2(B, -A)
                if abs(phi) < 1e-14:
                    continue
                c, s = math.cos(phi), math.sin(phi)
                Q[:, i], Q[:, j] = c * qi + s * qj, -s * qi + c * qj
                qi = Q[:, i]
                ui, uj = U[:, i].copy(), U[:, j]
                U[:, i], U[:, j] = c * ui + s * uj, -s * ui + c * uj
        if grad < GRAD_TOL:
            return U, sweep
    return U, MAX_SWEEPS


def functional(Q: np.ndarray, atom_of: np.ndarray, p: int = EXPONENT) -> float:
    """sum_i sum_A (n_A^i)^p of the columns of Q."""
    return float((populations(Q, atom_of, int(atom_of.max()) + 1) ** p).sum())


def populations(Q: np.ndarray, atom_of: np.ndarray, natoms: int) -> np.ndarray:
    """(orbitals, atoms): each orbital's IAO population on each atom (its
    columns of Q squared, summed per atom; each row sums to 1)."""
    out = np.zeros((Q.shape[1], natoms))
    np.add.at(out.T, atom_of, Q ** 2)
    return out


def _spin_sets(mos: list) -> list:
    """[(spin label, electrons per orbital, orbitals sorted by energy)]: the
    sets localized separately (each spin channel; xtb's spin-averaged open
    shells: the pairs and the singly occupied)."""
    beta = [m for m in mos if m["spin"].startswith("b")]
    if beta:
        alpha = [m for m in mos if not m["spin"].startswith("b")]
        return [(lab, 1.0, sorted([m for m in ch if m["occ"] > 0.5], key=lambda m: m["energy"]))
                for lab, ch in (("α", alpha), ("β", beta))]
    occ = sorted([m for m in mos if m["occ"] > 0.5], key=lambda m: m["energy"])
    pairs = [m for m in occ if m["occ"] >= PAIR_OCC]
    single = [m for m in occ if m["occ"] < PAIR_OCC]
    return [("", 2.0, pairs)] + ([("single", 1.0, single)] if single else [])


class BondOrbitals:
    """The IBOs of one geometry: `C` (orbitals x basis functions, in the
    shown orientation), `Q` (in IAOs), populations `pop` (orbitals x atoms),
    per orbital its spin set, electrons and energy (the canonical energies
    averaged with its weights), and IAO partial charges per atom."""

    def __init__(self, text: str, method: str, charge: int = 0, symbols: Optional[list] = None, shown_A=None,
                 align: Optional[dict] = None):
        mol = orb.parse_molden(text)
        if symbols is not None:
            # The elements as given: g-xTB's molden files carry each atom's valence charge where the
            # atomic number belongs (O as 6, C as 4).
            if len(symbols) != len(mol["atoms"]):
                raise orb.OrbitalError("the orbitals belong to another geometry")
            zs = {s.lower(): i for i, s in enumerate(orb._SYMBOLS)}
            mol = {**mol, "atoms": [(zs.get(str(s).lower(), z), p) for s, (z, p) in zip(symbols, mol["atoms"])]}
        if align is not None and shown_A is not None:
            mol = turned(mol, method, shown_A, align)
        self.mol, self.method = mol, method
        Z = np.array([z for z, _ in mol["atoms"]])
        self.natoms = len(Z)
        b1 = fo.basis(mol, method)
        S1 = fo.overlap(b1, b1)
        occ_e = sum(m["occ"] for m in mol["mos"])
        valence = float(sum(z - 2 * _core_pairs(int(z)) for z in Z))
        # Which electrons the file holds: all of them, or only the valence ones (xtb).
        all_electron = abs(Z.sum() - occ_e - charge) < abs(valence - occ_e - charge)
        self.all_electron = bool(all_electron)
        S12, S2, ref_atom = _reference(mol, b1, S1, self.all_electron)
        self.orbitals: list = []
        Cs, Qs, pops = [], [], []
        charges = (Z if self.all_electron else Z - 2 * np.array([_core_pairs(int(z)) for z in Z])).astype(float)
        self.sweeps = 0
        for spin, electrons, mos in _spin_sets(mol["mos"]):
            if not mos:
                continue
            C = np.array([m["coef"] for m in mos]).T
            e = np.array([m["energy"] for m in mos])
            # IAOs from every filled orbital of this spin channel (all of them for a spin-averaged file).
            same = [m for m in mol["mos"] if m["occ"] > 0.5 and (m["spin"][:1] == mos[0]["spin"][:1] or spin in ("", "single"))]
            A = iaos(S1, S12, S2, np.array([m["coef"] for m in same]).T)
            Q = A.T @ S1 @ C
            ncore = sum(_core_pairs(int(z)) for z in Z) if self.all_electron and spin != "single" else 0
            if ncore:
                n_e = populations(Q[:, :ncore], ref_atom, self.natoms) * electrons
                charges -= n_e.sum(axis=0)
            Cv, Qv, ev = C[:, ncore:], Q[:, ncore:], e[ncore:]
            U, sweeps = localize(Qv, ref_atom)
            self.sweeps = max(self.sweeps, sweeps)
            Cl, Ql = Cv @ U, Qv @ U
            pl = populations(Ql, ref_atom, self.natoms)
            charges -= electrons * pl.sum(axis=0)
            en = (U ** 2).T @ ev
            for k in range(Ql.shape[1]):
                self.orbitals.append({"spin": spin, "electrons": electrons, "energy_ev": round(float(en[k]) * orb.EV, 3)})
            Cs.append(Cl.T)
            Qs.append(Ql.T)
            pops.append(pl)
        if not Cs:
            raise orb.OrbitalError("no filled orbitals to localize")
        self.C = np.vstack(Cs)
        self.Q = np.vstack(Qs)
        self.pop = np.vstack(pops)
        self.ref_atom = ref_atom
        # Partial charges: valid when the electrons add up to the total charge.
        self.charges = charges if abs(charges.sum() - charge) < 0.05 else None
        self.xyz_A = np.array([p for _, p in mol["atoms"]]) * orb.BOHR
        # Each orbital's centroid <phi|r|phi> (Vidossich & Lledos): where its electrons sit -- off the
        # atom, into the lone pair's lobe, for a lone pair; along the bond for a bond.
        D = dipoles(b1)
        centres = np.einsum("ia,kab,ib->ik", self.C, D, self.C) * orb.BOHR
        for k, o in enumerate(self.orbitals):
            o["atoms"] = sites(self.pop[k])
            o["centre"] = [round(float(x), 3) for x in centres[k]]

    def summary(self) -> dict:
        return {"method": self.method, "all_electron": self.all_electron, "orbitals": self.orbitals,
                "charges": None if self.charges is None else [round(float(x), 3) for x in self.charges]}

    def cube(self, k: int, max_points: int = orb.MAX_POINTS) -> str:
        """A cube file of orbital k on a grid around the geometry."""
        xyz = np.array([p for _, p in self.mol["atoms"]])
        origin, h, n = orb.grid_for(xyz, max_points=max_points)
        axes = [origin[i] + h * np.arange(n[i]) for i in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        (v,) = orb.evaluate(self.mol, [self.C[k]], pts, orb.D_NORM[self.method], orb.NORMALIZED_PRIMS[self.method])
        return orb.cube_text(f"IBO {k + 1}", self.mol, origin, h, n, v)

    # Kept on disk: what a path and the cubes need, without the molden file.
    def to_json(self) -> dict:
        mol = {"atoms": [[int(z), [float(x) for x in p]] for z, p in self.mol["atoms"]],
               "shells": [[int(a), int(l), [[float(x), float(c)] for x, c in prims]] for a, l, prims in self.mol["shells"]]}
        return {"v": STATE_VERSION, "method": self.method, "mol": mol, "all_electron": self.all_electron,
                "orbitals": self.orbitals, "C": np.round(self.C, 9).tolist(), "Q": np.round(self.Q, 9).tolist(),
                "pop": np.round(self.pop, 6).tolist(), "ref_atom": self.ref_atom.tolist(),
                "charges": None if self.charges is None else self.charges.tolist(), "sweeps": self.sweeps}

    @classmethod
    def from_json(cls, d: dict) -> "BondOrbitals":
        if d.get("v") != STATE_VERSION:
            raise ValueError("kept bond orbitals of another version")
        self = cls.__new__(cls)
        self.method, self.all_electron, self.orbitals = d["method"], d["all_electron"], d["orbitals"]
        self.mol = {"atoms": [(z, np.array(p)) for z, p in d["mol"]["atoms"]],
                    "shells": [(a, l, [tuple(x) for x in prims]) for a, l, prims in d["mol"]["shells"]], "mos": []}
        self.natoms = len(self.mol["atoms"])
        self.C, self.Q, self.pop = np.array(d["C"]), np.array(d["Q"]), np.array(d["pop"])
        self.ref_atom = np.array(d["ref_atom"])
        self.charges = None if d["charges"] is None else np.array(d["charges"])
        self.xyz_A = np.array([p for _, p in self.mol["atoms"]]) * orb.BOHR
        self.sweeps = d.get("sweeps", 0)
        return self

    def save(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(json.dumps(self.to_json()).encode(), 5))
        tmp.replace(fp)

    @classmethod
    def load(cls, fp: Path) -> "BondOrbitals":
        return cls.from_json(json.loads(gzip.decompress(Path(fp).read_bytes())))


def default_pick(B: "BondOrbitals") -> int:
    """The orbital shown first at a single geometry: the most spread out
    when one spans three or more atoms (a transition state's forming and
    breaking bonds), else the highest in energy (the most reactive bond or
    lone pair, as the HOMO would be)."""
    spread = 1.0 / np.maximum((B.pop ** 2).sum(axis=1), 1e-12)     # effective number of atoms
    if spread.max() >= 2.5:
        return int(np.argmax(spread))
    return int(np.argmax([o["energy_ev"] for o in B.orbitals]))


def sites(pop_row: np.ndarray) -> list:
    """[[atom, share]] of the atoms an orbital sits on: the largest shares
    until SITE_SHARE of it is covered (each at least SITE_MIN), at most 4."""
    total = float(pop_row.sum()) or 1.0
    out, acc = [], 0.0
    for a in np.argsort(-pop_row):
        w = float(pop_row[a]) / total
        if out and (acc >= SITE_SHARE or w < SITE_MIN or len(out) >= 4):
            break
        out.append([int(a), round(w, 3)])
        acc += w
    return out


# ------------------------------------------------------------------ along a path

def follow(frames: list) -> dict:
    """Every orbital of the first frame (of `frames`, BondOrbitals or None
    where a frame failed) followed along the path: in each frame the
    orbital it became -- of the same spin set, matched one to one by the
    largest |overlap| with the previous frame's (their IAO coefficients:
    IAOs are the same atomic functions from frame to frame), with the sign
    that keeps the overlap positive. Per orbital: its local index, sign,
    change and charge centre per frame (None where a frame failed), and
    the smallest overlap with which it was matched (low: frames too far
    apart to tell). Its change at frame k is the RMS change of its
    populations over the atoms from the first frame, sqrt(sum_A (n_A(k) -
    n_A(0))^2), which decides whether it is active (ACTIVE); `moved` is the
    share of it that has moved to other atoms, sum_A |n_A(k) - n_A(0)| / 2,
    the number shown."""
    from scipy.optimize import linear_sum_assignment

    first = next((f for f in frames if f is not None), None)
    if first is None:
        raise orb.OrbitalError("no frame has bond orbitals")
    n = len(first.orbitals)
    spins = [o["spin"] for o in first.orbitals]
    nf = len(frames)
    local = [[None] * nf for _ in range(n)]
    sign = [[None] * nf for _ in range(n)]
    quality = [1.0] * n
    prev, prev_idx, prev_sign = None, None, None
    for k, fr in enumerate(frames):
        if fr is None:
            continue
        if [o["spin"] for o in fr.orbitals] != spins or fr.Q.shape != first.Q.shape:
            continue   # a different electronic structure (another number of orbitals): not comparable
        if prev is None:
            idx, sg = list(range(n)), [1] * n
        else:
            idx, sg = [None] * n, [1] * n
            O = prev.Q @ fr.Q.T            # (previous orbitals, this frame's)
            for s in set(spins):
                rows = [i for i in range(n) if spins[i] == s]
                cols = rows                # the same spin set has the same local positions in every frame
                sub = np.abs(O[np.ix_(rows, cols)])
                r, c = linear_sum_assignment(-sub)
                match = {rows[a]: cols[b] for a, b in zip(r, c)}
                for t in range(n):
                    if spins[t] != s:
                        continue
                    pi = prev_idx[t]
                    j = match[pi]
                    idx[t] = j
                    sg[t] = prev_sign[t] * (1 if O[pi, j] >= 0 else -1)
                    quality[t] = min(quality[t], float(abs(O[pi, j])))
        for t in range(n):
            local[t][k], sign[t][k] = idx[t], sg[t]
        prev, prev_idx, prev_sign = fr, idx, sg
    k0 = next(k for k, f in enumerate(frames) if f is not None)
    orbits = []
    for t in range(n):
        p0 = first.pop[local[t][k0]]
        change, moved, centre = [], [], []
        for k, fr in enumerate(frames):
            j = local[t][k]
            if fr is None or j is None:
                change.append(None)
                moved.append(None)
                centre.append(None)
                continue
            d = fr.pop[j] - p0
            change.append(round(float(np.sqrt((d ** 2).sum())), 4))
            moved.append(round(float(0.5 * np.abs(d).sum()), 4))
            centre.append(fr.orbitals[j]["centre"])
        last = max(k for k in range(nf) if local[t][k] is not None)
        orbits.append({"id": t, "spin": spins[t], "electrons": first.orbitals[t]["electrons"],
                       "local": local[t], "sign": sign[t], "change": change, "moved": moved, "centre": centre,
                       "max_change": max(c for c in change if c is not None),
                       "max_moved": max(c for c in moved if c is not None), "matched": round(quality[t], 3),
                       "start": first.orbitals[local[t][k0]]["atoms"], "end": frames[last].orbitals[local[t][last]]["atoms"]})
    active = sorted([o["id"] for o in orbits if o["max_change"] >= ACTIVE], key=lambda t: -orbits[t]["max_change"])
    charges = [None if f is None or f.charges is None else [round(float(x), 3) for x in f.charges] for f in frames]
    return {"orbitals": orbits, "active": active, "charges": charges}
