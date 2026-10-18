"""Frontier molecular orbitals (HOMO, LUMO) of one geometry, for viewing.

A single point with xtb's `--molden` (g-xTB, or GFN2-xTB), whose orbitals
are then evaluated on a grid around the molecule and written as Gaussian
cube files (what 3Dmol.js draws as isosurfaces). Along with each orbital:
its energy and how much of it sits on each atom (|psi|^2 over the grid,
each point counted for its nearest atom), so the orbitals also say where a
molecule gives or takes electrons.

The orbitals come from the semi-empirical method's minimal basis: right in
shape and ordering for what they are used for here (where the HOMO and LUMO
sit, how they change along a path), not a basis-set-converged picture.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

BOHR = 0.529177210903       # Angstrom
EV = 27.211386245988        # eV per Hartree
METHODS = ("gxtb", "gfn2")
MAX_POINTS = 56 ** 3        # grid points per orbital (a cube of ~2 MB of text, ~0.5 MB gzipped)
MARGIN_BOHR = 5.0           # grid beyond the outermost atoms
SPACING_BOHR = 0.3          # finest grid spacing
_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
# Molden's Cartesian component order per shell.
_CART = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1)],
}
_SYMBOLS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
            "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"]


class OrbitalError(RuntimeError):
    pass


# ------------------------------------------------------------------ molden

def parse_molden(text: str) -> dict:
    """{atoms: [(Z, xyz_bohr)], shells: [(atom, l, [(alpha, coef)])], mos: [{energy, spin, occ, coef}]}
    from a molden file with Cartesian functions (what xtb writes)."""
    if re.search(r"^\[(5D|5D7F|7F|9G|5D10F)\]", text, re.M | re.I):
        raise OrbitalError("spherical-harmonic molden basis is not supported (xtb writes Cartesian functions)")
    sections = re.split(r"^\[", text, flags=re.M)
    atoms, shells, mos = [], [], []
    for sec in sections:
        head, _, body = sec.partition("]")
        name = head.strip().lower()
        lines = body.splitlines()
        if name == "atoms":
            scale = 1.0 / BOHR if "angs" in lines[0].lower() else 1.0
            for ln in lines[1:]:
                p = ln.split()
                if len(p) >= 6:
                    atoms.append((int(p[2]), np.array([float(x) for x in p[3:6]]) * scale))
        elif name == "gto":
            i, atom = 1, None
            while i < len(lines):
                p = lines[i].split()
                i += 1
                if not p:
                    atom = None
                    continue
                if atom is None and len(p) >= 1 and p[0].isdigit():
                    atom = int(p[0]) - 1
                    continue
                if p[0].lower() in _L:
                    l, nprim = _L[p[0].lower()], int(p[1])
                    prims = []
                    for _ in range(nprim):
                        a, c = (float(x.replace("D", "E")) for x in lines[i].split()[:2])
                        prims.append((a, c))
                        i += 1
                    shells.append((atom, l, prims))
        elif name == "mo":
            cur = None
            for ln in lines:
                if "=" in ln:
                    k, v = (x.strip() for x in ln.split("=", 1))
                    k = k.lower()
                    if k == "sym" or (cur is None):
                        cur = {"energy": None, "spin": "alpha", "occ": 0.0, "coef": {}}
                        mos.append(cur)
                    if k == "ene":
                        cur["energy"] = float(v.replace("D", "E"))
                    elif k == "spin":
                        cur["spin"] = v.lower()
                    elif k == "occup":
                        cur["occ"] = float(v.replace("D", "E"))
                else:
                    p = ln.split()
                    if len(p) == 2 and cur is not None:
                        cur["coef"][int(p[0]) - 1] = float(p[1].replace("D", "E"))
    nbf = sum(len(_CART[l]) for _, l, _ in shells)
    for mo in mos:
        c = np.zeros(nbf)
        for k, v in mo["coef"].items():
            if k < nbf:
                c[k] = v
        mo["coef"] = c
    if not atoms or not shells or not mos:
        raise OrbitalError("the molden file has no atoms, basis or orbitals")
    return {"atoms": atoms, "shells": shells, "mos": mos}


def _dfact(n: int) -> int:
    return 1 if n <= 0 else n * _dfact(n - 2)


def _self_overlap(alphas: np.ndarray, cs: np.ndarray, lx: int, ly: int, lz: int) -> float:
    """<phi|phi> of sum_k c_k x^lx y^ly z^lz exp(-a_k r^2) (unnormalized primitives)."""
    L = lx + ly + lz
    fac = _dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1) / (2 ** L)
    a = alphas[:, None] + alphas[None, :]
    return float((np.outer(cs, cs) * fac * (math.pi / a) ** 1.5 / a ** L).sum())


# How the two programs scale a shell's Cartesian components (found by
# integrating every orbital on a grid: each comes out with norm 1 only so):
# GFN2-xTB gives each component unit norm; g-xTB gives all of them the norm
# of the x^l one.
D_NORM = {"gfn2": "own", "gxtb": "axis"}


def evaluate(mol: dict, coefs: list, points: np.ndarray, d_norm: str = "own") -> np.ndarray:
    """The orbitals with coefficient vectors `coefs` at `points` (bohr, (n, 3)).
    xtb's contraction coefficients are for unnormalized primitives; each
    contracted function is scaled as `d_norm` says (see D_NORM)."""
    out = np.zeros((len(coefs), len(points)))
    C = np.asarray(coefs)
    k = 0
    for atom, l, prims in mol["shells"]:
        centre = mol["atoms"][atom][1]
        d = points - centre
        r2 = np.einsum("ij,ij->i", d, d)
        comps = _CART[l]
        alphas = np.array([a for a, _ in prims])
        cs = np.array([c for _, c in prims])
        keep = r2 < (40.0 / alphas.min())          # exp(-40): nothing left beyond this
        if not keep.any():
            k += len(comps)
            continue
        dk = d[keep]
        radial = np.exp(-np.outer(r2[keep], alphas)) @ cs
        axis = _self_overlap(alphas, cs, l, 0, 0)
        for lx, ly, lz in comps:
            norm = axis if d_norm == "axis" else _self_overlap(alphas, cs, lx, ly, lz)
            phi = radial * (dk[:, 0] ** lx * dk[:, 1] ** ly * dk[:, 2] ** lz) / math.sqrt(norm)
            w = C[:, k]
            if np.any(w):
                out[:, keep] += np.outer(w, phi)
            k += 1
    return out


# ------------------------------------------------------------------ the grid

def grid_for(xyz_bohr: np.ndarray, spacing: float = SPACING_BOHR, margin: float = MARGIN_BOHR,
             max_points: int = MAX_POINTS) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """(origin, spacing, (nx, ny, nz)) of a box around the atoms; the spacing
    grows until the box has at most `max_points` points."""
    lo, hi = xyz_bohr.min(axis=0) - margin, xyz_bohr.max(axis=0) + margin
    span = hi - lo
    h = spacing
    while True:
        n = tuple(int(math.ceil(s / h)) + 1 for s in span)
        if n[0] * n[1] * n[2] <= max_points:
            return lo, h, n
        h *= 1.1


def cube_text(title: str, mol: dict, origin: np.ndarray, h: float, n: tuple, values: np.ndarray) -> str:
    """A Gaussian cube file (bohr) of `values` on the grid, x slowest, z fastest."""
    lines = [f"mepd {title}", "frontier orbital, from xtb --molden",
             f"{len(mol['atoms']):5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}",
             f"{n[0]:5d} {h:12.6f} {0:12.6f} {0:12.6f}",
             f"{n[1]:5d} {0:12.6f} {h:12.6f} {0:12.6f}",
             f"{n[2]:5d} {0:12.6f} {0:12.6f} {h:12.6f}"]
    for z, p in mol["atoms"]:
        lines.append(f"{z:5d} {float(z):12.6f} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}")
    v = values.reshape(n)
    body = []
    for ix in range(n[0]):
        for iy in range(n[1]):
            row = v[ix, iy]
            for s in range(0, n[2], 6):
                body.append(" ".join(f"{x:.4e}" for x in row[s:s + 6]))
    return "\n".join(lines + body) + "\n"


# ------------------------------------------------------------------ run

def _executable(method: str) -> list[str]:
    if method == "gxtb":
        exe = os.getenv("GXTB_EXECUTABLE") or shutil.which("gxtb")
        if not exe:
            from mepd.programs import gxtb_executable

            exe = gxtb_executable()
        if not exe:
            raise OrbitalError("g-xTB is not available")
        return [exe] + ([] if Path(exe).name == "gxtb" else ["--gxtb"])
    from mepd.programs import xtb_executable

    exe = xtb_executable()
    if not exe:
        raise OrbitalError("xtb is not available")
    return [exe, "--gfn", "2"]


def run_molden(symbols: list, coords_A: np.ndarray, charge: int = 0, multiplicity: int = 1,
               method: str = "gxtb", timeout_s: float = 120.0) -> str:
    """A single point with `--molden`; the molden file's text."""
    from mepd.engines.gxtb import _run_gxtb_process

    if method not in METHODS:
        raise OrbitalError(f"orbitals come from one of {', '.join(METHODS)}, not {method!r}")
    base = _executable(method)
    with tempfile.TemporaryDirectory(prefix="mepd-mo-") as tmp:
        tmpd = Path(tmp)
        xyz = "\n".join([str(len(symbols)), "mepd orbitals"]
                        + [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(symbols, coords_A)]) + "\n"
        (tmpd / "mol.xyz").write_text(xyz)
        cmd = [base[0], "mol.xyz", *base[1:], "--molden", "--chrg", str(int(charge))]
        if int(multiplicity) > 1:
            cmd += ["--uhf", str(int(multiplicity) - 1)]
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        done = _run_gxtb_process(cmd, cwd=tmpd, env=env, timeout_s=timeout_s)
        out = tmpd / "molden.input"
        if done.returncode != 0 or not out.exists():
            tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-4:])
            raise OrbitalError(f"{'g-xTB' if method == 'gxtb' else 'GFN2-xTB'} failed on this geometry: {tail}")
        return out.read_text()


def _frontier(mos: list) -> tuple[list, int, int]:
    """(the orbitals of one spin channel, sorted by energy, HOMO index, LUMO
    index). Open shell: xtb writes the orbitals spin-averaged (a SOMO holds
    e.g. 1.0, a degenerate pair 1.5 each); the highest one holding more than
    half an electron is the HOMO (SOMO). A file with separate spin channels: alpha."""
    alpha = [m for m in mos if m["spin"].startswith("a")] or mos
    alpha = sorted(alpha, key=lambda m: m["energy"])
    occ = [i for i, m in enumerate(alpha) if m["occ"] > 0.5]
    if not occ:
        raise OrbitalError("no occupied orbitals")
    homo = occ[-1]
    lumo = homo + 1 if homo + 1 < len(alpha) else None
    return alpha, homo, lumo


@lru_cache(maxsize=64)
def _cached(key: str, symbols: tuple, coords: tuple, charge: int, multiplicity: int, method: str, extra: int) -> dict:
    coords_A = np.array(coords).reshape(-1, 3)
    mol = parse_molden(run_molden(list(symbols), coords_A, charge, multiplicity, method))
    orbs, homo, lumo = _frontier(mol["mos"])
    xyz = np.array([p for _, p in mol["atoms"]])
    origin, h, n = grid_for(xyz)
    axes = [origin[i] + h * np.arange(n[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    picks = {}
    for name, idx in (("homo", homo), ("lumo", lumo)):
        if idx is not None:
            picks[name] = idx
    for j in range(1, extra + 1):
        if homo - j >= 0:
            picks[f"homo-{j}"] = homo - j
        if lumo is not None and lumo + j < len(orbs):
            picks[f"lumo+{j}"] = lumo + j
    names = list(picks)
    vals = evaluate(mol, [orbs[picks[k]]["coef"] for k in names], pts, D_NORM[method])
    # Where each orbital sits: |psi|^2 over the grid, each point for its nearest atom.
    nearest = np.argmin(((pts[:, None, :] - xyz[None, :, :]) ** 2).sum(-1), axis=1) if len(xyz) * len(pts) <= 6e7 else \
        np.concatenate([np.argmin(((c[:, None, :] - xyz[None]) ** 2).sum(-1), axis=1) for c in np.array_split(pts, 16)])
    result = {"method": method, "spacing_bohr": h, "points": int(n[0] * n[1] * n[2]),
              "open_shell": int(multiplicity) > 1, "orbitals": {}}
    for k, v in zip(names, vals):
        dens = v * v
        tot = float(dens.sum()) or 1.0
        share = np.bincount(nearest, weights=dens, minlength=len(xyz)) / tot
        m = orbs[picks[k]]
        result["orbitals"][k] = {
            "energy_ev": round(m["energy"] * EV, 3), "occupation": m["occ"],
            "atoms": [round(float(x), 4) for x in share],
            "norm": round(tot * h ** 3, 3),
            "cube": cube_text(k.upper(), mol, origin, h, n, v),
        }
    if "homo" in result["orbitals"] and "lumo" in result["orbitals"]:
        result["gap_ev"] = round(result["orbitals"]["lumo"]["energy_ev"] - result["orbitals"]["homo"]["energy_ev"], 3)
    return result


def frontier_orbitals(symbols: list, coords_A, charge: int = 0, multiplicity: int = 1,
                      method: str = "gxtb", extra: int = 0) -> dict:
    """HOMO and LUMO (and `extra` more on each side) of one geometry: each
    with its energy (eV), occupation, share on each atom, and a cube file.
    Results are cached per geometry (rounded to 1e-4 Angstrom)."""
    coords = tuple(round(float(x), 4) for x in np.asarray(coords_A, dtype=float).ravel())
    key = hashlib.sha1(repr((tuple(symbols), coords, charge, multiplicity, method, extra)).encode()).hexdigest()
    return _cached(key, tuple(symbols), coords, int(charge), int(multiplicity), method, int(extra))
