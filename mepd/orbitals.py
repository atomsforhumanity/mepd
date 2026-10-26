"""Frontier molecular orbitals (HOMO, LUMO) of one geometry, for viewing.

A single point that writes a molden file -- TeraChem (the method and basis
of the session's TeraChem profile), or xtb's `--molden` (g-xTB, or
GFN2-xTB) -- whose orbitals are then evaluated on a grid around the molecule and written as Gaussian
cube files (what 3Dmol.js draws as isosurfaces). Along with each orbital:
its energy and how much of it sits on each atom (|psi|^2 over the grid,
each point counted for its nearest atom), so the orbitals also say where a
molecule gives or takes electrons.

xtb's orbitals come from the semi-empirical method's minimal basis: right
in shape and ordering for what they are used for here (where the HOMO and
LUMO sit, how they change along a path), not a basis-set-converged picture;
TeraChem's are those of its own method and basis.

Orbitals a calculation already has: with $MEPD_ORBITALS_DIR set (the web
app sets it per job), the g-xTB / GFN2-xTB and TeraChem engines keep the
molden file of every SCF they run there (the newest KEEP_RUNS), indexed by
geometry. Asking for a geometry the job has computed -- a TS optimization's
frame as it runs -- then reads its file instead of running anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

BOHR = 0.529177210903       # Angstrom
EV = 27.211386245988        # eV per Hartree
METHODS = ("gxtb", "gfn2", "terachem")
MAX_POINTS = 56 ** 3        # grid points per orbital (a cube of ~2 MB of text, ~0.5 MB gzipped)
LIVE_POINTS = 38 ** 3       # ...when following a running calculation: a third of the points, built ~3x faster
MARGIN_BOHR = 5.0           # grid beyond the outermost atoms
ORBITALS_ENV = "MEPD_ORBITALS_DIR"
KEEP_RUNS = 1000            # molden files kept per job (the newest; ~25 kB each for 20 atoms)
MATCH_A = 2e-3              # same geometry: every coordinate within this (Angstrom)
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
                    # A new orbital at Sym= (xtb) or, without one, at Ene= once the current one has its
                    # coefficients (TeraChem writes no Sym= line).
                    if cur is None or k == "sym" or (k == "ene" and (cur["coef"] or cur["energy"] is not None)):
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


# How each program writes its basis (found by integrating every orbital on a
# grid: each comes out with norm 1 only so). xtb's contraction coefficients
# are for unnormalized primitives, TeraChem's (the Molden convention) for
# normalized ones; GFN2-xTB and TeraChem give each Cartesian component unit
# norm, g-xTB gives all of a shell's the norm of its x^l one.
D_NORM = {"gfn2": "own", "gxtb": "axis", "terachem": "own"}
NORMALIZED_PRIMS = {"gfn2": False, "gxtb": False, "terachem": True}


def _prim_norm(a: float, lx: int, ly: int, lz: int) -> float:
    """The norm of x^lx y^ly z^lz exp(-a r^2)."""
    L = lx + ly + lz
    return (2 * a / math.pi) ** 0.75 * (4 * a) ** (L / 2) / math.sqrt(
        _dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1))


def evaluate(mol: dict, coefs: list, points: np.ndarray, d_norm: str = "own", normalized_prims: bool = False) -> np.ndarray:
    """The orbitals with coefficient vectors `coefs` at `points` (bohr, (n, 3)).
    xtb's contraction coefficients are for unnormalized primitives, TeraChem's
    for normalized ones (`normalized_prims`); each contracted function is
    scaled as `d_norm` says (see D_NORM). Single precision on the grid (plenty
    for drawing), each atom's distances computed once for all its shells, and
    every shell only where it has not died off."""
    C = np.asarray(coefs, dtype=float)
    out = np.zeros((len(C), len(points)), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    # Each shell's first function index, so shells can be taken atom by atom.
    starts, k = [], 0
    for _, l, _ in mol["shells"]:
        starts.append(k)
        k += len(_CART[l])
    by_atom: dict = {}
    for sh, start in zip(mol["shells"], starts):
        by_atom.setdefault(sh[0], []).append((sh, start))
    for atom, shells in by_atom.items():
        if not any(np.any(C[:, start:start + len(_CART[l])]) for (_, l, _), start in shells):
            continue
        d = pts - np.asarray(mol["atoms"][atom][1], dtype=np.float32)
        r2 = np.einsum("ij,ij->i", d, d)
        for (_, l, prims), start in shells:
            comps = _CART[l]
            w_all = C[:, start:start + len(comps)]
            if not np.any(w_all):
                continue
            alphas = np.array([a for a, _ in prims])
            cs = np.array([c for _, c in prims])
            if normalized_prims:   # coefficients of normalized primitives (the s-type norm per exponent)
                cs = cs * np.array([_prim_norm(a, l, 0, 0) for a in alphas])
            keep = np.nonzero(r2 < (40.0 / alphas.min()))[0]   # exp(-40): nothing left beyond this
            if not len(keep):
                continue
            dk, r2k = d[keep], r2[keep]
            radial = np.exp(-np.outer(r2k, alphas.astype(np.float32))) @ cs.astype(np.float32)
            axis = _self_overlap(alphas, cs, l, 0, 0)
            # The angular factor of each component, times its coefficient in each orbital.
            ang = np.zeros((len(C), len(keep)), dtype=np.float32)
            for (lx, ly, lz), w in zip(comps, w_all.T):
                if not np.any(w):
                    continue
                norm = axis if d_norm == "axis" else _self_overlap(alphas, cs, lx, ly, lz)
                part = np.ones(len(keep), dtype=np.float32)
                for axis_i, power in enumerate((lx, ly, lz)):
                    if power:
                        part *= dk[:, axis_i] ** power
                ang += np.outer((w / math.sqrt(norm)).astype(np.float32), part)
            out[:, keep] += ang * radial
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
    for z, p in mol.get("atoms_shown", mol["atoms"]):
        lines.append(f"{z:5d} {float(z):12.6f} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}")
    # One format string per z-row (6 values a line, the cube convention): C-level formatting.
    nz = n[2]
    fmt = "\n".join(" ".join(["%.4e"] * min(6, nz - k)) for k in range(0, nz, 6)) + "\n"
    rows = np.asarray(values, dtype=float).reshape(-1, nz)
    return "\n".join(lines) + "\n" + "".join(fmt % tuple(r) for r in rows)


# ------------------------------------------------------------------ run

def _executable(method: str) -> list[str]:
    if method == "gxtb":
        exe = os.getenv("GXTB_EXECUTABLE") or shutil.which("gxtb")
        if not exe:
            from mepd.programs import gxtb_executable

            exe = gxtb_executable()
        if not exe:
            raise OrbitalError("g-xTB is not available")
        from mepd.programs import standalone_gxtb

        return [exe] + ([] if standalone_gxtb(exe) else ["--gxtb"])
    from mepd.programs import xtb_executable

    exe = xtb_executable()
    if not exe:
        raise OrbitalError("xtb is not available")
    return [exe, "--gfn", "2"]


_TERACHEM = threading.Lock()     # one GPU: one TeraChem orbital run at a time


def _xyz_text(symbols, coords_A) -> str:
    return "\n".join([str(len(symbols)), "mepd orbitals"]
                     + [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(symbols, coords_A)]) + "\n"


def run_terachem_molden(symbols: list, coords_A: np.ndarray, charge: int = 0, multiplicity: int = 1,
                        engine_kwds: Optional[dict] = None, timeout_s: float = 600.0) -> str:
    """A TeraChem single point (`run energy`) with the engine's method, basis
    and keywords (a profile's terachem_engine_kwds); the molden file TeraChem
    writes to its scratch folder."""
    import dataclasses

    from mepd.engines.gxtb import _run_gxtb_process
    from mepd.engines.terachem import TeraChemEngine, _failure

    names = {f.name for f in dataclasses.fields(TeraChemEngine)}
    engine = TeraChemEngine(**{k: v for k, v in (engine_kwds or {}).items() if k in names})
    with _TERACHEM, tempfile.TemporaryDirectory(prefix="mepd-mo-tc-") as tmp:
        tmpd = Path(tmp)
        (tmpd / "geom.xyz").write_text(_xyz_text(symbols, coords_A))
        (tmpd / "tc.in").write_text(engine.tc_input(int(charge), int(multiplicity), False, None)
                                    .replace("run gradient", "run energy"))
        done = _run_gxtb_process([*engine._cmd, "tc.in"], cwd=tmpd, env=os.environ.copy(), timeout_s=timeout_s)
        found = sorted((tmpd / "scr").glob("*.molden"))
        if not found:
            raise OrbitalError(f"TeraChem wrote no orbitals: {_failure(done.stdout + done.stderr)}")
        return found[0].read_text()


def terachem_label(engine_kwds: Optional[dict]) -> str:
    from mepd.engines.terachem import TeraChemEngine

    kw = engine_kwds or {}
    return f"TeraChem {kw.get('method', TeraChemEngine.method)}/{kw.get('basis', TeraChemEngine.basis)}"


def run_molden(symbols: list, coords_A: np.ndarray, charge: int = 0, multiplicity: int = 1,
               method: str = "gxtb", timeout_s: float = 120.0) -> str:
    """A single point with xtb's `--molden`; the molden file's text."""
    from mepd.engines.gxtb import _run_gxtb_process

    if method not in ("gxtb", "gfn2"):
        raise OrbitalError(f"orbitals come from one of {', '.join(METHODS)}, not {method!r}")
    base = _executable(method)
    with tempfile.TemporaryDirectory(prefix="mepd-mo-") as tmp:
        tmpd = Path(tmp)
        (tmpd / "mol.xyz").write_text(_xyz_text(symbols, coords_A))
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
def _cached(key: str, symbols: tuple, coords: tuple, charge: int, multiplicity: int, method: str, extra: int,
            engine_json: str = "{}", max_points: int = MAX_POINTS) -> dict:
    coords_A = np.array(coords).reshape(-1, 3)
    if method == "terachem":
        kwds = json.loads(engine_json)
        text = run_terachem_molden(list(symbols), coords_A, charge, multiplicity, kwds)
        label = terachem_label(kwds)
    else:
        text = run_molden(list(symbols), coords_A, charge, multiplicity, method)
        label = "g-xTB" if method == "gxtb" else "GFN2-xTB"
    out = orbitals_from_molden(text, method, extra, multiplicity, max_points)
    out.update(label=label, source="single point")
    return out


def orbitals_from_molden(text: str, method: str, extra: int = 0, multiplicity: int = 1,
                         max_points: int = MAX_POINTS, shown_A: Optional[np.ndarray] = None,
                         align: Optional[dict] = None) -> dict:
    """The frontier orbitals of a molden file written by `method` (whose basis
    conventions D_NORM / NORMALIZED_PRIMS say how to read it). With
    `shown_A` and `align` (find_run_orbitals): on a grid around the geometry
    as shown, each point evaluated where it lies in the molden file's own
    orientation, so a turned frame gets its orbitals turned with it."""
    mol = parse_molden(text)
    orbs, homo, lumo = _frontier(mol["mos"])
    if shown_A is not None and align is not None:
        shown = np.asarray(shown_A, dtype=float) / BOHR
        mol = {**mol, "atoms_shown": [(z, p) for (z, _), p in zip(mol["atoms"], shown)]}
    xyz = np.array([p for _, p in mol.get("atoms_shown", mol["atoms"])])
    origin, h, n = grid_for(xyz, max_points=max_points)
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
    at = pts
    if "atoms_shown" in mol:   # grid points of the shown frame -> where they are in the molden file's frame
        R = np.asarray(align["R"])
        at = (pts - np.asarray(align["from"]) / BOHR) @ R + np.asarray(align["to"]) / BOHR
    vals = evaluate(mol, [orbs[picks[k]]["coef"] for k in names], at, D_NORM[method], NORMALIZED_PRIMS[method])
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


# ------------------------------------------------------------------ a run's own orbitals

def _parse_xyz(text: str) -> tuple[list, list]:
    lines = text.strip().splitlines()
    n = int(lines[0].split()[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    return [r[0] for r in rows], [[float(x) for x in r[1:4]] for r in rows]


PRUNE_EVERY = 100          # saves between prunings of a folder (each lists the whole folder)
_SAVES: dict = {}
_SAVES_LOCK = threading.Lock()


def _due_for_pruning(d: Path) -> bool:
    """True on a folder's first save in this process and every PRUNE_EVERY-th after: listing and sorting
    ~1000 files after every SCF held up every thread of a parallel search (24 threads: 11 of them were in
    here, 5 running g-xTB). The folder may hold up to PRUNE_EVERY beyond `keep` in between."""
    key = str(d)
    with _SAVES_LOCK:
        n = _SAVES.get(key, 0)
        _SAVES[key] = n + 1
    return n % PRUNE_EVERY == 0


def save_run_orbitals(molden: Path, xyz_text: str, method: str, label: str, where: Optional[Path] = None,
                      keep: Optional[int] = -1, source: str = "run") -> Optional[dict]:
    """Keep a finished SCF's molden file in `where` (default $MEPD_ORBITALS_DIR;
    nothing when neither is set) with its geometry, and drop the oldest beyond
    `keep` (default KEEP_RUNS; None: keep all). Returns the index entry. Never raises: a
    calculation must not fail over a viewer's cache."""
    where = where or os.environ.get(ORBITALS_ENV, "").strip()
    if not where or not molden.is_file():
        return None
    try:
        import time

        d = Path(where)
        d.mkdir(parents=True, exist_ok=True)
        symbols, coords = _parse_xyz(xyz_text)
        name = f"{time.time_ns()}-{os.getpid()}.molden"
        shutil.copyfile(molden, d / name)
        entry = {"file": name, "method": method, "label": label, "source": source, "symbols": symbols,
                 "coords": [[round(x, 5) for x in c] for c in coords]}
        with open(d / "index.jsonl", "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        keep = KEEP_RUNS if keep == -1 else keep
        if keep is not None and _due_for_pruning(d):
            names = sorted(e.name for e in os.scandir(d) if e.name.endswith(".molden"))
            for old in names[:max(0, len(names) - keep)]:
                (d / old).unlink(missing_ok=True)
                (d / old).with_suffix(".json.gz").unlink(missing_ok=True)
        return entry
    except Exception:
        return None


_INDEX: dict = {}   # index.jsonl path -> ((mtime, size), entries)


def _index(fp: Path) -> list:
    try:
        st = fp.stat()
    except OSError:
        return []
    key = (st.st_mtime_ns, st.st_size)
    hit = _INDEX.get(str(fp))
    if hit and hit[0] == key:
        return hit[1]
    entries = []
    for ln in fp.read_text().splitlines():
        try:
            e = json.loads(ln)
            e["xyz"] = np.asarray(e["coords"], dtype=float)
            entries.append(e)
        except Exception:
            continue
    _INDEX[str(fp)] = (key, entries)
    return entries


def _aligned(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """(RMSD after the best rotation and translation, R) with (a - a0) @ R ~ (b - b0) (Kabsch)."""
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    R = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((ac @ R - bc) ** 2).sum(axis=1).mean())), R


def find_run_orbitals(dirs, symbols: list, coords_A, method: Optional[str] = None,
                      rigid: bool = True) -> Optional[dict]:
    """The newest kept SCF (index entry plus its folder) at this geometry in
    any of `dirs` (and of `method`, if given), or None. With `rigid`, also
    one at the same geometry turned and moved (an IRC's frames have their
    rigid motion taken out): then `align` = {"R", "from", "to"} maps the asked
    geometry onto the kept one, (x - from) @ R + to (Angstrom)."""
    xyz = np.asarray(coords_A, dtype=float).reshape(-1, 3)
    best = None
    for d in dirs:
        d = Path(d)
        for e in reversed(_index(d / "index.jsonl")):
            if e["symbols"] != list(symbols) or e["xyz"].shape != xyz.shape or (method and e.get("method") != method):
                continue
            if not (d / e["file"]).is_file():
                continue
            if np.abs(e["xyz"] - xyz).max() <= MATCH_A:
                hit = {**e, "dir": str(d), "align": None}
            elif rigid and len(xyz) > 2:
                rmsd, R = _aligned(xyz, e["xyz"])
                if rmsd > MATCH_A:
                    continue
                hit = {**e, "dir": str(d), "align": {"R": R, "from": xyz.mean(axis=0), "to": e["xyz"].mean(axis=0)}}
            else:
                continue
            if best is None or (best["align"] is not None and hit["align"] is None) or \
                    ((best["align"] is None) == (hit["align"] is None) and e["file"] > best["file"]):
                best = hit
            break
    return best


_PREFETCH = threading.Lock()


def prefetch(run_dir, extra: int = 0, multiplicity: int = 1, newest: int = 2, max_points: int = LIVE_POINTS) -> None:
    """Build (in a background thread) the orbitals of the newest SCFs kept in
    `run_dir` that are not built yet: a running calculation's next frame is
    then ready before the viewer asks for it."""
    def work():
        if not _PREFETCH.acquire(blocking=False):
            return                      # one prefetcher at a time
        try:
            d = Path(run_dir)
            for e in list(reversed(_index(d / "index.jsonl")))[:newest]:
                fp = d / e["file"]
                if fp.is_file() and e.get("method") in METHODS:
                    try:
                        _from_file(str(fp), e["method"], e["label"], int(extra), int(multiplicity), int(max_points))
                    except Exception:
                        pass
        finally:
            _PREFETCH.release()
    threading.Thread(target=work, daemon=True).start()


def built(entry: dict) -> Optional[bytes]:
    """The full-grid orbitals of a kept SCF as stored gzipped JSON (see
    build_stored), or None when not built yet."""
    fp = (Path(entry["dir"]) / entry["file"]).with_suffix(".json.gz")
    try:
        return fp.read_bytes()
    except OSError:
        return None


def build_stored(entry: dict, multiplicity: int = 1) -> Optional[bytes]:
    """Build a kept SCF's orbitals on the full grid once and store them next
    to its molden file (gzipped JSON, what /api/orbitals sends as it is)."""
    import gzip

    done = built(entry)
    if done is not None:
        return done
    out = dict(_from_file(str(Path(entry["dir"]) / entry["file"]), entry["method"], entry["label"], 0, int(multiplicity)))
    out["source"] = entry.get("source", "run")
    data = gzip.compress(json.dumps(out).encode(), 5)
    fp = (Path(entry["dir"]) / entry["file"]).with_suffix(".json.gz")
    tmp = fp.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(fp)
    return data


@lru_cache(maxsize=256)
def _from_file(path: str, method: str, label: str, extra: int, multiplicity: int, max_points: int = MAX_POINTS) -> dict:
    out = orbitals_from_molden(Path(path).read_text(), method, extra, multiplicity, max_points)
    out.update(label=label, source="run")
    return out


@lru_cache(maxsize=512)
def _from_file_turned(path: str, method: str, label: str, extra: int, multiplicity: int, max_points: int,
                      shown: tuple) -> dict:
    """A kept SCF's orbitals for the same geometry turned and moved (e.g. an IRC frame)."""
    shown_A = np.array(shown).reshape(-1, 3)
    entry_xyz = None
    for e in _index(Path(path).parent / "index.jsonl"):
        if e["file"] == Path(path).name:
            entry_xyz = e["xyz"]
    _, R = _aligned(shown_A, entry_xyz)
    align = {"R": R, "from": shown_A.mean(axis=0), "to": entry_xyz.mean(axis=0)}
    out = orbitals_from_molden(Path(path).read_text(), method, extra, multiplicity, max_points, shown_A, align)
    out.update(label=label, source="run")
    return out


def frame_build(run_dir, symbols: list, coords_A, multiplicity: int = 1, max_points: int = LIVE_POINTS) -> Optional[bytes]:
    """A frame's orbitals from a job's kept SCFs, built once and kept on disk
    (`<run_dir>/built/`, gzipped JSON as /api/orbitals sends it): every later
    look is a file read. None when the job never computed this geometry."""
    import gzip

    xyz = np.asarray(coords_A, dtype=float).reshape(-1, 3)
    key = hashlib.sha1(repr((list(symbols), np.round(xyz, 4).tolist(), int(multiplicity), int(max_points))).encode()).hexdigest()
    fp = Path(run_dir) / "built" / f"{key}.json.gz"
    try:
        return fp.read_bytes()
    except OSError:
        pass
    hit = find_run_orbitals([run_dir], symbols, xyz)
    if hit is None or hit["method"] not in METHODS:
        return None
    path = str(Path(hit["dir"]) / hit["file"])
    if hit["align"] is None:
        out = _from_file(path, hit["method"], hit["label"], 0, int(multiplicity), int(max_points))
    else:
        out = _from_file_turned(path, hit["method"], hit["label"], 0, int(multiplicity), int(max_points),
                                tuple(round(float(x), 4) for x in xyz.ravel()))
    data = gzip.compress(json.dumps(out).encode(), 5)
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(fp)
    except OSError:
        pass
    return data


def prebuild_frames(run_dir, frames_xyz: list, multiplicity: int = 1, workers: int = 2) -> None:
    """In the background: build (frame_build) the orbitals of a finished
    job's result frames -- its TS, IRC and path -- so viewing them later is
    only reading files."""
    from concurrent.futures import ThreadPoolExecutor

    def one(text):
        try:
            sym, xyz = _parse_xyz(text)
            frame_build(run_dir, sym, xyz, multiplicity)
        except Exception:
            pass

    def work():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, frames_xyz))
    threading.Thread(target=work, name="mepd-orbital-frames", daemon=True).start()


def frontier_orbitals(symbols: list, coords_A, charge: int = 0, multiplicity: int = 1,
                      method: str = "gxtb", extra: int = 0, engine_kwds: Optional[dict] = None,
                      run_dirs: Optional[list] = None, max_points: int = MAX_POINTS) -> dict:
    """HOMO and LUMO (and `extra` more on each side) of one geometry: each
    with its energy (eV), occupation, share on each atom, and a cube file;
    `label` names the method. A geometry a calculation has already computed
    (its SCF kept in one of `run_dirs`, see save_run_orbitals) is read from
    there (`source` "run"); otherwise a single point is run: `method`
    "terachem" with `engine_kwds` (a profile's terachem_engine_kwds), else xtb.
    Results are cached per geometry (rounded to 1e-4 Angstrom) and method."""
    if method not in METHODS:
        raise OrbitalError(f"orbitals come from one of {', '.join(METHODS)}, not {method!r}")
    if run_dirs:
        hit = find_run_orbitals(run_dirs, symbols, coords_A)
        if hit and hit["method"] in METHODS:
            if hit["align"] is None:
                return _from_file(str(Path(hit["dir"]) / hit["file"]), hit["method"], hit["label"], int(extra),
                                  int(multiplicity), int(max_points))
            shown = tuple(round(float(x), 4) for x in np.asarray(coords_A, dtype=float).ravel())
            return _from_file_turned(str(Path(hit["dir"]) / hit["file"]), hit["method"], hit["label"], int(extra),
                                     int(multiplicity), int(max_points), shown)
    coords = tuple(round(float(x), 4) for x in np.asarray(coords_A, dtype=float).ravel())
    engine_json = json.dumps(engine_kwds or {}, sort_keys=True, default=str) if method == "terachem" else "{}"
    key = hashlib.sha1(repr((tuple(symbols), coords, charge, multiplicity, method, extra, engine_json)).encode()).hexdigest()
    return _cached(key, tuple(symbols), coords, int(charge), int(multiplicity), method, int(extra), engine_json,
                   int(max_points))
