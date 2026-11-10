"""Orbitals along a reaction path (`mepd orbitals`): how each molecular orbital
near the frontier changes from reactants through the TS to products.

Each frame chosen along the path gets a DFT single point with linus (its
Molden file kept). Each orbital is then followed from frame to frame by its
overlap with the orbitals of the frame before (analytic, translation taken
out), not by energy order, so a line keeps its character through a level
crossing and carries its sign (the lobes keep their colours). Each orbital
on each line is described by where it sits (Mulliken shares per atom) and,
for two atoms, whether it bonds or antibonds them (their overlap
population), so a line reads "lone pair on Cl -> C-Cl bonding".

A few sentences say what happens (`story`): the frontier at the start, the
TS and the end; the orbitals that move most on the way to the TS; filled
orbitals that end up empty (and the reverse); orbitals that change what they
are. Everything is written to <output>/orbitals.json after every frame, so a
page can follow the run; <output>/frames/<k>.molden.gz keeps each frame's
orbitals for drawing (`orbital_cube`).
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from mepd import fragment_orbitals as fo
from mepd import orbitals as orb

METHOD = "linus"
WINDOW = 6          # orbitals shown each side of the frontier: HOMO-5 .. HOMO, LUMO .. LUMO+5
MATCH = 10          # ... matched against this many each side in the next frame
FOLLOW = 0.5        # |overlap| with the next frame's orbital below which a line ends
SHARE = 0.15        # an atom takes part in an orbital from this Mulliken share
LONE = 0.6          # ... and holds it alone from this share (when no partner bonds with it)
BOND_OP = 0.02      # |overlap population| from which two atoms' orbital counts as (anti)bonding
MOVED_EV = 0.5      # orbital energy changes the story mentions, from this many eV
CUBE_POINTS = 48 ** 3


# ------------------------------------------------------------------ frames along the path

def read_frames(text: str) -> tuple[list, list]:
    """(symbols, [coords (natoms, 3) in Angstrom per frame]) of a multi-frame xyz file."""
    lines = text.splitlines()
    frames, symbols, i = [], None, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        syms = [r[0] for r in rows]
        if symbols is None:
            symbols = syms
        elif syms != symbols:
            raise ValueError("the frames of a path must hold the same atoms in the same order")
        frames.append(np.array([[float(x) for x in r[1:4]] for r in rows]))
        i += 2 + n
    if not frames:
        raise ValueError("no frames in the path file")
    return symbols, frames


def arc_lengths(frames: list) -> np.ndarray:
    """Cumulative distance along the path (Angstrom, all atoms' displacements together)."""
    steps = [float(np.linalg.norm(b - a)) for a, b in zip(frames, frames[1:])]
    return np.concatenate([[0.0], np.cumsum(steps)])


def pick_frames(frames: list, count: int, ts_index: Optional[int] = None) -> list[int]:
    """`count` frames spread evenly along the path's length, always with both ends and the TS."""
    n = len(frames)
    if count >= n:
        return list(range(n))
    s = arc_lengths(frames)
    want = np.linspace(0.0, s[-1], max(count, 2))
    picked = {int(np.argmin(np.abs(s - w))) for w in want} | {0, n - 1}
    if ts_index is not None and 0 <= ts_index < n:
        picked.add(int(ts_index))
    return sorted(picked)


# ------------------------------------------------------------------ one frame's orbitals

def linus_molden(symbols: list, coords_A: np.ndarray, charge: int, multiplicity: int, functional: str,
                 basis: str, threads: int = 4, previous=None):
    """A linus single point with its orbitals: (molden text, energy in Eh, the output, to start the next
    frame's SCF from). `previous`: the frame before's output (its orbitals start this SCF)."""
    import linus
    from linus.qccodec import propagate_wfn
    from qcdata import ProgramInput, Structure

    s = Structure(symbols=list(symbols), geometry=(np.asarray(coords_A) / orb.BOHR).tolist(),
                  charge=int(charge), multiplicity=int(multiplicity))
    inp = ProgramInput(program="linus", structure=s, calctype="energy",
                       model={"method": functional, "basis": basis},
                       keywords={"jk": "df", "threads": int(threads), "molden": True})
    if previous is not None:
        inp = propagate_wfn(previous, inp)
    out = linus.compute(inp, raise_exc=False)
    if not out.success:
        raise RuntimeError(str(getattr(out, "traceback", None) or "linus failed")[-400:])
    text = out.results.files.get("molden.input")
    if not text:
        raise RuntimeError("linus wrote no Molden file")
    return text, float(out.results.energy), out


# Covalent radii (Angstrom, Cordero 2008) for "bonded at this frame": within BONDED x their sum.
_RCOV = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05,
         "Cl": 1.02, "Br": 1.20, "I": 1.39, "Li": 1.28, "Na": 1.66, "K": 2.03, "Mg": 1.41, "Al": 1.21, "Se": 1.20}
BONDED = 1.45       # loose enough for a TS's forming and breaking bonds (SN2: C-Cl 2.38 A = 1.34 x)
PI_AXIAL = 0.25     # a bond orbital is pi when its p part on both atoms lies off the bond axis (and it has little s)


class Frame:
    """One frame's orbitals (fragment_orbitals._Side) with each orbital's place: Mulliken shares per atom,
    overlap populations between atoms, and the direction of each atom's p part (sigma or pi)."""

    def __init__(self, molden: str, symbols: list):
        self.side = fo._Side(molden, METHOD)
        self.symbols = symbols
        mol = self.side.mol
        self.atom_of = np.concatenate([[atom] * len(orb._CART[l]) for atom, l, _ in mol["shells"]]).astype(int)
        self.l_of = np.concatenate([[l] * len(orb._CART[l]) for _, l, _ in mol["shells"]]).astype(int)
        self.p_shells, at = [], 0          # (atom, first function) of each p shell: px, py, pz in that order
        for atom, l, _ in mol["shells"]:
            if l == 1:
                self.p_shells.append((atom, at))
            at += len(orb._CART[l])
        self.S = fo.overlap(self.side.basis, self.side.basis)
        self.natoms = len(mol["atoms"])
        self.xyz = np.array([p for _, p in mol["atoms"]]) * orb.BOHR

    def window(self, half: int) -> list[int]:
        side = self.side
        lo = max(0, side.homo - half + 1)
        hi = min(len(side.orbs), side.homo + 1 + half)
        return list(range(lo, hi))

    def bonded(self, a: int, b: int) -> bool:
        r = float(np.linalg.norm(self.xyz[a] - self.xyz[b]))
        return r <= BONDED * (_RCOV.get(self.symbols[a], 1.2) + _RCOV.get(self.symbols[b], 1.2))

    def _axial(self, c: np.ndarray, atom: int, u: np.ndarray) -> float:
        """The share of `atom`'s p part along unit vector `u` (1: all along, sigma-like; 0: all across, pi-like)."""
        T = np.zeros((3, 3))
        for a, start in self.p_shells:
            if a == atom:
                v = c[start:start + 3]
                T += np.outer(v, v)
        tr = float(np.trace(T))
        return float(u @ T @ u) / tr if tr > 1e-12 else 1.0

    def _s_share(self, c: np.ndarray, sc: np.ndarray, atom: int) -> float:
        m = self.atom_of == atom
        tot = float(c[m] @ sc[m])
        return float(c[m & (self.l_of == 0)] @ sc[m & (self.l_of == 0)]) / tot if abs(tot) > 1e-12 else 0.0

    def character(self, i: int) -> dict:
        """Where orbital `i` sits: {kind: bond|antibond (with sym sigma|pi)|lone|lones|atom|spread, atoms
        (0-based), shares}. Two atoms bond or antibond only when bonded at this frame (or forming or
        breaking a bond: BONDED); the occupied out-of-phase combination of two unbonded atoms' lone
        pairs is "lones", not an antibond."""
        c = self.side.C[i]
        sc = self.S @ c
        q = np.bincount(self.atom_of, weights=c * sc, minlength=self.natoms)
        order = [int(a) for a in np.argsort(-q) if q[a] >= SHARE][:4]
        occ = float(self.side.orbs[i]["occ"]) > 0.5
        if not order:
            order = [int(np.argmax(q))]
        if len(order) >= 2:
            a, b = sorted(order[:2])
            if self.bonded(a, b):
                ma, mb = self.atom_of == a, self.atom_of == b
                op = 2.0 * float(c[ma] @ self.S[np.ix_(ma, mb)] @ c[mb])
                if abs(op) >= BOND_OP and q[a] + q[b] >= 0.5:
                    u = self.xyz[b] - self.xyz[a]
                    u = u / (np.linalg.norm(u) or 1.0)
                    heavy = self.symbols[a] != "H" and self.symbols[b] != "H"
                    pi = heavy and max(self._axial(c, a, u), self._axial(c, b, u)) < PI_AXIAL \
                        and max(self._s_share(c, sc, a), self._s_share(c, sc, b)) < 0.1
                    return {"kind": "bond" if op > 0 else "antibond", "sym": "pi" if pi else "sigma", "atoms": [a, b],
                            "shares": [round(float(q[a]), 2), round(float(q[b]), 2)], "op": round(op, 3)}
            elif occ and q[a] >= 0.25 and q[b] >= 0.25 and q[a] + q[b] >= 0.7:
                return {"kind": "lones", "atoms": [a, b], "shares": [round(float(q[a]), 2), round(float(q[b]), 2)]}
        if q[order[0]] >= LONE:
            return {"kind": "lone" if occ else "atom", "atoms": [order[0]], "shares": [round(float(q[order[0]]), 2)]}
        return {"kind": "spread", "atoms": order[:3], "shares": [round(float(q[a]), 2) for a in order[:3]]}


def _moved_basis(frame: Frame, onto: Frame) -> list:
    """`frame`'s basis translated so its atoms' centroid sits on `onto`'s (a path frame may drift)."""
    ca = np.mean([p for _, p in frame.side.mol["atoms"]], axis=0)
    cb = np.mean([p for _, p in onto.side.mol["atoms"]], axis=0)
    return [(c + (cb - ca), l, a, cs, sc) for c, l, a, cs, sc in frame.side.basis]


# ------------------------------------------------------------------ following orbitals along the path

class Follower:
    """Lines through the frames, built one frame at a time (see the module docstring): each line is
    [[frame k, orbital index, sign], ...]; a line goes on while its match overlaps by FOLLOW or more."""

    def __init__(self, window: int = WINDOW):
        self.window, self.lines, self.live, self.prev = window, [], {}, None

    def add(self, k: int, frame: Optional[Frame]) -> None:
        from scipy.optimize import linear_sum_assignment

        if frame is None:
            self.live, self.prev = {}, None
            return
        shown = set(frame.window(self.window))
        nxt: dict = {}
        if self.prev is not None:
            a, b = self.prev.window(MATCH), frame.window(MATCH)
            O = self.prev.side.C[a] @ fo.overlap(self.prev.side.basis, _moved_basis(frame, self.prev)) @ frame.side.C[b].T
            rows, cols = linear_sum_assignment(-np.abs(O))
            for r, c in zip(rows, cols):
                i, j = a[r], b[c]
                if i in self.live and abs(O[r, c]) >= FOLLOW:
                    line = self.live[i]
                    sign = line["points"][-1][2] * (1 if O[r, c] >= 0 else -1)
                    line["points"].append([k, int(j), int(sign)])
                    line["overlaps"].append(round(float(abs(O[r, c])), 3))
                    nxt[j] = line
        for j in sorted(shown):
            if j not in nxt:
                line = {"points": [[k, int(j), 1]], "overlaps": []}
                self.lines.append(line)
                nxt[j] = line
        self.live, self.prev = nxt, frame

    def shown_lines(self, frames: list) -> list:
        """Lines that are shown somewhere (inside some frame's window), each with an id."""
        out = []
        for line in self.lines:
            if any(f is not None and p[1] in set(f.window(self.window)) for p in line["points"]
                   for f in [frames[p[0]]]):
                out.append(line)
        for n, line in enumerate(out):
            line["id"] = n
        return out


# ------------------------------------------------------------------ the analysis, written as it goes

def _level(f: Frame, i: int) -> dict:
    o = f.side.orbs[i]
    return {"i": int(i), "rel": int(i - f.side.homo), "e": round(o["energy"] * orb.EV, 3),
            "occ": round(float(o["occ"]), 3), "what": f.character(i)}


def _label(symbols: list, what: dict) -> str:
    at = [f"{symbols[a]}{a + 1}" for a in what["atoms"]]
    kind = what["kind"]
    if kind in ("bond", "antibond"):
        sym = "π" if what.get("sym") == "pi" else "σ"
        return f"{at[0]}–{at[1]} {sym}{'*' if kind == 'antibond' else ''}"
    if kind == "lones":
        return f"lone-pair combination on {at[0]} and {at[1]}"
    if kind == "lone":
        return f"lone pair on {at[0]}"
    if kind == "atom":
        return f"empty orbital on {at[0]}"
    return f"orbital spread over {', '.join(at)}"


def _name(rel: int) -> str:
    return "HOMO" if rel == 0 else "LUMO" if rel == 1 else f"HOMO−{-rel}" if rel < 0 else f"LUMO+{rel - 1}"


_FUNCTIONALS = {"wb97x-d3bj": "ωB97X-D3BJ", "b3lyp": "B3LYP", "pbe0": "PBE0", "pbe": "PBE", "bp86": "BP86",
                "tpssh": "TPSSh", "r2scan": "r²SCAN", "m06-2x": "M06-2X", "cam-b3lyp": "CAM-B3LYP", "wb97m-v": "ωB97M-V",
                "b97m-v": "B97M-V", "hf": "HF"}


def level_label(functional: str, basis: str) -> str:
    """'wb97x-d3bj', 'def2-svp' -> 'ωB97X-D3BJ/def2-SVP' (as papers write them)."""
    import re

    b = re.sub(r"^def2-(\w+)", lambda m: "def2-" + m.group(1).upper(), basis)
    b = re.sub(r"^(aug-)?cc-p(\w+)", lambda m: (m.group(1) or "") + "cc-p" + m.group(2).upper().replace("V", "V"), b)
    b = re.sub(r"^(\d[\d+-]*)g", lambda m: m.group(1) + "G", b)
    return f"{_FUNCTIONALS.get(functional, functional.upper())}/{b}"


def _short(symbols: list, what: dict) -> str:
    """The diagram's notation: σ C1–Br2, σ* C1–Br2, π* C1–O5, n Cl6, n O2,O5, Cl6 empty, C1·Br2·Cl6."""
    at = [f"{symbols[a]}{a + 1}" for a in what["atoms"]]
    kind, g = what["kind"], "π" if what.get("sym") == "pi" else "σ"
    if kind == "bond":
        return f"{g} {at[0]}–{at[1]}"
    if kind == "antibond":
        return f"{g}* {at[0]}–{at[1]}"
    if kind == "lone":
        return f"n {at[0]}"
    if kind == "lones":
        return f"n {at[0]},{at[1]}"
    if kind == "atom":
        return f"{at[0]} empty"
    return "·".join(at)


def story(doc: dict) -> list:
    """What happens, tersely ({kind, text, detail (the numbers, for a tooltip), lines, frame}), most telling
    first: the HOMO's and the LUMO's journeys (start -> TS -> end), filled orbitals that end up empty,
    orbitals that turn into others, and the largest moves on the way to the TS."""
    frames, lines, sym = doc["frames"], doc["lines"], doc["symbols"]
    done = [k for k, f in enumerate(frames) if f is not None]
    if len(done) < 2:
        return []
    ts = doc.get("ts_frame")
    first, last = done[0], done[-1]
    stops = [(first, "start"), *([(ts, "TS")] if ts is not None and frames[ts] is not None else []), (last, "end")]
    out = []

    def at(line, k):
        return next((p for p in line["points"] if p[0] == k), None)

    def level(k, i):
        return next((o for o in frames[k]["orbitals"] if o["i"] == i), None)

    def line_of(k, i):
        return next((ln["id"] for ln in lines if at(ln, k) and at(ln, k)[1] == i), None)

    # The HOMO and the LUMO, start -> TS -> end: what they are at each stop.
    for rel, name in ((0, "HOMO"), (1, "LUMO")):
        seq = []
        for k, where in stops:
            o = next((o for o in frames[k]["orbitals"] if o["rel"] == rel), None)
            if o is not None:
                seq.append((k, where, o))
        if len(seq) < 2:
            continue
        words = [_short(sym, o["what"]) + (" at the TS" if where == "TS" else "") for k, where, o in seq]
        if len(set(_short(sym, o["what"]) for _, _, o in seq)) == 1:
            text = f"{name}: {words[0].removesuffix(' at the TS')} all along"
        else:
            text = f"{name}: " + " → ".join(words)
        out.append({"kind": "frontier", "frame": ts if ts is not None else first, "text": text,
                    "lines": [x for x in (line_of(k, o["i"]) for k, _, o in seq) if x is not None],
                    "detail": "; ".join(f"{where}: {_label(sym, o['what'])}, {o['e']:.2f} eV" for _, where, o in seq)})
    # Filled orbitals that end up empty (and the reverse): the mark of a reaction hard by symmetry.
    for ln in lines:
        a, b = ln["points"][0], ln["points"][-1]
        la, lb = level(a[0], a[1]), level(b[0], b[1])
        if la is None or lb is None or a[0] == b[0]:
            continue
        if (la["occ"] > 0.5) != (lb["occ"] > 0.5):
            out.append({"kind": "crossing", "lines": [ln["id"]],
                        "text": f"{_short(sym, la['what'])} {'empties' if la['occ'] > 0.5 else 'fills'}: it crosses the "
                                f"gap",
                        "detail": f"{la['e']:.2f} → {lb['e']:.2f} eV. A filled and an empty orbital trade places, the "
                                  f"mark of a reaction that is hard by orbital symmetry."})
    # Orbitals that turn into others (start -> end of a line), among the filled ones and the LUMO, each once.
    seen = set()
    for ln in lines:
        a, b = ln["points"][0], ln["points"][-1]
        la, lb = level(a[0], a[1]), level(b[0], b[1])
        if la is None or lb is None or a[0] == b[0] or max(la["rel"], lb["rel"]) > 1:
            continue
        wa, wb = la["what"], lb["what"]
        key = (_short(sym, wa), _short(sym, wb))
        if key[0] != key[1] and {wa["kind"], wb["kind"]} != {"spread"} and key not in seen:
            seen.add(key)
            same = [x["id"] for x in lines if x["points"][0][0] != x["points"][-1][0]
                    and (lambda p, q: p is not None and q is not None and (_short(sym, p["what"]), _short(sym, q["what"])) == key)(
                        level(*x["points"][0][:2]), level(*x["points"][-1][:2]))]
            out.append({"kind": "change", "lines": same, "frame": b[0],
                        "text": f"{key[0]} becomes {key[1]}" + (f" (×{len(same)})" if len(same) > 1 else ""),
                        "detail": f"{_label(sym, wa)} ({_name(la['rel'])}, {la['e']:.2f} eV) at the start → "
                                  f"{_label(sym, wb)} ({_name(lb['rel'])}, {lb['e']:.2f} eV) "
                                  f"{'at the end' if b[0] == last else f'by frame {b[0] + 1}'}"})
    # The largest moves on the way to the TS (or across the path); degenerate orbitals that move together once.
    k0, k1 = first, (ts if ts is not None and frames[ts] is not None else last)
    to = "the TS" if k1 == ts else "the end"
    moves = []
    for ln in lines:
        pa, pb = at(ln, k0), at(ln, k1)
        if pa and pb:
            la, lb = level(k0, pa[1]), level(k1, pb[1])
            if la and lb and abs(lb["e"] - la["e"]) >= MOVED_EV and max(la["rel"], lb["rel"]) <= 1:
                moves.append({"d": lb["e"] - la["e"], "ids": [ln["id"]], "la": [la], "text": _short(sym, la["what"])})
    groups: list = []
    for m in sorted(moves, key=lambda m: m["la"][0]["e"]):
        g = next((g for g in groups if g["text"] == m["text"] and abs(g["d"] - m["d"]) < 0.3
                  and abs(g["la"][-1]["e"] - m["la"][0]["e"]) < 0.2), None)
        if g:
            g["ids"] += m["ids"]
            g["la"] += m["la"]
        else:
            groups.append(m)
    for g in sorted(groups, key=lambda g: -abs(g["d"]))[:3]:
        n, d = len(g["ids"]), g["d"]
        out.append({"kind": "move", "lines": g["ids"], "frame": k1,
                    "text": f"{g['text']}{f' ×{n}' if n > 1 else ''} {('falls' if d < 0 else 'rises') if n == 1 else ('fall' if d < 0 else 'rise')} {abs(d):.1f} eV "
                            f"to {to}",
                    "detail": ", ".join(f"{_name(x['rel'])} {x['e']:.2f} eV" for x in g["la"]) + f" at the start"})
    return out


def analyze(symbols: list, frames_A: list, charge: int, multiplicity: int, functional: str, basis: str,
            output: Path, *, ts_index: Optional[int] = None, count: int = 41, threads: int = 4,
            window: int = WINDOW, log: Callable[[str], None] = print) -> dict:
    """Run the frames and write <output>/orbitals.json (after every frame) and the frames' Molden files."""
    output = Path(output)
    (output / "frames").mkdir(parents=True, exist_ok=True)
    picked = pick_frames(frames_A, count, ts_index)
    s_all = arc_lengths(frames_A)
    ts_k = picked.index(ts_index) if ts_index is not None and ts_index in picked else None
    doc = {"version": 1, "method": "linus", "level": {"functional": functional, "basis": basis,
                                                      "label": level_label(functional, basis)},
           "symbols": symbols, "charge": int(charge), "multiplicity": int(multiplicity),
           "window": window, "path_frames": len(frames_A), "picked": picked,
           "s": [round(float(s_all[i]), 4) for i in picked], "ts_frame": ts_k,
           "frames": [None] * len(picked), "energies": [None] * len(picked), "lines": [], "story": [],
           "status": "running", "seconds": 0.0}
    follower, parsed, previous, t0 = Follower(window), [None] * len(picked), None, time.time()
    # The frames computed, as one xyz file (what the page draws the orbitals on).
    (output / "frames.xyz").write_text("".join(
        f"{len(symbols)}\npath frame {idx + 1}\n" + "".join(f"{a} {x:.6f} {y:.6f} {z:.6f}\n"
                                                         for a, (x, y, z) in zip(symbols, frames_A[idx]))
        for idx in picked))

    def write():
        doc["seconds"] = round(time.time() - t0, 1)
        tmp = output / "orbitals.json.tmp"
        tmp.write_text(json.dumps(doc))
        tmp.replace(output / "orbitals.json")

    for k, idx in enumerate(picked):
        t = time.time()
        try:
            text, energy, previous = linus_molden(symbols, frames_A[idx], charge, multiplicity, functional, basis,
                                                  threads, previous)
            (output / "frames" / f"{k}.molden.gz").write_bytes(gzip.compress(text.encode(), 6))
            f = Frame(text, symbols)
            parsed[k] = f
            doc["frames"][k] = {"homo": int(f.side.homo), "orbitals": [_level(f, i) for i in f.window(window)]}
            doc["energies"][k] = energy
            log(f"frame {k + 1}/{len(picked)} (path frame {idx + 1}): E = {energy:.6f} Eh, "
                f"HOMO {f.side.orbs[f.side.homo]['energy'] * orb.EV:.2f} eV ({time.time() - t:.1f} s)")
        except Exception as exc:   # one frame failing breaks the lines there, not the run
            previous = None
            log(f"frame {k + 1}/{len(picked)} (path frame {idx + 1}) failed: {exc}")
        follower.add(k, parsed[k])
        doc["lines"] = [{"id": ln.get("id"), "points": ln["points"], "overlaps": ln["overlaps"]}
                        for ln in follower.shown_lines(parsed)]
        write()
    doc["story"] = story(doc)
    doc["status"] = "done"
    write()
    return doc


def orbital_cube(output: Path, k: int, i: int, sign: int = 1, max_points: int = CUBE_POINTS) -> str:
    """Orbital `i` (index in the frame's sorted alpha orbitals) of picked frame `k` as cube text, times `sign`."""
    text = gzip.decompress((Path(output) / "frames" / f"{int(k)}.molden.gz").read_bytes()).decode()
    side = fo._Side(text, METHOD)
    xyz = np.array([p for _, p in side.mol["atoms"]])
    origin, h, n = orb.grid_for(xyz, max_points=max_points)
    axes = [origin[a] + h * np.arange(n[a]) for a in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    (v,) = orb.evaluate(side.mol, [side.orbs[int(i)]["coef"] * (1 if sign >= 0 else -1)], pts,
                        orb.D_NORM[METHOD], orb.NORMALIZED_PRIMS[METHOD])
    return orb.cube_text(f"orbital {i}", side.mol, origin, h, n, v)


__all__ = ["analyze", "orbital_cube", "read_frames", "pick_frames", "story", "Follower", "Frame"]
