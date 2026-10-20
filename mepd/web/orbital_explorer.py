"""The orbital explorer's data (mepd.orbital_levels), kept per session.

A geometry or a path is analyzed once: `<workspace>/orbitals/explorer/<key>/`
holds levels.json (every frame's numbers and the lines through the path),
the molden files of each frame (the whole system, each fragment), and the
cube files built from them as they are asked for. Reopening the explorer,
playing the path, or comparing its ends are then file reads. The same
analysis asked for twice at once runs once. The folder keeps its newest
MAX_BYTES.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np

MAX_BYTES = 1024 * 2 ** 20
_LOCKS: dict = {}
_GUARD = threading.Lock()
_writes = 0


def folder(ws) -> Path:
    from mepd.web import orbital_store

    return orbital_store.store_dir(ws) / "explorer"


def _lock(key: str) -> threading.Lock:
    with _GUARD:
        if len(_LOCKS) > 1024:
            for k in [k for k, v in _LOCKS.items() if not v.locked()][:512]:
                del _LOCKS[k]
        return _LOCKS.setdefault(key, threading.Lock())


def key_of(symbols: list, frames: list, charge: int, multiplicity: int, method: str, engine_kwds: dict) -> str:
    return hashlib.sha1(json.dumps([list(symbols), [[[round(float(x), 4) for x in row] for row in np.asarray(f)]
                                                    for f in frames], int(charge), int(multiplicity), method,
                                    engine_kwds], default=str, sort_keys=True).encode()).hexdigest()[:24]


def levels(ws, symbols: list, frames: list, *, charge: int, multiplicity: int, method: str, engine_kwds: dict,
           job: Optional[str] = None) -> tuple[str, bytes]:
    """(key, gzipped levels JSON) of a geometry or path ([coords] in
    Angstrom): kept, or analyzed now (once, however many ask)."""
    from mepd import orbital_levels as ol
    from mepd import orbitals
    from mepd.web import orbital_store

    key = key_of(symbols, frames, charge, multiplicity, method, engine_kwds)
    d = folder(ws) / key
    fp = d / "levels.json.gz"
    if fp.is_file():
        return key, fp.read_bytes()
    with _lock(key):
        if fp.is_file():
            return key, fp.read_bytes()
        d.mkdir(parents=True, exist_ok=True)
        # The whole system's own orbitals where a calculation computed the frame, else a single point.
        runs = ([Path(ws.jobs_dir) / job / "orbitals"] if job else []) + \
            [orbital_store.store_dir(ws)] + orbital_store.run_dirs(ws)
        wholes = []
        for coords in frames:
            hit = None
            try:
                hit = orbitals.find_run_orbitals(runs, symbols, coords, method, rigid=False)
            except Exception:
                pass
            wholes.append((Path(hit["dir"]) / hit["file"]).read_text() if hit else None)

        def keep(k, whole, moldens):
            fd = d / f"f{k}"
            fd.mkdir(exist_ok=True)
            (fd / "whole.molden").write_text(whole)
            for i, t in enumerate(moldens):
                (fd / f"frag{i}.molden").write_text(t)

        out = ol.analyze_path(symbols, frames, charge, multiplicity, method, engine_kwds, wholes=wholes, keep=keep)
        out["label"] = orbitals.terachem_label(engine_kwds) if method == "terachem" else \
            "g-xTB" if method == "gxtb" else "GFN2-xTB"
        out["key"] = key
        (d / "geometry.json").write_text(json.dumps({"symbols": list(symbols),
                                                     "frames": [np.asarray(f).tolist() for f in frames]}))
        data = gzip.compress(json.dumps(out).encode(), 5)
        tmp = fp.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(fp)
        _trim(folder(ws))
        return key, data


def cube(ws, key: str, frame: int, system: str, index: int, sign: int, max_points: int, method: str) -> bytes:
    """A gzipped {"cube"} of one orbital of a kept analysis: `system`
    "whole" or a fragment's number, `index` its orbital, drawn as sign*psi on
    a grid around the whole frame."""
    from mepd import orbital_levels as ol

    d = folder(ws) / key
    name = "whole" if system == "whole" else f"frag{int(system)}"
    fp = d / f"f{int(frame)}" / f"{name}.{int(index)}.{1 if sign >= 0 else -1}.{int(max_points)}.json.gz"
    if fp.is_file():
        return fp.read_bytes()
    molden = d / f"f{int(frame)}" / f"{name}.molden"
    geo = json.loads((d / "geometry.json").read_text())
    text = ol.cube(molden.read_text(), method, int(index), geo["symbols"], geo["frames"][int(frame)], sign, max_points)
    data = gzip.compress(json.dumps({"cube": text}).encode(), 5)
    tmp = fp.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(fp)
    return data


def _trim(root: Path) -> None:
    """Keep the newest MAX_BYTES of analyses (checked every 20)."""
    global _writes
    import shutil

    _writes += 1
    if _writes % 20:
        return
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    total = 0
    for p in dirs:
        total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if total > MAX_BYTES:
            shutil.rmtree(p, ignore_errors=True)
