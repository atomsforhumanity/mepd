"""Bond orbitals of a session, kept (mepd.bond_orbitals): what
/api/bond-orbitals and /api/bond-orbitals/path send.

Each geometry's intrinsic bond orbitals are built once from the orbitals the
whole-system view uses (the session's store, a calculation's own SCF at that
geometry -- turned with it when the calculation ran it turned --, else a
single point) and kept in `<workspace>/orbitals/bonds/` (gzipped JSON with
the coefficients, so cubes are drawn without the molden file). A path's
frames are built WORKERS at a time; following the orbitals along it is then
arithmetic on the kept results. The folder keeps its newest MAX_BYTES.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

MAX_BYTES = 500 * 2 ** 20
WORKERS = 3                 # a path's frames built at once (xtb runs single-threaded)
MEMORY = 400                # results kept in memory as objects
_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()
_MEM: OrderedDict = OrderedDict()
_MEM_GUARD = threading.Lock()
_writes = 0


def folder(ws) -> Path:
    from mepd.web import orbital_store

    return orbital_store.store_dir(ws) / "bonds"


def _lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if len(_LOCKS) > 4096:
            for k in [k for k, v in _LOCKS.items() if not v.locked()][:2048]:
                del _LOCKS[k]
        return _LOCKS.setdefault(key, threading.Lock())


def _remember(key: str, value) -> None:
    with _MEM_GUARD:
        _MEM[key] = value
        _MEM.move_to_end(key)
        while len(_MEM) > MEMORY:
            _MEM.popitem(last=False)


def ensure(ws, symbols: list, coords, *, charge: int, multiplicity: int, method: str, engine_kwds: dict,
           job: Optional[str] = None):
    """(BondOrbitals, label, source) of this geometry: kept, or built now
    (once, however many ask)."""
    from mepd import bond_orbitals as bo
    from mepd import orbitals
    from mepd.web import orbital_store

    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    key = hashlib.sha1(json.dumps([list(symbols), np.round(xyz, 4).tolist(), int(charge), int(multiplicity), method,
                                   engine_kwds, bo.STATE_VERSION], default=str, sort_keys=True).encode()).hexdigest()
    with _MEM_GUARD:
        hit = _MEM.get(key)
    if hit is not None:
        return hit
    fp = folder(ws) / f"{key}.json.gz"
    meta = fp.with_name(f"{key}.meta.json")
    with _lock(key):
        with _MEM_GUARD:
            hit = _MEM.get(key)
        if hit is not None:
            return hit
        if fp.is_file() and meta.is_file():
            try:
                m = json.loads(meta.read_text())
                out = (bo.BondOrbitals.load(fp), m["label"], m["source"])
                _remember(key, out)
                return out
            except Exception:
                pass                       # an older layout or a broken file: build again
        runs = ([Path(ws.jobs_dir) / job / "orbitals"] if job else []) + \
            [orbital_store.store_dir(ws)] + orbital_store.run_dirs(ws)
        run = orbitals.find_run_orbitals(runs, symbols, xyz, method)
        if run is not None:
            text, kind = (Path(run["dir"]) / run["file"]).read_text(), run["method"]
            label, source, align = run["label"], run.get("source", "run"), run["align"]
        elif method == "terachem":
            kind = method
            text = orbitals.run_terachem_molden(list(symbols), xyz, charge, multiplicity, engine_kwds)
            label, source, align = orbitals.terachem_label(engine_kwds), "single point", None
        else:
            kind = method
            text = orbitals.run_molden(list(symbols), xyz, charge, multiplicity, method)
            label, source, align = ("g-xTB" if method == "gxtb" else "GFN2-xTB"), "single point", None
        B = bo.BondOrbitals(text, kind, int(charge), list(symbols),
                            shown_A=xyz if align is not None else None, align=align)
        try:
            B.save(fp)
            meta.write_text(json.dumps({"label": label, "source": "run" if source != "single point" else source}))
            _trim(fp.parent)
        except OSError:
            pass
        out = (B, label, "run" if source != "single point" else source)
        _remember(key, out)
        return out


def frames(ws, geoms: list, **kw) -> list:
    """Every frame of a path ([(symbols, coords)]) built and kept, WORKERS at
    a time: [(BondOrbitals, label, source) or None where a frame failed]."""
    def one(g):
        try:
            return ensure(ws, g[0], g[1], **kw)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(one, geoms))


def _trim(d: Path) -> None:
    """Keep the newest MAX_BYTES of results (checked every 50 writes)."""
    global _writes
    _writes += 1
    if _writes % 50:
        return
    files = sorted(d.glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = 0
    for f in files:
        total += f.stat().st_size
        if total > MAX_BYTES:
            f.unlink(missing_ok=True)
            f.with_name(f.name.replace(".json.gz", ".meta.json")).unlink(missing_ok=True)
