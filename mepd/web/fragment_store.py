"""Fragment orbitals of a session, kept (mepd.fragment_orbitals): what
/api/fragment-orbitals sends.

Each result (one geometry, its cut into molecules, the levels drawn) is built
once and kept on disk in `<workspace>/orbitals/fragments/` as the gzipped
JSON the endpoint sends, plus a small summary (the overlap tables) for path
profiles. Asking again -- playing a path back, reopening it after a
restart -- is a file read. The same result asked twice at once is built
once (the viewer preloads a path's frames while its profile fills them).
The folder keeps its newest MAX_BYTES.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

MAX_BYTES = 500 * 2 ** 20
WORKERS = 3                 # a path's frames built at once (xtb runs single-threaded)
_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()
_writes = 0


def folder(ws) -> Path:
    from mepd.web import orbital_store

    return orbital_store.store_dir(ws) / "fragments"


def _lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if len(_LOCKS) > 4096:
            for k in [k for k, v in _LOCKS.items() if not v.locked()][:2048]:
                del _LOCKS[k]
        return _LOCKS.setdefault(key, threading.Lock())


def reference_for(symbols: list, refs: list, charge: int,
                  prefer: Optional[str] = None) -> tuple[Optional[np.ndarray], str, list]:
    """(coordinates, "start" / "end", the ends that could be cut) for a path
    (`refs`: [(symbols, coords)] of its first and last frame): `prefer`'s
    end if its bonding holds more than one molecule, else the first end that
    does; (None, "shown", []) if neither does: the shown geometry is cut."""
    from mepd.discovery.nanoreactor import components, perceive_bonds

    ends = {}
    for (rsym, rxyz), end in zip(refs[:2], ("start", "end")):
        if list(rsym) == list(symbols):
            xyz = np.asarray(rxyz, dtype=float)
            if len(components(len(symbols), perceive_bonds(list(symbols), xyz))) > 1:
                ends[end] = xyz
    if not ends:
        return None, "shown", []
    end = prefer if prefer in ends else next(iter(ends))
    return ends[end], end, list(ends)


def build(ws, symbols: list, coords, **kw) -> bytes:
    """The gzipped JSON of mepd.fragment_orbitals.analyze for this geometry
    (plus "label"): kept, or built now (once, however many ask)."""
    return ensure(ws, symbols, coords, **kw).read_bytes()


def ensure(ws, symbols: list, coords, *, charge: int, multiplicity: int, method: str, engine_kwds: dict,
           reference=None, cut_from: str = "shown", cut_options=(), fragments=None, groups=None, pick=None,
           max_points: int,
           job: Optional[str] = None) -> Path:
    """The kept result's file, built first if need be."""
    from mepd import fragment_orbitals as fo
    from mepd import orbitals
    from mepd.web import orbital_store

    rnd = lambda a, d: None if a is None else [[round(float(x), d) for x in row] for row in np.asarray(a).reshape(-1, 3)]  # noqa: E731
    key = hashlib.sha1(json.dumps([list(symbols), rnd(coords, 4), int(charge), int(multiplicity), method, engine_kwds,
                                   rnd(reference, 3), cut_from, list(cut_options), fragments, groups, pick,
                                   int(max_points),
                                   "cut-v2"],
                                  default=str, sort_keys=True).encode()).hexdigest()
    fp = folder(ws) / f"{key}.json.gz"
    if fp.is_file():
        return fp
    with _lock(key):
        if fp.is_file():                     # built while this one waited
            return fp
        whole = None
        runs = ([Path(ws.jobs_dir) / job / "orbitals"] if job else []) + \
            [orbital_store.store_dir(ws)] + orbital_store.run_dirs(ws)
        try:
            hit = orbitals.find_run_orbitals(runs, symbols, coords, method, rigid=False)
            if hit is not None:
                whole = (Path(hit["dir"]) / hit["file"]).read_text()
            elif method != "terachem":
                whole = orbitals.run_molden(symbols, np.asarray(coords, dtype=float), charge, multiplicity, method)
        except Exception:
            whole = None
        out = fo.analyze(symbols, coords, charge, multiplicity, method, engine_kwds, fragments=fragments,
                         groups=groups,
                         reference_A=reference, whole_molden=whole, pick=pick, max_points=max_points)
        # Which geometry the molecules were cut from (the analysis says whether the shown one splits the same way).
        out["cut"] = {**out.get("cut", {}), "from": "given" if fragments else cut_from,
                      "options": [] if fragments else list(cut_options)}
        out["label"] = orbitals.terachem_label(engine_kwds) if method == "terachem" else \
            "g-xTB" if method == "gxtb" else "GFN2-xTB"
        data = gzip.compress(json.dumps(out).encode(), 5)
        fp.parent.mkdir(parents=True, exist_ok=True)
        summary = [{k: p[k] for k in ("donor", "acceptor", "table")} for p in out["pairs"]]
        (fp.parent / f"{key}.pairs.json").write_text(json.dumps(summary))
        tmp = fp.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(fp)
        _trim(fp.parent)
        return fp


def profile(ws, frames: list, **kw) -> list:
    """Every frame of a path ([(symbols, coords)]) built and kept, so playing
    it back reads files, WORKERS at a time; per frame its pairs' overlap
    tables [[occ, virt, |S|]], or None where the frame failed."""
    def one(geom):
        try:
            fp = ensure(ws, geom[0], geom[1], **kw)
            pairs = json.loads(fp.with_name(fp.name.replace(".json.gz", ".pairs.json")).read_text())
        except Exception:
            return None
        return [{"donor": p["donor"], "acceptor": p["acceptor"],
                 "table": [[t["occ"], t["virt"], t["s"]] for t in p["table"]]} for p in pairs]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(one, frames))


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
            f.with_name(f.name.replace(".json.gz", ".pairs.json")).unlink(missing_ok=True)
