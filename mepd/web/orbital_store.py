"""Every structure of a session has its frontier orbitals ready (mepd.orbitals).

A keeper thread per open session makes sure each structure and conformer in
the workspace has orbitals at the session's level of theory, kept in
`<workspace>/orbitals/` (never evicted) and already built on the full grid:
the calculation's own SCF when a job computed that geometry (its
`jobs/<id>/orbitals/`), else a single point. Showing orbitals in the UI is
then only a choice of what to draw.
"""
from __future__ import annotations

import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from mepd import orbitals

log = logging.getLogger(__name__)
SWEEP_S = 3.0          # look for structures without orbitals this often
PER_SWEEP = 6          # new orbitals made per sweep at most (stays out of the way of jobs)
MAX_ATOMS = 300        # larger structures: orbitals on request only
MAX_CONFORMERS = 10    # per structure
SETTLE_S = 8.0         # a structure this new may be about to be optimized: wait (its job's SCF will do)


def session_method(ws) -> tuple[str, dict]:
    """(method, TeraChem engine kwds) of the session's level: TeraChem's own
    or g-xTB's orbitals at those levels, GFN2-xTB's at any other."""
    import tomli

    try:
        profile = tomli.loads(ws.read_profile(ws.level_profile))
    except Exception:
        profile = {}
    engine = str(profile.get("engine_name") or "gxtb").lower()
    method = "terachem" if engine == "terachem" else "gxtb" if engine == "gxtb" else "gfn2"
    return method, dict(profile.get("terachem_engine_kwds") or {})


def store_dir(ws) -> Path:
    return Path(ws.root) / "orbitals"


def run_dirs(ws) -> list:
    """Every job's kept SCFs, most recently written first."""
    return sorted(Path(ws.jobs_dir).glob("*/orbitals"), key=lambda p: p.stat().st_mtime, reverse=True)


def ensure(ws, xyz_text: str, charge: int, multiplicity: int, method: str, engine_kwds: dict) -> Optional[dict]:
    """The stored orbitals (index entry, built) of this geometry at `method`:
    already there, copied from the job that computed it, or computed now."""
    symbols, coords = orbitals._parse_xyz(xyz_text)
    store = store_dir(ws)
    hit = orbitals.find_run_orbitals([store], symbols, coords, method)
    if hit is None:
        run = orbitals.find_run_orbitals(run_dirs(ws), symbols, coords, method)
        if run is not None:
            entry = orbitals.save_run_orbitals(Path(run["dir"]) / run["file"], xyz_text, method, run["label"],
                                               where=store, keep=None, source=run.get("source", "run"))
        else:
            if method == "terachem":
                text = orbitals.run_terachem_molden(symbols, coords, charge, multiplicity, engine_kwds)
                label = orbitals.terachem_label(engine_kwds)
            else:
                text = orbitals.run_molden(symbols, coords, charge, multiplicity, method)
                label = "g-xTB" if method == "gxtb" else "GFN2-xTB"
            with tempfile.NamedTemporaryFile("w", suffix=".molden", delete=False) as fh:
                fh.write(text)
            try:
                entry = orbitals.save_run_orbitals(Path(fh.name), xyz_text, method, label, where=store, keep=None,
                                                   source="single point")
            finally:
                Path(fh.name).unlink(missing_ok=True)
        if entry is None:
            return None
        hit = {**entry, "dir": str(store)}
    orbitals.build_stored(hit, multiplicity)
    return hit


class OrbitalKeeper:
    """Background thread: gives every structure of `ws` its orbitals."""

    def __init__(self, ws, *, max_atoms: int = MAX_ATOMS, allow_terachem: bool = True):
        self.ws, self.max_atoms, self.allow_terachem = ws, max_atoms, allow_terachem
        self._stop = threading.Event()
        self._failed: set = set()        # (geometry file, method) that could not be done: not retried
        self._thread = threading.Thread(target=self._loop, name="mepd-orbitals", daemon=True)

    def start(self) -> "OrbitalKeeper":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(SWEEP_S):
            try:
                self.sweep()
            except Exception:                       # never take the server down over orbitals
                log.debug("orbital sweep failed", exc_info=True)

    def todo(self) -> list:
        """(xyz file, charge, multiplicity) of every structure and conformer to cover."""
        out = []
        for rec in self.ws.snapshot()["structures"].values():
            if rec.get("qmmm") or rec.get("status") == "optimizing" or int(rec.get("natoms") or 0) > self.max_atoms:
                continue
            if time.time() - float(rec.get("created") or 0) < SETTLE_S:
                continue
            paths = [self.ws.structure_path(rec["id"])]
            for conf in (rec.get("conformers") or [])[:MAX_CONFORMERS]:
                paths.append(self.ws.conformer_path(rec["id"], conf["id"]))
            for p in paths:
                out.append((Path(p), int(rec.get("charge") or 0), int(rec.get("multiplicity") or 1)))
        return out

    def sweep(self) -> int:
        """Make the missing orbitals, at most PER_SWEEP; returns how many were made."""
        method, kwds = session_method(self.ws)
        if method == "terachem" and not self.allow_terachem:
            return 0
        store = store_dir(self.ws)
        made = 0
        for path, charge, mult in self.todo():
            if self._stop.is_set() or made >= PER_SWEEP:
                break
            if (str(path), method) in self._failed or not path.is_file():
                continue
            try:
                xyz = path.read_text()
                symbols, coords = orbitals._parse_xyz(xyz)
                have = orbitals.find_run_orbitals([store], symbols, coords, method)
                if have is not None and orbitals.built({**have, "dir": str(store)}) is not None:
                    continue
                ensure(self.ws, xyz, charge, mult, method, kwds)
                made += 1
            except Exception as exc:
                self._failed.add((str(path), method))
                log.info("no orbitals for %s: %s", path.name, exc)
        return made
