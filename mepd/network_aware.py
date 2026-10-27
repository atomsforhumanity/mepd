"""Network-aware recursive search: don't expand the network where the route
from the start is kinetically out of reach.

A recursive path search (MSMEP) splits a path at intermediates and searches
each piece. Without a notion of where it is in the network, it searches a
piece 150 kcal/mol above the reactants as eagerly as one at the bottom, and
the tree grows combinatorially. Here every piece is measured against one
reference (the floor: the root search's start, or `network_floor_hartree`
when a follow-up search continues someone else's network) and against the
route that leads to it:

  * the route's *span*: the largest climb out of a preceding well along the
    route from the start, i.e. the effective barrier of the route so far
    (the energetic-span idea: after falling into a deep well, the next
    barrier is measured from that well, not from the start);
  * its *well*: the lowest minimum on the route so far.

Pruning uses only bounds that hold for every route in the network (the
floor is the lowest reactant, so reaching anything at energy E above it
climbs at least E): a piece whose start lies more than the cutoff above the
floor is not searched (leaf "pruned_above_cutoff"); an elementary step whose
TS, optimized and confirmed, lies more than the cutoff above the floor is
kept as a path but marked kinetically irrelevant ("above_cutoff": no TS
optimization, IRC or channel follow-ups). A piece from a start within reach
is always searched, wherever it heads: the search is also a probe that finds
intermediates on its way. The route state (span, well) along each search
path is recorded for the log; it does not prune, because the same species or
TS may be reached by a lower route found by another search.

The cutoff is the barrier crossed with a half-life of `cutoff_time_s` at
`cutoff_temperature` (Eyring), plus `cutoff_margin_kcal` for the method's
error (electronic vs free energies, xTB vs a reference level):

    E_cut = R T ln(k_B T t / (h ln 2)) + margin

so a higher temperature or a longer time admits more of the network. Or it
is given directly as `barrier_cutoff_kcal`. Without either, nothing changes.

References: kinetics-interlaced exploration, which only traverses paths
from the reactants below a maximum barrier, M. Bensberg, M. Reiher,
Isr. J. Chem. 63, e202200123 (2023), doi:10.1002/ijch.202200123;
energetic span, S. Kozuch, S. Shaik,
Acc. Chem. Res. 44, 101 (2011), doi:10.1021/ar1000956.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from scipy.constants import Boltzmann, Planck, gas_constant

from mepd.units import HARTREE_TO_KCAL

_KCAL = 4184.0
_LOG_LOCK = threading.Lock()


def eyring_cutoff_kcal(temperature: float, time_s: float) -> float:
    """The barrier (kcal/mol) a first-order step crosses with a half-life of
    `time_s` at `temperature` (K)."""
    T = float(temperature)
    rt = gas_constant * T / _KCAL
    return rt * math.log(Boltzmann * T * float(time_s) / (Planck * math.log(2.0)))


def resolve_cutoff(path_min_inputs: Any) -> Optional[float]:
    """The barrier cutoff (kcal/mol) the path-min inputs ask for, or None
    (network-aware pruning off)."""
    direct = getattr(path_min_inputs, "barrier_cutoff_kcal", None)
    if direct is not None:
        return float(direct) if float(direct) > 0 else None
    T = getattr(path_min_inputs, "cutoff_temperature", None)
    if not T:
        return None
    t = float(getattr(path_min_inputs, "cutoff_time_s", 86400.0) or 86400.0)
    margin = float(getattr(path_min_inputs, "cutoff_margin_kcal", 10.0) or 0.0)
    return eyring_cutoff_kcal(float(T), t) + margin


@dataclass(frozen=True)
class RouteState:
    """Along the route from the start up to some point: the lowest well so
    far and the largest climb out of a preceding well (kcal/mol, relative to
    the floor)."""
    well: float = 0.0
    span: float = 0.0

    def through(self, e: float) -> "RouteState":
        """The route continued to a minimum (or any point) at energy `e`."""
        return RouteState(well=min(self.well, e), span=max(self.span, e - self.well))

    def over(self, peak: float) -> "RouteState":
        """The route continued over a barrier top at `peak` (the well stays)."""
        return RouteState(well=self.well, span=max(self.span, peak - self.well))

    @classmethod
    def at(cls, e: float) -> "RouteState":
        """A route that starts at a minimum `e` above the floor: it climbed at
        least to `e` to get there."""
        return cls(well=min(0.0, e), span=max(0.0, e))

    def as_dict(self) -> dict:
        return {"well_kcal": round(self.well, 3), "span_kcal": round(self.span, 3)}


def rel_kcal(energy_hartree: Optional[float], floor_hartree: float) -> Optional[float]:
    if energy_hartree is None:
        return None
    return (float(energy_hartree) - float(floor_hartree)) * HARTREE_TO_KCAL


def log_event(event: dict) -> None:
    """Append one line to $MEPD_NETWORK_AWARE_LOG (written as the search
    goes, so a killed run still says what it explored and what it cut).
    Never raises."""
    path = os.environ.get("MEPD_NETWORK_AWARE_LOG", "").strip()
    if not path:
        return
    try:
        line = json.dumps({"t": round(time.time(), 2), **event}, default=float)
        with _LOG_LOCK, open(path, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def accessibility(steps, energies_hartree, floor_hartree: float, start: int = 0) -> dict:
    """{species index: RouteState of its best route from `start`} over every
    verified step found so far (each step `(a, b, ts_energy_hartree)` or an
    object with .a, .b, .ts_energy; usable both ways). "Best" = the smallest
    span (largest climb out of a preceding well), then the deepest well.
    Species no known route reaches are absent. This is the network-wide
    picture a search path alone does not have: a species is within reach if
    ANY route found by ANY search gets there under the cutoff."""
    def parts(st):
        if isinstance(st, (tuple, list)):
            return int(st[0]), int(st[1]), float(st[2])
        return int(st.a), int(st.b), float(st.ts_energy)

    rel = [rel_kcal(e, floor_hartree) for e in energies_hartree]
    edges = [parts(st) for st in steps]
    acc = {int(start): RouteState.at(rel[int(start)])}
    for _ in range(len(rel) + 1):
        changed = False
        for a, b, ts in edges:
            peak = rel_kcal(ts, floor_hartree)
            for u, v in ((a, b), (b, a)):
                if u not in acc or v >= len(rel):
                    continue
                s = acc[u].through(rel[u]).over(max(peak, rel[u], rel[v])).through(rel[v])
                if v not in acc or (round(s.span, 9), round(s.well, 9)) < (round(acc[v].span, 9), round(acc[v].well, 9)):
                    acc[v] = s
                    changed = True
        if not changed:
            break
    return acc
