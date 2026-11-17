"""Add a molecule to a finished network: the kinetics, the read-out and the
sensitivity analysis (docs/add-molecule.md).

Everything here works on a plain network (a dict, the kinetics part of the
query file, so it needs no workspace and no geometry):

    {"temperature": K, "energies": "free" | "electronic",
     "species": [{"id", "name", "g"}, ...],                  # kcal/mol, one absolute scale
     "steps": [{"id", "label", "reactants": [ids], "products": [ids], "g_ts", "status"}, ...]}

`added` puts a molecule C into it from a list of estimates (where they come
from is not this module's business): C's binding to species and to the two
ends and the TS of parent steps, new species holding C, new steps. `readout`
compares the parent's kinetics with the variant's; `limits` says what
controls a target (degree of rate control, or, for a parent that makes none
of it, its lowest-peak routes); `first_order` is the estimate
d ln Q = sum X_i (-ddG_i / RT) to show beside the full kinetics.

Free energies are at the 1 M standard state. A species' chemical potential
at an amount c is g + RT ln(c / 1 M); a step's apparent barrier is its TS
above its reactants' chemical potentials, so that its forward rate is
(k_B T / h) (1 M) exp(-apparent / RT): the barrier "from the network's own
resting state at the stated amounts".
"""
from __future__ import annotations

import copy
import heapq
import math
from collections import Counter
from typing import Optional, Sequence

import numpy as np

from mepd import microkinetics as mk

# Below this share of the starting material a target's amount is integration noise (the kinetics'
# absolute tolerance is 1e-14 M): its degree of control is then not defined.
RESOLUTION = 1e-9
# A share of the added molecule below this counts as none (fate rules).
TRACE = 1e-3
USED = ("est", "verified")


# ---------------------------------------------------------------- the model
def _net(network: dict) -> tuple[mk.Network, dict]:
    pos = {s["id"]: k for k, s in enumerate(network["species"])}
    steps = [mk.Step([pos[x] for x in st["reactants"]], [pos[x] for x in st["products"]], float(st["g_ts"]),
                     st.get("label") or st["id"]) for st in network["steps"]]
    return mk.Network([s.get("name") or s["id"] for s in network["species"]],
                      np.array([float(s["g"]) for s in network["species"]]), steps), pos


def _value(e: Optional[dict], energies: str) -> Optional[float]:
    """The number of an estimate {"electronic", "free"} the network's energies call for; the electronic
    one where no free energy was estimated."""
    if not e:
        return None
    v = e.get(energies)
    return float(v) if v is not None else (float(e["electronic"]) if e.get("electronic") is not None else None)


def complex_id(species: str, molecule: str) -> str:
    return f"{species}.{molecule}"


def added(network: dict, molecule: dict, estimates: dict) -> dict:
    """The network with the molecule C in it.

    `molecule`: {"id", "name"} (its own free energy cancels in every rate; 0 unless "g" is given).
    `estimates` (every entry with "status": "est" | "verified"; others are left out):
      "species": [{"species": id, "binding": {"electronic", "free", "spread"}}]
          -> a species X.C with g = g(X) + g(C) + binding and the barrierless step X + C <=> X.C;
      "steps": [{"step": parent step id, "binding": {"r": {...}, "ts": {...}, "p": {...}}}]
          -> the parent step with C beside it, its TS at g_ts + g(C) + binding(ts): from X.C to Y.C where an
             end is one molecule (its complex made from binding(r) / binding(p), the lowest known), from the
             separated molecules + C where an end is several. The TS is counted from the separated
             molecules, so binding(r) never enters the rate from a dilute C twice;
      "new_species": [{"id", "name", "formed_from": [ids], "released": [ids], "energy": {...}}]
          -> g = sum g(formed_from) - sum g(released) + energy;
      "new_steps": [{"id", "reactants": [ids], "products": [ids], "barrier": {...}, "parent_step": id}]
          -> a step with its TS `barrier` above its reactants (ids may name C, complexes "<X>.<C>", new
             species). C on both sides: it takes part and is returned.
    """
    out = copy.deepcopy(network)
    mode = out.get("energies") or "free"
    sp = {s["id"]: s for s in out["species"]}
    by_id = {st["id"]: st for st in out["steps"]}
    c = molecule["id"]
    if c in sp:
        raise ValueError(f"{c} is already a species of the network")
    sp[c] = {"id": c, "name": molecule.get("name") or c, "g": float(molecule.get("g") or 0.0), "holds": 1,
             "kind": "added", **({"smiles": molecule["smiles"]} if molecule.get("smiles") else {})}
    notes = []

    def flag(e, what):
        if mode == "free" and e and e.get("free") is None:
            notes.append(f"{what}: no free-energy estimate, the electronic number is used")

    # Complexes: the lowest binding known for each species.
    bind: dict = {}

    def offer(x, e, est):
        v = _value(e, mode)
        if v is None or x not in sp or x == c:
            return
        if x not in bind or v < bind[x][0]:
            bind[x] = (v, e, est)

    used = lambda est: est.get("status") in USED   # noqa: E731
    for est in estimates.get("species") or []:
        if used(est):
            offer(est["species"], est.get("binding"), est)
    for est in estimates.get("steps") or []:
        st = by_id.get(est.get("step"))
        if st is None or not used(est):
            continue
        for end, key in ((st["reactants"], "r"), (st["products"], "p")):
            if len(end) == 1:
                offer(end[0], (est.get("binding") or {}).get(key), est)
    for x, (v, e, est) in bind.items():
        cid = complex_id(x, c)
        flag(e, f"{sp[x].get('name') or x} + {sp[c]['name']}")
        sp[cid] = {"id": cid, "name": f"{sp[x].get('name') or x} · {sp[c]['name']}", "g": sp[x]["g"] + sp[c]["g"] + v,
                   "holds": 1, "kind": "complex", "of": [x], "binding": v, "status": est["status"],
                   "spread": (e or {}).get("spread"), "level": est.get("level"), "source": est.get("source")}
        out["steps"].append({"id": f"assoc:{cid}", "label": f"{sp[x].get('name') or x} + {sp[c]['name']} ⇌ complex",
                             "reactants": [x, c], "products": [cid], "g_ts": max(sp[x]["g"] + sp[c]["g"], sp[cid]["g"]),
                             "kind": "association", "status": est["status"], "level": est.get("level")})

    # Parent steps with C beside them.
    for est in estimates.get("steps") or []:
        st = by_id.get(est.get("step"))
        e_ts = (est.get("binding") or {}).get("ts")
        if st is None or not used(est) or _value(e_ts, mode) is None:
            continue
        flag(e_ts, f"{st.get('label') or st['id']} with {sp[c]['name']}")
        side = lambda end: [complex_id(end[0], c)] if len(end) == 1 and end[0] in bind else list(end) + [c]   # noqa: E731
        out["steps"].append({"id": f"{st['id']}+{c}", "label": f"{st.get('label') or st['id']} with {sp[c]['name']}",
                             "reactants": side(st["reactants"]), "products": side(st["products"]),
                             "g_ts": float(st["g_ts"]) + sp[c]["g"] + _value(e_ts, mode), "kind": "bystander",
                             "parent_step": st["id"], "status": est["status"], "spread": e_ts.get("spread"),
                             "level": est.get("level"), "source": est.get("source")})

    # New species and steps.
    for n in estimates.get("new_species") or []:
        if not used(n):
            continue
        made, gone = list(n.get("formed_from") or []), list(n.get("released") or [])
        missing = [x for x in made + gone if x not in sp]
        if missing:
            raise ValueError(f"new species {n['id']}: unknown species {', '.join(missing)}")
        flag(n.get("energy"), n.get("name") or n["id"])
        sp[n["id"]] = {"id": n["id"], "name": n.get("name") or n["id"], "kind": "new",
                       "g": sum(sp[x]["g"] for x in made) - sum(sp[x]["g"] for x in gone) + _value(n.get("energy"), mode),
                       "holds": sum(sp[x].get("holds", 0) for x in made) - sum(sp[x].get("holds", 0) for x in gone),
                       **{k: n[k] for k in ("smiles", "status", "spread", "level", "source") if n.get(k) is not None}}
    for n in estimates.get("new_steps") or []:
        if not used(n):
            continue
        missing = [x for x in n["reactants"] + n["products"] if x not in sp]
        if missing:
            raise ValueError(f"new step {n['id']}: unknown species {', '.join(missing)} (a complex needs its binding "
                             "among the estimates)")
        flag(n.get("barrier"), n.get("label") or n["id"])
        out["steps"].append({"id": n["id"], "label": n.get("label") or n["id"], "reactants": list(n["reactants"]),
                             "products": list(n["products"]), "kind": "new",
                             "g_ts": sum(sp[x]["g"] for x in n["reactants"]) + _value(n.get("barrier"), mode),
                             **{k: n[k] for k in ("parent_step", "status", "spread", "level", "source")
                                if n.get(k) is not None}})
    out["species"] = list(sp.values())
    out["added"] = c
    out["notes"] = list(dict.fromkeys((out.get("notes") or []) + notes))
    return out


# ---------------------------------------------------------------- kinetics
def run(network: dict, initial: dict, time_s: float, held: Sequence[str] = ()) -> dict:
    """The kinetics of a network from `initial` ({id: M}) over `time_s`: per species the amount at the end,
    its largest and its time average; per step the integrated net extent and the apparent barriers."""
    net, pos = _net(network)
    T = float(network["temperature"])
    c0 = np.zeros(len(pos))
    for sid, v in (initial or {}).items():
        if sid in pos:
            c0[pos[sid]] = float(v)
    res = mk.simulate(net, c0, T, float(time_s), held=[pos[s] for s in held if s in pos])
    mean = np.trapezoid(res.c, res.times, axis=0) / float(time_s)
    rt = mk.rt_kcal(T)
    with np.errstate(divide="ignore"):
        mu = net.g + rt * np.log(mean)                       # -inf where there never was any
    apparent = []
    for st in net.steps:
        row = []
        for side in (st.reactants, st.products):
            ref = float(mu[side].sum())
            row.append(st.g_ts - ref if math.isfinite(ref) else None)
        apparent.append(row)
    return {"net": net, "pos": pos, "c0": c0, "res": res, "mean": mean, "apparent": apparent,
            "times": res.times, "initial": {s: float(v) for s, v in (initial or {}).items() if s in pos}}


def _targets(target) -> list:
    """A target is one species or several counted together (the stereoisomers of a product)."""
    return [target] if isinstance(target, str) else list(target)


def time_to(times, series, amount: float) -> Optional[float]:
    """When a series first holds `amount` (s; between output times by a straight line in log t)."""
    if not amount > 0:
        return None
    for k in range(1, len(times)):
        a, b = series[k - 1], series[k]
        if b >= amount:
            if a >= amount or b == a:
                return float(times[k - 1])
            f = (amount - a) / (b - a)
            if times[k - 1] <= 0:
                return float(times[k] * f)
            return float(math.exp(math.log(times[k - 1]) + f * (math.log(times[k]) - math.log(times[k - 1]))))
    return None


def _cargo(network: dict) -> dict:
    """What each species carries besides the added molecule: a complex X.C carries X, C nothing, any other
    species itself. A step that leaves the cargo of both sides the same is a pure association."""
    return {s["id"]: ([] if s.get("kind") == "added" else list(s["of"]) if s.get("kind") == "complex" else [s["id"]])
            for s in network["species"]}


def fate(network: dict, k: dict, routes: Sequence[dict] = (), peak: Optional[dict] = None) -> Optional[dict]:
    """What became of the added molecule C in the kinetics `k` (of `run`), as two facts side by side and what
    follows from them. `routes`: `flux_routes` of the target in these kinetics; `peak`: its lowest-peak route.

    * `on_route`: whether C is on the routes that carry the target's material (at least half of it goes
      through steps with a C-holding species; None when no material was made), with `route_share` (that
      share), `route_peak` (the highest point of the lowest-peak route to the target, kcal/mol above the
      mixture) and `route_peak_with_added` (whether that route has C in it);
    * `turnovers` (= `returns`): how often, per C put in, C came back from a step it took part in, in the
      run's time.

    By amounts (per C put in):

    * `shares` at the end: `free` (C itself), `bound` (in complexes, which fall apart without a barrier),
      `covalent` (in new species that hold C);
    * `returns`: how often a step that changed what C carries handed C back free or in a complex
      (sum over such steps of their net extent x the C on their product side held that way) / c0(C);
    * `resting_state`: the C-holding species with the largest time-averaged amount.

    Rules, in this order. C took part when it is on the route, or `returns` or `covalent` reach 0.001. When
    it did not: "bound and inert" if at the end at least half of C is bound, or a complex holds at least as
    much of a species as is left of it free; else "no part". When it did: "regenerated" if returns >= covalent
    (C came back at least as often as it is held up at the end), else "consumed". A slow catalyst reads
    "regenerated" with few turnovers: read the two facts, not the word alone.
    """
    c = network.get("added")
    pos = k["pos"]
    if c is None or c not in pos or not k["c0"][pos[c]] > 0:
        return None
    c0 = float(k["c0"][pos[c]])
    sp = {s["id"]: s for s in network["species"]}
    holders = [s for s in network["species"] if s.get("holds")]
    end = k["res"].c[-1]
    shares = {"free": 0.0, "bound": 0.0, "covalent": 0.0}
    for s in holders:
        key = {"added": "free", "complex": "bound"}.get(s.get("kind"), "covalent")
        shares[key] += s["holds"] * float(end[pos[s["id"]]]) / c0
    cargo = _cargo(network)
    back = 0.0
    for st, ext in zip(network["steps"], k["res"].extent):
        r, p = (st["reactants"], st["products"]) if ext >= 0 else (st["products"], st["reactants"])
        if not any(sp[x].get("holds") for x in r):
            continue
        if Counter(y for x in r for y in cargo[x]) == Counter(y for x in p for y in cargo[x]):
            continue                                     # an association or its reverse
        back += abs(float(ext)) * sum(sp[x]["holds"] for x in p if sp[x].get("kind") in ("added", "complex"))
    returns = back / c0
    rest = max(holders, key=lambda s: k["mean"][pos[s["id"]]] * s["holds"])
    carried = [r for r in routes if r.get("share") is not None]
    share = sum(r["share"] for r in carried if r.get("with_added")) if carried else None
    on_route = None if share is None else bool(share >= 0.5)
    holds_c = {s["id"] for s in holders}
    by_id = {st["id"]: st for st in network["steps"]}
    peak_with = bool(peak) and any(holds_c & set(by_id[x["step"]]["reactants"] + by_id[x["step"]]["products"])
                                   for x in peak["steps"])
    if not on_route and returns < TRACE and shares["covalent"] < TRACE:
        holds = any(s.get("kind") == "complex" and end[pos[s["id"]]] > TRACE * c0
                    and end[pos[s["id"]]] >= end[pos[s["of"][0]]] for s in holders)
        verdict = "bound and inert" if shares["bound"] >= 0.5 or holds else "no part"
    else:
        verdict = "regenerated" if returns >= shares["covalent"] else "consumed"
    return {"fate": verdict, "on_route": on_route, "route_share": share,
            "route_peak": peak["peak"] if peak else None, "route_peak_with_added": peak_with if peak else None,
            "turnovers": returns, "shares": shares, "returns": returns,
            "resting_state": {"id": rest["id"], "name": rest.get("name") or rest["id"],
                              "share": rest["holds"] * float(k["mean"][pos[rest["id"]]]) / c0}}


def flux_routes(network: dict, k: dict, target, *, limit: int = 5) -> list:
    """Where the target's material came from: routes (each a list of steps from starting species to the
    target) and the amount each carried, by a greedy split of the steps' net extents: the step that made
    most of the target, then for each of its reactants that was not there at the start the step that made
    most of that, and so on; the route carries the smallest extent on it, which is taken off its steps
    before the next route is read. Shares are of the target's gross formation."""
    pos = k["pos"]
    goal = [x for x in _targets(target) if x in pos]
    if not goal:
        return []
    left = [abs(float(x)) for x in k["res"].extent]
    sides = [((st["reactants"], st["products"]) if x >= 0 else (st["products"], st["reactants"]))
             for st, x in zip(network["steps"], k["res"].extent)]
    start = {s for s, v in k["initial"].items() if v > 0}
    made = sum(left[j] for j, (r, p) in enumerate(sides) for x in goal if x in p and x not in r)
    floor = max(made, 1e-300) * 1e-6
    sp = {s["id"]: s for s in network["species"]}
    routes = []
    for _ in range(limit):
        makes = lambda x: [j for j, (r, p) in enumerate(sides) if x in p and x not in r and left[j] > floor]   # noqa: E731
        first = max(goal, key=lambda x: max([left[j] for j in makes(x)] or [0.0]))
        chosen, seen, todo, ok = [], set(), [first], True
        while todo:
            x = todo.pop()
            if x in seen or x in start:
                continue
            seen.add(x)
            makers = [j for j, (r, p) in enumerate(sides) if x in p and x not in r and left[j] > floor and j not in chosen]
            if not makers:
                ok = ok and any(x in sides[j][1] for j in chosen)
                continue
            j = max(makers, key=lambda j: left[j])
            chosen.append(j)
            todo += sides[j][0]
        if not ok or not chosen:
            break
        amount = min(left[j] for j in chosen)
        for j in chosen:
            left[j] -= amount
        chosen.reverse()
        routes.append({"steps": [network["steps"][j]["id"] for j in chosen],
                       "labels": [network["steps"][j].get("label") or network["steps"][j]["id"] for j in chosen],
                       "amount": amount, "share": amount / made if made > 0 else None,
                       "with_added": any(sp[x].get("holds") for j in chosen for x in sides[j][0] + sides[j][1])})
    return routes


def peak_routes(network: dict, initial: dict, target, *, limit: int = 3, max_steps: int = 8,
                max_states: int = 50000) -> list:
    """The routes from the starting mixture to the target with the lowest highest point, each as its steps
    in order with the height of every TS above the mixture.

    A route is followed as molecules in hand. A step may run (either way) when each of its reactants is a
    starting species (as many as needed, at its chemical potential g + RT ln c0) or in hand; it puts its
    products in hand, together with what was made along with its reactants (one lot: the free energy of
    making an intermediate is that of making it and its co-products, and it is counted once however
    many of them a later step uses). The height of a TS is the cost of the lots its reactants come from
    plus the TS above its reactants. Molecules made by an unrelated side reaction are a lot of their own
    and change no height; a route ends with the target's lot alone in hand. Lowest peak first, then
    fewest steps; one route per set of steps. The search stops after `max_states` sets of molecules in
    hand (it then returns the routes it has)."""
    T = float(network["temperature"])
    rt = mk.rt_kcal(T)
    g = {s["id"]: float(s["g"]) for s in network["species"]}
    start = {s: g[s] + rt * math.log(float(v)) for s, v in (initial or {}).items() if s in g and float(v) > 0}
    goal = {x for x in _targets(target) if x in g}
    if not goal or goal & set(start):
        return []
    moves = []                                   # (step index, reactants, products, direction)
    for j, st in enumerate(network["steps"]):
        moves.append((j, st["reactants"], st["products"], "forward"))
        moves.append((j, st["products"], st["reactants"], "reverse"))
    need: dict = {}
    free_moves = []
    for m, (_j, r, _p, _d) in enumerate(moves):
        inner = [x for x in r if x not in start]
        if not inner:
            free_moves.append(m)
        for x in set(inner):
            need.setdefault(x, []).append(m)
    heap = [(-math.inf, 0, (), (), ())]           # peak, steps, lots in hand, their costs, path
    best: dict = {}
    out, found, n = [], [], 0
    while heap and len(out) < limit and n < max_states:
        peak, depth, lots, costs, path = heapq.heappop(heap)
        if best.get(lots, (math.inf, 0)) < (peak, depth):
            continue
        n += 1
        if any(goal & set(lot) for lot in lots):
            used = {j for j, _d, _h in path}
            if len(lots) == 1 and used not in found:
                found.append(used)
                out.append({"peak": peak, "steps": [{"step": network["steps"][j]["id"],
                                                     "label": network["steps"][j].get("label") or network["steps"][j]["id"],
                                                     "direction": d, "height": h} for j, d, h in path]})
            continue
        if depth >= max_steps:
            continue
        for m in dict.fromkeys(free_moves + [m for lot in lots for x in lot for m in need.get(x, ())]):
            j, r, p, d = moves[m]
            if path and path[-1][0] == j and path[-1][1] != d:
                continue                          # straight back
            left = [Counter(lot) for lot in lots]
            taken = set()
            for x in r:
                if x in start:
                    continue
                i = next((i for i, lot in enumerate(left) if lot[x] > 0), None)
                if i is None:
                    break
                left[i][x] -= 1
                taken.add(i)
            else:
                ref = sum(start.get(x, g[x]) for x in r)
                top = max(float(network["steps"][j]["g_ts"]), sum(g[x] for x in r), sum(g[x] for x in p))
                paid = sum(costs[i] for i in taken)
                height = paid + top - ref
                lot = tuple(sorted([x for i in taken for x in left[i].elements()] + [x for x in p if x not in start]))
                rest = [(lots[i], costs[i]) for i in range(len(lots)) if i not in taken]
                if lot:
                    rest.append((lot, paid + sum(start.get(x, g[x]) for x in p) - ref))
                rest.sort(key=lambda b: b[0])
                new_lots, new_costs = tuple(b[0] for b in rest), tuple(b[1] for b in rest)
                item = (max(peak, height), depth + 1)
                if not any(goal & set(b) for b in new_lots):      # (every arrival at the target is a route of its own)
                    if best.get(new_lots, (math.inf, 0)) <= item:
                        continue
                    best[new_lots] = item
                heapq.heappush(heap, (item[0], item[1], new_lots, new_costs, path + ((j, d, height),)))
    return out


# ---------------------------------------------------------------- sensitivity
def limits(network: dict, initial: dict, target, time_s: float, *, held: Sequence[str] = (),
           k: Optional[dict] = None) -> dict:
    """What limits the target's amount at `time_s`, and the order to work on the parent's steps.

    Where the kinetics make the target (at least 1e-9 of the starting material): the degree of rate
    control X of every TS and the thermodynamic degree of control of every species
    (mepd.microkinetics.degree_of_control). `order`: the steps by |X|, largest first; X > 0 "lower" (a
    lower TS gives more target), X < 0 "do not lower" (a competing step). `traps`: species with X < 0
    (stabilizing them costs target).

    Where they make none, X is not defined ("defined": False, the reason in "why", "control" None):
    then the lowest-peak routes to the target (`peak_routes`) and `order` = their steps, the highest
    above the access cutoff (the Eyring barrier crossed in `time_s`) first, all "lower". Nothing is
    known then about competing steps."""
    from mepd.network_aware import eyring_cutoff_kcal

    T = float(network["temperature"])
    k = k or run(network, initial, time_s, held)
    net, pos = k["net"], k["pos"]
    cutoff = eyring_cutoff_kcal(T, float(time_s))
    out = {"target": target, "cutoff_kcal": cutoff, "defined": False, "control": None, "order": [], "traps": []}
    goal = [x for x in _targets(target) if x in pos]
    if not goal:
        out["why"] = "the target is not a species of the network"
        return out
    amounts = [float(k["res"].c[-1, pos[x]]) for x in goal]
    base = sum(amounts)
    total = float(k["c0"].sum())
    if base > RESOLUTION * total:
        # Of several species counted together: d ln(sum Q) = sum Q_k d ln Q_k / sum Q.
        xs, xsp = np.zeros(len(net.steps)), np.zeros(len(net.names))
        for x, q in zip(goal, amounts):
            if q > RESOLUTION * total:
                drc = mk.degree_of_control(net, k["c0"], T, float(time_s), pos[x],
                                           held=[pos[s] for s in held if s in pos])
                xs += q / base * np.array(drc["steps"])
                xsp += q / base * np.array(drc["species"])
        steps = {st["id"]: float(x) for st, x in zip(network["steps"], xs) if math.isfinite(x)}
        species = {s["id"]: float(x) for s, x in zip(network["species"], xsp) if math.isfinite(x)}
        out.update(defined=True, basis="degree of control", control={"steps": steps, "species": species},
                   order=[{"step": sid, "control": x, "role": "lower" if x > 0 else "do not lower"}
                          for sid, x in sorted(steps.items(), key=lambda kv: -abs(kv[1])) if abs(x) > 1e-6],
                   traps=[{"species": sid, "control": x} for sid, x in sorted(species.items(), key=lambda kv: kv[1])
                          if x < -1e-6])
        return out
    out["why"] = (f"the network makes none of the target in {time_s:g} s at {T:g} K (below 1e-9 of the starting "
                  "material): a degree of control is not defined")
    out["basis"] = "lowest-peak routes"
    out["routes"] = peak_routes(network, k["initial"], target)
    seen: dict = {}
    for route in out["routes"]:
        for st in route["steps"]:
            if st["step"] not in seen or st["height"] > seen[st["step"]]:
                seen[st["step"]] = st["height"]
    out["order"] = [{"step": sid, "height": h, "above_cutoff": h - cutoff, "role": "lower"}
                    for sid, h in sorted(seen.items(), key=lambda kv: -kv[1])]
    return out


def shifts(network: dict, molecule_id: str, estimates: dict, amount: float) -> dict:
    """The estimates as shifts of the parent's own TSs and species, for the first-order estimate: with C
    at `amount` (M, taken as all free) a parent step gains a second channel through TS.C, as if its TS
    were lower by RT ln(1 + c exp(-binding(ts) / RT)); a species X held as X.C likewise for its binding.
    New species and new steps have no parent to shift: they are not in it."""
    mode = network.get("energies") or "free"
    rt = mk.rt_kcal(float(network["temperature"]))
    low = lambda b: -rt * math.log1p(float(amount) * math.exp(-b / rt))   # noqa: E731
    with_c = added(network, {"id": molecule_id}, {k: estimates.get(k) for k in ("species", "steps")})
    out = {"steps": {}, "species": {}}
    for s in with_c["species"]:
        if s.get("kind") == "complex":
            out["species"][s["of"][0]] = low(s["binding"])
    for est in estimates.get("steps") or []:
        b = _value((est.get("binding") or {}).get("ts"), mode)
        if est.get("status") in USED and b is not None:
            out["steps"][est["step"]] = low(b)
    return out


def first_order(lim: dict, shift: dict, temperature: float) -> Optional[dict]:
    """d ln Q = sum X_i (-ddG_i / RT) over the shifted TSs and species, from `limits`' degrees of control:
    {"dlnq", "factor" (Q'/Q), "terms"}. None where X is not defined."""
    if not lim.get("defined"):
        return None
    rt = mk.rt_kcal(float(temperature))
    terms = []
    for kind in ("steps", "species"):
        for sid, ddg in (shift.get(kind) or {}).items():
            x = lim["control"][kind].get(sid)
            if x is not None and ddg:
                terms.append({"kind": kind[:-1] if kind == "steps" else kind, "id": sid, "control": x,
                              "shift": ddg, "dlnq": x * (-ddg / rt)})
    total = sum(t["dlnq"] for t in terms)
    terms.sort(key=lambda t: -abs(t["dlnq"]))
    return {"dlnq": total, "factor": math.exp(min(total, 700.0)), "terms": terms}


# ---------------------------------------------------------------- the read-out
def _series(k: dict, target):
    cols = [k["pos"][x] for x in _targets(target) if x in k["pos"]]
    return k["res"].c[:, cols].sum(axis=1) if cols else None


def _side(network: dict, k: dict, target, fraction: float) -> dict:
    series = _series(k, target)
    if series is None:
        return {"final": None, "time_to_fraction_s": None}
    final = float(series[-1])
    resolved = final > RESOLUTION * float(k["c0"].sum())
    return {"final": final, "resolved": resolved,
            "time_to_fraction_s": time_to(k["times"], series, fraction * final) if resolved else None}


def readout(parent: dict, variant: dict, initial: dict, amount: float, target, time_s: float, *,
            fraction: float = 0.5, held: Sequence[str] = ()) -> dict:
    """The parent's kinetics and the variant's (the same mixture plus `amount` M of the added molecule),
    side by side: the target, every parent step's apparent barrier before and after (read the way its
    material goes, `direction`; after: the lowest of the step itself and its versions with C, and which,
    `through`; `barrier`: its TS above that side at 1 M), the routes that carry the target's material,
    C's resting state and fate (`fate`). Each step row has a `group`: "through" (its lowest version has C in
    it: the table to show), "amounts" (its own TS; the apparent barrier moved only because amounts moved),
    "none". "resolved" False: an amount below the kinetics' resolution."""
    c = variant["added"]
    kp = run(parent, initial, time_s, held)
    kv = run(variant, {**initial, c: float(amount)}, time_s, held)
    tp, tv = _side(parent, kp, target, fraction), _side(variant, kv, target, fraction)
    tgt = {"id": target, "fraction": fraction, "parent": tp, "with": tv, "factor": None, "time_to_parent_amount_s": None}
    if tp.get("resolved") and tv.get("final") is not None:
        tgt["factor"] = tv["final"] / tp["final"]
        # When the variant holds what the parent takes `time_to_fraction_s` to make.
        tgt["time_to_parent_amount_s"] = time_to(kv["times"], _series(kv, target), fraction * tp["final"])
    versions: dict = {}
    for j, st in enumerate(variant["steps"]):
        versions.setdefault(st.get("parent_step") or st["id"], []).append(j)
    cargo = _cargo(variant)

    def way(st, version) -> int:
        """0 if the version runs the way its parent step is written, 1 if the other way (by what its
        two sides carry; a molecule on both sides of the parent counts on neither)."""
        r, p = Counter(st["reactants"]), Counter(st["products"])
        r, p = r - p, p - r
        vr = Counter(y for x in version["reactants"] for y in cargo[x])
        vp = Counter(y for x in version["products"] for y in cargo[x])
        return 1 if (vr - vp) == p and r != p else 0

    steps = []
    for j, st in enumerate(parent["steps"]):
        mine = [(i, way(st, variant["steps"][i])) for i in versions.get(st["id"], [])]
        # Read the way the material goes (with C; without it where nothing goes with C).
        went = sum(float(kv["res"].extent[i]) * (1 - 2 * w) for i, w in mine) or float(kp["res"].extent[j])
        d = 1 if went < 0 else 0
        before = kp["apparent"][j][d]
        cand = [(kv["apparent"][i][d ^ w], variant["steps"][i]) for i, w in mine]
        cand = [x for x in cand if x[0] is not None]
        after = min(cand, key=lambda x: x[0]) if cand else (None, None)
        side = st["products"] if d else st["reactants"]
        steps.append({"step": st["id"], "label": st.get("label") or st["id"],
                      "direction": "reverse" if d else "forward",
                      **({"mirror": st.get("mirror") or st.get("mirror_of")} if st.get("mirror") or st.get("mirror_of")
                         else {}),
                      "barrier": float(st["g_ts"]) - sum(float(parent["species"][kp["pos"][x]]["g"]) for x in side),
                      "before": before, "after": after[0],
                      "change": after[0] - before if after[0] is not None and before is not None else None,
                      "through": after[1]["id"] if after[1] is not None else None,
                      # "through": its lowest version has C in it; "amounts": its own TS, the barrier moved only
                      # because its reactants' amounts moved; "none": unchanged
                      "group": ("through" if after[1] is not None and after[1]["id"] != st["id"] else
                                "amounts" if after[0] is not None and before is not None
                                and abs(after[0] - before) > 0.05 else "none"),
                      "status": (after[1].get("status") or "verified") if after[1] is not None else None,
                      "spread": (after[1] or {}).get("spread")})
    new = [{"id": s["id"], "name": s.get("name"), "kind": s["kind"], "final": float(kv["res"].c[-1, kv["pos"][s["id"]]]),
            "max": float(kv["res"].c[:, kv["pos"][s["id"]]].max()), "mean": float(kv["mean"][kv["pos"][s["id"]]]),
            "status": s.get("status")} for s in variant["species"] if s.get("kind") in ("complex", "new")]
    routes = {"parent": flux_routes(parent, kp, target), "with": flux_routes(variant, kv, target)}
    peaks = {"parent": peak_routes(parent, kp["initial"], target, limit=1),
             "with": peak_routes(variant, kv["initial"], target, limit=1)}
    made = _series(kv, target)
    carrying = routes["with"] if made is not None and float(made[-1]) > 1e-13 else ()   # (below: integration noise)
    # the levels of the steps the comparison rests on: the routes that carry the target's material with C, and the
    # lowest-peak routes without and with it
    level_of = {st["id"]: st.get("level") for st in variant["steps"] if st.get("kind") != "association"}
    on = [x for r in routes["with"] for x in r["steps"]] + [x["step"] for side in peaks.values() for r in side
                                                           for x in r["steps"]]
    by_level: dict = {}
    for sid in dict.fromkeys(x for x in on if x in level_of):
        by_level.setdefault(str(level_of[sid]), []).append(sid)
    return {"target": tgt, "steps": steps, "routes": routes, "peak_routes": peaks,
            "added": fate(variant, kv, carrying, peaks["with"][0] if peaks["with"] else None), "new_species": new,
            # True: the steps compared are not all at one level of theory (which, under `levels`)
            "mixed_levels": len(by_level) > 1, "levels": by_level, "notes": variant.get("notes") or [],
            "conditions": {"temperature": float(parent["temperature"]), "time_s": float(time_s),
                           "initial": kp["initial"], "amount": float(amount), "energies": parent.get("energies")}}


# ---------------------------------------------------------------- the bystander layer (stand-in)
def _rigid_g(symbols, coords, temperature: float) -> float:
    from mepd import thermo

    return float(thermo.qrrho([], list(symbols), np.asarray(coords, dtype=float), float(temperature))["g_corr"])


def bystander(frames: dict, molecule: dict, engine, *, temperature: float = 298.15, starts: Sequence[dict] = (),
              seed: int = 0, poses: int = 6) -> dict:
    """The molecule C beside the frames of one parent step, the step's own atoms held: C's binding to the
    reactant end, the TS and the product end, and the barrier and reaction-energy shifts.

    This is the stand-in until the measured placer lands (the contract both keep is in docs/add-molecule.md):
    `poses` random packed placements of C around the TS (mepd.complexes.packed; its conformers take
    turns), each carried unchanged to both ends, plus the proposed `starts`; C relaxed with the frame held
    (mepd.substituents.relax_group); per frame the lowest complex whose C kept its bonds. Its numbers move
    by kcal/mol with the placements (see `spread`): estimates to rank work by, not results.

    frames: {"symbols", "charge", "multiplicity", "r", "ts" (or None), "p"}, Angstrom, one atom order.
    molecule: {"symbols", "charge", "multiplicity", "conformers": [coords, ...]}, minima at the engine's level.
    starts: [{"frame": "r" | "ts" | "p", "coords": C's atoms in that frame's own Cartesian frame}].
    """
    import time

    from mepd.cli_variant import _angstrom, _node
    from mepd.complexes import packed
    from mepd.network_variants import adjacency
    from mepd.substituents import relax_group

    t0 = time.perf_counter()
    sym, q, mult = list(frames["symbols"]), int(frames.get("charge", 0)), int(frames.get("multiplicity", 1))
    csym, cq = list(molecule["symbols"]), int(molecule.get("charge", 0))
    keys = [k for k in ("r", "ts", "p") if frames.get(k) is not None]
    x = {k: np.asarray(frames[k], dtype=float) for k in keys}
    n = len(sym)
    confs = [np.asarray(c, dtype=float) for c in molecule["conformers"]]
    e_conf = [float(e) for e in engine.compute_energies([_node(csym, c, cq, molecule.get("multiplicity", 1))
                                                         for c in confs])]
    e_c, low = min(e_conf), confs[int(np.argmin(e_conf))]
    e_frame = dict(zip(keys, (float(e) for e in engine.compute_energies([_node(sym, x[k], q, mult) for k in keys]))))
    anchor = "ts" if "ts" in x else "r"
    placed = []                                       # C's coordinates, in the frames' Cartesian frame
    for k in range(int(poses)):                       # (the conformers of C take turns)
        a = _node(sym, x[anchor], q, mult).structure
        pk = _angstrom(_node_of(packed([a, _node(csym, confs[k % len(confs)], cq, 1).structure], seed=seed + k)))
        # back into the anchor frame's own coordinates (packing moved both)
        pa, xa = pk[:n] - pk[:n].mean(axis=0), x[anchor] - x[anchor].mean(axis=0)
        u, _s, vt = np.linalg.svd(pa.T @ xa)
        rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        placed.append((pk[n:] - pk[:n].mean(axis=0)) @ rot + x[anchor].mean(axis=0))
    placed += [np.asarray(s["coords"], dtype=float) for s in starts if s.get("frame") in keys]
    free = list(range(n, n + len(csym)))
    found: dict = {k: [] for k in keys}               # per frame: (energy, pose index, coords, moved)
    reacted = {k: 0 for k in keys}
    for i, xc in enumerate(placed):
        for k in keys:
            start = np.vstack([x[k], xc])
            before = adjacency(sym + csym, start)
            try:
                node = relax_group(engine, _node(sym + csym, start, q + cq, mult), free)
            except Exception:
                continue
            end = _angstrom(node)
            if not np.array_equal(before[free], adjacency(sym + csym, end)[free]):
                reacted[k] += 1
                continue
            found[k].append((float(node.energy), i, end, float(np.linalg.norm(end[n:] - xc, axis=1).max())))
    out = {"status": "ok", "why": None, "binding": {}, "complexes": {}, "molecule_energy_hartree": e_c,
           "free_method": "electronic binding + the change of the rigid-rotor and translational free energy at "
                          "1 M (no vibrations: the six new intermolecular modes are not counted, so the cost of "
                          "binding is overstated by their free energy)",
           "spread_method": "difference between the lowest binding over the even and over the odd placements",
           "level": type(engine).__name__, "poses": len(placed), "gradient_calls": None}
    for k in keys:
        if not found[k]:
            out["status"] = "reacted" if reacted[k] else "failed"
            out["why"] = (f"the molecule changed bonds in every placement beside the {_FRAME[k]}" if reacted[k]
                          else f"no placement beside the {_FRAME[k]} could be relaxed")
            continue
        if max(f[3] for f in found[k]) < 1e-3 and out["status"] == "ok":
            out["status"], out["why"] = "unmoved", f"the molecule did not move from where it was placed beside the {_FRAME[k]}"
        e, _i, xyz, _m = min(found[k], key=lambda f: f[0])
        halves = [[f[0] for f in found[k] if f[1] % 2 == h] for h in (0, 1)]
        el = (e - e_frame[k] - e_c) * 627.509474
        rigid = _rigid_g(sym + csym, xyz, temperature) - _rigid_g(sym, x[k], temperature) - _rigid_g(csym, low, temperature)
        out["binding"][k] = {"electronic": el, "free": el + rigid,
                             "spread": abs(min(halves[0]) - min(halves[1])) * 627.509474 if all(halves) else None}
        out["complexes"][k] = xyz.tolist()

    def shift(k):
        a, b = out["binding"].get(k), out["binding"].get("r")
        if not a or not b:
            return None
        return {"electronic": a["electronic"] - b["electronic"], "free": a["free"] - b["free"],
                "spread": None if a["spread"] is None or b["spread"] is None else a["spread"] + b["spread"]}

    out["barrier_shift"], out["reaction_shift"] = shift("ts"), shift("p")
    out["wall_s"] = time.perf_counter() - t0
    return out


_FRAME = {"r": "reactant end", "ts": "TS", "p": "product end"}


def _node_of(structure):
    from mepd.nodes.node import StructureNode

    return StructureNode(structure=structure)


# ---------------------------------------------------------------- the query and result files
QUERY, RESULT, VERSION = "mepd-addmol-query", "mepd-addmol-result", 1
UNITS = {"energy": "kcal/mol", "length": "angstrom", "amount": "mol/L", "time": "s", "temperature": "K"}
ADDED_ID = "added"


def match(species: Sequence[dict], text: str) -> list:
    """The ids of the species `text` names: an id, a name, or a SMILES. A SMILES without stereo marks
    names every stereoisomer (they are then counted together)."""
    hit = [s["id"] for s in species if text in (s["id"], s.get("name"))]
    if hit:
        return hit
    from rdkit import Chem

    def canon(smiles, flat):
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None:
            return None
        if flat:
            Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)

    flat = not any(ch in text for ch in "@/\\")
    want = canon(text, flat)
    return [s["id"] for s in species if want is not None and s.get("smiles") and canon(s["smiles"], flat) == want]


def query(net: dict, *, target: str, initial: dict, time_s: float, held=(), fraction: float = 0.5,
          level: Optional[dict] = None) -> dict:
    """The query file (docs/add-molecule.md) from a network whose species carry `atoms` and `coordinates`
    and whose steps carry `geometry` (mepd.web.added.query for a workspace, mepd.discovery.parent.query for an
    explorer run): the kinetics at these conditions, each species' amounts over time, each step's apparent
    barriers and extent, and what limits `target`. `target` and the keys of `initial`: ids, names or SMILES.
    ValueError when one names no species (or `initial` several)."""
    goal = match(net["species"], target)
    if not goal:
        raise ValueError(f"{target!r} is no species of the network's kinetic model")
    start = {}
    for text, amount in initial.items():
        ids = match(net["species"], text)
        if len(ids) != 1:
            raise ValueError(f"{text!r} names {len(ids)} species of the network's kinetic model")
        start[ids[0]] = float(amount)
    hold = [match(net["species"], x)[0] for x in held]
    k = run(net, start, time_s, hold)
    lim = limits(net, start, goal, time_s, held=hold, k=k)
    control = lim.get("control") or {"steps": {}, "species": {}}
    series = _series(k, goal)
    final = float(series[-1])
    resolved = final > RESOLUTION * float(k["c0"].sum())
    for s in net["species"]:
        i = k["pos"][s["id"]]
        s.update(amount={"initial": float(k["c0"][i]), "final": float(k["res"].c[-1, i]),
                         "max": float(k["res"].c[:, i].max()), "mean": float(k["mean"][i])},
                 series=k["res"].c[:, i].tolist(), control=control["species"].get(s["id"]))
    g = {s["id"]: s["g"] for s in net["species"]}
    for j, st in enumerate(net["steps"]):
        st.update(barrier={"forward": st["g_ts"] - sum(g[x] for x in st["reactants"]),
                           "reverse": st["g_ts"] - sum(g[x] for x in st["products"])},
                  apparent_barrier={"forward": k["apparent"][j][0], "reverse": k["apparent"][j][1]},
                  extent=float(k["res"].extent[j]), control=control["steps"].get(st["id"]))
    free = net["energies"] == "free"
    return {"schema": QUERY, "version": VERSION, "units": UNITS,
            "temperature": float(net["temperature"]), "energies": net["energies"],
            "energy_reference": net.get("energy_reference") or (
                "g and g_ts are absolute at the level (electronic energy"
                + (" + thermal correction, 1 M standard state" if free else "")
                + "): only differences between the two sides of a step mean anything.")
            + " barrier: the TS above the step's separated reactants (forward) or products (reverse). "
              "apparent_barrier: the TS above their chemical potentials at their time-averaged amounts.",
            "source": net.get("source") or "workspace", "level": level or {"key": None, "profile": None},
            "conditions": {"temperature": float(net["temperature"]), "time_s": float(time_s), "initial": start,
                           "held": hold},
            "target": {"species": goal, "asked": target, "fraction": fraction, "final": final, "resolved": resolved,
                       "time_to_fraction_s": time_to(k["times"], series, fraction * final) if resolved else None},
            **({"reference": net["reference"]} if net.get("reference") else {}),
            "times": k["times"].tolist(), "species": net["species"], "steps": net["steps"], "limits": lim,
            "notes": net.get("notes") or [], "left_out": len(net.get("excluded") or [])}


def _target_spread(net: dict, molecule: dict, estimates: dict, cond: dict, target, amount: float, tgt: dict
                   ) -> Optional[dict]:
    """How far the target's amount with the molecule moves when every estimate that carries a spread is moved by
    it, all the same way: {"with_low", "with_high"} (every such energy up by its spread, down by it) and the
    factors where the parent's amount is resolved. None when no estimate has a spread. An interval from the
    estimates' own spreads moved together, not a confidence interval."""
    def moved(sign):
        est = copy.deepcopy(estimates)
        n = 0

        def shift(e):
            nonlocal n
            if isinstance(e, dict) and e.get("spread"):
                n += 1
                for k in ("electronic", "free"):
                    if e.get(k) is not None:
                        e[k] = float(e[k]) + sign * float(e["spread"])

        for x in est.get("species") or []:
            shift(x.get("binding"))
        for x in est.get("steps") or []:
            for e in (x.get("binding") or {}).values():
                shift(e)
        for x in est.get("new_species") or []:
            shift(x.get("energy"))
        for x in est.get("new_steps") or []:
            shift(x.get("barrier"))
        if not n:
            return None
        variant = added(net, molecule, est)
        k = run(variant, {**cond["initial"], molecule["id"]: amount}, cond["time_s"], cond.get("held") or ())
        return float(_series(k, target)[-1])

    low = moved(+1.0)
    if low is None:
        return None
    high = moved(-1.0)
    base = (tgt.get("parent") or {}).get("final") if (tgt.get("parent") or {}).get("resolved") else None
    return {"with_low": low, "with_high": high, "factor_low": low / base if base else None,
            "factor_high": high / base if base else None,
            "what": "every estimate with a spread moved up, and down, by its spread, all together"}


def _level_labels(query: dict) -> dict:
    level = query.get("level") or {}
    key, label = level.get("key"), level.get("label") or level.get("key")
    if key is None:
        return {}
    return {key: label, f"{key}+steer": f"{label} + the run's steering single point",
            f"{key}+steer (lent)": f"{label} + a steering shift lent from its neighbours"}


def result(query: dict, candidate: dict, estimates: dict, *, status: str = "done", layers: Sequence[dict] = (),
           failed: Sequence[dict] = (), fraction: Optional[float] = None) -> dict:
    """The result file of one candidate: what was asked, every estimate with its layer, level and
    `est` / `verified`, what failed, and the read-out at the query's conditions (`readout`, plus the
    first-order estimate beside the full kinetics where a degree of control is defined)."""
    import time

    cond = query["conditions"]
    target = query["target"]["species"]
    molecule = {"id": ADDED_ID, "name": candidate.get("name") or candidate.get("smiles") or "added molecule",
                "smiles": candidate.get("smiles")}
    doc = {"schema": RESULT, "version": VERSION, "units": UNITS, "status": status,
           "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "query": {"target": query["target"], "conditions": cond},
           # what each `level` of a step stands for, in words
           "level_labels": _level_labels(query),
           "candidate": candidate, "layers": list(layers), "estimates": estimates, "failed": list(failed)}
    try:
        # (the kinetics need no geometry: leave the coordinates and time series behind)
        net = {"temperature": query["temperature"], "energies": query["energies"],
               "species": [{k: s.get(k) for k in ("id", "name", "g", "smiles")} for s in query["species"]],
               "steps": [{k: st.get(k) for k in ("id", "label", "reactants", "products", "g_ts", "status", "level",
                                                  "mirror", "mirror_of")}
                         for st in query["steps"]]}
        variant = added(net, molecule, estimates)
        out = readout(net, variant, cond["initial"], float(candidate["amount"]), target, cond["time_s"],
                      fraction=fraction or query["target"].get("fraction", 0.5), held=cond.get("held") or ())
        lim = query["limits"]
        fo = first_order(lim, shifts(net, ADDED_ID, estimates, float(candidate["amount"])), cond["temperature"])
        out["first_order"] = fo if fo is not None else {"factor": None, "why": lim.get("why")}
        if fo is not None:
            out["first_order"]["kinetics_factor"] = out["target"]["factor"]
        out["target"]["spread"] = _target_spread(net, molecule, estimates, cond, target, float(candidate["amount"]),
                                                 out["target"])
        doc["readout"] = out
    except Exception as exc:
        doc["readout"] = None
        doc["failed"].append({"layer": "kinetics", "why": f"{type(exc).__name__}: {exc}"})
    return doc


def write_json(path, doc: dict) -> None:
    """Write whole or not at all (a page or a model may be polling the file)."""
    import json
    import os
    from pathlib import Path

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    os.replace(tmp, path)
