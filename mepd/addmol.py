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


def fate(network: dict, k: dict) -> Optional[dict]:
    """What became of the added molecule C in the kinetics `k` (of `run`), by amounts (per C put in):

    * `shares` at the end: `free` (C itself), `bound` (in complexes, which fall apart without a barrier),
      `covalent` (in new species that hold C);
    * `returns`: how often a step that changed what C carries handed C back free or in a complex
      (sum over such steps of their net extent x the C on their product side held that way) / c0(C);
    * `resting_state`: the C-holding species with the largest time-averaged amount.

    Rules, in this order: `returns` and `covalent` both below 0.001 -> C took no part in any step:
    "bound and inert" if at the end at least half of C is bound, or a complex holds at least as much of a
    species as is left of it free; else "no part". Otherwise
    "regenerated" if returns >= covalent (C came back at least as often as it is held up at the end),
    else "consumed".
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
    if returns < TRACE and shares["covalent"] < TRACE:
        holds = any(s.get("kind") == "complex" and end[pos[s["id"]]] > TRACE * c0
                    and end[pos[s["id"]]] >= end[pos[s["of"][0]]] for s in holders)
        verdict = "bound and inert" if shares["bound"] >= 0.5 or holds else "no part"
    else:
        verdict = "regenerated" if returns >= shares["covalent"] else "consumed"
    return {"fate": verdict, "shares": shares, "returns": returns,
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
    side by side: the target, every parent step's apparent barrier before and after (after: the lowest
    of the step itself and its versions with C, and which), the routes that carry the target's material,
    C's resting state and fate (`fate`). "resolved" False: an amount below the kinetics' resolution."""
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
    steps = []
    for j, st in enumerate(parent["steps"]):
        before = kp["apparent"][j]
        cand = [(kv["apparent"][i][0], variant["steps"][i]) for i in versions.get(st["id"], [])
                if kv["apparent"][i][0] is not None]
        after = min(cand, key=lambda x: x[0]) if cand else (None, None)
        steps.append({"step": st["id"], "label": st.get("label") or st["id"],
                      "barrier": float(st["g_ts"]) - sum(float(parent["species"][i]["g"]) for i in
                                                         (kp["pos"][x] for x in st["reactants"])),
                      "before": before[0], "after": after[0],
                      "change": after[0] - before[0] if after[0] is not None and before[0] is not None else None,
                      "through": after[1]["id"] if after[1] is not None else None,
                      "status": (after[1] or {}).get("status") or "verified" if after[1] is not None else None,
                      "spread": (after[1] or {}).get("spread")})
    new = [{"id": s["id"], "name": s.get("name"), "kind": s["kind"], "final": float(kv["res"].c[-1, kv["pos"][s["id"]]]),
            "max": float(kv["res"].c[:, kv["pos"][s["id"]]].max()), "mean": float(kv["mean"][kv["pos"][s["id"]]]),
            "status": s.get("status")} for s in variant["species"] if s.get("kind") in ("complex", "new")]
    return {"target": tgt, "steps": steps,
            "routes": {"parent": flux_routes(parent, kp, target), "with": flux_routes(variant, kv, target)},
            "peak_routes": {"parent": peak_routes(parent, kp["initial"], target, limit=1),
                            "with": peak_routes(variant, kv["initial"], target, limit=1)},
            "added": fate(variant, kv), "new_species": new, "notes": variant.get("notes") or [],
            "conditions": {"temperature": float(parent["temperature"]), "time_s": float(time_s),
                           "initial": kp["initial"], "amount": float(amount), "energies": parent.get("energies")}}
