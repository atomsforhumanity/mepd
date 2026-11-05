"""Explore a mixture: networks of every combination of the feed molecules,
merged into one kinetic model of molecules.

The explorer (`mepd discovery expand`) follows the atoms of its seed: a
cyclopentadiene + butadiene seed makes vinylnorbornene but never the
butadiene dimer, and no two-molecule seed makes a trimer. A mixture is
therefore explored as several seeds:

- generation 0: every combination of the feed molecules of 1 to `copies`
  molecules, copies of one molecule included (CPD, BD, CPD+CPD, CPD+BD,
  BD+BD for copies=2), so reactants that react with themselves instead of
  with each other are in the same model as the target reaction;
- generation g > 0: every product molecule the merged kinetics populate
  (peak concentration >= `threshold` x the largest feed concentration),
  next to each feed molecule (VNB + CPD -> trimer).

Each seed is an ordinary explorer run in a folder of its own (rerunning
the mixture skips seeds that finished and resumes ones that stopped). Their
steps are then merged into one network of molecules:

- every species is split into its molecules (`FreeEnergies.parts`); a
  molecule that is the same before and after a step (same atoms, same stereo
  SMILES) and is not in contact with the others in the TS (partial bonds
  count: TS_CONTACT_SCALE) is a spectator and
  leaves the step, which then runs between the molecules that react (a
  rearrangement found next to a spectator is unimolecular, not bimolecular).
  One in contact in the TS stays on both sides (a shuttle: water relaying a
  proton, a catalyst);
- the TS keeps only the reacting molecules' atoms, re-optimized on them
  (it was a saddle next to the spectator, not alone) and checked by IRC to
  join the same molecules (else the spectator mattered: it stays, as a
  shuttle); its free energy is recomputed on them (electronic energy +
  quasi-RRHO from the engine's own Hessian), so a spectator's lost
  translation doesn't count against it;
- each molecule (stereo SMILES, `mepd.nodes.nodehelpers._stereo_smiles`) gets
  the free energy of its lowest-energy geometry found in any seed (the barrier
  floor); a TS found in several seeds, in either direction, counts once;
- mirror images as in the explorer (`mepd.discovery.mirror`): a chiral TS
  counts with its twin (twice between achiral molecules, else a step to the
  mirror images of its ends, which join with the same free energy); a chiral
  conformer of an achiral molecule gets -RT ln 2.

The merged kinetics (`mepd.microkinetics`, mass action in units of the standard
state) start from the feed's concentrations and decide
which products the next generation pairs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

BOHR_PER_ANGSTROM = 1.8897259886
HARTREE_TO_KCAL = 627.5094740631
GAP_ANGSTROM = 4.5          # between the molecules of a seed, closest atoms
# Two TSs of one reaction (either direction) are one TS when their electronic energies are this close and
# their sorted interatomic distances (atom order and mirror image drop out) differ by less than SAME_TS_ANGSTROM
# RMS: a TS found again in another seed was optimized next to other molecules and differs slightly.
SAME_TS_KCAL = 0.5
SAME_TS_ANGSTROM = 0.15          # as mepd.discovery.mirror; formic acid dehydration alone vs next to a second acid: 0.06
# A molecule touches the others in a TS within this x the covalent radii: partial bonds of a relay (O...H
# 1.3-1.45 A in formic acid's dimer-assisted decarboxylation) count; hydrogen bonds (~1.8 A) don't.
TS_CONTACT_SCALE = 1.5


def compositions(n_feed: int, copies: int) -> list[tuple[int, ...]]:
    """Every combination (with repeats) of 1..`copies` of `n_feed` feed molecules."""
    return [c for k in range(1, copies + 1) for c in itertools.combinations_with_replacement(range(n_feed), k)]


def place(structures: Sequence) -> object:
    """The molecules side by side in one Structure: centroids on a circle,
    neighbours GAP_ANGSTROM apart at their closest atoms (one molecule stays put)."""
    xyzs = [np.asarray(s.geometry, dtype=float) / BOHR_PER_ANGSTROM for s in structures]
    xyzs = [x - x.mean(axis=0) for x in xyzs]
    n = len(xyzs)
    if n > 1:
        reach = max(float(np.linalg.norm(x, axis=1).max()) for x in xyzs)
        r = (reach + GAP_ANGSTROM / 2) / math.sin(math.pi / n)
        xyzs = [x + r * np.array([math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n), 0.0])
                for k, x in enumerate(xyzs)]
    from qcdata import Structure

    return Structure(symbols=[str(a) for s in structures for a in s.symbols],
                     geometry=np.vstack(xyzs) * BOHR_PER_ANGSTROM,
                     charge=sum(int(s.charge) for s in structures),
                     multiplicity=1 + sum(int(s.multiplicity) - 1 for s in structures))


def _sub(structure, atoms: Sequence[int], charge: int, multiplicity: int):
    from qcdata import Structure

    xyz = np.asarray(structure.geometry, dtype=float)[list(atoms)]
    return Structure(symbols=[str(structure.symbols[k]) for k in atoms], geometry=xyz,
                     charge=int(charge), multiplicity=int(multiplicity))


def molecule_key(structure) -> str:
    """Stereo SMILES of one molecule (the network's own identity test)."""
    from mepd.nodes.node import StructureNode
    from mepd.nodes.nodehelpers import _stereo_smiles

    return _stereo_smiles(StructureNode(structure=structure))[0]


def _distances(structure) -> dict:
    """Sorted interatomic distances (Angstrom) per pair of elements."""
    sy = [str(a) for a in structure.symbols]
    X = np.asarray(structure.geometry, dtype=float) / BOHR_PER_ANGSTROM
    out: dict = {}
    for i in range(len(sy)):
        for j in range(i):
            out.setdefault(tuple(sorted((sy[i], sy[j]))), []).append(float(np.linalg.norm(X[i] - X[j])))
    return {k: np.sort(v) for k, v in out.items()}


def _same_distances(a: dict, b: dict) -> bool:
    if a.keys() != b.keys() or any(len(a[k]) != len(b[k]) for k in a):
        return False
    diff = np.concatenate([a[k] - b[k] for k in a])
    return float(np.sqrt((diff ** 2).mean())) < SAME_TS_ANGSTROM


def _geometry_hash(structure) -> str:
    xyz = np.round(np.asarray(structure.geometry, dtype=float), 4) + 0.0
    return hashlib.sha1(repr(([str(a) for a in structure.symbols], xyz.tolist())).encode()).hexdigest()


@dataclass
class Found:
    """One verified explorer step, reduced to the molecules that react."""
    reactants: list          # molecule keys
    products: list
    ts: object               # Structure: the reacting atoms of the TS
    source: str              # seed folder / step label
    shuttles: list = field(default_factory=list)
    full_ts: object = None   # the TS as found, when spectators were taken out (`settle`)
    spectators: list = field(default_factory=list)


@dataclass
class Merged:
    names: list                       # molecule keys (mirror images the seeds didn't find included)
    g: np.ndarray                     # kcal/mol
    steps: list                       # mepd.microkinetics.Step
    mirror_of: dict                   # key -> its mirror image's key
    added_mirrors: list               # keys only known as a found molecule's mirror image
    notes: list
    geometry: dict = field(default_factory=dict)   # key -> its lowest-energy geometry found


def read_seed(seed_dir: Path, parts: Callable, key: Callable = molecule_key):
    """(molecules, steps) of one finished explorer run in `seed_dir`/run:
    molecules as {key: [Structure, ...]} (every geometry each species gives),
    steps as `Found`. `parts(structure, ts)` splits a structure into
    (atoms, charge, multiplicity) per molecule."""
    from mepd.discovery.free_energy import molecules as contact_groups
    from mepd.structures import load_structure

    summary = json.loads((seed_dir / "run" / "summary.json").read_text())
    spin = json.loads((seed_dir / "seed.json").read_text()) if (seed_dir / "seed.json").is_file() else {}
    charge, mult = int(spin.get("charge", 0)), int(spin.get("multiplicity", 1))
    load = lambda f: load_structure(seed_dir / f, charge, mult)
    species = [load(s["file"]) for s in summary["species"]]
    split: list[list[tuple]] = []         # per species: (atoms frozenset, key, Structure, charge, mult)
    found_molecules: dict[str, list] = {}
    for st in species:
        mols = []
        for atoms, q, m in parts(st, False):
            sub = _sub(st, atoms, q, m)
            k = key(sub)
            mols.append((frozenset(atoms), k, sub, q, m))
            found_molecules.setdefault(k, []).append(sub)
        split.append(mols)
    steps = []
    for rec in summary.get("steps", []):
        files = rec.get("files") or {}
        if not files.get("verified") or not files.get("ts"):
            continue
        a, b = split[rec["a"]], split[rec["b"]]
        ts = load(files["ts"])
        after = {(atoms, k) for atoms, k, *_ in b}
        groups = [frozenset(g) for g in contact_groups([str(x) for x in ts.symbols],
                                                       np.asarray(ts.geometry, dtype=float) / BOHR_PER_ANGSTROM,
                                                       scale=TS_CONTACT_SCALE)]
        spectators, shuttles = [], []
        for atoms, k, sub, q, m in a:
            if (atoms, k) not in after:
                continue
            alone = all(g <= atoms or not (g & atoms) for g in groups)
            (spectators if alone else shuttles).append((atoms, k, q, m))
        gone = set().union(*[s[0] for s in spectators]) if spectators else set()
        unchanged = {s[0] for s in spectators + shuttles}        # shuttles join both sides in `merge`
        reactants = sorted(k for atoms, k, *_ in a if atoms not in unchanged)
        products = sorted(k for atoms, k, *_ in b if atoms not in unchanged)
        if reactants == products:
            continue                      # a conformer change
        core = [i for i in range(len(ts.symbols)) if i not in gone]
        core_ts = _sub(ts, core, int(ts.charge) - sum(s[2] for s in spectators),
                       int(ts.multiplicity) - sum(s[3] - 1 for s in spectators))
        steps.append(Found(reactants, products, core_ts, f"{seed_dir.name}/{rec.get('label', '')}",
                           [s[1] for s in shuttles], ts if spectators else None, [s[1] for s in spectators]))
    return found_molecules, steps


def settle(found_steps: list, engine, parts: Callable, cache_dir: Path, workers: int = 1,
           key: Callable = molecule_key) -> list:
    """A TS whose spectators were taken out was optimized next to them: re-
    optimize it on its own atoms (one imaginary mode) and check by IRC that it
    still joins the same molecules. Where that fails the spectators mattered:
    the step keeps the TS as found, with them as shuttles. Results are kept
    in `cache_dir` (one file per stripped TS). Returns notes."""
    from mepd.cli_common import _fork_map

    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in found_steps if f.full_ts is not None]

    def run(f):
        fp = cache_dir / f"{_geometry_hash(f.ts)}.json"
        if fp.is_file():
            return json.loads(fp.read_text())
        import mepd
        from mepd.nodes.node import StructureNode

        def keys(st):
            return sorted(key(_sub(st, atoms, q, m)) for atoms, q, m in parts(st, False))

        try:
            ts = mepd.find_ts(StructureNode(structure=f.ts), engine)
            ends = [keys(e.structure) for e in mepd.irc(ts, engine).ends]
            want = [sorted(f.reactants + f.shuttles), sorted(f.products + f.shuttles)]
            out = ({"ok": True, "xyz": ts.structure.to_xyz()} if sorted(ends) == sorted(want) else
                   {"ok": False, "why": f"IRC of the TS alone ends at {ends[0]} / {ends[1]}"})
        except Exception as exc:
            out = {"ok": False, "why": f"TS alone: {type(exc).__name__}: {exc}"[:300]}
        fp.write_text(json.dumps(out))
        return out

    notes = []
    for f, res in zip(todo, _fork_map(run, todo, max(1, int(workers)))):
        if isinstance(res, dict) and res.get("ok"):
            from qcdata import Structure

            f.ts = Structure.from_xyz(res["xyz"])          # charge and multiplicity are in the file
        else:
            why = res.get("why") if isinstance(res, dict) else str(res)
            notes.append(f"{f.source}: kept with its neighbours {f.spectators} as shuttles ({why})")
            f.ts, f.shuttles = f.full_ts, f.shuttles + f.spectators
        f.full_ts, f.spectators = None, []
    return notes


def merge(found_molecules: dict, found_steps: list, *, temperature: float, energies: Callable,
          corrections: Callable, chiral: Callable, mirror_key: Callable) -> Merged:
    """One network of molecules from every seed's molecules and steps.
    `energies(structures)` (Hartree), `corrections(structures, ts_flags)`
    (kcal/mol, None where the Hessian failed), `chiral(structure)` (bool or
    None), `mirror_key(structure)` (the key of its mirror image)."""
    from mepd.microkinetics import Step, rt_kcal

    notes = []
    best = {}
    for k, structs in found_molecules.items():
        uniq = list({_geometry_hash(s): s for s in structs}.values())
        e = energies(uniq)
        i = int(np.argmin(e))
        best[k] = (uniq[i], float(e[i]))
    keys = sorted(best)
    corr = corrections([best[k][0] for k in keys], [False] * len(keys))
    rt = rt_kcal(temperature)
    g, mirror_of = {}, {}
    for k, c in zip(keys, corr):
        st, e = best[k]
        if c is None:
            notes.append(f"{k}: no thermal correction (Hessian failed); electronic energy used")
        g[k] = e * HARTREE_TO_KCAL + (c or 0.0)
        mk = mirror_key(st)
        mirror_of[k] = mk
        if mk == k and chiral(st):
            g[k] -= rt * math.log(2.0)        # a chiral conformer of an achiral molecule: two mirror minima
    added = []
    for k in keys:
        mk = mirror_of[k]
        if mk not in g:
            g[mk], mirror_of[mk] = g[k], k
            added.append(mk)

    def twin(rxn):
        return tuple(sorted(mirror_of.get(k, k) for k in rxn[0])), tuple(sorted(mirror_of.get(k, k) for k in rxn[1]))

    # one TS per reaction (up to its mirror image) and electronic energy
    ts_e = energies([f.ts for f in found_steps]) if found_steps else []
    unique: dict[tuple, list] = {}
    for f, e in zip(found_steps, ts_e):
        rxn = (tuple(f.reactants), tuple(f.products))
        back = (rxn[1], rxn[0])
        canon = min(rxn, twin(rxn), back, twin(back))
        same = unique.setdefault(canon, [])
        d = _distances(f.ts)
        j = next((j for j, (_, e2, _, d2) in enumerate(same)
                  if abs(e - e2) * HARTREE_TO_KCAL < SAME_TS_KCAL and _same_distances(d, d2)), None)
        if j is None:
            same.append((f, float(e), rxn, d))
        elif e < same[j][1]:
            same[j] = (f, float(e), rxn, d)            # the lowest of one TS found several times
    flat = [item[:3] for items in unique.values() for item in items]
    ts_corr = corrections([f.ts for f, _, _ in flat], [True] * len(flat))
    names = sorted(g)
    index = {k: i for i, k in enumerate(names)}
    steps = []
    for (f, e, rxn), c in zip(flat, ts_corr):
        if c is None:
            notes.append(f"{f.source}: TS without thermal correction (Hessian failed); electronic energy used")
        g_ts = e * HARTREE_TO_KCAL + (c or 0.0)
        shuttle = [index[k] for k in f.shuttles]
        info = {"source": f.source}
        steps.append(Step([index[k] for k in rxn[0]] + shuttle, [index[k] for k in rxn[1]] + shuttle, g_ts,
                          label=f.source, info=info))
        if chiral(f.ts):
            tw = twin(rxn)
            steps.append(Step([index[k] for k in tw[0]] + shuttle, [index[k] for k in tw[1]] + shuttle, g_ts,
                              label=f"{f.source} (mirror image)", info={**info, "mirror_image": True}))
    return Merged(names, np.array([g[k] for k in names]), steps, mirror_of, added, notes,
                  {k: st for k, (st, _) in best.items()})


def kinetics(merged: Merged, c0: dict, temperature: float, time_s: float):
    """Mass-action kinetics of the merged network from `c0` ({key: M})."""
    from mepd.microkinetics import Network, simulate

    net = Network(list(merged.names), merged.g, merged.steps)
    start = np.zeros(len(merged.names))
    for k, c in c0.items():
        start[merged.names.index(k)] = c
    return net, simulate(net, start, temperature, time_s)


def _seed_name(keys: Sequence[str]) -> str:
    return hashlib.sha1("|".join(sorted(keys)).encode()).hexdigest()[:10]


def run_seed(folder: Path, structure, expand_args: Sequence[str], workers: int) -> dict:
    """One explorer run in `folder` (skipped when finished, resumed when stopped)."""
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "run" / "summary.json").is_file():
        return {"status": "done before"}
    (folder / "seed.xyz").write_text(structure.to_xyz())
    (folder / "seed.json").write_text(json.dumps({"charge": int(structure.charge),
                                                  "multiplicity": int(structure.multiplicity)}))
    cmd = [sys.executable, "-m", "mepd.cli", "discovery", "expand", "seed.xyz", "--charge", str(int(structure.charge)),
           "--multiplicity", str(int(structure.multiplicity)), *expand_args, "--workers", str(workers), "-o", "run"]
    if (folder / "run" / "expand_state.pkl").is_file():
        cmd.append("--resume")
    t0 = time.time()
    with open(folder / "expand.log", "a") as log:
        rc = subprocess.run(cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT).returncode
    ok = (folder / "run" / "summary.json").is_file()
    return {"status": "done" if ok else f"failed (exit {rc}, see expand.log)", "seconds": round(time.time() - t0)}


def explore_mixture(feed: Sequence[tuple], output: Path, expand_args: Sequence[str], engine, *,
                    temperature: float, time_s: float, standard_state: str = "1M", copies: int = 2,
                    generations: int = 2, threshold: float = 0.01, workers: int = 4, seed_workers: int = 8,
                    continuous: bool = True, log: Callable[[str], None] = print) -> dict:
    """Explore `feed` [(name, Structure, concentration M), ...] (see the module
    docstring); writes `output`/mixture.json and returns its record.

    `continuous`: no generation waits for the one before it to finish. Each
    time a seed finishes, the seeds finished so far are merged and their
    kinetics run; a product they populate is paired with each feed molecule
    at once (generation = 1 + the lowest generation of a seed that found it).
    Lower generations start first, and a queued seed starts only while its
    product is still populated in the latest merge -- otherwise it is iced
    and asked again after every merge, so a product that a later seed's
    competing channel drains costs nothing. False: generation by generation."""
    import concurrent.futures

    import mepd
    from mepd.discovery.free_energy import FreeEnergies
    from mepd.discovery.mirror import is_chiral, mirrored
    from mepd.nodes.node import StructureNode

    output.mkdir(parents=True, exist_ok=True)
    free = FreeEnergies(engine, temperature, standard_state, workers)
    feed_keys = [molecule_key(s) for _, s, _ in feed]
    c0: dict = {}
    for k, (_, _, c) in zip(feed_keys, feed):
        c0[k] = c0.get(k, 0.0) + float(c)
    best: dict = {k: s for k, (_, s, _) in zip(feed_keys, feed)}      # a geometry per molecule, for new seeds
    seeds: dict = {}

    def add_seed(keys, generation, product=None):
        name = _seed_name(keys)
        new = name not in seeds
        if new:
            seeds[name] = {"molecules": list(keys), "generation": generation,
                           **({"product": product} if product is not None else {})}
        return name, new

    for comp in compositions(len(feed), copies):
        add_seed([feed_keys[i] for i in comp], 0)
    per = max(1, min(seed_workers, workers))
    width = max(1, workers // per)

    def one(name):
        s = seeds[name]
        st = place([best[k] for k in s["molecules"]])
        return name, run_seed(output / "seeds" / name, st, expand_args, per)

    def merge_finished(gen_done: int) -> tuple:
        """Merge every finished seed; kinetics; write mixture.json. Returns (record, populated products,
        {molecule: lowest generation of a seed that found it})."""
        found_molecules: dict = {}
        found_steps: list = []
        mol_gen: dict = {}
        for name, s in seeds.items():
            folder = output / "seeds" / name
            if not (folder / "run" / "summary.json").is_file():
                continue
            mols, steps = read_seed(folder, free.parts)
            for k, structs in mols.items():
                found_molecules.setdefault(k, []).extend(structs)
                mol_gen[k] = min(mol_gen.get(k, s["generation"]), s["generation"])
            found_steps += steps
        settled = settle(found_steps, engine, free.parts, output / "settled_ts", workers)
        energy = lambda structs: mepd.energies([StructureNode(structure=s) for s in structs], engine)
        merged = merge(found_molecules, found_steps, temperature=temperature, energies=energy,
                       corrections=free.corrections, chiral=is_chiral,
                       mirror_key=lambda s: molecule_key(mirrored(s)))
        best.update({k: st for k, st in merged.geometry.items() if k not in c0})
        net, res = kinetics(merged, c0, temperature, time_s)
        cmax, cend = res.c.max(axis=0), res.c[-1]
        scale = max(c0.values())
        populated = [k for k, cm in zip(merged.names, cmax)
                     if k not in c0 and cm >= threshold * scale and k in found_molecules]
        barriers = net.barriers()
        record = {
            "feed": [{"name": n, "molecule": k, "concentration": c} for (n, _, c), k in zip(feed, feed_keys)],
            "settings": {"temperature_K": temperature, "time_s": time_s, "standard_state": standard_state,
                         "copies": copies, "generations": generations, "threshold": threshold,
                         "expand_args": list(expand_args), "continuous": bool(continuous)},
            "seeds": [{"name": n, **s} for n, s in seeds.items()],
            "molecules": [{"molecule": k, "g_kcal": float(gk), "max_concentration": float(cm),
                           "final_concentration": float(ce), "feed": k in c0,
                           "mirror_image_of": merged.mirror_of.get(k) if k in merged.added_mirrors else None}
                          for k, gk, cm, ce in zip(merged.names, merged.g, cmax, cend)],
            "reactions": [{"reactants": [merged.names[i] for i in st.reactants],
                           "products": [merged.names[i] for i in st.products],
                           "barrier_kcal": float(bf), "reverse_barrier_kcal": float(br),
                           "net_extent": float(x), **st.info}
                          for st, (bf, br), x in zip(merged.steps, barriers, res.extent)],
            "notes": settled + merged.notes + free.notes,
            "generation": gen_done}
        (output / "mixture.json").write_text(json.dumps(record, indent=1))
        log(f"Merged {sum(1 for s in seeds.values() if s.get('status', '').startswith('done'))} seed(s): "
            f"{len(merged.names)} molecules, {len(merged.steps)} steps; {len(populated)} product(s) populated.")
        return record, populated, mol_gen

    record: dict = {}
    if not continuous:
        for gen in range(generations):
            todo = [n for n, s in seeds.items() if s["generation"] == gen]
            log(f"Generation {gen}: {len(todo)} seed(s), {width} at a time, {per} workers each.")
            with ThreadPoolExecutor(max_workers=width) as pool:
                for name, res in pool.map(one, todo):
                    seeds[name].update(res)
                    log(f"  seed {name} ({' + '.join(seeds[name]['molecules'])}): {res['status']}")
            record, populated, _ = merge_finished(gen)
            if gen + 1 < generations:
                for k in populated:
                    for fk in dict.fromkeys(feed_keys):
                        add_seed([k, fk], gen + 1, product=k)
        return record

    log(f"{len(seeds)} feed seed(s), {width} at a time, {per} workers each; products are seeded as the "
        f"merged kinetics populate them (up to generation {generations - 1}).")
    queued = list(seeds)              # not started yet, in the order they were added
    populated_now: set = set()
    running: dict = {}
    with ThreadPoolExecutor(max_workers=width) as pool:
        while True:
            # lowest generation first; a product's seed only while the latest merge still populates it
            queued.sort(key=lambda n: seeds[n]["generation"])
            for name in list(queued):
                if len(running) >= width:
                    break
                prod = seeds[name].get("product")
                if prod is not None and prod not in populated_now:
                    seeds[name]["status"] = "iced (its product is not populated now)"
                    continue
                queued.remove(name)
                seeds[name]["status"] = "running"
                running[pool.submit(one, name)] = name
            if not running:
                break
            done, _ = concurrent.futures.wait(tuple(running), return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                name = running.pop(fut)
                try:
                    _, res = fut.result()
                except Exception as exc:
                    res = {"status": f"failed ({type(exc).__name__}: {exc})"}
                seeds[name].update(res)
                log(f"  seed {name} ({' + '.join(seeds[name]['molecules'])}, generation "
                    f"{seeds[name]['generation']}): {res['status']}")
            gen_done = max((s["generation"] for s in seeds.values() if s.get("status", "").startswith("done")),
                           default=0)
            record, populated, mol_gen = merge_finished(gen_done)
            populated_now = set(populated)
            for k in populated:
                g = mol_gen.get(k, 0) + 1
                if g < generations:
                    for fk in dict.fromkeys(feed_keys):
                        name, new = add_seed([k, fk], g, product=k)
                        if new:
                            queued.append(name)
    for name in queued:   # never started: their products stayed unpopulated
        seeds[name]["status"] = "not run (its product was not populated)"
    if queued and record:
        record["seeds"] = [{"name": n, **s} for n, s in seeds.items()]
        (output / "mixture.json").write_text(json.dumps(record, indent=1))
    return record
