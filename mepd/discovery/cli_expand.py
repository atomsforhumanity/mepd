"""`mepd discovery expand`: reaction network expansion from a structure by
graph rules (see mepd.discovery.network_expansion)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Optional

import typer

from mepd.chain import Chain
from mepd.discovery.cli import _progress, _validation_kwargs, discovery_app
from mepd.cpus import available_cpus


def _generator_options(generator: str, values: List[str], output: Path) -> dict:
    """--generator-option values; the nanoreactor's, the reactor's and collide's MD go into
    <output>/reactor/run_kk (the web UI's Reactor view reads it there)."""
    out = _parse_options(values)
    if str(generator).strip().lower() in ("nanoreactor", "reactor", "collide"):
        out.setdefault("workdir", str((Path(output) / "reactor").resolve()))
    return out


def _parse_options(values: List[str]) -> dict:
    out = {}
    for item in values or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--generator-option takes key=value, got {item!r}.")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def _ts_connector(run_inputs, output: Path, workers: int):
    """`connect` for flux steering: a recursive path search (MSMEP) for each
    (source, product) pair, then TS optimization + IRC of every leaf's TS
    guess. Returns one record per step whose IRC runs between two different
    minima: its two IRC ends and TS. Resumable: finished pair trees and
    TS/IRC files are reused from `output`."""
    from mepd.channel_search import _load_ts_and_irc_from_disk
    from mepd.cli_common import _fork_map, _optimize_ts_and_irc, _run_msmep_pairs, _ts_guess_tasks_from_tree
    from mepd.inputs import ChainInputs
    from mepd.TreeNode import TreeNode

    pairs_dir, ts_dir = output / "pairs", output / "ts"

    def connect(pairs, nodes):
        pairs_dir.mkdir(parents=True, exist_ok=True)
        ts_dir.mkdir(parents=True, exist_ok=True)
        charge, mult = int(nodes[0].structure.charge), int(nodes[0].structure.multiplicity)
        from mepd.network_aware import resolve_cutoff

        if resolve_cutoff(run_inputs.path_min_inputs) is not None \
                and getattr(run_inputs.path_min_inputs, "network_floor_hartree", None) is None:
            # Every path search prunes against one floor: the seed (species 0).
            if nodes[0]._cached_energy is None:
                run_inputs.engine.compute_energies([nodes[0]])
            run_inputs.path_min_inputs.network_floor_hartree = float(nodes[0].energy)
        _run_msmep_pairs([n.copy() for n in nodes], list(pairs), pairs_dir, run_inputs,
                         parallel=False, parallel_workers=None, workers=workers)
        tasks = []
        for i, j in pairs:
            tree_dir = pairs_dir / f"pair_{i}_{j}" / "tree"
            if not (tree_dir / "node_0.xyz").exists():
                typer.echo(f"No path found for reaction {i} -> {j}; it carries no flux.")
                continue
            try:
                tree = TreeNode.read_from_disk(tree_dir, chain_parameters=ChainInputs(), charge=charge,
                                               multiplicity=mult, ends_only=True)
            except Exception as exc:
                typer.echo(f"Could not read the path search for {i} -> {j}: {type(exc).__name__}: {exc}")
                continue
            tasks.extend(_ts_guess_tasks_from_tree(tree, f"pair_{i}_{j}_"))

        def ts_and_irc(task):
            label, guess = task
            if _load_ts_and_irc_from_disk(ts_dir, label, charge, mult) is None:
                _optimize_ts_and_irc(guess, run_inputs, ts_dir, run_irc=True, label=label)
            return label

        _fork_map(ts_and_irc, tasks, workers)
        found = []
        for label, _ in tasks:
            res = _load_ts_and_irc_from_disk(ts_dir, label, charge, mult)
            if res is None or res.irc_chain is None or len(res.irc_chain) < 2:
                continue
            ts = res.ts_node
            if ts._cached_energy is None:
                run_inputs.engine.compute_energies([ts])
            ends = [res.irc_chain[0], res.irc_chain[-1]]
            missing = [n for n in ends if n._cached_energy is None]
            if missing:
                run_inputs.engine.compute_energies(missing)
            found.append({"start": ends[0], "end": ends[1], "ts": ts, "label": label,
                          "files": {"ts": str(ts_dir / f"{label}.xyz"), "irc": str(ts_dir / f"{label}_irc.xyz")}})
        return found

    return connect


def _fast_connector(run_inputs, output: Path, workers: int, fallback):
    """`connect` without a path search: for each (source, product) pair, a
    geodesic interpolation, its highest image, one TS optimization (frequency-
    checked). Pairs run in parallel. A pair whose TS search fails goes to
    `fallback` (the recursive search + TS + IRC). Steps found this way are
    not IRC-verified yet (files["verified"] = False): `_irc_verifier` checks
    the ones the kinetics need."""
    import mepd
    from mepd.cli_common import _fork_map

    ts_dir = output / "ts"

    def connect(pairs, nodes):
        ts_dir.mkdir(parents=True, exist_ok=True)
        eng = run_inputs.engine

        def one(pair):
            i, j = pair
            label = f"fast_{i}_{j}"
            fp = ts_dir / f"{label}.xyz"
            if fp.exists():
                return label
            try:
                path = mepd.interpolate(nodes[i], nodes[j], nimages=11)
                en = mepd.energies(list(path.nodes), eng)
                k = max(range(1, len(en) - 1), key=lambda m: en[m])
                ts = mepd.find_ts(path.nodes[k], eng)
                mepd.write_xyz(fp, ts, comments=[f"E={ts.energy:.10f}"])
                return label
            except Exception as exc:
                (ts_dir / f"{label}.failed").write_text(f"{type(exc).__name__}: {exc}\n")
                return None

        labels = _fork_map(one, list(pairs), workers)
        found, failed = [], []
        for (i, j), label in zip(pairs, labels):
            if label is None:
                failed.append((i, j))
                continue
            ts = mepd.load(ts_dir / f"{label}.xyz")
            eng.compute_energies([ts])
            found.append({"start": nodes[i], "end": nodes[j], "pair": (i, j), "ts": ts, "label": label,
                          "files": {"ts": str(ts_dir / f"{label}.xyz"), "verified": False}})
        if failed:
            typer.echo(f"Fast connect: {len(failed)} of {len(pairs)} pair(s) had no TS from the interpolation; "
                       "running the full path search for those.")
            for rec in fallback(failed, nodes):
                rec.setdefault("files", {})["verified"] = True
                found.append(rec)
        return found

    return connect


def _horizontal(run_inputs, output: Path, workers: int, fast_ts_steps: int = 300):
    """`connect` as one shared job queue (mepd.discovery.horizontal): a fast TS
    attempt per pair, path-search pieces queued as splits happen, leaf TSs as
    jobs; no pair or branch waits for another."""
    from mepd.discovery.horizontal import HorizontalConnector

    return HorizontalConnector(run_inputs, output, workers, fast_ts_steps=fast_ts_steps, say=typer.echo)


def _channels_refiner(run_inputs, output: Path, workers: int, options: dict, max_depth: int = 0,
                      prefetch_steps: bool = False):
    """`refine` for the explorer: `mepd channels` (conformers of both ends, `explore_mechanisms` atom
    mappings, a TS + IRC per distinct path) between the two species of each step. The steps' searches run
    at once, `workers` shared between them (one step's slowest path search would otherwise idle the rest).
    Per step: {"direct": [...], "other": [...], "conformers": [...]} (records like connect's), or None if
    the search failed.

    `prefetch_steps`: `refine.prefetch(items)` starts sampling steps early, in a background process at a lower priority
    (the explorer calls it while it verifies the round's steps, whose tail leaves most cores idle);
    `refine` then reads what it finished, waits for what it is still sampling, and samples a step again
    if its species changed in between.

    `max_depth` (a profile's own recursive_split_max_depth wins): how many levels of a step's path searches
    split here. The pieces of a split at that depth come back unminimized under "deferred" ({"start",
    "end"}): the explorer searches each only once the kinetics put material at one of its ends (0: every
    piece past a mechanism's first split)."""
    import copy
    import multiprocessing
    import time

    import mepd
    from mepd import cli_common
    from mepd.cli_common import _fork_map
    from mepd.discovery.horizontal import _CoreShare

    ctx = multiprocessing.get_context("fork")
    # path searches running in all the steps, prefetched or not (shared with the forked step processes): a
    # step starts more of its searches, and a search runs its images on more threads, when others leave
    # cores idle
    searches = ctx.Value("i", 0)
    background_searches = ctx.Value("i", 0)
    background = {"proc": None, "stop": None}

    def fingerprint(a, b) -> str:
        import hashlib

        import numpy as np

        h = hashlib.sha1()
        for n in (a, b):
            h.update(np.round(np.asarray(n.coords, dtype=float), 6).tobytes())
            h.update(f"{tuple(n.structure.symbols)}{n.structure.charge}{n.structure.multiplicity}".encode())
        h.update(repr(sorted(options.items())).encode())
        h.update(f"depth={max_depth}".encode())
        return h.hexdigest()

    def widths(n: int, total: int) -> tuple[int, int]:
        # a step's path searches (one per mechanism; ~3 with explore_mechanisms 2) at once and fewer steps
        # at a time: with one worker per step its searches ran in turn, and the round waited on the slowest
        # step's three in a row
        each = max(min(3, total), total // max(1, n))
        return each, max(1, min(n, total // each))

    def sampler(each: int, stop=None):
        def sample(item):
            import pickle

            a, b, label = item
            folder = output / "channels" / f"pair_{label}"
            done_fp = folder / "explorer_record.pkl"   # this step's finished sampling, for a resumed run
            busy_fp = folder / "explorer_prefetching"
            key = fingerprint(a, b)
            if stop is None:   # the explorer's own call: a step the background is still sampling, it waits for
                while busy_fp.is_file() and _alive(background_pid(folder)):
                    time.sleep(5)
                busy_fp.unlink(missing_ok=True)
            if done_fp.is_file():
                try:
                    rec = pickle.loads(done_fp.read_bytes())
                except Exception:
                    rec = False
                if rec is not False and rec.get("key", key) == key:
                    return rec
                if rec is not False:   # sampled for other structures of these species: start over
                    k = 0
                    while folder.with_name(f"{folder.name}.stale-{k}").exists():
                        k += 1
                    folder.rename(folder.with_name(f"{folder.name}.stale-{k}"))
            if stop is not None:   # background: take a step only while the explorer hasn't asked for them
                with stop.get_lock():
                    if stop.value:
                        return None
                    folder.mkdir(parents=True, exist_ok=True)
                    busy_fp.write_text(str(os.getpid()))
            try:
                return _sample(a, b, label, folder, done_fp, key, each, background=stop is not None)
            finally:
                if stop is not None:
                    busy_fp.unlink(missing_ok=True)
        return sample

    def background_pid(folder) -> int:
        try:
            return int((folder / "explorer_prefetching").read_text())
        except (OSError, ValueError):
            return 0

    def _sample(a, b, label, folder, done_fp, key, each, background=False):
        import pickle

        profile = copy.deepcopy(run_inputs)
        if getattr(profile.engine, "n_parallel", None) is not None:
            profile.engine.n_parallel = _CoreShare(int(workers), background_searches if background else searches)
        path_min = getattr(profile, "path_min_inputs", None)
        if path_min is not None:
            # a split's pieces at the depth limit go back to the explorer unminimized, which runs each once
            # material reaches it (some searches recursed 39 levels through bands peaking hundreds of
            # kcal/mol up and held a round for an hour)
            path_min.defer_split_pieces = True
            if getattr(path_min, "recursive_split_max_depth", None) is None:
                path_min.recursive_split_max_depth = int(max_depth)
        try:
            # the 3 best-scoring conformer pairs of each mechanism (`mepd channels`' own default): every pair
            # (pairs_per_mechanism=0) searched 140 pairs on each of two Cp + methyl acrylate side steps nothing
            # reaches (174 min vs 47 min), with the same endo:exo (33:67 vs 36:64) and Suzuki product either way
            extra = {"parallel_workers": BACKGROUND_BRANCHES} if background else {}
            res = mepd.channels(a, b, output=folder, profile=profile,
                                **{"workers": each, "explore_mechanisms": 2, "pairs_per_mechanism": 3,
                                   **extra, **options})
        except Exception as exc:
            typer.echo(f"WARNING: channel sampling of step {label} failed ({type(exc).__name__}: {exc}); "
                       "it keeps its single TS.")
            return None
        rec = {"direct": [], "other": [], "conformers": [], "key": key, "deferred": _deferred_pieces(
            folder, int(a.structure.charge), int(a.structure.multiplicity))}
        for kind, chans in (("direct", res.direct), ("other", res.multistep + res.offtarget)):
            for ch in chans:
                for st in ch.steps:
                    if st.irc is None or st.ts is None:
                        continue
                    if st.ts._cached_energy is None:
                        profile.engine.compute_energies([st.ts])
                    rec[kind].append({"start": st.irc.path[0], "end": st.irc.path[-1], "ts": st.ts,
                                      "label": f"ch_{label}_{st.folder.parent.name}_{st.folder.name}",
                                      "files": {"ts": str(st.folder / "ts.xyz"), "verified": True,
                                                "irc": str(st.folder / "irc.xyz"), "channels": str(folder)}})
        for pool in (res.start_conformers, res.end_conformers):
            with_e = [n for n in pool if getattr(n, "_cached_energy", None) is not None]
            if with_e:
                rec["conformers"].append(min(with_e, key=lambda n: float(n.energy)))
        typer.echo(f"Channels of step {label}: {len(res.direct)} direct, {len(res.multistep)} multistep, "
                   f"{len(res.offtarget)} off-target exit(s), {len(rec['deferred'])} piece(s) left at the "
                   "depth limit.")
        tmp = done_fp.with_name(done_fp.name + ".tmp")
        folder.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(pickle.dumps(rec))
        tmp.replace(done_fp)
        return rec

    def run(items, stop=None):
        total = max(1, int(workers) // 2) if stop is not None else int(workers)
        each, at_once = widths(len(items), total)
        # the background (beside the explorer's own verification) gets half the lanes, counted apart: with all
        # of them, and each search's branch pool forking ~cores/3 processes, it held 57 GiB in ~1300
        # processes and the run was killed at its memory limit
        cli_common._SEARCH_SHARE = (total, background_searches if stop is not None else searches)
        try:
            return _fork_map(sampler(each, stop), items, at_once)
        finally:
            cli_common._SEARCH_SHARE = None

    def stop_background(end: bool) -> None:
        """No new steps for the background; `end`: also end it (its steps the explorer asked for are done by
        then, so what it still runs is a step the kinetics no longer need: left to resume from its folder)."""
        proc = background["proc"]
        if proc is None:
            return
        with background["stop"].get_lock():
            background["stop"].value = 1
        if end:
            if proc.is_alive():
                proc.terminate()   # its workers follow it (parent-death signal)
            proc.join()
            background["proc"] = None

    def prefetch(items):
        stop_background(end=True)
        items = list(items)
        if not items:
            return
        stop = ctx.Value("i", 0)

        def body():
            cli_common._fork_worker_init(os.getppid())
            os.nice(10)   # the explorer's own work first: this fills the cores it leaves idle
            run(items, stop)

        proc = ctx.Process(target=body, name="channels-prefetch")
        proc.start()
        if not background.get("atexit"):
            import atexit

            # not a daemon (those can't start the pools it runs), so Python would wait for it at exit: an
            # explorer that ends early (an error) ends it instead
            atexit.register(stop_background, True)
            background["atexit"] = True
        background.update(proc=proc, stop=stop)

    def refine(items):
        stop_background(end=False)
        try:
            return run(list(items))
        finally:
            stop_background(end=True)

    def pairs(item):
        """For the explorer's queue (schedule async): a step's conformers (minimized) and the atom-mapped
        conformer pairs of its mechanisms, no path search -- the queue runs each pair as a fast TS, and a
        path search only where that fails. {"start", "end", "pairs"}."""
        from mepd.channel_search import ChannelsOptions, run_channels

        a, b, label = item
        profile = copy.deepcopy(run_inputs)
        if getattr(profile.engine, "n_parallel", None) is not None:
            profile.engine.n_parallel = refine.width
        opts = ChannelsOptions(**{"workers": refine.width, "explore_mechanisms": 2, "pairs_per_mechanism": 3,
                                  **options, "conformers_only": True})
        return run_channels(a, b, profile, opts, output / "channels" / f"pair_{label}")

    # at most half the workers: two sampling jobs fit the queue at once (3 of 4 workers left the 4th idle or
    # oversubscribed behind each of them, (HCOOH)4)
    refine.pairs, refine.width = pairs, min(3, max(1, int(workers) // 2))
    if prefetch_steps:
        # off by default: the steps it guesses from the kinetics before the IRC checks are mostly not the
        # ones asked for after them (Suzuki run 30: 114 sampled, 41 asked for), and it took two thirds of
        # the cores while the verification waited for them
        refine.prefetch = prefetch
    return refine


BACKGROUND_BRANCHES = 4


def _deferred_pieces(folder: Path, charge: int = 0, multiplicity: int = 1) -> list:
    """The pieces the path searches under a channels `folder` handed back unminimized at their depth limit
    (defer_split_pieces), as {"start", "end"} nodes (each piece's band ends)."""
    from mepd.nodes.node import StructureNode
    from mepd.qcdata_structure_helpers import read_multiple_structure_from_file

    out = []
    for fp in sorted(Path(folder).glob("pairs/*/tree/node_*_deferred_*.xyz")):
        frames = read_multiple_structure_from_file(fp, charge=charge, spinmult=multiplicity)
        if len(frames) >= 2:
            out.append({"start": StructureNode(structure=frames[0]), "end": StructureNode(structure=frames[-1])})
    return out   # branch processes per path search the channels background runs (see `run`)


def _alive(pid: int) -> bool:
    """Whether process `pid` is running (a zombie, dead but not yet reaped, is not)."""
    if pid <= 0:
        return False
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _irc_verifier(run_inputs, output: Path, workers: int):
    """IRCs of unverified steps' TSs, in parallel: [(step index, start, end) or None]. Each IRC is kept
    whole, minimized ends included, with its energies (ts/<label>_irc.xyz + .energies): the path through
    the TS is what the web draws for the step."""
    import numpy as np

    import mepd
    from mepd.cli_common import _fork_map

    ts_dir = output / "ts"

    def verify(items):
        eng = run_inputs.engine

        def cut_of(ts_file, ts):
            """The TS's subsystem (network_expansion.subsystem), from the ends its search started between."""
            from mepd.discovery.network_expansion import subsystem

            ends = Path(ts_file).with_name(f"{Path(ts_file).stem}_ends.xyz")
            if not ends.exists():
                return None
            frames = mepd.read_frames(ends)
            return subsystem(frames[0], frames[-1], eng, ref=ts)

        def one(item):
            idx, ts_file = item
            label = Path(ts_file).stem
            fp = ts_dir / f"{label}_irc.xyz"
            ts = mepd.load(ts_file)
            cut = cut_of(ts_file, ts)
            if not fp.exists():
                try:
                    # ends minimized without the frequency check: its two Hessians were a fifth of an IRC
                    # check's time (7-10 s of 30-50 s, 20-atom TSs) and the connectivity is what it decides.
                    # A TS with spectators: the IRC of its subsystem, put back among them.
                    sub = cut.sub(ts) if cut else ts
                    path = mepd.irc(sub, eng, end_keywords={"check_frequencies": False}).path
                    if cut is not None and any(cut.clashes(cut.whole(n, sub)) for n in (path.nodes[0], path.nodes[-1])):
                        # an end put back in a spectator's room (Subsystem.clashes) would be minimized whole into
                        # a species made across the cut: the IRC of the whole box instead
                        cut = None
                        path = mepd.irc(ts, eng, end_keywords={"check_frequencies": False}).path
                    missing = [n for n in path.nodes if n._cached_energy is None]
                    if missing:
                        eng.compute_energies(missing)
                    if cut is None:
                        mepd.write_xyz(fp, path.nodes)
                        path.write_ene_info_to_disk(fp)
                    else:
                        mepd.write_xyz(fp, [cut.whole(n, sub) for n in path.nodes])   # (in the TS's frame)
                        np.savetxt(fp.with_suffix(".energies"), np.asarray(path.energies) + cut.env)
                except Exception as exc:
                    (ts_dir / f"{label}_irc.failed").write_text(f"{type(exc).__name__}: {exc}\n")
                    return None
            if cut is None:
                return str(fp), None
            frames = mepd.read_frames(fp)
            return str(fp), [float(e) + cut.env for e in eng.compute_energies([cut.sub(frames[0]), cut.sub(frames[-1])])]

        out = []
        for (idx, _), done in zip(items, _fork_map(one, list(items), workers)):
            if done is None:
                out.append((idx, None, None))
                continue
            fp, energies = done
            frames = mepd.read_frames(fp)
            a, b = frames[0], frames[-1]
            if energies is None:
                eng.compute_energies([a, b])
            else:   # a subsystem's IRC: its ends' energies as the TS's (no calculation on the whole)
                a._cached_energy, b._cached_energy = energies
            out.append((idx, a, b))
        return out

    return verify


@discovery_app.command("expand")
def expand(
    structure: str = typer.Argument(..., help="Seed structure: a path to an xyz file, or a SMILES string."),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True, help="RunInputs TOML (level of theory). Built-in defaults if omitted."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Override the seed's molecular charge."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="Override the seed's spin multiplicity."),
    rounds: int = typer.Option(
        1, "--rounds", help="Expansion depth: round 2 proposes products of round 1's new species, and so on. "
        "Not used by --schedule async, which grows until the kinetics ask for nothing more or a budget runs out."),
    steer: str = typer.Option(
        "auto", "--steer",
        help="Which species the next round expands. 'flux': find a verified TS (path search, TS opt, IRC) for "
        "every reaction, run microkinetics from the seed, and expand only species whose concentration flux "
        "reaches --flux-threshold, so the network stops growing on its own. 'access': verified TSs too, and "
        "expand every species some route from the seed reaches within the barrier cutoff (--cutoff-temperature "
        "/ --barrier-cutoff), over all steps found so far. 'window': every new species within --energy-window. "
        "'auto' (default): access with a cutoff, else flux when --rounds > 1, else window."),
    target: Optional[str] = typer.Option(
        None, "--target",
        help="A product (xyz in the seed's atom order, or SMILES): the seed -> target pair is searched first "
        "(two-ended recursive search); expansion then proceeds from every species within reach."),
    barrier_cutoff: Optional[float] = typer.Option(
        None, "--barrier-cutoff", help="--steer access: the cutoff (kcal/mol) directly."),
    cutoff_temperature: Optional[float] = typer.Option(
        None, "--cutoff-temperature",
        help="--steer access: the cutoff is the barrier with a half-life of --cutoff-time at this temperature "
        "(K), plus --cutoff-margin."),
    cutoff_time: float = typer.Option(86400.0, "--cutoff-time", help="Reaction time scale (s) for the cutoff."),
    cutoff_margin: float = typer.Option(10.0, "--cutoff-margin", help="kcal/mol added to the cutoff for the method's error."),
    connect_mode: str = typer.Option(
        "fast", "--connect-mode",
        help="--steer access: how each proposed reaction is connected. 'fast': one shared job queue -- a TS "
        "optimization from the peak of a geodesic interpolation per reaction, and where that fails a path search "
        "whose pieces are queued as it splits (no branch waits for another); IRCs only for steps the kinetics need. "
        "'full': recursive path search + TS + IRC for every reaction."),
    schedule: str = typer.Option(
        "rounds", "--schedule",
        help="--steer access with --connect-mode fast: 'rounds' expands the species the kinetics populate "
        "once per round. 'async': no rounds -- a species' proposals start the moment the live kinetics populate "
        "it within reach, and their reactions, IRC checks, channel sampling and steering single points join the "
        "same job queue as they are asked for. It ends when the kinetics ask for nothing more (converged), when the "
        "predicted products settled (--settle-kcal), or a budget runs out (--time-budget, --max-species, a stop "
        "request); --rounds and --max-expand do not apply."),
    time_budget: float = typer.Option(
        0.0, "--time-budget",
        help="--schedule async: seconds of wall time; then no new job starts, the running ones are cancelled and the "
        "network found so far is written with its kinetics. 0 (default): no limit."),
    settle_kcal: float = typer.Option(
        0.0, "--settle-kcal",
        help="--schedule async: stop once the predicted product distribution has settled ('settled'): no populated "
        "product's ratio to another (RT ln, kcal/mol) moved by more than this over the last 30 s and the last 10 "
        "live kinetics, the seed's proposals are all searched and no steering single point is out. Then it ends "
        "as on --time-budget. 0 (default): off, run until converged or a budget runs out. 0.3 is the value tested "
        "on recorded runs; not yet gated on the cluster."),
    max_expand: int = typer.Option(
        10, "--max-expand",
        help="--steer access: at most this many species expanded per round (the most populated first, among "
        "those within reach and with peak concentration or flux >= --flux-threshold)."),
    reroute_attempts: int = typer.Option(
        3, "--reroute-attempts",
        help="--target with --steer access: while the target can't be reached within the cutoff (the route found "
        "climbs too high), this many of the most accessible species per round get a two-ended search towards it, "
        "and the target's side is expanded too. 0 = off."),
    temperature: float = typer.Option(298.15, "--temperature", help="--steer flux: temperature (K) for the rates."),
    time_s: float = typer.Option(3600.0, "--time", help="--steer flux: simulated time (s), starting from pure seed."),
    kinetics_energy: str = typer.Option(
        "free", "--kinetics-energy",
        help="--steer flux/access: rates from 'free' energies (quasi-RRHO from the engine's own Hessians, a species "
             "of several molecules counted as those molecules apart) or 'electronic' energies."),
    refine_mode: str = typer.Option(
        "channels", "--refine",
        help="--steer access: 'channels' (default) runs `mepd channels` -- conformers of both ends and 2 atom "
             "mappings (mechanisms), the 3 best conformer pairs of each searched (--channels-option "
             "pairs_per_mechanism=0 searches all) -- on every "
             "step the kinetics need and every pair whose TS failed its IRC check; its channels replace the "
             "step's single TS and its off-target exits and multistep routes join the network. 'none' keeps "
             "single TSs."),
    channels_option: List[str] = typer.Option(
        [], "--channels-option", help="key=value for the channel searches (`mepd channels` option names with "
                                      "underscores, e.g. n_conformers=20). Repeatable."),
    channels_prefetch: bool = typer.Option(
        False, "--channels-prefetch/--no-channels-prefetch",
        help="Start sampling the channels of the steps the kinetics need already while their IRCs are checked "
             "(in the background, half the cores). Off: the IRC checks change which steps are needed."),
    channels_max_depth: int = typer.Option(
        0, "--channels-max-depth",
        help="How many levels the channel searches split a path themselves (a profile's "
             "recursive_split_max_depth wins). The pieces of a split at that depth are searched later by the "
             "connector, each only once the kinetics put material at one of its ends (0: every piece past a "
             "mechanism's first split)."),
    standard_state: str = typer.Option("1M", "--standard-state",
                                       help="With --kinetics-energy free: '1M' (solution) or '1atm' (gas)."),
    mirror_images: bool = typer.Option(
        True, "--mirror-images/--no-mirror-images",
        help="Count every species and transition state with its mirror image (x -> -x, same energy): a chiral TS "
        "whose twin was not found gets one, leading to the mirror images of its ends; a chiral conformer of an achiral "
        "molecule gets -RT ln 2. The explorer finds one twin or both by chance; this removes that bias."),
    flux_threshold: float = typer.Option(
        0.01, "--flux-threshold",
        help="--steer flux: expand a species once the material that flowed into it (fraction of the seed) "
        "reaches this. --steer access: tau of the molecule-level criteria, relative to the conversion (a molecule's "
        "own reactions once its net flux, an encounter once its two molecules' peak concentrations multiplied, "
        "reach tau x the fraction of the seed converted)."),
    exact: bool = typer.Option(
        False, "--exact",
        help="--steer access: keep the system exactly as given, whatever --copies says."),
    n_copies: int = typer.Option(
        1, "--copies",
        help="--steer access: start from this many copies of the given system side by side, 1 (the default for now) "
        "or 2 (each molecule meets a copy of itself and of the others, and what one copy makes meets the other's "
        "molecules); one copy when two would exceed 50 atoms or the system is charged or open-shell."),
    concentration: float = typer.Option(
        1.0, "--concentration",
        help="--steer access: each of the seed's molecules' starting concentration, in the standard state's units "
        "(M for 1M, atm for 1atm), for the molecule-level mass-action kinetics."),
    n_break: int = typer.Option(2, "--n-break", help="Most bonds broken in one proposed step."),
    n_form: int = typer.Option(2, "--n-form", help="Most bonds formed in one proposed step."),
    form_distance: float = typer.Option(
        4.0, "--form-distance", help="Only form bonds between atoms at most this far apart (Angstrom) in the source."),
    max_products: int = typer.Option(
        50, "--max-products", help="Proposals kept per source species (fewest bond changes first)."),
    energy_window: float = typer.Option(
        60.0, "--energy-window",
        help="Only species within this many kcal/mol of the seed are expanded in the next round."),
    allow_radicals: bool = typer.Option(
        False, "--allow-radicals/--no-allow-radicals",
        help="Also propose products with unpaired electrons beyond the multiplicity (e.g. carbenes, diradicals)."),
    allow_zwitterions: bool = typer.Option(
        False, "--allow-zwitterions", help="Also propose products that need separated formal charges."),
    generator: str = typer.Option(
        "auto", "--generator",
        help="Which tool proposes the products: 'nanoreactor' with --generator-option forces=piston,collide, "
        "forces=piston or forces=collide (Langevin MD on axtb with those forces: the piston squeezes the wall, "
        "collide pulls atoms or molecules together; piston,collide is 'reactor', collide alone 'collide'); "
        "'nanoreactor' without forces (xtb's hot piston MD of the species; with --steer access "
        "the default, 8 trajectories x 5 ps at 2500 K unless --generator-option says otherwise), 'bond-rules' "
        "(built in: the ZStruct/YARP break-and-form enumeration, reimplemented), or 'package.module:function' for "
        "your own: function(structure, **options) returning product Structures (or xyz paths) in the same atom "
        "order; 'collide' (the Reactor's Collide: steered GFN2 MD pulling atom pairs together, with a bond dragged "
        "apart); 'reactor' (the Reactor unattended: piston MD with random collisions, 2200 K, one 5 ps trajectory per "
        "worker). 'auto' (default): with --steer access, collide for one molecule and nanoreactor for several; "
        "else bond-rules."),
    generator_option: List[str] = typer.Option(
        [], "--generator-option", help="key=value passed to a custom --generator (values parsed as JSON). Repeatable."),
    products: Optional[Path] = typer.Option(
        None, "--products", exists=True, dir_okay=False,
        help="Import products another tool proposed (autodE, Chemoton, ...): a multi-frame xyz, "
        "atom-mapped to the seed (same atoms, same order). Replaces the generator for the seed."),
    maxiter: int = typer.Option(500, "--maxiter", help="Maximum geometry-optimization steps per proposal."),
    proposal_fmax: float = typer.Option(
        0.05, "--proposal-fmax",
        help="Minimize proposals only to this largest force (eV/Angstrom): enough to tell the species and its "
        "energy, and a floppy complex no longer holds up its round. Species the free-energy kinetics need are "
        "minimized tightly first. 0: tight from the start."),
    fast_ts_steps: int = typer.Option(
        300, "--fast-ts-steps",
        help="Steps for the single TS optimization tried first per pair (--steer access --connect fast); "
        "one that hasn't converged by then fails and its pair gets a path search."),
    refine_energies: Optional[Path] = typer.Option(
        None, "--refine-energies", exists=True, dir_okay=False,
        help="A profile (.toml) of a higher level of theory, e.g. DFT: after the exploration, every species and TS "
        "the kinetics use gets its energy at that level and the kinetics run again (refined_energies.json). "
        "See --refine-energies-mode."),
    refine_energies_mode: str = typer.Option(
        "single-point", "--refine-energies-mode",
        help="'single-point' (default): the explorer's geometries and thermal corrections, the higher level's "
        "electronic energies. 'reoptimize': minima and TSs re-converged at the higher level (each TS with an IRC "
        "that must reach the same species) and its own Hessians -- much more expensive."),
    steer_energies: str = typer.Option(
        "auto", "--steer-energies",
        help="--steer flux/access: a profile (.toml) of a higher level of theory whose single points steer the "
        "exploration: the kinetics that decide which species are expanded (and which steps get channel "
        "sampling) use G = G_explorer + (E_high - E_low) on every species and TS they treat, each geometry "
        "computed once (kept in steer_single_points/). 'auto' (default): the --refine-energies profile when "
        "given; 'none': the explorer's own energies."),
    max_species: int = typer.Option(400, "--max-species", help="Stop adding species beyond this many."),
    workers: int = typer.Option(
        min(4, max(1, available_cpus() // 2)), "--workers",
        help="Processes at once: product guesses, the web live view's animations, path-search jobs, and "
        "engine calculations (g-xTB) unless the engine's own n_parallel is set."),
    validate_minima_with_hessian: bool = typer.Option(
        False, "--validate-minima-with-hessian/--no-validate-minima-with-hessian", "-H/-noH",
        help="Hessian-check every new species (rescue push along an unstable mode; dropped if still not a minimum)."),
    hessian_minimum_frequency_cutoff: float = typer.Option(
        -20.0, "--hessian-minimum-frequency-cutoff", help="Minimum allowed frequency (cm^-1)."),
    hessian_minima_rescue_displacement: float = typer.Option(
        0.3, "--hessian-minima-rescue-displacement", help="Rescue push along the unstable mode, both ways (bohr)."),
    connect: bool = typer.Option(
        False, "--connect/--no-connect",
        help="Then run a recursive path search (MSMEP, as in `mepd network-splits`) for every proposed "
        "reaction that landed on a different species, and build the network from the paths found."),
    max_pairs: int = typer.Option(50, "--max-pairs", help="Cap on path searches for --connect."),
    parallel: bool = typer.Option(False, "--parallel", help="--connect: evaluate MSMEP branches in parallel."),
    parallel_workers: Optional[int] = typer.Option(None, "--parallel-workers", help="Workers for --parallel."),
    output: Path = typer.Option(Path("mepd_expand_output"), "--output", "-o", help="Directory to write results into."),
    known: List[Path] = typer.Option(
        [], "--known", file_okay=False,
        help="The output folder of an earlier `mepd discovery expand` at the same level of theory (its "
        "provenance.json's level_key), repeatable: its species (no re-optimization), steps with their TS and IRC "
        "(a pair with a step gets no new TS or path search; an unverified one still gets its IRC check), channel "
        "sampling, steering single points (at the same steering level) and Hessians (same temperature and standard "
        "state) are taken as known, and the kinetics start from its network. The seed must be one of its species "
        "(its atoms are mapped onto the seed's); a folder at another level, or one that can't be mapped, is said "
        "and skipped. Species it took from them don't count towards --max-species."),
    resume: bool = typer.Option(
        False, "--resume",
        help="Carry on an exploration that stopped (a time limit, a crash) from what it saved in --output "
        "(expand_state.pkl, written at every stage of a round and every couple of minutes while connecting): "
        "only the path searches of pairs it had not finished run again. Give the same settings; --rounds can "
        "be raised to grow a finished network further."),
) -> None:
    """Reaction network expansion: propose products of the seed by breaking
    and forming bonds (no Hessian sampling), optimize them, and repeat from
    the new species; optionally connect each proposed reaction by a path
    search. With --rounds > 1 the network grows only where the kinetics
    say material flows (--steer flux).

    Based on published methods, reimplemented in mepd (no code from them is
    used): the break/form enumeration on the bond graph is that of ZStruct
    (Zimmerman, J. Comput. Chem. 2013) and YARP (Zhao & Savoie, Nat. Comput.
    Sci. 2021); the Lewis-structure filter is RDKit DetermineBondOrders, i.e.
    xyz2mol (Kim & Kim, Bull. Korean Chem. Soc. 2015); flux steering follows
    Bensberg & Reiher (Isr. J. Chem. 2023) and the rate-based enlargement of
    Susnow et al. (J. Phys. Chem. A 1997). Product guesses use a restrained
    relaxation of mepd's own. Full references in summary.json.

    Writes species.xyz (the seed first, then every species found),
    species/species_<k>.xyz, proposals.xyz (every product guess, before
    optimization), rejected.xyz (if Hessian-checked) and summary.json (the
    species, and one record per proposed reaction: source, target, bonds
    broken/formed, outcome). With --connect, also pairs/ and network.json.
    """
    from mepd.cli_common import (
        _completed_tree_dirs, _echo_run_inputs_summary, _load_structure_from_smiles_or_xyz, _open_run_inputs,
        _run_msmep_pairs,
    )
    from mepd.discovery.network_expansion import REFERENCES, SETTLE_KCAL, expand_network
    from mepd.discovery.kinetics import negative_barrier_warning, raw_barriers
    from mepd.nodes.node import StructureNode

    for name, value in (("--rounds", rounds), ("--max-products", max_products), ("--maxiter", maxiter),
                        ("--max-species", max_species), ("--max-pairs", max_pairs)):
        if value <= 0:
            raise typer.BadParameter(f"{name} must be a positive integer.")
    if n_break < 0 or n_form < 0 or n_break + n_form == 0:
        raise typer.BadParameter("--n-break and --n-form must be >= 0 and not both 0.")

    from mepd.discovery.generators import get_generator, is_named, resolve_forces

    # Network-aware exploration proposes products by nanoreactor MD, several
    # short trajectories per species (bond-change enumeration capped at N
    # proposals missed relevant first steps; see netaware_toy/README.md).
    steer_requested = str(steer).strip().lower()
    wants_access = steer_requested == "access" or (
        steer_requested == "auto" and (isinstance(barrier_cutoff, (int, float)) or isinstance(cutoff_temperature, (int, float))))
    if str(generator).strip().lower() == "auto":
        generator = "bond-rules"
        if wants_access:
            # One molecule: collisions (the Reactor's Collide) -- they make the
            # unimolecular decompositions piston MD rarely does (formic acid:
            # nothing at 2500 K). Several molecules: multi-seed nanoreactor.
            import numpy as np

            from mepd.discovery.free_energy import molecules

            st = _load_structure_from_smiles_or_xyz(structure, charge, multiplicity)
            n_mol = len(molecules([str(x) for x in st.symbols], np.asarray(st.geometry) / 1.8897259886))
            generator = "collide" if n_mol == 1 else "nanoreactor"
        if generator == "nanoreactor":
            given = {o.partition("=")[0].strip() for o in (generator_option or [])}
            defaults = {"n_seeds": "8", "time_ps": "5", "temperature": "2500", "parallel": str(max(1, workers))}
            generator_option = list(generator_option or []) + [f"{k}={v}" for k, v in defaults.items() if k not in given]

    try:   # nanoreactor forces=...: the reactor or collide (generators.resolve_forces)
        generator, generator_option = resolve_forces(generator, generator_option)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--generator-option") from None
    if str(generator).strip().lower() in ("collide", "reactor"):
        # its processes follow --workers (a pod's os.cpu_count() is the node's, not the pod's)
        given = {o.partition("=")[0].strip() for o in (generator_option or [])}
        if "parallel" not in given:
            generator_option = list(generator_option or []) + [f"parallel={max(1, workers)}"]
    if products is None and is_named(generator):
        try:
            get_generator(generator).check()
        except ImportError as exc:
            raise typer.BadParameter(str(exc), param_hint="--generator") from None
    elif products is None and ":" not in generator and "." not in generator:
        try:
            get_generator(generator)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--generator") from None

    from mepd.network_aware import resolve_cutoff

    num = lambda v, default=None: float(v) if isinstance(v, (int, float)) else default  # OptionInfo when called directly
    # Sella moves its large eigensolves to the GPU whenever torch sees one (dimension >= 200: ~70+ atoms). In this
    # process that initialises CUDA, after which it can't fork its workers and every path search runs one at a
    # time (a 48-core pod at ~3 cores). Off unless the user set it; a GPU steering level has a process of its own.
    os.environ.setdefault("SELLA_DISABLE_GPU", "1")
    from mepd import stackdump

    stackdump.start()
    steer = steer.strip().lower()
    if steer not in ("auto", "window", "flux", "access"):
        raise typer.BadParameter("--steer must be auto, window, flux or access.")
    if str(refine_energies_mode) not in ("single-point", "reoptimize"):
        raise typer.BadParameter("--refine-energies-mode must be single-point or reoptimize.")
    high_engine = _open_run_inputs(refine_energies).engine if isinstance(refine_energies, Path) else None
    steer_with = str(steer_energies).strip() if isinstance(steer_energies, str) else "auto"
    if steer_with.lower() == "auto":
        steer_engine = high_engine
    elif steer_with.lower() == "none":
        steer_engine = None
    else:
        if not Path(steer_with).is_file():
            raise typer.BadParameter(f"{steer_with} is not a file.", param_hint="--steer-energies")
        steer_engine = _open_run_inputs(Path(steer_with)).engine
    run_inputs = _open_run_inputs(inputs)
    pmi = run_inputs.path_min_inputs
    if num(barrier_cutoff) is not None:
        pmi.barrier_cutoff_kcal = num(barrier_cutoff)
    if num(cutoff_temperature) is not None:
        pmi.cutoff_temperature = num(cutoff_temperature)
        pmi.cutoff_time_s = num(cutoff_time, 86400.0)
        pmi.cutoff_margin_kcal = num(cutoff_margin, 10.0)
    access_cutoff = resolve_cutoff(pmi)
    if steer == "auto":
        steer = "access" if access_cutoff is not None else ("flux" if rounds > 1 else "window")
    if steer == "access" and access_cutoff is None:
        raise typer.BadParameter("--steer access needs --cutoff-temperature or --barrier-cutoff.")
    if steer == "access" and num(cutoff_temperature) is not None:
        temperature, time_s = num(cutoff_temperature), num(cutoff_time, 86400.0)   # the kinetics report's conditions
    if steer in ("flux", "access") and (temperature <= 0 or time_s <= 0 or flux_threshold <= 0):
        raise typer.BadParameter("--temperature, --time and --flux-threshold must be positive.")
    if access_cutoff is not None:
        os.environ.setdefault("MEPD_NETWORK_AWARE_LOG", str(Path(output) / "network_aware.jsonl"))
        Path(output).mkdir(parents=True, exist_ok=True)
        typer.echo(f"Network-aware exploration: barrier cutoff {access_cutoff:.1f} kcal/mol.")
    state_fp = Path(output) / "expand_state.pkl"
    if resume is True and not state_fp.is_file():
        raise typer.BadParameter(f"nothing to resume: {state_fp} doesn't exist.", param_hint="--resume")
    if resume is not True and state_fp.is_file():
        typer.echo(f"Note: {output} holds a saved exploration; this run starts over (--resume carries it on).")
    _echo_run_inputs_summary(run_inputs)
    if getattr(run_inputs.engine, "holds_torch_model", False) and hasattr(run_inputs.engine, "server"):
        # A GPU model as the explorer's own level: one process of its own holds it, and this process and its
        # workers send it their structures (CUDA here would stop every parallel stage from forking).
        from mepd.engines.fairchem import serve

        serve(run_inputs.engine)
        typer.echo(f"Model server for {run_inputs.engine!r} started.")
    seed_structure, copies = _load_structure_from_smiles_or_xyz(structure, charge, multiplicity), 1
    n_copies = n_copies if isinstance(n_copies, int) else 1
    conc_given = concentration if isinstance(concentration, (int, float)) else 1.0
    if n_copies not in (1, 2):
        raise typer.BadParameter("--copies takes 1 or 2")
    if not math.isfinite(conc_given) or conc_given <= 0:
        raise typer.BadParameter("--concentration must be a number above 0")
    if steer != "access":
        asked = [o for o, given in (("--copies", n_copies != 1), ("--exact", exact is True),
                                    ("--concentration", conc_given != 1.0)) if given]
        if asked:
            typer.echo(f"WARNING: {', '.join(asked)} only act with --steer access (here: --steer {steer}); ignored.")
    if steer == "access":
        from mepd.discovery.molecules import one_copy_why, seed_box

        eng = run_inputs.engine
        boxed = getattr(eng, "region", None) is not None or bool(getattr(eng, "restraints", None))
        seed_structure, copies = seed_box(seed_structure, exact=exact is True or n_copies == 1 or boxed)
        if n_copies == 2 and copies == 1 and exact is not True:
            why = ("its engine is QM/MM or restrained (they number the given system's atoms)" if boxed
                   else one_copy_why(seed_structure))
            typer.echo(f"WARNING: --copies 2: one copy, as {why}.")
        typer.echo(f"Seed box: {'two copies of the given system side by side' if copies == 2 else 'the system as given'}"
                   f" ({len(seed_structure.symbols)} atoms{'; --exact keeps it as given' if copies == 2 else ''}).")
    seed = StructureNode(structure=seed_structure)
    target_node = (StructureNode(structure=_load_structure_from_smiles_or_xyz(target, charge, multiplicity))
                   if isinstance(target, str) and target else None)
    # A plain line after the settings table: what runs next (the live view
    # shows the latest line, and a first step such as CREST can be silent).
    typer.echo(f"Expanding the network from the seed: {rounds} round(s), products proposed by "
               f"{products and 'the --products file' or generator}.")
    validation = _validation_kwargs(validate_minima_with_hessian, hessian_minimum_frequency_cutoff,
                                    hessian_minima_rescue_displacement)

    from rich.console import Console

    console = Console()
    with _progress() as progress:
        task = {"id": None}

        def on_event(event, payload):
            if event == "proposed":
                if task["id"] is not None:
                    progress.remove_task(task["id"])
                task["id"] = progress.add_task(f"Round {payload['round']}: optimizing {payload['total']} proposal(s)",
                                               total=payload["total"] or None)
            elif event == "candidate_done" and task["id"] is not None:
                progress.update(task["id"], completed=payload["index"])
            elif event == "steer_single_points":
                progress.console.print(f"Round {payload['round']}: {payload['total']} steering single point(s)",
                                       highlight=False)
            elif event == "warning":
                progress.console.print(f"[bold red]WARNING[/bold red] {payload['message']}", highlight=False)
            elif event == "resumed":
                progress.console.print(f"Resuming round {payload['round']} ({payload['stage']}): "
                                       f"{payload['species']} species, {payload['steps']} steps saved.",
                                       highlight=False)
            elif event == "resumed_connect":
                progress.console.print(f"Resuming the connect call it was cut off in: {payload['pairs']} pair(s) "
                                       f"left, {payload['finished']} finished.", highlight=False)
            elif event == "known":
                progress.console.print(f"Known: {payload['message']}", highlight=False)
            elif event == "species_found":
                progress.console.print(f"[green]✓ species {payload['index']}[/green] {payload['smiles']}  "
                                       f"ΔE={payload['rel_energy_kcal']:+.1f} kcal/mol", highlight=False)

        def checkpoint(res):
            # The network after each round, in summary.json's own keys: a run
            # stopped later (a time limit) still has a network and its kinetics.
            e0 = float(res.species[0].node.energy)
            part = {"partial": True, "seed_energy": e0, "rounds": res.rounds,
                    "species": [{"index": k, "smiles": sp.smiles, "round": sp.round, "energy": float(sp.node.energy),
                                 "rel_energy_kcal": sp.rel_energy_kcal, "t_found": getattr(sp, "t_found", None)}
                                for k, sp in enumerate(res.species)],
                    # (t_found, t_opened, irc, placeholder_reason: each step's record so far)
                    "steps": [{"a": st.a, "b": st.b, "ts_energy": st.ts_energy, "label": st.label, "files": st.files,
                               **getattr(st, "rec", {})} for st in res.steps]}
            output.mkdir(parents=True, exist_ok=True)
            tmp = output / "summary_partial.json.tmp"
            tmp.write_text(json.dumps(part, default=str))
            tmp.replace(output / "summary_partial.json")

        if getattr(run_inputs.engine, "n_parallel", None) == 0:
            # at most --workers engine processes at once (os.cpu_count() is the node's, not a container's limit)
            run_inputs.engine.n_parallel = max(1, int(workers))
        known_spec = None
        if isinstance(known, list) and known:
            from mepd.discovery.known import read_known
            from mepd.web.workspace import level_key

            runs, skipped = read_known(known, level_key(Path(inputs).read_text() if inputs else None), output,
                                       say=lambda m: progress.console.print(f"Known: {m}", highlight=False))
            known_spec = {"runs": runs, "skipped": skipped, "dir": output / "ts"}
        try:
            result = expand_network(
                seed, run_inputs.engine, rounds=rounds, energy_window_kcal=energy_window, generator=generator,
                generator_options=_generator_options(generator, generator_option, output), products_file=str(products) if products else None,
                maxiter=maxiter, n_break=n_break, n_form=n_form, form_distance=form_distance,
                max_products=max_products, allow_radicals=allow_radicals, allow_zwitterions=allow_zwitterions,
                max_species=max_species, workers=workers, on_event=on_event, steer=steer,
                connect=(_horizontal(run_inputs, output, workers,
                                     int(fast_ts_steps) if isinstance(fast_ts_steps, int) else 300)
                         if steer == "access" and str(connect_mode).lower() == "fast"
                         else _ts_connector(run_inputs, output, workers) if steer in ("flux", "access") else None),
                verify=(_irc_verifier(run_inputs, output, workers)
                        if steer == "access" and str(connect_mode).lower() == "fast" else None),
                target=target_node, access_cutoff_kcal=access_cutoff if steer == "access" else None,
                reroute_attempts=int(reroute_attempts) if isinstance(reroute_attempts, int) else 3,
                max_expand_per_round=int(max_expand) if isinstance(max_expand, int) else 10,
                schedule=str(schedule).lower() if isinstance(schedule, str) else "rounds",
                time_budget_s=float(time_budget) if isinstance(time_budget, (int, float)) and time_budget > 0 else None,
                settle_kcal=float(settle_kcal) if isinstance(settle_kcal, (int, float)) else None,
                checkpoint=checkpoint,
                proposal_fmax=float(proposal_fmax) if isinstance(proposal_fmax, (int, float)) else 0.05,
                refine=(_channels_refiner(run_inputs, output, workers,
                                          _parse_options(channels_option) if isinstance(channels_option, list) else {},
                                          int(channels_max_depth) if isinstance(channels_max_depth, int) else 0,
                                          channels_prefetch is True)
                        if steer == "access" and str(refine_mode).lower() == "channels" else None),
                kinetics={"temperature": temperature, "time_s": time_s, "threshold": flux_threshold,
                          "free_energy": str(kinetics_energy).lower() == "free" if isinstance(kinetics_energy, str)
                          else True,
                          "standard_state": standard_state if isinstance(standard_state, str) else "1M",
                          "mirror_images": mirror_images is not False,
                          "concentration": float(concentration) if isinstance(concentration, (int, float)) else 1.0,
                          "copies": copies},
                steer_engine=steer_engine if steer in ("flux", "access") else None,
                steer_save_dir=output / "steer_single_points",
                state_file=output / "expand_state.pkl", resume=resume is True, known=known_spec,
                validate_minima={"frequency_cutoff": validation["hessian_minimum_frequency_cutoff"],
                                 "rescue_displacement": validation["hessian_minima_rescue_displacement"]}
                if validation else None,
            )
        except Exception as exc:
            typer.echo(f"Network expansion failed: {type(exc).__name__}: {exc}")
            if state_fp.is_file():
                typer.echo(f"What it had done is saved: rerun with --resume to carry on from {state_fp}.")
            raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    write_qcio = bool(getattr(run_inputs, "write_qcio", False))

    def write(nodes, fp):
        if nodes:
            Chain.model_validate({"nodes": [n.copy() for n in nodes], "parameters": run_inputs.chain_inputs}
                                 ).write_to_disk(fp, write_qcio=write_qcio)
            return str(fp)
        return None

    species_nodes = [s.node for s in result.species]
    files = {"species": write(species_nodes, output / "species.xyz"),
             "proposals": write([StructureNode(structure=e.proposal.structure) for e in result.edges
                                 if e.proposal.structure is not None], output / "proposals.xyz"),
             "rejected": write(result.rejected, output / "rejected.xyz")}
    (output / "species").mkdir(exist_ok=True)
    species_files = [output / "species" / f"species_{k}.xyz" for k in range(len(species_nodes))]
    for node, fp in zip(species_nodes, species_files):
        write([node], fp)

    pairs = result.connections()
    summary = {
        "structure": structure, "inputs": str(inputs) if inputs else None,
        "settings": {"rounds": rounds, "generator": generator if products is None else f"file:{products}",
                     "n_break": n_break, "n_form": n_form, "form_distance_angstrom": form_distance,
                     "max_products": max_products, "energy_window_kcal": energy_window,
                     "allow_radicals": allow_radicals, "allow_zwitterions": allow_zwitterions,
                     "hessian_validation": bool(validation)},
        "seed_energy": float(result.species[0].node.energy) if result.species[0].node._cached_energy is not None else None,
        "species": [{"index": k, "smiles": s.smiles, "round": s.round, "energy": float(s.node.energy),
                     "rel_energy_kcal": s.rel_energy_kcal, "loosely_minimized": s.loose, "validation": s.validation, "file": str(species_files[k]),
                     **({"known": s.known} if getattr(s, "known", None) else {})}
                    for k, s in enumerate(result.species)],
        "reactions": [{"source": e.source, "target": e.target, "outcome": e.outcome, "proposed_smiles": e.proposal.smiles,
                       "broken": [list(b) for b in e.proposal.broken], "formed": [list(f) for f in e.proposal.formed],
                       "landed_as_proposed": e.intended, "error": e.error or None,
                       # recorded for the measuring instruments (docs/network-expansion.md, "What a run records")
                       "t": getattr(e, "t", None), "batch": getattr(e, "batch", None),
                       "proposed_key": getattr(e.proposal, "key", None),
                       "not_searched": getattr(e, "not_searched", None)} for e in result.edges],
        # A named generator swaps in its own enumeration reference; imported
        # products and custom generators only share the animation.
        "methods": {**({k: v for k, v in REFERENCES.items() if k != "kinetics"}
                       if products is None and is_named(generator)
                       else {k: v for k, v in REFERENCES.items() if k == "animation"}),
                    **(get_generator(generator).references if products is None and is_named(generator) else {}),
                    **({"kinetics": REFERENCES["kinetics"]} if steer in ("flux", "access") else {})},
        "rounds": result.rounds,
        "steering": {"mode": steer, **({"temperature_K": temperature, "time_s": time_s, "flux_threshold": flux_threshold}
                                        if steer == "flux" else
                                        {"cutoff_kcal": access_cutoff, "temperature_K": temperature, "time_s": time_s}
                                        if steer == "access" else {"energy_window_kcal": energy_window}),
                     **({"kinetics_energy": str(kinetics_energy).lower() if isinstance(kinetics_energy, str) else "free",
                         "standard_state": standard_state if isinstance(standard_state, str) else "1M",
                         "mirror_images": mirror_images is not False}
                        if steer in ("flux", "access") else {}),
                     **({"kinetics": getattr(result, "kinetics_model", "box")} if steer == "access" else {}),
                     **({"copies": copies, "concentration": float(concentration) if isinstance(concentration, (int, float))
                         else 1.0} if steer == "access" else {})},
        # barrier_kcal: as computed (a negative one is a problem, flagged in `warning`);
        # kinetics_barrier_kcal: what the rates used (the TS never below an end); null for a step out of the
        # kinetics' reach (closed: no rate).
        "steps": [{"a": st.a, "b": st.b, "ts_energy": st.ts_energy, "barrier_kcal": list(raw),
                   "kinetics_barrier_kcal": list(bar) if bar is not None else None, "label": st.label,
                   "files": st.files,
                   "warning": negative_barrier_warning(st.label, st.a, st.b, *raw,
                                                       verified=st.files.get("verified") is not False)}
                  for st, bar, raw in zip(result.steps, result.kinetics.barriers_kcal if result.kinetics else [],
                                          raw_barriers(result.steps, [float(s.node.energy) for s in result.species]))],
        "connections": [list(p) for p in pairs],
        # each step channel sampling ran on: its TSs before and after (how far a first TS overestimated)
        "channels_log": getattr(result, "channels_log", []),
        "output_files": files,
    }
    # What the run recorded for the measuring instruments (docs/network-expansion.md, "What a run records")
    record = getattr(result, "record", None) or {}
    for k, row in enumerate(summary["species"]):
        row["t_found"] = getattr(result.species[k], "t_found", None)
        row.update(record["species"][k] if k < len(record.get("species", ())) else {})
    for m, row in enumerate(summary["steps"]):
        row.update(record["steps"][m] if m < len(record.get("steps", ())) else {})
    summary["steps_removed"] = getattr(result, "steps_removed", [])
    if "error" in record:
        summary["record_error"] = record["error"]
    if steer in ("flux", "access"):
        from mepd.discovery.energy_refine import _level

        def level(engine, profile):
            import hashlib

            return {"engine": type(engine).__name__, "key": _level(engine),
                    "profile_sha256": hashlib.sha256(Path(profile).read_bytes()).hexdigest() if profile else None}

        summary["steering"]["level"] = {
            "explorer": level(run_inputs.engine, inputs),
            "steer": level(steer_engine, steer_with if steer_engine is not high_engine else refine_energies)
            if steer_engine is not None else None}
        summary["steering"]["steer_fallbacks"] = record.get("steer_fallbacks")
    if getattr(result, "known", None) is not None:
        # the earlier runs given as --known: used (and how much was taken from each) or skipped, and why
        summary["known"] = result.known
    counts = {}
    for e in result.edges:
        counts[e.outcome] = counts.get(e.outcome, 0) + 1
    summary["outcomes"] = counts
    summary["warnings"] = [st["warning"] for st in summary["steps"] if st.get("warning")]
    for w in summary["warnings"]:
        typer.echo(f"WARNING: {w}")

    if result.kinetics is not None:
        for rec, f, cmax, cend in zip(summary["species"], result.kinetics.flux, result.kinetics.max_concentration,
                                      result.kinetics.final_concentration):
            rec.update(flux=f, max_concentration=cmax, final_concentration=cend)
        if result.kinetics.mirrors is not None:
            # the mirror images the explorer did not find (mepd.discovery.mirror), after its own species
            k = result.kinetics
            summary["mirror_images"] = {**k.mirrors, "virtual_species": [
                {**v, "flux": k.flux[v["index"]], "max_concentration": k.max_concentration[v["index"]],
                 "final_concentration": k.final_concentration[v["index"]]} for v in k.mirrors["virtual_species"]]}
        # next to what was asked (kinetics_energy): what the final kinetics rested on, and the structures within
        # reach that had no free-energy correction (each round's: rounds[i]["kinetics"])
        k_rec = next((r["kinetics"] for r in reversed(result.rounds) if r.get("kinetics")), None)
        if k_rec is not None and "energy_basis" in k_rec and "kinetics_energy" in summary["steering"]:
            summary["steering"]["kinetics_energy_used"] = k_rec["energy_basis"]
            summary["steering"]["kinetics_energy_basis"] = k_rec["energies"]
            if "free_energy_gaps" in k_rec:
                summary["steering"]["free_energy_gaps"] = k_rec["free_energy_gaps"]
            if summary["steering"]["kinetics_energy"] == "free" and k_rec["energy_basis"] != "free":
                summary["warnings"].append(f"free-energy kinetics were asked for, but they rest on electronic "
                                           f"energies ({k_rec['energies']})")
                typer.echo(f"WARNING: {summary['warnings'][-1]}")
    network_note = ""
    if steer in ("flux", "access"):
        connect = True   # the pair trees exist already: network.json is built from them below
    if connect and pairs:
        if len(pairs) > max_pairs:
            typer.echo(f"{len(pairs)} reactions to connect, capping at --max-pairs={max_pairs}.")
            pairs = pairs[:max_pairs]
        pairs_dir = output / "pairs"
        pairs_dir.mkdir(exist_ok=True)
        if steer not in ("flux", "access"):
            _run_msmep_pairs([s.node.copy() for s in result.species], pairs, pairs_dir, run_inputs,
                             parallel=parallel, parallel_workers=parallel_workers)
        tree_dirs = _completed_tree_dirs(pairs_dir)
        if tree_dirs:
            from mepd.inputs import NetworkInputs
            from mepd.NetworkBuilder import NetworkBuilder

            try:
                pot = NetworkBuilder(data_dir=output, network_inputs=NetworkInputs()).create_rxn_network_from_paths(tree_dirs)
                pot.write_to_disk(output / "network.json")
                files["network"] = str(output / "network.json")
                network_note = f", network.json: {pot.number_of_nodes} nodes / {pot.graph.number_of_edges()} edges"
            except Exception as exc:
                typer.echo(f"Network construction failed: {type(exc).__name__}: {exc}")
        else:
            typer.echo("No path search completed; no network.json written.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if high_engine is not None and steer in ("flux", "access") and result.steps:
        typer.echo(f"Refining energies ({refine_energies_mode}) with {refine_energies}...")
        try:
            _write_refined(output, run_inputs.engine, high_engine, str(refine_energies_mode), workers,
                           level=str(refine_energies), mirror_images=mirror_images is not False)
        except Exception as exc:
            typer.echo(f"Energy refinement failed: {type(exc).__name__}: {exc}")

    mol = (next((r["kinetics"] for r in reversed(result.rounds) if r.get("kinetics")), None) or {}).get("molecules") \
        if steer == "access" else None
    if mol and mol.get("names"):
        # the answer: the molecule-level kinetics' (summary.json's species concentrations are the boxes')
        seed_keys = {mol["names"][i] for i in mol["seed"]}
        made = sorted(((c, key) for key, c in zip(mol["names"], mol["final"]) if key not in seed_keys), reverse=True)
        tot = sum(c for c, _ in made)
        typer.echo(f"Answer (molecule-level kinetics, {mol['temperature_K']:g} K, {mol['time_s']:g} s; "
                   f"{100 * mol['conversion']:.3g} % of the seed converted"
                   + (f"; {mol['counts']['complexes']} complex(es) kept together" if mol["counts"].get("complexes")
                      else "") + "):")
        for c, key in made[:8]:
            if tot > 0 and c / tot >= 0.001:
                typer.echo(f"  {key}: {100 * c / tot:.3g} % of the converted material ({c:.3g})")
    console.print(f"[bold green]{len(result.species) - 1} new species[/bold green] from "
                  f"{len(result.edges)} proposed reactions ({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))}); "
                  f"{len(result.connections())} distinct reactions to connect{network_note}. Results in {output}/",
                  highlight=False)


def _write_refined(run_dir: Path, low_engine, high_engine, mode: str, workers: int, level: str = "",
                   mirror_images: bool = True) -> dict:
    """refine_run on an expand output; writes refined_energies.json and
    prints what changed."""
    from rich.console import Console
    from rich.table import Table

    from mepd.discovery.energy_refine import refine_run

    out = refine_run(run_dir, low_engine=low_engine, high_engine=high_engine, mode=mode, workers=workers,
                     mirror_images=mirror_images,
                     on_note=lambda m: typer.echo(f"WARNING: {m}"))
    out["level"] = level
    (Path(run_dir) / "refined_energies.json").write_text(json.dumps(out, indent=2, default=str))
    table = Table(title=f"Refined {out['basis']} (kcal/mol from the seed, {mode}) and final amounts")
    for col in ("", "species / step", "explorer", "refined", "final explorer", "final refined"):
        table.add_column(col)
    kin = out["kinetics"]
    for row in out["refined"]:
        if row["kind"] == "species":
            k = row["index"]
            if max(kin["low"]["final_concentration"][k], kin["high"]["final_concentration"][k]) < 0.01 and k:
                continue
            table.add_row(str(k), row.get("smiles") or "", f"{row['g_low_kcal']:.1f}", f"{row['g_high_kcal']:.1f}",
                          f"{kin['low']['final_concentration'][k]:.3f}", f"{kin['high']['final_concentration'][k]:.3f}")
        else:
            table.add_row("TS", f"{row['a']} -> {row['b']}", f"{row['g_low_kcal']:.1f}", f"{row['g_high_kcal']:.1f}",
                          "", "")
    for v in (out.get("mirror_images") or {}).get("virtual_species", []):
        if max(v["final_low"], v["final_high"]) >= 0.01:
            table.add_row(str(v["index"]), f"mirror image of {v['mirror_of']}", "", "",
                          f"{v['final_low']:.3f}", f"{v['final_high']:.3f}")
    Console().print(table)
    typer.echo(f"Refined energies in {Path(run_dir) / 'refined_energies.json'}")
    return out


@discovery_app.command("refine-energies")
def refine_energies_cmd(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False,
                                   help="A `mepd discovery expand` output folder (with summary.json)."),
    inputs: Path = typer.Option(..., "--inputs", "-i", exists=True, dir_okay=False,
                                help="Profile (.toml) of the higher level of theory, e.g. DFT."),
    explorer_inputs: Optional[Path] = typer.Option(
        None, "--explorer-inputs", exists=True, dir_okay=False,
        help="The exploration's own profile (default: the one in summary.json, else built-in defaults)."),
    mode: str = typer.Option("single-point", "--mode", help="single-point or reoptimize (see `expand --refine-energies-mode`)."),
    workers: int = typer.Option(1, "--workers", help="Processes at once; each higher-level calculation uses the "
                                                     "profile's own threads."),
    mirror_images: bool = typer.Option(True, "--mirror-images/--no-mirror-images", help="Count every species and transition state with its mirror image (x -> -x, same energy): a chiral TS "
        "whose twin was not found gets one, leading to the mirror images of its ends; a chiral conformer of an achiral "
        "molecule gets -RT ln 2. The explorer finds one twin or both by chance; this removes that bias."),
):
    """Energies of an explored network at a higher level of theory, and the
    kinetics on them: the explorer's level finds the species and TSs, the
    higher level decides which of them win."""
    from mepd.cli import _open_run_inputs

    if mode not in ("single-point", "reoptimize"):
        raise typer.BadParameter("--mode must be single-point or reoptimize.")
    summary = json.loads((run_dir / "summary.json").read_text())
    low = explorer_inputs or (Path(summary["inputs"]) if summary.get("inputs") and Path(summary["inputs"]).exists()
                              else None)
    _write_refined(run_dir, _open_run_inputs(low).engine, _open_run_inputs(inputs).engine, mode, workers,
                   level=str(inputs), mirror_images=mirror_images)
