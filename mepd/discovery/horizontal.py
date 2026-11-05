"""Connecting proposed reactions as one shared queue of small jobs.

A recursive path search (MSMEP) waits for every branch of a split before
it returns, and a batch of searches waits for its slowest one, so most
workers sit idle. Here every unit of work is a job in one pool shared by
all pairs:

  * fast-TS  -- geodesic interpolation between the two species, one TS
                optimization from its highest image (frequency-checked);
  * piece    -- one split level of a path search (MSMEP._guarded_step):
                its child pieces are queued the moment it splits, an
                elementary leaf queues a TS job;
  * TS       -- a TS optimization from a leaf path's highest image.

A pair starts with a fast-TS job; only if that fails does its path search
start, piece by piece. Every TS found becomes a step that has not been
IRC-checked yet (files["verified"] = False): the explorer runs IRCs only
for the steps its kinetics need (network_expansion, `verify`). An
elementary piece whose TS search fails (a nearly barrierless association
has no saddle to find) is kept as a step at its path's highest image
(files["ts_kind"] = "path_peak"; an upper bound on its barrier).

`max_pieces_per_pair` caps one pair's search (a pathological pair can't
take the round); what it cut is logged.

A piece below a split runs only once the caller's kinetics put material at
one of its ends (`relevant`). One that nothing reaches when the queue runs
dry is iced, not dropped: it stays in `iced` with its place in its pair's
search, and every later call queues it again, so a step found afterwards
that sends flux its way thaws it (`ices`). `queue_state()` / `restore()`
carry the iced pieces and each pair's piece numbering over a resumed run.

The caller can run jobs of its own in the same pool (`register`, `submit`)
and add pairs while a call runs (`add_pairs`): the explorer's proposals for
a species start the moment its kinetics populate it, and their products'
pairs join the queue as they come in, so nothing waits for a round.

Jobs run in forked worker processes: as threads of one process they were
held to about five busy cores whatever the worker count (each engine call's
Python-side work is serial under the GIL). The connector reaches the
children through fork, so only a job's arguments and result are pickled.
"""
from __future__ import annotations

import concurrent.futures
import concurrent.futures.process
import copy
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from mepd.network_aware import log_event
from mepd.cpus import available_cpus

_CONNECTOR: dict = {}   # the connector whose jobs the forked workers run


def _run_job(method: str, args):
    self = _CONNECTOR["self"]
    if method.startswith("own:"):    # a job kind the caller registered
        return self._jobs[method[4:]][0](*args)
    return getattr(self, method)(*args)


class _CoreShare:
    """An engine's `n_parallel` that follows the queue: the run's cores over the jobs running right now
    (read from shared memory the parent updates), so a job left on its own late in a stage -- a deep path
    search, say -- spreads its images over the cores the finished jobs left idle instead of using one."""

    def __init__(self, total: int, running):
        self.total, self.running = int(total), running

    def __int__(self) -> int:
        return max(1, self.total // max(1, int(self.running.value)))

    __index__ = __int__

    def __bool__(self) -> bool:
        return True

    def __eq__(self, other): return int(self) == other
    def __hash__(self): return id(self)
    def __lt__(self, other): return int(self) < other
    def __le__(self, other): return int(self) <= other
    def __gt__(self, other): return int(self) > other
    def __ge__(self, other): return int(self) >= other
    def __floordiv__(self, other): return int(self) // other
    def __rfloordiv__(self, other): return other // int(self)
    def __mul__(self, other): return int(self) * other
    __rmul__ = __mul__

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):   # pickled (a spawned process): its value now
        return int, (int(self),)

    def __repr__(self) -> str:
        return f"<{int(self)} of {self.total} cores>"


class HorizontalConnector:
    live = True    # takes on_found / relevant: the caller's network grows while the queue runs
    reports_pairs = True   # takes on_pair_done
    ices = True    # keeps pieces nothing reaches (`iced`) and queues them again on every call

    def __init__(self, run_inputs, output: Path, workers: int, *, max_pieces_per_pair: int = 40,
                 max_depth: Optional[int] = 6, nimages: int = 11, fast_ts_steps: int = 300, say=print):
        from mepd.msmep import _clone_run_inputs_for_worker

        self.ts_dir = Path(output) / "ts"
        self.workers = max(1, int(workers))
        self.max_pieces = int(max_pieces_per_pair)
        self.max_depth = max_depth
        self.nimages = int(nimages)
        # A fast-TS job is a shortcut: one that hasn't converged in this many steps
        # fails, and its pair gets the path search it would have had anyway.
        self.fast_ts_steps = int(fast_ts_steps)
        self.say = say
        self.run_inputs = run_inputs
        # One engine process per job: the pool is the parallelism.
        self.job_inputs = _clone_run_inputs_for_worker(run_inputs)
        # jobs running now, in memory the forked workers share: their engines size their parallelism by it
        import multiprocessing

        self._running = multiprocessing.get_context("fork").RawValue("i", 0)
        if hasattr(self.job_inputs.engine, "n_parallel"):
            # the run's own budget
            budget = int(getattr(run_inputs.engine, "n_parallel", 0) or 0) or available_cpus()
            self.job_inputs.engine.n_parallel = _CoreShare(budget, self._running)
        self._lock = threading.Lock()
        # optimizer keywords for a leaf TS's piece ends, minimized by its job (None: left to the caller)
        self.end_keywords: Optional[dict] = None
        # pieces nothing reached yet: (pair, (chain, index, depth)), queued again by every call
        self.iced: list = []
        # per pair, across calls: pieces started (the cap) and the next piece index (its leaves' TS labels)
        self._pieces: dict = {}
        self._next_index: dict = {}
        # the caller's FreeEnergies: a TS job also computes the TS's thermal correction (`free_cache`)
        self.free = None
        self._jobs: dict = {}     # the caller's own job kinds: kind -> (function, worker slots it takes)
        # the caller's budget: once it returns True no job starts; the ones running finish and the call ends
        self.stop_when = None
        self._live = None         # while a call runs: its add_pairs / submit

    def register(self, kind: str, fn, slots: int = 1) -> None:
        """A job kind of the caller's own: `fn(*args)` runs in a worker like the queue's jobs, taking `slots`
        of the workers (a job that runs its own parallel processes). Register before the call whose
        workers should run it: they fork with the function in place."""
        self._jobs[kind] = (fn, max(1, min(int(slots), self.workers)))

    def submit(self, kind: str, args: tuple, on_done, label: str = "") -> None:
        """While a call runs (from one of its callbacks): queue a registered job; `on_done(result or None)` is
        called in the caller's thread when it ends (None: it raised). `label`: what the live view calls it."""
        self._live.submit(kind, args, on_done, label)

    def add_pairs(self, pairs, search: bool = False) -> None:
        """While a call runs (from one of its callbacks): connect these pairs too (`nodes` must hold them)."""
        self._live.add_pairs(pairs, search)

    def add_channel(self, pair, a, b, label: str) -> None:
        """While a call runs: connect species `pair` once more, from structures `a` and `b` of its two
        species (a conformer pair, atom-mapped) -- a fast TS as `label`, a path search if that fails."""
        self._live.add_channel(pair, a, b, label)

    def queue_state(self) -> dict:
        """What a resumed run needs to carry on: the iced pieces and the piece numbering."""
        return {"iced": list(self.iced), "pieces": dict(self._pieces), "next_index": dict(self._next_index)}

    def restore(self, state: Optional[dict]) -> None:
        if state:
            self.iced = list(state.get("iced", []))
            self._pieces.update(state.get("pieces", {}))
            self._next_index.update(state.get("next_index", {}))

    # --- jobs -------------------------------------------------------------
    def _write_ts(self, ts, label: str, ends=None) -> str:
        """ts/<label>.xyz, and with `ends` (the two nodes its search started between, in its atom order)
        ts/<label>_ends.xyz: what tells a TS's spectators from its reacting complex until it has an IRC
        (free_energy.step_ends)."""
        import mepd

        fp = self.ts_dir / f"{label}.xyz"
        mepd.write_xyz(fp, ts, comments=[f"E={ts.energy:.10f}"])
        if ends is not None:
            mepd.write_xyz(self.ts_dir / f"{label}_ends.xyz", list(ends))
        return str(fp)

    def _ts_correction(self, ts, label: str, ends=None):
        """(cache key, thermal correction) of the TS as the caller's kinetics read it back from its file, or
        None: computed here, in the worker, so the caller's free-energy kinetics don't stop the queue for
        a Hessian each time a TS comes in."""
        if self.free is None:
            return None
        from mepd.discovery.network_expansion import _open_structure

        try:
            st = _open_structure(self.ts_dir / f"{label}.xyz").model_copy(
                update={"charge": ts.structure.charge, "multiplicity": ts.structure.multiplicity})
            ends = [n.structure for n in ends] if ends is not None else None
            return self.free._key(st, True, ends), float(self.free._correction(st, True, ends))
        except Exception:
            return None   # the caller computes it, or keeps the electronic energy (said)

    def _fast_ts(self, i: int, j: int, a, b, label: Optional[str] = None):
        import mepd

        label = label or f"fast_{i}_{j}"
        if (self.ts_dir / f"{label}.xyz").exists():
            ts = mepd.load(self.ts_dir / f"{label}.xyz")
            return ts, label, self._ts_correction(ts, label, (a, b))
        path = mepd.interpolate(a, b, nimages=self.nimages)
        en = mepd.energies(list(path.nodes), self.job_inputs.engine)
        k = max(range(1, len(en) - 1), key=lambda m: en[m])
        ts = mepd.find_ts(path.nodes[k], self.job_inputs.engine, maxiter=self.fast_ts_steps, continue_unconverged=0)
        self._write_ts(ts, label, (a, b))
        return ts, label, self._ts_correction(ts, label, (a, b))

    def _piece(self, chain, index: int, depth: int):
        from mepd.msmep import _parallel_recursive_step_worker

        return _parallel_recursive_step_worker(self.job_inputs, chain, index, depth, self.max_depth, None)

    def _leaf_ts(self, chain, label: str):
        import mepd

        ts = mepd.find_ts(chain.get_ts_node(), self.job_inputs.engine)
        self._write_ts(ts, label, (chain[0], chain[-1]))
        corr = self._ts_correction(ts, label, (chain[0], chain[-1]))
        if self.end_keywords is None:
            return ("with_ends", ts, None, corr)
        # The piece's two ends minimized as the caller would (its `end_keywords`), here in the worker: done
        # by the caller as each TS comes in, they held up the queue (no job starts while it works).
        from mepd.nodes.node import StructureNode

        ends = []
        for node in (chain[0], chain[-1]):
            try:
                ends.append(self.job_inputs.engine.compute_geometry_optimization(
                    StructureNode(structure=node.structure), keywords=dict(self.end_keywords))[-1])
            except Exception:
                return ("with_ends", ts, None, corr)
        return ("with_ends", ts, ends, corr)

    def _path_peak(self, chain, label: str):
        """The highest image of a finished elementary piece (written as
        ts/<label>.xyz), or None when the piece has no energies."""
        try:
            nodes = list(chain.nodes)
            if any(getattr(n, "_cached_energy", None) is None for n in nodes):
                self.job_inputs.engine.compute_energies(nodes)
            peak = max(nodes, key=lambda n: float(n.energy)).copy()
            self._write_ts(peak, label, (nodes[0], nodes[-1]))
            return peak
        except Exception:
            return None

    def _root_chain(self, a, b):
        from mepd.chain import Chain
        from mepd.interpolation import initial_chain

        seed = Chain.model_validate({"nodes": [a.copy(), b.copy()],
                                     "parameters": copy.deepcopy(self.run_inputs.chain_inputs)})
        return initial_chain(seed, self.run_inputs.chain_inputs, self.run_inputs.gi_inputs)

    # --- the pool -----------------------------------------------------------
    def _pool(self):
        """Forked worker processes for the jobs (threads where forking is unsafe).
        Workers fork when jobs are first submitted, so they see the connector as
        it is then (e.g. the network floor set at the start of a call)."""
        import multiprocessing

        from mepd.cli_common import _unsafe_to_fork

        _CONNECTOR["self"] = self
        unsafe = _unsafe_to_fork() if self.workers > 1 else "one worker"
        if unsafe:
            return concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        from mepd.cli_common import _fork_worker_init

        return concurrent.futures.ProcessPoolExecutor(max_workers=self.workers,
                                                      mp_context=multiprocessing.get_context("fork"),
                                                      initializer=_fork_worker_init, initargs=(os.getpid(),))

    def _broken(self, pool):
        """A fresh pool in place of one a dead worker broke."""
        from mepd.cli_common import _warn

        _warn("A connect worker process died without reporting an error (killed by the OS, or a crash "
              "inside the engine); the jobs it took down run again in a fresh pool.")
        pool.shutdown(wait=False, cancel_futures=True)
        return self._pool()

    # --- the queue ----------------------------------------------------------
    def __call__(self, pairs, nodes, search: bool = False, on_found=None, relevant=None,
                 on_pair_done=None, free_cache: Optional[dict] = None, start=None, on_drain=None) -> list[dict]:
        """Connect `pairs` of `nodes`. `search`: start every pair's path search
        at once (its single TS was already tried and belonged to another step).

        `on_found(record)`: called the moment a TS is found, so the caller's
        network (and kinetics) grows while the queue runs. `relevant(chain,
        final)`: asked, each time a worker frees up, whether a waiting piece of
        a path search should run -- True: run it; anything else: it waits.
        When only waiting pieces are left, each is asked once more with
        final=True and those still not relevant are iced (`iced`, logged):
        the next call queues them again, after the pieces of its own pairs. Pieces
        are taken best first: the order `relevant` is asked in is the order
        they were queued, the first relevant one runs. `on_pair_done(pair)`:
        called once a pair has nothing left queued or running (its TSs all
        reported), so a caller that saves its progress can skip it on resume.
        `free_cache`: where each TS's thermal correction (computed by its job,
        with `free` set) goes, as the caller's FreeEnergies cache keys it.
        `start()`: called once the queue runs, to submit the caller's own jobs.
        `on_drain()`: called when nothing runs and only pieces nothing reaches
        wait, before they are iced; True if it queued more work (the call goes
        on), so the call ends only when the caller has nothing left to add.
        `nodes` may grow while the call runs (pairs added by index)."""
        from mepd.network_aware import resolve_cutoff

        self.ts_dir.mkdir(parents=True, exist_ok=True)
        pmi = self.job_inputs.path_min_inputs
        if resolve_cutoff(pmi) is not None and getattr(pmi, "network_floor_hartree", None) is None:
            # Every piece prunes against one floor, the seed's energy -- not its own start's.
            if nodes[0]._cached_energy is None:
                self.job_inputs.engine.compute_energies([nodes[0]])
            pmi.network_floor_hartree = float(nodes[0].energy)
        found: list[dict] = []
        pieces = self._pieces            # pair -> pieces started
        counts = {"fast_ok": 0, "fast_failed": 0, "pieces": 0, "leaf_ts_ok": 0, "leaf_ts_failed": 0, "path_peak": 0,
                  "capped": 0,
                  "searched": 0, "deferred": 0, "iced": 0, "thawed": 0}
        running: dict = {}
        waiting: list = []     # (kind, pair, args) not started yet, in queue order
        next_index = self._next_index
        crashed: dict = {}     # id(job args) -> times its worker died under it
        outstanding: dict = {}  # pair -> its jobs queued or running

        def record(rec):
            found.append(rec)
            if on_found is not None:
                on_found(rec)

        pool = self._pool()
        try:
            def slots(kind) -> int:
                return self._jobs[kind[4:]][1] if kind.startswith("own:") else 1

            def busy() -> int:
                return sum(slots(job[0]) for job in running.values())

            def run(kind, pair, args):
                nonlocal pool
                method = {"fast": "_fast_ts", "piece": "_piece", "ts": "_leaf_ts"}.get(kind, kind)
                try:
                    fut = pool.submit(_run_job, method, args)
                except (concurrent.futures.process.BrokenProcessPool, RuntimeError):
                    pool = self._broken(pool)
                    fut = pool.submit(_run_job, method, args)
                running[fut] = (kind, pair, args)
                self._running.value = busy()

            def queue(kind, pair, *args):
                outstanding[pair] = outstanding.get(pair, 0) + 1
                waiting.append((kind, pair, args))

            def settle(pair):
                """One of `pair`'s jobs is over (done, failed or dropped); its last one ends the pair."""
                outstanding[pair] -= 1
                if outstanding[pair] == 0 and on_pair_done is not None:
                    on_pair_done(pair)

            def start_search(pair, a=None, b=None):
                i, j = pair
                try:
                    root = self._root_chain(nodes[i] if a is None else a, nodes[j] if b is None else b)
                except Exception as exc:
                    log_event({"event": "pair_failed", "pair": [i, j], "why": f"{type(exc).__name__}: {exc}"})
                    if not outstanding.get(pair) and on_pair_done is not None:
                        on_pair_done(pair)    # nothing of it ever ran
                    return
                # a pair searched again (after its single TS) numbers on from its earlier pieces
                pieces[pair] = pieces.get(pair, 0) + 1
                idx = next_index.get(pair, 0)
                next_index[pair] = idx + 1
                counts["pieces"] += 1
                queue("piece", pair, root, idx, 0)

            def gated(job) -> bool:
                """A piece below a split waits until material can reach one of its ends."""
                kind, _, args = job
                return kind == "piece" and args[2] > 0 and relevant is not None

            def fill(final=False):
                if self.stop_when is not None and waiting and self.stop_when():
                    counts["dropped"] = counts.get("dropped", 0) + len(waiting)
                    log_event({"event": "queue_stopped", "dropped": len(waiting)})
                    dropped = list(waiting)
                    waiting.clear()
                    for kind, pair, args in dropped:
                        if kind.startswith("own:"):
                            own_done.pop(id(args), None)
                            own_label.pop(id(args), None)
                        else:
                            settle(pair)
                    return
                k = 0
                while k < len(waiting) and busy() < self.workers:
                    job = waiting[k]
                    if running and busy() + slots(job[0]) > self.workers:
                        k += 1      # a wide job waits for room; narrower ones behind it may start
                        continue
                    if gated(job) and relevant(job[2][0], final) is not True:
                        k += 1
                        continue
                    waiting.pop(k)
                    run(*job)
                if final and not running and on_drain is not None and on_drain():
                    return fill()    # the caller added work (e.g. proposals for a species now populated)
                if final and not running:
                    # nothing runs and nothing waiting can be reached yet: ice them for a later call
                    for job in waiting:
                        counts["iced"] += 1
                        self.iced.append((job[1], job[2]))
                        log_event({"event": "piece_iced", "pair": list(job[1]), "depth": job[2][2]})
                    iced = [job[1] for job in waiting]
                    waiting.clear()
                    for pair in iced:
                        settle(pair)

            for i, j in pairs:
                if search:
                    counts["searched"] += 1
                    start_search((i, j))
                else:
                    queue("fast", (i, j), i, j, nodes[i], nodes[j])
            # pieces earlier calls iced: gated like any waiting piece, so they run once material reaches them
            thawing, self.iced = self.iced, []
            for pair, args in thawing:
                queue("piece", pair, *args)
            own_done: dict = {}   # id(args) of a registered job -> its on_done
            own_label: dict = {}  # ... and what the live view calls it
            finished: dict = {}   # job kind -> (done, failed), for the live view
            shown = [0.0]

            def what(kind, pair, args) -> str:
                if kind == "fast":
                    return f"{pair[0]}→{pair[1]}" + (f" · {args[4]}" if len(args) > 4 and args[4] else "")
                if kind == "piece":
                    return f"{pair[0]}→{pair[1]} · depth {args[2]}"
                if kind == "ts":
                    return str(args[1])
                return own_label.get(id(args), "")

            def show(force=False):
                """The queue as the live view shows it (at most once a second)."""
                import time

                from mepd.progress import write_live_state

                if not force and time.monotonic() - shown[0] < 1.0:
                    return
                shown[0] = time.monotonic()
                queued: dict = {}
                for kind, _, _ in waiting:
                    queued[kind.removeprefix("own:")] = queued.get(kind.removeprefix("own:"), 0) + 1
                write_live_state("queue", "queue", {
                    "workers": self.workers, "busy": busy(), "waiting": queued, "iced": len(self.iced),
                    "running": [{"kind": k.removeprefix("own:"), "what": what(k, p, a)} for k, p, a in running.values()],
                    "finished": {k: {"done": d, "failed": f} for k, (d, f) in finished.items()},
                    "counts": dict(counts), "ended": force})

            def add_pairs(new, search_=False):
                for i, j in new:
                    if search_:
                        counts["searched"] += 1
                        start_search((i, j))
                    else:
                        queue("fast", (i, j), i, j, nodes[i], nodes[j])

            def submit(kind, args, on_done, label=""):
                own_done[id(args)] = on_done
                own_label[id(args)] = label
                waiting.append(("own:" + kind, None, args))

            def add_channel(pair, a, b, label):
                queue("fast", tuple(pair), pair[0], pair[1], a, b, label)

            self._live = SimpleNamespace(add_pairs=add_pairs, submit=submit, add_channel=add_channel)
            if start is not None:
                start()

            while True:
                fill()
                show()
                if not running:
                    # only pieces nobody reaches yet: the network is as complete as this call makes it
                    counts["deferred"] = max(counts["deferred"], len(waiting))
                    fill(final=True)
                    if not running:
                        break
                done, _ = concurrent.futures.wait(tuple(running), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    kind, pair, args = running.pop(fut)
                    self._running.value = busy()
                    rerun = False
                    try:
                        i, j = pair if pair is not None else (None, None)
                        try:
                            out = fut.result()
                        except concurrent.futures.process.BrokenProcessPool:
                            # A worker died (OOM killer, a crashing engine): the jobs it took down run
                            # again in a fresh pool, once -- a job that kills its worker twice fails.
                            if getattr(pool, "_broken", False):
                                pool = self._broken(pool)
                            crashed[id(args)] = crashed.get(id(args), 0) + 1
                            if crashed[id(args)] < 2:
                                run(kind, pair, args)
                                rerun = True     # still the pair's job: it settles when this run ends
                                continue
                            out = None
                        except Exception:
                            out = None
                        d, f = finished.get(kind.removeprefix("own:"), (0, 0))
                        finished[kind.removeprefix("own:")] = (d + (out is not None), f + (out is None))
                        if kind.startswith("own:"):
                            own_label.pop(id(args), None)
                            own_done.pop(id(args))(out)
                            continue
                        if kind == "piece" and args[2] > 0 and any(p == pair and a is args for p, a in thawing):
                            counts["thawed"] += 1
                        if kind == "fast":
                            if out is not None:
                                ts, label, corr = out
                                if corr is not None and free_cache is not None:
                                    free_cache[corr[0]] = corr[1]
                                counts["fast_ok"] += 1
                                record({"start": nodes[i], "end": nodes[j], "pair": (i, j), "ts": ts, "label": label,
                                        "files": {"ts": str(self.ts_dir / f"{label}.xyz"), "verified": False,
                                                  "pair": [i, j]}})
                            else:
                                counts["fast_failed"] += 1
                                start_search(pair, *args[2:4])   # from the structures it was given
                        elif kind == "piece":
                            if out is None:
                                continue
                            history, children = out
                            if children:
                                for child in children:
                                    if pieces[pair] >= self.max_pieces:
                                        counts["capped"] += 1
                                        continue
                                    pieces[pair] += 1
                                    counts["pieces"] += 1
                                    idx = next_index[pair]
                                    next_index[pair] += 1
                                    queue("piece", pair, child, idx, args[2] + 1)
                            else:
                                data = getattr(history, "data", None)
                                traj = getattr(data, "chain_trajectory", None) or []
                                if traj and not getattr(history, "skip_ts", False):
                                    queue("ts", pair, traj[-1], f"pair_{i}_{j}_leaf_{history.index}")
                        elif kind == "ts":
                            chain, label = args
                            if out is None:
                                counts["leaf_ts_failed"] += 1
                                peak = self._path_peak(chain, label)
                                if peak is not None:
                                    # An elementary piece whose TS search failed -- typically a nearly barrierless
                                    # association, which has no saddle to find. Dropping it would cut the network
                                    # where it starts (e.g. a catalyst meeting its substrate), so it is kept with
                                    # its path's highest image as the barrier: an upper bound, and its two ends
                                    # are the piece's own minimized endpoints (nothing for an IRC to check).
                                    counts["path_peak"] += 1
                                    record({"start": chain[0], "end": chain[-1], "ts": peak, "label": label,
                                            "files": {"ts": str(self.ts_dir / f"{label}.xyz"), "verified": True,
                                                      "ts_kind": "path_peak"}})
                                continue
                            counts["leaf_ts_ok"] += 1
                            ends = None
                            if isinstance(out, tuple) and out and out[0] == "with_ends":
                                _, out, ends, corr = out
                                if corr is not None and free_cache is not None:
                                    free_cache[corr[0]] = corr[1]
                            rec = {"start": chain[0], "end": chain[-1], "ts": out, "label": label,
                                   "files": {"ts": str(self.ts_dir / f"{label}.xyz"), "verified": False}}
                            if ends is not None:
                                rec["ends"] = ends
                            # The TS search can land on a saddle of some other motion (a rotation next to a
                            # barrierless association): if its IRC says so, the piece falls back to this.
                            peak = self._path_peak(chain, f"{label}_peak")
                            if peak is not None:
                                rec["peak"] = peak
                                rec["files"]["peak"] = str(self.ts_dir / f"{label}_peak.xyz")
                            record(rec)
                    finally:
                        if not rerun and pair is not None:
                            settle(pair)
        finally:
            self._live = None
            try:
                show(force=True)
            except Exception:
                pass
            pool.shutdown(wait=True, cancel_futures=True)
            _CONNECTOR.pop("self", None)
        log_event({"event": "horizontal_connect", "pairs": len(pairs), **counts})
        self.say(f"Connected {len(pairs)} pair(s): " + (f"{counts['searched']} by path search only, " if search else "")
                 + f"{counts['fast_ok']} by one TS optimization, "
                 f"{counts['fast_failed']} by path search ({counts['pieces']} pieces, {counts['leaf_ts_ok']} TSs"
                 + (f", {counts['path_peak']} steps whose TS search failed kept at their path's highest image"
                    if counts["path_peak"] else "")
                 + (f", {counts['capped']} pieces over the per-pair cap not run" if counts["capped"] else "")
                 + (f", {counts['thawed']} iced pieces thawed" if counts["thawed"] else "")
                 + (f", {len(self.iced)} pieces nothing reaches yet iced" if self.iced else "")
                 + (f"; out of budget, {counts['dropped']} queued jobs not run" if counts.get("dropped") else "") + ").")
        return found
