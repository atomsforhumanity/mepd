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
"""
from __future__ import annotations

import concurrent.futures
import copy
import os
import threading
from pathlib import Path
from typing import Optional

from mepd.network_aware import log_event


class HorizontalConnector:
    live = True    # takes on_found / relevant: the caller's network grows while the queue runs

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
        if hasattr(self.job_inputs.engine, "n_parallel"):
            # the run's own budget (os.cpu_count() is the node's, not a container's limit)
            budget = int(getattr(run_inputs.engine, "n_parallel", 0) or 0) or (os.cpu_count() or 1)
            self.job_inputs.engine.n_parallel = max(1, budget // self.workers)
        self._lock = threading.Lock()

    # --- jobs -------------------------------------------------------------
    def _write_ts(self, ts, label: str) -> str:
        import mepd

        fp = self.ts_dir / f"{label}.xyz"
        mepd.write_xyz(fp, ts, comments=[f"E={ts.energy:.10f}"])
        return str(fp)

    def _fast_ts(self, i: int, j: int, a, b):
        import mepd

        label = f"fast_{i}_{j}"
        if (self.ts_dir / f"{label}.xyz").exists():
            return mepd.load(self.ts_dir / f"{label}.xyz"), label
        path = mepd.interpolate(a, b, nimages=self.nimages)
        en = mepd.energies(list(path.nodes), self.job_inputs.engine)
        k = max(range(1, len(en) - 1), key=lambda m: en[m])
        ts = mepd.find_ts(path.nodes[k], self.job_inputs.engine, maxiter=self.fast_ts_steps, continue_unconverged=0)
        self._write_ts(ts, label)
        return ts, label

    def _piece(self, chain, index: int, depth: int):
        from mepd.msmep import _parallel_recursive_step_worker

        return _parallel_recursive_step_worker(self.job_inputs, chain, index, depth, self.max_depth, None)

    def _leaf_ts(self, chain, label: str):
        import mepd

        ts = mepd.find_ts(chain.get_ts_node(), self.job_inputs.engine)
        self._write_ts(ts, label)
        return ts

    def _path_peak(self, chain, label: str):
        """The highest image of a finished elementary piece (written as
        ts/<label>.xyz), or None when the piece has no energies."""
        try:
            nodes = list(chain.nodes)
            if any(getattr(n, "_cached_energy", None) is None for n in nodes):
                self.job_inputs.engine.compute_energies(nodes)
            peak = max(nodes, key=lambda n: float(n.energy)).copy()
            self._write_ts(peak, label)
            return peak
        except Exception:
            return None

    def _root_chain(self, a, b):
        from mepd.chain import Chain
        from mepd.interpolation import initial_chain

        seed = Chain.model_validate({"nodes": [a.copy(), b.copy()],
                                     "parameters": copy.deepcopy(self.run_inputs.chain_inputs)})
        return initial_chain(seed, self.run_inputs.chain_inputs, self.run_inputs.gi_inputs)

    # --- the queue ----------------------------------------------------------
    def __call__(self, pairs, nodes, search: bool = False, on_found=None, relevant=None) -> list[dict]:
        """Connect `pairs` of `nodes`. `search`: start every pair's path search
        at once (its single TS was already tried and belonged to another step).

        `on_found(record)`: called the moment a TS is found, so the caller's
        network (and kinetics) grows while the queue runs. `relevant(chain,
        final)`: asked, each time a worker frees up, whether a waiting piece of
        a path search should run -- True: run it; anything else: it waits.
        When only waiting pieces are left, each is asked once more with
        final=True and those still not relevant are dropped (logged). Pieces
        are taken best first: the order `relevant` is asked in is the order
        they were queued, the first relevant one runs."""
        from mepd.network_aware import resolve_cutoff

        self.ts_dir.mkdir(parents=True, exist_ok=True)
        pmi = self.job_inputs.path_min_inputs
        if resolve_cutoff(pmi) is not None and getattr(pmi, "network_floor_hartree", None) is None:
            # Every piece prunes against one floor, the seed's energy -- not its own start's.
            if nodes[0]._cached_energy is None:
                self.job_inputs.engine.compute_energies([nodes[0]])
            pmi.network_floor_hartree = float(nodes[0].energy)
        found: list[dict] = []
        pieces = {}            # pair -> pieces started
        counts = {"fast_ok": 0, "fast_failed": 0, "pieces": 0, "leaf_ts_ok": 0, "leaf_ts_failed": 0, "path_peak": 0,
                  "capped": 0,
                  "searched": 0, "deferred": 0, "dropped": 0}
        running: dict = {}
        waiting: list = []     # (kind, pair, args) not started yet, in queue order
        next_index = {}

        def record(rec):
            found.append(rec)
            if on_found is not None:
                on_found(rec)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            def run(kind, pair, args):
                fut = pool.submit(getattr(self, {"fast": "_fast_ts", "piece": "_piece", "ts": "_leaf_ts"}[kind]),
                                  *args)
                running[fut] = (kind, pair, args)

            def queue(kind, pair, *args):
                waiting.append((kind, pair, args))

            def start_search(pair):
                i, j = pair
                try:
                    root = self._root_chain(nodes[i], nodes[j])
                except Exception as exc:
                    log_event({"event": "pair_failed", "pair": [i, j], "why": f"{type(exc).__name__}: {exc}"})
                    return
                pieces[pair] = 1
                next_index[pair] = 1
                counts["pieces"] += 1
                queue("piece", pair, root, 0, 0)

            def gated(job) -> bool:
                """A piece below a split waits until material can reach one of its ends."""
                kind, _, args = job
                return kind == "piece" and args[2] > 0 and relevant is not None

            def fill(final=False):
                k = 0
                while k < len(waiting) and len(running) < self.workers:
                    job = waiting[k]
                    if gated(job) and relevant(job[2][0], final) is not True:
                        k += 1
                        continue
                    waiting.pop(k)
                    run(*job)
                if final and not running:
                    # nothing runs and nothing waiting can be reached: drop them
                    for job in waiting:
                        counts["dropped"] += 1
                        log_event({"event": "piece_dropped", "pair": list(job[1]), "depth": job[2][2]})
                    waiting.clear()

            for i, j in pairs:
                if search:
                    counts["searched"] += 1
                    start_search((i, j))
                else:
                    queue("fast", (i, j), i, j, nodes[i], nodes[j])

            while running or waiting:
                fill()
                if not running:
                    # only pieces nobody reaches yet: the network is as complete as this call makes it
                    counts["deferred"] = max(counts["deferred"], len(waiting))
                    fill(final=True)
                    if not running:
                        break
                done, _ = concurrent.futures.wait(tuple(running), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    kind, pair, args = running.pop(fut)
                    i, j = pair
                    try:
                        out = fut.result()
                    except Exception:
                        out = None
                    if kind == "fast":
                        if out is not None:
                            ts, label = out
                            counts["fast_ok"] += 1
                            record({"start": nodes[i], "end": nodes[j], "pair": (i, j), "ts": ts, "label": label,
                                    "files": {"ts": str(self.ts_dir / f"{label}.xyz"), "verified": False,
                                              "pair": [i, j]}})
                        else:
                            counts["fast_failed"] += 1
                            start_search(pair)
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
                        rec = {"start": chain[0], "end": chain[-1], "ts": out, "label": label,
                               "files": {"ts": str(self.ts_dir / f"{label}.xyz"), "verified": False}}
                        # The TS search can land on a saddle of some other motion (a rotation next to a
                        # barrierless association): if its IRC says so, the piece falls back to this.
                        peak = self._path_peak(chain, f"{label}_peak")
                        if peak is not None:
                            rec["peak"] = peak
                            rec["files"]["peak"] = str(self.ts_dir / f"{label}_peak.xyz")
                        record(rec)
        log_event({"event": "horizontal_connect", "pairs": len(pairs), **counts})
        self.say(f"Connected {len(pairs)} pair(s): " + (f"{counts['searched']} by path search only, " if search else "")
                 + f"{counts['fast_ok']} by one TS optimization, "
                 f"{counts['fast_failed']} by path search ({counts['pieces']} pieces, {counts['leaf_ts_ok']} TSs"
                 + (f", {counts['path_peak']} steps whose TS search failed kept at their path's highest image"
                    if counts["path_peak"] else "")
                 + (f", {counts['capped']} pieces over the per-pair cap not run" if counts["capped"] else "")
                 + (f", {counts['dropped']} pieces nothing reaches dropped" if counts["dropped"] else "") + ").")
        return found
