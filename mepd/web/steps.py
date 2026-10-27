"""One elementary step of a multi-step TS search, as a calculation of its own.

A `mepd ts` search whose path splits (MSMEP) finds a route start ->
intermediate(s) -> end: one path, TS and IRC per step (tree/node_<i>,
ts_leaf_<i>, ts_leaf_<i>_irc). When one step goes into Explore as an edge,
that edge gets a job holding only that step -- its folder laid out like a
single-step search (mep_output, ts, irc) -- so its result page, its barrier
(from the step's own start) and every follow-up (solvent, substituents,
Sample more paths, VRI) see that step alone, not the whole route.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

# A step's entries in a TS search's result: its path, TS and IRC (`first_`:
# shown on a page that merges Sample more paths runs).
STEP_ENTRY = re.compile(r"^(?:first_)?(?:ts_)?leaf_(\d+)(?:_irc)?$")
_CHAIN_PARTS = (".xyz", ".energies", ".gradients", "_grad_shapes.txt")


def tree_leaves(tree_dir: Path) -> list[int]:
    """Node indices of an MSMEP split tree's leaves, in path order."""
    import numpy as np

    try:
        adj = np.atleast_2d(np.loadtxt(Path(tree_dir) / "adj_matrix.txt"))
    except (OSError, ValueError):
        return []
    n = adj.shape[0]
    leaves: list[int] = []

    def walk(i: int) -> None:
        kids = [j for j in range(i + 1, n) if adj[i, j] and adj[j].any()]
        if not kids:
            leaves.append(i)
        for j in kids:
            walk(j)

    walk(0)
    return leaves


def step_of_entry(job: dict, entry_id: str) -> Optional[tuple[int, int, int]]:
    """(leaf node, step number from 1, steps) when `entry_id` is one step of
    this TS search's multi-step route; else None (a single-step search, or
    an entry that is the whole route)."""
    m = STEP_ENTRY.match(str(entry_id))
    if job.get("op") != "ts" or m is None:
        return None
    leaves = tree_leaves(Path(job["output_dir"]) / "tree")
    leaf = int(m.group(1))
    if len(leaves) < 2 or leaf not in leaves:
        return None
    return leaf, leaves.index(leaf) + 1, len(leaves)


def _copy_chain(src_dir: Path, stem: str, dst_dir: Path, new_stem: str) -> bool:
    """A chain's files (xyz, energies, gradients, shapes) under a new name."""
    copied = False
    for part in _CHAIN_PARTS:
        fp = src_dir / f"{stem}{part}"
        if fp.exists():
            shutil.copyfile(fp, dst_dir / f"{new_stem}{part}")
            copied = True
    return copied


def write_step_folder(src: Path, leaf: int, dst: Path) -> None:
    """The step's files from a multi-step search's output `src`, laid out as
    a single-step search in `dst`: its path (tree/node_<leaf>) as
    mep_output, its TS as ts, its IRC as irc."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    _copy_chain(src / "tree", f"node_{leaf}", dst, "mep_output")
    _copy_chain(src, f"ts_leaf_{leaf}", dst, "ts")
    _copy_chain(src, f"ts_leaf_{leaf}_irc", dst, "irc")
    failed = src / f"ts_leaf_{leaf}_irc_failed.txt"
    if failed.exists():
        shutil.copyfile(failed, dst / "irc_failed.txt")
    check = src / f"ts_leaf_{leaf}_check_failed.txt"
    if check.exists():
        shutil.copyfile(check, dst / "ts_check_failed.txt")


def point_edge_at_step(ws, edge_id: str, step: dict) -> None:
    """The edge's origin becomes its step job: the barrier then comes from
    that job's result (from the step's own start, not the route's), and its
    links, TS and follow-ups are the step's."""
    parent = step["step_of"]["job"]
    with ws._lock:
        edge = ws._data["edges"].get(edge_id)
        if edge is None:
            return
        old = edge.get("origin") or {}
        edge["origin"] = {"kind": "job", "job": step["id"], "entry": "ts_irc", "group": "irc",
                          "has_ts": bool(old.get("has_ts", True)), "barrier_kcal": None,
                          "headline": old.get("headline", ""), "step_of": parent}
        ws._save()


def adopt_step_edges(manager) -> list[str]:
    """Edges added from one step of a multi-step TS search before step jobs
    existed (their origin is the whole search): each gets its step job now.
    Returns the new job ids."""
    snap = manager.ws.snapshot()
    made = []
    for eid, edge in snap["edges"].items():
        o = edge.get("origin") or {}
        parent = manager.jobs.get(o.get("job") or "")
        if o.get("kind") != "job" or parent is None or parent.get("status") != "done" or not o.get("entry"):
            continue
        found = step_of_entry(parent, o["entry"])
        if found is None:
            continue
        try:
            made.append(manager.add_step_job(parent, found, eid, [edge["source"], edge["target"]])["id"])
        except Exception:
            continue
    return made
