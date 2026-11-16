"""The mapping stage of `mepd channels` alone, on the conformers of an earlier run.

    PYTHONPATH=<code> python benchmarks/diagnostics/addmol/mapping_stage.py CHANNELS_OUT PROFILE.toml OUT_DIR \
        [--explore 20] [--workers 8] [--molecule SMILES]

Reads CHANNELS_OUT/conformers/start.xyz and end.xyz (the pools the searches of that run were paired from),
pairs every start with every end and runs `channel_search._expand_pairs_by_mechanism` as `run_channels` does
(3 pairs per mechanism). Writes OUT_DIR/pair_mechanisms.json (the stage's own table), OUT_DIR/mapping.json
(per kept search: start conformer, end conformer, mechanism, score, and the sha256 of the mapped end
structure's symbols and coordinates) and the wall time. Two code trees give the same mapping when both files
are the same; mapping.json sorted by (mechanism, start, end) compares trees that queue in different orders.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("channels_out", type=Path)
ap.add_argument("profile", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--explore", type=int, default=20)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--molecule")
a = ap.parse_args()

import mepd
from mepd import channel_search as cs
from mepd.chain import Chain
from mepd.inputs import ChainInputs, RunInputs

run_inputs = RunInputs.open(a.profile)
run_inputs.atom_mapping_inputs.explore_mechanisms = a.explore
if a.molecule:
    run_inputs.atom_mapping_inputs.explore_molecule = a.molecule
start = list(Chain.from_xyz(a.channels_out / "conformers" / "start.xyz", ChainInputs(), charge=0, spinmult=1))
end = list(Chain.from_xyz(a.channels_out / "conformers" / "end.xyz", ChainInputs(), charge=0, spinmult=1))
structures = start + end
candidates = [(i, len(start) + j) for i in range(len(start)) for j in range(len(end))]
a.out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
new_structures, new_candidates, summary = cs._expand_pairs_by_mechanism(
    structures, candidates, True, run_inputs, pairs_per_mechanism=3, workers=a.workers, output=a.out)
seconds = time.perf_counter() - t0
table = [r for r in json.loads((a.out / "pair_mechanisms.json").read_text()) if not r.get("not_in_this_run")]
rows = []
for r in table:
    s = new_structures[int(r["pair"].rsplit("_", 1)[1])].structure
    digest = hashlib.sha256((" ".join(s.symbols) + np.asarray(s.geometry, dtype=float).round(8).tobytes().hex()).encode()).hexdigest()
    rows.append({"start_conformer": r["start_conformer"], "end_structure": r["end_structure"], "mechanism": r["mechanism"],
                 "score": r["score"], "n_symmetry_variants": r["n_symmetry_variants"], "mapped_end_sha256": digest})
(a.out / "mapping.json").write_text(json.dumps({
    "code": mepd.__file__, "pairs": len(candidates), "searches": len(new_candidates), "seconds": round(seconds, 2),
    "workers": a.workers, "queue": [r["mechanism"] for r in table],
    "sorted": sorted(rows, key=lambda r: (r["mechanism"], r["start_conformer"], r["end_structure"]))}, indent=1))
print(f"{len(candidates)} pairs, {summary.get('n_mechanisms')} mechanisms, {len(new_candidates)} searches, {seconds:.1f} s")
