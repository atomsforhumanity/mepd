"""The mechanisms `mepd channels --explore-mechanisms N` would search for a reaction, from an earlier run's conformers.

    PYTHONPATH=<code> python benchmarks/diagnostics/addmol/explored_list.py CHANNELS_OUT [--explore 20] [--molecule SMILES]

Reads the first start and end conformer of CHANNELS_OUT/conformers and prints `suggest_mechanism_candidates`' keys in
ranked order with their number of symmetry variants (mapping only; nothing is searched).
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--explore", type=int, default=20)
ap.add_argument("--molecule")
a = ap.parse_args()

import mepd
from mepd.atom_mapping import suggest_mechanism_candidates
from mepd.chain import Chain
from mepd.inputs import ChainInputs

s = Chain.from_xyz(a.out / "conformers" / "start.xyz", ChainInputs(), charge=0, spinmult=1)[0].structure
e = Chain.from_xyz(a.out / "conformers" / "end.xyz", ChainInputs(), charge=0, spinmult=1)[0].structure
groups = suggest_mechanism_candidates(s, e, explore=a.explore, **({"explore_molecule": a.molecule} if a.molecule else {}))
print(f"# {a.out} explore {a.explore} molecule {a.molecule} code {mepd.__file__}")
for n, (k, v) in enumerate(groups.items(), start=1):
    print(f"{n:2d}. {k}  ({len(v)} variants)")
