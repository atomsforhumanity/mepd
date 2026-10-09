# MEPD

Minimum Energy Path Discovery tools for efficient reaction path calculations. 

This repository contains:

- NEB and related path minimizers
- MLPGI path support
- MSMEP recursive path splitting
- elementary-step analysis utilities

## Installation

```bash
pip install "mepd[web] @ git+https://github.com/mtzgroup/mepd.git"   # everything, web app included
pip install "git+https://github.com/mtzgroup/mepd.git"               # the core library and CLI only
```

For local development, a plain `uv sync` installs everything (the web app and every optional
feature); `uv run` keeps it that way:

```bash
uv sync
uv run pytest
```

Orb and MACE models are the exception: they can't share an environment with FAIR-Chem, so each
goes in an environment of its own (see `docs/mlip.md`).

### macOS on Apple Silicon: building `vina`

The web app's docking uses AutoDock Vina, which has no arm64 macOS wheel on PyPI, so it is built
from source and needs Boost and SWIG. Its build only looks for Boost in a conda env or
`/usr/local`, and it compiles as C++11 while Homebrew's Boost needs C++14. To build it against
Homebrew's Boost:

```bash
brew install boost swig
cat > /tmp/cxx14 <<'EOF'
#!/bin/bash
args=(); for a in "$@"; do [[ $a == -std=c++11 ]] && a=-std=c++14; args+=("$a"); done
exec /usr/bin/clang++ "${args[@]}"
EOF
chmod +x /tmp/cxx14
CC=/tmp/cxx14 CXX=/tmp/cxx14 CONDA_DEFAULT_ENV=homebrew CONDA_PREFIX=/opt/homebrew uv sync
```

uv caches the built wheel, so later syncs need none of this until the cache is cleared or the
`vina` version changes.

## Quick Start

```python
from mepd import Chain, ChainInputs, NEB, NEBInputs, StructureNode
from mepd.engines.qccompute import QCComputeEngine
from mepd.optimizers.cg import ConjugateGradient
import mepd.chainhelpers as ch
from qcdata import Structure

start = Structure.open("start.xyz")
end = Structure.open("end.xyz")

engine = QCComputeEngine(compute_program="chemcloud")

start_node = StructureNode(structure=start)
end_node = StructureNode(structure=end)
start_opt = engine.compute_geometry_optimization(start_node)[-1]
end_opt = engine.compute_geometry_optimization(end_node)[-1]

chain = Chain.model_validate({
    "nodes": [start_opt, end_opt],
    "parameters": ChainInputs(k=0.1, delta_k=0.09),
})
initial_chain = ch.run_geodesic(chain, nimages=15)

neb = NEB(
    initial_chain=initial_chain,
    parameters=NEBInputs(v=True),
    optimizer=ConjugateGradient(timestep=0.5),
    engine=engine,
)
result = neb.optimize_chain()
```

## CLI

The public CLI exposes path minimization and NEB refinement commands:

```bash
mepd run --start start.xyz --end end.xyz --inputs inputs.toml
mepd channels --reaction "CC(=O)C>>CC(O)=C" --inputs inputs.toml   # both ends as one reaction SMILES
mepd run-refine examples/oxycope.xyz -i expensive.toml -ci cheap.toml --mode neb
mepd refine previous_result.xyz --inputs expensive.toml --mode neb
mepd init --output inputs.toml
mepd-elementarystep path.xyz
```

Use the Python API for lower-level NEB, MLPGI, MSMEP, and elementary-step workflows.

## Web interface

```bash
uv sync                      # or: pip install "mepd[web] @ git+https://github.com/mtzgroup/mepd.git"
mepd web my_workspace
```

A browser UI over the same commands. You build a structure library and reaction graph from SMILES,
XYZ or earlier results. You then run TS searches, channel sampling, Hessian exploration or all-pairs
networks on any selection of structures and edges. Jobs are queued, streamed live, resumable, and
their results can be pulled back into the graph. See [docs/web.md](docs/web.md).

## Maintainers

For questions, contact:

- [Jan](mailto:jdep@stanford.edu)
