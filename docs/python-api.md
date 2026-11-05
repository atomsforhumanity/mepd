# Python API

mepd's command line is built on a small set of Python functions. Use them to
write your own workflows: a benchmark over a dataset, a screen, a one-off
analysis. Everything here is importable from `mepd` directly and is stable:
names without a leading underscore keep working across releases, and a
change to one is announced in the CHANGELOG. If something you need is only
reachable through a private (`_`) function, that is a gap in this API:
please open an issue or add it here (see [Contributing](#contributing)).

```python
import mepd

eng = mepd.engine("gxtb")                              # an engine by name
reactant = mepd.load("reactant.xyz")                   # xyz files are Angstrom
product = mepd.load("product.xyz")
reactant, product = mepd.optimize([reactant, product], eng)

path = mepd.interpolate(reactant, product, nimages=12)  # geodesic guess
mepd.energies(path.nodes, eng)
ts = mepd.find_ts(path.get_ts_node(), eng)             # highest image -> saddle
irc = mepd.irc(ts, eng)
if mepd.connects(irc, reactant, product):              # both sides, either direction
    print(f"barrier {mepd.barrier_kcal(ts, reactant, eng):.1f} kcal/mol")
mepd.write_xyz("irc.xyz", irc.nodes)
```

## Conventions

- **Structures** are `mepd.StructureNode`s, made with `mepd.load`,
  `mepd.read_frames` or `mepd.from_smiles`. They carry symbols, coordinates,
  charge, multiplicity, a molecular graph and, once computed, an energy.
- **Units.** Inside mepd, coordinates are Bohr and energies are Hartree
  (`node.coords`, `node.energy`, `mepd.energies`). Files are Angstrom. A number
  in any other unit carries it in its name: `barrier_kcal`. `mepd.units` holds
  the conversion constants; use those instead of literals.
- **Engines** compute energies and gradients and, where they can, minimize,
  find transition states and follow IRCs. Get one with `mepd.engine(name, ...)`
  or `mepd.engine_from_profile("profile.toml")` (the same file `--inputs` takes).
- **Errors.** Functions raise instead of returning `None`:
  `mepd.UnsupportedOperation` (a `NotImplementedError`) when the engine cannot do
  it, `mepd.ConvergenceError` when the calculation ran but did not reach a usable
  result, `ValueError` for inputs that cannot work.
- **No side effects.** Functions print nothing and write no files, except
  `mepd.write_xyz` and `mepd.channels` (which works in its output folder).
  mepd logs progress through Python's `logging` (loggers under `mepd`;
  `logging.basicConfig(level=logging.INFO)` shows it) and warnings appear on
  stderr. Not yet: path searches (NEB, GSM, MSMEP) still print their
  step-by-step progress.
- **Threads.** `mepd.set_threads(1)` before the first calculation, when
  running many in parallel.

## Structures

| Function | Returns |
|---|---|
| `load(path, *, charge=None, multiplicity=None)` | the (first) structure in an xyz file |
| `read_frames(path, *, charge=None, multiplicity=None)` | every frame of a multi-frame xyz file |
| `from_smiles(smiles, *, charge=None, multiplicity=None)` | a 3D structure; several molecules (`"C=C.O"`) in one |
| `write_xyz(path, structures, comments=None)` | writes one or several frames (Angstrom); returns the path |

qcdata's comment line stores charge and multiplicity, so `write_xyz` then
`load` round-trips them. `charge=`/`multiplicity=` override what a file says.

## Engines

| Function | |
|---|---|
| `engine(name="gxtb", *, program=None, solvation=None, **options)` | `"gxtb"`, `"xtb"` (`method="gfn2"`), `"mlip"` (`model="aimnet2-rxn"`), `"fairchem"`, `"ase"` (`calculator="pkg.module:Class"`), `"psi4"`, `"orca"` (ORCA's own optimizers and analytic Hessians; `method`, `basis`, `threads`, `maxcore`), `"terachem"`, `"qccompute"`/`"chemcloud"` (`program="psi4"`, plus its keywords) |
| `engine_from_profile(path)` | the engine a profile `.toml` describes |
| `set_threads(n)` | OMP/MKL/OpenBLAS threads for this process and the programs it starts |

## Calculations

| Function | Returns |
|---|---|
| `energies(structures, engine)` | Hartree, one per structure (also cached on each) |
| `optimize(structures, engine, **keywords)` | the minimized structure(s); batched when the engine can |
| `minimize(structure, engine, *, check_frequencies=True, escape_saddles=3, **keywords)` | a true minimum: converged, frequency-checked, pushed off a saddle along its imaginary mode if it stopped on one (`.frequencies`, `.saddle_escapes`) |
| `frequencies(structure, engine)` | a `Frequencies`: `.wavenumbers_cm` (imaginary negative), `.modes`, `.n_imaginary(cutoff_cm=10)` |
| `find_ts(guess, engine, *, check_frequencies=True, continue_unconverged=1, **keywords)` | the transition state: converged, with exactly one imaginary mode (`.frequencies`) |
| `irc(ts, engine, *, minimize_ends=True, **keywords)` | an `IRCResult`: `.path` (a Chain, minimum to minimum), `.irc_path` (the IRC alone), `.ends`, `.nodes` |
| `interpolate(start, end, *, nimages=15, **options)` | a geodesic path (a Chain); start and end in the same atom order |
| `barrier_kcal(ts, reactants, engine=None)` | TS energy minus the lowest of `reactants`, kcal/mol |

`keywords`/`options` go to the engine's optimizer or the geodesic smoother
(e.g. `fmax` in eV/Å, default 0.001; `maxiter`). Nothing returns an
unconverged structure: an optimization that runs out of steps raises
`ConvergenceError` (its `.structure` is where it stopped), and `find_ts`
first goes on from there `continue_unconverged` times. A converged TS without
exactly one imaginary mode, or a minimum still on a saddle, raises
`WrongStationaryPointError`. With g-xTB, TS searches start from g-xTB's own
Hessian (`exact_hessian=True`, `hessian_every=n` to renew it) and
minimizations use Sella with that Hessian (`optimizer="geometric"`, `"lbfgs"`;
`GXTBCalculator(minimizer="xtb")` for xtb's own optimizer). `irc` minimizes
both ends by default, since an IRC stops short of the minimum.
`barrier_kcal` takes every reactant structure you have, e.g. all conformers
found, and measures from the lowest, so barriers from different searches
share a floor.

## Channel search

| Function | Returns |
|---|---|
| `channels(start, end, *, output, profile=None, engine=None, **options)` | a `ChannelsResult`: every way from start to end found |
| `read_channels(folder)` | the `ChannelsResult` of an earlier search (`mepd.channels` or `mepd channels`) |

```python
result = mepd.channels(reactant, product, output="runs/claisen", profile="gsm.toml",
                       n_conformers=5, search_budget=1800, direct_only=True)
best = result.lowest()                     # the route with the lowest highest TS
print(best.kind, mepd.barrier_kcal(best.ts, result.start_conformers))
for channel in result.routes:              # direct, then multistep
    for step in channel.steps:
        print(step.connects, step.ts.energy, len(step.alternatives))
```

The search samples conformers of both ends, searches paths between pairs of
them, finds a TS and IRC for every path's peak, and sorts the steps into
`result.direct` (one step from start to end), `result.multistep` (through
intermediates) and `result.offtarget` (steps that leave the route). Each is
a list of `Channel(kind, steps, folder, species)`; a `Step` has `ts`, `irc`
(an `IRCResult`), `connects` and `alternatives` (other TSs for that step).
`result.start_conformers`/`end_conformers` are the pools searched, so
`mepd.barrier_kcal(channel.ts, result.start_conformers)` measures from the
lowest reactant conformer.

`options` are `mepd.ChannelsOptions` fields: each is the `mepd channels`
option of the same name with underscores (`n_conformers` is `--n-conformers`),
with the same default; `mepd channels --help` describes them. The search works
in `output` (conformers, pair trees, TS/IRC files) and resumes when run into
the same folder again. It logs progress on the `mepd.channel_search` logger
(`logging.basicConfig(level=logging.INFO)` shows it), and raises
`mepd.ChannelSearchError` when it cannot run to the end.

## Comparing structures

| Function | Returns |
|---|---|
| `same_molecule(a, b, *, stereo=True, metal_ligand_contacts=True)` | same bond graph (atom order and conformation ignored) and, with `stereo`, the same stereochemistry |
| `compare_structures(a, b, *, metal_ligand_contacts=True)` | a `Comparison`: `.graph`, `.stereo` (True, False, or None when it cannot be decided), `.smiles`, `.reason`, `.method`, `.metal_ligand_changed` |
| `connects(path, start, end, *, stereo=True, metal_ligand_contacts=True)` | a `Match`: `.start`, `.end`, `.both`, `.any`, `.none`, `.reversed`, `.undetermined`, `.comparisons`; `bool(match)` is `.both` |

With `metal_ligand_contacts=False`, bonds to transition metals (and lanthanides, actinides) are left out of
both the bond graphs and the stereochemistry: a ligand that binds through another atom, or comes off the
metal, is the same species as long as its covalent bonds and stereochemistry are, and
`Comparison.metal_ligand_changed` says whether the contacts differ. Off by default. The idea of keeping
dative metal-ligand contacts apart from covalent bonds follows Z. Li, H.-H. Hsu and B. M. Savoie, "Finding
the Roads Less Calculated in Organometallics with Automated Transition State Searches", ChemRxiv (2026),
[doi:10.26434/chemrxiv.15006374/v1](https://doi.org/10.26434/chemrxiv.15006374/v1) (preprint, not peer
reviewed).

`same_molecule` is mepd's one definition of "the same species"; every
command uses it. Stereochemistry is compared on stereo SMILES built from the
full bond graph (hydrogens included, so a tautomer differs), with bond orders
and stereo from 3D. When that cannot be decided (bond orders RDKit cannot
assign, so E/Z would be invisible), OpenBabel's stereo SMILES are a second
opinion that can only confirm "same"; otherwise the structures are not the
same molecule, and `Match.undetermined` says that side failed on this. `connects` takes an `IRCResult`, a Chain, or any sequence of
structures, and tries both directions.

## Contributing

The functions here are thin and composable on purpose. To add one:

1. Put the computation in a library module (`mepd/compute.py`,
   `mepd/compare.py`, `mepd/structures.py`, or a new one). It takes
   structures and an engine, returns results, raises `mepd` errors, and
   neither prints nor writes files.
2. Export it in `mepd/__init__.py` (`_API_EXPORTS`), document it here, and test
   it in `tests/test_public_api.py`: fast tests with toy inputs, plus one with a
   real engine that skips when the program is missing.
3. Make the command line call it rather than keeping its own copy.

To make an engine support minimization, TS search or IRC, give it any of:
`compute_geometry_optimization(node, keywords=None) -> list[StructureNode]`
(the trajectory, last = minimum), `compute_geometry_optimizations(nodes,
keywords=None)` (batched), `compute_transition_state(node, keywords=None) ->
StructureNode`, and `compute_irc_chain(ts_node, keywords=None) -> Chain`.
An engine without its own IRC gets geomeTRIC's.
