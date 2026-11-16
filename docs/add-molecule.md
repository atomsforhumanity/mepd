# Add a molecule to a finished network

You have a reaction network (an exploration, adopted into a workspace) and a product you want more of. `mepd addmol`
answers: what does one more molecule C in the flask (a possible catalyst, an additive) do to that product? It works
without the web UI, on files a program or a generative model can read and write:

- the **query** file: the network and what limits the target today;
- one **result** file per candidate molecule, rewritten as numbers come in, cheapest first.

```bash
mepd addmol query  RUN/ --target 'CC(=O)CC(C)O' --initial 'CC(C)=O=1.0' --initial 'CC=O=1.0' \
                   --temperature 298.15 --time 86400 -o query.json    # RUN/: a finished `mepd discovery expand` run,
                                                                      # or a workspace folder
mepd addmol test   query.json -c 'OC(=O)[C@@H]1CCCN1' --amount 0.1 --workers 4 -o results/
mepd addmol test   query.json --candidates candidates.json -o results/     # many, with poses or estimates
```

In Python: `mepd.discovery.parent` (an explorer run as the parent: `network`, `query`, `ts_step`),
`mepd.web.added` (a workspace as the parent: `network`, `query`, `put`, `adopt_run`) and `mepd.addmol` (the
kinetics, on plain dicts: `added`, `readout`, `limits`, `first_order`, `bystander`, `query`, `result`).

What exists today is the format, the kinetics and one layer of estimates (the bystander layer, with a stand-in
placer whose numbers move by kcal/mol between repeats). Layers that verify a step with the molecule present, find
steps in which it takes part, or find new species it forms, write their numbers into the same result file as they
are built; until then such numbers can be given with a candidate (`estimates`, below).

## The parent network

- **An explorer run** (`mepd addmol query RUN/`, `mepd.discovery.parent.network`): the explorer's own
  molecule-level network, built by its own code (`mepd.discovery.molecules.build`): molecules with free energies
  fitted over the run's boxes, a step found beside another molecule counted through its reacting molecules alone
  where the explorer counts it so, mirror images as the run decided them. Energies are the run's: electronic +
  its recorded thermal correction and steering shift, relative to the starting box's molecules. Every
  IRC-verified step is in it, also those the run's own kinetics had closed (at 298 K an uncatalysed parent may
  have no open step at all). A thermal correction the run did not record (it records none for most closed steps)
  is computed now with the run's own code and profile (`--inputs`, default the `.toml` beside the run); a step
  for which that is not possible is left out (`notes` says how many): the molecules' free energies are fitted
  over the boxes, and one box on an electronic energy would move them all. Each step's
  `geometry` holds the reacting molecules only (`atom_indices_in_run`: their indices in the run's files). A
  mirror-image twin of a step (`mirror_of`; its original names it under `mirror`) has the original's frames
  reflected (`reflected`): for a molecule that is its own mirror image the two are one step (the bystander layer
  computes the pair once and copies the row), with a chiral molecule they are two different steps, each
  computed, and the read-out's two rows name each other (`mirror`). A pose applies to the step it names.
- **A workspace** (`mepd addmol query WORKSPACE/`, `mepd.web.added.network`): the workspace's kinetic model
  (*Analyze › Kinetics*: IRC-verified steps, free energies at the 1 M standard state from quasi-RRHO on GFN2-xTB
  Hessians; [kinetics.md](kinetics.md)), for networks built in the web. An explorer run adopted into a workspace
  loses part of what the run knew (last section): use the run folder itself.

The query file says which it is (`source`).

## The kinetics with a molecule added

`mepd.addmol.added` puts C into the parent network from a list of estimates. It does not care where they come
from.

| estimate | what it adds |
|---|---|
| `species`: C's binding to a species X | a species **X.C** with G = G(X) + G(C) + binding, and the barrierless step X + C ⇌ X.C |
| `steps`: C's binding to the reactant end, TS and product end of a parent step | the parent step with C beside it, its TS at G(TS) + G(C) + binding(TS). From X.C to Y.C where an end is one molecule; from the separated molecules + C where an end is several molecules |
| `new_species`: a species formed from named species (some released) | that species, G = Σ G(formed from) − Σ G(released) + energy; it holds C when its makers do |
| `new_steps`: reactants, products, the TS above the reactants | that step. C on both sides: it takes part and is returned |

Each estimate has `status` `est` or `verified`; anything else (a bystander run in which C reacted, did not move,
or failed) is listed under `failed` and not used. Each energy is `{"electronic": x, "free": y, "spread": s}`: the
kinetics take `free` when the network is on free energies and fall back to `electronic` with a note.

Two consequences of counting the TS from the separated molecules:

- C's binding to the reactant end never enters a rate twice. From dilute X and C the rate is
  (k_B T/h) exp(−(barrier + binding(TS))/RT) [X][C], whatever binding(reactant) is. (The substituent variants'
  `estimate_ts` adds a barrier *shift* to the parent's TS height; for a molecule that is on both sides of the step
  that would count the barrier from the complex as if it were from the separated molecules. Added-molecule
  estimates do not go through it.)
- Saturation, the resting state and product inhibition come out of the kinetics: when X.C is deep, C sits in it
  and the rate stops growing with X; when Y.C is deep, C stays on the product.

G(C) cancels in every rate, so C needs no free energy of its own.

**A step given with its TS.** A `new_steps` entry may come with the TS itself instead of a barrier:
`"ts": {"atoms", "coordinates", "charge", "multiplicity", "ends": [reactant end, product end]}`, all in one atom
order (a verified TS and its IRC ends, a channel search's). `mepd addmol test` then puts it on the parent's scale
as a whole (`mepd.discovery.parent.ts_step`, `on_parent_scale`):

- G(TS) by the explorer's free-energy code (quasi-RRHO at 1 M), the TS counted as the explorer counts a TS file
  (what its file says it was optimized on; without that record, by its ends: molecules that take no part apart);
- minus **one reference per molecule, its lowest known form**: every parent molecule at the parent network's own
  free energy (for a run through its starting box, whose free energy and geometry the query keeps under
  `reference`; this needs the TS's parent molecules to be, together, the starting box's atoms, which a parent
  step with the molecule added always is), and the added molecule at its lowest free energy over the candidate's
  conformers and the form found in the complex. Not each molecule as it happened to sit in the complex: the
  entry keeps that number too (`free_from_molecules_minimized_alone`) and says which reference it used
  (`reference`, `added_form`).

The entry's `reactants` and `products` are read from the geometry (the molecules that take part; C as `added`),
`spectators` lists the others, and `parent_step` is the parent's step between the same molecules. A barrier
"from the complex" is never added to a separately estimated binding; only bystander estimates (not stationary
points) use the placer's free-energy term.

**Levels.** Every step carries its `level`. A run's own steps: `explorer` (the run's profile), `explorer+steer`
(its TS carries the run's steering single point) or `explorer+steer (lent)`. A step given with its TS is at
`explorer` after the above. `mepd addmol test --steer STEERING_PROFILE` then puts it on the parent's steering
level with the run's own single-point code: one single point of the TS (counted as above), of the added
molecule's reference form and of the starting box; the barrier moves by their difference (`steer` on the entry),
the level becomes `explorer+steer`. This is the slow part (a minute per 34-atom single point on one core), so
it runs after the low-level answer is written, the steps on the routes that carry the target's material first,
the file rewritten after each. The read-out's `mixed_levels` is `true` when the steps the comparison rests on
(the routes that carry the target's material, the lowest-peak routes without and with C) are not all at one
level; `levels` lists them by level. Steering a step with C beside a parent whose own steps on the route carry
no single point makes the comparison mixed, and the file says so.

## What you read (`readout`, in every result file)

Parent and variant are run from the same mixture, the variant with C at its amount.

- **`target`**: its amount at the end without (`parent`) and with (`with`) C, `factor` (with / without), the time to
  a stated fraction of each run's own final amount (`time_to_fraction_s`) and the time at which the variant holds
  the amount the parent needs that long for (`time_to_parent_amount_s`). `resolved: false`: the amount is below
  1e-9 of the starting material, which is integration noise; factors and times are then `null`.
- **`steps`**, one row per parent step: `before` and `after` are **apparent barriers**: the TS above its reactants'
  chemical potentials, G + RT ln(c / 1 M), at their time-averaged amounts. The forward rate is
  (k_B T/h)(1 M) exp(−apparent/RT), so this is the barrier from the network's own resting state at the stated
  amounts: an intermediate that is hardly there makes it high, a reactant held in a complex makes it higher.
  `after` is the lowest of the step itself and its versions with C; `through` says which, `status` whether that is
  `est` or `verified`, `direction` which way the step is read (the way its material goes). `barrier` is the plain
  TS above that side at 1 M. `group` sorts the rows: `through` (the lowest version has C in it: the table to
  show), `amounts` (the step's own TS; its apparent barrier moved only because its reactants' amounts moved: a
  second, folded group), `none`.
- **`routes`**: where the target's material came from, as lists of steps with the share each carried, by a greedy
  split of the steps' net extents (`flux_routes`); `with_added` marks routes with C in them.
- **`peak_routes`**: the route to the target with the lowest highest point, without and with C (see below).
- **`added`**: C's `resting_state` (the C-holding species with the largest time-averaged amount) and its fate, as
  two facts side by side and a word:

  | number | meaning |
  |---|---|
  | `on_route`, `route_share` | whether at least half of the target's material goes through steps with a C-holding species, and that share (`null` when no material was made) |
  | `route_peak`, `route_peak_with_added` | the highest point of the lowest-peak route to the target with C in the flask, and whether that route has C in it |
  | `turnovers` (= `returns`) | how often, per C put in, a step that changed what C carries handed C back free or in a complex, in the run's time |
  | `shares` | where C is at the end: `free`, `bound` (in complexes), `covalent` (in new species that hold it) |

  `fate`: C took part when it is on the route, or `turnovers` or `covalent` reach 0.001. When it did not: **bound
  and inert** if at least half of it is bound or a complex holds at least as much of a species as is left free,
  else **no part**. When it did: **regenerated** if `turnovers` ≥ `covalent`, else **consumed**. A slow catalyst
  on the only route reads `regenerated` with few turnovers, and one held up on its product reads `regenerated`
  with the product complex as resting state: read the numbers with the word.
- **`mixed_levels`**, **`levels`**: whether the steps compared are at one level of theory (above).
- **`first_order`**: the estimate below, beside the kinetics' own `factor`.
- **`new_species`**: the complexes and new species with their amounts.

## Sensitivity: what limits the target, and the order of work

`limits` (in the query under `limits`):

- Where the network makes the target: the degree of rate control X of every TS and the thermodynamic degree of
  control of every species for the target's amount at the end, X = d ln Q / d(−G/RT)
  (`mepd.microkinetics.degree_of_control`, central differences of ±0.2 kcal/mol). `order`: the steps by |X|,
  largest first; X > 0 `lower` (a lower TS gives more target), X < 0 `do not lower` (a competing step). `traps`:
  species with X < 0 (stabilizing them costs target). A target of several species (stereoisomers) is their sum.
- Where it makes none of it (below 1e-9 of the starting material) X is **not defined**: `defined: false`, the
  reason in `why`, `control: null`. Then `routes` holds the lowest-peak routes from the mixture to the target, and
  `order` their steps, the highest above the access cutoff first (`cutoff_kcal`: the Eyring barrier crossed with a
  half-life of the run's time), all `lower`. Nothing is known then about competing steps.

A lowest-peak route is followed as molecules in hand: a step may run when each reactant is a starting species (at
its chemical potential) or in hand, and puts its products in hand together with whatever was made along with its
reactants. The height of a TS is the cost of what its reactants were made with plus the TS above them, so two
products of one step used together later are paid for once, and an unrelated side reaction changes no height.

The **first-order estimate** from a set of shifts is d ln Q = Σ X_i (−ΔΔG_i/RT). For an added molecule at amount c
(taken as all free) a parent step's shift is −RT ln(1 + c exp(−binding(TS)/RT)) and a species' likewise from its
binding (`shifts`). It is local: right while the target's amount is linear in the rate constants, wrong near full
conversion or when C saturates, and silent about new steps and new species. It is `null` where X is not defined.

## The query file (`mepd-addmol-query`, version 1)

Plain JSON. Energies in kcal/mol, lengths in Å, amounts in mol/L, times in s (`units`).

| key | content |
|---|---|
| `schema`, `version`, `units` | |
| `conditions` | `temperature`, `time_s`, `initial` {species id: amount}, `held` |
| `temperature`, `energies` | `free` or `electronic`; `energy_reference` says what the numbers are relative to |
| `level` | `key` of the level of theory and its `profile` (the TOML text; `mepd addmol test` runs the bystander layer with it) |
| `target` | `species` (ids, counted together), `final` amount, `resolved`, `fraction`, `time_to_fraction_s` |
| `times` | the output times of every `series` |
| `species[]` | `id`, `name`, `smiles`, `charge`, `multiplicity`, `g`, `atoms`, `coordinates` (its own atom order), `amount` {`initial`, `final`, `max`, `mean`}, `series`, `control` (X, or null), `status`, `level` |
| `source` | `explorer run` or `workspace` |
| `reference` | a run only: its starting box (`seed_g_hartree`, `seed_symbols`, `seed_coordinates`, `seed_molecules`), what a TS with an added molecule is measured from |
| `steps[]` | `id`, `label`, `reactants`, `products` (ids, repeated for 2 A), `shuttles` (on both sides), `g_ts`, `barrier` {`forward`, `reverse`}, `apparent_barrier`, `extent`, `control`, `status`, `level`, `geometry`; from a run also `run_step`, `mirror_of` |
| `steps[].geometry` | `atoms`, `charge`, `multiplicity`, `reactant`, `ts`, `product`: three frames in **one atom order** (the TS's; the two ends are its IRC's ends), and `bonds_broken`, `bonds_formed` as pairs of 0-based indices into that order. `null` when the step has no TS and IRC files |
| `limits` | the section above: `defined`, `why`, `basis`, `control`, `order`, `traps`, `routes`, `cutoff_kcal` |
| `notes`, `left_out` | how the free energies were made; the number of reactions without a verified TS (not in the model) |

`g` and `g_ts` are on one scale per file (`energy_reference` says which: absolute at the level for a workspace,
relative to the starting box's molecules for a run); only differences between the two sides of a step mean
anything. Species ids are the workspace's ids, or for a run the explorer's molecule keys (stereo SMILES). `status`
and `level` tag every number of a record: `verified` (an IRC-verified TS, a minimum) or `est`. The file is also the
network `mepd.addmol` runs: its top-level `temperature`, `energies`, `species` and `steps` are all the kinetics
need.

## Candidates

`-c` takes a SMILES or an xyz file; `--candidates` a JSON list:

```json
[{"name": "L-proline", "smiles": "OC(=O)[C@@H]1CCCN1", "amount": 0.1},
 {"name": "from a model", "atoms": ["O", "H", "H"], "coordinates": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
  "charge": 0, "multiplicity": 1, "amount": 0.05,
  "pose": {"step": "r_1a2b3c4d", "frame": "ts", "coordinates": [[1.2, 0.3, 2.9], [1.9, 0.9, 3.1], [0.5, 0.8, 3.3]]},
  "estimates": {"new_steps": [], "new_species": [], "steps": [], "species": []}}]
```

`coordinates` are in the candidate's own atom order. A `pose` gives the candidate's atoms beside one step, in the
Cartesian frame of that step's `geometry` (frame `r`, `ts` or `p`): one more starting placement for the bystander
layer, never a replacement for the generic ones. `estimates` are numbers from elsewhere, in the form of the table
above, each with `status`, and ideally `level` and `source`; a step may name the added molecule as `added` and a
complex as `<species id>.added`.

## The result file (`mepd-addmol-result`, version 1)

One per candidate, `<output>/<name>.json`, written whole each time (a reader never sees half a file).

| key | content |
|---|---|
| `schema`, `version`, `units`, `updated` | |
| `status` | `running`, `done` or `failed` |
| `query` | the `target` and `conditions` it was tested at |
| `candidate` | as given |
| `layers[]` | `name`, `status`, `done` / `total`, `wall_s`, `level`, `what` |
| `estimates` | every number used, in the form `added` takes, each with `status` (`est` / `verified`), `layer`, `level`, `spread` and where given `source` |
| `failed[]` | what gave no usable number, and `why` |
| `readout` | the section "What you read"; `null` with an entry in `failed` if the kinetics could not be built |
| `timings` | seconds per part |

## The bystander layer and its contract

For every parent step with a geometry (in the query's order of work) and every species that holds material, C is
placed beside the reactant end, the TS and the product end, relaxed with the step's own atoms held, and its binding
to each frame recorded. `mepd.addmol.bystander` is the one function the CLI calls; today it is a stand-in:

```python
bystander(frames, molecule, engine, *, temperature=298.15, starts=(), seed=0, poses=6) -> dict
```

- `frames`: `{"symbols", "charge", "multiplicity", "r", "ts" (or None), "p"}`, Å, one atom order.
- `molecule`: `{"symbols", "charge", "multiplicity", "conformers": [coords, ...]}`, minima at the engine's level.
- `starts`: `[{"frame", "coords"}]`, proposed poses, added to the generic placements.
- returns `status` (`ok`, `reacted`, `unmoved`, `failed`) with `why`; `binding` per frame
  `{"electronic", "free", "spread"}` with `electronic` = E(frame + C) − E(frame) − E(C) for the lowest complex whose
  C kept its bonds, `free` the binding free energy at 1 M (G(X.C) = G(X) + G(C) + free); `barrier_shift`,
  `reaction_shift`; `complexes` (frame atoms first, then C's); `free_method`, `spread_method`, `poses`, `wall_s`.

The stand-in places C at random around the TS (`mepd.complexes.packed`), carries each placement to both ends, and
estimates `free` as the electronic binding plus the rigid-rotor and translational free-energy change, which
overstates the cost of binding (the six new intermolecular vibrations are not counted). Its `spread` is the
difference between the lowest binding over two halves of the placements. Treat its numbers as a way to order
work.

## Adopting an exploration into a workspace, and what is lost on the way

For `mepd addmol` use the run folder itself (above). To look at a finished run in `mepd web`,
`mepd addmol adopt RUN/ WORKSPACE/` (`mepd.web.added.adopt_run`) reads a finished run's `summary.json` as the web reads a running
one: every species becomes a node, every step goes onto the edge of its pair, the lowest IRC-verified one counting;
then each molecule of the run's mixtures is minimized on its own. The workspace's kinetics are not the explorer's:

- The explorer's species are whole boxes; the workspace's are molecules. A step found in a box with another
  molecule beside it becomes a reaction with that molecule on both sides (`shuttles`), first order in it, and its
  TS's thermal correction is the whole box's: the bystander pays for its lost translation. The explorer's rule that
  counts such a step through the reacting molecules alone is not applied.
- One step per pair of species: other TSs of the same pair (conformers, other mechanisms) are not kept, and steps
  whose IRC was not verified are left out of the kinetics.
- Free energies are recomputed (GFN2-xTB Hessians at the stored geometries), not taken from the run; the run's
  higher-level single points and its record of why a step is closed are not carried.
- Mirror images are separate species (stereo SMILES); the explorer counts them as one. Name a target without
  stereo marks to count them together.
- One concentration per species, as given to `query`; the run's own kinetics are not read.

See also [network-variants.md](network-variants.md) (substituent variants, which this sits beside as the variant
kind `added_molecule`: `mepd.web.added.put`) and [kinetics.md](kinetics.md).
