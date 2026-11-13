# Reaction network expansion

`mepd discovery expand` proposes the products of a structure by rewriting its
bond graph instead of sampling Hessians, optimizes them, and repeats from each
new species. With `--connect` it then runs a recursive path search for every
proposed reaction and builds the network from the paths it finds. In the web
UI this is the **Reaction network expansion** operation for a selected
structure.

```bash
mepd discovery expand "C=CCOC=C" -i gxtb.toml                 # one round: products of the seed
mepd discovery expand seed.xyz -i gxtb.toml --rounds 2 -H     # products of the products, Hessian-checked
mepd discovery expand seed.xyz -i gxtb.toml --connect         # then run a path search per reaction
```

## How products are proposed (`--generator bond-rules`, built in)

1. **Enumerate.** Every combination of up to `--n-break` (2) bond breaks and
   `--n-form` (2) bond formations is tried. A bond can only form between
   atoms that are at most `--form-distance` (4 Å) apart in the source.
2. **Filter.** Each atom must keep a normal number of bonded neighbours
   (C 1–4, N 1–4, O 1–2, H 1, halogens 0–1, …). The new graph must also have
   a Lewis structure with the system's total charge (RDKit
   `DetermineBondOrders`).
   - By default the product must be closed-shell, with exactly
     `multiplicity − 1` unpaired electrons. `--allow-radicals` also admits
     carbenes and diradicals.
   - By default products with separated formal charges are rejected.
     `--allow-zwitterions` admits them; that is needed for isocyanides
     (HCN → HNC) and CO.
3. **Deduplicate and rank.** Products are deduplicated by canonical SMILES
   and ranked with the fewest bond changes first, then by how close the
   atoms that form bonds are. At most `--max-products` (50) are kept for
   each source species.
4. **Build a guess.** Each product's 3D guess is built from the source
   geometry by a restrained relaxation onto the new bonds. It keeps the
   source's atom order, so it can go straight into a path search. Randomly
   kicked restarts get it out of symmetric traps; for example, an H
   migrating along a linear molecule has to go around the middle atom.
5. **Optimize.** The guess is optimized at the level of theory in `--inputs`.
6. **Classify.** Each optimized guess is recorded as one of:
   - a new species;
   - a known species, merged with it (the lowest conformer is kept);
   - `reverted` to its source;
   - `not_minimum`: it failed the Hessian check (`-H`).

   Species identity is connectivity plus stereochemistry.

Species within `--energy-window` (60 kcal/mol) of the seed are expanded in
the next round. If a later proposal matches a species already found, it is
recorded as an edge to that species without being optimized again.

## Stopping the network from growing forever: flux steering

With `--rounds` > 1 the expansion is steered by kinetics by default
(`--steer auto`; `--steer window` restores the plain energy window). Every
round:

1. **Propose and optimize products** of the species chosen for expansion (as above).
2. **Find a TS for every new reaction.** mepd runs a recursive path search
   (MSMEP, with the profile's path method), then TS optimization and IRC for
   every step it splits into. A step counts only if its IRC runs between two
   different minima. IRC ends that are no known species are minimized and
   join the network as intermediates. Everything goes in `pairs/` and `ts/`.
3. **Simulate the kinetics.**
   - Eyring rates come from each verified TS:
     k = (k<sub>B</sub>T/h) exp(−ΔE<sup>‡</sup>/RT).
   - Barriers are electronic energies, measured from each species' lowest
     conformer found.
   - Reverse rates come from the same TS, so detailed balance holds.
   - The first-order network is integrated from pure seed for `--time`
     seconds at `--temperature`.
4. **Expand by flux.** Only species whose *concentration flux* (the total
   material that flowed into them, as a fraction of the seed) reaches
   `--flux-threshold` are expanded next round. When none does, the network
   has converged and the run stops, even if `--rounds` allows more.

```bash
mepd discovery expand seed.xyz -i gxtb.toml --rounds 5                        # 298 K, 1 h, threshold 0.01
mepd discovery expand seed.xyz -i gxtb.toml --rounds 5 --temperature 600 --time 10
```

`summary.json` records:
- `steering`: the settings;
- `steps`: each verified step, its barriers in both directions, and its TS/IRC files;
- each species' `flux` and its `max_concentration` / `final_concentration`;
- each round's `kinetics`.

Limits:
- Each proposed product is one species, a pair of fragments included, so
  every step is first order. There is no bimolecular recombination.
- Barriers are electronic, with no thermal or entropic corrections.
- Fast equilibria count their back-and-forth traffic as flux, so
  equilibrating species always pass the threshold, as they should.
- The cost is one path search plus TS/IRC per reaction; `--workers`
  parallelizes the TS/IRC stage and the pair searches.

## Methods and references

| stage | method | reference |
|---|---|---|
| Enumeration | break ≤ n and form ≤ m bonds on the molecular graph, filtered by coordination | ZStruct: P. M. Zimmerman, *J. Comput. Chem.* **34**, 1385–1392 (2013), [doi:10.1002/jcc.23271](https://doi.org/10.1002/jcc.23271); the "b2f2" enumeration of YARP: Q. Zhao, B. M. Savoie, *Nat. Comput. Sci.* **1**, 479–490 (2021), [doi:10.1038/s43588-021-00101-3](https://doi.org/10.1038/s43588-021-00101-3) |
| Lewis-structure filter | bond orders and formal charges from connectivity (RDKit `DetermineBondOrders`) | xyz2mol: Y. Kim, W. Y. Kim, *Bull. Korean Chem. Soc.* **36**, 1769–1777 (2015), [doi:10.1002/bkcs.10334](https://doi.org/10.1002/bkcs.10334) |
| Product guess geometry | restrained relaxation of the source geometry onto the product bonds | mepd's own heuristic, not a published method |
| Flux steering | Eyring rates from verified TSs, first-order microkinetics, expand by concentration flux | concentration-flux-steered exploration: M. Bensberg, M. Reiher, *Isr. J. Chem.* **63**, e202200123 (2023), [doi:10.1002/ijch.202200123](https://doi.org/10.1002/ijch.202200123); rate-based model enlargement (as in RMG): R. G. Susnow, A. M. Dean, W. H. Green, P. Peczak, L. J. Broadbelt, *J. Phys. Chem. A* **101**, 3731–3740 (1997), [doi:10.1021/jp9637690](https://doi.org/10.1021/jp9637690) |
| Live-view animation | geodesic interpolation from source to product | X. Zhu, K. C. Thompson, T. J. Martínez, *J. Chem. Phys.* **150**, 164103 (2019), [doi:10.1063/1.5090303](https://doi.org/10.1063/1.5090303) |

The same list is written to `summary.json` (`methods`) and shown with the
results in the web UI.

**What is ours and what isn't.** The expansion is built on these published
methods, and anything that uses or reports it should cite them. mepd
reimplements them; it contains no code from ZStruct, YARP or Chemoton. In
particular, the product enumeration is YARP's b2f2 scheme (Zhao & Savoie,
2021), built on the same break-and-form idea as ZStruct (Zimmerman, 2013). The flux-steered
growth is Bensberg & Reiher's concentration-flux criterion (2023), in the
spirit of RMG's rate-based enlargement (Susnow et al., 1997). What mepd
adds is the glue:
- the atom-mapped 3D guess;
- optimization and species merging;
- TS verification through its own path search, TS optimization and IRC;
- the live view.

**Compared with YARP's own filters.** mepd's enumeration only forms bonds
between atoms within `--form-distance` (4 Å), and it filters with a
coordination check and RDKit's Lewis-structure check instead of YARP's
Lewis-score and formal-charge filters. So the two propose somewhat different
sets. On 6-hydroxyhexanal (`OCCCCC=O`, b2f2), mepd proposes 48 species and
YARP 65, with 39 in common. YARP's extras include ring closures between
atoms that start more than 4 Å apart; a larger `--form-distance` brings those
in. mepd in turn keeps some products that YARP's filters drop.

## CREST msreact (`--generator crest-msreact`)

CREST's mass-spectrometry fragment generator (`crest <xyz> --msreact`,
CREST 3) adds repulsive potentials on bonds (and attractive ones between H
and lone pairs), optimizes with GFN2-xTB, and keeps the distinct fragments
and isomers. Every product keeps the input's atoms in order, so mepd
re-optimizes them at the run's level of theory, classifies and connects
them like bond-rule products. Use it for a molecule's likely fragments (read
backwards: precursors) and nearby isomers without enumerating bond changes.

    mepd discovery expand seed.xyz -i inputs.toml --generator crest-msreact \
        --generator-option mode=fragments      # all | fragments | isomers

Options: `mode`, `nbonds` (how many bonds apart the repulsive potential
reaches, default 3), `nshifts` (extra optimizations from randomly shifted
atoms). `--max-products` keeps the lowest GFN2-xTB energies. It needs the
`crest` and `xtb` programs on PATH (msreact runs `xtb` for its constrained
optimizations); CREST runs single-threaded. CREST's `-reactor` mode is not
used: it only analyses an xtb metadynamics trajectory made beforehand.

## Nanoreactor MD (`--generator nanoreactor`)

Hot molecular dynamics of the structure inside a spherical wall that
periodically squeezes it (the piston of `mepd discovery nanoreactor`, with
xtb's MD), followed with the nanoreactor's bond-change detection. The
structure right after each reaction event is a product: whole, in the
input's atom order, one per distinct bond graph, earliest first. States in
which an atom lost all its bonds (the end of a run hot enough to tear the
structure apart) are skipped.

    mepd discovery expand seed.xyz -i inputs.toml --generator nanoreactor \
        --generator-option temperature=3000 --generator-option time_ps=20

Options: `temperature` (K, default 2500), `time_ps` (10), `compress`
(narrow/wide wall radius, 0.7), `period_ps` (1), `md_method` (auto: GFN2 with
xtb, else g-xTB), `seed`. Unlike the Nanoreactor operation, nothing is cut
out: the whole structure is the product. On a QM/MM system it runs on the
capped QM region (see qmmm.md); in the web UI it is offered as a method of
the network card for QM/MM systems.

On a QM/MM system, `--generator-option embedded=true` runs the MD with
QM/MM forces instead (mepd/discovery/qmmm_reactor.py): the environment is
present all along, the QM atoms are thermostatted at `temperature`, the
moving environment near `environment_temperature` (default 300 K, stronger
friction, so it takes up the heat), frozen atoms stay fixed, and the piston
wall acts on the QM atoms only. Events are detected on the QM atoms; each
product is the whole system right after an event, unless that frame changed
a bond in the MM region or stretched a QM/MM cut bond (both are counted and
reported). It costs one QM/MM gradient per MD step (about 0.07-0.1 s for a
small QM region in a GFN-FF water shell: tens of minutes per 10 ps).

In the web UI the expansion is one form in four stages, each chosen on its
own: **Grow** (bond enumeration, CREST msreact, or the Nanoreactor, whose forces
you switch on: Piston, Collide, or both, each trajectory streamed live), **Connect** (Fast, path search, or no TSs),
**Steer** (within reach, where material flows, or an energy window) and **Run**
(the live queue or rounds). Choices that cannot run together are greyed out with
the reason, and a stage moves to the nearest allowed choice when another rules
it out. Hessian sampling and basin hopping are other methods of the same card.

## Other generators

Generators are registered in `mepd/discovery/generators.py`: `bond-rules`,
`crest-msreact` and `nanoreactor` are built in. To add one (autodE, Chemoton, …), write a
`propose(symbols, coords, edges, **settings)` that returns
`(proposals, stats)` (or, with `kind="structures"`,
`propose(structure, *, max_products, options)` returning product
Structures) and register it in `GENERATORS`.

These plug in in place of the built-in generator:

- **Your own code:** `--generator package.module:function`. It is called as
  `function(structure, **options)`, with options from repeatable
  `--generator-option key=value` (values parsed as JSON). It must return
  product `qcdata.Structure`s, or xyz paths, **in the source's atom
  order**.
- **Another tool's output:** `--products products.xyz` imports products that
  another tool (autodE, Chemoton, …) already proposed for the seed. The
  file is a multi-frame xyz, atom-mapped to the seed. mepd then optimizes,
  classifies, and (with `--connect`) connects them as above.

## Output

| file | contents |
|---|---|
| `species.xyz` | the seed first, then every species found (lowest conformer) |
| `species/species_<k>.xyz` | one file per species, e.g. to pass to `mepd network-splits` or `channels` |
| `proposals.xyz` | every product guess, before optimization |
| `rejected.xyz` | species that failed the Hessian check |
| `summary.json` | the methods and references used; the species (SMILES, round, energy relative to the seed, Hessian record); one record per proposed reaction (source, target, bonds broken and formed, outcome, whether it landed on the proposed connectivity); rounds; `connections` |
| `pairs/`, `network.json` | with `--connect`: one MSMEP tree per reaction, and the network built from them (as in `mepd network-splits`) |

## What a run records

Beside its results, `summary.json` records why each number is what it is, for whoever measures the explorer
afterwards. Recording changes nothing a run does: no extra calculation, no decision reads these fields. Times
(`t`, `t_*`) are epoch seconds, as in the event log (`network_aware.jsonl`). `null` means the run does not have
the value; the field next to it says why.

Per species (`species[*]`):

| field | type | meaning | null when |
|---|---|---|---|
| `t_found` | number | when the species joined the network | a run resumed from a state saved before this field existed |
| `energy_terms` | object | what the final kinetics added to its electronic `energy`: `thermal_correction_kcal`; `steer_shift_kcal` (the steering level's single point minus the explorer's, relative to the seed's); `steer`: `"single_point"`, `"failed"` (with `steer_error`; the explorer's energy was used), `"not_asked"` (out of the kinetics' reach), `"seed_failed"` (the seed's single point failed: no shift is used anywhere) | no kinetics ran; `thermal_correction_kcal` null: not computed (out of reach, or electronic kinetics) or failed (`steering.free_energy_gaps`); `steer` null: no steering level |
| `molecules` | list of `{key, atoms}` | the box's molecules as the molecule-level kinetics name them (stereo SMILES; a radical's bond graph; a complex's key), with their atoms | the kinetics are not molecule-level (`steering.kinetics`) |

Per step (`steps[*]`):

| field | type | meaning | null when |
|---|---|---|---|
| `free_energy_barrier_kcal` | `[from a, from b]` | the step's barrier on the basis the final kinetics use (`steering.kinetics_energy_basis`: thermal corrections, steering shifts, mirror-image terms), not clamped. For a step the kinetics use it is what `kinetics_barrier_kcal` was made from (equal unless the TS lies below an end). For a closed step it is given only when the run computed every term of it. For an unverified step or a path peak (`files.verified`, `files.ts_kind`) it is that structure's, not a checked TS's | a closed step with a term the run never computed: `free_energy_barrier_missing` lists them (e.g. `"steering single point of the TS"`); nothing is computed to fill it |
| `free_energy_barrier_missing` | list of strings | the terms the run never computed | the barrier is given |
| `ts_energy_terms` | object | as `energy_terms`, for the TS | as there |
| `closed` | object | why the final kinetics leave the step out: `reason`, `by` (`"box kinetics"` or `"molecule kinetics"`), and the numbers compared, `energy_used`, `value_kcal`, `cutoff_kcal` (null where the rule compares none) | the final kinetics use the step |
| `irc` | object | its IRC check: `ok`, `t`, `ends` (the species its two ends are), `asked` (the pair its TS search was asked for), `searched_between` (the pair it was filed under before), `new_species`, `path_search_asked_for` (the pair sent to a path search because the IRC joins others) | no IRC check of this run filed it (unverified, a path peak, a TS from channel sampling or a known run) |
| `placeholder_reason` | object | for `files.ts_kind: "path_peak"`: `why` is `"leaf_ts_failed"` (with `error`, and `fast_ts_error`, the pair's single TS optimization's) or `"leaf_ts_irc_failed"` / `"leaf_ts_irc_went_elsewhere"` (with `ts`, `irc_ends`, `error`) | the step has a TS |
| `t_found`, `t_opened` | number | when the step joined the network; when it first took part in a kinetics, live or recorded (open in the box kinetics and, at molecule level, a reaction of the molecule network) | `t_opened`: it never did |
| `reacting` | object | the step read as a reaction of molecules: `atoms` (of every reacting molecule), `reactants`, `products` (molecule keys), `spectators` (`{key, atoms}`), `with_neighbours` when the kinetics also keep the spectators its whole-box TS touches; or `{"unreadable": reason}` | the kinetics are not molecule-level |
| `ts_group` | integer | the lowest index of the steps the kinetics count as one TS with it: mirror-image twins both found, one reaction found in several boxes (its own index when alone) | no kinetics ran |

`closed.reason` is one of:

| reason | rule | numbers |
|---|---|---|
| `end_out_of_reach` | no route from the seed reaches an end (`species`) within the access cutoff | `value_kcal`: the largest climb out of a well on its best route, electronic energies at the explorer's level (null: no route); `ts_above_seed_kcal` |
| `ts_above_cutoff` | both ends are within reach, the TS lies above the cutoff | `value_kcal`: the TS above the seed, electronic, explorer's level |
| `no_thermal_correction` | the TS (`of: "ts"`) or an end has no thermal correction (`why`) | none |
| `no_irc_check` | the run ended (time, stop, species cap, settled) before its IRC check | none |
| `reached_only_through_unchecked_steps` | closed with the unchecked steps that alone lead to it | none |
| `path_peak` | its TS search failed: a path's highest image is no saddle (molecule kinetics) | none |
| `no_reaction` | its ends hold the same molecules (a conformer change, another arrangement) | none |
| `spectator_mismatch` | the molecules that look on differ between its ends | none |
| `whole_box_ts_unchecked` | a TS of the whole box with spectators that no IRC checked | none |
| `duplicate` | the same reaction with the same TS found in another box: `of` is the step kept | none |
| `ts_above_closed_threshold` | its TS lies more than 0.5 Hartree above both ends | `value_kcal`, `cutoff_kcal` |

Per proposed reaction (`reactions[*]`): `t` (when its outcome was known), `batch` (a batched generator's batch,
from 1; null for the seed, whose batches are one job, and for generators without batches), `proposed_key` (async
schedule, a structures generator such as the reactor: the stereo SMILES of the product as the generator made it,
before it was minimized at the explorer's level; null otherwise or when it can't be built) and `not_searched`
(null, or why its pair got no TS search: `{"reason": "product_above_cutoff", "energy_used", "value_kcal",
"cutoff_kcal"}`).

Run level:

- `steps_removed`: every TS found that is no step of the network, or stopped being one: `{t, label, a, b,
  reason}` with `reason` one of `ends_one_species`, `end_not_a_species` (an end's minimization failed, or the
  species cap), `same_ts` (`of`: the step that has this TS), `path_peak_not_lowest`, `path_peak_superseded`
  (`by`), `irc_failed` (`irc.why`: the IRC job's error), `irc_ends_one_species`, `irc_end_not_a_species`,
  `same_ts_after_irc` (`of`), `superseded_by_channel_sampling`. The IRC ones carry `irc` as above. Each is also a
  `step_removed` line of the event log when it happens, and each passed IRC check an `irc` line.
- `steering.level`: `explorer` and `steer` (null without a steering level), each `{engine, key, profile_sha256}`:
  the engine class, the level of theory as the run's caches key it (program, method, basis and the other
  settings that change an energy), and the SHA-256 of the profile file.
- `steering.steer_fallbacks`: the structures the final kinetics use on the explorer's energy because their
  steering single point is missing: `{kind: "species", index, why}`, `{kind: "ts", step, label, why}`, or
  `{kind: "all", why}` when the seed's failed. Empty: every structure the kinetics use has its single point.
- `record_error`: present only if this record could not be made (the run's results are unaffected).

`summary_partial.json` carries `t_found` per species and `t_found`, `t_opened`, `irc`, `placeholder_reason` per
step as they stand; every line of `live/events.jsonl` has its time `t`.

## Limits

- The enumeration is combinatorial. `--n-break 2 --n-form 2` on a molecule
  with 30 atoms reaches the `max_combinations` cap (500k); `stats.clipped` in
  `summary.json` reports when that happens. Lowering `--form-distance`, or
  using `--n-form 1`, keeps it small.
- The rules propose graph changes, not mechanisms. A proposal that optimizes
  back to its source says nothing about the barrier. `--connect` is what
  checks that a path (TS plus IRC) exists.
