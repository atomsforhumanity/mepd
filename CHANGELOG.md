# Changelog

All notable changes to mepd are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Add each change under `Unreleased` as you make it, in the group it belongs to
(Added, Changed, Deprecated, Removed, Fixed, Security). On a release, rename
`Unreleased` to the version and date, and start a new empty `Unreleased`.

## [Unreleased]

### Fixed

- **A running expansion's step whose IRC joins other species goes to them.** The explorer reports a step under the pair it searched before its IRC is checked; when the IRC's ends (read by molecular graph) are other species, the step leaves that pair's edge, saying where its IRC runs, and joins the edge of the two species it does connect (barriers from its TS's energy) once the run has both. Before, it stayed on the searched edge as unverified: the Wittig run's "ylide + acetone -> phosphine oxide + isobutylene" carried the oxaphosphetane's ring-opening TS (1.1 kcal/mol), the reaction page played that, and kinetics including unverified barriers used it.

- **Explorer (async): a generator's proposal job reserves no more workers than it runs.** With `n_seeds` set, the reactor runs that many trajectories at a time, but its job held `parallel` (= `--workers`) slots: 4 MD runs with 10 workers showed 10/10 busy and kept the other 6 from anything else.

### Changed

- **Kinetics from an expansion's own start just works.** A network expansion queues one minimization job (per charge and spin) for the molecules of its mixtures that have no energy at its level -- its start's too, e.g. acetone and an ylide it was run on together -- as they appear: kinetics counts a reaction from its molecules apart, and the run optimizes each mixture whole. The Kinetics page now says first when the starting mixture is not in the model and why (no energy yet: **Compute them**), and when steps of a running expansion are left out until their IRC check (**Include them now**).
- **Web: the network expansion's steering cards say what they do.** "Material flow, within reach" (the default: the live kinetics' flux threshold, and only species some route reaches under the barrier cutoff), "Material flow, no cutoff" (the older rounds-only mode, a full path search per reaction) and "Energy window"; the manual and "?" entries say the same.

- **Web: network expansion in four stages, each chosen on its own.** The card's fixed presets (Bond rules, CREST msreact, "Reactor, live", Nanoreactor) are gone: **Grow** (bond enumeration, CREST msreact, collision MD, Reactor MD, the nanoreactor on QM/MM), **Connect** (Fast, path search, no TSs, plus channel sampling), **Steer** (within reach, where material flows, energy window) and **Run** (live queue or rounds) are chosen as cards, with the combinations the explorer cannot run greyed out with the reason; a stage moves to the nearest allowed choice when another rules it out, and says so. The grower is always named on the command line (the CLI's 'auto' picked an MD generator under access steering, so "Bond rules" with access steering ran collisions). A manual page (/manual/network-expansion) and the "?" entries explain each choice.
- **Explorer (async): the reactor's products join the network after every MD batch.** Each batch's products are minimized and their pairs queued for TS searches at once, while the species' next batch runs; batches, seeds and the convergence rule are unchanged (`reactor_generator.reactor_batch`, a generator's optional `propose_batch`). Before, nothing grew until all of a species' batches had converged. A species interrupted mid-batches restarts its batches on resume.

- **Web: a reaction page goes back where it was opened from.** Its breadcrumb names that page (Landscape, Kinetics, Reactions, a live network, Explore) and returns there, as does Esc (now at once: the page's own selection no longer takes a first press); the page's address keeps it (`?from=`), so reload and Back do too. Back on the Landscape, the route opened last is marked.

- **Network expansions: only steps whose IRC joins their two species count; the reaction page plays the step you opened; species no Lewis structure fits are drawn.**
  - A running expansion reports each TS under the pair it searched, before its IRC is checked; the check then moves it to the species its IRC really joins, or drops it (e.g. a search that slid into acetaldehyde's methyl rotation, a "TS" ~100 kcal/mol below the species it was reported between: the negative barriers). The web took those reports as they came. Now a live step is unverified (kinetics leaves it out by default) until its IRC's ends read as its two species (by molecular graph, `mepd/web/step_check.py`); the finished or stopped run's summary settles every edge: its real steps on their real pairs, and an edge that held a dropped or moved step back to proposed, saying why. Workspaces from before are set right when opened.
  - Each species' final, lowest structure from the run becomes a conformer of its node (the barrier floor), so no TS sits below the species it starts from.
  - A path's highest point (no TS found) reads as an upper bound on the reaction page.
  - The reaction page plays the very step on that edge (its own IRC, `GET /api/edges/<id>/step-path`, energies from its start's node), not the run's lowest path. Opened from the Landscape, it reads the route's direction and shows the kinetics' barrier beside the path's.
  - The live network shows a finished run's real steps only, and species no Lewis structure fits (ylides, hypervalent atoms: the reactor's "unusual bonding") drawn from their geometry with their formula (`mepd/web/depict_graph.py`). While the reactor's MD runs, "Reactor found so far" draws its products from the frames they formed in.

- **Network expansions count in kinetics.** After an expansion (finished, or stopped with Stop & analyze), Kinetics said no reaction had a barrier.
  - An expansion's steps now keep their TS's own energy on their edges (kinetics measures every barrier from it), from the live step events and from the finished summary. Workspaces from before get it back when opened: from the summary, and for the steps a stopped run's summary leaves out, from the run's step events.
  - The expansion's own optimized seed gives the seed's node its energy at the run's level (one more conformer). A seed that came in without one (a reactor's molecule, a drawing) dropped out of kinetics, and with it everything it starts.
  - Reactions to separated molecules (an expansion optimizes each product mixture as a whole) need each molecule's energy: Kinetics now names the molecules missing one and offers **Compute them** (one minimization per charge and multiplicity, seconds with g-xTB), then **Run kinetics again**.
  - A startup backfill no longer overwrites a known TS energy with "not found".

- **Web: the live network's steps say what they are, and its key reads in words.**
  - Steps end at the cards with an arrow the way material flows. The routes that carry material are drawn strong, with their barrier on them ("‡ 28.6"); steps that carry next to nothing fade back. In runs that did not record flows, the likely routes are marked instead: from the start to each species holding material, the path whose highest barrier is lowest.
  - Hovering a step shows it in words: from → to, the barrier and the barrier back, ΔG, the share of what leaves its start that goes that way, the energy level and the IRC check.
  - The key sits under the graph, each entry with a drawn sample, and lists only what the frame shows. Dashed lines and cards (g-xTB energy, DFT check to come) appear only in runs that have DFT energies.
  - From the live network, a simulation opens on itself in the calculation's Live tab (played and scrolled into view). The Live tab's kinetics table shows perceived SMILES for species the explorer left without one.

- **Web: a network expansion lands on its live network, which grows while it runs.**
  - Opening a network expansion (from Calculations, or right after Run) shows the live network, not the calculation page. Species appear as the explorer finds them, from the first one, before any kinetics.
  - Works with the explorer as it is on this branch: its kinetics refreshes are recorded with the job's clock, steps come from `live/events.jsonl`, and SMILES the explorer leaves empty (radicals) are perceived from each species' xyz (cached in `live/perceived_smiles.json`).
  - The side panel says what the explorer is doing now (busy workers, what each runs, work done) and has the ways in for a closer look: Reactor MD, Path searches, Results, Log, and the latest simulations as reactant → product, each opening its live view. The calculation page links back ("‹ The live network").
  - Runs without DFT steering are no longer greyed as "steering pending". Small networks stay together (capped spring length).

- **Web: the live network draws species as their structures, and works for any run.**
  - Each species is its 2D structure on a card whose area grows with the material it held (46–112 px). Hover a card for a large view with its name, energy and peak. Labels sit above the cards.
  - The side panel is "What forms" for every run: the main products as structures, named where they have an everyday name ("water + CO"), with their shares and how much of the start is left. A benchmark's measured selectivity shows only underneath, as validation.
  - A benchmark's named product (e.g. endo, both mirror twins) is one card. Species the explorer could not identify show only once they hold material.
  - Runs recorded before per-step flux (`step_list`) replay too: species from the kinetics snapshots, steps from `events.jsonl` (one per pair, its lowest barrier), edges without flux.
  - Three more examples from different chemistry: formic acid, acetic acid and methyl formate breaking down (1500 K explorer runs).

- **Web: the live network.** An explorer run (`mepd discovery expand`) watched as it grows, corrects itself and moves material: "Watch the network ›" on its calculation page, or the new example "Cp + methyl acrylate: watch the network grow" on Home.
  - One layout for the whole run, so species appear in place, newcomers ringed.
  - Node area grows with the material a species held. The starting mixture is a fixed mark, and a benchmark's products are named (endo, exo).
  - Open steps are coloured by the barrier the way material flows, as wide as their net flux, with dots flowing along them. Closed and replaced steps stay as faint dashes, and corrections flash on what they moved.
  - A selectivity gauge with the experimental range (benchmarks), counts, and a feed of corrections. A replay slider covers the whole run.
  - Frames whose energies wait for their DFT single points are greyed and tagged "steering pending". The end state is the last steered frame.
  - Running jobs record each kinetics refresh as a frame (`live/network_frames.jsonl`, plus `live/selectivity_frames.jsonl`). Endpoints: `GET /api/jobs/<id>/network-timeline` and `GET /api/examples/<slug>/timeline`. Replays pack with `python -m mepd.web.live_network` (`mepd/web/live_network.py`).

- **Web: the Reactor's Collide is the unbiased network generator's move.**
  - Between molecules, it pushes the two whole molecules into each other, where it used to drag one atom of each to their midpoint. The push is a capped force along one atom of each, spread over each molecule by mass (`langevin.pull_pairs` with `(i, j, A, B)`). With one molecule, or 30 % of the time, two of its atoms are pulled together.
  - The pick is `reactor_generator.collision_pair` itself, run inside the reactor process and timed in simulated fs (300 fs pushes). "keep colliding" repeats it on the generator's own schedule (gaps of ~200 fs).
  - The page shows the two molecules being pushed (blue and orange) and the line the push acts along.
  - New reactor commands `collide` and `auto_collide`. The browser-side picker is gone.

- **Web: theme switch.** Settings has Auto, Light and Dark, next to "All tools".

- **Web: a calculation reads as one sentence.** Each tool's card opens on what it will do, on what, and at which level: "Transition state from chloride + CBr to bromide + chloromethane at gxtb". It also shows how long it usually takes here (the median of earlier runs). Every setting stays one click away under Settings, which marks how many differ from the defaults.

- **Web: Explore › Landscape, the routes that matter as one energy diagram.**
  - Built from the last kinetics run: the starting mixture is a bar, and every step that carries material (at least 1 % of the busiest) is a hump over its TS to the next mixture.
  - Line width shows how much went that way, colour the barrier band, and each product carries its final share. Energies are relative to the start, at the basis the kinetics used.
  - Click a route to open its reaction.
  - The SN2 example now ships with its molecules optimized, so kinetics runs on it.

- **Web: Design's Move tool.**
  - Drag an atom to place it by hand: in the screen plane, or toward and away from you with Alt-drag. Nothing else moves or relaxes, so a TS guess or a contact stays as drawn.
  - New design edit `move` (`mepd/web/design.py`).

- **Web: the inspector has three tabs: Tools · Results · Method.**
  - **Tools:** what you can run on the selection.
  - **Results:** what is known (the detail, the 3D view, its calculations).
  - **Method:** how the selection's numbers were computed: level of theory, profile, wall time, what the calculation does and its pipeline, the settings that differ from the defaults (all of them on demand), caveats, and the command that reproduces it, with links to the calculation, its files, the manual and what to cite.
  - The chosen tab stays as the selection changes.

- **Web: reaction pages and the reaction player.**
  - Double-click a reaction (in Explore or the list), or use "Open: play its path", for its own page at `#/reaction/<edge>`.
  - The page shows the forward and reverse barriers and plays the path: 3D, energy profile and the bonds that form (blue) and break (orange), with their live lengths, all on one scrubber.
  - Space plays and the arrow keys step; there are Start / TS / End jumps.

- **Web: Explore shows what matters.**
  - Edges are coloured by barrier on one scale: under 10 kcal/mol teal, 10–20 amber, above 20 red and thinner.
  - After a kinetics run, a node's area grows with the material that passes through it and an edge's width with its net extent, both capped.
  - Reactions and Kinetics are now the canvas's view tabs, so the Analyze page lost its own switch.

- **Web: Home has three doors.**
  - **Learn:** the step-by-step path, as before.
  - **Start working:** type a SMILES, a name or a reaction and land in Explore; recent sessions; an "All tools" switch.
  - **Examples:** finished studies that open as your own copy. The first is the SN2 study.
  - Examples ship in `mepd/web/showcase/` and are packed with `python -m mepd.web.showcase_tool pack`; `GET /api/examples`, `POST /api/examples/<slug>/open`.
- **Web: lighter Explore.**
  - Structures show one status mark (● checked minimum, ◌ not optimized, ‡ TS, ✕ failed), with the words in the tooltip.
  - The graph toolbar is icons.
  - The legend has three entries.
  - With nothing selected, the right panel shows the workspace's counts and its fastest step.

- **Web: the Workbench shell.** One screen instead of seven top tabs.
  - **Layout:** a thin top bar (mepd → Home, the workspace, Learn, Settings); the structure list and inspector only on the workspace pages (Explore, Reactions, Kinetics, Calculations); a Calculations bar along the bottom (what is running, the level of theory, units).
  - **Drawers:** Settings (profiles, level of theory) and Learn (manual, references, your tools) open over the canvas, so the page underneath stays.
  - **Modes:** Design and the reactor get a mode bar with one way out (Done / Exit).
  - **Addresses:** every page has one (`#/explore`, `#/calc/<id>/live`, `#/design?d=settings`, …), so it can be linked, bookmarked and reloaded. Back and forward work as before.
  - **Esc:** goes up one level: the selection, then a drawer, then the page.
  - **Quests:** they still unlock tools, but no longer lock whole pages or bounce to the first screen.
  - **Removed:** the Playground layout and its font, because the Workbench replaces both layouts.
  - Design tokens (spacing, type, motion, chart colours) start in `static/shell.css`.

### Added

- **Explorer, async: the rotamers of every TS the kinetics use.** Once a step's TS has its IRC (open within the
  cutoff, touching a populated species), its rotatable bonds outside the reacting core (the bonds its IRC ends
  don't share) are turned -- 180 degrees, or 120 and 240 when both atoms are saturated; each bond alone and each
  pair together -- and each guess is a TS search of the queue (`rotamer` jobs, rank 1, in its forked workers).
  A rotamer TS that keeps the core's distances (0.3 A, as `mepd conformers --ts`) is one more TS of the step,
  IRC-checked like any; it gets no rotamers of its own, and a failed IRC drops it without a path search.
  `mepd.discovery.network_expansion.ts_rotamer_guesses`.
  - Why: an adduct's single TS is the conformer its proposal happened to be in. In the F0007 baseline (Cp +
    methyl acrylate, 10 runs) the endo step went through endo s-trans (1.5 kcal/mol over endo s-cis) and the exo
    step through the ester-E conformer (7 kcal/mol over exo s-cis): +5.5 kcal/mol on endo:exo. Turning those
    two TSs' acrylate and ester bonds (6 TS searches, ~3 s each with axtb) gives endo s-cis, exo s-cis and exo
    s-trans, the four conformers of the tracing ceiling; conformer-pair channel sampling of the same steps
    gave endo s-cis once in 16 pairs.

- **Explorer, async: a species first found as an IRC's end is connected from the species the search was asked
  from.** When a fast TS (or a conformer pair's) of pair (i, j) has an IRC ending in a species no proposal made,
  within the cutoff, the pair (i, that species) joins the queue like a proposal's. In the develop baseline 11 of
  the 16 channels found but not connected (F0007 6, isoprene_ma 2, cb_diester 2, cp_mbl 1) were such species:
  adducts that entered only as the end of a TS between two other species, 48-81 kcal/mol up.

- **g-xTB from axtb, in process** (`[gxtb_engine_kwds] backend = "axtb"`, with `axtb_conventions = "gxtb"` to
  reproduce the g-xTB 2.0.1 program). axtb (atomsforhumanity/axtb) is AFH's Rust reimplementation of g-xTB 2.0.1:
  energies and gradients come from this process with no program start per call, and Hessians are analytic.
  - With `"gxtb"` conventions it matches the program to ~1e-12 Eh (energies), ~1e-9 Eh/bohr (gradients) and
    ~0.2 cm-1 (frequencies).
  - On a formic acid explorer run (reactor, async, 4 workers, same seed) it gave the same network and selectivity
    (27.62 against 27.61). Time inside g-xTB fell from 305 s to 13.5 s (median call 34 ms to 1.1 ms), and the run
    from 282 s to 176 s.
  - Needs the `axtb` package (a wheel; not a mepd dependency). It takes only `--etemp/--acc/--iterations` in
    `extra_args` and doesn't run the program's own optimizer (`minimizer = "xtb"`).
  - The Docker image builds axtb from source (github.com/atomsforhumanity/axtb at `AXTB_COMMIT`;
    `--build-arg AXTB_COMMIT=` leaves it out), and `mepd-entrypoint check` compares its g-xTB energy with the
    program's.
  - The default backend is still the program (`"xtb"`). In the web app: Profiles › g-xTB › Backend (with a "?"
    explanation), and a note when axtb isn't installed. A workspace counts both backends as one g-xTB level.
  - A job's kept orbitals (the viewer's Whole, Fragments and Bonds) are axtb's own Molden files, kept as method
    `axtb` with the label g-xTB: a g-xTB session finds them, and each is read with axtb's conventions. Read as
    the program's, formamide's HOMO was off by up to 44% of its peak value; read as axtb's it matches the program's to 2e-8.

- **Molden files with spherical d and f functions in the orbital viewer** (`[5D]`, `[7F]`, `[5D7F]`,
  `[5D10F]`, in any order and case, e.g. PySCF's `[5d][7f][9g]`): `parse_molden` rewrites each orbital
  over the Cartesian functions of the same shells (Molden's order d0, d+1, d-1, d+2, d-2 and f0 ... f-3,
  normalized real solid harmonics), so `evaluate` and the fragment-orbital overlaps are unchanged; it
  used to refuse such files. Spherical g functions are still refused. A new method `axtb` reads the
  files axtb writes (GFN2-xTB, g-xTB v1.1/2.0.1; spherical by default, Cartesian with
  `molden="cartesian"`) with the Molden convention (normalized primitives, each component normalized, as
  TeraChem); `run_molden` runs no axtb single point, the files come from kept runs. A spherical file and
  the Cartesian file of the same calculation give the same orbital values on a grid to 1e-7 relative
  (identical in single precision) for CH3Cl (GFN2-xTB, g-xTB 2.0.1) and CeH4 (g-xTB 2.0.1, f functions);
  the gfn2, gxtb and terachem readings are byte-identical to before.
- **Bond orbitals** (Orbitals › Bonds, beside Whole and Fragments, in every 3D view and in Design): the
  whole system's filled orbitals as its bonds and lone pairs (Knizia's intrinsic bond orbitals, built on
  intrinsic atomic orbitals; xtb's own minimal valence shells, d and f as pure functions, or MINAO as the
  reference, exponent 2), from the same orbitals the Whole view uses, Cartesian or spherical files (axtb's). On a finished path every orbital is followed from frame to frame;
  the few that change are the mechanism's curly arrows, drawn as the path of each one's centroid, with how
  much each has moved frame by frame and the atoms' charge change (Knizia & Klein's electron flow). Kept in
  `<workspace>/orbitals/bonds/`. `POST /api/bond-orbitals`, `POST /api/bond-orbitals/path`,
  `mepd.bond_orbitals`; manual: /manual/orbitals#bond-orbitals.
- **Bond orbitals explained in place**: a colour-keyed "Arrows" legend under the viewer (click an entry to draw
  that orbital), hover tags naming each arrow and how far its pair has moved, and "?" explanations of the bond
  orbitals, the arrows and Δq, each linking to the manual and to the References tab. The manual's new
  "What the arrows mean" section has a labelled diagram (tail, head, dot) and a table of every part. A "?"
  explanation (static/explain.json) can now link its manual section (`manual`: "page#anchor"; checked by
  tests/test_manual.py).
- **Harmonic distance restraints** (`[[restraints]]` in a profile: `atoms = [i, j]`, `distance` in Å,
  `force_constant` in kcal/mol/Å², 500 by default) on any engine; on QM/MM inside the QM/MM energy, so
  microiterative optimizations hold them (QM atoms cannot be frozen). E.g. hold a C-H at 1.10 Å while a
  Michaelis complex minimizes. `mepd optimize` reports each restraint's distance and energy. (#26)
- **Counterions in `mepd qmmm protein-build`** (on by default; `--no-counterions`, `--ion-min-distance`):
  Na+/Cl- neutralize the force-field charge acting on the QM region (QM side chains' own charge left
  out), each ion beside a charged group at least 10 Å from the QM region; the system then carries the QM
  region's charge. The command reports that charge before and after and the system charge. Without them the
  KSI dimer's net -12 acted on the QM region unscreened. (#28)
- **`mepd channels` on QM/MM steps**: conformers of the QM region in its environment, by MD of the QM
  atoms with QM/MM forces (`--qmmm-temperature`, `--qmmm-md-ps`) and QM/MM minimization; pairs searched
  with the environment matched and marched as in `mepd run`; no atom mapping or mirror-image merge. (#27)
- **A check that a TS is the reaction's**: after `run --use-tsopt` on QM/MM, the TS's imaginary mode
  must overlap the path tangent (QM-region Hessian, environment held), and after any IRC its two ends
  must be different species. A TS that fails is reported (`<label>_check_failed.txt`, web results). (#25)
- **A mepd agent skill** packaged as a Claude Code plugin (`plugin/`: SKILL.md, LESSONS.md) for the AFH
  research-skills catalog. (#16)

- **Explorer, async: it stops when the kinetics converge or a budget runs out, not at a depth.** `--rounds` no
  longer caps how far from the seed a species may be expanded: the queue ends when the live kinetics ask for
  nothing more ("converged"), or when `--time-budget` (seconds), `--max-species` or a stop request (the web's
  Stop & analyze) runs out -- then no new job starts, the running ones are cancelled, and the network so far is
  written with its kinetics. The round record's `stopped` says which. Web: "Time budget (h)" in the form.
- **Explorer: IRC paths are kept whole and drawn.** An IRC check writes the full path through the TS
  (minimized ends included) with its energies (`ts/<label>_irc.xyz` + `.energies`; it kept only the two
  ends) and links it to its step, so the web shows each step's IRC, not only its TS. A running async
  expansion writes its network so far (`summary_partial.json`) as it saves, and the web reads it.
- **Web: the reactor generator's MD, live.** Each trajectory is written as it runs into
  `<output>/reactor/run_kk` (segments, piston schedule, events), so the job's Reactor tab plays it.
- **Web: run the async explorer live.** Network expansion has a "Reactor, live" method (the reactor
  generator, access steering, `--schedule async`; cutoff temperature and time, standard state, and a saved
  profile to steer with). Its Live tab shows the job queue (workers busy; running, waiting, done and failed
  jobs of each kind; what each running job is) and the live kinetics (each species' population, which are
  populated and expanded, how many steering single points are back), beside the species and steps that
  already join Explore as they are found.
- **Explorer, async: channel sampling is conformer pairs in the queue.** A step's sampling job only builds
  conformers and atom-mapped conformer pairs (`run_channels` with `conformers_only` returns them); each pair
  is then connected like any pair, a fast TS first and a path search (queue pieces) only if that fails.
- **Explorer, `--schedule async`: no stage waits for another.** IRC checks, channel sampling, the pieces
  channel searches leave at their depth limit and steering single points are jobs of the connector's one
  queue, each queued the moment the live kinetics ask for it; the end-of-cycle stages (and the cycles)
  are gone. The live kinetics that pick what runs next are steered by the single points back so far
  (they were the explorer's own level until a cycle ended). Formic acid at 1500 K, 2 workers: 7 min.
- **Generators tell diastereomers apart, and the reactor generator stops when its products converge.**
  The reactor, nanoreactor and collide generators key products by bond graph plus stereo SMILES
  (endo and exo were one product). The reactor generator runs batches of `n_seeds` trajectories until at
  most `unseen` (5%) of its events gave a product seen only once (Good-Turing), at most `max_batches` (4).
- **Explorer: the `reactor` generator, the interactive Reactor unattended** (`mepd/discovery/reactor_generator.py`).
  Langevin MD on GFN2-xTB in the nanoreactor's wall with the piston swinging it (the Reactor's cycle), while
  Collide every so often (200 fs apart on average, 300 fs each) pushes two molecules picked at random
  together, or pulls two atoms of one molecule toward their moving midpoint (whole molecules: see Changed).
  Nothing chooses the chemistry; events are read with the nanoreactor's detection,
  and the structure after each is a proposal. Within one molecule any pair two or more bonds apart may be
  pulled, H-H included (the Reactor's own rule left formic acid one pair, an H shift). `--generator reactor`;
  formic acid at 1500 K: CO + H2O and CO2 + H2 from 16 trajectories of 5 ps in ~10 s.
- **Explorer: channel sampling for every proposed product.** Each pair a generator proposes from a
  populated species gets `mepd channels` (conformers of both ends, mechanisms), whatever its single TS
  found. Formic acid: the single TS for HCOOH -> CO + H2O had landed on the carbene's step and the direct
  four-centre TS was never searched; its channels find it.
- **Explorer: channel searches ice the pieces of a split** (`path_min_inputs.defer_split_pieces` in MSMEP;
  `--channels-max-depth` now defaults to 0). Each mechanism's path is minimized; where it splits, its pieces
  come back unminimized and each is searched only once the kinetics put material at one of its ends, like
  the connector's own pieces.
- **Explorer: path-search pieces nothing reaches are iced, not dropped** (`mepd/discovery/horizontal.py`). A
  piece below a split waits until the kinetics put material at one of its ends; one still unreached when the
  connect queue runs dry is kept with its place in its pair's search, and every later connect call (each
  round's, and calls with no pairs of their own) queues it again, so a step found later that sends flux its
  way thaws it. Pieces the channel searches leave at their depth limit are kept the same way and asked again
  every round. Both survive `--resume` (`expand_state.pkl`). Rounds record `iced_pieces` and the deferred
  pieces still `iced`.
- **Explorer without rounds** (`discovery expand --schedule async`, with `--steer access` and the fast
  connect queue): a species' proposals are a job in the connect queue's pool, started the moment the live
  kinetics populate it within reach, and the pairs to its products join the same queue as they come in. The
  queue ends when nothing runs and nothing more can be reached; IRC checks and channel sampling then run on
  what it found, and the queue runs again if they open anything (one `rounds` record per such cycle).
  `--rounds` caps how many steps from the seed a species may lie to be expanded. Not yet for `--target`,
  QM/MM systems or a products file. Formic acid (1500 K, collide): the same network and branching as rounds,
  46 s instead of 58 s on 4 workers. The queue takes the caller's own jobs (`register`, `submit`) and new
  pairs while it runs (`add_pairs`).
- **Mixtures without generation barriers** (`discovery mixture --continuous`, the default; `--by-generation`
  restores the old order): each finished seed is merged and its kinetics run at once, and a product they
  populate is paired with each feed molecule right away. Lower generations start first; a queued product
  seed starts only while the latest merge still populates its product, else it is iced and asked again after
  each merge (never run if it stays unpopulated). Seeds run `--schedule async` unless `--target` is given.
- **Explorer: the live gate that decides which waiting pieces run uses free energies** when the kinetics do
  (quasi-RRHO at the explorer's level, not the steering level's single points); it used electronic energies.
  Each TS's thermal correction is computed by the worker job that found it, so the gate doesn't stop the queue
  for a Hessian each time a TS comes in.

- **ORCA engine** (`engine_name = "orca"`, `orca_engine_kwds`; `mepd/engines/orca.py`): minimizations and TS
  searches run in ORCA's own optimizers (OptTS from an analytic Hessian, renewed every `ts_hessian_every`
  steps), Hessians are ORCA's analytic ones, and the IRC is mepd's (Sella) on ORCA gradients. Inputs and
  outputs go through qccodec; ORCA runs by its full path. The executable is `executable`, `$MEPD_ORCA`,
  `/opt/orca/orca` or `orca` on PATH, only with ORCA's own programs beside it (`orca` is also GNOME's screen
  reader). A `[solvation]` table uses ORCA's CPCM. Default level wB97X-D3BJ/def2-TZVP with RIJCOSX. ORCA is
  licensed, so it is not in the public image.

- **The explorer's kinetics count mirror images** (`discovery expand` and `refine-energies`,
  `--mirror-images/--no-mirror-images`, on by default; `mepd/discovery/mirror.py`). A chiral transition state
  has a twin of the same energy that the explorer finds by chance or not at all. Now every chiral TS whose
  twin is not in the network gets one, between the mirror images of its ends: a second channel when both ends
  are achiral, else a route to the enantiomer, which joins the kinetics as a species if it was not found. A
  chiral conformer of an achiral molecule gets -RT ln 2. On Cp + methyl acrylate (F0007) the endo:exo ratio
  stays 75:25 with DFT, but the enantiomers come out racemic (they were 88:12) and conversion goes from
  49% to 74%. `summary.json` and `refined_energies.json` list what was added (`mirror_images`).

- **A cut-off exploration resumes** (`discovery expand --resume`): the whole exploration is saved in
  `expand_state.pkl` at every stage of a round, and as each pair finishes while connecting; a run stopped by a
  time limit or a crash carries on from there, and only the path searches of the pairs it had not finished run
  again. Each step's finished channel sampling is kept too. `--rounds` can be raised to grow a finished network.

- **The explorer steers on a higher level's single points** (`discovery expand --steer-energies PROFILE`;
  by default the `--refine-energies` profile): the kinetics that decide which species are expanded, and which
  steps get channel sampling, use G = G_explorer + (E_high - E_low) on every species and TS they treat. Each
  geometry is computed once and kept in `steer_single_points/`. The barrier cutoff's reach stays at the
  explorer's level. An ML potential on the GPU (`engine_name = "fairchem"`, e.g. UMA with `device = "cuda"`)
  steers from one process of its own, which loads the model once, so the explorer's other workers still fork
  (CUDA does not survive a fork). On the Suzuki oxidative-addition pilot (89 atoms), UMA steering plus DFT
  refinement gives the same major product as DFT steering, with no DFT during the exploration.

- **Higher-level energies for an explored network** (`mepd discovery refine-energies RUN_DIR`, `discovery
  expand --refine-energies PROFILE [--refine-energies-mode single-point|reoptimize]`): the species and open
  steps the explorer's free-energy kinetics use get single points on the explorer's geometries, or are
  re-minimized / have their TS re-found with an IRC check; the kinetics are rerun at both levels
  (`refined_energies.json`, geometries and RMSDs under `refined/`). Every result is kept as it finishes and a
  rerun picks up from what was kept at the same level of theory.
- **`mepd qmmm reactor`**: the nanoreactor with QM/MM forces over several trajectories, one process per seed
  (`--seeds`, `--workers`). The QM region runs hot inside its protein or solvent, which stays near room
  temperature. Products are whole systems, pooled over seeds (one per distinct QM bond graph), for
  `mepd discovery expand --products`. The MD runs the QM level at `--etemp` (3000 K, as the vacuum
  nanoreactor). Network expansion's embedded nanoreactor takes `n_seeds` and `electronic_temperature_k`.
- **Protein protonation states by hand**: `mepd qmmm protein-sites --protonation "A:ASP103=ASH"` (ASH, GLH,
  LYN, HID, HIE, HIP, CYX) instead of the pH, e.g. a buried catalytic acid.
- **Every docked orientation at a site**: sites.json lists each site's distinct poses (`poses`: more than 2 Å
  RMSD apart), and `mepd qmmm protein-build --pose k` builds one. A molecule that looks alike from both ends
  docks either way round in one pocket; before, only the best Vina score was kept, which for
  androstenedione in ketosteroid isomerase (1OH0) was the pose turned end to end from the crystal's.
- **`mepd channels --max-path-searches N`**: a cap on path searches (each pair's first search and every piece a
  split adds, counted in pair order) instead of a wall-clock budget. Past it, no pair starts and no search is
  split further, as with `--search-budget` (pieces already queued still run once), but the same inputs stop at
  the same search on any machine (with one pair at a time), so a search is reproducible for a given seed: two
  runs of a Transition1x-TMC search capped at 8 wrote byte-identical paths (11 searches each). Leaves cut by it have status
  `path_search_budget`; `stats.json` records `path_searches` and `max_path_searches_reached`.
- **Transition-state conformers: `mepd conformers ts.xyz --ts [--freeze 0-12,3-11]`, and Conformers on a TS in
  the web app.** CREST samples the TS with its reacting core held at the TS distances (xtb `$constrain`,
  `CrestInputs.constrain_pairs`, force constant 1), and each kept conformer is TS-optimized with the frequency
  check, so the lowest TS of a reaction can be found. The core defaults to the pairs the imaginary mode
  stretches most (`mepd.conformers.ts_reactive_core`). CREST's topology checks are off for this (partial bonds
  count as topology), so mepd drops a conformer whose held distances moved or whose other bonds changed. CREST
  now samples in the profile's ALPB/GBSA solvent. In the web app the sampler defaults to Automatic: RDKit for a
  minimum, CREST for a transition state. Manual page: /manual/ts-conformers. (#15)
- **Comparing species without metal-ligand contacts** (`metal_ligand_contacts=False` on `compare_structures`,
  `same_molecule` and `connects`). Bonds to transition metals, lanthanides and actinides are left out of the
  bond graphs and the stereochemistry, so a change of coordination alone (a ligand binding through another
  atom, or coming off the metal) is not another species; `Comparison.metal_ligand_changed` records it. Off by
  default: every command still compares all bonds. Following the treatment of dative metal-ligand contacts as
  zero-order bonds, kept apart from covalent bond changes, in Z. Li, H.-H. Hsu, B. M. Savoie, "Finding the
  Roads Less Calculated in Organometallics with Automated Transition State Searches", ChemRxiv (2026),
  doi:10.26434/chemrxiv.15006374/v1 (preprint). In the Transition1x-TMC screen, a quarter of the reaction
  ends whose bond graphs differed from the dataset's differed only in metal-ligand contacts.
- **`mepd run --recursive/--parallel --max-depth N --search-budget SECONDS`.** Caps for a recursive MSMEP that
  keeps splitting (59 nanoreactor pairs: 14 still recursing after 4 h, median 14 min). Past the depth, a path
  is converged and kept as one step; past the budget, no search is split further. Either way `--use-tsopt` and
  `--irc` run on what was found. These are the profile's `recursive_split_max_depth` and `mepd channels`'s
  `--search-budget`, now on `mepd run` too. (#13)


- **xtb (GFN2/GFN1) as a QM level retries a failed SCC** with `--etemp 1000`, more iterations, then
  `--etemp 5000` (as the solvation correction does), and says so in the log; `etemp` sets a fixed
  electronic temperature instead.

- **`mepd.minimize`, `mepd.frequencies`, `mepd.compare_structures`** (public API): a frequency-checked minimum
  that escapes saddles along their imaginary mode; harmonic frequencies and modes; a structure comparison that
  says why (`Comparison.reason`) and whether stereochemistry could be decided. `mepd.WrongStationaryPointError`.

- **Fragment orbitals** (`mepd/fragment_orbitals.py`, `POST /api/fragment-orbitals`, Orbitals › Fragments).
  The geometry is cut into its molecules (bonding of a path's first frame, so a TS keeps its reactants;
  charges and spins from Lewis structures). Each molecule gets its own single point where it sits. For every
  donor → acceptor pair, the bar shows the overlap |⟨occupied|empty⟩| and the gap of their frontier levels,
  the direction with the smaller HOMO→LUMO gap first (within it, the levels with the largest S²/ΔE), and draws both orbitals together (donor dark, acceptor pale, in phase). Overlaps
  are analytic (Obara–Saika over the molden basis; every orbital checks orthonormal to 1e-10 for g-xTB,
  GFN2-xTB and TeraChem). Degenerate levels count as one level, so hydroxide attacking CH₃Cl gives from
  its σ lone pair (HOMO−2), not its π pair. A path shows |S| frame by frame
  (`/api/fragment-orbitals/profile`). The tooltip gives the whole system's HOMO / LUMO as fragment
  orbitals. Suprafacial [2+2] HOMO→LUMO overlap comes out exactly 0, and Diels–Alder's does not.
  Paths behave like the whole system's orbitals. Each result is kept on disk
  (`<workspace>/orbitals/fragments/`, newest 500 MB, `mepd/web/fragment_store.py`), and the profile request
  builds every frame. The viewer loads frames ahead in playing order, with the same orbitals chosen at every
  frame. A frame's surfaces stay until the next frame's arrive, and the phase is carried from frame to frame.
  On a 65-frame IRC (GFN2), every frame was ready 4.5 s after opening, and playback always drew each frame's
  own surfaces. The bar reads "Electrons from [molecule, filled level] into [molecule, empty level] ·
  overlap · gap"; molecules go by formula and charge (Br⁻), degenerate levels say "3-fold", and any number
  of molecules is offered.
  Radicals' singly occupied levels (SOMOs) are offered on both sides, so two radicals' bond shows as
  "Pairs X SOMO with Y SOMO". In ethanol → C₂H₅O• + H• the SOMO–SOMO overlap is 0.43 → 0.17 → 0.04 along
  the IRC; before, only filled → empty pairs (≤ 0.016 at the IRC midpoint) were offered.
  The bar says which geometry the molecules were cut from ("cut at path start", "cut at path end", "cut
  here"), and warns "⚠ bonded at this frame" where the frame shown does not split into the same molecules:
  there the orbitals are of pieces with broken bonds, each computed alone, not of molecules that exist at
  that frame. When both ends of a path hold more than one molecule (A + B → C + D), the note is a menu:
  "cut at path end" follows the products' molecules instead of the reactants'. With three or more
  molecules, **group molecules** assigns each a letter: molecules with the same letter are computed together
  as one fragment (ammonia | a water dimer), with any two groups compared in the menus. In Design the
  molecules are as drawn (its bonds and formal charges), so a placed ion is its own fragment however close it
  sits: Mg²⁺, Li⁺ or Na⁺ next to water were merged with it ("H₂MgO") by the distance rule and never got orbitals
  of their own. A bare cation (no electrons left) is an acceptor only, and the default levels are now the
  frontier-most pair that overlaps at least half as well as the best one (water's σ lone pair → Mg²⁺ 3s, not its
  3p set).
- **Orbitals come from the calculation itself.** Every SCF a calculation runs with g-xTB / GFN2-xTB or
  TeraChem keeps its orbitals (molden) in the job's `orbitals/` folder (the newest 400; `--molden` costs
  g-xTB ~2 ms a call). The viewer reads the orbitals of the geometry it shows from there, so a TS
  optimization's frames show their orbitals without a new single point ("· this run" after the method);
  only a geometry no calculation has computed gets one. A running calculation's live view follows its
  frames by default, on a coarser grid (~0.1 s a frame to build; 0.12-0.2 s per frame in the browser
  while a search ran), and the next frames are built in the background. The grid evaluation is ~2x faster.
- **Orbitals from TeraChem.** In a session whose level of theory is TeraChem, the HOMO / LUMO come
  from a TeraChem single point with the profile's method, basis and keywords (TeraChem's own molden
  file), not from xtb. The bar now names where the orbitals come from (e.g. "TeraChem b3lyp/6-31gs",
  "g-xTB"). TeraChem's molden (no `Sym=` lines, normalized primitives) is read with its own conventions;
  every orbital checked integrates to 1. Not in the demo.
- **Frontier orbitals (HOMO / LUMO).** An "Orbitals" button under the 3D view of a structure (Explore's
  inspector, a reaction's ends and TS), a calculation's paths and live optimizations, and in Design draws
  the HOMO or LUMO as a wireframe isosurface (+ lobe blue, − lobe red) enclosing 85 % of its density
  (any share in "% density"; remembered per browser), with both energies, the gap, and the atoms the shown
  orbital sits on. On a path, "follow" recomputes them for each frame shown, e.g. a TS optimization as it
  runs; Design recomputes after each edit. They come from a g-xTB single point when the session's level is
  g-xTB, else GFN2-xTB (`xtb --molden`), evaluated on a grid in `mepd.orbitals` (every orbital checked to
  integrate to 1; the two programs scale Cartesian d functions differently). `POST /api/orbitals`.
- **Interactive reactor: a piston.** A Piston switch squeezes the wall in and out again and again, as the
  nanoreactor does (by default every 1 ps: open 75 % of the cycle, then in to 70 % of the Wall radius over
  100 fs). "cycle" opens the period, how far in and how long open; the Wall slider stays the open radius.
  The `piston` sandbox command takes `value`, `period_ps`, `compress` and `duty`.
- **Interactive reactor: Collide.** One button pulls two molecules into each other for a moment (what dragging an
  atom does by hand), and "keep going" repeats it every few seconds. It is on by default until you have caught a
  first reaction: left alone, the reactor rarely made any (glycine, 3 Å wall, 2000 K: none in 6.5 ps). With it,
  formaldehyde + water made 3 reactions in 20 s.
- **Web UI: finding what was just added.** The sidebar has a Structures | Reactions switch; Reactions lists
  every Explore arrow, newest first, with both ends drawn and its barrier. Structures and reactions a
  calculation added since you last looked carry a blue dot in the lists and a soft glow in the graph until
  selected; "N new" shows only those, and "Mark seen" clears them (per workspace, per browser). Clicking a
  structure or a reaction in the list centres and zooms the graph on it and pulses it (it used to need a
  double-click). A finished calculation's toast has "Show in Explore".

- **A public Python API** ([docs/python-api.md](docs/python-api.md)), so workflows are scripts on `mepd`
  instead of on private helpers: `mepd.load`, `read_frames`, `from_smiles`, `write_xyz`; `mepd.engine(name,
  **options)`, `engine_from_profile`, `set_threads`; `mepd.energies`, `optimize`, `find_ts`, `irc`,
  `interpolate`, `barrier_kcal`; `mepd.same_molecule` (mepd's one "same species" test) and `mepd.connects`
  (which sides of an IRC match, in either direction); `mepd.units`; errors `mepd.UnsupportedOperation` and
  `mepd.ConvergenceError`. They take structures and an engine, return results, raise instead of returning
  None, and neither print nor write files. The command line now calls them for loading structures and for
  TS optimization + IRC.
- **Channel search from Python**: `mepd.channels(start, end, output=..., profile=..., **options)` runs the
  search `mepd channels` runs and returns a `ChannelsResult` (direct, multistep and off-target channels, each
  with its TS/IRC steps, plus the conformer pools searched); `mepd.read_channels(folder)` reads an earlier
  run's folder the same way. The search itself moved from the command into `mepd/channel_search.py`
  (`ChannelsOptions` holds its settings, same names and defaults as the options); the command is now a thin
  wrapper, with unchanged options, output and folders.
- **The channel search and the shared run steps log instead of printing** (loggers `mepd.channel_search`,
  `mepd.cli_common`; problems at warning level), so `mepd.channels` from Python is quiet unless logging is
  set up. The commands print the same lines as before. Path searches' step-by-step progress still prints.

- **A guided first run in the web UI** (`Welcome.js`, `quests.js`). An empty session opens on a first screen:
  choose a ready-made molecule (glycine, ethanol, formaldehyde, HCN) or build one in Design, put it in the
  interactive reactor, catch a reaction. A step bar (Molecule › Reactor › Reaction › Barrier › Explore) follows
  you. Stop & analyze opens a plain-words **What happened** page (`ReactorStory.js`): one card per reaction,
  its barrier search, and forward and reverse barriers as separate arrows. The tabs and tools then unlock as
  they are used, shown as icon tiles under **Your tools** (the mepd logo opens it). **I know mepd: show
  everything** (or Settings › Show everything) unlocks every tab and tool at once.
- **Edit in Design from an edge**: a selected edge (or reaction arrow) opens in Design as a reaction, its start
  as the reactant and its end as the product, in the conformers saved on the edge, atoms matched as for an xyz
  reaction (`POST /api/design/load-edge`).
- **Products: a new reaction by clicking** (`NewReaction.js`). On a selected molecule (or in Analyze ›
  New reaction), what it can become appears at once as product tiles (bond rules, simplest first); click the
  ones to try and add them as reactions, each with its Find TS. Add a partner molecule, pick products from the
  graph, or draw one in Design. Products already in the graph, or already a reaction, are marked.
- **"?" on every implementation choice**: click it next to an operation, a form field, a Settings field or a
  Kinetics option for what it changes, how mepd does it and with which program, the default, what could stand
  in (and where to switch it, or where it would plug in), caveats, the code and command behind it, and a link
  to what to cite (`static/explain.json`, `Why.js`). Nothing shows until it is clicked.
- References: the quasi-RRHO and single-point-Hessian citations behind Kinetics › free energies.
- References: the recursive autosplitting paper (J. D. Estrada Pabón, M. Stöhr, D. Hait, T. J. Martínez,
  ChemRxiv 2026) in References › Path search, linked from the splitting options' "?".

- **Run provenance**: every `mepd` command given `--output` (so every web job) writes `provenance.json` there,
  or `<file>.provenance.json` for a file output. It records the command line, the profile text and level key,
  mepd's version and git commit/dirty state, external binaries with sha256 (xtb, gxtb-xtb, crest, gsm and any
  the profile names), library versions, host, timing and exit code. `MEPD_NO_PROVENANCE=1` turns it off.
  Study scripts can call `mepd.provenance.run_manifest` directly. This is what makes results publishable as
  findings with captured, rather than reconstructed, provenance.
- **Stop & analyze** for Find TS, Reaction channels and Sample more paths: next to Cancel on a running search.
  Each running path search (NEB, FSM, FSM-TS, GSM, MLP-GI) ends at its next step with the path it has, no new
  search starts and none is split further; then TS optimization, IRC and channel classification run on those
  paths and the run finishes as done, marked as stopped early. Cancel still discards the run.
  From a shell: run with `MEPD_STOP_FILE=<file>` and create that file (`mepd/search_control.py`).
  Also for the **Nanoreactor** and its Run longer: the MD ends where it is (xtb is stopped mid-segment, its
  frames kept), then events, species and reactions come from the trajectory so far; the result says where it
  stopped, and Run longer continues from there.
- **Export a whole workspace** (Analyze › Reactions › Export all; `GET /api/workspace/export`). Covers every
  reaction Analyze lists, with every channel found for each. Two files:
  - `report (.html)`: one self-contained page. An index of the reactions (lowest barrier, ΔE, channels, level;
    filter and sort) opens each reaction's own page: energy diagram, every channel's TS and IRC in 3D.
    One copy of 3Dmol.js serves all pages; it opens offline and has the CSV in it.
  - `data (.csv)`: one row per elementary step of every channel, and one row for a reaction with none. Fixed
    columns with units in their names (`workspace_export.CSV_COLUMNS`): SMILES, charge and multiplicity,
    channel and step barriers, energies in Hartree, level of theory, and the TS and both step ends as an
    element list and flat Angstrom coordinates in one atom order.
- **Network variants** (`mepd/network_variants.py`, `mepd/web/variants.py`, `mepd variant`). Edit a species in
  Design (hydrogens for groups, element or charge swaps), then **Apply to network**. The edit is carried
  through every reaction the species and its products take part in, then each one is recomputed. The result
  is a variant of the network, kept next to the parent in the same workspace.
  - Atoms are matched by graph isomorphism of each species onto each reaction's end geometries, all
    automorphisms included. Placements that differ in the reaction (regio- and diastereomers) become separate
    reactions; mirror images count as one. A wrong stereoisomer of the edited species is never produced.
  - A reaction's frames come from the TS kinetics uses (that job's IRC through it). Otherwise they come from
    the edge's pinned ends; SLAPMapper renumbers an end whose order does not fit the start.
  - A step in which an edited hydrogen takes part is `blocked`. Reactions without an edited species are shared.
  - Of equivalent placements on a molecule that does not react (which methyl of NMe4+ gets an OH), the one
    nearest the reacting atoms is used. Of two parent records of one step, the variant starts from the one
    with a TS (the lowest).
  - Edits stack: Apply to network from a species opened while Explore shows a variant builds a variant on top
    of it (e.g. choline for NMe4+, then iodide for bromide), carried through that variant's re-optimized
    reactions and compared with it.
  - Edits are made on one molecule. A design opened from a complex offers buttons to open each of its
    molecules instead.
  - Each variant reaction gets a fast estimate first: only the new atoms are relaxed, then single points, and
    the barrier shift is applied to the parent's TS. It is then re-optimized: both ends are minimized, the TS
    is optimized from the substituted one, and its IRC must connect the substituted ends. If that fails, the
    parent's own path search runs on the variant's ends.
  - Kinetics has a network picker (parent or a variant) and **compare with parent**: the same mixture, with
    end amounts and per-step barriers side by side. Explore has a network picker; the records a variant
    replaces are faded.
- **Container image of the full web app** (`Dockerfile`, `.dockerignore`,
  `deploy/docker/`). `docker build -t mepd .` builds one image with:
  - mepd and every `web` extra, from `uv.lock`;
  - g-xTB 2.0.1 (release build, SHA-256 checked);
  - xtb 6.7.1, CREST 3.0.2, Packmol 20.15.3 and Psi4 1.11 from conda-forge, via micromamba;
  - molecularGSM built from source with its ASE backend, against MKL;
  - the retrosynthesis data and an AiZynthFinder 4.4.1 environment;
  - AIMNet2 and ANI-2x weights, and ReactionT5's models in a layer of their own that doesn't depend on `uv.lock`;
  - FAIR-Chem UMA `uma-s-1p2p1` (uma-s-1.2.1), in a layer of its own that doesn't depend on `uv.lock`: it is
    downloaded with `huggingface_hub` alone, so a dependency change neither downloads nor pushes it again.

  UMA is gated on Hugging Face. A build needs a token from an approved account, as a build secret
  (`--secret id=hf_token,env=HF_TOKEN`); it is never stored in the image. A build without one fails with
  instructions, unless it opts out with `--build-arg FAIRCHEM_MODELS=`. The image check loads each baked
  model with Hugging Face offline. The image includes a C compiler because Triton, which runs UMA on the
  GPU, compiles a helper on first use.

  The base image is Ubuntu 26.04. Every program and data set is checked when the image is built
  (`mepd-entrypoint check`), as the user the app runs as. See `deploy/docker/README.md`.
- **Image entrypoint** (`deploy/docker/entrypoint.sh`): `serve` runs the web app in `demo` mode (shared
  password, a private workspace per visitor) or `full` mode (`MEPD_MODE`). `check` verifies the image. Any
  other argument runs as a command.
- **Data mirror for the retrosynthesis files** at `https://public.atomsforhumanity.org/mepd/retro/`:
  - `deploy/docker/retro-data.txt` pins each file's SHA-256 and upstream URL. The checksums were verified
    against Zenodo's and figshare's published MD5s.
  - `deploy/docker/fetch_data.py` fetches from the mirror, falls back to Zenodo/figshare, and accepts only
    matching checksums. It also lays out the mirror directory. Builds no longer wait 30–45 minutes on Zenodo.
- **Pinned AiZynthFinder environment** (`deploy/docker/aizynth-constraints.txt`). It is taken from a working
  install, because an unpinned one breaks as its dependencies move (see Fixed).
- **Prebuilt retrosynthesis stock indexes in the image.** These are the 80 MB sorted ZINC InChIKey array and
  the PaRoutes InChIKeys, which mepd would otherwise build on first use after every container restart. The
  ZINC stock now loads in 0.1 s.
- **Kubernetes manifest for the beta** (`deploy/k8s/mepd-demo.yaml`, `deploy/k8s/README.md`). It runs demo
  mode on afh-c04 as a single replica. A new version rolls out without downtime: the old pod serves until
  the new one is ready. The pod reserves little (2 cores, 8 GB, and the GPU seen through
  `NVIDIA_VISIBLE_DEVICES=all` rather than requested), yet can use the whole node. State and scratch
  live on the node's data disk (`/nvme1/mepd`), and an Ingress serves `mepd.atomsforhumanity.org`. The pod
  sets two things the first deploy needed:
  - `runtimeClassName: nvidia`: without it the GPU was assigned but never reached the container.
  - `enableServiceLinks: false`: Kubernetes' `MEPD_PORT` for the Service `mepd` overrode the server's
    port, and the server would not start.
- **Demo limits are configurable from the environment.** Any number in `DemoPolicy` (`mepd/web/demo.py`) can
  be overridden as `MEPD_DEMO_<FIELD>`, e.g. `MEPD_DEMO_GLOBAL_CONCURRENCY=64` to run 64 jobs at once across
  all testers instead of 4. Invalid values stop `mepd web --demo` at start-up with a message.
- **Continuous deployment** (`.github/workflows/build-deploy.yml`, set up in `deploy/ci/README.md`).
  - **On a push to `main`:** build `ghcr.io/atomsforhumanity/mepd:main-<run>-<sha>`, test it inside the
    image (a failed test is rerun once, alone), push it, deploy it to the beta cluster, and check the public
    site. It rolls back automatically if the new version doesn't become ready.
  - **On a tag `v1.2.3`:** publish image `1.2.3`. "Run workflow" with a tag redeploys or rolls back to it.
  - It runs on a self-hosted runner on afh-services (a warm build cache, room for the image, the cluster on
    the LAN), and deploys with k3s's `kubectl` and the cluster's kubeconfig on that node.
  - `deploy/k8s` is now applied with kustomize (`kustomization.yaml` carries the image tag, which CI sets).
- **`uv.lock` is committed.** The build uses `uv sync --locked`, so a `pyproject.toml` change without an
  updated lock fails instead of building untested versions.
- **This changelog.**
- **FSM-TS path method** (`mepd/pathminimizers/fsm_ts.py`). The freezing string method grows the path,
  its highest image is optimized to a TS, and a pseudo-IRC (one Hessian, a push both ways along the imaginary
  mode, two minimizations) decides whether the pair is one elementary step. If it is, the path ends as
  [reactant, TS, product], so `--use-tsopt`, channels and the web start from that TS; a new species splits the
  pair there. A TS that isn't one, doesn't converge, or doesn't connect is discarded and the pair is searched
  again with NEB.

### Changed

- **Collide prefers no molecule when it picks two to collide,** in both the reactor generator
  (`reactor_generator.collision_pair`) and the interactive Reactor's automatic Collide (`Sandbox.js collisionPair`).
  It used to avoid pairing two waters: water was never the first partner when there was another molecule, and a
  non-water second partner was preferred 60% of the time. That was a rule about one species, so it's gone (user,
  2026-11-05). Any two molecules are now equally likely. Generator trajectories of seeds with several molecules
  change, since one random draw fewer is used per pick.

- **Collide and the reactor push whole molecules into each other; the [4+2] double pulls are gone.** Between two
  molecules a pull no longer drags one atom of A onto one atom of B: the capped force along the line between two
  atoms, one of each (collide: every such pair, H included; the reactor: a random heavy atom of each), acts on each
  whole molecule, spread over its atoms by mass (`langevin.pull_pairs` with `(i, j, A, B)`). The two atoms only set
  the direction: until they are within 2 x cap / `k` (1.9 A at the default `k`) the force is the cap (95 kcal/mol/A),
  a constant push, and which atoms meet is up to the orientation and the dynamics. Within one molecule the atom pull
  stays (formic acid without pulls: no reaction in 48 trajectories of 5 ps at 2200 K).
  `collide_generator.cycloaddition_moves`, a reaction-class template, is removed. `LangevinMD.step` adds the bias
  outside its SCF retry, so a pull that raises is not retried as an SCF failure (a test checks that pushes between
  molecules lose no step of a reactor trajectory). Measured with 300 fs pulls unless said; cyclopentadiene + methyl
  acrylate (F0007), GFN2-xTB:
  - one collision at 1000 K: endo and exo adducts each in 3% of collisions (8% and 6% with 600 fs pulls). Pushing
    the two centres of mass together instead sends every encounter at the acrylate's ester: endo 8, exo 1 in 288
    collisions, the hetero-Diels-Alder on C=O in a quarter.
  - collide generator (96 collisions, 1000 K): both adducts proposed in 4 of 10 runs (9 of 10 with `pull_fs=600`).
  - reactor (5 ps trajectories): the Diels-Alder adduct is now the seed's most frequent first reaction; endo in 21%
    and exo in 6% of trajectories at 1000 K, 14% and 3% at 1500 K, 13% and 0% at 2200 K (before: 2 adducts in 72
    trajectories of 10 ps at 1200-2200 K, both exo).
  - Formic acid: pulls within one molecule are unchanged, but once a molecule has come apart (a reaction's products,
    or a hot H briefly beyond bonding distance) its pieces are pushed together as wholes too, so a monomer's
    trajectory matches the old one only until then (seed 7 at 2200 K: up to 0.62 ps). Monomer, CO + H2O: in 24% of
    trajectories at 2200 K (before 19%), 3% at 1500 K (before 9%; most of it came from pulling the H2 of CO2 + H2
    back onto CO2). (HCOOH)4 at 2200 K: 3.1 events per 5 ps instead of 4.6; CO + H2O proposed from 16 trajectories in
    84% of resamplings (before 100%), from 32 in 98%; the seed's reactions now nearly converge (10% seen once, before
    ~30%). Cost per trajectory unchanged.
- **Explorer, async: the queue refines the steps the flux runs through first.** Channel sampling of a step and
  the TS searches of its conformer pairs wait in order of the height above the seed of the step's lowest TS so
  far -- with none yet (its single TS failed or belongs to another step), of the higher of its two species, which
  no TS between them can lie below -- asked each time the queue orders its jobs
  (`HorizontalConnector.submit/add_channel(..., priority=)`, a number or a function); proposals of species after
  the seed wait behind them (rank 3, with path-search pieces). In the F0007 baseline the seed's 35-42 channel jobs were queued with its proposals, before any of
  their TSs existed, and ran in species order: the endo and exo ones waited 4-12 minutes; and in 9 of 10 runs
  the reactor for the most populated product then took all 32 workers for the budget's last 6-13 minutes,
  while the TS searches of 213-396 conformer pairs never ran.

- **Reactor generator: products the access cutoff can't reach don't hold up its convergence.** With `--steer access`,
  the explorer passes the reactor what is left of the access cutoff above the species it expands; a product whose
  quenched GFN2-xTB energy is further above that species (a step to it can't be lower than its reaction energy) is
  still proposed but not counted by Good-Turing. No new option. (HCOOH)4 at 2200 K, 64 trajectories of 5 ps,
  cutoff 87.9 kcal/mol: 8 of 70 reactions of the seed fall above it, and products seen once among those counted
  drop from 29% to 19% -- still above 5%, as the low-lying tail is real (ring acetals, glyoxylic acid, ...). The
  monomer has none above it: unchanged (converges after a median 28 trajectories).

- **The reactor generator runs at 2200 K by default** (was 800 K; `reactor_generator.OPTIONS`, the web form's
  "MD temperature"). Of the temperatures measured, 2200 K with 5 ps trajectories was the fastest to propose both
  HCOOH channels (CO + H2O and CO2 + H2) reliably, from the monomer and the (HCOOH)4 cluster (98-100% of runs).
  At 1800 K CO + H2O was missing from 12-16% of runs.

- **mepd.afh.org allows 300 atoms per structure** (`MEPD_DEMO_MAX_ATOMS`, was 150; also the QM region's
  cap). To be lowered again if larger jobs crowd the node.
- **`mepd qmmm protein-build` writes an FSM-TS profile in a caged environment** (`path_min_method = "FSM-TS"`,
  `qmmm_environment = "cage"`, NEB fallback), not NEB with the shell relaxed. In a relaxed shell the TS
  search can end on an environment saddle (ketosteroid isomerase: a Tyr57 OH rotation, both IRC ends the
  same species). Caged FSM-TS gave all three KSI steps' TSs with connecting IRCs, each in under 25 min.
  Solvent profiles keep NEB.
- **IRCs stop at 0.01 eV/Å by default, up to 3000 steps per direction** (Sella; was 0.1 and 1000), and the
  geomeTRIC IRC at the same largest force (its own default is about 0.023 eV/Å). At 0.1 an IRC can stop on a
  shoulder well above the minimum, and minimizing from there can slide into another isomer: in a
  Transition1x-TMC reaction (TMC_20796_Pt), the proton of a Pt-bound HCl moved to the chloride trans to the
  alkyl chloride, 13.9 kcal/mol lower, which the bond-graph comparison could not see; at 0.01 the IRC itself
  reaches that minimum. Pass `fmax` (and `steps`) in the IRC keywords for the old behaviour.
- **The demo moves to mepd.afh.org.** The Ingress serves only `mepd.afh.org`; Cloudflare redirects
  `mepd.atomsforhumanity.org` there. The deploy now checks `https://mepd.afh.org/login`.
  The retrosynthesis data mirror is now `https://public.afh.org/mepd/retro/` (the image's `DATA_MIRROR`).

- **Deleting molecules says which complexes go with them.** Explore draws a complex as its molecules, so a
  complex is removed with any of them (as before). The delete confirmation (Explore selection and the
  Inspector) now names those complexes. (#14)

- **`thermo.qrrho(ts=True)` says when it could not leave out the reaction coordinate.** A Hessian at another
  level than the TS (GFN2-xTB at a g-xTB TS) often has no negative mode; the soft mode then stayed in silently
  and G(TS) came out too low. The result now has `ts_mode_dropped` and a `warning`, `ts_mode=` names the mode
  to leave out, and Kinetics' free energies list such TSs in their warnings. (#12)
- **One Langevin integrator for the Reactor, collide and the reactor generator** (`mepd/discovery/langevin.py`).
  The interactive Reactor's MD loop, each collision of the collide generator and the reactor generator's
  trajectories had their own copy of the GFN2-xTB (tblite) BAOAB loop, wall and capped pulls; they now share
  `LangevinMD`. A collision whose SCF fails is stepped back and continued as in the Reactor, instead of
  failing.
- **Explorer**: the connect queue (single TS optimizations, path-search pieces, leaf TSs) runs its jobs in
  worker processes, not threads of one process, which held it to about five busy cores whatever `--workers`
  was. Species proposed at once (collide, nanoreactor) share `--workers` instead of each taking all of them.

- **Explorer**: a path piece whose TS search fails, or whose TS belongs to another motion by its IRC, keeps its
  path's highest image as an upper-bound barrier (`ts_kind = "path_peak"`), so a barrierless association no
  longer cuts the seed off the network; such pairs get channel sampling, and the explorer's channel searches
  sample every conformer pair of every mechanism by default.
- **Psi4**: a `[solvation]` table uses Psi4's own PCM instead of xTB solvation on top; Hessians are central
  differences of gradients, kept per geometry.

- **Transition states, minima and IRCs are checked, not assumed** (KAIST 500 benchmark on g-xTB, 2026-10-19).
  Defaults everywhere mepd optimizes, including `channels`, `ts`, `run --use-tsopt`, the force, solvent and
  substituent commands and the web app:
  - Optimizations converge to a largest force of **0.001 eV/Å** (was 0.01). At 0.01, 14 of 500 g-xTB TSs had a
    spurious second imaginary mode, and 13 sat 1–34 kcal/mol above the saddle they belong to.
  - g-xTB TS searches start Sella from **g-xTB's own Hessian** instead of a gradient-based estimate (the estimate
    reached a different, 3–6 kcal/mol higher saddle in 4 of 500; exact: 500/500 first-order, 4.5 s median).
  - A converged TS must have **exactly one imaginary mode** (below −10 cm⁻¹, `mepd.IMAGINARY_CUTOFF_CM`), else
    `WrongStationaryPointError`. An optimization that runs out of steps **raises** (it used to return its last
    frame); `find_ts` first goes on from where it stopped (`continue_unconverged`, default 1).
  - **IRC ends are minimized** (`mepd.irc(..., minimize_ends=True)`): IRC ends sat ~1.9 kcal/mol (median) above
    their minima, and ~6% became another species when minimized. The path now runs minimum → TS → minimum.
  - g-xTB **minimizations** use Sella with the exact Hessian to 0.001 eV/Å (`GXTBCalculator.minimizer`;
    100% converged and 99.4% true minima on 1000 IRC ends; xtb's own optimizer, `minimizer="xtb"`, stops at
    ~0.05 eV/Å). Conformer minimizations are slower (seconds each, not under a second).
  - **Stereochemistry that cannot be compared is not a match**: `same_molecule`/`connects` used to pass when the
    stereo SMILES failed. RDKit decides; when it cannot assign bond orders, OpenBabel's stereo SMILES can confirm
    "same" (30 of 33 such cases); otherwise the side is reported as undetermined (`Match.undetermined`).
  - QM/MM (frozen atoms) TS searches and IRCs are unchanged: no frequency check, no end minimization.
  - Profiles get `[ts_inputs]` (`mepd.inputs.TSInputs`: fmax, maxiter, continue_unconverged, check_frequencies,
    imaginary_cutoff_cm, exact_hessian, hessian_every), also in the web profile form ("TS optimization").

- **Orbitals are data, the Orbitals button is a view choice.** Each open session keeps every structure's
  (and conformer's) HOMO / LUMO ready in `<workspace>/orbitals/` at the session's level, built on the
  full grid ahead of time: from the job that computed the geometry when there is one, else a single
  point (`mepd.web.orbital_store`; a background keeper, up to 6 a sweep, never evicted). The viewer
  loads them with the geometry (from the store in ~40-100 ms), and Orbitals only draws them; it is
  remembered for every view (on in one, on everywhere), and HOMO/LUMO and % density only change the
  drawing. Paths show the orbitals of the frame on screen; "follow" and "this one" are gone. CLI runs
  with `--output` keep their SCFs' orbitals in `<output>/orbitals`. Switching the session's level
  (e.g. to TeraChem) makes the keeper redo them at that level.
- **Interactive reactor: Collide also pulls two atoms of one molecule together** (3 or more bonds apart:
  a ring closing, an H moving across), about a third of the time and always once everything is one
  molecule, where it used to do nothing.
- **FSM-TS is the default path method** (`RunInputs.path_min_method`, `mepd init`, a new web session's
  default profile, the demo's starter profile): the cheapest in the path-method benchmark at about GSM's
  quality. Profiles that name a method keep it; existing workspaces keep their saved profiles. QM/MM
  profiles still default to NEB (FSM regrows its string by geodesic interpolation, slow for hundreds of
  solvent atoms).
- **Interactive reactor: "Find their barriers" sits under the reactions it acts on** (it was at the top
  of the page). The top keeps "Stop & analyze" only while there are no reactions yet.
- **Web UI: the first screen says what mepd does.** "Compute how molecules react", then one card per
  scale the tools are grouped by: Molecule (build one, find its stable shapes), Reaction (what it turns
  into and the energy each step takes, drawn as an energy profile) and Network (the steps connected, drawn
  as a small graph). Each card holds your step there (pick a molecule, put it in the reactor, open
  Explore), so what mepd does and how to try it are the same three steps.
- **Web UI: first results say what they are.** A TS search that found no transition state joining its ends
  is no longer shown as a barrier: Explore draws it as a grey "no TS" arrow (not green), the Calculations
  list and job pages say "No TS" / "No transition state found (path top ≈ x kcal/mol)", Analyze puts it
  under "No TS yet", and the reaction card's energy diagram leaves the TS out. A path top above 100 kcal/mol
  is called out as a search that likely went wrong. The TS results overview then shows the path itself, not
  another saddle's IRC (which drew a reaction that was not this one).
- **Web UI: one reaction count everywhere.** The sidebar's Reactions list now also lists reactions with no
  arrow in Explore (barrierless, unstable product, ...; a click opens them in Analyze), so it matches the
  reactor, Analyze and "Its reactions". While a reactor run is being read, the story page says "N so far".
- **Web UI: after the reactor, always a next step.** Story cards say each outcome in words ("Happens on
  contact: no barrier", "The product falls back when relaxed"), offer "Try in <solvent>" for reactions with
  no TS ends and "Sample more paths" when no TS was found; "See them in Explore" and "Run the reactor again"
  show even when no barrier was found. Explore opens after a first reaction, not a first barrier.
- **Web UI: tool ticks are earned.** Barrier needs a TS whose IRC joins the ends; Method is done by changing
  the level of theory or saving a profile, and Kinetics by opening it (opening their tabs no longer counts).
  Unlock toasts use the tools' short names ("New tools: Barrier · Explore").
- **Web UI: names and wording.** Ions and zwitterions of a named molecule are named after it ("protonated
  glycine", "deprotonated glycine"); the reactor's TS searches are titled with species names, not SMILES.
  Reactor: "keep colliding", "Save this frame", a running-mean temperature (the instantaneous one swings
  far around the setting), bigger reaction pictures that stay inside the panel, and a note on why it runs
  GFN2-xTB. Reaction card: "seen once in the reactor", "In the reactor", "Path top"; the relaxed-ends note
  moved into a tooltip. Analyze's "Shuttled" filter is "With a carrier". Names and equations are no longer
  in monospace. Design shows one instruction at a time and its minimize/TS/clean buttons only once there is
  an atom; Settings' intro is one plain sentence; "Optimize N more" reads "N not optimized · Optimize".

- **A first visit is quieter** (from a first-time-user review):
  - **Reactor:** reactions are drawn as molecules, not spelled as SMILES. H swaps between identical molecules are
    a one-line count, not entries. The list and the badge count the same thing. The status line is time and
    temperature (the rest on hover). One button leaves: "Find their barriers" (or "Stop & analyze"); Discard is a
    link and "Add frame to Explore" is "Keep this moment" among the controls. The step bar ticks Reaction once one
    is caught.
  - **The reactor's summary page** hides the run controls behind "⋯", and the command and the other tabs behind
    "All results". "Show everything" keeps the full page. A finished run you are looking at no longer pops a toast.
  - **Library rows:** "radical", "triplet" or the charge instead of "+0 / 2"; where a structure came from is in its
    tooltip, not on every row. The "shuttle" badge reads "carrier".
  - **Welcome, coming back:** the steps tick from the workspace too; "Your last reactor run" links to it; tools
    not found yet are one "+N to find" tile instead of a wall of "?"; the interactive reactor no longer counts as
    "Find new reactions". Calculations is a tab as soon as there is one.
  - **Explore:** labels grow as you zoom out, so they stay readable. The how-it-works panel is a picture key, a line
    each. On phones, Tree and Select all are off the toolbar, and the toolbar and tabs fade where they scroll on.
- **Structures a calculation finds have everyday names** instead of their SMILES, for about 120 small molecules,
  radicals and ions: hydroxyl, formyl, hydroxymethyl, formic acid, H atom, H2 and so on. A complex is named after its
  molecules. Names you typed are kept; existing structures still named by their SMILES are renamed when the
  workspace loads.
- **FSM-TS converges its TS optimization to 0.001 eV/Å** (was 0.01; new option `ts_fmax`). At 0.01 a soft mode
  (20–45 cm⁻¹) could still be unsettled, so a real TS read as a higher-order saddle and the pair fell back to GSM.
  All 6 such fallbacks in a 20-reaction KAIST run became first-order TSs at 0.001, 0.07–0.48 kcal/mol lower,
  with the reaction mode within 20%. The g-xTB Hessian was not the cause: its frequencies agree at every SCF
  accuracy and finite-difference step tried.

- **`mepd.irc` is now the IRC function**; the geomeTRIC IRC module it shadowed is `mepd.irc_geometric`.
  `GeometryOptimizationNotConvergedError` is also a `mepd.ConvergenceError`.

- **Faster NEB defaults.** The conjugate-gradient chain optimizer grows its step 1.5x (was 1.2x) after 3 correlated
  steps (`positive_steps_thre`, was 15), and the TS-region stop needs the barrier steady over 15 steps
  (`ts_converged_window`, was 50). On the 6 KAIST reactions where NEB spent longest, channels found the same
  lowest channel (within 1 kcal/mol) on all 6 with about 80k g-xTB calls instead of at least 208k. Faster step
  growth alone lost a channel; FIRE and L-BFGS found fewer (0/6, 2/6). Set the old values in
  `[optimizer_kwds]` / `[path_min_inputs]` to reproduce earlier NEB runs.
- **Geodesic interpolation ~2.5x faster, same paths**: the least-squares solve runs in a numba kernel that
  repeats scipy's trf/lsmr steps operation for operation, and the Jacobians are built vectorized. On set30
  every path is bit-identical; median call 0.39 → 0.16 s. `numba` is now a dependency; without it (or for
  options the kernel doesn't cover) it falls back to `scipy.optimize.least_squares`.
- **Retry a barrierless reaction in a solvent** (`mepd discovery nanoreactor-retry`, web: the reaction card).
  A reactor reaction with no ends to search between in the gas phase (barrierless, products falling back,
  falling apart) can be tried again in an implicit solvent: its ends are re-optimized from the run's saved MD
  frames at the same level plus the solvent, saved as a profile of its own ("default in water"). If they hold,
  they join Explore as an edge at that level with Find TS; the card lists each solvent tried and its outcome.
- **Back means back in the web UI**: every page (a tab, a calculation, one of its runs) is a browser history
  step, so the browser's back and forward, a phone's back gesture and the job page's ← walk through what you
  opened; a reload stays on the page. A follow-up's page (substituent effects, solvent, a VRI check) says which
  results it came **from**, with a link; queuing one offers **Open it**, and its card links to its page.
  Substituent results number atoms from 1, like the picker and every viewer on the page.
- **Edit in Design is a tile**: under Molecule for a molecule, under Reaction for an edge (both ends).
- **Atom mapping in plain words**: candidates are named for what they are (Fewest bond changes, Best geometric
  fit, Equivalent atoms swapped, As given, Other mechanism), one line says whether the end was renumbered, and
  the fit reads "ends RMSD 1.44 bohr". The method details moved to hover text and a "?".
- **Substituent effects picked on the molecule**: tap atoms of the transition state in 3D to choose where
  groups go (a hydrogen, or an atom carrying one), pick groups from a short palette (More groups for the rest),
  and see how many variants that makes. A link loads the reaction into Design to change one group by hand.
  Replaces the wall of group buttons and the box for 0-based atom numbers.
- **A path search's results open on an overview**, like a reaction's report: the lowest barrier, how many
  channels, ΔE and whether the IRC connects the ends; an energy diagram with every channel; and a table of them.
  Click a channel for its 3D path below. Details (search summary, warnings, the atom mapping, now folded until
  clicked) and the follow-up calculations move to the bottom.
- **Web UI tools grouped by scale**: a selection's tools sit under **Molecule** (Optimize, Conformers, Complex),
  **Reaction** (Pick products, Transition state, Channels, Valley-ridge inflection) and **Network** (Reactor,
  Network expansion, Retrosynthesis), the same groups in the same order for any selection.
  Above them, a **surroundings switch** (Gas phase · Solvent · Explicit · Protein) applies to every scale:
  Solvent opens Explore › Conditions, Explicit and Protein build the QM/MM system (or go to the one already
  built), and a QM/MM system's tools are the same groups, run embedded. Retrosynthesis has its own card; Optimize
  offers Minimum or Transition state (the former "TS from a guess"); Edit in Design and Its reactions are
  buttons above the tools.

- **The demo needs no password** (mepd.atomsforhumanity.org). Opening the page makes a private workspace
  at once; `MEPD_DEMO_PASSWORD` / `--demo-password` is now optional and gates the demo only when set (the
  k8s manifest no longer sets it; `deploy/demo/run.sh` still uses one). The workspace is made on the page's
  first API call, so crawlers leave nothing on disk. In the open demo, Log out is **Delete workspace** (with
  the same warning page). mepd no longer generates or reads `.demo_password` itself.
- **Demo: logging out deletes the visitor's workspace.** Log out now opens a warning page; its button stops
  the visitor's calculations and interactive reactors and deletes everything under `visitors/<id>/`. Before,
  logging out only dropped the cookie and left the workspace on disk, unreachable. A visitor who closes the tab
  instead keeps their workspace. A GET of `/logout` only shows the warning; deletion needs the page's POST.
- Web UI on phones: the first screen shows the molecule's 3D preview right under the choices; the reactor's 3D
  view leaves room for its controls, counts reactions on the view and offers Find its barrier under it; job
  pages stack their header, scroll their tabs sideways and wrap the step bar; Explore shows the graph above
  the structure list; a new page opens at its top. Reloading while an interactive reactor runs returns to it.
- The interactive reactor says how to make a reaction when none has happened after 4 ps (left alone, few
  do), and a run with none offers Run it again with the same mix and tips. Unlock toasts skip the first steps.
- Web UI, less text and fewer controls: the selected molecule's actions are icon tiles grouped under short
  headings, one form open at a time under its group; field help moved into tooltips; square 4 px corners
  instead of pills; the reverse barrier is its own arrow in Explore; the interactive reactor has a wall-size
  slider and works by touch. Removed the top bar's New calculation and the sidebar's Add (molecules come from
  Design, or drop .xyz files on the sidebar); Level of theory and Confirm minima moved to Settings.

- **FAIR-Chem (UMA) no longer compiles the model by default** (`fairchem_engine_kwds.inference_settings`).
  mepd's `default` is now the uncompiled model; `compile` is fairchem's own default (merged experts plus
  torch.compile), and `turbo` stays. Compiling cost 20-40 s in every process that loads the model, and again
  for each new batch size, while mepd starts one per job and per parallel worker (the demo log showed 7
  compiles in one Find TS). On an RTX 5090 (21 atoms): ready in 8 s instead of 21-40 s, 0.025 s per gradient
  instead of 0.010 s once compiled, the same energies to 1e-7 Eh. A Find TS (FSM-TS, allyl vinyl ether, same
  endpoints, two runs each): 148 and 172 s instead of 278 and 191 s. Saved profiles with `default` get the new
  behavior.

- **Faster analysis after the path searches** (Reaction channels, Sample more paths, Stop & analyze). Each
  pair's TS optimizations and IRCs start as soon as its search finishes, in processes of their own beside the
  searches still running, instead of all waiting for the last search. Once no search is left, they run on as
  many processes as the searches had (pairs x branches x images) instead of `--workers`. A g-xTB IRC runs its
  two branches at once (the same path, bit for bit, in about half the time).
  A TS optimization that failed is recorded (`ts/<label>_tsopt_failed.txt`), so Sample more paths does not
  spend the same time failing again. Allyl vinyl ether -> pent-4-enal (2 workers): 109 s -> 92 s, 33 s -> 2 s
  left after the last search.
- **Nanoreactor refinement runs species and reactions at once** (`--refine-workers`, default one per core,
  for g-xTB and QCOP engines; others one at a time, as before). Live minimization streams are now per thread.

- **FSM-TS falls back to GSM, not NEB** (`[path_min_inputs] fallback = "GSM" | "NEB" | "none"`,
  overrides in `gsm_inputs` / `neb_inputs`). On 20 KAIST reactions GSM found the lowest channel more often
  than NEB. Without a GSM binary, or when GSM fails, the pair goes to NEB. `neb_fallback = false` still
  turns the fallback off.
- **FSM-TS defaults to a coarse string: 3 beads (`min_images`) and 3 minimization steps per growth
  (`max_min_iter`)**, down from FSM's 10 and 100. FNEB keeps its defaults.
- **Kinetics: the degree-of-control bars say what they mean.** Each reads "3.2× more" or "1.6× less": what
  lowering that TS, or stabilizing that species, by 1 kcal/mol does to the target (X is in the tooltip).
  "No barrier controls it" says equilibrium only when the run has stopped changing.
- **Kinetics: the amounts chart and the result use the time unit the run was set up in**, not always seconds.
- **`mepd web` no longer handles HTTPS; it is a plain HTTP app.** TLS and HTTPS policy belong to the proxy in
  front (here, Cloudflare). Removed from demo mode:
  - the redirect of plain-HTTP requests to HTTPS (it read `X-Forwarded-Proto`/`CF-Visitor`);
  - the `Strict-Transport-Security` header.

  Removed from both modes: login cookies were marked `Secure` based on the same headers. Set "Always Use
  HTTPS" and HSTS in Cloudflare instead.
- **No lockout after failed logins, in either mode.** Logins used to be blocked after 10 failures per
  address in 10 minutes. Behind a proxy every visitor can share one address, so a few typos from anyone
  locked every tester out. A wrong password still waits 0.5 s. The protection is a long random password;
  rate limiting, if wanted, belongs at the proxy (e.g. a Cloudflare rule), which sees real addresses.

- **Gated models (UMA, eSEN) are usable once their weights are on the machine, in demo mode too.** Demo mode
  refused every gated model in testers' profiles, and the model menu labelled them "Hugging Face access
  needed". That holds only when the weights still have to be downloaded with a login. A gated model whose
  weights are already in fairchem's cache, as UMA is in the image, now runs like any other model
  (`mepd.engines.mlip.weights_available`).
- **The "FAIR-Chem model" engine works in demo mode.** It refused `engine_name = "fairchem"`, although it is
  the same engine (`FAIRChemEngine`) a `mlip` profile with a UMA model uses, with FAIR-Chem's own settings
  (`task`, `inference_settings`). Its model is checked like any other: blank means `uma-s-1p2p1`. The demo's
  Settings menu now shows the engines the demo doesn't run greyed out, with the reason, instead of letting a
  tester pick one and fail on save.
- **Demo mode blocks only what would hand a tester the server.** It worked from an allowlist that new
  features had fallen behind. Unblocked:
  - the operations Conformers, Substituent effects, Mechanical force, Minimize the design, Optimize the design
    as a TS, Sample more paths, Place in a protein (QM/MM) and QM/MM system at this site;
  - the Psi4 engine;
  - the ReactionT5 and AiZynthFinder retrosynthesis methods, the ZINC stock, and step verification.

  Still blocked:
  - reading server files;
  - the ASE, QCCompute and ChemCloud engines;
  - programs, paths and credentials in profiles, now including Psi4's `python`;
  - local-LLM retrosynthesis.

  A test now fails if an operation is added without a decision for the demo.
- **Demo limits are fully configurable.** As well as numbers, the tables of per-operation caps and timeouts,
  profile caps, interactive-reactor and QM/MM limits take JSON merged into the defaults
  (`MEPD_DEMO_CAPS='{"channels": {"workers": 8}}'`), and misspelled names stop the server. New limit
  `psi4_max_memory_gb` (16 GB): Psi4's memory is text, which the numeric caps did not see. The beta manifest
  sizes the limits for afh-c04.
- **Demo mode offers only models that are installed.** It used to accept Orb and MACE, which can't be
  installed alongside fairchem and so aren't in the image; a tester's job with them failed at run time.

### Fixed

- **`mepd optimize` on a single atom** (an ion such as Cl⁻, Br⁻, H⁻) failed with an AssertionError in the optimizer, which left halide ions without energies and kept them out of kinetics. An atom now gets one single point and counts as a minimum by definition (no Hessian).

- **Explorer, async: a run stopped by `max_species` ends with what it has**, as one out of time does: no IRC checks or
  whole channel searches after its queue (unbudgeted), and no steering single point queued into the closed queue
  (`AttributeError: 'NoneType' object has no attribute 'submit'`). F0007 with a version of this branch that
  channel-sampled every IRC-end pair reached 200 species at 1459 s of 1800, then ran channel searches for 40 minutes
  and failed without a summary.

- **Kinetics crashed on networks slower than their time span** ("Values in `t_eval` are not within `t_span`",
  `mepd.discovery.kinetics.simulate` and `mepd.conditions.simulate_first_order`). With only slow steps, as just
  after an explorer's first TS, the time samples ran from `time_s` to `time_s`, and `geomspace` could put one a
  rounding error past the end, which `solve_ivp` refuses. This depended on the value: 2 ms (methyl formate),
  3, 7 and 20 ms and 7200 s failed; 1 ms did not.
  - Samples past the end are now dropped and the last is pinned to `time_s`, as `mepd.microkinetics` already did.

- **Orbitals in busy workspaces are fast again.** Looking up a calculation's own SCF scanned every line its
  index.jsonl ever got (346,218 lines in one workspace's 15 jobs, for about 15,000 molden files still kept),
  fitting a rotation to each: 1.4 s per frame, serialized across threads. Now only the entries whose files
  exist are read (then the index is read on from where it stopped), their positions compared as one array,
  and a rotation fitted only where every atom's distance to the centroid agrees: 0.4 ms, the same answers on
  all 512 test lookups there. Bond orbitals' integrals and localization are compiled (numba, threads run them
  at once): building a 20-atom path's frames went from 2.6 s to 0.07 s each (30 frames: 77 s to 2 s), a
  75-atom frame from 4.2 s to 1.2 s. Whole and Fragments use the same lookup.
- **Async explorer with the reactor: species past the seed were never expanded** (from 07c1cd9, the reactor's
  energy cutoff). The proposal job read the species it expands from the explorer's network inside the queue's
  pool worker, whose copy of the network is the one from when the pool was forked (only the seed): every
  proposal job for another species raised IndexError, and the network grew only by the seed's own proposals,
  IRC ends and channel sampling. The reactor's options for a species are now made when its proposals are
  queued, in the main process. A failed proposal job is now listed under the round's `proposals_failed`
  instead of its `sources` (it had counted as an expansion, so a run could end "converged" with expansions
  missing), and the queue's `job` event in network_aware.jsonl carries the failure (`why`). A test runs the
  queue's jobs in a forked worker, as the real pool does.

- **Explorer free energies: a TS's spectators also count bound, when that is lower.** Counting a
  molecule whose bonds don't change apart (at the standard state) drops whatever it does to the TS
  electronically, so a catalyst acting only through an H-bond, an ion pair or a dative contact lost its
  catalysis. A TS with spectators now also gets its correction bound (the whole complex, one molecule, as
  before 9bdac62), and the TS takes the lower of the two (`free_energy.both_channels`). Not the two rates
  added: the apart free energy already lets a spectator roam the whole standard-state volume, the positions
  next to the TS included, so a sum would count a spectator that does nothing for the TS twice (a rate up to
  2x too high where the two are close: RT ln 2, 1.4 kcal/mol at 1000 K). One extra Hessian per TS with spectators, cached with the TS (and in the TS
  job's precomputed correction). The steering single points (and `refine_energies`) move each channel by its
  own single point -- reacting complex and spectators apart, or the whole TS -- before taking the lower
  (`channels_shift`). On j_4e10ef7a (g-xTB, 1 M, recomputed offline): at 1000 K, apart wins on all 40 steps
  with spectators (by 5.7-51.7 kcal/mol), so nothing changes; at 500 K, bound wins on 15 of them (by up to
  20.6 kcal/mol, e.g. `pair_0_9_leaf_3` 0->1: 70.6 apart -> 51.5). Unimolecular formic acid is unchanged.

- **Species identity: one molecule no longer splits in two by its atom numbering.** The stereo SMILES that
  tells species apart (`_stereo_smiles`, used by `same_molecule`, the reactor's and nanoreactor's product keys
  and the mixture's molecule keys) puts the atoms in a canonical order that fixes them only up to the graph's
  symmetry, and a tie in the bond orders falls on whichever equivalent atom comes first. Dihydroxycarbene
  HO-C-OH is written [C-]=[OH+], so its formal C=O landed on either O and its syn/anti conformer read as E or
  Z by numbering alone: every HCOOH run held the carbene twice (species 2 and 4, same energy to 1e-10 Eh).
  When a SMILES has E/Z marks, each equivalent order of the heavy atoms (up to 24) is now read too and the
  distinct SMILES are kept, sorted and joined by "|": one key per geometry whatever the numbering. Where they
  agree, nothing changes, so real E/Z isomers stay apart and so do the carbene's three conformers. Rescored
  offline, the HCOOH selectivity is unchanged at 27.6: the carbene steps are out of reach.

- **Explorer: one TS found twice counts as one channel.** A TS asked for one pair whose IRC lands on the ends
  of a step with that very TS (or a channel search finding it again) was kept as a parallel channel, and each
  copy got its own mirror-image twin: formic acid's dehydration TS counted 3x (develop) or 4x (reactor at
  2200 K) against decarboxylation's 2x, so dehydration:decarboxylation read 41.4 or 55.2 where the network
  gives 27.6 (g-xTB, 1500 K). A step is now the same TS as another between the same two species when their
  energies agree within 0.1 kcal/mol and their geometries superimpose (proper rotation, identical atoms in
  any order); it is checked when the step is added and again when its IRC gives it its real ends. A mirror
  image (the twin TS) and another conformer's TS stay distinct channels. `mirror.superimposable` now matches
  identical atoms in any order (the isomorphisms of the two structures' graphs, not only one structure's
  automorphisms), so a twin found under another numbering is recognised as that twin. Rescored offline:
  develop's HCOOH run 41.4 -> 27.6, integration's 2200 K run 55.2 -> 27.6, integration at 1800 K 27.6
  unchanged.

- **Explorer free energies: a TS's spectators count apart, as in its species.** A species of several molecules
  counts them apart at the standard state; a TS counted as one complex, so in a cluster seed a step between two
  molecules paid its spectators' association entropy on top of its barrier (about +20 to +50 kcal/mol at
  1000 K on (HCOOH)4). Now a TS's reacting complex -- the connected pieces of the union of its step's two ends'
  bond graphs that hold a bond made or broken -- stays whole and loses its reaction mode, and each spectator
  counts apart with its own electronic energy, as the species' molecules do (`free_energy.ts_groups`). The ends
  are the step's IRC ends, else the two structures its TS search started from (written by the TS job as
  `ts/<label>_ends.xyz`), else its two species; ends whose bonds don't bracket the TS's leave it whole (said).
  Not from the TS geometry alone: partly formed bonds are long, and a distance split can cut the reacting
  complex in two. The steering single points (`_steer_shifts`) and `refine_energies` split TSs the same way.
  Channel steps now record their IRC (`files["irc"]`). On j_4e10ef7a (recomputed offline), e.g. step
  `ch_0_2_6` 0->2: 80.2 -> 48.4 kcal/mol; steps through species 4: 120-157 -> 75-109. A step whose
  molecules all react (`fast_0_2`: an addition with three concerted proton transfers round the tetramer) and
  unimolecular formic acid are unchanged.
- **Explorer summary: a step out of the kinetics' reach has no kinetics barrier, not ~800 kcal/mol.** Such a
  step is closed (its TS + 1 Hartree, so it carries no rate) and `kinetics_barrier_kcal` reported that
  placeholder as its barrier (818-893 kcal/mol for the six closed steps of j_4e10ef7a, 810 for formic acid's
  CO2 + H2 -> CO + H2O). It is now null.

- **Explorer, async: the time budget is a real limit, and the jobs the kinetics need start first.** At the
  budget (or Stop & analyze) the running jobs are cancelled, their worker processes killed with what they
  started, instead of awaited, and the run goes straight to its analysis: the final IRC pass and the final
  channel sampling (full path searches) no longer run after the queue, unbudgeted (they ran 50 min past a
  1 h budget in a (HCOOH)4 run). Waiting jobs start by rank, in queue order within one: steering single
  points, then IRC checks, then proposals, TS searches and channel sampling, then path-search pieces. A
  job that needs several workers (proposals: all of them; channel sampling 3) no longer waits while
  narrower jobs queued after it take each free worker: after 60 s (`wide_wait_s`) the jobs behind it wait
  until half its workers are free, and it starts on those. A channel-sampling job takes at most half the
  workers (it took 3 of 4, so one sat idle beside it). Proposals for species the kinetics populate never ran in a 15-min (HCOOH)2 run for that. IRC checks minimize their ends without the
  frequency check (a fifth of their time). `network_aware.jsonl` logs each job's wait and run time.
- **The reactor generator's products are molecules, not atom numberings or hot snapshots.** Each event's
  structure is quenched (collide's 200 steepest-descent steps) and keyed by its stereo SMILES alone; the bond
  set by atom number made the same reaction in another of a cluster's identical molecules a new product, and
  a hot contact that relaxes back to the seed a product at all. Convergence (Good-Turing) counts the reactions
  out of the seed's state, and needs at least 1/`unseen` of them; proposals come most-often-reached first.
  (HCOOH)4 at 1800 K, 48 trajectories of 5 ps: 284 "products" in 332 events (76% seen once) were 79 states,
  27% of events fell back to the seed; the seed's own reactions: 15 products in 54 events (17% seen once).
- **MLIP endpoint minimizations in implicit solvent failed, and runs went on from unminimized endpoints**
  ("Endpoint batch minimization failed (ElectronicStructureError: Electronic structure failed.)").
  Two causes. The endpoints are optimized in threads that share one UMA model on the GPU, and torch's
  lazy load of its CUDA linear algebra is not thread-safe ("lazy wrapper should be called at most
  once"): FAIR-Chem forward passes now take turns. And the `LBFGSLineSearch` optimizer stalls on an MLIP's
  float32 forces where the solvation correction cancels most of the force (`LineSearch failed!`): in ALPB
  water, 6 small molecules x 2 repeats, 9 of 12 converged with UMA and 2 of 12 with AIMNet2, against 12
  of 12 each with `LBFGS`. The FAIR-Chem and MLIP engines (and their Settings defaults) now use `LBFGS`.
- **The Docker image runs GFN-xTB with the official xtb 6.7.1 release** (`XTB_EXECUTABLE=/opt/xtb/bin/xtb`,
  checksum-verified), not conda-forge's, whose CPCM-X gave one energy for every solvent (the test below failed
  in CI, so nothing deployed since 2026-10-28). Not the g-xTB build either: its CPCM-X gives other values
  (HCN in hexane +0.5 kcal/mol, the release -2.2). CREST keeps conda-forge's xtb.
- **CPCM-X solvation energies were the same for every solvent.** With `--cpcmx`, xtb's `energy` file holds the
  conductor-like SCF energy CPCM-X starts from (on HCN: -7.0 kcal/mol in water, THF and hexane alike); mepd now
  reads the printed CPCM-X total (water -5.65, hexane -4.12, as xtb's own dG_solv). xtb also needs CPCM-X's
  full solvent names: DMSO, DMF, CH2Cl2, THF and ether stopped after the SCF (`--cpcmx thf`), and now run as
  `--cpcmx tetrahydrofuran` etc.; methanol, acetone and dioxane, not in CPCM-X's database, no longer offer it.
  xtb's CPCM-X has no gradient (the gas phase's is returned), so profiles and `mepd solvent --mode reoptimize`
  refuse it with that reason; it stays for single points on gas-phase geometries. (#15)
- **The IRC-ends check called identity reactions "no reaction".** Cl- + 1-chloropentane (SN2, the other Cl
  bonded at each end) was flagged "both IRC ends are the same species"; ends now count as the same only when
  the same atoms are bonded.
- **"Electronic structure failed." without the cause** from optimizations and TS searches through ASE: the
  program's error is now part of the message.

- **`run --use-tsopt` after FSM-TS** optimized the TS again from the string's interpolated peak (on QM/MM)
  instead of taking the TS FSM-TS had converged and checked; on KSI step 1 that search ended 108 kcal/mol
  up on another reaction. It now takes FSM-TS's TS and runs only the IRC. (#22)
- **NEB steps that blow up are rejected**: a step that raises an image by more than 1 Eh or a gradient
  100-fold is undone and retried smaller (dt halved); after three, the branch fails instead of MSMEP
  splitting on the wreck (a KSI NEB went to 1e10 kcal/mol in three steps). (#23)
- **xtb no longer mixes electronic temperatures within a path**: once an SCC needed Fermi smearing, the
  engine holds that temperature for every later calculation and computes the rest of the batch (the
  path's images, a Hessian's displacements) again at it. (#24)
- **SMILES endpoints embed the same way every time** (RDKit seeded): the unseeded embedding followed the
  process's random stream, so `mepd channels --start <SMILES>` could start from another geometry and
  conformer pool from run to run. (#29)
- **Web results of QM/MM jobs** were read with a process-wide QM region that another thread reading
  another job's result could reset mid-read (the whole system then perceived as one molecule). (#29)
- **Orbitals of a whole QM/MM system froze the web app.** The 3D viewer asks for the orbitals (and, with
  Fragments on, each molecule's orbitals) of the geometry it shows; for a protein system that meant xtb on all
  3,951 atoms (it crashed after 3 s) and Lewis structures of the protein's fragments tried charge by charge for
  minutes, holding the browser's connections so every other page waited. `/api/orbitals` and
  `/api/fragment-orbitals` now refuse a whole QM/MM system at once.
- **Parallel workers** (`_fork_map`): a worker OOM-killed mid-run no longer leaves its siblings' own workers
  orphaned for good (each forked worker now dies with its parent, on Linux), and forked workers let go of the
  parent's multiprocessing resource tracker. Together these hung a finished `discovery expand` with GPU
  steering for 6 hours: Python waits for that tracker at exit, and the orphans kept it alive.

- **Explorer**: labelling a steering level (which saved single points match it) no longer loads an ML
  potential's model; it did, putting CUDA in the explorer's own process, and every later parallel step ran
  serially. ML levels' labels now name their model. `discovery expand` sets `SELLA_DISABLE_GPU=1` unless set:
  Sella's GPU eigensolves (>= 200 coordinates) had the same effect.

- **Channels**: mechanisms whose chosen atom numbering is the input's own share one path search; run once
  each, the copies wrote the same pair folder at once and failed (`OSError: Directory not empty`).

- **Network kinetics** (`mepd.discovery.kinetics.simulate`): rates spanning ~1e13 to ~1e-80 /s no longer stall
  the stiff solver for hours or create material through rounding (flux-form right-hand side).
- Explorer element tables come from RDKit, so Pd and other metals work in collide, free energies and the
  nanoreactor.
- **QM/MM TS searches with a caged environment ran on the whole protein**: with no environment atom free
  to relax (`qmmm_environment = "cage"`), optimizations and TS searches skipped the microiterations and
  gave Sella every atom (3,951 atoms: ~4 min a step). They now see the QM region alone, as in a relaxed
  environment.
- **FSM and FSM-TS crashed on a QM/MM system in a protein** ("cannot reshape array ... into shape (N,3)"):
  a mostly frozen system is interpolated as a small sub-system, and the geodesic smoother kept that
  sub-system's path, which FSM turned into whole-system nodes. The smoother now holds the whole system's
  path. FSM-TS used to fall back to NEB on every protein pair, and an NEB crashed when MSMEP split a branch.
- **QM/MM nanoreactor in a protein stalled after its first segment**: the live events were found over the
  whole system (bond detection and molecule labels over 3,900 protein and water atoms, the whole trajectory
  again after every segment). They are now found among the QM atoms (indices still the system's), at most
  every 30 s.
- **QM side-chain charge from the residue's state**: a protonated Asp (ASH) or Glu (GLH), neutral Lys or
  doubly protonated His in the QM region was counted with its name's usual charge.
- **One molecule with a metal centre no longer compares as two stereoisomers** (`mepd.compare_structures`,
  `same_molecule`, `connects`). RDKit's stereo tag on a metal (@SP, @TB, @OH) records which coordination
  polytope it fitted to the geometry, and for a fluxional centre that changed between two geometries of one
  molecule (IrCl4L as @TB6 in one, @OH7 in the other). The tag is now dropped when all of the centre's
  ligands are alike but at most one (MA_n, MA_nB), which have no stereoisomers; cis/trans and fac/mer
  centres keep it. In a 200-reaction Transition1x-TMC screen, 10 more reactions were kept and none fewer.
- **Alkali and heavier alkaline-earth cations are ions in bond perception** (Li, Na, K, Rb, Cs, Ca, Sr, Ba;
  `mepd.discovery.nanoreactor.perceive_bonds`, the nanoreactor's bond history, fragment cuts). Their Cordero
  radii made a K+ "bond" to every O within 3.1 A, so a reactor with KOMe glued urea, methoxide and methanol
  into one "unusual bonding" species and reported coordination changes as reactions. They now come out as
  their own ions (`[K+]`), always charged (K+ + MeO-, never two radicals). Be and Mg stay covalent. (#10)

- **g-xTB `extra_args` with a space no longer run silently in the gas phase.** xtb takes `"--alpb water"` given
  as one argument as an unknown flag, exits 0 and returns the gas-phase energy. `GXTBCalculator` and CREST's
  `extra_args` now split such entries into one argument per word. A g-xTB engine also refuses implicit-solvent
  flags (`--alpb`, `--gbsa`, `--cosmo`, `--cpcmx`, `--gbe`): g-xTB has no solvent parameters, so use a
  `[solvation]` table (GFN2-xTB solvation on top). GFN2-xTB through the same engine keeps them. (#11)

- **An uploaded ion pair splits into ions, not two radicals** (`mepd/web/compose.py`). Without charges, the
  molecules of a split complex were made neutral where they could be, so pyridinium trifluoroacetate came out
  as two radicals. When a closed-shell structure splits into radicals, GFN2-xTB partial charges (when xtb is
  installed) now decide each molecule's charge, as in the nanoreactor. (#14)

- **Nanoreactor (xtb/g-xTB MD): a reactor no longer falls apart into atoms when xtb's SCC breaks down.**
  xtb's MD goes on after a failed SCC with zero energy and zero forces (every later step fails too, and so
  does every later segment, from the broken charges in `xtbrestart`), so the atoms fly off in straight lines;
  an SCC that lands on a wrong solution gives a jump of tens of Eh, a kick to ~10^4 K, "MD is unstable", and
  every later segment restarts from those velocities. Seen with a ketosteroid isomerase active site (1OH0,
  QM: the steroid + Tyr16/Asp40/Tyr57/Asp103, charge -2) at 3000 K: atoms after 0.3 ps, no products. Now
  such a run is cut at its last good frame (zero energy, a jump of more than 1 Eh between frames, or an
  unstable stop) and the segment goes on from there with a fresh SCC guess and new velocities at the
  temperature; a segment that breaks down 6 times keeps its good frames and the next starts from the last
  of them, and the piston closes 15% less far from then on (each time: a reactor squeezed past what xtb can
  converge is squeezed less rather than failing the expansion; 1OH0 at compress = 0.5 failed with "xtb's MD
  broke down 6 times in each of 3 segments in a row"). If 3 segments in a row still get nowhere, the MD ends
  early and the trajectory so far is analyzed. The job log says each of these.

- **Nanoreactor on a QM/MM system (QM region in vacuum): the QM region's side chains stay anchored.** The
  link H (where the protein's backbone was) and the QM atoms they cap (CB) are held at their distances to
  each other (xtb `$constrain`), and the piston pushes only the molecules without an anchor (the substrate,
  into the residues), the anchored ones kept inside the wide wall. Before, nothing held the residues: in the
  hot MD they drifted off and lost their caps (1OH0 at 2500 K: the anchor frame moved 3-6 Å and CB-H
  bonds broke, 27 of 29 reactor states torn; now within 1 Å, caps kept, no xtb breakdowns). Generator option
  `anchors` (0-based atoms of the QM model, `none` for none) overrides it. The wall's atoms go one
  contiguous range per sphere line: conda-forge's xtb 6.7.1 (the Docker image's) aborts with "free():
  invalid pointer" on a comma in a `$wall` atom list, which failed the first version of this on the server.

- **QM/MM nanoreactor (`embedded=true`): hours faster.** After each MD segment the live events were
  perceived over the whole system (protein and water included), re-reading every segment so far: hours per
  segment for a protein. They are now found among the QM atoms only (indices still the system's), each
  segment read once. The products step reads the QM atoms, and whole frames only for products. The MD's
  xtb SCC is smeared at the reactor's electronic temperature (3000 K) from the start and starts from the
  previous step's charges (0.26 → 0.17 s per step on 1OH0, and no 300 K SCC that fails first); a step whose
  energy jumps is redone from scratch.

- **GFN2-/GFN1-xTB through the xtb program (`engine_name = "xtb"`, QM/MM low levels): an SCC that does
  not converge is retried** with Fermi smearing (`--etemp 1000`, then `3000`, 1000 iterations), as the g-xTB
  engine already did. Before, one such geometry (e.g. a bond breaking in a 3000 K QM/MM nanoreactor) ended the
  whole run. Its errors now name the method that ran (they said "GFN-FF" for GFN2 runs too).

- **`mepd.find_ts` without `fmax`/`maxiter` used the engine's optimizer default** (0.01 eV/Å for the
  ASE engines; mepd#6), looser than `mepd ts` (`TSInputs`: 0.001 eV/Å, 1000 steps), at which soft modes read
  as extra imaginary frequencies. It now defaults to `mepd ts`'s values, which `mepd force`, the substituent
  screen and the solvent tools inherit. Through QCCompute this means geomeTRIC's `convergence_gmax`
  1.94e-5 Hartree/bohr, tighter (and slower) than geomeTRIC's own default. A given value wins; an engine
  whose `compute_transition_state` takes no `keywords` is called as before. (`mepd ts` itself, the frequency
  check and minimized IRC ends were already on `develop` from the checked-TS work.)

- **A `gxtb` symlink to the g-xTB 2.0.1 build ran GFN2-xTB** (mepd#4). mepd left out `--gxtb` for any
  binary named `gxtb`, but the 2.0.1 release is an xtb 6.7.1 build that runs GFN2 without it (exit 0, no
  warning). The flag is now decided from the program (`mepd.programs.standalone_gxtb`, which reads
  `--version` once per binary), in the engine, orbitals, nanoreactor and solvation. A run meant as g-xTB
  whose output reports a GFN Hamiltonian now raises instead of returning a GFN2 energy. `find_gxtb` also
  finds `~/.local/opt/xtb-gxtb-*/bin/xtb`. mepd's image passes `/opt/gxtb/bin/xtb` and was not affected.

- **QCCompute TS optimizations ignored their keywords** (mepd#5). `QCComputeEngine.compute_transition_state`
  never passed `keywords` on, so FSM-TS's `ts_fmax`/`ts_maxiter` and `mepd.find_ts(..., **keywords)` with
  ORCA, Psi4 or TeraChem ran geomeTRIC's default convergence. Keywords now reach geomeTRIC: `maxiter`
  (or `steps`), and ASE's `fmax` in eV/Å as `convergence_gmax` in Hartree/bohr (0.001 eV/Å → 1.94e-5);
  an explicit `convergence_gmax` wins. The default is still 1000 steps.

- **TeraChem frozen-atom optimizations froze nothing** (mepd#3). With `new_minimizer yes`, which mepd always
  sets, TeraChem ignores a `$constraints / atom N` group without an error; geomeTRIC reads
  `$constraint_freeze / xyz N,M` after an `end` line. QCCompute TeraChem optimizations with
  `frozen_atom_indices` now write that group (`$constraints` stays for `new_minimizer no`). A QM/MM `tc.in`
  may give frozen atoms either way; a constraint other than whole frozen atoms is rejected, no longer dropped.

- **Memory of reading a finished search back** (`mepd/neb.py`, `mepd/TreeNode.py`, `mepd/cli_common.py`,
  `mepd/discovery/cli_expand.py`). Collecting TS guesses after the path searches reloaded every pair tree with
  its whole written history, building a bond graph for every node of every string: 5.8 GiB for one long
  22-atom search (9,621 strings), only to use each leaf's last string. `TreeNode.read_from_disk(ends_only=True)`
  and `NEB.read_from_disk(ends_only=True)` read just each search's first and last string; the TS-guess
  readers use it (same guesses, 0.3 GiB, under a second instead of 82 s).
- **Memory of a g-xTB search** (`mepd/engines/_sella.py`, `mepd/pathminimizers/gsm.py`). Sella imported torch
  (~400 MiB) only to look for a CUDA GPU; with no GPU visible (no `/dev/nvidia*`, or `SELLA_DISABLE_GPU=1`)
  Sella is now imported without it, and torch stays importable for anything else. GSM's history kept a
  molecular graph for every node of every live string (each gradient batch, the whole run: ~600 MiB on a
  20-node, 20-atom search); those snapshots now carry none, and the one a stopped search analyzes gets them
  back. Results are unchanged.

- Explorer element tables come from RDKit, so Pd and other metals work in collide, free energies and the
  nanoreactor.

- **geomeTRIC IRC gradients**: `mepd/irc_geometric.py` multiplied every gradient it handed geomeTRIC by
  Bohr→Å (0.529), inconsistent with the energies. Affected IRCs of engines without their own (QCCompute) and
  `mepd discovery`'s geomeTRIC IRC; g-xTB, ASE, MLIP and FAIR-Chem engines use Sella's IRC and were not affected.

- **A multi-step result was drawn as parallel channels from reactant to product**, although each IRC joins
  only neighbours (start ⇌ intermediate, intermediate ⇌ end). The results overview now draws such a route as one
  chain, start → TS 1 → intermediate 1 → TS 2 → … → end. Its table lists each step from the species before it,
  with that step's own barrier and the route's hardest step. This covers TS searches whose IRC-verified route has
  several steps, channel searches with only multi-step routes (a picker when there are several), and unverified
  paths that split into pieces. An IRC that reaches an intermediate is no longer listed as a channel of a
  one-step result either. The results carry the ordered route (`routes`), and cached results are rebuilt.
- **Orbital phase flipped from frame to frame.** An orbital's sign is arbitrary and each SCF picks its own,
  so the blue and red lobes swapped along a path (16 of 40 neighbouring IRC frames for the HOMO). Each
  frame is now drawn with the sign that makes its overlap with the frame before it positive (sampled
  around every atom in both grids), carried along the path from its first frame; a running calculation's
  frames against the last one drawn. As drawn: 0 flips.
- **A flaky reactor test failed a deploy.** The interactive reactor packs its molecules from a clock-based
  seed, and from 2 packings in 12 the H atoms of `test_reaction_events_are_found_as_the_nanoreactor_finds_them`
  never met within its 15 s. `sandbox.start` takes a `seed`; the test uses a fixed one (1.8 s, 6 of 6).
- **Orbitals while playing an IRC.** IRC frames have their rigid motion taken out, so they never matched
  the IRC's own SCFs and each frame waited for a single point (2 of 40 frames shown with their own
  orbitals). Kept SCFs now match a frame turned and moved (Kabsch fit), and the orbitals are evaluated in
  the frame's orientation (same energies and atom shares as a single point on it); with orbitals shown,
  a path's frames load in the background, so playback draws each frame's own (40 of 40). A finished
  job's result frames (TS, IRC, path) get their orbitals built right away and kept on disk
  (`jobs/<id>/orbitals/built/`), and the viewer loads 4 at once: all 41 IRC frames in ~0.5 s
  (~5 ms a frame on the server).
- **Live orbitals vanished with every chain update.** A live view reloads the geometry on each update,
  which cleared the orbital surface, and a result that arrived after the chain had moved on was never
  drawn. The surface is now redrawn after every reload, and while following, the newest orbitals stay
  drawn until the next frame's arrive (on screen in 60 of 60 samples over 30 s of a running TS search).
- **Design: Add molecule on empty space.** With Add molecule picked (e.g. AlCl3), a click on an empty
  canvas dropped a carbon (methane) instead; it now puts the chosen molecule there, and a click on empty
  space next to an existing design adds it at that spot (`place` edit with `point` and no `atom`), as Add
  atom does; clicking an atom still puts it next to that atom. The hints say empty space first.
- **A reaction's Report (.html) / Data (.zip) could leave out its energies and paths.** With ends that have
  no energies at one level of theory (e.g. added from SMILES and never optimized; every edge with no reaction),
  the diagram said "No energies"; it now uses the search's own (its reactant at 0, its path's end). A search
  that found no TS joining the ends is shown as its path top ("no TS found"), with the path in 3D and as
  `path_*.xyz`, never as a TS; with Sample more paths added, its off-target saddles were taken for the route.
  An edge added from a search's page (Add to graph) now reports that IRC or path, and every TS search channel
  carries the path it searched. The export is also on a TS search's or channel search's result page, next to
  All files (.zip); a search on no edge exports on its own.
- **An interactive reactor outlived its server**: a web server that was killed (SIGKILL, `fuser -k`) left its
  reactors' GFN2-xTB dynamics running at a full CPU each, with no one to stop them, for days. A reactor now
  exits with its server, as the result-parsing workers already did.
- **Reactor reaction ends were read as ions**: "H + OH" was stored and drawn as `[H+].[OH-]`, because the complex
  was read as one geometry. A complex now takes its molecules' SMILES. **A lone H atom was read as H2** (`[HH]`);
  one-atom structures are now perceived as a bare atom or ion. Existing workspaces are repaired when loaded.
- **Reactor story: a barrier search that found no transition state offered "Find its barrier" again**, as if
  it had never run. It now says "No transition state found" with a link to the search. "See them in
  Explore" moved from below every reaction card to the top, and it selects those reactions.

- **mepd failed to import with qcconst 0.3** (`KeyError: 'Hartree energy'` in `mepd.engines.engine`), which
  any install without uv.lock got: qcconst 0.3 renamed its CODATA keys and is deprecated (renamed qcref).
  mepd now depends on **qcref** (>=0.3) instead of qcconst and reads the new key names; constants are unchanged.

- **A charged reaction's channels were classified as neutral when `--charge` was not given**: the TS and IRC
  files read back for classification took charge 0 and multiplicity 1 instead of the endpoints' own.

- **Reaction channels no longer crash when a path search ended without energies** (e.g. g-xTB's SCF failed on
  it): that leaf is skipped with a note, and the other leaves are still analyzed. It cost NEB its result on one
  benchmark reaction.
- **Geodesic interpolation ran on Bohr coordinates with Angstrom parameters** (since the first mepd import).
  Its covalent radii, 3 A pair cutoff, 1.8 A bonded threshold and nudges are Angstrom, but `run_geodesic` passed
  mepd's Bohr coordinates. Every bond read as 1.9x stretched (C-H w = 0.2 instead of ~1), the pair cutoff acted as
  1.6 A, and about half the atom pairs (those defining bond angles) were missing, so some motions cost nothing in
  the metric. `run_geodesic` now converts to Angstrom and back. On 30 benchmark reactions the 12-image
  geodesic's highest g-xTB energy is lower in 23 (median -22 kcal/mol; two old paths reached 2400 and 4000
  kcal/mol). This changes every geodesic path (NEB/FNEB/MSMEP starting paths, GSM seeding): results from
  earlier versions that used geodesic interpolation should be regenerated.
- Retrosynthesis with building blocks "Automatic" plus your own file failed (`--stock auto` was not a stock);
  `mepd retro plan --stock auto` now means the default stocks.
- Help text: explicit solvent's point-charge embedding works with xTB, Psi4 or TeraChem (not only Psi4);
  `mepd discovery vri-surface` described its axes wrongly; a comment in the CG optimizer named the wrong
  formula (it is Polak–Ribière+).
- Analyze › New reaction could add the same reaction twice.

- **Kinetics started from a whole complex as if it were one species** (the default starting mixture, taken from
  the latest nanoreactor run). A complex there now starts as its molecules, 1 M each (a molecule twice in it: 2 M).

- **Molecules packed into one geometry could come out bonded to each other** (`mepd.complexes.packed`: a
  reaction SMILES with several molecules, packed complexes). Packmol keeps them 2 Å apart, which is still bonding
  range for e.g. a bromide and a carbon; such a packing is now redone further apart. Seen only where Packmol is
  installed, as in the container image, where it failed the network-variants tests.
- **Network variants: a new group could land within bonding range of a neighbouring molecule** (a methyl 1.5 Å
  from CO2) and the frame then read as one molecule. A group is now turned about its bond whenever a new atom
  is within bonding range of an existing one, not only when closer than 1 Å.
- **The public demo showed no Interactive reactor.** The server has allowed it since the per-visitor limits
  (two at once, one per visitor), but Explore still hid its buttons, and the top bar's way back to it, in demo
  mode.
- A solvated engine (g-xTB + ALPB correction) ran one geometry optimization per core at once, ignoring the
  profile's **Calls at once** (`n_parallel`): a channels run minimizing its conformer pools started 24
  optimizations, each with a g-xTB and two xtb processes.
- Design lost the charge of a bare ion (Br⁻ loaded as a neutral Br atom), so swapping its element was refused
  as leaving an unpaired electron.
- `channels` reported one TS as two identical channels when its half-formed bonds confused snap-RMSD
  (Cl⁻ + CH₃Br): matching interatomic distances now count as the same TS, and results already on disk
  merge such duplicates when shown.
- **NEB stopped on a barrier that was still falling.** The TS-region stop (`ts_converged_stop`) checked only
  that the barrier moved less than `barrier_thre` since the last step; a band still relaxing drifts by less
  than that (0.07 kcal/mol a step on a KAIST pair) and stopped 25 kcal/mol high (114 vs 90 kcal/mol). The
  barrier must now also have moved under `ts_converged_window_thre` (0.5 kcal/mol) over the last
  `ts_converged_window` (50) steps.
- **Kinetics: "the kinetics did not integrate".** Runs failed or crawled at the top of the default
  250–650 K sweep, and some at room temperature (the CO2 + epoxide network at 298 K for 2 h). Causes: the
  per-step extents' rounding noise steered BDF's step size, and amounts were clipped at zero inside the rates.
  Now the extents don't steer the step, the rates use the raw amounts, and the solver is Radau, which finished
  400 of 400 random mixtures, temperatures and times on that network where BDF failed 12. A year-long run no
  longer ends a rounding error past its span.
- **IRC through Sella spun the molecule at the minimum.** Sella kept rotations in its Hessian, and noise made
  one negative, so the IRC stepped along it to its step limit (~720 frames of the whole system rotating). The
  curvature is now judged on vibrations only, and frames are aligned onto the previous one. The same g-xTB
  IRC takes 363 frames, not 1262.
- **`mepd retro plan --method aizynthfinder` found almost no routes.** mepd rewrote AiZynthFinder's
  `config.yml` with keys sorted alphabetically, which put the ring-breaking-only `ringbreaker` expansion
  policy first. `aizynthcli` used that policy, so most targets got no expansions; paracetamol, for example,
  gave 0 routes. mepd now keeps the config's order and passes `--policy uspto` (`mepd/retro/aizynth.py`).
  Paracetamol now gives 23 solved routes, led by acetic anhydride + 4-aminophenol.

### Known issues

- **A new AiZynthFinder environment is broken.** `mepd retro setup --aizynthfinder` now installs
  pyarrow 26 (released 2026-10), which needs NumPy 2. AiZynthFinder's RDKit needs NumPy 1, so it fails to
  import. The image pins a working set; installs made outside the image are not yet pinned.
- **The image's GSM is upstream molecularGSM, not mepd's patched build.** `early_stop_on_minima` is
  unavailable: mepd warns and runs GSM to convergence.
- **GSM crashes on linear molecules.** HCN → HNC aborts with heap corruption (`munmap_chunk(): invalid
  pointer`) in GSM's exact-TS search, while nonlinear cases such as keto–enol converge. This is likely
  upstream GSM, but it has not been compared against a known-good build.
- **`tests/test_sandbox.py::test_reaction_events_are_found_as_the_nanoreactor_finds_them` is flaky.** It
  fails about 1 run in 3–4, both in and outside the image.
- **`tests/test_qmmm_web.py::test_adding_a_qmmm_result_to_explore_reads_only_its_qm_region` is flaky** under
  parallel runs (`-n 8`). It passed 5 of 5 times alone in the image.

