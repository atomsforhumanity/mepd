# Changelog

All notable changes to mepd are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Add each change under `Unreleased` as you make it, in the group it belongs to
(Added, Changed, Deprecated, Removed, Fixed, Security). On a release, rename
`Unreleased` to the version and date, and start a new empty `Unreleased`.

## [Unreleased]

### Added

- **Comparing species without metal-ligand contacts** (`metal_ligand_contacts=False` on `compare_structures`,
  `same_molecule` and `connects`). Bonds to transition metals, lanthanides and actinides are left out of the
  bond graphs and the stereochemistry, so a change of coordination alone (a ligand binding through another
  atom, or coming off the metal) is not another species; `Comparison.metal_ligand_changed` records it. Off by
  default: every command still compares all bonds. Following the treatment of dative metal-ligand contacts as
  zero-order bonds, kept apart from covalent bond changes, in Z. Li, H.-H. Hsu, B. M. Savoie, "Finding the
  Roads Less Calculated in Organometallics with Automated Transition State Searches", ChemRxiv (2026),
  doi:10.26434/chemrxiv.15006374/v1 (preprint). In the Transition1x-TMC screen, a quarter of the reaction
  ends whose bond graphs differed from the dataset's differed only in metal-ligand contacts.

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

- **Deleting molecules says which complexes go with them.** Explore draws a complex as its molecules, so a
  complex is removed with any of them (as before). The delete confirmation (Explore selection and the
  Inspector) now names those complexes. (#14)

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

- **An uploaded ion pair splits into ions, not two radicals** (`mepd/web/compose.py`). Without charges, the
  molecules of a split complex were made neutral where they could be, so pyridinium trifluoroacetate came out
  as two radicals. When a closed-shell structure splits into radicals, GFN2-xTB partial charges (when xtb is
  installed) now decide each molecule's charge, as in the nanoreactor. (#14)

- **One molecule with a metal centre no longer compares as two stereoisomers** (`mepd.compare_structures`,
  `same_molecule`, `connects`). RDKit's stereo tag on a metal (@SP, @TB, @OH) records which coordination
  polytope it fitted to the geometry, and for a fluxional centre that changed between two geometries of one
  molecule (IrCl4L as @TB6 in one, @OH7 in the other). The tag is now dropped when all of the centre's
  ligands are alike but at most one (MA_n, MA_nB), which have no stereoisomers; cis/trans and fac/mer
  centres keep it. In a 200-reaction Transition1x-TMC screen, 10 more reactions were kept and none fewer.

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

