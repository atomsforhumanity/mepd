# mepd web

A browser interface over the mepd CLI. It is meant for three kinds of use:

- **One-off questions.** For example, "the TS between these two", with endpoints given as SMILES or XYZ.
- **Exploration** around a structure.
- **Hand-built reaction networks.** Draw the network, then run any calculation on some or all of its edges.

The web UI is an optional part of the package. Its `web` extra installs everything its features
use (ASE/Sella, tblite, SLAPMapper, the MLIPs, retrosynthesis), and the page itself ships inside
the package. g-xTB, the default level of a new workspace, is not a Python package: if it is not
installed, mepd downloads the official release (about 40 MB) the first time it is needed.

```bash
uv sync                      # dev checkout (installs everything, web app included)
pip install "mepd[web] @ git+https://github.com/mtzgroup/mepd.git"   # or from git
mepd web my_workspace        # opens http://127.0.0.1:8765
mepd web my_workspace --max-jobs 4 --port 9000 --no-open
```

On a remote machine, run it there and use an SSH tunnel:
`ssh -L 8765:localhost:8765 cluster`. There is no authentication. `--host 0.0.0.0` exposes job
submission and file reading to anyone who can reach the port. The page's JavaScript libraries are
vendored under `mepd/web/static/vendor/`, so it works without internet access.

## The Workbench

The page is one screen:

| Part | What it holds |
|---|---|
| **Top bar** | `mepd` (back to Home), the workspace (switch sessions), **Learn** (manual, references, your tools), the gear (**Settings**: level of theory, profiles, All tools, theme). |
| **Home** (`#/home`) | The landing. Pick a molecule (four presets, *Draw one*, or a typed name or SMILES) and **Grow its network**: a network expansion at 1000 K, 1 min simulated (hot, so that something forms), opened in its live network view. Beside it, a finished network as mepd shows it (glycine, replayed from `mepd/web/showcase/glycine-live`), and the finished networks and studies of Examples (`mepd/web/showcase/`). *Open the workbench* turns on every tool and opens Explore; the *All tools* switch is there too. |
| **Structure list** (left) | Molecules and reactions, one status mark each (● checked minimum, ◌ not optimized, ‡ TS, ✕ failed; the words are the tooltip), *+ Add* (SMILES, xyz, QM/MM system) and *Draw*. |
| **Canvas** (centre) | Explore (the network map, `NetworkMap.js`: one arrow per step with small molecules as tags, molecules in columns by the easiest way in from the start, a barrier histogram that filters, a molecule's neighbourhood and its easiest route; *Graph* switches to the editable graph, `Graph.js`: connect, arrange, conditions, variants, and after a kinetics run nodes and edges sized by the material that passes; manual page `explore`), Reactions, Kinetics, Landscape (the routes that carry material as one energy diagram), Add a molecule (what one more molecule in the flask does to a product you want more of: [below](#add-a-molecule)), a reaction's page, a calculation's page. |
| **Inspector** (right) | **Tools** (what you can run on the selection; each card reads as one sentence, settings folded), **Results** (what is known), **Method** (level, pipeline, settings vs defaults, the command that reproduces it). Nothing selected: the workspace's counts and its fastest step. |
| **Calculations bar** (bottom) | What is running, the level of theory, units; click it for the full list. |

**Design and the Reactor are modes** over the canvas, each with one way out (Done / Exit). Design's Move tool drags atoms into place by hand.

**Addresses:** every page has one: `#/explore`, `#/reactions`, `#/kinetics`, `#/landscape`, `#/add-molecule`, `#/reaction/<edge>`, `#/calculations`, `#/calc/<job>[/<tab>]`, `#/design`, `#/reactor/<id>`, `#/home`. Add `?d=settings` or `?d=learn` for a drawer. Back and forward walk them, a reload stays, and a link opens the same page. **Esc** goes up one level: the selection, then a drawer, then the page.

**The live network** (`#/network/<job>`, or an example's `#/network/example/<slug>`) replays an explorer run: species as structures sized by the material they held, steps coloured by barrier and as wide as their flux, corrections as they happen, and what forms.

**Reaction pages** (double-click a reaction) play its path: 3D, energy profile and the bonds that form and break, on one scrubber (Space plays, ←/→ step).

## Mental model

| Concept | What it is |
|---|---|
| **Workspace** | A directory. It holds a structure library, a reaction graph, compute profiles and job folders. Everything is plain files; copy or archive the directory to share a project. |
| **Structure** | A library entry and, equally, a graph node. It is added from SMILES (one per line, `SMILES name`), pasted XYZ (multi-frame means several structures), dropped `.xyz`/`.smi` files, or imported from a result. |
| **Edge** | A pair of structures declared as connected, with a direction (start → end). It is drawn with **Connect** (or the `C` key), made by *Connect as edge* on two selected structures, created implicitly by running a pair calculation, or imported from a result. Atom counts must match. |
| **Operation** | A mepd command the UI can run: TS / MEP (`mepd run`), reaction channels (`mepd channels`), TS from a guess (`mepd ts`), reaction network expansion (`mepd discovery expand`, `hessian-sample`, `hessian-global`: one card: the expansion's form in four stages -- Grow, Connect, Steer, Run -- beside Hessian sampling and Basin hopping), and all-pairs network (`mepd network-splits`). |
| **Profile** | A `RunInputs` TOML (the `--inputs` file): engine, level of theory, path minimizer and thresholds. It is edited and validated in the Settings drawer; *Validate* actually builds the engine in a scratch process. |
| **Job** | One `mepd …` subprocess with its own folder. |
| **Level of theory** | Each workspace has one: a chosen profile (Structures panel → *Level of theory*). Its fingerprint is the engine, program, method and basis settings. Every structure records the level its geometry and energy belong to. |
| **Known minimum** | A structure that is a minimum at a stated level: optimized on entry, or imported from a minima or conformer result of a job. A pair job skips endpoint minimization only when both endpoints are minima **at that job's level**; anything else (force-field embeddings, raw XYZ, structures from another level or an external output) is re-minimized first. |
| **TS structure** | A saddle point imported from a result (role `ts`). It is never minimized, because that would destroy it. |

### One level of theory per workspace

- **Structures added from SMILES or XYZ are minimized at the workspace level** with `mepd optimize`, one job per charge/multiplicity group (the add box has a checkbox to skip this). A SMILES embedding is only an RDKit/MMFF94 force-field geometry.
- **Status is shown on the structure.** It appears straight away as *optimizing…*, and its geometry, SMILES and energy are replaced in place when the job finishes. Its chip shows the level (e.g. `gxtb`), or *not optimized*, *other level*, *opt failed* or *not a minimum*.
- **Hessian check (on by default).** *Verify minima (Hessian)* makes every optimized structure pass a Hessian check: no frequency below the cutoff (default −20 cm⁻¹, so soft torsions and rotors pass).
  - A structure that stopped on a saddle point is pushed along its unstable mode by the rescue push (0.3 bohr), both ways, and re-optimized. Larger pushes are not retried: on the KAIST set they mostly fell back onto the same saddle.
  - If both pushes fail, it is kept but flagged *not a minimum* and never trusted as one.
  - Why 0.3 bohr: eclipsed ethane (−334 cm⁻¹) falls back onto its saddle from 0.1 bohr; 0.3 reaches staggered.
  - The same rescue is used by mepd everywhere minima are Hessian-checked: recursive path searches, `mepd optimize`, Hessian sampling and basin hopping.
- **Changing the level** marks structures from the old level as off-level. *Re-optimize N off-level* redoes them, skipping TS structures.
- **Jobs check levels too.** A job run with a profile at a different level shows a warning, and pair jobs re-minimize off-level endpoints themselves. Duplicate detection on import never compares energies from different levels.
- **Path searches (TS, channels, all-pairs network).** *Validate minima with Hessian* is a basic option: every intermediate minimum a recursive split proposes must pass the same check. Its cutoff and push size are under Advanced.
- **Hessian sampling and basin hopping.** The same option (on by default in the web) checks every minimum the sampler reports.
  - Rescued minima replace the originals and are de-duplicated again.
  - Structures that are still saddles are dropped from the minima and listed under *Rejected: not minima* (`rejected.xyz`).
  - Basin hopping only accepts, and seeds later rounds from, validated minima.
  - Imported minima carry their check, and only ones that passed count as known minima.
  - On the CLI: `mepd discovery hessian-sample|hessian-global --validate-minima-with-hessian`. It is off by default there, so existing scripts are unchanged.

### Selection drives what you can run

The right-hand panel lists exactly the operations that fit the current selection:

| Selection | Offered |
|---|---|
| One structure | Explore around it: TS optimization from a guess, Hessian sampling, basin hopping |
| Two structures (the first clicked is the start) or one edge | TS / MEP, reaction channels |
| Several edges | The same pair calculation, **one job per edge** (batch) |
| Several structures | Batch exploration, or the all-pairs network |

Each operation's form is generated from its parameter model:
- Basic knobs are shown directly, with the rest under *Advanced*.
- A knob that only applies when another option is on (for example, CREST settings when the backend is RDKit) is hidden until that option is on.
- *Show command* prints the exact `mepd …` command the job will run.

The last-used parameters and profile are remembered in the browser, per operation.

### Results flow back into the graph

A job's **Results** tab shows everything the command produced, in groups. For a channels run, for
example, the groups are direct channels, multi-step channels, off-target exits, conformer pools and
every TS search.

Each entry has:
- a 3D viewer;
- a frame slider;
- an energy profile (click or drag it to scrub).

From any entry you can:
- **Add ends + edge to graph.** The two path ends become structures and are joined by an edge that remembers the result and its barrier. A structure already in the library is reused (same connectivity and energy within 0.05 kcal/mol).
- **Add TS** or **Add this frame.** The structure becomes a node you can explore from.

Edges are coloured by their state: not computed, queued, running, done or failed. A done edge is
labelled with the best barrier any job or import found for it.

**Barriers** are referenced to the lowest reactant-side energy found anywhere in that result:
- for channels, reactant conformers and reactant-side IRC ends;
- for `run`, the path start and IRC ends of the same connectivity.

They are never referenced to the search's own starting geometry.

**Log warnings.** Problems mepd only reports on stdout (for example, an IRC that failed while the TS
was kept, or an NEB that never converged) are shown above the results.

### Transition states: one card

*Transition state* has two methods: **These endpoints** (one path search between the two structures, `mepd run`), or **Sample conformers and mappings** (reaction channels, `mepd channels`). Quick start has the same choice as a checkbox. A finished TS search offers **Sample more paths**: a reaction-channels run on the same two structures, conformers and level of theory; the edge keeps the lowest barrier found by either.

### Reaction SMILES

`reactants>>products` (or `reactants>agents>products`; agents are dropped) gives both ends of a reaction with their atoms matched (`mepd/reaction_smiles.py`). Map numbers in the SMILES are used when every heavy atom has one. Otherwise SLAPMapper maps it, and hydrogens (and ties between mappings) go to the smallest endpoint RMSD. That is fast, not exhaustive; a path search with atom mapping on still weighs the other mappings.

- **Explore › + Add, and Quick start:** adds both ends and the edge between them, pinned to the conformers built for this reaction.
- **Design:** opens the reactant and the product side by side. With *Edit both* on (the default), an edit on one is made on the matching atom of the other, e.g. to swap a substituent on both ends and compare barriers. *Search TS* / *Reaction channels* add both ends and their edge, then run the search. `mepd/web/design_reaction.py` carries the atom match through edits: atoms an edit does not touch keep it; new atoms are matched by element, bonded neighbours and the closest fit in 3D.
- **CLI:** `mepd run --reaction '...'` and `mepd channels --reaction '...'`.

### Design

Build or edit a structure in 3D: from scratch (click empty space to drop an atom, with its hydrogens; click a hydrogen to grow a new atom in its place), from SMILES, a reaction SMILES (above), an XYZ file (upload, drop or paste, with its charge), or a graph node. *Add molecule* places a preset, any SMILES, or an uploaded XYZ (kept rigid, as in the file) next to the atom you click.

A new, empty workspace opens on Home; an empty Explore points there too.

### QM/MM systems

A molecule in explicit solvent (*Put in explicit solvent (QM/MM)*), or a whole system added with *Add › QM/MM system*, becomes one node named by its QM region ("NCC(=O)O · in water"). A structure with exactly that system's atoms joins it automatically. Calculations on it run embedded, at their own level ("… / QM/MM"), and only the operations that work embedded are offered. Every 3D view draws such a structure by region and can colour the environment by motion. Each result path gets *QM/MM checks* (frozen drift, MM bond changes, cut bonds, contacts) and an energy split. See [qmmm.md](qmmm.md).

### Add a molecule

*Add a molecule* (the canvas tab beside Kinetics and Landscape; or select a species in Explore › *Make more of it*)
is the page over `mepd addmol` ([add-molecule.md](add-molecule.md); manual page `add-molecule`):

1. **Target and conditions** → *What limits it today?* builds the query file. Where every verified step of the
   workspace belongs to one finished Network expansion (`addpage.source_run`), the query is the exploration's own
   molecule-level network (`mepd addmol query RUN`, run in a process of its own; measured on a 274-step run with 2
   processes: 33 s at the run's temperature, 410 s at another, where every thermal correction is recomputed); otherwise
   the workspace's (`mepd.web.added.query`; the first time it computes the GFN2-xTB Hessians of every species and
   TS, then seconds). The page says which. It shows the
   degree-of-rate-control order (lower these, do not lower these, traps), or, where the network makes none of the
   target, the easiest route with each step's height against what the conditions allow.
2. **Candidates**: SMILES lines (`SMILES name amount`), a candidates file (JSON), an `.xyz` file, or the molecule
   in Design; several at once. *Test* queues one hidden operation, `addmol-test` (`mepd addmol test`), for all of
   them. *Download the query file* gives a model what it needs.
3. **A test as it runs.** `mepd addmol test` runs quick estimate → verification → scan (the form's *Scans*: how
   many scans for the molecule taking part; the endpoint also takes `steps` and `verify`). The detail leads with
   the answer (route peak without and with, the target's amount with its spread interval, the molecule's fate,
   what was not scanned and why, and *Not a verdict yet* while anything is open), then three stage boxes
   (`addpage.progress`), the step table (rows keyed and in the order of work while it runs; an estimate hollow and
   dashed, verified rows *beside* or *takes part*), and the scans (`scan.steps[]`; an entry may name several
   steps; several may run at once, and so may the stages). A verified TS is shown from its `ts_file` / `irc_file`
   (relative to the result file, or taken out of `files.tgz` beside it); where they are gone the page says so. The
   route peak's history (`route_peak`) is drawn as a step curve. A finished test is replayed by its rows' `t_s`:
   `GET …/results/{file}?t=<seconds>` is the result as it stood then (`addpage.replay`: the rows in by then, the
   kinetics run again on them, each scan as far as its arrivals had come). The estimate's uncertainty is read from
   the quick-estimate layer's `uncertainty`; *led to other species* rows show the `ends` of each pose.
4. **Results** arrive without a reload: the job rewrites one result file per candidate as layers finish, and the
   page reads them every 3 s while it runs (and on the job's events). One line per candidate, ranked by the gain of
   the target (target, how soon, returned or not and where it rests, the largest step change with its spread),
   every number tagged `est.` or `verified`; the detail has the barrier of each step without and with the molecule
   in two groups (through the molecule; only because amounts moved), what is new, the layers, what failed and why,
   and the TS in 3D. Ticked candidates are compared step by step.
5. **Show on the map** keeps a candidate as a variant of kind `added_molecule` (`mepd.web.added.put`, with what
   the map marks under `marks`); the network map colours the steps that go through the molecule by the change of
   their barrier (dashed: an estimate), doubles the border of molecules it binds and fades the rest. The record follows its result
   file when the job finishes.

Files, all under `<workspace>/addmol/<query id>/`: `query.json`, `results/<name>.json`. A result file from a
command-line run copied into `results/` is shown like any other. Code: `mepd/web/addpage.py`,
`static/js/components/AddMolecule.js`, `AddMoleculeMarks.js`, `static/addmol.css`.

Where the variants flow is reused, and where not: the record sits beside the substituent variants and Explore's
Network picker lists and deletes it; *Update* is not offered for it (nothing to carry), and Kinetics' *Network*
menu does not list it (its kinetics, on `mepd.addmol`, are on the page; `mepd.web.kinetics.analyze` does not run
an added-molecule variant).

### Sessions, cleanup, downloads

- **Sessions.** A session is a workspace directory.
  - *New session* starts over in a fresh, empty workspace and copies your compute profiles into it. A bare name creates the directory next to the current workspace; a full path puts it anywhere.
  - *Open…* lists recent sessions, or opens any workspace directory by path.
  - Switching never stops anything: the other session's running jobs keep running, and they are there when you switch back.
  - The recent list lives in `~/.config/mepd/web_sessions.json`.
- **Removing elements.** Select several structures and/or edges (shift/⌘-click, shift-drag, or *Select all*), then press *Delete* in the graph toolbar or the Delete key. Removing a structure removes its edges; calculation outputs are never deleted this way.
- **Downloads.**
  - On a result: a TS table (CSV: group, label, barrier, TS energy, what it connects), all TS geometries as one multi-frame XYZ, and the whole output folder with inputs and logs as a zip.
  - On any entry: the path, the TS or the current frame as XYZ. Comment lines carry the label, frame and energies (Eh and relative kcal/mol).
  - Selected structures: one multi-frame XYZ.

### Jobs

- **Queue.** At most `--max-jobs` jobs run at once. The rest wait, first in, first out.
- **Live tab.** Shows the latest status line and the path's energy profile as it optimizes, plus per-branch mini-plots for MSMEP. For `channels` it also shows the stats.json counters.
- **Optimization tree tab** (`ts`, `channels`, `network-splits`). Shows every MSMEP path search as a tree: why each node split (the criterion) or stopped (elementary, unresolved and why, failed with its error and traceback, skipped). Pick a node to see its optimization step by step: all steps' energy profiles overlaid, a step scrubber with play, the barrier, how far the path moved and the gradient at the highest image per step, and the 3D geometry of any image.
  - While a search runs, MSMEP records each node as it starts and finishes (`mepd/tree_log.py`, into `<job>/live/trees/<stream>/`). So a running search, or one that crashed or was cancelled before writing its tree, can be inspected too; the node that never finished is marked *interrupted*.
  - Finished trees also write `tree.json` next to `adj_matrix.txt`. Older runs without it get outcomes inferred from their files.
- **Cancel** kills the job's whole process group, which includes `--workers` children and CREST.
- **Resume / Rerun** runs the same command into the same output folder. `channels`, `ts` and `network-splits` skip what is already on disk.
- **Server restart.** Queued jobs stay queued. Running jobs are marked *interrupted* and can be resumed.
- **Open existing output…** registers a mepd output directory from anywhere on disk as a read-only job. Past CLI runs can then be browsed and imported into the graph. The directory is never modified. Its level of theory is the workspace profile picked under *Computed at*, else the profile `.toml` kept next to the output (in its folder or the one above). Its structures join Explore at that level, and a result added with *Add ends + edge* is listed on that edge, together with its barrier. With no level, its energies are not compared with any others.

## Architecture

```
browser (Preact + htm, Cytoscape, 3Dmol; no build step)
   │  REST: mutations          SSE /api/events: workspace / job / progress
   ▼
FastAPI app (mepd/web/app.py)
   ├── Workspace (workspace.py)   workspace.json + structures/*.xyz + profiles/*.toml
   ├── JobManager (jobs.py)       queue → `python -m mepd.cli <argv>` per job, own process group
   │                              progress from stdout, MEPD_DRIVE_PROGRESS_LOG,
   │                              MEPD_DRIVE_CHAIN_JSON, stats.json (polled 1 s)
   ├── OPERATIONS (operations.py) params model + argv builder per operation
   └── collectors (results.py)    output dir → {headline, barrier, groups[entries[frames]]},
                                  run in a spawned worker process (never on the server's threads)
```

Jobs are subprocesses rather than in-process calls, for four reasons:
- mepd's helpers raise `typer.Exit`, fork process pools, and set process-global state when `RunInputs` is built.
- A crash or hung QM call stays contained in its job.
- Killing the process group cancels everything the job started.
- Rerunning the CLI into the same output folder resumes it.

A job's folder:

```
jobs/<id>/job.json     record: argv, status, params, profile, targets, summary
jobs/<id>/inputs/      snapshot of the structures and profile the job used
jobs/<id>/output/      the command's --output directory
jobs/<id>/stdout.log   progress.log   chain.json   result.json (cache)
```

Every job snapshots its inputs, so editing or deleting a library structure or profile later never
changes what a past job ran with.

## Adding an operation

1. In `mepd/web/operations.py`, write a `Params` model. Each knob is a `P(default, title, help, cli="--flag", kind=..., group=..., advanced=..., requires=...)`. `kind` is one of:
   - `value`: `--flag v`
   - `switch`: `--flag` when true
   - `toggle`: `--flag` / `--no-flag`
   - `custom`: handled by the builder
2. Add an `Operation(key, title, summary, target, category, Params, build)`. `target` is `structure`, `pair` or `set`. `build(ctx, params)` returns the argv after `mepd`; `JobContext` has helpers for endpoints, charge/multiplicity and the profile.
3. If the command writes a new output layout, add a collector to `COLLECTORS` in `mepd/web/results.py`, and a rule to `detect_operation` so its output folders can be opened. Bump `RESULT_VERSION` when collectors change.

The form, the action panel, batching, the queue, progress and the log need no other change. For
example, a future `mepd discovery nanoreactor` command becomes one `Params` model plus one
`Operation` with `target="structure"`.

## HTTP API (for scripts)

| Method and path | Purpose |
|---|---|
| `GET /api/state` | Workspace, jobs, operations (with JSON schemas), profiles |
| `GET /api/events` | SSE: `workspace`, `job`, `job_deleted`, `progress`, `profiles` |
| `POST /api/structures` `{text, name?, charge?, multiplicity?}` | Add from SMILES or XYZ text |
| `POST /api/structures/upload` (multipart `files`) | Add from files |
| `PATCH` / `DELETE /api/structures/{id}` | Rename, set charge/multiplicity, delete (and its edges) |
| `POST /api/edges` `{source, target}` | Add an edge. `PATCH` takes `{label, reverse}`; `DELETE` removes it |
| `POST /api/jobs` `{op, structures[], edges[], params, profile, dry_run?}` | Submit. One job per edge or structure for batchable ops. `dry_run` returns the commands only |
| `POST /api/jobs/import` `{path, op?, charge, multiplicity}` | Open an existing output directory |
| `GET /api/jobs/{id}/result` | Parsed result (cached once finished) |
| `POST /api/jobs/{id}/import-entry` `{entry, frames: endpoints\|ts\|one\|all, frame?, connect}` | Result → library/graph |
| `POST /api/jobs/{id}/cancel` \| `retry` | Queue control. `DELETE /api/jobs/{id}` removes the job |
| `POST /api/jobs/{id}/stop-analyze` | Stop & analyze (Find TS, Reaction channels, Sample more paths): the path searches end now with the paths they have, then TS optimization, IRC and the channels run on them; (Nanoreactor, Run longer): the MD ends now and its trajectory so far is analyzed and refined (`mepd.search_control`) |
| `GET /api/jobs/{id}/log?which=stdout\|progress&offset=` | Incremental log (`X-Log-Size` header) |
| `GET/PUT/DELETE /api/profiles/{name}`, `POST /api/profiles/validate` | Profiles |
| `POST /api/delete` `{structures[], edges[]}` | Bulk removal (one save, one event) |
| `GET /api/structures-export?ids=a,b` | Structures as a multi-frame xyz |
| `GET /api/jobs/{id}/archive` | Output folder, inputs and logs as a zip |
| `GET /api/reactions/{id}/export?format=html\|zip` | One reaction's page (self-contained HTML) or its data |
| `GET /api/workspace/export?format=html\|csv` | Every reaction Analyze lists: one browsable page, or one CSV row per elementary step (`mepd/web/workspace_export.py`, `CSV_COLUMNS`) |
| `GET /api/addmol`, `POST /api/addmol/query` `{target, initial, held?, temperature, time_s}` | Add a molecule: the questions asked; ask one (builds or reopens its query file) |
| `GET /api/addmol/{q}`, `GET /api/addmol/{q}/query.json` | A question: what limits the target, one line per candidate; the query file |
| `POST /api/addmol/{q}/test` `{text, filename?, amount?, scan?, dry_run?, workers?, steps?, verify?}` | Candidates (SMILES lines, a candidates file or an xyz file as text): read them (`dry_run`), or start one job that tests them |
| `GET` / `DELETE /api/addmol/{q}/results/{file}`, `POST …/keep` | One candidate's detail; remove its result; keep it as a variant for the map |
| `GET /api/addmol/{q}/geometry?step=&result=&through=` | A step's ends and TS as xyz frames, with the molecule where the estimate kept it |
| `GET /api/sessions`, `POST /api/sessions/new` `{name \| path}`, `POST /api/sessions/open` `{path}` | Sessions |

## Performance notes

- **The path search dominates wall time.** The web layer adds milliseconds per action and about 2 s of process start-up per job. Progress reporting costs about 3% of an NEB step.
- **The path method matters most.** On a 14-atom test pair with g-xTB (a sampled allyl vinyl ether minimum to pent-4-enal, run with `--recursive --use-tsopt --irc`):

  | Profile | Wall time |
  |---|---|
  | Default NEB | 308 s (~1,000 steps) |
  | FNEB | 37 s |
  | GSM | 53 s (but it found a different product) |

- **Endpoints should be relaxed.** With one endpoint left unrelaxed, the same NEB search found a spurious intermediate and was still running at 400+ s.
- **Result parsing reads only final chains**, never NEB histories, and runs in a worker process, so the UI stays responsive while a large output is read. After the first read, results come from the cache.

## Limits worth knowing

- Nanoreactor and graph enumeration are not available until mepd has commands for them.
- Barrier floors are per result. Comparing results from different jobs (for example, RDKit vs CREST channels) still needs one shared floor across them, which the UI does not compute.
- Result parsing loads every chain in an output folder. Large `channels` runs take a few seconds the first time; after that they come from the cache.
- Profiles hold engine settings. Endpoint charge and multiplicity come from the structures, and both endpoints of a pair must agree.
