# Changelog

All notable changes to mepd are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Add each change under `Unreleased` as you make it, in the group it belongs to
(Added, Changed, Deprecated, Removed, Fixed, Security). On a release, rename
`Unreleased` to the version and date, and start a new empty `Unreleased`.

## [Unreleased]

### Added

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
  mode on afh-c04 as a single replica that is never rolled over (`Recreate`), with the GPU. State and scratch
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

