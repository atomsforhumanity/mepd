# Changelog

All notable changes to mepd are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Add each change under `Unreleased` as you make it, in the group it belongs to
(Added, Changed, Deprecated, Removed, Fixed, Security). On a release, rename
`Unreleased` to the version and date, and start a new empty `Unreleased`.

## [Unreleased]

### Added

- **Container image of the full web app** (`Dockerfile`, `.dockerignore`,
  `deploy/docker/`). `docker build -t mepd .` builds one image with:
  - mepd and every `web` extra, from `uv.lock`;
  - g-xTB 2.0.1 (release build, SHA-256 checked);
  - xtb 6.7.1, CREST 3.0.2, Packmol 20.15.3 and Psi4 1.11 from conda-forge, via micromamba;
  - molecularGSM built from source with its ASE backend, against MKL;
  - the retrosynthesis data and an AiZynthFinder 4.4.1 environment;
  - AIMNet2, ANI-2x and ReactionT5 weights;
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

### Changed

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

