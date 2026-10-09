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
  - FAIR-Chem UMA `uma-s-1p2p1` (uma-s-1.2.1), in a layer of its own.

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
  live on the node's data disk (`/nvme1/mepd`), and an Ingress serves `mepd.atomsforhumanity.org`.
- **Demo limits are configurable from the environment.** Any number in `DemoPolicy` (`mepd/web/demo.py`) can
  be overridden as `MEPD_DEMO_<FIELD>`, e.g. `MEPD_DEMO_GLOBAL_CONCURRENCY=64` to run 64 jobs at once across
  all testers instead of 4. Invalid values stop `mepd web --demo` at start-up with a message.
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
- **`uv.lock` is gitignored**, so builds from a fresh clone (e.g. CI) resolve dependencies afresh. Commit it
  for reproducible images.
