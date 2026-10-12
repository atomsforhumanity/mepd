# mepd in a container

One image with the whole web app and everything it runs, for hosting on a
server (and, next, building in CI and deploying to Kubernetes).

```bash
read -rs HF_TOKEN && export HF_TOKEN                    # a Hugging Face token with UMA access (see below)
docker build --secret id=hf_token,env=HF_TOKEN -t mepd . # from the repository root
docker run -d --name mepd -p 8765:8765 -v mepd-data:/data \
  -e MEPD_DEMO_PASSWORD='choose-one' mepd                 # open http://localhost:8765
```

Add `--gpus all` on a machine with NVIDIA GPUs and the NVIDIA container
toolkit. The machine-learned potentials then run on the GPU; without it they
run on the CPU. The same image serves both.

## What is in the image

| Piece | From | Where | How mepd finds it |
|---|---|---|---|
| mepd + every `web` extra (FastAPI, ASE/Sella, tblite, fairchem-core, AIMNet2, TorchANI, OpenMM, Vina, PDBFixer, onnxruntime, rdchiral, transformers, SLAPMapper, PyTorch with CUDA) | `uv sync --locked` from `uv.lock` | `/opt/mepd/venv` | on `PATH` |
| g-xTB 2.0.1 (also runs GFN1/GFN2-xTB) | GitHub release, SHA-256 checked | `/opt/gxtb` | `GXTB_EXECUTABLE` |
| xtb 6.7.1, CREST 3.0.2, Packmol 20.15.3 | conda-forge (micromamba) | `/opt/tools` | on `PATH` |
| Psi4 1.11 | conda-forge (micromamba), own Python | `/opt/psi4` | `MEPD_PSI4_PYTHON` |
| molecularGSM (ASE backend), MKL | built from source; MKL from conda-forge | `/opt/gsm` | `GSM_EXECUTABLE` |
| Retrosynthesis data (USPTO templates, policy/filter ONNX, PaRoutes stocks) and their prebuilt lookup indexes | our mirror, else Zenodo (SHA-256 checked) | `/opt/mepd/retro` | `MEPD_RETRO_DATA` |
| AiZynthFinder 4.4.1 in its own Python 3.11 env, its public models and ZINC stock | PyPI, its download script | `/opt/mepd/retro/aizynth-*` | `MEPD_RETRO_DATA` |
| AIMNet2 (rxn, general, nse, 2025), ANI-2x, ReactionT5v2 weights | their registries, Hugging Face | `/home/mepd/.cache`, `.local/share` | default caches |
| FAIR-Chem UMA `uma-s-1p2p1` (gated) | Hugging Face, with a token at build time | `/home/mepd/.cache/fairchem` | fairchem's cache |
| A C compiler | Ubuntu | `/usr/bin/cc` | Triton compiles a helper for UMA on the GPU |

The build finishes with `mepd-entrypoint check`, which fails the build if any
of these is missing or does not run. Run it any time with
`docker run --rm mepd check`.

Not included: Orb and MACE (they conflict with fairchem-core; see
`docs/mlip.md`), TeraChem/ChemCloud (remote services), and a local LLM
server for `mepd retro --method local-llm`.

**The upstream molecularGSM, not mepd's patched build.** mepd's GSM
early-stopping (`early_stop_on_minima`) needs a patched GSM that writes every
iteration's string. The image builds upstream molecularGSM at a pinned commit,
so mepd logs a warning and runs GSM to convergence instead. Everything else
about GSM works. Once the patch lives somewhere, point `GSM_REPO`/`GSM_COMMIT`
at it.

## Build options

```bash
docker build --secret id=hf_token,env=HF_TOKEN -t mepd .        # everything, UMA included
docker build --build-arg FAIRCHEM_MODELS= -t mepd .             # no UMA, so no token needed
docker build --build-arg PREFETCH_MODELS=0 ...                  # also skip AIMNet2/ANI/ReactionT5 weights
```

- **The gated FAIR-Chem models (`FAIRCHEM_MODELS`, default `uma-s-1p2p1`).**
  - **Token:** `HF_TOKEN` must belong to a Hugging Face account that has been granted access to
    [facebook/UMA](https://huggingface.co/facebook/UMA): `huggingface@atomsforhumanity.org`, with a
    fine-grained token that may only read gated repos.
  - **It is a build secret,** so it is never stored in a layer. Secrets are not part of BuildKit's
    cache key, so the model list is: a build that names models but has no token **fails** rather than
    silently leaving them out.
  - **Don't add `uma-m-1p1`.** It is a single 11 GB file, over GHCR's 10 GB limit per layer.
  - **Keep the image private.** It now contains Meta's licensed weights.
- **`uv.lock` pins the Python environment** (committed). The build uses
  `uv sync --locked`, so a `pyproject.toml` change without an updated lock
  fails the build: run `uv lock` and commit both.
- **Layer caching.** The downloads sit in stages that change rarely:
  - `retro`: the retrosynthesis data and AiZynthFinder (~2 GB, from Zenodo,
    which can be slow) depend only on the base image and
    `mepd/retro/{data,aizynth,stock}.py`.
  - `models`: the MLIP/ReactionT5 weights depend on the locked Python
    environment, but not on mepd's code.

  A change to `mepd/` rebuilds only the last layers. In CI, keep all stages'
  layers between runs (`cache-to: type=registry,ref=...,mode=max` with
  `docker/build-push-action`). `mode=max` is needed because these are
  intermediate stages. GitHub's `type=gha` cache is capped at 10 GB per repo,
  which this image exceeds.
- **Data mirror.** The retrosynthesis data (~800 MB, `retro-data.txt`) comes
  from our mirror, `DATA_MIRROR` (default
  `https://public.atomsforhumanity.org/mepd/retro`). Zenodo/figshare are the
  fallback for anything the mirror lacks. Every file is checked against the
  SHA-256 pinned in `retro-data.txt`, so the mirror is never trusted on its
  own. `--build-arg DATA_MIRROR=` skips it. See "The data mirror" below.
- Versions: `GSM_REPO`, `GSM_COMMIT`, `GXTB_URL`, `GXTB_SHA256`, and the
  base images (`UBUNTU_IMAGE`, `MICROMAMBA_IMAGE`, `UV_IMAGE`) are build
  args. The conda-forge versions are pinned in the Dockerfile.

## Running

| Variable | Default | |
|---|---|---|
| `MEPD_MODE` | `demo` | `demo`: many users with one shared password. Each browser gets its own workspace (`/data/demo/visitors/<id>`), and the limits in `mepd/web/demo.py` apply (40 atoms, 30 min per job, ...). Compute profiles are read-only (`/data/demo/profiles/*.toml`, editable on the volume). `full`: the unrestricted app over one workspace (`/data/workspace`). Anyone logged in can read files and run programs in the container. |
| `MEPD_DEMO_PASSWORD` | generated, printed in the log, kept in `/data/demo/.demo_password` | demo mode's shared password |
| `MEPD_WEB_PASSWORD` | a generated token, printed in the log | full mode's password |
| `MEPD_MAX_JOBS` | 4 | calculations at once |
| `MEPD_PORT` | 8765 | |
| `OMP_NUM_THREADS` | 1 | threads per calculation. Parallelism comes from running jobs side by side. |
| `HF_TOKEN` | | only for FAIR-Chem models that are not baked in (downloaded on first use) |

Everything the app writes is under the `/data` volume: workspaces, job
outputs, logins and session state (`/data/state`). The container runs as
uid 1000 (`mepd`). It needs `/tmp` to allow exec, because GSM runs a helper
script from its working directory.

Other commands run in the same environment:

```bash
docker run --rm mepd mepd models            # the MLIPs and whether each is ready
docker run --rm mepd mepd retro methods
docker run --rm -it -v mepd-data:/data mepd bash
```

The image is about 20 GB, mostly PyTorch/CUDA, MKL, Psi4, and the AiZynthFinder
data. Give a demo of 4 concurrent jobs ~16 GB of memory.

## The data mirror

`https://public.atomsforhumanity.org/mepd/retro/` serves the files of
`retro-data.txt` by their mirror names, plus `SHA256SUMS` and a `README.md`
that credits each dataset and its license (the CC-BY files require
attribution). It is public, which is allowed: everything there is CC-BY-4.0
or MIT. Never put gated files there, such as the FAIR-Chem weights.

Lay the directory out, from a local cache, the old mirror, or upstream:

```bash
python3 -I deploy/docker/fetch_data.py deploy/docker/retro-data.txt ~/mepd-mirror/mepd/retro --flat \
  --from ~/.cache/mepd/retro          # optional: reuse local copies (checked like any other source)
```

Copy `~/mepd-mirror/mepd/` to where the web server serves `/mepd/`. Check it
through the public URL, which downloads every file and verifies it:

```bash
python3 -I deploy/docker/fetch_data.py deploy/docker/retro-data.txt /tmp/mirror-check --flat \
  --mirror https://public.atomsforhumanity.org/mepd/retro
```

Every line should say `from mirror`. **When a file changes** (e.g. a new
AiZynthFinder version), add its line to `retro-data.txt`, re-run the layout
command, and copy again. Old files can stay; nothing references them. If a
build runs before the mirror has a new file, it still works: that file comes
from upstream.
