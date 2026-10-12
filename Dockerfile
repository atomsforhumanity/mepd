# syntax=docker/dockerfile:1.7
#
# The full mepd web app in one image: the Python package with every extra the
# web app uses (`uv sync`, from uv.lock), plus the programs it runs that are
# not Python packages, plus the model weights and data it would otherwise
# download on first use.
#
#   docker build -t mepd .
#   docker run --rm -p 8765:8765 -v mepd-data:/data -e MEPD_DEMO_PASSWORD=... mepd
#
# See deploy/docker/README.md for what is inside, the build arguments, and
# how to run it (demo vs full mode, GPUs, the gated FAIR-Chem models).
#
# Stages (independent ones build in parallel):
#   conda     micromamba envs from conda-forge: CREST/xtb/packmol, Psi4, MKL
#   gsm       molecularGSM compiled with its ASE backend (not on conda-forge)
#   gxtb      the g-xTB release build, checksum-verified
#   pybase    uv and its managed Pythons (3.12 for mepd, 3.11 for AiZynthFinder)
#   retro     retrosynthesis data (deploy/docker/retro-data.txt, from our mirror
#             or Zenodo/figshare) and AiZynthFinder's env. Depends only on
#             pybase, the manifest and mepd/retro/{data,aizynth,stock}.py.
#   deps      a Python venv with mepd's locked dependencies, not mepd itself
#   stock     the retrosynthesis stocks' lookup indexes (ZINC: 80 MB)
#   models    AIMNet2 and ANI-2x weights; needs the deps to load
#             the models, but not mepd's code
#   uma       FAIR-Chem UMA weights (gated: needs a Hugging Face token); on
#             pybase, so uv.lock changes don't download or push them again
#   reactiont5  ReactionT5's models (public), likewise on pybase
#   runtime   the final image; mepd's own code is its last, small layer
#
# For CI, keep every stage's layers between runs (BuildKit `mode=max`
# cache, e.g. type=registry), so retro and models are downloaded only when
# their inputs change.

ARG UBUNTU_IMAGE=ubuntu:26.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.3.3
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.5
# The FAIR-Chem models baked in (stage `uma`). They are gated on Hugging Face,
# so a build needs a token of an account with access, as a build secret:
#   docker build --secret id=hf_token,env=HF_TOKEN -t mepd .
# `--build-arg FAIRCHEM_MODELS=` builds without them.
ARG FAIRCHEM_MODELS="uma-s-1p2p1"

FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------- conda
FROM ${UBUNTU_IMAGE} AS conda
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
# CREST and xtb (CREST msreact, conformers, docking, solvation, GFN-FF QM/MM)
# and Packmol (complex building). Only their executables go on PATH, so this
# env's own libraries never shadow anything else. Packmol stays on 20.15: a
# seeded packing differs between versions, and with 21.x the rebuilt product
# of tests/test_reaction_smiles.py's proline aldol comes out stereo-inverted.
RUN micromamba create -y -p /opt/tools -c conda-forge crest=3.0.2 xtb=6.7.1 packmol=20.15.3 \
 && micromamba clean -afy
# Psi4 runs as a worker process in its own Python (MEPD_PSI4_PYTHON): it is
# distributed through conda-forge only, so it cannot live in mepd's venv.
RUN micromamba create -y -p /opt/psi4 -c conda-forge psi4=1.11 python=3.12 \
 && micromamba clean -afy
# MKL for molecularGSM, which includes <mkl.h> on Linux. The GSM binary is
# installed into this env and loads MKL from it at run time (rpath).
RUN micromamba create -y -p /opt/gsm -c conda-forge mkl-devel=2025.3.1 \
 && micromamba clean -afy

# ----------------------------------------------------------------------- gsm
FROM conda AS gsm
ARG GSM_REPO=https://github.com/ZimmermanGroup/molecularGSM.git
ARG GSM_COMMIT=cd668e0cc1c8b6c40ee9f8f2fa0fd1b1294de774
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*
# -DGSM_ENABLE_ASE=1: GSM asks mepd for energies/gradients through the
# ./grad.py helper mepd writes (mepd/pathminimizers/gsm.py). Sequential MKL:
# results do not depend on thread count. This is the upstream code, not
# mepd's patched build, so GSM's early_stop_on_minima is unavailable (mepd
# warns and runs GSM to the end).
RUN git clone -q "$GSM_REPO" /src/gsm \
 && git -C /src/gsm checkout -q "$GSM_COMMIT" \
 && MKLROOT=/opt/gsm cmake -S /src/gsm -B /src/gsm/build \
      -DCMAKE_BUILD_TYPE=Release -DGSM_ENABLE_ASE=1 \
      -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_PREFIX_PATH=/opt/gsm \
      -DCMAKE_INSTALL_PREFIX=/opt/gsm -DCMAKE_INSTALL_RPATH=/opt/gsm/lib \
 && cmake --build /src/gsm/build -j"$(nproc)" \
 && cmake --install /src/gsm/build \
 && /opt/gsm/bin/gsm 2>&1 | head -3 || true
# Keep only what the binary loads: drop headers, static and cluster/GPU MKL libs.
RUN rm -rf /opt/gsm/include /opt/gsm/conda-meta /opt/gsm/share/doc \
 && find /opt/gsm/lib \( -name '*.a' -o -name 'libmkl_sycl*' -o -name 'libmkl_blacs*' \
      -o -name 'libmkl_scalapack*' -o -name 'libmkl_cdft*' -o -name 'libmkl_*ilp64*' \) -delete \
 && test -x /opt/gsm/bin/gsm

# ---------------------------------------------------------------------- gxtb
FROM ${UBUNTU_IMAGE} AS gxtb
# The same build mepd downloads itself on first use (mepd/programs.py).
ARG GXTB_URL=https://github.com/grimme-lab/g-xtb/releases/download/v2.0.1/xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz
ARG GXTB_SHA256=cdfb81164f8d3300fc451dbbf2414528b533e71f7a30ec824b564cd30d6d9e5c
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/gxtb.tar.xz "$GXTB_URL" \
 && echo "$GXTB_SHA256  /tmp/gxtb.tar.xz" | sha256sum -c - \
 && mkdir /opt/gxtb && tar -xJf /tmp/gxtb.tar.xz -C /opt/gxtb --strip-components=1 \
 && rm /tmp/gxtb.tar.xz && test -x /opt/gxtb/bin/xtb

# -------------------------------------------------------------------- pybase
FROM ${UBUNTU_IMAGE} AS pybase
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
# 3.12 for mepd; 3.11 for AiZynthFinder's own env (it needs Python < 3.13).
RUN uv python install 3.12 3.11

# --------------------------------------------------------------------- retro
FROM pybase AS retro
ENV MEPD_RETRO_DATA=/opt/mepd/retro
# The files pinned (SHA-256) in retro-data.txt: from our mirror, else from
# Zenodo/figshare, and only ever if the checksum matches. DATA_MIRROR="" skips
# the mirror. A file used twice (mepd's and AiZynthFinder's copy) is hard-linked.
ARG DATA_MIRROR=https://public.atomsforhumanity.org/mepd/retro
COPY deploy/docker/retro-data.txt deploy/docker/fetch_data.py /tmp/build/
RUN "$(uv python find 3.12)" -I /tmp/build/fetch_data.py /tmp/build/retro-data.txt "$MEPD_RETRO_DATA" \
      --mirror "$DATA_MIRROR"
# Every file mepd's template method asks for must be among them.
COPY mepd/retro/data.py mepd/retro/aizynth.py mepd/retro/stock.py deploy/docker/aizynth-constraints.txt /tmp/build/
RUN "$(uv python find 3.12)" -c "import importlib.util as u, sys; \
s = u.spec_from_file_location('retro_data', sys.argv[1]); m = u.module_from_spec(s); \
sys.modules[s.name] = m; s.loader.exec_module(m); \
missing = [n for n in m.FILES if m.path(n, download=False) is None]; \
sys.exit(f'add to deploy/docker/retro-data.txt: {missing}' if missing else 0)" /tmp/build/data.py
# AiZynthFinder in its own env, set up as `mepd retro setup --aizynthfinder`
# does (mepd/retro/aizynth.py) but pinned throughout (aizynth-constraints.txt;
# it conflicts, on purpose, with any other aizynthfinder VERSION), with its config.yml written from its own
# download_public_data's file list and template (so a version that renames a
# file fails here, not at run time), and the ZINC stock as InChIKeys for mepd.
RUN set -eu; \
    ver="$(sed -n 's/^VERSION = "\(.*\)"/\1/p' /tmp/build/aizynth.py)"; \
    keys="$(sed -n 's/^ZINC_KEYS = "\(.*\)"/\1/p' /tmp/build/stock.py)"; \
    env="$MEPD_RETRO_DATA/aizynth-env"; data="$MEPD_RETRO_DATA/aizynth-data"; \
    uv venv --no-config --python 3.11 "$env"; \
    uv pip install --no-config --no-cache --python "$env/bin/python" -c /tmp/build/aizynth-constraints.txt \
      "aizynthfinder==$ver" "numpy<2"; \
    "$env/bin/python" -c "import os, sys; \
from aizynthfinder.tools.download_public_data import FILES_TO_DOWNLOAD as F, YAML_TEMPLATE as T; \
p = lambda k: os.path.join(sys.argv[1], F[k]['filename']); \
missing = [p(k) for k in F if not os.path.exists(p(k))]; \
sys.exit(f'add to deploy/docker/retro-data.txt: {missing}') if missing else None; \
open(os.path.join(sys.argv[1], 'config.yml'), 'w').write(T.format(*map(p, ['policy_model_onnx', \
'template_file', 'ringbreaker_model_onnx', 'ringbreaker_templates', 'filter_policy_onnx', 'stock'])))" "$data"; \
    "$env/bin/python" -c "import pandas as pd, sys; df = pd.read_hdf(sys.argv[1], 'table'); \
open(sys.argv[2], 'w').write('\n'.join(df['inchi_key'].astype(str)) + '\n')" \
        "$data/zinc_stock.hdf5" "$MEPD_RETRO_DATA/$keys"

# ---------------------------------------------------------------------- deps
FROM pybase AS deps
ENV UV_PROJECT_ENVIRONMENT=/opt/mepd/venv
WORKDIR /src
# uv.lock pins every package. --locked: a pyproject.toml change without an
# updated lock (`uv lock`) fails here, rather than building untested versions.
COPY pyproject.toml .python-version uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# -------------------------------------------------------------------- models
FROM deps AS models
ENV HOME=/home/mepd
COPY deploy/docker/prefetch.py /tmp/build/prefetch.py
# Model weights into $HOME's caches: AIMNet2, ANI-2x (ReactionT5 and UMA: own stages).
ARG PREFETCH_MODELS=1
RUN if [ "$PREFETCH_MODELS" = 1 ]; then /opt/mepd/venv/bin/python /tmp/build/prefetch.py models; fi \
 && mkdir -p "$HOME/.cache"

# ----------------------------------------------------------------------- uma
# FAIR-Chem UMA weights (gated; see FAIRCHEM_MODELS at the top), in a stage
# and a layer of their own. On pybase, with only huggingface_hub (pinned here,
# not from uv.lock): a dependency change neither downloads nor pushes them
# again. A secret is not part of BuildKit's cache key, so the step's key is the
# model list: a build that names models and has no token fails here, rather
# than reusing a step that ran without one.
FROM pybase AS uma
ENV HOME=/home/mepd
COPY deploy/docker/fetch_uma.py /tmp/build/fetch_uma.py
ARG FAIRCHEM_MODELS
RUN --mount=type=secret,id=hf_token,required=false --mount=type=cache,target=/root/.cache/uv \
    mkdir -p "$HOME/.cache/fairchem"; \
    if [ -z "$FAIRCHEM_MODELS" ]; then \
      echo "FAIRCHEM_MODELS is empty: no FAIR-Chem models in this image"; \
    elif [ ! -s /run/secrets/hf_token ]; then \
      echo "FAIRCHEM_MODELS=\"$FAIRCHEM_MODELS\" needs a Hugging Face token: build with" \
           "--secret id=hf_token,env=HF_TOKEN, or --build-arg FAIRCHEM_MODELS= to leave them out" >&2; \
      exit 1; \
    else \
      HF_TOKEN="$(cat /run/secrets/hf_token)" UV_CACHE_DIR=/root/.cache/uv \
        uv run --no-project --no-config --python 3.12 --with huggingface_hub==1.33.0 \
        python /tmp/build/fetch_uma.py $FAIRCHEM_MODELS; \
    fi

# ---------------------------------------------------------------- reactiont5
# ReactionT5's models (public, ~1.6 GB), for `mepd retro --method reactiont5`.
# Like `uma`: on pybase with only huggingface_hub, so uv.lock changes don't
# download or push them again.
FROM pybase AS reactiont5
ENV HOME=/home/mepd
COPY deploy/docker/fetch_reactiont5.py /tmp/build/fetch_reactiont5.py
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv uv run --no-project --no-config --python 3.12 --with huggingface_hub==1.33.0 \
      python /tmp/build/fetch_reactiont5.py \
 && rm -rf "$HOME/.cache/huggingface/xet"      # download logs and staging, not weights

# --------------------------------------------------------------------- stock
# The lookup indexes mepd builds the first time a retrosynthesis uses a stock
# (mepd/retro/stock.py): ZINC's 10.5 M InChIKeys as a sorted uint64 array
# (80 MB) and the PaRoutes stocks' InChIKeys. Built here, they are not rebuilt
# after every container restart. Needs only mepd/retro (mepd's __init__ is lazy).
FROM deps AS stock
ENV MEPD_RETRO_DATA=/opt/mepd/retro
COPY --from=retro /opt/mepd/retro /opt/mepd/retro
COPY mepd/__init__.py /tmp/stock/mepd/__init__.py
COPY mepd/retro /tmp/stock/mepd/retro
RUN PYTHONPATH=/tmp/stock /opt/mepd/venv/bin/python -c "from mepd.retro.stock import load; \
print(load(['zinc', 'paroutes-n1', 'paroutes-n5']).describe())"

# ------------------------------------------------------------------- runtime
FROM ${UBUNTU_IMAGE} AS runtime
ARG FAIRCHEM_MODELS
# Links the image to its repository on GHCR (and its access to the repo's).
LABEL org.opencontainers.image.source=https://github.com/atomsforhumanity/mepd \
      org.opencontainers.image.description="mepd web app with its chemistry programs, models and data" \
      org.opencontainers.image.licenses=MIT
# libgomp/libgfortran: g-xTB and GSM. The X libraries: the Cairo bundled in
# openbabel-wheel (its plugins fail to load without them). gcc + libc headers:
# Triton (UMA on the GPU) compiles a small C helper on first use. tini: PID 1
# that forwards signals and reaps the job processes mepd starts.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 libgfortran5 libstdc++6 \
      libxrender1 libxext6 libx11-6 libexpat1 gcc libc6-dev tini \
 && rm -rf /var/lib/apt/lists/* \
 && userdel -r ubuntu 2>/dev/null || true \
 && useradd --uid 1000 --create-home --home-dir /home/mepd --shell /bin/bash mepd \
 && mkdir -p /data && chown mepd:mepd /data

# Layers from the most to the least stable, so a code change ships as one
# small layer on top and every layer below it is reused (in the cache, the
# registry and on the nodes that already pulled an earlier image).
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=pybase /opt/python /opt/python
COPY --from=conda /opt/tools /opt/tools
COPY --from=conda /opt/psi4 /opt/psi4
COPY --from=gsm /opt/gsm /opt/gsm
COPY --from=gxtb /opt/gxtb /opt/gxtb
COPY --from=retro --chown=mepd:mepd /opt/mepd/retro /opt/mepd/retro
# Only the indexes from `stock` (newer than their sources, so mepd uses them).
COPY --from=stock --chown=mepd:mepd /opt/mepd/retro/zinc_stock_inchikeys.u64.npy \
     /opt/mepd/retro/stock_n1.inchikeys /opt/mepd/retro/stock_n5.inchikeys /opt/mepd/retro/
COPY --from=models --chown=mepd:mepd /home/mepd /home/mepd
# fairchem keeps its weights in a cache of its own, not Hugging Face's.
COPY --from=uma --chown=mepd:mepd /home/mepd/.cache/fairchem /home/mepd/.cache/fairchem
COPY --from=reactiont5 --chown=mepd:mepd /home/mepd/.cache/huggingface /home/mepd/.cache/huggingface
COPY --from=deps /opt/mepd/venv /opt/mepd/venv
RUN ln -s /opt/tools/bin/crest /opt/tools/bin/xtb /opt/tools/bin/packmol /usr/local/bin/
COPY --chmod=755 deploy/docker/entrypoint.sh /usr/local/bin/mepd-entrypoint

ENV PATH=/opt/mepd/venv/bin:/usr/local/bin:/usr/bin:/bin \
    HOME=/home/mepd \
    GXTB_EXECUTABLE=/opt/gxtb/bin/xtb \
    GSM_EXECUTABLE=/opt/gsm/bin/gsm \
    MEPD_PSI4_PYTHON=/opt/psi4/bin/python \
    MEPD_RETRO_DATA=/opt/mepd/retro \
    MEPD_NO_DOWNLOAD=1 \
    MEPD_WEB_STATE_DIR=/data/state \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    OMP_STACKSIZE=1G \
    TORCHDYNAMO_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    MEPD_FAIRCHEM_MODELS=${FAIRCHEM_MODELS} \
    MEPD_MODE=demo MEPD_PORT=8765 MEPD_MAX_JOBS=4

# mepd itself, into the venv above (its dependencies are all there already):
# a layer of a few MB, the only one most commits change.
RUN --mount=type=bind,target=/tmp/src,rw \
    uv pip install --no-deps --no-cache --compile-bytecode --python /opt/mepd/venv/bin/python /tmp/src

USER mepd
# Checked at build time, as the user it runs as: every program and data set
# is there and runs (and the caches the check fills stay writable by mepd).
RUN mepd-entrypoint check
WORKDIR /data
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"MEPD_PORT\"]}/login', timeout=4)"
ENTRYPOINT ["/usr/bin/tini", "--", "mepd-entrypoint"]
CMD ["serve"]
