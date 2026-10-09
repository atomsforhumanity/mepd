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
#   models    MLIP and ReactionT5 weights (Hugging Face etc.); needs the deps
#             to load the models, but not mepd's code
#   runtime   the final image; mepd's own code is its last, small layer
#
# For CI, keep every stage's layers between runs (BuildKit `mode=max`
# cache, e.g. type=registry), so retro and models are downloaded only when
# their inputs change.

ARG UBUNTU_IMAGE=ubuntu:26.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.3.3
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.5

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
# uv.lock pins every package; it is gitignored today, so it is optional here
# (without it uv resolves afresh). Commit it for reproducible builds.
COPY pyproject.toml .python-version uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then lock=--frozen; else lock=; fi \
 && uv sync $lock --no-dev --no-install-project

# -------------------------------------------------------------------- models
FROM deps AS models
ENV HOME=/home/mepd
COPY deploy/docker/prefetch.py /tmp/build/prefetch.py
# Model weights into $HOME's caches: AIMNet2, ANI-2x, ReactionT5.
ARG PREFETCH_MODELS=1
RUN if [ "$PREFETCH_MODELS" = 1 ]; then /opt/mepd/venv/bin/python /tmp/build/prefetch.py models; fi
# FAIR-Chem UMA/eSEN are gated on Hugging Face: only with a token, passed as a
# build secret so it never lands in a layer:
#   docker build --secret id=hf_token,env=HF_TOKEN -t mepd .
ARG FAIRCHEM_MODELS="uma-s-1p2p1"
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -s /run/secrets/hf_token ]; then \
      HF_TOKEN="$(cat /run/secrets/hf_token)" /opt/mepd/venv/bin/python /tmp/build/prefetch.py fairchem $FAIRCHEM_MODELS; \
    else echo "no hf_token secret: FAIR-Chem models not baked in (set HF_TOKEN at run time instead)"; fi \
 && rm -rf "$HOME/.cache/uv" && mkdir -p "$HOME/.cache"

# ------------------------------------------------------------------- runtime
FROM ${UBUNTU_IMAGE} AS runtime
# Links the image to its repository on GHCR (and its access to the repo's).
LABEL org.opencontainers.image.source=https://github.com/atomsforhumanity/mepd \
      org.opencontainers.image.description="mepd web app with its chemistry programs, models and data" \
      org.opencontainers.image.licenses=MIT
# libgomp/libgfortran: g-xTB and GSM. The X libraries: the Cairo bundled in
# openbabel-wheel (its plugins fail to load without them). tini: PID 1 that
# forwards signals and reaps the job processes mepd starts.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 libgfortran5 libstdc++6 \
      libxrender1 libxext6 libx11-6 libexpat1 tini \
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
COPY --from=models --chown=mepd:mepd /home/mepd /home/mepd
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
    MEPD_MODE=demo MEPD_PORT=8765 MEPD_MAX_JOBS=4

# mepd itself, into the venv above (its dependencies are all there already):
# a layer of a few MB, the only one most commits change.
RUN --mount=type=bind,target=/tmp/src,rw \
    uv pip install --no-deps --no-cache --compile-bytecode --python /opt/mepd/venv/bin/python /tmp/src

# The app was checked at build time: every external program resolves.
RUN mepd-entrypoint check

USER mepd
WORKDIR /data
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"MEPD_PORT\"]}/login', timeout=4)"
ENTRYPOINT ["/usr/bin/tini", "--", "mepd-entrypoint"]
CMD ["serve"]
