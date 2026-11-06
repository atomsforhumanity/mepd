#!/usr/bin/env bash
# Entrypoint of the mepd image.
#
#   serve   (default) the web app on 0.0.0.0:$MEPD_PORT, data under /data
#   check   verify every external program and data set the app uses is here
#   ...     anything else runs as a command, e.g. `docker run IMAGE mepd models`
#
# MEPD_MODE picks how `serve` runs:
#   demo  (default) open to anyone who opens the page, or behind one shared
#         password if MEPD_DEMO_PASSWORD is set. Each browser gets its own private
#         workspace under /data/demo/visitors, with the limits of
#         mepd/web/demo.py and read-only compute profiles in /data/demo/profiles.
#   full  the unrestricted app over one shared workspace, /data/workspace.
#         Anyone who logs in can read files and run programs in the container.
#         Log in with MEPD_WEB_PASSWORD, or the token printed at start-up.
set -euo pipefail

check() {
  local bad=0
  ok() { printf '  %-10s %s\n' "$1" "$2"; }
  fail() { printf '  %-10s MISSING: %s\n' "$1" "$2"; bad=1; }
  # Output captured first: with pipefail, `prog | grep -q` fails when grep
  # stops reading early and prog dies of SIGPIPE.
  says() { local out; out="$("${@:2}" 2>&1 || true)"; grep -q "$1" <<<"$out"; }
  echo "mepd image check:"
  for exe in "$GXTB_EXECUTABLE" "$GSM_EXECUTABLE" "$(command -v xtb)" "$(command -v crest)"; do
    # The real path: conda's binaries find their libraries relative to it ($ORIGIN).
    exe="$(readlink -f "$exe")"
    local missing; missing="$(ldd "$exe" 2>/dev/null | grep 'not found' | tr -d '\t' | tr '\n' ' ' || true)"
    [ -z "$missing" ] || fail "$exe" "unresolved libraries: $missing"
  done
  says "xtb version" "$GXTB_EXECUTABLE" --version && ok g-xtb "$GXTB_EXECUTABLE" || fail g-xtb "$GXTB_EXECUTABLE does not run"
  says "xtb version" xtb --version && ok xtb "$(command -v xtb)" || fail xtb "xtb does not run"
  says "Version" crest --version && ok crest "$(command -v crest)" || fail crest "crest does not run"
  [ -x "$(command -v packmol)" ] && ok packmol "$(command -v packmol)" || fail packmol "not on PATH"
  [ -x "$GSM_EXECUTABLE" ] && ok gsm "$GSM_EXECUTABLE" || fail gsm "$GSM_EXECUTABLE"
  # Triton (UMA on the GPU) compiles a C helper on first use.
  if command -v cc >/dev/null && cc --version >/dev/null 2>&1; then ok cc "$(command -v cc)"
  else fail cc "no C compiler (Triton, UMA on GPUs, needs one)"; fi
  "$MEPD_PSI4_PYTHON" -c "import psi4" 2>/dev/null && ok psi4 "$MEPD_PSI4_PYTHON" || fail psi4 "$MEPD_PSI4_PYTHON cannot import psi4"
  python -c "import mepd.web.app" && ok mepd "$(python -c 'import mepd, os; print(os.path.dirname(mepd.__file__))')" \
    || fail mepd "mepd.web.app does not import"
  local methods
  methods="$(mepd retro methods 2>&1)"
  for m in templates reactiont5 aizynthfinder; do
    grep -qE "^$m +ok" <<<"$methods" && ok "retro" "$m" || fail retro "$m: $(grep -A1 "^$m" <<<"$methods" | tail -1)"
  done
  # Stock indexes, prebuilt: present and newer than their sources (else mepd rebuilds them).
  local idx src
  for pair in zinc_stock_inchikeys.u64.npy:zinc_stock_inchikeys.txt stock_n1.inchikeys:stock_n1.txt \
              stock_n5.inchikeys:stock_n5.txt; do
    idx="$MEPD_RETRO_DATA/${pair%%:*}"; src="$MEPD_RETRO_DATA/${pair##*:}"
    if [ "$idx" -nt "$src" ]; then ok stock "${pair%%:*}"; else fail stock "$idx missing or older than $src"; fi
  done
  # ReactionT5's models: their tokenizer and config load from the image alone.
  if out="$(HF_HUB_OFFLINE=1 python -c "from transformers import AutoConfig, AutoTokenizer
for repo in ('sagawa/ReactionT5v2-retrosynthesis', 'sagawa/ReactionT5v2-forward'):
    AutoTokenizer.from_pretrained(repo); AutoConfig.from_pretrained(repo)" 2>&1)"; then
    ok reactiont5 "offline"
  else
    fail reactiont5 "does not load offline: $(tail -1 <<<"$out")"
  fi
  # FAIR-Chem models baked in (MEPD_FAIRCHEM_MODELS, from the build): each must
  # load from the image alone, with Hugging Face offline.
  local m out
  for m in ${MEPD_FAIRCHEM_MODELS:-}; do
    if out="$(HF_HUB_OFFLINE=1 python -c "import sys; from fairchem.core import pretrained_mlip; \
pretrained_mlip.get_predict_unit(sys.argv[1], device='cpu')" "$m" 2>&1)"; then
      ok fairchem "$m"
    else
      fail fairchem "$m does not load offline: $(tail -1 <<<"$out")"
    fi
  done
  # axtb (g-xTB in process, backend = "axtb"), unless built with AXTB_COMMIT=:
  # the same g-xTB energy as the program.
  if [ -n "${MEPD_AXTB_COMMIT:-}" ]; then
    if out="$(python -c "
from qcdata import Structure
from mepd.engines.gxtb import GXTBCalculator
from mepd.nodes.node import StructureNode
xyz = '3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n'
e = [GXTBCalculator(backend=b, n_parallel=1).compute_energies([StructureNode(structure=Structure.from_xyz(xyz))])[0]
     for b in ('axtb', 'xtb')]
assert abs(e[0] - e[1]) < 1e-8, e
print(f'{e[0]:.8f}')" 2>&1)"; then
      ok axtb "${MEPD_AXTB_COMMIT:0:7}, water $(tail -1 <<<"$out") Eh as the g-xTB program"
    else
      fail axtb "$(tail -1 <<<"$out")"
    fi
  else
    ok axtb "not in this image (built with AXTB_COMMIT=): backend = \"axtb\" unavailable"
  fi
  # Installed is not enough: AiZynthFinder's own env must import (it broke on
  # a NumPy 1/2 mismatch once while `aizynthcli` was still there).
  local az="$MEPD_RETRO_DATA/aizynth-env/bin/python" err
  if err="$("$az" -c "import aizynthfinder.interfaces.aizynthcli, rxnutils.routes.base" 2>&1)"; then
    ok aizynth "imports"
  else
    fail aizynth "$(tail -1 <<<"$err")"
  fi
  [ "$bad" = 0 ] || { echo "image check failed"; exit 1; }
}

serve() {
  local port="${MEPD_PORT:-8765}" jobs="${MEPD_MAX_JOBS:-4}"
  case "${MEPD_MODE:-demo}" in
    demo) exec mepd web /data/demo --demo --host 0.0.0.0 --port "$port" --no-open --max-jobs "$jobs" ;;
    full) mkdir -p /data/workspace
          exec mepd web /data/workspace --host 0.0.0.0 --port "$port" --no-open --max-jobs "$jobs" ;;
    *) echo "MEPD_MODE must be demo or full (got ${MEPD_MODE})"; exit 2 ;;
  esac
}

case "${1:-serve}" in
  serve) serve ;;
  check) check ;;
  *) exec "$@" ;;
esac
