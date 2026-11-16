#!/bin/sh
# The team41 core demonstration, one recorded step per call:
#   demo_aldol.sh adopt RUN WS | query WS QUERY | query-run RUN QUERY WORKERS | candidates QUERY ENOL_REPORT CC_REPORT OUT |
#                 candidates-ts QUERY ENOL_OUT CC_OUT OUT | test QUERY CANDS OUT WORKERS
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
here=$(dirname "$0")
what=$1; shift
case $what in
  adopt) python -m mepd.cli addmol adopt "$1" "$2" ;;
  query) python -m mepd.cli addmol query "$1" --target 'CC(=O)CC(C)O' --initial 'CC(C)=O=1.0' --initial 'CC=O=1.0' \
           --temperature 298.15 --time 86400 -o "$2" ;;
  query-run) python -m mepd.cli addmol query "$1" --target 'CC(=O)CC(C)O' --initial 'CC(C)=O=1.0' --initial 'CC=O=1.0' \
           --temperature 298.15 --time 86400 --workers "$3" -o "$2" ;;
  candidates) python "$here/demo_candidates.py" "$@" ;;
  candidates-ts) python "$here/demo_candidates_ts.py" "$@" ;;
  test) python -m mepd.cli addmol test "$1" --candidates "$2" --workers "$4" -o "$3" ;;
esac
