#!/bin/bash
# Formic acid with water added (team41 core round 5): the landed explorer run of HCOOH as the parent, two
# targets (CO2 + H2, decarboxylation; CO + H2O, dehydration), water and methane (control) as candidates.
#   hcooh_water.sh copy SRC_RUN DST_DIR | query RUN TARGET QUERY WORKERS | test QUERY OUT WORKERS
# Conditions fixed here for both targets: 700 K, 1 h, formic acid 1 M; each candidate at 1 M.
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
here=$(dirname "$(readlink -f "$0")")
what=$1; shift
case $what in
  # (a copy, so the frequencies the query computes are kept beside the copy, not in another team's run)
  copy) mkdir -p "$2"; cp -r "$1" "$2/run" ;;
  query) python -m mepd.cli addmol query "$1" --target "$2" --initial 'O=CO=1.0' --temperature 700 --time 3600 \
           --inputs "$here/../../hcooh_profile.toml" --workers "$4" -o "$3" ;;
  test) python -m mepd.cli addmol test "$1" -c O -c C --amount 1.0 --scan 25 \
          --scan-inputs "$here/../../hcooh_profile.toml" --inputs "$here/../../hcooh_profile.toml" \
          --workers "$3" -o "$2" ;;
esac
