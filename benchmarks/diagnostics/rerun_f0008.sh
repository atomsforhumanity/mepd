#!/bin/bash
# F0008 recorded rerun, local steps. Every measurement goes through the explorer team's recorder.
#   rerun_f0008.sh full   SYS ORACLE   oracle.py end to end (TS search + IRC, ceiling with local ORCA single points)
#   rerun_f0008.sh export SYS ORACLE   TS search + IRC + thermal corrections; steering single points to sp_todo.json
#   rerun_f0008.sh import SYS ORACLE   the ceiling from the cluster's sp_done.json
# SYS is a folder under $F (made by oracle_inputs.py); ORACLE is gxtb, gfn2 or uma (uma on the local GPU).
set -eo pipefail
F=${F:?set F to the rerun folder}
W=$(cd "$(dirname "$0")/../.." && pwd)
P=${P:-python}
REC=${REC:?set REC to the recorder script (recorded.py)}
export PYTHONPATH=$W OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mode=$1; sys=$2; K=$3
D=$F/$sys; O=$D/oracle_$K
dev=(); [ "$K" = uma ] && dev=(--device cuda)
mkdir -p "$O"
cd "$D"
case $mode in
  full)   python3 $REC "$O" --repo "$W" -- $P $W/benchmarks/diagnostics/oracle.py "$D" --oracle $K "${dev[@]}" --workers 3 --prefill 3 ;;
  export) python3 $REC "$O" --repo "$W" -- $P $W/benchmarks/diagnostics/oracle.py "$D" --oracle $K "${dev[@]}" --workers 3 --export-sp "$O"
          mv "$O/provenance" "$O/provenance_export" ;;
  import) python3 $REC "$O" --repo "$W" -- $P $W/benchmarks/diagnostics/oracle.py "$D" --oracle $K "${dev[@]}" --import-sp "$O/sp_done.json" ;;
esac
