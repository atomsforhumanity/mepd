#!/bin/bash
# Submit run NEXT and grow its Job parallelism within CAP pods, behind the FIRST runs (which keep up to FIRST_CAP pods
# while any of them has a task left to start) and the OTHER runs (whose running pods count).
# A FIRST run not submitted yet counts as having tasks left. Parallelism is only raised (lowering deletes pods).
# Hard guard (coordinator 2026-11-09): every submit and every parallelism raise first asks bench.py's pod guard,
# which counts ALL of jan's committed pods on the cluster (any run, any agent); it goes ahead only if those plus
# NEXT's new total stay <= 8 (256 threads), else this script waits and asks again.
# Usage: follow_branch.sh "FIRST..." "OTHER..." NEXT CAP FIRST_CAP    (log on stdout); SUBMIT_FLAGS for bench.py submit
# Environment: HARNESS, PY, MEPD_SCOREBOARD (defaults below).
set -u
firsts=$1; others=$2; next=$3; cap=$4; first_cap=$5
HARNESS=${HARNESS:-$(cd "$(dirname "$0")/../.." && pwd)}   # the checkout whose bench.py submits
PY=${PY:-python}
dir=${MEPD_SCOREBOARD:-$HARNESS/scoreboard}/afh/$next
par=0
pods() { afh ssh "kubectl -n jobs get pods -l afh/run=$1 --no-headers -o custom-columns=P:.status.phase 2>/dev/null" | grep -cE "Running|Pending"; }
while true; do
  active=0; unstarted=0
  for r in $firsts; do
    s=$(afh status "$r" 2>/dev/null)
    state=$(echo "$s" | awk '/^state:/{print $2}')
    pending=$(echo "$s" | awk '/^pending:/{print $2}')
    case "$state" in
      finished) ;;
      running) a=$(pods "$r"); active=$((active + a)); unstarted=$((unstarted + pending - a)) ;;
      *) unstarted=$((unstarted + 1)) ;;
    esac
  done
  oth=0
  for r in $others; do oth=$((oth + $(pods "$r"))); done
  if [ "$unstarted" -gt 0 ]; then room=$((cap - first_cap - oth)); else room=$((cap - active - oth)); fi
  if [ "$room" -gt "$par" ]; then
    if ! (cd $HARNESS && $PY benchmarks/cluster/bench.py podguard "$next" "$room"); then
      sleep 30; continue
    fi
    if [ "$par" -eq 0 ]; then
      sed -i "s/^parallelism: .*/parallelism: $room/" "$dir/run.yaml"
      (cd $HARNESS && $PY benchmarks/cluster/bench.py submit ${SUBMIT_FLAGS:-} "$next"); rc=$?
      if [ $rc -eq 75 ]; then sleep 30; continue; fi
      [ $rc -eq 0 ] || exit 1
    else
      afh ssh "kubectl -n jobs patch job jan-$next --type merge -p '{\"spec\":{\"parallelism\":$room}}'" || exit 1
    fi
    par=$room
    echo "$(date -u +%H:%M:%S) first runs: $active pods, $unstarted to start; others $oth pods -> $next parallelism $par"
  fi
  [ "$par" -ge "$cap" ] && { echo "$(date -u +%H:%M:%S) done: $next at parallelism $cap"; exit 0; }
  sleep 30
done
