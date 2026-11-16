#!/bin/bash
# build_relay_run.sh RUN_DIR WORKTREE COMMIT TASKS.jsonl: an afh run folder for relay_task.sh (code = `git archive COMMIT mepd`)
set -eu
RUN=$1; W=$2; C=$3; TASKS=$4
B=$(dirname "$(readlink -f "$0")"); BIN=$(dirname "$(readlink -f "$(command -v recorded.py 2>/dev/null || echo "$RECORDED")")")
mkdir -p "$RUN/code-$C"
git -C "$W" archive "$C" mepd | tar -x -C "$RUN/code-$C"
echo "$(git -C "$W" rev-parse "$C")" > "$RUN/code-$C/COMMIT"
cp "$B/relay_task.sh" "$RUN/task.sh"; cp "$B/profile_channels.py" "$B/read_channels.py" "$RUN/"
cp "$W/benchmarks/hcooh_profile.toml" "$RUN/"; cp "$BIN/recorded.py" "$BIN/manifest.py" "$RUN/"
cp "$TASKS" "$RUN/tasks.jsonl"
cat > "$RUN/run.yaml" <<Y
description: "team41 relay: profiled mepd channels --explore-mechanisms on parent steps of the aldol network with L-proline on both sides (31-34 atoms), 32 workers per pod (code $C, g-xTB axtb, FSM-TS)"
image: 192.168.8.6:5000/mepd-orca-uma-axtb:2026-11-09-main
command: "bash \$RUN_DIR/task.sh"
threads_per_task: 32
cpus_per_pod: 32
memory_per_pod: 32Gi
shards: $(wc -l < "$TASKS")
parallelism: 8
task_timeout: 3600
priority: batch
env: {SELLA_DISABLE_GPU: 1, OMP_NUM_THREADS: 1, MKL_NUM_THREADS: 1, OPENBLAS_NUM_THREADS: 1}
Y
