#!/bin/bash
# build_addmol_run.sh RUN_DIR WORKTREE COMMIT QUERY.json TASKS.jsonl RECORDED_BIN: an afh run folder for addmol_task.sh
# (code = `git archive COMMIT mepd`; one 32-thread pod per task)
set -eu
RUN=$1; W=$2; C=$3; QUERY=$4; TASKS=$5; BIN=$6
B=$(dirname "$(readlink -f "$0")")
mkdir -p "$RUN/code-$C"
git -C "$W" archive "$C" mepd | tar -x -C "$RUN/code-$C"
git -C "$W" rev-parse "$C" > "$RUN/code-$C/COMMIT"
cp "$B/addmol_task.sh" "$RUN/task.sh"; cp "$W/benchmarks/hcooh_profile.toml" "$RUN/"
cp "$BIN/recorded.py" "$BIN/manifest.py" "$RUN/"; cp "$QUERY" "$RUN/query.json"; cp "$TASKS" "$RUN/tasks.jsonl"
cat > "$RUN/run.yaml" <<Y
description: "team41 core: mepd addmol test end to end (quick bystander estimate, verification, relay scans) of a candidate molecule on the aldol parent network (acetone + acetaldehyde, 274 steps, 34 atoms with L-proline), one 32-thread pod (code $C, g-xTB axtb)"
image: 192.168.8.6:5000/mepd-orca-uma-axtb:2026-11-09-main
command: "bash \$RUN_DIR/task.sh"
threads_per_task: 32
cpus_per_pod: 32
memory_per_pod: 32Gi
shards: $(wc -l < "$TASKS")
parallelism: 1
task_timeout: 7200
priority: batch
env: {SELLA_DISABLE_GPU: 1, OMP_NUM_THREADS: 1, MKL_NUM_THREADS: 1, OPENBLAS_NUM_THREADS: 1}
Y
