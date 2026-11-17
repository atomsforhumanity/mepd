# One `mepd addmol test` in a pod (team41 core): the task's JSON has code (folder under $RUN_DIR), candidate
# (SMILES), amount, scan (number of relay scans), query (file under $RUN_DIR), and optionally steps / verify.
set -e
CODE=$(python3 -c "import json,os;print(json.load(open(os.environ['TASK_JSON']))['code'])")
cp -r $RUN_DIR/$CODE code
export PYTHONPATH=$PWD/code PYTHONUNBUFFERED=1
python3 -c "import mepd;print(mepd.__file__)"
python3 - > args.txt <<'P'
import json, os
t = json.load(open(os.environ["TASK_JSON"]))
run = os.environ["RUN_DIR"]
args = ["addmol", "test", f"{run}/{t['query']}", "-c", t["candidate"], "--amount", str(t["amount"]), "--scan", str(t["scan"]),
        "--scan-inputs", f"{run}/hcooh_profile.toml", "--workers", os.environ["NTHREADS"], "-o", os.getcwd() + "/rec/results"]
for k in ("steps", "verify"):
    if t.get(k) is not None:
        args += [f"--{k}", str(t[k])]
print("\n".join(args))
P
mapfile -t ARGS < args.txt
# What the run writes goes to $OUT_DIR as it comes (the result file, the scans' records and logs, every TS and IRC
# file), so a task stopped by its time limit loses nothing: copied every 30 s, and once more on TERM and at the end.
pack() {
  cp rec/results/*.json $OUT_DIR/ 2>/dev/null || true
  (cd rec/results 2>/dev/null && find . \( -path '*_relay/*' \( -name searches.json -o -name channels.log -o -name start.xyz -o -name end.xyz \) \) \
       -o \( -path '*_relay/*/out/ts/*' -name '*.xyz' \) -o \( -path '*_verify/*/reopt/*' -name '*.xyz' \) | tar -czf $OUT_DIR/files.tgz.tmp -T - 2>/dev/null \
     && mv $OUT_DIR/files.tgz.tmp $OUT_DIR/files.tgz) || true
  tail -c 20000 test.log > $OUT_DIR/test_tail.log 2>/dev/null || true
}
# (on TERM the command is told first: it writes its result file with status "cut" and stops its scans)
trap 'pkill -TERM -f "mepd.cli addmol test" || true; sleep 25; pack; exit 124' TERM
(while true; do sleep 30; pack; done) &
LOOP=$!
s=$(date +%s)
python3 $RUN_DIR/recorded.py $PWD/rec -- python3 -m mepd.cli "${ARGS[@]}" > test.log 2>&1 &
wait $! || echo "addmol test exit $?"
echo "wall $(( $(date +%s)-s )) s"
kill $LOOP 2>/dev/null || true
pack
cp -r rec/provenance $OUT_DIR/ || true
