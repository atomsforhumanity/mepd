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
s=$(date +%s)
python3 $RUN_DIR/recorded.py $PWD/rec -- python3 -m mepd.cli "${ARGS[@]}" > test.log 2>&1 || echo "addmol test exit $?"
echo "wall $(( $(date +%s)-s )) s"
cp rec/results/*.json $OUT_DIR/ || true
cp -r rec/provenance $OUT_DIR/ || true
tail -c 20000 test.log > $OUT_DIR/test_tail.log
# the scans' own records and logs (not their path-search folders)
(cd rec/results && find . -path '*_relay/*' \( -name searches.json -o -name channels.log -o -name 'start.xyz' -o -name 'end.xyz' \) -print0 \
   | tar -czf $OUT_DIR/relay_records.tgz.tmp --null -T - && mv $OUT_DIR/relay_records.tgz.tmp $OUT_DIR/relay_records.tgz) || true
(cd rec/results && tar -czf $OUT_DIR/ts_files.tgz.tmp --ignore-failed-read $(find . -name 'ts*.xyz' -path '*_relay/*/out/ts/*' -o -name '*.xyz' -path '*_verify/*/reopt/*' | head -4000) \
   && mv $OUT_DIR/ts_files.tgz.tmp $OUT_DIR/ts_files.tgz) || true
