# One profiled `mepd channels --explore-mechanisms` run in a pod (team41 relay). The task's JSON: reaction, budget,
# code (folder under $RUN_DIR), args (extra channels arguments, a list), stop_after (seconds: the run and all its
# processes are sent SIGTERM then, and what it has written is still read and packed: a stopped run).
set -e
CODE=$(python3 -c "import json,os;print(json.load(open(os.environ['TASK_JSON']))['code'])")
cp -r $RUN_DIR/$CODE code
export PYTHONPATH=$PWD/code PYTHONUNBUFFERED=1
python3 -c "import mepd;print(mepd.__file__)"
python3 - > args.txt <<'P'
import json, os
t = json.load(open(os.environ["TASK_JSON"]))
for a in ["channels", "--reaction", t["reaction"], "-i", os.environ["RUN_DIR"] + "/hcooh_profile.toml", "--workers",
          os.environ["NTHREADS"], "--parallel-workers", "1", "--search-budget", str(t["budget"]),
          "--explore-mechanisms", str(t.get("explore", 20)), *t.get("args", []), "-o", os.getcwd() + "/rec/out"]:
    print(a)
P
mapfile -t ARGS < args.txt
STOP=$(python3 -c "import json,os;print(json.load(open(os.environ['TASK_JSON'])).get('stop_after', 0))")
s=$(date +%s)
timeout -k 30 $STOP python3 $RUN_DIR/recorded.py $PWD/rec -- python3 $RUN_DIR/profile_channels.py $PWD/rec/prof "--" "${ARGS[@]}" > channels.log 2>&1 || echo "channels exit $?"
echo "wall $(( $(date +%s)-s )) s"
python3 $RUN_DIR/profile_channels.py --report rec/prof rec/out > $OUT_DIR/profile.txt 2>&1 || true
python3 $RUN_DIR/read_channels.py rec/out --out $OUT_DIR/report.json > $OUT_DIR/report.txt || true
date +%s > $OUT_DIR/packed_at.txt
cp rec/out/*.json $OUT_DIR/ || true
cp -r rec/provenance rec/prof $OUT_DIR/ || true
tail -c 40000 channels.log > $OUT_DIR/channels_tail.log
(cd rec/out && ls -l --time-style=+%s ts > $OUT_DIR/ts_listing.txt) || true
tar -czf $OUT_DIR/channels_out.tgz.tmp -C rec/out channels alternate-channels ts conformers 2>/dev/null && mv $OUT_DIR/channels_out.tgz.tmp $OUT_DIR/channels_out.tgz || true
