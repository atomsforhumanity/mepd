set -e
cp -r $RUN_DIR/code-34473de code
export PYTHONPATH=$PWD/code PYTHONUNBUFFERED=1
RXN=$(python3 -c "import json,os;print(json.load(open(os.environ['TASK_JSON']))['reaction'])")
python3 -c "import mepd;print(mepd.__file__)"
s=$(date +%s)
python3 $RUN_DIR/recorded.py $PWD/rec -- python3 -m mepd.cli channels --reaction "$RXN" -i $RUN_DIR/hcooh_profile.toml --workers $NTHREADS --parallel-workers 1 --search-budget 900 --explore-mechanisms 20 -o $PWD/rec/out > channels.log 2>&1 || echo "channels exit $?"
echo "wall $(( $(date +%s)-s )) s"
python3 $RUN_DIR/read_channels.py rec/out --out $OUT_DIR/report.json > $OUT_DIR/report.txt || true
cp rec/out/stats.json rec/out/pair_mechanisms.json $OUT_DIR/ || true
cp -r rec/provenance $OUT_DIR/ || true
tail -c 20000 channels.log > $OUT_DIR/channels_tail.log
tar -czf $OUT_DIR/channels_out.tgz.tmp -C rec/out channels alternate-channels ts conformers 2>/dev/null && mv $OUT_DIR/channels_out.tgz.tmp $OUT_DIR/channels_out.tgz || true
