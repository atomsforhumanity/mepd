"""Cluster task: `n` reactor trajectories of one system, `width` at a time (the proposal job's batch), on one MD engine:
"tblite" (code_tblite: mepd 1a58d2d, GFN2-xTB from tblite), or md_method "gfn2" / "gxtb-v2.0.1" / "uma" (code_axtb).
Writes OUT_DIR/s_SEED.json per trajectory (scripts/sample.py) and OUT_DIR/batch.json (wall times)."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
run = os.environ["RUN_DIR"]; t = json.load(open("task.json"))
code = run + ("/code_tblite" if t["engine"] == "tblite" else "/code_axtb")
sys.path.insert(0, code); sys.path.insert(0, run + "/scripts")
os.environ["PYTHONPATH"] = code
import sample
out = os.environ.get("OUT_DIR", "out"); os.makedirs(out, exist_ok=True)
opts = dict(t.get("opts", {}))
if t["engine"] != "tblite":
    opts["md_method"] = t["engine"]
t0 = time.time()
if t["engine"] == "uma" and int(t["width"]) > 1:
    from mepd.discovery.langevin import uma_server
    opts["md_server"] = uma_server()
t_server = time.time() - t0
waves = []
with ProcessPoolExecutor(int(t["width"])) as ex:
    seeds = list(range(int(t["seed0"]), int(t["seed0"]) + int(t["n"])))
    for k in range(0, len(seeds), int(t["width"])):
        w0 = time.time()
        list(ex.map(sample.one, [(run + "/" + t["xyz"], out, s, opts) for s in seeds[k:k + int(t["width"])]]))
        waves.append(time.time() - w0)
json.dump({"engine": t["engine"], "width": t["width"], "n": t["n"], "server_start_s": t_server, "waves_s": waves,
           "total_s": time.time() - t0}, open(out + "/batch.json", "w"))
