"""Cluster task: one reactor trajectory (sample.one) of mepd's reactor generator on one MD level.

task.json: {"id", "xyz": seed file under $RUN_DIR, "md": "tblite" | "gfn2" | "gxtb-v2.0.1", "seed": int,
            "opts": {"temperature": K, "time_ps": ps, "pull_fs": fs}}
"tblite": mepd's own LangevinMD (GFN2-xTB from tblite); "gfn2" / "gxtb-v2.0.1": the same MD with axtb forces
(md_axtb.install). Writes $OUT_DIR/s_SEED.json (sample.py)."""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
t = json.load(open("task.json"))
if t["md"] != "tblite":
    os.environ["MD"] = t["md"]
sys.path.insert(0, here)
import sample  # noqa: E402

out = os.environ.get("OUT_DIR", "out")
os.makedirs(out, exist_ok=True)
sample.one((os.path.join(os.environ["RUN_DIR"], t["xyz"]), out, int(t["seed"]), t["opts"]))
