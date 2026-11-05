"""One dft_ladder structure as an afh cluster task: $TASK_JSON = {"id", "name", "structure": {...dft_ladder
structure, with "xyz_text" in place of a file}, "spec": {...dft_ladder spec without "structures"}}.
Threads: every ORCA level runs with %pal nprocs $NTHREADS. Writes $OUT_DIR/<name>.json (dft_ladder's record) and
the ORCA work directories' outputs (orca.out only) under $OUT_DIR/orca/."""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dft_ladder  # noqa: E402

raw = os.environ.get("TASK_JSON", "")
task = json.loads(Path(raw).read_text() if raw and Path(raw).is_file() else (raw or Path("task.json").read_text()))
out = Path(os.environ.get("OUT_DIR", "out"))
nthreads = int(os.environ.get("NTHREADS", "1"))
spec = dict(task["spec"])
spec["workdir"] = "work"
for lev in [spec["opt"]] + list(spec.get("sp", [])):
    lev["threads"] = nthreads
    lev.setdefault("maxcore", 3000)
s = dict(task["structure"])
Path("input.xyz").write_text(s.pop("xyz_text"))
s["xyz"] = str(Path("input.xyz").resolve())
os.environ.setdefault("MEPD_ORCA", "/opt/orca/orca")
rec = dft_ladder.one(task["name"], s, spec, Path.cwd())
for f in Path("work").glob("*"):
    shutil.copy(f, out / f.name)
print(json.dumps({k: v for k, v in rec.items() if k not in ("trace",)}, indent=1)[:4000])
sys.exit(1 if rec.get("error") else 0)
