"""Copy what the oracle test starts from (one system's ceiling: spec, profiles, the g-xTB transition states and their
mirror twins, the seed) into a fresh folder, with no caches or results.

    python oracle_inputs.py SRC_SYSTEM_DIR DST_SYSTEM_DIR
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import twins_of  # noqa: E402


def main(src, dst):
    src, dst = Path(src).resolve(), Path(dst).resolve()
    dst.mkdir(parents=True, exist_ok=False)
    spec = json.loads((src / "ceiling_spec.json").read_text())
    files = {spec["profile"], spec["steer"], spec["species"][0]["xyz"], "ts_refined/distinct.json"}
    for st in spec["steps"]:
        d = Path(st["xyz"]).parent
        files |= {str(d / "ts.xyz"), str(d / "result.json")}
        for tw in twins_of(src, d.name):
            files |= {f"ts_refined/{tw}/ts.xyz", f"ts_refined/{tw}/result.json"}
    for f in sorted(files):
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dst / f)
    spec["cache"] = "ceiling_cache.json"
    (dst / "ceiling_spec.json").write_text(json.dumps(spec, indent=1))
    print(len(files) + 1, "files")


if __name__ == "__main__":
    main(*sys.argv[1:3])
