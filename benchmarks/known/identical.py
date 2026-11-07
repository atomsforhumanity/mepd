"""Without --known, does an expansion give what it gave before? Runs the same deterministic `mepd discovery
expand` (bond rules, rounds, g-xTB on axtb) with the code of two commits and compares their summary.json
(species, energies, reactions, steps; file paths aside).

    python benchmarks/known/identical.py BASE_COMMIT NEW_COMMIT PROFILE.toml OUT_DIR [SMILES] [extra expand args...]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = "/home/user/astera/mepd/.venv/bin/python"


def code(commit: str, dst: Path) -> Path:
    d = dst / f"code-{commit}"
    if not d.exists():
        d.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", commit, "mepd"], cwd=REPO, capture_output=True, check=True).stdout
        subprocess.run(["tar", "x", "-C", str(d)], input=archive, check=True)
    return d


def strip(x, out=""):
    """The summary without what names its own run (its folder, paths, wall-clock times)."""
    if isinstance(x, dict):
        return {k: strip(v, out) for k, v in x.items() if k not in ("file", "output_files", "inputs", "t_wall")}
    if isinstance(x, list):
        return [strip(v, out) for v in x]
    if isinstance(x, str) and out:
        return x.replace(out, "<run>")
    return x


def main(argv):
    base, new, profile, out = argv[0], argv[1], Path(argv[2]).resolve(), Path(argv[3]).resolve()
    smiles = argv[4] if len(argv) > 4 else "OC=O"
    extra = argv[5:] or ["--steer", "window", "--rounds", "2", "--max-products", "12"]
    got = {}
    for commit in (base, new):
        c = code(commit, out)
        o = out / f"run-{commit}"
        env = dict(os.environ, PYTHONPATH=str(c), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MEPD_NO_PROVENANCE="1")
        probe = subprocess.run([PY, "-c", "import mepd; print(mepd.__file__)"], env=env, capture_output=True, text=True)
        assert probe.stdout.strip().startswith(str(c)), probe.stdout
        cmd = [PY, "-m", "mepd.cli", "discovery", "expand", smiles, "-i", str(profile), "--workers", "4", "-o", str(o),
               *extra]
        with open(out / f"run-{commit}.log", "w") as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=out).returncode
        assert rc == 0, f"{commit}: exit {rc}"
        got[commit] = strip(json.loads((o / "summary.json").read_text()), str(o))
    a, b = got[base], got[new]
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    rec = {"base": base, "new": new, "smiles": smiles, "extra": extra, "identical": not differ, "differ": differ,
           "species": len(b.get("species", [])), "reactions": len(b.get("reactions", [])),
           "steps": len(b.get("steps", []))}
    (out / "identical.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return 0 if not differ else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
