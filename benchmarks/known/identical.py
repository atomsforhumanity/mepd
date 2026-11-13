"""Without --known, does an expansion give what it gave before? Runs the same deterministic `mepd discovery
expand` (bond rules, rounds, g-xTB on axtb) with the code of two commits and compares their summary.json
(species, energies, reactions, steps; file paths aside, and without the fields a run only records beside its
results, RECORDED: a commit that adds some must leave everything else as it was).

    python benchmarks/known/identical.py BASE_COMMIT NEW_COMMIT PROFILE.toml OUT_DIR [SMILES] [extra expand args...]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


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


# What a run records beside its results (docs/network-expansion.md, "What a run records"), by where it is: times,
# reasons and the terms of numbers the summary already had. Stripped only there (rounds[*].kinetics.molecules, the
# answer, stays compared).
RECORDED = {"species": ("t_found", "energy_terms", "molecules"),
            "steps": ("free_energy_barrier_kcal", "free_energy_barrier_missing", "closed", "ts_energy_terms",
                      "reacting", "ts_group", "t_found", "t_opened", "irc", "placeholder_reason"),
            "reactions": ("t", "batch", "proposed_key", "not_searched"),
            "steering": ("level", "steer_fallbacks")}


def without_record(summary: dict) -> dict:
    out = {k: v for k, v in summary.items() if k != "steps_removed"}
    for section, keys in RECORDED.items():
        rows = out.get(section)
        if isinstance(rows, dict):
            out[section] = {k: v for k, v in rows.items() if k not in keys}
        elif isinstance(rows, list):
            out[section] = [{k: v for k, v in row.items() if k not in keys} for row in rows]
    return out


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
        got[commit] = without_record(strip(json.loads((o / "summary.json").read_text()), str(o)))
    a, b = got[base], got[new]
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    recorded = {s: sorted({k for row in (rows if isinstance(rows, list) else [rows]) for k in row} & set(RECORDED[s]))
                for s, rows in ((s, json.loads((out / f"run-{new}" / "summary.json").read_text()).get(s) or [])
                                for s in RECORDED)}
    rec = {"base": base, "new": new, "smiles": smiles, "extra": extra, "identical": not differ, "differ": differ,
           "recorded_fields_in_new": recorded,
           "species": len(b.get("species", [])), "reactions": len(b.get("reactions", [])),
           "steps": len(b.get("steps", []))}
    (out / "identical.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return 0 if not differ else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
