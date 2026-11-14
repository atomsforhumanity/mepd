"""One local `mepd discovery expand` of a suite system as the scoreboard runs it (benchmarks/cluster/bench.py:
COMMON arguments, the system's conditions, linus steering), with `--procedure`: a check of the staged procedure
on the small systems, not a benchmark (the cluster runs are the bench agent's).

    python benchmarks/staged/run_local.py INPUTS_DIR SYSTEM PROCEDURE OUT_DIR [--budget 600] [--seed 0] [--workers 8]

INPUTS_DIR: a scoreboard run's inputs/ (the seed xyz files, hcooh_profile.toml, linus_steer_linus2.toml).
SYSTEM: hcooh or acetic_acid (1500 K, 1 ms, 1 atm: the two 5- and 8-atom gas-phase systems).
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs"), ap.add_argument("system", choices=["hcooh", "acetic_acid"])
    ap.add_argument("procedure", choices=["classic", "staged"]), ap.add_argument("out")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import mepd

    print("mepd:", mepd.__file__, flush=True)
    cmd = [sys.executable, "-m", "mepd.cli", "discovery", "expand", f"{a.inputs}/{a.system}.xyz",
           "--charge", "0", "--multiplicity", "1", "--steer", "access", "--schedule", "async", "--generator", "reactor",
           "--steer-energies", f"{a.inputs}/linus_steer_linus2.toml", "--inputs", f"{a.inputs}/hcooh_profile.toml",
           "--standard-state", "1atm", "--cutoff-temperature", "1500", "--cutoff-time", "0.001",
           "--temperature", "1500", "--time", "0.001", "--time-budget", f"{a.budget:g}", "--max-species", "400",
           "--generator-option", f"seed={a.seed}", "--channels-option", f"random_seed={a.seed}",
           "--workers", str(a.workers), "-o", os.path.join(a.out, "run")]
    if a.procedure != "classic":     # (the classic arm's argument list is the scoreboard's, unchanged)
        cmd += ["--procedure", a.procedure]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "expand.log"), "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=a.out).returncode


if __name__ == "__main__":
    sys.exit(main())
