# Explorer status, 2026-11-09 (paused)

Where the reaction-network explorer (`mepd discovery expand`, async schedule) stands. Work is paused here. This
file records what is on develop, what has been checked, and what comes next.

## Goal

Fast (≤ 30 min on 32 cores, ≤ 25 atoms), accurate, unbiased reaction networks. "Accurate" is judged only
against experiment, on 8 systems with measured outcomes: formic acid, methyl formate and acetic acid
decomposition; cyclopentadiene + methylene butyrolactone, isoprene + methyl acrylate, furan + maleimide,
cyclobutene diester and furan-dione (F0007). No reaction templates.

## How a change is checked

`benchmarks/cluster/` runs each version 5 times on each of the 8 systems (80 runs of up to 1800 s on 32 threads,
about 500–1,000 core-hours) and compares the new version against the base with `gate.py`:

- **Accuracy:** the median error of the predicted product ratio against experiment, in kcal/mol. Tolerance 0.5.
- **Recall:** whether a run finds every product experiment reports. Two-stage: a drop counts only if a Fisher test
  agrees; otherwise more replicates are needed.
- **Settle time:** when the predicted ratio stopped changing. Two-stage as well: a slower median fails only if a
  one-sided Mann–Whitney test gives p < 0.05. Isoprene's settle time moves ~350 s between identical runs.
- **Arms:** both arms run on the same image. A base arm from an earlier run is reused only when the cluster load
  (`contention.py`) matched.
- **Isoprene:** gets 15 runs per arm when its 5-run result is inconclusive.
- **Pod guard:** `bench.py podguard` keeps all submissions at ≤ 8 pods (256 threads).

## On develop now

Gated and passed (also on main unless noted):

- Overlapping-molecule fix: a subsystem TS or IRC put back on top of a spectator is redone on the whole box.
- Parallel minimization: a propose job minimizes and Hessian-checks on all the workers it holds.
- linus steering (ωB97X-D3BJ/def2-TZVP, RI-JK): 5–8× faster per point than ORCA, accuracy and recall unchanged.
  It is the default steering in the web app where linus is installed; deploying it needs a CI secret for the
  private linus repository.
- Nanoreactor with switchable forces (piston, collide).
- Mirror-end pool for channel sampling (team19). It passed, but did not make one sampling find both approach
  directions more often, which was its aim.
- The axtb move to 533d623: round-off differences only (energies ≤ 2e-13 Eh). The smoke run passed.

On develop, not gated:

- **Settled stop** (`--settle-kcal`). **Off by default.** It stops when the product ratios stay still (0.3 kcal/mol
  over 30 s and 10 kinetics updates), the seed's reactions are all searched, and no steering point is out.
  Replayed on 320 recorded runs it lost nothing but saved only ~5 % (8 % with linus). The open problem: a
  settled answer can come before a structure that changes it has been searched. For example, furan's exo
  adduct turns up after minutes of an endo-only answer. The rule also does not wait for the seed's
  conformer sampling.
- **Seed-first queue order:** jobs on the seed's pairs start before all other work. Only the order changes,
  not what is computed. Path-search pieces had the lowest rank and waited up to ~29 min behind steering, IRC
  and channel jobs. Replayed together with the settled stop, it would save ~22 % (methyl formate 82 %). The
  replay could not decide 7 runs (furan exo, isoprene meta or para).
- `--known <folder>`: a follow-up expansion reuses earlier runs. Opt-in; stage 1 of its check has not been run.
- Benchmark tooling: `benchmarks/cluster` (scoreboard and gate), `benchmarks/known`, `benchmarks/diagnostics`.

## Open problems

- **Acetic acid fails in every run:** decarboxylation is found but never connected. Its error stays at a 1209
  sentinel in both arms of every gate, so this system is held out of the gate.
- **Runs rarely converge:** of 40 runs, 13 converged; the rest ended at 1800 s (15) or at 400 species (12).
  Predicted ratios often settle long before that, but settling is not proof that nothing relevant remains.
- **Idle reserved threads:** about 31 % of reserved threads sat idle in the 5–9 Nov checks. The largest part is
  the tail after a capped run's last job, where the final kinetics recompute steering energies on one thread.

## Next directions (proposed, not started)

1. **Stop on completeness, not stability.** End a run when every reaction of every species populated above a
   small share (~1 %) has been fully searched: TS, path search, conformer sampling and steering. Also order the
   queue by the population of each job's starting species, of which the seed-first order is a special case.
   Test it offline on the recorded runs before any cluster time.
2. **Count electronic-structure calls as the cost.** Gradients, Hessians and single points, at each level, per
   run and per job kind. Then find the parameters that are excessive for the accuracy they buy: conformers per
   channel, reactor MD trajectories and batches, IRC checks, steering points, path-search depth. Each one is cut
   only if accuracy and recall against experiment hold.

Run-by-run notes for the team (not in the repository) are in the explorer team's `OPEN_ISSUES.md` and the
scoreboard's `NOTES.md`.
