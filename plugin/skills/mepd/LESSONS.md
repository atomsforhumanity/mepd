# Lessons learned with mepd

What cost time with mepd and isn't obvious from `--help`. Each entry: what we saw, the evidence,
what to do. Add entries by pull request (a human reviews them). Facts about the programs mepd
drives (g-xTB, xtb, CREST, ORCA, ...) belong in the research-skills program skills; link there.

## Path searches

- **Use FSM-TS (the default), not NEB, unless there is a reason.** On 20 KAIST reactions it was
  about GSM's quality at half the wall time; on a protein QM/MM step (KSI, caged) under 25 min
  against hours for NEB. A profile that names NEB is not a reason by itself. (2026-10-26)
- **Judge a path search by its TS: optimized, one imaginary mode, IRC connecting the ends.** A
  converged band says nothing more. (2026-10)
- **Pre-mapped datasets (start/end/TS share atom order): run `mepd channels --no-atom-mapping`.**
  Atom mapping re-numbers ends that are already right. (2026-10)

## Transition states and IRCs

- **IRCs at fmax 0.01 eV/Å everywhere** (the default); looser IRCs stopped short of minima.
- **Barriers from the lowest reactant conformer found**, shared across compared runs, not from
  each run's own reactant.
- **`*_check_failed.txt` beside a TS means it is not the reaction's**: its IRC ends are one
  species, or (QM/MM) its imaginary mode hardly overlaps the path. KSI step 1 in a relaxed
  environment ended on a Tyr57 OH rotation 9 kcal/mol below the path's peak (mepd#25).

## QM/MM

- **In a protein, `qmmm_environment = "cage"`.** Relaxed environments let TS searches slide to
  environment saddles; `mepd qmmm protein-build` writes a caged FSM-TS profile. (2026-10-26)
- **Counterions:** `protein-build` neutralizes the protein (Na+/Cl- beside charged groups ≥ 10 Å
  from the QM region). Builds made before 2026-10-27 have the bare net charge (KSI dimer: -12)
  acting on the QM region, which over-stabilized a dienolate (mepd#28).
- **xtb at one electronic temperature per run.** A failed SCC is retried with Fermi smearing;
  mepd now holds that temperature for the rest of the run and recomputes the batch, but set
  `[xtb_engine_kwds] electronic_temperature = 1000` from the start for hot or stretched systems:
  mixed 300/1000 K images blew up a KSI NEB to 1e10 kcal/mol (mepd#23, #24).
- **QM atoms cannot be frozen; restrain them** with `[[restraints]]` (mepd#26).

## Engines

- **g-xTB's own Hessian is unreliable at its default accuracy** (upstream g-xtb#25); mepd uses
  `hessian_acc = 0.001` or gradient differences. Don't trust a g-xTB `--hess` run outside mepd.
- **CREST single-threaded** (`-T 1`), parallel across molecules: see research-skills `crest`.
