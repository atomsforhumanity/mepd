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
- **Cap a search by path searches, not time: `mepd channels --max-path-searches N`.** A wall-clock
  budget stops wherever the machine got to, so counts depend on its speed and load and a rerun
  elsewhere finds other routes. (2026-11)
- **GSM is reproducible only when it converges.** A run that hits its iteration limit returns a
  snapshot that moves with a 1e-9 Å nudge or a rotation (8 to 148 kcal/mol on one KAIST pair).
  Compare methods on converged runs. The patched build (github.com/jandestrada/molecularGSM, branch
  `mepd`) writes full-precision geometries and is bit-identical across thread counts; with the
  stock binary mepd turns GSM's early stop off and warns. (2026-09)
- **Geodesic paths before 5b18bda (2026-10-14) were built in the wrong units** (Bohr read as Å).
  TS + IRC results from given structures are unaffected; which TS a search found, its cost and
  success rates may change. Regenerate such runs. (2026-10)

## Transition states and IRCs

- **IRCs at fmax 0.01 eV/Å everywhere** (the default); looser IRCs stopped short of minima.
- **Barriers from the lowest reactant conformer found**, shared across compared runs, not from
  each run's own reactant.
- **`*_check_failed.txt` beside a TS means it is not the reaction's**: its IRC ends are one
  species, or (QM/MM) its imaginary mode hardly overlaps the path. KSI step 1 in a relaxed
  environment ended on a Tyr57 OH rotation 9 kcal/mol below the path's peak (mepd#25).
- **A TS counts only if both IRC ends keep their species when minimized**, not just if the minimized
  ends match. Soft saddles (e.g. −51 cm⁻¹) whose IRC barely moves then minimize ~100 kcal/mol down
  into whatever matches; 7 of 20 "lowest route" winners in one gallery were such cases. The same
  rule holds for every step of a route: retry a failing step's IRC at 0.01 eV/Å and require the
  same two species. Before publishing outliers, look at the imaginary frequency and the end
  drops. (2026-10)

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

## Explorer (`mepd discovery expand`)

- **Prune by barriers and kinetics only.** Filters on bond-change or fragment counts remove relay
  and shuttle steps (one elementary step, many bonds); gate searches on the live kinetics
  instead. (2026-10)
- **Run `mepd channels` on every kinetically relevant step.** One TS per step misses the lowest TS
  conformer and nearby exits (Cp + methyl acrylate: barrier too high, one diastereomer). (2026-10)
- **Mirror images count twice** (`--mirror-images`, on by default): each chiral TS with its twin,
  not a lowest-TS formula; enantiomers share reach and the lower energy. (2026-10)

## Web and tests

- **`mepd web DIR` records DIR in `~/.config/mepd/web_sessions.json`.** For test servers set
  `XDG_CONFIG_HOME` to a scratch directory, or the user's session menu fills with test
  workspaces; stop test servers by PID. (2026-10)
- **`uv sync --all-extras` can fail building `vina`** (web extra) from source without the Boost
  headers; a plain `uv sync` (the `app` group installs everything) worked. (2026-11)
- **Run the suite with and without `GXTB_EXECUTABLE` set**: the conftest stubs the g-xTB lookup
  differently, and a first install must work without it. (2026-10)
- **GFN-FF (xtb) needs ~1.06e-6·N² GB** (7.6 GB at 2,685 atoms); use AMBER (OpenMM) for protein
  environments. (2026-10)
