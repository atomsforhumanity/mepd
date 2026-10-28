# Transition-state conformers

A transition state found from one pair of conformers is one TS among many: the
same reaction can go through the same bond changes with the rest of the
molecule arranged differently (an ester turned, a chain folded, a counter-ion
on the other side). The barrier is set by the lowest of them. **Conformers** on
a transition state (`mepd conformers ts.xyz --ts`) samples those arrangements
around the TS's reacting core and TS-optimizes each.

## What it does

1. **Finds the reacting core.** These are the atom pairs whose distance the
   TS's imaginary mode changes fastest: the forming and breaking bonds, and a
   moving atom's distances to where it comes from and where it goes. Every
   pair within 1.8 times its covalent radius sum that changes at least half
   as fast as the fastest one counts. This needs a Hessian at the session's
   level of theory. On the command line, `--freeze 0-12,3-11` (0-based atom
   pairs) names the core instead, and the Hessian is skipped.
2. **Samples with CREST around it.** The core's distances are held at their
   TS values with xtb distance constraints (`$constrain`, `distance: i, j,
   auto`, force constant 1 Eh/bohr²) while CREST's metadynamics moves
   everything else. CREST runs with `--nci` when the TS is more than one
   molecule, and in the profile's implicit solvent when that is ALPB or GBSA.
3. **Checks what comes back.** CREST's own topology checks are off for this
   (`--notopo --noreftopo`): it counts partial TS bonds as topology and
   would otherwise discard nearly every structure. mepd checks instead and
   drops a conformer if a held distance moved more than 0.05 Å or if any
   other bond formed or broke (bonds perceived from covalent radii; alkali
   and heavier alkaline-earth cations are ions, never bonded).
4. **TS-optimizes each one** at the session's level (as **Transition state ›
   Optimize**), with the frequency check: exactly one imaginary mode. The
   result lists each TS's energy relative to the lowest and how far its core
   distances moved from the input TS. One whose core moved more than 0.3 Å has
   probably gone to another TS: it is flagged and not added to this TS's
   conformers. TS conformers that optimize onto the same TS are marked as
   such, and so are mirror images of another.

The TS conformers join the TS node. The lowest one at the session's level then
represents it, so its barriers use the lowest TS found.

**Why constraints rather than sampling freely:** a TS is a saddle point, so an
unconstrained metadynamics run (or a minimization) slides off it to the
reactant or the product. Holding the core keeps every sampled geometry a TS
guess for the same reaction. TS-optimizing each guess afterwards puts back
what the constraints fixed.

**Why CREST only:** RDKit's embedding has no notion of a partial bond, and its
force field would close or open the core.

## Measured

Three g-xTB transition states, CREST at GFN2-xTB (one thread each), each kept
conformer TS-optimized at g-xTB (2026-10-28):

| TS | atoms | core held | CREST frames kept | distinct TSs | spread (kcal/mol) | lowest vs input | wall time |
|---|---|---|---|---|---|---|---|
| Cp + methyl acrylate, endo (Diels-Alder) | 23 | 2 forming C-C | 3 of 3 | 2 | 1.6 | same | 5 min |
| Cl⁻ + 1-chloropentane (SN2) | 18 | 2 C-Cl | 5 of 5 | 6 (one mirror pair) | 2.3 | 0.9 lower | 4 min |
| the same with K⁺ on the leaving Cl | 19 | 2 C-Cl | 9 of 17 | 7 (two mirror pairs) | 1.2 | 0.4 lower | 9 min |

- No frame broke the held distances or any other bond. With K⁺, 8 frames came
  back with the ion elsewhere (another species) and were dropped.
- After TS optimization the core distances moved at most 0.03 Å, so every
  conformer stayed on the same reaction.
- The Diels-Alder TS's second conformer, 1.6 kcal/mol up, is the other endo
  TS found independently in the Cp + methyl acrylate study (the ester turned):
  the same energy to 10⁻⁹ Eh.
- Mirror images: an achiral reaction has each TS conformer and its
  enantiomer, at one energy. They are kept as distinct saddle points and marked
  as each other's mirror image.

## Limits and possible improvements

- **Only around the seed.** The sampling keeps the core's distances, not just
  its bonds, so TSs with a different core geometry (endo vs exo, another
  approach angle, a counter-ion bridging differently) are not reached. Seed
  from several TSs for those.
- **The core from the imaginary mode is a heuristic.** A strongly coupled mode
  can pull in a neighboring pair, or leave out a bond that changes late along
  the IRC. Check the reported core, and give `--freeze` when it is wrong.
- **CREST's level.** Sampling is at GFN2-xTB (or GFN-FF), not the session's
  level. The TS optimizations re-rank at the session's level, but a
  conformer CREST never kept within its energy window is not tried.
- **CREST builds.** Some CREST 3.0.2 builds fail on a `force constant=` line;
  mepd then runs with xtb's default constant (0.05), and the result says so.
  CREST reads a word after `--notopo` as atoms to exclude and then checks
  topology after all, which is why mepd never passes it last.
