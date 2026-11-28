# Nanoreactor

The Nanoreactor puts molecules you choose in a box, heats them and squeezes
them, and reads off what reacts. Open it from the first screen (**Network ›
Nanoreactor**) or at `#/reactor/new`.

## What goes in

One row per species:

- **SMILES**, or an everyday name (water, hydroxide, O2, glycine).
- **How many** copies go in the box.
- **Charge**, read from the SMILES: write `[OH-]` for hydroxide and
  `[NH4+]` for ammonium.
- **Spin** of each copy, as a multiplicity (1 singlet, 2 doublet,
  3 triplet). It starts from the SMILES: one more than its unpaired
  electrons, so `[OH]` is a doublet. O₂ written `O=O` starts as a triplet,
  its ground state.

A list can be pasted instead: `2 CC=O, 4 O, [OH-], O=O triplet`. The count
goes before or after a species (`2 CC=O`, `CC=O*2`), and a spin word
anywhere in it.

Under the rows are the box's atoms, its total charge and its **spin**:

- **From the species:** the unpaired electrons of every copy added up, as
  if their spins were parallel. One O₂ in water makes a triplet box, and
  two make a quintet.
- **Lowest:** a singlet for an even number of electrons, a doublet for an
  odd one. Two radicals can then pair up and bond.
- **Other:** any multiplicity. An even number of electrons takes an odd
  multiplicity and an odd number an even one; any other spin is refused.

## Two ways to run it

**Launch nanoreactor** runs on its own (`mepd discovery nanoreactor`). The
molecules are packed at random inside a spherical wall and heated, by
default to 2000 K for 5 ps. Once a picosecond the wall squeezes in to the
**piston squeeze** fraction of its radius for a quarter of the period.

The trajectory is then read as reaction events. Each one keeps only the
molecules whose bonds change; those atoms are cut out and optimized, and
each molecule is optimized alone. Reactions are refined while the MD runs
and join Explore as they are found. **Find each reaction's TS** also starts
a TS search on each one. The page opens on the run; its Explore tab is the
network.

The MD runs at your session's level of theory. With g-xTB a few dozen atoms
take about 4 minutes per simulated picosecond; an MLIP is much faster.
**Run longer** on the result continues the same MD.

**Steer by hand** starts the interactive reactor on the same box: a live
GFN2-xTB MD you can drag, squeeze, switch the piston on and off in, and
push molecules together in (Collide). **Stop & analyze** reads its
trajectory the same way. It keeps up with 120 atoms at most, and it stops
15 minutes after nobody looks at it.

## Starting over

**Sessions › Start fresh** opens an empty session on the first screen. The
session you were in stays in the list as it was.

## Limits and possible improvements

- **The reactions are refined at their own lowest spin.** The box's spin
  drives the MD only. Each reaction cut out of it, and each species, is
  optimized as a singlet or a doublet, so a triplet O₂ found in the
  results is listed as a singlet.
- **"From the species" assumes parallel spins.** Two doublet radicals give
  a triplet box, which keeps them from bonding; choose **Lowest** to let
  them pair.
- **Collide is only in Steer by hand.** The unattended run has only the
  piston wall.
- **The live network page does not show a Nanoreactor run.** That page
  plays a network expansion's kinetics; a Nanoreactor run shows its
  reactions in Explore instead.
- **The demo** runs at most 10 ps, and one interactive reactor per visitor,
  which stops after 3 minutes unwatched.
