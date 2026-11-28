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

**Collide** (under Forces) adds the Reactor's pushes: every so often (about
every 200 fs) two molecules, or two atoms of one, are pulled together for
300 fs, as network growth and Steer by hand do it. A Collide run's MD is
mepd's own (`mepd reactor-run`), on GFN2-xTB or g-xTB through axtb; its
reactions are found and refined once the MD ends, and Run longer cannot
continue it.

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
trajectory the same way. It takes up to 500 atoms. Small boxes run
GFN2-xTB, whose steps cost about eight times more each time the atoms
double; from 100 atoms, on a machine with a GPU, the reactor runs UMA, a
machine-learned potential, instead. Measured on a laptop:

| Atoms | GFN2-xTB | UMA on the GPU |
|---|---|---|
| 30 | ~200 MD steps a second | (not used) |
| 120 | 6 | 40 |
| 240 | 0.7 (1.6 on 4 threads) | 22 |
| 480 | one every 11 s (one every 4 s on 4 threads) | 11 |

Without a GPU (or with three UMA reactors already on it), boxes from 150
atoms run GFN2-xTB on 4 threads. UMA is also given up when it turns out
slower than GFN2-xTB would be: a GPU that another job is using can make
it fifty times slower. The reactor's header names what it runs, and why.

A launched run can drive its MD with UMA too (**MD driven by: UMA**), with
the same checks; at most two such runs share the GPU.
A big box shows from the start, settling while its packing is relaxed (up
to 60 steps, cut short after 15 s once 10 are done). The reactor stops 15
minutes after nobody looks at it.

## Many runs, one list

One run is one packing of the box at one temperature, and what it finds
depends on both. To see what the box does more broadly, give the setup page
several temperatures, separated by commas (`1800, 2200, 2600`), and a
number of **repeats**: one run at each temperature for each repeat, each
with the molecules packed anew (its own seed, kept with the run).

Their reactions come together under **Reactions › Nanoreactor runs**:

- **Each reaction once.** Two runs find the same reaction when its two
  sides hold the same molecules (SMILES, charge and spin), whichever way
  each run saw it. A reaction with a molecule whose bonding could not be
  read stays with its own run.
- **Where it was seen.** Each row says under which conditions it was seen
  and how often (`1800 K ×3`), and **Seen** counts every time, this way
  and back.
- **How far it got.** *Seen in the MD* (not refined yet), *refined* (its
  molecules and reaction complex optimized, with a ΔE), or *TS* (a TS
  search found its barrier).
- **Sorting.** Click a column to sort by barrier, ΔE, times seen or the
  number of runs that found it.

A row's barrier, card and TS search belong to one run's copy of the
reaction, the one from the run that saw it most whose reaction complex has
TS endpoints. The row reads in that copy's direction. **Find the TS of
the … refined without one** queues one search per reaction, however many
runs found it. A run still going is listed as far as it has got: its
events after each simulated picosecond, and its reactions as they are
refined.

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
- **UMA is not GFN2-xTB.** A big box steered by hand moves on UMA's
  forces; its reactions are refined at your session's level of theory
  either way. UMA's weights are Meta's (FAIR Chemistry licence).
- **A Collide run finds its reactions only at the end.** Without Collide,
  reactions are refined while the MD runs.
- **The live network page does not show a Nanoreactor run.** That page
  plays a network expansion's kinetics; a Nanoreactor run shows its
  reactions in Explore instead.
- **Each run keeps its own copy of a reaction it found.** The list shows
  them as one, but Explore and Reactions › *By what they do* still show one
  copy per run, and a TS search on one copy does not carry over to the
  others.
- **No rates yet.** The list counts how often the MD saw a reaction; it
  does not turn counts or barriers into rates at each temperature.
- **The demo** runs at most 10 ps, and one interactive reactor per visitor,
  which stops after 3 minutes unwatched.
