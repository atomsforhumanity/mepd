# The bystander estimate

What does a molecule C do to a reaction step when it sits next to it and takes
no part in its bond changes: a hydrogen-bond donor, a Lewis acid, an ion, a
solvent molecule? `mepd.bystander.bystander` answers that for one step of a
finished network without a new path search:

```python
import mepd
from mepd import bystander as B

engine = mepd.engine_from_profile("profile.toml")
water = B.molecule_from_smiles("O", engine)                  # conformers, minimized
frames = {"symbols": [...], "charge": 0, "multiplicity": 1,   # one atom order, Angstrom
          "r": [...], "ts": [...], "p": [...]}                # the TS and its two IRC ends
res = B.bystander(frames, water, engine)
res["barrier_shift"]    # {"electronic": -3.1, "free": -2.9, "spread": 0.4, ...} kcal/mol
res["binding"]["ts"]    # C's binding to the TS: electronic, free (1 mol/L), spread, contact, ...
res["complexes"]["ts"]  # the lowest complex found, the step's atoms first
```

`bystander_steps(steps, molecule, make_engine, workers=8)` runs it for many
steps in parallel processes, in the order given, and calls `on_result` as each
step finishes.

## What it does

1. **Places** C next to each of the three frames. The default placer,
   `"seeded"`, reads GFN2-xTB partial charges: it takes the step's atoms whose
   charge differs most between the TS and either end, and its most polar
   atoms, and puts one of C's three most oppositely charged atoms against each,
   C turned to face it from outside, spun at random. No placer knows a
   reaction class, a functional group or a mechanism. Others: `"random"`
   (`mepd.complexes.packed`), `"screen-ff"` and `"screen-sp"` (200 rigid poses
   on the step's surface, the best by point charges + a 12-6 term, or by single
   points, relaxed).
2. **Relaxes** C in every pose with the step's atoms held where they are (as
   `mepd variant --mode fast` holds the old atoms), to 0.05 eV/Å.
3. **Carries** the three lowest poses of each frame to the other two frames and
   relaxes them there, so that the frames are compared at the same sites; then
   **polishes** the two lowest poses of each frame to 0.01 eV/Å and carries the
   best once more. Most of the lowest complexes come out of these two stages,
   not out of the first relaxation.
4. Repeats all of it with other random numbers (`repeats`, default 3). The
   numbers returned are from the lowest complex of each frame over all repeats;
   `spread` is the range of each repeat's own value.

The barrier shift is binding(TS) − binding(reactant end), the reaction-energy
shift binding(product end) − binding(reactant end), with binding(X) =
E(X·C) − E(X) − E(C, lowest conformer).

**Free energies.** `binding[frame]["free"]` adds a correction at 1 mol/L and
`temperature` without relaxing the complex: C's and the step's translation and
rotation are replaced by the complex's and by the six rigid-body vibrations of
C against the step (the complex's Hessian projected on them, quasi-RRHO as
`mepd.thermo.qrrho`). Internal vibrations are taken as unchanged and symmetry
numbers as 1. On 39 frames it was 1.1 kcal/mol below a full quasi-RRHO of the
relaxed complex on average (mean absolute 1.1; +10 to +13 kcal/mol for
proline and AlCl3, +7 to +9 for water), so a gas-phase electronic binding of
−11 kcal/mol is about 0 at 1 mol/L.

**Solvent.** Give the engine of the network's level: where that has an
implicit solvent the whole search runs in it. `final_engine` instead adds
single points at a second level on the three lowest poses of each frame and
repeat (`res["final"]`).

## Status

- `ok`.
- `reacted`: in some frame every pose, or a pose lower than the lowest
  complex, changed C's own bonds or the step's own bonds. C does not stay a
  bystander there; the numbers given are from the poses that kept their bonds.
- `unmoved`: no pose of some frame moved on relaxing (C was out of range of
  the step). A failure, never `ok`.
- `failed`: nothing could be relaxed, or no pose stayed in contact (closest
  pair further than 1.3 × the van der Waals radii); `why` says which.

Pairs of one atom of each within bonding range are listed as `cross_bonds` of
the lowest complex. Distance alone cannot tell a coordinated Lewis acid
(Al–O) from a new covalent bond, so they are reported and not judged.

## What the number is, and is not

- It is an **estimate for ordering work**, not a barrier. Against the same step
  verified with C present (TS optimized from the estimate's TS complex, IRC,
  ends minimized: `mepd variant --mode reoptimize`), on ten steps (proline on
  aldol steps, water on HCOOH → H2O + CO, AlCl3 on isoprene + methyl acrylate
  steps; g-xTB, gas phase) it had the verified shift's sign on eight and lay
  within 2 kcal/mol on five; the largest difference was 4.4 kcal/mol. The
  verified shift is usually the more negative one: a held TS cannot respond to
  C.
- **The verification is cheaper than the estimate** for a flexible C: 14–19 s
  against 4–5 min of one core at 34 atoms. Verify every step that matters.
- **The search is not converged for a flexible, strongly binding C.** With
  L-proline (17 atoms, three conformers) on 17-atom steps the barrier shift of
  a single repeat has a standard deviation of 0.7–0.8 kcal/mol on most steps
  and of 2–5 kcal/mol on some, where one frame has a deep complex that few
  searches find. Read `spread`; where it is several kcal/mol, the value is
  the lowest of what was found and more repeats can still move it. For water
  and AlCl3 the spread is 0.1–0.6 kcal/mol.
- **It cannot see C taking part.** A proton relay through C, a covalent
  intermediate, or C moving the TS to another saddle are found only by the
  verification (3 of 13 verified steps ended that way: the IRC did not connect
  the estimate's ends) or by a mapping scan (`mepd channels
  --explore-mechanisms`).
- **Gas-phase binding of polar molecules is large** and is not a binding in
  solution: use the solvated level where the network has one.
- One C per step; C's protonation form and conformers are the caller's
  (`molecule_from_smiles` gives a few RDKit conformers minimized at the
  engine's level; a form that is no minimum at that level is refused).
- Symmetry numbers are 1 in the free-energy estimate (RT ln σ: 0.4 kcal/mol
  for water, 1.1 for AlCl3, in binding only; it cancels in the shifts).

Measurements, with hardware and commits:
`benchmarks/diagnostics/addmol/place_*.py` and the team's notes.
