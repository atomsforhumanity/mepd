# Orbitals

Any 3D view of a structure or a path can show orbitals: switch on
**Orbitals** under the viewer to draw the HOMO or LUMO of what is shown.
**Explore…** opens the [orbital explorer](#orbital-explorer) for everything
else:

- **Whole orbitals**: the orbitals of the complete system shown (a molecule,
  a complex, a TS).
- **Fragment orbitals**: each molecule's own orbitals, computed separately,
  and how a filled orbital of one overlaps an empty orbital of another. This
  is the frontier-orbital picture of why two molecules react.
- How both change along a path, frame by frame.

Both kinds come from the same calculation: a single point that writes the
orbitals to a molden file, at the session's level of theory. What differs is
which system that calculation is run on and what is done with the result.

## Whole orbitals

### Where they come from

The first of these that has the shown geometry:

1. **The session's store.** A background thread keeps orbitals for every
   structure and conformer in the workspace (`<workspace>/orbitals/`), so
   showing them is a file read.
2. **The calculation's own SCF.** Every job keeps the orbitals of each SCF it
   runs (`jobs/<id>/orbitals/`): xtb runs with `--molden`, and TeraChem's own
   molden file is kept. A kept file is used when every coordinate matches the
   shown geometry within 0.002 Å. An IRC frame that is the same geometry
   turned and moved also counts, and its orbitals are turned with it. The bar
   then says "· this run".
3. **A new single point** at the session's method, if no calculation has
   computed this geometry.

The method is that of the session's level of theory: TeraChem (the profile's
method, basis and keywords) in a TeraChem session, g-xTB in a g-xTB session,
and GFN2-xTB otherwise.

**Why:** the orbitals should be those of the calculation that produced the
geometry, not of a different method run afterwards.

### Which orbital is the HOMO

The orbitals are sorted by energy. The HOMO is the highest one holding more
than half an electron, and the LUMO is the next one up. xtb writes open-shell
orbitals spin-averaged, so a singly occupied orbital holds 1.0 electron (it
shows as **SOMO**).

### How they are drawn

- Each orbital is evaluated on a box around the molecule (0.3 bohr spacing,
  5 bohr margin, at most 56³ points; 38³ for the frames of a path).
- Every program writes its basis set slightly differently. xtb's contraction
  coefficients are for unnormalized Gaussians and TeraChem's for normalized
  ones, and g-xTB and GFN2-xTB scale Cartesian d functions differently. Each
  is read its own way. The check is that every orbital integrates to 1.
- The surface is drawn where it encloses the chosen share of the orbital's
  density (**% density** in the explorer, 85 % by default, used in every
  view): blue for the + lobe, red for the − lobe.
- Where the orbital sits (in the tooltip of its energy, "C4 29 %, …"): |ψ|² summed over the grid, each
  point counted for its nearest atom.

### Along a path

The sign of an orbital is arbitrary: each SCF may return +ψ or −ψ, so the
colours could swap from frame to frame. Each frame's orbital is compared with
the frame before it (sampled around every atom) and flipped where needed, so
the colours stay consistent from the first frame on. When a calculation
finishes, the orbitals of its result frames are built ahead, and the viewer
loads four frames at a time in playing order.

## Orbital explorer

**Explore…** in the orbital bar opens the orbital explorer over the whole
window (Esc closes it). It has three parts that work together:

- **The 3D view** draws what you click in either diagram. Shift-click adds a
  second orbital: the first is drawn dark, the second pale, and the readout
  gives their **overlap** and **gap**. The second is drawn with its sign
  chosen so that the overlap is positive: **where the same colours meet, the
  two overlap in phase** (bonding).
- **At this frame** is an MO interaction diagram. The first molecule's own
  orbitals are on the left, the whole system's in the middle, and the other
  molecules' on the right, all on one energy scale (eV). Solid lines are
  filled orbitals, dashed lines are empty, and a dot marks a singly occupied
  one. The lines between columns show how much of each whole orbital each
  molecule's orbital makes up; the thicker the line, the larger the share.
  Hovering over an orbital gives its energy, occupation, and for a whole
  orbital what it is made of. **More levels** shows HOMO−4 … LUMO+4 instead
  of HOMO−1 … LUMO+1.
- **Along the path** is a correlation diagram: each orbital's energy at every
  frame. Each line is *one orbital followed through the path*, not "the
  HOMO" at each frame, so a line keeps its character where two levels cross.
  A line that changes from solid to dashed is a filled orbital becoming an
  empty one, an orbital-symmetry crossing. The buttons above the diagram show
  or hide each system. Click a line to draw that orbital, or click anywhere
  else to go to that frame.

**Start ↔ end** shows the drawn orbitals at the first and last frame side by
side: what each one looks like before and after the reaction.

When it opens on two or more molecules, the explorer draws the likeliest
donation first: the filled frontier orbital of one molecule and the empty
(or half-filled) frontier orbital of another with the smallest gap between
them. With one molecule, it draws the whole system's HOMO.

### How orbitals are followed along a path

Consecutive frames' orbitals are matched by their overlap, calculated
exactly from the two frames' basis functions after taking out the system's
translation. An orbital continues its line into the next frame's orbital it
overlaps most (an assignment over the orbitals near the frontier), as long as
that overlap is at least 0.5. The sign of each SCF's orbitals is arbitrary,
so where the match comes out negative the line's sign flips: its lobes keep
their colours from frame to frame.

### What is computed, and where it is kept

When the explorer opens, every frame of the path gets the whole system's
orbitals (the calculation's own SCF where it kept one, else a single point)
and each molecule's orbitals on its own. These are numbers only: energies,
occupations, contributions and overlaps. Surfaces are built only for the
orbitals drawn, and then for the frames ahead in playing order.

Everything is kept in `<workspace>/orbitals/explorer/` (newest 1 GB), so
opening the explorer again, playing the path or comparing its ends reads
files. A 65-frame IRC with two molecules took 3–4 s with GFN2-xTB.

## Fragment orbitals

### 1. Cutting the system into molecules

Bonds are perceived from distances: two atoms are bonded if they are closer
than 1.25 × the sum of their covalent radii. Each connected piece is one
molecule.

On a path, the cut is made at an end of the path that holds more than one
molecule (the first frame, else the last), and every frame keeps that cut.
This way a transition state, or a frame near the products, is still split
into the reactants. When both ends are one molecule, only the whole system
is shown.

### 2. Charge and spin of each molecule

These come from Lewis structures. Of the charge assignments that add up to
the system's total charge, mepd prefers neutral molecules, then the fewest
charged ones, then the fewest unpaired electrons. Chloromethane + bromide
becomes CH₃Cl (neutral, singlet) + Br⁻ (−1, singlet). A molecule with an odd
number of electrons is a doublet (shown with •).

### 3. Each molecule is computed alone

Each molecule gets its own single point, at exactly its coordinates in the
shown frame, with the session's method. The other molecules are not present:
there is no polarization by them, no charge transfer, and no solvent. See
[What "computed alone" means](#what-computed-alone-means).

**Why:** frontier-orbital theory explains bonding from the orbitals of the
separate, unperturbed molecules. One gives electrons from a filled orbital
into another's empty orbital, and how strongly depends on how much the two
orbitals overlap and how close they are in energy. The whole system's
orbitals have already mixed, so they cannot show this directly.

### 4. Radicals

A molecule with an unpaired electron has a singly occupied orbital
(**SOMO**). It can give its electron or take one. Two radicals bond by
pairing their SOMOs, one electron from each; the readout then says "pair
overlap". In ethanol → C₂H₅O• + H•, that SOMO–SOMO overlap is 0.43 where the
C–H bond is intact, 0.17 halfway along the IRC and 0.04 apart. In the
interaction diagram the two SOMOs combine into the whole system's HOMO (the
C–H bond) and LUMO (σ\*). For a SOMO pair the gap is the difference of the
two SOMO energies: a sign of how polar the bond will be, not an excitation.

### 5. Degenerate orbitals

Orbitals of the same energy (bromide's three 4p lone pairs, hydroxide's π
pair) are drawn side by side at one height. Any mix of them is an equally
valid choice, so one of them alone can have any overlap from zero to the
maximum. Look at all of them: the overlap of the whole set with a partner
orbital is √(sum of their squares). Hydroxide's π pair has exactly zero
overlap with CH₃Cl's σ\* in a backside approach; the attack comes from its σ
lone pair (HOMO−2).

### 6. Overlap

The **overlap** is how much two orbitals share the same space: 0 means not at
all, and 1 means identical. It is calculated exactly from the two molecules'
basis functions (Gaussian overlap integrals, Obara–Saika), not on a grid.
Overlaps below 0.005 are shown as "< 0.005".

An overlap of 0 means the two orbitals cannot mix, however close they are in
energy. Two ethylenes face to face (a suprafacial [2+2]) have exactly zero
HOMO → LUMO overlap by symmetry, while butadiene + ethylene (Diels–Alder) do
not.

The check that each program's files are read correctly: every molecule's
orbitals come out orthonormal through these integrals to 1e-10 (g-xTB,
GFN2-xTB; 1e-4 for TeraChem's 6-31G\*).

### 7. Gap

The **gap** is the energy difference of the two drawn orbitals, each from its
own molecule's calculation. For an anion it can be small or even negative:
gas-phase anion orbitals sit high (hydroxide's HOMO is at −0.2 eV with
GFN2-xTB, above CH₃Cl's LUMO at −4.5 eV).

## Why the whole and fragment gaps differ

They measure different things. The **whole gap** is the LUMO minus the HOMO
of one calculation on the whole system: the lowest empty level minus the
highest filled level, wherever they are. A **fragment gap** is one molecule's
empty orbital minus another molecule's filled orbital, from two separate
calculations: how far apart the two orbitals are that would mix in a
donation.

Butadiene above ethylene (GFN2-xTB, eV):

| Separation | Whole HOMO | Whole LUMO | Whole gap | Butadiene HOMO → ethylene LUMO |
|---|---|---|---|---|
| 6.0 Å | −10.34 (butadiene) | −6.85 (butadiene) | 3.49 | 4.59 |
| 2.6 Å | −10.39 (butadiene) | −6.83 (butadiene) | 3.55 | 4.59 |
| 2.2 Å | −9.94 (mixed) | −6.74 (butadiene) | 3.20 | 4.59 |

- **Far apart**, the whole orbitals are the molecules' own: the whole HOMO is
  the higher of the two molecules' HOMOs and the whole LUMO the lower of their
  LUMOs. At 6 Å both sit on butadiene, so the whole gap is butadiene's own
  (3.49 eV), not the donation's (4.59 eV).
- **Close up, filled and empty orbitals mix**: the filled one is pushed down
  and the empty one up, by about (βS)²/ΔE (β the interaction strength, S the
  overlap, ΔE the fragment gap). That stabilization is the driving force
  frontier-orbital theory counts, and the fragment gap is its ΔE.
- **Close up, filled orbitals also repel**: two filled orbitals split into
  an in-phase and an out-of-phase combination, and the out-of-phase one rises
  (Pauli repulsion). At 2.2 Å the whole HOMO (−9.94 eV) sits above both
  molecules' HOMOs (−10.36 and −11.24 eV), which is why the whole gap shrinks.
- **Electrostatics** shift each molecule's levels in the other's field. This
  is large for ions; see the anion example above.

Use the fragment gap with the overlap to judge whether and which way two
molecules' orbitals mix. Use the whole gap as a property of the system as it
is, already mixed. Both are orbital energies, not excitation energies, and
with g-xTB or GFN2-xTB they come from a minimal basis: compare trends rather
than absolute values.

## What "computed alone" means

A molecule's orbitals in the explorer are those of the molecule at its exact
shape and position in that frame, with the other molecules removed. They
include the **distortion** along the path: a C–Cl σ\* falling as the bond
stretches is real. The **overlap** is computed with both molecules in place,
so it is their true overlap. What they leave out:

1. **Polarization**: the other molecules' charges shift and distort the
   orbitals (large for ions).
2. **Pauli repulsion**: each molecule's filled orbitals must stay orthogonal
   to the partner's filled orbitals, which squeezes and raises them up close.
3. **Charge transfer**: the donation itself. Including it fully gives the
   whole system's orbitals.

Far from the transition state these are small, and the separate molecules'
orbitals are the right starting point for frontier-orbital arguments. Near
the transition state they are large: a molecule's orbital there describes
the distorted molecule, not the electronic structure that is actually
present. The trend of overlap and gap along the path still shows where the
interaction switches on. The whole system's orbitals, and how they are made
of the molecules' (the diagram's lines), show what the mixing produced.

## When HOMO → LUMO is not the picture

The frontier picture says the reacting pair is a filled orbital of one
molecule and an empty one of the other, usually but not always the HOMO and
LUMO. The cases where it is not:

- **Symmetry.** The HOMO can have exactly zero overlap with the partner's
  LUMO in the geometry of the reaction: hydroxide's π lone pairs against the
  C–Cl σ\* (the attack uses HOMO−2), or a suprafacial [2+2]. Look at the
  overlap first: a small gap with zero overlap does nothing.
- **Radicals.** Two radicals pair their SOMOs (see above). A radical adding to
  a closed-shell molecule uses its SOMO with the partner's HOMO or LUMO,
  whichever is closer in energy.
- **Charge control.** Between hard ions or very polar molecules, the
  interaction is mostly electrostatic (Klopman's charge-controlled
  reactions): the overlap stays small and the orbitals explain little.
  Orbitals matter most between soft, polarizable partners.
- **Several orbitals at once.** Metals use d orbitals and back-donation (a
  filled d into the ligand's π\* while the ligand gives into an empty d), so
  two directions matter together. Draw both: each molecule's filled orbital
  with the other's empty one.
- **Distance.** Overlap falls off exponentially with distance. Far apart
  (a reactant complex) every value is small; compare values along the path,
  not with a fixed threshold. The plot under a path shows where it grows.
- **One molecule.** An intramolecular reaction has no fragments to compare.

Also keep in mind that the orbitals are of isolated molecules: in a strongly
polarizing environment (an ion pair, a solvent) the real orbitals shift.

## Limits and possible improvements

**Whole orbitals**

- xtb's orbitals are those of a minimal basis: right in shape and ordering,
  not converged with basis set size.
- Not covered yet: QM/MM systems, TeraChem through ChemCloud/QCCompute (falls
  back to GFN2-xTB), and the intermediate steps of xtb's internal
  optimizations.

**Fragment orbitals**

- **Isolated molecules.** No polarization by the partner, no Pauli
  repulsion, no charge transfer (see [What "computed alone"
  means](#what-computed-alone-means)).
- **No solvent.** Anion orbital energies are gas-phase and can give negative
  gaps. Implicit solvent for the fragment single points would make gaps
  between ions meaningful.
- **Cut by distance only.** An intramolecular reaction (one molecule) has no
  fragments. The API accepts a hand-made cut (atom lists), but the viewer has
  no way to pick one yet.
- **Charges from Lewis structures.** Ion pairs could be assigned the wrong way
  round. The whole system's partial charges could steer the assignment.
- **Following orbitals.** A line ends where no orbital of the next frame
  overlaps it by 0.5 (after taking out translation, not rotation): large
  steps or a turning molecule can break lines. Near-degenerate orbitals that
  mix along the path can swap lines.
- **Cost on a path.** Every frame is one whole-system and one or more fragment
  single points: cheap with xtb (3–4 s for 65 frames of 16 atoms), slow with
  TeraChem.
- **Ways to include the partner, in increasing cost:** fragments embedded in
  the partner's point charges (polarization only; xtb supports point-charge
  embedding); block-localized orbitals (ALMO, BLW, ELMO), optimized in the
  full system but kept on their own molecule (polarization and Pauli), which
  also give a charge-transfer energy per direction; these need an SCF program
  with access to the Fock matrix (e.g. PySCF), not xtb.

## References

- Frontier orbitals: K. Fukui, T. Yonezawa, H. Shingu, *J. Chem. Phys.* 20,
  722 (1952).
- Overlap and gap (second-order perturbation): G. Klopman, *J. Am. Chem.
  Soc.* 90, 223 (1968); L. Salem, *J. Am. Chem. Soc.* 90, 543 (1968).
- Orbital symmetry: R. B. Woodward, R. Hoffmann, *Angew. Chem. Int. Ed.* 8,
  781 (1969).
- Whole-system orbitals in fragment orbitals: S. Dapprich, G. Frenking,
  *J. Phys. Chem.* 99, 9352 (1995).
- Overlap integrals: S. Obara, A. Saika, *J. Chem. Phys.* 84, 3963 (1986).
