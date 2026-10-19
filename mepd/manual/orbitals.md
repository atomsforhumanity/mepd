# Orbitals

Any 3D view of a structure or a path can show frontier orbitals: switch on
**Orbitals** under the viewer. There are two kinds:

- **Whole**: the orbitals of the complete system shown (a molecule, a
  complex, a TS).
- **Fragments**: each molecule's own orbitals, computed separately, and how
  a filled orbital of one overlaps an empty orbital of another. This is the
  frontier-orbital picture of why two molecules react.

Both come from the same kind of calculation: a single point that writes the
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
  density (**% density**, 85 % by default): blue for the + lobe, red for the −
  lobe.
- "on C4 29 %, …" is where the orbital sits: |ψ|² summed over the grid, each
  point counted for its nearest atom.

### Along a path

The sign of an orbital is arbitrary: each SCF may return +ψ or −ψ, so the
colours could swap from frame to frame. Each frame's orbital is compared with
the frame before it (sampled around every atom) and flipped where needed, so
the colours stay consistent from the first frame on. When a calculation
finishes, the orbitals of its result frames are built ahead, and the viewer
loads four frames at a time in playing order.

## Fragment orbitals

### 1. Cutting the system into molecules

Bonds are perceived from distances: two atoms are bonded if they are closer
than 1.25 × the sum of their covalent radii. Each connected piece is one
molecule.

On a path, the cut is made at an end of the path that holds more than one
molecule (the first frame, else the last), and every frame keeps that cut.
This way a transition state, or a frame near the products, is still split
into the reactants.

### 2. Charge and spin of each molecule

These come from Lewis structures. Of the charge assignments that add up to
the system's total charge, mepd prefers neutral molecules, then the fewest
charged ones, then the fewest unpaired electrons. Chloromethane + bromide
becomes CH₃Cl (neutral, singlet) + Br⁻ (−1, singlet). A molecule with an odd
number of electrons is a doublet (shown with •).

### 3. Each molecule is computed alone

Each molecule gets its own single point, at exactly its coordinates in the
shown frame, with the session's method. The other molecules are not present:
there is no polarization by them, no charge transfer, and no solvent.

**Why:** frontier-orbital theory explains bonding from the orbitals of the
separate, unperturbed molecules. One gives electrons from a filled orbital
into another's empty orbital, and how strongly depends on how much the two
orbitals overlap and how close they are in energy. The whole system's
orbitals have already mixed, so they cannot show this directly.

These are always new single points: no calculation in the workspace ran an
isolated molecule. They are cached per geometry.

### 4. Energy levels

Each molecule's orbitals are sorted by energy as for the whole system.
Orbitals within 0.02 eV of each other are one **level**. Bromide's three 4p
lone pairs, for example, are one level: "HOMO (3-fold)". The next level down
is then HOMO−3, because orbitals are still counted one by one (the three 4p
orbitals are HOMO, HOMO−1 and HOMO−2). Three filled and three empty levels
are kept for each molecule.

**Why levels:** any mix of degenerate orbitals is an equally valid choice, and
one of them alone can have any overlap from zero to the maximum depending on
that arbitrary choice. Hydroxide's HOMO is a degenerate π pair, which has
exactly zero overlap with CH₃Cl's σ\* in a backside approach. The attack comes
from its σ lone pair (HOMO−2). Treating the pair as one level makes the
number independent of the arbitrary mix.

### 5. Overlap

The **overlap** is how much two orbitals share the same space: 0 means not at
all, and 1 means identical. It is calculated exactly from the two molecules'
basis functions (Gaussian overlap integrals, Obara–Saika), not on a grid. For
two levels, it is the square root of the sum of squares over their orbitals,
which does not depend on how a degenerate set is mixed.

An overlap of 0 means the two orbitals cannot mix, however close they are in
energy. Two ethylenes face to face (a suprafacial [2+2]) have exactly zero
HOMO → LUMO overlap by symmetry, while butadiene + ethylene (Diels–Alder) do
not.

The check that each program's files are read correctly: every molecule's
orbitals come out orthonormal through these integrals to 1e-10 (g-xTB,
GFN2-xTB; 1e-4 for TeraChem's 6-31G\*).

### 6. Gap

The **gap** is the energy of the empty level minus that of the filled level,
each from its own molecule's calculation. For an anion it can be small or
negative: gas-phase anion orbitals sit high (hydroxide's HOMO is at −0.2 eV
with GFN2-xTB, above CH₃Cl's LUMO at −4.5 eV).

### 7. What is shown first

- **Which way electrons go.** Of the possible directions (molecule X's filled
  orbitals into molecule Y's empty ones), the one with the smallest HOMO →
  LUMO gap comes first, the frontier-control rule. Directions whose overlap is
  below 0.001 at every level go last.
- **Which levels.** Within a direction, the default is the pair of levels with
  the largest overlap² / gap, with gaps below 1 eV counted as 1 eV. Both menus
  in the bar can pick any other.

**Why not rank everything by overlap² / gap:** in a minimal basis, an anion's
empty levels include many-fold d sets 20+ eV up. Summed over their orbitals,
they outscored the real donation (bromide's HOMO → the C–Cl σ\*).

### 8. What is drawn

The donor's filled level is drawn in dark blue/red and the acceptor's empty
level in pale blue/red, both in place. Within degenerate levels, the mix of
each that overlaps the other best is drawn. Signs are chosen so that the
overlap is positive: **where the same colours meet, the two overlap in
phase** (bonding).

On a path:

- The same levels are drawn at every frame (the first frame's choice, until
  changed).
- Every frame is built on the server when the path is opened, and kept on
  disk (`<workspace>/orbitals/fragments/`, newest 500 MB).
- Frames load ahead in playing order, and a frame's surfaces stay on screen
  until the next frame's arrive.
- The phase is carried from frame to frame as for the whole orbitals.
- A small plot shows the overlap at every frame, with a dot for the frame on
  screen.

### 9. "Whole-system mix"

The tooltip gives the whole system's HOMO and LUMO in terms of the molecules'
levels: the weight of each level is |⟨fragment orbital|whole orbital⟩|². The
molecules' orbitals are not orthogonal to each other, so the weights describe
the mix rather than split it exactly (the idea of charge decomposition
analysis).

## Limits and possible improvements

**Whole orbitals**

- xtb's orbitals are those of a minimal basis: right in shape and ordering,
  not converged with basis set size.
- Degenerate whole-system levels are not grouped as fragment levels are, so
  one member of a degenerate set is shown.
- Not covered yet: QM/MM systems, TeraChem through ChemCloud/QCCompute (falls
  back to GFN2-xTB), and the intermediate steps of xtb's internal
  optimizations.

**Fragment orbitals**

- **Isolated molecules.** No polarization by the partner and no charge
  transfer. Orbitals optimized in the presence of the partner but kept on
  their own molecule would show how each molecule's orbitals respond, and
  give charge-transfer energies per direction. Extremely localized molecular
  orbitals (ELMOs) and block-localized wavefunctions do this, but they need
  an SCF program with access to the Fock matrix (e.g. PySCF), not xtb.
- **No solvent.** Anion orbital energies are gas-phase and can give negative
  gaps. Implicit solvent for the fragment single points would make gaps
  between ions meaningful.
- **Cut by distance only.** An intramolecular reaction (one molecule) has no
  fragments. The API accepts a hand-made cut (atom lists), but the viewer has
  no way to pick one yet.
- **Charges from Lewis structures.** Ion pairs could be assigned the wrong way
  round. The whole system's partial charges could steer the assignment.
- **Ranking constants** (0.02 eV degeneracy, overlap 0.001, 1 eV gap floor)
  were chosen on a handful of cases (Diels–Alder, SN2 with halides and
  hydroxide), not validated broadly.
- **Cost on a path.** Every frame is two or more extra single points. Cheap
  with xtb (about 0.5 s per frame for 16 atoms), slow with TeraChem.

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
