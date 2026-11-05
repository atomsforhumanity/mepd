# Orbitals

Any 3D view of a structure or a path can show frontier orbitals: switch on
**Orbitals** under the viewer. There are three kinds:

- **Whole**: the orbitals of the complete system shown (a molecule, a
  complex, a TS).
- **Fragments**: each molecule's own orbitals, computed separately, and how
  a filled orbital of one overlaps an empty orbital of another. This is the
  frontier-orbital picture of why two molecules react.
- **Bonds**: the whole system's filled orbitals rewritten as its bonds and
  lone pairs (intrinsic bond orbitals). Along a path, the few that change
  are the curly arrows of the mechanism, read off the wavefunction.

All three come from the same kind of calculation: a single point that writes
the orbitals to a molden file, at the session's level of theory. What differs
is which system that calculation is run on and what is done with the result.

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

The bar says which geometry was cut: **cut at path start**, **cut at path
end** (when the start is one molecule) or **cut here** (a single structure).
Where the frame on screen does not split into the same molecules on its own,
because they have bonded, it warns **⚠ bonded at this frame**. The orbitals
shown there are still those of the cut's molecules, each computed alone at
that frame's geometry: pieces with broken bonds, not molecules that exist at
that frame. Their overlap and gap say how far the separate molecules'
orbitals have merged, which is useful along the path, but they are not two
molecules interacting. The choice of geometry decides what "the molecules"
are, so keep it in mind when comparing paths: a path stored product-first is
cut at its end. When both ends of a path hold more than one molecule (A + B
→ C + D), the note is a menu: choose **cut at path end** to follow the
products' molecules instead of the reactants'.

**In Design**, the molecules are as drawn: Design's own bonds and formal
charges decide them, however close they sit, so a placed ion or molecule is
its own fragment (the bar says "molecules as drawn"). A bare cation (Mg²⁺,
Li⁺, H⁺) has only empty orbitals, so it is offered only as an acceptor.

**Grouping molecules.** With three or more molecules, **group molecules**
assigns each a letter (A, B, C, …). Molecules with the same letter are one
fragment, computed together at their positions in the frame: for example
ammonia as A and a water dimer as B in a proton relay, where a single water
is not the acceptor. A group's charge is the sum of its molecules' charges;
it is a singlet with an even number of electrons, else a doublet. With three
or more groups, the two menus pick any two of them, e.g. donor → bridge and
then bridge → acceptor. **Each molecule separately** goes back to one
fragment per molecule.

### 2. Charge and spin of each molecule

These come from Lewis structures. Of the charge assignments that add up to
the system's total charge, mepd prefers neutral molecules, then the fewest
charged ones, then the fewest unpaired electrons. Chloromethane + bromide
becomes CH₃Cl (neutral, singlet) + Br⁻ (−1, singlet). A molecule with an odd
number of electrons is a doublet (shown with •).

### 3. Each molecule is computed alone

Each molecule gets its own single point, at exactly its coordinates in the
shown frame, with the session's method. The other molecules are not present:
there is no polarization by them, no charge transfer, and no solvent (see
[What "computed alone" means](#what-computed-alone-means)).

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

**Radicals.** A molecule with an unpaired electron has a singly occupied
level (**SOMO**). It can give its electron or take one, so it is offered on
both sides. Two radicals bond by pairing their SOMOs, one electron from each
("Pairs C₂H₅O• SOMO with H• SOMO"). In ethanol → C₂H₅O• + H•, that SOMO–SOMO
overlap is 0.43 where the C–H bond is intact, 0.17 halfway along the IRC and
0.04 apart, while no filled → empty pair exceeds 0.12. For a SOMO pair the
"gap" is the difference of the two SOMO energies: a sign of how polar the
bond will be, not an excitation.

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
with GFN2-xTB, above CH₃Cl's LUMO at −4.5 eV). It is not the whole system's
HOMO–LUMO gap: see [Why the whole and fragment gaps
differ](#why-the-whole-and-fragment-gaps-differ).

### 7. What is shown first

- **Which way electrons go.** Of the possible directions (molecule X's filled
  orbitals into molecule Y's empty ones), the one with the smallest HOMO →
  LUMO gap comes first, the frontier-control rule. Directions whose overlap is
  below 0.001 at every level go last.
- **Which levels.** Within a direction, the default is the frontier-most pair
  that really overlaps: the lowest empty level, then the highest filled one,
  whose overlap is at least half the best overlap of any pair. Both menus in
  the bar can pick any other.

**Why not rank everything by overlap² / gap:** in a minimal basis, an anion's
empty levels include many-fold d sets 20+ eV up, and a bare cation's levels
sit far below a molecule's (every gap negative). Either way, many-fold p or d
sets won on overlap alone over the real donation (bromide's HOMO → the C–Cl
σ\*; water's σ lone pair → Mg²⁺ 3s).

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

## Bond orbitals

The canonical orbitals (HOMO, HOMO−1, …) of a molecule are spread over the
whole molecule, and any rotation among the filled ones describes exactly the
same electrons. **Bonds** shows the rotation in which every filled orbital
sits on as few atoms as possible: Knizia's intrinsic bond orbitals (IBOs).
They come out as the Lewis structure: one orbital per bond (a double bond as
a σ and a π orbital) and per lone pair. Where no single Lewis structure fits
(a transition state's forming and breaking bonds, a bridging hydrogen), they
spread over three or more atoms. Water gives two O–H bonds and two lone pairs
on O; ethylene gives four C–H bonds and the C–C σ and π.

### How they are made

1. **The whole system's orbitals**, from the same source as **Whole**: the
   session's store, the calculation's own SCF at this geometry (turned with
   the frame when the calculation ran it turned), else a single point.
2. **Intrinsic atomic orbitals (IAOs).** A minimal set of atomic orbitals,
   polarized by the molecule, that holds the filled orbitals exactly (Knizia
   2013). "How much of an orbital is on atom A" is its population in A's
   IAOs. This does not depend on the basis set the way Mulliken or Löwdin
   populations do. The reference minimal basis:
   - xtb's basis (g-xTB, GFN2-xTB) is already a minimal valence basis. The
     reference is its own minimal valence shells per atom (s for H and the
     alkali and alkaline-earth metals; s and p for the p block; s and d for
     transition metals). Polarization shells, such as Cl 3d, are left out.
   - All-electron bases (TeraChem): MINAO, the minimal basis Knizia made from
     the cc-pVTZ sets (as distributed with PySCF), H to Kr except K. Core
     orbitals (C 1s, Cl 1s–2p, …) are left out of the bond orbitals.
3. **Localization.** The filled orbitals are rotated, two at a time, to
   maximize Σ over orbitals and atoms of (population on the atom)^p (Knizia
   2013, with the 2×2 rotation angles of his reference code), with **p = 2**.
   Knizia used 4 and found 2 effectively the same except in aromatic rings.
   IboView's default has been 2 since 2015. Along a g-xTB Diels–Alder IRC
   (an aromatic transition state) p = 4 flipped back and forth between two
   localizations within ten frames, while p = 2 changed smoothly. Benzene's π
   orbitals came out the same with either.
4. **Open shells.** Files with separate α and β orbitals give separate α and
   β bond orbitals (shown with α/β). xtb writes open shells spin-averaged:
   the doubly and singly occupied orbitals are localized separately (a
   singly occupied orbital on one atom shows as a **radical**).

The bar lists every bond orbital: those spread over three or more atoms first
(marked "3c"), then from the highest energy down. The first one is drawn by
default. "on C3 47%, …" is the orbital's IAO population per atom. The energy
in the tooltip is the canonical orbital energies it mixes, weighted. It is a
guide to which bond or lone pair is most reactive, not a measurable energy.
**charges** gives the IAO partial charges of the atoms.

### Bonds along a path

On a finished path (an IRC, a path's frames), every frame's bond orbitals
are built and each orbital of the first frame is followed through the
frames. This is the analysis of Knizia & Klein (2015). Most orbitals stay
where they are. The few that change are drawn as arrows, the curly arrows of
the mechanism.

### What the arrows mean

<figure class="manual-figure">
<svg viewBox="0 0 720 240" role="img" aria-labelledby="arrows-fig-title">
<title id="arrows-fig-title">How to read the arrows, on an SN2 step</title>
<defs>
<marker id="hd-orange" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#e07b00"/></marker>
<marker id="hd-teal" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#12907e"/></marker>
</defs>
<line x1="140" y1="130" x2="340" y2="130" stroke-width="3" stroke-dasharray="7 6" style="stroke: var(--muted)"/>
<line x1="340" y1="130" x2="560" y2="130" stroke-width="3" style="stroke: var(--muted)"/>
<text x="240" y="156" text-anchor="middle" font-size="12" style="fill: var(--muted)">Nu···C forming</text>
<text x="400" y="118" text-anchor="middle" font-size="12" style="fill: var(--muted)">C–X breaking</text>
<ellipse cx="183" cy="130" rx="19" ry="11" fill="none" stroke="#e07b00" stroke-width="1.2" stroke-dasharray="3 3"/>
<circle cx="140" cy="130" r="22" stroke-width="1.5" style="fill: var(--bg); stroke: var(--text)"/>
<circle cx="340" cy="130" r="22" stroke-width="1.5" style="fill: var(--bg); stroke: var(--text)"/>
<circle cx="560" cy="130" r="22" stroke-width="1.5" style="fill: var(--bg); stroke: var(--text)"/>
<text x="140" y="135" text-anchor="middle" font-size="15" style="fill: var(--text)">Nu</text>
<text x="340" y="135" text-anchor="middle" font-size="15" style="fill: var(--text)">C</text>
<text x="560" y="135" text-anchor="middle" font-size="15" style="fill: var(--text)">X</text>
<path d="M186 121 Q215 52 240 118" fill="none" stroke="#e07b00" stroke-width="3.5" marker-end="url(#hd-orange)"/>
<circle cx="214" cy="86" r="6" fill="#e07b00"/>
<path d="M452 138 Q487 205 532 143" fill="none" stroke="#12907e" stroke-width="3.5" marker-end="url(#hd-teal)"/>
<circle cx="488" cy="172" r="6" fill="#12907e"/>
<g font-size="11" font-weight="700" text-anchor="middle">
<circle cx="168" cy="100" r="9" stroke="#e07b00" stroke-width="1.5" style="fill: var(--bg)"/><text x="168" y="104" fill="#e07b00">1</text>
<circle cx="258" cy="98" r="9" stroke="#e07b00" stroke-width="1.5" style="fill: var(--bg)"/><text x="258" y="102" fill="#e07b00">2</text>
<circle cx="214" cy="60" r="9" stroke="#e07b00" stroke-width="1.5" style="fill: var(--bg)"/><text x="214" y="64" fill="#e07b00">3</text>
<circle cx="436" cy="160" r="9" stroke="#12907e" stroke-width="1.5" style="fill: var(--bg)"/><text x="436" y="164" fill="#12907e">1</text>
<circle cx="548" cy="166" r="9" stroke="#12907e" stroke-width="1.5" style="fill: var(--bg)"/><text x="548" y="170" fill="#12907e">2</text>
<circle cx="488" cy="200" r="9" stroke="#12907e" stroke-width="1.5" style="fill: var(--bg)"/><text x="488" y="204" fill="#12907e">3</text>
</g>
</svg>
<figcaption>An SN2 step, drawn by hand to show how the arrows read. <b>1</b>
tail: where the electron pair sits at the first frame. <b>2</b> head: where
it sits at the last frame. <b>3</b> dot: where it is at the frame on screen.
Orange: the nucleophile's lone pair becomes the new Nu–C bond. Teal: the C–X
bond becomes a lone pair on the leaving group X. In mepd the lines are the
computed paths of each pair's centre, and the atoms move with the
frames.</figcaption>
</figure>

Each arrow is one electron pair (one bond orbital) that changes along the
path. It is a curly arrow of the mechanism, but computed from the
wavefunction instead of drawn by hand.

| Part | What it shows |
|---|---|
| Tail | Where the pair sits at the first frame: in a lone pair's lobe, just off its atom, or along a bond, between its two atoms. |
| Head | Where the pair sits at the last frame: between the two atoms of a new bond, or on the atom of a new lone pair. |
| Line | The route of the pair's centre, frame by frame. |
| Dot | Where the pair is at the frame on screen. Drag the slider and the dots move along the arrows. |
| Thick arrow | The orbital chosen in the menu, the one drawn as blue/red surfaces. |
| Colour | The same in the legend under the viewer ("Arrows") and in the plot. Click a legend entry to draw that orbital. |

Hover an arrow in the view to see its name ("C1–C2 → C1–C6") and how much
of the pair has moved at this frame. Only orbitals that change get an arrow
(a change of 0.1 or more; see below). Every other bond and lone pair stays
where it is.

The pair's centre is the orbital's centroid ⟨φ|r|φ⟩, the average position of
its two electrons, computed exactly from the basis at every frame. Drawing
these centroids as arrows follows Vidossich & Lledós (2014).

**How they differ from arrows drawn by hand:**

- **They are short.** A pair's centre moves about 0.5–1 Å, so zoom in.
- **A head ends between two atoms** (the new bond), not on an atom.
- **They can bend near the transition state**, where an orbital is briefly
  spread over three atoms and its centre sits between them.
- **A short arrow inside one bond is polarization**: the pair shifts toward
  one atom, and no bond forms or breaks. Its name ends in "(polarizes)".
- **A ring of arrows can go either way.** For a pericyclic step
  (Diels–Alder) the arrows could be drawn clockwise or counterclockwise, and
  the orbitals pick one direction.

**Timing.** The plot next to the menu shows, for every arrow in its colour,
how much of the pair has moved at each frame. Arrows that rise together
are synchronous. One that rises well before the others is a bond change
that runs ahead of the rest within the same elementary step. In HCN → HNC
(below), the hydrogen's two arrows are half done before the TS, and the N
lone pair's arrow only after it.

### How the orbitals are followed

- **Following.** Each frame's orbitals are matched one to one with the
  previous frame's by the largest overlap: an assignment (Hungarian method)
  within each spin set, with the sign chosen so that the overlap is
  positive, so the colours do not flip. The overlap uses the orbitals' IAO
  coefficients. IAOs are the same atomic functions from frame to frame, as
  IboView treats the atomic orbitals. If the smallest overlap with which an
  orbital was followed is below 0.8 (IboView's warning level), the bar says
  **⚠ loosely followed**. The frames may be too far apart, or the orbital
  mixes with an equivalent one, such as lone pairs around a bond axis, where
  any mix is as localized as another.
- **Which orbitals change.** An orbital's change at a frame is
  √(Σ_A (n_A − n_A(first frame))²), with n_A its population on atom A (the
  measure of Klein & Knizia and IboView). Orbitals whose change reaches 0.1
  at some frame are **changing**: they are listed first and drawn as arrows,
  largest change first. The rest are listed as unchanged. IboView sets no
  threshold and only ranks; 0.1 separated the reacting orbitals from the
  rest in every case tried here (unchanged ones stayed below 0.05).
- **"moved"** is the share of the orbital that has moved to other atoms
  since the first frame: half the sum over atoms of the change in its
  population. A lone pair on A that becomes an even A–B bond has moved 50 %.
  The option list gives each changing orbital's largest value.
- **The plot** shows every changing orbital's "moved" frame by frame, in its
  arrow's colour. Whether the arrows move together or one after another says
  whether the bond changes are synchronous or stepwise within one
  elementary step.
- **Δq** gives the atoms whose IAO charge at this frame differs most from
  the first frame's (+ lost electrons, − gained).

Until every frame is built, the frame on screen shows its own bond orbitals
("following the path…").

### Examples

g-xTB IRCs, read from the start of the path:

- **HCN → HNC** (44 frames, TS at frame 21): the C–H bond becomes a lone pair
  on C (43 % moved), one of the C–N bonds becomes the N–H bond (49 %), and the N
  lone pair becomes a C–N bond (32 %). The first two are half done by frame
  17, before the TS. The lone pair is half done at frame 32, after it: the
  hydrogen moves first and the lone pair follows. A fourth orbital, a C–N
  bond, changes its polarization (14 %). The fifth stays below 3 %.
- **Butadiene + ethylene → cyclohexene** (164 frames): three orbitals change,
  each by 50 %. Each C=C π bond becomes one of the two new C–C σ bonds or
  the new π bond. The other fourteen move less than 4 %. A pericyclic
  reaction can be drawn with its arrows going either way round the ring,
  and the bond orbitals pick one.

### Reading them for design

Each **?** in the bar explains its part in place: the bond orbitals, the
arrows, and Δq. Each one links back here and to the References tab.

- **Which bonds react.** The changing orbitals name the bonds and lone pairs
  that do the chemistry, and everything else is a spectator. Substituents
  and catalysts act on those atoms.
- **Where charge builds up.** Δq at the TS says which atoms gain or lose
  electrons on the way up. An electron-withdrawing group on an atom that
  gains electrons, or a donor on one that loses them, stabilizes the TS
  (the reasoning behind Hammett-type substituent effects). A Lewis acid or a
  hydrogen-bond donor helps where a lone pair builds up. The charges are a
  guide: check a suggestion by computing the substituted path.
- **Timing.** If one arrow moves well before the others (a proton before a
  C–C bond), the steps are asynchronous. A substituent that speeds up the
  first event may change which event limits the rate.
- **With Fragments.** Bonds shows what the electrons do. Fragments shows why,
  for two molecules: the donor's filled and the acceptor's empty orbital,
  their overlap and gap. A changing lone pair or π bond in Bonds is usually
  the donor level that Fragments ranks first.

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

In **Fragments**, a molecule's orbitals are those of the molecule at its exact
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
interaction switches on.
**Whole** shows what the mixing produced, and the bar's "whole-system mix"
tooltip says how the whole HOMO and LUMO are made of the molecules' levels.

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
  two directions matter together. Read both directions in the menu.
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

**Bond orbitals**

- **xtb's minimal basis.** When the calculation's basis is itself minimal,
  the IAO step has nothing to correct. The IAOs are then the
  orthonormalized atomic orbitals, and the bond orbitals are Pipek–Mezey
  orbitals on Löwdin populations. They are right in shape (the textbook
  bonds and lone pairs above). No published work validates IBOs from
  tight-binding wavefunctions, so check a conclusion that matters in a
  TeraChem session (all-electron, MINAO reference).
- **Exponent.** Only p = 2 is offered. p = 4 picks one Kekulé structure in
  aromatic rings at a single geometry, but flips between them along a path.
- **Following.** The matching assumes the frames are aligned, as an IRC's
  are, and close enough. Mapping orbitals as sets between structures (Bensberg
  & Reiher 2023) would be more robust for distant frames.
- **The 0.1 threshold** for changing orbitals was set on three paths
  (an SN2 by interpolation, HCN → HNC, Diels–Alder), not validated broadly.
- **Not built yet**, from the same literature:
  - natural reaction orbitals: the occupied and empty orbital pairs that mix
    as the atoms move along the IRC, with a curve that peaks where the
    electrons reorganize (Ebisawa et al. 2022);
  - the activation strain model along the path: the energy of distorting
    the molecules against their interaction, plotted against a forming bond
    length (Bickelhaupt & Houk 2017). The fragment single points of
    **Fragments** already give most of it;
  - occupied–occupied overlaps in **Fragments**: Pauli repulsion, which
    Lewis acid catalysis lowers (Vermeeren et al. 2020);
  - charge-transfer orbitals with energies (ETS-NOCV, ALMO-EDA), which need
    an SCF program with access to the Fock matrix (e.g. PySCF).
- **Cost.** Each frame needs the whole system's orbitals: free when the
  calculation computed that frame, else a single point. The Diels–Alder IRC
  above (164 frames, 16 atoms, g-xTB) took 15–25 s the first time on a
  laptop, 3 frames at a time. After that it is a file read.

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
- Intrinsic atomic and bond orbitals: G. Knizia, *J. Chem. Theory Comput.* 9,
  4834 (2013).
- Electron flow along IRCs: G. Knizia, J. E. M. N. Klein, *Angew. Chem. Int.
  Ed.* 54, 5518 (2015).
- The orbital change measure and following orbitals frame to frame: J. E. M.
  N. Klein, G. Knizia, *Angew. Chem. Int. Ed.* 57, 11913 (2018); J. E. M. N.
  Klein, G. Knizia, H. S. Rzepa, *ChemistryOpen* 8, 1244 (2019).
- Orbital centroids as arrows: G. Vidossich, A. Lledós, *Dalton Trans.* 43,
  11145 (2014).
- Pipek–Mezey localization: J. Pipek, P. G. Mezey, *J. Chem. Phys.* 90, 4916
  (1989).
- MINAO and IBOs in PySCF: Q. Sun et al., *J. Chem. Phys.* 153, 024109
  (2020).
- Mapping orbitals between structures: J. Bensberg, M. Reiher, *J. Phys.
  Chem. Lett.* 14, 2112 (2023).
- Natural reaction orbitals: T. Ebisawa, M. Hasebe, T. Tsutsumi, T. Tsuneda,
  T. Taketsugu, *Phys. Chem. Chem. Phys.* 24, 3532 (2022).
- Activation strain model: F. M. Bickelhaupt, K. N. Houk, *Angew. Chem. Int.
  Ed.* 56, 10070 (2017); Lewis acids lower Pauli repulsion: P. Vermeeren, T.
  A. Hamlin, I. Fernández, F. M. Bickelhaupt, *Angew. Chem. Int. Ed.* 59,
  6201 (2020).
