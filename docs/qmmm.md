# QM/MM in mepd

A QM/MM system is a fixed list of atoms, such as a molecule in a shell of
explicit solvent or an active site in a protein. A few of its atoms (the
**QM region**) are computed at the profile's level of theory, with any mepd
engine (g-xTB, an MLIP, an ASE calculator...). The rest (the **environment**)
is computed with a force field. Every mepd command runs on a QM/MM system:
endpoint minimization, NEB/FNEB/GSM, TS optimization, IRC, Hessians and
reaction network expansion.

This replaces neb-dynamics' experimental `QMMMEngine`. That engine needed
TeraChem for both levels; this one does not. TeraChem can be the QM level,
and TeraChem's own QM/MM is kept as an option (see *TeraChem* below).

## The model

* **Embedding.** Two low-level choices:
  * `mm = "gfnff"` (default), `"gfn2"` or `"gfn1"` use the subtractive
    (ONIOM) scheme, `E = E_QM(model) + E_low(real) - E_low(model)`. GFN-FF
    needs no parameters, so any structure can be embedded as it is. It runs
    in axtb, in this process, when axtb is installed (xtb 6.7.1's force
    field, reproduced term by term: no xtb process or `gfnff_topo` file per
    call, an analytic Hessian, and none of xtb's memory ceiling, which ran
    out at about 1700 waters; 1728 waters take 0.8 GB and 13 s per gradient
    on one core). `[qmmm] gfnff_backend = "xtb"` (or
    `MEPD_GFNFF_BACKEND=xtb`) runs the xtb program instead; GFN1/GFN2 low
    levels and the mean-force sampler's dynamics always do.
  * `mm = "amber"` uses the additive scheme,
    `E = E_QM(model) + E_MM(everything except QM-QM terms)`, through OpenMM,
    from a prmtop (or a PDB of standard residues plus force-field XMLs).

  Both are *mechanical* embeddings. The QM region feels the environment's
  sterics, dispersion and the low level's charges, but its electrons are not
  polarized by them.
* **Link atoms.** A covalent bond cut by the boundary is capped with a
  hydrogen on the bond, at the QM atom's C–H distance. Its force goes back
  to the two atoms of the cut bond. Prefer to cut C–C bonds: polar cuts,
  cut X–H bonds and several cuts on one atom are reported as problems.
* **Frozen atoms.** QM/MM optimizations move the QM atoms and a shell of
  the environment (`active_radius`, Å; small molecules such as waters are
  taken whole). The rest stays exactly where it is. Frozen atoms are held
  in every optimization (ASE `FixAtoms`), in every path (their gradient is
  zero, geodesic interpolation keeps them fixed, and the ends are never
  rotated onto each other), and they are left out of Hessians. Hessians are
  partial: they are taken over the QM atoms only (partial Hessian
  vibrational analysis).
* **Species identity.** Molecular graphs, elementary-step checks and SMILES
  use the QM region only (capped with its link hydrogens). A water that
  moves does not count as a reaction.
* **One surface for the whole network.** GFN-FF's bonded terms are taken
  from the system's *reference* geometry (the system as set up), for every
  structure. Bonds breaking in the QM region therefore never switch
  force-field terms mid-path, and every species is on the same surface.

Frozen atoms also work without QM/MM: `chain_inputs.frozen_atom_indices`
wraps any engine in the same `FrozenAtomsEngine`.

## Command line

```bash
# A solute in explicit water; the solute is the QM region.
mepd qmmm build "NCC(=O)O" --solvent water --shell 6 --active-radius 5 -o gly_water

# A region on your own structure (xyz or PDB).
mepd qmmm region system.pdb --qm "0-23" --qm-charge -1 --active-radius 8 -o site

# A gas-phase reaction in solvent: the reactant is solvated, the product
# (and a TS) are put into that same solvent shell (same atom order).
mepd qmmm reaction --start reactant.xyz --end product.xyz --ts ts.xyz --solvent water -o rx
mepd optimize rx/system.xyz rx/product.xyz -i rx/qmmm_profile.toml -H
mepd ts --guess rx/ts.xyz -i rx/qmmm_profile.toml --irc

# Another geometry of the solute (a product, a TS) into an existing system.
mepd qmmm embed product.xyz --system gly_water -o emb      # --into: a minimized system structure

# An old neb-dynamics / TeraChem input (tc.in + prmtop + rst7 + qmindices).
# By default it runs now, with AMBER through OpenMM and the QM region at the
# profile's level; --mm terachem keeps TeraChem's own QM/MM.
mepd qmmm from-tc path/to/tc.in -o converted

# Any command, through the profile written next to the region:
mepd optimize gly_water/system.xyz -i gly_water/qmmm_profile.toml
mepd run --start a.xyz --end b.xyz -i gly_water/qmmm_profile.toml
mepd discovery expand gly_water/system.xyz -i gly_water/qmmm_profile.toml --allow-zwitterions

# Check a structure or path, and split its energy into QM and environment.
mepd qmmm inspect gly_water/region.json path.xyz -i gly_water/qmmm_profile.toml -o report.json
```

The profile carries the region in a `[qmmm]` table. Files named in it are
relative to the profile:

```toml
engine_name = "gxtb"        # the QM level
[qmmm]
file = "region.json"        # or: reference = "system.xyz", qm_atoms = "0-11", active_radius = 6.0, mm = "gfnff"
```

`[qmmm]` and `[solvation]` do not combine: put explicit solvent in the
environment instead.

## Taking a gas-phase reaction into solvent

Both ends must be one QM/MM system: the same solvent molecules, in the
same order, only the solute's geometry differing. Solvating the two ends
separately would give two unrelated solvent shells, and a path between them
would mostly move solvent. So the start is solvated once and every other
geometry of the solute (the end, a TS) is *embedded*: aligned onto the
solute where it sits in the system and moved out of contact with the
solvent, keeping its own geometry (a TS keeps its partial bonds), the
solvent left exactly where it is (`mepd.qmmm_build.embed`). The region
records which atoms are the solute (`solute_atoms`), so this works after
the QM region was grown to include solvent molecules too.

For a known reaction, re-optimizing the embedded gas-phase TS (with its
IRC) is the most direct route; a path search between the solvated ends
works as well but, like in the gas phase, FNEB can overshoot the barrier:
optimize the TS from its result. Checked on acetaldehyde → vinyl alcohol
(g-xTB QM, GFN-FF water): gas-phase barrier 74.9 kcal/mol; the embedded
TS re-optimized in water, IRC connecting both solvated ends, 77.1 kcal/mol.

### From a converged path to a TS

`mepd run` on a QM/MM system first matches the end's solvent molecules to
the start's and marches the solvent along the interpolated QM path
(`mepd.qmmm_path`). `mepd ts --guess path.xyz` (a path's frames, with its
`.energies`) starts from the interpolated energy maximum. With
microiterations (an OpenMM environment), the TS search then computes an exact
Hessian of the QM region there, with the environment held, and starts Sella
on it. With a QM engine that has analytic Hessians (g-xTB in axtb) and
mechanical embedding, that Hessian is assembled from them:
`J^T H_QM(model) J` plus the environment's block (`J^T H_low J` terms for
GFN-FF in axtb, analytic; central differences of the cheap force-field
gradient for AMBER and xtb), where `J` maps the model system, link hydrogens
included, onto the real atoms (constant: a link sits at a fixed fraction of
its cut bond). One QM Hessian then replaces 6 QM/MM gradients per QM atom,
and no displaced SCF adds noise. Otherwise (electrostatic embedding, other QM
engines) it is finite differences of the QM/MM gradient (6 gradients per QM
atom). Partial Hessians for frequencies take the same route.
Of its negative modes it climbs the one along the path's tangent; any others
are made positive. Sella's model Hessian with only the tangent as a first
guess lost the reaction mode within a few steps on the environment-relaxed
surface and slid into a minimum. Pass `exact_hessian: false` in the TS
keywords to skip the Hessian.

After the search (`mepd run --use-tsopt`), the TS is checked: a second
QM-region Hessian there, whose imaginary mode must overlap the path tangent
by at least 0.3, and, after the IRC, its two ends must be different QM
species. A TS that fails says why (`✗ ...` in the log, and
`<label>_check_failed.txt` beside it, shown in the web results); it is kept,
not deleted, so you can look at it. After FSM-TS, `--use-tsopt` takes the TS
FSM-TS already optimized and checked (one imaginary mode, pseudo-IRC joining
the ends) and runs only the IRC from it.

Checked on Menshutkin NMe3 + CH3Cl in 56 moving TIP3P waters: B3LYP/def2-SVP
(Psi4) TS C–N 2.13 / C–Cl 2.23 Å, 15.0 kcal/mol above the IRC's reactant end
(gas phase 33.6), 33 min for TS and IRC; GFN2-xTB (`engine_name = "xtb"`)
the whole `mepd run --use-tsopt --irc` in 3 min, TS C–N 2.04 / C–Cl 2.12 Å,
9.4 kcal/mol. Both IRCs connect the path's reactant and product.

When a path search's output goes to a file instead of a terminal, every step
is logged as one line: step, gradients, the peak (kcal/mol, image) and the
end energy relative to the start, and the time the step took.

## Reactions in a protein

The species (a molecule, or a complex of several) is docked into a protein,
you choose the site, and the system built there is a QM/MM system like a
solvated one (`mepd/qmmm_protein.py`):

```bash
# 1. Prepare the protein (chains kept, ligands and crystal water removed,
#    hydrogens at --ph) and dock the species rigid, in its own geometry,
#    with AutoDock Vina in 20 Å boxes covering it; poses grouped into sites.
mepd qmmm protein-sites chorismate.xyz --protein 2CHT --chains A,B,C --charge -2 -o cm_sites
# 2. The species at a site as a QM/MM system: the protein (AMBER ff14SB) and a
#    TIP3P water shell are the environment, their charges act on the QM
#    region. Side chains can join the QM region; --end/--ts bring a reaction
#    along (put into the same site, as `mepd qmmm reaction` does in solvent).
mepd qmmm protein-build cm_sites --site 0 --qm-residues "A:ARG90" --end prephenate.xyz -o cm_site0
```

* `--protonation "A:ASP103=ASH"` (protein-sites) sets a residue's state by
  hand instead of by `--ph` (AMBER names: ASH, GLH, LYN, HID, HIE, HIP,
  CYX), e.g. a buried catalytic acid that pH 7 would deprotonate. A side
  chain in the QM region counts with the charge of the state it is in.
* Each site lists its distinct poses (`poses` in sites.json: more than 2 Å
  RMSD apart, best Vina score first, the protein-sites output prints them);
  `protein-build --pose k` builds that one. A molecule that looks alike from
  both ends docks either way round in the same pocket, and Vina's scores
  cannot tell them apart: androstenedione in ketosteroid isomerase (1OH0)
  scores -9.1 kcal/mol turned end to end from the crystal's equilenin and
  -8.7 the crystal's way. Compare such poses in one environment with QM/MM
  (put each pose's coordinates into the same system and minimize) or check
  them against a crystal ligand.

* Electrostatic embedding: the QM level must take point charges
  (`engine_name = "xtb"` for GFN2-xTB, or `"psi4"`); the written profile uses
  xtb.
* The force-field atoms come first (the PDB's order: protein, then water),
  the species last. It has no force-field parameters: its atoms get UFF
  Lennard-Jones only, and its electrostatics are the QM calculation's.
* Moving: water and protein atoms within `--active-radius` (6 Å) of the QM
  region; everything else is frozen (`--freeze-protein`: the whole protein).
  The frozen protein still acts on the QM region through its charges.
* The force field is cut off at `--cutoff` (12 Å, reaction field); the QM
  region feels every charge regardless. Between QM steps the environment is
  relaxed on a sub-system (the moving atoms and everything within the cutoff
  of them), whose forces on the moving atoms are exactly the full system's.
* Counterions (`--counterions`, on by default): the force-field charge that
  acts on the QM region (protein and water, without side chains that are
  QM atoms) is neutralized with one Na+ per excess negative charge and one
  Cl- per positive one, each beside a charged group (Asp, Glu, C terminus / Lys,
  Arg, N terminus) at least `--ion-min-distance` (10 Å) from every QM atom,
  nearest first, so they screen the charges that act most on the QM region.
  The ions are AMBER atoms of the environment (frozen beyond the active
  radius). Without them the protein's net charge acts on the QM region
  unscreened: the KSI dimer (1OH0) is -12, and its dienolate came out far too
  stable. The command reports the environment's charge before and after and
  the system charge; ions that find no group far enough away are reported
  and left out (lower `--ion-min-distance`), never put by the QM region.
* At build time, the protein's hydrogens and the water are minimized around
  the species (GFN2-xTB charges on it; protein heavy atoms held).
* Docking is rigid on purpose: the species keeps the geometry it has (a
  reactant optimized with QM, a pre-reactive conformer). Vina's score ranks
  sites; the QM/MM energies that follow are what count.

Checked on chorismate mutase (*B. subtilis*, PDB 2CHT, one trimer, 5663
atoms): the three best sites of 20 are the three active sites (1.9–3.5 Å
from the crystal's transition-state analog), lined by Arg7, Glu78 and
Arg90; one QM/MM gradient (GFN2-xTB, 24 QM atoms, 5.7k point charges)
takes 0.1 s.

The reaction, chorismate → prephenate at GFN2-xTB (24 QM atoms; results in
`qmmm/external/chorismate_mutase_gfn2`):

| | TS: C4–C8 / O5–C6 (Å) | barrier (kcal/mol) |
|---|---|---|
| gas phase (dianion) | 1.92 / 1.56 | 31.1 from extended chorismate |
| enzyme, NEB → TS → IRC between the minimized ends | 1.95 / 1.61 | 8.3–9.4 |
| enzyme, the gas-phase TS re-optimized in the site | 1.93 / 1.58 | 10.8 |
| water (TIP3P, 258 waters), the gas-phase TS re-optimized there | 1.88 / 1.59 | 23–31 |

In the gas phase the reactive (pseudo-diaxial) conformer is 14.7 kcal/mol up
and not a minimum; in the active site chorismate minimizes to it (C4–C8
2.9 Å), part of how the enzyme lowers the barrier. These are potential
energies from single minima: the environment's local minimum moves them by
1–2 kcal/mol in the enzyme (compare its two rows), and by up to ~10 in
water, whose 107 moving molecules have many local minima: 23 from the
near-attack chorismate in the TS's own water, 31 from extended chorismate
minimized separately. In water, the NEB route's IRC was too noisy to connect
the ends. Free energies need sampling.

In the web UI: select the species, *Place in a protein (QM/MM)* (a PDB ID
or a file path); the result shows the protein with the species at every
site, best first. Pick a site, click residues to add their side chains to
the QM region, optionally bring one of the species' reactions along, and
*Build QM/MM system here*: the system joins Explore as a QM/MM node (and,
with a reaction, its other end and the edge), minimized embedded.

### The nanoreactor in a protein (or solvent), with QM/MM forces

```bash
# 8 trajectories, 4 at a time: the QM region at 2500 K inside the protein
# (moving shell near 300 K), 5 ps each with the piston; products pooled.
mepd qmmm reactor site0/minimized.xyz -i site0/qmmm_profile.toml --seeds 8 --workers 4 \
    --temperature 2500 --time 5 -o reactor
# then optimize, classify and connect them in the same system:
mepd discovery expand site0/minimized.xyz -i site0/qmmm_profile.toml --products reactor/products.xyz
```

Each seed is its own process (the QM/MM engine is rebuilt from the profile),
with its MD, live events and products in `seed_<k>/`. A product is the whole
system right after an event, unless its frame changed a bond in the MM region
or stretched a cut bond (both counted in summary.json). The QM level runs at
`--etemp` (3000 K) during the MD, as the vacuum nanoreactor does, so a bond
breaking homolytically does not stop the SCC; products are re-optimized at
the profile's own settings. Events are found among the QM atoms only.

Ketosteroid isomerase (1OH0 chains A+B, Asp103 as ASH) with
androst-5-ene-3,17-dione and Tyr16, Asp40, Asp103, Tyr57 in the QM region
(94 QM atoms with link H, GFN2-xTB, 3951 atoms): one QM/MM gradient 1.4 s,
one MD step about 0.2 s on one core. 8 seeds x 5 ps gave 39 distinct
products; after `expand`, the clean ones were the enzyme's own chemistry
(Asp40's H to C6: the Δ4 product; Asp40's H back to C4; O3 protonated by
Tyr16, Asp103 or Tyr57), the rest 2500 K damage 20-56 kcal/mol up.

#### Steps between the products

`protein-build` writes a profile for this: FSM-TS in a caged environment
(`qmmm_environment = "cage"`, NEB as FSM-TS's fallback).

```bash
mepd run --start expand/species/species_0.xyz --end expand/species/species_35.xyz \
    -i site0/qmmm_profile.toml --recursive --use-tsopt --irc -o step2
```

- Why the cage: with the shell relaxed, a TS search can let the environment
  relax under the reaction coordinate and end on an environment saddle (KSI
  step 1: a Tyr57 OH rotation 9 kcal/mol below the NEB peak, both IRC ends
  the same species). In the cage every energy of a step, the TS and its IRC
  share the start's environment.
- Speed: each caged FSM-TS search took under 25 min for the KSI steps (FSM
  ~7 min, then the TS on the QM atoms with an exact QM-region Hessian of 540
  gradients at 0.2 s each, then the pseudo-IRC).
- If xtb's SCC needs its fallback anywhere along a path, set one electronic
  temperature for the whole run (`[xtb_engine_kwds] electronic_temperature =
  1000`). Otherwise
  images sit on different surfaces, and one KSI NEB blew up.
- `--use-tsopt` re-optimizes the TS from the path's highest image even after
  FSM-TS has converged one. For KSI step 1 that second search slid to a wrong
  saddle. Take FSM-TS's TS (the path's middle node) and run `mepd ts --guess
  ... --irc` on it.

KSI results (GFN2-xTB/ff14SB, caged, etemp 1000 K; barriers from the
minimized IRC ends, each with one imaginary mode):

| Step | Barrier | ΔE (kcal/mol) |
|---|---|---|
| Asp40 takes the C4 H (Δ5 → dienolate) | 0.7 | −3.0 |
| Tyr16's H to O3 (dienolate ⇌ dienol) | 0.1 | +0.1 |
| Asp40 gives the H to C6 (dienolate → Δ4) | 13.0 | +13.0 |

The barriers are far from experiment's ~10-12 kcal/mol for step 1, and the
product lies above the intermediate, which experiment does not show. 1OH0 was
crystallized with an intermediate analog, and the protein's −12 charge acts
without counterions; the dienolate is over-stabilized at B3LYP-D3/6-31G* too.

## The environment along a path: relaxed, cage, or free energy

`qmmm_environment` in a profile (Settings › Advanced › QM/MM systems) says
what the environment does while a path is searched:

- `relaxed` (default): the moving shell relaxes with every structure. Each
  image sits in its own local minimum of the solvent, so a path's energies
  include solvent rearrangements that have nothing to do with the reaction.
- `cage`: the whole environment frozen; only the QM region moves, in the
  environment's field. The usual approximation in a protein pocket. Every
  image sees the reactant's solvent arrangement, which can favour the start.
- `mean_force`: the QM region's free-energy surface (potential of mean
  force). At each geometry the solute is held and the environment is run as
  dynamics (`[mean_force]`: `temperature`, `equilibrate_ps`, `sample_ps`,
  `frames`) -- xtb's for GFN-FF/GFN1/GFN2 environments, OpenMM's for AMBER
  and TIP3P; the gradient is the mean force on the solute
  (Hu, Lu, Yang, JCTC 3, 390 (2007), doi:10.1021/ct600240y). Any path
  method relaxes in that space (NEB, FSM, GSM, MLP-GI). When the path is
  done its energies are replaced by the free-energy profile: the mean force
  integrated along it, sampled again inside each segment (Simpson's rule,
  panels doubled per segment until its integral changes by less than 0.5
  kcal/mol: a bond breaking within one segment needs several). During the
  search a path's images get the mean force integrated along it (trapezoid
  rule), so tangents and the climbing image follow the free energy.

```toml
qmmm_environment = "mean_force"
[mean_force]
equilibrate_ps = 0.5
sample_ps = 1.0
frames = 10
```

HCN → HNC in a 93-atom water shell (GFN2-xTB QM region, GFN-FF water, NEB
with 8 images and 30 steps), kcal/mol:

| | barrier | reaction |
|---|---|---|
| gas-phase QM along the same geometries | 73.2 | 20.3 |
| cage | 77.3 | 25.8 |
| free energy (mean force) | 74.9 | 23.1 |

Electrostatic embedding (TIP3P, or AMBER with `embedding =
"electrostatic"`): the QM energy depends on where the environment's charges
are, so the dynamics see the QM region as fixed point charges (the QM
program's atomic charges at that geometry; Hu, Lu and Yang use ESP charges)
and every sampled frame gets a full QM/MM gradient in its own field: `frames`
QM calculations per geometry. With mechanical embedding the QM energy of the
held solute is the same in every frame: one QM gradient per geometry.

Menshutkin reaction, NH3 + CH3Cl -> CH3NH3+ Cl- (contact ion pair), GFN2-xTB
QM region in a 300-atom TIP3P droplet, electrostatic embedding, NEB with 12
images and 120 steps, 1 ps of TIP3P dynamics and 10 QM gradients per
geometry (8 minutes on 8 cores), kcal/mol:

| | barrier | reaction |
|---|---|---|
| gas-phase QM along the same path | 42.4 | +31.3 |
| cage (water as around the reactant) | 45.7 | -5.3 |
| free energy (three profiles of the same path) | 20 ± 1.3 | -13 ± 1.5 |

The barrier is in the 20-30 kcal/mol range of earlier QM/MM free-energy
studies and sits earlier along the path than in the gas phase (C-N 1.9 Å,
C-Cl 2.3 Å). The spread between profiles is the sampling noise of 1 ps
per geometry; sample longer for tighter numbers.

Cost (mechanical): one QM gradient per geometry plus `equilibrate_ps +
sample_ps` of environment dynamics (about 20 s for 1.3 ps of 230 atoms of
GFN-FF on one core; OpenMM is much faster) and `frames` low-level gradients;
images run `n_parallel` at a time. Limits: no Hessians, TS optimizations or
IRCs on this surface (use the
climbing image, or the cage for a TS optimization); minimizations and
elementary-step checks see a noisy surface (switch `do_elem_step_checks`
off for a single path); the forces are averages over `frames`, so their
noise (about 2e-3 Eh/bohr at the defaults) limits how far a path converges.

## Restraints

Harmonic distance restraints hold a distance while everything else
optimizes (QM atoms cannot be frozen), e.g. to test whether a Michaelis
complex is a minimum with its C-H kept at 1.10 Å:

```toml
[[restraints]]
atoms = [3951, 3967]      # 0-based
distance = 1.10           # Å
force_constant = 500      # kcal/mol/Å² (default)
```

They go into the QM/MM energy itself, so microiterative optimizations hold
them; on QM/MM they may involve only QM atoms and link partners.
`mepd optimize` reports each restraint's distance and energy. Any engine
takes them (`mepd/restraints.py`).

## Channels on a QM/MM step

`mepd channels -i <QM/MM profile> --start a.xyz --end b.xyz` (whole systems,
the same atoms) samples conformers of the QM region in its environment:
Langevin MD of the QM atoms with QM/MM forces, the environment held
(`--qmmm-temperature`, 400 K; `--qmmm-md-ps`, 1 ps), frames whose QM bonds
are still the start's minimized with the profile's engine, then minima told
apart by the QM atoms' RMSD (`--rmsd-cutoff`) and capped at
`--n-conformers`. Every reactant x product conformer pair is then searched
as in solvent (environment matched and marched along the path), with TS and
IRC per leaf. No atom mapping (both ends are one system) and no mirror-image
merging; `--pair-from` must be `both`.

## Checking that a result is not nonsense

`mepd qmmm inspect` (and the web UI's *QM/MM checks*) report, frame by frame:

| check | should be |
|---|---|
| frozen atoms moved | 0 Å |
| bonds made or broken in the MM region | none (the force field cannot describe a reaction) |
| cut-bond stretch | near 1× the covalent length |
| closest non-bonded QM–MM contact | not a clash (≳ 1.5 Å) |
| QM / environment RMSD from frame 0 | where the motion is |

With `-i`, each frame's energy is also split into the QM region and its
environment. This shows whether a barrier comes from the reaction itself or
from the solvent or protein around it.

## Web UI

* **Put in explicit solvent (QM/MM)**: an operation on any molecule. It
  builds the solvent shell, adds the system to Explore as a QM/MM node, and
  minimizes it embedded.
* **Model in solvent (QM/MM)**, on a gas-phase edge or two gas-phase ends
  (same atoms, same order): the start is solvated, the end put into that
  same solvent shell, both minimized embedded and joined by a QM/MM edge.
  If the edge has an IRC-verified TS, it is put in too and re-optimized as a
  TS in the solvent, with its IRC.
* **Put into this QM/MM system**, on a QM/MM structure plus a gas-phase
  structure of its solute (e.g. a product or a TS from the gas phase): the
  structure goes into that system's solvent in place of the solute; a
  minimum is minimized embedded and joined to the QM/MM structure by an
  edge, a TS is re-optimized as a TS with its IRC.
* **Add › QM/MM system**: upload a whole system (XYZ/PDB) with its QM atoms,
  or point to a TeraChem input on the server.
* **Define a QM region (QM/MM)** (Inspector, any structure) and **Edit the QM
  region** (QM/MM structures): click atoms in 3D to add or remove them, and
  set the QM charge, spin, moving shell and environment level. The cut
  bonds, moving and frozen counts and any problems update as you click.
* A QM/MM node is named by its QM region ("NCC(=O)O · in water"). Only the
  operations that run embedded are offered (optimize, TS search, TS
  optimization, Hessian sampling, network expansion, paths between
  isomers). Its energies are at their own level ("… / QM/MM", keyed by the
  region), never compared with gas-phase ones or with another region's.
* **3D views** of a QM/MM system, wherever one is shown (Explore, results,
  live paths, optimization trees), draw the QM atoms as balls and sticks,
  cut bonds as dashed orange bonds with their link hydrogens, the moving
  environment as sticks and the frozen environment as grey lines. You can
  show all, the moving part only, or the QM atoms only. *Motion* colours the
  environment by how far it moved from the first frame; a frozen atom that
  moved turns magenta.
* **QM/MM checks** under every result path: the table above, per-frame
  curves, and *Split energies: QM vs environment*.

## Limits

* Electrostatic embedding (the environment's charges polarizing the QM
  region) needs fixed environment charges (TIP3P water, AMBER) and a QM
  level that takes point charges: `engine_name = "xtb"` (GFN2-/GFN1-xTB,
  xtb's own embedding, ~0.05 s per gradient for an 18-atom QM region in
  ~320 water charges), `"psi4"` (any Psi4 method; B3LYP/def2-SVP ~8 s) or
  `"terachem"` (see *TeraChem* below). xtb retries an SCC that does not
  converge with `--etemp 1000`, then `--etemp 3000` (1000 iterations; the
  log says when). Once a temperature was needed, the engine holds it for
  every later calculation and computes the rest of that batch (a path's
  images, a Hessian's displacements) again at it, so a path never mixes
  surfaces; `xtb_engine_kwds.electronic_temperature` sets one from the
  start. A smeared energy differs from the 300 K one where the gap is
  small: check the log before comparing such energies.
  g-xTB and the GFN-FF environment are mechanical embedding only, so
  charged or zwitterionic QM regions in polar environments are described
  less well with them.
* The boundary is fixed: atoms do not move between the QM and MM regions
  (no adaptive QM/MM).
* Implicit solvent cannot be combined with QM/MM.
* Network expansion's whole-molecule generators (CREST msreact, the
  nanoreactor, an imported generator, a products file of the QM model) run
  on the QM model:
  the QM atoms capped with their link hydrogens, at the QM charge and spin.
  Each product is aligned back onto the QM atoms, relaxed briefly against
  the nearby environment and put into the system, which the QM/MM
  optimization then relaxes. A product that breaks a cut bond (its link
  hydrogen no longer caps its QM atom) is dropped, and the run says so.
* Not offered on QM/MM systems in the web UI (they need whole molecules):
  reaction channels (on the command line they run, see above), conformers, complexes,
  retrosynthesis, VRI, solvent/substituent follow-ups. The Nanoreactor
  operation (several molecules, reactions cut out with only the molecules
  they need) is replaced on a QM/MM system by the `nanoreactor` generator of
  network expansion: hot piston MD of the capped QM region in vacuum; its
  state after each reaction event goes back into the environment. The
  environment is not present during that MD unless `embedded=true` (web:
  "MD with QM/MM forces"), which runs the hot MD with QM/MM forces and the
  solvent held near room temperature; see network-expansion.md.
* The TeraChem routes below are tested against a stand-in that writes
  TeraChem's input and output formats, not yet against a live TeraChem:
  the installed docker image has no GPU code for this laptop's RTX 5090
  (Blackwell, SM 12.0). Run the live check once a Blackwell build is in
  place (see below).

## TeraChem

`engine_name = "terachem"` runs a local TeraChem as the QM level:

```toml
engine_name = "terachem"

[terachem_engine_kwds]
method = "b3lyp"          # TeraChem's names; "u" is added for open shells
basis = "6-31gs"
# command = "terachem"    # default: $MEPD_TERACHEM, else terachem on PATH
# timeout = 3600          # s per run
[terachem_engine_kwds.keywords]   # any other tc.in lines
dftd = "d3"
```

* The command can be a native install or the docker wrapper for the
  mtzgroup/terachem images. The license is read from
  `$TERACHEM_LICENSE_PATH`; when that is unset, mepd uses
  `~/.config/terachem/license.key` (a `key ...` / `server host:port` file).
* Each gradient is one TeraChem run. The orbitals of the previous run of the
  same molecule (`c0`, or `ca0`/`cb0`) are the next run's `guess`.
* Electrostatic embedding: the environment's charges go to TeraChem as a
  `pointcharges` file (`charge x y z` in Å, with
  `pointcharges_self_interaction false`, because the force field has the MM–MM
  part). The gradient on the charges is read from TeraChem's "MM / Point
  charge part" block, so TeraChem is a QM level for TIP3P water and the
  protein workflow, the same as xtb and Psi4. The Mulliken charges
  (`scr/charge_mull.xls`) are what the environment sees during
  microiterations.
* TeraChem's own QM/MM (`[qmmm] mm = "terachem"`, prmtop + rst7, e.g. from
  `mepd qmmm from-tc tc.in`) runs locally with `engine_name = "terachem"`
  (method/basis from `terachem_engine_kwds` unless the region has a tc.in
  template). With other engines it goes through ChemCloud. Its
  optimizations, TS searches and IRCs run through ASE/Sella on TeraChem
  gradients.
* Each run starts a new TeraChem process. With docker, that adds container
  start-up and a license check to every gradient. TeraChem's server mode
  (TCPB) would remove that, but its protocol carries MM point charges only
  for interactive MD, so it is not used yet.
* Live check: `MEPD_TERACHEM_LIVE=1 pytest tests/test_terachem.py` checks,
  by finite differences on water with two point charges, that TeraChem's
  energy and its gradients on the atoms and the charges agree (units, signs,
  the point-charge file).
