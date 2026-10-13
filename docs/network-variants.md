# Network variants

Edit one species of a reaction network, and recompute every reaction it
takes part in, plus every reaction of what it becomes, with the edit in
place. Then compare the variant's kinetics with the parent's.

Example (`organocatalysis/1_co2_carbonate`): kinetics suggest that a more
stable epoxide gives more carbonate. To test it, open ethylene oxide in
Design, put a methyl on one hydrogen, and **Apply to network**. The variant
has propylene oxide, two bromoalkoxides (bromide attacks the substituted or
the unsubstituted carbon), two bromocarbonates and one propylene carbonate.
It also has every TS between them.

## In the web UI

1. In Explore, open the species with **Edit in Design**. Every atom keeps
   its index in that species (as its atom-map number), so Design knows what
   the edit changed. Open a single molecule, not a complex or a TS: a design
   opened from a complex offers buttons to open each of its molecules instead.
2. Edit it. You can swap hydrogens for groups (any `[*]SMILES`), change the
   element or charge of atoms (Design re-adds hydrogens), and combine both.
   **Across the network › Apply to network…** shows a preview, without
   calculating anything:
   - the new species;
   - per parent reaction, its variants: *recompute* (it has a TS to start
     from), *path search* (it has none), or *blocked*;
   - the reactions it could not reach, and why.
3. **Run** adds the variant's records to the workspace and starts the jobs.
   - New species are minimized.
   - Each variant reaction gets a fast estimate, then a re-optimization.
   - A reaction without a usable parent TS gets a path search at once.
4. Explore's **Network** picker shows the parent or a variant. In a variant,
   what it replaces is faded. An edge reads `est. +1.2` until its
   re-optimized TS is in. Update (reactions added since) and ✕ (delete) sit
   next to the picker.
5. To stack edits, pick a variant in Explore's **Network** first, then Edit in
   Design and Apply to network: the new edit is carried through that
   variant's reactions (from their re-optimized TSs) and becomes a variant
   built on it, e.g. choline for NMe4+ first, then iodide for bromide on top.
   A variant something else is built on cannot be deleted first.
6. Kinetics › **Network**: run the parent or a variant. **Compare with
   parent** runs both from the parent's mixture (a stacked variant runs
   against the variant it is built on): amounts of a replaced species go to
   its variant (split evenly over several). It shows end amounts and each
   step's barrier side by side.

## How the edit is carried

| Step | What happens | Code |
|---|---|---|
| Edit | The tagged design is read against the original: element/charge swaps on kept atoms, hydrogens removed, new groups (each hangs off one original atom and usually takes a hydrogen's place, paired by bond direction). Refused, with the reason: deleting heavy atoms, making or breaking bonds between original atoms, a group bonded to two original atoms, removing a hydrogen from an unchanged atom, or an odd change in the electron count. | `derive_edit` |
| Frames | A parent reaction's geometries in one atom order: the TS kinetics uses for its edge, with that job's IRC ends. Every job on the edge is searched, because "Sample more paths" runs keep their IRCs in their own folders. If the IRC does not run between the edge's two ends (the TS is one step of a longer route), or there is no TS, the edge's pinned ends are used. When those differ in more than 6 bonds, SLAPMapper renumbers the end. | `variants.parent_network` |
| Placement | The edited species is matched onto each fragment of the reaction's end that is that species, under every graph isomorphism. Only placements that rebuild the edited species with its configurations (or their mirror image) are kept. They are reduced to one per distinct substituted reaction: a canonical graph with formed and broken bonds marked, plus the configurations at both ends; mirror images count as one. Of placements that give the same reaction (e.g. which of NMe4+'s twelve hydrogens gets an OH), the one nearest the reacting atoms is kept, where a group can act on the reaction. | `propagate`, `reaction_key` |
| Geometry | The same edit builds the reactant end, the TS and the product end. New atoms are fitted on the anchor atom's neighbourhood (only neighbours still bonded to it in that frame; in the TS, bonds kept by both ends). A group that replaces a hydrogen lies along that frame's own X–H bond, at the design's bond length, turned to clear the rest. The replaced hydrogen's index becomes the group's first atom, and other new atoms are appended. | `apply_edit` |
| Onward | The fragments of the other end that hold edited atoms are variants of that end's species, keyed by connectivity, configurations (mirror images merged) and charge. Two routes that reach the same product share one node. Those variants then enter their own reactions, in both directions, until nothing new appears. | `propagate` |
| Blocked | A reaction where an edited hydrogen (replaced or removed) changes its bonds has no analogue in the variant. It is dropped and listed. | `_carry` |

A species that has variants is replaced in the variant network, together
with every parent reaction it takes part in. Reactions with no edited
species are shared. Limitation: a species that also forms without the edited
one is still replaced.

Identical variant reactions from two parent reactions are computed once.
For example, two parent records of the same step from different searches
both carry the edit to the same reaction.

## `mepd variant`

One variant reaction, from its spec (`<workspace>/variants/<id>/<edge>.json`;
Angstrom):

```json
{"symbols": [...], "charge": 0, "multiplicity": 1,
 "r": [[x, y, z], ...], "ts": [...] , "p": [...],
 "bonds_r": [[i, j], ...], "bonds_p": [...],
 "free": {"r": [...], "ts": [...], "p": [...]},
 "parent": {"symbols": [...], "r": ..., "ts": ..., "p": ..., "energies": [E_r, E_ts, E_p]}}
```

- `--mode fast` writes `<output>/fast/`. Only the `free` atoms (the new ones)
  are relaxed, then single points. It gives `barrier_shift` and
  `reaction_shift` against the parent's frames (their energies are
  recomputed when the spec has none). It reports `reacted` if the new atoms
  formed or broke bonds while relaxing.
- `--mode reoptimize` writes `<output>/reopt/`:
  - Both ends are minimized; they must keep `bonds_r`/`bonds_p` and their
    configurations, or the status is `ends_changed`.
  - The TS is optimized from the substituted TS (the fast one if it exists),
    then its IRC is run. The IRC's ends must have exactly the substituted
    ends' bonds (`other_minima` otherwise).
  - The barrier is measured from the lower of the minimized reactant end and
    the matching IRC end, as for `mepd ts`.

Kinetics uses a variant edge's re-optimized TS when it is verified. Before
that, it uses the fast estimate: the parent's TS height above its reactants
plus `barrier_shift`, added to the variant reactants' own energies. Steps
built this way are marked `est.`.

## Failure modes, and what happens

| Case | Handling |
|---|---|
| A symmetric species (two equivalent carbons) | Every placement; distinct ones become separate reactions |
| A constitution-only automorphism would give the wrong diastereomer | The placement is dropped (configurations are checked) |
| Conformers or representatives numbered differently from the edge | Only TS-job frames or the edge's pinned conformers are used |
| Ends from different sources, numbered differently | SLAPMapper renumbering (needs `mepd[aam]`) |
| An edited hydrogen is transferred | `blocked` |
| A group squeezed against the rest | It is turned about its bond. The preview shows `crowded` below 1.2 Å |
| A group that reacts while relaxing | Fast status `reacted`; re-optimization still runs |
| The TS re-optimizes to another saddle, or its IRC misses the ends | The parent's path search (`ts`/`channels`, its parameters) on the variant's minimized ends |
| The parent TS is one step of a longer route | Path search between the substituted ends |
| QM/MM species or reactions | Not carried (listed) |
