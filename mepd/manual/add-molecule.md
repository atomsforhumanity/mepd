# Add a molecule

You have a reaction network and a product you want more of. **Add a molecule**
(the tab beside Kinetics and Landscape; or select a species in Explore and
choose *Make more of it*) answers: what would one more molecule in the flask,
a possible catalyst or an additive, do to that product?

The page reads files that `mepd addmol` writes ([the format](https://github.com/atomsforhumanity/mepd/blob/main/docs/add-molecule.md)),
so everything it shows can also be made, and read, without the web app.

## The question

Pick the **target** (the species you want more of) and the **conditions**: the
starting mixture, the temperature and the time. They start from what
Kinetics last ran, else from what the latest exploration started from, 1 M
each. A target that exists as several stereoisomers is several species of the
network; they are counted together unless you untick the box.

Which network is asked depends on where the workspace's network came from
(the line under *What limits it today* says which):

- **One finished exploration** (every verified step is a step of the same
  Network expansion): the exploration's own network. Its species are
  molecules, its free energies are the run's own, and every IRC-verified
  step it found counts, also the ones it closed as too high. A question
  takes about half a minute at the exploration's own temperature (the
  thermal corrections it did not record are computed each time) and several
  minutes at another one (every correction is computed again).
- **Anything else** (steps from several calculations, a hand-built
  network, a question with a fixed amount): the workspace's network, the one
  Kinetics runs: one step per pair of species, free energies at the 1 M
  standard state from GFN2-xTB Hessians. The first question computes those
  Hessians (minutes for a hundred steps); they are kept, and later questions
  take seconds.

A question is kept in the workspace. *Earlier questions* opens one again with
its candidates.

## What limits the target

Before any candidate, the page shows what holds the target back today.

- **Where the network makes the target:** the degree of rate control of every
  transition state, as bars: what lowering it by 1 kcal/mol does to the
  target's amount. Blue steps are the ones a molecule should lower. Red steps
  compete with the target: a molecule that lowers them costs target. Below
  them the **traps**: species whose stabilization costs target, so a molecule
  that binds them works against you. Candidates are worked in this order.
- **Where it makes none** (less than 1e-9 of the starting material, which is
  integration noise) a degree of rate control is not defined. The page then
  draws the easiest route from the starting mixture to the target: how high
  each transition state stands above the mixture, against a line at the
  barrier these conditions allow (the Eyring barrier crossed with a
  half-life of the run's time). A molecule has to bring every step of some
  route under that line. Which steps compete cannot be said while nothing
  forms.

## Candidates

Type or paste one molecule per line: a SMILES, then optionally a name and an
amount in mol/L (`OC(=O)[C@@H]1CCCN1 L-proline 0.1`). **Load a file** takes
SMILES lines, an `.xyz` structure (every frame one candidate), or a
candidates file (JSON) as a generative model writes it, which may carry a
proposed pose beside a step and estimates from elsewhere. **From Design**
adds the molecule that is open in Design. What was read is listed, with a
reason for anything that cannot be used, before you press **Test**.

All candidates of one click run as one calculation (*Calculations* lists it;
cancel it there). They are tested one after another, each in layers, the
cheapest first. Today one layer exists: the **bystander** layer puts the
molecule beside the two ends and the transition state of every step, the
step's own atoms held, and records how strongly it binds to each. Its
placer is a stand-in whose numbers move by kcal/mol between repeats; that
spread is shown beside every number.

## Reading a candidate

Each candidate is one line, the largest gain of the target first. The page
reads the result files every few seconds while a calculation runs, so
numbers appear and settle without a reload.

- **Target**: its amount at the end with the molecule against without, as a
  factor. Where the network makes none today: the amount with the molecule,
  or, if that is still none, how far the easiest route came down.
- **How soon**: when the mixture with the molecule holds what today's
  mixture takes half its run to make.
- **Returned?** Two facts side by side: whether steps with the molecule carry
  the target's material, and how often each molecule came back in the run
  (turnovers). *Returned*: it came back at least as often as it is held up
  at the end. *Used up*: it ends in new species. *Held, does nothing*: it
  sits in complexes and takes part in no step. *Takes no part* is said only
  when none of these holds. Below it, the species that holds most of the
  molecule over the run.
- **Largest step change**: the step whose barrier changes most through the
  molecule, with its spread. A change no larger than its spread is flagged:
  it is not a signal.

Every number carries **est.** (built on estimates) or **verified** (built
only on transition states optimized with the molecule present, IRC checked).
A molecule that changes nothing is a result, and the page says so.

Click a candidate for its detail:

- the facts above with their numbers, and, where a degree of rate control is
  defined, the first-order estimate beside the full kinetics' factor;
- **Steps that go through the molecule**: each step's apparent barrier (the
  transition state above its reactants at the amounts they have in the run)
  without and with the molecule on one axis, the change, its spread, where
  the number is from (layer and level), and **3D**: the transition state,
  with the molecule where the estimate kept its geometry;
- folded below, the steps whose own transition state is unchanged but whose
  barrier moved because their reactants hold more or less material;
- **What is new**: steps in which the molecule takes part, and the complexes
  and new species that hold it, with their amounts;
- which layers are done, running or failed, and every calculation that gave
  no number, with the reason.

Tick several candidates to see them **side by side**: one row per step, one
column per candidate, the change of the barrier on a blue (lower) to red
(higher) scale.

## On the network map

**Show on the map** keeps the candidate as a variant of the network (Explore's
*Graph* view lists it in its Network picker) and opens the map with it:

- steps whose fastest version goes through the molecule are coloured by the
  change of their barrier, blue lower, red higher, dashed while that is an
  estimate and solid once verified;
- a molecule with a doubled border has a complex with the added molecule
  that holds material;
- new species, and steps with no counterpart in the network, are counted in
  the legend: they have no place on the parent's map yet.

The variant follows its result file when the calculation finishes. Deleting
it (✕ beside the Network picker, or in the legend to go back to the parent)
does not delete the result.

## For a generative model

*Download the query file* gives the question as one JSON file that needs no
mepd to read: conditions, every species with SMILES, coordinates and amounts
over time, every step with both ends and the transition state in one atom
order and the bonds that change, barriers, and what limits the target. A
model proposes candidates from it; test them here (Load a file) or with
`mepd addmol test query.json --candidates candidates.json -o results/`.
Result files written elsewhere and copied into
`<workspace>/addmol/<query>/results/` show up on the page like any other.

## Limits and possible improvements

- One layer of estimates exists, with a stand-in placer. Layers that verify a
  step with the molecule present, find steps in which it takes part, or find
  the new species it forms are not built yet; until then such numbers can
  come with a candidate (`estimates` in a candidates file) and are shown
  with their source.
- Steps of an exploration's own network are drawn on the map through the
  workspace's edges; a step the workspace keeps no edge for (a second
  transition state of the same pair) is counted but not drawn.
- The target's factor has no spread of its own: the spread shown is that of
  the steps underneath it.
- Kinetics' *Network* menu does not list added-molecule variants: their
  kinetics are on this page.
- A question is asked of the parent network, not of a substituent variant.
