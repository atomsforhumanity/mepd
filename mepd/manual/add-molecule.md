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
cancel it there), one candidate after another.

## A test as it runs

A test has three stages. They may run one after another or at the same time
(the scans start while steps are still being verified), so the page shows
each as a box of its own, with its state, its progress and the time its first
number came:

- **Quick estimate.** The molecule is placed beside the two ends and the
  transition state of every step, the step's own atoms held: seconds per
  step. A row from it is an *estimate*: drawn hollow and dashed, in muted
  figures, with the uncertainty the result file gives for such estimates
  (how far they lay from the verified value where both were computed).
- **Verification.** Each step's transition state is optimized with the
  molecule beside it and its IRC is run. A verified row replaces the step's
  estimate (*verified · beside*); its spread is the difference between its
  two poses. A verification that leads to other species is listed with the
  species each of its poses joined (*led to other species*), and the step
  keeps its estimate.
- **Scan.** A search for transition states in which the molecule itself
  takes part (a proton relay, an exchange): one scan per reaction of the
  network (it stands for every step between the same molecules), several
  side by side, in the order of what the target would gain. Its rows read
  *verified · takes part*. *Scans* in the form sets how many reactions are
  scanned (0: none).

**The route's highest point as the test went** is the curve to watch: the
highest point of the easiest route to the target with the molecule, stepping
down as rows arrive, against the same without the molecule (dashed) and
what the conditions allow (the line the target forms under).

The page reads the result file every few seconds. While a test runs the step
table keeps one order (the steps that limit the target first, then by step),
so a row that is replaced by a lower one stays where it is and is lit for a
moment; once the test is over the largest change comes first. *Scans for the
molecule taking part* lists every scan: the reaction, its lane, its state,
its searches, how many transition states with the molecule it found, the
lowest, and how many led to other species.

**Replay.** Every row of a result carries the time it arrived, so a finished
test can be played again: the bar under the candidate's name (or a click in
the curve) shows the page as it stood at that time, from the rows that had
arrived and the kinetics run again on them. A row that a lower one later
replaced, and the estimate of a step that was later verified, are not in the
final file and so not in the replay.

**The answer comes first**: the easiest route's highest point without and
with the molecule, the target's amount with the interval it has if every
number with a spread is off by its spread, what became of the molecule, and
**Not scanned**: the steps that were wanted for a scan and got none, with
the reason. While anything is still running, rests on an estimate, or was
not scanned, the page says so (*Not a verdict yet*; for a molecule that
already helps, *Not the whole picture yet*): "no effect found" is then not
"no effect". Under it, one sentence says what the answer is: what this
network gives at its level of theory, not a prediction of an experiment; a
route the network does not hold is not in it.

A verified transition state opens in 3D from its own file, beside the result
file or in the `files.tgz` a remote run leaves beside it. Where the file is
gone, the page shows the network's own transition state and says so, with
the path.

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
  at the end. *Used up*: it ends in new species. *Held, inert*: it
  sits in complexes and takes part in no step. *Takes no part* is said only
  when none of these holds. Below it, the species that holds most of the
  molecule over the run.
- **Largest step change**: the step whose barrier changes most through the
  molecule, with its spread. A change no larger than its spread is flagged:
  it is not a signal.

**Built on** says what the row's kinetics rest on: **est.** (estimates) or
**verified** (only transition states optimized with the molecule present, IRC
checked); the step's own number has its own tag. A row is two lines, a
verdict and one fact; hover a cell for the rest (the route share, where the
molecule rests, the step and its barriers), or open the detail.
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

- the steps that matter for the target (those that limit it, and those on
  its easiest route with the molecule) and whose fastest version goes
  through the molecule are coloured by the change of their barrier, blue
  lower, red higher, dashed while that is an estimate and solid once
  verified; the legend counts the other steps that go through it;
- everything else is faded, so the few marked steps can be found; *Details*
  in the legend goes back to the candidate in the list, ✕ to the network
  without it;
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

- New species the molecule forms (a covalent intermediate that is no version
  of a step of the network) are not searched for; such numbers can come with
  a candidate (`estimates` in a candidates file) and are shown with their
  source.
- A replay runs the kinetics again for each time shown: seconds per frame on
  a network of a few hundred steps.
- Steps of an exploration's own network are drawn on the map through the
  workspace's edges; a step the workspace keeps no edge for (a second
  transition state of the same pair) is counted but not drawn.
- Kinetics' *Network* menu does not list added-molecule variants: their
  kinetics are on this page.
- A question is asked of the parent network, not of a substituent variant.
