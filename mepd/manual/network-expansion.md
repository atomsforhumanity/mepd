# Network expansion

A network expansion (`mepd discovery expand`) grows a reaction network from one
species: it proposes what that species can turn into, finds a transition state
for each proposed reaction, decides from the barriers which new species to grow
from next, and repeats. In the web app each of those is a stage you choose on
its own. Choices that cannot run together are greyed out with the reason, and a
stage moves to the nearest allowed choice when another rules it out.

## Grow

How each species' possible products are proposed. Everything after that
(minimization, species identity, the network) is the same for every choice.

- **Bond enumeration.** Every combination of up to *n* bonds broken and *m*
  formed between nearby atoms, kept if a Lewis structure fits (the ZStruct /
  YARP enumeration, reimplemented). Fast and systematic, blind to energy:
  fewest bond changes first.
- **CREST msreact.** Biased GFN2-xTB optimizations find the fragments and
  isomers a molecule can reach. One molecule at a time.
- **Collision MD.** Langevin MD that pulls two atoms (or two molecules)
  together for a moment, over many pairs: decompositions, rearrangements,
  additions.
- **Reactor MD.** Langevin MD in a periodically squeezing wall with random
  collisions; nothing picks the chemistry. It runs batches of trajectories until
  few reaction events give a product not seen before (5 %). In the live queue
  each batch's products join the network at once, while the next batch runs.
- **Nanoreactor** (QM/MM systems). Hot MD of the QM region in a squeezing wall,
  each product put back into its environment.

A species no Lewis structure fits (an ylide, a hypervalent atom: "unusual
bonding") is kept and drawn from its geometry.

## Connect

How each proposed reaction gets its transition state.

- **Fast** (with *Material flow, within reach*). One shared job queue: a TS optimized from the
  peak of a geodesic interpolation, and where that fails a path search whose
  pieces are queued as it splits. IRCs run for the steps the kinetics need; a
  step counts only once its IRC ends are its two species.
- **Path search.** A recursive path search, TS optimization and IRC for every
  proposed reaction. Thorough, slow.
- **No TSs** (with *Energy window*). Products only, to connect later by hand.

**Sample mechanism channels** runs `mepd channels` (conformers of both ends,
several atom mappings) on the steps the kinetics need; their lowest channels
replace the single TS.

## Steer

Which species have their products proposed next.

- **Material flow, within reach** (the default). Microkinetics from the start,
  refreshed as the network grows, decide what grows: a species once its peak
  concentration or the flux into it reaches the threshold. It must also be
  reachable: some route from the start gets there with every barrier under a
  cutoff, the barrier that reacts within the cutoff time at the cutoff
  temperature, plus 10 kcal/mol for the method's error (about 33 kcal/mol at
  298 K over an hour). The cutoff keeps the run from growing behind barriers
  nothing could cross in that time.
- **Material flow, no cutoff.** The same flux rule without the cutoff, in
  rounds, with a full path search for every reaction: the older, slower mode,
  without the live queue or channel sampling.
- **Energy window.** Every new species within a window of the start's energy;
  no kinetics, no TSs needed.

Experimental results never steer a run: they are only for checking it after
the fact.

## Run

- **Live queue** (with *Material flow, within reach* and *Fast*). No rounds: a species grows
  the moment the live kinetics reach it, and every proposal batch, TS, IRC and
  channel search runs as soon as it is asked for. It ends when the kinetics ask
  for nothing more, the time budget runs out, or you press Stop & analyze. Its
  page is the live network.
- **Rounds.** Propose for the species the last round reached, connect, check,
  repeat.

## Limits and possible improvements

- Only the Reactor MD hands over products batch by batch; collision MD, CREST
  msreact and the nanoreactor return all their products at once.
- A species interrupted between its MD batches starts its batches again on
  resume (with the same seeds, so its earlier products come back as known).
- Energy-window growth sees no barriers: it can grow far from where any
  material would go.
