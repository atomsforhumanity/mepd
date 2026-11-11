# Explore: the network map

Explore opens on a map of the workspace's network: its molecules and the
steps between them, drawn so that a network with hundreds of reactions stays
readable. *Graph* (top right) switches to the editable graph, which you need
for connecting two structures, arranging by hand, experimental conditions and
network variants. The choice is remembered.

## One arrow per step

A step such as glycine → methylamine + CO₂ is drawn as one arrow between the
two molecules it is about. On each side that is the molecule taking part in
the fewest steps; a molecule added by hand comes first, and if two tie, the
larger one. The other molecules ride on the arrow as tags: **+** taken in,
**−** given off, **via** for one given back unchanged, such as a water relay.
In a network grown from glycine, H₂ takes part in 145 of 237 steps and water
in 108. Drawn as nodes, they joined nearly every step to the same two points.
As tags they leave the picture. Molecules that are only ever tags are listed
under **Riding along**; click one to light up its steps.

## Columns: the easiest way in

The overview stands each molecule in a column by how hard it is to reach. Of
all the routes from the start that go through searched steps, mepd takes the
one whose highest barrier is lowest (if two tie, the one with fewer steps),
and reads off that highest barrier. The columns are 5 kcal/mol wide.
Molecules no searched route reaches stand in their own column. Where the
network starts is the molecules added by hand. If nearly all were, it is the
ones that no step makes.

Rows are ordered so that molecules joined by a step sit near each other. A
crowded column folds into staggered sub-columns, as many as make the drawing
the shape of the canvas.

## The barrier scale and filter

Each arrow's colour and width follow the step's forward barrier, on one scale
that spans the workspace's own barriers: the lowest barrier is dark and thick,
the highest pale and thin. The histogram above the map counts the barriers in
5 kcal/mol bins and is also the filter. Drag it, and steps above the limit
fade, along with the molecules they alone reach. Steps that were not searched
yet (proposals of a network expansion), and steps searched without a TS
joining their ends, are hidden until you tick them. A step being searched is
drawn dashed in blue.

## A molecule's neighbourhood

Clicking a molecule selects it (the inspector offers its calculations) and
opens its neighbourhood. What it is made from stands on the left, what it
turns into on the right. Each partner's label gives the step's barrier, the
number of steps between the two when there are several, and its tags. A
dashed arrow back means the reverse step is known too. Clicking a partner
walks on, the trail above the map leads back, and **Esc** returns to the
whole network.

**Show route** lays out the easiest way in from the start as a strip, each
step with its barrier and tags.

## Limits and possible improvements

- **The arrow is a choice of drawing.** A step making two molecules of similar
  size is drawn to the one in fewer steps, and the other becomes a tag. It is
  still counted for routes, and the step's page shows the full equation.
- **The forward barrier only.** The scale and the routes use each step's
  forward barrier, at the workspace's level when the step has one. A step
  known only in reverse counts once the reverse is searched.
- **Barriers, not rates.** The columns rank by barrier, not by a rate at given
  conditions; Kinetics and Landscape answer what forms.
- **Layout is recomputed, not saved.** Positions follow the data. To arrange
  molecules by hand, use the editable graph.
