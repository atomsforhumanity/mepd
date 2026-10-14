// The map: mepd unfolds as it is discovered. Each quest opens tabs,
// operations and features once the quests before it are done, and is done
// itself the first time what it opened is used. Nothing is gated for good:
// a workspace's own history counts (its calculations complete the quests
// they belong to), and "Show everything" opens it all at once.
import { useStore, prefs, state, set, select, toast, update, openTab } from './store.js';

// opens: {tabs, ops, features}; uses: job ops that complete it (default: its ops);
// start: where "Try it" goes (select something, open an operation's card).
export const QUESTS = [
  // The first chain (Welcome.js steps).
  { id: 'molecule', short: 'Molecule', title: 'A molecule', blurb: 'Pick one, or build your own.', after: [], opens: { tabs: ['design'] }, start: { tab: 'design' } },
  { id: 'reactor', short: 'Reactor', title: 'The reactor', blurb: 'Heat it, squeeze it, pull atoms.', after: ['molecule'], start: { tab: 'welcome' } },
  { id: 'reaction', short: 'Reaction', title: 'A reaction', blurb: 'Catch bonds changing.', after: ['reactor'], start: { tab: 'welcome' } },
  { id: 'barrier', short: 'Barrier', title: 'Its barrier', blurb: 'How hard the step is.', after: ['reaction'],
    // Also everything else that makes a barrier: between two ends you pick, or from a reaction drawn in Design.
    opens: { tabs: ['jobs'], ops: ['ts', 'tsopt', 'design-tsopt', 'complex'], features: ['connect', 'new-calculation', 'compose', 'design-reaction', 'complex'] },
    start: { tab: 'jobs' } },
  { id: 'network', short: 'Explore', title: 'Explore', blurb: 'Your molecules and reactions, as a graph.', after: ['barrier'], opens: { tabs: ['graph'], ops: ['optimize'] }, start: { tab: 'graph' } },
  // Branches.
  { id: 'shapes', short: 'Conformers', title: 'Conformers', blurb: 'The same molecule, folded differently.', after: ['network'],
    opens: { ops: ['conformers'] }, start: { pick: 'molecule', op: 'conformers' } },
  { id: 'method', short: 'Method', title: 'A better method', blurb: 'Choose how energies are computed.', after: ['barrier'],
    opens: { tabs: ['profiles'], features: ['level'] }, start: { tab: 'profiles' } },
  { id: 'kinetics', short: 'Kinetics', title: 'Kinetics', blurb: 'Which products a mixture ends up with.', after: ['network'],
    opens: { tabs: ['analyze'] }, start: { analyze: 'kinetics' } },
  { id: 'discover', short: 'Discover', title: 'Find new reactions', blurb: 'Let mepd propose products for a molecule.', after: ['network'],
    opens: { ops: ['graph-enumeration', 'nanoreactor', 'nanoreactor-more', 'network-splits', 'hessian-sample', 'hessian-global'] },
    start: { pick: 'molecule', op: 'graph-enumeration' } },
  { id: 'solvent', short: 'Solvent', title: 'Solvent', blurb: 'How a solvent changes a barrier.', after: ['network'],
    opens: { ops: ['solvent'] }, start: { job: 'barrier', op: 'solvent' } },
  { id: 'channels', short: 'Reaction channels', title: 'Reaction channels', blurb: 'Every way the same reaction can go.', after: ['shapes'],
    opens: { ops: ['channels', 'channels-more'] }, start: { pick: 'ends', op: 'channels' } },
  // (Catalysis: no one-step tool yet, only Design by hand, so not on the map until there is one.)
  { id: 'retro', short: 'Retrosynthesis', title: 'Retrosynthesis', blurb: 'Routes to a molecule from what you can buy.', after: ['shapes'],
    opens: { ops: ['retrosynthesis'] }, start: { pick: 'molecule', op: 'retrosynthesis' } },
  { id: 'variants', short: 'What if', title: 'What if…', blurb: 'Swap a group and recompute the network.', after: ['solvent'],
    opens: { ops: ['substituents', 'variant-fast', 'variant-reopt'], features: ['variants'] }, start: { job: 'barrier', op: 'substituents' } },
  // (Mechanical force: off in the web UI for now, so not on the map.)
  { id: 'explicit', short: 'Surroundings', title: 'Real surroundings', blurb: 'Explicit water, or a protein pocket.', after: ['solvent'],
    opens: { ops: ['qmmm-build', 'qmmm-inspect', 'qmmm-reaction', 'qmmm-protein-sites', 'qmmm-protein-build', 'qmmm-embed'], features: ['qmmm'] },
    start: { pick: 'molecule', op: 'qmmm-build' } },
  { id: 'bifurcation', short: 'Bifurcations', title: 'Bifurcations', blurb: 'Paths that split after the TS (valley-ridge inflections).', after: ['channels', 'explicit'],
    opens: { ops: ['vri', 'vri-check', 'vri-surface'] }, start: { pick: 'ends', op: 'vri' } },
];
const BY_ID = Object.fromEntries(QUESTS.map((q) => [q.id, q]));

// Every op and tab some quest opens; anything else is always there.
const GATED_OPS = new Set(QUESTS.flatMap((q) => q.opens?.ops || []));
const GATED_TABS = new Set(QUESTS.flatMap((q) => q.opens?.tabs || []));
const GATED_FEATURES = new Set(QUESTS.flatMap((q) => q.opens?.features || []));

const fromSandbox = (j) => j.op === 'nanoreactor' && /[\\/]sandbox[\\/]/.test(j.params?.trajectory || '');

// Which quests are done, from what this browser saw and what the workspace holds.
function doneSet(s) {
  const tour = s.tour || {};
  const jobs = Object.values(s.jobs || {});
  const ws = s.workspace || {};
  const ran = new Set(jobs.filter((j) => j.status === 'done').map((j) => j.op));
  const done = new Set();
  const nEdges = Object.keys(ws.edges || {}).length;
  const rules = {
    molecule: () => Object.keys(ws.structures || {}).length > 0,
    reactor: () => jobs.some(fromSandbox),
    reaction: () => Object.keys(ws.reactions || {}).length > 0,
    barrier: () => jobs.some((j) => j.op === 'ts' && j.status === 'done' && j.summary?.barrier_kcal != null),
    // A graph someone has been building by hand: they know Explore.
    network: () => nEdges >= 2 && jobs.length >= 3,
    product: () => Boolean(ws.design?.reaction),
  };
  for (const q of QUESTS) {
    const uses = q.uses || q.opens?.ops || [];
    if (tour[q.id] || rules[q.id]?.() || uses.some((op) => ran.has(op))) done.add(q.id);
  }
  return done;
}

// The map as it stands: done, open (available, not done yet) and locked quests.
export function mapOf(s = state) {
  const all = Boolean(s.everything);
  const done = doneSet(s);
  const open = new Set();
  for (const q of QUESTS) if (!done.has(q.id) && (all || q.after.every((a) => done.has(a)))) open.add(q.id);
  const reachable = new Set([...done, ...open]);
  const opened = (kind) => new Set(QUESTS.filter((q) => all || reachable.has(q.id)).flatMap((q) => q.opens?.[kind] || []));
  return { all, done, open, ops: opened('ops'), tabs: opened('tabs'), features: opened('features') };
}

const key = (s) => {
  const m = mapOf(s);
  return `${m.all}|${[...m.done].join(',')}|${[...m.open].join(',')}`;
};

// A hook: re-renders when the map changes; returns helpers.
export function useMap() {
  useStore(key);
  const m = mapOf(state);
  return {
    ...m,
    op: (k) => m.all || !GATED_OPS.has(k) || m.ops.has(k),
    tab: (k) => m.all || !GATED_TABS.has(k) || m.tabs.has(k),
    feature: (f) => m.all || !GATED_FEATURES.has(f) || m.features.has(f),
  };
}
export const opOpen = (k, s = state) => { const m = mapOf(s); return m.all || !GATED_OPS.has(k) || m.ops.has(k); };
export const tabOpen = (k, s = state) => { const m = mapOf(s); return m.all || !GATED_TABS.has(k) || m.tabs.has(k); };

export function markQuest(id) {
  if (state.tour?.[id]) return;
  update((s) => { s.tour = { ...(s.tour || {}), [id]: true }; });
  prefs.set('tour', state.tour);
}

export function setEverything(on) {
  prefs.set('everything', on);
  set({ everything: on });
}

// Announce each quest once, when it opens (not the ones open at page load).
let announced = null;
export function announceUnlocks() {
  const m = mapOf(state);
  if (m.all) return;
  const seen = new Set(prefs.get('seenQuests', []));
  if (announced == null) {   // the first look after loading: remember silently
    announced = true;
    for (const id of m.open) seen.add(id);
    prefs.set('seenQuests', [...seen]);
    return;
  }
  const fresh = [...m.open].filter((id) => !seen.has(id));
  if (!fresh.length) return;
  for (const id of fresh) seen.add(id);
  prefs.set('seenQuests', [...seen]);
  const names = fresh.map((id) => BY_ID[id].title);
  toast(`Unlocked: ${names.join(', ')}`, 'ok', 8000, { label: 'See tools', run: () => openTab('welcome') });
}

// The most recent finished barrier search (its job), if any.
function lastBarrierJob(ops = ['ts']) {
  const src = new Set(ops);
  return Object.values(state.jobs).filter((j) => src.has(j.op) && j.status === 'done' && j.summary?.barrier_kcal != null)
    .sort((a, b) => (b.finished || 0) - (a.finished || 0))[0] || null;
}

// "Try it": take the user to where the quest's feature lives.
export function startQuest(id) {
  const q = BY_ID[id];
  const go = q?.start || {};
  const ws = state.workspace;
  if (go.tab) { openTab(go.tab); return; }
  if (go.analyze) { set({ analyze: { ...(state.analyze || {}), compose: false, view: go.analyze }, view: { tab: 'analyze', jobId: null } }); return; }
  // A follow-up of a finished barrier (Reaction conditions on its results page).
  if (go.job) {
    const op = state.operations.find((o) => o.key === go.op);
    const job = lastBarrierJob(op?.source_ops || ['ts']);
    if (!job) { toast('First find a barrier: it starts from one.', 'info', 6000); return; }
    set({ view: { tab: 'job', jobId: job.id, jobTab: 'result' }, focusOp: { key: go.op, at: Date.now() } });
    return;
  }
  if (go.design) {
    if (ws.design?.natoms && !confirm('Replace the molecule open in Design with an example reaction?')) return;
    import('./api.js').then(({ api, attempt }) => attempt(() => api.post('/api/design/new', { smiles: go.design })))
      .then((out) => { if (out) { openTab('design'); toast('Two sides: edit the product on the right.', 'info', 8000); } });
    return;
  }
  // Something to act on: the biggest molecule, the reaction with a barrier, a TS.
  const recs = Object.values(ws.structures || {});
  let picked = null;
  if (go.pick === 'molecule') {
    const mols = recs.filter((r) => r.role !== 'ts' && r.role !== 'complex' && !r.qmmm).sort((a, b) => (b.natoms || 0) - (a.natoms || 0));
    if (mols[0]) picked = { structures: [mols[0].id] };
  } else if (go.pick === 'reaction') {
    const rx = Object.values(ws.reactions || {}).find((r) => r.edge && ws.edges[r.edge]);
    const e = rx?.edge || Object.keys(ws.edges || {})[0];
    if (e) picked = { edges: [e] };
  } else if (go.pick === 'ends') {
    // Two ends of a reaction that has a barrier (else any edge's): a pair to connect.
    const job = lastBarrierJob(['ts']);
    const e = ws.edges[job?.targets.edges[0]] || Object.values(ws.edges || {})[0];
    if (e) picked = { structures: [e.source, e.target] };
  } else if (go.pick === 'ts') {
    const ts = recs.find((r) => r.role === 'ts');
    if (ts) picked = { structures: [ts.id] };
  }
  set({ view: { tab: 'graph', jobId: null } });
  if (!picked) { toast(`First you need ${{ ts: 'a transition state', ends: 'a reaction', reaction: 'a reaction', molecule: 'a molecule' }[go.pick]}.`, 'info', 6000); return; }
  select(picked);
  if (go.op) set({ focusOp: { key: go.op, at: Date.now() } });
}

export const questById = (id) => BY_ID[id];
