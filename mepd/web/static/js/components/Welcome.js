// The first screen (the landing): pick a molecule and grow its reaction network, beside a finished
// network as mepd shows it. Also the steps a newcomer follows elsewhere (StepBar, the reactor helpers)
// and the map of tools (QuestMap).
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, openTab, prefs, set, state, update, useStore } from '../store.js';
import { QUESTS, questById, setEverything, startQuest, useMap } from '../quests.js';
import { QuestIcon } from '../icons.js';
import { depictUrl } from '../util.js';

// Ready-made molecules, with enough copies and water to give the reactor
// something to do (MMFF geometries; the reactor needs no minimum).
export const PRESETS = [
  { key: 'glycine', name: 'Glycine', smiles: 'NCC(=O)O', formula: 'C₂H₅NO₂', n: 2, water: 4,
    xyz: '10\nglycine\nN -0.713 0.422 -1.219\nC -0.470 -0.599 -0.186\nC 0.786 -0.292 0.617\nO 1.533 -1.110 1.130\nO 0.991 1.032 0.809\nH -1.678 0.355 -1.536\nH -0.610 1.341 -0.780\nH -1.314 -0.640 0.509\nH -0.348 -1.571 -0.671\nH 1.823 1.062 1.326' },
  { key: 'ethanol', name: 'Ethanol', smiles: 'CCO', formula: 'C₂H₆O', n: 2, water: 4,
    xyz: '9\nethanol\nC -0.863 0.220 0.203\nC 0.406 -0.375 -0.370\nO 1.527 0.404 0.022\nH -1.738 -0.359 -0.104\nH -0.988 1.255 -0.132\nH -0.821 0.242 1.297\nH 0.551 -1.401 -0.019\nH 0.365 -0.387 -1.463\nH 1.561 0.400 0.994' },
  { key: 'formaldehyde', name: 'Formaldehyde', smiles: 'C=O', formula: 'CH₂O', n: 4, water: 3,
    xyz: '4\nformaldehyde\nC -0.012 0.002 0.000\nO 1.191 -0.226 -0.001\nH -0.763 -0.804 0.000\nH -0.416 1.027 0.001' },
  { key: 'hcn', name: 'Hydrogen cyanide', smiles: 'C#N', formula: 'HCN', n: 5, water: 2,
    xyz: '3\nhcn\nC -0.032 0.001 0.000\nN 1.128 -0.038 0.000\nH -1.096 0.037 0.000' },
];

const TEMPERATURE = 800;

// ---- the steps (per browser) ----------------------------------------------
const STEPS = [['molecule', 'Molecule'], ['reactor', 'Reactor'], ['reaction', 'Reaction'], ['barrier', 'Barrier'], ['network', 'Explore']];

export function markStep(step) {
  if (state.tour?.[step]) return;
  update((s) => { s.tour = { ...(s.tour || {}), [step]: true }; });
  prefs.set('tour', state.tour);
}

// "1 Molecule › 2 Reactor › 3 Reaction": where you are, what is done. Quiet,
// and gone for good once closed.
export function StepBar({ at }) {
  const tour = useStore((s) => s.tour || {});
  if (tour.hidden) return null;
  const hide = () => { update((s) => { s.tour = { ...(s.tour || {}), hidden: true }; }); prefs.set('tour', state.tour); };
  const here = STEPS.findIndex(([k]) => k === at);
  const done = (k, i) => i < here || !!tour[k];   // being at a step means the ones before it are done
  // Later steps show up as they are earned: always three, then one past the furthest reached.
  const reached = Math.max(here, ...STEPS.map(([k], i) => (tour[k] ? i : -1)));
  const shown = STEPS.slice(0, Math.max(3, reached + 2));
  return html`<nav class="step-bar small" aria-label="Getting started">
    ${shown.map(([k, label], i) => html`${i > 0 && html`<span class="step-sep">›</span>`}
      <span class=${`step ${k === at ? 'on' : ''} ${done(k, i) ? 'done' : ''}`}>${done(k, i) && k !== at ? '✓' : i + 1} ${label}</span>`)}
    <button class="btn-link small step-close" title="Hide these steps" onClick=${hide}>hide</button>
  </nav>`;
}

// ---- into the reactor -------------------------------------------------------
// How much water joins a molecule of `natoms` atoms (n copies), so the box
// stays small enough to run smoothly.
export function mixFor(natoms) {
  const n = natoms <= 12 ? 2 : 1;
  return { n, water: Math.max(0, Math.min(4, Math.floor((36 - n * natoms) / 3))) };
}

async function water() {
  const [rec] = await api.post('/api/structures', { text: 'O water', optimize: false });
  return rec.id;
}

// Start the reactor on these species ({id: count}) and open it.
async function launch(counts) {
  const box = await api.post('/api/sandbox', { counts, temperature: TEMPERATURE });
  markStep('molecule');
  markStep('reactor');
  set({ view: { tab: 'sandbox', jobId: null, sandbox: box } });
  return box;
}

// A graph structure (just added from Design) into the reactor, with water.
// Several molecules (a complex) go in as they are.
export function reactorFor(rec) {
  return attempt(async () => {
    if ((rec.members || []).length > 1) {
      const box = await api.post('/api/sandbox', { structure: rec.id, temperature: TEMPERATURE });
      markStep('molecule'); markStep('reactor');
      set({ view: { tab: 'sandbox', jobId: null, sandbox: box } });
      return box;
    }
    // A first-screen molecule keeps its own mix; anything else gets one sized to it.
    const preset = PRESETS.find((x) => x.smiles === rec.smiles);
    const { n, water: w } = preset ? { n: preset.n, water: preset.water } : mixFor(rec.natoms);
    const counts = { [rec.id]: n };
    if (w) { const wid = await water(); counts[wid] = (counts[wid] || 0) + w; }
    return launch(counts);
  });
}

// ---- the map ----------------------------------------------------------------
// What the first chain opened: done quests, ones to try now, and the next
// ones as silhouettes; the rest stays a count, to discover.
// Done first, then what is new, then what comes next, then the unknown.
const ORDER = { done: 0, open: 1, all: 1, locked: 2, far: 3 };

export function QuestMap() {
  const map = useMap();
  const near = (q) => q.after.some((a) => map.done.has(a));
  // "Show everything": every tool unlocked and usable, none marked new.
  const stateOf = (q) => (map.done.has(q.id) ? 'done' : map.all ? 'all' : map.open.has(q.id) ? 'open' : near(q) ? 'locked' : 'far');
  const found = QUESTS.filter((q) => map.done.has(q.id)).length;
  const far = QUESTS.filter((q) => stateOf(q) === 'far').length;
  return html`<section class="quest-map">
    <div class="quest-head"><h4 class="section-title">Your tools</h4><span class="small muted mono">${map.all ? `all ${QUESTS.length}` : `${found}/${QUESTS.length}`}</span></div>
    <ul class="quests">${QUESTS.map((q) => [q, stateOf(q)]).filter(([, st]) => st !== 'far')
      .sort((a, b) => ORDER[a[1]] - ORDER[b[1]]).map(([q, st]) => {
      const need = q.after.filter((a) => !map.done.has(a)).map((a) => questById(a).short).join(' + ');
      const tip = st === 'far' ? 'Not found yet' : st === 'locked' ? `${q.title}: after ${need}` : `${q.title}: ${q.blurb}`;
      return html`<li><button class=${`quest ${st}`} title=${tip} aria-label=${tip} disabled=${st === 'locked' || st === 'far'}
        onClick=${() => startQuest(q.id)}>
        ${st === 'far' ? html`<span class="q-unknown">?</span>` : html`<${QuestIcon} id=${q.id} />`}
        <span class="q-name">${st === 'far' ? '' : q.short}</span>
        ${st === 'done' && html`<span class="q-badge">✓</span>`}
        ${st === 'open' && html`<span class="q-new" aria-hidden="true"></span>`}
      </button></li>`;
    })}${far > 0 && html`<li><span class="quest far more" title="Found as you go: each tool appears once the ones before it are used">+${far}<span class="q-name">to find</span></span></li>`}</ul>
  </section>`;
}

// The landing (2026-11-10 design, canvas "mepd landing: grow a network"): pick a molecule and grow its
// network, beside a finished network exactly as mepd shows it (a real run, replayed from Examples), and
// the finished networks to open. The guided run is hot (1000 K, 1 min simulated): at room temperature
// glycine's lowest barrier is ~46 kcal/mol and nothing forms, so the guide would end on "nothing".
const GUIDED = { temperature: 1000, time_s: 60 };
const HERO = { slug: 'glycine-live', img: '/static/img/landing-glycine-network.png' };

async function grow(text, label) {
  return attempt(async () => {
    const added = await api.post('/api/structures', { text, optimize: false });
    if (!added?.length) throw new Error('nothing to grow from');
    const mol = added[0];
    const jobs = await api.post('/api/jobs', { op: 'graph-enumeration', structures: [mol.id], params: GUIDED,
      label: `Reaction network expansion: ${label || mol.name}` });
    markStep('molecule');
    openJob(jobs[0].id);
    return jobs;
  });
}

const possessive = (name) => `${name}${name.endsWith('s') ? "'" : "'s"}`;

export function WelcomeView() {
  const [pick, setPick] = useState('glycine');
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const map = useMap();
  const ex = useExamples();
  const p = PRESETS.find((x) => x.key === pick);
  const typed = text.trim();
  const short = (x) => (x.key === 'hcn' ? 'HCN' : x.name);
  const name = typed || (p.key === 'hcn' ? 'HCN' : p.name.toLowerCase());
  const go = async (e) => {
    e?.preventDefault();
    setBusy(true);
    await (typed ? grow(typed, typed) : grow(`${p.smiles} ${p.name.toLowerCase()}`, p.name.toLowerCase()));
    setBusy(false);
  };
  const draw = async () => {
    const out = await attempt(() => api.post('/api/design/new', { scratch: true }));
    if (out) openTab('design');
  };
  const workbench = () => { setEverything(true); openTab('graph'); };
  const hero = ex.find((x) => x.slug === HERO.slug);
  const openHero = (e) => { e.preventDefault(); if (hero) openExample(hero); };
  return html`<div class="land">
    <section class="land-start" aria-label="Start">
      <h1>Predict what reactions will happen in real time</h1>
      <p class="land-lede">Pick a molecule. mepd heats it until bonds break, finds every reaction it can take, checks the energies, and shows what it becomes, live.</p>
      <form onSubmit=${go}>
        <h2 class="land-label">Pick a molecule</h2>
        <div class="land-picks" role="radiogroup" aria-label="Molecule">
          ${PRESETS.map((x) => html`<button type="button" role="radio" aria-checked=${!typed && x.key === pick ? 'true' : 'false'}
            class=${`land-pick ${!typed && x.key === pick ? 'on' : ''}`} onClick=${() => { setPick(x.key); setText(''); }}>
            <img src=${depictUrl(x.smiles, 168, 124)} alt="" /><span>${short(x)}</span></button>`)}
          <button type="button" class="land-pick land-draw" onClick=${draw}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20l4-1 11-11-3-3L5 16l-1 4z" /><path d="M14 6l3 3" /></svg>Draw one</button>
        </div>
        <label class="land-type" for="land-text"><span>or type one</span>
          <input id="land-text" class="mono" placeholder="name or SMILES, e.g. aspirin" value=${text} onInput=${(e) => setText(e.target.value)} /></label>
        <button class="btn primary land-go" type="submit" disabled=${busy}>
          ${busy ? 'Starting…' : `Grow ${possessive(name)} network`}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7" /></svg></button>
      </form>
      <p class="land-note">First reactions in about 2 minutes, in the view on the right. It runs hot (${GUIDED.temperature} K, ${GUIDED.time_s / 60} min simulated) so that something forms.</p>
      <div class="land-foot">
        <a href="#/explore" onClick=${(e) => { e.preventDefault(); workbench(); }}>Know what you want? Open the workbench</a>
        <label class="door-switch" title="Every tool now, without the step-by-step unlocking">
          <input type="checkbox" class="switch" checked=${map.all} onChange=${(e) => setEverything(e.target.checked)} /><span>All tools</span></label>
      </div>
    </section>
    <section class="land-hero" aria-label="A finished network, as mepd shows it">
      <div class="land-card">
        <div class="land-card-head">
          <h2 class="land-label">A finished network, as mepd shows it</h2>
          ${hero && html`<a href=${`#/network/example/${HERO.slug}`} onClick=${openHero}>Open it ›</a>`}
        </div>
        <div class="land-card-title"><span class="land-serif">Glycine</span><span class="muted">1000 K · 1 min simulated</span>
          <span class="land-count"><b>400</b> species <b>1556</b> reactions</span></div>
        <a class="land-shot" href=${`#/network/example/${HERO.slug}`} onClick=${openHero} aria-label="Open glycine's finished network">
          <img src=${HERO.img} alt="mepd's network view of glycine's finished run: glycine on the left, the species it reaches in columns by reactions from the start, routes drawn by how fast they go, and what forms per glycine on the right" /></a>
        <p class="land-forms">What forms, per glycine: <b>CO₂ + methylamine 0.99</b> · the rest (CO + formaldehyde + ammonia, methyl isocyanate + water) under 0.01 · glycine all used up.</p>
        <p class="land-method">How it was computed: <b>g-xTB</b> structures · energies <b>ωB97X-D3BJ/def2-TZVP</b> (DFT single points on all 1084 structures) · rates from <b>free energies</b> (1 M), gas phase.</p>
        <p class="land-note">This is the view your own run opens in, starting from your molecule alone. The first reactions appear after about 2 minutes; the network then fills in over hours (this one: 2.5 h on 16 workers).</p>
      </div>
    </section>
    ${ex.length > 0 && html`<section class="land-examples" aria-label="Finished networks">
      <h2 class="land-label">Finished networks</h2>
      <ul>${ex.map((x) => html`<li><button class="land-example" onClick=${() => openExample(x)} title=${x.blurb || ''}>
        <b>${x.title}</b><span>${x.tag}</span></button></li>`)}</ul>
    </section>`}
  </div>`;
}

// Examples: finished studies that open as your own copy (mepd/web/examples).
function useExamples() {
  const [list, setList] = useState([]);
  useEffect(() => { api.get('/api/examples').then((r) => setList(r || [])).catch(() => setList([])); }, []);
  return list;
}
function openExample(x) {
  if (x.kind === 'replay') { set({ view: { tab: 'network', jobId: null, example: x.slug } }); return null; }
  return attempt(() => api.post(`/api/examples/${encodeURIComponent(x.slug)}/open`), `Opened “${x.title}” as your own copy`);
}
