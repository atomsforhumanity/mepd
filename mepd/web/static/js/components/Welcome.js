// The first screen of an empty session, and the steps a newcomer follows:
// a molecule (ready-made, or built in Design), then the interactive reactor.
// Everything deeper (barriers, networks) opens from what happens there.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, openTab, prefs, set, state, update, useStore } from '../store.js';
import { Viewer3D } from './Viewer3D.js';
import { QUESTS, questById, setEverything, startQuest, useMap } from '../quests.js';
import { QuestIcon } from '../icons.js';
import { PHONE_QUERY } from '../util.js';

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

function startPreset(p) {
  return attempt(async () => {
    const added = await api.post('/api/structures', { text: `${p.smiles} ${p.name}\nO water`, optimize: false });
    const [mol, wat] = added;
    const counts = { [mol.id]: p.n };
    counts[wat.id] = (counts[wat.id] || 0) + p.water;
    return launch(counts);
  });
}

// ---- the map ----------------------------------------------------------------
// What the first chain opened: done quests, ones to try now, and the next
// ones as silhouettes; the rest stays a count, to discover.
// Done first, then what is new, then what comes next, then the unknown.
const ORDER = { done: 0, open: 1, all: 1, locked: 2, far: 3 };

function QuestMap() {
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

// ---- the first screen ---------------------------------------------------------
// What mepd does is the way in: one column per scale its tools are grouped
// by (Actions.js), each saying what mepd does there and holding your step.
// What a reaction step is to mepd: up over a barrier, down to the product.
function ProfilePic() {
  return html`<svg class="scale-pic" viewBox="0 0 240 120" role="img" aria-label="An energy profile: reactant, transition state, product">
    <path class="sp-line" d="M14 78 H58 C88 78 96 24 120 24 C144 24 152 92 182 92 H226" />
    <path class="sp-level" d="M14 78 H58 M182 92 H226 M104 24 H136" />
    <path class="sp-arrow" d="M36 74 V30" marker-end="url(#sp-head)" />
    <defs><marker id="sp-head" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="sp-head" /></marker></defs>
    <text x="42" y="54" class="sp-num">barrier</text>
    <text x="36" y="108" class="sp-cap" text-anchor="middle">start</text>
    <text x="120" y="14" class="sp-cap" text-anchor="middle">transition state</text>
    <text x="204" y="108" class="sp-cap" text-anchor="middle">product</text>
  </svg>`;
}

// What a network is: molecules joined by the reactions between them.
function NetworkPic() {
  const nodes = [[30, 60], [92, 26], [96, 92], [160, 54], [212, 22], [210, 98]];
  const edges = [[0, 1, '18'], [0, 2, '25'], [1, 3, '12'], [2, 3, ''], [3, 4, '31'], [3, 5, '9']];
  return html`<svg class="scale-pic" viewBox="0 0 240 120" role="img" aria-label="A reaction network: molecules joined by reactions">
    ${edges.map(([a, b, t]) => { const [x1, y1] = nodes[a], [x2, y2] = nodes[b];
      return html`<line x1=${x1} y1=${y1} x2=${x2} y2=${y2} class=${`sp-edge ${t ? '' : 'idle'}`} />
        ${t && html`<text x=${(x1 + x2) / 2} y=${(y1 + y2) / 2 + 3} class="sp-enum" text-anchor="middle">${t}</text>`}`; })}
    ${nodes.map(([x, y], i) => html`<rect x=${x - 11} y=${y - 9} width="22" height="18" rx="3" class=${`sp-node ${i ? '' : 'start'}`} />`)}
  </svg>`;
}

function Scale({ n, icon, name, what, done, on, later, children }) {
  return html`<li class=${`scale ${on ? 'on' : ''} ${done ? 'done' : ''} ${later ? 'scale-later' : ''}`}>
    <div class="scale-head">
      <span class="ws-icon"><${QuestIcon} id=${icon} size=${26} /></span>
      <b>${name}</b>
      <span class="step-num" title=${done ? 'Done' : `Step ${n}`}>${done ? '✓' : n}</span>
    </div>
    <p class="scale-what">${what}</p>
    <div class="scale-do">${children}</div>
  </li>`;
}

function Recent() {
  const [data, setData] = useState(null);
  useEffect(() => { api.get('/api/sessions').then(setData).catch(() => setData(null)); }, []);
  const others = (data?.sessions || []).filter((s) => !s.current && (s.structures || s.jobs)).slice(0, 4);
  if (!others.length) return null;
  return html`<div class="welcome-recent">
    <h4 class="section-title">Recent</h4>
    <ul>${others.map((s) => html`<li><a href="#" title=${s.path}
      onClick=${(e) => { e.preventDefault(); attempt(() => api.post('/api/sessions/open', { path: s.path })); }}>${s.name}</a>
      <span class="small muted">${s.structures} molecules</span></li>`)}</ul>
  </div>`;
}

// Phones: one column, so the preview sits right under the choices.
function usePhone() {
  const [phone, setPhone] = useState(() => window.matchMedia(PHONE_QUERY).matches);
  useEffect(() => {
    const mq = window.matchMedia(PHONE_QUERY);
    const f = () => setPhone(mq.matches);
    mq.addEventListener('change', f);
    return () => mq.removeEventListener('change', f);
  }, []);
  return phone;
}

export function WelcomeView() {
  const phone = usePhone();
  const [pick, setPick] = useState('glycine');
  const [busy, setBusy] = useState(false);
  const map = useMap();
  // A step is done once it was done here, or the workspace shows it (a reload, another browser).
  const tour = { molecule: map.done.has('molecule'), reactor: map.done.has('reactor'), reaction: map.done.has('reaction') };
  const jobs = useStore((s) => s.jobs);
  const lastRun = Object.values(jobs).filter((j) => j.op === 'nanoreactor' && /[\\/]sandbox[\\/]/.test(j.params?.trajectory || ''))
    .sort((a, b) => (b.created || 0) - (a.created || 0))[0];
  const p = PRESETS.find((x) => x.key === pick);
  const start = async () => { setBusy(true); await startPreset(p); setBusy(false); };
  const build = async () => {
    const out = await attempt(() => api.post('/api/design/new', { scratch: true }));
    if (out) openTab('design');
  };
  return html`<div class="welcome">
    <header class="welcome-hero">
      <div class="welcome-brand">
        <svg class="logo" viewBox="0 0 32 32" aria-hidden="true"><polygon points="16,3 27,9.5 27,22.5 16,29 5,22.5 5,9.5" fill="none" stroke="currentColor" stroke-width="2.6"/><circle cx="16" cy="16" r="3.2" fill="currentColor"/></svg>
        <h1>mepd</h1>
      </div>
      <p class="welcome-tag">Compute how molecules react.</p>
      <p class="welcome-sub">Try it here in a few minutes, one step at a time.</p>
    </header>
    <ol class="welcome-scales">
      <${Scale} n=${1} icon="molecule" name="Molecule" what="Build one and find its stable shapes." on=${!tour.molecule} done=${tour.molecule}>
        <div class="welcome-preview"><${Viewer3D} key=${p.key} xyz=${p.xyz} height=${phone ? 130 : 150} /></div>
        <div class="chips">
          ${PRESETS.map((x) => html`<button class=${`chip ${x.key === pick ? 'on' : ''}`} onClick=${() => setPick(x.key)}>${x.name}</button>`)}
          <button class="chip add" onClick=${build} title="Draw one atom by atom in Design">✎ Build your own</button>
        </div>
      <//>
      <${Scale} n=${2} icon="reaction" name="Reaction" what="Find what it turns into, and the energy each step takes." on=${tour.molecule && !tour.reaction} done=${tour.reaction}>
        <${ProfilePic} />
        <button class="btn primary" disabled=${busy} onClick=${start}>${busy ? 'Starting…' : `Put ${p.name.toLowerCase()} in the reactor`}</button>
        <span class="small muted">${tour.reaction ? (map.done.has('barrier') ? 'Again, or with another molecule.' : 'Next: find its barrier') : `${p.n} ${p.name.toLowerCase()} + ${p.water} water at ${TEMPERATURE} K: watch bonds break and form.`}</span>
      <//>
      <${Scale} n=${3} icon="network" name="Network" what="Connect the steps to see what a mixture becomes." on=${tour.reaction} later=${!tour.reaction}>
        <${NetworkPic} />
        ${map.tab('graph')
          ? html`<button class="btn" onClick=${() => openTab('graph')}>Open Explore</button>`
          : html`<span class="small muted">Opens with your first reaction.</span>`}
      <//>
    </ol>
    <div class="welcome-main">
      <p class="small muted welcome-more">
        ${lastRun && html`<a href="#" onClick=${(e) => { e.preventDefault(); openJob(lastRun.id, 'story'); }}>Your last reactor run</a>${' · '}`}
        <a href="#" onClick=${(e) => { e.preventDefault(); openTab('design'); }}>Open a SMILES or .xyz</a>
      </p>
      ${(map.done.has('reactor') || map.all) && html`<${QuestMap} />`}
      <p class="small welcome-all">${map.all
        ? html`Showing everything. <a href="#" onClick=${(e) => { e.preventDefault(); setEverything(false); }}>Show only what I've found</a>`
        : html`<a href="#" title="Every tab and tool at once" onClick=${(e) => { e.preventDefault(); setEverything(true); }}>I know mepd: show everything</a>`}</p>
    </div>
    <aside class="welcome-side"><${Recent} /></aside>
  </div>`;
}
