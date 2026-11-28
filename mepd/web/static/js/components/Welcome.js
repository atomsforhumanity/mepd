// The first screen (the landing): the three scales mepd works at (a molecule, a path, a network), each a
// finished run as mepd shows it and one way to start your own.
// Also the steps a newcomer follows elsewhere (StepBar, the reactor helpers) and the map of tools (QuestMap).
import { html, useEffect, useState } from '../lib.js';
import { api, attempt, refreshState } from '../api.js';
import { openJob, openReactorSetup, openTab, prefs, set, state, toast, update, useStore } from '../store.js';
import { QUESTS, questById, setEverything, startQuest, useMap } from '../quests.js';
import { QuestIcon } from '../icons.js';

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

// The landing (2026-11-23): one row, small to large, so the scales read at a glance (Jan: "the main message
// … the varying scales of information: molecules, paths, networks"). Each scale is a picture of a real run's
// own page (never staged; a click opens that page), one line, and one action; method details are hover text.
// Molecule and Path show the SN2 study shipped in web/showcase/sn2, its TS page: the 3D view's Fragments
// orbitals at the TS (chloride's HOMO into CH3Br's LUMO), and the first search beside the paths a channels run
// sampled after it. Network replays glycine. Image recipe: data/runs/landing-demos/{shot2.py,crop.py}; the
// network's, shot_stage.py (the stage at a 1530 px window, 2x, cropped round glycine). The network run is hot
// (1000 K, 1 min simulated): at room temperature glycine's lowest barrier is ~46 kcal/mol and nothing forms.
const GUIDED = { temperature: 1000, time_s: 60 };
const HERO = { slug: 'glycine-live', img: '/static/img/landing-scale-network.webp' };
const SN2 = { slug: 'sn2', ts: 'j_85e27dbc' };
const EXAMPLE_REACTION = '[Cl-].CBr>>ClC.[Br-]';

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

// A reaction SMILES into Explore (both ends and their edge), then a Reaction channels run on that edge,
// opened. The run minimizes the two ends itself (--minimize-ends), so nothing else is queued.
async function sample(text) {
  return attempt(async () => {
    if (!text.includes('>>')) throw new Error(`A reaction is written reactants>>products, e.g. ${EXAMPLE_REACTION}`);
    const added = await api.post('/api/structures', { text, optimize: false });
    const edge = (added || []).find((a) => a.edge)?.edge;
    if (!edge) throw new Error('Both sides of the reaction are the same structure');
    const jobs = await api.post('/api/jobs', { op: 'channels', edges: [edge], params: {} });
    openJob(jobs[0].id);
    return jobs;
  });
}

// One scale: its name and size, a picture of a finished run (a click opens it), one line, what the picture
// is (method in the hover), and one action.
function Scale({ name, size, img, alt, href = '#', onOpen, line, caption, how, children }) {
  const open = (e) => { e.preventDefault(); onOpen?.(); };
  return html`<li class="land-scale">
    <div class="land-scale-head"><span class="land-label">${name}</span><span class="land-scale-n">${size}</span></div>
    <a class="land-shot" href=${href} onClick=${open} title=${how} aria-label=${`Open: ${line}`}><img src=${img} alt=${alt} /></a>
    <h2>${line}</h2>
    <p class="land-cap" title=${how}>${caption}${caption && onOpen ? ' ' : ''}${onOpen && html`<a href=${href} onClick=${open}>Open ›</a>`}</p>
    <div class="land-act">${children}</div>
  </li>`;
}

const Arrow = () => html`<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7" /></svg>`;

// The Molecule picture opened: the TS page with the 3D view's orbitals on, as Fragments (the viewer reads both
// choices from prefs when it mounts), scrolled to that view once it is drawn.
async function openFragments(sn2) {
  prefs.set('orbitalsOn', true);
  prefs.set('orbitalKind', 'fragments');
  const out = await openExample(sn2, SN2.ts);
  const reveal = (tries) => {
    const el = document.querySelector('.mo-help');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    else if (tries > 0) setTimeout(() => reveal(tries - 1), 200);
  };
  if (out) reveal(25);
}

export function WelcomeView() {
  const map = useMap();
  const ex = useExamples();
  const hero = ex.find((x) => x.slug === HERO.slug);
  const sn2 = ex.find((x) => x.slug === SN2.slug);
  const [rx, setRx] = useState(EXAMPLE_REACTION);
  const [mol, setMol] = useState('glycine');
  const [busy, setBusy] = useState(null);
  const run = (key, fn) => async (e) => { e?.preventDefault(); setBusy(key); await fn(); setBusy(null); };
  const draw = run('draw', async () => {
    const out = await attempt(() => api.post('/api/design/new', { scratch: true }));
    if (out) openTab('design');
  });
  const sampleIt = run('path', () => sample(rx.trim()));
  const presetOf = (typed) => PRESETS.find((x) => x.name.toLowerCase() === typed.toLowerCase() || x.key === typed.toLowerCase());
  const growIt = run('network', () => {
    const typed = mol.trim();
    if (/[,;]/.test(typed)) { toast('Grow it starts from one molecule; a mixture goes in the Nanoreactor', 'error', 7000); return null; }
    const p = presetOf(typed);
    return p ? grow(`${p.smiles} ${p.name.toLowerCase()}`, p.name.toLowerCase()) : grow(typed, typed);
  });
  // The Nanoreactor setup, filled in with what is typed (a first-screen molecule with its usual water).
  const nanoreactor = () => {
    const typed = mol.trim();
    const p = presetOf(typed);
    openReactorSetup(p ? `${p.n} ${p.smiles}, ${p.water} O` : typed);
  };
  const workbench = () => { setEverything(true); openTab('graph'); };
  return html`<div class="land">
    <header class="land-top">
      <h1>Minimum Energy Path Dynamics (MEPD)</h1>
    </header>
    <ol class="land-scales">
      <${Scale} name="Molecule" size="1 structure" img="/static/img/landing-scale-molecule.webp"
        onOpen=${sn2 && (() => openFragments(sn2))}
        line="Structures, conformers, orbitals" caption="Cl⁻'s HOMO meets CH₃Br's LUMO at the TS"
        how="Orbitals › Fragments: each molecule's own orbitals, computed alone where it sits. At the SN2 transition state chloride's HOMO (dark) overlaps bromomethane's empty σ* C–Br orbital (pale), overlap 0.16. Click to open it."
        alt="Chloride's filled p orbital meeting the empty σ* orbital of bromomethane behind the carbon, at the SN2 transition state">
        <button class="btn primary land-go" disabled=${busy === 'draw'} onClick=${draw}>Draw a molecule<${Arrow} /></button>
      <//>
      <${Scale} name="Path" size="1 reaction" img="/static/img/landing-scale-path.webp"
        onOpen=${sn2 && (() => openExample(sn2, SN2.ts))}
        line="Find alternate and competing transition states" caption=""
        how="Cl⁻ + CH₃Br: the first path search crossed a frontside-attack TS (53.2 kcal/mol); sampling conformers and atom mappings found the backside attack (8.6). g-xTB, every TS followed by an IRC, electronic barriers from the lowest reactant geometry found."
        alt="Energy diagram: the first search's route over 53.2 kcal/mol and the sampled route over 8.6">
        <form class="land-row" onSubmit=${sampleIt}>
          <input class="mono" spellcheck="false" aria-label="Reaction SMILES" placeholder="reactants>>products (SMILES)"
            value=${rx} onInput=${(e) => setRx(e.target.value)} />
          <button class="btn primary land-go" type="submit" disabled=${busy === 'path' || !rx.trim()}
            title="Reaction channels: conformers and atom mappings of both ends, a path search per pair, every TS checked by an IRC">Find its TSs<${Arrow} /></button>
        </form>
      <//>
      <${Scale} name="Network" size="1,556 reactions" img=${HERO.img} href=${`#/network/example/${HERO.slug}`}
        onOpen=${hero && (() => openExample(hero))}
        line="Unbiased prediction of reaction products" caption="Glycine at 1000 K, 400 species"
        how="g-xTB structures, ωB97X-D3BJ/def2-TZVP single points on all 1084 structures, rates from free energies (1 M), gas phase. What forms per glycine: CO₂ + methylamine 0.99. 2.5 h on 16 workers."
        alt="mepd's network view of glycine's finished run, from glycine on the left to what forms on the right">
        <form class="land-row" onSubmit=${growIt}>
          <input class="mono" spellcheck="false" aria-label="Molecule, name or SMILES" placeholder="molecule, name or SMILES"
            value=${mol} onInput=${(e) => setMol(e.target.value)} />
          <button class="btn primary land-go" type="submit" disabled=${busy === 'network' || !mol.trim()}
            title=${`Runs hot (${GUIDED.temperature} K, ${GUIDED.time_s / 60} min simulated) so that something forms; first reactions in about 2 minutes`}>Grow it<${Arrow} /></button>
        </form>
        <div class="land-alt">
          <button class="btn land-go-alt" type="button" onClick=${nanoreactor}
            title="A box of the molecules you choose, so many of each, with their charges and spins: hot MD finds what they do together">Nanoreactor<${Arrow} /></button>
          <span class="small muted">several molecules, so many of each</span>
        </div>
      <//>
    </ol>
    <div class="land-foot">
      ${ex.length > 0 && html`<span class="land-label">Examples</span>
        ${ex.map((x) => html`<button class="land-example" onClick=${() => openExample(x)} title=${x.blurb || x.tag || ''}>${x.title}</button>`)}`}
      <span class="land-foot-end">
        <a href="#/explore" onClick=${(e) => { e.preventDefault(); workbench(); }}>Open the workbench</a>
        <label class="door-switch" title="Every tool now, without the step-by-step unlocking">
          <input type="checkbox" class="switch" checked=${map.all} onChange=${(e) => setEverything(e.target.checked)} /><span>All tools</span></label>
      </span>
    </div>
  </div>`;
}

// Examples: finished studies that open as your own copy (mepd/web/examples). A replay opens its live
// network view. A study opens the copy you made before if it still has `job` (else a new copy), and lands
// on that calculation's page when one is given.
function useExamples() {
  const [list, setList] = useState([]);
  useEffect(() => { api.get('/api/examples').then((r) => setList(r || [])).catch(() => setList([])); }, []);
  return list;
}
async function openExample(x, job = null) {
  if (x.kind === 'replay') { set({ view: { tab: 'network', jobId: null, example: x.slug } }); return null; }
  const out = await attempt(() => api.post(`/api/examples/${encodeURIComponent(x.slug)}/open`, { reuse: true, job }),
    `Opened “${x.title}”, your own copy`);
  if (out && job) {
    await refreshState().catch(() => {});
    if (state.jobs[job]) openJob(job);
  }
  return out;
}
