// Nanoreactor reactions, wherever they are shown (a job's results, its live
// reactor, the Inspector): what each one's TS search stands at, and one
// click to start it on the reaction's subsystem (the hidden edge between
// its optimized reactant and product sides).
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, select, set, state, useStore } from '../store.js';
import { depictUrl, edgeStatus, fmtKcal, jobHeadline, levelBarrierText } from '../util.js';
import { ProfilePicker, defaultProfile } from './Actions.js';
import { ParamForm, clampToSchema, defaultsFor } from './ParamForm.js';
import { Viewer3D } from './Viewer3D.js';
import { EventPlayer } from './ReactorLive.js';
import { openAnalyze } from './Analyze.js';
import { Why } from './Why.js';

export function reactionOf(workspace, jobId, index) {
  return Object.values(workspace.reactions || {}).find((r) => r.origin?.job === jobId && r.origin?.index === index) || null;
}

// Why a reaction has no TS endpoints (its optimized reactant or product
// complex changed bonds), short for tables, in full for tooltips and cards.
export const NO_TS_SHORT = {
  barrierless: 'barrierless', recombines: 'barrierless (recombines)', reverts: 'unstable product',
  falls_apart: 'falls apart', rearranges: 'rearranges', failed: 'optimization failed',
};
// The same, as an outcome in plain words (the reactor's story cards).
export const NO_TS_OUTCOME = {
  barrierless: 'Happens on contact: no barrier', recombines: 'Recombines on contact: no barrier',
  reverts: 'The product falls back when relaxed', falls_apart: 'Falls apart when relaxed',
  rearranges: 'Rearranges when relaxed', failed: 'Relaxing its molecules failed',
};
// A search that found no TS, with the path's highest point as a hint (a
// very high one means the search went wrong, not that the step is hard).
export function noTsHint(v) {
  return v > 100 ? `the path went implausibly high (≈${fmtKcal(v)} kcal/mol): the search likely went wrong`
    : `the path's highest point was ≈${fmtKcal(v)} kcal/mol, but no TS there joins the two ends`;
}
// A finished search's page with its "Sample more paths" card open.
export function sampleMore(jobId) {
  set({ view: { tab: 'job', jobId, jobTab: 'result' }, focusOp: { key: 'channels-more', at: Date.now() } });
}
export function noTsWhy(r) {
  const why = r.complex_error && !/no instance kept/.test(r.complex_error) ? r.complex_error
    : 'its reactant or product complex changed bonds when optimized';
  return `No TS search: ${why}.`;
}

export async function findTs(r) {
  if (!r?.edge) return null;
  return attempt(() => api.post('/api/jobs', { op: 'ts', edges: [r.edge], params: {}, profile: defaultProfile() }),
    'TS search queued on the reaction’s subsystem');
}

export function showInExplore(r) {
  if (!r?.edge) return;
  select({ edges: [r.edge] });
  set({ view: { tab: 'graph', jobId: null } });
}

// "TS 31.2" / "running…" / a Find TS button: what can be done about this
// reaction's barrier right now. `r` is the workspace reaction (or null).
export function TsCell({ r, why = 'not in Explore (deleted?)' }) {
  useStore((s) => s.jobs);   // re-render as its searches run
  const edges = useStore((s) => s.workspace.edges);
  if (!r) return html`<span class="small muted" title=${why}>—</span>`;
  if (!r.edge || !edges[r.edge]) {
    return html`<span class="small muted" title=${noTsWhy(r)}>${NO_TS_SHORT[r.complex_reason] || 'no TS endpoints'}</span>`;
  }
  const st = edgeStatus(edges[r.edge], state.jobs);
  if (st.status === 'running' || st.status === 'queued') {
    const j = Object.values(state.jobs).find((x) => x.targets.edges.includes(r.edge) && ['running', 'queued'].includes(x.status));
    return html`<a href="#" class="small" title="Watch it live" onClick=${(e) => { e.preventDefault(); if (j) openJob(j.id, 'live'); }}>${st.status}…</a>`;
  }
  const label = st.barrier != null ? `ΔE‡ ${fmtKcal(st.barrier)}` : st.barrierUnverified != null ? 'no TS' : null;
  // Several levels of theory: each level's lowest, each opening its own calculation.
  const perLevel = st.levels?.length > 1 && html`${st.levels.map((l, i) => html`${i ? ' · ' : ''}${l.job
    ? html`<a href="#" class="mono" title="Open the calculation behind this barrier" onClick=${(e) => { e.preventDefault(); openJob(l.job); }}>${levelBarrierText(l)}</a>`
    : html`<span class="mono">${levelBarrierText(l)}</span>`}`)}`;
  return html`<span class="ts-cell">
    ${perLevel || label && (st.barrierJob
    ? html`<a href="#" class="mono" onClick=${(e) => { e.preventDefault(); openJob(st.barrierJob); }}
        title=${`${st.barrier != null ? 'Barrier from the reactant side, TS + IRC verified' : `No transition state found: ${noTsHint(st.barrierUnverified)}`}. Open its TS search (TS, IRC, sample more paths)`}>${label}</a>`
    : html`<span class="mono" title=${st.barrier != null ? 'Barrier from the reactant side, TS + IRC verified' : `No transition state found: ${noTsHint(st.barrierUnverified)}`}>${label}</span>`)}
    ${label
    ? html`<button class="btn small" onClick=${() => openAnalyze({ reaction: r.id, retry: Date.now() })}
        title="Open its settings (from the last search) to change what to try, then search again">Search again…</button>`
    : html`<button class="btn small" onClick=${() => findTs(r)} title="Path search, TS optimization and IRC on only this reaction’s molecules">Find TS</button>`}
  </span>`;
}

// The TS searches a nanoreactor run started (one job per reaction).
export function SpawnedSearches({ job }) {
  const jobs = useStore((s) => s.jobs);
  const ids = job.spawned || [];
  if (!ids.length) return null;
  const list = ids.map((id) => jobs[id]).filter(Boolean);
  const n = (st) => list.filter((j) => j.status === st).length;
  return html`<div class="rx-spawned">
    <div class="section-title">TS searches started by this run · ${n('done')} done${n('running') ? `, ${n('running')} running` : ''}${n('queued') ? `, ${n('queued')} queued` : ''}${n('failed') ? `, ${n('failed')} failed` : ''}</div>
    <ul>${list.map((j) => html`<li>
      <a href="#" onClick=${(e) => { e.preventDefault(); openJob(j.id, j.status === 'running' ? 'live' : undefined); }}>${j.title.replace(/^TS: /, '')}</a>
      <span class=${`pill ${j.status}`}>${j.status}</span>
      ${j.summary?.barrier_kcal != null && (j.summary.barrier_verified === false
        ? html`<span class="small muted" title=${noTsHint(j.summary.barrier_kcal)}>no TS</span>`
        : html`<span class="mono small">ΔE‡ ${fmtKcal(j.summary.barrier_kcal)}</span>`)}
    </li>`)}</ul>
  </div>`;
}

// The reactions of one nanoreactor job, at the top of its results: click a
// row to open its reaction card underneath.
export function ReactionTable({ job, reactions }) {
  const ws = useStore((s) => s.workspace);
  const [open, setOpen] = useState(null);
  if (!reactions.length) return null;
  const openRx = open == null ? null : reactionOf(ws, job.id, open);
  return html`<div class="rx-table-wrap">
    <p class="small muted">Click a reaction to see its species, energies, reaction complex and MD event, and to find its TS.</p>
    <table class="rx-table">
      <thead><tr><th>Reaction</th><th title="Products minus reactants, each optimized alone">ΔE</th><th>Seen</th><th>TS</th></tr></thead>
      <tbody>${reactions.map((rx) => {
        const r = reactionOf(ws, job.id, rx.id);
        return html`<tr class=${`clickable ${open === rx.id ? 'on' : ''}`} onClick=${() => setOpen(open === rx.id ? null : rx.id)}>
          <td class="rx-eq">${rx.label}</td>
          <td class="mono">${rx.delta_e_kcal != null ? `${rx.delta_e_kcal >= 0 ? '+' : ''}${fmtKcal(rx.delta_e_kcal)}` : '—'}</td>
          <td class="mono">${rx.count}${rx.reverse_count ? `/${rx.reverse_count}` : ''}</td>
          <td onClick=${(e) => e.stopPropagation()}><${TsCell} r=${r} /></td>
        </tr>
        ${open === rx.id && html`<tr class="rx-card-row"><td colspan="4">
          ${openRx ? html`<${ReactionCard} key=${openRx.id} r=${openRx} />` : html`<p class="small muted">This reaction is not in Explore (deleted?).</p>`}
        </td></tr>`}`;
      })}</tbody>
    </table>
  </div>`;
}

// ------------------------------------------------------------ reaction card
// Everything about one reaction in one place: what reacts (structures),
// how the energy goes (separated -> complex -> TS -> complex -> separated,
// all over the same atoms), the reaction complex in 3D (optimized ends,
// the MD event, the TS), and its TS search.

function speciesTiles(ids, structures, shuttles) {
  const count = {};
  ids.forEach((id) => { count[id] = (count[id] || 0) + 1; });
  return Object.entries(count).map(([id, n]) => ({ id, n, rec: structures[id], shuttle: shuttles.has(id) }));
}

function Tile({ t }) {
  const img = t.rec?.smiles ? depictUrl(t.rec.smiles, 120, 90) : null;
  return html`<button class=${`rc-tile ${t.shuttle ? 'shuttle' : ''}`} title=${`${t.rec?.name || '?'}${t.shuttle ? ' (shuttle: comes out unchanged)' : ''} — select in Explore`}
    onClick=${() => { select({ structures: [t.id] }); set({ view: { tab: 'graph', jobId: null } }); }}>
    ${img ? html`<img src=${img} alt=${t.rec?.name} />` : html`<span class="mono">${t.rec?.name || '?'}</span>`}
    <span class="rc-name">${t.n > 1 ? `${t.n} × ` : ''}${t.rec?.name || '?'}</span>
    ${t.shuttle && html`<span class="rc-badge" title="Takes part and comes out unchanged: it carries an atom (often H) across">carrier</span>`}
  </button>`;
}

export function Equation({ r, structures }) {
  const shuttles = new Set(r.shuttles || []);
  const side = (ids) => speciesTiles(ids, structures, shuttles);
  const join = (tiles) => tiles.map((t, i) => html`${i ? html`<span class="rc-op">+</span>` : ''}<${Tile} t=${t} />`);
  return html`<div class="rc-eq">${join(side(r.reactants))}<span class="rc-op arrow">→</span>${join(side(r.products))}</div>`;
}

// Energy levels, kcal/mol from the separated reactants.
// Energies from the graph itself (composed reactions, or when the run's own
// are missing): every species and both complexes at one level of theory.
function ladderFromGraph(r, structures) {
  const recs = [...r.reactants, ...r.products].map((id) => structures[id]);
  const cx = (r.complexes || []).map((id) => structures[id]);
  const all = [...recs, ...cx].filter(Boolean);
  if (!recs.every((x) => x && x.energy != null)) return null;
  const key = recs[0].level?.key;
  if (!all.every((x) => x.level?.key === key)) return null;
  const k = 627.509474, sum = (ids) => ids.reduce((a, id) => a + structures[id].energy, 0);
  const base = sum(r.reactants);
  const e = (x) => (x && x.energy != null && x.level?.key === key ? (x.energy - base) * k : null);
  return { reactants: 0, reactant_complex: e(cx[0]), product_complex: e(cx[1]), products: (sum(r.products) - base) * k };
}

function Ladder({ r, st, structures }) {
  const L = (r.ladder && r.ladder.reactant_complex != null ? r.ladder : ladderFromGraph(r, structures)) || r.ladder || {};
  const ts = st.barrier != null && L.reactant_complex != null ? L.reactant_complex + st.barrier
    : null;   // (a path top no TS confirmed is not drawn as one)
  const levels = [
    // One molecule on a side: its "complex" is the molecule itself, no extra level.
    { key: 'reactants', label: r.reactants.length > 1 ? 'separated' : 'reactant', v: L.reactants },
    r.reactants.length > 1 && { key: 'rc', label: 'reactant complex', v: L.reactant_complex },
    { key: 'ts', label: st.barrier != null ? 'TS' : st.barrierUnverified != null ? 'no TS found' : 'TS (not searched)', v: ts },
    r.products.length > 1 && { key: 'pc', label: 'product complex', v: L.product_complex },
    { key: 'products', label: r.products.length > 1 ? 'separated' : 'product', v: L.products },
  ].filter(Boolean);
  const known = levels.filter((l) => l.v != null);
  if (known.length < 2) return html`<p class="small muted">No energies for this reaction yet.</p>`;
  const W = 520, H = 150, pad = 26, lo = Math.min(...known.map((l) => l.v)), hi = Math.max(...known.map((l) => l.v));
  const y = (v) => pad + (H - 2 * pad) * (hi === lo ? 0.5 : (hi - v) / (hi - lo));
  const x = (i) => 20 + i * ((W - 40) / (levels.length - 1));
  const pts = levels.map((l, i) => ({ ...l, x: x(i), y: l.v != null ? y(l.v) : null }));
  const drawn = pts.filter((p) => p.y != null);
  return html`<svg class="rc-ladder" viewBox=${`0 0 ${W} ${H + 18}`} role="img" aria-label="Energy along the reaction">
    ${drawn.slice(1).map((p, i) => html`<line x1=${drawn[i].x + 30} y1=${drawn[i].y} x2=${p.x - 30} y2=${p.y} class="rc-link" />`)}
    ${pts.map((p) => (p.y == null
      ? html`<text x=${p.x} y=${H / 2} class="rc-missing" text-anchor="middle">—</text>`
      : html`<g class=${`rc-level ${p.key}`}>
          <line x1=${p.x - 30} x2=${p.x + 30} y1=${p.y} y2=${p.y} />
          <text x=${p.x} y=${p.y - 6} text-anchor="middle" class="rc-v">${p.v >= 0 ? '+' : ''}${p.v.toFixed(1)}</text>
        </g>`))}
    ${pts.map((p) => html`<text x=${p.x} y=${H + 12} text-anchor="middle" class="rc-l">${p.label}</text>`)}
  </svg>`;
}

function useText(url) {
  const [t, setT] = useState(null);
  useEffect(() => {
    let live = true;
    setT(null);
    if (url) api.get(url).then((x) => live && setT(x)).catch(() => live && setT(''));
    return () => { live = false; };
  }, [url]);
  return t;
}

// The settings of a TS search on a reaction: the last search's (to change
// what went wrong), else the defaults; any compute profile (path method,
// images and level of theory live there).
function SearchSettings({ r, onDone }) {
  const op = useStore((s) => s.operations.find((o) => o.key === 'ts'));
  const prev = Object.values(state.jobs).filter((j) => j.op === 'ts' && j.targets.edges.includes(r.edge))
    .sort((a, b) => b.created - a.created)[0];
  const schema = op?.schema;
  const [values, setValues] = useState(() => (schema ? clampToSchema(schema, { ...defaultsFor(schema), ...(prev?.params || {}) }) : {}));
  const [profile, setProfile] = useState(prev?.profile ?? defaultProfile());
  const [busy, setBusy] = useState(false);
  if (!schema) return html`<p class="small muted">The TS search is not available here.</p>`;
  const run = async () => {
    setBusy(true);
    const out = await attempt(() => api.post('/api/jobs', { op: 'ts', edges: [r.edge], params: values, profile }), 'TS search queued');
    setBusy(false);
    if (out) onDone(out[0]);
  };
  return html`<div class="rc-settings">
    ${prev && html`<p class="small muted">Last search: ${jobHeadline(prev) || prev.status} · profile <b>${prev.profile || 'built-in defaults'}</b>.
      Change what to try: another profile (its path method, e.g. GSM, or more NEB images), or a single path instead of recursive splitting.</p>`}
    <${ParamForm} op="ts" schema=${schema} values=${values} onChange=${setValues} />
    <${ProfilePicker} value=${profile} onChange=${setProfile} />
    <div class="op-run">
      <button class="btn primary" disabled=${busy} onClick=${run}>${busy ? 'Queuing…' : 'Run TS search'}</button>
      <button class="btn-link small" onClick=${() => onDone(null)}>Cancel</button>
    </div>
  </div>`;
}

// A reactor reaction with no ends to search between (barrierless, products
// falling back...) in the gas phase can have them in a solvent, which
// stabilizes charged and polar species: try it, one solvent at a time
// (POST /api/reactions/{rid}/retry), and see each try's outcome here.
const FIRST_SOLVENTS = ['water', 'methanol', 'dmso', 'acetonitrile', 'thf', 'toluene'];
export function RetryInSolvent({ r }) {
  const solvents = useStore((s) => s.solvents) || [];
  const operations = useStore((s) => s.operations);
  const jobsKey = useStore((s) => Object.values(s.jobs).filter((j) => j.op === 'reaction-retry' || j.op === 'ts')
    .map((j) => `${j.id}:${j.status}`).join('|'));
  const [solvent, setSolvent] = useState('water');
  const [busy, setBusy] = useState(false);
  if (!r.origin?.nanoreactor || !operations.some((o) => o.key === 'reaction-retry' && o.available)) return null;
  // The latest try in each solvent.
  const seen = new Set();
  const tries = Object.values(state.jobs).filter((j) => j.op === 'reaction-retry' && j.params?.rid === r.id)
    .sort((a, b) => b.created - a.created).filter((j) => !seen.has(j.params?.solvent) && seen.add(j.params?.solvent));
  const outcome = r.retries || {};
  const ordered = [...FIRST_SOLVENTS.map((k) => solvents.find((x) => x.key === k)).filter(Boolean),
    ...solvents.filter((x) => !FIRST_SOLVENTS.includes(x.key))];
  const name = (k) => solvents.find((x) => x.key === k)?.label || k;
  const go = async () => {
    setBusy(true);
    await attempt(() => api.post(`/api/reactions/${r.id}/retry`, { solvent }), `Trying it in ${name(solvent).toLowerCase()}`);
    setBusy(false);
  };
  const tsOn = (eid) => Object.values(state.jobs).filter((j) => j.op === 'ts' && j.targets.edges.includes(eid))
    .sort((a, b) => b.created - a.created)[0];
  const line = (j) => {
    const o = outcome[j.id];
    const who = html`<b>${name(j.params?.solvent)}</b>`;
    if (!o) {
      if (['queued', 'running'].includes(j.status)) return html`${who}: optimizing its ends… <a href="#" onClick=${(e) => { e.preventDefault(); openJob(j.id); }}>watch</a>`;
      if (j.status === 'failed') return html`${who}: the try failed (<a href="#" onClick=${(e) => { e.preventDefault(); openJob(j.id, 'log'); }}>log</a>)`;
      return html`${who}: reading the outcome…`;
    }
    if (!o.edge) return html`${who}: still no ends to search between: ${o.error}.`;
    const ts = tsOn(o.edge);
    const tsDone = ts?.status === 'done' && ts.summary?.barrier_kcal != null;
    return html`${who}: the ends hold${o.relaxed ? html` (as ${o.relaxed})` : ''}${o.delta_e_kcal != null ? `, ΔE ${o.delta_e_kcal > 0 ? '+' : ''}${fmtKcal(o.delta_e_kcal)}` : ''}.
      ${tsDone ? html` <a href="#" onClick=${(e) => { e.preventDefault(); openJob(ts.id); }}>ΔE‡ ${fmtKcal(ts.summary.barrier_kcal)} kcal/mol →</a>`
        : ts && ['queued', 'running'].includes(ts.status) ? html` TS search ${ts.status}… <a href="#" onClick=${(e) => { e.preventDefault(); openJob(ts.id, 'live'); }}>watch</a>`
          : html` <button class="btn small primary" onClick=${() => attempt(() => api.post('/api/jobs', { op: 'ts', edges: [o.edge], params: {}, profile: o.profile }),
            `TS search queued in ${name(j.params?.solvent).toLowerCase()}`)}>Find TS</button>`}`;
  };
  return html`<div class="rc-retry small">
    ${tries.length > 0 && html`<ul>${tries.map((j) => html`<li key=${j.id}>${line(j)}</li>`)}</ul>`}
    <div class="rc-retry-row">
      <span class="muted">${tries.length ? 'Another solvent?' : 'That is in the gas phase. A solvent can stabilize charged or polar ends.'}</span>
      <select value=${solvent} onChange=${(e) => setSolvent(e.target.value)} aria-label="Solvent">
        ${ordered.map((x) => html`<option value=${x.key}>${x.label}</option>`)}</select>
      <button class="btn small" disabled=${busy} onClick=${go}
        title="Optimize this reaction's ends again from its MD frames, at the same level plus this implicit solvent">Try in ${name(solvent).toLowerCase()}</button>
      <${Why} id="op.reaction-retry" label="Retry in a solvent" />
    </div>
  </div>`;
}

export function ReactionCard({ r, event = null, compact = false }) {
  const nav = useStore((s) => s.analyze) || {};
  const [settings, setSettings] = useState(() => Boolean(nav.retry && nav.reaction === r.id));
  useEffect(() => { if (nav.retry && nav.reaction === r.id) setSettings(true); }, [nav.retry]);
  const structures = useStore((s) => s.workspace.structures);
  const edges = useStore((s) => s.workspace.edges);
  useStore((s) => s.jobs);
  const edge = r.edge ? edges[r.edge] : null;
  const st = edge ? edgeStatus(edge, state.jobs) : { status: 'idle' };
  const events = r.events || [];
  const [ev, setEv] = useState(event ?? events[0] ?? null);
  const [view, setView] = useState(ev != null ? 'event' : 'rc');
  const [rc, pc] = r.complexes || [];
  const xyzR = useText(view === 'rc' && rc ? `/api/structures/${rc}/xyz` : null);
  const xyzP = useText(view === 'pc' && pc ? `/api/structures/${pc}/xyz` : null);
  // The structure behind the barrier: the verified TS, or (unverified) the path's highest point.
  const tsUrl = st.barrierJob ? `/api/jobs/${st.barrierJob}/route-ts${r.origin?.job === st.barrierJob && r.origin?.index != null ? `?reaction=${r.origin.index}` : ''}` : null;
  const xyzTS = useText(view === 'ts' ? tsUrl : null);
  const verified = st.barrier != null;
  const job = r.origin?.job;
  const running = st.status === 'running' || st.status === 'queued';
  const runningJob = running ? Object.values(state.jobs).find((j) => j.targets.edges.includes(r.edge) && ['running', 'queued'].includes(j.status)) : null;
  const views = [
    ev != null && job && ['event', 'In the reactor'],
    rc && ['rc', 'Reactant complex'],
    pc && ['pc', 'Product complex'],
    st.barrierJob && ['ts', verified ? 'TS' : 'Path top'],
  ].filter(Boolean);
  const height = compact ? 230 : 340;
  return html`<div class=${`reaction-card ${compact ? 'compact' : ''}`}>
    <${Equation} r=${r} structures=${structures} />
    <div class="small muted rc-meta" title="Energies in kcal/mol, all over the same atoms">${r.origin?.kind === 'composed' ? 'composed by hand' : r.retro ? `retrosynthesis step · ${r.retro.method}${r.retro.reagents?.length ? ` · reagents ${r.retro.reagents.join(', ')}` : ''}` : html`seen ${(r.count || 0) === 1 ? 'once' : `${r.count || 0}×`} in the reactor${r.reverse_count ? `, reverse ${r.reverse_count}×` : ''}`}
      ${compact && html` · <a href="#" onClick=${(e) => { e.preventDefault(); openAnalyze({ reaction: r.id }); }}>open in Analyze</a>`}</div>
    ${r.relaxed_from && html`<p class="small muted rc-relaxed" title=${`The reactor showed ${r.relaxed_from}. ${(r.relaxed_why || []).join(' ')} This is the reaction between the relaxed ends.`}>
      <span class="has-tip">Changed when relaxed</span></p>`}
    <${Ladder} r=${r} st=${st} structures=${structures} />

    <div class="rc-ts">
      ${!edge ? html`<p class="small warn-box">${noTsWhy(r)}
          ${['barrierless', 'recombines'].includes(r.complex_reason) ? ' There is no barrier to find here; its ΔE is the reaction energy.' : ''}</p>
          <${RetryInSolvent} r=${r} />`
        : running ? html`<p class="small">TS search ${st.status}…
            ${runningJob && html` <a href="#" onClick=${(e) => { e.preventDefault(); openJob(runningJob.id, 'live'); }}>watch it live</a>`}</p>`
          : html`<div class="rc-ts-row">
            ${st.levels?.length > 1 ? html`<span class="rc-barrier">${st.levels.map((l, i) => html`${i ? ' · ' : ''}${l.job
              ? html`<a href="#" title="Its TS search at this level of theory" onClick=${(e) => { e.preventDefault(); openJob(l.job); }}>${levelBarrierText(l)}</a>`
              : levelBarrierText(l)}`)} kcal/mol</span>`
            : st.barrierJob ? html`<a href="#" class="rc-barrier" onClick=${(e) => { e.preventDefault(); openJob(st.barrierJob); }}
                title="Its TS search: TS, IRC, energy profile, sample more paths">
                ${st.barrier != null ? `ΔE‡ ${fmtKcal(st.barrier)} kcal/mol from the reactant complex →` : 'No transition state found →'}</a>
                ${st.barrier == null && st.barrierUnverified != null && html`<span class="small muted rc-nots">${noTsHint(st.barrierUnverified)}.</span>`}`
              : html`<span class="small muted">No TS yet.</span>`}
            <span class="spacer"></span>
            ${st.barrierJob
              ? html`<button class="btn" onClick=${() => setSettings(!settings)} aria-expanded=${settings}
                  title="Change the settings of the last search (path method, profile, ...) and search again">Search again…</button>`
              : html`<button class="btn primary" onClick=${() => findTs(r)}
                  title="Path search, TS optimization and IRC between the optimized reactant and product complexes (only this reaction's molecules), with the default settings">
                  Find transition state</button>
                <button class="btn-link small" onClick=${() => setSettings(!settings)} aria-expanded=${settings}>settings…</button>`}
          </div>
          ${settings && html`<${SearchSettings} r=${r} onDone=${(job) => { setSettings(false); if (state.analyze?.retry) set({ analyze: { ...state.analyze, retry: null } }); }} />`}`}
    </div>

    ${views.length > 0 && html`<div class="rc-view">
      <div class="segmented wide">
        ${views.map(([k, l]) => html`<button class=${view === k ? 'on' : ''} onClick=${() => setView(k)}>${l}</button>`)}
      </div>
      ${view === 'event' && events.length > 1 && html`<div class="rc-events small">MD occurrence:
        ${events.map((e, i) => html`<button class=${`btn-link small ${e === ev ? 'on' : ''}`} onClick=${() => setEv(e)}>${i + 1}</button>`)}</div>`}
      ${view === 'event' && ev != null && job && html`<${EventPlayer} key=${`${job}:${ev}`} jobId=${job} event=${ev} height=${height} />`}
      ${view === 'rc' && html`<${Viewer3D} xyz=${xyzR} height=${height} />
        <p class="small muted">${r.origin?.kind === 'composed' ? 'The reactants packed together, then optimized together.'
          : 'Only the molecules this reaction needs, optimized together (from the MD frame before it).'}</p>`}
      ${view === 'pc' && html`<${Viewer3D} xyz=${xyzP} height=${height} />`}
      ${view === 'ts' && (xyzTS === ''
        ? html`<p class="small muted">No geometry for this search's barrier (its output is gone or holds no path).</p>`
        : html`<${Viewer3D} xyz=${xyzTS} height=${height} orbitals=${structures[rc] || null} />
          <p class="small muted">${verified ? 'The transition state (TS optimization, IRC-verified).'
            : 'The highest point of the path: no TS optimization or IRC has confirmed it, so the barrier is approximate.'}</p>`)}
    </div>`}
    <${ReactionExport} r=${r} />
  </div>`;
}

// This reaction as files to share, with every channel found for it: one
// self-contained page (opens offline, embeds in an iframe) or its data for SI.
function ReactionExport({ r }) {
  const url = (format) => `/api/reactions/${r.id}/export?format=${format}`;
  return html`<div class="rc-export small">
    <span class="muted">Export</span>
    <a class="btn small" href=${url('html')} download
      title="One HTML file: equation, energy diagram with every channel found, each channel's TS and IRC in 3D, energies and level of theory. Opens without a server or internet; embeds in an iframe.">Report (.html)</a>
    <a class="btn small" href=${url('zip')} download
      title="The data behind it: an XYZ per structure, each channel's TS and IRC, energies as CSV and JSON, the diagram as SVG, the profiles used, a README, and the report">Data (.zip)</a>
  </div>`;
}
