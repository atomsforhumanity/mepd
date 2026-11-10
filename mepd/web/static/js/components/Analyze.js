// Analyze: the workspace's reactions as a list to compare, filter and act on
// (Explore is the map; this is where barriers are looked at and asked for).
//  * reactions are grouped by what they do (net reactants -> net products),
//    so parallel routes -- direct, water-shuttled, ... -- sit together with
//    their barriers;
//  * the selected reaction's card: species, energy ladder, complexes, MD
//    event, TS;
//  * New reaction: click molecules of the graph, then the products they can
//    become (NewReaction.js).
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, select, set, state, toast, useStore } from '../store.js';
import { edgeStatus, fmtKcal, isSpecies, levelBarrierText } from '../util.js';
import { NO_TS_SHORT, ReactionCard, findTs, noTsHint, noTsWhy } from './Reactions.js';
import { LandscapeView } from './Landscape.js';
import { KineticsView } from './Kinetics.js';
import { ReactionMaker } from './NewReaction.js';
import { markQuest } from '../quests.js';

const count = (ids) => ids.reduce((m, id) => ({ ...m, [id]: (m[id] || 0) + 1 }), {});

// What a reaction does, shuttles taken out: the key its parallel routes share.
function netKey(r) {
  const sh = count(r.shuttles || []);
  const side = (ids) => {
    const c = count(ids);
    return Object.keys(c).sort().flatMap((id) => Array(Math.max(0, c[id] - (sh[id] || 0))).fill(id)).join('+');
  };
  return `${side(r.reactants)}>${side(r.products)}`;
}

function netLabel(r, structures) {
  const sh = count(r.shuttles || []);
  const side = (ids) => {
    const c = count(ids);
    return Object.entries(c).filter(([id, n]) => n - (sh[id] || 0) > 0)
      .map(([id, n]) => `${n - (sh[id] || 0) > 1 ? `${n - (sh[id] || 0)} ` : ''}${structures[id]?.name || '?'}`).join(' + ');
  };
  return `${side(r.reactants)} → ${side(r.products)}`;
}

function routeLabel(r, structures) {
  const sh = [...new Set(r.shuttles || [])].map((id) => structures[id]?.name || '?');
  return sh.length ? `via ${sh.join(', ')}` : 'direct';
}

function tsState(r, edges, jobs) {
  const edge = r.edge ? edges[r.edge] : null;
  if (!edge) return { kind: 'none', text: NO_TS_SHORT[r.complex_reason] || 'no TS endpoints', title: noTsWhy(r) };
  const st = edgeStatus(edge, jobs);
  if (st.status === 'running' || st.status === 'queued') return { kind: 'busy', text: `${st.status}…` };
  if (st.levels?.length > 1) {   // each level of theory its own lowest; sorted by the workspace level's
    return { kind: st.barrier != null ? 'ts' : 'unverified', text: st.levels.map((l) => levelBarrierText(l)).join(' · '),
      v: st.barrier, title: 'Lowest barrier at each level of theory (not comparable across levels)' };
  }
  if (st.barrier != null) return { kind: 'ts', text: `ΔE‡ ${fmtKcal(st.barrier)}`, v: st.barrier };
  if (st.barrierUnverified != null) return { kind: 'unverified', text: 'no TS', v: null, title: `No transition state found: ${noTsHint(st.barrierUnverified)}` };
  if (edge.origin?.qmmm_ts) return { kind: 'busy', text: 'TS, barrier pending', title: edge.origin.headline || '' };
  return { kind: 'todo', text: 'no TS yet' };
}

// Where a reaction came from, in a word (a nanoreactor's: how often the MD saw it).
function sourceLabel(r) {
  if (r.origin?.kind === 'edge') return 'TS on this edge';
  if (r.origin?.kind === 'composed') return 'composed';
  if (r.origin?.retro) return 'retrosynthesis';
  return r.count ? `seen ${r.count}×` : '';
}

// The whole workspace as files: fetched first (building it takes a while on a
// large workspace), so the link can say so -- and say why if it fails -- rather
// than look dead until the download starts.
const EXPORTS = [
  ['html', 'report (.html)', 'One HTML file to browse every reaction: an index (lowest barrier, ΔE, channels), each opening its page with the energy diagram and every channel\'s TS and IRC in 3D. Opens without a server or internet.'],
  ['csv', 'data (.csv)', 'One row per elementary step of every channel found (and one per reaction with none): fixed columns with units, SMILES, energies, the TS and step-end geometries inline. For loading in bulk.'],
];
function ExportAll() {
  const [busy, setBusy] = useState(null);
  const run = async (format) => {
    if (busy) return;
    setBusy(format);
    toast('Preparing the export: on a large workspace this takes a minute.', 'info', 4000);
    await attempt(async () => {
      const res = await api.raw(`/api/workspace/export?format=${format}`);
      const blob = await res.blob();
      const name = /filename="([^"]+)"/.exec(res.headers.get('content-disposition') || '')?.[1] || `workspace.${format}`;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      return name;
    }, (name) => `Saved ${name}`);
    setBusy(null);
  };
  return html`Export all: ${EXPORTS.map(([f, label, title], i) => html`${i ? ', ' : ''}<button class="btn-link small"
    title=${title} disabled=${!!busy} onClick=${() => run(f)}>${busy === f ? html`<span class="spinner" aria-hidden="true"></span>preparing ${label}…` : label}</button>`)}`;
}

// ------------------------------------------------------------ the view
export function AnalyzeView() {
  const mode = useStore((s) => (s.analyze || {}).view) || 'reactions';
  useEffect(() => { if (mode === 'kinetics') markQuest('kinetics'); }, [mode]);
  // (Reactions and Kinetics are the canvas's own view tabs: Shell.js CanvasHeader.)
  if (mode === 'kinetics') return html`<div class="analyze-wrap"><${KineticsView} /></div>`;
  if (mode === 'landscape') return html`<div class="analyze-wrap"><${LandscapeView} /></div>`;
  return html`<div class="analyze-wrap"><${ReactionsAnalyze} /></div>`;
}

function ReactionsAnalyze() {
  const ws = useStore((s) => s.workspace);
  const jobs = useStore((s) => s.jobs);
  const nav = useStore((s) => s.analyze) || {};
  const [q, setQ] = useState('');
  const [only, setOnly] = useState('all');   // all | todo | ts | shuttle
  const structures = ws.structures, edges = ws.edges;
  // Edges no reaction stands for (two structures you connected) that a TS
  // search ran on, or that carry a barrier of their own (e.g. a TS attached
  // in a QM/MM solvent): listed too, as a reaction of their two ends.
  const ofEdge = new Set(Object.values(ws.reactions || {}).map((r) => r.edge).filter(Boolean));
  const computed = (e) => edgeStatus(e, jobs).count > 0 || e.origin?.barrier_kcal != null || !!e.origin?.qmmm_ts;
  const edgeRows = Object.values(edges).filter((e) => !ofEdge.has(e.id) && isSpecies(structures[e.source])
    && isSpecies(structures[e.target]) && computed(e))
    .map((e) => ({ id: `edge:${e.id}`, reactants: [e.source], products: [e.target], edge: e.id, origin: { kind: 'edge' } }));
  const all = [...Object.values(ws.reactions || {}), ...edgeRows];
  const filterIds = nav.species || null;      // from Explore: reactions of these species
  const shown = all.filter((r) => {
    const ids = [...r.reactants, ...r.products];
    if (filterIds && !filterIds.every((id) => ids.includes(id))) return false;
    if (q) {
      const text = ids.map((id) => `${structures[id]?.name} ${structures[id]?.smiles}`).join(' ').toLowerCase();
      if (!text.includes(q.toLowerCase())) return false;
    }
    const t = tsState(r, edges, jobs);
    if (only === 'todo' && !['todo', 'unverified'].includes(t.kind)) return false;   // (no TS found: worth another try)
    if (only === 'ts' && t.kind !== 'ts') return false;
    if (only === 'shuttle' && !(r.shuttles || []).length) return false;
    return true;
  });
  const groups = {};
  for (const r of shown) (groups[netKey(r)] ||= []).push(r);
  const ordered = Object.values(groups).map((rs) => rs.sort((a, b) => (tsState(a, edges, jobs).v ?? 1e9) - (tsState(b, edges, jobs).v ?? 1e9)))
    .sort((a, b) => b.length - a.length || (tsState(a[0], edges, jobs).v ?? 1e9) - (tsState(b[0], edges, jobs).v ?? 1e9));
  const selected = nav.reaction && ws.reactions?.[nav.reaction] ? ws.reactions[nav.reaction] : null;
  const pick = (rid) => {
    if (rid.startsWith('edge:')) {   // no reaction card: its calculation
      const edge = edges[rid.slice(5)];
      const job = edge && edgeStatus(edge, jobs).barrierJob;
      if (job) openJob(job); else { select({ structures: [], edges: [rid.slice(5)] }); set({ view: { tab: 'graph', jobId: null } }); }
      return;
    }
    set({ analyze: { ...nav, reaction: rid, compose: false } });
  };
  const todo = shown.filter((r) => r.origin?.kind !== 'edge' && tsState(r, edges, jobs).kind === 'todo');
  const findAll = async () => {
    if (!window.confirm(`Queue ${todo.length} TS search${todo.length > 1 ? 'es' : ''}? They run a few at a time.`)) return;
    for (const r of todo) await findTs(r);
  };

  if (!all.length && !nav.compose) {
    return html`<div class="analyze empty-hint">
      <h3>No reactions yet</h3>
      <p>Grow a network with the <b>Nanoreactor</b> (Explore › select a molecule, or a complex of several › Network expansion), or compose a reaction
        from species you already have.</p>
      <button class="btn primary" onClick=${() => set({ analyze: { ...nav, compose: true } })}>New reaction</button>
    </div>`;
  }
  return html`<div class="analyze">
    <div class="an-list">
      <div class="an-bar">
        <input type="search" placeholder="Filter by species" value=${q} onInput=${(e) => setQ(e.target.value)} />
        <div class="segmented">
          ${[['all', 'All'], ['todo', 'No TS yet'], ['ts', 'With TS'], ['shuttle', 'With a carrier']].map(([k, l]) => html`
            <button class=${only === k ? 'on' : ''} onClick=${() => setOnly(k)}>${l}</button>`)}
        </div>
        <button class="btn primary small" onClick=${() => set({ analyze: { ...nav, compose: true, reaction: null } })}>New reaction</button>
      </div>
      ${filterIds && html`<p class="small an-filter">Reactions of ${filterIds.map((id) => structures[id]?.name || '?').join(' + ')}
        <button class="btn-link small" onClick=${() => set({ analyze: { ...nav, species: null } })}>show all</button></p>`}
      <div class="small muted an-count">${shown.length} reaction${shown.length === 1 ? '' : 's'} · ${ordered.length} transformation${ordered.length === 1 ? '' : 's'}
        ${todo.length > 0 && html` · <button class="btn-link small" onClick=${findAll}>find TS for the ${todo.length} without one</button>`}
        ${' · '}<${ExportAll} /></div>
      <div class="an-groups">
        ${ordered.map((rs) => html`<div class="an-group">
          <div class="an-net">${netLabel(rs[0], structures)}${rs.length > 1 ? html` <span class="small muted">· ${rs.length} routes</span>` : ''}</div>
          ${rs.map((r) => {
            const t = tsState(r, edges, jobs);
            return html`<button class=${`an-route ${selected?.id === r.id ? 'on' : ''}`} onClick=${() => pick(r.id)}>
              <span class="an-via">${routeLabel(r, structures)}</span>
              <span class=${`an-ts ${t.kind}`} title=${t.title || ''}>${t.text}</span>
              <span class="an-de mono small">${r.delta_e_kcal != null ? `ΔE ${r.delta_e_kcal >= 0 ? '+' : ''}${fmtKcal(r.delta_e_kcal)}` : ''}</span>
              <span class="an-src small muted">${sourceLabel(r)}</span>
            </button>`;
          })}
        </div>`)}
      </div>
    </div>
    <div class="an-detail">
      ${nav.compose ? html`<${ReactionMaker} initial=${nav.composeReactants}
          onDone=${(rid, stay) => set({ analyze: { ...nav, compose: !!stay, composeReactants: stay ? nav.composeReactants : null, reaction: rid || nav.reaction } })} />`
        : selected ? html`<${ReactionCard} key=${selected.id} r=${selected} />`
          : html`<div class="empty-hint"><p>Pick a reaction on the left: routes that do the same thing are grouped, the lowest barrier first.</p></div>`}
    </div>
  </div>`;
}

// From Explore: open Analyze on a reaction, on the reactions of some
// species, or with the composer started from them.
export function openAnalyze(patch) {
  set({ analyze: { ...(state.analyze || {}), compose: false, view: 'reactions', ...patch }, view: { tab: 'analyze', jobId: state.view.jobId } });
}
