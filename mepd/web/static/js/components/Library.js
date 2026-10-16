// Left sidebar: add structures (SMILES / xyz text / dropped files) and
// browse the library. Library entries and graph nodes are the same thing.
import { html, useState } from '../lib.js';
import { api, attempt, refreshState } from '../api.js';
import { isNew, markAllSeen, openTab, prefs, select, set, state, useStore } from '../store.js';
import { cls, depictUrl, edgeStatus, fmtKcal, isComplex, isSpecies, levelStatus } from '../util.js';
import { QmmmAdd } from './QMMM.js';
import { markQuest, useMap } from '../quests.js';
import { Why } from './Why.js';

// Point the graph at a node or an arrow: centre it, close enough to read, and
// pulse it. From another tab, Explore opens first (and is laid out) for a reaction.
function findInGraph(id, open = false) {
  const go = () => window.dispatchEvent(new CustomEvent('mepd:graph', { detail: { cmd: 'center', id, zoom: true, flash: true } }));
  if (state.view.tab === 'graph') go();
  else if (open) { openTab('graph'); setTimeout(go, 300); }
}

// "Added 2 structures · 1 new conformer of CCO · 1 already there"
function addedSummary(list) {
  const fresh = list.filter((a) => !a.merged).length;
  const conf = list.filter((a) => a.merged && !a.duplicate);
  const dup = list.filter((a) => a.duplicate).length;
  const parts = [];
  if (fresh) parts.push(`Added ${fresh} structure${fresh === 1 ? '' : 's'}`);
  if (conf.length) parts.push(`${conf.length} new conformer${conf.length === 1 ? '' : 's'} of ${[...new Set(conf.map((a) => a.name))].join(', ')}`);
  if (dup) parts.push(`${dup} already in the graph`);
  const edges = list.filter((a) => a.edge).length;
  if (edges) parts.push(`${edges} reaction edge${edges === 1 ? '' : 's'}`);
  return parts.join(' · ') || 'Nothing new';
}

export async function uploadFiles(files, { charge = null, multiplicity = null } = {}) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  fd.append('optimize', prefs.get('optimizeOnAdd', true) ? 'true' : 'false');
  if (charge !== null && charge !== '') fd.append('charge', charge);
  if (multiplicity !== null && multiplicity !== '') fd.append('multiplicity', multiplicity);
  return attempt(() => api.post('/api/structures/upload', fd),
    (added) => `Added ${added.length} structure${added.length === 1 ? '' : 's'}`);
}

export function AddBox({ onDone }) {
  const [text, setText] = useState('');
  const [charge, setCharge] = useState('');
  const [mult, setMult] = useState('');
  const [busy, setBusy] = useState(false);
  const [optimize, setOptimize] = useState(() => prefs.get('optimizeOnAdd', true));
  const validate = useStore((s) => s.validateMinima);
  const level = useStore((s) => s.levels[s.levelProfile ?? '']);
  const isXyz = /^\s*\d+\s*\n/.test(text);
  const n = isXyz ? null : text.split('\n').filter((l) => l.trim() && !l.trim().startsWith('#')).length;
  const rxns = !isXyz && text.split('\n').some((l) => ((l.trim().split(/\s/)[0] || '').match(/>/g) || []).length === 2);

  const add = async () => {
    if (!text.trim()) return;
    setBusy(true);
    const added = await attempt(() => api.post('/api/structures', {
      text, charge: charge === '' ? null : +charge, multiplicity: mult === '' ? null : +mult, optimize,
    }), (a) => `${addedSummary(a)}${optimize && a.some((x) => !x.duplicate) ? ` · optimizing at ${level?.label ?? 'the workspace level'}` : ''}`);
    setBusy(false);
    if (added) {
      setText('');
      select({ structures: [...new Set(added.map((a) => a.id))] });
      onDone?.();
    }
  };
  return html`
    <div class="add-box">
      <textarea rows="3" value=${text} spellcheck="false"
        placeholder=${'SMILES, one per line (a name may follow a space); a reaction\nSMILES (reactants>>products) adds both ends and their edge;\nor pasted XYZ. You can also drop .xyz / .smi files.'}
        onInput=${(e) => setText(e.target.value)}
        onKeyDown=${(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) add(); }} />
      <div class="add-row">
        <input type="number" class="tiny" placeholder="Charge" value=${charge} onInput=${(e) => setCharge(e.target.value)} title="Charge (default: from SMILES / xyz)" />
        <input type="number" class="tiny" placeholder="Mult." min="1" value=${mult} onInput=${(e) => setMult(e.target.value)} title="Spin multiplicity (default: from SMILES / xyz)" />
        <button class="btn primary" disabled=${busy || !text.trim()} onClick=${add}>
          ${busy ? 'Embedding…' : isXyz ? 'Add XYZ' : rxns && n === 1 ? 'Add reaction' : n > 1 ? `Add ${n}` : 'Add'}
        </button>
      </div>
      <label class="small opt-toggle" title="Minimize at the workspace level of theory so every structure in the graph lives on the same potential energy surface">
        <input type="checkbox" checked=${optimize} onChange=${(e) => { setOptimize(e.target.checked); prefs.set('optimizeOnAdd', e.target.checked); }} />
        <span>Minimize at the level of theory first${validate ? ', then check with a Hessian' : ''}</span>
      </label>
      <details class="qmmm-add">
        <summary class="small">QM/MM system (a protein or solvated system with a QM region)…</summary>
        <${QmmmAdd} onDone=${(sid) => { if (sid) select({ structures: [sid] }); onDone?.(); }} />
      </details>
    </div>`;
}

export function LevelChip({ rec }) {
  useStore((s) => [s.levelProfile, s.levels]);
  const st = levelStatus(rec);
  return html`<span class=${`level-chip level-${st.kind}`} title=${st.title}>${st.text}</span>`;
}

// `status`: only the line saying what is not at the level yet (the sidebar's);
// the choice itself lives in Settings.
export function LevelBar({ status = false, always = false }) {
  const map = useMap();
  const levelProfile = useStore((s) => s.levelProfile);
  const profiles = useStore((s) => s.profiles);
  const levels = useStore((s) => s.levels);
  const validate = useStore((s) => s.validateMinima);
  const structures = useStore((s) => s.workspace.structures);
  // TS structures (kind 'ts') are never counted: minimizing them would destroy them.
  const off = Object.values(structures).filter((r) => !isComplex(r) && ['other', 'none', 'failed'].includes(levelStatus(r, levels[levelProfile ?? '']?.key).kind));
  const busy = Object.values(structures).filter((r) => r.status === 'optimizing').length;
  const setLevel = (profile) => attempt(() => api.put('/api/level', { profile: profile || null })).then((r) => { if (r) markQuest('method'); });
  const reopt = () => attempt(() => api.post('/api/structures/reoptimize', { structures: off.map((r) => r.id) }),
    (js) => `Re-optimizing ${off.length} structure${off.length > 1 ? 's' : ''} (${js.length} job${js.length > 1 ? 's' : ''})`);
  if (!always && !map.feature('level')) return null;   // opens with "A better method" (quests.js)
  const note = (busy > 0 || off.length > 0) && html`<div class="level-status small">
    ${busy > 0 && html`<span class="muted">${busy} minimizing…</span>`}
    ${off.length > 0 && html`<span class="muted">${off.length} not optimized ·</span> <button class="btn-link small" onClick=${reopt}
      title=${`Minimize every structure that is not yet a minimum at the workspace level, ${levels[levelProfile ?? '']?.label ?? 'this level'} (force-field embeddings, other levels, failed optimizations)`}>
      Optimize</button>`}
  </div>`;
  if (status) return note ? html`<div class="level-bar">${note}</div>` : null;
  return html`
    <div class="level-bar">
      <label class="level-field" title="Every structure in this session is minimized at this level of theory, so all energies are comparable. Edit a profile's engine and method under Settings.">
        <span class="field-label">Level of theory<${Why} id="level.profile" /></span>
        <select value=${levelProfile ?? ''} onChange=${(e) => setLevel(e.target.value)}>
          ${profiles.map((p) => html`<option value=${p}>${levels[p]?.label ?? p} · ${p}</option>`)}
          <option value="">${levels['']?.label ?? ''} · mepd defaults</option>
        </select>
      </label>
      <label class="check small" title="After a structure is minimized, compute its Hessian and require no imaginary frequency beyond 20i cm⁻¹. A structure stuck on a saddle point is pushed along its unstable mode and re-minimized; if that fails it is flagged 'not a minimum'.">
        <input type="checkbox" class="switch" checked=${validate}
          onChange=${(e) => attempt(() => api.put('/api/level', { validate_minima: e.target.checked }))
            .then((r) => r && refreshState())} />
        Confirm minima with a Hessian<${Why} id="level.validate_minima" />
      </label>
      ${note}
    </div>`;
}

function Card({ rec, selected, order, fresh }) {
  const url = depictUrl(rec.smiles, 180, 110);
  const origin = rec.origin?.kind;
  return html`
    <li class=${cls('card', selected && 'selected')} draggable="false"
      onClick=${(e) => {
        const several = e.shiftKey || e.metaKey || e.ctrlKey;
        select({ structures: [rec.id] }, several);
        if (several) window.dispatchEvent(new CustomEvent('mepd:graph', { detail: { cmd: 'center', id: rec.id, ifHidden: true } }));
        else findInGraph(rec.id);
      }}
      onDblClick=${() => findInGraph(rec.id, true)}
      title=${`${rec.smiles || rec.formula} · ${origin === 'job' ? `from ${rec.origin.label || 'a calculation'}` : origin === 'smiles' ? 'entered as SMILES' : 'from an .xyz'}`}>
      <div class="card-img">
        ${url ? html`<img src=${url} alt="" loading="lazy" />` : html`<span class="muted small">${rec.formula}</span>`}
        ${order > 0 && html`<span class="order-badge" title="Selection order (1 = start)">${order}</span>`}
      </div>
      <div class="card-body">
        <div class="card-name">${fresh && html`<span class="new-dot" title="New: a calculation added it since you last looked"></span>`}${rec.name}</div>
        <div class="card-meta">
          <span>${rec.formula}</span>
          ${(rec.conformers || []).length > 1 && html`<span class="badge" title="Conformers of this molecule; the lowest-energy one represents it">${rec.conformers.length} conformers</span>`}
          ${rec.charge !== 0 && html`<span class="badge" title=${`Charge ${rec.charge > 0 ? '+' : ''}${rec.charge}`}>${rec.charge > 0 ? '+' : '−'}${Math.abs(rec.charge) > 1 ? Math.abs(rec.charge) : ''}</span>`}
          ${rec.multiplicity !== 1 && html`<span class="badge" title=${`Spin multiplicity ${rec.multiplicity}: ${rec.multiplicity - 1} unpaired electron${rec.multiplicity > 2 ? 's' : ''}`}>${rec.multiplicity === 2 ? 'radical' : rec.multiplicity === 3 ? 'triplet' : `mult. ${rec.multiplicity}`}</span>`}
          ${levelStatus(rec).kind !== 'ok' && html`<${LevelChip} rec=${rec} />`}
        </div>
      </div>
    </li>`;
}

// One reaction (an Explore arrow): both ends drawn, its barrier, newest first.
function ReactionRow({ e, selected, fresh }) {
  const structures = useStore((s) => s.workspace.structures);
  useStore((s) => s.jobs);   // its barrier as searches finish
  const a = structures[e.source], b = structures[e.target];
  if (!a || !b) return null;
  const st = edgeStatus(e);
  const tag = st.status === 'running' || st.status === 'queued' ? html`<span class="muted">searching…</span>`
    : st.barrier != null ? html`<b>${fmtKcal(st.barrier)}</b> <span class="muted">kcal/mol</span>`
    : st.barrierUnverified != null ? html`<span class="muted" title="Highest point on the path; no transition state confirmed it">no TS found</span>`
    : st.status === 'failed' ? html`<span class="warn-text">failed</span>`
    : html`<span class="muted">no barrier yet</span>`;
  const pic = (r) => { const u = depictUrl(r.smiles, 150, 90); return u ? html`<img src=${u} alt="" loading="lazy" />` : html`<span class="small muted">${r.formula}</span>`; };
  return html`<li class=${cls('card rxn-row', selected && 'selected')} title=${`${a.name} → ${b.name}`}
    onClick=${() => { select({ edges: [e.id] }); findInGraph(e.id, true); }}>
    <div class="rxn-pics">${pic(a)}<span class="rxn-arrow">→</span>${pic(b)}</div>
    <div class="rxn-meta small">${fresh && html`<span class="new-dot" title="New: a calculation added it since you last looked"></span>`}
      <span class="rxn-names">${a.name} → ${b.name}</span><span class="rxn-tag">${tag}</span></div>
  </li>`;
}

// A reaction with no arrow in Explore (barrierless, or its ends changed when
// relaxed): it is still a reaction, so it is listed; its card is in Analyze.
const NO_ARROW = { barrierless: 'no barrier', recombines: 'no barrier', reverts: 'unstable product',
  falls_apart: 'falls apart', rearranges: 'rearranges', failed: 'relaxing failed' };
function ArrowlessRow({ r }) {
  const structures = useStore((s) => s.workspace.structures);
  const side = (ids) => ids.map((id) => structures[id]).filter(Boolean);
  const a = side(r.reactants || []), b = side(r.products || []);
  if (!a.length || !b.length) return null;
  const pic = (recs) => { const u = depictUrl(recs.map((x) => x.smiles).join('.'), 150, 90); return u ? html`<img src=${u} alt="" loading="lazy" />` : html`<span class="small muted">${recs.map((x) => x.formula).join(' + ')}</span>`; };
  const names = (recs) => Object.entries(recs.reduce((m, x) => ({ ...m, [x.name]: (m[x.name] || 0) + 1 }), {}))
    .map(([n, k]) => (k > 1 ? `${k} ${n}` : n)).join(' + ');
  return html`<li class="card rxn-row" title=${`${names(a)} → ${names(b)} · open it in Analyze`}
    onClick=${() => set({ analyze: { ...(state.analyze || {}), compose: false, view: 'reactions', reaction: r.id }, view: { tab: 'analyze', jobId: null } })}>
    <div class="rxn-pics">${pic(a)}<span class="rxn-arrow">→</span>${pic(b)}</div>
    <div class="rxn-meta small"><span class="rxn-names">${names(a)} → ${names(b)}</span>
      <span class="rxn-tag muted">${NO_ARROW[r.complex_reason] || 'no TS ends'}</span></div>
  </li>`;
}

export function Library() {
  const structures = useStore((s) => s.workspace.structures);
  const selected = useStore((s) => s.selection.structures);
  const [q, setQ] = useState('');
  const edges = useStore((s) => s.workspace.edges);
  const selEdges = useStore((s) => s.selection.edges);
  useStore((s) => s.seenTick);
  const [what, setWhat] = useState('structures');   // or 'reactions'
  const [onlyNew, setOnlyNew] = useState(false);
  const [drag, setDrag] = useState(false);
  // A reaction's hidden subsystem ends (role "complex") are not library entries.
  const list = Object.values(structures)
    .filter((r) => !isComplex(r))
    .filter((r) => !q || `${r.name} ${r.smiles} ${r.formula}`.toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => b.created - a.created);
  const rxns = Object.values(edges || {}).filter((e) => structures[e.source] && structures[e.target])
    .filter((e) => !q || `${structures[e.source].name} ${structures[e.target].name}`.toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => (b.created || 0) - (a.created || 0));
  const reactions = useStore((s) => s.workspace.reactions) || {};
  const arrowless = Object.values(reactions).filter((r) => !r.edge || !edges?.[r.edge])
    .filter((r) => !q || [...(r.reactants || []), ...(r.products || [])].some((id) => (structures[id]?.name || '').toLowerCase().includes(q.toLowerCase())));
  const nRxns = rxns.length + arrowless.length;   // the same count as Analyze and the reactor
  const newS = list.filter(isNew).length, newE = rxns.filter(isNew).length;
  const nNew = what === 'reactions' ? newE : newS;
  const showNew = onlyNew && nNew > 0;

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
  };
  const empty = !Object.values(structures).some((r) => !isComplex(r));
  // Nodes that are the same molecule (graphs from before conformers were
  // merged into one node): offer to fold them together.
  const seen = {};
  let dupes = 0;
  for (const r of Object.values(structures)) {
    if (!isSpecies(r) || !r.smiles || r.reacted) continue;
    const k = `${r.smiles}|${r.charge}|${r.multiplicity}|${r.qmmm || ''}`;   // a QM/MM system is its own world
    if (seen[k]) dupes += 1; else seen[k] = true;
  }
  return html`
    <aside class=${cls('library', drag && 'drag')}
      onDragOver=${(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave=${(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDrag(false); }}
      onDrop=${onDrop}>
      <div class="pane-head">
        ${nRxns || what === 'reactions'
    ? html`<div class="lib-switch" role="tablist">
            <button role="tab" class=${cls(what === 'structures' && 'active')} onClick=${() => setWhat('structures')}>
              Structures <span class="count">${Object.values(structures).filter((r) => !isComplex(r)).length}</span>${newS > 0 && html`<span class="new-dot"></span>`}</button>
            <button role="tab" class=${cls(what === 'reactions' && 'active')} onClick=${() => setWhat('reactions')}>
              Reactions <span class="count">${nRxns}</span>${newE > 0 && html`<span class="new-dot"></span>`}</button>
          </div>`
    : html`<h2>Structures${!empty && html` <span class="count">${Object.values(structures).filter((r) => !isComplex(r)).length}</span>`}</h2>`}
        ${nNew > 0 && html`<div class="new-bar small">
          <button class=${cls('new-filter', showNew && 'active')} onClick=${() => setOnlyNew(!showNew)}
            title="Show only what calculations added since you last looked"><span class="new-dot"></span>${nNew} new</button>
          <button class="btn-link small" onClick=${() => { markAllSeen(); setOnlyNew(false); }} title="Clear the new marks">Mark seen</button>
        </div>`}
      </div>
      <${LevelBar} status />
      ${dupes > 0 && html`<p class="level-note small">${dupes} structure${dupes > 1 ? 's are' : ' is'} the same molecule as another node.
        <button class="btn-link small" onClick=${() => attempt(() => api.post('/api/structures/merge-duplicates'), 'Merged into one node per molecule')}>Merge into conformers</button></p>`}
      ${(what === 'reactions' ? Object.keys(edges || {}).length : Object.values(structures).filter((r) => !isComplex(r)).length) > 12 && html`
        <input class="search" type="search" placeholder="Filter" title="By name, SMILES or formula" value=${q} onInput=${(e) => setQ(e.target.value)} />`}
      ${what === 'reactions'
        ? html`<ul class="cards rxn-list">
            ${(showNew ? rxns.filter(isNew) : rxns).map((e) => html`<${ReactionRow} key=${e.id} e=${e} selected=${selEdges.includes(e.id)} fresh=${isNew(e)} />`)}
            ${!showNew && arrowless.map((r) => html`<${ArrowlessRow} key=${r.id} r=${r} />`)}
          </ul>`
        : empty
        ? html`<p class="empty-hint small muted">Molecules you <a href="#" onClick=${(e) => { e.preventDefault(); openTab('design'); }}>build or open in Design</a> show up here. You can also drop .xyz files here.</p>`
        : html`<ul class="cards">
            ${(showNew ? list.filter(isNew) : list).map((r) => html`<${Card} key=${r.id} rec=${r} fresh=${isNew(r)} selected=${selected.includes(r.id)}
              order=${selected.length > 1 ? selected.indexOf(r.id) + 1 : 0} />`)}
          </ul>`}
      <div class="drop-overlay">Drop .xyz / .smi files to add them</div>
    </aside>`;
}
