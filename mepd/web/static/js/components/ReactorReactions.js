// Reactions › Nanoreactor runs (#/reactions/nanoreactor): every Nanoreactor run of the session and the
// reactions they found as one table (GET /api/nanoreactor/reactions, mepd/web/reactor_list.py). Each reaction
// is one row however many runs found it: how often, under which conditions, how far it got (seen in the MD,
// refined, a TS), sortable by barrier, ΔE or how often it was seen. A row's barrier and card are its
// representative's (the session reaction with TS endpoints from the run that saw it most), so finding TSs
// from here starts one search per reaction, not one per run.
import { html, useEffect, useState } from '../lib.js';
import { api } from '../api.js';
import { openJob, openReactorSetup, set, state, useStore } from '../store.js';
import { fmtKcal } from '../util.js';
import { ReactionCard, TsCell, findTs, tsState } from './Reactions.js';

const STAGES = [['all', 'All'], ['seen', 'Seen in the MD'], ['refined', 'Refined, no TS yet'], ['ts', 'With a TS']];
const SPIN_WORD = { 2: 'doublet', 3: 'triplet', 4: 'quartet', 5: 'quintet' };
const RUNNING = new Set(['running', 'queued']);

// The list, read again whenever a run or the session's reactions change, and every 8 s while a run goes on.
function useReactorList() {
  const runsKey = useStore((s) => Object.values(s.jobs).filter((j) => j.op === 'nanoreactor' || j.op === 'nanoreactor-more' || j.op === 'ts')
    .map((j) => `${j.id}:${j.status}:${j.rev ?? ''}`).join(','));
  const rxKey = useStore((s) => Object.keys(s.workspace.reactions || {}).length);
  const busy = useStore((s) => Object.values(s.jobs).some((j) => j.op.startsWith('nanoreactor') && RUNNING.has(j.status)));
  const [data, setData] = useState(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    api.get('/api/nanoreactor/reactions').then((d) => { if (alive) setData(d); }).catch(() => {});
    return () => { alive = false; };
  }, [runsKey, rxKey, tick]);
  useEffect(() => {
    if (!busy) return undefined;
    const t = setInterval(() => setTick((k) => k + 1), 8000);
    return () => clearInterval(t);
  }, [busy]);
  return data;
}

// 'CC=O', 'O', 'O' -> 'acetaldehyde + 2 water' (the session's names for its molecules).
function side(list, names) {
  const n = {};
  for (const s of list) n[s] = (n[s] || 0) + 1;
  return Object.entries(n).map(([s, k]) => `${k > 1 ? `${k} ` : ''}${names[s] || s}`).join(' + ');
}

// The box of the newest run, as the setup page's list ('2 CC=O, 4 O, O=O triplet'), for "More runs".
function boxOf(job, structures) {
  const ids = job?.targets?.structures || [];
  const counts = String(job?.params?.copies ?? '').split(',').map((x) => parseInt(x, 10));
  return ids.map((id, i) => {
    const s = structures[id];
    if (!s?.smiles) return null;
    const n = counts.length === ids.length && counts[i] > 0 ? counts[i] : counts.length === 1 && counts[0] > 0 ? counts[0] : 1;
    return `${n} ${s.smiles}${SPIN_WORD[s.multiplicity] ? ` ${SPIN_WORD[s.multiplicity]}` : ''}`;
  }).filter(Boolean).join(', ');
}

export function ReactorReactions() {
  const data = useReactorList();
  const ws = useStore((s) => s.workspace);
  const jobs = useStore((s) => s.jobs);
  const [stage, setStage] = useState('all');
  const [q, setQ] = useState('');
  const [sort, setSort] = useState({ by: 'barrier', dir: 1 });
  const [picked, setPicked] = useState(null);
  if (!data) return html`<p class="boot">Reading the runs…</p>`;

  const names = {};
  for (const s of Object.values(ws.structures)) if (s.smiles && !(s.smiles in names)) names[s.smiles] = s.name;
  const runs = data.runs;
  const runOf = Object.fromEntries(runs.map((r) => [r.run, r]));
  const rows = data.reactions.map((row) => {
    const r = row.rep ? ws.reactions?.[row.rep] : null;
    const t = r ? tsState(r, ws.edges, jobs) : null;
    const kind = t?.kind === 'ts' ? 'ts' : row.stage;
    const byCond = {};
    for (const m of row.members) {
      const label = runOf[m.run]?.label || '?';
      byCond[label] = (byCond[label] || 0) + m.n + m.n_rev;
    }
    return { ...row, r, t, kind, seen: row.n + row.n_rev, label: `${side(row.smiles[0], names)} → ${side(row.smiles[1], names)}`,
      smilesLabel: `${row.smiles[0].join(' + ')} → ${row.smiles[1].join(' + ')}`, byCond };
  });
  const shown = rows.filter((x) => {
    if (stage === 'seen' && x.stage !== 'seen') return false;
    if (stage === 'refined' && !(x.stage === 'refined' && x.kind !== 'ts')) return false;
    if (stage === 'ts' && x.kind !== 'ts') return false;
    return !q || `${x.label} ${x.smilesLabel}`.toLowerCase().includes(q.toLowerCase());
  });
  const val = {
    barrier: (x) => x.t?.v ?? null,
    de: (x) => x.delta_e ?? null,
    seen: (x) => -x.seen,
    runs: (x) => -x.runs,
    first: (x) => x.first_fs ?? null,
  };
  const cmp = (a, b) => {
    const va = val[sort.by](a), vb = val[sort.by](b);
    if (va == null && vb == null) return b.seen - a.seen;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (va - vb) * sort.dir || b.seen - a.seen;
  };
  shown.sort(cmp);
  const todo = shown.filter((x) => x.r && x.t?.kind === 'todo');
  const findAll = async () => {
    if (!window.confirm(`Queue ${todo.length} TS search${todo.length > 1 ? 'es' : ''}, one per reaction? They run a few at a time.`)) return;
    for (const x of todo) await findTs(x.r);
  };
  const head = (by, label, title) => html`<th title=${title}><button class=${`nrl-sort ${sort.by === by ? 'on' : ''}`}
    onClick=${() => setSort(sort.by === by ? { by, dir: -sort.dir } : { by, dir: 1 })}>${label}${sort.by === by ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`;
  const going = runs.filter((r) => RUNNING.has(r.status)).length;
  const newest = Object.values(jobs).filter((j) => j.op === 'nanoreactor' && !j.params?.trajectory).sort((a, b) => b.created - a.created)[0];
  const sel = picked && rows.find((x) => x.key === picked);

  return html`<div class="analyze nrl">
    <div class="an-list nrl-list">
      <div class="nrl-runs">
        <span class="small muted">${runs.length} run${runs.length === 1 ? '' : 's'}${going ? ` · ${going} going` : ''}</span>
        ${runs.map((r) => html`<button class=${`nrl-run ${RUNNING.has(r.status) ? 'busy' : r.status === 'failed' ? 'failed' : ''}`}
          title=${`${r.title}: ${r.status}${r.final ? '' : ' (so far)'}. Open the run.`} onClick=${() => openJob(r.run)}>${r.name || r.label}</button>`)}
        <button class="btn small" title="The setup page with the newest run's box" onClick=${() => openReactorSetup(boxOf(newest, ws.structures))}>More runs ›</button>
      </div>
      <div class="an-bar">
        <input type="search" placeholder="Filter by species" value=${q} onInput=${(e) => setQ(e.target.value)} />
        <div class="segmented">${STAGES.map(([k, l]) => html`<button class=${stage === k ? 'on' : ''} onClick=${() => setStage(k)}>${l}</button>`)}</div>
      </div>
      <div class="small muted an-count">${shown.length} reaction${shown.length === 1 ? '' : 's'}
        ${todo.length > 0 && html` · <button class="btn-link small" onClick=${findAll}>find the TS of the ${todo.length} refined without one</button>`}</div>
      <div class="nrl-scroll">
        <table class="rx-table nrl-table">
          <thead><tr>
            <th>Reaction</th>
            ${head('barrier', 'Barrier', 'ΔE‡ of its TS search, kcal/mol')}
            ${head('de', 'ΔE', 'Products minus reactants, each optimized alone, kcal/mol')}
            ${head('seen', 'Seen', 'Times the MD saw it, all runs (this way / back)')}
            ${head('runs', 'Runs', 'How many runs found it')}
          </tr></thead>
          <tbody>${shown.map((x) => html`<tr class=${`clickable ${picked === x.key ? 'on' : ''}`} onClick=${() => setPicked(picked === x.key ? null : x.key)}>
            <td class="nrl-eq"><span title=${x.smilesLabel}>${x.label}</span>
              <span class="nrl-conds">${Object.entries(x.byCond).map(([c, k]) => html`<span class="nrl-cond">${c} ×${k}</span>`)}
                <span class=${`nrl-stage ${x.kind}`}>${x.kind === 'ts' ? 'TS' : x.stage === 'refined' ? 'refined' : 'seen in the MD'}</span></span></td>
            <td onClick=${(e) => e.stopPropagation()}>${x.r ? html`<${TsCell} r=${x.r} />` : html`<span class="small muted" title="Not refined yet: no TS search to run">—</span>`}</td>
            <td class="mono">${x.delta_e != null ? `${x.delta_e >= 0 ? '+' : ''}${fmtKcal(x.delta_e)}` : '—'}</td>
            <td class="mono">${x.n}${x.n_rev ? html`<span class="muted">/${x.n_rev}</span>` : ''}</td>
            <td class="mono">${x.runs}</td>
          </tr>`)}</tbody>
        </table>
        ${!shown.length && html`<p class="small muted nrl-empty">${rows.length ? 'None of this kind.' : going ? 'Nothing seen yet: the list fills in as the runs go (every simulated ps).' : 'These runs found no reactions.'}</p>`}
      </div>
    </div>
    <div class="an-detail">
      ${sel ? html`<div class="nrl-found">
          <h3 class="section-title">Found in</h3>
          <ul>${sel.members.map((m) => html`<li><a href="#" onClick=${(e) => { e.preventDefault(); openJob(m.run); }}>${runOf[m.run]?.name || m.run}</a>
            <span class="mono small"> ${m.n}${m.n_rev ? `/${m.n_rev}` : ''}×</span>
            <span class="small muted"> ${m.refined ? 'refined' : 'seen in the MD'}${m.reaction === sel.rep ? ' · shown below' : ''}</span></li>`)}</ul>
        </div>
        ${sel.r ? html`<${ReactionCard} key=${sel.r.id} r=${sel.r} />`
          : html`<p class="small muted">Seen in the MD, not refined yet. Each run refines its reactions while the MD goes on and at its end; the reaction's card and its TS search show here then.</p>`}`
        : html`<div class="empty-hint"><p>Every reaction the runs found, once each, lowest barrier first. Pick one for its card and its TS search; the conditions under each say which runs saw it.</p></div>`}
    </div>
  </div>`;
}
