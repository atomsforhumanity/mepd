// The inspector's Method tab: exactly how a number was computed, for whoever
// wants to check it (after Figma's Dev Mode). Level of theory, what the
// calculation does and how, the settings that differ from the defaults, the
// command that reproduces it, and where to read more. Facts come from the job
// record (its params, level, command) and the operation's schema and
// explain.json entry; nothing here is computed anew.
import { html, useState } from '../lib.js';
import { openJob, set, state, useStore } from '../store.js';
import { copy, edgeStatus, fmtDuration, isTs, steeringOf } from '../util.js';
import { useExplain } from './Why.js';

const fmtVal = (v) => (v === true ? 'on' : v === false ? 'off' : v == null || v === '' ? '—' : Array.isArray(v) ? v.join(', ') || '—' : String(v));
const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

// The job behind what is selected: an edge's barrier, a structure's origin.
export function methodJobFor(sel, ws = state.workspace, jobs = state.jobs) {
  if (sel.edges.length === 1 && !sel.structures.length) {
    const e = ws.edges?.[sel.edges[0]];
    if (!e) return null;
    const st = edgeStatus(e, jobs);
    if (st.barrierJob) return st.barrierJob;
    if (e.origin?.kind === 'job') return e.origin.job;
    return null;
  }
  if (sel.structures.length === 1 && !sel.edges.length) {
    const r = ws.structures?.[sel.structures[0]];
    if (!r) return null;
    const opt = Object.values(jobs).filter((j) => j.op === 'optimize' && j.status === 'done' && j.targets?.structures?.includes(r.id))
      .sort((a, b) => (b.finished || 0) - (a.finished || 0))[0];
    return opt?.id || (r.origin?.kind === 'job' ? r.origin.job : null);
  }
  return null;
}

export function MethodPane({ jobId, rec = null }) {
  const job = useStore((s) => (jobId ? s.jobs[jobId] : null));
  const ops = useStore((s) => s.operations);
  const wsLevel = useStore((s) => s.levels[s.levelProfile ?? '']);
  const explain = useExplain();
  const [all, setAll] = useState(false);
  if (!job) {
    return html`<div class="method-pane">
      <div class="m-row"><span class="m-k">Level of theory</span><span class="m-v">${rec?.level?.label || wsLevel?.label || '—'}${rec && !rec.level ? html` <span class="muted">(${isTs(rec) ? 'TS as found' : 'not optimized yet'})</span>` : ''}</span></div>
      <p class="small muted">${rec?.origin?.kind === 'smiles' ? html`From SMILES <code>${rec.origin.input}</code>, embedded with RDKit.` : rec ? 'Geometry as given.' : 'Nothing computed for this selection yet.'}</p>
    </div>`;
  }
  const op = ops.find((o) => o.key === job.op);
  const props = op?.schema?.properties || {};
  const params = job.params || {};
  const rows = Object.entries(params).filter(([k]) => props[k]).map(([k, v]) => ({
    k, label: props[k].title || k.replace(/_/g, ' '), v, d: props[k].default, changed: !same(v, props[k].default),
  }));
  const changed = rows.filter((r) => r.changed);
  const ex = explain?.[`op.${job.op}`];
  const took = job.finished && job.started ? fmtDuration(job.finished - job.started) : null;
  return html`<div class="method-pane">
    <div class="m-head">
      <b>${op?.title || job.op}</b>
      <a href=${`#/calc/${job.id}`} onClick=${(e) => { e.preventDefault(); openJob(job.id); }}>Calculation ›</a>
    </div>
    <div class="m-chips">
      <span class="chip-k" title="Level of theory this ran at">${job.level?.label || '—'}</span>
      <span class="chip-k" title="Compute profile">profile ${job.profile || job.level?.profile || 'default'}</span>
      ${steeringOf(job) && html`<span class="chip-k" title="The level whose single points steer the expansion's kinetics">steering: ${steeringOf(job)}</span>`}
      ${took && html`<span class="chip-k" title="Wall time">${took}</span>`}
    </div>
    ${ex?.does && html`<p class="m-does">${ex.does.split(/(?<=\.)\s/)[0]}</p>`}
    ${Array.isArray(ex?.how) && ex.how.length > 0 && html`<details class="m-fold"><summary>Pipeline · ${ex.how.length} steps</summary>
      <ol class="m-steps">${ex.how.map((h) => html`<li>${h}</li>`)}</ol></details>`}
    <h4 class="section-title">Settings vs defaults</h4>
    ${changed.length
      ? html`<dl class="m-kv">${changed.map((r) => html`<dt>${r.label}</dt><dd><b>${fmtVal(r.v)}</b> <span class="muted">default ${fmtVal(r.d)}</span></dd>`)}</dl>`
      : html`<p class="small muted">All defaults.</p>`}
    ${rows.length > changed.length && html`<button class="btn-link small" onClick=${() => setAll(!all)}>${all ? 'Hide' : 'All'} ${rows.length} settings</button>
      ${all && html`<dl class="m-kv all">${rows.map((r) => html`<dt>${r.label}</dt><dd>${fmtVal(r.v)}</dd>`)}</dl>`}`}
    ${Array.isArray(ex?.caveats) && ex.caveats.length > 0 && html`<details class="m-fold"><summary>Keep in mind</summary><ul>${ex.caveats.map((c) => html`<li>${c}</li>`)}</ul></details>`}
    ${job.command && html`<h4 class="section-title">Reproduce</h4>
      <div class="m-cmd"><code>${job.command}</code><button class="btn-icon" title="Copy" onClick=${() => copy(job.command)}>⧉</button></div>`}
    <p class="small m-more">
      <a href="#" onClick=${(e) => { e.preventDefault(); set({ view: { ...state.view, drawer: 'learn' } }); }}>How it works ›</a>
      ${ex?.refs && html` · <a href="#" onClick=${(e) => { e.preventDefault(); set({ view: { ...state.view, drawer: 'learn' }, refsFocus: ex.refs }); }}>Cite ›</a>`}
      · <a href=${`#/calc/${job.id}/files`} onClick=${(e) => { e.preventDefault(); openJob(job.id, 'files'); }}>Files ›</a>
    </p>
  </div>`;
}
