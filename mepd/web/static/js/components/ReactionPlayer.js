// A reaction, opened: its path played as a movie, with the 3D view, the
// energy profile and the bonds that change all on one scrubber (linked
// representations, after PhET and ChemTube3D). The path is the lowest
// barrier result of the reaction's own calculation.
import { html, useEffect, useMemo, useState } from '../lib.js';
import { api } from '../api.js';
import { openJob, select, set, state, useStore } from '../store.js';
import { cls, edgeStatus, fmtKcal } from '../util.js';
import { Viewer3D } from './Viewer3D.js';
import { EnergyPlot } from './EnergyPlot.js';

const COV = { H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, P: 1.07, S: 1.05, Cl: 1.02, Br: 1.2, I: 1.39, B: 0.84, Si: 1.11, Li: 1.28, Na: 1.66, K: 2.03, Mg: 1.41 };
const cov = (e) => COV[e] ?? 1.2;

function parseXyz(text) {
  const lines = text.trim().split('\n');
  const n = +lines[0];
  return lines.slice(2, 2 + n).map((l) => { const [e, x, y, z] = l.trim().split(/\s+/); return { e, p: [+x, +y, +z] }; });
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// Bonds that exist at one end of the path and not the other: forming
// (absent at the start, present at the end) and breaking (the reverse).
export function bondChanges(frames) {
  if (!frames || frames.length < 2) return [];
  const first = parseXyz(frames[0]), last = parseXyz(frames[frames.length - 1]);
  const out = [];
  for (let i = 0; i < first.length; i += 1) {
    for (let j = i + 1; j < first.length; j += 1) {
      const s = cov(first[i].e) + cov(first[j].e);
      const d0 = dist(first[i].p, first[j].p), d1 = dist(last[i].p, last[j].p);
      const bonded0 = d0 < 1.2 * s, bonded1 = d1 < 1.2 * s;
      if (bonded0 !== bonded1) out.push({ i, j, label: `${first[i].e}${i + 1}–${first[j].e}${j + 1}`, kind: bonded1 ? 'form' : 'break' });
    }
  }
  return out.slice(0, 6);
}

// The best path a calculation found: the lowest barrier among entries with frames.
function bestEntry(result) {
  const all = (result?.groups || []).flatMap((g) => g.entries).filter((e) => e.frames?.length > 1);
  const hasE = (e) => e.frames.some((f) => f.energy_kcal != null);
  const rank = (e) => [hasE(e) ? 0 : 1, e.barrier_kcal ?? Infinity];
  return [...all].sort((a, b) => { const x = rank(a), y = rank(b); return x[0] - y[0] || x[1] - y[1]; })[0] || null;
}

// `stepEdge`: play the very step a network expansion put on that edge (its own
// IRC, from `start`, energies from that node's energy), not the run's lowest path.
export function ReactionPlayer({ jobId, compact = false, stepEdge = null, start = null }) {
  const job = useStore((s) => s.jobs[jobId]);
  // The orbitals of each frame, from this calculation's own run (Viewer3D's Orbitals bar).
  const chem = job ? { charge: job.charge, multiplicity: job.multiplicity, job: job.id } : null;
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    let alive = true;
    setResult(null); setError(null);
    const url = stepEdge ? `/api/edges/${encodeURIComponent(stepEdge)}/step-path${start ? `?start=${encodeURIComponent(start)}` : ''}`
      : `/api/jobs/${jobId}/result`;
    api.get(url).then((r) => { if (alive) setResult(r); }).catch((e) => alive && setError(e.message));
    return () => { alive = false; };
  }, [jobId, stepEdge, start]);
  const entry = useMemo(() => (stepEdge ? (result?.frames?.length > 1 ? { frames: result.frames.map((f, k) => ({ ...f, path_length: k })), ts_index: result.ts_index } : null)
    : bestEntry(result)), [result, stepEdge]);
  const frames = useMemo(() => entry?.frames.map((f) => f.xyz), [entry]);
  const xs = useMemo(() => entry?.frames.map((f) => f.path_length), [entry]);
  const ys = useMemo(() => entry?.frames.map((f) => f.energy_kcal) ?? [], [entry]);
  const changes = useMemo(() => bondChanges(frames), [frames]);
  const coords = useMemo(() => frames?.map((x) => parseXyz(x).map((a) => a.p)), [frames]);
  useEffect(() => { if (entry) { setFrame(entry.ts_index ?? 0); setPlaying(false); } }, [entry]);
  useEffect(() => {
    if (!playing || !entry) return undefined;
    const id = setInterval(() => setFrame((f) => {
      if (f + 1 >= entry.frames.length) { setPlaying(false); return f; }
      return f + 1;
    }), 70);
    return () => clearInterval(id);
  }, [playing, entry]);
  // Space plays, arrows step (when nothing else has the keyboard).
  useEffect(() => {
    if (!entry) return undefined;
    const onKey = (e) => {
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || state.modal) return;
      if (e.key === ' ') { e.preventDefault(); setPlaying((p) => { if (!p && frame >= entry.frames.length - 1) setFrame(0); return !p; }); }
      if (e.key === 'ArrowRight') { e.preventDefault(); setPlaying(false); setFrame((f) => Math.min(entry.frames.length - 1, f + 1)); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); setPlaying(false); setFrame((f) => Math.max(0, f - 1)); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [entry, frame]);

  if (error) return stepEdge ? html`<p class="muted">${error}</p>` : html`<p class="error-box">${error}</p>`;
  if (!result) return html`<p class="muted small player-loading">Loading the path…</p>`;
  if (!entry) return html`<p class="muted">This calculation kept no path to play.</p>`;
  const n = entry.frames.length;
  const ts = entry.ts_index;
  const e = ys[frame];
  const where = frame === 0 ? 'Start' : frame === n - 1 ? 'End' : frame === ts ? 'TS ‡' : '';
  const go = (i) => { setPlaying(false); setFrame(i); };
  const toggle = () => { if (!playing && frame >= n - 1) setFrame(0); setPlaying(!playing); };
  return html`<div class=${cls('reaction-player', compact && 'compact')}>
    <div class="rp-stage">
      <${Viewer3D} frames=${frames} frame=${frame} height=${compact ? 220 : 340} labelBase=${1} orbitals=${chem} />
      <div class="rp-where">
        ${where && html`<span class=${cls('rp-tag', frame === ts && 'ts')}>${where}</span>`}
        ${e != null && html`<span class="rp-e">${e > 0 ? '+' : ''}${e.toFixed(1)} kcal/mol</span>`}
      </div>
      <div class="rp-key"><span><i class="rp-form"></i>forming</span><span><i class="rp-break"></i>breaking</span></div>
    </div>
    ${changes.length > 0 && html`<div class="rp-bonds">
      ${changes.map((c) => {
        const d = coords ? dist(coords[frame][c.i], coords[frame][c.j]) : null;
        return html`<span class=${`rp-bond ${c.kind}`} title=${c.kind === 'form' ? 'Forms along the path' : 'Breaks along the path'}>
          ${c.label} <b>${d != null ? d.toFixed(2) : '–'} Å</b> ${c.kind === 'form' ? '↓' : '↑'}</span>`;
      })}
    </div>`}
    <${EnergyPlot} xs=${xs} ys=${ys} current=${frame} tsIndex=${ts} onPick=${go} height=${compact ? 110 : 150} />
    <div class="rp-controls">
      <button class="btn primary rp-play" onClick=${toggle}>${playing ? 'Pause' : 'Play'}</button>
      <button class="btn small" onClick=${() => go(0)}>Start</button>
      ${ts != null && html`<button class="btn small" onClick=${() => go(ts)}>TS ‡</button>`}
      <button class="btn small" onClick=${() => go(n - 1)}>End</button>
      <input type="range" min="0" max=${n - 1} value=${frame} onInput=${(ev) => go(+ev.target.value)} aria-label="Position along the path" />
    </div>
  </div>`;
}

// The reaction page (#/reaction/<edge>): what it is, its barriers, its movie.
export function ReactionFocus({ edgeId }) {
  const ws = useStore((s) => s.workspace);
  const jobs = useStore((s) => s.jobs);
  const ctx = useStore((s) => (s.reactionContext?.edgeId === edgeId ? s.reactionContext : null));
  const edge = ws.edges?.[edgeId];
  useEffect(() => { if (edge && !state.selection.edges.includes(edgeId)) select({ structures: [], edges: [edgeId] }); }, [edgeId, !!edge]);
  if (!edge) return html`<div class="focus-missing"><p>This reaction is not in this workspace.</p>
    <button class="btn" onClick=${() => set({ view: { tab: 'graph', jobId: null } })}>Back to Explore</button></div>`;
  const name = (id) => ws.structures?.[id]?.name || id;
  const st = edgeStatus(edge, jobs);
  // Read the way it was opened from (the Landscape's route), else source -> target.
  const flip = ctx?.start && ctx.start === edge.target;
  const [from, to] = flip ? [edge.target, edge.source] : [edge.source, edge.target];
  const b1 = st.barrierFwd ?? st.barrier, b2 = st.barrierRev;
  const fwd = flip ? b2 : b1, rev = flip ? b1 : b2;
  // An expansion's step: play that very step (its own IRC), not its run's lowest path.
  const o = edge.origin || {};
  const ownStep = o.label && !o.proposed && jobs[o.job]?.op === 'graph-enumeration';
  const band = (v) => `band-${v < 10 ? 'low' : v < 20 ? 'mid' : 'high'}`;
  return html`<div class="focus reaction-focus">
    <nav class="crumbs"><a href="#/explore" onClick=${(ev) => { ev.preventDefault(); set({ view: { tab: 'graph', jobId: null } }); }}>Explore</a>
      <span>›</span><b>${name(from)} → ${name(to)}</b></nav>
    <div class="focus-chips">
      ${ctx?.barrier != null && html`<span class="chip-k" title=${`What the Landscape drew: the kinetics' barrier (${ctx.basis}${ctx.T ? `, ${ctx.T} K` : ''}), ${ctx.separated ? 'from the molecules apart' : 'from the lowest structure found'}`}>
        <b class=${band(ctx.barrier)}>‡ ${fmtKcal(ctx.barrier)}</b> in the kinetics</span>`}
      ${fwd != null && html`<span class="chip-k" title="This step's barrier on its path (electronic), from the lowest start found"><b class=${band(fwd)}>ΔE‡ ${fmtKcal(fwd)}</b> →</span>`}
      ${rev != null && html`<span class="chip-k" title="Reverse barrier: the same TS from the other side"><b class=${band(rev)}>${fmtKcal(rev)}</b> ←</span>`}
      ${o.barrier_verified === false && html`<span class="chip-k muted" title=${o.headline || ''}>IRC not checked yet</span>`}
      ${o.ts_kind === 'path_peak' && html`<span class="chip-k muted" title=${o.headline || ''}>upper bound: no TS found</span>`}
      ${st.barrier == null && st.barrierUnverified != null && html`<span class="chip-k muted">no TS</span>`}
      ${st.barrierJob && html`<a class="chip-k link" href=${`#/calc/${st.barrierJob}`} onClick=${(ev) => { ev.preventDefault(); openJob(st.barrierJob); }}>Calculation ›</a>`}
    </div>
    ${o.rejected_step ? html`<p class="small muted">${o.headline}</p>` : ''}
    ${ownStep ? html`<${ReactionPlayer} jobId=${o.job} stepEdge=${edgeId} start=${from} key=${`${edgeId}:${from}`} />`
      : st.barrierJob
      ? html`<${ReactionPlayer} jobId=${st.barrierJob} />`
      : html`<div class="focus-empty"><p>No path computed for this reaction yet.</p><p class="small muted">Pick Transition state in the panel on the right.</p></div>`}
  </div>`;
}
