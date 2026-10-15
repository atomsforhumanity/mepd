// "What happened": the first tab of an interactive reactor's analysis. It
// carries on the getting-started steps (Welcome.js) in plain words: one card
// per reaction the reactor made, what its barrier search is doing, and once a
// barrier is known, a small energy diagram with the forward and the reverse
// barrier as arrows of their own. The full results stay on the other tabs.
import { html, useEffect, useState } from '../lib.js';
import { openJob, select, set, toast, useStore } from '../store.js';
import { edgeStatus, fmtKcal } from '../util.js';
import { Equation, NO_TS_SHORT, ReactionCard, TsCell, findTs, noTsWhy } from './Reactions.js';
import { StepBar, markStep, reactorFor } from './Welcome.js';

// Reactant, TS and product levels; an arrow up to the TS from each side.
function EnergyArrows({ fwd, rev }) {
  const W = 280, H = 120, pad = 16;
  const prod = fwd - rev;
  const lo = Math.min(0, prod), hi = fwd;
  const y = (v) => pad + ((hi - v) / ((hi - lo) || 1)) * (H - 2 * pad);
  const bars = [[14, 74, 0, 'start'], [110, 170, fwd, 'TS'], [206, 266, prod, 'end']];
  return html`<svg class="energy-arrows" viewBox=${`0 0 ${W} ${H + 14}`} role="img"
    aria-label=${`Barrier ${fmtKcal(fwd)} kcal/mol forward, ${fmtKcal(rev)} back`}>
    <defs>${['fwd', 'rev'].map((k) => html`<marker id=${`ea-${k}`} viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" class=${`ea-head ${k}`} /></marker>`)}</defs>
    <line class="ea-link" x1="74" y1=${y(0)} x2="110" y2=${y(fwd)} />
    <line class="ea-link" x1="170" y1=${y(fwd)} x2="206" y2=${y(prod)} />
    ${bars.map(([x1, x2, v, l]) => html`<line class="ea-level" x1=${x1} x2=${x2} y1=${y(v)} y2=${y(v)} />
      <text class="ea-cap" x=${(x1 + x2) / 2} y=${H + 10} text-anchor="middle">${l}</text>`)}
    <line class="ea-arrow fwd" x1="92" y1=${y(0)} x2="92" y2=${y(fwd) + 3} marker-end="url(#ea-fwd)" />
    <text class="ea-num fwd" x="88" y=${(y(0) + y(fwd)) / 2 + 4} text-anchor="end">${fmtKcal(fwd)}</text>
    <line class="ea-arrow rev" x1="188" y1=${y(prod)} x2="188" y2=${y(fwd) + 3} marker-end="url(#ea-rev)" />
    <text class="ea-num rev" x="192" y=${(y(prod) + y(fwd)) / 2 + 4}>${fmtKcal(rev)}</text>
  </svg>`;
}

function StoryCard({ r }) {
  const edges = useStore((s) => s.workspace.edges);
  const jobs = useStore((s) => s.jobs);
  const structures = useStore((s) => s.workspace.structures);
  const [more, setMore] = useState(false);
  const edge = r.edge && edges[r.edge];
  const st = edge ? edgeStatus(edge, jobs) : null;
  const searching = st && ['running', 'queued'].includes(st.status);
  const searchJob = searching && Object.values(jobs).find((j) => j.targets.edges.includes(r.edge) && ['running', 'queued'].includes(j.status));
  let status;
  if (r.live) status = html`<span class="muted">Checking the molecules…</span>`;
  else if (!edge) status = html`<span class="muted" title=${noTsWhy(r)}>No barrier: ${NO_TS_SHORT[r.complex_reason] || 'no stable ends'}</span>`;
  else if (searching) {
    status = html`<span class="story-busy">Finding the barrier…</span>${searchJob && html` <a href="#" class="small"
      onClick=${(e) => { e.preventDefault(); openJob(searchJob.id, 'live'); }}>watch</a>`}`;
  } else if (st.barrierFwd != null && st.barrierRev != null) status = html`<b>Barrier ${fmtKcal(st.barrierFwd)} kcal/mol</b>
    <span class="muted">· ${fmtKcal(st.barrierRev)} back</span>`;
  else if (st.barrier != null) status = html`<b>Barrier ${fmtKcal(st.barrier)} kcal/mol</b>`;
  else if (st.barrierUnverified != null) {
    status = html`<span class="muted" title=${`The highest point on the path (≈${fmtKcal(st.barrierUnverified)} kcal/mol) did not lead to a transition state joining the two ends`}>No transition state found.</span>
      ${st.barrierJob && html` <a href="#" class="small" onClick=${(e) => { e.preventDefault(); openJob(st.barrierJob); }}>see why</a>`}`;
  } else if (st.status === 'failed') status = html`<span class="muted">No barrier found.</span> <${TsCell} r=${r} />`;
  else status = html`<button class="btn small primary" onClick=${() => findTs(r)}>Find its barrier</button>`;
  return html`<li class="story-card">
    <${Equation} r=${r} structures=${structures} />
    <div class="story-status">${status}</div>
    ${st?.barrierFwd != null && st?.barrierRev != null && html`<${EnergyArrows} fwd=${st.barrierFwd} rev=${st.barrierRev} />`}
    <button class="btn-link small story-more" onClick=${() => setMore(!more)}>${more ? 'Less' : 'Details'}</button>
    ${more && html`<${ReactionCard} r=${r} compact=${true} />`}
  </li>`;
}

export function ReactorStory({ job }) {
  const reactionsById = useStore((s) => s.workspace.reactions || {});
  const edges = useStore((s) => s.workspace.edges);
  const jobs = useStore((s) => s.jobs);
  const mine = Object.values(reactionsById).filter((r) => r.origin?.job === job.id)
    .sort((a, b) => (a.origin.index ?? 0) - (b.origin.index ?? 0));
  const found = mine.filter((r) => r.edge && edges[r.edge] && edgeStatus(edges[r.edge], jobs).barrier != null);
  const busy = job.status === 'running' || job.status === 'queued';
  useEffect(() => { if (found.length) markStep('barrier'); }, [found.length]);
  const toExplore = () => {
    markStep('network');
    window.dispatchEvent(new CustomEvent('mepd:graph', { detail: { cmd: 'fit-to', ids: found.map((r) => r.edge) } }));
    set({ view: { tab: 'graph', jobId: null } });
    select({ edges: found.map((r) => r.edge) });
    const tap = window.matchMedia?.('(pointer: coarse)').matches ? 'Tap' : 'Click';
    toast(`Your reactions: each arrow carries its barrier. ${tap} one to explore it.`, 'info', 7000);
  };
  const n = mine.length;
  // The same molecule into a new reactor: the run's first species that is not water.
  const structures = useStore((s) => s.workspace.structures);
  const again = (job.targets?.structures || []).map((id) => structures[id]).find((r) => r && r.smiles !== 'O') || null;
  const headline = !n ? (busy ? 'Looking at what your reactor made…' : 'Your reactor made no lasting reactions')
    : `Your reactor made ${n} reaction${n > 1 ? 's' : ''}`;
  return html`<div class="story">
    <${StepBar} at=${found.length ? 'network' : n ? 'barrier' : 'reaction'} />
    <h3 class="story-head">${headline}</h3>
    ${n > 0 && html`<p class="small muted">${busy ? 'Each one is checked, then its barrier is searched.'
      : `${found.length} of ${n} with a barrier.`}</p>`}
    ${found.length > 0 && html`<button class="rx-bulb story-bulb" onClick=${toExplore}>
      <span class="bulb-dot"></span><span><b>See ${found.length > 1 ? 'them' : 'it'} in Explore</b>
      <span class="small muted"> · with your other molecules</span></span></button>`}
    <ul class="story-list">${mine.map((r) => html`<${StoryCard} key=${r.id} r=${r} />`)}</ul>
    ${!busy && !n && html`<div class="story-empty">
      <p class="small muted">Left alone, molecules rarely react within a few picoseconds. Next time:</p>
      <ul><li>drag an atom of one molecule into another and hold it there a moment;</li>
        <li>shrink the Wall to squeeze them together;</li><li>turn the Temperature up.</li></ul>
      <div class="row">${again && html`<button class="btn primary" onClick=${() => reactorFor(again)}>Run it again</button>`}</div>
    </div>`}
    ${!busy && html`<p class="small muted story-again"><a href="#" onClick=${(e) => { e.preventDefault(); set({ view: { tab: 'welcome', jobId: null } }); }}>Try another molecule</a></p>`}
  </div>`;
}
