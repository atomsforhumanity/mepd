// The live network (#/network/<job> or #/network/example/<slug>): an explorer
// run as it grows, corrects itself and moves material (mepd/web/live_network.py).
// One layout for the whole run, so species appear in place; per frame:
// - a node's area grows with the most material it ever held (capped both ways);
// - an edge is an open step, coloured by its barrier the way material goes,
//   as wide as its net flux, with dots flowing along it;
// - closed, replaced and dropped steps stay as faint dashes;
// - corrections since the last frame flash on what they moved.
// Frames whose energies wait for their DFT single points ("steering pending")
// are marked, and never taken as the answer.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { openJob, set, useStore } from '../store.js';

const W = 900, H = 600, CX = W / 2, CY = H / 2;
const band = (v) => (v == null ? 'var(--edge-idle)' : v < 10 ? 'var(--chart-3)' : v < 20 ? 'var(--chart-4)' : 'var(--chart-5)');

// A deterministic spring layout of every species the run ever showed, the seed in the middle.
function layoutOf(tl) {
  const ids = [], seen = new Set(), links = new Map();
  for (const f of tl.frames) {
    for (const s of f.species) if (!seen.has(s[0])) { seen.add(s[0]); ids.push(s[0]); }
    for (const st of f.steps) links.set(`${st[1]}|${st[2]}`, [st[1], st[2]]);
    for (const st of f.gone || []) links.set(`${st[1]}|${st[2]}`, [st[1], st[2]]);
  }
  const seed = tl.frames[0]?.species[0]?.[0] ?? ids[0];
  const n = ids.length;
  const idx = Object.fromEntries(ids.map((id, i) => [id, i]));
  const pos = ids.map((id, i) => {
    const a = (i / Math.max(1, n)) * Math.PI * 2 * 2.39996;   // golden-angle spiral start
    const r = id === seed ? 0 : 60 + 220 * Math.sqrt(i / Math.max(1, n));
    return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
  });
  const E = [...links.values()].filter(([a, b]) => a in idx && b in idx).map(([a, b]) => [idx[a], idx[b]]);
  const k = Math.sqrt((W * H) / Math.max(1, n)) * 0.55;
  for (let it = 0; it < 260; it += 1) {
    const t = 18 * (1 - it / 260) + 0.5;
    const d = pos.map(() => [0, 0]);
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
        const dd = Math.max(1, Math.hypot(dx, dy)), f = (k * k) / dd;
        d[i][0] += (dx / dd) * f; d[i][1] += (dy / dd) * f; d[j][0] -= (dx / dd) * f; d[j][1] -= (dy / dd) * f;
      }
    }
    for (const [a, b] of E) {
      const dx = pos[a][0] - pos[b][0], dy = pos[a][1] - pos[b][1];
      const dd = Math.max(1, Math.hypot(dx, dy)), f = (dd * dd) / k;
      d[a][0] -= (dx / dd) * f; d[a][1] -= (dy / dd) * f; d[b][0] += (dx / dd) * f; d[b][1] += (dy / dd) * f;
    }
    for (let i = 0; i < n; i += 1) {
      if (ids[i] === seed) { pos[i] = [CX, CY]; continue; }
      const dl = Math.max(1, Math.hypot(d[i][0], d[i][1]));
      pos[i][0] = Math.min(W - 40, Math.max(40, pos[i][0] + (d[i][0] / dl) * Math.min(dl, t)));
      pos[i][1] = Math.min(H - 40, Math.max(40, pos[i][1] + (d[i][1] / dl) * Math.min(dl, t)));
    }
  }
  return { at: Object.fromEntries(ids.map((id, i) => [id, pos[i]])), seed };
}

const short = (smi) => (smi && smi.length > 22 ? `${smi.slice(0, 20)}…` : smi || '?');
const fmtMin = (t) => `${(t / 60).toFixed(1)} min`;
const sign = (x) => (x > 0 ? `+${x.toFixed(1)}` : x.toFixed(1));

function eventText(ev, smiles) {
  const [, kind, ids, d] = ev;
  if (kind === 'barrier_moved') {
    const delta = d.to != null && d.from != null ? d.to - d.from : null;
    const why = /gxtb/.test(d.why || '') ? 'steering pending' : d.src === 'dft' ? 'DFT back' : d.src === 'borrowed' ? 'borrowed' : '';
    return [delta != null && delta < 0 ? 'good' : 'warn', `Barrier ${delta != null ? sign(delta) : ''}${d.to != null ? ` → ${d.to.toFixed(1)}` : ''}${why ? ` · ${why}` : ''}`];
  }
  if (kind === 'species_rereferenced') return ['good', `Lower conformer${smiles[ids[0]] ? `: ${short(smiles[ids[0]])}` : ''}${d.to != null && d.from != null ? ` ${sign(d.to - d.from)}` : ''}`];
  if (kind === 'step_replaced') return ['info', `Step replaced by ${d.by?.length || 1} better`];
  if (kind === 'step_closed') return ['muted', 'Step closed: out of reach'];
  if (kind === 'step_reopened') return ['info', 'Step reopened'];
  return ['muted', kind];
}

export function LiveNetworkView({ jobId = null, example = null }) {
  const [tl, setTl] = useState(null);
  const [error, setError] = useState(null);
  const [i, setI] = useState(null);           // frame shown (null: follow the latest / the final)
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const running = useStore((s) => (jobId ? ['running', 'queued'].includes(s.jobs[jobId]?.status) : false));
  const url = example ? `/api/examples/${encodeURIComponent(example)}/timeline` : `/api/jobs/${jobId}/network-timeline`;
  useEffect(() => {
    let alive = true;
    const load = () => api.get(url).then((d) => { if (alive) { setTl(d); setError(null); } }).catch((e) => alive && setError(e.message));
    load();
    const t = running ? setInterval(load, 5000) : null;
    return () => { alive = false; if (t) clearInterval(t); };
  }, [url, running]);
  const lay = useMemo(() => (tl?.frames?.length ? layoutOf(tl) : null), [tl]);
  const n = tl?.frames?.length || 0;
  const cur = i == null ? (running ? n - 1 : tl?.final ?? n - 1) : Math.min(i, n - 1);
  useEffect(() => {
    if (!playing || !n) return undefined;
    const id = setInterval(() => setI((k) => {
      const nk = (k == null ? 0 : k) + 1;
      if (nk >= n) { setPlaying(false); return n - 1; }
      return nk;
    }), 450 / speed);
    return () => clearInterval(id);
  }, [playing, n, speed]);
  const flowKey = useRef(0);

  if (error) return html`<div class="ln-empty"><p>${error}</p>${jobId && html`<button class="btn" onClick=${() => openJob(jobId)}>Back to the calculation</button>`}</div>`;
  if (!tl || !lay) return html`<p class="boot">Loading the network…</p>`;
  const f = tl.frames[cur];
  const prev = cur > 0 ? tl.frames[cur - 1] : null;
  const prevIds = new Set(prev ? prev.species.map((s) => s[0]) : []);
  const tPrev = prev ? prev.t : -1;
  const recent = tl.events.filter((e) => e[0] > tPrev - 1e-6 && e[0] <= f.t + 1e-6);
  const moved = new Map(), reref = new Map();
  for (const e of recent) {
    if (e[1] === 'barrier_moved' && e[3].to != null && e[3].from != null) for (const id of e[2]) moved.set(id, e[3].to - e[3].from);
    if (e[1] === 'species_rereferenced' && e[3].to != null && e[3].from != null) for (const id of e[2]) reref.set(id, e[3].to - e[3].from);
  }
  // Sizes compare what forms: the starting mixture (always 100 %) is a fixed mark.
  const peakMax = Math.max(1e-30, ...f.species.filter((s) => s[0] !== lay.seed).map((s) => s[2] || 0));
  const radius = (peak, id) => (id === lay.seed ? 16 : Math.min(30, Math.max(5, 30 * Math.sqrt((peak || 0) / peakMax))));
  const fluxMax = Math.max(1e-30, ...f.steps.map((s) => Math.abs(s[7] || 0)));
  const at = lay.at;
  const edges = f.steps.map((s) => {
    const [id, a, b, dGf, dGr, src, irc, flux] = s;
    const fwd = (flux || 0) >= 0;
    const [p, q] = fwd ? [at[a], at[b]] : [at[b], at[a]];
    const barrier = fwd ? dGf : dGr;
    const w = Math.min(9, Math.max(1, 9 * Math.sqrt(Math.abs(flux || 0) / fluxMax)));
    return { id, p, q, barrier, w, src, irc, flux, moved: moved.get(id) };
  }).filter((e) => e.p && e.q);
  const gone = (f.gone || []).map(([id, a, b, state]) => ({ id, p: at[a], q: at[b], state })).filter((e) => e.p && e.q);
  const nodes = f.species.map(([id, G, peak, fin, pop, dft]) => ({ id, G, peak, fin, pop, dft, at: at[id], r: radius(peak, id),
    fresh: prev && !prevIds.has(id), dE: reref.get(id) })).filter((x) => x.at);
  // Labels: a benchmark's products by name (endo, exo), the start, and whatever else holds real material.
  const productOf = {};
  for (const [name, ids] of Object.entries(tl.selectivity?.species || {})) for (const id of ids) productOf[id] = name;
  const labelOf = (id) => (id === lay.seed ? 'start' : productOf[id] || short(tl.smiles[id]));
  const labelled = new Set(nodes.filter((x) => x.id === lay.seed || productOf[x.id] || (x.peak || 0) >= 0.05 * peakMax).map((x) => x.id));

  // Selectivity: against experiment when the run is a benchmark, else the products' shares.
  const sel = tl.selectivity;
  const point = sel ? [...sel.series].reverse().find((r) => r[0] <= f.t + 1e-6) : null;
  const ref = sel?.reference;
  const share = (r) => (r == null ? null : r / (1 + r));
  const pct = point ? share(point[1]) : null;
  const refLo = ref?.range ? share(ref.range[0]) : ref?.value ? share(ref.value) : null;
  const refHi = ref?.range ? share(ref.range[1]) : refLo;
  const names = sel?.products?.length === 2 ? sel.products : null;
  const tops = !sel ? nodes.filter((x) => x.id !== lay.seed).sort((a, b) => (b.fin || 0) - (a.fin || 0)).slice(0, 3) : [];
  const finTotal = tops.reduce((t, x) => t + Math.max(0, x.fin || 0), 0);
  // The corrections so far, newest first (moves under 0.05 kcal/mol left out).
  const sizable = (e) => !(['barrier_moved', 'species_rereferenced'].includes(e[1]) && e[3].to != null && e[3].from != null && Math.abs(e[3].to - e[3].from) < 0.05);
  const feed = tl.events.filter((e) => e[0] <= f.t + 1e-6 && sizable(e)).slice(-8).reverse();
  flowKey.current += 0;
  return html`<div class="live-network">
    <div class="ln-main">
      <div class="ln-head">
        <h2>${example ? 'Live network · example' : 'Live network'}</h2>
        ${!f.steered && html`<span class="ln-pending" title="These energies wait for their DFT single points (the seed just moved to a lower conformer): not a real swing">steering pending</span>`}
        <span class="spacer"></span>
        <span class="small muted">${f.basis}</span>
        ${jobId && html`<a href=${`#/calc/${jobId}`} onClick=${(e) => { e.preventDefault(); openJob(jobId); }}>Calculation ›</a>`}
      </div>
      <div class=${`ln-stage ${f.steered ? '' : 'unsteered'}`}>
        <svg viewBox=${`0 0 ${W} ${H}`} width="100%" role="img" aria-label="The network: species as circles sized by the material they held, steps as lines coloured by barrier and as wide as their flux">
          ${gone.map((e) => html`<line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} class=${`ln-gone ${e.state}`}><title>${e.id}: ${e.state}</title></line>`)}
          ${edges.map((e) => html`<g class="ln-edge">
            <line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} style=${`stroke:${band(e.barrier)};stroke-width:${e.w}`} class=${`ln-line ${e.src === 'g' ? 'gxtb' : ''}`} />
            ${Math.abs(e.flux || 0) > 1e-3 * fluxMax && html`<line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} class="ln-flow" style=${`stroke:${band(e.barrier)};stroke-width:${Math.max(2, e.w * 0.8)}`} />`}
            ${e.moved != null && Math.abs(e.moved) >= 0.1 && html`<text x=${(e.p[0] + e.q[0]) / 2} y=${(e.p[1] + e.q[1]) / 2 - 6} class=${`ln-delta ${e.moved < 0 ? 'down' : 'up'}`}>${sign(e.moved)}</text>`}
            <title>${e.id}: barrier ${e.barrier != null ? e.barrier.toFixed(1) : '?'} kcal/mol${e.src === 'd' ? ' (DFT)' : e.src === 'b' ? ' (borrowed)' : ' (g-xTB)'}${e.irc === 1 ? ' · IRC ✓' : ''} · net flux ${(e.flux ?? 0).toPrecision(2)}</title>
          </g>`)}
          ${nodes.map((x) => html`<g class="ln-node">
            ${x.fresh && html`<circle cx=${x.at[0]} cy=${x.at[1]} r=${x.r + 9} class="ln-fresh" />`}
            <circle cx=${x.at[0]} cy=${x.at[1]} r=${x.r} class=${`ln-dot ${x.id === lay.seed ? 'seed' : x.pop ? 'pop' : ''} ${x.dft ? '' : 'gxtb'}`} />
            ${labelled.has(x.id) && html`<text x=${x.at[0]} y=${x.at[1] + x.r + 14} class=${`ln-label ${productOf[x.id] ? 'product' : ''}`}>${labelOf(x.id)}</text>`}
            ${x.dE != null && Math.abs(x.dE) >= 0.1 && html`<text x=${x.at[0] + x.r + 4} y=${x.at[1] - x.r} class="ln-delta down">${sign(x.dE)}</text>`}
            <title>${tl.smiles[x.id]} · ${x.G != null ? `${x.G.toFixed(1)} kcal/mol` : ''} · peak ${(x.peak ?? 0).toPrecision(2)} · final ${(x.fin ?? 0).toPrecision(2)}${x.dft ? ' · DFT' : ' · g-xTB'}</title>
          </g>`)}
        </svg>
        <div class="ln-key small">
          <span class="lg-scale"><i class="lg-band low"></i><i class="lg-band mid"></i><i class="lg-band high"></i>${'barrier <10 · 10–20 · >20'}</span>
          <span>size = material it held</span><span>width = flux</span><span>◌ energy still g-xTB</span><span class="ln-k-gone">┄ closed or replaced</span>
        </div>
      </div>
      <div class="ln-controls">
        <button class="btn primary" onClick=${() => { if (!playing && cur >= n - 1) setI(0); setPlaying(!playing); }}>${playing ? 'Pause' : 'Replay'}</button>
        <input type="range" min="0" max=${n - 1} value=${cur} onInput=${(e) => { setPlaying(false); setI(+e.target.value); }} aria-label="Exploration time" />
        <span class="mono small">${fmtMin(f.t)} / ${fmtMin(tl.frames[n - 1].t)}</span>
        <div class="segmented">${[1, 4].map((s) => html`<button class=${speed === s ? 'on' : ''} onClick=${() => setSpeed(s)}>${s}×</button>`)}</div>
        ${cur !== tl.final && !running && html`<button class="btn small" onClick=${() => { setPlaying(false); setI(tl.final); }} title="The end state: the last frame with every energy steered">End</button>`}
      </div>
    </div>
    <aside class="ln-side">
      ${sel ? html`<section>
        <span class="section-title">${names ? `${names[0]} : ${names[1]}` : 'Selectivity'}</span>
        <b class=${`ln-ratio ${f.steered ? '' : 'pending'}`}>${pct != null ? `${Math.round(pct * 100)} : ${100 - Math.round(pct * 100)}` : '— : —'}</b>
        <div class="ln-gauge">
          <span class="ln-gauge-fill" style=${`width:${pct != null ? pct * 100 : 0}%`}></span>
          ${refLo != null && html`<span class="ln-gauge-ref" style=${`left:${refLo * 100}%;width:${Math.max(0.4, (refHi - refLo) * 100)}%`} title=${ref?.citation_text || ref?.citation || 'experiment'}></span>`}
        </div>
        ${refLo != null && html`<span class="small muted">▮ measured ${Math.round(share(ref.value) * 100)} : ${100 - Math.round(share(ref.value) * 100)}${point?.[2] ? ' · settled' : ''}</span>`}
      </section>` : html`<section>
        <span class="section-title">What forms</span>
        ${tops.map((x) => html`<div class="ln-prod"><span class="mono small">${short(tl.smiles[x.id])}</span><b>${finTotal > 0 ? `${Math.round(100 * Math.max(0, x.fin) / finTotal)}%` : '—'}</b></div>`)}
      </section>`}
      <div class="big-counts ln-counts">
        <span><b>${f.counts.species}</b>found</span>
        <span><b>${f.steps.length}</b>carry</span>
        <span><b>${(f.counts.closed || 0) + (f.counts.replaced || 0)}</b>closed</span>
      </div>
      <section>
        <span class="section-title">Corrections</span>
        <ul class="ln-feed">${feed.map((e) => { const [cls, txt] = eventText(e, tl.smiles); return html`<li class=${cls}><span class="mono">${fmtMin(e[0])}</span> ${txt}</li>`; })}</ul>
      </section>
    </aside>
  </div>`;
}

export function openLiveNetwork({ jobId = null, example = null }) {
  set({ view: { tab: 'network', jobId, example } });
}
