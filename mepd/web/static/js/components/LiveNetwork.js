// The live network (#/network/<job> or #/network/example/<slug>): an explorer
// run as it grows, corrects itself and moves material (mepd/web/live_network.py).
// One layout for the whole run, so species appear in place; per frame:
// - a species is its 2D structure, on a card whose area grows with the most
//   material it ever held (capped both ways);
// - an edge is an open step, coloured by its barrier the way material goes,
//   as wide as its net flux, with dots flowing along it;
// - closed, replaced and dropped steps stay as faint dashes;
// - corrections since the last frame flash on what they moved.
// Frames whose energies wait for their DFT single points ("steering pending")
// are marked, and never taken as the answer.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { openJob, set, useStore } from '../store.js';
import { depictUrl } from '../util.js';

const W = 1100, H = 720, CX = W / 2, CY = H / 2, M = 64;
const BOX_MIN = 46, BOX_MAX = 112, BOX_SEED = 88;   // card widths (px); height is 3/4 of it
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
  const deg = ids.map(() => 0);
  for (const [a, b] of E) { deg[a] += 1; deg[b] += 1; }
  // The largest card each species ever gets (as the frames size them), so cards don't land on each other.
  const big = ids.map(() => BOX_MIN);
  for (const f of tl.frames) {
    const pm = Math.max(1e-30, ...f.species.filter((x) => x[0] !== seed).map((x) => x[2] || 0));
    for (const x of f.species) {
      const j = idx[x[0]];
      big[j] = Math.max(big[j], x[0] === seed ? BOX_SEED : Math.min(BOX_MAX, BOX_MAX * Math.sqrt((x[2] || 0) / pm)));
    }
  }
  const k = Math.sqrt((W * H) / Math.max(1, n)) * 0.62;
  for (let it = 0; it < 260; it += 1) {
    const t = 18 * (1 - it / 260) + 0.5;
    const d = pos.map(() => [0, 0]);
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
        const dd = Math.max(1, Math.hypot(dx, dy));
        const clear = (big[i] + big[j]) / 2 + 14;   // card half-widths plus a gap
        // Repulsion reaches only nearby species (else a crowd pushes loners to the walls); overlap pushes hard.
        const f = (dd < 3 * k ? (k * k) / dd : 0) + (dd < clear ? (clear - dd) * 8 : 0);
        if (!f) continue;
        d[i][0] += (dx / dd) * f; d[i][1] += (dy / dd) * f; d[j][0] -= (dx / dd) * f; d[j][1] -= (dy / dd) * f;
      }
    }
    for (const [a, b] of E) {
      const dx = pos[a][0] - pos[b][0], dy = pos[a][1] - pos[b][1];
      const dd = Math.max(1, Math.hypot(dx, dy)), f = (dd * dd) / k;
      d[a][0] -= (dx / dd) * f; d[a][1] -= (dy / dd) * f; d[b][0] += (dx / dd) * f; d[b][1] += (dy / dd) * f;
    }
    for (let i = 0; i < n; i += 1) {
      // A gentle pull to the middle, so species with no step among the shown ones stay near, not in the corners.
      const g = deg[i] ? 0.09 : 0.4;
      d[i][0] += (CX - pos[i][0]) * g; d[i][1] += (CY - pos[i][1]) * g * 1.5;
      if (ids[i] === seed) { pos[i] = [CX, CY]; continue; }
      const dl = Math.max(1, Math.hypot(d[i][0], d[i][1]));
      pos[i][0] = Math.min(W - M, Math.max(M, pos[i][0] + (d[i][0] / dl) * Math.min(dl, t)));
      pos[i][1] = Math.min(H - M, Math.max(M, pos[i][1] + (d[i][1] / dl) * Math.min(dl, t)));
    }
  }
  return { at: Object.fromEntries(ids.map((id, i) => [id, pos[i]])), seed };
}

const short = (smi) => (smi && smi.length > 22 ? `${smi.slice(0, 20)}…` : smi || '?');
const fmtMin = (t) => `${(t / 60).toFixed(1)} min`;
const sign = (x) => (x > 0 ? `+${x.toFixed(1)}` : x.toFixed(1));

function eventText(ev, smiles, names = {}) {
  const [, kind, ids, d] = ev;
  if (kind === 'barrier_moved') {
    const delta = d.to != null && d.from != null ? d.to - d.from : null;
    const why = /gxtb/.test(d.why || '') ? 'steering pending' : d.src === 'dft' ? 'DFT back' : d.src === 'borrowed' ? 'borrowed' : '';
    return [delta != null && delta < 0 ? 'good' : 'warn', `Barrier ${delta != null ? sign(delta) : ''}${d.to != null ? ` → ${d.to.toFixed(1)}` : ''}${why ? ` · ${why}` : ''}`];
  }
  if (kind === 'species_rereferenced') return ['good', `Lower conformer${names[ids[0]] || smiles[ids[0]] ? `: ${names[ids[0]] || short(smiles[ids[0]])}` : ''}${d.to != null && d.from != null ? ` ${sign(d.to - d.from)}` : ''}`];
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
  const [hover, setHover] = useState(null);   // species id under the pointer: drawn large on top
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
  // A named product (e.g. both mirror twins of endo) is one card, at its first species' place.
  const productOf = {};
  for (const [name, ids] of Object.entries(tl.selectivity?.species || {})) for (const id of ids) productOf[id] = name;
  const keyOf = (id) => (productOf[id] ? `p:${productOf[id]}` : id);
  const merged = new Map();
  for (const row of f.species) {
    const m = merged.get(keyOf(row[0]));
    if (!m) merged.set(keyOf(row[0]), [...row]);
    else { m[2] = (m[2] || 0) + (row[2] || 0); m[3] = (m[3] || 0) + (row[3] || 0); m[4] = m[4] || row[4]; }
  }
  const nodeOf = {};
  for (const row of f.species) nodeOf[row[0]] = merged.get(keyOf(row[0]))[0];
  // Sizes compare what forms: the starting mixture (always 100 %) is a fixed mark.
  const peakMax = Math.max(1e-30, ...[...merged.values()].filter((s) => s[0] !== lay.seed).map((s) => s[2] || 0));
  const box = (peak, id) => (id === lay.seed ? BOX_SEED : Math.min(BOX_MAX, Math.max(BOX_MIN, BOX_MAX * Math.sqrt((peak || 0) / peakMax))));
  const at = lay.at;
  const nodes = [...merged.values()].map(([id, G, peak, fin, pop, dft]) => {
    const w = box(peak, id);
    return { id, G, peak, fin, pop, dft, at: at[id], w, h: w * 0.75, fresh: prev && !prevIds.has(id), dE: reref.get(id) };
  // A species the explorer could not identify shows only once it holds material.
  }).filter((x) => x.at && (tl.smiles[x.id] || x.id === lay.seed || (x.peak || 0) >= 0.01 * peakMax))
    // Small cards first, so the ones that matter sit on top; the start last.
    .sort((a, b) => (a.id === lay.seed) - (b.id === lay.seed) || a.w - b.w);
  const shown = new Set(nodes.map((x) => x.id));
  const ends = (a, b) => {
    const [x, y] = [nodeOf[a], nodeOf[b]];
    return x && y && x !== y && shown.has(x) && shown.has(y) ? [at[x], at[y]] : null;
  };
  const fluxMax = Math.max(1e-30, ...f.steps.map((s) => Math.abs(s[7] || 0)));
  const noFlux = !!f.legacy;   // runs before per-step flux was recorded: barriers only
  const edges = f.steps.map((s) => {
    const [id, a, b, dGf, dGr, src, irc, flux] = s;
    const fwd = (flux || 0) >= 0;
    const pq = ends(fwd ? a : b, fwd ? b : a);
    if (!pq) return null;
    const barrier = fwd ? dGf : dGr;
    const w = noFlux ? 2 : Math.min(9, Math.max(1, 9 * Math.sqrt(Math.abs(flux || 0) / fluxMax)));
    return { id, p: pq[0], q: pq[1], barrier, w, src, irc, flux, moved: moved.get(id) };
  }).filter(Boolean);
  const gone = (f.gone || []).map(([id, a, b, state]) => { const pq = ends(a, b); return pq && { id, p: pq[0], q: pq[1], state }; }).filter(Boolean);
  // Labels: a benchmark's products by name (endo, exo), the start, and whatever else holds real material.
  const names = tl.names || {};
  const labelOf = (id) => (id === lay.seed ? names[id] || 'start' : productOf[id] || names[id]);
  // The structure says what it is; a name only where one exists and the species matters.
  const labelled = new Set(nodes.filter((x) => x.id === lay.seed || productOf[x.id] || (names[x.id] && (x.peak || 0) >= 0.05 * peakMax)).map((x) => x.id));

  // Selectivity: against experiment when the run is a benchmark, else the products' shares.
  const sel = tl.selectivity;
  const point = sel ? [...sel.series].reverse().find((r) => r[0] <= f.t + 1e-6) : null;
  const ref = sel?.reference;
  const share = (r) => (r == null ? null : r / (1 + r));
  const pct = point ? share(point[1]) : null;
  const refLo = ref?.range ? share(ref.range[0]) : ref?.value ? share(ref.value) : null;
  const refHi = ref?.range ? share(ref.range[1]) : refLo;
  const pair = sel?.products?.length === 2 ? sel.products : null;
  // What forms: every run has this; shares of what the start turned into.
  const formed = nodes.filter((x) => x.id !== lay.seed && (x.fin || 0) > 0);
  const finTotal = formed.reduce((t, x) => t + Math.max(0, x.fin || 0), 0);
  // Species that read the same (one product's mirror twins, or the benchmark's named product) count as one.
  const groups = new Map();
  for (const x of formed) {
    const k = productOf[x.id] || tl.smiles[x.id] || x.id;
    const g = groups.get(k) || { ...x, fin: 0 };
    g.fin += Math.max(0, x.fin || 0);
    groups.set(k, g);
  }
  const tops = [...groups.values()].sort((a, b) => b.fin - a.fin).slice(0, 4).filter((x) => finTotal > 0 && x.fin / finTotal >= 0.005);
  const seedNode = nodes.find((x) => x.id === lay.seed);
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
        <svg viewBox=${`0 0 ${W} ${H}`} width="100%" role="img" aria-label="The network: species as their structures, on cards sized by the material they held; steps as lines coloured by barrier and as wide as their flux">
          ${gone.map((e) => html`<line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} class=${`ln-gone ${e.state}`}><title>${e.id}: ${e.state}</title></line>`)}
          ${edges.map((e) => html`<g class="ln-edge">
            <line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} style=${`stroke:${band(e.barrier)};stroke-width:${e.w}`} class=${`ln-line ${e.src === 'g' ? 'gxtb' : ''}`} />
            ${!noFlux && Math.abs(e.flux || 0) > 1e-3 * fluxMax && html`<line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} class="ln-flow" style=${`stroke:${band(e.barrier)};stroke-width:${Math.max(2, e.w * 0.8)}`} />`}
            ${e.moved != null && Math.abs(e.moved) >= 0.1 && html`<text x=${(e.p[0] + e.q[0]) / 2} y=${(e.p[1] + e.q[1]) / 2 - 6} class=${`ln-delta ${e.moved < 0 ? 'down' : 'up'}`}>${sign(e.moved)}</text>`}
            <title>${e.id}: barrier ${e.barrier != null ? e.barrier.toFixed(1) : '?'} kcal/mol${e.src === 'd' ? ' (DFT)' : e.src === 'b' ? ' (borrowed)' : ' (g-xTB)'}${e.irc === 1 ? ' · IRC ✓' : ''} · net flux ${(e.flux ?? 0).toPrecision(2)}</title>
          </g>`)}
          ${nodes.map((x) => html`<g class="ln-node" key=${x.id} transform=${`translate(${x.at[0]},${x.at[1]})`}
              onMouseEnter=${() => setHover(x.id)} onMouseLeave=${() => setHover((h) => (h === x.id ? null : h))}>
            ${x.fresh && html`<rect x=${-x.w / 2 - 6} y=${-x.h / 2 - 6} width=${x.w + 12} height=${x.h + 12} rx="6" class="ln-fresh" />`}
            <rect x=${-x.w / 2} y=${-x.h / 2} width=${x.w} height=${x.h} rx="4" class=${`ln-card ${x.id === lay.seed ? 'seed' : x.pop ? 'pop' : ''} ${x.dft ? '' : 'gxtb'}`} />
            ${tl.smiles[x.id] ? html`<image href=${depictUrl(tl.smiles[x.id], 160, 120)} x=${-x.w / 2 + 2} y=${-x.h / 2 + 2} width=${x.w - 4} height=${x.h - 4} preserveAspectRatio="xMidYMid meet" />`
              : html`<text x="0" y="5" class="ln-unknown">?</text>`}
            ${x.dE != null && Math.abs(x.dE) >= 0.1 && html`<text x=${x.w / 2 + 4} y=${-x.h / 2} class="ln-delta down">${sign(x.dE)}</text>`}
            <title>${tl.smiles[x.id] || 'structure not perceived'} · ${x.G != null ? `${x.G.toFixed(1)} kcal/mol` : ''} · peak ${(x.peak ?? 0).toPrecision(2)} · final ${(x.fin ?? 0).toPrecision(2)}${x.dft ? ' · DFT' : ' · g-xTB'}</title>
          </g>`)}
          ${(() => {
            const x = hover && nodes.find((y) => y.id === hover);
            if (!x) return null;
            const w = Math.max(x.w, 180), h = w * 0.75 + 34;
            const cx = Math.min(W - w / 2 - 4, Math.max(w / 2 + 4, x.at[0])), cy = Math.min(H - h / 2 - 4, Math.max(h / 2 + 4, x.at[1]));
            const nm = labelOf(x.id) || (x.id === lay.seed ? 'start' : '');
            return html`<g class="ln-hover" transform=${`translate(${cx},${cy})`} pointer-events="none">
              <rect x=${-w / 2} y=${-h / 2} width=${w} height=${h} rx="4" class="ln-card pop" />
              ${tl.smiles[x.id] ? html`<image href=${depictUrl(tl.smiles[x.id], 160, 120)} x=${-w / 2 + 4} y=${-h / 2 + 4} width=${w - 8} height=${w * 0.75 - 8} preserveAspectRatio="xMidYMid meet" />` : html`<text x="0" y="0" class="ln-unknown">?</text>`}
              <text x="0" y=${h / 2 - 12} class="ln-hover-text">${nm ? `${nm} · ` : ''}${x.G != null ? `${x.G.toFixed(1)} kcal/mol` : ''}${x.id !== lay.seed ? ` · peak ${(x.peak || 0) < 0.005 ? '<1' : Math.round(100 * Math.min(1, x.peak))}%` : ''}</text>
            </g>`;
          })()}
          ${nodes.filter((x) => labelled.has(x.id) && labelOf(x.id) && x.id !== hover).map((x) => html`<text x=${x.at[0]} y=${x.at[1] + x.h / 2 + 15} class=${`ln-label ${productOf[x.id] ? 'product' : ''}`}>${labelOf(x.id)}</text>`)}
        </svg>
        <div class="ln-key small">
          <span class="lg-scale"><i class="lg-band low"></i><i class="lg-band mid"></i><i class="lg-band high"></i>${'barrier <10 · 10–20 · >20'}</span>
          <span>card size = material it held</span>${noFlux ? html`<span title="This run is from an explorer version that did not record each step's flux">no flux recorded</span>` : html`<span>width = flux</span>`}<span>┆ energy still g-xTB</span><span class="ln-k-gone">┄ closed or replaced</span>
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
      <section>
        <span class="section-title">What forms</span>
        ${tops.length ? tops.map((x) => html`<div class="ln-prod" title=${tl.smiles[x.id]}>
          ${tl.smiles[x.id] ? html`<img class="ln-prod-mol" src=${depictUrl(tl.smiles[x.id], 160, 120)} alt=${tl.smiles[x.id]} />` : html`<span class="ln-prod-mol">?</span>`}
          <span class="ln-prod-name">${productOf[x.id] || names[x.id] || ''}</span>
          <b>${Math.round(100 * Math.max(0, x.fin) / finTotal)}%</b>
        </div>`) : html`<span class="small muted">Nothing yet</span>`}
        ${seedNode && html`<span class="small muted">start left: ${Math.round(100 * Math.min(1, Math.max(0, seedNode.fin || 0)))}%</span>`}
      </section>
      ${sel && html`<section>
        <span class="section-title">${pair ? `${pair[0]} : ${pair[1]}` : 'Selectivity'}</span>
        <b class=${`ln-ratio ${f.steered ? '' : 'pending'}`}>${pct != null ? `${Math.round(pct * 100)} : ${100 - Math.round(pct * 100)}` : '— : —'}</b>
        <div class="ln-gauge">
          <span class="ln-gauge-fill" style=${`width:${pct != null ? pct * 100 : 0}%`}></span>
          ${refLo != null && html`<span class="ln-gauge-ref" style=${`left:${refLo * 100}%;width:${Math.max(0.4, (refHi - refLo) * 100)}%`} title=${ref?.citation_text || ref?.citation || 'experiment'}></span>`}
        </div>
        ${point?.[2] && html`<span class="small muted" title="The run's own answer has stopped moving (judged from its own recent values, not from any experiment)">answer stable</span>`}
        ${refLo != null && html`<span class="small muted" title=${`Held-out validation: the run never sees this value. ${ref?.citation_text || ref?.citation || ''}`}>▮ measured ${Math.round(share(ref.value) * 100)} : ${100 - Math.round(share(ref.value) * 100)} · validation only</span>`}
      </section>`}
      <div class="big-counts ln-counts">
        <span><b>${f.counts.species}</b>found</span>
        <span><b>${f.steps.length}</b>carry</span>
        <span><b>${(f.counts.closed || 0) + (f.counts.replaced || 0)}</b>closed</span>
      </div>
      ${tl.events.length > 0 && html`<section>
        <span class="section-title">Corrections</span>
        <ul class="ln-feed">${feed.map((e) => { const [cls, txt] = eventText(e, tl.smiles, names); return html`<li class=${cls}><span class="mono">${fmtMin(e[0])}</span> ${txt}</li>`; })}</ul>
      </section>`}
    </aside>
  </div>`;
}

export function openLiveNetwork({ jobId = null, example = null }) {
  set({ view: { tab: 'network', jobId, example } });
}
