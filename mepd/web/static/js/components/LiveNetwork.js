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
import { openJob, set, update, useStore } from '../store.js';
import { depictUrl } from '../util.js';

const W = 1250, H = 820, CX = W / 2, CY = H / 2 - 20, M = 72;
const BOX_MIN = 46, BOX_MAX = 112, BOX_SEED = 88;   // card widths (px); height is 3/4 of it
const band = (v) => (v == null ? 'var(--edge-idle)' : v < 10 ? 'var(--chart-3)' : v < 20 ? 'var(--chart-4)' : 'var(--chart-5)');
const bandKey = (v) => (v == null ? 'idle' : v < 10 ? 'low' : v < 20 ? 'mid' : 'high');

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
  const k = Math.min(170, Math.sqrt((W * H) / Math.max(1, n)) * 0.85);   // capped: a small network stays together
  for (let it = 0; it < 260; it += 1) {
    const t = 18 * (1 - it / 260) + 0.5;
    const d = pos.map(() => [0, 0]);
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
        const dd = Math.max(1, Math.hypot(dx, dy));
        // Card half-widths plus a gap; a wide one between big cards (the start and the products), so they stand apart.
        const clear = (big[i] + big[j]) / 2 + (Math.min(big[i], big[j]) > 1.5 * BOX_MIN ? 190 : 40);
        // Repulsion reaches only nearby species (else a crowd pushes loners to the walls); overlap pushes hard.
        const f = (dd < 3 * k ? (k * k) / dd : 0) + (dd < clear ? (clear - dd) * 8 : 0);
        if (!f) continue;
        d[i][0] += (dx / dd) * f; d[i][1] += (dy / dd) * f; d[j][0] -= (dx / dd) * f; d[j][1] -= (dy / dd) * f;
      }
    }
    for (const [a, b] of E) {
      const dx = pos[a][0] - pos[b][0], dy = pos[a][1] - pos[b][1];
      // Springs between big cards (the products and the start) are looser, so what matters spreads out;
      // so are a hub's (one species with many steps), else a dense core knots up.
      const dd = Math.max(1, Math.hypot(dx, dy)), f = (dd * dd) / k / ((big[a] + big[b]) / (2 * BOX_MIN)) / Math.sqrt(Math.max(deg[a], deg[b]));
      d[a][0] -= (dx / dd) * f; d[a][1] -= (dy / dd) * f; d[b][0] += (dx / dd) * f; d[b][1] += (dy / dd) * f;
    }
    for (let i = 0; i < n; i += 1) {
      // A gentle pull to the middle, so species with no step among the shown ones stay near, not in the corners.
      const g = deg[i] ? 0.04 : 0.35;
      d[i][0] += (CX - pos[i][0]) * g; d[i][1] += (CY - pos[i][1]) * g * 1.5;
      if (ids[i] === seed) { pos[i] = [CX, CY]; continue; }
      const dl = Math.max(1, Math.hypot(d[i][0], d[i][1]));
      pos[i][0] = Math.min(W - M, Math.max(M, pos[i][0] + (d[i][0] / dl) * Math.min(dl, t)));
      pos[i][1] = Math.min(H - M - 40, Math.max(M, pos[i][1] + (d[i][1] / dl) * Math.min(dl, t)));
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
  const [hoverEdge, setHoverEdge] = useState(null);   // step id under the pointer: its details
  const job = useStore((s) => (jobId ? s.jobs[jobId] : null));
  const running = ['running', 'queued'].includes(job?.status);
  const url = example ? `/api/examples/${encodeURIComponent(example)}/timeline` : `/api/jobs/${jobId}/network-timeline`;
  useEffect(() => {
    let alive = true;
    const load = () => api.get(url).then((d) => { if (alive) { setTl(d); setError(null); } }).catch((e) => alive && setError(e.message));
    load();
    const t = running ? setInterval(load, 4000) : null;
    return () => { alive = false; if (t) clearInterval(t); };
  }, [url, running]);
  const lay = useMemo(() => (tl?.frames?.length ? layoutOf(tl) : null), [tl]);
  const anyDft = useMemo(() => !!tl?.frames?.some((fr) => fr.species.some((x) => x[5]) || fr.steps.some((x) => x[5] === 'd')), [tl]);
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
  if (!tl) return html`<p class="boot">Loading the network…</p>`;
  if (!lay) {
    return html`<div class="live-network">
      <div class="ln-main">
        <div class="ln-head"><h2>Live network</h2>${running && html`<span class="ln-live">● live</span>`}</div>
        <div class="ln-stage ln-waiting">
          <div class="ln-seed-pulse"></div>
          <p>${running ? 'Looking for the first reactions…' : 'This calculation found no network.'}</p>
          <p class="small muted">${running ? 'Species appear here as the explorer finds them.' : ''}</p>
        </div>
      </div>
      <aside class="ln-side">${jobId && html`<${Closer} jobId=${jobId} job=${job} activity=${tl.activity} running=${running} />`}</aside>
    </div>`;
  }
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
  const sizeOf = Object.fromEntries(nodes.map((x) => [x.id, x]));
  const ends = (a, b) => {
    const [x, y] = [nodeOf[a], nodeOf[b]];
    return x && y && x !== y && shown.has(x) && shown.has(y) ? [x, y] : null;
  };
  // Where the segment from a's centre to b's leaves a's card (plus a gap).
  const clip = (from, to, gap) => {
    const [x0, y0] = at[from], [x1, y1] = at[to], n = sizeOf[from];
    const dx = x1 - x0, dy = y1 - y0;
    const t = Math.min(Math.abs((n.w / 2 + gap) / (dx || 1e-9)), Math.abs((n.h / 2 + gap) / (dy || 1e-9)));
    return [x0 + dx * Math.min(t, 0.45), y0 + dy * Math.min(t, 0.45)];
  };
  const fluxMax = Math.max(1e-30, ...f.steps.map((s) => Math.abs(s[7] || 0)));
  const noFlux = !!f.legacy;   // runs before per-step flux was recorded: barriers only
  // Without flux: read every step outward from the start (its barrier that way); hops from the seed.
  const hops = {};
  if (noFlux) {
    hops[lay.seed] = 0;
    let front = [lay.seed];
    while (front.length) {
      const next = [];
      for (const s of f.steps) {
        for (const [u, v] of [[nodeOf[s[1]], nodeOf[s[2]]], [nodeOf[s[2]], nodeOf[s[1]]]]) {
          if (front.includes(u) && v && hops[v] == null) { hops[v] = hops[u] + 1; next.push(v); }
        }
      }
      front = next;
    }
  }
  // Without flux, the likely routes: from the start to each species holding material, the path whose
  // highest barrier is lowest (as the Landscape reads it); their steps are the main ones, read that way.
  const route = new Map();   // step id -> its direction along a route (true: from -> to)
  if (noFlux) {
    const best = { [lay.seed]: -Infinity }, via = {};
    const todo = new Set([lay.seed]);
    while (todo.size) {
      const u = [...todo].reduce((x, y) => (best[x] <= best[y] ? x : y));
      todo.delete(u);
      for (const st of f.steps) {
        const [id, a, b, dGf, dGr] = st;
        for (const [x, y, bar, dir] of [[nodeOf[a], nodeOf[b], dGf, true], [nodeOf[b], nodeOf[a], dGr, false]]) {
          if (x !== u || !y || !shown.has(y) || bar == null) continue;
          const c = Math.max(best[u], bar);
          if (best[y] == null || c < best[y]) { best[y] = c; via[y] = [id, x, dir]; todo.add(y); }
        }
      }
    }
    for (const x of nodes) {
      if (x.id === lay.seed || !x.pop) continue;
      for (let v = x.id, guard = 0; via[v] && guard < 50; v = via[v][1], guard += 1) route.set(via[v][0], via[v][2]);
    }
  }
  const edges = f.steps.map((s) => {
    const [id, a, b, dGf, dGr, src, irc, flux] = s;
    const fwd = noFlux ? (route.has(id) ? route.get(id) : (hops[nodeOf[a]] ?? 99) <= (hops[nodeOf[b]] ?? 99)) : (flux || 0) >= 0;
    const ab = ends(fwd ? a : b, fwd ? b : a);
    if (!ab) return null;
    const barrier = fwd ? dGf : dGr, back = fwd ? dGr : dGf;
    // Main routes: carry ≥ 5 % of the busiest step's material (no flux: touch a species that holds material).
    const main = noFlux ? route.has(id) : Math.abs(flux || 0) >= 0.05 * fluxMax;
    const w = noFlux ? (main ? 2.4 : 1.4) : Math.min(9, Math.max(1.2, 9 * Math.sqrt(Math.abs(flux || 0) / fluxMax)));
    const p = clip(ab[0], ab[1], 3), q = clip(ab[1], ab[0], 9);
    return { id, from: ab[0], to: ab[1], p, q, barrier, back, w, src, irc, flux, main, moved: moved.get(id),
      flow: noFlux ? null : Math.abs(flux || 0) };
  }).filter(Boolean);
  // The share of what leaves a species that takes this step.
  const outOf = {};
  for (const e of edges) outOf[e.from] = (outOf[e.from] || 0) + (e.flow || 0);
  for (const e of edges) e.share = e.flow == null || !outOf[e.from] ? null : e.flow / outOf[e.from];
  // Barrier labels on the main routes, the busiest first (at most 8, so they stay readable).
  // (greedy: none within 40 px of one already placed)
  const labelledEdges = new Set();
  const placed = [];
  for (const e of edges.filter((x) => x.main && x.barrier != null).sort((a, b) => (b.flow ?? 0) - (a.flow ?? 0) || a.barrier - b.barrier)) {
    const m = [(e.p[0] + e.q[0]) / 2, (e.p[1] + e.q[1]) / 2];
    if (labelledEdges.size >= 8 || placed.some((o) => Math.hypot(o[0] - m[0], o[1] - m[1]) < 40)) continue;
    placed.push(m); labelledEdges.add(e.id);
  }
  const gone = (f.gone || []).map(([id, a, b, state]) => { const ab = ends(a, b); return ab && { id, p: at[ab[0]], q: at[ab[1]], state }; }).filter(Boolean);
  const dashed = (src) => anyDft && src === 'g';
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
        <h2>${example ? 'Live network · example' : job?.title?.replace(/^Reaction network expansion:\s*/, 'Network: ') || 'Live network'}</h2>
        ${running && html`<span class="ln-live" title="Updating as the explorer works">● live</span>`}
        ${f.growing && running && html`<span class="small muted" title="Found since the last kinetics refresh: their amounts come with the next one">new species waiting for kinetics</span>`}
        ${!f.steered && html`<span class="ln-pending" title="These energies wait for their DFT single points (the seed just moved to a lower conformer): not a real swing">steering pending</span>`}
        <span class="spacer"></span>
        <span class="small muted">${f.basis}</span>

      </div>
      <div class=${`ln-stage ${f.steered ? '' : 'unsteered'}`}>
        <svg viewBox=${`0 0 ${W} ${H}`} width="100%" role="img" aria-label="The network: species as their structures, on cards sized by the material they held; steps as lines coloured by barrier and as wide as their flux">
          <defs>${['low', 'mid', 'high', 'idle'].map((k) => html`<marker id=${`ln-arr-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" class=${`ln-arr ${k}`} /></marker>`)}</defs>
          ${gone.map((e) => html`<line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} class=${`ln-gone ${e.state}`} />`)}
          ${[...edges].sort((a, b) => a.main - b.main).map((e) => {
            const k = bandKey(e.barrier), on = hoverEdge === e.id;
            return html`<g class=${`ln-edge ${e.main ? 'main' : 'minor'} ${on ? 'on' : ''}`} key=${e.id}
                onMouseEnter=${() => setHoverEdge(e.id)} onMouseLeave=${() => setHoverEdge((h) => (h === e.id ? null : h))}>
              <line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} class="ln-hit" />
              <line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} style=${`stroke:${band(e.barrier)};stroke-width:${on ? e.w + 2 : e.w}`}
                class=${`ln-line ${dashed(e.src) ? 'gxtb' : ''}`} marker-end=${e.main || on ? `url(#ln-arr-${k})` : null} />
              ${!noFlux && e.main && html`<line x1=${e.p[0]} y1=${e.p[1]} x2=${e.q[0]} y2=${e.q[1]} class="ln-flow" style=${`stroke-width:${Math.max(2, e.w * 0.6)}`} />`}
              ${(labelledEdges.has(e.id) || on) && (() => {
                const mx = (e.p[0] + e.q[0]) / 2, my = (e.p[1] + e.q[1]) / 2, txt = `‡ ${e.barrier.toFixed(1)}`, tw = txt.length * 6.6 + 10;
                return html`<g class="ln-elabel" transform=${`translate(${mx},${my})`}>
                  <rect x=${-tw / 2} y="-9" width=${tw} height="18" rx="3" style=${`stroke:${band(e.barrier)}`} />
                  <text y="4">${txt}</text></g>`;
              })()}
              ${e.moved != null && Math.abs(e.moved) >= 0.1 && html`<text x=${(e.p[0] + e.q[0]) / 2} y=${(e.p[1] + e.q[1]) / 2 - 14} class=${`ln-delta ${e.moved < 0 ? 'down' : 'up'}`}>${sign(e.moved)}</text>`}
            </g>`;
          })}
          ${nodes.map((x) => html`<g class="ln-node" key=${x.id} transform=${`translate(${x.at[0]},${x.at[1]})`}
              onMouseEnter=${() => setHover(x.id)} onMouseLeave=${() => setHover((h) => (h === x.id ? null : h))}>
            ${x.fresh && html`<rect x=${-x.w / 2 - 6} y=${-x.h / 2 - 6} width=${x.w + 12} height=${x.h + 12} rx="6" class="ln-fresh" />`}
            <rect x=${-x.w / 2} y=${-x.h / 2} width=${x.w} height=${x.h} rx="4" class=${`ln-card ${x.id === lay.seed ? 'seed' : x.pop ? 'pop' : ''} ${anyDft && !x.dft ? 'gxtb' : ''}`} />
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
        ${(() => {
          const e = hoverEdge && edges.find((x) => x.id === hoverEdge);
          if (!e) return null;
          const nm = (id) => labelOf(id) || (tl.smiles[id] ? short(tl.smiles[id]) : 'unidentified');
          const dG = e.barrier != null && e.back != null ? e.barrier - e.back : null;
          const mx = (e.p[0] + e.q[0]) / 2, my = (e.p[1] + e.q[1]) / 2;
          return html`<div class="ln-edge-card" style=${`left:${(100 * mx) / W}%;top:${(100 * my) / H}%`}>
            <div class="ln-ec-route"><b>${nm(e.from)}</b> <span class="muted">→</span> <b>${nm(e.to)}</b></div>
            <div class="ln-ec-nums">
              <span><b style=${`color:${band(e.barrier)}`}>${e.barrier?.toFixed(1) ?? '?'}</b>barrier</span>
              ${e.back != null && html`<span><b>${e.back.toFixed(1)}</b>back</span>`}
              ${dG != null && html`<span><b>${dG > 0 ? '+' : ''}${dG.toFixed(1)}</b>ΔG</span>`}
              ${e.share != null && html`<span><b>${e.share >= 0.01 ? `${Math.round(100 * e.share)}%` : '<1%'}</b>of what leaves ${nm(e.from)}</span>`}
            </div>
            ${e.barrier != null && e.barrier < 0 && html`<div class="small muted">Below zero: its TS lies under the energy recorded for ${nm(e.from)} (often a lower conformer found later).</div>`}
            <div class="small muted">kcal/mol · ${e.src === 'd' ? 'DFT energies' : e.src === 'b' ? 'borrowed from a twin step' : 'g-xTB energies'}${e.irc === 1 ? ' · IRC connects ✓' : e.irc === 0 ? ' · IRC does not connect' : ''}</div>
          </div>`;
        })()}
      </div>
      <div class="ln-legend small" aria-label="How to read the network">
        <span class="ln-lg"><svg width="64" height="14"><line x1="2" y1="7" x2="18" y2="7" class="lg-l low" /><line x1="23" y1="7" x2="39" y2="7" class="lg-l mid" /><line x1="44" y1="7" x2="60" y2="7" class="lg-l high" /></svg>
          ${'barrier to cross: low (<10) · medium · high (>20 kcal/mol)'}</span>
        <span class="ln-lg"><svg width="52" height="16"><rect x="1" y="2" width="50" height="12" rx="3" class="lg-pill" /><text x="26" y="11.5" class="lg-pill-t">‡ 18.2</text></svg>its height, on the busiest steps</span>
        ${noFlux ? html`<span class="ln-lg" title="This run comes from an explorer version that did not record how much material each step carries"><svg width="40" height="14"><line x1="2" y1="7" x2="30" y2="7" class="lg-l mid" /><path d="M30,2 L38,7 L30,12 z" class="lg-arr" /></svg>likely route: the lowest barriers from the start to each product (flows not recorded)</span>`
          : html`<span class="ln-lg"><svg width="40" height="14"><line x1="2" y1="7" x2="30" y2="7" class="lg-l thick" /><path d="M30,2 L38,7 L30,12 z" class="lg-arr" /></svg>material flows this way; thicker = more</span>`}
        <span class="ln-lg"><svg width="40" height="16"><rect x="1" y="5" width="12" height="9" rx="2" class="lg-card" /><rect x="18" y="1" width="20" height="14" rx="2" class="lg-card" /></svg>bigger card: more of it formed</span>
        ${anyDft && html`<span class="ln-lg"><svg width="30" height="14"><line x1="2" y1="7" x2="28" y2="7" class="lg-l dash" /></svg>dashed: g-xTB energy, DFT check to come</span>`}
        ${gone.length > 0 && html`<span class="ln-lg"><svg width="30" height="14"><line x1="2" y1="7" x2="28" y2="7" class="lg-l gone" /></svg>grey: step closed or replaced</span>`}
        <span class="ln-lg muted">hover a step or a card for details</span>
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
      ${jobId && html`<${Closer} jobId=${jobId} job=${job} activity=${tl.activity} running=${running} />`}
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

// What the explorer is doing now, and the ways in for a closer look: the
// simulations under the network (reactor MD, path searches), its results, log.
const WORK = { propose: 'proposing', fast: 'quick paths', channels: 'channels', irc: 'IRCs', piece: 'path pieces', ts: 'TS searches', rotamer: 'rotamers' };
function Closer({ jobId, job, activity, running }) {
  const q = activity?.queue;
  const sims = activity?.sims || [];
  const reactor = ['nanoreactor', 'reactor'].includes(job?.params?.generator);
  const openSim = (stream) => {
    update((s) => { s.networkFocus = { jobId, stream }; });
    openJob(jobId, 'live');
  };
  const done = q?.finished ? Object.entries(q.finished).filter(([, v]) => v.done || v.failed) : [];
  return html`<section class="ln-closer">
    ${running && q && html`<span class="section-title">Now</span>
      <div class="ln-workers" title=${`${q.busy || 0} of ${q.workers} workers busy`}>
        ${Array.from({ length: q.workers || 0 }, (_, k) => html`<i class=${k < (q.busy || 0) ? 'on' : ''}></i>`)}
        <span class="small muted">${q.busy || 0}/${q.workers} busy</span>
      </div>
      ${(q.running || []).slice(0, 4).map((r) => html`<div class="small ln-now">${WORK[r.kind] || r.kind} · <span class="mono">${r.what}</span></div>`)}`}
    ${done.length > 0 && html`<span class="section-title">${running ? 'Done so far' : 'Work done'}</span><div class="ln-done small muted">${done.map(([k, v]) => html`<span title=${v.failed ? `${v.failed} failed` : ''}><b>${v.done}</b> ${WORK[k] || k}</span>`)}</div>`}
    <span class="section-title">Look closer</span>
    <div class="ln-doors">
      ${reactor && html`<button class="btn small" onClick=${() => openJob(jobId, 'reactor')} title="The molecular dynamics that proposes the reactions">Reactor MD</button>`}
      <button class="btn small" onClick=${() => openJob(jobId, 'live')} title="Each proposed reaction and its path search, live">Path searches</button>
      <button class="btn small" onClick=${() => openJob(jobId, 'result')} disabled=${job?.status === 'queued'}>Results</button>
      <button class="btn small" onClick=${() => openJob(jobId, 'log')}>Log</button>
    </div>
    ${sims.length > 0 && html`<div class="ln-sims">${sims.slice(0, 5).map((m) => html`<button class="ln-sim" onClick=${() => openSim(m.stream)} title=${`${m.label}${m.caption ? ` · ${m.caption}` : ''}: open it`}>
      ${m.from ? html`<img src=${depictUrl(m.from, 120, 90)} alt=${m.from} />` : html`<span>?</span>`}
      <span class="ln-sim-arrow">→</span>
      ${m.to ? html`<img src=${depictUrl(m.to, 120, 90)} alt=${m.to} />` : html`<span>?</span>`}
      <span class="small muted">${(m.label || '').replace(/ from species.*/, '')}</span>
    </button>`)}</div>`}
  </section>`;
}
