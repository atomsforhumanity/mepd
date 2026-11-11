// Explore's network map: the workspace's molecules and the steps between them,
// drawn so that a dense network stays readable.
//  - A step is one arrow between the two molecules it is about (on each side
//    the least shared one); water, H2 and the like ride on the arrow as tags
//    (+ taken in, − given off) instead of each drawing a line to a hub.
//  - The overview stands molecules in columns by how hard they are to reach:
//    the highest barrier on the easiest route from the start (5 kcal/mol bins).
//  - One barrier scale, the workspace's own range, in one hue: a low barrier
//    is dark and thick. Its histogram is the filter.
//  - Clicking a molecule opens its neighbourhood: what it is made from on the
//    left, what it turns into on the right, each partner labelled with its
//    barrier; clicking a partner walks on. A route shows as a strip.
// The editable graph (connect, arrange, conditions, variants) is Graph.js,
// one click away (setExploreView).
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import cytoscape from '../../vendor/cytoscape.esm.min.js';
import { clearSelection, openReaction, prefs, select, set, setExploreView, state, useStore } from '../store.js';
import { PHONE_QUERY, depictUrl, edgeStatus, edgeStatusKey, isComplex, reactionOfEdge, variantScope } from '../util.js';
import { uploadFiles } from './Library.js';

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const SUB = { 0: '₀', 1: '₁', 2: '₂', 3: '₃', 4: '₄', 5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉' };
const subs = (t) => t.replace(/\d/g, (d) => SUB[d]);
const fmt = (v) => (v == null ? '—' : v.toFixed(1));
const FORMULA = /^([A-Z][a-z]?\d*)+$/;
function heavy(f) { let n = 0; for (const [, el, c] of (f || '').matchAll(/([A-Z][a-z]?)(\d*)/g)) if (el !== 'H') n += +c || 1; return n; }
// A molecule's name on the map: its common name, else its formula (never a SMILES).
export function mapLabel(m) {
  if (!m) return '?';
  const named = m.name && m.name !== m.smiles && (/[a-z]{3,}/.test(m.name) || FORMULA.test(m.name));
  if (named) return FORMULA.test(m.name) ? subs(m.name) : m.name;
  return subs(m.formula || m.smiles || m.name || '?');
}
function rgb(c) {
  const m = c.trim();
  if (m.startsWith('#')) return [1, 3, 5].map((i) => parseInt(m.slice(i, i + 2), 16));
  const v = m.match(/[\d.]+/g) || [0, 0, 0];
  return v.slice(0, 3).map(Number);
}
const mix = (a, b, t) => `rgb(${rgb(a).map((v, i) => Math.round(v + (rgb(b)[i] - v) * t)).join(',')})`;

// ---- the model -----------------------------------------------------------
function buildModel(ws, jobs, network) {
  const scope = variantScope(ws, network);
  const mols = {};
  for (const s of Object.values(ws.structures)) {
    if (isComplex(s) || s.role === 'ts') continue;
    if (s.variant && !scope.keep.has(s.variant) && !scope.mine.has(s.id)) continue;
    mols[s.id] = s;
  }
  const side = (sid) => { const s = ws.structures[sid]; return isComplex(s) ? s.members || [] : [sid]; };
  const steps = [];
  for (const e of Object.values(ws.edges)) {
    if (e.variant && !scope.keep.has(e.variant)) continue;
    const r = reactionOfEdge(ws, e.id);
    const sh = new Set(r?.shuttles || []);
    const ins = (r ? r.reactants : side(e.source)).filter((x) => mols[x] && !sh.has(x));
    const outs = (r ? r.products : side(e.target)).filter((x) => mols[x] && !sh.has(x));
    if (!ins.length || !outs.length) continue;
    const st = edgeStatus(e, jobs);
    const fwd = st.barrierFwd ?? st.barrier;
    const kind = ['running', 'queued'].includes(st.status) ? 'run'
      : fwd != null ? 'b' : ['done', 'failed'].includes(st.status) ? 'nots' : 'ns';
    steps.push({ id: e.id, ins, outs, via: [...sh].filter((x) => mols[x]), kind, fwd: kind === 'b' ? fwd : null,
      rev: st.barrierRev, dE: r?.delta_e_kcal ?? null, label: r?.label || '' });
  }
  const deg = {};
  for (const s of steps) for (const x of new Set([...s.ins, ...s.outs, ...s.via])) deg[x] = (deg[x] || 0) + 1;
  const flagged = new Set(Object.values(mols).filter((m) => m.origin && !m.origin.parent && !m.origin.job).map((m) => m.id));
  const pick = (xs) => [...new Set(xs)].sort((x, y) => (flagged.has(y) - flagged.has(x)) || (deg[x] - deg[y])
    || (heavy(mols[y].formula) - heavy(mols[x].formula)) || (x < y ? -1 : 1))[0];
  const tagCount = {};
  for (const s of steps) {
    s.a = pick(s.ins);
    const outs = s.outs.filter((x) => x !== s.a);
    s.b = pick(outs.length ? outs : s.outs);
    const rest = (xs, main) => { const c = {}; xs.forEach((x) => { if (x !== main) c[x] = (c[x] || 0) + 1; }); return c; };
    s.tagIn = rest(s.ins, s.a);
    s.tagOut = rest(s.outs, s.b);
    for (const x of [...Object.keys(s.tagIn), ...Object.keys(s.tagOut), ...s.via]) tagCount[x] = (tagCount[x] || 0) + 1;
  }
  const real = steps.filter((s) => s.a !== s.b);
  const bars = real.filter((s) => s.kind === 'b').map((s) => s.fwd);
  const lo = bars.length ? Math.floor(Math.min(...bars)) : 0, hi = bars.length ? Math.ceil(Math.max(...bars)) : 1;
  // Molecules drawn: every one that is the main molecule of a step, and every one in no step at all.
  const inStep = new Set(real.flatMap((s) => [...s.ins, ...s.outs, ...s.via]));
  const main = new Set(real.flatMap((s) => [s.a, s.b]));
  const nodes = new Set(Object.keys(mols).filter((id) => main.has(id) || !inStep.has(id) || flagged.has(id)));
  // Where the network starts: the molecules added by hand; when (nearly) all
  // were, the ones nothing here makes.
  let starts = [...flagged].filter((id) => nodes.has(id) && main.has(id));
  const made = new Set(real.map((s) => s.b));
  if (!starts.length || starts.length > 0.6 * main.size) {
    const src = [...main].filter((id) => !made.has(id));
    if (src.length) starts = src;
    else if (!starts.length && main.size) starts = [[...main][0]];
  }
  return { mols, steps: real, byId: Object.fromEntries(real.map((s) => [s.id, s])), tagCount, lo, hi, nodes, starts, inStep, any: bars.length > 0 };
}
const tagText = (M, c, sign) => Object.entries(c).map(([x, k]) => `${sign}${k > 1 ? `${k} ` : ''}${mapLabel(M.mols[x])}`);
const stepTags = (M, s) => [...tagText(M, s.tagIn, '+'), ...tagText(M, s.tagOut, '−'), ...s.via.map((x) => `via ${mapLabel(M.mols[x])}`)];
const drawn = (s, o) => s.kind === 'b' || s.kind === 'run' || (s.kind === 'ns' && o.ns) || (s.kind === 'nots' && o.nots);
const visible = (s, o, thr) => drawn(s, o) && (s.kind !== 'b' || thr == null || s.fwd <= thr + 1e-9);
const stepsOf = (M, id) => M.steps.filter((s) => s.ins.includes(id) || s.outs.includes(id) || s.via.includes(id));
const barText = (s) => (s.kind === 'b' ? fmt(s.fwd) : s.kind === 'run' ? 'running' : s.kind === 'ns' ? 'not searched' : 'no TS');

// The easiest way in: per molecule, the route from a start whose highest
// barrier is lowest (then the fewest steps), over the searched steps shown.
function reach(M, o) {
  const adj = {};
  M.steps.filter((s) => s.kind === 'b' && drawn(s, o)).forEach((s) => {
    for (const b of [s.b, ...Object.keys(s.tagOut)]) if (b !== s.a) (adj[s.a] ||= []).push([b, s]);
  });
  const cost = {}, prev = {}, done = new Set();
  const better = (x, y) => !y || x[0] < y[0] - 1e-9 || (Math.abs(x[0] - y[0]) < 1e-9 && x[1] < y[1]);
  const open = [...M.starts];
  M.starts.forEach((s) => { cost[s] = [0, 0]; });
  while (open.length) {
    let bi = 0;
    for (let i = 1; i < open.length; i += 1) if (better(cost[open[i]], cost[open[bi]])) bi = i;
    const a = open.splice(bi, 1)[0];
    if (done.has(a)) continue;
    done.add(a);
    for (const [b, s] of adj[a] || []) {
      const c = [Math.max(cost[a][0], s.fwd), cost[a][1] + 1];
      if (better(c, cost[b])) { cost[b] = c; prev[b] = [a, s]; open.push(b); }
    }
  }
  return { cost, prev };
}
function routeTo(M, R, t) {
  if (!R.cost[t] || M.starts.includes(t)) return null;
  const steps = [], mols = [t];
  for (let x = t; R.prev[x]; x = R.prev[x][0]) { steps.unshift(R.prev[x][1]); mols.unshift(R.prev[x][0]); }
  return { steps, mols, top: R.cost[t][0] };
}

// ---- layouts ---------------------------------------------------------------
const BIN = 5;
const LONE = 2e6, UNREACHED = 1e6;
function layoutAll(M, R, o, aspect = 1.4) {
  const st = M.steps.filter((s) => drawn(s, o));
  const linked = new Set(st.flatMap((s) => [s.a, s.b]));
  // Molecules only in steps not shown (not searched, no TS) wait with them.
  const nodes = [...M.nodes].filter((n) => linked.has(n) || M.starts.includes(n) || !M.inStep.has(n));
  const key = (n) => (M.starts.includes(n) ? -1 : R.cost[n] ? Math.floor(R.cost[n][0] / BIN) : linked.has(n) ? UNREACHED : LONE);
  const keys = [...new Set(nodes.map(key))].sort((a, b) => a - b);
  const cols = keys.map((k) => nodes.filter((n) => key(n) === k));
  const nb = {};
  st.forEach((s) => { (nb[s.a] ||= []).push(s.b); (nb[s.b] ||= []).push(s.a); });
  const y = {}, ci = {};
  cols.forEach((c, i) => {
    c.sort((p, q) => ((R.cost[p]?.[1] ?? 99) - (R.cost[q]?.[1] ?? 99)) || ((R.cost[p]?.[0] ?? 1e9) - (R.cost[q]?.[0] ?? 1e9)) || (p < q ? -1 : 1));
    c.forEach((n, r) => { y[n] = r - (c.length - 1) / 2; ci[n] = i; });
  });
  // Rows: barycentre sweeps, neighbours in nearer columns weighing more.
  for (let sweep = 0; sweep < 12; sweep += 1) {
    const order = sweep % 2 ? cols.map((_, i) => i).reverse() : cols.map((_, i) => i);
    for (const k of order) {
      const bc = {};
      cols[k].forEach((n) => {
        const ps = (nb[n] || []).filter((p) => ci[p] !== k);
        const w = (p) => 1 / (1 + Math.abs(ci[p] - k));
        bc[n] = ps.length ? ps.reduce((a, p) => a + y[p] * w(p), 0) / ps.reduce((a, p) => a + w(p), 0) : y[n];
      });
      cols[k].sort((p, q) => bc[p] - bc[q]);
      cols[k].forEach((n, r) => { y[n] = r - (cols[k].length - 1) / 2; });
    }
  }
  // A crowded column folds into staggered sub-columns, as many as make the
  // picture the canvas's shape (the largest drawing that fits).
  const COL = 170, SUBW = 100, DY = 96;
  const size = (m) => {
    let w = 0, h = 0;
    cols.forEach((c) => { const k = Math.max(1, Math.ceil(c.length / m)); w += (k - 1) * SUBW + COL; h = Math.max(h, Math.ceil(c.length / k)); });
    return Math.min(aspect / Math.max(1, w), 1 / Math.max(1, h * DY));
  };
  let maxRows = 1;
  for (let m = 1; m <= Math.max(1, ...cols.map((c) => c.length)); m += 1) if (size(m) > size(maxRows) * 1.0001) maxRows = m;
  const pos = {}, heads = [];
  let x = 0;
  cols.forEach((c, i) => {
    const k = Math.max(1, Math.ceil(c.length / maxRows)), rows = Math.ceil(c.length / k);
    c.forEach((n, r) => {
      const sub = r % k, row = Math.floor(r / k);
      pos[n] = { x: x + sub * SUBW, y: (row - (rows - 1) / 2) * DY + (k > 1 ? (sub - (k - 1) / 2) * DY / k : 0) };
    });
    const kk = keys[i];
    heads.push({ x: x + (k - 1) * SUBW / 2, key: kk,
      text: kk === -1 ? 'Start' : kk === UNREACHED ? 'No searched route' : kk === LONE ? 'In no step yet' : `${kk * BIN}–${kk * BIN + BIN}` });
    x += (k - 1) * SUBW + COL;
  });
  return { pos, heads, waiting: M.nodes.size - nodes.length };
}
function egoOf(M, id, o, thr) {
  const outP = new Map(), inP = new Map();
  for (const s of stepsOf(M, id).filter((x) => visible(x, o, thr))) {
    const out = s.ins.includes(id) || s.via.includes(id);
    const p = out ? (s.b !== id ? s.b : s.a) : (s.a !== id ? s.a : s.b);
    if (p === id || !M.nodes.has(p)) continue;
    const m = out ? outP : inP;
    if (!m.has(p)) m.set(p, []);
    m.get(p).push(s);
  }
  for (const [p, ss] of inP) if (outP.has(p)) { outP.get(p).back = ss; inP.delete(p); }
  const best = (ss) => ss.slice().sort((a, b) => (a.kind === 'b' ? a.fwd : 1e9) - (b.kind === 'b' ? b.fwd : 1e9))[0];
  const side = (m) => [...m.entries()].map(([p, ss]) => [p, ss, best(ss)]).sort((a, b) => (a[2].kind === 'b' ? a[2].fwd : 1e9) - (b[2].kind === 'b' ? b[2].fwd : 1e9));
  return { out: side(outP), in: side(inP), best };
}
function layoutEgo(M, id, o, thr, look) {
  const E = egoOf(M, id, o, thr);
  const DY = 104, X = 300, SUBW = 200;
  const pos = { [id]: { x: 0, y: 0 } }, labels = {}, edges = [], heads = [];
  const place = (list, sign) => {
    const k = Math.max(1, Math.ceil(list.length / 6)), rows = Math.ceil(list.length / k);
    list.forEach(([p, ss, b], r) => {
      const sub = r % k, row = Math.floor(r / k);
      pos[p] = { x: sign * (X + sub * SUBW), y: (row - (rows - 1) / 2) * DY + (k > 1 ? (sub - (k - 1) / 2) * DY / k : 0) };
      const tags = stepTags(M, b).filter((t) => !t.endsWith(mapLabel(M.mols[id])));
      labels[p] = `${mapLabel(M.mols[p])}\n${barText(b)}${ss.length > 1 ? `  (${ss.length} ways)` : ''}${tags.length ? `  ${tags.join(' ')}` : ''}`;
      const lk = look(b);
      const out = sign > 0;
      edges.push({ group: 'edges', classes: 'ego', data: { id: `ego:${p}`, step: b.id, partner: p, source: out ? id : p, target: out ? p : id,
        col: lk.col, w: lk.w + 0.4, kind: b.kind } });
      if (ss.back) {
        const bb = E.best(ss.back), l2 = look(bb);
        edges.push({ group: 'edges', classes: 'ego back', data: { id: `egob:${p}`, step: bb.id, partner: p, source: p, target: id, col: l2.col, w: 1.2, kind: bb.kind } });
      }
    });
    if (list.length) heads.push({ x: sign * (X + (k - 1) * SUBW / 2), text: sign > 0 ? `Turns into · ${list.length}` : `Made from · ${list.length}` });
  };
  place(E.out, 1);
  place(E.in, -1);
  return { pos, labels, edges, heads };
}
function layoutRoute(M, r, look) {
  // A long route snakes in rows of four, so it stays readable at canvas width.
  const pos = {}, labels = {}, edges = [], PER = 4;
  r.mols.forEach((m, i) => {
    const row = Math.floor(i / PER), k = i % PER;
    pos[m] = { x: (row % 2 ? PER - 1 - k : k) * 280, y: row * 230 };
    labels[m] = mapLabel(M.mols[m]);
  });
  r.steps.forEach((s, i) => {
    const lk = look(s);
    edges.push({ group: 'edges', classes: 'ego rt', data: { id: `rt:${i}`, step: s.id, source: r.mols[i], target: r.mols[i + 1],
      col: lk.col, w: lk.w + 0.5, kind: s.kind, lab: [fmt(s.fwd), ...stepTags(M, s)].join('\n') } });
  });
  return { pos, labels, edges, heads: [] };
}

function styles() {
  const font = css('--font-ui') || 'sans-serif';
  return [
    { selector: 'node', style: { shape: 'round-rectangle', width: 84, height: 63, 'corner-radius': 4, 'background-color': css('--node-bg'),
      'border-width': 1, 'border-color': css('--border-strong'), 'background-image': 'data(img)', 'background-fit': 'contain',
      'background-width': '92%', 'background-height': '92%', 'background-clip': 'node', 'background-image-containment': 'inside',
      label: 'data(label)', 'font-family': font, 'font-size': 12, color: css('--heading'), 'text-valign': 'bottom', 'text-margin-y': 5,
      'text-max-width': 104, 'text-wrap': 'ellipsis', 'min-zoomed-font-size': 7,
      'text-background-color': css('--panel'), 'text-background-opacity': 0.9, 'text-background-padding': 2 } },
    // No drawing (no SMILES): the name inside the card, dark on the card's light ground in both themes.
    { selector: 'node[?noimg]', style: { 'background-image': 'none', 'text-valign': 'center', 'text-margin-y': 0, 'text-background-opacity': 0, color: '#202a33' } },
    { selector: 'node.two', style: { 'text-wrap': 'wrap', 'text-max-width': 190, 'line-height': 1.3 } },
    { selector: 'node[?start]', style: { 'border-width': 2.5, 'border-color': css('--heading'), 'font-weight': 700 } },
    { selector: 'node.center', style: { 'border-width': 3, 'border-color': css('--accent'), width: 104, height: 78, 'font-size': 14, 'font-weight': 700 } },
    { selector: 'node.picked', style: { 'border-width': 3, 'border-color': css('--accent'), 'underlay-color': css('--accent'), 'underlay-opacity': 0.15, 'underlay-padding': 5, 'underlay-shape': 'round-rectangle' } },
    { selector: 'edge', style: { 'curve-style': 'bezier', width: 'data(w)', 'line-color': 'data(col)', 'target-arrow-color': 'data(col)',
      'target-arrow-shape': 'triangle', 'arrow-scale': 0.7, opacity: 0.85 } },
    { selector: 'edge[kind = "ns"]', style: { 'line-style': 'dashed', 'line-dash-pattern': [5, 4], opacity: 0.6 } },
    { selector: 'edge[kind = "nots"]', style: { 'line-style': 'dotted', opacity: 0.7 } },
    { selector: 'edge[kind = "run"]', style: { 'line-style': 'dashed', 'line-dash-pattern': [7, 4], 'line-color': css('--accent'), 'target-arrow-color': css('--accent'), width: 2.4, opacity: 1 } },
    { selector: 'edge.ego', style: { 'curve-style': 'unbundled-bezier', 'control-point-distances': [0], 'control-point-weights': [0.5], opacity: 1 } },
    { selector: 'edge.back', style: { 'curve-style': 'bezier', 'control-point-step-size': 26, 'line-style': 'dashed', 'line-dash-pattern': [4, 4], opacity: 0.8 } },
    { selector: 'edge.rt', style: { label: 'data(lab)', 'text-wrap': 'wrap', 'font-family': font, 'font-size': 13, color: css('--heading'),
      'text-margin-y': -18, 'text-background-color': css('--panel'), 'text-background-opacity': 0.95, 'text-background-padding': 3, 'line-height': 1.35 } },
    { selector: 'edge.over', style: { opacity: 0.04, events: 'no' } },
    { selector: 'node.lonely', style: { opacity: 0.28 } },
    { selector: '.fade', style: { opacity: 0.07 } },
    { selector: 'node.fade', style: { opacity: 0.15 } },
    { selector: '.hl', style: { opacity: 1 } },
    { selector: 'edge.picked', style: { 'underlay-color': css('--accent'), 'underlay-opacity': 0.25, 'underlay-padding': 5 } },
    { selector: 'edge.hot', style: { 'line-color': css('--route'), 'target-arrow-color': css('--route'), width: 4, opacity: 1, 'z-index': 10 } },
    { selector: 'node.hot', style: { 'border-color': css('--route'), 'border-width': 3, opacity: 1 } },
    { selector: '.gone', style: { display: 'none' } },
  ];
}

// The barrier filter: a histogram of the workspace's barriers that is also the legend.
function BarrierFilter({ M, thr, setThr, look }) {
  const bars = M.steps.filter((s) => s.kind === 'b').map((s) => s.fwd);
  if (!bars.length) return html`<span class="nm-filter-empty small muted">No barriers computed yet</span>`;
  const W = 5, b0 = Math.floor(M.lo / W) * W, nb = Math.max(1, Math.ceil((M.hi - b0 + 0.001) / W));
  const counts = Array(nb).fill(0);
  bars.forEach((v) => { counts[Math.min(nb - 1, Math.max(0, Math.floor((v - b0) / W)))] += 1; });
  const mx = Math.max(...counts);
  const t = thr ?? M.hi;
  const shown = bars.filter((v) => v <= t + 1e-9).length;
  return html`<div class="nm-filter">
    <div class="nm-filter-lab">Barriers up to <b>${thr == null ? 'any' : Math.round(thr)}</b> kcal/mol <span>· ${shown} of ${bars.length} steps</span></div>
    <div class="nm-hist">
      <svg viewBox=${`0 0 ${nb * 10} 24`} preserveAspectRatio="none" aria-hidden="true">
        ${counts.map((c, i) => {
          const h = c ? Math.max(2, 22 * c / mx) : 0;
          const on = b0 + i * W < t;
          return html`<rect x=${i * 10 + 1} y=${24 - h} width="8" height=${h} rx="1.5"
            fill=${on ? look({ kind: 'b', fwd: b0 + (i + 0.5) * W }).col : css('--border')}><title>${`${b0 + i * W}–${b0 + (i + 1) * W} kcal/mol: ${c} step${c === 1 ? '' : 's'}`}</title></rect>`;
        })}
      </svg>
      <input type="range" min=${b0} max=${b0 + nb * W} step="1" value=${t} aria-label="Highest barrier shown (kcal/mol)"
        onInput=${(e) => { const v = +e.target.value; setThr(v >= b0 + nb * W ? null : v); }} />
    </div>
  </div>`;
}

const isPhone = () => window.matchMedia(PHONE_QUERY).matches;
const fitCapped = (c) => { if (!c) return; c.maxZoom(1.1); c.fit(c.elements(':visible'), 40); c.maxZoom(2.5); };

export function NetworkMap() {
  const host = useRef(null);
  const cyRef = useRef(null);
  const headRef = useRef(null);
  const tipRef = useRef(null);
  const workspace = useStore((s) => s.workspace);
  const statusKey = useStore((s) => edgeStatusKey(s.jobs));
  const network = useStore((s) => s.network);
  const selection = useStore((s) => s.selection);
  const connectMode = useStore((s) => s.connectMode);
  const [view, setView] = useState({ mode: 'all' });
  const [trail, setTrail] = useState([]);
  const [thr, setThr] = useState(null);
  const [o, setO] = useState(() => ({ ns: false, nots: false, ...prefs.get('mapShow', {}) }));
  const [q, setQ] = useState('');
  const [hi, setHi] = useState(0);
  const [drag, setDrag] = useState(false);
  const setOpt = (patch) => { const v = { ...o, ...patch }; setO(v); prefs.set('mapShow', v); };
  const M = useMemo(() => buildModel(workspace, state.jobs, network), [workspace, statusKey, network]);
  const R = useMemo(() => reach(M, o), [M, o.ns, o.nots]);
  const L = useMemo(() => {
    const c = cyRef.current;
    const aspect = c && c.width() > 50 && c.height() > 50 ? c.width() / c.height() : 1.4;
    return layoutAll(M, R, o, aspect);
  }, [M, R]);
  const live = useRef({});
  const look = (s) => {
    if (s.kind !== 'b') return { col: s.kind === 'run' ? css('--accent') : css('--edge-idle'), w: 1.2 };
    const t = M.hi > M.lo ? Math.min(1, Math.max(0, (s.fwd - M.lo) / (M.hi - M.lo))) : 0;
    return { col: mix(css('--ramp-lo'), css('--ramp-hi'), Math.pow(t, 0.8)), w: 0.9 + 2.8 * (1 - t) };
  };
  live.current = { M, R, L, view, thr, o, look };

  // Go to a molecule's neighbourhood (or, for a molecule that only rides along, highlight its steps).
  const goMol = (id, keepTrail = false) => {
    const { M: m } = live.current;
    if (!m.nodes.has(id)) { if (m.tagCount[id]) { setView({ mode: 'all', tag: id }); setTrail([]); } return; }
    setTrail((t) => { const at = t.indexOf(id); return at >= 0 ? t.slice(0, at + 1) : keepTrail ? [...t, id] : [...t, id]; });
    setView({ mode: 'ego', mol: id });
  };
  const toAll = () => { setView({ mode: 'all' }); setTrail([]); };

  // --- cytoscape, once
  useEffect(() => {
    const c = cytoscape({ container: host.current, style: styles(), minZoom: 0.06, maxZoom: 2.5, boxSelectionEnabled: false });
    cyRef.current = c;
    document.fonts?.ready.then(() => { if (cyRef.current === c) c.style(styles()); });
    const tip = tipRef.current;
    c.on('mouseover', 'node', (e) => { const v = live.current.view; if (v.mode === 'all' && !v.tag && !v.steps) highlight({ mol: e.target.id() }); });
    c.on('mouseout', 'node', () => { if (live.current.view.mode === 'all') highlight(null); });
    c.on('mouseover', 'edge', (e) => {
      const { M: m } = live.current;
      const s = m.byId[e.target.data('step') || e.target.id()];
      if (!s || !tip) return;
      const tags = stepTags(m, s);
      tip.textContent = `${mapLabel(m.mols[s.a])} → ${mapLabel(m.mols[s.b])}   ${s.kind === 'b' ? `${fmt(s.fwd)} kcal/mol` : barText(s)}${tags.length ? `   ${tags.join(' ')}` : ''}`;
      tip.hidden = false;
    });
    c.on('mousemove', (e) => {
      const ev = e.originalEvent;
      if (ev && tip && !tip.hidden) { tip.style.left = `${ev.clientX + 14}px`; tip.style.top = `${ev.clientY + 12}px`; }
    });
    c.on('mouseout', 'edge', () => { if (tip) tip.hidden = true; });
    c.on('tap', (e) => {
      if (tip) tip.hidden = true;
      const ev = e.originalEvent;
      const additive = ev && (ev.shiftKey || ev.metaKey || ev.ctrlKey);
      const v = live.current.view;
      if (e.target === c) {
        if (v.tag || v.steps) setView({ mode: 'all' });
        if (v.mode === 'all') clearSelection();
        return;
      }
      if (e.target.isNode()) {
        const id = e.target.id();
        if (additive) { select({ structures: [id] }, true); return; }
        select({ structures: [id] });
        if (id !== v.mol) goMol(id);
        return;
      }
      select({ edges: [e.target.data('step') || e.target.id()] }, additive);
    });
    c.on('dbltap', 'edge', (e) => { const id = e.target.data('step') || e.target.id(); if (state.workspace.edges[id]) openReaction(id); });
    c.on('viewport', () => placeHeads());
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onTheme = () => { c.style(styles()); render(false); };
    mq.addEventListener('change', onTheme);
    const onRestyle = (e) => { if (e.detail === 'restyle') onTheme(); };
    window.addEventListener('mepd:graph', onRestyle);
    let seen = '';
    const ro = new ResizeObserver(([entry]) => {
      const size = `${Math.round(entry.contentRect.width)}x${Math.round(entry.contentRect.height)}`;
      if (size === seen) return;
      const first = !seen || seen.startsWith('0x');
      seen = size;
      c.resize();
      if (first && entry.contentRect.width > 0) fitCapped(c);
    });
    ro.observe(host.current);
    return () => { mq.removeEventListener('change', onTheme); window.removeEventListener('mepd:graph', onRestyle); ro.disconnect(); c.destroy(); };
  }, []);

  // --- elements follow the model (positions kept, so a change animates)
  useEffect(() => {
    const c = cyRef.current;
    if (!c) return;
    const old = Object.fromEntries(c.nodes().map((n) => [n.id(), { ...n.position() }]));
    const els = [];
    for (const id of M.nodes) {
      const m = M.mols[id];
      const img = depictUrl(m.smiles, 200, 150);
      // A new molecule found from one already drawn grows out of it.
      const from = old[id] || old[m.origin?.parent] || L.pos[id] || { x: 0, y: 0 };
      els.push({ group: 'nodes', data: { id, name: mapLabel(m), label: mapLabel(m), img: img || 'none', noimg: !img, start: M.starts.includes(id) }, position: { ...from } });
    }
    for (const s of M.steps) {
      if (!drawn(s, o) || !M.nodes.has(s.a) || !M.nodes.has(s.b)) continue;
      const lk = look(s);
      els.push({ group: 'edges', data: { id: s.id, source: s.a, target: s.b, col: lk.col, w: lk.w, kind: s.kind } });
    }
    c.batch(() => { c.elements().remove(); c.add(els); });
    let v = view;
    if (v.mode !== 'all' && !M.nodes.has(v.mol)) { v = { mode: 'all' }; setView(v); setTrail([]); }
    render(Object.keys(old).length > 0, !Object.keys(old).length);
  }, [M, L]);

  // --- the view: overview, a neighbourhood or a route
  useEffect(() => { render(true); }, [view]);
  useEffect(() => { render(false, false, true); }, [thr]);

  function render(anim = true, fitNow = false, thrOnly = false) {
    const c = cyRef.current;
    if (!c || !c.nodes().length) { placeHeads([]); return; }
    const { M: m, R: r, L: l, view: v, thr: t, o: op, look: lk } = live.current;
    if (thrOnly && v.mode === 'all') { applyThr(); return; }
    let P;
    c.batch(() => {
      c.remove('.ego');
      c.elements().removeClass('gone fade hl hot two lonely over center');
      c.nodes().forEach((n) => { if (n.data('label') !== n.data('name')) n.data('label', n.data('name')); });
      if (v.mode === 'all') { P = { pos: l.pos, heads: l.heads }; c.nodes().forEach((n) => { if (!l.pos[n.id()]) n.addClass('gone'); }); return; }
      const route = v.mode === 'route' ? routeTo(m, r, v.mol) : null;
      P = route ? layoutRoute(m, route, lk) : layoutEgo(m, v.mol, op, t, lk);
      c.add(P.edges);
      c.elements().forEach((el) => { if (!el.hasClass('ego') && !(el.isNode() && P.pos[el.id()])) el.addClass('gone'); });
      for (const [id, text] of Object.entries(P.labels)) { const n = c.getElementById(id); n.data('label', text); if (text.includes('\n')) n.addClass('two'); }
      c.getElementById(v.mol).addClass('center');
    });
    live.current.heads = P.heads;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    c.stop();
    c.maxZoom(1.1);   // fitting a few molecules never blows them up to poster size
    c.nodes().filter((n) => P.pos[n.id()]).layout({
      name: 'preset', positions: (n) => P.pos[n.id()], animate: anim && !reduced, animationDuration: 480,
      animationEasing: 'ease-in-out-cubic', fit: true, padding: v.mode === 'all' ? 40 : 56,
      stop: () => c.maxZoom(2.5),
    }).run();
    if (v.mode === 'all') applyThr();
    reflectSelection();
    placeHeads();
    setTimeout(() => placeHeads(), anim && !reduced ? 520 : 0);
    if (fitNow) fitCapped(c);
  }
  function applyThr() {
    const c = cyRef.current;
    const { M: m, R: r, thr: t, o: op } = live.current;
    c.batch(() => {
      c.edges(':not(.ego)').forEach((e) => { const s = m.byId[e.id()]; e.toggleClass('over', !!s && !visible(s, op, t)); });
      c.nodes().forEach((n) => {
        const cost = r.cost[n.id()];
        const out = t != null && (!cost || cost[0] > t + 1e-9) && !n.data('start') && !n.connectedEdges().some((e) => !e.hasClass('over'));
        n.toggleClass('lonely', out);
      });
    });
    highlight(null);
    placeHeads();
  }
  function highlight(hover) {
    const c = cyRef.current;
    const { M: m, view: v, thr: t, o: op } = live.current;
    if (!c || v.mode !== 'all') return;
    const f = hover || (v.tag ? { tag: v.tag } : v.steps ? { steps: v.steps } : null);
    c.batch(() => {
      c.elements().removeClass('fade hl hot');
      if (!f) return;
      const ss = (f.mol ? stepsOf(m, f.mol) : f.tag ? stepsOf(m, f.tag) : f.steps.map((id) => m.byId[id]).filter(Boolean)).filter((s) => visible(s, op, t));
      const keep = new Set(f.mol ? [f.mol] : []);
      ss.forEach((s) => { keep.add(s.id); keep.add(s.a); keep.add(s.b); });
      c.elements().forEach((el) => el.addClass(keep.has(el.id()) ? 'hl' : 'fade'));
      if (f.steps) f.steps.forEach((id) => c.getElementById(id).addClass('hot'));
    });
  }
  function reflectSelection() {
    const c = cyRef.current;
    if (!c) return;
    const sel = state.selection;
    c.batch(() => {
      c.elements().removeClass('picked');
      sel.structures.forEach((id) => c.getElementById(id).addClass('picked'));
      sel.edges.forEach((id) => { c.getElementById(id).addClass('picked'); c.edges('.ego').filter((e) => e.data('step') === id).addClass('picked'); });
    });
  }
  function placeHeads(force) {
    const el = headRef.current, c = cyRef.current;
    if (!el || !c) return;
    const heads = force || live.current.heads || [];
    const { view: v, thr: t } = live.current;
    const z = c.zoom(), p = c.pan();
    let last = -1e9;
    // Skip a header that would run into the one before it.
    const shown = heads.filter((h) => { const x = h.x * z + p.x, w = h.text.length * 7.5 + 14; if (x - w / 2 < last) return false; last = x + w / 2; return true; });
    el.innerHTML = (v.mode === 'all' && heads.length > 1 ? '<em>Easiest way in: highest barrier, kcal/mol</em>' : '')
      + shown.map((h) => `<span style="left:${Math.round(h.x * z + p.x)}px" class="${v.mode === 'all' && t != null && h.key >= 0 && h.key < UNREACHED && h.key * BIN > t ? 'off' : ''}">${h.text}</span>`).join('');
  }

  // --- the store's selection: shown, and a single molecule picked elsewhere (Library, search) opens its neighbourhood
  const hadSelection = useRef(false);
  useEffect(() => {
    reflectSelection();
    const { structures, edges } = selection;
    const any = structures.length + edges.length > 0;
    if (structures.length === 1 && !edges.length && M.nodes.has(structures[0]) && !(view.mode !== 'all' && view.mol === structures[0])) goMol(structures[0]);
    else if (!any && hadSelection.current && view.mode !== 'all') toAll();   // Esc: back to the whole network
    hadSelection.current = any;
  }, [selection]);

  // Phones open on where the network starts: the overview is too small to read there.
  const phoneOpened = useRef(false);
  useEffect(() => {
    if (phoneOpened.current || !M.starts.length) return;
    phoneOpened.current = true;
    if (isPhone() && M.nodes.size > 12 && view.mode === 'all') goMol(M.starts[0]);
  }, [M]);

  // Connecting two structures, or experimental conditions, happen on the editable graph.
  useEffect(() => { if (connectMode) setExploreView('graph'); }, [connectMode]);
  useEffect(() => {
    const on = (e) => {
      const d = typeof e.detail === 'string' ? { cmd: e.detail } : (e.detail || {});
      if (d.cmd === 'conditions') setExploreView('graph');
      else if (d.cmd === 'center' && d.id && live.current.M.nodes.has(d.id)) goMol(d.id);
      else if (d.cmd === 'fit-to' && d.ids?.length) {
        setView({ mode: 'all', steps: d.ids });
        setTrail([]);
        setTimeout(() => {
          const c = cyRef.current;
          let els = c.collection();
          d.ids.forEach((id) => { els = els.union(c.getElementById(id)); });
          els = els.union(els.connectedNodes());
          if (els.nonempty()) c.animate({ fit: { eles: els, padding: 80 }, duration: 450 });
        }, 520);
      } else if (d.cmd === 'fit') fitCapped(cyRef.current);
    };
    window.addEventListener('mepd:graph', on);
    return () => window.removeEventListener('mepd:graph', on);
  }, []);
  useEffect(() => { highlight(null); }, [view.tag, view.steps]);

  // --- find
  const hits = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!t) return [];
    return Object.values(M.mols).filter((m) => (M.nodes.has(m.id) || M.tagCount[m.id])
      && [m.name, m.smiles, m.formula, mapLabel(m)].some((x) => (x || '').toLowerCase().includes(t))).slice(0, 8);
  }, [q, M]);
  const go = (id) => { setQ(''); select({ structures: [id] }); if (!M.nodes.has(id)) goMol(id); };

  const route = view.mode !== 'all' ? routeTo(M, R, view.mol) : null;
  const shelf = Object.entries(M.tagCount).filter(([id]) => !M.nodes.has(id)).sort((a, b) => b[1] - a[1]).slice(0, 7);
  const nNs = M.steps.filter((s) => s.kind === 'ns').length, nNots = M.steps.filter((s) => s.kind === 'nots').length;
  const empty = M.nodes.size === 0;
  const crumbs = view.mode === 'all' ? null : html`<nav class="nm-crumbs" aria-label="Where you are">
    <button onClick=${() => { clearSelection(); toAll(); }}>Whole network</button>
    ${trail.map((id, i) => html`<i>›</i><button onClick=${() => { setTrail(trail.slice(0, i + 1)); setView({ mode: 'ego', mol: id }); select({ structures: [id] }); }}>${mapLabel(M.mols[id])}</button>`)}
    ${view.mode === 'route' && html`<i>›</i><span>Easiest route</span>`}
  </nav>`;
  return html`<div class=${`nm-wrap ${drag ? 'drag' : ''}`}
      onDragOver=${(e) => { e.preventDefault(); setDrag(true); }} onDragLeave=${() => setDrag(false)}
      onDrop=${(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]); }}>
    <div class="nm-bar">
      <div class="nm-find">
        <input type="search" placeholder="Find a molecule" value=${q} aria-label="Find a molecule"
          onInput=${(e) => { setQ(e.target.value); setHi(0); }}
          onKeyDown=${(e) => {
            if (e.key === 'ArrowDown') { setHi(Math.min(hits.length - 1, hi + 1)); e.preventDefault(); }
            else if (e.key === 'ArrowUp') { setHi(Math.max(0, hi - 1)); e.preventDefault(); }
            else if (e.key === 'Enter' && hits[hi]) go(hits[hi].id);
            else if (e.key === 'Escape') { setQ(''); e.stopPropagation(); }
          }} />
        ${hits.length > 0 && html`<ul class="nm-hits">${hits.map((m, i) => html`<li class=${i === hi ? 'on' : ''} onMouseDown=${(e) => { e.preventDefault(); go(m.id); }}>
          ${m.smiles && html`<img src=${depictUrl(m.smiles, 120, 90)} alt="" />`}<span>${mapLabel(m)}<small>${m.smiles || ''}</small></span></li>`)}</ul>`}
      </div>
      <${BarrierFilter} M=${M} thr=${thr} setThr=${setThr} look=${look} />
      <div class="nm-toggles">
        ${nNs > 0 && html`<label title="Steps a network expansion proposed that no path search has tried"><input type="checkbox" checked=${o.ns} onChange=${(e) => setOpt({ ns: e.target.checked })} /> not searched (${nNs})</label>`}
        ${nNots > 0 && html`<label title="Searched, but no transition state joins the two ends (or the search failed)"><input type="checkbox" checked=${o.nots} onChange=${(e) => setOpt({ nots: e.target.checked })} /> no TS found (${nNots})</label>`}
        ${L.waiting > 0 && view.mode === 'all' && html`<span title="Molecules whose only steps are hidden: tick the boxes above to see them">${L.waiting} molecule${L.waiting > 1 ? 's' : ''} only in these</span>`}
      </div>
      <a class="nm-help" href="/manual/explore" target="_blank" rel="noopener" title="How the map is drawn (manual)">?</a>
      <div class="nm-switch" role="group" aria-label="Explore view">
        <button class="on" aria-pressed="true" title="The network map">Map</button>
        <button aria-pressed="false" title="The editable graph: connect structures, arrange, conditions, variants" onClick=${() => setExploreView('graph')}>Graph</button>
      </div>
    </div>
    <div class="nm-canvas">
      <div class="nm-cy" ref=${host}></div>
      <div class="nm-heads" ref=${headRef} aria-hidden="true"></div>
      ${crumbs}
      ${view.mode === 'ego' && !M.starts.includes(view.mol) && html`<div class="nm-route">${route
        ? html`<span>Easiest way here: <b>${route.steps.length} step${route.steps.length > 1 ? 's' : ''}</b>, highest barrier <b class="mono">${fmt(route.top)}</b></span>
          <button class="btn small" onClick=${() => setView({ mode: 'route', mol: view.mol })}>Show route</button>`
        : html`<span class="muted">No searched route from the start reaches it.</span>`}</div>`}
      ${view.mode === 'route' && route && html`<div class="nm-route"><span>${route.steps.length} step${route.steps.length > 1 ? 's' : ''} from ${mapLabel(M.mols[route.mols[0]])}; none higher than <b class="mono">${fmt(route.top)}</b> kcal/mol</span>
        <button class="btn small" onClick=${() => setView({ mode: 'ego', mol: view.mol })}>Back</button></div>`}
      <button class="btn small nm-fit" title="Fit in view" onClick=${() => cyRef.current?.animate({ fit: { eles: cyRef.current.elements(':visible'), padding: 40 }, duration: 300 })}>Fit</button>
      ${shelf.length > 0 && html`<div class="nm-shelf"><span>Riding along:</span>${shelf.map(([id, k]) => html`<button
        class=${`nm-chip ${view.tag === id ? 'on' : ''}`} title=${`The ${k} steps that take in or give off ${mapLabel(M.mols[id])}`}
        onClick=${() => { setTrail([]); setView(view.tag === id ? { mode: 'all' } : { mode: 'all', tag: id }); }}>${mapLabel(M.mols[id])}<small>${k}</small></button>`)}</div>`}
      ${empty && html`<div class="graph-empty">
        <h3>Your reaction network</h3>
        <p>Molecules you make or find show up here.</p>
        <button class="btn primary" onClick=${() => set({ view: { tab: 'welcome', jobId: null } })}>Get started</button>
      </div>`}
    </div>
    <div class="nm-tip" ref=${tipRef} hidden></div>
  </div>`;
}
