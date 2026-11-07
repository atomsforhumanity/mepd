// The routes that matter, as one energy diagram (Explore › Landscape).
// Built from the last kinetics run: the starting mixture is a bar at its
// energy; every step that carries material (≥ 1 % of the busiest one) is a
// hump over its TS to the next mixture; line width is how much went that way,
// colour the barrier's band. Everything else the network holds is one count.
// Energies are the ones the kinetics used (free or electronic, as it says),
// relative to the separated molecules; nothing is drawn that was not computed.
import { html, useMemo } from '../lib.js';
import { openReaction, set, state, useStore } from '../store.js';

const W = 960, H = 440, PAD = { l: 64, r: 30, t: 30, b: 46 };
const band = (v) => (v < 10 ? 'var(--chart-3)' : v < 20 ? 'var(--chart-4)' : 'var(--chart-5)');

// A mixture: species ids (repeats allowed), kept sorted so equal ones match.
const key = (ids) => [...ids].sort().join('+');
// A step between several molecules: its barrier is from them apart (the kinetics' reference).
const st0 = (via) => via.from.length > 1 || via.to.length > 1;

function buildTree(res, ws) {
  const g = Object.fromEntries((res.species || []).map((s) => [s.id, s.g]));
  const name = Object.fromEntries((res.species || []).map((s) => [s.id, s.name]));
  const final = Object.fromEntries((res.species || []).map((s) => [s.id, s.final]));
  const start = (res.species || []).filter((s) => s.c0 > 0).map((s) => s.id);
  if (!start.length) return null;
  const sides = (st) => {
    if (st.kind === 'reaction') {
      const r = ws.reactions?.[st.reaction];
      return r ? [r.reactants, r.products] : null;
    }
    const e = ws.edges?.[st.edge];
    return e ? [[e.source], [e.target]] : null;
  };
  const steps = (res.steps || []).map((st) => {
    const sd = sides(st);
    if (!sd) return null;
    const fwd = (st.extent || 0) >= 0;
    // The edge's source is the reaction's (or the edge's) first side: the route enters by it when it runs forward.
    const e = ws.edges?.[st.edge];
    return { from: fwd ? sd[0] : sd[1], to: fwd ? sd[1] : sd[0], barrier: fwd ? st.barrier_fwd : st.barrier_rev,
      extent: Math.abs(st.extent || 0), edge: st.edge, label: st.label, start: e ? (fwd ? e.source : e.target) : null };
  }).filter((s) => s && s.from.every((x) => x in g) && s.to.every((x) => x in g));
  const top = Math.max(0, ...steps.map((s) => s.extent));
  const relevant = steps.filter((s) => top > 0 && s.extent >= 0.01 * top);
  const energy = (ids) => ids.reduce((t, x) => t + (g[x] ?? 0), 0);
  const root = { ids: start, e: energy(start), depth: 0, kids: [] };
  const seen = new Set([key(start)]);
  const nodes = [root];
  let frontier = [root];
  for (let depth = 1; depth <= 3 && frontier.length; depth += 1) {
    const next = [];
    for (const n of frontier) {
      const opts = relevant.filter((s) => {
        const pool = [...n.ids];
        return s.from.every((x) => { const i = pool.indexOf(x); if (i < 0) return false; pool.splice(i, 1); return true; });
      }).sort((a, b) => b.extent - a.extent).slice(0, 5);
      for (const s of opts) {
        const ids = [...n.ids];
        for (const x of s.from) ids.splice(ids.indexOf(x), 1);
        ids.push(...s.to);
        const k = key(ids);
        if (seen.has(k)) continue;
        seen.add(k);
        const child = { ids, e: energy(ids), depth, kids: [], via: s, ts: n.e + s.barrier,
          made: s.to.filter((x) => !start.includes(x)) };
        n.kids.push(child);
        nodes.push(child);
        next.push(child);
      }
    }
    frontier = next;
  }
  // Relative to the starting mixture (the kinetics' energies are absolute).
  const e0 = root.e;
  for (const n of nodes) { n.e -= e0; if (n.ts != null) n.ts -= e0; }
  // The share of the material each new species ends up with (its final amount over all made species').
  const madeTotal = Object.entries(final).filter(([id]) => !start.includes(id)).reduce((t, [, v]) => t + Math.max(0, v || 0), 0);
  return { root, nodes, top, name, final, madeTotal, hidden: steps.length - relevant.length };
}

function layout(tree) {
  const depthMax = Math.max(...tree.nodes.map((n) => n.depth));
  const es = tree.nodes.flatMap((n) => [n.e, n.ts ?? n.e]);
  let lo = Math.min(...es), hi = Math.max(...es);
  const span = Math.max(10, hi - lo);
  lo -= span * 0.12; hi += span * 0.15;
  const X = (d) => PAD.l + 20 + (d / Math.max(1, depthMax)) * (W - PAD.l - PAD.r - 160);
  const Y = (e) => PAD.t + (hi - e) / (hi - lo) * (H - PAD.t - PAD.b);
  // Spread the bars of one column a little so equal energies don't overlap.
  const cols = {};
  for (const n of tree.nodes) (cols[n.depth] ||= []).push(n);
  for (const col of Object.values(cols)) {
    col.sort((a, b) => a.e - b.e);
    col.forEach((n, i) => { n.x = X(n.depth) + (col.length > 1 ? (i - (col.length - 1) / 2) * 34 : 0); n.y = Y(n.e); });
  }
  const ticks = [];
  const step = span > 60 ? 20 : span > 25 ? 10 : 5;
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
  return { Y, ticks };
}

export function LandscapeView() {
  const res = useStore((s) => s.kinetics?.res || null);
  const last = useStore((s) => s.reactionContext?.edgeId || null);   // the route opened last: marked on return
  const ws = useStore((s) => s.workspace);
  const tree = useMemo(() => (res ? buildTree(res, ws) : null), [res, ws]);
  if (!res || !tree) {
    return html`<div class="landscape-empty">
      <h3>The routes that matter</h3>
      <p class="muted">Run Kinetics from a starting mixture: this view then draws the steps that carry material, over their barriers.</p>
      <button class="btn primary" onClick=${() => set({ analyze: { ...(state.analyze || {}), view: 'kinetics' }, view: { tab: 'analyze', jobId: null } })}>Open Kinetics</button>
    </div>`;
  }
  const { Y, ticks } = layout(tree);
  const BAR = 74;
  const paths = tree.nodes.filter((n) => n.via).map((n) => {
    const p = tree.nodes.find((m) => m.kids.includes(n));
    const x0 = p.x + BAR / 2, x1 = n.x - BAR / 2, yT = Y(n.ts), mid = (x0 + x1) / 2;
    const d = `M${x0},${p.y} C${x0 + (mid - x0) * 0.6},${p.y} ${mid - (mid - x0) * 0.5},${yT} ${mid},${yT} C${mid + (x1 - mid) * 0.5},${yT} ${x1 - (x1 - mid) * 0.6},${n.y} ${x1},${n.y}`;
    const w = Math.max(1.5, 9 * Math.sqrt(n.via.extent / (tree.top || 1)));
    return { n, d, w, mid, yT, color: band(n.via.barrier) };
  });
  const label = (n) => n.ids.map((x) => tree.name[x] || x).join(' + ');
  const share = (n) => {
    if (!n.made?.length || !tree.madeTotal) return null;
    const v = n.made.reduce((t, x) => t + Math.max(0, tree.final[x] || 0), 0) / tree.madeTotal;
    return v >= 0.005 ? `${Math.round(v * 100)}%` : '~0%';
  };
  const basis = res.energies === 'free' ? 'free energy' : 'energy';
  return html`<div class="landscape">
    <svg viewBox=${`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Energy diagram of the routes that carry material in the last kinetics run">
      ${ticks.map((t) => html`<g><line x1=${PAD.l} x2=${W - PAD.r} y1=${Y(t)} y2=${Y(t)} class=${t === 0 ? 'ls-zero' : 'ls-grid'} />
        <text x=${PAD.l - 8} y=${Y(t) + 4} class="ls-tick">${t > 0 ? '+' : ''}${t}</text></g>`)}
      <text x="8" y="18" class="ls-axis">${basis}, kcal/mol</text>
      <text x=${W - PAD.r} y=${H - 10} class="ls-axis end">start → over the barrier → product</text>
      ${paths.map((p) => html`<g class=${`ls-route ${p.n.via.edge && p.n.via.edge === last ? 'last' : ''}`} onClick=${() => p.n.via.edge && openReaction(p.n.via.edge,
        { start: p.n.via.start, barrier: p.n.via.barrier, basis: res.energies === 'free' ? 'free energy' : 'electronic energy',
          separated: st0(p.n.via), T: res.temperature })}>
        <path d=${p.d} class="ls-path" style=${`stroke:${p.color};stroke-width:${p.w}`} />
        <path d=${p.d} class="ls-flow" style=${`stroke:${p.color};stroke-width:${Math.max(2, p.w * 0.7)}`} />
        <text x=${p.mid} y=${p.yT - 8} class="ls-ts" style=${`fill:${p.color}`}>‡ ${p.n.via.barrier.toFixed(1)}</text>
        <title>${p.n.via.label}: barrier ${p.n.via.barrier.toFixed(1)} kcal/mol — open it</title>
      </g>`)}
      ${tree.nodes.map((n) => html`<g>
        <line x1=${n.x - BAR / 2} x2=${n.x + BAR / 2} y1=${n.y} y2=${n.y} class=${n.depth ? 'ls-bar' : 'ls-bar start'} />
        <text x=${n.x} y=${n.y + 18} class="ls-name">${label(n)}</text>
        ${share(n) && html`<text x=${n.x} y=${n.y + 38} class="ls-share">${share(n)}</text>`}
      </g>`)}
    </svg>
    <p class="small muted ls-foot">relative to the start · ${res.temperature} K · ${tree.nodes.length - 1} step${tree.nodes.length === 2 ? '' : 's'} shown${tree.hidden ? ` · ${tree.hidden} carry almost nothing` : ''} · width = how much went that way</p>
  </div>`;
}
