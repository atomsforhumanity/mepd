// The routes that matter, as one energy diagram (Explore › Landscape).
// Built from the last kinetics run: the starting mixture is a bar at its
// energy; every step that carries material (≥ 1 % of the busiest one) is a
// hump over its TS to the next mixture; line width is how much went that way,
// colour the barrier's band. Everything else the network holds is one count.
// Under each state: the share of the starting mass that ends there (each
// species counted once, at the state its main route makes it), so the drawn
// states, what is left of the start and what goes beyond the drawn routes add
// up to 100 %. The tree follows the flow as deep as it goes (up to 8 steps).
// Energies are the ones the kinetics used (free or electronic, as it says),
// relative to the separated molecules; nothing is drawn that was not computed.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { openReaction, set, state, useStore } from '../store.js';
import { CoverageWarning, networkKey, runKineticsWith, settingsKey } from './Kinetics.js';

const MAX_DEPTH = 8, MAX_NODES = 28, PER_STATE = 4;
const W = 960, H = 440, PAD = { l: 64, r: 30, t: 30, b: 46 };
const band = (v) => (v < 10 ? 'var(--chart-3)' : v < 20 ? 'var(--chart-4)' : 'var(--chart-5)');

// A mixture: species ids (repeats allowed), kept sorted so equal ones match.
const key = (ids) => [...ids].sort().join('+');
// A step between several molecules: its barrier is from them apart (the kinetics' reference).
const st0 = (via) => via.from.length > 1 || via.to.length > 1;
// What a step makes: its products less its reactants (a molecule on both sides only looks on).
const netMade = (s) => { const pool = [...s.from]; return s.to.filter((x) => { const i = pool.indexOf(x); if (i < 0) return true; pool.splice(i, 1); return false; }); };

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
  // The routes that carry material, plus the next few side routes however little they carry (drawn faint,
  // with their real share): a 0.04 % side product is shown as one, not rounded away.
  const byFlow = [...steps].sort((a, b) => b.extent - a.extent);
  const relevant = byFlow.filter((s, k) => top > 0 && (s.extent >= 0.01 * top || (k < 5 && s.extent > 0)));
  for (const s of relevant) s.faint = s.extent < 0.01 * top;
  const energy = (ids) => ids.reduce((t, x) => t + (g[x] ?? 0), 0);
  const root = { ids: start, e: energy(start), depth: 0, kids: [], flow: Infinity };
  const seen = new Set([key(start)]);
  const nodes = [root];
  let frontier = [root];
  // Breadth first, the busiest states first, as deep as material goes (MAX_DEPTH), at most MAX_NODES states.
  for (let depth = 1; depth <= MAX_DEPTH && frontier.length && nodes.length < MAX_NODES; depth += 1) {
    const next = [];
    for (const n of [...frontier].sort((a, b) => b.flow - a.flow)) {
      const opts = relevant.filter((s) => {
        const pool = [...n.ids];
        return s.from.every((x) => { const i = pool.indexOf(x); if (i < 0) return false; pool.splice(i, 1); return true; });
      }).sort((a, b) => b.extent - a.extent).slice(0, PER_STATE);
      for (const s of opts) {
        if (nodes.length >= MAX_NODES) break;
        const ids = [...n.ids];
        for (const x of s.from) ids.splice(ids.indexOf(x), 1);
        ids.push(...s.to);
        const k = key(ids);
        if (seen.has(k)) continue;
        seen.add(k);
        const child = { ids, e: energy(ids), depth, kids: [], via: s, ts: n.e + s.barrier, flow: Math.min(n.flow, s.extent),
          made: netMade(s).filter((x) => !start.includes(x)) };
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
  // Where the starting mass ends: each species' final amount times its mass, over the whole (mass is conserved;
  // a species without a mass counts as one). Each species is credited once, to the state its busiest drawn step
  // makes it at; what no drawn state makes is "beyond the drawn routes".
  const mass = Object.fromEntries((res.species || []).map((s) => [s.id, s.mass || 1]));
  const amount = (id) => Math.max(0, final[id] || 0) * (mass[id] ?? 1);
  const total = Object.keys(final).reduce((t, id) => t + amount(id), 0) || 1;
  const home = {};
  for (const n of nodes.slice(1)) {
    for (const x of n.made) if (!home[x] || home[x].via.extent < n.via.extent) home[x] = n;
  }
  for (const n of nodes) n.share = n === root ? null : n.made.filter((x) => home[x] === n).reduce((t, x) => t + amount(x), 0) / total;
  const unreacted = start.reduce((t, x) => t + amount(x), 0) / total;
  const beyondList = Object.keys(final).filter((id) => !start.includes(id) && !home[id] && amount(id) / total >= 0.005)
    .map((id) => ({ id, name: name[id] || id, share: amount(id) / total })).sort((a, b) => b.share - a.share);
  const drawn = nodes.reduce((t, n) => t + (n.share || 0), 0);
  const beyond = Math.max(0, 1 - drawn - unreacted);
  const faint = relevant.filter((s) => s.faint).length;
  const shownSteps = new Set(nodes.map((n) => n.via).filter(Boolean));
  return { root, nodes, top, name, unreacted, beyond, beyondList, hidden: steps.length - shownSteps.size, faint };
}

function layout(tree) {
  const depthMax = Math.max(...tree.nodes.map((n) => n.depth));
  const es = tree.nodes.flatMap((n) => [n.e, n.ts ?? n.e]);
  let lo = Math.min(...es), hi = Math.max(...es);
  const span = Math.max(10, hi - lo);
  lo -= span * 0.12; hi += span * 0.15;
  const Wd = Math.max(W, 150 * depthMax + 250);   // room for every column's labels, however deep
  const X = (d) => PAD.l + 20 + (d / Math.max(1, depthMax)) * (Wd - PAD.l - PAD.r - 160);
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
  return { Y, ticks, Wd };
}

export function LandscapeView() {
  const res = useStore((s) => s.kinetics?.res || null);
  const kin = useStore((s) => s.kinetics || null);
  const draft = useStore((s) => s.kineticsDraft || null);
  const last = useStore((s) => s.reactionContext?.edgeId || null);   // the route opened last: marked on return
  const ws = useStore((s) => s.workspace);
  const tree = useMemo(() => (res && !res.compare ? buildTree(res, ws) : null), [res, ws]);
  // Follow Kinetics: when its settings (as now in its form) or the network (steps found, moved, checked;
  // energies) differ from what this was drawn from, run them again -- at most every 10 s while a run grows it.
  const netKey = useMemo(() => networkKey(ws), [ws]);
  const settings = draft || kin;
  const stale = !!settings?.mix?.length && !settings.cmp
    && (!kin?.res || settingsKey(settings) !== settingsKey(kin) || netKey !== kin.netKey);
  const [updating, setUpdating] = useState(false);
  const lastRun = useRef(0);
  useEffect(() => {
    if (!stale || updating) return undefined;
    const wait = Math.max(300, 10000 - (Date.now() - lastRun.current));
    const t = setTimeout(async () => {
      setUpdating(true);
      lastRun.current = Date.now();
      try { await runKineticsWith(settings); } finally { setUpdating(false); }
    }, wait);
    return () => clearTimeout(t);
  }, [stale, updating, netKey, settingsKey(settings || {})]);
  if ((!res || !tree) && updating) return html`<div class="landscape-empty"><p class="muted">Running the kinetics…</p></div>`;
  if (!res || !tree) {
    return html`<div class="landscape-empty">
      <h3>The routes that matter</h3>
      <p class="muted">Run Kinetics from a starting mixture: this view then draws the steps that carry material, over their barriers.</p>
      <button class="btn primary" onClick=${() => set({ analyze: { ...(state.analyze || {}), view: 'kinetics' }, view: { tab: 'analyze', jobId: null } })}>Open Kinetics</button>
    </div>`;
  }
  const { Y, ticks, Wd } = layout(tree);
  const BAR = 74;
  const paths = tree.nodes.filter((n) => n.via).map((n) => {
    const p = tree.nodes.find((m) => m.kids.includes(n));
    const x0 = p.x + BAR / 2, x1 = n.x - BAR / 2, yT = Y(n.ts), mid = (x0 + x1) / 2;
    const d = `M${x0},${p.y} C${x0 + (mid - x0) * 0.6},${p.y} ${mid - (mid - x0) * 0.5},${yT} ${mid},${yT} C${mid + (x1 - mid) * 0.5},${yT} ${x1 - (x1 - mid) * 0.6},${n.y} ${x1},${n.y}`;
    const w = Math.max(1.5, 9 * Math.sqrt(n.via.extent / (tree.top || 1)));
    return { n, d, w, mid, yT, color: band(n.via.barrier), faint: n.via.faint };
  });
  // Barrier labels that would overlap (humps side by side at similar heights) step up out of each other's way.
  const placed = [];
  for (const p of [...paths].sort((a, b) => a.yT - b.yT)) {
    let y = p.yT - 8;
    while (placed.some((q) => Math.abs(q.x - p.mid) < 52 && Math.abs(q.y - y) < 14)) y -= 14;
    placed.push({ x: p.mid, y });
    p.ly = y;
  }
  const label = (n) => n.ids.map((x) => tree.name[x] || x).join(' + ');
  // whole percent where that says it; small shares as they are (0.04 %), not '~0 %'
  const pct = (v) => (v >= 0.995 && v < 1 ? '>99%' : v >= 0.01 ? `${Math.round(v * 100)}%` : v >= 1e-4 ? `${(v * 100).toPrecision(1)}%` : v > 0 ? '<0.01%' : '0%');
  const share = (n) => (n.share == null || !n.made?.length ? null : pct(n.share));
  // States whose new species no exploration expanded: what they do next was never searched.
  const unexplored = new Set((res.coverage?.frontier || []).map((f) => f.id));
  const openEnd = (n) => n.made?.filter((x) => unexplored.has(x)) || [];
  const basis = res.energies === 'free' ? 'free energy' : 'energy';
  const startNames = (res.species || []).filter((x) => x.c0 > 0).map((x) => x.name).join(' + ');
  return html`<div class=${`landscape ${updating ? 'ls-updating' : ''}`}>
    <${CoverageWarning} res=${res} compact />
    ${(updating || stale) && html`<p class="small ls-status">${updating ? 'Updating to what Kinetics has now…' : 'Kinetics changed: updating shortly…'}</p>`}
    <div class="ls-scroll"><svg viewBox=${`0 0 ${Wd} ${H}`} width="100%" style=${`min-width:${Math.round(Wd * 0.75)}px`} role="img" aria-label="Energy diagram of the routes that carry material in the last kinetics run">
      ${ticks.map((t) => html`<g><line x1=${PAD.l} x2=${Wd - PAD.r} y1=${Y(t)} y2=${Y(t)} class=${t === 0 ? 'ls-zero' : 'ls-grid'} />
        <text x=${PAD.l - 8} y=${Y(t) + 4} class="ls-tick">${t > 0 ? '+' : ''}${t}</text></g>`)}
      <text x="8" y="18" class="ls-axis">${basis}, kcal/mol</text>
      <text x=${Wd - PAD.r} y=${H - 10} class="ls-axis end">start → over the barrier → product</text>
      ${paths.map((p) => html`<g class=${`ls-route ${p.n.via.edge && p.n.via.edge === last ? 'last' : ''} ${p.faint ? 'faint' : ''}`} onClick=${() => p.n.via.edge && openReaction(p.n.via.edge,
        { start: p.n.via.start, barrier: p.n.via.barrier, basis: res.energies === 'free' ? 'free energy' : 'electronic energy',
          separated: st0(p.n.via), T: res.temperature })}>
        <path d=${p.d} class="ls-path" style=${`stroke:${p.color};stroke-width:${p.w}`} />
        <path d=${p.d} class="ls-flow" style=${`stroke:${p.color};stroke-width:${Math.max(2, p.w * 0.7)}`} />
        <text x=${p.mid} y=${p.ly} class="ls-ts" style=${`fill:${p.color}`}>‡ ${p.n.via.barrier.toFixed(1)}</text>
        <title>${p.n.via.label}: barrier ${p.n.via.barrier.toFixed(1)} kcal/mol — open it</title>
      </g>`)}
      ${tree.nodes.map((n) => html`<g>
        <line x1=${n.x - BAR / 2} x2=${n.x + BAR / 2} y1=${n.y} y2=${n.y} class=${n.depth ? 'ls-bar' : 'ls-bar start'} />
        <text x=${n.x} y=${n.y + 18} class="ls-name">${label(n)}</text>
        ${share(n) && html`<text x=${n.x} y=${n.y + 38} class="ls-share"><title>Share of the starting mass that ends here</title>${share(n)}</text>`}
        ${openEnd(n).length > 0 && html`<text x=${n.x} y=${n.y + (share(n) ? 54 : 34)} class="ls-open">unexplored ›<title>${openEnd(n).map((x) => tree.name[x] || x).join(', ')}: no exploration expanded ${openEnd(n).length === 1 ? 'it' : 'them'}, so what ${openEnd(n).length === 1 ? 'it does' : 'they do'} next was never searched</title></text>`}
      </g>`)}
    </svg></div>
    <p class="small ls-mass">Where the starting mass ends: ${pct(1 - tree.unreacted - tree.beyond)} in the states drawn · ${pct(tree.unreacted)} unreacted${tree.beyond >= 1e-4
      ? html` · <b>${pct(tree.beyond)} beyond the drawn routes</b>${tree.beyondList.length ? ` (${tree.beyondList.slice(0, 4).map((b) => `${b.name} ${pct(b.share)}`).join(', ')}${tree.beyondList.length > 4 ? ', …' : ''})` : ''}` : ''}</p>
    <p class="small muted ls-foot">from ${startNames} · relative to the start · ${res.temperature} K · ${tree.nodes.length - 1} step${tree.nodes.length === 2 ? '' : 's'} shown${tree.faint ? ` · faint: side routes under 1 % of the main flow` : ''}${tree.hidden ? ` · ${tree.hidden} more carry less` : ''} · width = how much went that way</p>
  </div>`;
}
