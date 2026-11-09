// Prototype designs for the live network views (dev only: add ?design=a, b or c to the page's
// address, before the #). The default views are unchanged. All three share:
// - a stable layout: a species is placed once, where it was found (next to the species it came
//   from), from what was known when it appeared; nothing placed ever moves, so a replay and the
//   live run look the same;
// - a summary strip: time, what the explorer is doing, counts, the leading products with their
//   amounts (as % of the start) and how long the answer has held;
// - speed at the run's temperature in place of fixed barrier bands: each step's half-life
//   (Eyring, from its free-energy barrier) against the time the kinetics simulate;
// - colour with one job each: the three leading products keep a hue (blue, orange, aqua) wherever
//   they show; a grey ramp says how fast a step is; magenta is what runs now.
// A: Map (a calm node-link map with a side panel). B: Scheme (left to right by steps from the
// start, the products in an outcome lane). C: Rings (growth rings around the start, with what was
// just found).
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { depictUrl } from '../util.js';

export const designParam = () => {
  try { return new URLSearchParams(location.search).get('design'); } catch { return null; }
};

// ---------------------------------------------------------------- chemistry helpers
const R_KCAL = 1.987204e-3, KB_H = 2.083661912e10;   // kcal/mol/K; Boltzmann/Planck, 1/(s K)
export function halfLife(barrier, T) {
  if (barrier == null || !T) return null;
  return Math.LN2 / (KB_H * T * Math.exp(-barrier / (R_KCAL * T)));
}
export function fmtTime(s) {
  if (s == null || !Number.isFinite(s)) return '—';
  if (s < 1e-12) return '< 1 ps';
  if (s > 3.15e9) return '> 100 years';
  const u = [[1e-12, 'ps'], [1e-9, 'ns'], [1e-6, 'µs'], [1e-3, 'ms'], [1, 's'], [60, 'min'], [3600, 'h'], [86400, 'days'], [3.15e7, 'years']];
  let k = 0;
  for (let i = 0; i < u.length; i += 1) if (s >= u[i][0]) k = i;
  const v = s / u[k][0];
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${u[k][1]}`;
}
const fmtClock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
const pct = (x) => (x == null ? '—' : x >= 0.995 ? `${Math.round(100 * x)} %` : x >= 0.0995 ? `${Math.round(100 * x)} %`
  : x >= 0.001 ? `${(100 * x).toFixed(1)} %` : x > 0 ? '< 0.1 %' : '0 %');

// The run's temperature (from the kinetics basis) and simulated time (the job's --time), which set
// what "fast" means; without a time the speeds are read against 1 s and the key says so.
export function conditionsOf(basis, command, mixture = null) {
  const T = mixture?.temperature_K ?? (+(/([\d.]+)\s*K\b/.exec(basis || '')?.[1]) || null);
  const m = /--time\s+([\d.eE+-]+)/.exec(command || '');
  const tau = mixture?.time_s ?? (m ? +m[1] : null);
  const b = /--time-budget\s+([\d.]+)/.exec(command || '');
  return { T, tau: tau ?? 1, tauKnown: tau != null, budget: b ? +b[1] : null };
}
export function speedOf(barrier, cond) {
  const t = halfLife(barrier, cond.T);
  if (t == null) return null;
  return t < cond.tau / 100 ? 'fast' : t < cond.tau ? 'run' : 'slow';
}

const drawable = (smi) => !!smi && !smi.startsWith('?') && !/unusual bonding/.test(smi);
const short = (s, n = 26) => (s && s.length > n ? `${s.slice(0, n - 1)}…` : s || '');
const reduced = () => { try { return matchMedia('(prefers-reduced-motion: reduce)').matches; } catch { return false; } };

// ---------------------------------------------------------------- stable placement
// A spatial hash of what is placed, for "is this spot free".
class Grid {
  constructor(cell = 48) { this.cell = cell; this.m = new Map(); }
  key(x, y) { return `${Math.floor(x / this.cell)},${Math.floor(y / this.cell)}`; }
  add(x, y, r) { const k = this.key(x, y); if (!this.m.has(k)) this.m.set(k, []); this.m.get(k).push([x, y, r]); }
  free(x, y, r) {
    const cx = Math.floor(x / this.cell), cy = Math.floor(y / this.cell), reach = Math.ceil((r + 70) / this.cell);
    for (let i = cx - reach; i <= cx + reach; i += 1) {
      for (let j = cy - reach; j <= cy + reach; j += 1) {
        for (const [px, py, pr] of this.m.get(`${i},${j}`) || []) if (Math.hypot(px - x, py - y) < r + pr) return false;
      }
    }
    return true;
  }
}

// Map: next to where it was found, continuing outward along the branch, in the nearest free spot.
function placeMap(id, a, st) {
  if (a == null) { st.grid.add(0, 0, 70); return [0, 0]; }
  const [ax, ay] = st.pos.get(a);
  const aa = st.anchor.get(a);
  const n = st.kids.get(a) || 0;
  st.kids.set(a, n + 1);
  const base = aa == null ? n * 2.39996 : Math.atan2(ay - st.pos.get(aa)[1], ax - st.pos.get(aa)[0]);
  const r = 22;
  for (const d of aa == null ? [150, 190, 230, 270, 320, 380, 440] : [70, 95, 120, 150, 185, 225, 270]) {
    for (let k = 0; k < 24; k += 1) {
      const th = base + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (Math.PI / 12);
      const x = ax + d * Math.cos(th), y = ay + d * Math.sin(th);
      if (st.grid.free(x, y, r)) { st.grid.add(x, y, r); return [x, y]; }
    }
  }
  const x = ax + 300 * Math.cos(base), y = ay + 300 * Math.sin(base);
  st.grid.add(x, y, r);
  return [x, y];
}

// Rings: one band per step from the start; in a band, the free spot nearest the angle of the species
// it came from, on the innermost track that has one (a crowded band fills outward like a seed head).
const RING = (h) => (h <= 0 ? 0 : 175 + 200 * (h - 1));
const TRACK = 13, SEP = 15;
function placeRings(id, a, st) {
  if (a == null) { st.angle.set(id, 0); return [0, 0]; }
  const h = Math.min(st.hop.get(id), 5);
  const n = st.kids.get(a) || 0;
  st.kids.set(a, n + 1);
  const target = st.anchor.get(a) == null ? n * 2.39996 : st.angle.get(a) + (n % 2 ? 1 : -1) * 0.04 * Math.ceil(n / 2);
  const tryTrack = (j, reach) => {
    const r = RING(h) + j * TRACK, sep = SEP / r, used = st.ring.get(`${h}:${j}`) || [];
    for (let k = 0; k * sep * 0.5 <= reach; k += 1) {
      const th = target + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * sep * 0.5;
      if (used.every((o) => Math.abs(Math.atan2(Math.sin(th - o), Math.cos(th - o))) >= sep)) {
        used.push(th);
        st.ring.set(`${h}:${j}`, used);
        return [r, th];
      }
    }
    return null;
  };
  let got = null;
  for (let j = 0; j < 14 && !got; j += 1) got = tryTrack(j, 0.5);
  for (let j = 0; j < 30 && !got; j += 1) got = tryTrack(j, Math.PI);
  const [r, th] = got || [RING(h), target];
  st.angle.set(id, th);
  return [r * Math.cos(th), r * Math.sin(th)];
}

// Scheme: a column per step from the start. A species gets a row (a structure) once it matters (holds
// material, or a route runs through it), in the free row nearest the species it came from; until then it
// is one dot in its column's pile of what was explored, above. Rows and pile slots never move.
const SUB_W = 118, SUBS = 3, COL_W = SUB_W * SUBS + 90, ROW_H = 92, ROWS = 6, PILE = 9, PILE_W = 26, MAX_COL = 4;
function useScheme(key, disc, order, relevantOrder) {
  const ref = useRef(null);
  if (!ref.current || ref.current.key !== key) ref.current = { key, row: new Map(), pile: new Map(), cells: new Map(), piles: new Map() };
  const sc = ref.current;
  const col = (id) => Math.min(disc.hop.get(id) ?? 1, MAX_COL);
  const x0 = (c) => (c === 0 ? 0 : 150 + (c - 1) * COL_W);
  for (const id of order) {
    if (sc.pile.has(id) || !disc.pos.has(id) || disc.anchor.get(id) == null) continue;
    const c = col(id), k = sc.piles.get(c) || 0;
    sc.piles.set(c, k + 1);
    sc.pile.set(id, [x0(c) + (k % PILE_W) * PILE, -40 - Math.floor(k / PILE_W) * PILE]);
  }
  for (const id of relevantOrder) {
    if (sc.row.has(id) || !disc.pos.has(id)) continue;
    if (disc.anchor.get(id) == null) { sc.row.set(id, [0, 60]); continue; }
    const c = col(id);
    let a = disc.anchor.get(id);
    while (a != null && !sc.row.has(a)) a = disc.anchor.get(a);
    const ty = a == null ? 0 : Math.max(0, Math.round((sc.row.get(a)[1] - 60) / ROW_H));
    const used = sc.cells.get(c) || new Set();
    // the free cell nearest the anchor's row: first sub-column first, then down the block
    let best = null;
    for (let r = 0; r < 400 && !best; r += 1) {
      for (const dy of r ? [r, -r] : [0]) {
        const y = ty + dy;
        if (y < 0) continue;
        for (let j = 0; j < SUBS; j += 1) if (!used.has(`${y}:${j}`) && (y < ROWS || j === 0 || true)) { best = [y, j]; break; }
        if (best) break;
      }
    }
    used.add(`${best[0]}:${best[1]}`);
    sc.cells.set(c, used);
    sc.row.set(id, [x0(c) + 60 + best[1] * SUB_W, 60 + best[0] * ROW_H]);
  }
  return sc;
}

const PLACE = { a: placeMap, b: placeMap, c: placeRings };

// Positions kept across polls (a ref): only species not placed yet are placed, in the order they
// were found, each from what was placed before it.
function usePlacement(design, key, order, anchorOf) {
  const ref = useRef(null);
  if (!ref.current || ref.current.key !== key) {
    ref.current = { key, pos: new Map(), hop: new Map(), anchor: new Map(), kids: new Map(), angle: new Map(),
      ring: new Map(), rows: new Map(), grid: new Grid() };
  }
  const st = ref.current;
  for (const id of order) {
    if (st.pos.has(id)) continue;
    const a = anchorOf(id, st);
    st.anchor.set(id, a);
    st.hop.set(id, a == null ? 0 : (st.hop.get(a) ?? 0) + 1);
    st.pos.set(id, PLACE[design](id, a, st));
  }
  return st;
}

// The view's frame of reference: everything shown, padded; it only grows while a run is followed,
// and eases there (no camera jumps).
function useViewBox(box, follow) {
  const [vb, setVb] = useState(box);
  const cur = useRef(box);
  useEffect(() => {
    const target = follow ? [Math.min(cur.current[0], box[0]), Math.min(cur.current[1], box[1]),
      Math.max(cur.current[2], box[2]), Math.max(cur.current[3], box[3])] : box;
    if (reduced()) { cur.current = target; setVb(target); return undefined; }
    const from = cur.current, t0 = performance.now();
    let raf;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / 700), e = 1 - (1 - k) ** 3;
      cur.current = from.map((v, i) => v + (target[i] - v) * e);
      setVb(cur.current);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [box.join(','), follow]);
  return vb;
}

// ---------------------------------------------------------------- the model of one moment
// Products keep a hue while they lead: each of the three leading products holds its slot (blue,
// orange, aqua) for as long as it stays among the three; a newcomer takes a slot that fell free.
const HUES = 3;
function productHues(frames, seed, keyOf, upto) {
  const slot = new Map();
  for (let i = 0; i <= upto; i += 1) {
    const tops = new Map();
    for (const s of frames[i].species) {
      if (s[0] === seed || !(s[3] > 1e-4)) continue;
      tops.set(keyOf(s[0]), (tops.get(keyOf(s[0])) || 0) + s[3]);
    }
    const top = [...tops].sort((x, y) => y[1] - x[1]).slice(0, HUES).map((x) => x[0]);
    for (const k of [...slot.keys()]) if (!top.includes(k)) slot.delete(k);
    for (const k of top) {
      if (slot.has(k)) continue;
      const used = new Set(slot.values());
      slot.set(k, [0, 1, 2].find((x) => !used.has(x)));
    }
  }
  return slot;
}

// Cards (structures) must not cover each other: each, by importance, at its species or the nearest
// free spot outward from the start (a thin leader joins it to its species).
function cardSpots(ids, P, size, origin) {
  const placed = [], out = new Map();
  for (const id of ids) {
    const [x, y] = P(id), w = size(id), h = w * 0.75 + 22;
    const base = Math.atan2(y - origin[1], x - origin[0]);
    let spot = null;
    for (let r = 0; r <= 420 && !spot; r += 22) {
      for (let k = 0; k < (r ? 18 : 1); k += 1) {
        const th = base + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (Math.PI / 9);
        const cx = x + r * Math.cos(th), cy = y + r * Math.sin(th);
        if (placed.every(([px, py, pw, ph]) => Math.abs(px - cx) > (pw + w) / 2 + 10 || Math.abs(py - cy) > (ph + h) / 2 + 6)) { spot = [cx, cy]; break; }
      }
    }
    spot = spot || [x, y];
    placed.push([spot[0], spot[1], w, h]);
    out.set(id, spot);
  }
  return out;
}

// What the explorer runs now, as the network's species and pairs (job "what": "3→17 · ch_3_17_0",
// "species 4 · batch 1").
function activityOf(q) {
  const pairs = [], species = [];
  for (const r of q?.running || []) {
    const m = /^(\d+)→(\d+)/.exec(r.what || '');
    if (m) pairs.push([m[1], m[2], r.kind]);
    const s = /^species (\d+)/.exec(r.what || '');
    if (s && r.kind === 'propose') species.push(s[1]);
  }
  return { pairs, species };
}
const PHASE = { propose: 'Proposing reactions (MD)', fast: 'Searching transition states', piece: 'Path searches',
  ts: 'Transition states of path pieces', irc: 'Checking what each TS joins (IRC)', channels: 'Sampling conformers of a reaction',
  steer: 'Refining energies (DFT single points)', rotamer: 'Trying rotamers' };
function phaseOf(q) {
  if (!q) return null;
  const by = {};
  for (const r of q.running || []) by[r.kind] = (by[r.kind] || 0) + 1;
  const top = Object.entries(by).sort((a, b) => b[1] - a[1])[0];
  const waiting = Object.values(q.waiting || {}).reduce((a, b) => a + b, 0);
  if (q.ended) return { text: 'Finished its queue', detail: '' };
  if (!top) return { text: waiting ? `${waiting} jobs wait for a worker` : 'Waiting for the next kinetics', detail: '' };
  const rest = Object.entries(by).sort((a, b) => b[1] - a[1]).slice(1, 3).map(([k, n]) => `${n} ${(PHASE[k] || k).toLowerCase()}`);
  return { text: PHASE[top[0]] || top[0], detail: `${q.busy || 0} of ${q.workers} workers${rest.length ? ` · also ${rest.join(', ')}` : ''}${waiting ? ` · ${waiting} waiting` : ''}` };
}

// One moment of a network expansion's timeline (tl, frame cur).
export function expansionModel(tl, cur) {
  const frames = tl.frames, f = frames[cur], seed = frames[0]?.species[0]?.[0] ?? '0';
  const first = new Map(), parent = new Map();
  for (const fr of frames) for (const s of fr.species) if (!first.has(s[0])) first.set(s[0], fr.t);
  for (const [id, t, p] of tl.explored || []) {
    if (!first.has(id) || t < first.get(id)) first.set(id, t);
    if (p != null && p !== id) parent.set(id, p);
  }
  const nb = new Map();
  const link = (a, b) => { if (!nb.has(a)) nb.set(a, []); if (!nb.get(a).includes(b)) nb.get(a).push(b); };
  for (const fr of frames) for (const st of [...fr.steps, ...(fr.gone || [])]) { link(st[1], st[2]); link(st[2], st[1]); }
  const order = [...first.keys()].sort((a, b) => first.get(a) - first.get(b) || (+a - +b) || (a < b ? -1 : 1));
  const productOf = {};
  for (const [name, ids] of Object.entries(tl.selectivity?.species || {})) for (const id of ids) productOf[id] = name;
  const names = tl.names || {};
  const keyOf = (id) => (productOf[id] ? `p:${productOf[id]}` : drawable(tl.smiles[id]) ? tl.smiles[id] : names[id] || id);
  return { frames, f, seed, first, parent, nb, order, productOf, names, keyOf };
}

// The anchor a species is placed from: the species it was proposed from, else the first species a
// step joins it to that is already placed (nearest the start), else the start.
function anchorFor(m) {
  return (id, st) => {
    if (id === m.seed) return null;
    const p = m.parent.get(id);
    if (p != null && st.pos.has(p)) return p;
    const placed = (m.nb.get(id) || []).filter((x) => st.pos.has(x));
    if (placed.length) return placed.reduce((x, y) => ((st.hop.get(x) ?? 99) <= (st.hop.get(y) ?? 99) ? x : y));
    return m.seed;
  };
}

// Without per-step flux (runs on develop record none), the likely routes: from the start to each
// species holding material, the path whose highest barrier is lowest.
function likelyRoutes(steps, seed, holders) {
  const best = { [seed]: -Infinity }, via = {};
  const todo = new Set([seed]);
  while (todo.size) {
    const u = [...todo].reduce((x, y) => (best[x] <= best[y] ? x : y));
    todo.delete(u);
    for (const [id, a, b, dGf, dGr] of steps) {
      for (const [x, y, bar, dir] of [[a, b, dGf, true], [b, a, dGr, false]]) {
        if (x !== u || bar == null) continue;
        const c = Math.max(best[u], bar);
        if (best[y] == null || c < best[y]) { best[y] = c; via[y] = [id, x, dir]; todo.add(y); }
      }
    }
  }
  const route = new Map();
  for (const h of holders) for (let v = h, g = 0; via[v] && g < 50; v = via[v][1], g += 1) route.set(via[v][0], via[v][2]);
  return route;
}

// ---------------------------------------------------------------- the view
export function DesignNetwork({ tl, cur, design, jobId, example, job, running, follow, controls }) {
  const m = useMemo(() => expansionModel(tl, cur), [tl, cur]);
  const [hover, setHover] = useState(null);   // {kind: 'n'|'e', id}
  const key = `${jobId || example}|${design}`;
  const st = usePlacement(design, key, m.order, anchorFor(m));
  const f = m.f;
  const cond = conditionsOf(f.basis, tl.command ?? job?.command);
  const hues = useMemo(() => productHues(m.frames, m.seed, m.keyOf, cur), [tl, cur]);
  const hueOf = (id) => (id === m.seed ? null : hues.get(m.keyOf(id)));
  const startSmiles = drawable(tl.smiles[m.seed]) ? tl.smiles[m.seed] : null;
  const imgOf = (id, w = 240, h = 180) => (drawable(tl.smiles[id]) ? `${depictUrl(tl.smiles[id], w, h, id === m.seed ? null : startSmiles)}&bond=${Math.round(w / 6)}`
    : jobId && !jobId.startsWith("mix:") ? `/api/jobs/${jobId}/species-graph/${encodeURIComponent(id)}` : null);
  const nameOf = (id) => (id === m.seed ? m.names[id] || 'start' : m.productOf[id] || m.names[id] || '');
  const label = (id) => nameOf(id) || short(tl.smiles[id], 18) || `species ${id}`;

  // species now: every one found so far (dots), with amounts where the frame has them
  const amount = new Map(f.species.map((s) => [s[0], s]));
  const seen = m.order.filter((id) => m.first.get(id) <= f.t + 1e-6 && st.pos.has(id));
  const peakMax = Math.max(1e-30, ...f.species.filter((s) => s[0] !== m.seed).map((s) => s[2] || 0));
  const holders = f.species.filter((s) => s[0] !== m.seed && (s[2] || 0) >= 0.02 * peakMax).sort((a, b) => b[2] - a[2]);
  const fresh = (id) => id !== m.seed && cur > 0 && m.first.get(id) >= f.t - 12;
  const act = running && cur === m.frames.length - 1 ? activityOf(tl.activity?.queue) : { pairs: [], species: [] };
  const busy = new Set([...act.species, ...act.pairs.flatMap((x) => x.slice(0, 2))]);

  // steps now
  const noFlux = !!f.legacy;
  const fluxMax = Math.max(1e-30, ...f.steps.map((s) => Math.abs(s[7] || 0)));
  const route = noFlux ? likelyRoutes(f.steps, m.seed, holders.slice(0, 6).map((s) => s[0])) : null;
  const edges = f.steps.filter((s) => st.pos.has(s[1]) && st.pos.has(s[2]) && s[1] !== s[2]).map((s) => {
    const [id, a, b, dGf, dGr, src, irc, flux, below] = s;
    const fwd = noFlux ? (route.has(id) ? route.get(id) : (st.hop.get(a) ?? 9) <= (st.hop.get(b) ?? 9)) : (flux || 0) >= 0;
    const [from, to] = fwd ? [a, b] : [b, a];
    const barrier = fwd ? dGf : dGr, back = fwd ? dGr : dGf;
    const main = noFlux ? route.has(id) : Math.abs(flux || 0) >= 0.05 * fluxMax;
    const w = noFlux ? (main ? 3 : 1.2) : Math.min(10, Math.max(1.2, 10 * Math.sqrt(Math.abs(flux || 0) / fluxMax)));
    return { id, from, to, barrier, back, src, irc, flux, below: !!below, main, w, speed: below ? null : speedOf(barrier, cond), hue: main ? hueOf(to) : null, co: tl.co?.[id] };
  });
  const outOf = {};
  for (const e of edges) outOf[e.from] = (outOf[e.from] || 0) + Math.abs(e.flux || 0);
  for (const e of edges) e.share = noFlux || !outOf[e.from] ? null : Math.abs(e.flux || 0) / outOf[e.from];

  // what forms: amounts at the end of the simulated time, as a share of the start
  const groups = new Map();
  for (const s of f.species) {
    if (s[0] === m.seed || !(s[3] > 0)) continue;
    const k = m.keyOf(s[0]);
    const g = groups.get(k) || { id: s[0], key: k, fin: 0 };
    g.fin += s[3];
    groups.set(k, g);
  }
  const products = [...groups.values()].sort((a, b) => b.fin - a.fin).slice(0, 3);
  const startLeft = amount.get(m.seed)?.[3];
  const leadAt = (fr) => {
    let best = null;
    const g = new Map();
    for (const s of fr.species) if (s[0] !== m.seed && s[3] > 0) g.set(m.keyOf(s[0]), (g.get(m.keyOf(s[0])) || 0) + s[3]);
    for (const [k, v] of g) if (!best || v > best[1]) best = [k, v];
    return best;
  };
  const lead = leadAt(f);
  let steadySince = f.t;
  for (let i = cur; i >= 0 && lead; i -= 1) {
    const l = leadAt(m.frames[i]);
    if (!l || l[0] !== lead[0] || Math.abs(l[1] - lead[1]) > 0.02) break;
    steadySince = m.frames[i].t;
  }
  const series = products.map((p) => ({ ...p, hue: hues.get(p.key),
    pts: m.frames.slice(0, cur + 1).map((fr) => [fr.t, fr.species.filter((s) => m.keyOf(s[0]) === p.key).reduce((t, s) => t + (s[3] || 0), 0)]) }));

  // where things are drawn: A and C at their placed spot; B by its scheme (a row once it matters)
  const relevantOrder = useMemo(() => {
    if (design !== 'b') return [];
    const out = [], seenR = new Set();   // (one row per product: mirror twins share the first's)
    for (const fr of m.frames) {
      const pk = Math.max(1e-30, ...fr.species.filter((s) => s[0] !== m.seed).map((s) => s[2] || 0));
      for (const s of fr.species) if (!seenR.has(m.keyOf(s[0])) && (s[0] === m.seed || (s[2] || 0) >= 0.02 * pk)) { seenR.add(m.keyOf(s[0])); out.push([s[0], fr.t]); }
    }
    return out;
  }, [tl, design]);
  const relevantNow = new Set(relevantOrder.filter(([, t]) => t <= f.t + 1e-6).map(([id]) => id));
  const sc = useScheme(key, st, m.order, relevantOrder.map(([id]) => id));
  const P = design === 'b' ? (id) => (relevantNow.has(id) ? sc.row.get(id) : sc.pile.get(id)) || st.pos.get(id) : (id) => st.pos.get(id);
  // one card per product (mirror twins and a benchmark's named product are one): its most held species
  const byKey = new Map();
  for (const s of holders) if (!byKey.has(m.keyOf(s[0]))) byKey.set(m.keyOf(s[0]), s[0]);
  const groupFin = (id) => groups.get(m.keyOf(id))?.fin ?? amount.get(id)?.[3];
  const cardIds = design === 'b' ? new Set([...relevantNow].filter((id) => sc.row.has(id)))
    : new Set([m.seed, ...[...byKey.values()].slice(0, 6)]);
  const cardW = design === 'b' ? (id) => (id === m.seed ? 120 : 92)
    : (id) => (id === m.seed ? 112 : Math.max(80, Math.min(136, 136 * Math.sqrt((amount.get(id)?.[2] || 0) / peakMax))));
  const order = [...cardIds].filter((id) => seen.includes(id)).sort((a, b) => (b === m.seed) - (a === m.seed) || (amount.get(b)?.[2] || 0) - (amount.get(a)?.[2] || 0));
  // marks and text keep their size on screen as the view zooms out (the stage is about 1250 x 620 px)
  let b0 = [Infinity, Infinity, -Infinity, -Infinity];
  for (const id of seen) { const [x, y] = P(id); b0 = [Math.min(b0[0], x), Math.min(b0[1], y), Math.max(b0[2], x), Math.max(b0[3], y)]; }
  const scale = design === 'b' ? 1 : Math.max(1, (b0[2] - b0[0] + 300) / 1250, (b0[3] - b0[1] + 260) / 620);
  const spot = design === 'b' ? new Map(order.map((id) => [id, P(id)])) : cardSpots(order, P, (id) => cardW(id) * scale, P(m.seed));
  const C = (id) => spot.get(id) || P(id);   // where its mark is: its card, else its dot

  let box = [Infinity, Infinity, -Infinity, -Infinity];
  for (const id of seen) {
    const [x, y] = C(id), r = cardIds.has(id) ? (cardW(id) * scale) / 2 + 22 * scale : 14 * scale;
    box = [Math.min(box[0], x - r), Math.min(box[1], y - r), Math.max(box[2], x + r), Math.max(box[3], y + r + (cardIds.has(id) ? 18 : 0))];
  }
  if (design === 'b') box = [box[0] - 40, box[1] - 20, box[2] + 360, box[3] + 10];
  const vb = useViewBox([box[0], box[1], Math.max(700, box[2] - box[0]), Math.max(460, box[3] - box[1])], follow);

  const ends = (id, toward, gap) => {
    const [x0, y0] = C(id), [x1, y1] = C(toward);
    if (design === 'b' && cardIds.has(id)) {   // leave a card by its side
      const w = (cardW(id) * scale) / 2 + gap;
      return [x0 + (x1 >= x0 ? w : -w), y0];
    }
    const r = cardIds.has(id) ? cardW(id) * scale * 0.42 : 6 * scale;
    const d = Math.max(1e-6, Math.hypot(x1 - x0, y1 - y0)), k = Math.min(0.45, (r + gap) / d);
    return [x0 + (x1 - x0) * k, y0 + (y1 - y0) * k];
  };
  const pathOf = (e) => {
    const p = ends(e.from, e.to, 3 * scale), q = ends(e.to, e.from, 8 * scale);
    if (design === 'b') {
      const dx = Math.max(30, Math.abs(q[0] - p[0]) / 2), sgn = q[0] >= p[0] ? 1 : -1;
      if (Math.abs(q[0] - p[0]) < 10) return `M${p[0]},${p[1]} C${p[0] + 60},${p[1]} ${q[0] + 60},${q[1]} ${q[0]},${q[1]}`;
      return `M${p[0]},${p[1]} C${p[0] + sgn * dx},${p[1]} ${q[0] - sgn * dx},${q[1]} ${q[0]},${q[1]}`;
    }
    if (design === 'c') {
      const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2, nx = -(q[1] - p[1]) * 0.12, ny = (q[0] - p[0]) * 0.12;
      return `M${p[0]},${p[1]} Q${mx + nx},${my + ny} ${q[0]},${q[1]}`;
    }
    return `M${p[0]},${p[1]} L${q[0]},${q[1]}`;
  };
  // "found from": only beyond the first step (the start's own are implied by where they sit)
  const branch = (id) => {
    const a = st.anchor.get(id);
    if (a == null || a === m.seed || design === 'b') return null;
    const [x0, y0] = P(a), [x1, y1] = P(id);
    return `M${x0},${y0} L${x1},${y1}`;
  };
  const leader = (id) => { const [x, y] = P(id), [cx, cy] = C(id); return Math.hypot(cx - x, cy - y) > 2 ? `M${x},${y} L${cx},${cy}` : null; };
  // progressive disclosure: the routes to the products always; a species' own steps on hover; a step
  // just found for a moment
  // (a few at a time: a finished run's last frame, rebuilt from its summary, would flash them all)
  let newStep = new Set(cur > 0 ? f.steps.map((x) => x[0]).filter((id) => !m.frames[cur - 1].steps.some((y) => y[0] === id)) : []);
  if (newStep.size > 12) newStep = new Set();
  const shownEdges = edges.filter((e) => e.main || hover?.id === e.id || (hover?.kind === 'n' && (e.from === hover.id || e.to === hover.id))
    || (newStep.has(e.id) && !e.below))
    .filter((e) => design !== 'b' || (cardIds.has(e.from) && cardIds.has(e.to)) || hover?.id === e.id);

  const hovered = hover?.kind === 'n' ? hover.id : null;
  const hoveredEdge = hover?.kind === 'e' ? edges.find((e) => e.id === hover.id) : null;
  const near = hovered ? new Set([hovered, ...edges.filter((e) => e.from === hovered || e.to === hovered).flatMap((e) => [e.from, e.to])]) : null;
  const dim = (id) => near && !near.has(id);
  const dotIds = seen.filter((id) => !cardIds.has(id));
  const piles = design === 'b' ? [...sc.piles.entries()] : [];
  const lane = design === 'b' ? series.map((p, k) => ({ ...p, x: box[2] - 330, y: box[1] + 60 + k * 150 })) : [];
  // arrivals grow from where they came from: the species they were found from, or (B) their pile slot
  const origin = (id) => (design === 'b' && cardIds.has(id) ? sc.pile.get(id) : null) || P(st.anchor.get(id)) || C(id);
  const style = (id) => { const a = origin(id), c = C(id); return `--fx:${a[0] - c[0]}px;--fy:${a[1] - c[1]}px;transform:translate(${c[0]}px,${c[1]}px)`; };
  const since = new Map(relevantOrder);
  const freshCard = (id) => (design === 'b' ? cur > 0 && id !== m.seed && (since.get(id) ?? -1) >= f.t - 12 : fresh(id));

  const stage = html`<div class=${`ld-stage ld-${design}`}>
    <svg viewBox=${vb.join(' ')} preserveAspectRatio="xMidYMid meet" role="img" style=${`--s:${scale}`}
      aria-label="The network: each species placed where it was found; the start, the products and what holds material as structures; steps as lines, darker the faster at the run's temperature">
      <defs>${['fast', 'run', 'slow', 'none', 'h0', 'h1', 'h2'].flatMap((k) => [['s', 9], ['m', 14], ['l', 22]].map(([z, px]) => html`<marker id=${`ld-arr-${k}-${z}`} viewBox="0 0 10 10" refX="8" refY="5"
        markerWidth=${px * scale} markerHeight=${px * scale} markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L9,5 L0,9 z" class=${`ld-arrhead ${k}`} /></marker>`))}</defs>
      ${design === 'c' && [1, 2, 3, 4, 5].filter((h) => [...st.hop.values()].some((x) => Math.min(x, 5) === h)).map((h) => html`<g class="ld-ring" key=${`r${h}`}>
        <circle r=${RING(h) - 14} /><text y=${-(RING(h) + TRACK * Math.max(0, ...[...st.ring.keys()].filter((k) => k.startsWith(`${h}:`)).map((k) => +k.split(':')[1])) + 16)}>${h === 1 ? '1 step from the start' : `${h}${h === 5 ? '+' : ''} steps from the start`}</text></g>`)}
      ${piles.map(([c, k]) => html`<text class="ld-colhead left" key=${`h${c}`} x=${c === 0 ? 0 : 150 + (c - 1) * COL_W} y="-18">${c === 0 ? 'start' : `${c}${c === MAX_COL ? '+' : ''} step${c === 1 ? '' : 's'} · ${k} found`}</text>`)}
      <g class="ld-branches">${seen.map((id) => { const d = branch(id); return d && html`<path key=${`b${id}`} d=${d} class=${`ld-branch ${dim(id) ? 'dim' : ''}`} />`; })}</g>
      <g class="ld-steps">${[...shownEdges].sort((a, b) => a.main - b.main).map((e) => html`<g key=${e.id} class=${`ld-step ${e.main ? 'main' : ''} ${newStep.has(e.id) && !e.main ? 'new' : ''} ${e.speed || 'none'} ${e.below ? 'below' : ''} ${e.irc === -1 ? 'unchecked' : ''} ${(near && !(near.has(e.from) && near.has(e.to))) ? 'dim' : ''} ${hover?.id === e.id ? 'on' : ''}`}
          onMouseEnter=${() => setHover({ kind: 'e', id: e.id })} onMouseLeave=${() => setHover(null)}>
        <path d=${pathOf(e)} class="ld-hit" />
        <path d=${pathOf(e)} class="ld-line" style=${`stroke-width:${e.w * scale}px${e.hue != null ? `;stroke:var(--ld-h${e.hue})` : ''}`}
          marker-end=${e.main || hover?.id === e.id ? `url(#ld-arr-${e.hue != null ? `h${e.hue}` : e.speed || 'none'}-${e.w < 2.5 ? 's' : e.w < 6 ? 'm' : 'l'})` : null} />
        ${e.main && !noFlux && html`<path d=${pathOf(e)} class="ld-flow" style=${`stroke-width:${Math.max(2, e.w * 0.55) * scale}px`} />`}
        ${e.co?.with?.length > 0 && (e.main || hover?.id === e.id) && (() => {
          const [x0, y0] = C(e.from), [x1, y1] = C(e.to);
          return html`<text x=${x0 + (x1 - x0) * 0.5} y=${y0 + (y1 - y0) * 0.5 - 8 * scale} class="ld-co" style=${`font-size:${12 * scale}px`}>+ ${e.co.with.map(label).join(' + ')}</text>`;
        })()}
      </g>`)}</g>
      <g class="ld-activity">${act.pairs.filter(([a, b]) => P(a) && P(b) && a !== b).map(([a, b, kind], k) => html`<path key=${`p${a}-${b}-${k}`}
        d=${`M${C(a)[0]},${C(a)[1]} L${C(b)[0]},${C(b)[1]}`} class=${`ld-probe ${kind}`} />`)}</g>
      <g class="ld-leaders">${order.map((id) => { const d = leader(id); return d && html`<path key=${`l${id}`} d=${d} class="ld-leader" />`; })}</g>
      <g class="ld-nodes">${dotIds.map((id) => {
        const s = amount.get(id), hue = hueOf(id);
        return html`<g key=${`n${id}`} class=${`ld-dot ${s?.[4] ? 'pop' : ''} ${hue != null && s?.[3] > 1e-3 ? `h${hue}` : ''} ${fresh(id) ? 'fresh' : ''} ${busy.has(id) ? 'busy' : ''} ${dim(id) ? 'dim' : ''}`}
          style=${style(id)} onMouseEnter=${() => setHover({ kind: 'n', id })} onMouseLeave=${() => setHover(null)}>
          <g class="ld-grow"><circle r=${(design === 'b' ? 3 : s?.[4] ? 6.5 : 4.5) * scale} /><circle r=${(design === 'b' ? 5 : 12) * scale} class="ld-hit" /></g></g>`;
      })}</g>
      <g class="ld-cards">${[...order].reverse().map((id) => {
        const w = cardW(id) * scale, h = w * 0.75, hue = hueOf(id), img = imgOf(id), s = amount.get(id), nm = nameOf(id);
        return html`<g key=${`c${id}`} class=${`ld-card ${id === m.seed ? 'seed' : ''} ${hue != null ? `h${hue}` : ''} ${freshCard(id) ? 'fresh' : ''} ${busy.has(id) ? 'busy' : ''} ${dim(id) ? 'dim' : ''}`}
          style=${style(id)} onMouseEnter=${() => setHover({ kind: 'n', id })} onMouseLeave=${() => setHover(null)}>
          <g class="ld-grow">
            <rect x=${-w / 2} y=${-h / 2} width=${w} height=${h} rx=${6 * scale} class="ld-card-bg" />
            ${img ? html`<image href=${img} x=${-w / 2 + 3 * scale} y=${-h / 2 + 3 * scale} width=${w - 6 * scale} height=${h - 6 * scale} preserveAspectRatio="xMidYMid meet" />`
              : html`<text y=${5 * scale} class="ld-unknown">?</text>`}
            ${(nm || (id !== m.seed && groupFin(id) > 1e-4)) && html`<text y=${h / 2 + 15 * scale} class="ld-name" style=${`font-size:${12.5 * scale}px`}>${nm ? short(nm, design === 'b' ? 13 : 24) : ''}${id !== m.seed && groupFin(id) > 1e-4 ? html`<tspan class="ld-amt">${nm ? ' · ' : ''}${pct(groupFin(id))}</tspan>` : ''}</text>`}
          </g></g>`;
      })}</g>
      ${lane.map((p) => {
        const [x0, y0] = C(p.id) || [p.x, p.y], gx = p.x - 10;
        return html`<g class=${`ld-lane h${p.hue ?? 'x'}`} key=${`ln${p.key}`}>
          <path d=${`M${x0 + cardW(p.id) / 2},${y0} C${(x0 + gx) / 2 + 40},${y0} ${(x0 + gx) / 2},${p.y} ${gx},${p.y}`} class="ld-lane-link" />
          <rect x=${p.x} y=${p.y - 56} width="300" height="112" rx="8" class="ld-lane-bg" />
          <rect x=${p.x + 4} y=${p.y - 52} width="128" height="104" rx="6" class="ld-card-bg" />
          ${imgOf(p.id) && html`<image href=${imgOf(p.id)} x=${p.x + 6} y=${p.y - 48} width="124" height="96" preserveAspectRatio="xMidYMid meet" />`}
          <text x=${p.x + 144} y=${p.y - 18} class="ld-lane-pct">${pct(p.fin)}</text>
          <text x=${p.x + 144} y=${p.y + 2} class="ld-lane-sub">of the start, at the end</text>
          <text x=${p.x + 144} y=${p.y + 22} class="ld-lane-sub">${short(label(p.id), 22)}</text>
          <rect x=${p.x + 144} y=${p.y + 34} width="142" height="8" rx="4" class="ld-bar-bg" />
          <rect x=${p.x + 144} y=${p.y + 34} width=${Math.max(2, 142 * Math.min(1, p.fin))} height="8" rx="4" class="ld-bar" />
        </g>`;
      })}
    </svg>
    ${hovered && html`<${NodeCard} at=${C(hovered)} vb=${vb} s=${amount.get(hovered)} img=${imgOf(hovered, 320, 240)} name=${nameOf(hovered)}
      smiles=${tl.smiles[hovered]} first=${m.first.get(hovered)} seed=${hovered === m.seed} />`}
    ${hoveredEdge && html`<${EdgeCard} e=${hoveredEdge} at=${[(C(hoveredEdge.from)[0] + C(hoveredEdge.to)[0]) / 2, (C(hoveredEdge.from)[1] + C(hoveredEdge.to)[1]) / 2]} vb=${vb}
      nm=${label} cond=${cond} />`}
  </div>`;

  const strip = html`<${Strip} t=${f.t} tEnd=${m.frames[m.frames.length - 1].t} atEnd=${cur === m.frames.length - 1} running=${running} q=${tl.activity?.queue} counts=${f.counts} nSteps=${f.steps.length} products=${series}
    startLeft=${startLeft} steady=${f.t - steadySince} imgOf=${imgOf} nameOf=${label} cond=${cond} pending=${!f.steered} basis=${f.basis} sel=${tl.selectivity} />`;
  const keyEl = html`<${Key} cond=${cond} noFlux=${noFlux} design=${design} anyBelow=${edges.some((e) => e.below)} running=${running} />`;
  if (design === 'a') {
    return html`<div class="ld-page ld-page-a">${strip}<div class="ld-row">
      <div class="ld-col">${stage}${keyEl}${controls}</div>
      <aside class="ld-side"><${SidePanel} products=${series} imgOf=${imgOf} nameOf=${label} startLeft=${startLeft} q=${running ? tl.activity?.queue : null}
        recent=${seen.filter(fresh).slice(-6).reverse()} smiles=${tl.smiles} /></aside></div></div>`;
  }
  if (design === 'c') {
    const recent = seen.filter((id) => id !== m.seed).slice(-7).reverse();
    return html`<div class="ld-page ld-page-c">${strip}${stage}
      <div class="ld-ticker"><span class="ld-ticker-h">Just found</span>${recent.map((id) => html`<span class=${`ld-tick ${fresh(id) ? 'fresh' : ''}`} key=${id}
        onMouseEnter=${() => setHover({ kind: 'n', id })} onMouseLeave=${() => setHover(null)}>
        ${imgOf(id, 120, 90) ? html`<img src=${imgOf(id, 120, 90)} alt=${tl.smiles[id] || ''} />` : html`<i class="ld-tick-q">?</i>`}<span class="mono">${fmtClock(m.first.get(id))}</span></span>`)}</div>
      ${keyEl}${controls}</div>`;
  }
  return html`<div class="ld-page ld-page-b">${strip}${stage}${keyEl}${controls}</div>`;
}

// ---------------------------------------------------------------- pieces
function Spark({ series, w = 150, h = 34 }) {
  const t1 = Math.max(1, ...series.flatMap((s) => s.pts.map((p) => p[0])));
  return html`<svg class="ld-spark" width=${w} height=${h} viewBox=${`0 0 ${w} ${h}`} aria-label="Each leading product's amount over the run">
    <line x1="0" x2=${w} y1=${h - 1} y2=${h - 1} class="ld-spark-axis" />
    ${series.map((s) => html`<polyline class=${`ld-spark-line h${s.hue ?? 'x'}`} points=${s.pts.map(([t, v]) => `${(w * t) / t1},${h - 2 - (h - 4) * Math.min(1, v)}`).join(' ')} />`)}
  </svg>`;
}

function Strip({ t, tEnd, atEnd, running, q, counts, nSteps, products, startLeft, steady, imgOf, nameOf, cond, pending, basis, sel }) {
  const ph = running ? phaseOf(q) : null;
  return html`<div class="ld-strip" role="status">
    <div class="ld-s-time" title="Run time">
      <span class=${`ld-s-dot ${running ? 'live' : ''}`}></span>
      <b class="mono">${fmtClock(t)}</b>
      ${cond.budget && html`<span class="ld-s-budget" title=${`of a ${fmtClock(cond.budget)} time budget`}><i style=${`width:${Math.min(100, (100 * t) / cond.budget)}%`}></i></span>`}
    </div>
    <div class="ld-s-phase">
      <b>${ph ? ph.text : running ? (atEnd ? 'Starting' : 'Replaying the run so far') : atEnd ? 'Finished' : `Replay · ${fmtClock(t)} of ${fmtClock(tEnd)}`}</b>
      <span class="small muted">${ph ? ph.detail : ''}${ph ? '' : `${cond.T ? `${Math.round(cond.T)} K` : ''}${cond.tauKnown ? ` · ${fmtTime(cond.tau)} simulated` : ''}`}</span>
    </div>
    <div class="ld-s-counts"><span><b>${counts.species}</b> species</span><span><b>${counts.steps ?? nSteps}</b> steps</span></div>
    <div class="ld-s-products" title="What the start has turned into at the end of the simulated time, as a share of the start">
      ${products.map((p) => html`<span class=${`ld-s-prod h${p.hue ?? 'x'}`} title=${nameOf(p.id)}>
        ${imgOf(p.id, 120, 90) && html`<img src=${imgOf(p.id, 120, 90)} alt="" />`}<b>${pct(p.fin)}</b></span>`)}
      ${startLeft != null && html`<span class="ld-s-prod start" title="The start, left at the end of the simulated time"><span class="ld-s-pname">start</span><b>${pct(startLeft)}</b></span>`}
      ${products.length > 0 && html`<${Spark} series=${products} />`}
      <span class=${`small ld-s-steady ${pending ? 'pending' : ''}`} title="How long the leading product and its amount (within 2 points) have held, in run time">
        ${pending ? 'energies wait for DFT' : products.length ? html`answer held<br /><b>${steady >= 60 ? `${(steady / 60).toFixed(1)} min` : `${Math.round(steady)} s`}</b>` : 'nothing formed yet'}</span>
    </div>
  </div>`;
}

function SidePanel({ products, imgOf, nameOf, startLeft, q, recent, smiles }) {
  return html`<section><span class="section-title">What forms</span>
      <span class="small muted">at the end of the simulated time, as a share of the start</span>
      ${products.map((p) => html`<div class=${`ld-bar-row h${p.hue ?? 'x'}`}>
        ${imgOf(p.id, 120, 90) && html`<img src=${imgOf(p.id, 120, 90)} alt="" />`}
        <div><span>${short(nameOf(p.id) || smiles[p.id], 28)}</span><i><b style=${`width:${Math.max(1, 100 * Math.min(1, p.fin))}%`}></b></i></div>
        <b>${pct(p.fin)}</b></div>`)}
      ${startLeft != null && html`<div class="ld-bar-row start"><div><span>start left</span><i><b style=${`width:${Math.max(1, 100 * Math.min(1, startLeft))}%`}></b></i></div><b>${pct(startLeft)}</b></div>`}
    </section>
    ${q && html`<section><span class="section-title">Now</span>
      ${(q.running || []).slice(0, 5).map((r) => html`<div class="small ld-now"><i class="ld-now-dot"></i>${(PHASE[r.kind] || r.kind)} <span class="mono muted">${r.what.split(' · ')[0]}</span></div>`)}
      ${(q.running || []).length > 5 && html`<span class="small muted">and ${(q.running || []).length - 5} more</span>`}</section>`}
    ${recent.length > 0 && html`<section><span class="section-title">Just found</span>
      <div class="ld-recent">${recent.map((id) => html`<img src=${imgOf(id, 120, 90)} alt=${smiles[id] || ''} title=${smiles[id] || ''} />`)}</div></section>`}`;
}

function at(vb, p) { return [`${(100 * (p[0] - vb[0])) / vb[2]}%`, `${(100 * (p[1] - vb[1])) / vb[3]}%`]; }

function NodeCard({ at: p, vb, s, img, name, smiles, first, seed }) {
  const [l, t] = at(vb, p);
  return html`<div class="ld-pop" style=${`left:${l};top:${t}`}>
    ${img && html`<img src=${img} alt=${smiles || ''} />`}
    <div class="ld-pop-name">${name || (seed ? 'start' : '')}</div>
    <div class="mono small muted ld-pop-smi">${short(smiles || 'structure not perceived', 48)}</div>
    <div class="ld-pop-nums">
      ${s?.[1] != null && html`<span><b>${s[1] > 0 ? '+' : ''}${s[1].toFixed(1)}</b>kcal/mol (G)</span>`}
      ${!seed && html`<span><b>${pct(s?.[2] ?? 0)}</b>most at once</span><span><b>${pct(s?.[3] ?? 0)}</b>at the end</span>`}
    </div>
    <div class="small muted">${seed ? 'the start' : `found at ${fmtClock(first)}`}${s ? ` · ${s[5] ? 'DFT energy' : 'g-xTB energy'}` : ' · holds no material yet'}</div>
  </div>`;
}

function EdgeCard({ e, at: p, vb, nm, cond }) {
  const [l, t] = at(vb, p);
  const dG = e.barrier != null && e.back != null ? e.barrier - e.back : null;
  const hl = halfLife(e.barrier, cond.T);
  return html`<div class="ld-pop edge" style=${`left:${l};top:${t}`}>
    <div class="ld-pop-name">${nm(e.from)}${e.co?.with?.length ? ` + ${e.co.with.map(nm).join(' + ')}` : ''} <span class="muted">→</span> ${nm(e.to)}${e.co?.also?.length ? ` + ${e.co.also.map(nm).join(' + ')}` : ''}</div>
    ${e.co?.channels > 1 && html`<div class="small"><b>×${e.co.channels}</b> ${e.co.mirror ? 'mirror images, both counted' : 'channels, counted together'}</div>`}
    ${e.below ? html`<div class="small muted">TS below its ends: not this reaction${e.irc === 1 ? '' : ', unchecked'}. No barrier is drawn.</div>` : html`
    <div class="ld-pop-nums">
      <span><b>${e.barrier?.toFixed(1) ?? '?'}</b>barrier</span>
      ${e.back != null && html`<span><b>${e.back.toFixed(1)}</b>back</span>`}
      ${dG != null && html`<span><b>${dG > 0 ? '+' : ''}${dG.toFixed(1)}</b>ΔG</span>`}
      ${e.share != null && html`<span><b>${pct(e.share)}</b>of what leaves</span>`}
    </div>
    <div class="small">${hl != null ? html`Half-life at ${Math.round(cond.T)} K: <b>${fmtTime(hl)}</b>${cond.tauKnown ? ` (the kinetics simulate ${fmtTime(cond.tau)})` : ''}` : ''}</div>
    <div class="small muted">kcal/mol · ${e.src === 'd' ? 'DFT energies' : e.src === 'b' ? 'borrowed from a twin step' : 'g-xTB energies'}${e.irc === 1 ? ' · IRC joins these two ✓' : e.irc === -1 ? ' · no IRC check yet' : ''}</div>`}
  </div>`;
}

function Key({ cond, noFlux, anyBelow, running }) {
  const T = cond.T ? `${Math.round(cond.T)} K` : 'the run temperature';
  return html`<div class="ld-key small">
    <span class="ld-k"><svg width="78" height="12"><line x1="2" y1="6" x2="24" y2="6" class="ld-kl fast" /><line x1="28" y1="6" x2="50" y2="6" class="ld-kl run" /><line x1="54" y1="6" x2="76" y2="6" class="ld-kl slow" /></svg>
      how fast at ${T}: half-life under ${fmtTime(cond.tau / 100)} · under ${fmtTime(cond.tau)}${cond.tauKnown ? ' (the simulated time)' : ''} · longer</span>
    <span class="ld-k"><svg width="34" height="12"><line x1="2" y1="6" x2="24" y2="6" class="ld-kl main h0" /><path d="M24,1 L32,6 L24,11 z" class="ld-arrhead h0" /></svg>
      ${noFlux ? 'likely route to a product (lowest barriers; flows not recorded)' : 'material flows; thicker = more'}, in the product's colour</span>
    <span class="ld-k"><svg width="16" height="12"><circle cx="8" cy="6" r="4" class="ld-kdot" /></svg>a species found (hover for its structure)</span>
    <span class="ld-k"><svg width="30" height="12"><line x1="2" y1="6" x2="28" y2="6" class="ld-branch" /></svg>found from</span>
    ${anyBelow && html`<span class="ld-k"><svg width="30" height="12"><line x1="2" y1="6" x2="28" y2="6" class="ld-kl below" /></svg>TS below its ends: no barrier</span>`}
    ${running && html`<span class="ld-k"><svg width="30" height="12"><line x1="2" y1="6" x2="28" y2="6" class="ld-probe" /></svg>searched now</span>`}
  </div>`;
}

// ---------------------------------------------------------------- a reacting mixture, the same way
// A mixture (mixture.json) as a one-frame timeline of molecules: each reaction an arrow from its main
// reactant (the one that is not fed, else the first) to each product, its co-reactant written on it
// ("+ formic acid"), as wide as the amount it converted. A molecule is placed from the molecule it
// is made from (the reaction that makes the most of it); feed molecules sit at the start.
export function mixtureTimeline(mx, names) {
  const feed = new Set((mx.feed || []).map((f) => f.molecule));
  const fedTotal = (mx.feed || []).reduce((t, f) => t + (f.concentration || 0), 0) || 1;
  const mols = mx.molecules || [];
  const seed = (mx.feed || [])[0]?.molecule;
  const steps = [], co = {}, makes = new Map();
  (mx.reactions || []).forEach((r, k) => {
    const fwd = (r.net_extent || 0) >= 0;
    const ins = fwd ? r.reactants : r.products, outs = fwd ? r.products : r.reactants;
    const main = ins.find((x) => !feed.has(x)) || ins[0];
    const others = ins.filter((x, i) => i !== ins.indexOf(main));
    for (const out of [...new Set(outs)].filter((x) => !ins.includes(x) || x === main)) {
      if (out === main) continue;
      const id = `r${k}:${out}`;
      const bf = fwd ? r.barrier_kcal : r.reverse_barrier_kcal, br = fwd ? r.reverse_barrier_kcal : r.barrier_kcal;
      steps.push([id, main, out, bf, br, 'g', null, Math.abs(r.net_extent || 0) / fedTotal, 0]);
      co[id] = { with: others, also: outs.filter((x) => x !== out), channels: r.channels || 1, mirror: !!r.mirror_images, source: r.source };
      const prev = makes.get(out);
      if (!prev || Math.abs(r.net_extent || 0) > prev[1]) makes.set(out, [main, Math.abs(r.net_extent || 0)]);
    }
  });
  // order: the feed, then outward along what makes each
  const depth = new Map([...feed].map((x) => [x, 0]));
  for (let d = 0; d < 8; d += 1) for (const [out, [src]] of makes) if (!depth.has(out) && depth.get(src) === d) depth.set(out, d + 1);
  const explored = mols.map((x) => [x.molecule, depth.get(x.molecule) ?? 9, feed.has(x.molecule) ? (x.molecule === seed ? null : seed) : makes.get(x.molecule)?.[0] ?? seed]);
  const species = [...mols].sort((a, b) => (b.molecule === seed) - (a.molecule === seed) || (b.max_concentration || 0) - (a.max_concentration || 0))
    .map((x) => [x.molecule, x.g_kcal != null ? null : null, (x.max_concentration || 0) / fedTotal, (x.final_concentration || 0) / fedTotal,
      (x.max_concentration || 0) / fedTotal >= (mx.settings?.threshold ?? 0.01) ? 1 : 0, 0]);
  const T = mx.settings?.temperature_K, tau = mx.settings?.time_s;
  const frame = { t: 9, species, steps, gone: [], counts: { species: mols.length, steps: (mx.reactions || []).length }, steered: true,
    basis: `free energy (${T} K, ${mx.settings?.standard_state || ''})` };
  const smiles = Object.fromEntries(mols.map((x) => [x.molecule, x.molecule.split('|')[0]]));
  return { frames: [frame], smiles, names, explored, co, mixture: { feed: [...feed], tau, seeds: mx.seeds || [], generation: mx.generation },
    command: tau != null ? `--time ${tau}` : '' };
}

export function MixtureDesign({ mixture, names, design, jobId, job, running }) {
  const tl = useMemo(() => mixtureTimeline(mixture, names), [JSON.stringify(mixture)]);
  const seeds = tl.mixture.seeds;
  const done = seeds.filter((x) => String(x.status || '').startsWith('done')).length;
  const controls = html`<div class="ld-seeds small muted">${seeds.length} seed run${seeds.length === 1 ? '' : 's'} (${done} done) · generation ${tl.mixture.generation ?? 0}:
    each seed is one combination of molecules explored on its own; its steps join this network. Amounts as a share of the feed (${(mixture.feed || []).map((f) => `${f.name || names[f.molecule] || f.molecule} ${f.concentration} ${mixture.settings?.standard_state === '1atm' ? 'atm' : 'M'}`).join(', ')}).</div>`;
  return html`<${DesignNetwork} tl=${tl} cur=${0} design=${design} jobId=${`mix:${jobId}`} job=${job} running=${running} follow=${true} controls=${controls} />`;
}
