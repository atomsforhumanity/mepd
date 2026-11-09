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
import { depictUrl, steeringOf } from '../util.js';
import { Popover, pointBox, useHoverCard } from './Popover.js';
import { EventPlayer } from './ReactorLive.js';
import { api } from '../api.js';
import { openJob } from '../store.js';

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
const schemeX0 = (c) => (c === 0 ? 0 : 150 + (c - 1) * COL_W);
function useScheme(key, disc, order, relevantOrder) {
  const ref = useRef(null);
  if (!ref.current || ref.current.key !== key) ref.current = { key, row: new Map(), pile: new Map(), cells: new Map(), piles: new Map() };
  const sc = ref.current;
  const col = (id) => Math.min(disc.hop.get(id) ?? 1, MAX_COL);
  const x0 = schemeX0;
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
export function DesignNetwork({ tl, cur, design, jobId, example, job, running, follow, controls, apiJob = jobId, eventClock = null, onEvents = null }) {
  const m = useMemo(() => expansionModel(tl, cur), [tl, cur]);
  // the card under the pointer, focused, or pinned by a tap (Popover.js): a species or a step
  const hc = useHoverCard();
  const hover = hc.cur ? { kind: hc.cur.key[0], id: hc.cur.data } : null;
  const key = `${jobId || example}|${design}`;
  const st = usePlacement(design, key, m.order, anchorFor(m));
  const f = m.f;
  const cond = conditionsOf(f.basis, tl.command ?? job?.command);
  const hues = useMemo(() => productHues(m.frames, m.seed, m.keyOf, cur), [tl, cur]);
  const hueOf = (id) => (id === m.seed ? null : hues.get(m.keyOf(id)));
  // the reactor's events (the overlay), the one opened in full, and species that just caught one
  const rx = useReactorEvents(apiJob, running);
  useEffect(() => { if (onEvents && rx.events.length) onEvents(rx.events); }, [rx.events.length]);
  const [dialog, setDialog] = useState(null);
  const [caught, setCaught] = useState({});
  const followLive = running && cur === m.frames.length - 1;
  const nowOf = eventClock || (() => (followLive && job?.started ? Date.now() / 1000 - job.started : f.t));
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
  else {   // room around the start for the reactor's events
    const [sx, sy] = C(m.seed), r = 240 * scale;
    box = [Math.min(box[0], sx - r), Math.min(box[1], sy - r), Math.max(box[2], sx + r), Math.max(box[3], sy + r)];
  }
  // at least 700 x 460, the content in the middle
  const bw = Math.max(700, box[2] - box[0]), bh = Math.max(460, box[3] - box[1]);
  const vb = useViewBox([(box[0] + box[2]) / 2 - bw / 2, (box[1] + box[3]) / 2 - bh / 2, bw, bh], follow);

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

  const tNow = nowOf();
  const madeBy = new Map();
  for (const e of rx.events) if (e.prod != null && e.t <= tNow + 0.5) { if (!madeBy.has(e.prod)) madeBy.set(e.prod, []); madeBy.get(e.prod).push(e); }
  const openEvent = (ev) => { hc.close(); setDialog(ev); };
  const lastTen = rx.events.reduce((n, e) => n + (e.t <= tNow + 0.5 && e.t > tNow - WINDOW_S ? 1 : 0), 0);
  const seenSet = new Set(seen);
  const avoid = order.map((id) => { const [x, y] = C(id), w = cardW(id) * scale; return [x, y + 8 * scale, w + 16 * scale, w * 0.75 + 40 * scale]; });
  const settleOf = (ev, at) => {
    if (design === 'b') { const c = Math.min((st.hop.get(ev.src ?? m.seed) ?? 0) + 1, MAX_COL); return [schemeX0(c) + (PILE_W / 2) * PILE, -70]; }
    const o = C(m.seed), d = Math.max(1, Math.hypot(at[0] - o[0], at[1] - o[1]));
    return [at[0] + ((at[0] - o[0]) / d) * 60 * scale, at[1] + ((at[1] - o[1]) / d) * 60 * scale];
  };
  const onCatch = (id) => setCaught((c) => ({ ...c, [id]: performance.now() }));
  const isCaught = (id) => caught[id] && performance.now() - caught[id] < 1200;
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
          ...${hc.bind(`e:${e.id}`, e.id, (el, ev) => { const r = el.getBoundingClientRect(); return ev ? pointBox(ev.clientX, ev.clientY) : pointBox(r.left + r.width / 2, r.top + r.height / 2); })}>
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
        return html`<g key=${`n${id}`} class=${`ld-dot ${isCaught(id) ? 'caught' : ''} ${s?.[4] ? 'pop' : ''} ${hue != null && s?.[3] > 1e-3 ? `h${hue}` : ''} ${fresh(id) ? 'fresh' : ''} ${busy.has(id) ? 'busy' : ''} ${dim(id) ? 'dim' : ''}`}
          style=${style(id)} ...${hc.bind(`n:${id}`, id, (el) => el.querySelector('circle, .ld-card-bg').getBoundingClientRect())}>
          <g class="ld-grow"><circle r=${(design === 'b' ? 3 : s?.[4] ? 6.5 : 4.5) * scale} /><circle r=${(design === 'b' ? 5 : 12) * scale} class="ld-hit" /></g></g>`;
      })}</g>
      ${rx.events.length > 0 && html`<${ReactorOverlay} apiJob=${apiJob} events=${rx.events} now=${nowOf} live=${followLive || !!eventClock}
        posOf=${(id) => (seenSet.has(id) ? C(id) : null)} seenAt=${(id, t) => (m.first.get(id) ?? Infinity) <= t + 0.5}
        avoid=${avoid} settleOf=${(ev, at) => settleOf(ev, at)} scale=${scale} onOpen=${openEvent} onCatch=${onCatch} />`}
      <g class="ld-cards">${[...order].reverse().map((id) => {
        const w = cardW(id) * scale, h = w * 0.75, hue = hueOf(id), img = imgOf(id), s = amount.get(id), nm = nameOf(id);
        return html`<g key=${`c${id}`} class=${`ld-card ${isCaught(id) ? 'caught' : ''} ${id === m.seed ? 'seed' : ''} ${hue != null ? `h${hue}` : ''} ${freshCard(id) ? 'fresh' : ''} ${busy.has(id) ? 'busy' : ''} ${dim(id) ? 'dim' : ''}`}
          style=${style(id)} ...${hc.bind(`n:${id}`, id, (el) => el.querySelector('circle, .ld-card-bg').getBoundingClientRect())}>
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
    ${hovered && html`<${Popover} anchor=${hc.cur.anchor} className="ld-pop" onEnter=${hc.card.onEnter} onLeave=${hc.card.onLeave}>
      <${NodeCard} s=${amount.get(hovered)} img=${imgOf(hovered, 320, 240)} name=${nameOf(hovered)}
        smiles=${tl.smiles[hovered]} first=${m.first.get(hovered)} seed=${hovered === m.seed} made=${madeBy.get(hovered) || []} onEvent=${openEvent} /><//>`}
    ${hoveredEdge && html`<${Popover} anchor=${hc.cur.anchor} className="ld-pop edge" onEnter=${hc.card.onEnter} onLeave=${hc.card.onLeave}>
      <${EdgeCard} e=${hoveredEdge} nm=${label} cond=${cond} /><//>`}
    ${dialog && html`<${EventDialog} apiJob=${apiJob} ev=${dialog} nameOf=${label} onClose=${() => setDialog(null)} />`}
  </div>`;

  const strip = html`<${Strip} t=${f.t} tEnd=${m.frames[m.frames.length - 1].t} atEnd=${cur === m.frames.length - 1} running=${running} q=${tl.activity?.queue} counts=${f.counts} nSteps=${f.steps.length} products=${series}
    startLeft=${startLeft} steady=${f.t - steadySince} imgOf=${imgOf} nameOf=${label} cond=${cond} pending=${!f.steered} basis=${f.basis} sel=${tl.selectivity}
    live0=${followLive && job?.started ? job.started : null} steering=${steeringOf(job)} reactor=${rx.runs ? { running: rx.running, runs: rx.runs, recent: lastTen, events: rx.events.filter((e) => e.t <= tNow + 0.5).length,
      open: () => openJob(apiJob, apiJob === jobId ? 'reactor' : 'live') } : null} />`;
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
        ...${hc.bind(`n:${id}`, id)}>
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

// Run time that ticks while the run is followed live (frames come only with each kinetics refresh).
function LiveClock({ t0 }) {
  const [, tick] = useState(0);
  useEffect(() => { const id = setInterval(() => tick((x) => x + 1), 1000); return () => clearInterval(id); }, []);
  return fmtClock(Math.max(0, Date.now() / 1000 - t0));
}

function Strip({ t, tEnd, atEnd, running, q, counts, nSteps, products, startLeft, steady, imgOf, nameOf, cond, pending, basis, sel, steering, reactor, live0 = null }) {
  const ph = running ? phaseOf(q) : null;
  return html`<div class="ld-strip" role="status">
    <div class="ld-s-time" title="Run time">
      <span class=${`ld-s-dot ${running ? 'live' : ''}`}></span>
      <b class="mono">${live0 != null ? html`<${LiveClock} t0=${live0} />` : fmtClock(t)}</b>
      ${cond.budget && html`<span class="ld-s-budget" title=${`of a ${fmtClock(cond.budget)} time budget`}><i style=${`width:${Math.min(100, (100 * t) / cond.budget)}%`}></i></span>`}
    </div>
    <div class="ld-s-phase">
      <b>${ph ? ph.text : running ? (atEnd ? 'Starting' : 'Replaying the run so far') : atEnd ? 'Finished' : `Replay · ${fmtClock(t)} of ${fmtClock(tEnd)}`}</b>
      <span class="small muted">${ph ? ph.detail : ''}${ph ? '' : `${cond.T ? `${Math.round(cond.T)} K` : ''}${cond.tauKnown ? ` · ${fmtTime(cond.tau)} simulated` : ''}`}</span>
    </div>
    <div class="ld-s-counts"><span><b>${counts.species}</b> species</span><span><b>${counts.steps ?? nSteps}</b> steps</span></div>
    ${reactor && html`<button class=${`ld-s-reactor ${reactor.running ? 'on' : ''}`} onClick=${reactor.open}
        title="The reactor's molecular dynamics that proposes the reactions: open it">
      <b>Reactor MD ›</b><span class="small muted">${reactor.running ? `${reactor.running} trajector${reactor.running === 1 ? 'y' : 'ies'} running · ` : ''}${reactor.recent
        ? `${reactor.recent} event${reactor.recent === 1 ? '' : 's'} in the last ${WINDOW_S} s` : `${reactor.events} events`}</span></button>`}
    <div class="ld-s-products" title="What the start has turned into at the end of the simulated time, as a share of the start">
      ${products.map((p) => html`<span class=${`ld-s-prod h${p.hue ?? 'x'}`} title=${nameOf(p.id)}>
        ${imgOf(p.id, 120, 90) && html`<img src=${imgOf(p.id, 120, 90)} alt="" />`}<b>${pct(p.fin)}</b></span>`)}
      ${startLeft != null && html`<span class="ld-s-prod start" title="The start, left at the end of the simulated time"><span class="ld-s-pname">start</span><b>${pct(startLeft)}</b></span>`}
      ${products.length > 0 && html`<${Spark} series=${products} />`}
      <span class=${`small ld-s-steady ${pending ? 'pending' : ''}`} title="How long the leading product and its amount (within 2 points) have held, in run time">
        ${pending ? 'energies wait for DFT' : products.length ? html`answer held<br /><b>${steady >= 60 ? `${(steady / 60).toFixed(1)} min` : `${Math.round(steady)} s`}</b>` : 'nothing formed yet'}
        ${steering && html`<br /><span title="The level the live kinetics' energies are steered to (DFT single points)">energies: ${steering}</span>`}</span>
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

function NodeCard({ s, img, name, smiles, first, seed, made = [], onEvent }) {
  return html`${img && html`<img src=${img} alt=${smiles || ''} />`}
    <div class="ld-pop-name">${name || (seed ? 'start' : '')}</div>
    <div class="mono small muted ld-pop-smi">${short(smiles || 'structure not perceived', 48)}</div>
    <div class="ld-pop-nums">
      ${s?.[1] != null && html`<span><b>${s[1] > 0 ? '+' : ''}${s[1].toFixed(1)}</b>kcal/mol (G)</span>`}
      ${!seed && html`<span><b>${pct(s?.[2] ?? 0)}</b>most at once</span><span><b>${pct(s?.[3] ?? 0)}</b>at the end</span>`}
    </div>
    <div class="small muted">${seed ? 'the start' : `found at ${fmtClock(first)}`}${s ? ` · ${s[5] ? 'DFT energy' : 'g-xTB energy'}` : ' · holds no material yet'}</div>
    ${made.length > 0 && html`<div class="ld-made small"><span>Made in the reactor's MD by ${made.length} event${made.length === 1 ? '' : 's'}:</span>
      ${made.slice(-4).reverse().map((ev) => html`<button class="btn small" onClick=${() => onEvent(ev)} title=${ev.label}>▶ ${fmtClock(ev.t)}</button>`)}</div>`}`;
}

function EdgeCard({ e, nm, cond }) {
  const dG = e.barrier != null && e.back != null ? e.barrier - e.back : null;
  const hl = halfLife(e.barrier, cond.T);
  return html`<div class="ld-pop-name">${nm(e.from)}${e.co?.with?.length ? ` + ${e.co.with.map(nm).join(' + ')}` : ''} <span class="muted">→</span> ${nm(e.to)}${e.co?.also?.length ? ` + ${e.co.also.map(nm).join(' + ')}` : ''}</div>
    ${e.co?.channels > 1 && html`<div class="small"><b>×${e.co.channels}</b> ${e.co.mirror ? 'mirror images, both counted' : 'channels, counted together'}</div>`}
    ${e.below ? html`<div class="small muted">TS below its ends: not this reaction${e.irc === 1 ? '' : ', unchecked'}. No barrier is drawn.</div>` : html`
    <div class="ld-pop-nums">
      <span><b>${e.barrier?.toFixed(1) ?? '?'}</b>barrier</span>
      ${e.back != null && html`<span><b>${e.back.toFixed(1)}</b>back</span>`}
      ${dG != null && html`<span><b>${dG > 0 ? '+' : ''}${dG.toFixed(1)}</b>ΔG</span>`}
      ${e.share != null && html`<span><b>${pct(e.share)}</b>of what leaves</span>`}
    </div>
    <div class="small">${hl != null ? html`Half-life at ${Math.round(cond.T)} K: <b>${fmtTime(hl)}</b>${cond.tauKnown ? ` (the kinetics simulate ${fmtTime(cond.tau)})` : ''}` : ''}</div>
    <div class="small muted">kcal/mol · ${e.src === 'd' ? 'DFT energies' : e.src === 'b' ? 'borrowed from a twin step' : 'g-xTB energies'}${e.irc === 1 ? ' · IRC joins these two ✓' : e.irc === -1 ? ' · no IRC check yet' : ''}</div>`}`;
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
  // A finished mixture has no timeline: its reactor events replay at their own pace, over and over
  // (the strip and the dialog say when each happened); a running one shows them as they come.
  const t0 = useRef(performance.now());
  const rxEnd = useRef(1);
  const eventClock = running ? null : () => ((performance.now() - t0.current) / 1000) % rxEnd.current;
  const seeds = tl.mixture.seeds;
  const done = seeds.filter((x) => String(x.status || '').startsWith('done')).length;
  const controls = html`<div class="ld-seeds small muted">${seeds.length} seed run${seeds.length === 1 ? '' : 's'} (${done} done) · generation ${tl.mixture.generation ?? 0}:
    each seed is one combination of molecules explored on its own; its steps join this network. Amounts as a share of the feed (${(mixture.feed || []).map((f) => `${f.name || names[f.molecule] || f.molecule} ${f.concentration} ${mixture.settings?.standard_state === '1atm' ? 'atm' : 'M'}`).join(', ')}).</div>`;
  return html`<${DesignNetwork} tl=${tl} cur=${0} design=${design} jobId=${`mix:${jobId}`} apiJob=${jobId} job=${job} running=${running} follow=${true}
    controls=${controls} eventClock=${eventClock} onEvents=${(evs) => { rxEnd.current = Math.max(1, ...evs.map((e) => e.t)) + 6; }} />`;
}

// ---------------------------------------------------------------- the reactor's events, overlaid
// While a propose job's MD runs, the reactor finds reaction events (its live_events.json, per MD
// segment). Each shows faintly as a ghost of its subsystem's movie (the atoms that react, the few
// frames around the bond change, projected flat) at the species whose MD it was: it fades in, plays
// once, and drifts to the species it became when that is found (which pulses as it catches it), else
// settles away and fades. At most GHOSTS at once; the rest are counted. Click one for the full movie
// (the Reactor view's event player). Reduced motion: the latest one, still.
const GHOSTS = 3, LIFE_MS = 5200, WINDOW_S = 10;

export function useReactorEvents(apiJob, live) {
  const [st, setSt] = useState({ events: [], epoch: null, running: 0, runs: 0 });
  const ref = useRef(st);
  ref.current = st;
  useEffect(() => {
    if (!apiJob) return undefined;
    let alive = true, timer = null;
    const load = async () => {
      const cur = ref.current;
      try {
        const d = await api.get(`/api/jobs/${encodeURIComponent(apiJob)}/reactor-events?since=${cur.events.length}`);
        if (!alive) return;
        if (cur.epoch && d.epoch !== cur.epoch) {   // the server restarted: its numbering too
          ref.current = { events: [], epoch: d.epoch, running: d.running, runs: d.runs };
          setSt(ref.current);
          timer = setTimeout(load, 50);
          return;
        }
        let events = cur.events;
        const prods = Object.entries(d.prods || {}).filter(([i, pr]) => events[i] && events[i].prod !== pr);
        const srcs = Object.entries(d.srcs || {}).filter(([i, sr]) => events[i] && events[i].src !== sr);
        if (d.events.length || prods.length || srcs.length) {
          events = [...events];
          for (const [i, pr] of prods) events[i] = { ...events[i], prod: pr };
          for (const [i, sr] of srcs) events[i] = { ...events[i], src: sr };
          events.push(...d.events);
        }
        ref.current = { events, epoch: d.epoch, running: d.running, runs: d.runs };
        setSt(ref.current);
        if (d.events.length >= 4000) timer = setTimeout(load, 50);   // catching up
        else if (live) timer = setTimeout(load, 2000);
      } catch {
        if (alive && live) timer = setTimeout(load, 4000);
      }
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, [apiJob, live]);
  return st;
}

const COV = { H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, P: 1.07, S: 1.05, Cl: 1.02, Br: 1.2, B: 0.84, Si: 1.11 };
// An event's movie, small: ~14 frames from just before its reactant frame to just after its product
// frame, flattened onto the plane its atoms spread most in, fitted to a unit circle; bonds per frame.
function ghostFrames(x) {
  const n = x.symbols.length, F = x.frames;
  const a = Math.max(0, x.reactant_frame - 6), b = Math.min(F.length - 1, x.product_frame + 6);
  const idx = [];
  for (let k = 0; k < 14; k += 1) idx.push(Math.round(a + ((b - a) * k) / 13));
  const ref = F[Math.max(0, Math.min(F.length - 1, x.reactant_frame))];
  const c = [0, 1, 2].map((d) => { let t = 0; for (let i = 0; i < n; i += 1) t += ref[3 * i + d]; return t / n; });
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i += 1) for (let p = 0; p < 3; p += 1) for (let q = 0; q < 3; q += 1) cov[p][q] += (ref[3 * i + p] - c[p]) * (ref[3 * i + q] - c[q]);
  const top = (M, skip) => {   // power iteration, deflated against `skip`
    let v = [1, 0.7, 0.3];
    for (let it = 0; it < 40; it += 1) {
      let w = [0, 1, 2].map((p) => M[p][0] * v[0] + M[p][1] * v[1] + M[p][2] * v[2]);
      if (skip) { const d = w[0] * skip[0] + w[1] * skip[1] + w[2] * skip[2]; w = w.map((u, k) => u - d * skip[k]); }
      const l = Math.hypot(...w) || 1;
      v = w.map((u) => u / l);
    }
    return v;
  };
  const e1 = top(cov), e2 = top(cov, e1);
  const changes = new Map((x.changes || []).map((ch) => [`${Math.min(...ch.atoms)}-${Math.max(...ch.atoms)}`, ch.formed]));
  let ext = 0.5;
  const frames = idx.map((k) => {
    const f = F[k], pts = [];
    for (let i = 0; i < n; i += 1) {
      const d = [f[3 * i] - c[0], f[3 * i + 1] - c[1], f[3 * i + 2] - c[2]];
      const pnt = [d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2], d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2]];
      ext = Math.max(ext, Math.abs(pnt[0]), Math.abs(pnt[1]));
      pts.push(pnt);
    }
    const bonds = [];
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const dd = Math.hypot(f[3 * i] - f[3 * j], f[3 * i + 1] - f[3 * j + 1], f[3 * i + 2] - f[3 * j + 2]);
        const key = `${i}-${j}`;
        if (dd < 1.15 * ((COV[x.symbols[i]] ?? 0.8) + (COV[x.symbols[j]] ?? 0.8))) bonds.push([i, j, changes.has(key) ? (changes.get(key) ? 'f' : 'b') : '']);
      }
    }
    return { pts, bonds };
  });
  for (const fr of frames) fr.pts = fr.pts.map(([u, v]) => [u / ext, v / ext]);
  return { sym: x.symbols, frames };
}

const EVENT_CACHE = new Map();
function useGhostFrames(apiJob, ev) {
  const key = `${apiJob}|${ev.seed || ''}|${ev.run}|${ev.k}`;
  const [d, setD] = useState(EVENT_CACHE.get(key) || null);
  useEffect(() => {
    if (EVENT_CACHE.has(key)) { setD(EVENT_CACHE.get(key)); return undefined; }
    let alive = true;
    api.get(`/api/jobs/${encodeURIComponent(apiJob)}/reactor/events/${ev.k}?run=${ev.run}${ev.seed ? `&seed=${encodeURIComponent(ev.seed)}` : ''}`)
      .then((x) => {
        const g = ghostFrames(x);
        EVENT_CACHE.set(key, g);
        if (EVENT_CACHE.size > 60) EVENT_CACHE.delete(EVENT_CACHE.keys().next().value);
        if (alive) setD(g);
      }).catch(() => {});
    return () => { alive = false; };
  }, [key]);
  return d;
}

// One ghost's drawing: a flat ball-and-stick of frame k (bonds that form or break in the event marked).
function GhostDrawing({ g, k, r }) {
  const fr = g.frames[Math.min(k, g.frames.length - 1)];
  return html`<g class="ld-gmol">
    ${fr.bonds.map(([i, j, kind]) => html`<line x1=${fr.pts[i][0] * r} y1=${fr.pts[i][1] * r} x2=${fr.pts[j][0] * r} y2=${fr.pts[j][1] * r} class=${`ld-gb ${kind}`} />`)}
    ${fr.pts.map(([u, v], i) => html`<circle cx=${u * r} cy=${v * r} r=${(g.sym[i] === 'H' ? 0.055 : 0.09) * r} class=${`ld-ga el-${COV[g.sym[i]] != null ? g.sym[i] : 'x'}`} />`)}
  </g>`;
}

function Ghost({ apiJob, gh, scale, still, onOpen }) {
  const g = useGhostFrames(apiJob, gh.ev);
  const [k, setK] = useState(still ? 13 : 0);
  useEffect(() => {
    if (still || !g) return undefined;
    const id = setInterval(() => setK((x) => (x >= 13 ? 13 : x + 1)), 140);
    return () => clearInterval(id);
  }, [!!g, still]);
  if (!g) return null;
  const r = 30 * scale;
  return html`<g class=${`ld-ghost ${gh.phase} ${still ? 'still' : ''}`} style=${`transform:translate(${gh.at[0]}px,${gh.at[1]}px)`}
      onClick=${() => onOpen(gh.ev)} role="button" tabindex="0" aria-label=${`Reactor event: ${gh.ev.label}. Open its movie`}
      onKeyDown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(gh.ev); } }}>
    <g class="ld-ghost-in"><circle r=${r * 1.25} class="ld-ghost-bg" /><${GhostDrawing} g=${g} k=${k} r=${r} /></g>
    <title>${gh.ev.label} (reactor event; click for its movie)</title>
  </g>`;
}

// Where a ghost appears: beside the species it started from, clear of the cards (products holding
// material) and of the other ghosts.
function ghostSpot(src, avoid, scale, k) {
  const r = 42 * scale;
  for (const d of [70, 100, 135, 175].map((x) => x * scale)) {
    for (let j = 0; j < 12; j += 1) {
      const th = -Math.PI / 3 + ((j + k * 5) % 12) * (Math.PI / 6);
      const x = src[0] + d * Math.cos(th), y = src[1] + d * Math.sin(th);
      if (avoid.every(([ax, ay, aw, ah]) => x + r < ax - aw / 2 || x - r > ax + aw / 2 || y + r < ay - ah / 2 || y - r > ay + ah / 2)) return [x, y];
    }
  }
  return [src[0] + 80 * scale, src[1] - 80 * scale];
}

export function ReactorOverlay({ apiJob, events, now, live, posOf, seenAt, avoid, settleOf, scale, onOpen, onCatch }) {
  const still = reduced();
  const [ghosts, setGhosts] = useState([]);
  const shown = useRef(new Set());
  const st = useRef({});
  st.current = { events, now, live, posOf, seenAt, avoid, settleOf, scale, onCatch };
  useEffect(() => {
    if (still) return undefined;
    const tick = () => {
      const { events: evs, now: nowF, posOf: pos, seenAt: seen, avoid: av, settleOf: settle, scale: sc, onCatch: caught } = st.current;
      const t = nowF(), wall = performance.now();
      setGhosts((gs) => {
        let out = gs.filter((x) => wall - x.born < LIFE_MS);
        out = out.map((x) => {
          const age = wall - x.born;
          if (age > 2600 && x.phase === 'play') {
            const target = x.ev.prod != null && pos(x.ev.prod) && seen(x.ev.prod, t) ? pos(x.ev.prod) : null;
            if (target) caught?.(x.ev.prod);
            return { ...x, phase: target ? 'catch' : 'settle', at: target || settle(x.ev, x.at) || [x.at[0], x.at[1] - 40 * sc] };
          }
          return x;
        });
        if (out.length < GHOSTS) {
          const recent = [];
          for (let i = evs.length - 1; i >= 0 && recent.length < 400; i -= 1) {
            const e = evs[i];
            if (e.t > t + 0.5) continue;
            if (e.t < t - WINDOW_S) { if (st.current.live) break; continue; }
            if (!shown.current.has(e.i) && pos(e.src ?? '0')) recent.push(e);
          }
          // the newest first, those whose product is found before the rest
          recent.sort((a, b) => (b.prod != null) - (a.prod != null) || b.t - a.t);
          for (const e of recent.slice(0, GHOSTS - out.length)) {
            shown.current.add(e.i);
            const src = pos(e.src ?? '0');
            const at = ghostSpot(src, [...av, ...out.map((x) => [x.at[0], x.at[1], 84 * sc, 84 * sc])], sc, e.i);
            out.push({ ev: e, born: wall + 120 * out.length, phase: 'play', at, from: at });
          }
        }
        return out.length === gs.length && out.every((x, i) => x === gs[i]) ? gs : out;
      });
    };
    tick();
    const id = setInterval(tick, 400);
    return () => clearInterval(id);
  }, [still]);
  if (still) {   // reduced motion: the latest event, still, beside its species
    const t = now();
    const e = [...events].reverse().find((x) => x.t <= t + 0.5 && posOf(x.src ?? '0'));
    if (!e) return null;
    const at = ghostSpot(posOf(e.src ?? '0'), avoid, scale, 0);
    return html`<g class="ld-ghosts"><${Ghost} key=${e.i} apiJob=${apiJob} gh=${{ ev: e, at, phase: 'play' }} scale=${scale} still=${true} onOpen=${onOpen} /></g>`;
  }
  return html`<g class="ld-ghosts">${ghosts.map((gh) => html`<${Ghost} key=${gh.ev.i} apiJob=${apiJob} gh=${gh} scale=${scale} onOpen=${onOpen} />`)}</g>`;
}

// The full movie of one event: the Reactor view's event player, in a dialog over the page.
export function EventDialog({ apiJob, ev, nameOf, onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); onClose(); } };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, []);
  return html`<div class="ld-dialog-back" onClick=${onClose}>
    <div class="ld-dialog" role="dialog" aria-label="Reactor event" onClick=${(e) => e.stopPropagation()}>
      <div class="ld-dialog-head"><b>${ev.label}</b><span class="spacer"></span>
        <button class="btn small" onClick=${() => { onClose(); openJob(apiJob, 'reactor'); }}>Reactor MD ›</button>
        <button class="btn small" onClick=${onClose} aria-label="Close">✕</button></div>
      <div class="small muted">A reaction event in the reactor's MD at ${fmtClock(ev.t)} run time${ev.seed ? ` (seed ${ev.seed})` : ''}, ${ev.atoms} atoms
        · from ${nameOf(ev.src ?? '0')}${ev.prod != null ? ` · became ${nameOf(ev.prod)} (found and minimized)` : ' · not (yet) matched to a species'}</div>
      <${EventPlayer} jobId=${apiJob} event=${ev.k} run=${+ev.run} seed=${ev.seed} height=${360} />
    </div></div>`;
}
