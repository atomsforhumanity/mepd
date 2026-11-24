// The live network (#/network/<job>, #/network/example/<slug>) and a reacting mixture's network, drawn
// as a reaction scheme that grows while the run works (mepd/web/live_network.py):
// - the network grown from the start, in the middle: each species found beside the one it was found from,
//   tied to it by a thread, packed into the stage's shape; a species gets its structure once it matters
//   (holds 2 % of the largest amount), where it stands, and the routes from the start to the products;
// - beside it, what forms, per molecule (the five leading molecules and how much of each forms per starting
//   box at the end of the simulated time; the server counts each molecule over the box's states, so 2 H2 in
//   a box count twice), and a benchmark's measured selectivity; or, toggled, the outcomes as whole mixtures
//   (the box's states, as shares of the starting boxes);
// - one bar on top: run time, what the explorer does now, counts, how long the answer has held, a door to
//   the reactor's MD;
// - steps coloured by speed at the run's temperature (half-life against the simulated time), routes in
//   their product's colour; magenta is what runs now: searches marching between two species, arrivals,
//   a step just found; the reactor's events as faint ghosts of their movies, caught by the species
//   they verifiably made, and on opening the page those since you last looked, all at once;
// - nothing placed moves but to make room for a structure, so a replay grows in place. Hover, focus or tap
//   for details (Popover.js); a table view lists every species.
import { Component, html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { depictUrl } from '../util.js';
import { Popover, pointBox, useHoverCard } from './Popover.js';
import { EventPlayer } from './ReactorLive.js';
import { api } from '../api.js';
import { openJob, update } from '../store.js';

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
// An amount per starting box (or per mol of feed), never a percentage: a molecule can form more than once.
const amt = (x) => (x == null ? '—' : x >= 0.01 ? x.toFixed(2) : x >= 0.001 ? x.toFixed(3) : x > 0 ? '< 0.001' : '0');
const amtFine = (x) => (x != null && x > 0 && x < 0.001 ? x.toExponential(1) : amt(x));   // (tables)
// What the amounts are counted per: the start's one molecule, a pair, a box of several, or the feed.
function perText(tl) {
  if (tl.mixture) return 'mol per mol of feed';
  const start = tl.start || [];
  if (start.length === 1) return `per ${tl.molecules?.[start[0]]?.name || 'starting molecule'}`;
  return start.length === 2 ? 'per starting pair' : `per starting box of ${start.length} molecules`;
}
const LANE_N = 5;

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

// ---------------------------------------------------------------- the scheme's places
// The network grown from the start: the start in the middle, each species found beside the one it was found
// from, at the free spot nearest it that keeps the whole compact, in an ellipse the stage's shape, so the network
// fills the stage at every moment. It grows frame by frame, as the run went: a species that comes to hold
// material, or lies on a route to one, becomes a structure there (beside a structure it reacts with nearest the
// start, clear of the others) and the dots under it step aside. Every place is kept with the time it was taken
// (a ref), so a replay, scrubbed either way, shows each moment as it was, and nothing moves but to make room.
const DOT = 9, CARD_W = 104, SEED_W = 128, CELL = 48;
const RECT = { seed: [SEED_W / 2 + 14, 62, 78], card: [CARD_W / 2 + 12, 50, 80], dot: [DOT, DOT, DOT] };   // half width, above, below
const rectAt = (kind, x, y) => { const [w, u, d] = RECT[kind]; return [x - w, y - u, x + w, y + d]; };
function cellsOf(r) {
  const out = [];
  for (let i = Math.floor(r[0] / CELL); i <= Math.floor(r[2] / CELL); i += 1) for (let j = Math.floor(r[1] / CELL); j <= Math.floor(r[3] / CELL); j += 1) out.push(`${i}:${j}`);
  return out;
}

function useScheme(key) {
  const ref = useRef(null);
  if (!ref.current || ref.current.key !== key) {
    ref.current = { key, anchor: new Map(), hop: new Map(), at: new Map(), grid: new Map(), hist: new Map(), cardT: new Map(), cardKey: new Map(), upto: -1, sig: '', aspect: null };
  }
  return ref.current;
}

function hits(sc, r) {
  const out = new Set();
  for (const c of cellsOf(r)) {
    for (const id of sc.grid.get(c) || []) {
      const o = sc.at.get(id).r;
      if (o[0] < r[2] && o[2] > r[0] && o[1] < r[3] && o[3] > r[1]) out.add(id);
    }
  }
  return out;
}
function put(sc, id, kind, x, y, t) {
  const old = sc.at.get(id);
  if (old) for (const c of cellsOf(old.r)) sc.grid.get(c)?.delete(id);
  const r = rectAt(kind, x, y);
  sc.at.set(id, { x, y, kind, r });
  for (const c of cellsOf(r)) { if (!sc.grid.has(c)) sc.grid.set(c, new Set()); sc.grid.get(c).add(id); }
  const h = sc.hist.get(id) || [];
  if (h.length && h[h.length - 1][0] === t) h.pop();
  h.push([t, x, y, kind]);
  sc.hist.set(id, h);
}
// Where a species was at time t: [x, y, kind], or null before it was found.
function placeAt(sc, id, t) {
  const h = sc.hist.get(id);
  if (!h) return null;
  for (let i = h.length - 1; i >= 0; i -= 1) if (h[i][0] <= t + 1e-6) return h[i].slice(1);
  return null;
}
const hash = (s) => { let h = 2166136261; for (const ch of String(s)) h = Math.imul(h ^ ch.charCodeAt(0), 16777619); return (h >>> 0) / 4294967296; };

// The free spot for a species beside (ax, ay): rings round it, outwards, in the stage's shape (ellipses as wide
// as the stage); of the first free ones, the spot nearest it that keeps the network compact (near the start).
// blocks(id): whether a species already placed keeps it from a spot.
function spotNear(sc, id, kind, ax, ay, blocks) {
  const ex = Math.sqrt(sc.aspect), ey = 1 / ex, step = kind === 'dot' ? 2 * DOT + 2 : 24, ph = hash(id) * 2 * Math.PI;
  const round = Math.sqrt((ex * ex + ey * ey) / 2);
  let best = null, cost = Infinity, last = Infinity;
  for (let i = 0; i < 200 && i <= last; i += 1) {
    const rad = i * step, n = i ? Math.max(8, Math.round((2 * Math.PI * rad * round) / step)) : 1;
    for (let k = 0; k < n; k += 1) {
      const th = ph + (2 * Math.PI * k) / n, x = ax + rad * ex * Math.cos(th), y = ay + rad * ey * Math.sin(th);
      let free = true;
      for (const o of hits(sc, rectAt(kind, x, y))) if (o !== id && blocks(o)) { free = false; break; }
      if (!free) continue;
      const c = rad + 0.7 * Math.hypot(x / ex, y / ey);
      if (c < cost) { cost = c; best = [x, y]; }
      if (last === Infinity) last = i + 3;
    }
  }
  return best || [ax, ay];
}

// A structure near (ax, ay), clear of the other structures; the dots under it go beside the species they
// were found from.
function settleCard(sc, id, ax, ay, t) {
  const [x, y] = spotNear(sc, id, 'card', ax, ay, (q) => sc.at.get(q).kind !== 'dot');
  put(sc, id, 'card', x, y, t);
  for (const q of hits(sc, sc.at.get(id).r)) {
    if (q === id || sc.at.get(q).kind !== 'dot') continue;
    const a = sc.at.get(sc.anchor.get(q)) || sc.at.get(id);
    put(sc, q, 'dot', ...spotNear(sc, q, 'dot', a.x, a.y, () => true), t);
  }
}

// The frames not yet grown, in order (a live run's new ones as they come): the species found by each frame get
// their places, beside the species they were found from; those that matter (mattersAt) become structures,
// beside the structure they react with nearest the start.
const frameSig = (fr) => (fr ? `${fr.t}:${fr.species.length}:${fr.steps.length}` : '');
function growScheme(sc, m) {
  const anchorOf = anchorFor(m);
  if (!sc.at.has(m.seed)) { put(sc, m.seed, 'seed', 0, 0, -Infinity); sc.hop.set(m.seed, 0); }
  // (a live run's last frame can be rewritten in place, and is rebuilt from the summary at the end: grown again)
  const from = sc.upto >= 0 && frameSig(m.frames[sc.upto]) !== sc.sig ? sc.upto : sc.upto + 1;
  for (let i = from; i < m.frames.length; i += 1) {
    const fr = m.frames[i], t = fr.t;
    for (const id of m.order) {
      if (sc.at.has(id) || !(m.first.get(id) <= t + 1e-6)) continue;
      const a = anchorOf(id, sc);
      sc.anchor.set(id, a);
      sc.hop.set(id, a == null ? 0 : (sc.hop.get(a) ?? 0) + 1);
      const p = sc.at.get(a ?? m.seed);
      put(sc, id, 'dot', ...spotNear(sc, id, 'dot', p.x, p.y, () => true), t);
    }
    for (const id of mattersAt(m, fr).ids) {
      if (sc.cardKey.has(m.keyOf(id)) || !sc.at.has(id)) continue;
      sc.cardKey.set(m.keyOf(id), id);
      sc.cardT.set(id, t);
      const by = (m.nb.get(id) || []).filter((x) => sc.at.get(x) && sc.at.get(x).kind !== 'dot')
        .sort((x, y) => (sc.hop.get(x) ?? 99) - (sc.hop.get(y) ?? 99))[0];
      const o = sc.at.get(by ?? id);
      settleCard(sc, id, o.x, o.y, t);
    }
    sc.upto = i;
    sc.sig = frameSig(fr);
  }
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

// Without per-step flux (runs on develop record none), the likely routes: from the start to each
// species holding material, the path whose highest barrier is lowest.
function likelyRoutes(steps, seed, holders) {
  const out = new Map();   // each species' steps, in the steps' order: [step, other end, barrier, forward]
  const link = (x, e) => { if (!out.has(x)) out.set(x, []); out.get(x).push(e); };
  for (const [id, a, b, dGf, dGr] of steps) { link(a, [id, b, dGf, true]); link(b, [id, a, dGr, false]); }
  const best = { [seed]: -Infinity }, via = {};
  const todo = new Set([seed]);
  while (todo.size) {
    let u = null;
    for (const x of todo) if (u == null || best[x] < best[u]) u = x;
    todo.delete(u);
    for (const [id, y, bar, dir] of out.get(u) || []) {
      if (bar == null) continue;
      const c = Math.max(best[u], bar);
      if (best[y] == null || c < best[y]) { best[y] = c; via[y] = [id, u, dir]; todo.add(y); }
    }
  }
  const route = new Map();
  for (const h of holders) for (let v = h, g = 0; via[v] && g < 50; v = via[v][1], g += 1) route.set(via[v][0], via[v][2]);
  return route;
}

// The species that matter at a frame, one per product (a mirror twin shows on its twin's): those holding
// material (2 % of the largest amount), in the order found; then the ends of the routes (the steps carrying 5 %
// of the largest flux; without flux, the likely routes to the six largest holders), read the way material goes.
function mattersAt(m, fr, route = null) {
  const others = fr.species.filter((x) => x[0] !== m.seed);
  const peak = Math.max(1e-30, ...others.map((x) => x[2] || 0));
  const ids = [], keys = new Set();
  const add = (id) => { if (id === m.seed || keys.has(m.keyOf(id))) return; keys.add(m.keyOf(id)); ids.push(id); };
  for (const x of [...fr.species].sort((a, b) => (m.first.get(a[0]) ?? 0) - (m.first.get(b[0]) ?? 0))) if ((x[2] || 0) >= 0.02 * peak) add(x[0]);
  const seen = (id) => m.first.get(id) <= fr.t + 1e-6;
  if (fr.legacy && !route) route = likelyRoutes(fr.steps, m.seed, others.filter((x) => (x[2] || 0) >= 0.02 * peak).sort((a, b) => b[2] - a[2]).slice(0, 6).map((x) => x[0]));
  const fluxMax = Math.max(1e-30, ...fr.steps.map((x) => Math.abs(x[7] || 0)));
  for (const [id, a, b, , , , , flux] of fr.steps) {
    if (!seen(a) || !seen(b) || a === b || !(fr.legacy ? route.has(id) : Math.abs(flux || 0) >= 0.05 * fluxMax)) continue;
    for (const x of (fr.legacy ? route.get(id) : (flux || 0) >= 0) ? [a, b] : [b, a]) add(x);
  }
  return { ids, keys };
}

// The anchor a species is placed from: the species it was proposed from, else the first species a step
// joins it to that is already placed (nearest the start), else the start.
function anchorFor(m) {
  return (id, st) => {
    if (id === m.seed) return null;
    const p = m.parent.get(id);
    if (p != null && st.hop.has(p)) return p;
    const placed = (m.nb.get(id) || []).filter((x) => st.hop.has(x));
    if (placed.length) return placed.reduce((x, y) => ((st.hop.get(x) ?? 99) <= (st.hop.get(y) ?? 99) ? x : y));
    return m.seed;
  };
}

// ---------------------------------------------------------------- the view
const steeringOf = (job) => (job?.op === 'graph-enumeration' && (job.params?.steer ?? 'access') === 'access'
  ? job.params?.steer_profile || 'none (the run\'s own level)' : null);

export function LiveScheme({ tl, cur, jobId, example, job, running, follow, controls, apiJob = jobId, eventClock = null, onEvents = null, extra = null }) {
  const m = useMemo(() => expansionModel(tl, cur), [tl, cur]);
  const hc = useHoverCard();
  const hover = hc.cur ? { kind: hc.cur.key[0], id: hc.cur.data } : null;
  const [view, setView] = useState('scheme');
  const [outcome, setOutcome] = useState('mol');
  const [panel, setPanel] = useState(null);   // 'method' | 'key' | null: what opens under the bar
  const togglePanel = (p) => setPanel((x) => (x === p ? null : p));
  const [svgRef, svgPx] = useElementSize();
  const sc = useScheme(`${jobId || example}`);
  const f = m.f;
  const cond = conditionsFrom(tl, f.basis, job);
  const hues = useMemo(() => productHues(m.frames, m.seed, m.keyOf, cur), [tl, cur]);
  const hueOf = (id) => (id === m.seed ? null : hues.get(m.keyOf(id)));
  const rx = useReactorEvents(apiJob, running);
  useEffect(() => { if (onEvents && rx.events.length) onEvents(rx.events); }, [rx.events.length]);
  const [dialog, setDialog] = useState(null);
  const [caught, setCaught] = useState({});
  const [shower, setShower] = useState(null);   // the reactor's events at once: { id, events }
  const followLive = running && cur === m.frames.length - 1;
  const nowOf = eventClock || (() => (followLive && job?.started ? Date.now() / 1000 - job.started : f.t));
  const startSmiles = drawable(tl.smiles[m.seed]) ? tl.smiles[m.seed] : null;
  const imgOf = (id, w = 240, h = 180) => (drawable(tl.smiles[id]) ? `${depictUrl(tl.smiles[id], w, h, id === m.seed ? null : startSmiles)}&bond=${Math.round(w / 6)}`
    : jobId && !String(jobId).startsWith('mix:') ? `/api/jobs/${jobId}/species-graph/${encodeURIComponent(id)}` : null);
  const nameOf = (id) => (id === m.seed ? m.names[id] || 'start' : m.productOf[id] || m.names[id] || '');
  const label = (id) => nameOf(id) || short(tl.smiles[id], 18) || `species ${id}`;
  const knownIds = new Set(tl.known_ids || []);

  // species now
  const amount = new Map(f.species.map((s) => [s[0], s]));
  const seen = m.order.filter((id) => m.first.get(id) <= f.t + 1e-6);
  const peakMax = Math.max(1e-30, ...f.species.filter((s) => s[0] !== m.seed).map((s) => s[2] || 0));
  const holders = f.species.filter((s) => s[0] !== m.seed && (s[2] || 0) >= 0.02 * peakMax).sort((a, b) => b[2] - a[2]);
  const fresh = (id) => id !== m.seed && cur > 0 && m.first.get(id) >= f.t - 12;
  const act = followLive ? activityOf(tl.activity?.queue) : { pairs: [], species: [] };
  const busy = new Set([...act.species, ...act.pairs.flatMap((x) => x.slice(0, 2))]);

  // steps now, each read the way material goes
  const noFlux = !!f.legacy;
  const fluxMax = Math.max(1e-30, ...f.steps.map((s) => Math.abs(s[7] || 0)));
  const route = noFlux ? likelyRoutes(f.steps, m.seed, holders.slice(0, 6).map((s) => s[0])) : null;
  const known = new Set(seen);
  const edges = f.steps.filter((s) => known.has(s[1]) && known.has(s[2]) && s[1] !== s[2]).map((s) => {
    const [id, a, b, dGf, dGr, src, irc, flux, below] = s;
    const fwd = noFlux ? (route.has(id) ? route.get(id) : (sc.hop.get(a) ?? 9) <= (sc.hop.get(b) ?? 9)) : (flux || 0) >= 0;
    const [from, to] = fwd ? [a, b] : [b, a];
    const barrier = fwd ? dGf : dGr, back = fwd ? dGr : dGf;
    const main = noFlux ? route.has(id) : Math.abs(flux || 0) >= 0.05 * fluxMax;
    const w = noFlux ? (main ? 3 : 1.4) : Math.min(10, Math.max(1.4, 10 * Math.sqrt(Math.abs(flux || 0) / fluxMax)));
    return { id, from, to, barrier, back, src, irc, flux, below: !!below, main, w, speed: below ? null : speedOf(barrier, cond),
      hue: main ? hueOf(to) : null, co: tl.co?.[id] };
  });
  const outOf = {};
  for (const e of edges) outOf[e.from] = (outOf[e.from] || 0) + Math.abs(e.flux || 0);
  for (const e of edges) e.share = noFlux || !outOf[e.from] ? null : Math.abs(e.flux || 0) / outOf[e.from];

  // what forms, per molecule (f.yields: the server counts each molecule over the box's states), and as whole
  // mixtures (the box's states, as shares of the starting boxes); amounts at the end of the simulated time
  const groups = new Map();
  for (const s of f.species) {
    if (s[0] === m.seed || !(s[3] > 0)) continue;
    const k = m.keyOf(s[0]);
    const g = groups.get(k) || { id: s[0], key: k, fin: 0 };
    g.fin += s[3];
    groups.set(k, g);
  }
  const mixtures = [...groups.values()].sort((a, b) => b.fin - a.fin).slice(0, LANE_N);
  const molOf = tl.molecules || {}, per = perText(tl), isMixture = !!tl.mixture;
  // A molecule's name: its everyday or benchmark name, else its formula and, when others share that formula,
  // what tells it apart (the server's isomer labels: "C9H14O2, 1,4-isomer, 6-ring"). The lane and the strip,
  // beside its drawing, say what tells it apart; the table and its hover card, the formula too.
  const molNamed = (k) => {
    const sid = molOf[k]?.species;
    return molOf[k]?.name || (sid != null && m.productOf[sid] && !String(tl.smiles[sid] || '').includes('.') ? m.productOf[sid] : '');
  };
  const molTell = (k) => (molNamed(k) ? '' : molOf[k]?.tell || '');
  const molHead = (k) => molNamed(k) || molOf[k]?.formula || (String(k).startsWith('#') ? `species ${String(k).slice(1)}` : short(k, 24));
  const molName = (k) => (molTell(k) ? `${molHead(k)}, ${molTell(k)}` : molHead(k));
  const molImg = (k, w = 240, h = 180) => {
    const smi = isMixture ? tl.smiles[k] : k;   // (a mixture's species are its molecules)
    return drawable(smi) && !String(smi).startsWith('#') ? `${depictUrl(smi, w, h, null)}&bond=${Math.round(w / 3)}` : imgOf(molOf[k]?.species, w, h);
  };
  const molecules = (f.yields || []).slice(0, LANE_N).map(([key, a, sid]) => ({ key, fin: a, id: sid, mol: true }));
  const left = (f.left || []).map(([key, a]) => ({ key, fin: a, name: molName(key) }));
  const leadAt = (fr) => fr.yields?.[0] || null;
  const lead = leadAt(f);
  let steadySince = f.t;
  for (let i = cur; i >= 0 && lead; i -= 1) {
    const l = leadAt(m.frames[i]);
    if (!l || l[0] !== lead[0] || Math.abs(l[1] - lead[1]) > 0.02) break;
    steadySince = m.frames[i].t;
  }
  const ptsOf = (key) => m.frames.slice(0, cur + 1).map((fr) => [fr.t, (fr.yields || []).find((y) => y[0] === key)?.[1] || 0]);
  const series = molecules.map((p) => ({ ...p, hue: hueOf(p.id), pts: ptsOf(p.key) }));
  const outcomes = view === 'scheme' && outcome === 'mix' && !isMixture ? mixtures.map((p) => ({ ...p, hue: hues.get(p.key) })) : series;

  // the scheme: which species matter, placed
  const { ids: relevant, keys: seenKey } = mattersAt(m, f, route);
  if (!sc.aspect && svgPx?.[1] > 0) sc.aspect = Math.min(3, Math.max(0.6, svgPx[0] / svgPx[1]));   // the stage's shape, once
  const ready = !!sc.aspect;
  if (ready) growScheme(sc, m);
  if (ready && sc.born == null) sc.born = f.t;   // the moment the scheme was first drawn: what it placed then is not new
  // structures at this moment: what has held material by now (faded once it no longer does), and the ends of
  // the routes; one per product (a mirror twin shows on its twin's)
  const shownBy = new Map();
  for (const [id, t0] of sc.cardT) if (t0 <= f.t + 1e-6) shownBy.set(m.keyOf(id), id);
  for (const id of relevant) if (!shownBy.has(m.keyOf(id)) && placeAt(sc, id, f.t)) shownBy.set(m.keyOf(id), id);
  const cells = new Set(shownBy.values());
  const past = new Set([...cells].filter((id) => !seenKey.has(m.keyOf(id))));
  const cellOf = (id) => (cells.has(id) ? id : shownBy.get(m.keyOf(id)) ?? null);
  const at = (id) => { const o = placeAt(sc, id, f.t); return o ? [o[0], o[1]] : null; };
  const P = (id) => (id == null ? null : cellOf(id) != null ? at(cellOf(id)) : at(id));
  const cardW = (id) => (id === m.seed ? SEED_W : CARD_W);
  const isCard = (id) => id === m.seed || cellOf(id) != null;
  const groupFin = (id) => groups.get(m.keyOf(id))?.fin ?? amount.get(id)?.[3];

  // the frame: what is shown, and room round the start for the reactor's events (all there is while its first
  // MD runs), in the stage's shape
  const aspect = sc.aspect || 2;
  let box = [-150 * aspect, -150, 150 * aspect, 150];
  for (const id of ready ? seen : []) {
    const p = P(id);
    if (!p) continue;
    const r = rectAt(id === m.seed ? 'seed' : isCard(id) ? 'card' : 'dot', p[0], p[1]);
    box = [Math.min(box[0], r[0]), Math.min(box[1], r[1]), Math.max(box[2], r[2]), Math.max(box[3], r[3])];
  }
  let bw = box[2] - box[0] + 40, bh = box[3] - box[1] + 40;
  if (bw / bh < aspect) bw = bh * aspect; else bh = bw / aspect;
  const vb = useViewBox([(box[0] + box[2]) / 2 - bw / 2, (box[1] + box[3]) / 2 - bh / 2, bw, bh], follow);
  // the reactor's events are drawn at a size on screen, whatever the zoom: px per unit of the drawing
  const ppu = svgPx ? Math.min(svgPx[0] / vb[2], svgPx[1] / vb[3]) : 0.6;
  const ghostScale = Math.min(4, Math.max(0.8, GHOST_PX / (30 * ppu)));
  // (a shower's are smaller, and smaller still on a narrow stage)
  const showerScale = Math.min(4, Math.max(0.2, Math.min(SHOWER_PX, Math.max(14, (svgPx?.[0] || 1100) / 42)) / (30 * ppu)));

  // a line between two species leaves and enters each at the edge of what is drawn there, bowed a little
  const edgeAt = (id, p, q) => {
    const hw = isCard(id) ? cardW(id) / 2 + 4 : 6, hh = isCard(id) ? cardW(id) * 0.375 + 4 : 6;
    const dx = q[0] - p[0], dy = q[1] - p[1], k = Math.min(1, hw / Math.max(1e-9, Math.abs(dx)), hh / Math.max(1e-9, Math.abs(dy)));
    return [p[0] + dx * k, p[1] + dy * k];
  };
  const ends = (e) => { const a = P(e.from), b = P(e.to); return [edgeAt(e.from, a, b), edgeAt(e.to, b, a)]; };
  const pathOf = (e) => {
    const [p, q] = ends(e), mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2, dx = q[0] - p[0], dy = q[1] - p[1];
    return `M${p[0]},${p[1]} Q${mx - dy * 0.12},${my + dx * 0.12} ${q[0]},${q[1]}`;
  };
  const apart = (e) => { const a = P(e.from), b = P(e.to); return a && b && (a[0] !== b[0] || a[1] !== b[1]); };
  // progressive disclosure: the routes to the products always; a species' own steps on hover or focus;
  // a step just found, for a moment (a few at a time: a run's last frame, rebuilt from its summary,
  // would flash them all)
  let newStep = new Set(cur > 0 ? f.steps.map((x) => x[0]).filter((id) => !m.frames[cur - 1].steps.some((y) => y[0] === id)) : []);
  if (newStep.size > 12) newStep = new Set();
  const shownEdges = ready ? edges.filter((e) => apart(e) && (e.main || hover?.id === e.id || (hover?.kind === 'n' && (e.from === hover.id || e.to === hover.id))
    || (newStep.has(e.id) && !e.below && isCard(e.from) && isCard(e.to)))) : [];

  // corrections since the last frame: flashed on what they moved
  const prevT = cur > 0 ? m.frames[cur - 1].t : -1;
  const moved = new Set(), reref = new Set();
  for (const ev of tl.events || []) {
    if (!(ev[0] > prevT + 1e-6 && ev[0] <= f.t + 1e-6)) continue;
    if (ev[1] === 'barrier_moved') for (const id of ev[2]) moved.add(id);
    if (ev[1] === 'species_rereferenced') for (const id of ev[2]) reref.add(id);
  }

  const hovered = hover?.kind === 'n' ? hover.id : null;
  const hoveredEdge = hover?.kind === 'e' ? edges.find((e) => e.id === hover.id) : null;
  const hoveredMol = hover?.kind === 'm' ? (f.yields || []).find((y) => y[0] === hover.id)?.[2] : null;
  const near = hovered ? new Set([hovered, ...edges.filter((e) => e.from === hovered || e.to === hovered).flatMap((e) => [e.from, e.to])])
    : hoveredMol != null ? new Set([hoveredMol, cellOf(hoveredMol)]) : null;
  const dim = (id) => near && !near.has(id);
  const dotIds = ready ? seen.filter((id) => !isCard(id) && P(id)) : [];
  const cardIds = ready ? seen.filter((id) => (id === m.seed || cells.has(id)) && P(id)) : [];
  // the threads of discovery: each species to the one it was found from
  const threads = dotIds.concat(cardIds).filter((id) => id !== m.seed && P(sc.anchor.get(id)));
  // arrivals grow from where they came from: a dot from the species it was found from, a structure in place
  const origin = (id) => (isCard(id) ? P(id) : P(sc.anchor.get(id)) || P(id));
  const style = (id) => { const a = origin(id), c = P(id); return `--fx:${a[0] - c[0]}px;--fy:${a[1] - c[1]}px;transform:translate(${c[0]}px,${c[1]}px)`; };
  const freshCard = (id) => cur > 0 && id !== m.seed && (sc.cardT.get(id) ?? -1) >= f.t - 12 && (sc.cardT.get(id) > sc.born || cur < m.frames.length - 1);

  // the reactor's events
  const tNow = nowOf();
  const madeBy = new Map();
  for (const e of rx.events) if (e.prod != null && e.t <= tNow + 0.5) { if (!madeBy.has(e.prod)) madeBy.set(e.prod, []); madeBy.get(e.prod).push(e); }
  const openEvent = (ev) => { hc.close(); setDialog(ev); };
  const lastTen = rx.events.reduce((n, e) => n + (e.t <= tNow + 0.5 && e.t > tNow - WINDOW_S ? 1 : 0), 0);
  const avoid = cardIds.map((id) => { const [x, y] = P(id), w = cardW(id); return [x, y + 10, w + 16, w * 0.75 + 46]; });
  const onCatch = (ids) => setCaught((c) => { const o = { ...c }, t = performance.now(); for (const id of [].concat(ids)) o[id] = t; return o; });
  const isCaught = (id) => caught[id] && performance.now() - caught[id] < 1200;
  // the reactor's events since you last looked, at once: on opening the page (once its events are in), and on
  // coming back to it; what plays while you watch is seen. A replay sent back to the start ends a shower.
  const showerFrom = (evs) => {
    // (while it follows a running job, the last few seconds' events are the live ghosts')
    const lo = followLive ? nowOf() - WINDOW_S : Infinity;
    const list = evs.filter((e) => e.t <= lo);
    if (list.length && !reduced()) setShower({ id: performance.now(), events: list });
  };
  const rxRef = useRef(rx);
  rxRef.current = rx;
  const showerRef = useRef(null);
  showerRef.current = showerFrom;
  const lastT = (evs) => evs.reduce((t, e) => Math.max(t, e.t), -Infinity);
  const looked = useRef(false);
  useEffect(() => {
    if (!ready || !apiJob || rx.epoch == null || looked.current) return undefined;
    looked.current = true;
    const mark = seenMark(apiJob), evs = rx.events;
    setSeenMark(apiJob, lastT(evs));
    const id = setTimeout(() => showerRef.current(evs.filter((e) => e.t > mark)), 400);
    return () => clearTimeout(id);
  }, [ready, rx.epoch, apiJob]);
  useEffect(() => {
    if (looked.current && !document.hidden && rx.events.length) setSeenMark(apiJob, lastT(rx.events));
  }, [rx.events.length]);
  useEffect(() => {
    if (!apiJob) return undefined;
    const back = () => {
      if (document.hidden || !looked.current) return;
      const mark = seenMark(apiJob), evs = rxRef.current.events;
      setSeenMark(apiJob, lastT(evs));
      showerRef.current(evs.filter((e) => e.t > mark));
    };
    document.addEventListener('visibilitychange', back);
    return () => document.removeEventListener('visibilitychange', back);
  }, [apiJob]);
  const prevCur = useRef(cur);
  useEffect(() => { if (cur < prevCur.current) setShower(null); prevCur.current = cur; }, [cur]);
  const nodeAnchor = (el) => (el.querySelector('.ld-card-bg, circle') || el).getBoundingClientRect();
  const lineArrow = (e) => `url(#ld-arr-${e.hue != null ? `h${e.hue}` : e.speed || 'none'}-${e.w < 2.5 ? 's' : e.w < 6 ? 'm' : 'l'})`;

  const key = html`<${Key} cond=${cond} noFlux=${noFlux} anyBelow=${edges.some((e) => e.below && shownEdges.includes(e))}
    anyUnchecked=${shownEdges.some((e) => e.irc === -1)} known=${knownIds.size} running=${running} anyEvents=${rx.events.length > 0} />`;
  const sel = tl.selectivity;
  const laneMax = Math.max(1, ...outcomes.map((p) => p.fin || 0));
  // what forms, beside the network: a fixed size, whatever the network's zoom
  const side = html`<aside class="ld-out" aria-label="What forms">
    <div class="ld-out-head">
      <b title=${outcomes === series ? `At the end of the simulated time${cond.tauKnown ? ` (${fmtTime(cond.tau)})` : ''}, ${per}; a molecule counted in every state of the box that holds it`
        : 'The box\'s states: the share of the starting boxes that end as each whole mixture'}>${outcomes === series ? `What forms ${per}` : 'Whole mixtures'}</b>
      ${!isMixture && html`<div class="segmented small" role="tablist" aria-label="What forms, counted">
        <button role="tab" aria-selected=${outcome === 'mol'} class=${outcome === 'mol' ? 'on' : ''} onClick=${() => setOutcome('mol')}
          title="What forms, per molecule: each counted in every state of the box that holds it">Molecules</button>
        <button role="tab" aria-selected=${outcome === 'mix'} class=${outcome === 'mix' ? 'on' : ''} onClick=${() => setOutcome('mix')}
          title="What forms, as the box's states: how many of the starting boxes end as each whole mixture">Mixtures</button></div>`}
    </div>
    ${sel && html`<svg class="ld-out-gauge" viewBox="-4 -48 310 96"><${LaneGauge} sel=${sel} f=${f} x=${0} y=${0} /></svg>`}
    ${!outcomes.length && html`<p class="small muted">${tl.mixture?.pending ? 'Nothing merged yet: the seeds are running' : 'Nothing formed yet'}</p>`}
    ${outcomes.map((p) => {
      const nm = p.mol ? molName(p.key) : label(p.id), img = p.mol ? molImg(p.key) : imgOf(p.id);
      return html`<div class=${`ld-out-row h${p.hue ?? 'x'}`} key=${`ln${p.key}`} tabindex="0" ...${p.mol ? hc.bind(`m:${p.key}`, p.key, nodeAnchor) : hc.bind(`n:${p.id}`, p.id, nodeAnchor)}
          aria-label=${p.mol ? `${nm}: ${amt(p.fin)} ${per} at the end` : `${nm}: ${amt(p.fin)} of the starting boxes end as this`}>
        ${img ? html`<img src=${img} alt="" />` : html`<span class="ld-out-img"></span>`}
        <div class="ld-out-txt"><b class="ld-out-amt">${amt(p.fin)}</b>
          <span class="ld-out-name">${short(p.mol ? molTell(p.key) || molHead(p.key) : nm, 26)}${p.mol && molOf[p.key]?.mirror ? html` <span class="muted">+ mirror</span>` : ''}</span>
          <i class="ld-out-bar"><i style=${`width:${Math.max(1, (100 * (p.fin || 0)) / laneMax)}%`}></i></i></div></div>`;
    })}
    ${outcomes === series && left.length > 0 && html`<p class="ld-out-left" title=${`What is left of the start at the end of the simulated time, ${per}`}>${left.map((l) => `${short(l.name, 20)} left ${amt(l.fin)}`).join(' · ')}</p>`}
  </aside>`;
  const stage = html`<div class="ld-stage ld-b">
    <div class="ld-net">
    <div class="ld-stage-tools"><button class=${`btn small ${panel === 'key' ? 'on' : ''}`} aria-expanded=${panel === 'key'} onClick=${() => togglePanel('key')}
      title="How to read the network">Key</button>${rx.events.length > 0 && !reduced() && html`<button class="btn small"
      onClick=${() => showerFrom(rx.events)} title=${`The reactor's ${rx.events.length} reaction events at once${rx.events.length > SHOWER_MAX ? ` (${SHOWER_MAX} of them: some from each species)` : ''}`}>All reactions</button>`}</div>
    <svg ref=${svgRef} viewBox=${vb.join(' ')} preserveAspectRatio="xMidYMid meet" role="group" style="--s:1"
      aria-label="The network grown from the start, in the middle: each species beside the one it was found from; species that hold material as structures, the rest as dots">
      <defs>${['fast', 'run', 'slow', 'none', 'h0', 'h1', 'h2'].flatMap((k) => [['s', 10], ['m', 15], ['l', 22]].map(([z, px]) => html`<marker id=${`ld-arr-${k}-${z}`} viewBox="0 0 10 10" refX="8" refY="5"
        markerWidth=${px} markerHeight=${px} markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L9,5 L0,9 z" class=${`ld-arrhead ${k}`} /></marker>`))}</defs>
      <g class="ld-threads">${threads.map((id) => { const a = P(sc.anchor.get(id)), b = P(id);
        return html`<line key=${`t${id}`} x1=${a[0]} y1=${a[1]} x2=${b[0]} y2=${b[1]} class=${dim(id) ? 'dim' : ''} />`; })}</g>
      <g class="ld-steps">${[...shownEdges].sort((a, b) => a.main - b.main).map((e) => html`<g key=${e.id}
          class=${`ld-step ${e.main ? 'main' : ''} ${newStep.has(e.id) && !e.main ? 'new' : ''} ${moved.has(e.id) ? 'moved' : ''} ${e.speed || 'none'} ${e.below ? 'below' : ''} ${e.irc === -1 ? 'unchecked' : ''} ${(near && !(near.has(e.from) && near.has(e.to))) ? 'dim' : ''} ${hover?.id === e.id ? 'on' : ''}`}
          ...${hc.bind(`e:${e.id}`, e.id, (el, ev) => { const r = el.getBoundingClientRect(); return ev ? pointBox(ev.clientX, ev.clientY) : pointBox(r.left + r.width / 2, r.top + r.height / 2); })}
          aria-label=${`Step ${label(e.from)} to ${label(e.to)}${e.barrier != null ? `, barrier ${e.barrier.toFixed(1)} kcal/mol` : ''}`}>
        <path d=${pathOf(e)} class="ld-hit" />
        <path d=${pathOf(e)} class="ld-line" style=${`stroke-width:${e.w}px${e.hue != null ? `;stroke:var(--ld-h${e.hue})` : ''}`}
          marker-end=${e.main || hover?.id === e.id ? lineArrow(e) : null} />
        ${e.main && !noFlux && html`<path d=${pathOf(e)} class="ld-flow" style=${`stroke-width:${Math.max(2, e.w * 0.55)}px`} />`}
        ${e.co?.with?.length > 0 && (e.main || hover?.id === e.id) && (() => {
          const [p, q] = ends(e);
          return html`<text x=${(p[0] + q[0]) / 2} y=${(p[1] + q[1]) / 2 - 9} class="ld-co" style="font-size:12px">+ ${e.co.with.map(label).join(' + ')}</text>`;
        })()}
      </g>`)}</g>
      <g class="ld-activity">${act.pairs.filter(([a, b]) => P(a) && P(b) && a !== b).map(([a, b, kind], k) => html`<path key=${`p${a}-${b}-${k}`}
        d=${`M${P(a)[0]},${P(a)[1]} L${P(b)[0]},${P(b)[1]}`} class=${`ld-probe ${kind}`} />`)}</g>
      <g class="ld-nodes">${dotIds.map((id) => {
        const s = amount.get(id), hue = hueOf(id);
        return html`<g key=${`n${id}`} class=${`ld-dot ${isCaught(id) ? 'caught' : ''} ${s?.[4] ? 'pop' : ''} ${hue != null && s?.[3] > 1e-3 ? `h${hue}` : ''} ${fresh(id) ? 'fresh' : ''} ${busy.has(id) ? 'busy' : ''} ${dim(id) ? 'dim' : ''}`}
          style=${style(id)} ...${hc.bind(`n:${id}`, id, nodeAnchor)} tabindex="-1">
          <g class="ld-grow"><circle r="4.5" /><circle r="8" class="ld-hit" /></g></g>`;
      })}</g>
      ${ready && rx.events.length > 0 && html`<${ReactorOverlay} apiJob=${apiJob} events=${rx.events} now=${nowOf} live=${followLive || !!eventClock}
        posOf=${(id) => (known.has(id) ? P(id) : isMixture && id === '0' ? P(m.seed) : null)} seenAt=${(id, t) => (m.first.get(id) ?? Infinity) <= t + 0.5}
        avoid=${avoid} settleOf=${() => null} scale=${ghostScale} bounds=${vb} onOpen=${openEvent} onCatch=${onCatch} />`}
      ${ready && shower && html`<${GhostShower} key=${shower.id} apiJob=${apiJob} events=${shower.events}
        posOf=${(id) => (known.has(id) ? P(id) : isMixture && id === '0' ? P(m.seed) : null)} seenAt=${(id, t) => (m.first.get(id) ?? Infinity) <= t + 0.5}
        avoid=${avoid} scale=${showerScale} bounds=${vb} onOpen=${openEvent} onCatch=${onCatch} onDone=${() => setShower(null)} />`}
      <g class="ld-cards">${cardIds.map((id) => {
        const w = cardW(id), h = w * 0.75, hue = hueOf(id), img = imgOf(id), nm = nameOf(id) || short(tl.smiles[id], 16), fin = groupFin(id);
        return html`<g key=${`c${id}`} class=${`ld-card ${isCaught(id) ? 'caught' : ''} ${reref.has(id) ? 'reref' : ''} ${id === m.seed ? 'seed' : ''} ${past.has(id) ? 'past' : ''} ${hue != null ? `h${hue}` : ''} ${freshCard(id) ? 'fresh' : ''} ${busy.has(id) ? 'busy' : ''} ${dim(id) ? 'dim' : ''}`}
          style=${style(id)} ...${hc.bind(`n:${id}`, id, nodeAnchor)} role="button"
          aria-label=${`${id === m.seed ? 'The start' : label(id)}${fin > 1e-4 && id !== m.seed ? `, ${amt(fin)} ${isMixture ? per : 'of the starting boxes end as this'}` : ''}`}>
          <g class="ld-grow">
            <rect x=${-w / 2} y=${-h / 2} width=${w} height=${h} rx="6" class="ld-card-bg" />
            ${img ? html`<image href=${img} x=${-w / 2 + 3} y=${-h / 2 + 3} width=${w - 6} height=${h - 6} preserveAspectRatio="xMidYMid meet" />`
              : html`<text y="5" class="ld-unknown">?</text>`}
            ${knownIds.has(id) && html`<text x=${w / 2 - 8} y=${-h / 2 + 13} class="ln-known" title="reused from an earlier run">↺</text>`}
            <text y=${h / 2 + 15} class="ld-name" style="font-size:12.5px">${short(id === m.seed ? nameOf(id) || 'start' : nm, 17)}</text>
            ${id !== m.seed && fin > 1e-4 && html`<text y=${h / 2 + 30} class="ld-name ld-amt" style="font-size:12px">${amt(fin)}<title>${`${amt(fin)} ${isMixture ? per : 'of the starting boxes end as this'}`}</title></text>`}
          </g></g>`;
      })}</g>
    </svg>
    ${panel === 'key' && html`<div class="ld-key-pop">${key}<button class="btn small ghost" onClick=${() => setPanel(null)} aria-label="Close the key">✕</button></div>`}
    </div>
    ${view === 'scheme' && side}
    ${hovered && html`<${Popover} anchor=${hc.cur.anchor} className="ld-pop" onEnter=${hc.card.onEnter} onLeave=${hc.card.onLeave}>
      <${NodeCard} s=${amount.get(hovered)} img=${imgOf(hovered, 320, 240)} name=${nameOf(hovered)} known=${knownIds.has(hovered)}
        smiles=${tl.smiles[hovered]} first=${m.first.get(hovered)} seed=${hovered === m.seed} made=${madeBy.get(hovered) || []} onEvent=${openEvent}
        unit=${isMixture ? 'of the feed' : 'of boxes'} /><//>`}
    ${hover?.kind === 'm' && html`<${Popover} anchor=${hc.cur.anchor} className="ld-pop" onEnter=${hc.card.onEnter} onLeave=${hc.card.onLeave}>
      <${MoleculeCard} k=${hover.id} y=${(f.yields || []).find((x) => x[0] === hover.id)} img=${molImg(hover.id, 320, 240)} name=${molName(hover.id)} formula=${molOf[hover.id]?.formula}
        mirror=${molOf[hover.id]?.mirror} per=${per} holder=${(sid) => label(sid)} mixture=${isMixture} /><//>`}
    ${hoveredEdge && html`<${Popover} anchor=${hc.cur.anchor} className="ld-pop edge" onEnter=${hc.card.onEnter} onLeave=${hc.card.onLeave}>
      <${EdgeCard} e=${hoveredEdge} nm=${label} cond=${cond} /><//>`}
    ${dialog && html`<${EventDialog} apiJob=${apiJob} ev=${dialog} nameOf=${label} onClose=${() => setDialog(null)} />`}
  </div>`;

  const mixture = String(jobId).startsWith('mix:');
  const title = mixture ? null : example ? 'Live network · example' : job?.title?.replace(/^Reaction network expansion:\s*/, 'Network: ') || 'Live network';
  // (said once there are rates: before the first step, nothing runs on either)
  const electronic = tl.method?.asked === 'free' && tl.method?.used === 'electronic' && f.steps.length > 0;
  const tools = html`
    <div class="segmented small" role="tablist" aria-label="Show the network as">
      <button role="tab" aria-selected=${view === 'scheme'} class=${view === 'scheme' ? 'on' : ''} onClick=${() => setView('scheme')}>Scheme</button>
      <button role="tab" aria-selected=${view === 'table'} class=${view === 'table' ? 'on' : ''} onClick=${() => setView('table')}>Table</button></div>
    ${!mixture && html`<button class=${`btn small ${panel === 'method' ? 'on' : ''}`} aria-expanded=${panel === 'method'} onClick=${() => togglePanel('method')}
      title="Level of theory, rates and what was searched">How computed</button>`}`;
  const strip = html`<${Strip} title=${title} t=${f.t} tEnd=${m.frames[m.frames.length - 1].t} atEnd=${cur === m.frames.length - 1} running=${running} q=${tl.activity?.queue} counts=${f.counts} nSteps=${f.steps.length}
    formed=${series.length > 0} steady=${f.t - steadySince} cond=${cond} pending=${!f.steered}
    live0=${followLive && job?.started ? job.started : null} merged=${mixture ? (tl.mixture?.pending ? 'none' : 'some') : null}
    reactor=${rx.runs ? { running: rx.running, runs: rx.runs, recent: lastTen, events: rx.events.filter((e) => e.t <= tNow + 0.5).length,
      open: () => openJob(apiJob, apiJob === jobId ? 'reactor' : 'live') } : null}
    warn=${electronic && html`<button class="ld-s-warn" onClick=${() => setPanel('method')} title=${`Free energies were asked for, but ${running ? 'a frequency calculation failed or is not done yet' : 'a frequency calculation failed'}: the rates use electronic energies`}>Electronic energies</button>`}
    tools=${tools} />`;
  const table = html`<div class="ld-tables">${!isMixture && html`<h3 class="ld-table-h">What forms, ${per} <span class="muted small">at the end of the simulated time; a molecule counted in every state that holds it</span></h3>
    <${MoleculeTable} f=${f} molImg=${molImg} molName=${molName} molOf=${molOf} label=${label} hueOf=${hueOf} />
    <h3 class="ld-table-h">Outcomes as whole mixtures <span class="muted small">the box's states, as shares of the starting boxes</span></h3>`}
    <${SpeciesTable} m=${m} f=${f} imgOf=${imgOf} label=${label} madeBy=${madeBy} hueOf=${hueOf} smiles=${tl.smiles} unit=${isMixture ? per : 'share of the starting boxes'} /></div>`;
  return html`<div class="ld-page ld-page-b">
    ${strip}
    ${panel === 'method' && !mixture && html`<${MethodNote} m=${tl.method} cond=${cond} example=${!!example} steering=${steeringOf(job) || tl.method?.steering || null}
      running=${running} onClose=${() => setPanel(null)} />`}
    ${view === 'scheme' ? stage : table}
    ${controls}
    ${extra}
    <div class="ld-details">
      ${jobId && !mixture && html`<${Closer} jobId=${jobId} job=${job} activity=${tl.activity} running=${running} />`}
      <${Corrections} events=${tl.events || []} t=${f.t} smiles=${tl.smiles} names=${m.names} />
    </div>
  </div>`;
}

// An element's size in px, kept up to date (null until measured): a callback ref and the size.
function useElementSize() {
  const [el, setEl] = useState(null);
  const [size, setSize] = useState(null);
  useEffect(() => {
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([e]) => setSize([e.contentRect.width, e.contentRect.height]));
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, size];
}

// A benchmark's selectivity in the outcome lane: the run's ratio, and the measured one (validation only).
function LaneGauge({ sel, f, x, y }) {
  const point = [...sel.series].reverse().find((r) => r[0] <= f.t + 1e-6);
  const ref = sel.reference;
  const share = (r) => (r == null ? null : r / (1 + r));
  const p = point ? share(point[1]) : null;
  const lo = ref?.range ? share(ref.range[0]) : ref?.value ? share(ref.value) : null;
  const hi = ref?.range ? share(ref.range[1]) : lo;
  const pair = sel.products?.length === 2 ? sel.products : null;
  const W = 300;
  return html`<g class="ld-gauge" transform=${`translate(${x},${y})`}>
    <text class="ld-lane-sub" y="-30">${pair ? `${pair[0]} : ${pair[1]}` : 'selectivity'}${point?.[2] ? ' · answer stable' : ''}</text>
    <text class="ld-lane-pct" y="-4">${p != null ? `${Math.round(p * 100)} : ${100 - Math.round(p * 100)}` : '— : —'}</text>
    <rect y="8" width=${W} height="12" rx="4" class="ld-gauge-b" />
    <rect y="8" width=${p != null ? W * p : 0} height="12" rx="4" class="ld-gauge-a" />
    ${lo != null && html`<rect x=${W * lo - 1} y="2" width=${Math.max(3, W * (hi - lo))} height="24" class="ld-gauge-ref"><title>${ref?.citation_text || ref?.citation || 'experiment'}</title></rect>`}
    ${lo != null && html`<text class="ld-lane-sub" y="42">▮ measured ${Math.round(share(ref.value) * 100)} : ${100 - Math.round(share(ref.value) * 100)} · validation only</text>`}
  </g>`;
}

// What forms, per molecule, as a table: each molecule, how much forms per starting box, the state that holds
// most of it; then what is left of the start.
function MoleculeTable({ f, molImg, molName, molOf, label, hueOf }) {
  const rows = [...(f.yields || []).map(([k, a, sid]) => [k, a, sid, false]), ...(f.left || []).map(([k, a]) => [k, a, null, true])];
  if (!rows.length) return html`<p class="small muted">Nothing formed yet.</p>`;
  return html`<div class="ld-table-wrap"><table class="ld-table small">
    <caption class="sr-only">What forms, per molecule</caption>
    <thead><tr><th></th><th>molecule</th><th title="Per starting box, at the end of the simulated time">amount</th><th>most of it in</th></tr></thead>
    <tbody>${rows.map(([k, a, sid, isLeft]) => html`<tr key=${k} class=${sid != null && hueOf(sid) != null ? `h${hueOf(sid)}` : ''}>
      <td>${molImg(k, 120, 90) ? html`<img src=${molImg(k, 120, 90)} alt="" loading="lazy" />` : ''}</td>
      <td><b>${molName(k)}</b>${isLeft ? html` <span class="muted">(left of the start)</span>` : ''}${molOf[k]?.mirror ? html` <span class="muted">(with its mirror image)</span>` : ''}
        <div class="mono muted">${String(k).startsWith('#') ? '' : short(k, 40)}</div></td>
      <td class="mono">${amtFine(a)}</td><td>${sid != null ? label(sid) : ''}</td></tr>`)}</tbody></table></div>`;
}

function MoleculeCard({ k, y, img, name, formula, mirror, per, holder, mixture }) {
  return html`${img && html`<img src=${img} alt=${k} />`}
    <div class="ld-pop-name">${name}${mirror ? html` <span class="muted small">with its mirror image</span>` : ''}</div>
    <div class="mono small muted ld-pop-smi">${String(k).startsWith('#') ? 'structure not perceived' : `${formula && !name.startsWith(formula) ? `${formula} · ` : ''}${short(k, 48)}`}</div>
    <div class="ld-pop-nums"><span><b>${amt(y?.[1])}</b>${per}, at the end</span></div>
    ${!mixture && y?.[2] != null && html`<div class="small muted">Counted in every state of the box that holds it; most of it in <b>${holder(y[2])}</b></div>`}`;
}

// Every species the frame holds, as a table (keyboard and screen-reader friendly).
function SpeciesTable({ m, f, imgOf, label, madeBy, hueOf, smiles, unit }) {
  const rows = [...f.species].sort((a, b) => (b[0] === m.seed) - (a[0] === m.seed) || (b[3] || 0) - (a[3] || 0));
  return html`<div class="ld-table-wrap"><table class="ld-table small">
    <caption class="sr-only">Species of the run at this moment</caption>
    <thead><tr><th></th><th>species</th><th title="Free energy relative to the start">G (kcal/mol)</th><th title=${`The most there was at once (${unit})`}>most at once</th>
      <th title=${`At the end of the simulated time (${unit})`}>at the end</th><th>found at</th><th title="Reactor MD events verified to have made it">made in the MD</th></tr></thead>
    <tbody>${rows.map((s) => html`<tr key=${s[0]} class=${hueOf(s[0]) != null ? `h${hueOf(s[0])}` : ''}>
      <td>${imgOf(s[0], 120, 90) ? html`<img src=${imgOf(s[0], 120, 90)} alt="" loading="lazy" />` : ''}</td>
      <td><b>${s[0] === m.seed ? 'start' : label(s[0])}</b><div class="mono muted">${short(smiles[s[0]] || '', 40)}</div></td>
      <td class="mono">${s[1] != null ? `${s[1] > 0 ? '+' : ''}${s[1].toFixed(1)}` : '—'}</td>
      <td class="mono">${s[0] === m.seed ? '' : amtFine(s[2] || 0)}</td><td class="mono">${amtFine(s[3] || 0)}</td>
      <td class="mono">${s[0] === m.seed ? '' : fmtClock(m.first.get(s[0]) || 0)}</td>
      <td class="mono">${(madeBy.get(s[0]) || []).length || ''}</td></tr>`)}</tbody></table></div>`;
}

const sign = (x) => (x > 0 ? `+${x.toFixed(1)}` : x.toFixed(1));
function eventText(ev, smiles, names = {}) {
  const [, kind, ids, d] = ev;
  if (kind === 'barrier_moved') {
    const delta = d.to != null && d.from != null ? d.to - d.from : null;
    const why = /gxtb/.test(d.why || '') ? 'steering pending' : d.src === 'dft' ? 'DFT back' : d.src === 'borrowed' ? 'borrowed' : '';
    return [delta != null && delta < 0 ? 'good' : 'warn', `Barrier ${delta != null ? sign(delta) : ''}${d.to != null ? ` → ${d.to.toFixed(1)} kcal/mol` : ''}${why ? ` · ${why}` : ''}`];
  }
  if (kind === 'species_rereferenced') return ['good', `Lower conformer${names[ids[0]] || smiles[ids[0]] ? `: ${names[ids[0]] || short(smiles[ids[0]], 22)}` : ''}${d.to != null && d.from != null ? ` ${sign(d.to - d.from)} kcal/mol` : ''}`];
  if (kind === 'step_replaced') return ['info', `Step replaced by ${d.by?.length || 1} better`];
  if (kind === 'step_closed') return ['muted', 'Step closed: out of reach'];
  if (kind === 'step_reopened') return ['info', 'Step reopened'];
  return ['muted', kind];
}

// The corrections so far, quietly: newest first (moves under 0.05 kcal/mol left out).
function Corrections({ events, t, smiles, names }) {
  const sizable = (e) => !(['barrier_moved', 'species_rereferenced'].includes(e[1]) && e[3].to != null && e[3].from != null && Math.abs(e[3].to - e[3].from) < 0.05);
  const feed = events.filter((e) => e[0] <= t + 1e-6 && sizable(e));
  if (!feed.length) return null;
  return html`<details class="ld-corr"><summary class="small"><b>Corrections</b> <span class="muted">${feed.length} so far: barriers that moved, lower conformers, steps replaced</span></summary>
    <ul class="ln-feed">${feed.slice(-12).reverse().map((e) => { const [cls, txt] = eventText(e, smiles, names); return html`<li class=${cls}><span class="mono">${fmtClock(e[0])}</span> ${txt}</li>`; })}</ul></details>`;
}

// What the explorer does now, and the ways in for a closer look: the reactor's MD and what it made so
// far, the path searches, results, log.
const WORK = { propose: 'proposing', fast: 'quick paths', channels: 'channels', irc: 'IRCs', piece: 'path pieces', ts: 'TS searches', rotamer: 'rotamers', steer: 'DFT single points' };
const strip_ = (s) => String(s || '').replace(/ \(unusual bonding #\w+\)/g, '');
export function Closer({ jobId, job, activity, running }) {
  const q = activity?.queue;
  const sims = activity?.sims || [];
  const reactor = ['md', 'nanoreactor', 'reactor', 'collide'].includes(job?.params?.generator);
  const openSim = (stream) => { update((s) => { s.networkFocus = { jobId, stream }; }); openJob(jobId, 'live'); };
  const done = q?.finished ? Object.entries(q.finished).filter(([, v]) => v.done || v.failed) : [];
  const made = activity?.reactor?.distinct > 0 ? activity.reactor : null;
  return html`<section class="ln-closer">
    <div class="ln-doors"><span class="section-title">Look closer</span>
      ${reactor && html`<button class="btn small" onClick=${() => openJob(jobId, 'reactor')} title="The molecular dynamics that proposes the reactions">Reactor MD</button>`}
      <button class="btn small" onClick=${() => openJob(jobId, 'live')} title="Each proposed reaction and its path search, live">Path searches</button>
      <button class="btn small" onClick=${() => openJob(jobId, 'result')} disabled=${job?.status === 'queued'}>Results</button>
      <button class="btn small" onClick=${() => openJob(jobId, 'log')}>Log</button>
      ${(done.length > 0 || made || sims.length > 0) && html`<details class="ln-more"><summary class="small">${running ? 'Work so far' : 'Work done'}</summary>
      ${done.length > 0 && html`<div class="ln-done small muted">${done.map(([k, v]) => html`<span title=${v.failed ? `${v.failed} failed` : ''}><b>${v.done}</b> ${WORK[k] || k}</span>`)}</div>`}
      ${made && html`<div class="small muted">The reactor made ${made.distinct} kind${made.distinct === 1 ? '' : 's'} of product in ${made.events} events (${made.runs} MD runs)</div>`}
      ${made && html`<div class="ln-reactor">${made.products.slice(0, 6).map((p) => html`<div class="ln-rp" title=${`${p.products} · seen ${p.count}×`}>
        <img src=${`/api/jobs/${jobId}/reactor-product/${encodeURIComponent(p.run)}/${p.event}`} alt=${p.products} loading="lazy" />
        <span>${strip_(p.products)}${p.unusual ? html` <i title="No Lewis structure fits it: drawn from its geometry">unusual bonding</i>` : ''}</span><b>${p.count}×</b></div>`)}</div>`}
      ${sims.length > 0 && html`<div class="ln-sims">${sims.slice(0, 5).map((s) => html`<button class="ln-sim" onClick=${() => openSim(s.stream)} title=${`${s.label}${s.caption ? ` · ${s.caption}` : ''}: open it`}>
        ${drawable(s.from) ? html`<img src=${depictUrl(s.from, 120, 90, drawable(s.to) ? s.to : null)} alt=${s.from} />` : html`<span class="ln-sim-txt">${strip_(s.from || '?')}</span>`}
        <span class="ln-sim-arrow">→</span>
        ${drawable(s.to) ? html`<img src=${depictUrl(s.to, 120, 90, drawable(s.from) ? s.from : null)} alt=${s.to} />` : html`<span class="ln-sim-txt">${strip_(s.to || '?')}</span>`}
        <span class="small muted">${(s.label || '').replace(/ from species.*/, '')}</span></button>`)}</div>`}
    </details>`}
    </div>
  </section>`;
}

// ---------------------------------------------------------------- pieces
// Run time that ticks while the run is followed live (frames come only with each kinetics refresh).
function LiveClock({ t0 }) {
  const [, tick] = useState(0);
  useEffect(() => { const id = setInterval(() => tick((x) => x + 1), 1000); return () => clearInterval(id); }, []);
  return fmtClock(Math.max(0, Date.now() / 1000 - t0));
}

// The page's one bar: the run's name, its clock and what it does, its size, the reactor's MD, how settled the
// answer is, then the view's switches.
function Strip({ title, t, tEnd, atEnd, running, q, counts, nSteps, formed, steady, cond, pending, reactor, live0 = null, merged = false, warn = null, tools = null }) {
  const ph = running && atEnd ? phaseOf(q) : null;
  const phase = merged ? (merged === 'none' ? (running ? 'Seeds running; nothing merged yet' : 'Nothing was merged') : running ? 'Seeds running; merged so far' : 'Merged network')
    : ph ? ph.text : running ? (atEnd ? 'Starting' : 'Replay of the run so far') : atEnd ? 'Finished' : `Replay of ${fmtClock(tEnd)}`;
  const sub = ph ? ph.detail : [cond.T ? `${Math.round(cond.T)} K` : '', cond.tauKnown ? `${fmtTime(cond.tau)} simulated` : ''].filter(Boolean).join(' · ');
  return html`<div class="ld-strip" role="status" aria-live="off">
    ${title && html`<h2 class="ld-title" title=${title}>${title}</h2>`}
    <div class="ld-s-time" title=${merged ? 'The mixture as merged so far' : 'Run time'}>
      <span class=${`ld-s-dot ${running ? 'live' : ''}`} title=${running ? 'Live: updating as the explorer works' : ''}></span>
      ${!merged && html`<b class="mono">${live0 != null ? html`<${LiveClock} t0=${live0} />` : fmtClock(t)}</b>`}
      ${cond.budget && html`<span class="ld-s-budget" title=${`of a ${fmtClock(cond.budget)} time budget`}><i style=${`width:${Math.min(100, (100 * t) / cond.budget)}%`}></i></span>`}
    </div>
    <div class="ld-s-phase"><b>${phase}</b>${sub && html`<span class="small muted">${sub}</span>`}</div>
    <div class="ld-s-counts"><span><b>${counts.species}</b> species</span><span><b>${counts.steps ?? nSteps}</b> steps</span></div>
    ${reactor && html`<button class=${`ld-s-reactor ${reactor.running ? 'on' : ''}`} onClick=${reactor.open}
        title="The reactor's molecular dynamics that proposes the reactions: open it">
      <b>Reactor MD ›</b><span class="small muted">${reactor.running ? `${reactor.running} running · ` : ''}${reactor.recent
        ? `${reactor.recent} event${reactor.recent === 1 ? '' : 's'} in ${WINDOW_S} s` : `${reactor.events} events`}</span></button>`}
    ${pending ? html`<span class="ld-s-flag" title="The live kinetics wait for their DFT single points: the amounts can still move">energies wait for DFT</span>`
      : formed && !merged ? html`<span class="small ld-s-steady" title="How long the leading molecule and its amount (within 0.02) have held, in run time">answer held <b>${steady >= 60 ? `${(steady / 60).toFixed(1)} min` : `${Math.round(steady)} s`}</b></span>` : null}
    ${warn}
    <div class="ld-s-tools">${tools}</div>
  </div>`;
}

// How the numbers on this page were made (live_network.method_of), said where they are read: a surprising
// answer can then be traced to the level, the rates or what was searched, instead of guessed at. Rates from
// electronic energies when free ones were asked (a frequency calculation failed, or is not done yet) are said in a line.
const steerName = (x) => { const [prog, lvl] = String(x || '').replace(/\s*\(built in\)\s*$/, '').split(/\s*·\s*/); return lvl ? `${lvl} (DFT, ${prog})` : prog; };
const fmtSpan = (s) => (s == null ? '' : s >= 3600 ? `${+(s / 3600).toPrecision(3)} h` : s >= 60 ? `${+(s / 60).toPrecision(3)} min` : `${+s.toPrecision(3)} s`);
function MethodNote({ m, cond, example = false, steering = null, running = false, onClose }) {
  const close = html`<button class="btn small ghost ln-method-x" onClick=${onClose} aria-label="Close">✕</button>`;
  if (!m) {
    return html`<div class="ln-method small">${close}<p class="muted">${example
      ? 'How this was computed was not recorded with this replay (an earlier explorer version): its level of theory, the energies its rates come from and what it searched are unknown here.'
      : `How this was computed was not recorded with this run${steering ? `; its energies are steered to ${steerName(steering)}` : ''}.`}</p></div>`;
  }
  const T = m.T ?? cond?.T, t = m.time_s ?? cond?.time_s;
  const rates = m.used === 'free' ? `free energies (quasi-RRHO${m.standard_state ? `, ${m.standard_state}` : ''})` : m.used === 'electronic' ? 'electronic energies' : null;
  const mismatch = m.asked === 'free' && m.used === 'electronic';
  const where = m.solvent ? `in ${m.solvent}` : 'gas phase';
  return html`<div class="ln-method small">${close}
    ${mismatch && html`<p class="ln-method-warn"><b>Rates use electronic energies, not the free energies asked for:</b> ${running ? 'a frequency calculation failed or is not done yet' : 'a frequency calculation failed'}.</p>`}
    <dl>
      ${m.structures && html`<dt>Structures</dt><dd>Every species, transition state and IRC: <b>${m.structures}</b> (geometries, and the frequencies free energies need).</dd>`}
      <dt>Energies</dt><dd>${m.steering ? html`<b>${steerName(m.steering)}</b>: DFT single points at the ${m.structures || 'same'} geometries${m.steer_points ? `, ${m.steer_points} computed` : ''}. The kinetics use these; a structure still waiting for its own uses ${m.structures || 'the structures\' level'} meanwhile.`
        : html`The structures' own level, <b>${m.structures || 'as run'}</b> (no DFT steering).`}</dd>
      <dt>Rates</dt><dd>Transition-state theory (Eyring) from ${rates ? html`<b>${rates}</b>` : 'the energies above'}${m.asked && m.used && m.asked !== m.used ? ` (asked: ${m.asked})` : ''}; mass-action kinetics from the start alone${T ? ` at ${T} K` : ''}${t ? ` over ${fmtSpan(t)}` : ''}, ${where}.</dd>
      ${m.mode === 'access' && m.searched_kcal != null && html`<dt>Searched</dt><dd>Reactions out of species that hold material, with barriers up to${' '}<b>${m.searched_kcal.toFixed(0)} kcal/mol</b>: what crosses in that time at that temperature${m.margin_kcal ? ` (${m.cutoff_kcal.toFixed(0)} with a ${m.margin_kcal.toFixed(0)} kcal/mol margin for the method's error)` : ''}. Higher ones were never looked for.</dd>`}
      ${m.gaps > 0 && html`<dt>Gaps</dt><dd>${m.gaps} structure${m.gaps === 1 ? '' : 's'} had no free-energy correction (frequencies that failed)${m.gaps_closed
        ? `: ${m.gaps_closed} step${m.gaps_closed === 1 ? '' : 's'} through ${m.gaps === 1 ? 'it' : 'them'} closed` : ''}; the rest use free energies.</dd>`}
      ${m.threshold != null && html`<dt>Grown from</dt><dd>A species' own reactions are searched once ${(m.threshold * 100).toFixed(0)} % of the start has passed through it.</dd>`}
    </dl>
  </div>`;
}

function NodeCard({ s, img, name, smiles, first, seed, known, made = [], onEvent, unit = 'of boxes' }) {
  return html`${img && html`<img src=${img} alt=${smiles || ''} />`}
    <div class="ld-pop-name">${name || (seed ? 'start' : '')}${known ? html` <span class="ln-known" title="reused from an earlier run">↺ reused</span>` : ''}</div>
    <div class="mono small muted ld-pop-smi">${short(smiles || 'structure not perceived', 48)}</div>
    <div class="ld-pop-nums">
      ${s?.[1] != null && html`<span><b>${s[1] > 0 ? '+' : ''}${s[1].toFixed(1)}</b>kcal/mol (G)</span>`}
      ${!seed && html`<span><b>${amt(s?.[2] ?? 0)}</b>${unit}, most at once</span><span><b>${amt(s?.[3] ?? 0)}</b>${unit} at the end</span>`}
    </div>
    <div class="small muted">${seed ? 'the start' : `found at ${fmtClock(first)}`}${s ? ` · ${s[5] ? 'DFT energy' : 'g-xTB energy'}` : ' · holds no material yet'}</div>
    ${made.length > 0 && html`<div class="ld-made small"><span title="Each event's products, at its own MD frames, are this species' molecules (compared as the network compares species)">Made in the reactor's MD by ${made.length} event${made.length === 1 ? '' : 's'}:</span>
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
    <div class="small muted">kcal/mol · ${e.src === 'd' ? 'DFT energies' : e.src === 'b' ? 'borrowed from a twin step' : 'g-xTB energies'}${e.irc === 1 ? ' · IRC joins these two ✓' : e.irc === -1 ? '' : ''}</div>
    ${e.irc === -1 && html`<div class="small"><b>Unconfirmed:</b> no IRC check yet (a run that ends first leaves it out of its final kinetics).</div>`}`}`;
}

function Key({ cond, noFlux, anyBelow, anyUnchecked, known, running, anyEvents }) {
  const T = cond.T ? `${Math.round(cond.T)} K` : 'the run temperature';
  return html`<div class="ld-key small" aria-label="How to read the network">
    ${anyEvents && html`<span class="ld-k"><svg width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="7" class="ld-ghost-bg" /></svg>a reaction in the reactor's MD, as it happens, or all since you last looked at once (click: its movie)</span>`}
    <span class="ld-k"><svg width="78" height="12" aria-hidden="true"><line x1="2" y1="6" x2="24" y2="6" class="ld-kl fast" /><line x1="28" y1="6" x2="50" y2="6" class="ld-kl run" /><line x1="54" y1="6" x2="76" y2="6" class="ld-kl slow" /></svg>
      half-life at ${T}: under ${fmtTime(cond.tau / 100)} · under ${fmtTime(cond.tau)} · longer</span>
    <span class="ld-k"><svg width="34" height="12" aria-hidden="true"><line x1="2" y1="6" x2="24" y2="6" class="ld-kl main h0" /><path d="M24,1 L32,6 L24,11 z" class="ld-arrhead h0" /></svg>
      ${noFlux ? 'likely route to a product (lowest barriers)' : 'material flow, thicker = more'}, in the product's colour</span>
    <span class="ld-k"><svg width="30" height="12" aria-hidden="true"><line x1="2" y1="6" x2="22" y2="6" class="ld-kthread" /><circle cx="24" cy="6" r="3.5" class="ld-kdot" /></svg>a species found, tied to the one it was found from; holding little or nothing yet</span>
    ${anyBelow && html`<span class="ld-k"><svg width="30" height="12" aria-hidden="true"><line x1="2" y1="6" x2="28" y2="6" class="ld-kl below" /></svg>TS below its ends: no barrier</span>`}
    ${anyUnchecked && html`<span class="ld-k"><svg width="30" height="12" aria-hidden="true"><line x1="2" y1="6" x2="28" y2="6" class="ld-kl below" /></svg>unconfirmed: no IRC yet</span>`}
    ${known > 0 && html`<span class="ld-k"><b class="ln-known-k">↺</b>reused from an earlier run (${known})</span>`}
    ${running && html`<span class="ld-k"><svg width="30" height="12" aria-hidden="true"><line x1="2" y1="6" x2="28" y2="6" class="ld-probe" /></svg>searched now</span>`}
    <span class="ld-k muted">Hover or tap a structure or a route for its numbers</span>
  </div>`;
}

// The run's temperature, simulated time and budget: from the timeline (the server reads the job's
// command line), else from the job's command and the kinetics' basis.
function conditionsFrom(tl, basis, job) {
  const c = tl.conditions || {};
  const parsed = conditionsOf(basis, tl.command ?? job?.command);
  const tau = c.time_s ?? (parsed.tauKnown ? parsed.tau : null);
  return { T: c.T ?? parsed.T, tau: tau ?? 1, tauKnown: tau != null, budget: c.budget_s ?? parsed.budget };
}

// ---------------------------------------------------------------- a reacting mixture, the same way
// A mixture (mixture.json) as a one-frame timeline of molecules: each reaction an arrow from its main
// reactant (the one that is not fed, else the first) to each product, its co-reactant written on it
// ("+ formic acid"), as wide as the amount it converted. A molecule is placed from the molecule it
// is made from (the reaction that makes the most of it); feed molecules sit at the start.
export function mixtureTimeline(mx, names, labels = {}) {
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
  // per molecule (here every species is one): mol per mol of feed, the feed's own molecules left
  const fin = (x) => (x.final_concentration || 0) / fedTotal;
  frame.yields = mols.filter((x) => !feed.has(x.molecule) && fin(x) >= 1e-9).sort((a, b) => fin(b) - fin(a)).slice(0, 16).map((x) => [x.molecule, fin(x), x.molecule]);
  frame.left = mx.pending ? [] : mols.filter((x) => feed.has(x.molecule)).map((x) => [x.molecule, fin(x)]);
  const molecules = Object.fromEntries(mols.map((x) => [x.molecule, { name: names?.[x.molecule] || null, ...(labels[x.molecule] || {}), species: x.molecule }]));
  return { frames: [frame], smiles, names, explored, co, molecules, start: [...feed], mixture: { feed: [...feed], tau, seeds: mx.seeds || [], generation: mx.generation, pending: !!mx.pending },
    command: tau != null ? `--time ${tau}` : '' };
}

export function MixtureScheme({ mixture, names, labels = {}, jobId, job, running, extra = null }) {
  const tl = useMemo(() => mixtureTimeline(mixture, names, labels), [JSON.stringify(mixture), JSON.stringify(labels)]);
  // A finished mixture has no timeline: its reactor events replay at their own pace, over and over
  // (the strip and the dialog say when each happened); a running one shows them as they come.
  const t0 = useRef(performance.now());
  const rxEnd = useRef(1);
  const eventClock = running ? null : () => ((performance.now() - t0.current) / 1000) % rxEnd.current;
  return html`<${LiveScheme} tl=${tl} cur=${0} jobId=${`mix:${jobId}`} apiJob=${jobId} job=${job} running=${running} follow=${true}
    controls=${null} extra=${extra} eventClock=${eventClock} onEvents=${(evs) => { rxEnd.current = Math.max(1, ...evs.map((e) => e.t)) + 6; }} />`;
}

// ---------------------------------------------------------------- the reactor's events, overlaid
// While a propose job's MD runs, the reactor finds reaction events (its live_events.json, per MD
// segment). Each shows faintly as a ghost of its subsystem's movie (the atoms that react, the few
// frames around the bond change, projected flat) at the species whose MD it was: it fades in, plays
// once, and drifts to the species it became when that is found (which pulses as it catches it), else
// settles away and fades. At most GHOSTS at once, each drawn GHOST_PX in radius on screen whatever the zoom; the
// rest are counted. A replay plays the events its clock passes (between one frame and the next). Click
// one for the full movie (the Reactor view's event player). Reduced motion: the latest one, still.
const GHOSTS = 5, LIFE_MS = 5200, WINDOW_S = 10, GHOST_PX = 46;

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
    api.get(`/api/jobs/${encodeURIComponent(apiJob)}/reactor/events/${ev.k}?name=${encodeURIComponent(ev.run)}&frames=40${ev.seed ? `&seed=${encodeURIComponent(ev.seed)}` : ''}`)
      .then((x) => {
        const g = ghostFrames(x);
        EVENT_CACHE.set(key, g);
        if (EVENT_CACHE.size > 400) EVENT_CACHE.delete(EVENT_CACHE.keys().next().value);
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

// Where a ghost appears: beside the species it started from, inside the view, clear of the cards (products
// holding material) and of the other ghosts; null when there is no such place (the event is then skipped,
// not stacked on another).
function ghostSpot(src, avoid, scale, k, bounds = null) {
  const r = 42 * scale;
  const inside = (x, y) => !bounds || (x - r >= bounds[0] && x + r <= bounds[0] + bounds[2] && y - r >= bounds[1] && y + r <= bounds[1] + bounds[3]);
  for (const d of [70, 100, 135, 175, 220, 270].map((x) => x * scale)) {
    for (let j = 0; j < 12; j += 1) {
      const th = -Math.PI / 3 + ((j + k * 5) % 12) * (Math.PI / 6);
      const x = src[0] + d * Math.cos(th), y = src[1] + d * Math.sin(th);
      if (inside(x, y) && avoid.every(([ax, ay, aw, ah]) => x + r < ax - aw / 2 || x - r > ax + aw / 2 || y + r < ay - ah / 2 || y - r > ay + ah / 2)) return [x, y];
    }
  }
  return null;
}

export function ReactorOverlay({ apiJob, events, now, live, posOf, seenAt, avoid, settleOf, scale, onOpen, onCatch, bounds = null }) {
  const still = reduced();
  const [ghosts, setGhosts] = useState([]);
  const shown = useRef(new Set());
  const lastT = useRef(null);   // the clock at the last look: a replay plays what happened since
  const st = useRef({});
  st.current = { events, now, live, posOf, seenAt, avoid, settleOf, scale, onCatch, bounds };
  useEffect(() => {
    if (still) return undefined;
    const tick = () => {
      const { events: evs, now: nowF, posOf: pos, seenAt: seen, avoid: av, settleOf: settle, scale: sc, onCatch: caught } = st.current;
      const t = nowF(), wall = performance.now(), last = lastT.current;
      if (last != null && t < last - 1) shown.current.clear();   // the replay went back: its events may play again
      const lo = st.current.live || last == null || t < last ? t - WINDOW_S : Math.min(t - WINDOW_S, last);
      lastT.current = t;
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
            if (e.t < lo) { if (st.current.live) break; continue; }
            if (!shown.current.has(e.i) && pos(e.src ?? '0')) recent.push(e);
          }
          // the newest first, those whose product is found before the rest
          recent.sort((a, b) => (b.prod != null) - (a.prod != null) || b.t - a.t);
          for (const e of recent.slice(0, GHOSTS - out.length)) {
            shown.current.add(e.i);
            const src = pos(e.src ?? '0');
            const at = ghostSpot(src, [...av, ...out.map((x) => [x.at[0], x.at[1], 84 * sc, 84 * sc])], sc, e.i, st.current.bounds);
            if (!at) continue;
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
    const src = posOf(e.src ?? '0');
    const at = ghostSpot(src, avoid, scale, 0, bounds) || [src[0] + 80 * scale, src[1] - 80 * scale];
    return html`<g class="ld-ghosts"><${Ghost} key=${e.i} apiJob=${apiJob} gh=${{ ev: e, at, phase: 'play' }} scale=${scale} still=${true} onOpen=${onOpen} /></g>`;
  }
  return html`<g class="ld-ghosts">${ghosts.map((gh) => html`<${Ghost} key=${gh.ev.i} apiJob=${apiJob} gh=${gh} scale=${scale} onOpen=${onOpen} />`)}</g>`;
}

// ---------------------------------------------------------------- the reactor's events, all at once
// A shower: what the reactor found since you last looked (on opening the page, everything so far; on coming
// back to it, what came meanwhile; the "All reactions" button, everything), at once. Each event bursts out of
// the species it started from as a small ghost, plays its movie, and flies into what it became (which pulses)
// or fades. At most SHOWER_MAX: one from each species in turn, those whose product is found first, released
// in the order they happened over SPREAD_MS. One clock for all; each movie fetched once (6 at a time).
const SHOWER_MAX = 150, SHOWER_PX = 26, SPREAD_MS = 2600, STEP_MS = 90, FLY_MS = 1500, GONE_MS = 2900;

// When you last looked at a job's reactor events (the latest event's time then), in this browser.
const markKey = (apiJob) => `mepd:reactor-seen:${apiJob}`;
export function seenMark(apiJob) {
  try { const v = parseFloat(localStorage.getItem(markKey(apiJob))); return Number.isFinite(v) ? v : -Infinity; } catch { return -Infinity; }
}
export function setSeenMark(apiJob, t) {
  try { if (Number.isFinite(t)) localStorage.setItem(markKey(apiJob), String(t)); } catch { /* private window: every visit is the first */ }
}

export function pickShower(evs, max = SHOWER_MAX) {
  const byTime = (a, b) => a.t - b.t || a.i - b.i;
  if (evs.length <= max) return [...evs].sort(byTime);
  const bySrc = new Map();
  for (const e of [...evs].sort(byTime)) { const k = e.src ?? '0'; if (!bySrc.has(k)) bySrc.set(k, []); bySrc.get(k).push(e); }
  const lists = [...bySrc.values()].map((l) => l.sort((a, b) => (b.prod != null) - (a.prod != null) || b.t - a.t));
  const out = [], labels = new Set();
  while (out.length < max && lists.some((l) => l.length)) {
    for (const l of lists) {
      if (out.length >= max || !l.length) continue;
      const j = Math.max(0, l.findIndex((e) => !labels.has(e.label)));   // a reaction not shown yet, if it has one
      const [e] = l.splice(j, 1);
      labels.add(e.label);
      out.push(e);
    }
  }
  return out.sort(byTime);
}

// Where a shower ghost lands: a scatter round its species, inside the view, off the structures, overlapping
// the other ghosts as little as it can (a shower may overlap; it never hides a structure if it can help it).
function showerSpot(src, avoid, placed, scale, i, bounds) {
  const r = 30 * scale * 1.25;
  const rnd = (a, b) => { const h = Math.sin((i + 1) * a + b) * 43758.5453; return h - Math.floor(h); };
  let best = [src[0], src[1] - 2 * r], bestS = Infinity;
  for (let j = 0; j < 16; j += 1) {
    const th = 2 * Math.PI * rnd(12.9898, j * 78.233), d = r * (1.3 + 3.4 * rnd(39.3468, j * 11.135));
    const x = src[0] + d * Math.cos(th), y = src[1] + d * Math.sin(th);
    let s = d / (r * 30);
    if (bounds && (x - r < bounds[0] || x + r > bounds[0] + bounds[2] || y - r < bounds[1] || y + r > bounds[1] + bounds[3])) s += 100;
    for (const [ax, ay, aw, ah] of avoid) if (!(x + r < ax - aw / 2 || x - r > ax + aw / 2 || y + r < ay - ah / 2 || y - r > ay + ah / 2)) s += 10;
    for (const [px, py] of placed) { const dd = Math.hypot(x - px, y - py); if (dd < 2 * r) s += 1 - dd / (2 * r); }
    if (s < bestS) { bestS = s; best = [x, y]; }
  }
  return best;
}

// A ghost's drawing, redrawn only when its frame changes (a shower has many at once).
class StillDrawing extends Component {
  shouldComponentUpdate(p) { return p.k !== this.props.k || p.g !== this.props.g || p.r !== this.props.r; }
  render({ g, k, r }) { return html`<${GhostDrawing} g=${g} k=${k} r=${r} />`; }
}

export function GhostShower({ apiJob, events, posOf, seenAt, avoid, scale, bounds, onCatch, onOpen, onDone }) {
  const props = useRef({});
  props.current = { posOf, seenAt, onCatch, onDone, scale };
  const plan = useMemo(() => {
    const list = pickShower(events.filter((e) => posOf(e.src ?? '0')));
    const placed = [];
    return list.map((ev, n) => {
      const src = posOf(ev.src ?? '0'), at = showerSpot(src, avoid, placed, scale, ev.i, bounds);
      placed.push(at);
      return { ev, src, at, delay: list.length > 1 ? (SPREAD_MS * n) / (list.length - 1) : 0, phase: 'wait', born: 0, g: null };
    });
  }, []);
  const [now, setNow] = useState(0);
  useEffect(() => {   // the movies, in the order they are released
    let alive = true, next = 0;
    const one = async () => {
      while (alive && next < plan.length) {
        const gh = plan[next++], ev = gh.ev, key = `${apiJob}|${ev.seed || ''}|${ev.run}|${ev.k}`;
        if (EVENT_CACHE.has(key)) { gh.g = EVENT_CACHE.get(key); continue; }
        try {
          const x = await api.get(`/api/jobs/${encodeURIComponent(apiJob)}/reactor/events/${ev.k}?name=${encodeURIComponent(ev.run)}&frames=16${ev.seed ? `&seed=${encodeURIComponent(ev.seed)}` : ''}`);
          gh.g = ghostFrames(x);
          EVENT_CACHE.set(key, gh.g);
        } catch { gh.g = false; }
      }
    };
    for (let w = 0; w < 6; w += 1) one();
    return () => { alive = false; };
  }, []);
  useEffect(() => {
    const t0 = performance.now();
    const id = setInterval(() => {
      const t = performance.now() - t0, { posOf: pos, seenAt: seen, onCatch: caught, onDone: done, scale: sc } = props.current;
      const got = [];
      let left = 0;
      for (const gh of plan) {
        if (gh.phase === 'wait') {
          if (t >= gh.delay && gh.g) { gh.phase = 'out'; gh.born = t; } else if (gh.g === false || t > gh.delay + 3000) gh.phase = 'gone';
        } else if (gh.phase === 'out') gh.phase = 'play';   // drawn once at its species: now it flies out
        else if (gh.phase === 'play' && t - gh.born >= FLY_MS) {
          const p = gh.ev.prod != null && seen(gh.ev.prod, Infinity) ? pos(gh.ev.prod) : null;
          if (p) { got.push(gh.ev.prod); gh.phase = 'catch'; gh.to = p; } else { gh.phase = 'settle'; gh.to = [gh.at[0], gh.at[1] - 20 * sc]; }
        } else if ((gh.phase === 'catch' || gh.phase === 'settle') && t - gh.born >= GONE_MS) gh.phase = 'gone';
        if (gh.phase !== 'gone') left += 1;
      }
      if (got.length) caught([...new Set(got)]);
      if (!left) { clearInterval(id); done(); return; }
      setNow(t);
    }, STEP_MS);
    return () => clearInterval(id);
  }, []);
  const r = 30 * scale;
  return html`<g class="ld-ghosts shower">${plan.filter((gh) => gh.phase !== 'wait' && gh.phase !== 'gone').map((gh) => {
    const at = gh.phase === 'out' ? gh.src : gh.to || gh.at;
    const k = Math.max(0, Math.min(13, Math.floor((now - gh.born) / STEP_MS)));
    return html`<g key=${gh.ev.i} class=${`ld-ghost sh ${gh.phase}`} style=${`transform:translate(${at[0]}px,${at[1]}px)`}
        onClick=${() => onOpen(gh.ev)} aria-hidden="true">
      <g class="ld-ghost-in"><circle r=${r * 1.25} class="ld-ghost-bg" /><${StillDrawing} g=${gh.g} k=${k} r=${r} /></g>
      <title>${gh.ev.label} (reactor event; click for its movie)</title></g>`;
  })}</g>`;
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
      <${EventPlayer} jobId=${apiJob} event=${ev.k} runName=${ev.run} seed=${ev.seed} height=${360} />
    </div></div>`;
}
