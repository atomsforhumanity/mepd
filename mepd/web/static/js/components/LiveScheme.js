// The live network (#/network/<job>, #/network/example/<slug>) and a reacting mixture's network, drawn
// as a reaction scheme that grows while the run works (mepd/web/live_network.py):
// - left to right by reactions from the start: the start, then a column per reaction; a species gets a cell
//   with its structure once it matters (holds 2 % of the largest amount), lifting out of its column's
//   pile of what was explored, above, where every species found lands first; routes read left to
//   right, start to product (a species a route reaches through another moves right of it);
// - on the right an outcome lane: what forms, per molecule (the five leading molecules and how much of
//   each forms per starting box at the end of the simulated time; the server counts each molecule over
//   the box's states, so 2 H2 in a box count twice), and a benchmark's measured selectivity; or, toggled,
//   the outcomes as whole mixtures (the box's states, as shares of the starting boxes);
// - a strip on top: run time against the budget, what the explorer does now, counts, the three leading
//   molecules and what is left of the start, how long the answer has held, the steering level, a door to
//   the reactor's MD;
// - steps coloured by speed at the run's temperature (half-life against the simulated time), routes in
//   their product's colour; magenta is what runs now: searches marching between two species, arrivals,
//   a step just found; the reactor's events as faint ghosts of their movies, caught by the species
//   they verifiably made;
// - nothing placed ever moves except to keep a route reading left to right, so a replay and the live
//   run look the same. Hover, focus or tap for details (Popover.js); a table view lists every species.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
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
const LANE_N = 5, LANE_STEP = 86;

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
// The start at the left; a column per step from it (the last holds everything further); in a column,
// cells three across for the species that matter, and above them the pile of every species found
// there, one dot each, in the order found. Places are kept across polls (a ref) and only ever move
// right, when a route would otherwise read backwards.
const SUB_W = 136, SUBS = 3, COL_GAP = 104, COL_W = SUB_W * SUBS + COL_GAP, ROW_H = 122, PILE = 9, PILE_W = 30, MAX_COL = 5;
const CARD_W = 104, SEED_W = 128, TOP = 72;
const colX = (c) => (c === 0 ? 0 : 190 + (c - 1) * COL_W);   // the left edge of a column
// A column's head in plain words: how many species were found that many reactions from the start.
const columnHead = (c, k) => `${k} species ${c === 1 ? 'one reaction' : `${c}${c === MAX_COL ? ' or more' : ''} reactions`} from the start`;

function useScheme(key) {
  const ref = useRef(null);
  if (!ref.current || ref.current.key !== key) {
    ref.current = { key, anchor: new Map(), hop: new Map(), pile: new Map(), piles: new Map(), cell: new Map(), used: new Map(), since: new Map() };
  }
  return ref.current;
}

// The free cell nearest a row: down the column's first sub-column first (so a route coming in from the
// left never passes behind another cell of the column), then the next.
function freeCell(sc, c, row) {
  const used = sc.used.get(c) || new Set();
  for (let reach = 6; reach < 400; reach += 6) {
    for (let j = 0; j < SUBS; j += 1) {
      for (let r = 0; r <= reach; r += 1) {
        for (const y of r ? [row + r, row - r] : [row]) if (y >= 0 && !used.has(`${y}:${j}`)) return [y, j];
      }
    }
  }
  return [0, 0];
}

function take(sc, id, c, row) {
  const old = sc.cell.get(id);
  if (old) sc.used.get(old.c)?.delete(`${old.r}:${old.j}`);
  const [r, j] = freeCell(sc, c, row);
  if (!sc.used.has(c)) sc.used.set(c, new Set());
  sc.used.get(c).add(`${r}:${j}`);
  sc.cell.set(id, { c, r, j });
}

// One pass per frame: discovery anchors (the species each was found from), pile slots, cells for the
// species that now matter, then routes made to read left to right.
function placeScheme(sc, m, relevant, routeEdges, t) {
  const anchorOf = anchorFor(m);
  for (const id of m.order) {
    if (sc.hop.has(id)) continue;
    const a = anchorOf(id, sc);
    sc.anchor.set(id, a);
    sc.hop.set(id, a == null ? 0 : (sc.hop.get(a) ?? 0) + 1);
    if (a != null) {
      const c = Math.min(sc.hop.get(id), MAX_COL), k = sc.piles.get(c) || 0;
      sc.piles.set(c, k + 1);
      sc.pile.set(id, [colX(c) + 10 + (k % PILE_W) * PILE, -44 - Math.floor(k / PILE_W) * PILE]);
    }
  }
  const parentOn = new Map(routeEdges.map((e) => [e.to, e.from]));
  for (const id of relevant) {
    if (sc.cell.has(id) || id === m.seed || !sc.hop.has(id)) continue;
    const p = parentOn.get(id);
    const c = p != null && (p === m.seed || sc.cell.has(p)) ? Math.min((p === m.seed ? 0 : sc.cell.get(p).c) + 1, MAX_COL)
      : Math.max(1, Math.min(sc.hop.get(id), MAX_COL));
    let a = p ?? sc.anchor.get(id);
    while (a != null && a !== m.seed && !sc.cell.has(a)) a = sc.anchor.get(a);
    take(sc, id, c, a != null && a !== m.seed ? sc.cell.get(a).r : 0);
    sc.since.set(id, t);
  }
  // a route reads left to right: a species reached through another sits right of it
  for (let pass = 0; pass < 4; pass += 1) {
    let moved = false;
    for (const e of routeEdges) {
      const a = e.from === m.seed ? { c: 0, r: 0 } : sc.cell.get(e.from), b = sc.cell.get(e.to);
      if (!a || !b || b.c > a.c || a.c >= MAX_COL) continue;
      take(sc, e.to, a.c + 1, a.r);
      moved = true;
    }
    if (!moved) break;
  }
}

const cellXY = (cl) => [colX(cl.c) + cl.j * SUB_W + SUB_W / 2, TOP + cl.r * ROW_H];

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
  const sc = useScheme(`${jobId || example}`);
  const f = m.f;
  const cond = conditionsFrom(tl, f.basis, job);
  const hues = useMemo(() => productHues(m.frames, m.seed, m.keyOf, cur), [tl, cur]);
  const hueOf = (id) => (id === m.seed ? null : hues.get(m.keyOf(id)));
  const rx = useReactorEvents(apiJob, running);
  useEffect(() => { if (onEvents && rx.events.length) onEvents(rx.events); }, [rx.events.length]);
  const [dialog, setDialog] = useState(null);
  const [caught, setCaught] = useState({});
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

  // the scheme: which species matter (one cell per product: mirror twins share the first's), placed
  const relevant = [];
  const seenKey = new Set();
  for (const s of [...f.species].sort((a, b) => (m.first.get(a[0]) ?? 0) - (m.first.get(b[0]) ?? 0))) {
    if (s[0] === m.seed || (s[2] || 0) < 0.02 * peakMax || seenKey.has(m.keyOf(s[0]))) continue;
    seenKey.add(m.keyOf(s[0]));
    relevant.push(s[0]);
  }
  // (a route's end gets a cell unless its product has one: a mirror twin the route reaches lands on its twin's)
  for (const e of edges) {
    if (!e.main) continue;
    for (const id of [e.from, e.to]) {
      if (id === m.seed || relevant.includes(id) || seenKey.has(m.keyOf(id))) continue;
      seenKey.add(m.keyOf(id));
      relevant.push(id);
    }
  }
  const routeEdges = edges.filter((e) => e.main);
  placeScheme(sc, m, relevant, routeEdges, f.t);
  const cells = new Set(relevant.filter((id) => sc.cell.has(id)));
  const shownBy = new Map([...cells].map((id) => [m.keyOf(id), id]));
  const cellOf = (id) => (cells.has(id) ? id : shownBy.get(m.keyOf(id)) ?? null);
  const P = (id) => (id === m.seed ? [SEED_W / 2, TOP] : cellOf(id) != null ? cellXY(sc.cell.get(cellOf(id))) : sc.pile.get(id) || null);
  const cardW = (id) => (id === m.seed ? SEED_W : CARD_W);
  const isCard = (id) => id === m.seed || cellOf(id) != null;
  const groupFin = (id) => groups.get(m.keyOf(id))?.fin ?? amount.get(id)?.[3];

  // the frame: the scheme, its piles above, the outcome lane at the right
  let box = [-20, -60, SEED_W + 40, TOP + 140];
  for (const id of seen) {
    const p = P(id);
    if (!p) continue;
    const r = isCard(id) ? cardW(id) / 2 + 20 : 8;
    box = [Math.min(box[0], p[0] - r), Math.min(box[1], p[1] - r - 18), Math.max(box[2], p[0] + r), Math.max(box[3], p[1] + r + (isCard(id) ? 34 : 0))];
  }
  const sel = tl.selectivity;
  const laneX = box[2] + 70;
  const laneTop = sel ? 120 : 20;
  const lane = outcomes.map((p, k) => ({ ...p, x: laneX, y: laneTop + 82 + k * LANE_STEP }));
  const laneMax = Math.max(1, ...lane.map((p) => p.fin || 0));
  box = [box[0], Math.min(box[1], -30), laneX + 330, Math.max(box[3], laneTop + 50 + Math.max(1, lane.length) * LANE_STEP)];
  // room round the start for the reactor's events (all there is while its first MD runs)
  box = [Math.min(box[0], -40), Math.min(box[1], TOP - 230), Math.max(box[2], SEED_W + 300), Math.max(box[3], TOP + 230)];
  const bw = Math.max(760, box[2] - box[0]), bh = Math.max(440, box[3] - box[1]);
  const vb = useViewBox([(box[0] + box[2]) / 2 - bw / 2, (box[1] + box[3]) / 2 - bh / 2, bw, bh], follow);

  const side = (id, right) => {
    const [x, y] = P(id);
    return [x + (right ? 1 : -1) * (isCard(id) ? cardW(id) / 2 + 4 : 6), y];
  };
  const pathOf = (e) => {
    const p = side(e.from, true), q = side(e.to, false);
    if (q[0] < p[0] + 12) {   // (the last column, or a step shown on hover that goes back): around, not through
      const r = side(e.to, true), dx = 46;
      return `M${p[0]},${p[1]} C${p[0] + dx},${p[1]} ${r[0] + dx},${r[1]} ${r[0] + 6},${r[1]}`;
    }
    const dx = Math.max(30, (q[0] - p[0]) / 2);
    return `M${p[0]},${p[1]} C${p[0] + dx},${p[1]} ${q[0] - dx},${q[1]} ${q[0]},${q[1]}`;
  };
  // progressive disclosure: the routes to the products always; a species' own steps on hover or focus;
  // a step just found, for a moment (a few at a time: a run's last frame, rebuilt from its summary,
  // would flash them all)
  let newStep = new Set(cur > 0 ? f.steps.map((x) => x[0]).filter((id) => !m.frames[cur - 1].steps.some((y) => y[0] === id)) : []);
  if (newStep.size > 12) newStep = new Set();
  const shownEdges = edges.filter((e) => P(e.from) && P(e.to) && (e.main || hover?.id === e.id || (hover?.kind === 'n' && (e.from === hover.id || e.to === hover.id))
    || (newStep.has(e.id) && !e.below && isCard(e.from) && isCard(e.to))));

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
  const near = hovered ? new Set([hovered, ...edges.filter((e) => e.from === hovered || e.to === hovered).flatMap((e) => [e.from, e.to])]) : null;
  const dim = (id) => near && !near.has(id);
  const dotIds = seen.filter((id) => !isCard(id) && P(id));
  const cardIds = seen.filter((id) => (id === m.seed || cells.has(id)) && P(id));
  // arrivals grow from where they came from: a dot from the species it was found from, a cell from its pile slot
  const origin = (id) => (isCard(id) ? sc.pile.get(id) : P(sc.anchor.get(id))) || P(id);
  const style = (id) => { const a = origin(id), c = P(id); return `--fx:${a[0] - c[0]}px;--fy:${a[1] - c[1]}px;transform:translate(${c[0]}px,${c[1]}px)`; };
  const freshCard = (id) => cur > 0 && id !== m.seed && (sc.since.get(id) ?? -1) >= f.t - 12;

  // the reactor's events
  const tNow = nowOf();
  const madeBy = new Map();
  for (const e of rx.events) if (e.prod != null && e.t <= tNow + 0.5) { if (!madeBy.has(e.prod)) madeBy.set(e.prod, []); madeBy.get(e.prod).push(e); }
  const openEvent = (ev) => { hc.close(); setDialog(ev); };
  const lastTen = rx.events.reduce((n, e) => n + (e.t <= tNow + 0.5 && e.t > tNow - WINDOW_S ? 1 : 0), 0);
  const avoid = [...cardIds.map((id) => { const [x, y] = P(id), w = cardW(id); return [x, y + 10, w + 16, w * 0.75 + 46]; }),
    ...lane.map((p) => [p.x + 155, p.y, 320, LANE_STEP])];
  const settleOf = (ev) => { const c = Math.min((sc.hop.get(ev.src ?? m.seed) ?? 0) + 1, MAX_COL); return [colX(c) + (PILE_W / 2) * PILE, -60]; };
  const onCatch = (id) => setCaught((c) => ({ ...c, [id]: performance.now() }));
  const isCaught = (id) => caught[id] && performance.now() - caught[id] < 1200;
  const nodeAnchor = (el) => (el.querySelector('.ld-card-bg, circle') || el).getBoundingClientRect();
  const lineArrow = (e) => `url(#ld-arr-${e.hue != null ? `h${e.hue}` : e.speed || 'none'}-${e.w < 2.5 ? 's' : e.w < 6 ? 'm' : 'l'})`;

  const stage = html`<div class="ld-stage ld-b">
    <svg viewBox=${vb.join(' ')} preserveAspectRatio="xMidYMid meet" role="group" style="--s:1"
      aria-label="The network as a reaction scheme: the start at the left, a column per step; species that hold material as structures, the rest as dots in each column's pile; at the right, what forms per molecule">
      <defs>${['fast', 'run', 'slow', 'none', 'h0', 'h1', 'h2'].flatMap((k) => [['s', 10], ['m', 15], ['l', 22]].map(([z, px]) => html`<marker id=${`ld-arr-${k}-${z}`} viewBox="0 0 10 10" refX="8" refY="5"
        markerWidth=${px} markerHeight=${px} markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L9,5 L0,9 z" class=${`ld-arrhead ${k}`} /></marker>`))}</defs>
      ${[...sc.piles.entries()].map(([c, k]) => html`<text class="ld-colhead left" key=${`h${c}`} x=${colX(c) + 6} y="-22">${columnHead(c, k)}
        ${c === Math.min(...sc.piles.keys()) && html`<tspan x=${colX(c) + 6} dy="15" class="ld-colnote">(dots hold little or no material yet; hover for the structure)</tspan>`}</text>`)}
      <g class="ld-lane-links">${lane.map((p) => {
        const [x0, y0] = P(p.id) || [p.x, p.y], gx = p.x - 8, from = isCard(p.id) ? x0 + CARD_W / 2 + 4 : x0 + 4;
        return html`<path key=${`lk${p.key}`} d=${`M${from},${y0} C${(from + gx) / 2},${y0} ${(from + gx) / 2},${p.y} ${gx},${p.y}`} class=${`ld-lane-link h${p.hue ?? 'x'} ${hover?.kind === 'm' && hover.id !== p.key ? 'dim' : ''}`} />`;
      })}</g>
      <g class="ld-steps">${[...shownEdges].sort((a, b) => a.main - b.main).map((e) => html`<g key=${e.id}
          class=${`ld-step ${e.main ? 'main' : ''} ${newStep.has(e.id) && !e.main ? 'new' : ''} ${moved.has(e.id) ? 'moved' : ''} ${e.speed || 'none'} ${e.below ? 'below' : ''} ${e.irc === -1 ? 'unchecked' : ''} ${(near && !(near.has(e.from) && near.has(e.to))) ? 'dim' : ''} ${hover?.id === e.id ? 'on' : ''}`}
          ...${hc.bind(`e:${e.id}`, e.id, (el, ev) => { const r = el.getBoundingClientRect(); return ev ? pointBox(ev.clientX, ev.clientY) : pointBox(r.left + r.width / 2, r.top + r.height / 2); })}
          aria-label=${`Step ${label(e.from)} to ${label(e.to)}${e.barrier != null ? `, barrier ${e.barrier.toFixed(1)} kcal/mol` : ''}`}>
        <path d=${pathOf(e)} class="ld-hit" />
        <path d=${pathOf(e)} class="ld-line" style=${`stroke-width:${e.w}px${e.hue != null ? `;stroke:var(--ld-h${e.hue})` : ''}`}
          marker-end=${e.main || hover?.id === e.id ? lineArrow(e) : null} />
        ${e.main && !noFlux && html`<path d=${pathOf(e)} class="ld-flow" style=${`stroke-width:${Math.max(2, e.w * 0.55)}px`} />`}
        ${e.co?.with?.length > 0 && (e.main || hover?.id === e.id) && (() => {
          const p = side(e.from, true), q = side(e.to, false);
          return html`<text x=${(p[0] + q[0]) / 2} y=${(p[1] + q[1]) / 2 - 9} class="ld-co" style="font-size:12px">+ ${e.co.with.map(label).join(' + ')}</text>`;
        })()}
      </g>`)}</g>
      <g class="ld-activity">${act.pairs.filter(([a, b]) => P(a) && P(b) && a !== b).map(([a, b, kind], k) => html`<path key=${`p${a}-${b}-${k}`}
        d=${`M${P(a)[0]},${P(a)[1]} L${P(b)[0]},${P(b)[1]}`} class=${`ld-probe ${kind}`} />`)}</g>
      <g class="ld-nodes">${dotIds.map((id) => {
        const s = amount.get(id), hue = hueOf(id);
        return html`<g key=${`n${id}`} class=${`ld-dot ${isCaught(id) ? 'caught' : ''} ${s?.[4] ? 'pop' : ''} ${hue != null && s?.[3] > 1e-3 ? `h${hue}` : ''} ${fresh(id) ? 'fresh' : ''} ${busy.has(id) ? 'busy' : ''} ${dim(id) ? 'dim' : ''}`}
          style=${style(id)} ...${hc.bind(`n:${id}`, id, nodeAnchor)} tabindex="-1">
          <g class="ld-grow"><circle r="3" /><circle r="5" class="ld-hit" /></g></g>`;
      })}</g>
      ${rx.events.length > 0 && html`<${ReactorOverlay} apiJob=${apiJob} events=${rx.events} now=${nowOf} live=${followLive || !!eventClock}
        posOf=${(id) => (known.has(id) ? P(id) : isMixture && id === '0' ? P(m.seed) : null)} seenAt=${(id, t) => (m.first.get(id) ?? Infinity) <= t + 0.5}
        avoid=${avoid} settleOf=${settleOf} scale=${1} onOpen=${openEvent} onCatch=${onCatch} />`}
      <g class="ld-cards">${cardIds.map((id) => {
        const w = cardW(id), h = w * 0.75, hue = hueOf(id), img = imgOf(id), nm = nameOf(id) || short(tl.smiles[id], 16), fin = groupFin(id);
        return html`<g key=${`c${id}`} class=${`ld-card ${isCaught(id) ? 'caught' : ''} ${reref.has(id) ? 'reref' : ''} ${id === m.seed ? 'seed' : ''} ${hue != null ? `h${hue}` : ''} ${freshCard(id) ? 'fresh' : ''} ${busy.has(id) ? 'busy' : ''} ${dim(id) ? 'dim' : ''}`}
          style=${style(id)} ...${hc.bind(`n:${id}`, id, nodeAnchor)} role="button"
          aria-label=${`${id === m.seed ? 'The start' : label(id)}${fin > 1e-4 && id !== m.seed ? `, ${amt(fin)} ${isMixture ? per : 'of the starting boxes end as this'}` : ''}`}>
          <g class="ld-grow">
            <rect x=${-w / 2} y=${-h / 2} width=${w} height=${h} rx="6" class="ld-card-bg" />
            ${img ? html`<image href=${img} x=${-w / 2 + 3} y=${-h / 2 + 3} width=${w - 6} height=${h - 6} preserveAspectRatio="xMidYMid meet" />`
              : html`<text y="5" class="ld-unknown">?</text>`}
            ${knownIds.has(id) && html`<text x=${w / 2 - 8} y=${-h / 2 + 13} class="ln-known" title="reused from an earlier run">↺</text>`}
            <text y=${h / 2 + 15} class="ld-name" style="font-size:12.5px">${short(id === m.seed ? nameOf(id) || 'start' : nm, 17)}</text>
            ${id !== m.seed && fin > 1e-4 && html`<text y=${h / 2 + 30} class="ld-name ld-amt" style="font-size:12px">${amt(fin)} ${isMixture ? 'of the feed' : 'of boxes'}</text>`}
          </g></g>`;
      })}</g>
      ${sel && html`<${LaneGauge} sel=${sel} f=${f} x=${laneX} y=${laneTop - 40} />`}
      <text x=${laneX} y=${laneTop + 14} class="ld-lane-head">${outcomes === series ? `What forms, ${per}` : 'Outcomes as whole mixtures'}</text>
      <text x=${laneX} y=${laneTop + 34} class="ld-lane-sub">${outcomes === series
        ? (lane.length ? `at the end of the simulated time${cond.tauKnown ? ` (${fmtTime(cond.tau)})` : ''}` : tl.mixture?.pending ? 'nothing merged yet: the seeds are running' : 'nothing formed yet')
        : 'share of the starting boxes that end as each'}</text>
      ${lane.map((p) => {
        const nm = p.mol ? molName(p.key) : label(p.id), img = p.mol ? molImg(p.key) : imgOf(p.id);
        return html`<g class=${`ld-lane h${p.hue ?? 'x'}`} key=${`ln${p.key}`} ...${p.mol ? hc.bind(`m:${p.key}`, p.key, nodeAnchor) : hc.bind(`n:${p.id}`, p.id, nodeAnchor)}
          aria-label=${p.mol ? `${nm}: ${amt(p.fin)} ${per} at the end` : `${nm}: ${amt(p.fin)} of the starting boxes end as this`}>
          <rect x=${p.x} y=${p.y - 39} width="310" height="78" rx="8" class="ld-lane-bg" />
          <rect x=${p.x + 4} y=${p.y - 35} width="96" height="70" rx="6" class="ld-card-bg" />
          ${img && html`<image href=${img} x=${p.x + 6} y=${p.y - 33} width="92" height="66" preserveAspectRatio="xMidYMid meet" />`}
          <text x=${p.x + 112} y=${p.y - 8} class="ld-lane-pct">${amt(p.fin)}</text>
          <text x=${p.x + 112} y=${p.y + 13} class="ld-lane-name">${short(p.mol ? molTell(p.key) || molHead(p.key) : nm, 22)}</text>
          ${p.mol && molOf[p.key]?.mirror && html`<text x=${p.x + 300} y=${p.y - 13} class="ld-lane-sub end">+ mirror image</text>`}
          <rect x=${p.x + 112} y=${p.y + 24} width="186" height="7" rx="3.5" class="ld-bar-bg" />
          <rect x=${p.x + 112} y=${p.y + 24} width=${Math.max(2, (186 * (p.fin || 0)) / laneMax)} height="7" rx="3.5" class="ld-bar" />
        </g>`;
      })}
    </svg>
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

  const strip = html`<${Strip} t=${f.t} tEnd=${m.frames[m.frames.length - 1].t} atEnd=${cur === m.frames.length - 1} running=${running} q=${tl.activity?.queue} counts=${f.counts} nSteps=${f.steps.length} products=${series.slice(0, 3)}
    left=${left} per=${per} steady=${f.t - steadySince} imgOf=${molImg} nameOf=${molName} tellOf=${molTell} cond=${cond} pending=${!f.steered} basis=${f.basis}
    live0=${followLive && job?.started ? job.started : null} steering=${steeringOf(job)} merged=${String(jobId).startsWith('mix:') ? (tl.mixture?.pending ? 'none' : 'some') : null}
    reactor=${rx.runs ? { running: rx.running, runs: rx.runs, recent: lastTen, events: rx.events.filter((e) => e.t <= tNow + 0.5).length,
      open: () => openJob(apiJob, apiJob === jobId ? 'reactor' : 'live') } : null} />`;
  const table = html`<div class="ld-tables">${!isMixture && html`<h3 class="ld-table-h">What forms, ${per} <span class="muted small">at the end of the simulated time; a molecule counted in every state that holds it</span></h3>
    <${MoleculeTable} f=${f} molImg=${molImg} molName=${molName} molOf=${molOf} label=${label} hueOf=${hueOf} />
    <h3 class="ld-table-h">Outcomes as whole mixtures <span class="muted small">the box's states, as shares of the starting boxes</span></h3>`}
    <${SpeciesTable} m=${m} f=${f} imgOf=${imgOf} label=${label} madeBy=${madeBy} hueOf=${hueOf} smiles=${tl.smiles} unit=${isMixture ? per : 'share of the starting boxes'} /></div>`;
  const mixture = String(jobId).startsWith('mix:');
  return html`<div class="ld-page ld-page-b">
    ${!mixture && html`<div class="ln-head"><h2>${example ? 'Live network · example' : job?.title?.replace(/^Reaction network expansion:\s*/, 'Network: ') || 'Live network'}</h2>
      ${running && html`<span class="ln-live" title="Updating as the explorer works">● live</span>`}</div>`}
    ${strip}
    <div class="ld-viewbar">
      <div class="segmented small" role="tablist" aria-label="Show the network as">
        <button role="tab" aria-selected=${view === 'scheme'} class=${view === 'scheme' ? 'on' : ''} onClick=${() => setView('scheme')}>Scheme</button>
        <button role="tab" aria-selected=${view === 'table'} class=${view === 'table' ? 'on' : ''} onClick=${() => setView('table')}>Table</button></div>
      ${view === 'scheme' && !isMixture && html`<span class="small muted">What forms:</span><div class="segmented small" role="tablist" aria-label="What forms, counted">
        <button role="tab" aria-selected=${outcome === 'mol'} class=${outcome === 'mol' ? 'on' : ''} onClick=${() => setOutcome('mol')}
          title="Each molecule, counted in every state of the box that holds it">Molecules</button>
        <button role="tab" aria-selected=${outcome === 'mix'} class=${outcome === 'mix' ? 'on' : ''} onClick=${() => setOutcome('mix')}
          title="The box's states: how many of the starting boxes end as each whole mixture">Whole mixtures</button></div>`}
      <span class="small muted">${view === 'scheme' ? 'hover, focus or tap a structure or a route for details; tap again to let go' : isMixture ? `the ${f.species.length} molecules that matter most, by what there is at the end` : 'per molecule, and the box\'s states'}</span>
    </div>
    ${view === 'scheme' ? stage : table}
    ${view === 'scheme' && html`<${Key} cond=${cond} noFlux=${noFlux} anyBelow=${edges.some((e) => e.below && shownEdges.includes(e))}
      anyUnchecked=${shownEdges.some((e) => e.irc === -1)} known=${knownIds.size} running=${running} anyEvents=${rx.events.length > 0} />`}
    ${controls}
    ${extra}
    <div class="ld-details">
      ${jobId && !String(jobId).startsWith('mix:') && html`<${Closer} jobId=${jobId} job=${job} activity=${tl.activity} running=${running} />`}
      <${Corrections} events=${tl.events || []} t=${f.t} smiles=${tl.smiles} names=${m.names} />
    </div>
  </div>`;
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
  return html`<section class="ln-closer">
    <span class="section-title">Look closer</span>
    <div class="ln-doors">
      ${reactor && html`<button class="btn small" onClick=${() => openJob(jobId, 'reactor')} title="The molecular dynamics that proposes the reactions">Reactor MD</button>`}
      <button class="btn small" onClick=${() => openJob(jobId, 'live')} title="Each proposed reaction and its path search, live">Path searches</button>
      <button class="btn small" onClick=${() => openJob(jobId, 'result')} disabled=${job?.status === 'queued'}>Results</button>
      <button class="btn small" onClick=${() => openJob(jobId, 'log')}>Log</button>
    </div>
    ${done.length > 0 && html`<div class="ln-done small muted">${running ? 'Done so far: ' : 'Work done: '}${done.map(([k, v]) => html`<span title=${v.failed ? `${v.failed} failed` : ''}><b>${v.done}</b> ${WORK[k] || k}</span>`)}</div>`}
    ${activity?.reactor?.distinct > 0 && html`<span class="small muted">The reactor found ${activity.reactor.distinct} kind${activity.reactor.distinct === 1 ? '' : 's'} of product in ${activity.reactor.events} reaction events
        (${activity.reactor.runs} MD runs); the run optimizes and connects them next</span>
      <div class="ln-reactor">${activity.reactor.products.slice(0, 6).map((p) => html`<div class="ln-rp" title=${`${p.products} · seen ${p.count}×`}>
        <img src=${`/api/jobs/${jobId}/reactor-product/${encodeURIComponent(p.run)}/${p.event}`} alt=${p.products} loading="lazy" />
        <span>${strip_(p.products)}${p.unusual ? html` <i title="No Lewis structure fits it: drawn from its geometry">unusual bonding</i>` : ''}</span><b>${p.count}×</b></div>`)}</div>`}
    ${sims.length > 0 && html`<div class="ln-sims">${sims.slice(0, 5).map((s) => html`<button class="ln-sim" onClick=${() => openSim(s.stream)} title=${`${s.label}${s.caption ? ` · ${s.caption}` : ''}: open it`}>
      ${drawable(s.from) ? html`<img src=${depictUrl(s.from, 120, 90, drawable(s.to) ? s.to : null)} alt=${s.from} />` : html`<span class="ln-sim-txt">${strip_(s.from || '?')}</span>`}
      <span class="ln-sim-arrow">→</span>
      ${drawable(s.to) ? html`<img src=${depictUrl(s.to, 120, 90, drawable(s.from) ? s.from : null)} alt=${s.to} />` : html`<span class="ln-sim-txt">${strip_(s.to || '?')}</span>`}
      <span class="small muted">${(s.label || '').replace(/ from species.*/, '')}</span></button>`)}</div>`}
  </section>`;
}

// ---------------------------------------------------------------- pieces
function Spark({ series, w = 150, h = 34 }) {
  const t1 = Math.max(1, ...series.flatMap((s) => s.pts.map((p) => p[0])));
  const v1 = Math.max(1, ...series.flatMap((s) => s.pts.map((p) => p[1])));
  return html`<svg class="ld-spark" width=${w} height=${h} viewBox=${`0 0 ${w} ${h}`} role="img" aria-label="Each leading molecule's amount over the run">
    <line x1="0" x2=${w} y1=${h - 1} y2=${h - 1} class="ld-spark-axis" />
    ${series.map((s) => html`<polyline class=${`ld-spark-line h${s.hue ?? 'x'}`} points=${s.pts.map(([t, v]) => `${(w * t) / t1},${h - 2 - ((h - 4) * v) / v1}`).join(' ')} />`)}
  </svg>`;
}

// Run time that ticks while the run is followed live (frames come only with each kinetics refresh).
function LiveClock({ t0 }) {
  const [, tick] = useState(0);
  useEffect(() => { const id = setInterval(() => tick((x) => x + 1), 1000); return () => clearInterval(id); }, []);
  return fmtClock(Math.max(0, Date.now() / 1000 - t0));
}

function Strip({ t, tEnd, atEnd, running, q, counts, nSteps, products, left = [], per, steady, imgOf, nameOf, tellOf = () => '', cond, pending, steering, reactor, live0 = null, merged = false }) {
  const ph = running && atEnd ? phaseOf(q) : null;
  return html`<div class="ld-strip" role="status" aria-live="off">
    <div class="ld-s-time" title=${merged ? 'The mixture as merged so far' : 'Run time'}>
      <span class=${`ld-s-dot ${running ? 'live' : ''}`}></span>
      ${!merged && html`<b class="mono">${live0 != null ? html`<${LiveClock} t0=${live0} />` : fmtClock(t)}</b>`}
      ${cond.budget && html`<span class="ld-s-budget" title=${`of a ${fmtClock(cond.budget)} time budget`}><i style=${`width:${Math.min(100, (100 * t) / cond.budget)}%`}></i></span>`}
    </div>
    <div class="ld-s-phase">
      <b>${merged ? (merged === 'none' ? (running ? 'Seeds running; nothing merged yet' : 'Nothing was merged') : running ? 'Seeds running; merged so far' : 'Merged network') : ph ? ph.text : running ? (atEnd ? 'Starting' : `Replay · ${fmtClock(t)} of ${fmtClock(tEnd)} so far`) : atEnd ? 'Finished' : `Replay · ${fmtClock(t)} of ${fmtClock(tEnd)}`}</b>
      <span class="small muted">${ph ? ph.detail : `${cond.T ? `${Math.round(cond.T)} K` : ''}${cond.tauKnown ? ` · ${fmtTime(cond.tau)} simulated` : ''}`}</span>
    </div>
    <div class="ld-s-counts"><span><b>${counts.species}</b> species</span><span><b>${counts.steps ?? nSteps}</b> steps</span></div>
    ${reactor && html`<button class=${`ld-s-reactor ${reactor.running ? 'on' : ''}`} onClick=${reactor.open}
        title="The reactor's molecular dynamics that proposes the reactions: open it">
      <b>Reactor MD ›</b><span class="small muted">${reactor.running ? `${reactor.running} trajector${reactor.running === 1 ? 'y' : 'ies'} running · ` : ''}${reactor.recent
        ? `${reactor.recent} event${reactor.recent === 1 ? '' : 's'} in the last ${WINDOW_S} s` : `${reactor.events} events`}</span></button>`}
    <div class="ld-s-products" title=${`What forms at the end of the simulated time, ${per} (a molecule counted in every state of the box that holds it)`}>
      ${(products.length > 0 || left.length > 0) && html`<span class="small muted ld-s-per">${per}:</span>`}
      ${products.map((p) => html`<span class=${`ld-s-prod h${p.hue ?? 'x'}`} title=${`${nameOf(p.key)}: ${amt(p.fin)} ${per}`}>
        ${imgOf(p.key, 120, 90) && html`<img src=${imgOf(p.key, 120, 90)} alt=${nameOf(p.key)} />`}${tellOf(p.key) && html`<span class="ld-s-pname">${short(tellOf(p.key), 26)}</span>`}<b>${amt(p.fin)}</b></span>`)}
      ${left.map((l) => html`<span class="ld-s-prod start" title=${`${l.name}, left of the start at the end of the simulated time`}>
        <span class="ld-s-pname">${left.length > 1 ? `${short(l.name, 14)} ` : ''}left</span><b>${amt(l.fin)}</b></span>`)}
      ${products.length > 0 && html`<${Spark} series=${products} />`}
      <span class=${`small ld-s-steady ${pending ? 'pending' : ''}`} title="How long the leading molecule and its amount (within 0.02) have held, in run time">
        ${pending ? 'energies wait for DFT' : merged ? '' : products.length ? html`answer held <b>${steady >= 60 ? `${(steady / 60).toFixed(1)} min` : `${Math.round(steady)} s`}</b>` : 'nothing formed yet'}
        ${steering && html`<br /><span title="The level the live kinetics' energies are steered to (DFT single points)">energies: <b>${steering}</b></span>`}</span>
    </div>
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
    <span class="ld-k"><svg width="78" height="12" aria-hidden="true"><line x1="2" y1="6" x2="24" y2="6" class="ld-kl fast" /><line x1="28" y1="6" x2="50" y2="6" class="ld-kl run" /><line x1="54" y1="6" x2="76" y2="6" class="ld-kl slow" /></svg>
      how fast at ${T}: half-life under ${fmtTime(cond.tau / 100)} · under ${fmtTime(cond.tau)}${cond.tauKnown ? ' (the simulated time)' : ''} · longer</span>
    <span class="ld-k"><svg width="34" height="12" aria-hidden="true"><line x1="2" y1="6" x2="24" y2="6" class="ld-kl main h0" /><path d="M24,1 L32,6 L24,11 z" class="ld-arrhead h0" /></svg>
      ${noFlux ? 'likely route to a product (lowest barriers; flows not recorded)' : 'material flows; thicker = more'}, in the product's colour</span>
    <span class="ld-k"><svg width="16" height="12" aria-hidden="true"><circle cx="8" cy="6" r="3" class="ld-kdot" /></svg>a species found that holds little or no material yet (hover for the structure);
      each column holds the species found one, 2, … reactions from the start</span>
    ${anyBelow && html`<span class="ld-k"><svg width="30" height="12" aria-hidden="true"><line x1="2" y1="6" x2="28" y2="6" class="ld-kl below" /></svg>TS below its ends: no barrier</span>`}
    ${anyUnchecked && html`<span class="ld-k"><svg width="30" height="12" aria-hidden="true"><line x1="2" y1="6" x2="28" y2="6" class="ld-kl below" /></svg>unconfirmed: no IRC yet</span>`}
    ${known > 0 && html`<span class="ld-k"><b class="ln-known-k">↺</b>reused from an earlier run (${known})</span>`}
    ${running && html`<span class="ld-k"><svg width="30" height="12" aria-hidden="true"><line x1="2" y1="6" x2="28" y2="6" class="ld-probe" /></svg>searched now</span>`}
    ${anyEvents && html`<span class="ld-k"><svg width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="7" class="ld-ghost-bg" /></svg>a reaction event in the reactor's MD (click for its movie)</span>`}
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
  // per molecule (here every species is one): mol per mol of feed, the feed's own molecules left
  const fin = (x) => (x.final_concentration || 0) / fedTotal;
  frame.yields = mols.filter((x) => !feed.has(x.molecule) && fin(x) >= 1e-9).sort((a, b) => fin(b) - fin(a)).slice(0, 16).map((x) => [x.molecule, fin(x), x.molecule]);
  frame.left = mx.pending ? [] : mols.filter((x) => feed.has(x.molecule)).map((x) => [x.molecule, fin(x)]);
  const molecules = Object.fromEntries(mols.map((x) => [x.molecule, { name: names?.[x.molecule] || null, species: x.molecule }]));
  return { frames: [frame], smiles, names, explored, co, molecules, start: [...feed], mixture: { feed: [...feed], tau, seeds: mx.seeds || [], generation: mx.generation, pending: !!mx.pending },
    command: tau != null ? `--time ${tau}` : '' };
}

export function MixtureScheme({ mixture, names, jobId, job, running, extra = null }) {
  const tl = useMemo(() => mixtureTimeline(mixture, names), [JSON.stringify(mixture)]);
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
    api.get(`/api/jobs/${encodeURIComponent(apiJob)}/reactor/events/${ev.k}?name=${encodeURIComponent(ev.run)}&frames=40${ev.seed ? `&seed=${encodeURIComponent(ev.seed)}` : ''}`)
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
      <${EventPlayer} jobId=${apiJob} event=${ev.k} runName=${ev.run} seed=${ev.seed} height=${360} />
    </div></div>`;
}
