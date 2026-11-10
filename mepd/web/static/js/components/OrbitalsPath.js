// Orbitals along a path (`mepd orbitals`, op "orbitals"): the page of the calculation.
//
// Left, the correlation diagram: every orbital around the HOMO–LUMO gap as a line through the path's
// frames, followed by overlap (mepd/orbital_path.py), so a line keeps its character through a crossing.
// Filled orbitals solid, empty ones dashed, the gap shaded, the energy profile on top on the same axis.
// Right, the chosen orbital in 3D at the chosen frame, and what it is. On top, what happens, in words:
// hovering a sentence lights its orbitals, clicking it shows them. One frame cursor for everything.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, prefs, state, useStore } from '../store.js';
import { PHONE_QUERY } from '../util.js';
import { drawOrbital, isoFor } from './Viewer3D.js';
import { Why } from './Why.js';

const STICKS = { stick: { radius: 0.11 }, sphere: { scale: 0.22 } };
const HARTREE_KCAL = 627.509474;
// The drawing's own units: a phone draws a narrower diagram so its text stays readable.
const GEOM = { desk: { W: 780, L: 150, R: 128, PT: 84, PB: 30, H: 480, fs: 11.5 },
  phone: { W: 430, L: 104, R: 86, PT: 76, PB: 28, H: 430, fs: 10.5 } };
const COARSE = !!window.matchMedia?.('(pointer: coarse)').matches;
const FUNC = { 'wb97x-d3bj': 'ωB97X-D3BJ', b3lyp: 'B3LYP', pbe0: 'PBE0', pbe: 'PBE', bp86: 'BP86', tpssh: 'TPSSh',
  r2scan: 'r²SCAN', 'm06-2x': 'M06-2X', 'cam-b3lyp': 'CAM-B3LYP', 'wb97m-v': 'ωB97M-V', 'b97m-v': 'B97M-V', hf: 'HF' };
const prettyBasis = (b) => (b || '').replace(/^def2-(\w+)/i, (m, x) => `def2-${x.toUpperCase()}`)
  .replace(/^(aug-)?cc-p(v\w+)/i, (m, a, x) => `${a || ''}cc-p${x.toUpperCase().replace(/^V/, 'V')}`)
  .replace(/^(\d[\d+-]*)g/i, (m, x) => `${x}G`);
export const levelName = (lv) => `${FUNC[lv?.functional] || lv?.functional || '?'}/${prettyBasis(lv?.basis)}`;

// ------------------------------------------------------------------ words

const atomName = (sym, a) => `${sym[a]}${a + 1}`;
// Short (diagram labels): σ C1–Br2, σ* C1–Br2, π* C1–O5, n Cl6, n O2,O5, Cl6 empty, C1·Br2·Cl6.
function shortWhat(sym, w) {
  if (!w) return '';
  const at = w.atoms.map((a) => atomName(sym, a));
  const g = w.sym === 'pi' ? 'π' : 'σ';
  if (w.kind === 'bond') return `${g} ${at[0]}–${at[1]}`;
  if (w.kind === 'antibond') return `${g}* ${at[0]}–${at[1]}`;
  if (w.kind === 'lone') return `n ${at[0]}`;
  if (w.kind === 'lones') return `n ${at[0]},${at[1]}`;
  if (w.kind === 'atom') return `${at[0]} empty`;
  return at.join('·');
}
// Long (the card, tooltips), as the backend's story words it.
function longWhat(sym, w) {
  if (!w) return '';
  const at = w.atoms.map((a) => atomName(sym, a));
  const g = w.sym === 'pi' ? 'π' : 'σ';
  if (w.kind === 'bond') return `${at[0]}–${at[1]} ${g} bond`;
  if (w.kind === 'antibond') return `${at[0]}–${at[1]} ${g}* (antibonding)`;
  if (w.kind === 'lone') return `lone pair on ${at[0]}`;
  if (w.kind === 'lones') return `lone-pair combination on ${at[0]} and ${at[1]}`;
  if (w.kind === 'atom') return `empty orbital on ${at[0]}`;
  return `spread over ${at.join(', ')}`;
}
const relName = (rel) => (rel === 0 ? 'HOMO' : rel === 1 ? 'LUMO' : rel < 0 ? `HOMO−${-rel}` : `LUMO+${rel - 1}`);
const sameWhat = (a, b) => a && b && a.kind === b.kind && a.atoms.join() === b.atoms.join();

function splitXyz(text) {
  if (!text) return [];
  const lines = text.split('\n'), out = [];
  for (let i = 0; i < lines.length;) {
    const n = parseInt(lines[i], 10);
    if (!n) { i += 1; continue; }
    out.push(lines.slice(i, i + n + 2).join('\n') + '\n');
    i += n + 2;
  }
  return out;
}

// ------------------------------------------------------------------ the data, shaped for drawing

function useDoc(job) {
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);
  const live = ['queued', 'running'].includes(job.status);
  useEffect(() => {
    let stop = false, timer = null;
    const load = async () => {
      try {
        const d = await api.get(`/api/jobs/${job.id}/path-orbitals`);
        if (!stop) { setDoc(d.status === 'waiting' ? null : d); setError(null); }
      } catch (e) { if (!stop) setError(e.message); }
      if (!stop && live) timer = setTimeout(load, 1500);
    };
    load();
    return () => { stop = true; clearTimeout(timer); };
  }, [job.id, job.status]);
  return { doc, error, live };
}

function shape(doc) {
  const sym = doc.symbols;
  const level = new Map();      // `${k}:${i}` -> {e, occ, rel, what}
  doc.frames.forEach((f, k) => f && f.orbitals.forEach((o) => level.set(`${k}:${o.i}`, o)));
  const lines = doc.lines.map((ln) => {
    const pts = ln.points.map(([k, i, sign]) => ({ k, i, sign, ...(level.get(`${k}:${i}`) || {}) })).filter((p) => p.e != null);
    const a = pts[0], b = pts[pts.length - 1];
    return {
      id: ln.id, pts, a, b,
      changes: !!(a && b && !sameWhat(a.what, b.what)),
      crosses: !!(a && b && (a.occ > 0.5) !== (b.occ > 0.5)),
      frontier: pts.some((p) => p.rel === 0 || p.rel === 1),
    };
  }).filter((l) => l.pts.length);
  const done = doc.frames.map((f) => !!f);
  const E = doc.energies.filter((x) => x != null);
  const e0 = doc.energies.find((x) => x != null);
  const prof = doc.energies.map((x) => (x == null ? null : (x - e0) * HARTREE_KCAL));
  const allE = lines.flatMap((l) => l.pts.map((p) => p.e));
  // The view opens on HOMO−5 .. LUMO+2 (higher empty orbitals of a small basis crowd the top otherwise).
  const nearE = lines.flatMap((l) => l.pts.filter((p) => p.rel >= -5 && p.rel <= 3).map((p) => p.e));
  return { sym, lines, done, prof, hasE: E.length > 1,
    emin: Math.min(...allE), emax: Math.max(...allE), nmin: Math.min(...nearE), nmax: Math.max(...nearE),
    smax: doc.s[doc.s.length - 1] || 1 };
}

// ------------------------------------------------------------------ the correlation diagram

function Diagram({ doc, d, frame, setFrame, sel, setSel, lit, phone, all }) {
  const svg = useRef(null);
  const [hover, setHover] = useState(null);     // {id, k, x, y}
  const { W, L, R, PT, PB, H, fs } = GEOM[phone ? 'phone' : 'desk'];
  const pad = 0.6;
  const lo = Math.floor((all ? d.emin : d.nmin) - pad), hi = Math.ceil((all ? d.emax : d.nmax) + pad);
  const inView = (e) => e >= lo - 0.01 && e <= hi + 0.01;
  const X = (s) => L + (s / d.smax) * (W - L - R);
  const Y = (e) => PT + (1 - (e - lo) / (hi - lo)) * (H - PT - PB);
  const kAtX = (x) => {
    let best = 0, dist = Infinity;
    doc.s.forEach((s, k) => { if (d.done[k] && Math.abs(X(s) - x) < dist) { dist = Math.abs(X(s) - x); best = k; } });
    return best;
  };
  const toLocal = (ev) => {
    const r = svg.current.getBoundingClientRect();
    return [(ev.clientX - r.left) * (W / r.width), (ev.clientY - r.top) * (H / r.height)];
  };
  const lineNear = (k, y) => {
    let best = null, dist = COARSE ? 24 : 12;
    for (const ln of d.lines) {
      const p = ln.pts.find((q) => q.k === k);
      if (p && Math.abs(Y(p.e) - y) < dist) { dist = Math.abs(Y(p.e) - y); best = ln; }
    }
    return best;
  };
  const dragging = useRef(false);
  const onMove = (ev) => {
    const [x, y] = toLocal(ev);
    const k = kAtX(x);
    if (dragging.current) { setFrame(k); return; }
    const ln = y > PT - 4 ? lineNear(k, y) : null;
    setHover(ln ? { id: ln.id, k, x, y } : null);
  };
  const onDown = (ev) => {
    const [x, y] = toLocal(ev);
    const k = kAtX(x);
    const ln = y > PT - 4 ? lineNear(k, y) : null;
    if (ln) setSel(ln.id);
    setFrame(k);
    dragging.current = true;
    svg.current.setPointerCapture?.(ev.pointerId);
  };
  const onUp = () => { dragging.current = false; };

  // The gap: between the HOMO and the LUMO of every computed frame.
  const gap = useMemo(() => {
    const top = [], bot = [];
    doc.frames.forEach((f, k) => {
      if (!f) return;
      const h = f.orbitals.find((o) => o.rel === 0), u = f.orbitals.find((o) => o.rel === 1);
      if (h && u) { bot.push([X(doc.s[k]), Y(h.e)]); top.push([X(doc.s[k]), Y(u.e)]); }
    });
    if (bot.length < 2) return null;
    return [...bot, ...top.reverse()].map((p) => p.join(',')).join(' ');
  }, [doc, d, H, W, lo, hi]);

  // The energy profile, in the band on top.
  const prof = useMemo(() => {
    if (!d.hasE) return null;
    const vals = d.prof.filter((x) => x != null);
    const pmin = Math.min(...vals), pmax = Math.max(...vals, pmin + 1);
    const PY = (v) => 14 + (1 - (v - pmin) / (pmax - pmin)) * (PT - 40);
    const pts = d.prof.map((v, k) => (v == null ? null : [X(doc.s[k]), PY(v), v])).filter(Boolean);
    const kTs = doc.ts_frame != null && d.prof[doc.ts_frame] != null ? doc.ts_frame : null;
    const end = pts[pts.length - 1];
    const fmt = (v) => `${v > 0.05 ? '+' : v < -0.05 ? '−' : ''}${Math.abs(v).toFixed(1)}`;
    return { path: pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(''),
      area: `M${pts[0][0]},${PT - 22}` + pts.map((p) => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('') + `L${end[0]},${PT - 22}Z`,
      marks: [{ x: pts[0][0], y: pts[0][1], t: '0', a: 'start' },
        kTs != null && { x: X(doc.s[kTs]), y: PY(d.prof[kTs]), t: fmt(d.prof[kTs]), a: 'middle' },
        pts.length > 1 && { x: end[0], y: end[1], t: fmt(end[2]), a: 'end' }].filter(Boolean) };
  }, [doc, d, H, W]);

  // Labels at both ends, pushed apart so they never overlap.
  const edgeLabels = (end) => {
    const k = end ? d.done.lastIndexOf(true) : d.done.indexOf(true);
    const f = doc.frames[k];
    if (!f) return [];
    const raw = d.lines.map((ln) => ({ ln, p: ln.pts.find((q) => q.k === k) }))
      .filter((x) => x.p && x.p.rel >= (phone ? -2 : -4) && x.p.rel <= (phone ? 2 : 4) && inView(x.p.e))
      .map((x) => ({ ...x, text: shortWhat(d.sym, x.p.what), n: 1, y: Y(x.p.e) })).sort((a, b) => a.p.e - b.p.e);
    // Degenerate orbitals of one kind (a halide's three lone pairs) share one label: "n(Cl6) ×3".
    const items = [];
    for (const it of raw) {
      const last = items[items.length - 1];
      if (last && last.text === it.text && Math.abs(last.p.e - it.p.e) < 0.15) {
        last.n += 1;
        if (it.p.rel === 0 || it.p.rel === 1) last.p = it.p;
      } else items.push(it);
    }
    items.sort((a, b) => a.y - b.y);
    for (let i = 1; i < items.length; i += 1) items[i].y = Math.max(items[i].y, items[i - 1].y + fs + 2);
    return items;
  };
  const tsK = doc.ts_frame;
  const ticks = [];
  const tstep = hi - lo > 24 ? 5 : hi - lo > 12 ? 2 : 1;
  for (let e = Math.ceil(lo / tstep) * tstep; e <= hi; e += tstep) ticks.push(e);
  const cx = X(doc.s[frame] ?? 0);
  const active = (ln) => sel === ln.id || hover?.id === ln.id || lit.includes(ln.id);
  const dim = lit.length > 0 || hover != null;
  const segs = (ln) => {
    const out = [];
    for (let j = 1; j < ln.pts.length; j += 1) {
      const p = ln.pts[j - 1], q = ln.pts[j];
      out.push({ d: `M${X(doc.s[p.k]).toFixed(1)},${Y(p.e).toFixed(1)}L${X(doc.s[q.k]).toFixed(1)},${Y(q.e).toFixed(1)}`,
        occ: p.occ > 0.5 && q.occ > 0.5, key: j });
    }
    return out;
  };
  const hv = hover && d.lines.find((l) => l.id === hover.id);
  const hp = hv && hv.pts.find((p) => p.k === hover.k);
  return html`<div class="op-diagram">
    <svg ref=${svg} viewBox=${`0 0 ${W} ${H}`} style=${{ fontSize: `${fs}px` }} role="img" aria-label="Orbital energies along the path"
      onPointerMove=${onMove} onPointerDown=${onDown} onPointerUp=${onUp} onPointerLeave=${() => { setHover(null); onUp(); }}>
      ${prof && html`<g class="op-prof">
        <path class="op-prof-area" d=${prof.area} />
        <path class="op-prof-line" d=${prof.path} />
        <text x=${L - 8} y=${26} text-anchor="end">energy</text>
        <text x=${L - 8} y=${26 + fs + 2} text-anchor="end" class="op-sub">kcal/mol</text>
        ${prof.marks.map((m) => html`<text class="op-mark" x=${m.x} y=${m.y - 5} text-anchor=${m.a}>${m.t}</text>`)}
      </g>`}
      <line class="op-axis" x1=${L} x2=${W - R} y1=${PT - 16} y2=${PT - 16} />
      <defs><clipPath id="op-clip"><rect x=${L} y=${PT - 6} width=${W - L - R} height=${H - PT - PB + 6} /></clipPath></defs>
      ${ticks.map((e) => html`<g key=${e}><line class="op-grid" x1=${L} x2=${W - R} y1=${Y(e)} y2=${Y(e)} />
        <text class="op-tick" x=${L - 4} y=${Y(e) + 4} text-anchor="end">${e < 0 ? `−${-e}` : e}</text></g>`)}
      <text class="op-unit" x=${L - 8} y=${PT - 2} text-anchor="end">eV</text>
      <g clip-path="url(#op-clip)">
      ${gap && html`<polygon class="op-gap" points=${gap} />`}
      ${tsK != null && d.done[tsK] && html`<g class="op-ts"><line x1=${X(doc.s[tsK])} x2=${X(doc.s[tsK])} y1=${PT - 16} y2=${H - PB} /></g>`}
      ${tsK != null && d.done[tsK] && html`<text class="op-ts-label" x=${X(doc.s[tsK])} y=${H - PB + 18} text-anchor="middle">‡ TS</text>`}
      <text class="op-end" x=${L} y=${H - PB + 18}>start</text>
      <text class="op-end" x=${W - R} y=${H - PB + 18} text-anchor="end">end</text>
      ${(() => { const ln = d.lines.find((l) => l.id === sel); return ln && html`<g class="op-halo">${segs(ln).map((s) => html`<path key=${s.key} d=${s.d} />`)}</g>`; })()}
      ${d.lines.map((ln) => html`<g key=${ln.id} class=${`op-line ${active(ln) ? 'on' : dim ? 'dim' : ''} ${sel === ln.id ? 'sel' : ''} ${lit.includes(ln.id) ? 'lit' : ''}`}>
        ${segs(ln).map((s) => html`<path key=${s.key} d=${s.d} class=${s.occ ? 'occ' : 'virt'} />`)}
        ${ln.pts.length === 1 && html`<circle cx=${X(doc.s[ln.pts[0].k])} cy=${Y(ln.pts[0].e)} r="2" />`}
      </g>`)}
      </g>
      ${[false, true].map((end) => edgeLabels(end).map(({ ln, p, y, text, n }) => html`<text key=${`${end}${ln.id}`}
        class=${`op-lab ${active(ln) ? 'on' : ''} ${p.occ > 0.5 ? 'occ' : 'virt'}`} x=${end ? W - R + 6 : L - 28} y=${y + 4}
        text-anchor=${end ? 'start' : 'end'} onPointerDown=${(e) => { e.stopPropagation(); setSel(ln.id); }}>
        ${p.rel === 0 || p.rel === 1 ? html`<tspan class="op-fr">${relName(p.rel)} </tspan>` : ''}${text}${n > 1 ? ` ×${n}` : ''}</text>`))}
      <line class="op-cursor" x1=${cx} x2=${cx} y1=${8} y2=${H - PB} />
      ${(() => {
        const ln = d.lines.find((l) => l.id === sel);
        const p = ln && ln.pts.find((q) => q.k === frame);
        return p && html`<circle class="op-dot" cx=${cx} cy=${Y(p.e)} r="5" />`;
      })()}
    </svg>
    ${hv && hp && html`<div class="op-tip" style=${{ left: `${(hover.x / W) * 100}%`, top: `${(hover.y / H) * 100}%` }}>
      <b>${relName(hp.rel)}</b> ${hp.e.toFixed(2)} eV · ${hp.occ > 0.5 ? 'filled' : 'empty'}<br />${longWhat(d.sym, hp.what)}</div>`}
  </div>`;
}


// ------------------------------------------------------------------ where the camera looks from

const RCOV = { H: 0.31, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Br: 1.2, I: 1.39 };
function atomsOf(xyz) {
  const ln = xyz.trim().split('\n');
  return ln.slice(2, 2 + (+ln[0])).map((l) => { const [e, x, y, z] = l.trim().split(/\s+/); return { e, p: [+x, +y, +z] }; });
}
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit3 = (a) => { const n = Math.hypot(...a) || 1; return a.map((x) => x / n); };
// The main axis of a set of points (power iteration on their spread), optionally across `not`.
function axisOf(pts, not = null) {
  const c = [0, 1, 2].map((i) => pts.reduce((t, q) => t + q[i], 0) / pts.length);
  let v = not ? unit3(cross3(not, [0.31, 0.72, 0.62])) : [1, 0.5, 0.25];
  for (let it = 0; it < 40; it += 1) {
    const w = [0, 0, 0];
    for (const q of pts) {
      let r = sub3(q, c);
      if (not) r = sub3(r, not.map((x) => x * dot3(r, not)));
      const k = dot3(r, v);
      w[0] += r[0] * k; w[1] += r[1] * k; w[2] += r[2] * k;
    }
    if (Math.hypot(...w) < 1e-9) break;
    v = unit3(w);
  }
  return v;
}
// A camera (3Dmol's rotation quaternion) facing across the atoms whose bonds change between the path's
// ends, with that axis left to right and the start's side on the left, as in the diagram.
function cameraFor(frames) {
  if (frames.length < 2) return null;
  const A = atomsOf(frames[0]), B = atomsOf(frames[frames.length - 1]), M = atomsOf(frames[Math.floor(frames.length / 2)]);
  const bonded = (X, i, j) => Math.hypot(...sub3(X[i].p, X[j].p)) <= 1.3 * ((RCOV[X[i].e] || 1.2) + (RCOV[X[j].e] || 1.2));
  const moving = new Set();
  for (let i = 0; i < A.length; i += 1) for (let j = i + 1; j < A.length; j += 1) {
    if (bonded(A, i, j) !== bonded(B, i, j)) { moving.add(i); moving.add(j); }
  }
  const idx = moving.size >= 2 ? [...moving] : M.map((_, i) => i);
  let a = axisOf(idx.map((i) => M[i].p));
  // The start's side on the left: the atoms that lose a bond sit at +a... flip so they read right.
  const lose = idx.filter((i) => A.some((_, j) => j !== i && bonded(A, i, j) && !bonded(B, i, j)));
  if (lose.length) {
    const c = [0, 1, 2].map((k) => idx.reduce((t, i) => t + M[i].p[k], 0) / idx.length);
    const side = lose.reduce((t, i) => t + dot3(sub3(M[i].p, c), a), 0);
    if (side < 0) a = a.map((x) => -x);
  }
  const b = axisOf(M.map((q) => q.p), a);
  const c3 = unit3(cross3(a, b));
  const R = [a, unit3(cross3(c3, a)), c3];          // rows: model -> view
  const t = R[0][0] + R[1][1] + R[2][2];
  let q;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    q = [(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, s / 4];
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    const s = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2;
    q = [s / 4, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s];
  } else if (R[1][1] > R[2][2]) {
    const s = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2;
    q = [(R[0][1] + R[1][0]) / s, s / 4, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s];
  } else {
    const s = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2;
    q = [(R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, s / 4, (R[1][0] - R[0][1]) / s];
  }
  return q;
}

// ------------------------------------------------------------------ the 3D view of one orbital

function Orbital3D({ job, doc, d, frame, line, phone }) {
  const host = useRef(null), viewer = useRef(null), surfaces = useRef([]), shownFrame = useRef(null);
  const cache = useRef(new Map());
  const [pct, setPctState] = useState(() => prefs.get('orbitalPct', 85));
  const setPct = (v) => { setPctState(v); prefs.set('orbitalPct', v); };
  const [loading, setLoading] = useState(false);
  const touch = useMemo(() => !!window.matchMedia?.('(pointer: coarse)').matches, []);
  const [rotating, setRotating] = useState(false);
  const frames = useMemo(() => splitXyz(doc.xyz), [doc.xyz]);
  const p = line && line.pts.find((q) => q.k === frame);
  const height = phone ? 210 : 300;

  useEffect(() => {
    if (!window.$3Dmol || !host.current) return undefined;
    const bg = getComputedStyle(document.documentElement).getPropertyValue('--viewer-bg').trim() || '#ffffff';
    viewer.current = window.$3Dmol.createViewer(host.current, { backgroundColor: bg, antialias: true });
    const ro = new ResizeObserver(() => { viewer.current?.resize(); viewer.current?.render(); });
    ro.observe(host.current);
    return () => { ro.disconnect(); viewer.current?.clear(); viewer.current = null; };
  }, []);
  // The frame's atoms (the view kept from frame to frame), the orbital's atoms named.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !frames[frame]) return;
    const first = shownFrame.current == null;
    const view = first ? null : v.getView();
    v.removeAllModels();
    v.removeAllLabels();
    v.addModel(frames[frame], 'xyz');
    v.setStyle({}, STICKS);
    const atoms = p?.what?.atoms || [];
    const model = v.getModel();
    const all = model ? model.selectedAtoms({}) : [];
    atoms.forEach((a) => all[a] && v.addLabel(atomName(d.sym, a), { position: { x: all[a].x, y: all[a].y, z: all[a].z },
      fontSize: 11, fontColor: '#17324a', backgroundColor: '#ffffff', backgroundOpacity: 0.8, borderThickness: 0, inFront: true,
      screenOffset: { x: 12, y: -12 } }));
    if (first) {
      v.zoomTo();
      const q = cameraFor(frames);
      if (q) { const vw = v.getView(); v.setView([vw[0], vw[1], vw[2], vw[3], ...q]); }
      v.zoomTo();
      v.zoom(1.25);
    } else v.setView(view);
    shownFrame.current = frame;
    v.render();
  }, [frames, frame, p?.i, line?.id]);
  // The orbital: fetched once per (frame, orbital), drawn with the line's sign so its lobes keep their colours.
  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    if (!p) { drawOrbital(v, surfaces, null, 0); return; }
    const key = `${frame}_${p.i}`;
    let stop = false;
    const draw = (orb) => { if (!stop) drawOrbital(v, surfaces, orb, isoFor(orb, pct), p.sign); };
    const hit = cache.current.get(key);
    if (hit) { draw(hit); return () => { stop = true; }; }
    setLoading(true);
    api.get(`/api/jobs/${job.id}/path-orbitals/cube?k=${frame}&i=${p.i}`)
      .then((orb) => { cache.current.set(key, orb); draw(orb); })
      .catch(() => {}).finally(() => !stop && setLoading(false));
    return () => { stop = true; };
  }, [frame, p?.i, p?.sign, pct, frames]);
  // Ahead of the cursor: the next frames of the chosen orbital.
  useEffect(() => {
    if (!line) return;
    line.pts.filter((q) => q.k > frame && q.k <= frame + 2).forEach((q) => {
      const key = `${q.k}_${q.i}`;
      if (!cache.current.has(key)) {
        cache.current.set(key, null);
        api.get(`/api/jobs/${job.id}/path-orbitals/cube?k=${q.k}&i=${q.i}`).then((orb) => cache.current.set(key, orb)).catch(() => cache.current.delete(key));
      }
    });
  }, [frame, line?.id]);
  return html`<div class="op-3d">
    <div class="viewer-wrap">
      <div class="viewer" ref=${host} style=${{ height }}></div>
      ${touch && !rotating && html`<button class="viewer-guard" style=${{ height }} onClick=${() => setRotating(true)}
        aria-label="Rotate the molecule (a swipe here scrolls the page)"><span>Tap to rotate</span></button>`}
      ${touch && rotating && html`<button class="btn small viewer-done" onClick=${() => setRotating(false)}>Done</button>`}
      ${loading && html`<span class="op-loading small">drawing…</span>`}
      ${line && !p && html`<div class="op-none small">This orbital is not followed at this frame: it begins or ends elsewhere on the path.</div>`}
    </div>
    <div class="op-3d-tools small">
      <span><span class="op-swatch pos"></span>+ <span class="op-swatch neg"></span>− lobes</span>
      <label title="The surface encloses this share of the orbital's density (|ψ|²). Smaller: tighter lobes.">
        <input type="number" class="tiny" min="5" max="99" step="5" value=${pct}
          onChange=${(e) => { const v = +e.target.value; if (v > 0 && v < 100) setPct(v); }} /> % density</label>
    </div>
  </div>`;
}

// ------------------------------------------------------------------ the chosen orbital, in words

function OrbitalCard({ d, doc, line, frame, setFrame }) {
  if (!line) return html`<div class="op-card muted small">Click a line in the diagram to see that orbital.</div>`;
  const p = line.pts.find((q) => q.k === frame);
  const tsK = doc.ts_frame;
  const at = (k) => line.pts.find((q) => q.k === k);
  const stops = [['start', line.a], tsK != null && at(tsK) && ['‡ TS', at(tsK)], ['end', line.b]].filter(Boolean);
  return html`<div class="op-card">
    ${p ? html`<div class="op-card-head">
      <span class="op-card-name">${relName(p.rel)}</span>
      <span class="op-card-e">${p.e.toFixed(2)} eV</span>
      <span class=${`op-occ ${p.occ > 0.5 ? 'occ' : 'virt'}`}>${p.occ > 0.5 ? 'filled' : 'empty'}</span>
    </div>
    <div class="op-card-what">${longWhat(d.sym, p.what)}
      <span class="muted small">${p.what.atoms.map((a, j) => ` · ${atomName(d.sym, a)} ${Math.round(p.what.shares[j] * 100)} %`).join('')}</span></div>`
      : html`<div class="muted small">Not followed at this frame.</div>`}
    <div class="op-journey">
      ${stops.map(([where, q], j) => html`${j ? html`<span class="op-arrow">→</span>` : ''}<button class="op-stop" title=${`Go to the ${where}`}
        onClick=${() => setFrame(q.k)}><span class="op-stop-where">${where}</span>
        <span class="op-stop-what">${shortWhat(d.sym, q.what)}</span><span class="op-stop-e">${q.e.toFixed(1)} eV</span></button>`)}
    </div>
    ${line.crosses && html`<p class="op-flag small">Filled at one end and empty at the other: it crosses the frontier.</p>`}
  </div>`;
}

// ------------------------------------------------------------------ the page

const LEAD = 3;   // sentences shown before "more"

export function OrbitalsPath({ job }) {
  const { doc, error, live } = useDoc(job);
  const phone = useMemo(() => !!window.matchMedia?.(PHONE_QUERY).matches, []);
  const d = useMemo(() => (doc && doc.lines?.length ? shape(doc) : null), [doc]);
  const [frame, setFrameState] = useState(null);
  const [sel, setSel] = useState(null);
  const [lit, setLit] = useState([]);
  const [playing, setPlaying] = useState(false);
  const [more, setMore] = useState(false);
  const [all, setAll] = useState(false);
  const box = useRef(null);
  const doneKs = d ? d.done.map((x, k) => (x ? k : -1)).filter((k) => k >= 0) : [];
  // The keys work from the start (no click needed first).
  useEffect(() => { if (d) box.current?.focus({ preventScroll: true }); }, [!!d]);
  const setFrame = (k) => setFrameState(k);
  // Opens on the TS (or the last frame computed so far), on the orbital that is the HOMO there.
  useEffect(() => {
    if (!d || frame != null) return;
    const k0 = doc.ts_frame != null && d.done[doc.ts_frame] ? doc.ts_frame : doneKs[doneKs.length - 1];
    setFrameState(k0);
    const h = d.lines.find((ln) => ln.pts.some((p) => p.k === k0 && p.rel === 0));
    if (h) setSel(h.id);
  }, [d]);
  useEffect(() => {
    if (!playing || !doneKs.length) return undefined;
    const t = setInterval(() => setFrameState((k) => doneKs[(doneKs.indexOf(k) + 1) % doneKs.length]), 420);
    return () => clearInterval(t);
  }, [playing, doneKs.length]);
  const step = (dir) => {
    const j = doneKs.indexOf(frame);
    if (j >= 0) setFrame(doneKs[Math.min(Math.max(j + dir, 0), doneKs.length - 1)]);
  };
  // Up and down: the next orbital above or below the chosen one at this frame.
  const climb = (dir) => {
    if (!d) return;
    const here = d.lines.map((ln) => ({ ln, p: ln.pts.find((q) => q.k === frame) })).filter((x) => x.p).sort((a, b) => a.p.e - b.p.e);
    const j = here.findIndex((x) => x.ln.id === sel);
    const nx = here[j < 0 ? 0 : Math.min(Math.max(j + dir, 0), here.length - 1)];
    if (nx) setSel(nx.ln.id);
  };
  const onKey = (e) => {
    if (e.target.closest('input, select, textarea')) return;
    const k = { ArrowLeft: () => step(-1), ArrowRight: () => step(1), ArrowUp: () => climb(1), ArrowDown: () => climb(-1),
      ' ': () => setPlaying((x) => !x) }[e.key];
    if (k) { e.preventDefault(); k(); }
  };
  const line = d && d.lines.find((ln) => ln.id === sel);
  const total = doc?.frames?.length || 0;
  const computed = doc ? doc.frames.filter(Boolean).length : 0;
  const story = doc?.story || [];
  const lead = phone ? 2 : LEAD;
  const shown = more ? story : story.slice(0, lead);
  const goStory = (s) => {
    if (s.lines?.length) setSel(s.lines[0]);
    const ln = d?.lines.find((l) => l.id === s.lines?.[0]);
    const k = s.frame ?? (s.kind === 'move' && doc.ts_frame != null ? doc.ts_frame : s.kind === 'change' && ln ? ln.b.k : null);
    if (k != null) setFrame(k);
  };
  if (error) return html`<p class="warn-box small">${error}</p>`;
  if (!doc) return html`<div class="op-page"><p class="muted">${live ? 'Starting the first frame…' : 'No orbitals were written.'}</p></div>`;
  const tsHere = frame != null && frame === doc.ts_frame;
  return html`<div class="op-page" ref=${box} tabIndex="0" onKeyDown=${onKey}>
    <div class="op-top">
      <div class="op-level small">
        <span class="op-level-chip" title="Each frame: a DFT single point with linus, its orbitals followed to the next by their overlap">
          ${levelName(doc.level)}</span>
        <span class="muted" title=${`${computed} of ${total} frames computed, spread along the ${doc.path_frames} frames of ${doc.entry?.label || 'the path'}${doc.seconds ? `, in ${Math.round(doc.seconds)} s` : ''}`}>
          ${live ? `frame ${computed} of ${total}` : `${computed} frames`}${doc.entry?.label ? ` along ${doc.entry.label}` : ''}</span>
        ${doc.charge < 0 && !/(svpd|tzvpd|tzvppd|qzvpd|aug-|\+)/i.test(doc.level.basis || '')
          && html`<span class="op-caveat" title="An anion's extra electron needs diffuse functions (def2-SVPD, aug-cc-pVDZ, 6-31+G*). Without them its empty orbitals are confined by the basis: their energies and the gap are qualitative.">anion without diffuse functions: empty-orbital energies are rough</span>`}
        <${Why} id="op.orbitals" label="Orbitals along the path" />
      </div>
      ${live && html`<div class="op-progress" title=${`Frame ${computed} of ${total}`}><span style=${{ width: `${(100 * computed) / Math.max(total, 1)}%` }}></span></div>`}
    </div>
    ${story.length > 0 && html`<section class="op-story" aria-label="What happens">
      <h3>What happens</h3>
      <ul>${shown.map((s, j) => html`<li key=${j} class=${`op-s-${s.kind}`} onMouseEnter=${() => setLit(s.lines || [])} onMouseLeave=${() => setLit([])}
        onClick=${() => goStory(s)} title=${s.detail || 'Show these orbitals'}>${s.text}</li>`)}</ul>
      ${story.length > lead && html`<button class="btn-link small" onClick=${() => setMore(!more)}>${more ? 'Fewer' : `${story.length - lead} more`}</button>`}
    </section>`}
    ${d ? html`<div class="op-main">
      <div class="op-diag">
        <${Diagram} doc=${doc} d=${d} frame=${frame ?? 0} setFrame=${setFrame} sel=${sel} setSel=${setSel} lit=${lit} phone=${phone} all=${all} />
        <div class="op-bar">
          <button class="btn small" onClick=${() => setPlaying(!playing)} title="Play along the path (Space)">${playing ? '❚❚ Pause' : '▶ Play'}</button>
          <button class="btn small op-step" onClick=${() => step(-1)} title="Frame before (←)" aria-label="Frame before">‹</button>
          <button class="btn small op-step" onClick=${() => step(1)} title="Next frame (→)" aria-label="Next frame">›</button>
          <span class="small nowrap">frame ${doneKs.indexOf(frame) + 1} of ${doneKs.length}${tsHere ? html` · <b class="op-ts-tag">‡ TS</b>` : frame === doneKs[0] ? ' · start' : frame === doneKs[doneKs.length - 1] ? ' · end' : ''}</span>
          ${(d.emin < d.nmin - 0.3 || d.emax > d.nmax + 0.3) && html`<label class="small nowrap" title="Also the orbitals far below and above the gap">
            <input type="checkbox" checked=${all} onChange=${(e) => setAll(e.target.checked)} /> all orbitals</label>`}
          <span class="op-legend small" title=${phone ? 'Tap a line to choose an orbital; drag across to move along the path.'
            : 'Click a line to choose an orbital; drag across to move along the path. ← → frames, ↑ ↓ the orbital above or below, Space plays.'}>
            <span class="op-key occ"></span>filled <span class="op-key virt"></span>empty <span class="op-key gap"></span>gap</span>
        </div>
        <p class="op-hint small muted">${phone ? 'Tap a line · drag across' : 'Click a line · drag across · ← → ↑ ↓ · Space'}</p>
      </div>
      <div class="op-view"><${Orbital3D} job=${job} doc=${doc} d=${d} frame=${frame ?? 0} line=${line} phone=${phone} /></div>
      <div class="op-cardwrap"><${OrbitalCard} d=${d} doc=${doc} line=${line} frame=${frame ?? 0} setFrame=${setFrame} /></div>
    </div>` : html`<p class="muted">The first orbitals appear here once two frames are computed.</p>`}
  </div>`;
}

// ------------------------------------------------------------------ starting one, from a path in a result

const STATUS = { queued: 'queued', running: 'running', done: '', failed: 'failed', cancelled: 'cancelled', interrupted: 'stopped' };

// Under a path of a TS search or channels run: "Orbitals along this path", its level, and the runs on it.
export function PathOrbitalsLauncher({ job, entry }) {
  const op = useStore((s) => s.operations.find((o) => o.key === 'orbitals'));
  const runsKey = useStore((s) => Object.values(s.jobs).filter((j) => j.op === 'orbitals' && j.source_job === job.id)
    .map((j) => `${j.id}:${j.status}`).join('|'));
  const runs = useMemo(() => Object.values(state.jobs).filter((j) => j.op === 'orbitals' && j.source_job === job.id
    && (j.params?.entry || '') === entry.id).sort((a, b) => b.created - a.created), [runsKey, entry.id]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const props = op?.schema?.properties || {};
  const [values, setValues] = useState(() => prefs.get('pathOrbitals', {}));
  if (!op) return null;
  const v = (k) => values[k] ?? props[k]?.default;
  const setV = (k, x) => { const nv = { ...values, [k]: x }; setValues(nv); prefs.set('pathOrbitals', nv); };
  const labelOf = (k, x) => props[k]?.labels?.[x] || x;
  const run = async () => {
    setBusy(true);
    const what = (job.title || '').replace(/^(Transition state|TS|Reaction channels):\s*/, '');
    const out = await attempt(() => api.post('/api/jobs', { op: 'orbitals', source_job: job.id,
      label: `Orbitals: ${what}`, params: { entry: entry.id, functional: v('functional'), basis: v('basis'), frames: v('frames') } }));
    setBusy(false);
    if (out?.length) openJob(out[0].id);
  };
  const levelOf = (j) => levelName({ functional: j.params?.functional, basis: j.params?.basis });
  return html`<div class="op-launch">
    <div class="op-launch-row">
      <button class=${`btn small ${open ? '' : 'ghost'}`} disabled=${!op.available} title=${op.available ? op.summary : op.unavailable_reason}
        onClick=${() => setOpen(!open)}>Orbitals along this path${open ? '' : '…'}</button>
      ${runs.map((j) => html`<a key=${j.id} href="#" class="small" onClick=${(e) => { e.preventDefault(); openJob(j.id); }}>
        ${levelOf(j)}${STATUS[j.status] ? ` (${STATUS[j.status]})` : ''}</a>`)}
    </div>
    ${open && html`<div class="op-launch-form small">
      <label>Functional <select value=${v('functional')} onChange=${(e) => setV('functional', e.target.value)}>
        ${(props.functional?.enum || []).map((x) => html`<option value=${x}>${labelOf('functional', x)}</option>`)}</select></label>
      <label>Basis <select value=${v('basis')} onChange=${(e) => setV('basis', e.target.value)}>
        ${(props.basis?.enum || []).map((x) => html`<option value=${x}>${prettyBasis(x)}</option>`)}</select></label>
      <label title=${props.frames?.description || ''}>Frames <input type="number" class="tiny" min="2" value=${v('frames')}
        onChange=${(e) => setV('frames', Math.max(2, +e.target.value || 2))} /></label>
      <button class="btn small primary" disabled=${busy} onClick=${run}>${busy ? 'Starting…' : 'Run'}</button>
      <span class="muted">DFT single points with linus; seconds to minutes for small molecules.</span>
    </div>`}
  </div>`;
}
