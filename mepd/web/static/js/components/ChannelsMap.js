// Path map for `mepd channels`, read as mechanisms: each conformer pair's
// search becomes one route from reactant to product through its elementary
// steps (R -> I1 -> ... -> P), pairs through the same minima are one
// mechanism (one colour), and every route is drawn twice -- in two
// coordinates (the map) and as energy along its steps (the profile), hover
// and selection shared. The splits the search went through and the fitted
// energy surface are there on request. See mepd/web/channels_pes.py for the
// coordinates, the surface and how routes are read.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { prefs } from '../store.js';
import { Viewer3D } from './Viewer3D.js';
import { Why } from './Why.js';

const PALETTE = [[23, 50, 74], [31, 82, 120], [45, 118, 150], [80, 152, 160], [135, 183, 160], [196, 210, 160], [241, 229, 178], [250, 244, 222]];
const W = 660, H = 480, L = 56, R_ = 16, T = 14, B = 46;

function color(t) {
  t = Math.max(0, Math.min(1, t)) * (PALETTE.length - 1);
  const i = Math.min(PALETTE.length - 2, Math.floor(t)), f = t - i, a = PALETTE[i], b = PALETTE[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function ticks(a, b, n) {
  const span = b - a || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = []; for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(+t.toFixed(10)); return out;
}
function bilinear(G, xs, ys, x, y) {
  const nx = xs.length, ny = ys.length;
  const fx = (x - xs[0]) / (xs[nx - 1] - xs[0]) * (nx - 1), fy = (y - ys[0]) / (ys[ny - 1] - ys[0]) * (ny - 1);
  const i = Math.max(0, Math.min(nx - 2, Math.floor(fx))), j = Math.max(0, Math.min(ny - 2, Math.floor(fy)));
  const tx = fx - i, ty = fy - j, v = [G[j][i], G[j][i + 1], G[j + 1][i], G[j + 1][i + 1]];
  if (v.some((q) => q === null)) return null;
  return (1 - tx) * (1 - ty) * v[0] + tx * (1 - ty) * v[1] + (1 - tx) * ty * v[2] + tx * ty * v[3];
}
function contours(G, xs, ys, lev) {
  const seg = [];
  for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) {
    const c = [[xs[i], ys[j], G[j][i]], [xs[i + 1], ys[j], G[j][i + 1]], [xs[i + 1], ys[j + 1], G[j + 1][i + 1]], [xs[i], ys[j + 1], G[j + 1][i]]];
    if (c.some((p) => p[2] === null)) continue;
    const pts = [];
    for (let k = 0; k < 4; k++) {
      const a = c[k], b = c[(k + 1) % 4];
      if ((a[2] - lev) * (b[2] - lev) < 0) { const t = (lev - a[2]) / (b[2] - a[2]); pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]); }
    }
    if (pts.length === 2) seg.push(pts); else if (pts.length === 4) { seg.push([pts[0], pts[1]]); seg.push([pts[2], pts[3]]); }
  }
  return seg;
}


// Mechanism colours: a fixed categorical order (shell.css --mech-1..8, light
// and dark), A first; a ninth mechanism reuses the order.
const mechColor = (m) => `var(--mech-${(((m || 'A').charCodeAt(0) - 65) % 8 + 8) % 8 + 1})`;
// 'main' is the TS search that Sample more paths runs were added to.
const pairLabel = (pair) => {
  if (pair === 'main') return 'first search';
  const m = /^pair_(\d+)/.exec(pair || '');
  return m ? `pair ${+m[1] + 1}` : (pair || '').replace(/_/g, ' ');
};
const kcal = (v) => (v == null ? '—' : v.toFixed(1));
const seqText = (r) => r.minima.join(' → ') + (r.complete ? '' : ' → …');

// What the map and the profile need from a map: routes by pair, each path's
// place on a route, mechanisms with their routes, and which routes are lit.
function readRoutes(map, sel) {
  const routes = map?.routes || [];
  const byId = Object.fromEntries((map?.paths || []).map((p) => [p.id, p]));
  const onRoute = {};
  routes.forEach((r) => r.steps.forEach((s, i) => { onRoute[s.path] = { route: r, i }; }));
  const mechs = [];
  for (const r of routes) {
    let m = mechs.find((x) => x.id === r.mechanism);
    if (!m) mechs.push(m = { id: r.mechanism, routes: [] });
    m.routes.push(r);
  }
  for (const m of mechs) m.top = Math.min(...m.routes.map((r) => (r.top == null ? Infinity : r.top)));
  const lit = (r) => (sel.pair ? r.pair === sel.pair : sel.mech ? r.mechanism === sel.mech : true);
  const pairMech = Object.fromEntries(routes.map((r) => [r.pair, r.mechanism]));
  return { routes, byId, onRoute, mechs, lit, pairMech };
}

function Surface({ map, R, sel, onPick, hover, setHover, history, surface }) {
  const canvas = useRef(null);
  const { x: xs, y: ys, E } = map;
  const marks = map.ts || [];
  const ircPts = map.irc?.points || [];
  const shown = map.paths.filter((p) => history || R.onRoute[p.id]);
  const pts = shown.flatMap((p) => p.points).concat(marks.map((m) => [m.q[0], m.q[1], m.e]), ircPts);
  const allX = pts.map((q) => q[0]).concat(surface ? xs : []), allY = pts.map((q) => q[1]).concat(surface ? ys : []);
  const pad = (a, b) => { const d = (b - a) * 0.06 || 0.05; return [a - d, b + d]; };
  const [x0, x1] = pad(Math.min(...allX), Math.max(...allX)), [y0, y1] = pad(Math.min(...allY), Math.max(...allY));
  const energies = (E.flat ? E.flat() : []).filter((v) => v !== null).concat(pts.map((q) => q[2]).filter((v) => v !== null));
  const vmin = Math.min(...energies, 0), vmax = Math.max(Math.min(Math.max(...energies), vmin + 150), vmin + 5);
  const sx = (x) => L + (x - x0) / ((x1 - x0) || 1) * (W - L - R_), sy = (y) => T + (1 - (y - y0) / ((y1 - y0) || 1)) * (H - T - B);
  const ix = (px) => x0 + (px - L) / (W - L - R_) * (x1 - x0), iy = (py) => y0 + (1 - (py - T) / (H - T - B)) * (y1 - y0);

  useEffect(() => {
    const ctx = canvas.current?.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    if (!surface || !xs.length) return;
    const Ec = E.map((row) => row.map((v) => (v === null ? null : Math.min(v, vmax))));
    const img = ctx.createImageData(W - L - R_, H - T - B);
    for (let py = 0; py < H - T - B; py++) for (let px = 0; px < W - L - R_; px++) {
      const e = bilinear(Ec, xs, ys, ix(px + L), iy(py + T)), o = 4 * (py * (W - L - R_) + px);
      if (e === null) continue;
      const c = color((e - vmin) / (vmax - vmin)); img.data.set([c[0], c[1], c[2], 150], o);
    }
    ctx.putImageData(img, L, T);
  }, [map, surface, x0, x1, y0, y1]);

  const lev = surface ? ticks(vmin, vmax, 12) : [];
  const step = (lev[1] - lev[0]) || 10;
  const segs = surface && xs.length ? lev.flatMap((l) => contours(E, xs, ys, l)) : [];
  const poly = (points) => points.map((q) => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(' ');
  const routeOf = (p) => R.onRoute[p.id]?.route;
  const isLit = (p) => { const r = routeOf(p); return r ? R.lit(r) : false; };
  const anyDim = !!(sel.mech || sel.pair);
  // Off-route chains (the splits the search went through) first, then dim routes, then lit ones, the picked step last.
  const order = [...shown].sort((a, b) => (!!routeOf(a) - !!routeOf(b)) || (isLit(a) - isLit(b)) || ((a.id === sel.path) - (b.id === sel.path)));
  // Minima of the lit routes, one label per name and place.
  const minima = [];
  for (const r of R.routes) {
    if (!R.lit(r)) continue;
    for (const m of r.points) {
      const X = sx(m.q[0]), Y = sy(m.q[1]);
      const near = minima.find((o) => Math.hypot(o.X - X, o.Y - Y) < 9);
      if (near) { if (!near.names.includes(m.name)) near.names.push(m.name); continue; }
      minima.push({ X, Y, names: [m.name], e: m.e, mech: r.mechanism });
    }
  }
  return html`<div class="cmap-surf" style=${`aspect-ratio:${W}/${H}`}>
    <canvas ref=${canvas} width=${W} height=${H}></canvas>
    <svg viewBox=${`0 0 ${W} ${H}`} onMouseLeave=${() => setHover(null)}>
      <defs><clipPath id="cmapClip"><rect x=${L} y=${T} width=${W - L - R_} height=${H - T - B} /></clipPath></defs>
      <rect class="cmap-frame" x=${L} y=${T} width=${W - L - R_} height=${H - T - B} />
      ${!surface && ticks(x0, x1, 6).map((t) => html`<line class="cmap-grid" x1=${sx(t)} x2=${sx(t)} y1=${T} y2=${H - B} />`)}
      ${!surface && ticks(y0, y1, 6).map((t) => html`<line class="cmap-grid" x1=${L} x2=${W - R_} y1=${sy(t)} y2=${sy(t)} />`)}
      <g clip-path="url(#cmapClip)">
        ${segs.map(([a, b]) => html`<line x1=${sx(a[0])} y1=${sy(a[1])} x2=${sx(b[0])} y2=${sy(b[1])} class="cmap-contour" />`)}
        ${order.map((p) => {
          const r = routeOf(p);
          if (!r) {
            return html`<polyline points=${poly(p.points)} class="cmap-history" style="pointer-events:none" />`;
          }
          const lit = R.lit(r), on = p.id === sel.path, col = mechColor(r.mechanism);
          const dots = on || (hover?.p?.id === p.id);
          return html`<g style="cursor:pointer" class=${lit ? '' : 'cmap-dim'} onClick=${() => onPick({ mech: r.mechanism, pair: r.pair, path: on ? null : p.id })}>
            <polyline points=${poly(p.points)} class="cmap-casing" style=${`stroke-width:${on ? 7 : lit && anyDim ? 5.5 : 4.5}`} />
            <polyline points=${poly(p.points)} fill="none" style=${`stroke:${col};stroke-width:${on ? 3.6 : lit && anyDim ? 3 : 2.2}`}
              class=${p.active ? 'cmap-active' : ''} stroke-linecap="round" stroke-linejoin="round" />
            <polyline points=${poly(p.points)} class="cmap-hit" />
            ${p.points.map((q, k) => html`<circle cx=${sx(q[0])} cy=${sy(q[1])} r=${dots ? 3 : 5} class=${dots ? 'cmap-img' : 'cmap-img-hit'}
              style=${dots ? `fill:${col}` : ''} onMouseEnter=${() => setHover({ p, k, q })} />`)}
          </g>`;
        })}
        ${ircPts.length > 1 && html`<g style="pointer-events:none">
          <polyline points=${poly(ircPts)} class="cmap-casing" style="stroke-width:5" />
          <polyline points=${poly(ircPts)} fill="none" class="cmap-irc" />
        </g>`}
        ${marks.map((m) => {
          const X = sx(m.q[0]), Y = sy(m.q[1]), mech = R.pairMech[m.pair], r = 6;
          const lit = !anyDim || (sel.pair ? m.pair === sel.pair : mech === sel.mech);
          const dash = m.kind === 'direct' ? '' : m.kind === 'multi-step' ? '3 2' : '1.5 2';
          return html`<g class=${lit ? '' : 'cmap-dim'} opacity=${m.kind === 'other' ? 0.75 : 1}
            onMouseEnter=${() => setHover({ ts: m })} style="cursor:default">
            <path d=${`M${X - r},${Y - r}L${X + r},${Y + r}M${X - r},${Y + r}L${X + r},${Y - r}`} class="cmap-casing" style="stroke-width:5.5" stroke-linecap="round" />
            <path d=${`M${X - r},${Y - r}L${X + r},${Y + r}M${X - r},${Y + r}L${X + r},${Y - r}`} style=${`stroke:${mech ? mechColor(mech) : 'var(--muted)'};stroke-width:2.8`}
              stroke-linecap="round" stroke-dasharray=${dash} />
            <circle cx=${X} cy=${Y} r="10" fill="transparent" />
          </g>`;
        })}
        ${minima.map((m) => html`<g style="pointer-events:none">
          <circle cx=${m.X} cy=${m.Y} r="5" class="cmap-min" />
          <text x=${m.X + 8} y=${m.Y - 7} class="cmap-minlabel">${m.names.join(' · ')}</text>
        </g>`)}
      </g>
      ${ticks(x0, x1, 6).map((t) => html`<text x=${sx(t)} y=${H - B + 16} text-anchor="middle">${+t.toFixed(2)}</text>`)}
      ${ticks(y0, y1, 6).map((t) => html`<text x=${L - 6} y=${sy(t) + 4} text-anchor="end">${+t.toFixed(2)}</text>`)}
      <text x=${(L + W - R_) / 2} y=${H - 8} text-anchor="middle" class="cmap-axis">${map.axes.x}</text>
      <text x="14" y=${(T + H - B) / 2} transform=${`rotate(-90 14 ${(T + H - B) / 2})`} text-anchor="middle" class="cmap-axis">${map.axes.y}</text>
      ${hover?.ts && html`<g style="pointer-events:none">
        <text x=${Math.min(sx(hover.ts.q[0]) + 12, W - 320)} y=${Math.max(sy(hover.ts.q[1]) - 30, 24)} class="cmap-tip">
          TS · ${pairLabel(hover.ts.pair)} · ${hover.ts.e.toFixed(1)} kcal/mol · ${hover.ts.kind === 'direct' ? 'direct channel' : hover.ts.kind === 'multi-step' ? 'step of a multi-step channel' : 'IRC misses the ends'}</text>
        ${hover.ts.closest && html`<text x=${Math.min(sx(hover.ts.q[0]) + 12, W - 320)} y=${Math.max(sy(hover.ts.q[1]) - 30, 24) + 15} class="cmap-tip">
          its chain came within ${hover.ts.closest.rmsd.toFixed(2)} Å RMSD</text>`}
      </g>`}
      ${hover && !hover.ts && html`<g style="pointer-events:none">
        <circle cx=${sx(hover.q[0])} cy=${sy(hover.q[1])} r="7" class="cmap-hover" />
        <text x=${Math.min(sx(hover.q[0]) + 10, W - 200)} y=${Math.max(sy(hover.q[1]) - 10, 24)} class="cmap-tip">
          ${hoverText(hover, R)}</text>
      </g>`}
    </svg>
  </div>
    ${surface && html`<div class="cmap-legend small">
      <span>Energy</span> <span class="mono">${vmin.toFixed(0)}</span>
      <span class="cmap-bar" style=${`background:linear-gradient(90deg, ${PALETTE.map((c) => `rgb(${c.join(',')})`).join(',')})`}></span>
      <span class="mono">${vmax.toFixed(0)}</span> <span class="muted">kcal/mol · contours every ${+step.toFixed(2)} · fitted to the images, blank where nothing was computed</span>
    </div>`}`;
}

function hoverText(hover, R) {
  const at = R.onRoute[hover.p.id];
  const where = at ? `${at.route.mechanism} · ${pairLabel(at.route.pair)} · step ${at.i + 1}` : `${pairLabel(hover.p.pair)} · ${hover.p.monitor}`;
  return `${where} · image ${hover.k + 1}: ${hover.q[2] == null ? '—' : `${hover.q[2].toFixed(1)} kcal/mol`}`;
}

// Energy along each route, its steps side by side (step i spans i..i+1, the
// images spaced by how far the structure moves), so the routes' minima line
// up and their barriers can be compared step by step.
const PW = 660, PH = 210, PL = 56, PR = 16, PT = 16, PB = 26;
function Profile({ map, R, sel, onPick, hover, setHover }) {
  const lines = [];
  for (const r of R.routes) {
    const pts = [];
    r.steps.forEach((s, i) => {
      const p = R.byId[s.path];
      if (!p) return;
      const n = p.points.length;
      p.points.forEach((q, k) => { if (q[2] != null) pts.push({ x: i + (p.s?.[k] ?? k / Math.max(1, n - 1)), e: q[2], p, k, q }); });
    });
    lines.push({ r, pts });
  }
  const tsMarks = [];
  for (const { r } of lines) r.steps.forEach((s, i) => {
    if (!s.ts) return;
    const m = (map.ts || []).find((t) => t.label === s.ts);
    const p = R.byId[s.path];
    if (!m || !p) return;
    const k = p.points.reduce((b, q, j) => ((q[2] ?? -1e9) > (p.points[b][2] ?? -1e9) ? j : b), 0);
    tsMarks.push({ r, x: i + (p.s?.[k] ?? 0.5), e: m.e, m });
  });
  const es = lines.flatMap((l) => l.pts.map((p) => p.e)).concat(tsMarks.map((t) => t.e));
  if (!es.length) return null;
  const n = Math.max(1, ...R.routes.map((r) => r.steps.length));
  const e0 = Math.min(...es), e1 = Math.max(...es), pad = (e1 - e0) * 0.08 || 2;
  const y0 = e0 - pad, y1 = e1 + pad * 1.6;
  const sx = (x) => PL + x / n * (PW - PL - PR), sy = (e) => PT + (1 - (e - y0) / (y1 - y0)) * (PH - PT - PB);
  const anyDim = !!(sel.mech || sel.pair);
  const order = [...lines].sort((a, b) => R.lit(a.r) - R.lit(b.r) || ((a.r.pair === sel.pair) - (b.r.pair === sel.pair)));
  const labels = [];
  for (const { r } of order) {
    if (!R.lit(r)) continue;
    r.points.forEach((m, i) => {
      if (m.e == null) return;
      const X = sx(i), Y = sy(m.e);
      const near = labels.find((o) => Math.abs(o.X - X) < 4 && Math.abs(o.Y - Y) < 11);
      if (near) { if (!near.names.includes(m.name)) near.names.push(m.name); return; }
      labels.push({ X, Y, names: [m.name] });
    });
  }
  const hx = hover && !hover.ts && hover.p ? lines.flatMap((l) => l.pts).find((p) => p.p.id === hover.p.id && p.k === hover.k) : null;
  return html`<div class="cmap-prof">
    <svg viewBox=${`0 0 ${PW} ${PH}`} onMouseLeave=${() => setHover(null)}>
      <rect class="cmap-frame" x=${PL} y=${PT} width=${PW - PL - PR} height=${PH - PT - PB} />
      ${ticks(y0, y1, 5).map((t) => html`<line class="cmap-grid" x1=${PL} x2=${PW - PR} y1=${sy(t)} y2=${sy(t)} />
        <text x=${PL - 6} y=${sy(t) + 4} text-anchor="end">${+t.toFixed(1)}</text>`)}
      ${Array.from({ length: n + 1 }, (_, i) => html`<line class="cmap-grid" x1=${sx(i)} x2=${sx(i)} y1=${PT} y2=${PH - PB} />`)}
      ${Array.from({ length: n }, (_, i) => html`<text x=${sx(i + 0.5)} y=${PH - 8} text-anchor="middle" class="cmap-axis">step ${i + 1}</text>`)}
      <text x="14" y=${(PT + PH - PB) / 2} transform=${`rotate(-90 14 ${(PT + PH - PB) / 2})`} text-anchor="middle" class="cmap-axis">kcal/mol</text>
      ${order.map(({ r, pts }) => {
        const lit = R.lit(r), on = r.pair === sel.pair, col = mechColor(r.mechanism);
        const d = pts.map((p) => `${sx(p.x).toFixed(1)},${sy(p.e).toFixed(1)}`).join(' ');
        return html`<g class=${lit ? '' : 'cmap-dim'} style="cursor:pointer" onClick=${() => onPick({ mech: r.mechanism, pair: on && !sel.path ? null : r.pair, path: null })}>
          <polyline points=${d} class="cmap-casing" style=${`stroke-width:${on ? 6 : 4.5}`} />
          <polyline points=${d} fill="none" stroke-linejoin="round" stroke-linecap="round" style=${`stroke:${col};stroke-width:${on ? 3 : lit && anyDim ? 2.6 : 2}`} />
          <polyline points=${d} class="cmap-hit" />
          ${pts.map((p) => html`<circle cx=${sx(p.x)} cy=${sy(p.e)} r="5" class="cmap-img-hit" onMouseEnter=${() => setHover({ p: p.p, k: p.k, q: p.q })} />`)}
        </g>`;
      })}
      ${tsMarks.map((t) => {
        const X = sx(t.x), Y = sy(t.e), r = 5, lit = R.lit(t.r);
        return html`<g class=${lit ? '' : 'cmap-dim'} style="pointer-events:none">
          <path d=${`M${X - r},${Y - r}L${X + r},${Y + r}M${X - r},${Y + r}L${X + r},${Y - r}`} class="cmap-casing" style="stroke-width:5" stroke-linecap="round" />
          <path d=${`M${X - r},${Y - r}L${X + r},${Y + r}M${X - r},${Y + r}L${X + r},${Y - r}`} style=${`stroke:${mechColor(t.r.mechanism)};stroke-width:2.4`} stroke-linecap="round" />
        </g>`;
      })}
      ${labels.map((l) => html`<g style="pointer-events:none">
        <circle cx=${l.X} cy=${l.Y} r="4" class="cmap-min" />
        <text x=${l.X + (l.X > PW - 60 ? -7 : 7)} y=${l.Y + 15} text-anchor=${l.X > PW - 60 ? 'end' : 'start'} class="cmap-minlabel">${l.names.join(' · ')}</text>
      </g>`)}
      ${hx && html`<g style="pointer-events:none">
        <circle cx=${sx(hx.x)} cy=${sy(hx.e)} r="6" class="cmap-hover" />
        <text x=${Math.min(sx(hx.x) + 9, PW - 230)} y=${Math.max(sy(hx.e) - 9, 14)} class="cmap-tip">${hoverText(hover, R)}</text>
      </g>`}
    </svg>
  </div>`;
}

// The mechanisms, lowest highest point first; a picked one opens its pairs,
// a picked pair its steps.
function Mechanisms({ R, sel, onPick }) {
  return html`<ul class="cmap-mechs small">${R.mechs.map((m) => {
    const on = sel.mech === m.id, r0 = m.routes[0];
    return html`<li class=${on ? 'on' : ''}>
      <button type="button" class="cmap-mech" onClick=${() => onPick(on && !sel.pair ? {} : { mech: m.id })} aria-pressed=${on}>
        <span class="cmap-letter" style=${`background:${mechColor(m.id)}`}>${m.id}</span>
        <span class="cmap-seq">${seqText(r0)}</span>
        <span class="cmap-top mono" title="Highest point on the way (optimized TS where there is one), kcal/mol from the lowest reactant">‡ ${kcal(m.top === Infinity ? null : m.top)}</span>
        <span class="muted cmap-count">${r0.steps.length} step${r0.steps.length === 1 ? '' : 's'} · ${m.routes.length} pair${m.routes.length === 1 ? '' : 's'}</span>
      </button>
      ${on && html`<ul class="cmap-routes">${m.routes.map((r) => {
        const pon = sel.pair === r.pair;
        return html`<li class=${pon ? 'on' : ''}>
          <button type="button" onClick=${() => onPick(pon && !sel.path ? { mech: m.id } : { mech: m.id, pair: r.pair })} title=${r.pair}>
            ${pairLabel(r.pair)} <span class="mono muted">‡ ${kcal(r.top)}</span>${r.steps.some((s) => R.byId[s.path]?.active) ? html` <span class="badge">relaxing</span>` : ''}
          </button>
          ${pon && html`<ol class="cmap-steps">${r.steps.map((s, i) => html`<li class=${sel.path === s.path ? 'on' : ''}>
            <button type="button" onClick=${() => onPick({ mech: m.id, pair: r.pair, path: sel.path === s.path ? null : s.path })}>
              ${r.minima[i]} → ${r.minima[i + 1] || '…'} <span class="mono muted" title=${s.ts ? 'Optimized TS · rise from the step\'s start, kcal/mol' : 'No optimized TS for this step: its highest image · rise from the step\'s start, kcal/mol'}>${s.peak == null ? '' : `${s.ts ? '‡' : 'max'} ${kcal(s.peak)}`}${s.barrier == null ? '' : ` · +${kcal(s.barrier)}`}</span>
            </button></li>`)}</ol>`}
        </li>`;
      })}</ul>`}
    </li>`;
  })}</ul>`;
}

function PathDetail({ job, path, R, marks = [] }) {
  const [frames, setFrames] = useState(null);
  const [k, setK] = useState(0);
  useEffect(() => {
    let live = true;
    api.get(`/api/jobs/${job.id}/live/${path.pair}`).then((d) => {
      const m = (d.monitors || {})[path.monitor] || d;
      const f = m.geometry?.frames || [];
      if (live) { setFrames(f); setK(Math.max(0, path.points.reduce((best, q, i) => ((q[2] ?? -1e9) > (path.points[best][2] ?? -1e9) ? i : best), 0))); }
    }).catch(() => {});
    return () => { live = false; };
  }, [path.id, path.points.length]);
  const at = R.onRoute[path.id];
  const ts = marks.filter((m) => m.pair === path.pair && m.closest?.monitor === path.monitor);
  return html`<div class="cmap-detail">
    <h4>${at && html`<span class="cmap-letter" style=${`background:${mechColor(at.route.mechanism)}`}>${at.route.mechanism}</span>`}
      ${at ? `${pairLabel(path.pair)} · step ${at.i + 1}: ${at.route.minima[at.i]} → ${at.route.minima[at.i + 1] || '…'}` : `${pairLabel(path.pair)} · ${path.monitor}`}${path.active ? ' · relaxing now' : ''}</h4>
    ${path.bonds && html`<p class="small">${path.bonds.kind === 'break-form'
      ? html`breaks <b>${path.bonds.x.join(', ') || '—'}</b> · forms <b>${path.bonds.y.join(', ') || '—'}</b>`
      : html`${path.bonds.kind} <b>${path.bonds.x.join(', ')}</b> · then <b>${path.bonds.y.join(', ') || '—'}</b>`}</p>`}
    ${ts.map((m) => html`<p class="small">✕ TS ${m.e.toFixed(1)} kcal/mol · ${m.kind === 'direct' ? 'direct channel' : m.kind === 'multi-step' ? 'multi-step channel' : "IRC misses the ends"}${m.closest
        ? ` · chain within ${m.closest.rmsd.toFixed(2)} Å` : ''}</p>`)}
    ${frames && frames.length > 0 && html`
      <${Viewer3D} frames=${frames} frame=${Math.min(k, frames.length - 1)} height=${220} />
      <input type="range" min="0" max=${frames.length - 1} value=${Math.min(k, frames.length - 1)} onInput=${(e) => setK(+e.target.value)} aria-label="Image" />
      <div class="small muted">image ${Math.min(k, frames.length - 1) + 1} of ${frames.length}${path.points[k]?.[2] != null ? ` · ${path.points[k][2].toFixed(1)} kcal/mol` : ''}</div>`}
  </div>`;
}

const CAPTION = {
  bonds: 'x: bonds that break, y: bonds that form, each 0 at the reactant and 1 at the product (per pair). Concerted runs along the diagonal, stepwise along an edge.',
  distance: 'x, y: how far each structure is from the lowest reactant and product conformer (Å, independent of atom numbering). Paths through similar structures run close together.',
  irc: 'Each chain against one computed IRC (the grey line, its TS at 0): x along it, y how far off it.',
};

export function ChannelsMap({ job }) {
  const [mode, setModeState] = useState(() => prefs.get('channelsMapMode', 'bonds'));
  const setMode = (m) => { setModeState(m); prefs.set('channelsMapMode', m); };
  const [history, setHistoryState] = useState(() => prefs.get('channelsMapHistory', false));
  const setHistory = (v) => { setHistoryState(v); prefs.set('channelsMapHistory', v); };
  const [surface, setSurfaceState] = useState(() => prefs.get('channelsMapSurface', false));
  const setSurface = (v) => { setSurfaceState(v); prefs.set('channelsMapSurface', v); };
  const [map, setMap] = useState(null);
  const [ircTs, setIrcTs] = useState(null);
  const [err, setErr] = useState(null);
  const [sel, setSel] = useState({});
  const [hover, setHover] = useState(null);
  // Fetch on a view change at once; while the run goes, refresh every 3 s --
  // never with a request already in flight (a big run's map is not instant),
  // and never applying a late answer to an older request.
  const busy = useRef(false);
  const latest = useRef(0);
  const fetchMap = (force) => {
    if (busy.current && !force) return;
    const id = ++latest.current;
    busy.current = true;
    api.get(`/api/jobs/${job.id}/channels-map?mode=${mode}${mode === 'irc' && ircTs ? `&ts=${encodeURIComponent(ircTs)}` : ''}`)
      .then((m) => { if (id === latest.current) { setMap(m); setErr(null); } })
      .catch((e) => { if (id === latest.current) setErr(e.message); })
      .finally(() => { if (id === latest.current) busy.current = false; });
  };
  useEffect(() => { fetchMap(true); }, [mode, ircTs, job.status]);
  useEffect(() => {
    if (job.status !== 'running') return undefined;
    const t = setInterval(() => fetchMap(false), 3000);
    return () => clearInterval(t);
  }, [mode, ircTs, job.status]);
  const choices = map?.irc_choices || [];
  const R = useMemo(() => readRoutes(map, sel), [map, sel]);
  const pick = (s) => { setSel(s || {}); };
  const selPath = sel.path && map?.paths.find((p) => p.id === sel.path);
  const nHistory = map ? map.paths.filter((p) => !R.onRoute[p.id]).length : 0;
  const pairs = R.routes.length;
  return html`<div class="cmap">
    <div class="cmap-head">
      <div class="segmented">
        <button class=${mode === 'bonds' ? 'on' : ''} onClick=${() => setMode('bonds')} title="Each pair's own bonds, 0 to 1: broken vs formed (More O'Ferrall-Jencks)">Bond progress</button>
        <button class=${mode === 'distance' ? 'on' : ''} onClick=${() => setMode('distance')} title="Distance from the lowest reactant and product conformers, the same for every atom mapping">Distance</button>
        <button class=${mode === 'irc' ? 'on' : ''} disabled=${!choices.length && mode !== 'irc'} onClick=${() => setMode('irc')}
          title=${choices.length ? 'Post hoc: every chain measured against a computed IRC' : 'Needs a finished TS optimization with its IRC'}>Along an IRC</button>
      </div>
      ${mode === 'irc' && choices.length > 0 && html`<label class="small">IRC of
        <select value=${map?.irc?.ts || ''} onChange=${(e) => setIrcTs(e.target.value)}>
          ${choices.map((c) => html`<option value=${c.label}>${pairLabel(c.pair)} · ${c.e.toFixed(1)} kcal/mol · ${c.kind === 'direct' ? 'direct channel' : c.kind === 'multi-step' ? 'multi-step' : 'unconnected'}</option>`)}
        </select></label>`}
      <label class="cmap-toggle small" title="The chains the search split into steps, and steps off the easiest way"><input type="checkbox" checked=${history} onChange=${(e) => setHistory(e.target.checked)} /> Search history${nHistory ? ` (${nHistory})` : ''}</label>
      <label class="cmap-toggle small" title="An approximate surface fitted to the computed images, drawn only where they are"><input type="checkbox" checked=${surface} onChange=${(e) => setSurface(e.target.checked)} /> Energy surface</label>
      <span class="small muted">${map ? `${R.mechs.length} mechanism${R.mechs.length === 1 ? '' : 's'} · ${pairs} pair${pairs === 1 ? '' : 's'}` : ''}${job.status === 'running' ? ' · updating live' : ''}</span>
      <${Why} id="other.path-map" label="the path map" />
    </div>
    <p class="small muted cmap-caption">${CAPTION[mode] || ''}</p>
    ${err && html`<p class="error-box small">${err}</p>`}
    ${(map?.warnings || []).map((w) => html`<p class="level-note small">${w}</p>`)}
    ${map && !map.paths.length && html`<p class="small muted">No paths yet: they appear here as soon as the path searches start.</p>`}
    ${map && map.paths.length > 0 && html`<div class="cmap-body">
      <div class="cmap-main">
        <${Surface} map=${map} R=${R} sel=${sel} onPick=${pick} hover=${hover} setHover=${setHover} history=${history} surface=${surface} />
        <div class="cmap-key small muted">
          <span><svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="4" class="cmap-min" /></svg> minimum (R reactant, I intermediate, P product)</span>
          <span><svg width="12" height="12" viewBox="0 0 12 12"><path d="M2,2L10,10M2,10L10,2" style="stroke:var(--muted);stroke-width:2" /></svg> optimized TS</span>
          ${history && html`<span><svg width="18" height="8" viewBox="0 0 18 8"><line x1="0" y1="4" x2="18" y2="4" class="cmap-history" /></svg> search history</span>`}
        </div>
        <h4 class="cmap-subhead">Energy along each route</h4>
        <${Profile} map=${map} R=${R} sel=${sel} onPick=${pick} hover=${hover} setHover=${setHover} />
      </div>
      <div class="cmap-side">
        <h4 class="cmap-subhead">Mechanisms</h4>
        <${Mechanisms} R=${R} sel=${sel} onPick=${pick} />
        ${selPath ? html`<${PathDetail} job=${job} path=${selPath} R=${R} marks=${map.ts || []} />`
          : html`<p class="small muted">${sel.pair ? 'Pick a step to step through its images in 3D.' : sel.mech ? 'Pick a pair to see its steps.' : 'Pick a mechanism, or click a line, to follow one route.'}</p>`}
      </div>
    </div>`}
  </div>`;
}
