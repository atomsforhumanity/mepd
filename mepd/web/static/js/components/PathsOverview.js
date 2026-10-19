// The top of a path search's results, as in a reaction's report: the
// numbers that matter, an energy diagram with every channel (or TS) found,
// and a table of them. Clicking a channel shows it in the viewer below.
import { html, useState } from '../lib.js';
import { fmtKcal } from '../util.js';

// Signed, as on an energy diagram (+16.6, −30.7).
const sk = (x) => (x == null ? '–' : `${x > 0.05 ? '+' : ''}${x.toFixed(1)}`.replace('-', '−'));
// A short name under the diagram: the result's own (Channel 2 -> ch 2).
const short = (label) => label.replace(/^Channel /, 'ch ').replace(/^First search · /, '').slice(0, 14);
const HUES = ['#2f6fdb', '#e0662f', '#1f9e74', '#8a55c9', '#c8417a', '#7a8b2b', '#2b97b8', '#b8862b'];

// The paths worth comparing: the channels when there are (channels, sampled
// TS searches), else the IRC-verified TSs, else the paths themselves. With
// no TS joining the ends, the path itself: another saddle's IRC would draw a
// reaction that is not this one. Only IRCs that join the two ends are
// channels: one that reaches an intermediate is a step of a route (RouteView).
export function overviewPaths(result) {
  const groups = result.groups || [];
  const of = (kind) => groups.filter((g) => g && g.kind === kind).flatMap((g) => g.entries)
    .filter((e) => kind !== 'irc' || e.joins_ends !== false);
  const usable = (es) => es.filter((e) => e.frames.length > 1 && e.ts_index != null && e.frames[e.ts_index]?.energy_kcal != null);
  for (const kind of result.barrier_verified === false ? ['channel', 'path', 'irc'] : ['channel', 'irc', 'path']) {
    const es = usable(of(kind));
    if (es.length) return es.map((e) => {
      const fr = e.frames;
      return { id: e.id, label: e.label, note: e.note, barrier: e.barrier_kcal,
        start: fr[0].energy_kcal, ts: fr[e.ts_index].energy_kcal, end: fr[fr.length - 1].energy_kcal };
    }).sort((a, b) => (a.barrier ?? a.ts) - (b.barrier ?? b.ts));   // lowest first, as in a reaction's report
  }
  return [];
}

function Diagram({ paths, lowest, current, onPick, top = 'TS' }) {
  const W = 640, H = 210, padT = 26, padB = 44;
  const ys = paths.flatMap((p) => [p.start, p.ts, p.end, 0]);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const y = (e) => padT + (hi - e) / ((hi - lo) || 1) * (H - padT - padB);
  const n = paths.length;
  const xR = 60, xP = W - 60, colW = (xP - xR - 80) / Math.max(n, 1);
  const xTs = (i) => xR + 40 + colW * (i + 0.5);
  const prod = Math.min(...paths.map((p) => p.end));
  const bar = (x, e, w = 52) => `M${x - w / 2},${y(e)}H${x + w / 2}`;
  return html`<svg class="po-diagram" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label="Energy diagram">
    <path d=${bar(xR, 0)} class="po-level" />
    <text x=${xR} y=${y(0) - 7} class="po-num">${sk(0)}</text>
    <text x=${xR} y=${H - 24} class="po-lab">reactant</text>
    <path d=${bar(xP, prod)} class="po-level" />
    <text x=${xP} y=${y(prod) - 7} class="po-num">${sk(prod)}</text>
    <text x=${xP} y=${H - 24} class="po-lab">product</text>
    ${paths.map((p, i) => {
      const x = xTs(i), c = HUES[i % HUES.length], on = p.id === current;
      return html`<g class=${`po-ch ${on ? 'on' : ''}`} style=${`--c:${c}`} onClick=${() => onPick(p.id)}>
        <path d=${`M${xR + 26},${y(0)}L${x - 16},${y(p.ts)}H${x + 16}L${xP - 26},${y(prod)}`} class="po-line" />
        <text x=${x} y=${y(p.ts) - 7} class="po-num">${sk(p.ts)}</text>
        <text x=${x} y=${H - 24} class="po-lab">${n > 1 ? short(p.label) : top}</text>
        ${p.id === lowest && n > 1 && html`<text x=${x} y=${H - 10} class="po-lab b">lowest</text>`}
      </g>`;
    })}
  </svg>`;
}

// A multi-step route: start -> TS -> intermediate -> TS -> ... -> end, one
// chain (each IRC joins neighbours, never the reactant to the product).
// An intermediate sits at the lower of the two geometries found for it (the
// end of one step, the start of the next), and each step's barrier is
// measured from the species before it.
function chainOf(route) {
  const st = route.steps;
  const levels = [0];
  for (let i = 0; i < st.length - 1; i += 1) {
    const a = st[i].to_kcal, b = st[i + 1].from_kcal;
    levels.push(a == null ? b : b == null ? a : Math.min(a, b));
  }
  levels.push(st[st.length - 1].to_kcal);
  // A species between steps that is the start or end molecule in another shape (a conformer change follows).
  let k = 0;
  const mids = st.map((x) => x.to).slice(0, -1).map((n) => (n === 'start' || n === 'end'
    ? { name: `${n} (other shape)`, short: `${n}′` } : (k += 1, { name: `intermediate ${k}`, short: `int. ${k}` })));
  const names = mids.map((m) => m.name);
  const chain = st.map((x, i) => ({ ...x, lo: levels[i], hi: levels[i + 1], step: x.ts_kcal != null && levels[i] != null ? x.ts_kcal - levels[i] : null,
    fromName: i ? names[i - 1] : 'start', toName: i < st.length - 1 ? names[i] : 'end', fromSmiles: i ? st[i - 1].to : '', toSmiles: x.to }));
  return { chain, mids };
}

function ChainDiagram({ chain, mids, current, onPick, top }) {
  const W = 640, H = 210, padT = 26, padB = 44;
  const ys = chain.flatMap((s) => [s.lo, s.ts_kcal, s.hi]).filter((v) => v != null);
  const lo = Math.min(...ys, 0), hi = Math.max(...ys, 0);
  const y = (e) => padT + (hi - e) / ((hi - lo) || 1) * (H - padT - padB);
  const n = chain.length, x0 = 50, x1 = W - 50, dx = (x1 - x0) / (2 * n);
  const xs = (j) => x0 + dx * j;          // even j: a minimum, odd j: a TS
  const bar = (x, e, w) => `M${x - w / 2},${y(e)}H${x + w / 2}`;
  const w = Math.min(52, dx * 0.9);
  const levelLab = (j) => (j === 0 ? 'start' : j === 2 * n ? 'end' : mids[j / 2 - 1].short);
  const levelAt = (i) => (i === 0 ? chain[0].lo : chain[i - 1].hi);
  return html`<svg class="po-diagram" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label="Energy diagram of the route">
    ${[...Array(n + 1).keys()].map((i) => html`<g>
      <path d=${bar(xs(2 * i), levelAt(i) ?? 0, w)} class="po-level" />
      <text x=${xs(2 * i)} y=${y(levelAt(i) ?? 0) - 7} class="po-num">${sk(levelAt(i))}</text>
      <text x=${xs(2 * i)} y=${H - 24} class="po-lab">${levelLab(2 * i)}</text></g>`)}
    ${chain.map((s, i) => {
      const x = xs(2 * i + 1), on = s.id === current;
      return html`<g class=${`po-ch ${on ? 'on' : ''}`} style=${`--c:${HUES[0]}`} onClick=${() => onPick(s.id)}>
        <path d=${`M${xs(2 * i) + w / 2},${y(s.lo ?? 0)}L${x - 12},${y(s.ts_kcal)}H${x + 12}L${xs(2 * i + 2) - w / 2},${y(s.hi ?? 0)}`} class="po-line" />
        <text x=${x} y=${y(s.ts_kcal) - 7} class="po-num">${sk(s.ts_kcal)}</text>
        <text x=${x} y=${H - 24} class="po-lab">${top} ${i + 1}</text>
      </g>`;
    })}
  </svg>`;
}

function RouteView({ result, current, onPick }) {
  const routes = result.routes;
  const [pickRoute, setPickRoute] = useState(0);
  const route = routes[Math.min(pickRoute, routes.length - 1)];
  const { chain, mids } = chainOf(route);
  const nots = route.verified === false;
  const overall = Math.max(...chain.map((s) => s.ts_kcal ?? -Infinity));
  const worst = chain.reduce((a, b) => ((b.ts_kcal ?? -Infinity) > (a.ts_kcal ?? -Infinity) ? b : a));
  const pick = (id) => {
    onPick(id);
    setTimeout(() => document.querySelector('.result-body')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
  };
  const end = chain[chain.length - 1].hi;
  const nMid = mids.filter((m) => m.name.startsWith('intermediate')).length;
  return html`<section class="paths-overview">
    <div class="po-stats">
      ${nots ? html`<div title="The path splits into steps, but no TS there has an IRC joining the species on either side"><span>Transition state</span><b>None found</b></div>`
        : html`<div title="The highest TS on the route, from the start: the route is as hard as its hardest step"><span>Overall barrier</span><b>${fmtKcal(overall)} kcal/mol</b></div>`}
      <div title=${nots ? 'Pieces the path split into' : 'Each step is its own TS, with an IRC joining the species on either side'}><span>Steps</span><b>${chain.length}</b></div>
      <div><span>Reaction energy</span><b>${sk(end)} kcal/mol</b></div>
    </div>
    ${routes.length > 1 && html`<div class="segmented po-routes" role="tablist">${routes.map((r, i) => html`<button role="tab"
      class=${i === pickRoute ? 'on' : ''} onClick=${() => setPickRoute(i)} title=${r.label}>Route ${i + 1}${r.barrier_kcal != null ? ` · ${fmtKcal(r.barrier_kcal)}` : ''}</button>`)}</div>`}
    <div class="po-card"><${ChainDiagram} chain=${chain} mids=${mids} current=${current} onPick=${pick} top=${nots ? 'top' : 'TS'} /></div>
    <p class="small muted">${nots ? 'Not verified: the path split into these pieces, but no TS was confirmed. '
      : nMid ? `No single step joins start and end: the reaction goes through ${nMid} intermediate${nMid > 1 ? 's' : ''}. ` : ''}Electronic energies, kcal/mol from the lowest reactant geometry found. Click a step to see it in 3D.</p>
    <table class="po-table">
      <thead><tr><th>Step</th><th>From → to</th><th class="num" title="TS minus the species the step starts from">ΔE‡</th><th class="num" title="TS, from the start">TS</th><th class="num" title="What the step gives, minus what it starts from">ΔE</th></tr></thead>
      <tbody>${chain.map((s, i) => html`<tr class=${s.id === current ? 'on' : ''} onClick=${() => pick(s.id)}>
        <td>${i + 1}${s === worst && chain.length > 1 && !nots ? html` <span class="small muted">hardest</span>` : ''}</td>
        <td><span title=${s.fromSmiles}>${s.fromName}</span> → <span title=${s.toSmiles}>${s.toName}</span></td>
        <td class="num">${s.step != null ? fmtKcal(s.step) : '–'}</td><td class="num">${sk(s.ts_kcal)}</td>
        <td class="num">${s.hi != null && s.lo != null ? sk(s.hi - s.lo) : '–'}</td></tr>`)}</tbody>
    </table>
  </section>`;
}

export function PathsOverview({ result, current, onPick }) {
  const paths = overviewPaths(result);
  // No direct channel, and a route of several steps: draw the route.
  const direct = paths.length && !(result.n_steps > 1) && result.barrier_verified !== false;
  if (result.routes?.length && !direct && result.routes[0].steps?.length > 1) {
    return html`<${RouteView} result=${result} current=${current} onPick=${onPick} />`;
  }
  if (!paths.length) return null;
  const best = paths.reduce((a, b) => ((b.barrier ?? Infinity) < (a.barrier ?? Infinity) ? b : a));
  const barrier = result.barrier_kcal ?? best.barrier;
  const many = paths.length > 1;
  const nots = result.barrier_verified === false;
  const pick = (id) => {
    onPick(id);
    setTimeout(() => document.querySelector('.result-body')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
  };
  return html`<section class="paths-overview">
    <div class="po-stats">
      ${nots ? html`<div title=${`The path's highest point is ≈${barrier != null ? fmtKcal(barrier) : '?'} kcal/mol, but no TS there has an IRC joining the two ends`}><span>Transition state</span><b>None found</b></div>`
        : barrier != null && html`<div><span>Lowest barrier</span><b>${fmtKcal(barrier)} kcal/mol</b></div>`}
      <div><span>${many ? 'Channels found' : 'Paths found'}</span><b>${paths.length}</b></div>
      <div><span>Reaction energy</span><b>${sk(best.end)} kcal/mol</b></div>

    </div>
    <div class="po-card"><${Diagram} paths=${paths} lowest=${best.id} current=${current} onPick=${pick} top=${nots ? 'path top' : 'TS'} /></div>
    <p class="small muted">Electronic energies, kcal/mol from the lowest reactant geometry found. Click a ${many ? 'channel' : 'path'} to see it in 3D.</p>
    ${many && html`<table class="po-table">
      <thead><tr><th>Channel</th><th class="num">ΔE‡</th><th class="num">TS</th><th class="num">ΔE</th><th>Connects</th></tr></thead>
      <tbody>${paths.map((p, i) => html`<tr class=${p.id === current ? 'on' : ''} onClick=${() => pick(p.id)}>
        <td><i style=${`background:${HUES[i % HUES.length]}`}></i>${p.label}${p.id === best.id ? html` <span class="small muted">lowest</span>` : ''}</td>
        <td class="num">${p.barrier != null ? fmtKcal(p.barrier) : '–'}</td><td class="num">${sk(p.ts)}</td>
        <td class="num">${sk(p.end)}</td><td class="small muted">${(p.note || '').split(' · ')[0]}</td></tr>`)}</tbody>
    </table>`}
  </section>`;
}
