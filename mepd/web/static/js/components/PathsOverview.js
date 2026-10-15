// The top of a path search's results, as in a reaction's report: the
// numbers that matter, an energy diagram with every channel (or TS) found,
// and a table of them. Clicking a channel shows it in the viewer below.
import { html } from '../lib.js';
import { fmtKcal } from '../util.js';

// Signed, as on an energy diagram (+16.6, −30.7).
const sk = (x) => (x == null ? '–' : `${x > 0.05 ? '+' : ''}${x.toFixed(1)}`.replace('-', '−'));
// A short name under the diagram: the result's own (Channel 2 -> ch 2).
const short = (label) => label.replace(/^Channel /, 'ch ').replace(/^First search · /, '').slice(0, 14);
const HUES = ['#2f6fdb', '#e0662f', '#1f9e74', '#8a55c9', '#c8417a', '#7a8b2b', '#2b97b8', '#b8862b'];

// The paths worth comparing: the channels when there are (channels, sampled
// TS searches), else the IRC-verified TSs, else the paths themselves.
export function overviewPaths(result) {
  const groups = result.groups || [];
  const of = (kind) => groups.filter((g) => g.kind === kind).flatMap((g) => g.entries);
  const usable = (es) => es.filter((e) => e.frames.length > 1 && e.ts_index != null && e.frames[e.ts_index]?.energy_kcal != null);
  for (const kind of ['channel', 'irc', 'path']) {
    const es = usable(of(kind));
    if (es.length) return es.map((e) => {
      const fr = e.frames;
      return { id: e.id, label: e.label, note: e.note, barrier: e.barrier_kcal,
        start: fr[0].energy_kcal, ts: fr[e.ts_index].energy_kcal, end: fr[fr.length - 1].energy_kcal };
    }).sort((a, b) => (a.barrier ?? a.ts) - (b.barrier ?? b.ts));   // lowest first, as in a reaction's report
  }
  return [];
}

function Diagram({ paths, lowest, current, onPick }) {
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
        <text x=${x} y=${H - 24} class="po-lab">${n > 1 ? short(p.label) : 'TS'}</text>
        ${p.id === lowest && n > 1 && html`<text x=${x} y=${H - 10} class="po-lab b">lowest</text>`}
      </g>`;
    })}
  </svg>`;
}

export function PathsOverview({ result, current, onPick }) {
  const paths = overviewPaths(result);
  if (!paths.length) return null;
  const best = paths.reduce((a, b) => ((b.barrier ?? Infinity) < (a.barrier ?? Infinity) ? b : a));
  const barrier = result.barrier_kcal ?? best.barrier;
  const many = paths.length > 1;
  const pick = (id) => {
    onPick(id);
    setTimeout(() => document.querySelector('.result-body')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
  };
  return html`<section class="paths-overview">
    <div class="po-stats">
      ${barrier != null && html`<div><span>Lowest barrier</span><b>${fmtKcal(barrier)} kcal/mol</b></div>`}
      <div><span>${many ? 'Channels found' : 'Paths found'}</span><b>${paths.length}</b></div>
      <div><span>Reaction energy</span><b>${sk(best.end)} kcal/mol</b></div>
      ${result.barrier_verified != null && html`<div><span>IRC connects the ends</span><b>${result.barrier_verified ? 'Yes' : 'No'}</b></div>`}
    </div>
    <div class="po-card"><${Diagram} paths=${paths} lowest=${best.id} current=${current} onPick=${pick} /></div>
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
