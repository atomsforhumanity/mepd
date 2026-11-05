// Live view of an async network expansion (`--schedule async`): the one job
// queue everything runs in -- proposals, TS searches, path pieces, IRC
// checks, channel sampling, steering single points -- and the live kinetics
// that decide what it runs next. Both are whole-state streams the explorer
// rewrites as they change ("queue", "kinetics").
import { html, useState } from '../lib.js';
import { depictUrl } from '../util.js';

const KINDS = [
  ['propose', 'Proposals', 'the generator run on a species the kinetics populate'],
  ['fast', 'Fast TS', 'interpolation, then one TS optimization between two species (or two of their conformers)'],
  ['piece', 'Path pieces', 'one level of a path search; started only where the fast TS failed'],
  ['ts', 'Leaf TSs', 'a TS optimization from an elementary piece of a path search'],
  ['irc', 'IRC checks', 'which two species a TS really joins; only for steps touching a populated species'],
  ['channels', 'Channel sampling', 'conformers and atom mappings of a step; each conformer pair then joins the queue as a fast TS'],
  ['steer', 'Steering single points', 'the higher level on a species or TS the kinetics use'],
];

function fmt(v) {
  if (v == null) return '–';
  if (v === 0) return '0';
  return v >= 0.01 ? v.toFixed(3) : v.toExponential(1);
}

// log scale from 1e-4 (empty) to 1 (full)
function barWidth(v) {
  if (!v || v <= 1e-4) return 0;
  return Math.min(100, Math.max(2, (Math.log10(v) + 4) * 25));
}

function Queue({ q }) {
  const running = q.running || [];
  const byKind = {};
  for (const r of running) byKind[r.kind] = (byKind[r.kind] || 0) + 1;
  const busy = Math.min(q.busy || 0, q.workers || 1);
  const known = new Set(KINDS.map(([k]) => k));
  const kinds = [...KINDS, ...Object.keys({ ...q.waiting, ...q.finished, ...byKind }).filter((k) => !known.has(k)).map((k) => [k, k, ''])];
  return html`
    <div class="card-block async-queue">
      <h4>Job queue <span class="muted small">· ${busy} of ${q.workers} workers busy${q.iced ? ` · ${q.iced} path pieces iced (nothing reaches them yet)` : ''}${q.ended ? ' · finished' : ''}</span></h4>
      <div class="aq-meter"><div style=${`width:${100 * busy / Math.max(1, q.workers)}%`}></div></div>
      <table class="aq-table small">
        <thead><tr><th>Job</th><th>running</th><th>waiting</th><th>done</th><th>failed</th></tr></thead>
        <tbody>${kinds.map(([k, label, help]) => {
          const f = q.finished?.[k] || {};
          const any = byKind[k] || q.waiting?.[k] || f.done || f.failed;
          return any ? html`<tr key=${k} title=${help}><td>${label}</td><td class=${byKind[k] ? 'aq-run' : ''}>${byKind[k] || ''}</td>
            <td>${q.waiting?.[k] || ''}</td><td>${f.done || ''}</td><td class=${f.failed ? 'aq-fail' : ''}>${f.failed || ''}</td></tr>` : null;
        })}</tbody>
      </table>
      ${running.length > 0 && html`<div class="aq-running">${running.slice(0, 60).map((r, i) => html`
        <span key=${i} class="aq-chip" title=${r.kind}><b>${(KINDS.find(([k]) => k === r.kind) || [, r.kind])[1]}</b> ${r.what}</span>`)}
        ${running.length > 60 ? html`<span class="muted small"> +${running.length - 60} more</span>` : ''}</div>`}
    </div>`;
}

function Kinetics({ k }) {
  const [all, setAll] = useState(false);
  const rows = [...(k.species || [])]
    .map((s) => ({ ...s, load: Math.max(s.max_concentration || 0, s.flux || 0) }))
    .sort((a, b) => b.load - a.load || a.index - b.index);
  const shown = all ? rows : rows.filter((s) => s.populated);
  return html`
    <div class="card-block async-kinetics">
      <h4>Live kinetics <span class="muted small">· ${k.species?.length || 0} species, ${k.steps} steps · ${k.basis}${k.steered ? ` · ${k.steered} steering single points back` : ''}</span></h4>
      <p class="small muted">A species is populated once its largest concentration or the material that flowed through it reaches ${k.threshold} of the seed:
        only populated species are expanded, and only steps touching them get IRC checks and channel sampling.</p>
      <label class="small check"><input type="checkbox" class="switch" checked=${all} onChange=${(e) => setAll(e.target.checked)} /> Show unpopulated species too</label>
      <table class="ak-table small">
        <thead><tr><th>#</th><th></th><th>species</th><th>ΔE</th><th title="largest concentration, as a fraction of the seed">max</th><th title="material that flowed through it">flux</th><th>final</th><th></th></tr></thead>
        <tbody>${shown.map((s) => html`<tr key=${s.index} class=${s.populated ? 'ak-hot' : ''}>
          <td class="mono">${s.index}</td>
          <td>${s.smiles ? html`<img class="ak-depict" src=${depictUrl(s.smiles, 90, 60)} alt="" />` : ''}</td>
          <td class="mono ak-smiles" title=${s.smiles}>${s.smiles || '?'}
            <div class="ak-bar"><div style=${`width:${barWidth(s.load)}%`}></div></div></td>
          <td class="mono">${s.rel_kcal != null ? s.rel_kcal.toFixed(1) : ''}</td>
          <td class="mono">${fmt(s.max_concentration)}</td><td class="mono">${fmt(s.flux)}</td><td class="mono">${fmt(s.final)}</td>
          <td>${s.populated ? html`<span class="live-state running">populated</span>` : ''}${s.expanded ? html`<span class="live-state done">expanded</span>` : ''}</td>
        </tr>`)}</tbody>
      </table>
    </div>`;
}

export function AsyncLive({ progress }) {
  const q = progress?.streams?.queue?.data;
  const k = progress?.streams?.kinetics?.data;
  if (!q && !k) return null;
  return html`<div class="async-live">${q && html`<${Queue} q=${q} />`}${k && html`<${Kinetics} k=${k} />`}</div>`;
}
