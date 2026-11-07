// Analyze › Kinetics: the network as a microkinetic model (mepd.microkinetics
// via /api/kinetics). Start from a mixture, in a closed flask or with a
// constant feed, and read: how much of the target forms, which barriers
// control it (degree of rate control: lower these), which species are
// traps, where the material flows, and how it all moves with temperature.
import { html, useMemo, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { set, state, useStore } from '../store.js';
import { depictUrl, isSpecies, variantScope } from '../util.js';
import { Why } from './Why.js';

const fmtC = (x) => (x == null ? '—' : x === 0 ? '0' : Math.abs(x) >= 0.01 && Math.abs(x) < 1000 ? x.toPrecision(3) : x.toExponential(1));
const UNITS = { s: 1, min: 60, h: 3600, d: 86400 };
const RT = (T) => (8.314462618 * T) / 4184;

// ------------------------------------------------------------ charts
// Lines on log-log axes (time vs amount). Series colours: the validated
// categorical slots, in fixed order by the species' rank at the start.
function LineChart({ xs, series, xLabel, yLabel, floor = 1e-12, xlog = true, width = 560, height = 240 }) {
  const [hover, setHover] = useState(null);
  const pad = { l: 56, r: 12, t: 10, b: 34 };
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b;
  const lx = (x) => (xlog ? Math.log10(Math.max(x, 1e-300)) : x);
  const ly = (y) => Math.log10(Math.max(y, floor));
  const xsF = xs.filter((x) => !xlog || x > 0);
  const x0 = Math.min(...xsF.map(lx)), x1 = Math.max(...xsF.map(lx));
  const ys = series.flatMap((s) => s.values).filter((v) => v > 0);
  const y1 = Math.ceil(Math.log10(Math.max(...ys, floor * 10)));
  const y0 = Math.max(Math.floor(Math.log10(Math.min(...ys, 1))), Math.log10(floor));
  const X = (x) => pad.l + ((lx(x) - x0) / Math.max(1e-9, x1 - x0)) * W;
  const Y = (y) => pad.t + (1 - (ly(y) - y0) / Math.max(1e-9, y1 - y0)) * H;
  const ticksY = [];
  const stepY = Math.max(1, Math.ceil((y1 - y0) / 5));
  for (let e = y0; e <= y1; e += stepY) ticksY.push(e);
  const ticksX = [];
  if (xlog) { const sx = Math.max(1, Math.ceil((x1 - x0) / 6)); for (let e = Math.ceil(x0); e <= x1; e += sx) ticksX.push(e); }
  else { for (let k = 0; k <= 5; k += 1) ticksX.push(x0 + ((x1 - x0) * k) / 5); }
  const pts = (vals) => xs.map((x, k) => (xlog && x <= 0 ? null : `${X(x).toFixed(1)},${Y(vals[k]).toFixed(1)}`)).filter(Boolean).join(' ');
  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * width;
    let best = 0, bd = Infinity;
    xs.forEach((x, k) => { if (xlog && x <= 0) return; const d = Math.abs(X(x) - px); if (d < bd) { bd = d; best = k; } });
    setHover(best);
  };
  return html`<div class="kn-chart">
    <svg viewBox=${`0 0 ${width} ${height}`} role="img" aria-label=${`${yLabel} against ${xLabel}`}
      onMouseMove=${onMove} onMouseLeave=${() => setHover(null)}>
      ${ticksY.map((e) => html`<line class="kn-grid" x1=${pad.l} x2=${pad.l + W} y1=${Y(10 ** e)} y2=${Y(10 ** e)} />
        <text class="kn-tick" x=${pad.l - 6} y=${Y(10 ** e) + 4} text-anchor="end">1e${e}</text>`)}
      ${ticksX.map((e) => html`<text class="kn-tick" x=${X(xlog ? 10 ** e : e)} y=${pad.t + H + 16} text-anchor="middle">${xlog ? `1e${e}` : +e.toPrecision(3)}</text>`)}
      <text class="kn-axis" x=${pad.l + W / 2} y=${height - 2} text-anchor="middle">${xLabel}</text>
      <text class="kn-axis" transform=${`translate(12 ${pad.t + H / 2}) rotate(-90)`} text-anchor="middle">${yLabel}</text>
      ${series.map((s) => html`<polyline class="kn-line" style=${{ stroke: `var(--series-${s.slot})` }} points=${pts(s.values)} />`)}
      ${hover != null && html`<line class="kn-cross" x1=${X(xs[hover])} x2=${X(xs[hover])} y1=${pad.t} y2=${pad.t + H} />
        ${series.map((s) => html`<circle class="kn-dot" cx=${X(xs[hover])} cy=${Y(s.values[hover])} r="4" style=${{ fill: `var(--series-${s.slot})` }} />`)}`}
    </svg>
    <div class="kn-legend">${series.map((s) => html`<span><i style=${{ background: `var(--series-${s.slot})` }}></i>${s.name}</span>`)}</div>
    ${hover != null && html`<div class="kn-tip">${xLabel.replace(/ \((.*)\)$/, ` ${fmtC(xs[hover])} $1`)}:
      ${series.map((s) => html` <span><i style=${{ background: `var(--series-${s.slot})` }}></i>${s.name} ${fmtC(s.values[hover])}</span>`)}</div>`}
  </div>`;
}

// Diverging bars: positive (blue) speeds the target up when that energy is lowered.
// One bar per TS or species: what 1 kcal/mol lower (a lower barrier, a
// more stable species) does to the target -- "3.2× more" (blue) or "1.6×
// less" (red). The bar length is the degree of rate control X, the
// exponent of that factor (Campbell); the tooltip gives X itself.
function ControlBars({ rows, onPick, T }) {
  const max = Math.max(0.05, ...rows.map((r) => Math.abs(r.x)));
  return html`<div class="kn-bars">${rows.map((r) => {
    const w = (Math.abs(r.x) / max) * 50;
    const factor = Math.exp(r.x / RT(T));
    return html`<button class="kn-bar-row" onClick=${() => onPick && onPick(r)}
      title=${`Degree of rate control X = ${r.x.toFixed(2)}: 1 kcal/mol lower multiplies the target by ${fmtFactor(factor)}`}>
      <span class="kn-bar-label">${r.label}</span>
      <span class="kn-bar-track"><span class="kn-bar-mid"></span>
        <span class=${`kn-bar ${r.x >= 0 ? 'pos' : 'neg'}`} style=${{ width: `${w}%`, [r.x >= 0 ? 'left' : 'right']: '50%' }}></span></span>
      <span class="kn-bar-v mono">${factor >= 1 ? `${fmtFactor(factor)}× more` : `${fmtFactor(1 / factor)}× less`}</span>
    </button>`;
  })}</div>`;
}

const fmtFactor = (f) => (f < 1000 ? f.toPrecision(2) : f.toExponential(0));

// ------------------------------------------------------------ setup
function Mixture({ items, onChange, species }) {
  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const ql = q.trim().toLowerCase();
  const hits = ql && open ? species.filter((s) => `${s.name} ${s.smiles}`.toLowerCase().includes(ql) && !items.some((i) => i.id === s.id)).slice(0, 8) : [];
  const up = (k, patch) => onChange(items.map((it, i) => (i === k ? { ...it, ...patch } : it)));
  return html`<div class="kn-mix">
    ${items.map((it, k) => {
      const rec = state.workspace.structures[it.id];
      return html`<div class="kn-mix-row">
        ${rec?.smiles && html`<img src=${depictUrl(rec.smiles, 60, 45)} alt="" />`}
        <span class="kn-mix-name">${rec?.name || '?'}</span>
        <input type="number" min="0" step="0.1" value=${it.c} title="Starting amount (mol/L)" onInput=${(e) => up(k, { c: +e.target.value })} />
        <span class="small muted">M</span>
        <label class="small" title="Constant feed: held at this amount throughout"><input type="checkbox" checked=${it.held} onChange=${(e) => up(k, { held: e.target.checked })} /> fixed</label>
        <button class="btn-icon" title="Remove" onClick=${() => onChange(items.filter((_, i) => i !== k))}>✕</button>
      </div>`;
    })}
    <div class="cp-search"><input type="search" placeholder="add a species to the start…" value=${q}
      onInput=${(e) => { setQ(e.target.value); setOpen(true); }} onFocus=${() => setOpen(true)} onBlur=${() => setTimeout(() => setOpen(false), 150)} />
      ${hits.length > 0 && html`<ul class="cp-hits">${hits.map((s) => html`<li><button onMouseDown=${(e) => e.preventDefault()}
        onClick=${() => { onChange([...items, { id: s.id, c: 1, held: false }]); setQ(''); setOpen(false); }}>
        ${s.smiles && html`<img src=${depictUrl(s.smiles, 60, 45)} alt="" />`}<span>${s.name}</span></button></li>`)}</ul>`}</div>
  </div>`;
}

function defaultMixture() {
  // The start of the latest nanoreactor run, 1 M each.
  const jobs = Object.values(state.jobs).filter((j) => j.op === 'nanoreactor').sort((a, b) => b.created - a.created);
  // A complex is its molecules (a species twice in it: 2 M).
  const st = state.workspace.structures;
  const c = {};
  for (const id of jobs[0]?.targets?.structures || []) {
    const rec = st[id];
    if (!rec) continue;
    for (const m of rec.role === 'complex' && rec.members?.length ? rec.members : [id]) if (st[m]) c[m] = (c[m] || 0) + 1;
  }
  return Object.entries(c).map(([id, n]) => ({ id, c: n, held: false }));
}

// ------------------------------------------------------------ view
export function KineticsView() {
  const structures = useStore((s) => s.workspace.structures);
  const reactions = useStore((s) => s.workspace.reactions || {});
  const variants = useStore((s) => s.workspace.variants || {});
  const saved = state.kinetics || {};
  // Which network: '' the parent, else a variant (mepd.web.variants); `cmp`:
  // the parent and that variant side by side, from the parent's mixture.
  const [net, setNet] = useState(saved.net && variants[saved.net] ? saved.net : '');
  // `cmp`: true (with the parent), or for a stacked variant 'on' (the variant it is built on) or 'parent'.
  const [cmp, setCmp] = useState(saved.cmp && variants[saved.net] ? saved.cmp : false);
  const species = useMemo(() => {
    // Compared runs start from the base network's mixture (the parent's, or the variant it is built on).
    const scope = variantScope(state.workspace, !net ? '' : !cmp ? net : cmp === 'on' ? variants[net]?.on : '');
    const ids = new Set(Object.values(reactions).flatMap((r) => [...r.reactants, ...r.products]));
    return Object.values(structures).filter((s) => isSpecies(s) && (ids.has(s.id) || s.energy != null)
      && (!s.variant || scope.keep.has(s.variant) || scope.mine.has(s.id)) && !scope.replaced.has(s.id))
      .sort((a, b) => (ids.has(b.id) - ids.has(a.id)) || a.name.localeCompare(b.name));
  }, [structures, reactions, variants, net, cmp]);
  const [mix, setMix] = useState(saved.mix || defaultMixture());
  const [T, setT] = useState(saved.T ?? 298.15);
  const [tVal, setTVal] = useState(saved.tVal ?? 1);
  const [tUnit, setTUnit] = useState(saved.tUnit ?? 'h');
  const [target, setTarget] = useState(saved.target || '');
  const [what, setWhat] = useState(saved.what || 'amount');
  const [unverified, setUnverified] = useState(saved.unverified ?? true);
  const [free, setFree] = useState(saved.free ?? true);
  const [sweepOn, setSweepOn] = useState(saved.sweepOn ?? true);
  const [tLo, setTLo] = useState(saved.tLo ?? 250);
  const [tHi, setTHi] = useState(saved.tHi ?? 650);
  const [res, setRes] = useState(saved.res || null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    const sweep = sweepOn ? Array.from({ length: 9 }, (_, k) => tLo + ((tHi - tLo) * k) / 8) : [];
    const out = await attempt(() => api.post('/api/kinetics', {
      initial: Object.fromEntries(mix.map((m) => [m.id, m.c])), held: mix.filter((m) => m.held).map((m) => m.id),
      temperature: T, time_s: tVal * UNITS[tUnit], target: target || null, include_unverified: unverified,
      control: !!target, control_what: what, sweep, free_energy: free,
      network: net && !cmp ? net : null, compare: net && cmp ? net : null, compare_parent: cmp === 'parent' }));
    setBusy(false);
    const shown = out ? { ...out, tUnit } : res;
    setRes(shown);
    set({ kinetics: { mix, T, tVal, tUnit, target, what, unverified, free, sweepOn, tLo, tHi, res: shown, net, cmp } });
  };
  const openReaction = (rid) => set({ analyze: { ...(state.analyze || {}), view: 'reactions', reaction: rid } });

  return html`<div class="kinetics">
    <div class="kn-setup">
      ${Object.keys(variants).length > 0 && html`<div class="kn-row"><label>Network <select value=${net}
        onChange=${(e) => { setNet(e.target.value); if (!e.target.value) setCmp(false); }}>
        <option value="">parent</option>
        ${Object.values(variants).map((v) => html`<option value=${v.id}>${v.name}</option>`)}</select></label>
        ${net && !variants[net]?.on && html`<label class="small" title="Run the parent and this variant from the same mixture: amounts of a replaced species go to its variant">
          <input type="checkbox" checked=${!!cmp} onChange=${(e) => setCmp(e.target.checked)} /> compare with parent<${Why} id="kinetics.compare" /></label>`}
        ${net && variants[net]?.on && html`<label class="small" title="Run this variant and another network from the same mixture: amounts of a replaced species go to its variant">compare with
          <select value=${cmp || ''} onChange=${(e) => setCmp(e.target.value)}>
            <option value="">—</option><option value="on">${variants[variants[net].on]?.name || 'its base'}</option><option value="parent">parent</option></select></label>`}</div>`}
      <div class="section-title">Start<${Why} ids=${['kinetics.model']} label="The kinetic model" /></div>
      <${Mixture} items=${mix} onChange=${setMix} species=${species} />
      <p class="small muted">fixed: a constant feed<${Why} id="kinetics.held" label="Fixed amounts" /></p>
      <div class="kn-row"><label>Temperature <input type="number" value=${T} min="1" step="10" onInput=${(e) => setT(+e.target.value)} /> K</label><${Why} ids=${['kinetics.temperature']} /></div>
      <div class="kn-row"><label>Time <input type="number" value=${tVal} min="0" step="1" onInput=${(e) => setTVal(+e.target.value)} /></label><${Why} id="kinetics.time" />
        <select value=${tUnit} onChange=${(e) => setTUnit(e.target.value)}>${Object.keys(UNITS).map((u) => html`<option value=${u}>${u}</option>`)}</select></div>
      <div class="kn-row"><label>Target <select value=${target} onChange=${(e) => setTarget(e.target.value)}>
        <option value="">— none —</option>
        ${species.map((s) => html`<option value=${s.id}>${s.name}</option>`)}</select></label><${Why} ids=${['kinetics.target']} /></div>
      ${target && html`<div class="kn-row segmented">
        <${Why} id="kinetics.control_what" /><button class=${what === 'amount' ? 'on' : ''} onClick=${() => setWhat('amount')} title="How much of the target there is at the end">amount at the end</button>
        <button class=${what === 'rate' ? 'on' : ''} onClick=${() => setWhat('rate')} title="How fast it forms at the end (with a feed: the steady-state rate)">formation rate</button>
      </div>`}
      <label class="kn-row small"><input type="checkbox" checked=${sweepOn} onChange=${(e) => setSweepOn(e.target.checked)} />
        temperature sweep <input type="number" class="tiny" value=${tLo} onInput=${(e) => setTLo(+e.target.value)} />–<input type="number" class="tiny" value=${tHi} onInput=${(e) => setTHi(+e.target.value)} /> K<${Why} ids=${['kinetics.sweep', 'kinetics.temperature_sweep']} /></label>
      <label class="kn-row small" title="Barriers that are only a path maximum (no TS/IRC confirmed them)"><input type="checkbox" checked=${unverified} onChange=${(e) => setUnverified(e.target.checked)} /> include unverified barriers<${Why} ids=${['kinetics.unverified', 'kinetics.include_unverified', 'kinetics.include_unverified_barriers']} /></label>
      <label class="kn-row small" title="Quasi-RRHO free energies (zero-point energy, enthalpy and entropy from GFN2-xTB single-point Hessians, computed once per structure and kept). Without it, electronic energies stand in, and complexes look too stable."><input type="checkbox" checked=${free} onChange=${(e) => setFree(e.target.checked)} /> free energies<${Why} ids=${['kinetics.free_energy', 'kinetics.free_energies']} /></label>
      <button class="btn primary" disabled=${busy || !mix.length} onClick=${run}>${busy ? 'Running…' : 'Run kinetics'}</button>
    </div>
    <div class="kn-results">${res?.compare ? html`<${Compare} res=${res} T=${T} onReaction=${openReaction} />`
      : res ? html`<${Results} res=${res} T=${T} onReaction=${openReaction} onRerun=${run} />`
      : html`<div class="empty-hint"><p>Pick a starting mixture and (optionally) a target species, then run. The model uses every
          reaction with a barrier: the lower its TS, the faster it goes, forward and back.</p></div>`}</div>
  </div>`;
}

function Results({ res, T, onReaction, onRerun }) {
  const [showAll, setShowAll] = useState(false);
  const [logTime, setLogTime] = useState(false);
  if (res.error) return html`<p class="warn-box">${res.error}</p><${MissingEnergy} res=${res} onRerun=${onRerun} />${res.excluded?.length > 0 && html`<${Excluded} list=${res.excluded} />`}`;
  const byId = Object.fromEntries(res.species.map((s) => [s.id, s]));
  // Series: the species that ever reach a visible amount, largest first; fixed colour by that order.
  const ranked = [...res.species].sort((a, b) => b.max - a.max);
  const shown = ranked.filter((s) => s.max > 1e-12).slice(0, 6);
  const series = shown.map((s, k) => ({ name: s.name, slot: k + 1, values: res.series[s.id] }));
  // Log time: from a decade before anything visibly changes, not from the
  // integrator's first instant (1e-12 s), so the axis follows the run's length.
  let chartFrom = 0;
  if (logTime) {
    const top = Math.max(1e-30, ...series.flatMap((s) => s.values));
    const moved = res.times.findIndex((_, k) => series.some((s) => Math.abs(s.values[k] - s.values[0]) > 1e-4 * top));
    const t0 = moved > 0 ? res.times[moved] / 10 : 0;
    chartFrom = Math.max(0, res.times.findIndex((t) => t >= t0) - 1);
  }
  const chartT = res.times.slice(chartFrom);
  const tgt = res.target;
  const total = res.species.reduce((a, s) => a + s.c0, 0);
  const allCtl = res.control ? res.steps.map((st, k) => ({ ...st, x: res.control.steps[k] })).filter((r) => Number.isFinite(r.x)) : [];
  const ctl = allCtl.filter((r) => Math.abs(r.x) >= 0.01).sort((a, b) => Math.abs(b.x) - Math.abs(a.x)).slice(0, 8);
  const quiet = allCtl.length - ctl.length;
  // Intermediates and other products only: stabilizing a starting material always slows things (trivially).
  const traps = res.control ? res.species.map((s, k) => ({ label: s.name, x: res.control.species[k], id: s.id, c0: s.c0 }))
    .filter((r) => Number.isFinite(r.x) && Math.abs(r.x) > 0.02 && !(r.c0 > 0) && r.id !== tgt?.id)
    .sort((a, b) => Math.abs(b.x) - Math.abs(a.x)).slice(0, 6) : [];
  const flows = [...res.steps].sort((a, b) => Math.abs(b.extent) - Math.abs(a.extent));
  const unit = res.tUnit || 's';   // the unit the run was set up in
  const timeLabel = `${+(res.time_s / UNITS[unit]).toPrecision(3)} ${unit}`;
  return html`<div class="kn-out">
    <p class="small kn-summary">${tgt && html`<b>${tgt.name}</b>: ${fmtC(tgt.final)} M after ${timeLabel} at ${Math.round(res.temperature)} K${
      total ? ` (${fmtC((100 * tgt.final) / total)}% of the start)` : ''}${res.mode === 'constant feed' && res.steady ? `, forming at ${fmtC(tgt.rate)} M/s` : ''} · `}<span class="muted">${
      res.mode === 'batch' ? (res.steady ? 'finished: nothing changes any more' : 'still changing at the end')
        : res.steady ? 'constant feed, steady state' : 'constant feed, not yet steady: run longer'} · ${res.steps.length} steps</span></p>

    <section>
      <div class="section-title">Amounts over time</div>
      <label class="small kn-row"><input type="checkbox" checked=${logTime} onChange=${(e) => setLogTime(e.target.checked)} /> log time</label>
      <${LineChart} xs=${chartT.map((t) => t / UNITS[unit])} series=${series.map((s) => ({ ...s, values: s.values.slice(chartFrom) }))}
        xLabel=${`time (${unit})`} yLabel="amount (M)" xlog=${logTime} />
      <details class="small"><summary>Table</summary>
        <table class="rx-table"><thead><tr><th>Species</th><th>start (M)</th><th>end (M)</th><th>peak (M)</th></tr></thead>
          <tbody>${ranked.slice(0, showAll ? ranked.length : 12).map((s) => html`<tr><td class="rx-eq">${s.name}${s.held ? ' (fixed)' : ''}</td>
            <td class="mono">${fmtC(s.c0)}</td><td class="mono">${fmtC(s.final)}</td><td class="mono">${fmtC(s.max)}</td></tr>`)}</tbody></table>
        ${ranked.length > 12 && html`<button class="btn-link small" onClick=${() => setShowAll(!showAll)}>${showAll ? 'fewer' : `all ${ranked.length}`}</button>`}
      </details>
    </section>

    ${res.control && !ctl.length && html`<section>
      <div class="section-title">What controls ${tgt.name}</div>
      <p class="small">${res.steady
        ? `No barrier controls it here: by ${timeLabel} at ${Math.round(res.temperature)} K the network has reached ${
          res.mode === 'batch' ? 'equilibrium' : 'a steady state set by the feed'}, so the amount of ${tgt.name} is set by the species' energies. To make more: stabilize ${
          tgt.name} or destabilize what competes with it${traps.length ? ' (below)' : ''}. Use a shorter time or a lower temperature to see which barriers matter before that.`
        : `No barrier controls it here: every step that limits it has its TS at or below one of its sides, so it counts as barrierless and its pace is set by the species' energies${
          traps.length ? ' (below)' : ''}. Lowering a TS changes nothing; check those TSs (see the warnings at the bottom).`}</p>
      ${traps.length > 0 && html`<p class="small muted kn-sub">What making each intermediate or other product 1 kcal/mol more stable does to ${tgt.name}: red, less; blue, more.</p>
        <${ControlBars} rows=${traps} T=${res.temperature} />`}
    </section>`}
    ${ctl.length > 0 && html`<section>
      <div class="section-title">What controls ${tgt.name} (${res.control.what === 'rate' ? 'its formation rate' : 'its amount'})</div>
      <p class="small muted">What lowering each TS by 1 kcal/mol does to ${tgt.name}: blue, more of it; red, less (that step diverts material away).
        Click to open the reaction.</p>
      <${ControlBars} rows=${ctl} T=${res.temperature} onPick=${(r) => r.reaction && onReaction(r.reaction)} />
      ${quiet > 0 && html`<p class="small muted">${quiet} other step${quiet > 1 ? 's have' : ' has'} next to no effect on it.</p>`}
      ${traps.length > 0 && html`<p class="small muted kn-sub">What making each intermediate or other product 1 kcal/mol more stable does to ${tgt.name}: red, less (a trap or a competing product); blue, more.</p>
        <${ControlBars} rows=${traps} T=${res.temperature} />`}
    </section>`}


    ${res.sweep && html`<${Sweep} sw=${res.sweep} res=${res} />`}

    <section>
      <div class="section-title">Where the material went</div>
      <table class="rx-table"><thead><tr><th>Reaction</th><th title="Forward barrier from the separated reactants">ΔE‡</th><th title="Net amount converted over the run">converted (M)</th></tr></thead>
        <tbody>${flows.slice(0, 10).map((st) => html`<tr class=${st.reaction ? 'clickable' : ''} onClick=${() => st.reaction && onReaction(st.reaction)}>
          <td class="rx-eq">${st.label}${st.copies > 1 ? html` <span class="small muted">(${st.copies} runs; lowest TS)</span>` : ''}${st.estimate ? html` <span class="small muted" title="fast estimate: the parent's barrier shifted by the edit, until the re-optimized TS is in">est.</span>`
            : !st.verified ? html` <span class="small muted" title="path maximum, not TS/IRC-verified">≈</span>` : ''}</td>
          <td class="mono">${st.barrier_fwd.toFixed(1)}</td><td class="mono">${fmtC(st.extent)}</td></tr>`)}</tbody></table>
    </section>
    <${MissingEnergy} res=${res} onRerun=${onRerun} />
    ${res.excluded?.length > 0 && html`<${Excluded} list=${res.excluded} />`}
    ${res.warnings.map((w) => html`<p class="small muted">⚠ ${w}</p>`)}
  </div>`;
}

// Reactions left out only because some of their molecules have no energy at
// this level (e.g. the separate molecules of a network expansion's products,
// which it optimized together): one minimization each brings them in.
function MissingEnergy({ res, onRerun }) {
  const list = res.missing_energy || [];
  const [jobIds, setJobIds] = useState(null);
  const jobs = useStore((s) => (jobIds ? jobIds.map((id) => s.jobs[id]).filter(Boolean) : []));
  if (!list.length) return null;
  const n = new Set((res.excluded || []).filter((x) => x.missing?.length).map((x) => x.reaction || x.edge)).size;
  const running = jobs.filter((j) => ['queued', 'running'].includes(j.status)).length;
  const failed = jobs.filter((j) => j.status === 'failed').length;
  const compute = async () => {
    // One job per charge and multiplicity (an optimize job shares them).
    const st = state.workspace?.structures || {};
    const groups = {};
    for (const m of list) {
      const r = st[m.id] || {};
      (groups[`${r.charge ?? 0}/${r.multiplicity ?? 1}`] ||= []).push(m.id);
    }
    const ids = [];
    for (const g of Object.values(groups)) {
      const out = await attempt(() => api.post('/api/jobs', { op: 'optimize', structures: g, params: {} }));
      if (out) ids.push(...out.map((j) => j.id));
    }
    if (ids.length) setJobIds(ids);
  };
  const names = list.slice(0, 6).map((m) => m.name).join(', ') + (list.length > 6 ? `, +${list.length - 6}` : '');
  return html`<div class="kn-missing">
    <p><b>${n} reaction${n === 1 ? '' : 's'} left out</b> because ${list.length} molecule${list.length === 1 ? ' has' : 's have'} no energy
      at this level: <span class="muted">${names}</span>.</p>
    ${!jobIds ? html`<button class="btn small primary" onClick=${compute} title="One minimization per molecule, at the workspace's level (seconds each with g-xTB)">Compute ${list.length === 1 ? 'it' : 'them'}</button>`
      : running ? html`<span class="small muted">Computing… ${jobs.length - running} of ${jobs.length} done</span>`
      : html`<span class="small">${failed ? `${failed} failed (see Calculations). ` : 'Done. '}</span>
          <button class="btn small primary" onClick=${() => { setJobIds(null); onRerun?.(); }}>Run kinetics again</button>`}
  </div>`;
}

// The parent network and a variant, run from the same mixture: what changes
// in the outcome and in each step's barrier.
function Compare({ res, T, onReaction }) {
  const [open, setOpen] = useState('');
  const { parent: a, variant: b, match } = res;
  const v = state.workspace.variants?.[res.variant_id];
  // A variant built on another one is compared with that one.
  const base = res.base_id ? state.workspace.variants?.[res.base_id]?.name || 'base' : 'parent';
  const rx = state.workspace.reactions || {};
  if (a.error || b.error) return html`${a.error && html`<p class="warn-box">${base}: ${a.error}</p>`}${b.error && html`<p class="warn-box">Variant: ${b.error}</p>`}`;
  const endOf = (r, ids) => ids.reduce((t, id) => t + (r.species.find((s) => s.id === id)?.final || 0), 0);
  const rows = [...a.species].sort((x, y) => y.max - x.max).filter((s) => s.max > 1e-9).slice(0, 10)
    .map((s) => ({ s, ids: match[s.id] || [s.id] }));
  // A variant's step stands for the step it was made from (through each variant it is built on).
  const from = (rid, target) => { for (let r = rid, k = 0; r && k < 20; r = rx[r]?.parent, k += 1) if (r === target) return true; return false; };
  const steps = a.steps.map((st) => ({ st, mine: b.steps.filter((x) => x.reaction && from(x.reaction, st.reaction)) }));
  const tgt = a.target;
  const tb = tgt ? endOf(b, match[tgt.id] || [tgt.id]) : null;
  // Parent steps the variant has no barrier for yet (still computing, or not carried): the variant is missing them.
  const missing = steps.filter(({ mine }) => !mine.length).map(({ st }) => st.label);
  // And variant steps whose base step has no barrier (e.g. the base variant's re-optimization failed).
  const matched = new Set(steps.flatMap(({ mine }) => mine));
  const baseMissing = b.steps.filter((x) => !matched.has(x)).map((x) => x.label);
  const nEst = b.steps.filter((x) => x.estimate).length;
  return html`<div class="kn-out">
    ${missing.length > 0 && html`<p class="warn-box small">Not in the variant yet: ${missing.join('; ')}. Until these have
      barriers the variant is missing those steps, so the comparison is incomplete.</p>`}
    ${baseMissing.length > 0 && html`<p class="warn-box small">Not in ${base}: ${baseMissing.join('; ')}. ${base} is missing
      those steps, so the comparison is incomplete.</p>`}
    ${nEst > 0 && html`<p class="level-note small">${nEst} variant step${nEst > 1 ? 's are' : ' is'} a fast estimate (est.):
      re-optimizations are running.</p>`}
    <p class="small kn-summary">${tgt && html`<b>${tgt.name}</b>: ${fmtC(tgt.final)} M → <b>${fmtC(tb)} M</b> with ${v?.name || 'the variant'}${
      tgt.final > 0 && tb != null ? ` (${(tb / tgt.final).toPrecision(2)}×)` : ''} · `}<span class="muted">${v?.edit.join('; ')}</span></p>
    <section>
      <div class="section-title">Amounts at the end (M)</div>
      <table class="rx-table"><thead><tr><th>Species</th><th>${base}</th><th>variant</th></tr></thead>
        <tbody>${rows.map(({ s, ids }) => html`<tr><td class="rx-eq" title=${ids.map((id) => state.workspace.structures[id]?.name || id).join(' + ')}>${s.name}${ids.length > 1 ? html` <span class="small muted">(${ids.length})</span>` : ''}</td>
          <td class="mono">${fmtC(s.final)}</td><td class="mono">${fmtC(endOf(b, ids))}</td></tr>`)}</tbody></table>
    </section>
    <section>
      <div class="section-title">Barriers ΔE‡ (kcal/mol)</div>
      <table class="rx-table"><thead><tr><th>Step</th><th>${base}</th><th>variant</th></tr></thead>
        <tbody>${steps.map(({ st, mine }) => html`<tr class=${st.reaction ? 'clickable' : ''} onClick=${() => st.reaction && onReaction(st.reaction)}>
          <td class="rx-eq">${st.label}</td><td class="mono">${st.barrier_fwd.toFixed(1)}</td>
          <td class="mono">${mine.length ? mine.map((x, k) => html`${k ? ' / ' : ''}<span title=${x.label}>${x.barrier_fwd.toFixed(1)}${x.estimate ? ' est.' : ''}</span>`) : '—'}</td></tr>`)}</tbody></table>
    </section>
    <div class="kn-row segmented">
      <button class=${open === 'a' ? 'on' : ''} onClick=${() => setOpen(open === 'a' ? '' : 'a')}>${base} in full</button>
      <button class=${open === 'b' ? 'on' : ''} onClick=${() => setOpen(open === 'b' ? '' : 'b')}>variant in full</button>
    </div>
    ${open && html`<${Results} res=${{ ...(open === 'a' ? a : b), tUnit: res.tUnit }} T=${T} onReaction=${onReaction} />`}
    ${(res.notes || []).map((w) => html`<p class="small muted">⚠ ${w}</p>`)}
  </div>`;
}

function Sweep({ sw, res }) {
  const rows = sw.temperatures.map((t, k) => ({ t, rate: sw.rate[k], ea: sw.apparent_ea_kcal[k] }));
  const ranked = [...res.species].sort((a, b) => b.max - a.max).filter((s) => s.max > 1e-12).slice(0, 6);
  const idx = Object.fromEntries(res.species.map((s, k) => [s.id, k]));
  const series = ranked.map((s, k) => ({ name: s.name, slot: k + 1, values: sw.final.map((f) => f[idx[s.id]]) }));
  const mid = rows[Math.floor(rows.length / 2)];
  return html`<section>
    <div class="section-title">With temperature</div>
    ${res.target && rows[0].rate > 0 && rows[rows.length - 1].rate > 0 && html`<p class="small">${res.target.name} forms ${speedup(rows[rows.length - 1].rate / rows[0].rate)} faster at ${Math.round(rows[rows.length - 1].t)} K than at ${Math.round(rows[0].t)} K${mid?.ea != null ? ` · apparent activation energy ≈ ${mid.ea.toFixed(1)} kcal/mol around ${Math.round(mid.t)} K` : ''}.</p>`}
    <${LineChart} xs=${sw.temperatures} series=${series} xLabel="temperature (K)" yLabel="amount at the end (M)" xlog=${false} />
  </section>`;
}

function speedup(r) {
  if (!Number.isFinite(r) || r <= 0) return '—';
  return r < 1000 ? `${r.toPrecision(2)}×` : `10^${Math.round(Math.log10(r))} times`;
}

function Excluded({ list }) {
  return html`<details class="small kn-excl"><summary>${list.length} reaction${list.length > 1 ? 's' : ''} not in the model</summary>
    <ul>${list.map((e) => html`<li><span class="rx-eq">${e.label}</span> <span class="muted">— ${e.reason}</span></li>`)}</ul></details>`;
}
