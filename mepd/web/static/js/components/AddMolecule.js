// Add a molecule: "I want more of this species; what does one more molecule in
// the flask do?" (mepd/web/addpage.py over mepd.web.added and `mepd addmol test`;
// docs/add-molecule.md). The page is a thin reader of files:
//   1. a target and conditions make the query (what limits the target today);
//   2. candidates (SMILES, a candidates file, an xyz file) go to one calculation;
//   3. their result files are rewritten as layers finish, cheapest first, and
//      read here every few seconds: one line per candidate, then the detail.
// Every number says whether it is an estimate (est.) or verified, and its spread.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, openReaction, openTab, set, setExploreView, state, toast, useStore } from '../store.js';
import { cls, depictUrl, isSpecies, readFileText } from '../util.js';
import { ControlBars, Mixture, UNITS, fmtC, fmtTime } from './Kinetics.js';
import { Viewer3D } from './Viewer3D.js';
import { Why } from './Why.js';

const kcal = (x, sign = false) => (x == null ? '—' : `${sign && x > 0 ? '+' : ''}${x.toFixed(1)}`);
const fmtX = (f) => (f >= 1e6 || f < 0.1 ? f.toExponential(1).replace('e+', 'e') : f >= 100 ? Math.round(+f.toPrecision(2)).toLocaleString('en-US') : +f.toPrecision(2));
const pct = (v) => (v == null ? '—' : v >= 0.995 ? (v > 0.99999 ? '100%' : '>99%') : v >= 0.01 ? `${Math.round(v * 100)}%` : v > 0 ? '<1%' : '0%');
const flat = (smiles) => (smiles || '').replace(/@+/g, '').replace(/[/\\]/g, '');
// A step read the way its material goes ("A -> B" written the other way round when it runs in reverse).
const stepLabel = (label, direction) => {
  const parts = (label || '').split(' -> ');
  return (direction === 'reverse' && parts.length === 2 ? `${parts[1]} → ${parts[0]}` : (label || '').replace(' -> ', ' → '));
};
const LAYER = { given: 'given with the candidate', bystander: 'bystander' };

// est. / verified: the state of a number, in the app's own word for an estimate.
function Tag({ status, title }) {
  if (!status) return null;
  const v = status === 'verified';
  return html`<span class=${`am-tag ${v ? 'ver' : 'est'}`} title=${title || (v
    ? 'Verified: a transition state optimized with the molecule present, its IRC checked'
    : 'An estimate: not a transition state optimized with the molecule present')}>${v ? 'verified' : 'est.'}</span>`;
}

// ------------------------------------------------------------ setup
function defaultMixture() {
  const draft = state.kineticsDraft?.mix || state.kinetics?.mix;
  if (draft?.length) return draft.map((m) => ({ ...m }));
  // What the latest exploration started from, 1 M each (a complex is its molecules).
  const jobs = Object.values(state.jobs).filter((j) => ['nanoreactor', 'graph-enumeration', 'network-mixture'].includes(j.op))
    .sort((a, b) => b.created - a.created);
  const st = state.workspace.structures;
  const c = {};
  for (const id of jobs[0]?.targets?.structures || []) {
    const rec = st[id];
    for (const m of rec?.role === 'complex' && rec.members?.length ? rec.members : rec ? [id] : []) if (st[m]) c[m] = 1;
  }
  return Object.keys(c).map((id) => ({ id, c: 1, held: false }));
}

const condText = (c) => `${c.initial.map((m) => `${m.name} ${fmtC(m.amount)} M`).join(' + ')} · ${+c.temperature.toFixed(1)} K · ${fmtTime(c.time_s)}`;

function Setup({ q, busy, onAsk, queries, onOpen }) {
  const structures = useStore((s) => s.workspace.structures);
  const reactions = useStore((s) => s.workspace.reactions || {});
  const preset = useStore((s) => s.addmol?.target || '');
  const species = useMemo(() => {
    const ids = new Set(Object.values(reactions).flatMap((r) => [...r.reactants, ...r.products]));
    return Object.values(structures).filter((s) => isSpecies(s) && !s.variant && ids.has(s.id)).sort((a, b) => a.name.localeCompare(b.name));
  }, [structures, reactions]);
  const [target, setTarget] = useState(preset);
  const [together, setTogether] = useState(true);
  const [mix, setMix] = useState(defaultMixture);
  const [T, setT] = useState(state.kineticsDraft?.T ?? 298.15);
  const [tVal, setTVal] = useState(state.kineticsDraft?.tVal ?? 24);
  const [tUnit, setTUnit] = useState(state.kineticsDraft?.tUnit ?? 'h');
  useEffect(() => { if (preset) setTarget(preset); }, [preset]);
  // An earlier question opened: the form shows what it asked.
  useEffect(() => {
    if (!q) return;
    const c = q.conditions, a = q.asked;
    setTarget(a ? a.target : q.target.species[0]); setTogether(a ? a.together : q.target.species.length > 1);
    setMix(a ? Object.entries(a.initial).map(([id, amount]) => ({ id, c: amount, held: a.held.includes(id) }))
      : c.initial.map((m) => ({ id: m.id, c: m.amount, held: c.held.includes(m.id) })));
    setT(c.temperature);
    const u = ['d', 'h', 'min', 's'].find((k) => Number.isInteger(+(c.time_s / UNITS[k]).toFixed(6))) || 's';
    setTUnit(u); setTVal(+(c.time_s / UNITS[u]).toFixed(6));
  }, [q?.id]);
  const rec = structures[target];
  const twins = rec?.smiles ? species.filter((s) => s.id !== target && flat(s.smiles) === flat(rec.smiles)) : [];
  const ask = () => onAsk({ target, together: twins.length > 0 && together,
    initial: Object.fromEntries(mix.map((m) => [m.id, m.c])), held: mix.filter((m) => m.held).map((m) => m.id),
    temperature: T, time_s: tVal * UNITS[tUnit] });
  return html`
    <div class="section-title">Target<${Why} id="addmol.target" label="The target" /></div>
    <div class="kn-row"><label>I want more of <select value=${target} onChange=${(e) => setTarget(e.target.value)}>
      <option value="">— pick a species —</option>
      ${species.map((s) => html`<option value=${s.id}>${s.name}</option>`)}</select></label></div>
    ${rec?.smiles && html`<img class="am-target-img" src=${depictUrl(rec.smiles, 180, 110)} alt=${rec.name} />`}
    ${twins.length > 0 && html`<label class="kn-row small" title="The network keeps mirror images and other stereoisomers as separate species">
      <input type="checkbox" checked=${together} onChange=${(e) => setTogether(e.target.checked)} />
      count its ${twins.length + 1} stereoisomers together</label>`}
    <div class="section-title">Conditions</div>
    <${Mixture} items=${mix} onChange=${setMix} species=${species} />
    <div class="kn-row"><label>Temperature <input type="number" value=${T} min="1" step="10" onInput=${(e) => setT(+e.target.value)} /> K</label></div>
    <div class="kn-row"><label>Time <input type="number" value=${tVal} min="0" step="1" onInput=${(e) => setTVal(+e.target.value)} /></label>
      <select value=${tUnit} onChange=${(e) => setTUnit(e.target.value)}>${Object.keys(UNITS).map((u) => html`<option value=${u}>${u}</option>`)}</select></div>
    <button class="btn primary" disabled=${busy || !target || !mix.length || !(T > 0) || !(tVal > 0)} onClick=${ask}>
      ${busy ? html`<span class="spinner" aria-hidden="true"></span>Working out what limits it…` : 'What limits it today?'}</button>
    ${busy && html`<p class="small muted">The first time, the free energies of every species and transition state are computed (GFN2-xTB Hessians): a few minutes for a hundred steps. They are kept.</p>`}
    ${queries.length > 0 && html`<div class="kn-row"><label class="small">Earlier questions <select value=${q?.id || ''} onChange=${(e) => e.target.value && onOpen(e.target.value)}>
      <option value="">—</option>
      ${queries.map((x) => html`<option value=${x.id}>${x.target.names[0]}${x.target.names.length > 1 ? ` (+${x.target.names.length - 1})` : ''} · ${
        +x.conditions.temperature.toFixed(0)} K · ${fmtTime(x.conditions.time_s)} · ${x.results} candidate${x.results === 1 ? '' : 's'}</option>`)}</select></label></div>`}`;
}

// Candidates: typed or pasted lines, a candidates file, an xyz file, or the molecule in Design.
function Candidates({ q, onStarted }) {
  const design = useStore((s) => s.workspace.design);
  const [text, setText] = useState('');
  const [filename, setFilename] = useState('');
  const [amount, setAmount] = useState(0.1);
  const [read, setRead] = useState(null);
  const [busy, setBusy] = useState(false);
  const file = useRef(null);
  const body = (dry) => ({ text, filename, amount, dry_run: dry });
  const look = async (t = text, f = filename) => {
    if (!t.trim()) { setRead(null); return; }
    const out = await attempt(() => api.post(`/api/addmol/${q.id}/test`, { text: t, filename: f, amount, dry_run: true }));
    setRead(out ? out.candidates : null);
  };
  useEffect(() => { if (read) look(); }, [amount]);
  const load = async (f) => {
    const t = await readFileText(f);
    setText(t); setFilename(f.name); look(t, f.name);
  };
  const run = async () => {
    setBusy(true);
    const out = await attempt(() => api.post(`/api/addmol/${q.id}/test`, body(false)));
    setBusy(false);
    if (!out) return;
    toast(`${out.candidates} candidate${out.candidates === 1 ? '' : 's'}: one calculation started, cheapest numbers first`, 'ok');
    setText(''); setFilename(''); setRead(null);
    onStarted(out.job);
  };
  const bad = (read || []).filter((c) => c.error);
  const lines = text.trim() ? text.trim().split('\n').length : 0;
  const long = filename && lines > 12;
  return html`
    <div class="section-title">Candidates<${Why} id="addmol.candidates" label="Candidates" /></div>
    ${long ? html`<p class="small">${filename} <button class="btn-link small" onClick=${() => { setText(''); setFilename(''); setRead(null); }}>clear</button></p>`
      : html`<textarea class="am-text mono" rows="4" spellcheck="false" placeholder=${'One per line: SMILES, a name, an amount (mol/L)\nOC(=O)[C@@H]1CCCN1 L-proline 0.1\nCC(=O)O'}
        value=${text} onInput=${(e) => { setText(e.target.value); setFilename(''); setRead(null); }} onBlur=${() => look()}></textarea>`}
    <div class="kn-row"><label class="small" title="For every candidate that gives no amount of its own">Amount
      <input type="number" class="tiny" min="0" step="0.05" value=${amount} onInput=${(e) => setAmount(+e.target.value)} /> M each</label>
      <input type="file" ref=${file} hidden accept=".json,.smi,.txt,.xyz" onChange=${(e) => { if (e.target.files[0]) load(e.target.files[0]); e.target.value = ''; }} />
      <button class="btn small" title="A candidates file (JSON, as a generative model writes it), SMILES lines (.smi) or a structure (.xyz)"
        onClick=${() => file.current.click()}>Load a file…</button>
      ${design?.smiles && html`<button class="btn small" title=${`Add the molecule in Design: ${design.smiles}`}
        onClick=${() => { const t = `${text.trim() ? `${text.trim()}\n` : ''}${design.smiles} ${(design.name || 'design').replace(/\s+/g, '_')}`; setText(t); setFilename(''); look(t, ''); }}>From Design</button>`}
    </div>
    ${read && read.length > 0 && html`<table class="rx-table am-read"><tbody>${read.map((c) => html`<tr class=${c.error ? 'am-bad' : ''}>
      <td>${c.smiles && !c.error ? html`<img src=${depictUrl(c.smiles, 60, 45)} alt="" />` : ''}</td>
      <td class="rx-eq">${c.name}<div class="small muted">${c.error || `${c.formula || `${c.natoms} atoms`} · ${fmtC(c.amount)} M`}</div></td></tr>`)}</tbody></table>`}
    <button class="btn primary" disabled=${busy || !text.trim() || bad.length > 0 || !(amount > 0)} onClick=${run}>
      ${busy ? 'Starting…' : read?.length ? `Test ${read.length} candidate${read.length === 1 ? '' : 's'}` : 'Test'}</button>
    <p class="small muted">For a model: <a href=${`/api/addmol/${q.id}/query.json`} download>download the query file</a>${' '}
      (the network, its geometries and what limits the target, readable without mepd)<${Why} id="addmol.query" label="The query file" />.</p>`;
}

// ------------------------------------------------------------ what limits the target
// A route as its steps' heights against what the conditions allow: one hue, the cutoff a line.
function RouteBars({ route, cutoff, top }) {
  const max = Math.max(top, cutoff) * 1.04;
  return html`<div class="am-route">
    ${route.steps.map((s, k) => html`<div class="am-route-row" title=${`${stepLabel(s.label, s.direction)}: its transition state is ${kcal(s.height)} kcal/mol above the starting mixture, ${kcal(s.height - cutoff)} above what these conditions allow`}>
      <span class="am-route-n">${k + 1}</span>
      <span class="am-route-label">${stepLabel(s.label, s.direction)}</span>
      <span class="am-route-track"><span class=${cls('am-route-bar', s.height > cutoff && 'over')} style=${{ width: `${(Math.max(0, s.height) / max) * 100}%` }}></span>
        <span class="am-route-cut" style=${{ left: `${(cutoff / max) * 100}%` }}></span></span>
      <span class="am-route-v mono">${kcal(s.height)}${s.height > cutoff ? html` <span class="muted">(${kcal(s.height - cutoff, true)})</span>` : ''}</span>
    </div>`)}
  </div>`;
}

function Limits({ q }) {
  const lim = q.limits, t = q.target, c = q.conditions;
  const name = t.names.length > 1 ? `${t.names[0]} and its ${t.names.length - 1 === 1 ? 'stereoisomer' : `${t.names.length - 1} stereoisomers`}` : t.names[0];
  const total = c.initial.reduce((a, m) => a + m.amount, 0);
  const openStep = (r) => { if (r.edge) openReaction(r.edge); };
  const energy = `${q.source === 'explorer run' ? 'The exploration\'s own network' : 'The workspace\'s network'}: ${q.counts.steps} verified steps, ${q.energies === 'free' ? 'free energies at 1 M' : 'electronic energies'}`;
  const head = html`<p class="am-lede"><b>${name}</b>: ${t.resolved
    ? html`<span class="mono">${fmtC(t.final)} M</span> today (${fmtC((100 * t.final) / total)}% of the start), half of it after ${t.time_to_fraction_s != null ? fmtTime(t.time_to_fraction_s) : '—'}`
    : html`<b>none forms today</b> <span class="muted">(${fmtC(t.final)} M: below what the kinetics resolve)</span>`}
    <span class="muted"> · ${condText(c)}</span></p>`;
  if (lim.defined) {
    const rows = lim.order.filter((o) => Math.abs(o.control) >= 0.01).slice(0, 8).map((o) => ({ ...o, x: o.control, label: `${stepLabel(o.label)}${o.mirror ? ' (mirror image)' : ''}` }));
    const quiet = lim.order.length - rows.length;
    const traps = lim.traps.filter((s) => Math.abs(s.control) > 0.02).slice(0, 6).map((s) => ({ label: s.name, x: s.control }));
    return html`<section class="am-card">
      <div class="section-title">What limits it today<${Why} id="addmol.limits" label="What limits the target" /></div>
      ${head}
      ${rows.length ? html`<p class="small muted">What lowering each transition state by 1 kcal/mol does to the target: blue, more of it (a molecule should lower these);
        red, less (a competing step: do not lower it). Largest first: the order candidates are worked in.</p>
        <${ControlBars} rows=${rows} T=${c.temperature} onPick=${openStep} />
        ${quiet > 0 && html`<p class="small muted">${quiet} other step${quiet > 1 ? 's have' : ' has'} next to no effect on it.</p>`}`
        : html`<p class="small">No barrier controls it at these conditions: its amount is set by the species' energies.</p>`}
      ${traps.length > 0 && html`<p class="small muted kn-sub">Traps: what making each of these 1 kcal/mol more stable does to the target. A molecule that binds them costs target.</p>
        <${ControlBars} rows=${traps} T=${c.temperature} />`}
      <p class="small muted">Degree of rate control. ${energy}.</p>
    </section>`;
  }
  const [first, ...others] = lim.routes;
  return html`<section class="am-card">
    <div class="section-title">What limits it today<${Why} id="addmol.limits" label="What limits the target" /></div>
    ${head}
    ${first ? html`<p class="small muted">The easiest route to it, step by step: how high each transition state stands above the starting mixture (kcal/mol).
        The line is what ${+c.temperature.toFixed(0)} K over ${fmtTime(c.time_s)} allows (<span class="mono">${kcal(lim.cutoff_kcal)}</span>): a molecule has to bring every step under it. In brackets: how far above.</p>
      <${RouteBars} route=${first} cutoff=${lim.cutoff_kcal} top=${Math.max(...lim.routes.map((r) => r.peak))} />
      ${others.length > 0 && html`<details class="small"><summary>${others.length} other route${others.length > 1 ? 's' : ''}, peaking at ${others.map((r) => kcal(r.peak)).join(' and ')} kcal/mol</summary>
        ${others.map((r) => html`<${RouteBars} route=${r} cutoff=${lim.cutoff_kcal} top=${Math.max(...lim.routes.map((x) => x.peak))} />`)}</details>`}
      <p class="small muted">Which steps compete with the target cannot be said while none of it forms. ${energy}.</p>`
      : html`<p class="warn-box small">No route of verified steps leads from the starting mixture to it.</p>`}
  </section>`;
}

// ------------------------------------------------------------ one line per candidate
function effectText(r, q) {
  const t = r.target;
  if (t.factor != null) {
    if (t.factor >= 1.05) return { main: `${fmtX(t.factor)}× more`, kind: 'pos', sub: `${fmtC(t.parent.final)} → ${fmtC(t.with.final)} M` };
    if (t.factor <= 0.95) return { main: `${fmtX(1 / t.factor)}× less`, kind: 'neg', sub: `${fmtC(t.parent.final)} → ${fmtC(t.with.final)} M` };
    return { main: 'no change', kind: '', sub: `${fmtC(t.with.final)} M` };
  }
  if (t.with.resolved) return { main: `${fmtC(t.with.final)} M`, kind: 'pos', sub: 'none without it' };
  const drop = r.peak.parent != null && r.peak.with != null ? r.peak.with - r.peak.parent : 0;
  return { main: 'still none', kind: '', short: Math.abs(drop) > 0.05 ? `route ${kcal(r.peak.parent)} → ${kcal(r.peak.with)}` : 'route unchanged',
    sub: Math.abs(drop) > 0.05 ? `easiest route ${kcal(r.peak.parent)} → ${kcal(r.peak.with)} kcal/mol (${kcal(q.limits.cutoff_kcal)} needed)`
      : 'easiest route unchanged' };
}
function soonerText(r) {
  const t = r.target;
  if (t.time_to_parent_amount_s == null || t.parent.time_to_fraction_s == null) return null;
  return { main: `after ${fmtTime(t.time_to_parent_amount_s)}`, sub: `today: ${fmtTime(t.parent.time_to_fraction_s)}` };
}
// Two facts side by side: is the molecule on the route that carries the target's material, and how often it comes back.
function fateText(r) {
  const turn = r.returns ?? 0, on = r.on_route ?? 0;
  const turns = turn >= 0.001 ? `${fmtX(turn)} turnover${turn === 1 ? '' : 's'} each` : 'no turnover yet';
  // `fact`: the one short fact of the table's row; `sub`: the sentence of the detail and the row's tooltip.
  const short = turn >= 0.001 ? `${fmtX(turn)} turnovers` : 'no turnover yet';
  if (r.fate === 'consumed') return { main: 'used up', fact: `${pct(r.shares?.covalent)} in new species`, sub: `${pct(r.shares?.covalent)} ends in new species` };
  if (r.fate === 'bound and inert') return { main: 'held, inert', fact: `${pct(r.shares?.bound)} bound`, sub: `${pct(r.shares?.bound)} bound in complexes` };
  if (r.fate === 'regenerated') return { main: 'returned', fact: short, sub: `${turns}${on >= 0.001 ? ` · on the route of ${pct(on)} of the target` : ''}` };
  if (on >= 0.001) return { main: 'on the route', fact: short, sub: `of ${pct(on)} of the target · ${turns}` };
  return { main: 'takes no part', fact: '', sub: 'no step with it carries material' };
}
const restText = (r) => (r.resting_state ? { main: r.resting_state.id === 'added' ? 'free' : r.resting_state.name, sub: pct(r.resting_state.share) } : null);
const stateOf = (r) => (r.used?.est ? 'est' : r.used?.verified ? 'verified' : null);

function Progress({ r }) {
  if (r.status === 'queued') return html`<span class="am-run">queued</span>`;
  const lay = (r.layers || []).find((l) => l.status === 'running');
  if (r.status === 'running') {
    return html`<span class="am-run"><span class="spinner" aria-hidden="true"></span>${lay ? `${LAYER[lay.name] || lay.name} ${lay.done ?? 0}/${lay.total ?? '?'}` : 'starting'}</span>`;
  }
  if (r.status === 'interrupted') return html`<span class="am-run warn" title="Its calculation stopped before the last layer finished: the numbers are from the layers that did">stopped early</span>`;
  if (r.status === 'failed') return html`<span class="am-run bad">failed</span>`;
  return null;
}

// One glance per candidate: each cell is a verdict with its est./verified tag and one short fact below;
// the sentences are the cell's tooltip and the detail panel.
function Cell({ v, tag, title }) {
  if (!v) return html`<td class="muted">—</td>`;
  return html`<td title=${title || v.sub || ''}><span class=${cls('am-main', v.kind)}>${v.main}</span>${tag}<div class="small muted">${v.short ?? v.sub}</div></td>`;
}

function Table({ q, rows, sel, onPick, picked, onToggle }) {
  return html`<div class="rx-table-wrap"><table class="rx-table am-table">
    <thead><tr><th></th><th>Candidate</th>
      <th title="What this row's kinetics are built on: est., estimates; verified, only transition states optimized with the molecule present. It holds for the target, how soon and returned.">Built on</th>
      <th title="The target's amount at the end, with the molecule against without">Target</th>
      <th title="When the mixture with the molecule holds what today's takes half its run to make">How soon</th>
      <th title="Whether the molecule comes back: its turnovers in the run, whether it is on the route that carries the target's material, and where it rests (hover a row; the detail has it all)">Returned?</th>
      <th title="Among the steps that limit the target, the one whose barrier changes most through the molecule">Largest step change (kcal/mol)</th></tr></thead>
    <tbody>${rows.map((r) => {
      const s = stateOf(r);
      const b = r.best;
      const within = b && b.spread != null && b.spread >= Math.abs(b.change);
      const fate = r.readout && s ? fateText(r) : null, rest = r.readout ? restText(r) : null;
      const sooner = r.readout && s ? soonerText(r) : null;
      const there = r.status !== 'queued';   // (a result file exists)
      const none = r.why ? `No number: ${r.why}` : r.status === 'queued' ? 'Waiting its turn: the candidates of one calculation are tested one after another.'
        : r.status === 'running' ? 'No number yet: the first arrive within a minute.' : 'No layer gave a number.';
      return html`<tr class=${cls(there && 'clickable', sel === r.file && 'on')} onClick=${() => there && onPick(r.file)}>
        <td><input type="checkbox" checked=${picked.has(r.file)} disabled=${!r.readout || !s} title="Compare step by step" aria-label=${`Compare ${r.name}`}
          onClick=${(e) => e.stopPropagation()} onChange=${() => onToggle(r.file)} /></td>
        <td title=${r.name}><div class="am-cand">${r.smiles ? html`<img src=${depictUrl(r.smiles, 60, 45)} alt="" />` : ''}<div><b>${r.name}</b>
          <div class="small muted">${r.amount != null ? `${fmtC(r.amount)} M` : ''} <${Progress} r=${r} /></div></div></div></td>
        ${r.readout && s ? html`
          <td><${Tag} status=${s} /></td>
          <${Cell} v=${effectText(r, q)} />
          <${Cell} v=${sooner && { main: sooner.main.replace('after ', ''), sub: sooner.sub }} title=${sooner ? `As much as today's mixture makes in half its run, ${sooner.main} (${sooner.sub})` : ''} />
          <td title=${`${fate.sub}${rest ? `\nrests as ${rest.main} (${rest.sub} over the run)` : ''}`}><span class="am-main">${fate.main}</span><div class="small muted">${fate.fact}</div></td>
          ${b ? html`<td title=${`${stepLabel(b.label, b.direction)}\n${kcal(b.before)} → ${kcal(b.after)} kcal/mol${b.spread == null ? ', no spread came with this number' : `, spread ± ${b.spread.toFixed(1)}`}${
              within ? '\nThe change is no larger than its spread between repeats: not a signal.' : ''}${b.limits ? '' : '\nNo step that limits the target goes through it: this is the largest change anywhere.'}`}>
            <span class=${cls('am-main mono', within ? 'am-weak' : b.change < 0 ? 'pos' : 'neg')}>${kcal(b.change, true)}</span>
            <span class="mono small"> ${b.spread != null ? `± ${b.spread.toFixed(1)}` : '± ?'}</span>
            <div class="small muted"><${Tag} status=${b.status} />${within ? html` <span class="am-flag">within spread</span>` : ''}${b.limits ? '' : html` <span class="am-flag" title="No step that limits the target goes through it">off-route</span>`}</div></td>`
            : html`<td class="muted small">none through it</td>`}`
          : html`<td colspan="5" class=${cls('small', r.status === 'failed' || r.why ? 'am-warn' : 'muted')} title=${none}><div class="am-two">${none}</div></td>`}
      </tr>`;
    })}</tbody></table></div>`;
}

// ------------------------------------------------------------ one candidate
// Barrier before and after on one axis: hollow without the molecule, filled with it
// (blue lower, red higher), the whisker its spread.
function Dumbbell({ before, after, spread, lo, hi }) {
  const X = (v) => `${Math.min(100, Math.max(0, ((v - lo) / Math.max(1e-9, hi - lo)) * 100))}%`;
  const a = Math.min(before, after), b = Math.max(before, after);
  const kind = after < before ? 'pos' : 'neg';
  return html`<span class="am-db">
    ${spread != null && spread > 0 && html`<span class="am-db-spread" style=${{ left: X(after - spread), right: `calc(100% - ${X(after + spread)})` }}></span>`}
    <span class=${`am-db-bar ${kind}`} style=${{ left: X(a), right: `calc(100% - ${X(b)})` }}></span>
    <span class="am-db-dot before" style=${{ left: X(before) }}></span>
    <span class=${`am-db-dot ${kind}`} style=${{ left: X(after) }}></span>
  </span>`;
}

function StepRows({ rows, onView, viewing, through }) {
  const vals = rows.flatMap((s) => [s.before, s.after, s.after - (s.spread || 0), s.after + (s.spread || 0)]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max(1, (hi - lo) * 0.06);
  return html`<div class="rx-table-wrap"><table class="rx-table am-steps">
    <thead><tr><th>Step</th><th title="Apparent barrier: the transition state above its reactants at the amounts they have in the run">without</th>
      <th class="am-db-head"><span>${kcal(lo - pad)}</span><span>kcal/mol</span><span>${kcal(hi + pad)}</span></th><th>with</th><th>change</th>
      ${through && html`<th>From</th><th></th>`}</tr></thead>
    <tbody>${rows.map((s) => {
      const within = s.spread != null && s.spread >= Math.abs(s.change);
      return html`<tr class=${cls(viewing === s.step && 'on')}>
        <td class="rx-eq">${stepLabel(s.label, s.direction)}${s.via ? html`<div class="small muted">${s.via}</div>` : ''}</td>
        <td class="mono">${kcal(s.before)}</td>
        <td class="am-db-cell" title=${`${kcal(s.before)} → ${kcal(s.after)} kcal/mol${s.spread != null ? `, spread ± ${s.spread.toFixed(1)}` : ''}`}>
          <${Dumbbell} before=${s.before} after=${s.after} spread=${s.spread} lo=${lo - pad} hi=${hi + pad} /></td>
        <td class="mono">${kcal(s.after)}${through ? html` <${Tag} status=${s.status} />` : ''}</td>
        <td class="mono"><b>${kcal(s.change, true)}</b>${through ? html` <span class=${within ? 'am-warn' : 'muted'} title=${s.spread == null ? 'No spread came with this number' : within ? 'The change is no larger than its spread between repeats' : 'Spread between repeats'}>${
          s.spread != null ? `± ${s.spread.toFixed(1)}` : '± ?'}</span>` : ''}</td>
        ${through && html`<td class="small muted" title=${`${s.source || ''}${s.level ? `\nlevel: ${s.level}` : ''}`}>${LAYER[s.layer] || s.layer || '—'}${s.level ? html`<div class="am-level">${s.level}</div>` : ''}</td>
          <td>${(s.geometry || s.with_geometry) && html`<button class="btn small ghost" title="Its transition state in 3D" onClick=${() => onView(s)}>3D</button>`}</td>`}
      </tr>`;
    })}</tbody></table></div>`;
}

function StepViewer({ qid, file, step, onClose }) {
  const [g, setG] = useState(null);
  const [frame, setFrame] = useState(1);
  useEffect(() => {
    let alive = true;
    setG(null);
    attempt(() => api.get(`/api/addmol/${qid}/geometry?step=${encodeURIComponent(step.step)}&result=${encodeURIComponent(file)}&through=${encodeURIComponent(step.through || '')}`))
      .then((out) => { if (alive && out) { setG(out); setFrame(out.ts); } });
    return () => { alive = false; };
  }, [qid, file, step.step]);
  const words = g?.frames.length === 3 ? ['reactant end', 'TS', 'product end'] : ['TS'];
  return html`<div class="am-viewer">
    <div class="kn-row"><b>${stepLabel(step.label, step.direction)}</b><span class="spacer"></span>
      ${g && g.frames.length > 1 && html`<div class="segmented mini">${words.map((w, k) => html`<button class=${frame === k ? 'on' : ''} onClick=${() => setFrame(k)}>${w}</button>`)}</div>`}
      <button class="btn-icon" aria-label="Close" onClick=${onClose}>✕</button></div>
    ${g ? html`<${Viewer3D} frames=${g.frames} frame=${frame} height=${300} />
      <p class="small muted">${g.with_molecule ? 'With the molecule where this estimate left it.'
        : 'The network\'s own transition state, without the molecule: this estimate came with no geometry.'}</p>`
      : html`<p class="small muted">Loading the geometry…</p>`}
  </div>`;
}

function Detail({ q, file, tick, onGone }) {
  const [d, setD] = useState(null);
  const [view, setView] = useState(null);
  useEffect(() => { setD(null); setView(null); }, [file]);
  useEffect(() => {
    let alive = true;
    api.get(`/api/addmol/${q.id}/results/${encodeURIComponent(file)}`).then((out) => { if (alive) setD(out); }).catch(() => {});
    return () => { alive = false; };
  }, [q.id, file, tick]);
  if (!d) return html`<section class="am-card"><p class="small muted">Reading the result…</p></section>`;
  const name = d.name;
  const variant = (q.results.find((r) => r.file === file) || {}).variant;
  const show = async () => {
    const out = await attempt(() => api.post(`/api/addmol/${q.id}/results/${encodeURIComponent(file)}/keep`));
    if (!out) return;
    set({ network: out.variant });
    setExploreView('map');
    openTab('graph');
  };
  const remove = async () => {
    if (!window.confirm(`Remove the result of ${name}?`)) return;
    if (await attempt(() => api.del(`/api/addmol/${q.id}/results/${encodeURIComponent(file)}`))) onGone();
  };
  const head = html`<div class="am-detail-head">
    ${d.smiles && html`<img src=${depictUrl(d.smiles, 120, 90)} alt="" />`}
    <div><h3>${name}</h3><div class="small muted">${d.amount != null ? `${fmtC(d.amount)} M added` : ''}${d.smiles ? ` · ${d.smiles}` : ''}</div>
      <div class="am-layers">${(d.layers || []).map((l) => html`<span class=${`am-layer ${l.status === 'stopped early' ? 'failed' : l.status}`} title=${`${l.what || ''}${l.level ? `\nlevel: ${l.level}` : ''}`}>
        ${l.status === 'running' ? html`<span class="spinner" aria-hidden="true"></span>` : ''}${LAYER[l.name] || l.name}: ${l.status}${
          l.total ? ` ${l.done ?? 0}/${l.total}` : l.entries != null ? ` · ${l.entries} number${l.entries === 1 ? '' : 's'}` : ''}${l.wall_s ? ` · ${fmtTime(l.wall_s)}` : ''}</span>`)}
</div></div>
    <span class="spacer"></span>
    <div class="am-actions">
      ${d.readout && html`<button class="btn small" title="Keep it as a variant of the network and mark its changes on the network map" onClick=${show}>${variant ? 'Show on the map' : 'Show on the map…'}</button>`}
      ${d.job && html`<button class="btn small ghost" onClick=${() => openJob(d.job)}>Calculation</button>`}
      <button class="btn small ghost danger-text" title="Remove this result" onClick=${remove}>✕</button></div>
  </div>`;
  if (!d.readout) {
    return html`<section class="am-card">${head}
      <p class="warn-box">${d.failed.length ? 'No result: ' : d.status === 'queued' ? 'Waiting its turn.' : 'No number yet.'}${d.failed.map((f) => f.why).join(' ')}</p></section>`;
  }
  const eff = effectText(d, q), fate = fateText(d), s = stateOf(d);
  const t = d.target, fo = d.first_order_detail;
  const peak = d.peak;
  const moved = t.factor != null ? Math.abs(Math.log(t.factor)) > 0.05
    : t.with.resolved !== t.parent.resolved || Math.abs((peak.with ?? 0) - (peak.parent ?? 0)) > 0.05;
  const nothing = !d.through_steps.length && !moved;
  const quiet = d.new_species.filter((x) => !(x.max > 1e-9)).length + (d.new_species_total - d.new_species.length);
  const held = d.new_species.filter((x) => x.max > 1e-9);
  const why = {};
  d.failed.forEach((f) => { (why[f.why || f.status || 'failed'] ||= []).push(f.label || f.layer); });
  return html`<section class="am-card">${head}
    ${d.status === 'running' && html`<p class="level-note small">Still running: what follows is from the numbers that are in, and changes as more arrive.</p>`}
    ${d.status === 'interrupted' && html`<p class="warn-box small">Its calculation stopped before the last layer finished: what follows is from the numbers that were in.</p>`}
    ${d.status === 'failed' && html`<p class="warn-box small">A layer failed: ${d.failed.map((f) => f.why).join(' ')} What follows is from the layers that ran.</p>`}
    ${nothing && d.status !== 'running' && html`<p class="level-note">No effect at these conditions: ${name} lowers no step of the network${s ? '' : ' (no layer gave a number)'}, and the target's amount stays where it is. That is a result: ${
      d.used.est + d.used.verified} estimate${d.used.est + d.used.verified === 1 ? '' : 's'} went into it.</p>`}
    <dl class="am-facts">
      <dt>Target at the end</dt><dd><span class="mono">${t.parent.resolved ? fmtC(t.parent.final) : 'none'} → ${t.with.resolved ? `${fmtC(t.with.final)} M` : 'none'}</span>
        ${t.factor != null && html` <b class=${cls('am-main', eff.kind)}>${eff.main}</b>`} <${Tag} status=${s} />
        ${(!t.with.resolved || !t.parent.resolved) && html`<div class="small muted">None: below what the kinetics resolve, 1e-9 of the starting material (${!t.parent.resolved ? `without ${fmtC(t.parent.final)} M` : ''}${!t.parent.resolved && !t.with.resolved ? ', ' : ''}${!t.with.resolved ? `with ${fmtC(t.with.final)} M` : ''}).</div>`}
        ${fo?.factor != null && html`<div class="small muted">First-order estimate from the degrees of rate control: ${fmtX(fo.factor)}× (the full kinetics: ${t.factor != null ? `${fmtX(t.factor)}×` : '—'})<${Why} id="addmol.first_order" label="The first-order estimate" /></div>`}</dd>
      ${soonerText(d) && html`<dt>As much as today's half</dt><dd><span class="mono">${soonerText(d).main}</span> <span class="muted">(today: ${fmtTime(t.parent.time_to_fraction_s)})</span> <${Tag} status=${s} /></dd>`}
      <dt>Easiest route to it</dt><dd><span class="mono">${kcal(peak.parent)} → ${kcal(peak.with)} kcal/mol</span> at its highest point <${Tag} status=${peak.with !== peak.parent ? s : 'verified'} />
        <span class="muted"> · ${kcal(q.limits.cutoff_kcal)} is what these conditions allow</span></dd>
      <dt>Is it returned?<${Why} id="addmol.fate" label="Returned, used up or held" /></dt><dd><b>${fate.main}</b> <${Tag} status=${s} /> <span class="muted">${fate.sub}</span>
        <div class="small muted">On the target's route: ${d.on_route >= 0.001 ? `${pct(d.on_route)} of the target's material passes through steps with it` : 'no'}.
          Turnovers per molecule in ${fmtTime(q.conditions.time_s)}: <span class="mono">${d.returns != null ? fmtC(d.returns) : '—'}</span>.</div></dd>
      <dt>Where it is at the end</dt><dd>${d.shares ? html`free ${pct(d.shares.free)} · bound in complexes ${pct(d.shares.bound)} · in new species ${pct(d.shares.covalent)}` : '—'}
        ${d.resting_state && html`<span class="muted"> · rests as ${d.resting_state.id === 'added' ? 'the free molecule' : d.resting_state.name} (${pct(d.resting_state.share)} over the run)</span>`} <${Tag} status=${s} /></dd>
    </dl>

    <div class="section-title">Steps that go through ${name} (${d.through_steps.length})<${Why} id="addmol.steps" label="Barrier before and after" /></div>
    ${d.through_steps.length ? html`
      <p class="small muted am-key"><i class="am-db-dot before"></i>without <i class="am-db-dot pos"></i>with, lower <i class="am-db-dot neg"></i>with, higher <i class="am-db-spread"></i>spread between repeats
        · apparent barriers, ${q.energies === 'free' ? 'free energy' : 'electronic'}</p>
      <${StepRows} rows=${d.through_steps} through onView=${(x) => setView(view?.step === x.step ? null : x)} viewing=${view?.step} />
      ${view && html`<${StepViewer} qid=${q.id} file=${file} step=${view} onClose=${() => setView(null)} />`}`
      : html`<p class="small muted">${d.used.est + d.used.verified ? `None: every step of the network is still fastest without ${name}.` : 'No layer has given a number for a step yet.'}</p>`}
    ${d.moved_steps.length > 0 && html`<details class="small"><summary>${d.moved_steps.length} step${d.moved_steps.length > 1 ? 's' : ''} moved only because amounts moved</summary>
      <p class="small muted">Their own transition states are unchanged: their reactants hold more or less material with ${name} in the flask.</p>
      <${StepRows} rows=${d.moved_steps.slice(0, 30)} /></details>`}
    <p class="small muted">${d.unchanged} step${d.unchanged === 1 ? '' : 's'} of the network unchanged.</p>

    <div class="section-title">What is new</div>
    ${d.new_steps.length > 0 && html`<table class="rx-table"><thead><tr><th>Step with ${name}</th><th>barrier</th><th>From</th></tr></thead>
      <tbody>${d.new_steps.map((x) => {
        const v = x.barrier?.free ?? x.barrier?.electronic;
        return html`<tr><td class="rx-eq">${x.label || x.id}</td>
          <td class="mono">${kcal(v)} <span class="muted">${x.barrier?.spread != null ? `± ${x.barrier.spread.toFixed(1)} ` : ''}kcal/mol${x.barrier?.free == null ? ', electronic' : ''}</span> <${Tag} status=${x.status} /></td>
          <td class="small muted" title=${x.source || ''}>${LAYER[x.layer] || x.layer || '—'}${x.level ? html`<div class="am-level">${x.level}</div>` : ''}</td></tr>`;
      })}</tbody></table>`}
    ${held.length > 0 && html`<table class="rx-table"><thead><tr><th>Species holding ${name}</th><th>most at any time (M)</th><th>at the end (M)</th></tr></thead>
      <tbody>${held.slice(0, 8).map((x) => html`<tr><td class="rx-eq">${x.name} <span class="small muted">${x.kind === 'complex' ? 'complex' : 'new species'}</span></td>
        <td class="mono">${fmtC(x.max)} <${Tag} status=${x.status} /></td><td class="mono">${fmtC(x.final)}</td></tr>`)}</tbody></table>`}
    ${!d.new_steps.length && !held.length && html`<p class="small muted">Nothing that holds material.</p>`}
    ${quiet > 0 && html`<p class="small muted">${quiet} ${held.length ? 'other ' : ''}complex${quiet === 1 ? '' : 'es'} or new species hold${quiet === 1 ? 's' : ''} next to nothing (under 1e-9 M).</p>`}

    ${d.failed.length > 0 && html`<details class="small" open=${d.failed.length < 4}><summary>${d.failed.length} calculation${d.failed.length === 1 ? '' : 's'} gave no number</summary>
      <ul class="am-failed">${Object.entries(why).map(([w, labels]) => html`<li>${w}${labels.filter(Boolean).length ? html` <span class="muted">(${labels.filter(Boolean).slice(0, 4).map((l) => stepLabel(l)).join('; ')}${labels.length > 4 ? `; ${labels.length - 4} more` : ''})</span>` : ''}</li>`)}</ul></details>`}
    ${(d.notes || []).map((w) => html`<p class="small muted">⚠ ${w}</p>`)}
  </section>`;
}

// ------------------------------------------------------------ candidates side by side
// One row per step, one column per candidate: the change of its apparent barrier, blue lower, red higher.
function CompareSteps({ rows }) {
  const steps = {};
  rows.forEach((r) => (r.steps || []).forEach((s) => { (steps[s.step] ||= { ...s, by: {} }).by[r.file] = s; }));
  const list = Object.values(steps).sort((a, b) => Math.min(...Object.values(a.by).map((x) => x.change)) - Math.min(...Object.values(b.by).map((x) => x.change))).slice(0, 12);
  const max = Math.max(1, ...list.flatMap((s) => Object.values(s.by).map((x) => Math.abs(x.change))));
  if (!list.length) return html`<p class="small muted">No step goes through any of these candidates.</p>`;
  return html`<div class="rx-table-wrap"><table class="rx-table am-compare">
    <thead><tr><th>Step · change of its barrier (kcal/mol)</th>${rows.map((r) => html`<th title=${r.name}>${r.name}</th>`)}</tr></thead>
    <tbody>${list.map((s) => html`<tr><td class="rx-eq">${stepLabel(s.label, s.direction)}</td>
      ${rows.map((r) => {
        const x = s.by[r.file];
        if (!x) return html`<td class="muted am-heat" title="Its fastest version does not go through this candidate">—</td>`;
        const w = Math.round((Math.min(1, Math.abs(x.change) / max)) * 70);
        return html`<td class="am-heat mono" style=${{ background: `color-mix(in srgb, var(${x.change < 0 ? '--div-pos' : '--div-neg'}) ${w}%, transparent)` }}
          title=${`${r.name}: ${kcal(x.change, true)} kcal/mol${x.spread != null ? ` ± ${x.spread.toFixed(1)}` : ', spread not given'}, ${x.status === 'verified' ? 'verified' : 'an estimate'}`}>
          ${kcal(x.change, true)}${x.status !== 'verified' ? html`<span class="am-heat-est"> est.</span>` : ''}</td>`;
      })}</tr>`)}</tbody></table></div>`;
}

// ------------------------------------------------------------ the page
export function AddMoleculeView() {
  const saved = useStore((s) => s.addmol) || {};
  const jobKey = useStore((s) => Object.values(s.jobs).filter((j) => j.op === 'addmol-test').map((j) => `${j.id}:${j.status}`).join());
  const [q, setQ] = useState(null);
  const [queries, setQueries] = useState([]);
  const [busy, setBusy] = useState(false);
  const [sel, setSel] = useState(saved.sel || null);
  const [picked, setPicked] = useState(new Set());
  const [tick, setTick] = useState(0);
  const qid = q?.id || saved.qid || null;
  const refresh = async (id = qid) => {
    if (!id) return;
    try { setQ(await api.get(`/api/addmol/${id}`)); setTick((n) => n + 1); } catch { /* the next poll tries again */ }
  };
  const list = () => api.get('/api/addmol').then((out) => setQueries(out.queries)).catch(() => {});
  useEffect(() => { list(); if (saved.qid) refresh(saved.qid); }, []);
  useEffect(() => { if (qid) { refresh(); list(); } }, [jobKey]);
  const live = (q?.results || []).some((r) => ['running', 'queued'].includes(r.status));
  // Result files are rewritten as layers finish: read them again every few seconds while a calculation runs.
  useEffect(() => {
    if (!live) return undefined;
    const timer = setInterval(() => refresh(), 3000);
    return () => clearInterval(timer);
  }, [live, qid]);
  useEffect(() => { set({ addmol: { ...(state.addmol || {}), qid, sel } }); }, [qid, sel]);
  const ask = async (body) => {
    setBusy(true);
    const out = await attempt(() => api.post('/api/addmol/query', body));
    setBusy(false);
    if (out) { setQ(out); setSel(null); setPicked(new Set()); list(); }
  };
  const open = async (id) => { setQ(null); setSel(null); setPicked(new Set()); await refresh(id); };
  const rows = q?.results || [];
  const shown = sel && rows.some((r) => r.file === sel) ? sel : null;
  const toggle = (f) => { const p = new Set(picked); if (p.has(f)) p.delete(f); else p.add(f); setPicked(p); };
  const compared = rows.filter((r) => picked.has(r.file) && r.readout);
  return html`<div class="kinetics addmol">
    <div class="kn-setup">
      <${Setup} q=${q} busy=${busy} onAsk=${ask} queries=${queries} onOpen=${open} />
      ${q && html`<${Candidates} q=${q} onStarted=${() => refresh()} />`}
    </div>
    <div class="kn-results">
      ${!q && html`<div class="empty-hint">
        <p><b>Would one more molecule in the flask give you more of a product?</b></p>
        <p>Pick the species you want more of and the conditions. You see at once what holds it back in this network.
          Then add candidates (a possible catalyst, an additive): each is placed beside every step, and the kinetics are run with it.
          Numbers arrive cheapest first, each marked as an estimate or verified.</p>
        ${busy && html`<p class="small muted"><span class="spinner" aria-hidden="true"></span>Reading the network…</p>`}
      </div>`}
      ${q && html`<${Limits} q=${q} />
        <section class="am-card">
          <div class="section-title">Candidates (${rows.length})<${Why} id="addmol.read" label="What each column says" /></div>
          ${rows.length ? html`<p class="small muted">Largest gain of the target first. Click one for its steps; tick several to compare them.</p>
            <${Table} q=${q} rows=${rows} sel=${shown} onPick=${(f) => setSel(f === shown ? null : f)} picked=${picked} onToggle=${toggle} />`
            : html`<p class="small muted">None yet. Add candidates on the left: a SMILES per line, or a file from a model. A first number comes within a minute.</p>`}
          ${compared.length > 1 && html`<div class="section-title">Side by side</div><${CompareSteps} rows=${compared} />`}
        </section>
        ${shown && html`<${Detail} key=${shown} q=${q} file=${shown} tick=${tick} onGone=${() => { setSel(null); refresh(); }} />`}`}
    </div>
  </div>`;
}

// From Explore: the page with this species as the target.
export function openAddMolecule(target) {
  set({ addmol: { ...(state.addmol || {}), target, qid: null, sel: null },
    analyze: { ...(state.analyze || {}), compose: false, view: 'addmol' }, view: { tab: 'analyze', jobId: null } });
}
