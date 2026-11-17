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
import { PHONE_QUERY, cls, depictUrl, isSpecies, readFileText } from '../util.js';
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
const LAYER = { given: 'given with the candidate', steer: 'steering level', bystander: 'quick estimate', verify: 'verification', relay: 'scan' };

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
  const [scan, setScan] = useState(25);
  const [side, setSide] = useState(false);
  const [read, setRead] = useState(null);
  const [busy, setBusy] = useState(false);
  const file = useRef(null);
  const body = (dry) => ({ text, filename, amount, scan, scan_side_reactions: side, dry_run: dry });
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
    toast(`${out.candidates} candidate${out.candidates === 1 ? '' : 's'}: one calculation started (quick estimates, then verification${scan ? ', then scans' : ''})`, 'ok');
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
    <div class="kn-row"><label class="small" title="Searches for transition states in which the molecule takes part (a relay, an exchange), one per reaction, in the order of what the target would gain; they end by themselves once no reaction is left that could lower the target's route. This is the most they may take. Minutes each. 0: none.">Scans: at most
      <input type="number" class="tiny" min="0" max="200" step="1" value=${scan} onInput=${(e) => setScan(Math.max(0, Math.round(+e.target.value)))} /> reactions</label><${Why} id="addmol.scan" label="The scan" /></div>
    <label class="kn-row small" title="Reactions that do not lower the target's route but could divert material if the molecule opened them: listed, not scanned, unless this is ticked"><input type="checkbox" checked=${side} onChange=${(e) => setSide(e.target.checked)} /> also scan side reactions</label>
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
  return { main: r.settled ? 'no change' : r.progress?.open ? 'none found yet' : 'still none', kind: '', short: Math.abs(drop) > 0.05 ? `route ${kcal(r.peak.parent)} → ${kcal(r.peak.with)}` : 'route unchanged',
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
  if (r.fate === 'regenerated') return { main: r.route_role === 'takes part' ? 'returned, takes part' : 'returned', fact: short, sub: r.in_words || `${turns}${on >= 0.001 ? ` · on the route of ${pct(on)} of the target` : ''}` };
  if (on >= 0.001) return { main: 'on the route', fact: short, sub: r.in_words || `of ${pct(on)} of the target · ${turns}` };
  return { main: 'takes no part', fact: '', sub: 'no step with it carries material' };
}
const restText = (r) => (r.resting_state ? { main: r.resting_state.id === 'added' ? 'free' : r.resting_state.name, sub: pct(r.resting_state.share) } : null);
const stateOf = (r) => (r.used?.est ? 'est' : r.used?.verified ? 'verified' : null);

function Progress({ r }) {
  if (r.status === 'queued') return html`<span class="am-run">queued</span>`;
  const live = r.progress?.stages || [];
  if (r.status === 'running') {
    return html`<span class="am-run">${r.settled ? html`<b class="am-settled-word" title="No reaction is left that could lower the target's route">settled</b> · ` : ''}<span class="spinner" aria-hidden="true"></span>${live.length ? live.map((l) => `${LAYER[l.name] || l.name} ${l.done ?? 0}/${l.total ?? '?'}`).join(' · ') : r.readout ? 'running' : 'starting'}</span>`;
  }
  if (r.status === 'cut') return html`<span class="am-run warn" title="Stopped from outside before it was complete: the numbers are from what had arrived">cut short</span>`;
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
            <div class="small muted"><${Tag} status=${b.status} />${within ? html` <span class="am-flag">within spread</span>` : ''}${b.limits ? '' : html` <span class="am-flag" title="No step that limits the target goes through it">off-route</span>`}${
              b.via_reaction ? html` <span class="am-flag" title="Its number is the scan's of the whole reaction, found from another of its steps: a step that carries a molecule along gets the reaction's lower one, which is no effect of the added molecule">·r</span>` : ''}</div></td>`
            : html`<td class="muted small">none through it</td>`}`
          : html`<td colspan="5" class=${cls('small', r.status === 'failed' || r.why ? 'am-warn' : 'muted')} title=${none}><div class="am-two">${none}</div></td>`}
      </tr>`;
    })}</tbody></table></div>`;
}

// ------------------------------------------------------------ one candidate
// Barrier before and after on one axis: hollow without the molecule, filled with it
// (blue lower, red higher), the whisker its spread.
function Dumbbell({ before, after, spread, lo, hi, est = false }) {
  const X = (v) => `${Math.min(100, Math.max(0, ((v - lo) / Math.max(1e-9, hi - lo)) * 100))}%`;
  const a = Math.min(before, after), b = Math.max(before, after);
  const kind = after < before ? 'pos' : 'neg';
  return html`<span class="am-db">
    ${spread != null && spread > 0 && html`<span class="am-db-spread" style=${{ left: X(after - spread), right: `calc(100% - ${X(after + spread)})` }}></span>`}
    <span class=${`am-db-bar ${kind} ${est ? 'est' : ''}`} style=${{ left: X(a), right: `calc(100% - ${X(b)})` }}></span>
    <span class="am-db-dot before" style=${{ left: X(before) }}></span>
    <span class=${`am-db-dot ${kind} ${est ? 'est' : ''}`} style=${{ left: X(after) }}></span>
  </span>`;
}

// What kind of number a row is: a quick estimate (dashed), a verified transition state with the molecule
// beside the step, or one with the molecule taking part.
const KIND = {
  estimate: ['estimate', 'A quick estimate: the molecule placed beside the step\'s frozen frames, no transition state optimized with it.'],
  beside: ['beside', 'Verified: the transition state optimized with the molecule beside the step, its IRC joining the step\'s ends.'],
  'takes part': ['takes part', 'Verified: a transition state in which the molecule\'s own bonds change (a relay, an exchange), its IRC checked.'],
};
// What the result file says about its quick estimates (the quick-estimate stage's `uncertainty`), in one sentence.
const uncWords = (u) => {
  const b = u?.barrier_shift_kcal;
  if (!b) return 'The result file gives no measured uncertainty for its estimates.';
  return `Measured uncertainty of such an estimate: ${b.mean_abs_difference_to_verified} kcal/mol from the verified value on average, ${b.largest} at most, the same sign on ${b.same_sign}${u.what ? ` (${u.what})` : ''}.${u.quick_setting ? ` ${u.quick_setting[0].toUpperCase()}${u.quick_setting.slice(1)}.` : ''}`;
};

function StepRows({ rows, onView, viewing, through, fresh, unc }) {
  const mean = unc?.barrier_shift_kcal?.mean_abs_difference_to_verified ?? null;   // kcal/mol, from the file
  const sp = (s) => (s.spread != null ? s.spread : s.kind === 'estimate' ? mean || 0 : 0);
  const vals = rows.flatMap((s) => [s.before, s.after, s.after - sp(s), s.after + sp(s)]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max(1, (hi - lo) * 0.06);
  return html`<div class="rx-table-wrap"><table class="rx-table am-steps">
    <thead><tr><th>Step</th><th title="Apparent barrier: the transition state above its reactants at the amounts they have in the run">without</th>
      <th class="am-db-head" title="kcal/mol"><span>${kcal(lo - pad)}</span><span>${kcal(hi + pad)}</span></th><th>with</th><th>change</th>
      ${through && html`<th title="What the number is: a quick estimate, or a verified transition state with the molecule beside the step or taking part">Kind</th><th></th>`}</tr></thead>
    <tbody>${rows.map((s) => {
      const est = s.kind === 'estimate';
      const spread = sp(s);
      const within = through && spread > 0 && spread >= Math.abs(s.change);
      const [word, what] = KIND[s.kind] || [s.kind || '', ''];
      const label = `${stepLabel(s.label, s.direction)}${s.step.endsWith('~mirror') ? ' (mirror image)' : ''}`;
      return html`<tr key=${s.step} class=${cls(viewing === s.step && 'on', est && 'am-est-row', fresh?.has(s.step) && 'am-fresh')}>
        <td class="am-step" title=${`${label}${s.via ? `\n${s.via}` : ''}${s.rank != null ? '\nOne of the steps that limit the target.' : ''}${
          s.via_label ? `\nIts number is the scan's of the whole reaction, found from the step ${stepLabel(s.via_label)}: the lowest transition state between the same molecules.` : ''}${
          s.fixed_together?.length ? `\n${s.fixed_together.join(', ')}: free energy fixed only together with other species (the boxes never show it alone), so this barrier is not reliable.` : ''}`}>${
          s.rank != null ? html`<i class="am-limit" aria-label="limits the target"></i>` : ''}${s.fixed_together?.length ? html`<span class="am-flag" aria-label="rests on a shared free energy">shared G</span> ` : ''}${label}</td>
        <td class="mono">${kcal(s.before)}</td>
        <td class="am-db-cell" title=${`${kcal(s.before)} → ${kcal(s.after)} kcal/mol${spread ? `, ± ${spread.toFixed(1)}` : ''}`}>
          <${Dumbbell} before=${s.before} after=${s.after} spread=${spread || null} lo=${lo - pad} hi=${hi + pad} est=${est} /></td>
        <td class="mono">${kcal(s.after)}</td>
        <td class="mono"><b class=${within ? 'am-weak' : ''}>${kcal(s.change, true)}</b>${through ? html` <span class=${within ? 'am-warn' : 'muted'}
          title=${s.spread != null ? (est ? 'Spread over the estimate\'s repeats' : `Difference between the ${s.poses_verified || 2} verified poses`)
            : est ? `No spread of its own (one repeat). ${uncWords(unc)}` : 'One verified pose: no spread'}>${
          s.spread != null ? `± ${s.spread.toFixed(1)}` : est ? (mean != null ? `± ~${mean.toFixed(0)}` : '± ?') : ''}</span>` : ''}</td>
        ${through && html`<td class="small" title=${`${what}${s.via_label ? ' From the scan of the whole reaction (another step between the same molecules).' : ''}${est ? ` ${uncWords(unc)}` : ''}${s.t_s != null ? `\nArrived ${fmtTime(s.t_s)} into the test.` : ''}${s.level ? `\nLevel: ${s.level}` : ''}${s.source ? `\n${s.source}` : ''}`}><${Tag} status=${s.status} /> <span class="muted">${word}${s.via_label ? ' ·r' : ''}</span></td>
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
  }, [qid, file, step.step, step.through]);
  const words = g?.frames.length === 3 ? (g.ends === 'irc' ? ['one end', 'TS', 'other end'] : ['reactant end', 'TS', 'product end']) : ['TS'];
  return html`<div class="am-viewer">
    <div class="kn-row"><b>${stepLabel(step.label, step.direction)}</b><span class="spacer"></span>
      ${g && g.frames.length > 1 && html`<div class="segmented mini">${words.map((w, k) => html`<button class=${frame === k ? 'on' : ''} onClick=${() => setFrame(k)}>${w}</button>`)}</div>`}
      <button class="btn-icon" aria-label="Close" onClick=${onClose}>✕</button></div>
    ${g ? html`<${Viewer3D} frames=${g.frames} frame=${frame} height=${300} />
      ${g.missing ? html`<p class="warn-box small">This is the network's own transition state, without the molecule. The one with the molecule was written to
          <code class="am-path">${g.missing}</code>, which is not there any more (the calculation ran elsewhere and its files were not kept).</p>`
        : html`<p class="small muted">${g.with_molecule ? (g.ends === 'irc' ? 'The transition state optimized with the molecule, and the two ends of its IRC.' : 'With the molecule where this estimate left it.')
          : 'The network\'s own transition state, without the molecule: this estimate came with no geometry.'}</p>`}`
      : html`<p class="small muted">Loading the geometry…</p>`}
  </div>`;
}

// ------------------------------------------------------------ a test as it runs
const STAGES = [['bystander', 'Quick estimate', 'bystander', 'The molecule placed beside every step\'s two ends and transition state, the step\'s atoms held: seconds per step.'],
  ['verify', 'Verification', 'verified', 'Each step\'s transition state optimized with the molecule beside it, its IRC checked: replaces the step\'s estimate.'],
  ['relay', 'Scan', 'relay', 'A search for transition states in which the molecule takes part (relays, exchanges), on the steps that limit the target first: minutes per scan.']];

// Three stages in the order they run, each a fixed box: what it is doing, how far it is, when its first number came.
function Stages({ d }) {
  const by = Object.fromEntries((d.layers || []).map((l) => [l.name, l]));
  const sum = d.progress?.summary;
  const first = d.progress?.first || {};
  if (!STAGES.some(([k]) => by[k])) return null;
  const parts = sum ? [[sum.verified_beside, 'verified with it beside'], [sum.takes_part, 'with it taking part'], [sum.est, 'an estimate only'],
    [sum.leads_elsewhere, 'led to other species when verified'], [(sum.failed || 0), 'gave no number'], [sum.not_reached, 'not reached']].filter(([n]) => n > 0) : [];
  const word = { queued: 'queued', running: 'running', done: 'done', skipped: 'not asked', cut: 'cut short', 'stopped early': 'stopped early', failed: 'failed' };
  // Each box reads its own stage: they may run one after another or all at once.
  return html`<div class="am-stages-wrap">
    <div class="am-stages">${STAGES.map(([key, title, firstKey, what]) => {
      const l = by[key];
      const live = d.status === 'running';
      const status = l ? word[l.status] || l.status : live ? 'queued' : 'not asked';
      const frac = l?.total && l.done != null ? Math.min(1, l.done / l.total) : l?.status === 'done' ? 1 : 0;
      const unit = key === 'relay' ? 'reactions' : 'steps';
      const count = !l ? (live ? 'not started' : 'not part of this test')
        : l.done != null && l.total ? `${l.done} of ${l.total} ${unit}` : l.seen != null && l.status !== 'queued' ? `${l.seen} rows by then` : l.total ? `${l.total} ${unit} to do` : '—';
      return html`<div class=${cls('am-stage', l?.status === 'running' && 'running', l?.status === 'done' && 'done', ((!l && !live) || l?.status === 'skipped') && 'off',
        ['stopped early', 'cut'].includes(l?.status) && 'cut')} title=${`${what}${l?.what ? `\n${l.what}` : ''}`}>
        <div class="am-stage-head"><b>${title}</b>
          <span class="am-stage-status">${l?.status === 'running' ? html`<span class="spinner" aria-hidden="true"></span>` : ''}${status}</span></div>
        <div class=${cls('am-stage-bar', l?.status === 'running' && l.done == null && 'open')}><i style=${{ width: `${frac * 100}%` }}></i></div>
        <div class="small muted am-stage-foot">${count}${key === 'relay' && l?.lanes > 1 ? ` · ${l.lanes} lanes` : ''}${
          first[firstKey] != null ? ` · first after ${fmtTime(first[firstKey])}` : ''}</div>
      </div>`;
    })}</div>
    ${sum && html`<p class="small muted am-sum">How the ${sum.steps} steps stand: ${parts.map(([n, w]) => `${n} ${w}`).join(' · ') || 'none has a number yet'}.</p>`}
  </div>`;
}

// The route's highest point with the molecule, over the time of the test: the one curve to watch. A step line
// (the peak holds until a lower one arrives), the peak without the molecule and what the conditions allow as
// reference lines, a marker where the replay stands; hover reads a time.
function RoutePeak({ peak, cutoff, end, now, onPick, settled }) {
  const [hover, setHover] = useState(null);
  const pts = (peak?.with || []).filter((x) => x.peak != null);
  if (peak?.parent == null || !pts.length) return null;
  // (drawn at the size it is shown: a phone gets its own, narrower drawing, not a shrunken one)
  const small = useMemo(() => !!window.matchMedia?.(PHONE_QUERY).matches, []);
  const W = small ? 340 : 640, H = small ? 190 : 170, pad = { l: 34, r: small ? 82 : 96, t: 12, b: 28 };
  const tMax = Math.max(end || 0, pts[pts.length - 1].t_s, 1);
  const lo = Math.min(cutoff ?? Infinity, ...pts.map((x) => x.peak), peak.parent), hi = Math.max(peak.parent, ...pts.map((x) => x.peak));
  const y0 = Math.floor((lo - 3) / 5) * 5, y1 = Math.ceil((hi + 2) / 5) * 5;
  const X = (t) => pad.l + (Math.min(t, tMax) / tMax) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const upto = now ?? tMax;
  const at = (t) => { let v = peak.parent; for (const x of pts) if (x.t_s <= t) v = x.peak; return v; };
  let path = `M${X(0)},${Y(peak.parent)}`;
  let last = peak.parent;
  for (const x of pts) { if (x.t_s > upto) break; path += ` H${X(x.t_s)} V${Y(x.peak)}`; last = x.peak; }
  path += ` H${X(upto)}`;
  const unit = tMax >= 7200 ? 3600 : tMax >= 300 ? 60 : 1, uw = unit === 3600 ? 'h' : unit === 60 ? 'min' : 's';
  const stepT = [1, 2, 5, 10, 15, 20, 30, 60, 120].map((k) => k * unit).find((k) => tMax / k <= (small ? 4 : 7)) || tMax;
  const ticksX = []; for (let t = 0; t <= tMax + 1e-6; t += stepT) ticksX.push(t);
  const ticksY = []; for (let v = y0; v <= y1 + 1e-6; v += (y1 - y0) > 40 ? 10 : 5) ticksY.push(v);
  const drops = pts.filter((x, k) => x.t_s <= upto && (k === 0 ? x.peak < peak.parent - 0.05 : x.peak < pts[k - 1].peak - 0.05));
  const move = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const t = (((e.clientX - box.left) / box.width) * W - pad.l) / (W - pad.l - pad.r) * tMax;
    setHover(Math.max(0, Math.min(tMax, t)));
  };
  return html`<div class="am-peak">
    <svg viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${`Highest point of the easiest route to the target over the time of the test: ${kcal(peak.parent)} kcal/mol without the molecule, ${kcal(last)} with it`}
      onMouseMove=${move} onMouseLeave=${() => setHover(null)} onClick=${() => hover != null && onPick && onPick(hover)} style=${{ cursor: onPick ? 'pointer' : 'default' }}>
      ${ticksY.map((v) => html`<line class="kn-grid" x1=${pad.l} x2=${W - pad.r} y1=${Y(v)} y2=${Y(v)} /><text class="kn-tick" x=${pad.l - 6} y=${Y(v) + 3.5} text-anchor="end">${v}</text>`)}
      ${ticksX.map((t) => html`<text class="kn-tick" x=${X(t)} y=${H - pad.b + 14} text-anchor="middle">${+(t / unit).toFixed(1)}</text>`)}
      <text class="kn-axis" x=${W - pad.r} y=${H - 2} text-anchor="end">time into the test (${uw})</text>
      <text class="kn-axis" transform=${`translate(11 ${pad.t + (H - pad.t - pad.b) / 2}) rotate(-90)`} text-anchor="middle">kcal/mol</text>
      <line class="am-peak-ref" x1=${pad.l} x2=${W - pad.r} y1=${Y(peak.parent)} y2=${Y(peak.parent)} />
      <text class="am-peak-lab" x=${W - pad.r + 6} y=${Y(peak.parent) + 4}>without ${kcal(peak.parent)}</text>
      ${cutoff != null && cutoff >= y0 && html`<line class="am-peak-cut" x1=${pad.l} x2=${W - pad.r} y1=${Y(cutoff)} y2=${Y(cutoff)} />
        <text class="am-peak-lab" x=${W - pad.r + 6} y=${Y(cutoff) + 4}>allowed ${kcal(cutoff)}</text>`}
      <path class="am-peak-line" d=${path} />
      ${drops.map((x) => html`<circle class="am-peak-dot" cx=${X(x.t_s)} cy=${Y(x.peak)} r="4.5"><title>${`${kcal(x.peak)} kcal/mol from ${fmtTime(x.t_s)}`}</title></circle>`)}
      ${Math.abs(Y(last) - Y(peak.parent)) > 12 && (cutoff == null || Math.abs(Y(last) - Y(cutoff)) > 12) && html`<text class="am-peak-lab strong" x=${W - pad.r + 6} y=${Y(last) + 4}>with ${kcal(last)}</text>`}
      ${settled && settled.t_s <= upto && html`<line class="am-peak-settled" x1=${X(settled.t_s)} x2=${X(settled.t_s)} y1=${pad.t} y2=${H - pad.b} />
        <text class="am-peak-lab strong" x=${X(settled.t_s) + (X(settled.t_s) > W - pad.r - 90 ? -4 : 4)} y=${H - pad.b - 6}
          text-anchor=${X(settled.t_s) > W - pad.r - 90 ? 'end' : 'start'}>settled ${fmtTime(settled.t_s)}</text>`}
      ${now != null && now < tMax && html`<line class="am-peak-now" x1=${X(now)} x2=${X(now)} y1=${pad.t} y2=${H - pad.b} />`}
      ${hover != null && html`<line class="kn-cross" x1=${X(hover)} x2=${X(hover)} y1=${pad.t} y2=${H - pad.b} />`}
    </svg>
    <div class="am-peak-tip small">${hover != null ? html`<span class="mono">${fmtTime(hover)}</span> into the test: route peak with the molecule <span class="mono">${hover <= upto ? `${kcal(at(hover))} kcal/mol` : 'not replayed yet'}</span>${onPick ? ' · click to see the page as it stood then' : ''}`
      : html`<span class="muted">Highest point of the easiest route to the target, with the molecule, as its rows arrived (kcal/mol, free energy). Lower is faster; under the "allowed" line the target forms.</span>`}</div>
  </div>`;
}

// A finished test replayed by its rows' arrival times: a small bar under the candidate's name.
function Replay({ d, at, setAt }) {
  const [playing, setPlaying] = useState(false);
  const total = d.duration_s;
  const cur = at ?? total;
  useEffect(() => {
    if (!playing) return undefined;
    if (cur >= total) { setPlaying(false); return undefined; }
    if (at != null && d.replay?.t_s !== at) return undefined;      // (the next frame once this one is on the page)
    const next = setTimeout(() => setAt(Math.min(total, cur + total / 24) >= total ? null : Math.min(total, cur + total / 24)), 1200);
    return () => clearTimeout(next);
  }, [playing, d.replay?.t_s, at]);
  if (!total || d.status === 'running' && !d.replay) return null;
  return html`<div class="am-replay" role="group" aria-label="Replay of the test">
    <button class="btn small" title=${playing ? 'Pause' : 'Play the test again as its numbers arrived'} onClick=${() => { if (!playing && cur >= total) setAt(0); setPlaying(!playing); }}>${playing ? '❚❚' : '▶'} Replay</button>
    <input type="range" min="0" max=${Math.ceil(total)} step=${Math.max(1, Math.round(total / 200))} value=${Math.round(cur)} aria-label="Time into the test"
      onChange=${(e) => { setPlaying(false); setAt(+e.target.value >= Math.ceil(total) ? null : +e.target.value); }} />
    <span class="mono small">${fmtTime(cur)}</span><span class="small muted">of ${fmtTime(total)}</span>
    ${at != null && html`<button class="btn-link small" onClick=${() => { setPlaying(false); setAt(null); }}>as it ended</button>`}
  </div>`;
}

// The answer first: does this molecule help make the target? Never "no effect" while something was not looked at.
function verdict(d, q) {
  const t = d.target, peak = d.peak;
  const drop = peak.parent != null && peak.with != null ? peak.parent - peak.with : 0;
  const over = peak.with != null ? peak.with - q.limits.cutoff_kcal : null;
  if (t.factor != null && t.factor >= 1.05) return { kind: 'pos', text: `It helps: ${fmtX(t.factor)}× more of the target.` };
  if (t.factor != null && t.factor <= 0.95) return { kind: 'neg', text: `It costs target: ${fmtX(1 / t.factor)}× less.` };
  if (t.with.resolved && !t.parent.resolved) return { kind: 'pos', text: `It helps: the target forms (${fmtC(t.with.final)} M), where none does without it.` };
  if (drop >= 0.5) return { kind: '', text: `It lowers the easiest route by ${kcal(drop)} kcal/mol${over > 0 ? `, which is still ${kcal(over)} above what these conditions allow: no target forms` : ''}.` };
  if (d.settled) return { kind: '', text: 'No effect on the target\'s route: it is settled where it is without the molecule.' };
  return { kind: '', text: d.progress?.open ? 'No effect found so far.' : 'No effect at these conditions.' };
}

function Answer({ d, q }) {
  const v = verdict(d, q), s = stateOf(d), t = d.target, peak = d.peak, p = d.progress || {};
  const fate = fateText(d), sp = t.spread, fo = d.first_order_detail;
  const open = [];
  const live = (p.stages || []).map((l) => (STAGES.find(([k]) => k === l.name) || [0, l.name])[1].toLowerCase());
  if (d.status === 'running' && live.length) open.push(`the ${live.join(' and the ')} ${live.length > 1 ? 'are' : 'is'} still running`);
  if (d.status === 'running' && !live.length) open.push('the test is still running');
  if (['interrupted', 'cut'].includes(d.status)) open.push(`the calculation was stopped before it finished${d.cut?.why ? ` (${d.cut.why})` : ''}`);
  if (p.summary?.est) open.push(`${p.summary.est} step${p.summary.est === 1 ? ' rests' : 's rest'} on a quick estimate only`);
  // Reactions left unscanned because they could not lower the target's route are no reason to wait; those that could, are.
  const settled = d.settled;
  const sideN = (d.side_reactions?.listed || []).length;
  const left = settled ? 0 : (d.scan_left || []).reduce((a, x) => a + x.n, 0);
  if (left) open.push(`${left} reaction${left === 1 ? ' was' : 's were'} not scanned for the molecule taking part`);
  else if (d.status !== 'running' && !(d.scans || []).length && (d.layers || []).some((l) => l.name === 'bystander')) open.push('no step was scanned for the molecule taking part');
  const peakState = d.peak_rests_on || (peak.with !== peak.parent ? s : 'verified');
  const above = d.takes_part_above || [];
  return html`<div class=${cls('am-answer', v.kind, settled && 'settled')}>
    <p class="am-verdict"><b>${v.text}</b> <${Tag} status=${peakState} title=${peakState === 'verified' ? 'The route this rests on is made of verified steps only' : 'The route this rests on has steps that are estimates only'} /></p>
    ${settled && html`<p class="am-settled"><b>Settled after ${fmtTime(settled.t_s)}:</b> no reaction is left that could lower the target's route (every one that could, with no barrier, has been scanned).
      ${d.status === 'running' ? html` The quick estimates and the verification are still running: a verified row can still move the route's peak.` : ''}</p>`}
    ${open.length > 0 && html`<p class="am-open"><b>${settled ? (d.status === 'running' ? 'Still running:' : 'Left open:') : v.kind === 'pos' ? 'Not the whole picture yet:' : 'Not a verdict yet:'}</b> ${open.join('; ')}.${left ? ' A reaction that was not scanned may still have a faster way through the molecule.' : ''}</p>`}
    <p class="am-scope small">This is what this network gives at ${d.scope?.level || d.level_words || 'its level of theory'}${d.scope?.phase ? ` (${d.scope.phase})` : d.solvent ? '' : ', in the gas phase'}: not a prediction of an experiment.${' '}
      ${d.scope?.does_not_hold ? `${d.scope.does_not_hold[0].toUpperCase()}${d.scope.does_not_hold.slice(1)}.` : 'A route the network does not hold (a species nobody explored, an intermediate the molecule forms) is not in it.'}</p>
    ${above.length > 0 && html`<p class="am-control small"><b>${above.length} scan${above.length === 1 ? '' : 's'} found transition states with the molecule taking part, all above the same reaction with it beside</b>${' '}
      (${above.slice(0, 4).map((x) => `${kcal(x.takes_part)} against ${kcal(x.beside)}`).join('; ')}${above.length > 4 ? '; …' : ''} kcal/mol): its taking part there is no route anything goes by.</p>`}
    <dl class="am-facts">
      <dt>Easiest route to the target</dt><dd><span class="mono">${kcal(peak.parent)} → <b>${kcal(peak.with)}</b> kcal/mol</span> at its highest point <${Tag} status=${peakState} />
        <span class="muted"> · ${kcal(q.limits.cutoff_kcal)} is what these conditions allow${d.route_peak_with_added === false && peak.with !== peak.parent ? ' · the route\'s highest step is not one with the molecule' : ''}</span></dd>
      <dt>Target at the end</dt><dd><span class="mono">${t.parent.resolved ? fmtC(t.parent.final) : 'none'} → <b>${t.with.resolved ? `${fmtC(t.with.final)} M` : 'none'}</b></span>
        ${t.factor != null && html` <b class=${cls('am-main', effectText(d, q).kind)}>${effectText(d, q).main}</b>`} <${Tag} status=${s} />
        ${sp && html`<div class="small muted" title=${sp.what || ''}>Between <span class="mono">${fmtC(sp.with_low)}</span> and <span class="mono">${fmtC(sp.with_high)} M</span>${
          sp.factor_low != null ? ` (${fmtX(sp.factor_low)}× to ${fmtX(sp.factor_high)}×)` : ''} if every number with a spread is off by its spread, all one way.${
          sp.with_high != null && !t.with.resolved && sp.with_high > 1e-9 * q.conditions.initial.reduce((a, m) => a + m.amount, 0) ? ' The upper end would be a resolved amount.' : ''}</div>`}
        ${(!t.with.resolved || !t.parent.resolved) && html`<div class="small muted">None: below what the kinetics resolve, 1e-9 of the starting material (${!t.parent.resolved ? `without ${fmtC(t.parent.final)} M` : ''}${!t.parent.resolved && !t.with.resolved ? ', ' : ''}${!t.with.resolved ? `with ${fmtC(t.with.final)} M` : ''}).</div>`}
        ${fo?.factor != null && html`<div class="small muted">First-order estimate from the degrees of rate control: ${fmtX(fo.factor)}× (the full kinetics: ${t.factor != null ? `${fmtX(t.factor)}×` : '—'})<${Why} id="addmol.first_order" label="The first-order estimate" /></div>`}</dd>
      ${soonerText(d) && html`<dt>As much as today's half</dt><dd><span class="mono">${soonerText(d).main}</span> <span class="muted">(today: ${fmtTime(t.parent.time_to_fraction_s)})</span> <${Tag} status=${s} /></dd>`}
      <dt>The molecule itself<${Why} id="addmol.fate" label="Returned, used up or held" /></dt><dd><b>${fate.main}</b> <${Tag} status=${s} /> <span class="muted">${fate.sub}</span>
        <div class="small muted">${d.shares ? `At the end: free ${pct(d.shares.free)} · bound in complexes ${pct(d.shares.bound)} · in new species ${pct(d.shares.covalent)}. ` : ''}Turnovers per molecule in ${fmtTime(q.conditions.time_s)}: <span class="mono">${d.returns != null ? fmtC(d.returns) : '—'}</span>.</div></dd>
      ${sideN > 0 && html`<dt>Side reactions</dt><dd><details class="am-side"><summary><b>${sideN}</b> could divert material if the molecule opened them; ${d.side_reactions.asked_for ? 'scanned' : 'not scanned'}</summary>
        <ul>${d.side_reactions.listed.map((x) => html`<li title=${x.in_words || ''}>${stepLabel(x.label || x.reaction)}${x.steps?.length > 1 ? html` <span class="muted">· ${x.steps.length} steps</span>` : ''}
          <span class="muted"> · with no barrier, ${x.species_name} would change by <span class="mono">${x.change > 0 ? '+' : ''}${fmtC(x.change)} M</span></span></li>`)}</ul>
        ${!d.side_reactions.asked_for && html`<p class="small muted">They do not lower the target's route, so they were not scanned. To scan them, tick <i>also scan side reactions</i> before Test (minutes each).</p>`}</details></dd>`}
      ${(left > 0 || (d.scans || []).length > 0) && html`<dt>Not scanned</dt><dd>${settled && d.not_scanned ? html`<span class="muted">${d.not_scanned}.</span>` : left ? html`<b>${left}${d.scan_reactions ? ` of ${d.scan_reactions}` : ''} reaction${left === 1 ? '' : 's'}</b> <span class="muted">got no scan for the molecule taking part:</span>
          <ul class="am-left">${(d.scan_left || []).map((x) => html`<li title=${x.labels.map((l) => stepLabel(l)).join('\n') + (x.n > x.labels.length ? `\n… ${x.n - x.labels.length} more` : '')}><span class="mono">${x.n}</span> ${x.why}</li>`)}</ul>`
          : d.status === 'running' ? html`<span class="muted">The scans are still running (${d.scans.length} started${d.scan_reactions ? ` of ${d.scan_reactions} reactions` : ''}): what is left unscanned is known when they end.</span>`
            : html`<span class="muted">Every reaction that was wanted for a scan was scanned.</span>`}
        ${d.not_scanned && !left && !settled && html`<div class="small muted">${d.not_scanned}.</div>`}</dd>`}
    </dl>
  </div>`;
}

// The scans for the molecule taking part: one row each (a scan may stand for several steps of one reaction).
function Scans({ d }) {
  if (!(d.scans || []).length) return null;
  const word = { queued: 'queued', mapping: 'mapping atoms', searching: 'searching', done: 'done', failed: 'failed' };
  const lanes = new Set(d.scans.map((x) => x.lane).filter((x) => x != null)).size > 1;
  return html`<div class="section-title">Scans for the molecule taking part (${d.scans.filter((x) => x.status === 'done').length} of ${d.scans.length} started${d.scan_reactions ? `; the network has ${d.scan_reactions} reactions` : ''})<${Why} id="addmol.scan" label="The scan" /></div>
    <div class="rx-table-wrap"><table class="rx-table am-scans">
      <thead><tr><th title="One scan per reaction: it stands for every step of the network between the same molecules">Reaction scanned</th>${lanes ? html`<th title="Scans run side by side: which of the lanes this one ran in">lane</th>` : ''}<th>State</th><th title="Path searches finished, of those set up">searches</th>
        <th title="Verified transition states in which the molecule takes part">found</th><th title="The lowest of them, above the separated reactants and the molecule (kcal/mol, free energy)">lowest</th>
        <th title="Verified transition states that join other species than the step's two ends">elsewhere</th><th>time</th></tr></thead>
      <tbody>${d.scans.map((x) => {
        const live = ['mapping', 'searching'].includes(x.status) && d.status === 'running';
        const cut = ['mapping', 'searching', 'queued'].includes(x.status) && d.status !== 'running';
        return html`<tr key=${x.steps.join()}>
          <td class="am-step" title=${`${x.labels.map((l) => stepLabel(l)).join('\n')}\nWhy this one: ${x.why || '—'}`}>${stepLabel(x.labels[0] || '')}${x.steps[0]?.endsWith('~mirror') ? ' (mirror image)' : ''}${x.labels.length > 1 ? html` <span class="muted">· ${x.labels.length} steps</span>` : ''}</td>
          ${lanes ? html`<td class="mono small">${x.lane != null ? x.lane + 1 : '—'}</td>` : ''}
          <td class=${cls('small', cut && 'am-warn')} title=${x.why_failed || ''}>${live ? html`<span class="spinner" aria-hidden="true"></span>` : ''}${cut ? `cut short while ${word[x.status] || x.status}` : word[x.status] || x.status}</td>
          <td class="mono">${x.searches ? `${x.searches.searched ?? 0}/${x.searches.total ?? '?'}` : '—'}</td>
          <td class="mono">${x.with_molecule ?? '—'}</td>
          <td class="mono">${x.lowest?.with_molecule != null ? html`${kcal(x.lowest.with_molecule)} <${Tag} status="verified" />` : '—'}</td>
          <td class="mono">${x.elsewhere ?? '—'}</td>
          <td class="mono small">${x.wall_s != null ? fmtTime(x.wall_s) : '—'}</td></tr>`;
      })}</tbody></table></div>
    ${d.scan_rule && html`<details class="small"><summary>How the order of the scans is chosen</summary><p class="small muted">${d.scan_rule}.</p></details>`}`;
}

function Detail({ q, file, tick, onGone }) {
  const [d, setD] = useState(null);
  const [view, setView] = useState(null);
  const [order, setOrder] = useState(null);     // 'work' | 'change'; null: by whether the test still runs
  const [all, setAll] = useState(false);
  const [at, setAt] = useState(null);           // seconds into the test (a replay); null: the file as it is
  const [busy, setBusy] = useState(false);
  const seen = useRef({ rows: null, at: {} });
  useEffect(() => { setD(null); setView(null); setAt(null); seen.current = { rows: null, at: {} }; }, [file]);
  useEffect(() => {
    let alive = true;
    setBusy(at != null);
    api.get(`/api/addmol/${q.id}/results/${encodeURIComponent(file)}${at != null ? `?t=${at}` : ''}`)
      .then((out) => { if (alive) { setD(out); setBusy(false); } }).catch(() => { if (alive) setBusy(false); });
    return () => { alive = false; };
  }, [q.id, file, at, at == null ? tick : 0]);
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
      <div class="am-layers">${(d.layers || []).filter((l) => !STAGES.some(([k]) => k === l.name) || l.status === 'failed').map((l) => html`<span class=${`am-layer ${l.status === 'stopped early' ? 'failed' : l.status}`} title=${`${l.what || ''}${l.level ? `\nlevel: ${l.level}` : ''}`}>
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
  const s = stateOf(d);
  const quiet = d.new_species.filter((x) => !(x.max > 1e-9)).length + (d.new_species_total - d.new_species.length);
  const held = d.new_species.filter((x) => x.max > 1e-9);
  const why = {};
  d.failed.forEach((f) => { (why[f.why || f.status || 'failed'] ||= []).push(f.label || f.layer); });
  // Rows that appeared or were replaced since the last look (a poll, or a step of the replay) are lit;
  // nothing is lit on first sight.
  const now = Object.fromEntries(d.through_steps.map((x) => [x.step, `${x.through}:${x.after.toFixed(1)}`]));
  const stamp = d.replay ? `r${d.replay.t_s}` : `p${tick}`;
  if (seen.current.stamp !== stamp) {
    seen.current.fresh = new Set(seen.current.rows ? Object.keys(now).filter((k) => seen.current.rows[k] !== now[k]) : []);
    seen.current.rows = now; seen.current.stamp = stamp;
  }
  const fresh = seen.current.fresh || new Set();
  const playing = !!d.replay && !d.replay.over;
  // While a test runs the rows keep one order (the order of work, then by step), so a row that is replaced
  // stays where it is; once it is over they are read by the size of the change.
  const mode = order || (d.status === 'running' || playing ? 'work' : 'change');
  const sorted = [...d.through_steps].sort(mode === 'work'
    ? (a, b) => (a.rank ?? 1e9) - (b.rank ?? 1e9) || (a.step < b.step ? -1 : 1)
    : (a, b) => a.change - b.change || (a.step < b.step ? -1 : 1));
  // By default the steps that limit the target (the query's order of work); where none of them goes through
  // the molecule, the twelve first of the order chosen.
  const limiting = sorted.filter((x) => x.rank != null);
  const main = limiting.length ? limiting : sorted.slice(0, 12);
  const shown = all ? sorted : main;
  const nEst = d.through_steps.filter((x) => x.kind === 'estimate').length;
  return html`<section class="am-card">${head}
    <${Replay} d=${d} at=${at} setAt=${setAt} />
    ${d.replay && !d.replay.over && html`<p class="level-note small">${busy ? html`<span class="spinner" aria-hidden="true"></span>` : ''}A replay: the page as it stood <b>${fmtTime(d.replay.t_s)}</b> into the test, from the rows that had arrived by then and the kinetics run again on them.
      A row that a lower one later replaced, and the estimate of a step that was later verified, are not in the final file and so not in the replay.</p>`}
    ${d.status === 'running' && !d.replay && html`<p class="level-note small">Still running: what follows is from the numbers that are in, and changes as more arrive.</p>`}
    ${d.status === 'cut' && html`<p class="warn-box small">Cut short${d.cut?.t_s != null ? ` ${fmtTime(d.cut.t_s)} into the test` : ''}${d.cut?.why ? `: ${d.cut.why}` : ''}. What follows is from the numbers that had arrived.</p>`}
    ${d.status === 'interrupted' && html`<p class="warn-box small">Its calculation stopped before the last stage finished: what follows is from the numbers that were in.</p>`}
    ${d.status === 'failed' && html`<p class="warn-box small">A stage failed: ${d.failed.filter((f) => !f.step).map((f) => f.why).join(' ')} What follows is from the stages that ran.</p>`}
    <${Answer} d=${d} q=${q} />
    ${d.route_peak?.with?.length > 0 && html`<div class="section-title">The route's highest point as the test went</div>
      <${RoutePeak} peak=${d.route_peak} cutoff=${q.limits.cutoff_kcal} end=${d.duration_s} settled=${d.settled} now=${d.replay && !d.replay.over ? d.replay.t_s : null}
        onPick=${d.duration_s && (d.status !== 'running' || d.replay) ? (t) => setAt(Math.round(t)) : null} />`}
    <${Stages} d=${d} />

    <div class="section-title">Steps that go through ${name} (${d.through_steps.length})<${Why} id="addmol.steps" label="Barrier before and after" /></div>
    ${d.through_steps.length ? html`
      <div class="kn-row am-steps-bar">
        <div class="segmented mini" role="group" aria-label="Order of the rows">
          <button class=${mode === 'work' ? 'on' : ''} title="The steps that limit the target first, then by step: rows keep their place while numbers arrive" onClick=${() => setOrder('work')}>order of work</button>
          <button class=${mode === 'change' ? 'on' : ''} title="The largest lowering first" onClick=${() => setOrder('change')}>largest change</button></div>
        ${main.length > 0 && main.length < sorted.length && html`<button class="btn-link small" onClick=${() => setAll(!all)}>${all ? (limiting.length ? `only the ${main.length} that limit the target` : 'the first 12') : `all ${sorted.length}`}</button>`}
        <span class="small muted">${all ? 'every step' : limiting.length ? 'the steps that limit the target' : 'the first 12'}${nEst ? ` · ${nEst} of ${d.through_steps.length} are quick estimates` : ''}</span></div>
      <p class="small muted am-key"><i class="am-db-dot before"></i>without <i class="am-db-dot pos"></i>with, verified <i class="am-db-dot pos est"></i>with, a quick estimate <i class="am-db-spread"></i><span title=${uncWords(d.uncertainty)}>its uncertainty</span>
        <i class="am-limit"></i>limits the target · blue lower, red higher · apparent barriers, ${q.energies === 'free' ? 'free energy' : 'electronic'}</p>
      <${StepRows} rows=${shown} through fresh=${fresh} unc=${d.uncertainty} onView=${(x) => setView(view?.step === x.step ? null : x)} viewing=${view?.step} />
      ${view && html`<${StepViewer} qid=${q.id} file=${file} step=${d.through_steps.find((x) => x.step === view.step) || view} onClose=${() => setView(null)} />`}`
      : html`<p class="small muted">${d.used.est + d.used.verified ? `None: every step of the network is still fastest without ${name}.` : 'No stage has given a number for a step yet.'}</p>`}
    ${shown.some((x) => x.fixed_together?.length) && html`<p class="small muted"><span class="am-flag">shared G</span> the step makes or uses a species whose free energy the network fixes only together with others (the boxes never show it alone): its barrier is not reliable.</p>`}
    ${shown.some((x) => x.via_label) && html`<p class="small muted"><span class="muted">·r</span> the number is the scan's of the whole reaction: the lowest transition state between the same molecules, found from another of its steps. A step that carries a molecule along (a shuttle) then gets the reaction's lower one, which is no effect of the added molecule.</p>`}
    ${d.moved_steps.length > 0 && html`<details class="small"><summary>${d.moved_steps.length} step${d.moved_steps.length > 1 ? 's' : ''} moved only because amounts moved</summary>
      <p class="small muted">Their own transition states are unchanged: their reactants hold more or less material with ${name} in the flask.</p>
      <${StepRows} rows=${d.moved_steps.slice(0, 30)} /></details>`}
    <p class="small muted">${d.unchanged} step${d.unchanged === 1 ? '' : 's'} of the network unchanged.</p>

    <${Scans} d=${d} />

    ${(d.new_steps.length > 0 || held.length > 0) && html`<div class="section-title">What is new</div>`}
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
    ${quiet > 0 && html`<p class="small muted">${quiet} ${held.length ? 'other ' : ''}complex${quiet === 1 ? '' : 'es'} or new species hold${quiet === 1 ? 's' : ''} next to nothing (under 1e-9 M).</p>`}

    ${(d.elsewhere || []).length > 0 && html`<details class="small am-elsewhere"><summary>${d.elsewhere.length} step${d.elsewhere.length === 1 ? '' : 's'} led to other species when verified with ${name}</summary>
      <p class="small muted">The transition state optimized with the molecule present did not join the step's own two ends. Where each of its poses led instead (the two ends of its IRC); the step keeps its estimate.</p>
      <table class="rx-table"><thead><tr><th>Step of the network</th><th>pose</th><th>its transition state joins</th></tr></thead>
        <tbody>${d.elsewhere.flatMap((x) => (x.poses.length ? x.poses : [{ pose: '—', ends: [] }]).map((p, k) => html`<tr>
          ${k === 0 ? html`<td class="rx-eq" rowspan=${Math.max(1, x.poses.length)}>${stepLabel(x.label)}${x.t_s != null ? html`<div class="small muted">${fmtTime(x.t_s)} into the test</div>` : ''}</td>` : ''}
          <td class="mono">${p.pose}</td>
          <td class="rx-eq">${p.ends.length === 2 ? html`${p.ends[0].join(' + ')} <span class="muted">⇌</span> ${p.ends[1].join(' + ')}${
            p.ends[0].slice().sort().join() === p.ends[1].slice().sort().join() ? html` <span class="am-flag" title="Both ends of the IRC are the same species: the molecules only moved against each other">the same species both ways</span>` : ''}` : html`<span class="muted">not recorded</span>`}</td></tr>`))}</tbody></table></details>`}
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
  // The steps that limit the target first (a far-off step can change by more and matter less), then the largest lowering.
  const low = (x) => Math.min(...Object.values(x.by).map((y) => y.change));
  const list = Object.values(steps).sort((a, b) => (b.limits ? 1 : 0) - (a.limits ? 1 : 0) || low(a) - low(b)).slice(0, 12);
  const max = Math.max(1, ...list.flatMap((s) => Object.values(s.by).map((x) => Math.abs(x.change))));
  if (!list.length) return html`<p class="small muted">No step goes through any of these candidates.</p>`;
  return html`<div class="rx-table-wrap"><table class="rx-table am-compare">
    <thead><tr><th>Step · change of its barrier (kcal/mol)</th>${rows.map((r) => html`<th title=${r.name}>${r.name}</th>`)}</tr></thead>
    <tbody>${list.map((s) => html`<tr><td class="rx-eq">${s.limits ? html`<i class="am-limit" title="One of the steps that limit the target"></i>` : ''}${stepLabel(s.label, s.direction)}</td>
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
