// "What can I run on what I selected?" -- lists every operation whose
// target fits the current selection, grouped by intent, each expandable
// into its parameter form. Batch runs fall out naturally: N edges selected
// => one job per edge.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, openTab, prefs, select, useStore, state } from '../store.js';
import { useMap } from '../quests.js';
import { QuestIcon } from '../icons.js';

// The Inspector's actions as tiles (same drawings as Your tools), grouped by
// scale -- the same three groups, in the same order, whatever is selected --
// with the surroundings (gas, solvent, explicit, protein) as a switch above
// them, since any scale can be modelled in any of them.
const SCALES = [
  ['molecule', 'Molecule', 'The structure itself'],
  ['reaction', 'Reaction', 'One step: its barrier and path'],
  ['network', 'Network', 'What it can become, or come from'],
];
// Operations (or families) by scale; anything not listed is at the molecule's.
const SCALE_OF = {
  ts: 'reaction', channels: 'reaction', 'channels-more': 'reaction', vri: 'reaction', substituents: 'reaction',
  network: 'network', retrosynthesis: 'network', 'network-splits': 'network', 'nanoreactor-more': 'network',
};
// The operations that build surroundings: on the switch, not tiles.
const SWITCH_OPS = new Set(['qmmm-build', 'qmmm-protein-sites', 'qmmm-reaction']);
const TILE = {
  optimize: ['optimize', 'Optimize'], tsopt: ['barrier', 'TS from a guess'], conformers: ['shapes', 'Conformers'],
  retrosynthesis: ['retro', 'Retrosynthesis'], nanoreactor: ['reactor', 'Nanoreactor'], 'network-splits': ['network', 'Paths between isomers'],
  ts: ['connect', 'Transition state'], channels: ['channels', 'Reaction channels'], vri: ['bifurcation', 'Valley-ridge inflection'],
  'qmmm-build': ['explicit', 'Explicit solvent'], 'qmmm-reaction': ['solvent', 'Reaction in solvent'],
  'qmmm-protein-sites': ['protein', 'In a protein'], 'qmmm-embed': ['explicit', 'Into this system'], 'qmmm-inspect': ['explicit', 'Energy split'],
  'hessian-sample': ['discover', 'Hessian sampling'], 'hessian-global': ['discover', 'Basin hopping'], 'graph-enumeration': ['discover', 'Network expansion'],
};
// The first sentence, without the trailing CLI hint: what the tile's tooltip and form head say.
export const oneLine = (s) => (s || '').replace(/\s*\(`[^`]*`\)\s*$/, '').split(/(?<=[.;])\s/)[0];

export function ActTile({ icon, name, tip, on = false, primary = false, onClick }) {
  return html`<button type="button" class=${`act-tile ${on ? 'on' : ''} ${primary ? 'primary' : ''}`} title=${tip} aria-expanded=${on} onClick=${onClick}>
    <${QuestIcon} id=${icon} size=${28} /><span>${name}</span></button>`;
}

export function ActForm({ title, summary, why, onClose, children }) {
  return html`<div class="op-card open act-form">
    <div class="act-form-head"><div><b>${title}${why && html`<${Why} id=${why} label=${title} />`}</b>${summary && html`<span class="small muted">${summary}</span>`}</div>
      <button class="btn-icon" title="Close" aria-label="Close" onClick=${onClose}>✕</button></div>
    ${children}
  </div>`;
}
import { Why } from './Why.js';
import { ParamForm, clampToSchema, defaultsFor, rememberValues, rememberedValues } from './ParamForm.js';
import { conformerLabel, conformerRows, isComplex, isTs } from '../util.js';
import { ComplexMethodForm, memberCounts } from './ComplexBuild.js';

// How many jobs an operation would create for this selection (0 = not applicable).
export function applicability(op, sel) {
  if (op.hidden) return null;   // started by the app itself (e.g. a network variant's jobs)
  const nS = sel.structures.length, nE = sel.edges.length;
  // A QM/MM system takes only the operations that run embedded (the rest
  // need whole molecules); and is already embedded.
  const ws = state.workspace;
  const ids = [...sel.structures, ...sel.edges.flatMap((id) => (ws.edges[id] ? [ws.edges[id].source, ws.edges[id].target] : []))];
  const qmmm = ids.some((id) => ws.structures[id]?.qmmm);
  const mixed = qmmm && ids.some((id) => !ws.structures[id]?.qmmm);
  // A QM/MM structure with a gas-phase one: only putting the second into the first's system.
  if (op.key === 'qmmm-embed') {
    if (!(mixed && ids.length === 2)) return null;
  } else if (mixed) return null;
  if (qmmm && !op.qmmm) return null;
  if (!qmmm && op.key === 'qmmm-inspect') return null;
  if (op.target === 'pair') {
    let fit = null;
    if (nE > 0 && nS === 0) fit = { n: nE, what: nE === 1 ? 'this edge' : `${nE} edges` };
    else if (nE === 0 && nS === 2) {
      // Different atom counts cannot be the two ends of a reaction (the panel says so): nothing to run.
      const [a, b] = sel.structures.map((id) => state.workspace.structures[id]);
      if (a && b && a.natoms !== b.natoms) return null;
      fit = { n: 1, what: 'these two structures' };
    }
    if (fit && op.needs_route_ts) Object.assign(fit, routeTsCheck(sel));
    return fit;
  }
  if (op.target === 'structure') {
    if (!(nS > 0 && nE === 0)) return null;
    // e.g. a VRI search: offered only when every selected structure is a TS.
    if (op.structure_role === 'ts' && !sel.structures.every((id) => isTs(state.workspace.structures[id]))) return null;
    return { n: nS, what: nS === 1 ? 'this structure' : `${nS} structures` };
  }
  if (op.target === 'job') return null;   // follow-ups live on their source job's result page
  if (op.target === 'set') {
    if (!(nS >= op.min_structures && nE === 0)) return null;
    const fit = { n: 1, what: nS === 1 ? 'this structure' : `${nS} structures` };
    if (op.same_atoms) {
      const recs = sel.structures.map((id) => state.workspace.structures[id]).filter(Boolean);
      const key = (r) => `${r.formula}|${r.charge}|${r.multiplicity}`;
      if (new Set(recs.map(key)).size > 1) {
        fit.disabled = 'Needs isomers: the selected structures differ in atoms, charge or spin. The Nanoreactor handles molecules with different atoms.';
      }
    }
    return fit;
  }
  return null;
}

const IRC_GROUP_KINDS = ['irc', 'channel', 'alternate', 'offtarget'];

function sameSet(a, b) {
  return a.length === b.length && a.every((x) => b.includes(x));
}

// Does each selected edge (or the selected pair) have an IRC-verified TS
// from a finished TS / channels job? A VRI search starts from that TS.
function routeTsCheck(sel) {
  const ws = state.workspace;
  const pairs = sel.edges.length
    ? sel.edges.map((id) => ws.edges[id]).filter(Boolean).map((e) => ({ ids: [e.source, e.target], eid: e.id }))
    : [{ ids: sel.structures, eid: null }];
  const jobs = Object.values(state.jobs).filter((j) => j.op === 'ts' || j.op === 'channels');
  let running = 0, unverified = 0, missing = 0;
  const found = [];
  for (const p of pairs) {
    const mine = jobs.filter((j) => (p.eid && j.targets.edges.includes(p.eid)) || sameSet(j.targets.structures, p.ids));
    const ok = mine.filter((j) => j.status === 'done' && j.summary?.route_ts).map((j) => j.summary.route_ts);
    // An edge added from a result's IRC ("Add ends + edge to Graph") has that IRC's TS.
    const origin = p.eid ? ws.edges[p.eid]?.origin : null;
    if (origin?.kind === 'job' && origin.has_ts !== false
        && (IRC_GROUP_KINDS.includes(origin.group) || String(origin.entry || '').endsWith('_irc'))) {
      ok.push({ label: 'the TS of the IRC this edge was added from', barrier_kcal: origin.barrier_kcal ?? null });
    }
    ok.sort((x, y) => (x.barrier_kcal ?? 1e9) - (y.barrier_kcal ?? 1e9));
    if (ok.length) { found.push(ok[0]); continue; }
    if (mine.some((j) => j.status === 'queued' || j.status === 'running')) running += 1;
    else if (mine.some((j) => j.status === 'done')) unverified += 1;
    else missing += 1;
  }
  const blocked = running + unverified + missing;
  if (!blocked) {
    const ts = found.length === 1 ? found[0] : null;
    return { note: ts ? `Starts from ${ts.label}${ts.barrier_kcal != null ? ` (ΔE‡ ${ts.barrier_kcal.toFixed(1)} kcal/mol)` : ''}: the IRC-verified TS of this edge.` : 'Each edge starts from its IRC-verified TS.' };
  }
  if (pairs.length > 1) {
    return { disabled: `${blocked} of the ${pairs.length} selected edges have no IRC-verified transition state yet. Run 'Transition state' or 'Reaction channels' on them first, or select only edges that have one.` };
  }
  if (running) return { disabled: 'A transition-state search on this edge is still running. This becomes available once it finds a TS whose IRC connects the two structures.' };
  if (unverified) return { disabled: 'The searches on this edge found no TS whose IRC connects both structures (only a path maximum or other saddles), so there is no transition state to start from.' };
  return { disabled: "Needs a transition state for this edge first: run 'Transition state' or 'Reaction channels'. The VRI search starts from the TS whose IRC connects the two structures." };
}

// The two structures of a pair calculation, start first: an edge's ends, or
// two selected structures in click order.
function pairEnds(sel, ws) {
  if (sel.edges.length === 1 && !sel.structures.length) {
    const e = ws.edges[sel.edges[0]];
    return e ? [ws.structures[e.source], ws.structures[e.target]] : null;
  }
  if (sel.structures.length === 2 && !sel.edges.length) return sel.structures.map((id) => ws.structures[id]);
  return null;
}

function EndpointConformers({ ends, value, onChange }) {
  return html`<div class="field conf-picks" title="The geometry each end starts from. Saved on the edge, so later calculations on it use the same ones.">
    <span class="field-label">Endpoint conformers <span class="field-info" aria-hidden="true">ⓘ</span></span>
    ${ends.map((r, i) => r && html`<label class="small">
      <span class="muted">${i === 0 ? 'start' : 'end'} · ${r.name}</span>
      <select value=${value[r.id] || ''} disabled=${(r.conformers || []).length <= 1}
        onChange=${(e) => onChange({ ...value, [r.id]: e.target.value || null })}>
        <option value="">Lowest energy (default)</option>
        ${conformerRows(r).map((c) => html`<option value=${c.id}>${conformerLabel(c)}</option>`)}
      </select></label>`)}
  </div>`;
}

export function ProfilePicker({ value, onChange }) {
  const profiles = useStore((s) => s.profiles);
  const levels = useStore((s) => s.levels);
  const levelProfile = useStore((s) => s.levelProfile);
  const summary = useStore((s) => s.pathSummaries[value ?? '']);
  return html`
    <div class="field" title="The level of theory and path-search settings this runs with (a RunInputs TOML; edit under Settings)">
      <label class="field-label">Compute profile<${Why} id="level.profile" label="Compute profile" fallback=${html` <span class="field-info" aria-hidden="true">ⓘ</span>`} /></label>
      <select value=${value || ''} onChange=${(e) => onChange(e.target.value || null)}>
        <option value="">mepd built-in defaults · ${levels['']?.label ?? ''}</option>
        ${profiles.map((p) => html`<option value=${p}>${p} · ${levels[p]?.label ?? ''}${p === levelProfile ? ' (workspace level)' : ''}</option>`)}
      </select>
      ${summary && html`<span class="field-help">Path search: <b>${summary.text}</b></span>`}
      ${summary?.warnings.map((w) => html`<p class="level-note small">${w}</p>`)}
    </div>`;
}

// The schema without the fields a method fixes (they are not the user's to set there).
function withoutFields(schema, keys) {
  if (!schema || !keys.length) return schema;
  const properties = Object.fromEntries(Object.entries(schema.properties || {}).filter(([k]) => !keys.includes(k)));
  return { ...schema, properties };
}

// A done, running or queued job of `op` on the same structures with the same
// settings and profile: running it again only repeats it.
function sameCalculation(op, sel, params, profile) {
  const ws = state.workspace;
  const ends = (t) => [...new Set([...(t.structures || []), ...(t.edges || []).flatMap((id) => (ws.edges[id] ? [ws.edges[id].source, ws.edges[id].target] : []))])].sort().join(',');
  const want = ends(sel);
  return Object.values(state.jobs).find((j) => j.op === op.key && ['done', 'running', 'queued'].includes(j.status)
    && (j.profile ?? null) === (profile ?? null) && ends(j.targets || {}) === want
    && Object.entries(params).every(([k, v]) => JSON.stringify(j.params?.[k] ?? null) === JSON.stringify(v ?? null))) || null;
}

function OperationCard({ op, sel, fit, open, onToggle, fixed = null, embedded = false }) {
  const [values, setValues] = useState(() => rememberedValues(op));
  // Default to the session's level-of-theory profile, so a calculation runs
  // on the surface its structures were minimized on.
  const [profile, setProfile] = useState(() => defaultProfile());
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState(null);
  // A complex's Conformers: arrange its molecules (the complex builders), or
  // sample the whole as one structure (RDKit/CREST, as for a molecule).
  const [cxWay, setCxWay] = useState('arrange');

  useEffect(() => { setPreview(null); }, [sel, values, profile, JSON.stringify(fixed)]);

  // Drop remembered keys the schema no longer has.
  const clean = (v) => Object.fromEntries(Object.entries(v).filter(([k]) => k in (op.schema?.properties || {})));
  // A pair calculation: which conformer of each end (default: its lowest).
  const ws = useStore((s) => s.workspace);
  const ends = op.target === 'pair' ? pairEnds(sel, ws) : null;
  const edgePicks = sel.edges.length === 1 ? (ws.edges[sel.edges[0]]?.conformers || {}) : {};
  const [confs, setConfs] = useState(() => ({ ...edgePicks }));
  useEffect(() => { setConfs({ ...edgePicks }); }, [sel.edges.join(','), sel.structures.join(','), JSON.stringify(edgePicks)]);
  const showConfs = ends && ends.some((r) => (r?.conformers || []).length > 1);
  const params = () => ({ ...clean(values), ...(fixed || {}) });
  const body = (dry) => ({
    op: op.key, structures: sel.structures, edges: sel.edges,
    params: params(), profile, dry_run: dry,
    ...(showConfs ? { conformers: Object.fromEntries(ends.map((r) => [r.id, confs[r.id] || null])) } : {}),
  });

  const run = async () => {
    const twin = sameCalculation(op, sel, params(), profile);
    if (twin && !confirm(`This exact calculation ${twin.status === 'done' ? 'already ran' : `is already ${twin.status}`} `
      + `(${twin.title}): same structures, settings and profile, so it would ${twin.status === 'done' ? 'give the same result' : 'run twice'}. Run it anyway?`)) return;
    setBusy(true);
    rememberValues(op, clean(values));   // (a method's fixed params are not remembered as choices)
    prefs.set('profile', profile);
    const jobs = await attempt(() => api.post('/api/jobs', body(false)),
      (js) => (js.length === 1 ? `Queued: ${js[0].title}` : `Queued ${js.length} jobs`));
    setBusy(false);
    if (jobs?.length === 1) openJob(jobs[0].id);
  };
  const showCommand = async () => {
    const out = await attempt(() => api.post('/api/jobs', body(true)));
    if (out) setPreview(out.map((j) => j.command));
  };

  // Level-of-theory consistency between the chosen profile and the targets.
  const levels = useStore((s) => s.levels);
  const structures = useStore((s) => s.workspace.structures);
  const edges = useStore((s) => s.workspace.edges);
  const targetIds = sel.edges.length
    ? sel.edges.flatMap((id) => (edges[id] ? [edges[id].source, edges[id].target] : []))
    : sel.structures;
  const jobKey = levels[profile ?? '']?.key;
  const offLevel = targetIds.map((id) => structures[id]).filter((r) => r && !(r.optimized && r.level?.key === jobKey));
  const pairRecs = op.target === 'pair' && targetIds.length === 2 ? targetIds.map((id) => structures[id]) : null;
  const pairNotes = [];
  if (pairRecs && pairRecs.every(Boolean)) {
    for (const r of pairRecs) {
      if (r.reacted) pairNotes.push(`${r.name} reacted while being minimized (it was entered as ${r.reacted.from}), so it is not the structure you typed: at this level that input has no barrier to ${r.reacted.to}.`);
    }
    if (pairRecs[0].smiles && pairRecs[0].smiles === pairRecs[1].smiles) {
      pairNotes.push(`Both ends are ${pairRecs[0].smiles}: this searches a conformational change, not a reaction.`);
    }
  }
  const levelNote = !offLevel.length || op.key === 'optimize' || op.key === 'channels' ? null
    : op.target === 'pair'
      ? `${offLevel.length} endpoint(s) are not minima at ${levels[profile ?? '']?.label ?? 'this level'}; they will be re-minimized at this profile first, so the whole path is on one surface.`
      : op.key === 'tsopt' || op.structure_role === 'ts' ? null
        : `${offLevel.map((r) => r.name).join(', ')} ${offLevel.length > 1 ? 'are' : 'is'} not a minimum at ${levels[profile ?? '']?.label ?? 'this level'}. Optimize at this profile first, or results will mix levels of theory.`;

  if (fit.disabled) {
    return html`<div class="op-card disabled" aria-disabled="true">
      <div class="op-head"><span class="op-text"><span class="op-title">${op.title}</span>
        <span class="op-summary">${op.summary.replace(/\s*\(`[^`]*`\)\s*$/, '')}</span></span></div>
      <p class="small why-disabled">${fit.disabled}</p>
    </div>`;
  }
  if (!op.available) {
    return html`<div class="op-card disabled" title=${op.unavailable_reason}>
      <div class="op-head"><span class="op-title">${op.title}</span><span class="badge muted">not available</span></div>
      <p class="small muted">${op.unavailable_reason}</p>
    </div>`;
  }
  const formSchema = withoutFields(op.schema, Object.keys(fixed || {}));
  // The form sees the fixed values too, so fields that depend on them show or hide.
  const formValues = { ...values, ...(fixed || {}) };
  const opBody = html`
        <div class="op-body">
          <${ParamForm} op=${op.key} schema=${formSchema} values=${formValues} onChange=${(v) => setValues(Object.fromEntries(Object.entries(v).filter(([k]) => !(k in (fixed || {})))))} />
          <${ProfilePicker} value=${profile} onChange=${setProfile} />
          ${showConfs && html`<${EndpointConformers} ends=${ends} value=${confs} onChange=${setConfs} />`}
          ${fit.note && html`<p class="small muted">${fit.note}</p>`}
          ${levelNote && html`<p class="level-note small">${levelNote}</p>`}
          ${pairNotes.map((n) => html`<p class="level-note small">${n}</p>`)}
          <div class="op-run">
            <button class="btn primary" disabled=${busy} onClick=${run}>
              ${busy ? 'Queuing…' : fit.n > 1 ? `Run on ${fit.what} (${fit.n} jobs)` : `Run on ${fit.what}`}
            </button>
            <button class="btn-link small" onClick=${showCommand}>Show command</button>
          </div>
          ${preview && html`<pre class="cmd-preview">${preview.join('\n\n')}</pre>`}
        </div>`;
  const cxRec = op.key === 'conformers' && sel.structures.length === 1 && !sel.edges.length ? structures[sel.structures[0]] : null;
  const isCx = cxRec && isComplex(cxRec) && (cxRec.members || []).length > 1;
  const cardBody = !isCx ? opBody : html`<div class="op-body">
    <div class="segmented wide">
      <button class=${cxWay === 'arrange' ? 'on' : ''} onClick=${() => setCxWay('arrange')}>Arrange the molecules</button>
      <button class=${cxWay === 'sample' ? 'on' : ''} onClick=${() => setCxWay('sample')}>RDKit / CREST</button>
    </div>
    ${cxWay === 'arrange'
      ? html`<p class="small muted">Other arrangements of its molecules; each becomes a geometry of this complex.</p>
        <${ComplexMethodForm} counts=${memberCounts(cxRec.members)} runLabel="Find arrangements" />`
      : html`<p class="small muted">Samples the complex as one structure, as for a molecule: its conformers join this geometry.
        Plain CREST can let the molecules drift apart; Arrange › Ensemble (CREST's NCI mode) keeps them together.</p>${opBody}`}
  </div>`;
  if (embedded) return cardBody;
  return html`
    <div class=${`op-card ${open ? 'open' : ''}`}>
      <button type="button" class="op-head" onClick=${onToggle} aria-expanded=${open}>
        <span class="op-text">
          <span class="op-title">${op.title}</span>
          <span class="op-summary">${op.summary.replace(/\s*\(`[^`]*`\)\s*$/, '')}</span>
        </span>
        <span class="op-chevron" aria-hidden="true">${open ? '−' : '+'}</span>
      </button>
      ${open && cardBody}
    </div>`;
}

// Several operations as one card: a method switch, then the chosen
// method's form (e.g. the ways of exploring a reaction network).
function FamilyCard({ family, methods, sel, open, onToggle, embedded = false }) {
  const [pick, setPick] = useState(() => {
    const saved = prefs.get(`family:${family.key}`, null);
    return methods.some((m) => m.id === saved) ? saved : (methods.find((m) => m.available) || methods[0]).id;
  });
  const m = methods.find((x) => x.id === pick) || methods[0];
  const choose = (id) => { setPick(id); prefs.set(`family:${family.key}`, id); };
  // Sent here for one of its methods (a tile in Your tools): show that method.
  const focusOp = useStore((s) => s.focusOp);
  useEffect(() => {
    if (!focusOp || Date.now() - focusOp.at > 10000) return;
    const hit = methods.find((x) => x.op?.key === focusOp.key);
    if (hit) choose(hit.id);
  }, [focusOp?.at]);
  const body = html`<div class="family-body">
        ${[...new Set(methods.map((x) => x.group || ''))].map((g) => html`
        ${g && html`<div class="method-group small">${g}</div>`}
        <div class="method-switch" role="tablist">
          ${methods.filter((x) => (x.group || '') === g).map((x) => html`<button role="tab" aria-selected=${x.id === m.id} class=${`method ${x.id === m.id ? 'on' : ''} ${x.available ? '' : 'off'}`}
            title=${x.available ? x.summary : x.reason} onClick=${() => choose(x.id)}>${x.label}</button>`)}
        </div>`)}
        <p class="small method-summary" title=${m.summary}>${oneLine(m.summary)}<${Why} ids=${[`method.${m.id}`, `op.${m.op.key}`]} label=${m.label} /></p>
        ${!m.available ? html`<p class="warn-box small">Not available here: ${m.reason}.</p>`
          : m.fit.disabled ? html`<p class="small why-disabled">${m.fit.disabled}</p>`
            : html`<${OperationCard} key=${m.id} op=${m.op} sel=${sel} fit=${m.fit} open=${true} embedded fixed=${m.fixed} />`}
      </div>`;
  if (embedded) return body;
  return html`
    <div class=${`op-card family ${open ? 'open' : ''}`}>
      <button type="button" class="op-head" onClick=${onToggle} aria-expanded=${open}>
        <span class="op-text">
          <span class="op-title">${family.title}</span>
          <span class="op-summary">${methods.map((x) => x.label).join(' · ')}</span>
        </span>
        <span class="op-chevron" aria-hidden="true">${open ? '−' : '+'}</span>
      </button>
      ${open && body}
    </div>`;
}

export function defaultProfile() {
  if (state.levelProfile && state.profiles.includes(state.levelProfile)) return state.levelProfile;
  return prefs.get('profile', state.profiles.includes('default') ? 'default' : null);
}

// `extras`: actions that are not calculations ({key, scale, last, icon, name, tip, act() | render(close)}),
// first in their scale's group, or last with `last` (picking products and the reactor: network; a complex: molecule).
export function ActionPanel({ sel, extras = [] }) {
  const operations = useStore((s) => s.operations);
  const map = useMap();
  const focusOp = useStore((s) => s.focusOp);
  const [openKey, setOpenKey] = useState(null);
  // Operations the map has not opened yet stay out of sight (quests.js).
  const all = operations.filter((op) => map.op(op.key) && !SWITCH_OPS.has(op.key)).map((op) => ({ op, fit: applicability(op, sel) })).filter((x) => x.fit);
  // A quest's "Try it": open that operation's card and bring it into view.
  useEffect(() => {
    if (!focusOp || Date.now() - focusOp.at > 10000) return;
    const op = operations.find((o) => o.key === focusOp.key);
    if (!op || !all.some((x) => x.op.key === op.key)) return;
    setOpenKey(op.family ? op.family.key : op.key);
    setTimeout(() => {
      const card = document.querySelector('.actions .op-card.open');
      card?.scrollIntoView({ block: 'start', behavior: 'smooth' });
      card?.classList.add('focus-flash');
      setTimeout(() => card?.classList.remove('focus-flash'), 2600);
    }, 150);
  }, [focusOp?.at]);
  const surroundings = html`<${Surroundings} sel=${sel} />`;
  if (!all.length && !extras.length) return html`<section class="actions">${surroundings}</section>`;
  const fits = all.filter((x) => x.op.available);
  const later = all.filter((x) => !x.op.available);
  const byScale = {};
  const scaleOf = (key) => SCALE_OF[key] || 'molecule';
  // Operations of one family (e.g. network exploration) become one card, in its first member's place.
  const families = {};
  for (const x of fits) {
    const fam = x.op.family;
    if (!fam) { (byScale[scaleOf(x.op.key)] ||= []).push(x); continue; }
    if (!families[fam.key]) {
      families[fam.key] = { family: fam, methods: [] };
      (byScale[scaleOf(fam.key)] ||= []).push({ familyKey: fam.key });
    }
    const ws = state.workspace;
    const qmmmSel = [...sel.structures, ...sel.edges.flatMap((id) => (ws.edges[id] ? [ws.edges[id].source, ws.edges[id].target] : []))]
      .some((id) => ws.structures[id]?.qmmm);
    for (const [i, meth] of (x.op.methods?.length ? x.op.methods : [{ label: x.op.title, summary: x.op.summary, fixed: {}, available: true }]).entries()) {
      if (meth.qmmm_only && !qmmmSel) continue;   // e.g. the nanoreactor on a QM region (molecules have their own card)
      families[fam.key].methods.push({ ...meth, id: `${x.op.key}:${i}`, op: x.op, fit: x.fit });
    }
  }
  Object.values(families).forEach((f) => f.methods.sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0)));
  const toggle = (k) => setOpenKey(openKey === k ? null : k);
  const tileOf = (item) => {
    if (item.familyKey) {
      const f = families[item.familyKey].family;
      const first = families[item.familyKey].methods[0]?.op.key;
      const several = new Set(families[item.familyKey].methods.map((x) => x.op.key)).size > 1;
      const [icon, name] = TILE[f.key] || (several ? ['discover', f.title] : TILE[first]) || ['discover', f.title];
      return { key: item.familyKey, icon, name: f.key === first ? name : f.title.replace(/^Reaction network /, 'Network '), tip: families[item.familyKey].methods.map((x) => x.label).join(' · '),
        form: () => html`<${ActForm} title=${f.title} onClose=${() => setOpenKey(null)}>
          <${FamilyCard} family=${f} methods=${families[item.familyKey].methods} sel=${sel} open=${true} embedded /><//>` };
    }
    const { op, fit } = item;
    const [icon, name] = TILE[op.key] || ['molecule', op.title];
    return { key: op.key, icon, name, tip: oneLine(op.summary),
      form: () => html`<${ActForm} title=${op.title} summary=${oneLine(op.summary)} why=${`op.${op.key}`} onClose=${() => setOpenKey(null)}>
        <${OperationCard} op=${op} sel=${sel} fit=${fit} open=${true} embedded /><//>` };
  };
  const extraTile = (x) => ({ ...x,
    form: x.render && (() => html`<${ActForm} title=${x.name} why=${x.why} onClose=${() => setOpenKey(null)}>${x.render(() => setOpenKey(null))}<//>`) });
  const groups = SCALES.map(([key, name, tip]) => ({ name, tip,
    tiles: [...extras.filter((x) => (x.scale || 'molecule') === key && !x.last).map(extraTile), ...(byScale[key] || []).map(tileOf),
      ...extras.filter((x) => (x.scale || 'molecule') === key && x.last).map(extraTile)] }))
    .filter((g) => g.tiles.length);
  return html`
    <section class="actions">
      ${surroundings}
      ${groups.map((g) => {
        const open = g.tiles.find((t) => t.key === openKey);
        return html`<div class="action-group">
          <h3 class="act-cat" title=${g.tip}>${g.name}</h3>
          <div class="act-tiles">${g.tiles.map((t) => html`<${ActTile} key=${t.key} icon=${t.icon} name=${t.name} tip=${t.tip}
            on=${t.key === openKey} onClick=${() => (t.act ? t.act() : toggle(t.key))} />`)}</div>
          ${open && open.form && open.form()}
        </div>`;
      })}
      ${later.length > 0 && html`<p class="small muted later">Not available here: ${later.map(({ op }, i) =>
        html`${i ? ', ' : ''}<span title=${op.unavailable_reason}>${op.title}</span>`)}.</p>`}
    </section>`;
}

// Where the selection is modelled: gas phase, implicit solvent (Explore's
// Conditions, over the whole network), explicit solvent or a protein (QM/MM
// systems, their own nodes in the graph). The groups below then run there.
const SURROUNDINGS = [
  ['gas', 'Gas phase', 'Isolated molecules'],
  ['solvent', 'Solvent', 'Implicit solvent over the whole network: Explore › Conditions'],
  ['explicit', 'Explicit', 'Solvent molecules around it, as a QM/MM system'],
  ['protein', 'Protein', 'In a protein pocket, as a QM/MM system'],
];
function Surroundings({ sel }) {
  const operations = useStore((s) => s.operations);
  const ws = useStore((s) => s.workspace);
  const map = useMap();
  const [pick, setPick] = useState(null);
  useEffect(() => setPick(null), [sel.structures.join(','), sel.edges.join(',')]);
  const ids = [...sel.structures, ...sel.edges.flatMap((id) => (ws.edges[id] ? [ws.edges[id].source, ws.edges[id].target] : []))];
  const recs = ids.map((id) => ws.structures[id]).filter(Boolean);
  if (!recs.length) return null;
  const sysOf = (r) => (r.qmmm ? ws.qmmm_systems?.[r.qmmm] : null);
  const kindOf = (sys) => (!sys ? 'gas' : ['amber', 'terachem'].includes(sys.mm) ? 'protein' : 'explicit');
  const kinds = new Set(recs.map((r) => kindOf(sysOf(r))));
  if (kinds.size > 1) return null;   // gas with QM/MM: putting one into the other (a tile)
  const now = [...kinds][0];
  const one = sel.structures.length === 1 && !sel.edges.length ? recs[0] : null;
  // The same species elsewhere: its gas-phase node, or its QM/MM copy of a kind.
  const twin = (kind) => one && Object.values(ws.structures).find((r) => r.id !== one.id && r.smiles === one.smiles
    && r.charge === one.charge && kindOf(sysOf(r)) === kind);
  const opFor = (kind) => {
    const key = kind === 'protein' ? (one ? 'qmmm-protein-sites' : null)
      : kind === 'explicit' ? (one ? 'qmmm-build' : 'qmmm-reaction') : null;
    const op = key && operations.find((o) => o.key === key);
    if (!op || !map.op(op.key)) return null;
    const fit = applicability(op, sel);
    return fit ? { op, fit } : null;
  };
  // Shown once something besides the gas phase is open here (a quest, or "show everything").
  const solventOpen = map.op('solvent');
  // Surroundings the map has not opened yet stay out of sight, like the tools.
  const opened = { gas: true, solvent: solventOpen, explicit: map.op('qmmm-build') || map.op('qmmm-reaction'), protein: map.op('qmmm-protein-sites') };
  const choices = SURROUNDINGS.filter(([key]) => opened[key] || key === now).map(([key, label, tip]) => {
    const go = twin(key);
    const build = now === 'gas' && opFor(key);
    const can = key === now || (key === 'solvent' ? solventOpen && now === 'gas' : Boolean(go || build));
    return { key, label, tip, go, build, can };
  });
  if (now === 'gas' && !choices.some((c) => c.key !== 'gas' && c.can)) return null;
  const choose = (c) => {
    if (c.key === now) { setPick(null); return; }
    if (c.key === 'solvent') {
      openTab('graph');
      setTimeout(() => window.dispatchEvent(new CustomEvent('mepd:graph', { detail: { cmd: 'conditions' } })), 50);
      return;
    }
    if (c.go && !c.build) { select({ structures: [c.go.id] }); return; }
    setPick(pick === c.key ? null : c.key);
  };
  const open = choices.find((c) => c.key === pick);
  const label = (k) => SURROUNDINGS.find((x) => x[0] === k)[1].toLowerCase();
  return html`<div class="surroundings">
    <div class="surr-row"><span class="surr-label">In<${Why} id="other.surroundings" label="Surroundings" /></span>
      <div class="surr-switch" role="radiogroup" aria-label="Surroundings">
        ${choices.map((c) => html`<button type="button" role="radio" aria-checked=${c.key === now}
          class=${`${c.key === now ? 'on' : ''} ${c.key === pick ? 'picked' : ''}`} disabled=${!c.can}
          title=${c.can ? c.tip : `${c.tip} (not available for this selection${c.key === 'solvent' && now !== 'gas' ? ': already in explicit surroundings' : ''})`}
          onClick=${() => choose(c)}>${c.label}</button>`)}
      </div></div>
    ${open && html`<${ActForm} title=${open.build ? open.build.op.title : open.label} summary=${open.build && oneLine(open.build.op.summary)}
        why=${open.build && `op.${open.build.op.key}`} onClose=${() => setPick(null)}>
      ${open.go && html`<p class="small">It is already modelled ${open.key === 'gas' ? 'in the gas phase' : `in ${label(open.key) === 'explicit' ? 'explicit solvent' : 'a protein'}`}:${' '}
        <button class="btn-link" onClick=${() => select({ structures: [open.go.id] })}>go to it</button>${open.build ? ', or build another below.' : '.'}</p>`}
      ${open.build && html`<${OperationCard} op=${open.build.op} sel=${sel} fit=${open.build.fit} open=${true} embedded />`}
    <//>`}
  </div>`;
}
