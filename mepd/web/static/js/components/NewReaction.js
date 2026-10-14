// A new reaction by clicking: what the chosen molecules can become shows up
// at once (the bond rules of mepd.web.compose.propose, simplest first), and
// the products picked become reactions. Used in the Inspector (the selected
// molecules are the reactants) and in Analyze (pick them from the graph).
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openTab, set, state, useStore } from '../store.js';
import { depictUrl, isSpecies } from '../util.js';
import { useMap } from '../quests.js';
import { TsCell, findTs } from './Reactions.js';

const count = (ids) => ids.reduce((m, id) => ({ ...m, [id]: (m[id] || 0) + 1 }), {});
const usable = (s) => isSpecies(s) && !s.qmmm && s.smiles;

// Species of the graph as picture tiles: a click adds one (again: one more).
function SpeciesTiles({ chosen = [], onPick, exclude = [], limit = 18, autoFocus = false }) {
  const structures = useStore((s) => s.workspace.structures);
  const [q, setQ] = useState('');
  const all = useMemo(() => Object.values(structures).filter((s) => usable(s) && !exclude.includes(s.id))
    .sort((a, b) => (a.natoms - b.natoms) || a.name.localeCompare(b.name)), [structures, exclude.join()]);
  const ql = q.trim().toLowerCase();
  const hits = ql ? all.filter((s) => `${s.name} ${s.smiles} ${s.formula}`.toLowerCase().includes(ql)) : all;
  const c = count(chosen);
  return html`<div class="sp-pick">
    ${all.length > 8 && html`<input type="search" class="sp-search" placeholder=${`Search ${all.length} molecules`} value=${q}
      autoFocus=${autoFocus} onInput=${(e) => setQ(e.target.value)} />`}
    <div class="sp-tiles">${hits.slice(0, limit).map((s) => html`<button class=${`sp-tile ${c[s.id] ? 'on' : ''}`}
      title=${`${s.name} · ${s.smiles}`} onClick=${() => onPick(s.id)}>
      <img src=${depictUrl(s.smiles, 120, 90)} alt="" loading="lazy" />
      <span>${s.name}</span>${c[s.id] > 0 && html`<b class="sp-count">${c[s.id] > 1 ? `×${c[s.id]}` : '✓'}</b>`}
    </button>`)}</div>
    ${hits.length > limit && html`<p class="small muted">${hits.length - limit} more: search for them.</p>`}
    ${!hits.length && html`<p class="small muted">${all.length ? 'No match.' : 'No molecules in the graph yet.'}</p>`}
  </div>`;
}

// One reactant as a chip: picture, name, how many; − takes one away.
function ReactantChips({ ids, onChange, structures }) {
  const c = count(ids);
  return Object.entries(c).map(([id, n]) => html`<span class="nr-chip" title=${structures[id]?.smiles || ''}>
    <img src=${depictUrl(structures[id]?.smiles, 72, 54)} alt="" />
    <span>${n > 1 ? `${n} × ` : ''}${structures[id]?.name || '?'}</span>
    ${ids.length > 1 && html`<button class="btn-icon" title="One fewer"
      onClick=${() => { const k = ids.indexOf(id); onChange(ids.filter((_, i) => i !== k)); }}>−</button>`}
  </span>`);
}

function useProposals(reactants, rev) {
  const [out, setOut] = useState({ key: null, data: null, error: null });
  const key = `${[...reactants].sort().join(',')}#${rev}`;
  const seq = useRef(0);
  useEffect(() => {
    if (!reactants.length) { setOut({ key, data: null, error: null }); return undefined; }
    const n = ++seq.current;
    const t = setTimeout(() => {
      api.post('/api/reactions/propose', { reactants })
        .then((data) => { if (n === seq.current) setOut({ key, data, error: null }); })
        .catch((e) => { if (n === seq.current) setOut({ key, data: null, error: e.message || String(e) }); });
    }, 200);
    return () => clearTimeout(t);
  }, [key]);
  return out.key === key ? out : { key, data: null, error: null };
}

// "Breaks 1 bond, forms 1": a product's size of change, for its tooltip.
const change = (p) => `breaks ${p.broken.length} bond${p.broken.length === 1 ? '' : 's'}, forms ${p.formed.length}`;

export function ProductPicker({ reactants, onReactants, onAdded }) {
  const structures = useStore((s) => s.workspace.structures);
  const reactions = useStore((s) => s.workspace.reactions || {});
  const map = useMap();
  const [rev, setRev] = useState(0);              // asks again once reactions are added
  const { data, error } = useProposals(reactants, rev);
  const [picked, setPicked] = useState(new Set());
  const [partner, setPartner] = useState(false);
  const [own, setOwn] = useState(null);          // products picked by hand from the graph
  const [busy, setBusy] = useState(false);
  const [added, setAdded] = useState([]);        // reaction ids made here
  useEffect(() => { setPicked(new Set()); }, [data]);
  const props = data?.proposals || [];
  const toggle = (k) => { const n = new Set(picked); n.has(k) ? n.delete(k) : n.add(k); setPicked(n); };
  const openReaction = (rid) => set({ analyze: { ...(state.analyze || {}), view: 'reactions', compose: false, reaction: rid },
    view: { tab: 'analyze', jobId: null } });
  const add = async () => {
    setBusy(true);
    const made = [];
    if (own) {
      const r = await attempt(() => api.post('/api/reactions/compose', { reactants, products: own }));
      if (r) made.push(r.id);
    } else {
      for (const k of [...picked].sort((a, b) => a - b)) {
        const p = props[k];
        const r = await attempt(() => api.post('/api/reactions/compose', { reactants,
          proposal: { ...p, complex_xyz: data.complex_xyz, charge: data.charge, multiplicity: data.multiplicity } }));
        if (r) made.push(r.id);
      }
    }
    setBusy(false);
    if (!made.length) return;
    setAdded([...added, ...made]);
    setRev(rev + 1);
    setPicked(new Set());
    setOwn(null);
    onAdded?.(made);
  };
  const madeRs = added.map((id) => reactions[id]).filter(Boolean);
  const todo = madeRs.filter((r) => r.edge);
  const n = own ? (own.length ? 1 : 0) : picked.size;
  return html`<div class="newrx">
    <div class="nr-reactants">
      <${ReactantChips} ids=${reactants} onChange=${onReactants} structures=${structures} />
      <button class=${`nr-chip add ${partner ? 'on' : ''}`} title="Another molecule taking part (e.g. water relaying a proton)"
        onClick=${() => setPartner(!partner)}>＋ Partner</button>
    </div>
    ${partner && html`<div class="nr-partner"><${SpeciesTiles} chosen=${reactants} limit=${12} autoFocus=${true}
      onPick=${(id) => { onReactants([...reactants, id]); setPartner(false); }} /></div>`}

    ${own == null ? html`
      <div class="nr-head"><b>It can become</b>
        <span class="small muted">${error ? '' : !data ? 'working it out…' : props.length ? `${props.length} · click the ones to try` : ''}</span></div>
      ${error && html`<p class="error-box">${error}</p>`}
      ${!data && !error && html`<div class="nr-grid loading">${[0, 1, 2, 3].map(() => html`<span class="nr-prod ghost"></span>`)}</div>`}
      ${data && !props.length && html`<p class="small muted">Nothing within two bonds broken and two formed. Add a partner, or pick the products yourself.</p>`}
      ${props.length > 0 && html`<div class="nr-grid">${props.map((p, k) => html`<button key=${p.smiles}
        class=${`nr-prod ${picked.has(k) ? 'on' : ''} ${p.reaction ? 'made' : ''}`} aria-pressed=${picked.has(k)}
        title=${p.reaction ? `${p.smiles}\nAlready a reaction: click to open it` : `${p.smiles}\n${change(p)}`}
        onClick=${() => (p.reaction ? openReaction(p.reaction) : toggle(k))}>
        <span class="nr-frags">${p.fragments.map((f, i) => html`${i ? html`<i>+</i>` : ''}<img src=${depictUrl(f, 120, 90)} alt=${f} loading="lazy" />`)}</span>
        ${p.reaction ? html`<span class="nr-known">✓ added</span>`
          : p.known?.some(Boolean) && html`<span class="nr-known" title="Already in your graph">in graph</span>`}
        ${picked.has(k) && html`<span class="nr-check">✓</span>`}
      </button>`)}</div>`}
      <p class="small muted nr-alt"><a href="#" onClick=${(e) => { e.preventDefault(); setOwn([]); }}>Pick products from your graph</a>
        ${map.feature('design-tsopt') && html` · <a href="#" title="Draw the product atom by atom" onClick=${(e) => { e.preventDefault(); openTab('design'); }}>Draw it in Design</a>`}</p>`
    : html`
      <div class="nr-head"><b>Products</b><span class="small muted">click molecules from your graph; the atoms must balance</span></div>
      <${SpeciesTiles} chosen=${own} onPick=${(id) => setOwn([...own, id])} />
      ${own.length > 0 && html`<div class="nr-reactants"><span class="small muted">→</span>
        <${ReactantChips} ids=${own} onChange=${setOwn} structures=${structures} />
        <button class="btn-icon" title="Clear" onClick=${() => setOwn([])}>✕</button></div>`}
      <p class="small muted nr-alt"><a href="#" onClick=${(e) => { e.preventDefault(); setOwn(null); }}>Back to what it can become</a></p>`}

    <div class="nr-foot">
      <button class="btn primary" disabled=${busy || !n} onClick=${add}>
        ${busy ? 'Adding…' : n ? `Add ${n} reaction${n > 1 ? 's' : ''}` : 'Pick a product'}</button>
    </div>
    ${madeRs.length > 0 && html`<div class="nr-made">
      <div class="nr-head"><b>Added</b>${todo.length > 1 && html`<button class="btn-link small"
        onClick=${async () => { for (const r of todo) await findTs(r); }}>find all ${todo.length} barriers</button>`}</div>
      <ul>${madeRs.map((r) => html`<li><span>${r.label || r.id}</span><${TsCell} r=${r} /></li>`)}</ul>
    </div>`}
  </div>`;
}

// Analyze's version: pick the reactants from the graph first.
export function ReactionMaker({ initial, onDone }) {
  const [reactants, setReactants] = useState(initial || []);
  return html`<div class="newrx-wrap">
    <div class="nr-head big"><b>New reaction</b><button class="btn-link small" onClick=${() => onDone(null)}>Close</button></div>
    ${reactants.length
      ? html`<${ProductPicker} reactants=${reactants} onReactants=${setReactants} onAdded=${(ids) => onDone(ids[ids.length - 1], true)} />
        <p class="small muted nr-alt"><a href="#" onClick=${(e) => { e.preventDefault(); setReactants([]); }}>Start from other molecules</a></p>`
      : html`<div class="nr-head"><b>What reacts?</b><span class="small muted">click a molecule</span></div>
        <${SpeciesTiles} chosen=${reactants} onPick=${(id) => setReactants([...reactants, id])} limit=${24} />`}
  </div>`;
}
