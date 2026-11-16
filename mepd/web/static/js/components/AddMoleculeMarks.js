// The network map with a molecule added (Add a molecule › Show on the map): what the variant record's
// `marks` say (mepd/web/addpage.py marks), drawn over the parent's own map. A step's colour is the change of
// its barrier (steps whose fastest version goes through the molecule) on Kinetics' diverging pair (blue lower, red higher), dashed while that is an estimate;
// a molecule with a doubled border has a complex with the added molecule that holds material.
import { html } from '../lib.js';
import { set, state } from '../store.js';

const FULL = 10;   // kcal/mol at which the colour is saturated

export function marksOf(workspace, network) {
  const v = workspace.variants?.[network];
  return v?.kind === 'added_molecule' ? v.marks || null : null;
}

export function markLook(mark, css, mix) {
  const t = Math.min(1, Math.abs(mark.change) / FULL);
  return { col: mix(css('--div-mid'), css(mark.change < 0 ? '--div-pos' : '--div-neg'), 0.35 + 0.65 * t), w: 2.2 + 2.2 * t };
}

const markKind = (mark) => (!mark ? undefined : mark.status !== 'verified' ? 'est' : 'ver');

// What a map element carries while a candidate is shown (cytoscape data): `am` on a step through the molecule
// ('est' | 'ver') and on a molecule it binds; `amfade` on everything else, so the few marked steps can be
// found among hundreds (their two ends stay as they are).
export function markData(marks, steps) {
  if (!marks) return { node: () => ({}), edge: () => ({}) };
  const ends = new Set(steps.filter((s) => marks.edges[s.id]).flatMap((s) => [s.a, s.b]));
  return {
    node: (id) => (marks.species[id] ? { am: 1 } : ends.has(id) ? {} : { amfade: 1 }),
    edge: (s) => (marks.edges[s.id] ? { am: markKind(marks.edges[s.id]) } : { amfade: 1 }),
  };
}

export const markTip = (mark) => (!mark ? '' : `   with the molecule: ${mark.change > 0 ? '+' : ''}${mark.change.toFixed(1)} kcal/mol ${
  mark.status === 'verified' ? '(verified)' : `(est.${mark.spread != null ? `, ± ${mark.spread.toFixed(1)}` : ', spread not given'})`}`);

export function MarksLegend({ workspace, network }) {
  const v = workspace.variants?.[network];
  const m = marksOf(workspace, network);
  if (!m) return null;
  const n = Object.keys(m.edges).length, held = Object.keys(m.species).length;
  const back = () => set({ addmol: { ...(state.addmol || {}), qid: v.addmol?.query, sel: v.addmol?.result },
    analyze: { ...(state.analyze || {}), compose: false, view: 'addmol' }, view: { tab: 'analyze', jobId: null } });
  return html`<div class="nm-added" role="note">
    <b>${v.name}</b>
    ${n ? html`<span class="nm-added-scale" title="Steps whose fastest version goes through the molecule: the change of their barrier (kcal/mol)">−${FULL}<i></i>+${FULL} kcal/mol</span>
      <span><i class="nm-added-line est"></i>estimate</span><span><i class="nm-added-line"></i>verified</span><span class="muted">the rest is faded</span>`
      : html`<span class="muted">no step goes through it</span>`}
    ${m.moved > 0 && html`<span class="muted" title="Their own transition states are unchanged: their reactants hold more or less material">${m.moved} more moved with the amounts</span>`}
    ${held > 0 && html`<span title=${Object.values(m.species).map((s) => s.name).join('\n')}><i class="nm-added-ring"></i>binds it (${held})</span>`}
    ${m.new.length > 0 && html`<span class="muted" title=${m.new.map((x) => x.name).join('\n')}>${m.new.length} new species or step${m.new.length > 1 ? 's' : ''} not on this map</span>`}
    ${m.status === 'running' && html`<span class="muted">still running</span>`}
    <button class="btn-link small" title="Back to this candidate in the list" onClick=${back}>Details</button>
    <button class="btn-link small" title="Back to the network without it" onClick=${() => set({ network: '' })}>✕</button>
  </div>`;
}
