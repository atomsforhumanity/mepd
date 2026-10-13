// Network variants: carry a Design edit of a species (e.g. methyls for
// hydrogens) through every reaction of the network, recompute them, and
// compare the kinetics (mepd/web/variants.py). Shown in Design when the
// design was opened from a species in Explore.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openTab, set, state, toast } from '../store.js';

const STATUS = {
  ok: ['recompute', 'Fast estimate, then TS re-optimized from the substituted one (IRC checked)'],
  no_ts: ['path search', 'No TS of the parent to start from: a path search between the substituted ends'],
  blocked: ['blocked', 'The edited atoms take part in this step: it has no analogue in the variant'],
};

export function ApplyToNetwork({ design, busy }) {
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState('');
  useEffect(() => { setPreview(null); }, [design.molblock]);
  if (!design.source?.parent_molblock || design.reaction) return html`<${OneMolecule} design=${design} />`;
  const on = design.source.network ? state.workspace.variants?.[design.source.network] : null;

  const look = async () => {
    setLoading(true);
    const out = await attempt(() => api.post('/api/variants/preview'));
    setLoading(false);
    if (out) setPreview(out);
  };
  const run = async () => {
    setLoading(true);
    const out = await attempt(() => api.post('/api/variants', { name: name.trim() }));
    setLoading(false);
    if (!out) return;
    setPreview(null);
    set({ network: out.variant.id });
    toast(`Variant ${out.variant.name}: ${out.jobs.length} calculations started (fast estimates first)`, 'ok', 8000,
      { label: 'Explore', run: () => openTab('graph') });
  };

  return html`<div class="variant-box">
    <div class="section-title">Across the network</div>
    ${!preview && html`<p class="small muted">Make the same edit in every reaction of ${design.source.name}, and in what it
      becomes${on ? html`, in the variant <b>${on.name}</b> (on top of its edit)` : ''}, then recompute them and compare the kinetics.</p>
      <button class="btn small" disabled=${busy || loading} onClick=${look}>${loading ? 'Mapping…' : 'Apply to network…'}</button>`}
    ${preview && html`<${Preview} p=${preview} />
      <div class="variant-run">
        <input class="tiny-text" placeholder=${`${preview.base_name} edit`} value=${name} onInput=${(e) => setName(e.target.value)} />
        <button class="btn small primary" disabled=${loading || !preview.jobs} onClick=${run}>
          ${loading ? 'Starting…' : `Run · ${preview.jobs} reaction${preview.jobs === 1 ? '' : 's'}`}</button>
        <button class="btn-link small" onClick=${() => setPreview(null)}>cancel</button>
      </div>`}
  </div>`;
}

// Edits are carried per molecule: a design opened from a complex points at
// its molecules instead (an edit of one of them reaches the complex too).
function OneMolecule({ design }) {
  const src = design.source;
  const rec = src?.kind === 'graph' && !design.reaction ? state.workspace.structures?.[src.structure] : null;
  if (!rec || rec.role !== 'complex' || !rec.members?.length) return null;
  const open = (id) => attempt(() => api.post('/api/design/load', { structure: id, network: src.network || state.network || null }),
    'Loaded into Design');
  return html`<div class="variant-box">
    <div class="section-title">Across the network</div>
    <p class="small muted">Network edits are made on one molecule, then carried into every complex and reaction it is
      in. Open the one to change:</p>
    ${[...new Set(rec.members)].map((m) => html`<button class="btn small" onClick=${() => open(m)}>
      ${state.workspace.structures[m]?.name || m}</button> `)}
  </div>`;
}

function Preview({ p }) {
  return html`<div class="variant-preview small">
    <div class="mono">${p.edit.join('; ')}${p.on_name ? html` <span class="muted">· on top of ${p.on_name}</span>` : ''}</div>
    <div><span class="muted">Species</span> ${p.species.map((s) => html`<span class="chip" title=${`replaces ${s.parent_name}`}>${s.smiles || s.label}</span>`)}</div>
    <table class="rx-table"><tbody>${p.reactions.map((r) => html`<tr>
      <td class="rx-eq" title=${r.note || ''}>${r.label}${r.note ? ' *' : ''}</td>
      <td>${r.variants.map((v) => html`<span class=${`chip st-${v.status}`}
        title=${`${v.text}\n${v.placement}${v.why ? `\n${v.why}` : ''}${v.clash < 1.2 ? `\ncrowded: ${v.clash} Å` : ''}`}>${STATUS[v.status]?.[0] || v.status}</span>`)}</td>
    </tr>`)}</tbody></table>
    ${p.shared.length > 0 && html`<div class="muted" title=${p.shared.join('\n')}>${p.shared.length} reaction${p.shared.length > 1 ? 's' : ''} unchanged (shared)</div>`}
    ${p.problems.map((x) => html`<p class="warn-box small">${x.label}: ${x.why}</p>`)}
    ${p.warnings.map((w) => html`<p class="level-note small">${w}</p>`)}
  </div>`;
}

// Explore's toolbar, when a variant is shown: carry the edit into reactions
// added since, or remove the variant.
export function VariantTools({ vid }) {
  const v = state.workspace.variants?.[vid];
  if (!v) return null;
  const update = async () => {
    const out = await attempt(() => api.post(`/api/variants/${vid}/update`));
    if (out) toast(out.jobs.length ? `${out.jobs.length} new reactions in ${v.name}` : `${v.name} already covers every reaction`, 'ok');
  };
  const remove = async () => {
    if (!confirm(`Delete the variant ${v.name} and its species, reactions and results in Explore?`)) return;
    if (await attempt(() => api.del(`/api/variants/${vid}`))) set({ network: '' });
  };
  return html`<button class="btn small ghost" onClick=${update} title="Carry the edit into reactions added since">Update</button>
    <button class="btn small ghost danger-text" onClick=${remove} title="Delete this variant">✕</button>`;
}
