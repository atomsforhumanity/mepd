// Substituent effects, picked on the molecule: tap atoms of the transition
// state to choose where groups go, choose groups from a short palette (the
// rest one click away), and see how many variants that makes. One specific
// change by hand goes to Design instead.
import { html, useEffect, useMemo, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openTab, toast } from '../store.js';
import { Viewer3D } from './Viewer3D.js';
import { ParamForm } from './ParamForm.js';
import { Why } from './Why.js';

// Shown first: donors to acceptors, the electronic trend.
const COMMON = ['amino', 'hydroxyl', 'methoxy', 'methyl', 'fluoro', 'chloro', 'trifluoromethyl', 'cyano', 'nitro'];

const atomsOf = (xyz) => (xyz || '').split('\n').slice(2).map((l) => l.trim().split(/\s+/)).filter((t) => t.length >= 4)
  .map(([el, x, y, z]) => ({ el, p: [+x, +y, +z] }));
// A hydrogen, or a heavy atom with a hydrogen on it (within 1.25 Å): somewhere a group can go.
const hasH = (atoms, i) => atoms[i]?.el === 'H' || atoms.some((a, j) => j !== i && a.el === 'H'
  && Math.hypot(a.p[0] - atoms[i].p[0], a.p[1] - atoms[i].p[1], a.p[2] - atoms[i].p[2]) < 1.25);
const parseSites = (s) => [...new Set((s || '').split(/[,;\s]+/).filter((x) => /^\d+$/.test(x)).map(Number))];

export function SubstituentPicker({ op, job, values, onChange }) {
  const [ts, setTs] = useState(null);
  const [more, setMore] = useState(false);
  useEffect(() => {
    let live = true;
    api.get(`/api/jobs/${job.id}/result`).then((r) => { if (live) setTs(r.route_ts?.xyz || null); }).catch(() => {});
    return () => { live = false; };
  }, [job.id, job.result_rev]);
  const atoms = useMemo(() => atomsOf(ts), [ts]);
  const symbols = atoms.map((a) => a.el);
  const props = op.schema?.properties || {};
  const all = props.groups?.items?.enum || [];
  const labels = props.groups?.labels || {};
  const groups = values.groups || [];
  const sites = parseSites(values.sites);
  const set = (k, v) => onChange({ ...values, [k]: v });
  const toggleSite = (i) => (!sites.includes(i) && !hasH(atoms, i)
    ? toast(`${name(i)} has no hydrogen to replace: tap a hydrogen, or an atom that carries one`, 'info')
    : set('sites', (sites.includes(i) ? sites.filter((x) => x !== i) : [...sites, i]).join(', ')));
  const toggleGroup = (g) => set('groups', groups.includes(g) ? groups.filter((x) => x !== g) : [...groups, g]);
  const name = (i) => `${symbols[i] || '?'}${i + 1}`;
  const short = (g) => String(labels[g] ?? g).replace(/\s*\(.*\)$/, '');
  const shownGroups = more ? all : [...COMMON.filter((g) => all.includes(g)), ...groups.filter((g) => !COMMON.includes(g))];
  // The rest of the form (geometries, advanced): as any other operation's.
  const rest = useMemo(() => {
    const { groups: _g, sites: _s, ...others } = props;
    return { ...op.schema, properties: others };
  }, [op.schema]);
  const maxSites = values.max_sites ?? props.max_sites?.default ?? 8;
  const nSites = sites.length;
  const toDesign = job.targets?.edges?.[0];
  return html`<div class="subst-picker">
    <div class="field"><span class="field-label">Where<${Why} ids=${['param.substituents.sites']} label="Where" /></span>
      ${ts ? html`<${Viewer3D} xyz=${ts} height=${240} labels=${true} labelBase=${1} highlight=${sites} onAtomClick=${toggleSite} />`
        : html`<p class="small muted">Loading the transition state…</p>`}
      <div class="subst-sites small">
        ${nSites ? html`${sites.map((i) => html`<button type="button" class="chip on" title="Remove" onClick=${() => toggleSite(i)}>${name(i)} ✕</button>`)}
            <button type="button" class="btn-link small" onClick=${() => set('sites', '')}>clear</button>`
          : html`<span class="muted">Tap a hydrogen, or a heavy atom for all its hydrogens. None: every hydrogen not transferred, closest to the reaction first (up to ${maxSites}).</span>`}
      </div>
    </div>
    <div class="field"><span class="field-label">Groups<${Why} ids=${['param.substituents.groups']} label="Groups" /></span>
      <div class="chips" role="group">
        ${shownGroups.map((g) => html`<button type="button" aria-pressed=${groups.includes(g)} class=${`chip ${groups.includes(g) ? 'on' : ''}`}
          title=${labels[g] ?? g} onClick=${() => toggleGroup(g)}>${short(g)}</button>`)}
        <button type="button" class="btn-link small" onClick=${() => setMore(!more)}>${more ? 'Fewer' : `More groups (${all.length - COMMON.length})`}</button>
      </div>
    </div>
    <p class="small subst-count">${nSites ? `${nSites} site${nSites > 1 ? 's' : ''}` : `Up to ${maxSites} sites`} × ${groups.length} group${groups.length === 1 ? '' : 's'} = ${nSites ? nSites * groups.length : `up to ${maxSites * groups.length}`} variants
      ${toDesign && html`<span class="muted"> · or </span><a href="#" onClick=${(e) => { e.preventDefault();
        attempt(() => api.post('/api/design/load-edge', { edge: toDesign }), 'Reaction loaded into Design').then((d) => d && openTab('design')); }}
        title="Load this reaction into Design and swap one group by hand (Swap group tool), then recompute it">change one by hand in Design →</a>`}</p>
    <${ParamForm} op=${op.key} schema=${rest} values=${values} onChange=${onChange} />
  </div>`;
}
