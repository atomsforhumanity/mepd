// A reacting mixture (`mepd discovery mixture`) as it stands, at #/network/<job>: a network of
// molecules, not of one system's states. Every combination of the feed molecules (copies of one
// included: A + A) and, generation by generation, every populated product next to each feed molecule
// was an explorer run; their steps are merged into reactions between molecules with mass-action
// kinetics from the feed's concentrations. Drawn as the live network is (LiveScheme.js): the feed at
// the left, each molecule out from the molecule it is made from, a reaction an arrow from its main
// reactant to each product with its co-reactant written on it, amounts in mol per mol of feed, and the
// seeds' reactor events. Before the first merge, the feed alone (from the job's command), the seeds'
// reactor events playing on it. Experimental: the mixture explorer is not yet benchmarked.
import { html, useEffect, useState } from '../lib.js';
import { api } from '../api.js';
import { openJob, useStore } from '../store.js';
import { depictUrl } from '../util.js';
import { MixtureScheme } from './LiveScheme.js';

function fmtC(x) {
  if (x == null) return '—';
  if (x === 0) return '0';
  return Math.abs(x) >= 0.01 && Math.abs(x) < 1000 ? x.toPrecision(2) : x.toExponential(1);
}

export function MixtureView({ jobId }) {
  const job = useStore((s) => s.jobs[jobId]);
  const running = ['running', 'queued'].includes(job?.status);
  const [d, setD] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true;
    const load = () => api.get(`/api/jobs/${jobId}/mixture`).then((x) => { if (alive) { setD(x); setError(null); } })
      .catch((e) => alive && setError(e.message));
    load();
    const t = running ? setInterval(load, 5000) : null;
    return () => { alive = false; if (t) clearInterval(t); };
  }, [jobId, running]);

  if (error) return html`<div class="ln-empty"><p>${error}</p></div>`;
  if (!d) return html`<p class="boot">Loading the mixture…</p>`;
  // Before the first merge: the feed alone, as a mixture that has not reacted yet ("pending").
  const pre = !(d.mixture?.molecules || []).length && (d.feed || []).length ? {
    feed: d.feed, settings: d.settings || {}, seeds: d.seeds || [], reactions: [], notes: [], generation: null, pending: true,
    molecules: d.feed.map((f) => ({ molecule: f.molecule, max_concentration: f.concentration, final_concentration: f.concentration, feed: true })) } : null;
  const m = (d.mixture?.molecules || []).length ? d.mixture : pre;
  // The feed's own names (as given to the mixture), else everyday names, else the SMILES.
  const names = { ...(d.names || {}), ...Object.fromEntries(((m && m.feed) || []).filter((f) => f.name && f.name !== f.molecule).map((f) => [f.molecule, f.name])) };
  const nameOf = (k) => names[k] || k.split('|')[0];   // "A|B": forms counted as one species
  const unit = m?.settings?.standard_state === '1atm' ? 'atm' : 'M';
  const seeds = m?.seeds || d.seeds || [];
  const doneSeeds = seeds.filter((s) => String(s.status || '').startsWith('done')).length;
  const doors = html`<div class="ln-doors">
    <button class="btn small" onClick=${() => openJob(jobId, 'live')}>Seeds' progress</button>
    <button class="btn small" onClick=${() => openJob(jobId, 'result')} disabled=${job?.status === 'queued'}>Results</button>
    <button class="btn small" onClick=${() => openJob(jobId, 'log')}>Log</button></div>`;
  const head = html`<div class="ln-head"><h2>${job?.title?.replace(/^Reacting mixture[^:]*:\s*/, 'Mixture: ') || 'Reacting mixture'}</h2>
    <span class="ln-exp" title="The mixture explorer is not yet benchmarked against experiment">experimental</span>
    ${running && html`<span class="ln-live">● live</span>`}<span class="spacer"></span>
    ${m && html`<span class="small muted">${m.settings?.temperature_K} K · ${m.settings?.time_s} s · ${m.pending ? 'nothing merged yet' : `generation ${m.generation ?? 0}`}</span>`}</div>`;
  if (!m || !(m.molecules || []).length) {
    return html`<div class="ld-page">${head}
      <div class="ln-stage ln-waiting"><div class="ln-seed-pulse"></div>
        <p>${running ? 'The first seeds are running; the merged network appears after the first one finishes.' : 'This mixture has no merged network.'}</p>
        <p class="small muted">${seeds.length} seed${seeds.length === 1 ? '' : 's'} started · ${doneSeeds} done</p></div>
      <div class="ld-details"><section class="ln-closer"><span class="section-title">Look closer</span>${doors}</section></div></div>`;
  }
  const extra = html`<div class="ld-details">
    <section class="ln-closer"><span class="section-title">Feed</span>
      ${(m.feed || []).map((f) => html`<div class="ln-prod" title=${f.molecule}>
        <img class="ln-prod-mol" src=${depictUrl(f.molecule, 160, 120)} alt=${f.molecule} />
        <span class="ln-prod-name">${nameOf(f.molecule)}</span><b>${fmtC(f.concentration)} ${unit}</b></div>`)}
      <span class="small muted">${seeds.length} seed${seeds.length === 1 ? '' : 's'} (${doneSeeds} done)${m.pending ? '' : `, generation ${m.generation ?? 0}`}: each seed is one
        combination of molecules explored on its own; its steps join this network${m.pending ? ' when it finishes (none has yet)' : ''}.
        Amounts are in mol per mol of feed.</span>
      <span class="section-title">Look closer</span>${doors}</section>
    ${(m.notes || []).length > 0 && html`<section class="ln-closer"><span class="section-title">Notes</span>
      <ul class="ln-feed">${m.notes.slice(0, 4).map((n) => html`<li class="muted">${n}</li>`)}</ul></section>`}
  </div>`;
  return html`<div class="ld-mixture">${head}<${MixtureScheme} mixture=${m} names=${names} labels=${d.labels || {}} jobId=${jobId} job=${job} running=${running} extra=${extra} /></div>`;
}
