// A channels run with built endpoints (--pair-from): every built pair, ranked
// by its apparent barrier (the xtb peak along its interpolation, above the
// lowest start among them). The lowest were searched; any other can be
// searched from here (a Sample more paths run for that entry), and each
// one's interpolation viewed. Data: result.built_pairs
// (mepd.web.results.built_pairs_view), paths from built_pairs/paths/<label>.json.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { fmtKcal } from '../util.js';
import { EnergyPlot } from './EnergyPlot.js';
import { Viewer3D } from './Viewer3D.js';

const FROM = { start: 'reactant', end: 'product' };

// One entry's interpolation: 3D frames and its xtb profile above the floor.
function BuiltPath({ job, entry, floor }) {
  const [path, setPath] = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    setPath(null); setFrame(0); setPlaying(false);
    api.get(`/api/jobs/${job.id}/files/built_pairs/paths/${entry.label}.json`).then(setPath, () => setPath(false));
  }, [entry.label]);
  const frames = path?.frames || [];
  useEffect(() => {
    if (!playing || frames.length < 2) return undefined;
    const t = setInterval(() => setFrame((i) => {
      if (i >= frames.length - 1) { setPlaying(false); return i; }
      return i + 1;
    }), 250);
    return () => clearInterval(t);
  }, [playing, frames.length]);
  if (path === false) return html`<p class="small muted">Could not read this interpolation.</p>`;
  if (!path) return html`<p class="small muted">Reading…</p>`;
  const ys = (path.energies_kcal || []).map((e) => (floor != null ? e - floor : null));
  return html`
    <div class="mapping-interp">
      <div class="live-head"><span class="small"><b>#${entry.rank + 1}</b> ${entry.mechanism_text} · interpolation frame ${frame + 1} of ${frames.length}</span>
        <button class="btn small" onClick=${() => { if (!playing && frame >= frames.length - 1) setFrame(0); setPlaying(!playing); }}>${playing ? 'Pause' : 'Play'}</button></div>
      <${Viewer3D} frames=${frames} frame=${frame} height=${260} />
      <div class="frame-bar"><input type="range" min="0" max=${frames.length - 1} value=${frame}
        onInput=${(e) => { setPlaying(false); setFrame(+e.target.value); }} /></div>
      ${ys.some((y) => y != null) && html`<${EnergyPlot} ys=${ys} current=${frame} onPick=${(i) => { setPlaying(false); setFrame(i); }}
        yLabel="kcal/mol above floor" height=${140} />`}
    </div>`;
}

export function BuiltPairsPanel({ job, data }) {
  const [busy, setBusy] = useState(null);
  const [shown, setShown] = useState(null);
  const entries = data.entries || [];
  const n = entries.filter((e) => e.searched).length;
  const canSearch = job.op === 'channels' && job.status === 'done';
  const search = async (label) => {
    setBusy(label);
    await attempt(() => api.post('/api/jobs', { op: 'channels-more', params: { search_built: label }, source_job: job.id }),
      (js) => `Queued: ${js[0].title}`);
    setBusy(null);
  };
  const barrier = (e) => (e.barrier_kcal != null ? fmtKcal(e.barrier_kcal)
    : html`<span title=${`${data.metric}: ${e.score}`}>—</span>`);
  return html`
    <details class="built-pairs">
      <summary>Built pairs: ${n} of ${entries.length} searched</summary>
      <p class="small muted">Ranked by apparent barrier (xtb peak along the interpolation, above the lowest start). Click a mechanism to see its interpolation.</p>
      <div class="table-scroll"><table class="data">
        <thead><tr><th>#</th><th>Mechanism</th><th>Built from</th><th title="kcal/mol">Barrier</th><th></th></tr></thead>
        <tbody>${entries.map((e) => html`<tr>
          <td class="mono" title=${e.label}>${e.rank + 1}</td>
          <td>${e.has_path ? html`<a href="#" title="Show the interpolation" class=${shown === e.label ? 'active' : ''}
              onClick=${(ev) => { ev.preventDefault(); setShown(shown === e.label ? null : e.label); }}>${e.mechanism_text}</a>`
            : e.mechanism_text}</td>
          <td>${FROM[e.built_from] || e.built_from} ${e.conformer}</td>
          <td class="num">${barrier(e)}</td>
          <td>${e.searched ? html`<span class="small muted">searched</span>`
            : canSearch && html`<button class="btn small" disabled=${busy != null}
                onClick=${() => search(e.label)}>${busy === e.label ? 'Queuing…' : 'Search'}</button>`}</td>
        </tr>`)}</tbody>
      </table></div>
      ${shown && html`<${BuiltPath} job=${job} entry=${entries.find((e) => e.label === shown)} floor=${data.floor_kcal} />`}
    </details>`;
}
