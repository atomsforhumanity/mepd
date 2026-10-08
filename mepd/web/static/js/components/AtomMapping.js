// What the start/end atom mapping (SLAPMapper) did: the mapping used and
// the others it was compared with, each as a numbered reactant/product
// drawing (bonds that break / form highlighted) and an interpolation from
// the start to the end under that mapping. Data: result.atom_mapping
// (mepd.mapping_report via mepd.web.results.atom_mapping_view).
import { html, useEffect, useState } from '../lib.js';
import { Viewer3D } from './Viewer3D.js';

const ABOUT = {
  current: "the end's atoms as they were given",
  slapmapper: 'a mapping with the fewest bond changes, from SLAPMapper',
  symmetric: 'the same bond changes, with equivalent atoms (e.g. the H of a methyl) swapped',
  snap: "SLAPMapper's mapping, with its equivalent atoms placed to best match the two geometries",
  explored: 'a mapping with more bond changes: another mechanism, explored on purpose',
};

const KIND = {
  current: 'current numbering',
  slapmapper: 'SLAPMapper mapping',
  symmetric: 'symmetric relabeling',
  snap: 'snap placement of symmetric atoms',
  explored: 'explored mechanism',
};

function svgUrl(svg) {
  return svg ? `data:image/svg+xml;utf8,${encodeURIComponent(svg)}` : null;
}

function bondText(symbols, pairs) {
  return pairs.map(([i, j]) => `${symbols[i]}${i + 1}–${symbols[j]}${j + 1}`).join(', ');
}

function title(c, i) {
  if (c.kind === 'current') return 'Current numbering';
  const n = c.label.match(/mapping_(\d+)/);
  return `${KIND[c.kind] ? KIND[c.kind][0].toUpperCase() + KIND[c.kind].slice(1) : 'Mapping'} ${n ? +n[1] + 1 : i + 1}`;
}

export function AtomMappingPanel({ mapping }) {
  const cands = mapping.candidates || [];
  const [pick, setPick] = useState(0);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const c = cands[Math.min(pick, cands.length - 1)];
  const frames = c?.frames || [];
  useEffect(() => { setFrame(0); setPlaying(false); }, [pick]);
  useEffect(() => {
    if (!playing || frames.length < 2) return undefined;
    const t = setInterval(() => setFrame((i) => {
      if (i >= frames.length - 1) { setPlaying(false); return i; }
      return i + 1;
    }), 220);
    return () => clearInterval(t);
  }, [playing, frames.length]);
  if (!c) return null;
  const symbols = mapping.symbols || [];
  const several = cands.length > 1;
  return html`
    <section class="card-block atom-mapping">
      <h3 class="section-title">Atom mapping</h3>
      <p class="small">${mapping.summary}${mapping.n_candidates > cands.length
        ? ` ${cands.length} of the ${mapping.n_candidates} compared are shown (the chosen one, the current numbering, the best-scoring others).` : ''}</p>
      ${several && html`<div class="mapping-chips">
        ${cands.map((x, i) => html`<button type="button" key=${x.label} class=${`chip ${i === pick ? 'on' : ''}`}
            title=${`${KIND[x.kind] || x.kind}${x.score != null ? ` · score ${x.score.toFixed(4)}` : ''}`}
            onClick=${() => setPick(i)}>
          ${x.chosen ? '★ ' : ''}${title(x, i)}
          <span class="chip-detail"> · ${x.broken.length + x.formed.length} bond change${x.broken.length + x.formed.length === 1 ? '' : 's'}${x.score != null ? ` · ${x.score.toFixed(3)}` : ''}</span>
        </button>`)}
      </div>`}
      <div class="mapping-head small">
        <b>${title(c, pick)}</b>${c.chosen ? html` <span class="badge accent">used for the path</span>` : ''}
        <span class="muted"> · ${ABOUT[c.kind] || c.kind}${c.score != null ? ` · ${mapping.quantity || mapping.metric}: ${c.score.toFixed(4)}` : ''}</span>
      </div>
      ${!c.chosen && (c.differs_at || []).length > 0 && html`<p class="small">${`Differs from the mapping used at ${c.differs_at.map((k) => `${symbols[k]}${k + 1}`).join(', ')}: those start atoms end up as other atoms of the same element.`}</p>`}
      <div class="mapping-pair">
        <figure>
          ${c.svg?.start ? html`<img src=${svgUrl(c.svg.start)} alt="start, numbered" />` : html`<p class="small muted">(no drawing)</p>`}
          <figcaption class="small">Start${c.broken.length ? html`: <span class="map-broken">breaks ${bondText(symbols, c.broken)}</span>` : ''}</figcaption>
        </figure>
        <span class="mapping-arrow">→</span>
        <figure>
          ${c.svg?.end ? html`<img src=${svgUrl(c.svg.end)} alt="end, numbered as the start" />` : html`<p class="small muted">(no drawing)</p>`}
          <figcaption class="small">End, numbered as the start${c.formed.length ? html`: <span class="map-formed">forms ${bondText(symbols, c.formed)}</span>` : ''}</figcaption>
        </figure>
      </div>
      ${!c.broken.length && !c.formed.length && html`<p class="small muted">No bond breaks or forms under this mapping (a conformational change).</p>`}
      ${frames.length > 1 && html`
        <div class="mapping-interp">
          <div class="live-head"><span class="small"><b>Interpolation</b> under this mapping: ${c.interpolation} · frame ${frame + 1} of ${frames.length}</span>
            <button class="btn small" onClick=${() => { if (!playing && frame >= frames.length - 1) setFrame(0); setPlaying(!playing); }}>${playing ? 'Pause' : 'Play'}</button></div>
          <${Viewer3D} frames=${frames} frame=${frame} height=${240} labels=${true} labelBase=${1} />
          <div class="frame-bar"><input type="range" min="0" max=${frames.length - 1} value=${frame}
            onInput=${(e) => { setPlaying(false); setFrame(+e.target.value); }} /></div>
        </div>`}
      ${(mapping.notes || []).length > 0 && html`<details class="small"><summary>How the mapping was chosen</summary>
        <ul>${mapping.notes.map((n) => html`<li class="mono">${n}</li>`)}</ul></details>`}
      <p class="small muted">Atoms are numbered by the start's order; each mapping renumbers the end so atom k is the
        start's atom k. Hydrogens are drawn only where a bond to them changes.</p>
    </section>`;
}
