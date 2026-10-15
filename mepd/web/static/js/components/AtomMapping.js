// What the start/end atom mapping (SLAPMapper) did: the mapping used and
// the others it was compared with, each as a numbered reactant/product
// drawing (bonds that break / form highlighted) and an interpolation from
// the start to the end under that mapping. Data: result.atom_mapping
// (mepd.mapping_report via mepd.web.results.atom_mapping_view).
import { html, useEffect, useState } from '../lib.js';
import { Viewer3D } from './Viewer3D.js';
import { Why } from './Why.js';

// Each candidate in plain words; what it is exactly is its tooltip (and the "?").
const KIND = {
  current: 'As given',
  slapmapper: 'Fewest bond changes',
  symmetric: 'Equivalent atoms swapped',
  snap: 'Best geometric fit',
  explored: 'Other mechanism',
};
const ABOUT = {
  current: "The end's atoms in the order they were given.",
  slapmapper: 'SLAPMapper: the mapping with the fewest bonds broken and formed.',
  symmetric: 'The same bond changes, with equivalent atoms (e.g. the hydrogens of a methyl) swapped.',
  snap: "SLAPMapper's mapping, with equivalent atoms placed where the two geometries match best.",
  explored: 'More bond changes than the fewest: another mechanism, tried on purpose.',
};

function svgUrl(svg) {
  return svg ? `data:image/svg+xml;utf8,${encodeURIComponent(svg)}` : null;
}

function bondText(symbols, pairs) {
  return pairs.map(([i, j]) => `${symbols[i]}${i + 1}–${symbols[j]}${j + 1}`).join(', ');
}

function title(c, i) {
  return KIND[c.kind] || `Mapping ${i + 1}`;
}
// How far apart the two ends are under a mapping (what the choice minimizes).
const fit = (c, mapping) => (c.score == null ? ''
  : /rmsd/i.test(mapping.quantity || '') ? `RMSD ${c.score.toFixed(2)}${/bohr/.test(mapping.quantity) ? ' bohr' : ''}` : `score ${c.score.toFixed(2)}`);

export function AtomMappingPanel({ mapping }) {
  const cands = mapping.candidates || [];
  const [pick, setPick] = useState(0);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  // Folded until asked for: the path is what the page is about.
  const [open, setOpen] = useState(false);
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
  const used = cands.find((x) => x.chosen) || c;
  const nChange = used.broken.length + used.formed.length;
  return html`
    <details class="card-block atom-mapping" open=${open} onToggle=${(e) => setOpen(e.currentTarget.open)}>
      <summary><span class="section-title">Atom mapping</span>
        <span class="small muted"> · ${nChange} bond change${nChange === 1 ? '' : 's'}${several ? ` · ${cands.length} compared` : ''}</span></summary>
      ${open && html`<p class="small" title=${mapping.summary}>${used.kind === 'current' ? "The end's atoms were used as given." : "The end's atoms were renumbered to match the start."}
        ${several ? ' ★ marks the one used.' : ''}<${Why} id="other.atom-mapping" label="Atom mapping" /></p>
      ${several && html`<div class="mapping-chips">
        ${cands.map((x, i) => html`<button type="button" key=${x.label} class=${`chip ${i === pick ? 'on' : ''}`}
            title=${`${ABOUT[x.kind] || x.kind}${x.score != null ? ` Ends: ${fit(x, mapping)}.` : ''}`}
            onClick=${() => setPick(i)}>
          ${x.chosen ? '★ ' : ''}${title(x, i)}
          <span class="chip-detail"> · ${x.broken.length + x.formed.length} bond change${x.broken.length + x.formed.length === 1 ? '' : 's'}</span>
        </button>`)}
      </div>`}
      <div class="mapping-head small">
        <b title=${ABOUT[c.kind] || ''}>${title(c, pick)}</b>${c.chosen ? html` <span class="badge accent">used</span>` : ''}
        ${c.score != null && html`<span class="muted" title=${`${mapping.quantity || mapping.metric}: ${c.score.toFixed(4)} (lower: the two ends line up better)`}> · ends ${fit(c, mapping)}</span>`}
      </div>
      ${!c.chosen && (c.differs_at || []).length > 0 && html`<p class="small">${`Differs from the one used at ${c.differs_at.map((k) => `${symbols[k]}${k + 1}`).join(', ')}.`}</p>`}
      <div class="mapping-pair">
        <figure>
          ${c.svg?.start ? html`<img src=${svgUrl(c.svg.start)} alt="start, numbered" />` : html`<p class="small muted">(no drawing)</p>`}
          <figcaption class="small">Start${c.broken.length ? html`: <span class="map-broken">breaks ${bondText(symbols, c.broken)}</span>` : ''}</figcaption>
        </figure>
        <span class="mapping-arrow">→</span>
        <figure>
          ${c.svg?.end ? html`<img src=${svgUrl(c.svg.end)} alt="end, numbered as the start" />` : html`<p class="small muted">(no drawing)</p>`}
          <figcaption class="small">End${c.formed.length ? html`: <span class="map-formed">forms ${bondText(symbols, c.formed)}</span>` : ''}</figcaption>
        </figure>
      </div>
      ${!c.broken.length && !c.formed.length && html`<p class="small muted">No bonds change (a conformational change).</p>`}
      ${frames.length > 1 && html`
        <div class="mapping-interp">
          <div class="live-head"><span class="small" title=${`Interpolated from start to end with this mapping (${c.interpolation})`}><b>Start → end</b> · frame ${frame + 1}/${frames.length}</span>
            <button class="btn small" onClick=${() => { if (!playing && frame >= frames.length - 1) setFrame(0); setPlaying(!playing); }}>${playing ? 'Pause' : 'Play'}</button></div>
          <${Viewer3D} frames=${frames} frame=${frame} height=${240} labels=${true} labelBase=${1} />
          <div class="frame-bar"><input type="range" min="0" max=${frames.length - 1} value=${frame}
            onInput=${(e) => { setPlaying(false); setFrame(+e.target.value); }} /></div>
        </div>`}
      ${(mapping.notes || []).length > 0 && html`<details class="small"><summary>How the mapping was chosen</summary>
        <ul>${mapping.notes.map((n) => html`<li class="mono">${n}</li>`)}</ul></details>`}
      <p class="small muted">Both drawings use the start's atom numbers.</p>`}
    </details>`;
}
