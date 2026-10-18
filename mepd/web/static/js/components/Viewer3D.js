// 3Dmol.js wrapper.
//
// Two modes: a single `xyz`, or a path as `frames` (xyz strings) + `frame`
// index. A path is parsed into 3Dmol once (addModelsAsFrames); scrubbing
// then only calls setFrame, instead of re-parsing and re-styling the
// molecule on every step. Swapping between molecules with the same atom
// count keeps the camera.
//
// QM/MM: with `system` (a QM/MM system id) or `region` (a region object),
// a structure of that system is drawn by region (see ../qmmm.js), with a
// small bar to show all / only the moving / only the QM atoms, and to colour
// the environment by how far it moved from the first frame.
// `onAtomClick(index)` makes atoms clickable (the region editor);
// `highlight` marks atoms.
//
// Orbitals: with `orbitals` ({charge, multiplicity, job, live}), the shown
// geometry's HOMO and LUMO come with it (POST /api/orbitals: the session keeps
// every structure's ready; a calculation's frames come from its own SCFs), and
// the "Orbitals" button only chooses to draw one, with their energies and the
// atoms each sits on. A path shows the orbitals of the frame on screen,
// a running calculation's as it moves.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { prefs } from '../store.js';
import { drawLinks, ENV_MODES, parseXyz, styleRegion, useAutoSystem, useRegion } from '../qmmm.js';

function themeBackground() {
  return getComputedStyle(document.documentElement).getPropertyValue('--viewer-bg').trim() || '#ffffff';
}

const STYLES = {
  stick: { stick: { radius: 0.13 }, sphere: { scale: 0.24 } },
  sphere: { sphere: { scale: 0.32 }, stick: { radius: 0.12 } },
};

function atomCount(xyz) {
  return parseInt(xyz.trimStart().split('\n', 1)[0], 10) || 0;
}

export function Viewer3D({ xyz = null, frames = null, frame = 0, height = 260, labels = false, style = 'stick',
  system = null, region: regionIn = null, onAtomClick = null, highlight = null, labelBase = 0, orbitals = null }) {
  const host = useRef(null);
  const viewer = useRef(null);
  const lastAtoms = useRef(0);
  const loaded = useRef(null);  // the frames array / xyz string currently in the viewer
  const [env, setEnv] = useState('all');
  const [motion, setMotion] = useState(false);
  const source = frames && frames.length ? frames : xyz;
  const first = source ? (Array.isArray(source) ? source[0] : source) : null;
  // No system given: recognize one by its atoms (live paths, trees, maps...).
  const auto = useAutoSystem(regionIn || system || onAtomClick ? null : first);
  const fetched = useRegion(regionIn ? null : (system || auto));
  const region = regionIn || fetched;
  const qmmm = !!(region && first && atomCount(first) === region.natoms);
  // Parsed coordinates, only when colouring by motion needs them.
  const coords = useMemo(() => (qmmm && motion && Array.isArray(source) ? source.map(parseXyz) : null), [qmmm, motion, source]);
  const clickRef = useRef(onAtomClick);
  clickRef.current = onAtomClick;
  const mo = useOrbitals(orbitals && !qmmm ? orbitals : null, source, frames && frames.length ? Math.min(frame, frames.length - 1) : 0);
  const surfaces = useRef([]);

  useEffect(() => {
    if (!window.$3Dmol || !host.current) return undefined;
    viewer.current = window.$3Dmol.createViewer(host.current, { backgroundColor: themeBackground(), antialias: true });
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onTheme = () => { viewer.current?.setBackgroundColor(themeBackground()); viewer.current?.render(); };
    mq.addEventListener('change', onTheme);
    const ro = new ResizeObserver(() => { viewer.current?.resize(); viewer.current?.render(); });
    ro.observe(host.current);
    return () => { mq.removeEventListener('change', onTheme); ro.disconnect(); viewer.current?.clear(); viewer.current = null; };
  }, []);

  const drawLabels = (v) => {
    v.removeAllLabels();
    if (!labels) return;
    const model = v.getModel();
    if (!model) return;
    model.selectedAtoms({}).forEach((a, i) => v.addLabel(String(i + labelBase), {
      position: { x: a.x, y: a.y, z: a.z }, fontSize: 10, backgroundOpacity: 0.55, inFront: true,
    }));
  };

  const shown = () => (frames && frames.length ? Math.min(frame, frames.length - 1) : 0);

  const applyStyle = (v) => {
    if (qmmm) {
      styleRegion(v, region, { env, motion, frame: shown(), coords, base: STYLES[style], highlight });
      drawLinks(v, env === 'qm' ? null : region);
    } else {
      v.setStyle({}, STYLES[style] || STYLES.stick);
      v.removeAllShapes();
      if (highlight && highlight.length) v.addStyle({ index: highlight }, { sphere: { scale: 0.42, color: '#3868b8', opacity: 0.55 } });
    }
    if (clickRef.current) {
      v.setClickable({}, true, (atom) => clickRef.current?.(atom.index));
    } else {
      v.setClickable({}, false);
    }
  };

  // (Re)load geometry only when the molecule/path itself changes.
  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    if (source === loaded.current) return;
    loaded.current = source;
    const view = v.getView();
    v.removeAllModels();
    v.removeAllShapes();
    if (!source) { v.removeAllLabels(); v.render(); return; }
    if (Array.isArray(source)) {
      v.addModelsAsFrames(source.map((s) => (s.endsWith('\n') ? s : `${s}\n`)).join(''), 'xyz');
      v.setFrame(shown());
    } else {
      v.addModel(source, 'xyz');
    }
    applyStyle(v);
    const n = atomCount(first);
    if (n === lastAtoms.current) v.setView(view);
    else if (qmmm) v.zoomTo({ index: region.qm_atoms }); else v.zoomTo();
    lastAtoms.current = n;
    drawLabels(v);
    v.render();
  }, [xyz, frames, style]);

  // Region, display mode or highlight changed: restyle in place.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !loaded.current) return;
    applyStyle(v);
    v.render();
  }, [region, env, motion, coords, highlight, !!onAtomClick]);

  // A region arriving after the geometry: frame the QM atoms once.
  useEffect(() => {
    const v = viewer.current;
    if (v && qmmm && loaded.current) { v.zoomTo({ index: region.qm_atoms }); v.render(); }
  }, [qmmm && region?.id]);

  // Scrubbing: just switch frames.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !frames || !frames.length || loaded.current !== frames) return;
    Promise.resolve(v.setFrame(shown())).then(() => {
      if (qmmm) {
        if (motion) styleRegion(v, region, { env, motion, frame: shown(), coords, base: STYLES[style], highlight });
        drawLinks(v, env === 'qm' ? null : region);
      }
      if (labels) drawLabels(v);
      v.render();
    });
  }, [frame, frames]);

  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    drawLabels(v);
    v.render();
  }, [labels]);

  // The chosen orbital, when it belongs to what is shown.
  // (Also after every reload of the geometry -- a live chain update clears all shapes.)
  useEffect(() => {
    if (!viewer.current) return;
    drawOrbital(viewer.current, surfaces, mo.current, mo.iso);
    if (host.current) host.current._orbital = { surfaces: surfaces.current.length, fresh: mo.fresh };   // for tests
  }, [mo.current, mo.iso, xyz, frames, frame]);

  if (!window.$3Dmol) {
    return html`<div class="viewer viewer-missing" style=${{ height }}>3D viewer unavailable (3Dmol.js did not load)</div>`;
  }
  return html`<div class="viewer-wrap">
    <div class="viewer" ref=${host} style=${{ height }}></div>
    ${qmmm && html`<div class="qmmm-bar small">
      <span class="qmmm-tag" title=${region.summary}>QM/MM</span>
      <span class="segmented mini">${ENV_MODES.map(([k, label]) => html`<button class=${env === k ? 'on' : ''}
        title=${k === 'all' ? 'QM region, moving and frozen environment' : k === 'active' ? 'Hide the frozen environment' : 'Only the QM atoms'}
        onClick=${() => setEnv(k)}>${label}</button>`)}</span>
      ${Array.isArray(source) && source.length > 1 && html`<label title="Colour the environment by how far it moved from the first frame: grey still, red 1 Å or more; a frozen atom that moved turns magenta">
        <input type="checkbox" checked=${motion} onChange=${(e) => setMotion(e.target.checked)} /> motion</label>`}
      <span class="qmmm-legend"><i class="qm"></i>QM <i class="link"></i>cut bond <i class="act"></i>moving <i class="frz"></i>frozen</span>
    </div>`}
    ${orbitals && !qmmm && source && html`<${OrbitalBar} mo=${mo} path=${Array.isArray(source) && source.length > 1} labelBase=${labelBase || 1} />`}
  </div>`;
}

// ---- orbitals ---------------------------------------------------------------
const DENSITY_PCT = 85;   // the surface encloses this share of the orbital's density (|psi|^2), by default

// The |psi| at which an orbital's surface encloses `pct` % of its density on
// the grid: the largest |psi|^2 values, summed until they reach that share.
// (Sorted once per orbital and kept on it.)
export function isoFor(orb, pct) {
  if (!orb?.cube) return 0.05;
  if (!orb._dens) {
    const lines = orb.cube.split('\n');
    const natoms = Math.abs(parseInt(lines[2], 10));
    const vals = lines.slice(6 + natoms).join(' ').trim().split(/\s+/).map(Number);
    const d = Float64Array.from(vals, (x) => x * x).sort().reverse();
    let total = 0;
    for (let i = 0; i < d.length; i += 1) total += d[i];
    orb._dens = { d, total };
  }
  const { d, total } = orb._dens;
  const want = Math.min(Math.max(pct, 1), 99.9) / 100 * total;
  let acc = 0;
  for (let i = 0; i < d.length; i += 1) { acc += d[i]; if (acc >= want) return Math.sqrt(d[i]); }
  return Math.sqrt(d[d.length - 1]);
}

// An orbital as two wireframe isosurfaces (+ lobe blue, − lobe red) in viewer
// `v`, replacing the ones in `surfaces` (a ref); none when `orb` is null.
export function drawOrbital(v, surfaces, orb, iso) {
  (surfaces.current || []).forEach((sh) => { try { v.removeShape(sh); } catch { /* gone with the model */ } });
  surfaces.current = [];
  if (orb?.cube) {
    const look = { wireframe: true, linewidth: 1, opacity: 0.9, smoothness: 1 };
    surfaces.current = [
      v.addVolumetricData(orb.cube, 'cube', { ...look, isoval: iso, color: '#1f4fd1' }),
      v.addVolumetricData(orb.cube, 'cube', { ...look, isoval: -iso, color: '#d42a2a' }),
    ];
  }
  v.render();
}

// The orbitals of the shown geometry, fetched as it changes (one request at a time).
export function useOrbitals(chem, source, index = 0) {
  // Showing them is a choice of what to draw, remembered for every viewer; the data comes with the geometry.
  const [on, setOnState] = useState(() => prefs.get('orbitalsOn', false));
  const setOn = (v) => { setOnState(v); prefs.set('orbitalsOn', v); };
  const [which, setWhich] = useState('homo');
  const [pct, setPctState] = useState(() => prefs.get('orbitalPct', DENSITY_PCT));
  const setPct = (v) => { setPctState(v); prefs.set('orbitalPct', v); };
  const [data, setData] = useState(null);        // {xyz, result}
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const inFlight = useRef(null);
  const geom = source ? (Array.isArray(source) ? source[index] : source) : null;
  const path = Array.isArray(source) && source.length > 1;
  const fresh = !!data && data.xyz === geom;
  // A single geometry: loaded with it (the session keeps every structure's orbitals ready). A path's
  // frames: while orbitals are shown (each frame's, as it is shown; a running calculation's, as it moves).
  const want = !!geom && !fresh && (!path || on) && error?.xyz !== geom;
  useEffect(() => {
    if (!chem || !want || inFlight.current) return;
    const xyz = geom;
    inFlight.current = xyz;
    setBusy(true);
    api.post('/api/orbitals', { xyz, charge: chem.charge ?? 0, multiplicity: chem.multiplicity ?? 1,
      job: chem.job || undefined, follow: chem.live || undefined })
      .then((result) => { setData({ xyz, result }); setError(null); })
      .catch((e) => setError({ xyz, message: String(e.message || e) }))
      .finally(() => { inFlight.current = null; setBusy(false); });
  }, [want, geom, busy]);
  const result = data?.result || null;
  // On a path that moves on while a frame's are fetched (a running chain), the newest stay drawn
  // until the next frame's arrive; a single geometry shows only its own.
  const sameAtoms = !!data && !!geom && parseInt(data.xyz, 10) === parseInt(geom, 10);
  const shown = on && result && (fresh || (path && sameAtoms)) ? result.orbitals[which] || null : null;
  return { on, setOn, which, setWhich, pct, setPct, iso: isoFor(shown, pct), busy: busy && on, error: on ? error?.message : null,
    result: on ? result : null, fresh, current: shown,
    symbols: geom ? geom.trim().split('\n').slice(2).map((l) => l.trim().split(/\s+/)[0]) : [] };
}

export function OrbitalBar({ mo, path = false, labelBase = 1 }) {
  const r = mo.result;
  const open = r?.open_shell;
  const name = (k) => (k === 'homo' && open ? 'SOMO' : k.toUpperCase());
  const ev = (k) => r?.orbitals[k]?.energy_ev;
  const fmt = (x) => (x == null ? '–' : `${x > 0 ? '+' : ''}${x.toFixed(2)}`.replace('-', '−'));
  // Where the shown orbital sits: atoms with at least a tenth of it, largest first.
  const sites = mo.current ? mo.current.atoms.map((w, i) => [w, i]).filter(([w]) => w >= 0.1).sort((a, b) => b[0] - a[0]).slice(0, 4) : [];
  const method = r?.label || (r?.method === 'gfn2' ? 'GFN2-xTB' : 'g-xTB');
  return html`<div class="mo-bar small">
    <button class=${`btn small ${mo.on ? 'on' : ''}`} onClick=${() => mo.setOn(!mo.on)} aria-pressed=${mo.on}
      title="Show the highest occupied and lowest unoccupied molecular orbitals: where it gives and takes electrons (remembered for every view)">
      ${mo.busy ? html`<span class="spinner" aria-hidden="true"></span>` : ''}Orbitals</button>
    ${mo.on && r && html`
      <span class="segmented mini">${['homo', 'lumo'].filter((k) => r.orbitals[k]).map((k) => html`<button class=${mo.which === k ? 'on' : ''}
        onClick=${() => mo.setWhich(k)} title=${`${name(k)}: ${fmt(ev(k))} eV`}>${name(k)}</button>`)}</span>
      <label class="mo-pct" title=${`The surface encloses this share of the orbital's density (|ψ|²); drawn at |ψ| = ${mo.iso.toFixed(3)}. Smaller: tighter lobes.`}>
        <input type="number" class="tiny" min="1" max="99.9" step="5" value=${mo.pct}
          onChange=${(e) => { const v = +e.target.value; if (v > 0 && v < 100) mo.setPct(v); }} /> % density</label>
      <span class="mo-numbers" title=${`${method} orbital energies (eV)${r.method === 'terachem' ? '' : ', from its minimal basis'}`}>${name('homo')} ${fmt(ev('homo'))} · LUMO ${fmt(ev('lumo'))}${r.gap_ev != null ? html` · gap <b>${r.gap_ev.toFixed(2)}</b>` : ''} eV</span>
      <span class="mo-method muted" title=${r.source === 'run' ? 'From the calculation itself: the SCF it ran at this geometry' : 'From a single point at this level on the geometry shown (no calculation had run this geometry)'}>${method}${r.source === 'run' ? ' · this run' : ''}</span>
      ${sites.length > 0 && html`<span class="mo-sites" title="How much of the shown orbital sits on each atom">on ${sites.map(([w, i]) => `${mo.symbols[i]}${i + labelBase} ${Math.round(w * 100)}%`).join(', ')}</span>`}
      `}
    ${mo.error && html`<span class="warn-text" title=${mo.error}>no orbitals</span>`}
  </div>`;
}
