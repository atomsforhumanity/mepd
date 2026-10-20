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
// "Fragments" instead shows each molecule's own orbitals (run alone where it
// sits; POST /api/fragment-orbitals): one's HOMO and another's LUMO drawn
// together (the acceptor's in paler colours, its sign chosen so that same
// colours meeting is in-phase overlap), with |<HOMO|LUMO>|, the gap, and on
// a path |S| frame by frame.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { prefs, set } from '../store.js';
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
  system = null, region: regionIn = null, onAtomClick = null, highlight = null, labelBase = 0, orbitals = null,
  surfaces: surfaceList = null }) {
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
  const moIndex = frames && frames.length ? Math.min(frame, frames.length - 1) : 0;
  const mo = useOrbitals(orbitals && !qmmm ? orbitals : null, source, moIndex);
  const surfaces = useRef([]);
  const surfaces2 = useRef([]);

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
    // Orbitals given by the caller (the orbital explorer) instead of the bar's.
    const given = surfaceList || [];
    drawOrbital(viewer.current, surfaces, given.length ? given[0]?.orb || null : mo.current,
      given.length ? given[0]?.iso : mo.iso, given.length ? given[0]?.sign ?? 1 : mo.sign, given[0]?.colors);
    drawOrbital(viewer.current, surfaces2, given[1]?.orb || null, given[1]?.iso, given[1]?.sign ?? 1, given[1]?.colors || PALE);
    if (host.current) {   // for tests
      host.current._orbital = { surfaces: surfaces.current.length, fresh: mo.fresh, sign: mo.sign, orb: mo.current };
      host.current._given = { surfaces: surfaces.current.length + surfaces2.current.length, count: given.filter(Boolean).length };
    }
  }, [mo.current, mo.iso, mo.sign, surfaceList, xyz, frames, frame]);

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
    ${orbitals && !qmmm && source && html`<${OrbitalBar} mo=${mo} labelBase=${labelBase || 1}
      onExplore=${() => set({ modal: { kind: 'orbitals', frames: Array.isArray(source) ? source : [source], frame: moIndex,
        chem: { charge: orbitals.charge, multiplicity: orbitals.multiplicity, job: orbitals.job }, title: orbitals.name || '' } })} />`}
  </div>`;
}

// ---- orbitals ---------------------------------------------------------------
const DENSITY_PCT = 85;
const LOADERS = 4;        // a path's frames: this many orbital requests at once   // the surface encloses this share of the orbital's density (|psi|^2), by default

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
// `sign` -1: the orbital is drawn as -psi (its colours swapped), to keep a path's phase (orbitalSigns).
const STRONG = ['#1f4fd1', '#d42a2a'];
export const PALE = ['#8fb0f5', '#f2a0a0'];     // a second orbital drawn with the first (a fragment's LUMO)
export function drawOrbital(v, surfaces, orb, iso, sign = 1, colors = STRONG) {
  (surfaces.current || []).forEach((sh) => { try { v.removeShape(sh); } catch { /* gone with the model */ } });
  surfaces.current = [];
  if (orb?.cube) {
    const look = { wireframe: true, linewidth: 1, opacity: 0.9, smoothness: 1 };
    const [pos, neg] = sign < 0 ? [colors[1], colors[0]] : colors;
    surfaces.current = [
      v.addVolumetricData(orb.cube, 'cube', { ...look, isoval: iso, color: pos }),
      v.addVolumetricData(orb.cube, 'cube', { ...look, isoval: -iso, color: neg }),
    ];
  }
  v.render();
}

// ---- phase along a path ------------------------------------------------------
// An orbital's sign is arbitrary: each SCF may give +psi or -psi, so the
// lobes' colours could swap from frame to frame. Consecutive frames are made
// to agree: the sign that makes their overlap positive, sampled around every
// atom in both grids, carried along the path from its first frame.
function gridOf(orb) {
  if (!orb._grid) {
    const lines = orb.cube.split('\n');
    const natoms = Math.abs(parseInt(lines[2], 10));
    const head = (i) => lines[i].trim().split(/\s+/).map(Number);
    const o = head(2).slice(1, 4);
    const n = [head(3)[0], head(4)[0], head(5)[0]];
    const h = [head(3)[1], head(4)[2], head(5)[3]];
    const atoms = lines.slice(6, 6 + natoms).map((l) => l.trim().split(/\s+/).slice(2, 5).map(Number));
    const values = Float32Array.from(lines.slice(6 + natoms).join(' ').trim().split(/\s+/), Number);
    orb._grid = { o, n, h, atoms, values };
  }
  return orb._grid;
}

// psi at point p (bohr), trilinear between grid points; 0 outside the grid.
function sampleAt(g, p) {
  const f = [0, 1, 2].map((i) => (p[i] - g.o[i]) / g.h[i]);
  const i0 = f.map(Math.floor);
  if (i0.some((x, i) => x < 0 || x >= g.n[i] - 1)) return 0;
  const t = f.map((x, i) => x - i0[i]);
  const at = (a, b, c) => g.values[((i0[0] + a) * g.n[1] + (i0[1] + b)) * g.n[2] + (i0[2] + c)];
  let v = 0;
  for (const a of [0, 1]) for (const b of [0, 1]) for (const c of [0, 1]) {
    v += at(a, b, c) * (a ? t[0] : 1 - t[0]) * (b ? t[1] : 1 - t[1]) * (c ? t[2] : 1 - t[2]);
  }
  return v;
}

const PROBES = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].map((d) => d.map((x) => x * 0.8));

// The overlap's sign of two frames' orbitals (+1 / -1), at points around each atom of the second.
export function phaseBetween(a, b) {
  const ga = gridOf(a), gb = gridOf(b);
  let s = 0;
  for (const atom of gb.atoms) {
    for (const d of PROBES) {
      const p = [atom[0] + d[0], atom[1] + d[1], atom[2] + d[2]];
      s += sampleAt(ga, p) * sampleAt(gb, p);
    }
  }
  return s < 0 ? -1 : 1;
}

// The orbitals of the shown geometry, fetched as it changes (one request at a time).
export function useOrbitals(chem, source, index = 0) {
  // Showing them is a choice of what to draw, remembered for every viewer; the data comes with the geometry.
  const [on, setOnState] = useState(() => prefs.get('orbitalsOn', false));
  const setOn = (v) => { setOnState(v); prefs.set('orbitalsOn', v); };
  const [which, setWhich] = useState('homo');
  const [pct, setPctState] = useState(() => prefs.get('orbitalPct', DENSITY_PCT));
  const setPct = (v) => { setPctState(v); prefs.set('orbitalPct', v); };
  const [data, setData] = useState(null);        // the newest {xyz, result}
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [tick, setTick] = useState(0);           // a frame's orbitals arrived
  const cache = useRef(new Map());               // xyz -> result: a path's frames, for playing it back
  const failed = useRef(new Set());
  const inFlight = useRef(new Set());            // up to LOADERS frames load at once
  const geom = source ? (Array.isArray(source) ? source[index] : source) : null;
  const path = Array.isArray(source) && source.length > 1;
  // A single geometry: loaded with it (the session keeps every structure's orbitals ready). A path,
  // while orbitals are shown: the frame on screen first, then every other frame in playing order, so
  // playing it back draws each frame's own (a running calculation's: only the frame shown, as it moves).
  const needed = (f) => f && !cache.current.has(f) && !failed.current.has(f) && !inFlight.current.has(f);
  const targets = [];
  if (needed(geom) && (!path || on)) targets.push(geom);
  if (path && on && !chem?.live) {
    for (let k = 1; k < source.length && targets.length < LOADERS; k += 1) {
      const f = source[(index + k) % source.length];
      if (needed(f) && !targets.includes(f)) targets.push(f);
    }
  }
  useEffect(() => {
    if (!chem) return;
    for (const xyz of targets.slice(0, Math.max(0, LOADERS - inFlight.current.size))) {
      inFlight.current.add(xyz);
      setBusy(true);
      api.post('/api/orbitals', { xyz, charge: chem.charge ?? 0, multiplicity: chem.multiplicity ?? 1,
        job: chem.job || undefined, follow: (chem.live || path) || undefined })
        .then((result) => {
          cache.current.set(xyz, result);
          if (cache.current.size > 400) cache.current.delete(cache.current.keys().next().value);
          setData((d) => (xyz === geom || !d ? { xyz, result } : d));
          setError(null);
        })
        .catch((e) => { failed.current.add(xyz); if (xyz === geom) setError({ xyz, message: String(e.message || e) }); })
        .finally(() => { inFlight.current.delete(xyz); setBusy(inFlight.current.size > 0); setTick((t) => t + 1); });
    }
  }, [targets.join('|'), tick]);
  const own = geom ? cache.current.get(geom) : null;
  const fresh = !!own;
  const result = own || data?.result || null;
  // On a path that moves on while a frame's are fetched (a running chain), the newest stay drawn
  // until the next frame's arrive; a single geometry shows only its own.
  const sameAtoms = !!result && !!geom && (fresh || parseInt(data?.xyz, 10) === parseInt(geom, 10));
  const shown = on && result && (fresh || (path && sameAtoms)) ? result.orbitals[which] || null : null;
  // Its sign, so that a path's phase does not flip from frame to frame: a finished path's from its first
  // frame on (each frame against the one before it that has orbitals); a running one's against the last drawn.
  const last = useRef(null);
  let sign = 1;
  if (shown && path && !chem?.live) {
    let prev = null;
    for (let k = 0; k <= index; k += 1) {
      const orb = cache.current.get(source[k])?.orbitals[which];
      if (!orb) continue;
      if (prev && parseInt(source[k], 10) === parseInt(prev.xyz, 10)) {
        prev.orb._phase ||= new Map();
        if (!prev.orb._phase.has(orb)) prev.orb._phase.set(orb, phaseBetween(prev.orb, orb));
        sign *= prev.orb._phase.get(orb);
      }
      prev = { orb, xyz: source[k] };
    }
  } else if (shown && path) {
    const ref = last.current;
    if (ref && ref.orb === shown) sign = ref.sign;
    else if (ref && ref.which === which && sameAtoms) sign = ref.sign * phaseBetween(ref.orb, shown);
  }
  if (shown) last.current = { orb: shown, sign, which };
  return { on, setOn, which, setWhich, pct, setPct, iso: isoFor(shown, pct), sign, busy: busy && on, error: on ? error?.message : null,
    result: on ? result : null, fresh, current: shown,
    symbols: geom ? geom.trim().split('\n').slice(2).map((l) => l.trim().split(/\s+/)[0]) : [] };
}

// Under a viewer: switch orbitals on, pick HOMO or LUMO, and "Explore…" for everything else
// (the orbital explorer: each molecule's orbitals, the interaction and correlation diagrams).
export function OrbitalBar({ mo, labelBase = 1, onExplore = null }) {
  const r = mo.result;
  const open = r?.open_shell;
  const name = (k) => (k === 'homo' && open ? 'SOMO' : k.toUpperCase());
  const ev = (k) => r?.orbitals[k]?.energy_ev;
  const fmt = (x) => (x == null ? '–' : `${x > 0 ? '+' : ''}${x.toFixed(2)}`.replace('-', '−'));
  // Where the shown orbital sits: atoms with at least a tenth of it, largest first.
  const sites = mo.current ? mo.current.atoms.map((w, i) => [w, i]).filter(([w]) => w >= 0.1).sort((a, b) => b[0] - a[0]).slice(0, 4) : [];
  const method = r?.label || (r?.method === 'gfn2' ? 'GFN2-xTB' : 'g-xTB');
  const about = r ? [`${method} orbital energies (eV)${r.method === 'terachem' ? '' : ', from its minimal basis'}`,
    r.source === 'run' ? 'From the calculation itself (its SCF at this geometry)' : 'From a single point on the geometry shown',
    sites.length ? `${name(mo.which)} sits on ${sites.map(([w, i]) => `${mo.symbols[i]}${i + labelBase} ${Math.round(w * 100)}%`).join(', ')}` : '',
    r.gap_ev != null ? `Gap ${r.gap_ev.toFixed(2)} eV` : ''].filter(Boolean).join('\n') : '';
  return html`<div class="mo-bar small">
    <button class=${`btn small ${mo.on ? 'on' : ''}`} onClick=${() => mo.setOn(!mo.on)} aria-pressed=${mo.on}
      title="Show the highest occupied and lowest unoccupied molecular orbitals: where it gives and takes electrons (remembered for every view)">
      ${mo.busy ? html`<span class="spinner" aria-hidden="true"></span>` : ''}Orbitals</button>
    ${mo.on && r && html`
      <span class="segmented mini">${['homo', 'lumo'].filter((k) => r.orbitals[k]).map((k) => html`<button class=${mo.which === k ? 'on' : ''}
        onClick=${() => mo.setWhich(k)}>${name(k)}</button>`)}</span>
      <span class="mo-numbers" title=${about}>${fmt(ev(mo.which))} eV</span>`}
    ${mo.on && onExplore && html`<button class="btn small" onClick=${onExplore}
      title="Every orbital, each molecule's own orbitals and how they combine, and how they change along a path">Explore…</button>`}
    ${mo.error && html`<span class="warn-text" title=${mo.error}>no orbitals</span>`}
    ${mo.on && html`<a class="mo-help" href="/manual/orbitals#whole-orbitals" target="_blank" rel="noopener"
      title="How these orbitals are computed (manual)">?</a>`}
  </div>`;
}
