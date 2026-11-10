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
// "Bonds" shows the whole system's filled orbitals as bonds and lone pairs
// (intrinsic bond orbitals; POST /api/bond-orbitals). On a path every one is
// followed from frame to frame (POST /api/bond-orbitals/path): the few that
// change are the mechanism's curly arrows, drawn as the path of each one's
// charge centre, and the one chosen is drawn at every frame.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { prefs } from '../store.js';
import { drawLinks, ENV_MODES, parseXyz, styleRegion, useAutoSystem, useRegion } from '../qmmm.js';
import { Why } from './Why.js';

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
  // Touch screens: 3Dmol takes every touch on the view (to rotate), so a swipe that starts on it scrolled nothing,
  // and on a phone the view fills the middle of the page. A cover passes swipes to the page; a tap turns rotating on.
  const touch = useMemo(() => !!window.matchMedia?.('(pointer: coarse)').matches, []);
  const [rotating, setRotating] = useState(false);
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
  const fmo = useFragmentOrbitals(orbitals && !qmmm ? orbitals : null, source, moIndex, mo.on && mo.frag);
  const bmo = useBondOrbitals(orbitals && !qmmm ? orbitals : null, source, moIndex, mo.on && mo.bonds);
  const surfaces = useRef([]);
  const surfaces2 = useRef([]);
  const arrows = useRef([]);

  useEffect(() => {
    if (!window.$3Dmol || !host.current) return undefined;
    viewer.current = window.$3Dmol.createViewer(host.current, { backgroundColor: themeBackground(), antialias: true });
    host.current._v = viewer.current;   // (for tests)
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
    // `labels` an array: each atom's own number (e.g. a subsystem's atoms by their numbers in the whole system)
    model.selectedAtoms({}).forEach((a, i) => v.addLabel(String(Array.isArray(labels) ? labels[i] : i + labelBase), {
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
    if (qmmm) v.zoomTo({ index: region.qm_atoms }); else v.zoomTo();
    if (n === lastAtoms.current) {
      // The same molecule again (another of its structures or paths): keep the rotation and zoom you had,
      // but turn about these coordinates' centre -- the old view's centre was the previous structure's,
      // which sits elsewhere (an IRC and its TS, two species from different runs), so the molecule spun
      // about an empty point. getView: [centre x, y, z, zoom, rotation quaternion x, y, z, w].
      const c = v.getView();
      v.setView([c[0], c[1], c[2], view[3], view[4], view[5], view[6], view[7]]);
    }
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
    const frag = mo.on && mo.frag;
    const bonds = mo.on && mo.bonds;
    const [one, iso, sign] = frag ? [fmo.occ, isoFor(fmo.occ, mo.pct), fmo.sign]
      : bonds ? [bmo.current, isoFor(bmo.current, mo.pct), bmo.sign] : [mo.current, mo.iso, mo.sign];
    drawOrbital(viewer.current, surfaces, one, iso, sign);
    drawOrbital(viewer.current, surfaces2, frag ? fmo.virt : null, isoFor(fmo.virt, mo.pct), fmo.virtSign, PALE);
    drawArrows(viewer.current, arrows, bonds ? bmo.arrows : null, moIndex,
      (a) => arrowName(bmo.followed, a.id, mo.symbols, labelBase || 1, moIndex));
    if (host.current) {   // for tests
      host.current._orbital = { surfaces: surfaces.current.length, fresh: mo.fresh, sign: mo.sign, orb: mo.current };
      host.current._fragments = { surfaces: surfaces2.current.length + (frag ? surfaces.current.length : 0), pair: fmo.pair, fresh: fmo.fresh, sign: fmo.sign };
      host.current._bonds = { surfaces: bonds ? surfaces.current.length : 0, arrows: arrows.current.length, id: bmo.id, local: bmo.local, sign: bmo.sign, fresh: bmo.fresh,
        dots: bonds && bmo.arrows ? bmo.arrows.map((a) => a.points[Math.min(moIndex, a.points.length - 1)]) : [] };
    }
  }, [mo.current, mo.iso, mo.sign, mo.frag, mo.bonds, mo.on, mo.pct, fmo.occ, fmo.virt, fmo.sign, bmo.current, bmo.sign, bmo.arrows, xyz, frames, frame]);

  if (!window.$3Dmol) {
    return html`<div class="viewer viewer-missing" style=${{ height }}>3D viewer unavailable (3Dmol.js did not load)</div>`;
  }
  return html`<div class="viewer-wrap">
    <div class="viewer" ref=${host} style=${{ height }}></div>
    ${touch && !rotating && html`<button class="viewer-guard" style=${{ height }} onClick=${() => setRotating(true)}
      aria-label="Rotate the molecule (a swipe here scrolls the page)"><span>Tap to rotate</span></button>`}
    ${touch && rotating && html`<button class="btn small viewer-done" onClick=${() => setRotating(false)}
      title="Back to scrolling the page with a swipe here">Done</button>`}
    ${qmmm && html`<div class="qmmm-bar small">
      <span class="qmmm-tag" title=${region.summary}>QM/MM</span>
      <span class="segmented mini">${ENV_MODES.map(([k, label]) => html`<button class=${env === k ? 'on' : ''}
        title=${k === 'all' ? 'QM region, moving and frozen environment' : k === 'active' ? 'Hide the frozen environment' : 'Only the QM atoms'}
        onClick=${() => setEnv(k)}>${label}</button>`)}</span>
      ${Array.isArray(source) && source.length > 1 && html`<label title="Colour the environment by how far it moved from the first frame: grey still, red 1 Å or more; a frozen atom that moved turns magenta">
        <input type="checkbox" checked=${motion} onChange=${(e) => setMotion(e.target.checked)} /> motion</label>`}
      <span class="qmmm-legend"><i class="qm"></i>QM <i class="link"></i>cut bond <i class="act"></i>moving <i class="frz"></i>frozen</span>
    </div>`}
    ${orbitals && !qmmm && source && html`<${OrbitalBar} mo=${mo} fmo=${fmo} bmo=${bmo} index=${moIndex} path=${Array.isArray(source) && source.length > 1}
      nframes=${Array.isArray(source) ? source.length : 1} labelBase=${labelBase || 1} />`}
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
  // Which orbitals: the whole system's ('whole'), each molecule's own ('fragments'), or the whole
  // system's filled ones as bonds and lone pairs ('bonds').
  const [kind, setKindState] = useState(() => prefs.get('orbitalKind', prefs.get('orbitalFragments', false) ? 'fragments' : 'whole'));
  const setKind = (v) => { setKindState(v); prefs.set('orbitalKind', v); };
  const frag = kind === 'fragments';
  const setFrag = (v) => setKind(v ? 'fragments' : 'whole');
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
  if (needed(geom) && (!path || (on && kind === 'whole'))) targets.push(geom);
  if (path && on && kind === 'whole' && !chem?.live) {
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
  const shown = on && kind === 'whole' && result && (fresh || (path && sameAtoms)) ? result.orbitals[which] || null : null;
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
  return { on, setOn, kind, setKind, frag, setFrag, bonds: kind === 'bonds', which, setWhich, pct, setPct, iso: isoFor(shown, pct), sign, busy: busy && on, error: on ? error?.message : null,
    result: on ? result : null, fresh, current: shown,
    symbols: geom ? geom.trim().split('\n').slice(2).map((l) => l.trim().split(/\s+/)[0]) : [] };
}

// Each molecule's own frontier orbitals at the shown geometry (fetched while `enabled`): every
// donor -> acceptor pair's level overlaps, the two levels chosen (the server's best by default)
// drawn. A path works as the whole system's orbitals do: the choice stays the same at every frame
// (the first frame's best, until changed), the frames load ahead in playing order (the profile
// request has the server build and keep all of them, and returns |S| per frame), a frame's
// surfaces stay until the next frame's arrive, and the phase is carried from frame to frame.
export function useFragmentOrbitals(chem, source, index, enabled) {
  const [tick, setTick] = useState(0);
  const [choice, setChoice] = useState(null);     // {donor, acceptor, occ, virt}, or null: the strongest
  const [cutEnd, setCutEnd] = useState(null);     // a path cut at 'start' / 'end' (null: the start, if it can be)
  const [groups, setGroupsState] = useState(null); // molecules grouped into fragments ([[molecule numbers]]), or each its own
  const [profile, setProfile] = useState(null);   // {key, pairs: per frame [{donor, acceptor, table: [[occ, virt, s]]}]}
  const cache = useRef(new Map());                // `${pick}|${xyz}` -> result
  const inFlight = useRef(new Set());
  const last = useRef(null);                      // the result drawn last (kept on screen until the next frame's)
  const path = Array.isArray(source) && source.length > 1;
  const live = !!chem?.live;
  const geom = source ? (Array.isArray(source) ? source[index] : source) : null;
  const reference = path ? [source[0], source[source.length - 1]] : null;   // cut into the molecules of an end
  // A new path or geometry family: start from the server's best again.
  useEffect(() => { setChoice(null); setCutEnd(null); setGroupsState(null); last.current = null; }, [path ? source[0] : null]);
  // Cut at the other end, or grouped otherwise: other fragments, so the levels are chosen afresh.
  const setCut = (end) => { setCutEnd(end); setGroupsState(null); setChoice(null); last.current = null; };
  const setGroups = (g) => { setGroupsState(g); setChoice(null); last.current = null; };
  // Molecules given by the caller (Design: as drawn); a change (one added or removed) starts the choice afresh.
  const fragKey = chem?.fragments ? JSON.stringify(chem.fragments) : '';
  useEffect(() => { setChoice(null); setGroupsState(null); }, [fragKey]);
  const pickKey = `${fragKey}|${cutEnd}|${JSON.stringify(groups)}|${JSON.stringify(choice)}`;
  const keyOf = (xyz) => `${pickKey}|${xyz}`;
  const body = (xyz) => ({ xyz, reference: reference || undefined, cut: (path && cutEnd) || undefined, groups: groups || undefined,
    fragments: chem.fragments || undefined, charge: chem.charge ?? 0, multiplicity: chem.multiplicity ?? 1,
    job: chem.job || undefined, pick: choice || undefined, follow: path || live || undefined });
  const needed = (xyz) => xyz && !cache.current.has(keyOf(xyz)) && !inFlight.current.has(keyOf(xyz));
  const targets = [];
  if (enabled && chem) {
    if (needed(geom)) targets.push(geom);
    // Ahead of a finished path's playing, once the choice is fixed (the first frame's answer fixes it).
    if (path && !live && choice) {
      for (let k = 1; k < source.length && targets.length < LOADERS; k += 1) {
        const f = source[(index + k) % source.length];
        if (needed(f) && !targets.includes(f)) targets.push(f);
      }
    }
  }
  useEffect(() => {
    for (const xyz of targets.slice(0, Math.max(0, LOADERS - inFlight.current.size))) {
      const key = keyOf(xyz);
      inFlight.current.add(key);
      api.post('/api/fragment-orbitals', body(xyz))
        .then((r) => { cache.current.set(key, r); })
        .catch((e) => { cache.current.set(key, { error: String(e.message || e) }); })
        .finally(() => {
          inFlight.current.delete(key);
          if (cache.current.size > 600) cache.current.delete(cache.current.keys().next().value);
          setTick((t) => t + 1);
        });
    }
  }, [targets.join('|'), pickKey, tick]);
  // A finished path's profile: the server builds and keeps every frame for this choice (the frames
  // asked above then wait on the same builds), and sends |S| per frame.
  const profKey = path && !live && choice ? `${pickKey}|${source.length}|${source[0]}` : null;
  useEffect(() => {
    if (!enabled || !profKey || profile?.key === profKey) return;
    setProfile({ key: profKey, pairs: null });
    api.post('/api/fragment-orbitals/profile', { ...body(source[0]), xyz: undefined, frames: source })
      .then((x) => setProfile((p) => (p?.key === profKey ? { key: profKey, pairs: x.pairs } : p)))
      .catch(() => setProfile((p) => (p?.key === profKey ? { key: profKey, pairs: [] } : p)));
  }, [enabled, profKey]);
  const own = geom ? cache.current.get(keyOf(geom)) : null;
  const fresh = !!own && !own.error;
  // On a path, the answer on screen fixes the choice for every frame (stored under that choice too).
  useEffect(() => {
    if (!enabled || !path || choice || !fresh || !own.drawn) return;
    const c = { donor: own.drawn.donor, acceptor: own.drawn.acceptor, occ: own.drawn.occ, virt: own.drawn.virt };
    cache.current.set(`${fragKey}|${cutEnd}|${JSON.stringify(groups)}|${JSON.stringify(c)}|${geom}`, own);
    setChoice(c);
  }, [enabled, path, choice, fresh && own]);
  let r = fresh ? own : null;
  const kept = !r && path && last.current && parseInt(last.current.xyz, 10) === parseInt(geom, 10);
  if (kept) r = last.current.result;
  const drawn = enabled && r?.drawn ? r.drawn : null;
  if (drawn) {
    drawn._occ ||= { cube: drawn.occ_cube };
    drawn._virt ||= { cube: drawn.virt_cube };
  }
  // The pair's phase along a finished path: each frame's donor orbital against the frame before
  // it that has one (the acceptor's flips with it, so the overlap drawn stays in phase).
  let sign = kept ? last.current.sign : 1;
  if (drawn && fresh && path && !live) {
    let prev = null;
    for (let k = 0; k <= index; k += 1) {
      const d = cache.current.get(keyOf(source[k]))?.drawn;
      if (!d) continue;
      d._occ ||= { cube: d.occ_cube };
      d._virt ||= { cube: d.virt_cube };
      if (prev) {
        prev._occ._phase ||= new Map();
        if (!prev._occ._phase.has(d._occ)) prev._occ._phase.set(d._occ, phaseBetween(prev._occ, d._occ));
        sign *= prev._occ._phase.get(d._occ);
      }
      prev = d;
    }
  }
  if (drawn && fresh && path && live && last.current?.result?.drawn && !kept) {
    // A running calculation: against the last drawn.
    const prev = last.current.result.drawn;
    if (prev !== drawn && prev._occ) sign = last.current.sign * phaseBetween(prev._occ, drawn._occ);
    else if (prev === drawn) sign = last.current.sign;
  }
  if (fresh) last.current = { xyz: geom, result: own, sign };
  const pair = drawn ? r.pairs.find((p) => p.donor === drawn.donor && p.acceptor === drawn.acceptor) : null;
  const choose = (c) => setChoice({ donor: drawn?.donor, acceptor: drawn?.acceptor, occ: drawn?.occ, virt: drawn?.virt, ...c });
  return { result: enabled ? r : null, error: enabled && own?.error ? own.error : null,
    busy: enabled && !!geom && !own, pair, drawn, choose, fresh, setCut, setGroups,
    occ: drawn?._occ || null, virt: drawn?._virt || null, sign, virtSign: sign,
    profile: profile?.key === profKey ? profile.pairs : null };
}

// The overlap of the two levels at every frame of the path, the shown frame marked.
function Sparkline({ values, index }) {
  const W = 120, H = 22;
  const known = values.map((v, i) => [i, v]).filter(([, v]) => v != null);
  if (known.length < 2) return null;
  const top = Math.max(...known.map(([, v]) => v), 1e-3);
  const x = (i) => (values.length > 1 ? (i / (values.length - 1)) * (W - 4) + 2 : W / 2);
  const y = (v) => H - 2 - (v / top) * (H - 4);
  const at = values[index];
  return html`<svg class="mo-spark" width=${W} height=${H} viewBox=${`0 0 ${W} ${H}`} aria-label="overlap along the path">
    <title>${`Their overlap along the path; the dot is this frame (largest ${top.toFixed(3)})`}</title>
    <polyline fill="none" stroke="currentColor" stroke-width="1.2" points=${known.map(([i, v]) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} />
    ${at != null && html`<circle cx=${x(index)} cy=${y(at)} r="2.6" fill="var(--accent)" />`}
  </svg>`;
}

// A molecule as people write it: CH₃Cl, Br⁻, H₃O⁺, and its spin when it is a radical.
const SUB = '₀₁₂₃₄₅₆₇₈₉';
function fragLabel(f) {
  const q = f.charge;
  const charge = !q ? '' : `${Math.abs(q) > 1 ? Math.abs(q) : ''}${q > 0 ? '⁺' : '⁻'}`.replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
  const formula = f.formula.replace(/\d/g, (d) => SUB[d]);
  return `${f.members?.length > 1 ? `(${formula})` : formula}${charge}${f.multiplicity > 1 ? ' •' : ''}`;
}
const levelName = (l) => `${l.name.toUpperCase().replace('-', '−')}${l.degeneracy > 1 ? ` (${l.degeneracy}-fold)` : ''}`;

// Which geometry the molecules were cut from, said in the bar: it decides what "the molecules" are.
function cutNote(cut, nframes) {
  if (!cut?.from) return null;
  const where = { start: `the path's start (frame 1)`, end: `the path's end (frame ${nframes})`, shown: 'this geometry', given: 'a hand-made cut' }[cut.from];
  const label = { start: 'cut at path start', end: 'cut at path end', shown: 'cut here', given: 'molecules as drawn' }[cut.from];
  const tip = cut.from === 'given'
    ? 'The molecules as drawn here (their own bonds and formal charges), however close they sit to each other.'
    : cut.from === 'shown'
    ? 'The molecules are the separate pieces of the geometry shown (bonds perceived from distances).'
    : `The molecules are the separate pieces of ${where} (bonds perceived from distances), and every frame is split the same way, `
      + 'so the same molecules are followed through the whole path. For A + B → C cut at the start, a frame where A and B have '
      + `bonded is still split into A's and B's atoms.${cut.from === 'end' && !(cut.options || []).includes('start') ? ' (The start is one molecule, so the end is used.)' : ''}`;
  return { label, tip };
}

// Advanced: which molecules make up each fragment (shown only when there are three or more). Each
// molecule gets a letter; molecules with the same letter are computed together as one fragment.
const LETTERS = 'ABCDEF';
function GroupPicker({ cut, onApply }) {
  const mols = cut.molecules;
  const current = () => {
    const letter = mols.map((_, i) => i);
    (cut.groups || mols.map((_, i) => [i])).forEach((g, gi) => g.forEach((i) => { letter[i] = gi; }));
    return letter;
  };
  const [draft, setDraft] = useState(current);
  const used = new Set(draft);
  const apply = () => {
    const groups = [...new Set(draft)].sort((a, b) => a - b).map((gi) => draft.map((g, i) => (g === gi ? i : -1)).filter((i) => i >= 0));
    if (groups.length >= 2) onApply(groups.length === mols.length ? null : groups);
  };
  return html`<details class="mo-groups">
    <summary title="Choose which molecules make up each fragment: molecules with the same letter are computed together, as one (e.g. a water dimer as the acceptor)">${cut.groups ? `${cut.groups.length} groups` : 'group molecules'}</summary>
    <div class="mo-groups-pop">
      ${mols.map((m, i) => html`<div class="mo-groups-row"><span>${fragLabel(m)}</span>
        <span class="segmented mini">${LETTERS.slice(0, mols.length).split('').map((L, gi) => html`<button
          class=${draft[i] === gi ? 'on' : ''} onClick=${() => setDraft(draft.map((g, j) => (j === i ? gi : g)))}>${L}</button>`)}</span></div>`)}
      <div class="mo-groups-row">
        <button class="btn small" disabled=${used.size < 2} onClick=${apply}
          title=${used.size < 2 ? 'Use at least two letters' : 'Compute each group as one fragment'}>Use these groups</button>
        <button class="btn small" onClick=${() => { setDraft(mols.map((_, i) => i)); onApply(null); }}>Each molecule separately</button>
      </div>
    </div>
  </details>`;
}

function FragmentBar({ fmo, index, path, nframes = 1 }) {
  const r = fmo.result;
  if (fmo.error) return html`<span class="warn-text" title=${fmo.error}>${/one molecule/.test(fmo.error) ? 'one molecule: nothing to split' : 'no fragment orbitals'}</span>`;
  const { pair, drawn } = fmo;
  if (!r || !pair || !drawn) return null;
  const F = r.fragments;
  const cell = pair.table.find((t) => t.occ === drawn.occ && t.virt === drawn.virt);
  // Picking the giver: keep its level, take the best receiver for it; picking the receiver: its best level.
  const pickFrom = (v) => {
    const [d, occ] = v.split('|'); const donor = +d;
    if (donor === drawn.donor) return fmo.choose({ occ });
    const p = r.pairs.find((x) => x.donor === donor);
    if (p) fmo.choose({ donor, acceptor: p.acceptor, occ, virt: p.virt });
  };
  const pickInto = (v) => {
    const [a, virt] = v.split('|'); const acceptor = +a;
    fmo.choose({ acceptor, virt });
  };
  const ok = (d, a) => r.pairs.some((p) => p.donor === d && p.acceptor === a);
  const whole = r.whole && ['homo', 'lumo'].filter((k) => r.whole[k]?.length).map((k) => `The whole system's ${k.toUpperCase()} is ${
    r.whole[k].map((w) => `${Math.round(w.weight * 100)}% ${fragLabel(F[w.fragment])} ${w.level === 'other' ? '(other orbitals)' : w.level.toUpperCase().replace('-', '−')}`).join(' + ')}`).join('\n');
  const series = path && fmo.profile ? fmo.profile.map((fr) => fr?.find((p) => p.donor === pair.donor && p.acceptor === pair.acceptor)
    ?.table.find(([o, v]) => o === drawn.occ && v === drawn.virt)?.[2] ?? null) : null;
  const note = cutNote(r.cut, nframes);
  // Two radicals' singly occupied levels: one electron each, paired into a bond (not a donation).
  const pairing = drawn.occ === 'somo' && drawn.virt === 'somo';
  return html`
    <span class="mo-frag">
      <span class="muted" title=${pairing ? 'Two singly occupied orbitals (SOMOs), one electron each: overlapping, they pair into a bond' : ''}>${pairing ? 'Pairs' : 'Electrons from'}</span>
      <select class="tiny" value=${`${drawn.donor}|${drawn.occ}`} onChange=${(e) => pickFrom(e.target.value)}
        title="A filled orbital of one molecule, computed on its own (drawn dark)">
        ${F.map((f, i) => f.levels.filter((l) => l.occupied).map((l) => html`<option value=${`${i}|${l.name}`}>${fragLabel(f)} ${levelName(l)}</option>`))}</select>
      <span class="muted">${pairing ? 'with' : 'into'}</span>
      <select class="tiny" value=${`${drawn.acceptor}|${drawn.virt}`} onChange=${(e) => pickInto(e.target.value)}
        title="An empty (or half-filled: SOMO) orbital of another molecule, computed on its own (drawn pale)">
        ${F.map((f, i) => (i === drawn.donor || !ok(drawn.donor, i) ? [] : f.levels.filter((l) => !l.occupied)
          .map((l) => html`<option value=${`${i}|${l.name}`}>${fragLabel(f)} ${levelName(l)}</option>`)))}</select>
    </span>
    <span class="mo-numbers">
      <span title="How much the two orbitals share the same space (0 = not at all, 1 = identical). Electrons can only flow between orbitals that overlap: 0 means symmetry forbids it, however small the gap. Where same colours meet, they overlap in phase (bonding).">overlap <b>${cell ? cell.s.toFixed(3) : '–'}</b></span>
      <span title="Energy of the empty orbital minus the filled one (eV). The smaller the gap and the larger the overlap, the stronger the pull to bond (∝ overlap² / gap).">· gap ${cell ? cell.gap_ev.toFixed(2) : '–'} eV</span>
    </span>
    ${series && html`<${Sparkline} values=${series} index=${index} />`}
    ${path && fmo.profile === null && html`<span class="spinner" title="Computing every frame" aria-hidden="true"></span>`}
    ${note && (path && r.cut?.options?.length > 1
      ? html`<select class="tiny mo-cut" value=${r.cut.from} onChange=${(e) => fmo.setCut(e.target.value)}
          title=${`${note.tip}\n\nBoth ends of this path hold more than one molecule: choose which end's molecules to use.`}>
          <option value="start">cut at path start</option><option value="end">cut at path end</option></select>`
      : html`<span class="mo-sites" title=${note.tip}>${note.label}</span>`)}
    ${r.cut?.molecules?.length >= 3 && html`<${GroupPicker} key=${JSON.stringify(r.cut.groups)} cut=${r.cut} onApply=${fmo.setGroups} />`}
    ${r.cut && r.cut.same === false && r.cut.from !== 'given' && html`<span class="warn-text" title=${`At this frame the atoms do not split into the same molecules (bonds perceived from distances give ${r.cut.here === 1 ? 'one molecule' : `${r.cut.here} pieces`} here). The orbitals are still those of the molecules of the cut, each computed alone at this frame's geometry: where they have bonded, those are pieces with broken bonds, not molecules that exist here. Read overlap and gap here as how far the separate molecules' orbitals have merged.`}>⚠ bonded at this frame</span>`}
    ${whole && html`<span class="mo-sites" title=${whole}>whole-system mix</span>`}`;
}

// ---- bond orbitals -----------------------------------------------------------
// The colours of the curly arrows (the active orbitals' charge centres along a path), apart from
// the orbitals' blue and red.
const ARROW_COLORS = ['#e07b00', '#12907e', '#8a4fd3', '#c2185b', '#5d7f1e', '#8d6e63'];

// The whole system's filled orbitals as bonds and lone pairs, at the geometry shown (fetched while
// `enabled`). A finished path: the server follows every orbital through the frames once (its
// number and sign at each frame); the one chosen (by default the one that changes most) is drawn
// at every frame, frames loading ahead in playing order, and the ones that change are drawn as
// arrows. A single geometry (or a running calculation's frame): its list, and the one chosen.
export function useBondOrbitals(chem, source, index, enabled) {
  const [pick, setPick] = useState(null);         // a path: the followed orbital's id; one geometry: its number
  const [track, setTrack] = useState(null);       // {key, data}: the path's orbitals followed frame to frame
  const [tick, setTick] = useState(0);
  const cache = useRef(new Map());                // `${number}|${xyz}` -> result
  const inFlight = useRef(new Set());
  const last = useRef(null);                      // the result drawn last (kept on screen until the next frame's)
  const path = Array.isArray(source) && source.length > 1;
  const live = !!chem?.live;
  const geom = source ? (Array.isArray(source) ? source[index] : source) : null;
  const pathKey = path && !live ? `${source.length}|${source[0]}|${source[source.length - 1]}` : null;
  useEffect(() => { setPick(null); last.current = null; }, [pathKey || geom]);
  useEffect(() => {
    if (!enabled || !chem || !pathKey || track?.key === pathKey) return;
    setTrack({ key: pathKey, data: null });
    api.post('/api/bond-orbitals/path', { frames: source, charge: chem.charge ?? 0, multiplicity: chem.multiplicity ?? 1, job: chem.job || undefined })
      .then((d) => setTrack((t) => (t?.key === pathKey ? { key: pathKey, data: d } : t)))
      .catch((e) => setTrack((t) => (t?.key === pathKey ? { key: pathKey, data: { error: String(e.message || e) } } : t)));
  }, [enabled, pathKey]);
  const T = pathKey && track?.key === pathKey ? track.data : null;
  const followed = T && !T.error ? T : null;
  const byChange = followed ? [...followed.orbitals].sort((a, b) => b.max_moved - a.max_moved) : [];
  const id = followed ? (pick ?? followed.active[0] ?? byChange[0]?.id ?? null) : null;
  // The orbital's number at frame f (a followed path), else the one picked (null: the server's choice).
  // While a path is being followed, the frame on screen shows its own (the server's choice) meanwhile.
  const numberAt = (f) => (pathKey ? (followed && id != null ? followed.orbitals[id].local[f] : (!T && f === index ? null : undefined)) : pick);
  const keyOf = (xyz, k) => `${k ?? '-'}|${xyz}`;
  const targets = [];
  if (enabled && chem && geom) {
    for (let j = 0; j < (pathKey ? source.length : 1) && targets.length < LOADERS; j += 1) {
      const f = pathKey ? (index + j) % source.length : index;
      const xyz = pathKey ? source[f] : geom;
      const k = numberAt(f);
      if (k === undefined) continue;
      const key = keyOf(xyz, k);
      if (!cache.current.has(key) && !inFlight.current.has(key)) targets.push([xyz, k]);
    }
  }
  useEffect(() => {
    for (const [xyz, k] of targets.slice(0, Math.max(0, LOADERS - inFlight.current.size))) {
      const key = keyOf(xyz, k);
      inFlight.current.add(key);
      api.post('/api/bond-orbitals', { xyz, charge: chem.charge ?? 0, multiplicity: chem.multiplicity ?? 1, job: chem.job || undefined,
        orbital: k ?? undefined, follow: path || live || undefined })
        .then((r) => { cache.current.set(key, r); })
        .catch((e) => { cache.current.set(key, { error: String(e.message || e) }); })
        .finally(() => {
          inFlight.current.delete(key);
          if (cache.current.size > 600) cache.current.delete(cache.current.keys().next().value);
          setTick((t) => t + 1);
        });
    }
  }, [targets.map(([x, k]) => keyOf(x, k)).join('|'), tick]);
  const k = numberAt(index);
  const own = geom && k !== undefined ? cache.current.get(keyOf(geom, k)) : null;
  const fresh = !!own && !own.error;
  let r = fresh ? own : null;
  let shownAt = index;
  if (!r && path && last.current && parseInt(last.current.xyz, 10) === parseInt(geom, 10)) {
    r = last.current.result;
    shownAt = last.current.at;
  }
  if (fresh) last.current = { xyz: geom, result: own, at: index };
  const drawn = enabled && r?.drawn ? r.drawn : null;
  if (drawn) drawn._orb ||= { cube: drawn.cube };
  const sign = followed && id != null ? (followed.orbitals[id].sign[shownAt] ?? 1) : 1;
  const arrows = useMemo(() => (followed ? followed.active.slice(0, ARROW_COLORS.length).map((t, i) => ({
    id: t, color: ARROW_COLORS[i], points: followed.orbitals[t].centre, chosen: t === id })) : null), [followed, id]);
  const error = enabled ? (T?.error || own?.error || null) : null;
  return { result: enabled ? r : null, error, busy: enabled && !!geom && ((!!pathKey && !T) || (!own && !error)), following: enabled && !!pathKey && !T,
    followed: enabled ? followed : null, byChange, id, number: drawn?.index ?? null, local: k, current: enabled ? drawn?._orb || null : null,
    sign, arrows: enabled ? arrows : null, fresh, colorOf: (t) => (followed ? ARROW_COLORS[followed.active.indexOf(t)] : null),
    choose: setPick };
}

// The active orbitals' centroids along the path, as curly arrows (the chosen one thicker), each
// with a dot where it is at the frame shown. Hovering one tags it with its name (`describe`).
function drawArrows(v, shapes, arrows, frame, describe = () => '') {
  (shapes.current || []).forEach((sh) => {
    try { if (sh._tag) v.removeLabel(sh._tag); v.removeShape(sh); } catch { /* gone with the model */ }
  });
  shapes.current = [];
  for (const a of arrows || []) {
    const pts = [];
    for (const c of a.points) {
      if (!c) continue;
      const p = { x: c[0], y: c[1], z: c[2] };
      const q = pts[pts.length - 1];
      if (!q || Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z) >= 0.08) pts.push(p);   // (no loops from jitter)
    }
    const end = a.points.filter(Boolean).pop();
    const tail = pts[pts.length - 1];
    if (end && tail && (end[0] !== tail.x || end[1] !== tail.y || end[2] !== tail.z)) {
      if (pts.length > 1 && Math.hypot(end[0] - tail.x, end[1] - tail.y, end[2] - tail.z) < 0.04) pts.pop();
      pts.push({ x: end[0], y: end[1], z: end[2] });    // the arrow ends where the orbital ends
    }
    if (pts.length < 2) continue;
    if (pts.length === 2) pts.splice(1, 0, { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2, z: (pts[0].z + pts[1].z) / 2 });
    const radius = a.chosen ? 0.07 : 0.045;
    const head = pts[pts.length - 1];
    const tag = {
      hoverable: true,
      hover_callback: (shape, viewer) => {
        if (shape._tag) return;
        const text = describe(a);
        shape._tag = viewer.addLabel(text, { position: head, backgroundColor: a.color, backgroundOpacity: 0.92, fontColor: 'white',
          fontSize: 12, inFront: true, borderRadius: 4, padding: 3 });
        viewer._mepdHover = text;   // (for tests)
        viewer.render();
      },
      unhover_callback: (shape, viewer) => {
        if (!shape._tag) return;
        viewer.removeLabel(shape._tag);
        shape._tag = null;
        viewer.render();
      },
    };
    try {
      shapes.current.push(v.addCurve({ points: pts, radius, color: a.color, toArrow: true, smooth: 4, ...tag }));
      const at = a.points[Math.min(frame, a.points.length - 1)];
      if (at) shapes.current.push(v.addSphere({ center: { x: at[0], y: at[1], z: at[2] }, radius: a.chosen ? 0.15 : 0.1, color: a.color, ...tag }));
    } catch { /* a degenerate curve: skip it */ }
  }
  v.render();
}

// An orbital as people say it: "O3 lone pair", "C1–O3", "C1–H5–O3 (3c)", from the atoms it sits on.
function bondLabel(o, symbols, base) {
  const atoms = o?.atoms;
  if (!atoms?.length) return '?';
  const names = [...atoms].sort((x, y) => x[0] - y[0]).map(([a]) => `${symbols[a] ?? '?'}${a + base}`);
  const spin = o.spin === 'α' || o.spin === 'β' ? ` ${o.spin}` : '';
  if (names.length === 1) return `${names[0]} ${o.spin === 'single' ? 'radical' : 'lone pair'}${spin}`;
  return `${names.join('–')}${names.length > 2 ? ` (${names.length}c)` : ''}${o.spin === 'single' ? ' (1e)' : ''}${spin}`;
}

// A followed orbital as "C1–C2 → C1–C6" (where it starts → where it ends).
function pathLabel(F, t, symbols, base) {
  const x = F.orbitals[t];
  const a = bondLabel({ ...x, atoms: x.start }, symbols, base);
  const b = bondLabel({ ...x, atoms: x.end }, symbols, base);
  if (a !== b) return `${a} → ${b}`;
  return F.active.includes(t) ? `${a} (polarizes)` : a;   // changing, but in the same bond: shifts toward one atom
}

// What an arrow's hover tag says: the electron pair, and how far it has moved by this frame.
function arrowName(F, t, symbols, base, frame) {
  if (!F?.orbitals[t]) return '';
  const o = F.orbitals[t];
  const now = o.moved[frame];
  return `${pathLabel(F, t, symbols, base)} · electron pair, ${now == null ? '' : `${Math.round(now * 100)} % moved here, `}${Math.round(o.max_moved * 100)} % in all`;
}

// A little arrow in a colour (the legend's key to the arrows in the view).
const Swatch = ({ color }) => html`<svg class="mo-swatch" width="20" height="8" viewBox="0 0 20 8" aria-hidden="true">
  <line x1="1" y1="4" x2="13" y2="4" stroke=${color} stroke-width="2.4" /><path d="M12 0.5 L19 4 L12 7.5 Z" fill=${color} /></svg>`;

// How much of each active orbital has moved between atoms at every frame (its colour as its
// arrow's), the chosen one thicker, a dot at the frame shown.
function ChangePlot({ series, index }) {
  const W = 120, H = 24;
  const n = Math.max(0, ...series.map((s) => s.values.length));
  if (n < 2) return null;
  const top = Math.max(0.05, ...series.flatMap((s) => s.values.filter((v) => v != null)));
  const x = (i) => (i / (n - 1)) * (W - 4) + 2;
  const y = (v) => H - 2 - (v / top) * (H - 4);
  const chosen = series.find((s) => s.chosen);
  const at = chosen?.values[index];
  return html`<svg class="mo-spark" width=${W} height=${H} viewBox=${`0 0 ${W} ${H}`} aria-label="orbital change along the path">
    <title>${`How much of each changing orbital has moved to other atoms, frame by frame (top ${Math.round(top * 100)} %); the dot is this frame`}</title>
    ${series.map((s) => html`<polyline fill="none" stroke=${s.color} stroke-width=${s.chosen ? 1.8 : 1}
      points=${s.values.map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`)).filter(Boolean).join(' ')} />`)}
    ${at != null && html`<circle cx=${x(index)} cy=${y(at)} r="2.6" fill=${chosen.color} />`}
  </svg>`;
}

function BondBar({ bmo, index, path, symbols, labelBase }) {
  if (bmo.error) return html`<span class="warn-text" title=${bmo.error}>no bond orbitals</span>`;
  const r = bmo.result;
  const F = bmo.followed;
  const fmtq = (q) => `${q > 0 ? '+' : ''}${q.toFixed(2)}`.replace('-', '−');
  const name = (a) => `${symbols[a] ?? '?'}${a + labelBase}`;
  if (path && F) {
    const o = F.orbitals[bmo.id];
    if (!o) return null;
    const label = (t) => pathLabel(F, t, symbols, labelBase);
    const quiet = F.orbitals.filter((x) => !F.active.includes(x.id));
    const series = F.active.map((t) => ({ values: F.orbitals[t].moved, color: bmo.colorOf(t), chosen: t === bmo.id }));
    const q0 = F.charges.find(Boolean), q = F.charges[index];
    const dq = q && q0 ? q.map((v, a) => [v - q0[a], a]).filter(([d]) => Math.abs(d) >= 0.05).sort((u, w) => Math.abs(w[0]) - Math.abs(u[0])) : [];
    const moved = o.moved[index];
    return html`
      <select class="tiny" value=${bmo.id} onChange=${(e) => bmo.choose(+e.target.value)}
        title="The whole system's filled orbitals as bonds and lone pairs (intrinsic bond orbitals), each followed along the path. The ones that change are the curly arrows of the mechanism, drawn as the path of each one's charge centre; the rest stay put.">
        ${F.active.length > 0 && html`<optgroup label="Changing (arrows)">${F.active.map((t) => html`<option value=${t}>${label(t)} · ${Math.round(F.orbitals[t].max_moved * 100)} %</option>`)}</optgroup>`}
        <optgroup label="Unchanged">${quiet.map((x) => html`<option value=${x.id}>${label(x.id)}</option>`)}</optgroup>
      </select><${Why} id="other.bond-orbitals" label="Bond orbitals" />
      <span class="mo-numbers" title="How much of this orbital has moved to other atoms since the first frame: half the sum over atoms of the change in its share">moved <b>${moved == null ? '–' : `${Math.round(moved * 100)} %`}</b></span>
      ${series.length > 0 && html`<${ChangePlot} series=${series} index=${index} />`}
      ${o.matched < 0.8 && html`<span class="warn-text" title=${`Followed from frame to frame with an overlap as low as ${o.matched.toFixed(2)} (IboView warns below 0.8): the frames are far apart, or it mixes with an equivalent orbital (lone pairs around a bond axis).`}>⚠ loosely followed</span>`}
      ${dq.length > 0 && html`<span class="mo-sites" title=${`Charge on each atom (IAO charges) at this frame minus the first frame's (+: electrons lost, −: gained):\n${dq.map(([d, a]) => `${name(a)} ${fmtq(d)}`).join('\n')}`}>Δq ${dq.slice(0, 3).map(([d, a]) => `${name(a)} ${fmtq(d)}`).join(', ')}<${Why} id="other.bond-charges" label="Δq" /></span>`}
      ${F.failed?.length > 0 && html`<span class="warn-text" title=${`No orbitals at frame${F.failed.length > 1 ? 's' : ''} ${F.failed.map((i) => i + 1).join(', ')}`}>⚠ ${F.failed.length} frame${F.failed.length > 1 ? 's' : ''} missing</span>`}
      <span class="mo-legend">${F.active.length > 0
        ? html`<span class="muted" title="Each arrow is one electron pair that moves along the path: from where it sits at the first frame (tail) to where it sits at the last (head); the dot is where it is now">Arrows</span>
          ${F.active.slice(0, 6).map((t) => html`<button type="button" class=${t === bmo.id ? 'on' : ''} onClick=${() => bmo.choose(t)}
            title=${`${label(t)}: this electron pair moves ${Math.round(F.orbitals[t].max_moved * 100)} % onto other atoms along the path. Click to draw it.`}>
            <${Swatch} color=${bmo.colorOf(t)} />${label(t)}</button>`)}`
        : html`<span class="muted">No bond or lone pair changes along this path</span>`}<${Why} id="other.bond-arrows" label="The arrows" /></span>`;
  }
  if (!r) return bmo.following ? html`<span class="muted" title="Every frame's bond orbitals are built and each is followed from frame to frame">following the orbitals along the path…</span>` : null;
  const order = r.orbitals.map((x, i) => [x, i]).sort(([a], [b]) => (b.atoms.length > 2) - (a.atoms.length > 2) || b.energy_ev - a.energy_ev);
  const drawn = r.orbitals[r.drawn.index];
  return html`
    <select class="tiny" value=${r.drawn.index} onChange=${(e) => bmo.choose(+e.target.value)} disabled=${bmo.following}
      title="The whole system's filled orbitals as bonds and lone pairs (intrinsic bond orbitals): those spread over three or more atoms first (a transition state's forming and breaking bonds), then from the highest energy down">
      ${order.map(([x, i]) => html`<option value=${i}>${bondLabel(x, symbols, labelBase)}</option>`)}</select><${Why} id="other.bond-orbitals" label="Bond orbitals" />
    ${drawn && html`<span class="mo-sites" title=${`Its share on each atom (IAO populations); orbital energy ${drawn.energy_ev.toFixed(2)} eV (the canonical energies it mixes, weighted)`}>on ${drawn.atoms.map(([a, w]) => `${name(a)} ${Math.round(w * 100)}%`).join(', ')}</span>`}
    ${r.charges && html`<span class="mo-sites" title=${`Partial charges (IAO): ${r.charges.map((v, a) => `${name(a)} ${fmtq(v)}`).join(', ')}`}>charges</span>`}
    ${bmo.following && html`<span class="muted" title="Every frame's bond orbitals are built and each is followed from frame to frame; until then, this frame's own are shown">following the path…</span>`}`;
}

export function OrbitalBar({ mo, fmo = null, bmo = null, index = 0, path = false, nframes = 1, labelBase = 1 }) {
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
      ${mo.busy || (mo.frag && fmo?.busy) || (mo.bonds && bmo?.busy) ? html`<span class="spinner" aria-hidden="true"></span>` : ''}Orbitals</button>
    ${mo.on && (fmo || bmo) && html`<span class="segmented mini">
      <button class=${mo.kind === 'whole' ? 'on' : ''} onClick=${() => mo.setKind('whole')} title="The orbitals of the whole system">Whole</button>
      ${fmo && html`<button class=${mo.frag ? 'on' : ''} onClick=${() => mo.setKind('fragments')}
        title="Each molecule's own orbitals, computed on its own where it sits: a filled orbital of one and an empty orbital of another, and how much they overlap (the frontier-orbital picture of why molecules bond)">Fragments</button>`}
      ${bmo && html`<button class=${mo.bonds ? 'on' : ''} onClick=${() => mo.setKind('bonds')}
        title="The whole system's filled orbitals as its bonds and lone pairs (intrinsic bond orbitals). Along a path, the few that change are the curly arrows of the mechanism.">Bonds</button>`}</span>`}
    ${mo.on && mo.frag && fmo && html`<${FragmentBar} fmo=${fmo} index=${index} path=${path} nframes=${nframes} />`}
    ${mo.on && mo.bonds && bmo && html`<${BondBar} bmo=${bmo} index=${index} path=${path} symbols=${mo.symbols} labelBase=${labelBase} />`}
    ${mo.on && (mo.frag ? fmo : mo.bonds ? bmo : null) && html`<label class="mo-pct" title="The surfaces enclose this share of each orbital's density (|ψ|²). Smaller: tighter lobes.">
        <input type="number" class="tiny" min="1" max="99.9" step="5" value=${mo.pct}
          onChange=${(e) => { const v = +e.target.value; if (v > 0 && v < 100) mo.setPct(v); }} /> % density</label>`}
    ${mo.on && mo.kind === 'whole' && r && html`
      <span class="segmented mini">${['homo', 'lumo'].filter((k) => r.orbitals[k]).map((k) => html`<button class=${mo.which === k ? 'on' : ''}
        onClick=${() => mo.setWhich(k)} title=${`${name(k)}: ${fmt(ev(k))} eV`}>${name(k)}</button>`)}</span>
      <label class="mo-pct" title=${`The surface encloses this share of the orbital's density (|ψ|²); drawn at |ψ| = ${mo.iso.toFixed(3)}. Smaller: tighter lobes.`}>
        <input type="number" class="tiny" min="1" max="99.9" step="5" value=${mo.pct}
          onChange=${(e) => { const v = +e.target.value; if (v > 0 && v < 100) mo.setPct(v); }} /> % density</label>
      <span class="mo-numbers" title=${`${method} orbital energies (eV)${r.method === 'terachem' ? '' : ', from its minimal basis'}`}>${name('homo')} ${fmt(ev('homo'))} · LUMO ${fmt(ev('lumo'))}${r.gap_ev != null ? html` · gap <b>${r.gap_ev.toFixed(2)}</b>` : ''} eV</span>
      <span class="mo-method muted" title=${r.source === 'run' ? 'From the calculation itself: the SCF it ran at this geometry' : 'From a single point at this level on the geometry shown (no calculation had run this geometry)'}>${method}${r.source === 'run' ? ' · this run' : ''}</span>
      ${sites.length > 0 && html`<span class="mo-sites" title="How much of the shown orbital sits on each atom">on ${sites.map(([w, i]) => `${mo.symbols[i]}${i + labelBase} ${Math.round(w * 100)}%`).join(', ')}</span>`}
      `}
    ${mo.on && mo.bonds && bmo?.result && html`<span class="mo-method muted" title=${bmo.result.source === 'run' ? 'Built from the calculation\'s own orbitals at this geometry' : 'Built from a single point at this level on the geometry shown (no calculation had run this geometry)'}>${bmo.result.label}${bmo.result.source === 'run' ? ' · this run' : ''}</span>`}
    ${mo.error && mo.kind === 'whole' && html`<span class="warn-text" title=${mo.error}>no orbitals</span>`}
    ${mo.on && html`<a class="mo-help" href=${`/manual/orbitals#${mo.frag ? 'fragment-orbitals' : mo.bonds ? (path ? 'bonds-along-a-path' : 'bond-orbitals') : 'whole-orbitals'}`} target="_blank" rel="noopener"
      title="How these orbitals are computed (manual)">?</a>`}
  </div>`;
}
