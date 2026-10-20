// The orbital explorer: a full-window view of a geometry's or a path's orbitals,
// opened from a viewer's orbital bar ("Explore…"). Everything an expert needs,
// out of everyone else's way:
//
// - an interaction diagram of the frame on screen: each molecule's own orbitals
//   (computed alone) beside the whole system's, lines showing how much of each
//   whole orbital each molecule's orbital makes up;
// - along a path, a correlation diagram: every orbital's energy frame by frame,
//   each line one orbital followed by its overlap with the frame before (so it
//   keeps its character through crossings, and its colours);
// - the 3D view draws what is clicked in either (shift-click: a second one, e.g.
//   one molecule's HOMO with another's LUMO, with their overlap and gap), and
//   "Start ↔ end" shows the chosen orbitals at both ends of the path.
//
// Data: POST /api/orbital-explorer (numbers for every frame, kept per session);
// surfaces: GET /api/orbital-explorer/{key}/cube, only for what is drawn.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { prefs } from '../store.js';
import { PALE, Viewer3D, isoFor } from './Viewer3D.js';

const SYS_COLORS = ['#3868b8', '#c2621f', '#2f8a5b', '#8a4fb8', '#b83868', '#5b6b2f'];
const WHOLE_COLOR = '#6b7782';
const LOADERS = 4;
const SUB = '₀₁₂₃₄₅₆₇₈₉';

// A molecule as people write it: CH₃Cl, Br⁻, and • for a radical.
export function fragLabel(f) {
  const q = f.charge;
  const charge = !q ? '' : `${Math.abs(q) > 1 ? Math.abs(q) : ''}${q > 0 ? '⁺' : '⁻'}`.replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
  return `${f.formula.replace(/\d/g, (d) => SUB[d])}${charge}${f.multiplicity > 1 ? '•' : ''}`;
}

// HOMO, HOMO−2, LUMO+1, SOMO (a partly filled frontier orbital).
function orbName(o) {
  if (o.rel <= 0 && o.occ > 0.5 && o.occ < 1.95) return o.rel === 0 ? 'SOMO' : `SOMO−${-o.rel}`;
  if (o.rel <= 0) return o.rel === 0 ? 'HOMO' : `HOMO−${-o.rel}`;
  return o.rel === 1 ? 'LUMO' : `LUMO+${o.rel - 1}`;
}

const fmtE = (x) => `${x < 0 ? '−' : ''}${Math.abs(x).toFixed(2)}`;

export function OrbitalExplorer({ frames, frame = 0, chem = {}, title = '', onClose }) {
  const n = frames.length;
  const path = n > 1;
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [k, setK] = useState(Math.min(frame, n - 1));
  const [playing, setPlaying] = useState(false);
  const [picks, setPicks] = useState(null);          // [{sys: 'w' | fragment index, line: index | null, i}]
  const [compare, setCompare] = useState(false);
  const [more, setMore] = useState(() => prefs.get('orbitalMore', false));
  const [hidden, setHidden] = useState({});           // correlation diagram: systems switched off
  const [pct, setPctState] = useState(() => prefs.get('orbitalPct', 85));
  const setPct = (v) => { setPctState(v); prefs.set('orbitalPct', v); };
  const [tick, setTick] = useState(0);
  const cubes = useRef(new Map());
  const inFlight = useRef(new Set());
  const shownCubes = useRef([null, null]);            // per slot, kept until the next frame's arrive

  useEffect(() => {
    setData(null); setError(null);
    api.post('/api/orbital-explorer', { frames, charge: chem.charge ?? 0, multiplicity: chem.multiplicity ?? 1,
      job: chem.job || undefined })
      .then(setData).catch((e) => setError(String(e.message || e)));
  }, [frames]);
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
      if (path && e.key === 'ArrowRight') setK((x) => Math.min(n - 1, x + 1));
      if (path && e.key === 'ArrowLeft') setK((x) => Math.max(0, x - 1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [n]);
  useEffect(() => {
    if (!playing) return undefined;
    const t = setInterval(() => setK((x) => (x + 1) % n), 160);
    return () => clearInterval(t);
  }, [playing, n]);

  const frags = data?.fragments || [];
  const sysOf = (fr, sys) => (fr ? (sys === 'w' ? fr.whole : fr.fragments[sys]) : null);
  const linesOf = (sys) => (sys === 'w' ? data.lines.whole : data.lines.fragments[sys]) || [];
  const sysLabel = (sys) => (sys === 'w' ? 'Whole' : fragLabel(frags[sys]));
  const sysColor = (sys) => (sys === 'w' ? WHOLE_COLOR : SYS_COLORS[sys % SYS_COLORS.length]);
  const lineAt = (sys, i, kk) => linesOf(sys).findIndex((ln) => ln.points.some((p) => p[0] === kk && p[1] === i));
  // A pick at frame kk: {i, sign}, or null where its line does not reach.
  const at = (pick, kk) => {
    if (!pick) return null;
    if (pick.line == null) return kk === pick.k ? { i: pick.i, sign: 1 } : null;
    const p = linesOf(pick.sys)[pick.line]?.points.find((q) => q[0] === kk);
    return p ? { i: p[1], sign: p[2] } : null;
  };
  const orbAt = (pick, kk) => {
    const a = at(pick, kk);
    const o = a && sysOf(data?.frames[kk], pick.sys)?.orbitals.find((x) => x.i === a.i);
    return o ? { ...o, sign: a.sign } : null;
  };
  const pickOf = (sys, i, kk) => {
    const line = lineAt(sys, i, kk);
    return { sys, line: line >= 0 ? line : null, i, k: kk };
  };

  // First look: two molecules' most likely donation (smallest HOMO -> LUMO gap), else the whole HOMO.
  useEffect(() => {
    if (!data || picks) return;
    const fr = data.frames[k] || data.frames.find(Boolean);
    if (!fr) return;
    const kk = data.frames.indexOf(fr);
    let best = null;
    fr.fragments.forEach((d, di) => fr.fragments.forEach((a, ai) => {
      if (ai === di) return;
      const h = d.orbitals.find((o) => o.rel === 0);
      const somo = (o) => o.rel === 0 && o.occ < 1.95;
      const l = a.orbitals.find(somo) || a.orbitals.find((o) => o.rel === 1);
      if (h && l && (!best || l.energy_ev - h.energy_ev < best.gap)) best = { gap: l.energy_ev - h.energy_ev, di, ai, h, l };
    }));
    if (best) setPicks([pickOf(best.di, best.h.i, kk), pickOf(best.ai, best.l.i, kk)]);
    else {
      const h = fr.whole.orbitals.find((o) => o.rel === 0);
      setPicks(h ? [pickOf('w', h.i, kk)] : []);
    }
  }, [data]);

  const click = (sys, i, kk, add) => {
    const p = pickOf(sys, i, kk);
    setPicks((cur) => {
      const list = cur || [];
      const same = (q) => q.sys === p.sys && (q.line != null ? q.line === p.line : q.i === p.i);
      if (list.some(same)) return list.filter((q) => !same(q));
      return add && list.length ? [list[0], p] : [p];
    });
    if (kk !== k) setK(kk);
  };

  // Surfaces: the picks at the frame(s) shown, then their lines' other frames in playing order.
  const coarse = path ? 1 : 0;
  const cubeKey = (kk, pick, a) => `${kk}|${pick.sys}|${a.i}|${a.sign}`;
  const wanted = [];
  if (data && picks) {
    const ends = compare ? [0, n - 1] : [k];
    for (const kk of ends) for (const p of picks) { const a = at(p, kk); if (a) wanted.push([kk, p, a]); }
    if (path) {
      for (let d = 1; d < n && wanted.length < 40; d += 1) {
        const kk = (k + d) % n;
        for (const p of picks) { const a = at(p, kk); if (a) wanted.push([kk, p, a]); }
      }
    }
  }
  const todo = wanted.filter(([kk, p, a]) => !cubes.current.has(cubeKey(kk, p, a)) && !inFlight.current.has(cubeKey(kk, p, a)));
  useEffect(() => {
    for (const [kk, p, a] of todo.slice(0, Math.max(0, LOADERS - inFlight.current.size))) {
      const key = cubeKey(kk, p, a);
      inFlight.current.add(key);
      const sys = p.sys === 'w' ? 'whole' : String(p.sys);
      api.get(`/api/orbital-explorer/${data.key}/cube?frame=${kk}&system=${sys}&i=${a.i}&sign=${a.sign}&coarse=${coarse}`)
        .then((r) => { cubes.current.set(key, { cube: r.cube }); })
        .catch(() => { cubes.current.set(key, { cube: null }); })
        .finally(() => {
          inFlight.current.delete(key);
          if (cubes.current.size > 300) cubes.current.delete(cubes.current.keys().next().value);
          setTick((t) => t + 1);
        });
    }
  }, [todo.map(([kk, p, a]) => cubeKey(kk, p, a)).join(','), tick]);

  // The overlap of two picks at frame kk, signed as drawn (null if they are not two molecules' orbitals).
  const overlapAt = (kk) => {
    if (!picks || picks.length < 2) return null;
    const [p, q] = picks;
    if (p.sys === 'w' || q.sys === 'w' || p.sys === q.sys) return null;
    const a = at(p, kk), b = at(q, kk);
    const fr = data?.frames[kk];
    if (!a || !b || !fr) return null;
    const [x, y] = p.sys < q.sys ? [[p.sys, a.i], [q.sys, b.i]] : [[q.sys, b.i], [p.sys, a.i]];
    const hit = fr.overlaps.find((o) => o.a[0] === x[0] && o.a[1] === x[1] && o.b[0] === y[0] && o.b[1] === y[1]);
    return { s: hit ? hit.s * a.sign * b.sign : 0, listed: !!hit };
  };
  // Surfaces of frame kk: the first pick dark, the second pale; the second flipped so their overlap is in phase.
  const surfacesAt = (kk, slotKeep = false) => {
    if (!data || !picks) return [];
    const ov = overlapAt(kk);
    return picks.map((p, slot) => {
      const a = at(p, kk);
      let c = a ? cubes.current.get(cubeKey(kk, p, a)) : null;
      if (slotKeep) {
        if (c?.cube) shownCubes.current[slot] = c;
        else if (path && a) c = shownCubes.current[slot];   // the last drawn until this frame's arrive
      }
      if (!c?.cube) return null;
      return { orb: c, iso: isoFor(c, pct), sign: slot === 1 && ov && ov.s < 0 ? -1 : 1, colors: slot ? PALE : undefined };
    });
  };

  const fr = data?.frames[k];
  const ov = overlapAt(k);
  const shownOrbs = (picks || []).map((p) => orbAt(p, k));
  const pairing = shownOrbs.length === 2 && shownOrbs.every((o) => o && o.rel === 0 && o.occ < 1.95);

  return html`<div class="orbx" role="dialog" aria-label="Orbital explorer">
    <header class="orbx-head">
      <h2>Orbitals${title ? html` <span class="muted">· ${title}</span>` : ''}</h2>
      ${data && html`<span class="muted small">${data.label}${frags.length ? ` · ${frags.map(fragLabel).join(' + ')}, each computed alone` : ' · one molecule'}</span>`}
      <span class="spacer"></span>
      <a class="small" href="/manual/orbitals#orbital-explorer" target="_blank" rel="noopener">How this works</a>
      <button class="btn-icon" onClick=${onClose} title="Close (Esc)">✕</button>
    </header>
    ${error && html`<div class="orbx-msg warn-text">${error}</div>`}
    ${!data && !error && html`<div class="orbx-msg"><span class="spinner"></span> Computing the orbitals of ${n > 1 ? `${n} frames` : 'this geometry'}${frags.length ? '' : ' and of each molecule'}…</div>`}
    ${data && html`<div class="orbx-body">
      <section class="orbx-3d">
        ${compare && path
          ? html`<div class="orbx-pair">${[0, n - 1].map((kk) => html`<div>
              <div class="small muted">${kk === 0 ? 'Start' : 'End'} (frame ${kk + 1})</div>
              <${Viewer3D} xyz=${frames[kk]} height=${300} surfaces=${surfacesAt(kk)} />
              <div class="small">${(picks || []).map((p) => { const o = orbAt(p, kk); return o ? `${sysLabel(p.sys)} ${orbName(o)} ${fmtE(o.energy_ev)} eV` : `${sysLabel(p.sys)}: not followed here`; }).join(' · ')}</div>
            </div>`)}</div>`
          : html`<${Viewer3D} frames=${path ? frames : null} xyz=${path ? null : frames[0]} frame=${k} height=${380} surfaces=${surfacesAt(k, true)} />`}
        ${path && html`<div class="orbx-scrub">
          <button class="btn small" onClick=${() => setPlaying(!playing)} title=${playing ? 'Pause' : 'Play'}>${playing ? '❚❚' : '▶'}</button>
          <input type="range" min="0" max=${n - 1} value=${k} onInput=${(e) => { setPlaying(false); setK(+e.target.value); }} />
          <span class="small muted">${k + 1}/${n}</span>
          <button class=${`btn small ${compare ? 'on' : ''}`} onClick=${() => setCompare(!compare)} title="The chosen orbitals at the first and last frame, side by side">Start ↔ end</button>
        </div>`}
        <div class="orbx-readout">
          ${(picks || []).length === 0 && html`<span class="muted">Click an orbital in a diagram to draw it; shift-click a second to draw both.</span>`}
          ${(picks || []).map((p, slot) => { const o = shownOrbs[slot]; return html`<span class="orbx-pick">
            <i class=${`sw ${slot ? 'pale' : ''}`}></i>${sysLabel(p.sys)} ${o ? html`<b>${orbName(o)}</b> ${fmtE(o.energy_ev)} eV` : html`<span class="muted">not followed at this frame</span>`}
            <button class="btn-icon tiny" title="Stop drawing it" onClick=${() => setPicks(picks.filter((q) => q !== p))}>✕</button></span>`; })}
          ${ov && shownOrbs[0] && shownOrbs[1] && html`<span class="orbx-pair-numbers">
            <span title=${pairing ? 'Two singly occupied orbitals, one electron each: overlapping, they pair into a bond' : 'How much the two share the same space (0 none, 1 identical): electrons only flow between orbitals that overlap. Same colours meeting = in phase.'}>
              ${pairing ? 'pair' : ''} overlap <b>${ov.listed ? Math.abs(ov.s).toFixed(3) : '< 0.005'}</b></span>
            <span title="Energy of the higher one minus the lower one (eV): the smaller, the more they mix"> · gap ${Math.abs(shownOrbs[1].energy_ev - shownOrbs[0].energy_ev).toFixed(2)} eV</span></span>`}
          <label class="small muted orbx-pct" title="Each surface encloses this share of the orbital's density">
            <input type="number" class="tiny" min="1" max="99.9" step="5" value=${pct}
              onChange=${(e) => { const v = +e.target.value; if (v > 0 && v < 100) setPct(v); }} /> % density</label>
        </div>
      </section>
      <section class="orbx-diagram">
        <div class="orbx-title">At this frame <span class="muted small">· click to draw, shift-click to add a second</span>
          <label class="small muted"><input type="checkbox" checked=${more} onChange=${(e) => { setMore(e.target.checked); prefs.set('orbitalMore', e.target.checked); }} /> more levels</label></div>
        ${fr ? html`<${Interaction} fr=${fr} frags=${frags} k=${k} picks=${picks || []} at=${at} more=${more}
          sysLabel=${sysLabel} sysColor=${sysColor} onPick=${click} />` : html`<div class="muted small">This frame failed.</div>`}
      </section>
      ${path && html`<section class="orbx-corr">
        <div class="orbx-title">Along the path <span class="muted small">· each line one orbital, followed by its overlap from frame to frame; dashed = empty</span>
          <span class="orbx-chips">${['w', ...frags.map((_, i) => i)].map((sys) => html`<button
            class=${`chip ${hidden[sys] ? '' : 'on'}`} style=${{ '--c': sysColor(sys) }}
            onClick=${() => setHidden({ ...hidden, [sys]: !hidden[sys] })}>${sysLabel(sys)}</button>`)}</span></div>
        <${Correlation} data=${data} k=${k} picks=${picks || []} more=${more} hidden=${hidden} frags=${frags}
          sysColor=${sysColor} sysLabel=${sysLabel} onFrame=${(kk) => { setPlaying(false); setK(kk); }}
          onPick=${(sys, line, kk, add) => { setPlaying(false); click(sys, linesOf(sys)[line].points.find((p) => p[0] === kk)?.[1], kk, add); }} />
      </section>`}
    </div>`}
  </div>`;
}

// ---- interaction diagram ----------------------------------------------------------
function Interaction({ fr, frags, picks, at, k, more, sysLabel, sysColor, onPick }) {
  const W = 460, H = 380, top = 30, bottom = 16, left = 44;
  const lo = more ? -4 : -1, hi = more ? 5 : 2;         // HOMO-1 .. LUMO+1 (more: HOMO-4 .. LUMO+4)
  // Columns: the first molecule, the whole system, then the other molecules.
  const cols = frags.length ? [0, 'w', ...frags.slice(1).map((_, i) => i + 1)] : ['w'];
  const colX = (c) => left + ((cols.indexOf(c) + 0.5) * (W - left)) / cols.length;
  const seg = Math.min(64, (W - left) / cols.length - 26);
  const sysOf = (sys) => (sys === 'w' ? fr.whole : fr.fragments[sys]);
  // The frontier region (HOMO..LUMO of every column ± 2.5 eV; "more levels": everything): far-off levels
  // would squeeze it.
  const fe = cols.flatMap((c) => sysOf(c).orbitals.filter((o) => o.rel === 0 || o.rel === 1).map((o) => o.energy_ev));
  const near = (e) => more || (e >= Math.min(...fe) - 2.5 && e <= Math.max(...fe) + 2.5);
  const shown = (sys) => sysOf(sys).orbitals.filter((o) => o.rel >= lo && o.rel <= hi && near(o.energy_ev));
  const all = cols.flatMap((c) => shown(c).map((o) => o.energy_ev));
  const eMin = Math.min(...all) - 0.4, eMax = Math.max(...all) + 0.4;
  const y = (e) => top + ((eMax - e) / (eMax - eMin || 1)) * (H - top - bottom);
  const picked = (sys, i) => picks.findIndex((p) => p.sys === sys && at(p, k)?.i === i);
  // Degenerate orbitals side by side.
  const place = (sys) => {
    const os = shown(sys).slice().sort((a, b) => a.energy_ev - b.energy_ev);
    const out = [];
    for (let s = 0; s < os.length;) {
      let e = s + 1;
      while (e < os.length && os[e].energy_ev - os[s].energy_ev < 0.02) e += 1;
      const m = e - s, w = (seg - (m - 1) * 4) / m;
      for (let j = s; j < e; j += 1) out.push({ o: os[j], x0: colX(sys) - seg / 2 + (j - s) * (w + 4), w });
      s = e;
    }
    return out;
  };
  const placed = Object.fromEntries(cols.map((c) => [c, place(c)]));
  // Labels (HOMO, LUMO, SOMO and the picked ones), spread apart where levels are close.
  const labelY = {};
  for (const c of cols) {
    const want = placed[c].filter(({ o }) => o.rel === 0 || o.rel === 1 || picked(c, o.i) >= 0)
      .map(({ o }) => ({ i: o.i, y: y(o.energy_ev) + 3 })).sort((a, b) => b.y - a.y);
    for (let j = 1; j < want.length; j += 1) want[j].y = Math.min(want[j].y, want[j - 1].y - 11);
    for (const w of want) labelY[`${c}|${w.i}`] = w.y;
  }
  const pos = (sys, i) => placed[sys]?.find((p) => p.o.i === i);
  // Lines from the molecules' orbitals to the whole orbitals they make up (weight ≥ 0.1, or any when picked).
  const links = [];
  for (const { o, x0, w } of placed.w || []) {
    for (const c of o.from || []) {
      const f = pos(c.fragment, c.i);
      if (!f || (c.w < 0.1 && picked('w', o.i) < 0 && picked(c.fragment, c.i) < 0)) continue;
      const fromRight = cols.indexOf(c.fragment) < cols.indexOf('w');
      links.push({ x1: fromRight ? f.x0 + f.w : f.x0, y1: y(f.o.energy_ev), x2: fromRight ? x0 : x0 + w, y2: y(o.energy_ev), w: c.w, color: sysColor(c.fragment) });
    }
  }
  const ticks = [];
  for (let e = Math.ceil(eMin); e <= Math.floor(eMax); e += Math.max(1, Math.round((eMax - eMin) / 6))) ticks.push(e);
  const tip = (sys, o) => {
    let t = `${sysLabel(sys)} ${orbName(o)}: ${fmtE(o.energy_ev)} eV, ${o.occ.toFixed(2)} electrons`;
    if (sys === 'w' && o.from?.length) t += `\nMade of ${o.from.slice().sort((a, b) => b.w - a.w).map((c) => {
      const f = fr.fragments[c.fragment].orbitals.find((x) => x.i === c.i);
      return `${Math.round(c.w * 100)}% ${sysLabel(c.fragment)} ${f ? orbName(f) : `orbital ${c.i}`}`; }).join(' + ')}`;
    return `${t}\nClick to draw; shift-click to add`;
  };
  return html`<svg class="orbx-svg" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label="Orbital interaction diagram">
    ${ticks.map((e) => html`<g><line x1=${left - 4} x2=${left} y1=${y(e)} y2=${y(e)} class="axis" />
      <text x=${left - 6} y=${y(e) + 3} class="tick" text-anchor="end">${fmtE(e).replace('.00', '')}</text></g>`)}
    <text x="4" y="12" class="tick">eV</text>
    ${cols.map((c) => html`<text x=${colX(c)} y="14" text-anchor="middle" class="colhead" fill=${sysColor(c)}>${sysLabel(c)}</text>`)}
    ${links.map((l) => html`<line x1=${l.x1} y1=${l.y1} x2=${l.x2} y2=${l.y2} stroke=${l.color}
      stroke-width=${0.6 + 4 * l.w} stroke-opacity=${0.2 + 0.6 * l.w} />`)}
    ${cols.map((c) => placed[c].map(({ o, x0, w }) => {
      const sel = picked(c, o.i);
      const filled = o.occ > 0.5;
      const ly = labelY[`${c}|${o.i}`];
      const label = ly != null;
      return html`<g class="lvl" onClick=${(e) => onPick(c, o.i, k, e.shiftKey || e.metaKey || e.ctrlKey)}>
        <title>${tip(c, o)}</title>
        <rect x=${x0 - 2} y=${y(o.energy_ev) - 6} width=${w + 4} height="12" fill="transparent" />
        ${sel >= 0 && html`<rect x=${x0 - 3} y=${y(o.energy_ev) - 5} width=${w + 6} height="10" rx="2" class=${`hl ${sel ? 'pale' : ''}`} />`}
        <line x1=${x0} x2=${x0 + w} y1=${y(o.energy_ev)} y2=${y(o.energy_ev)} stroke=${sysColor(c)}
          stroke-width=${sel >= 0 ? 3 : 2} stroke-dasharray=${filled ? '' : '4 3'} />
        ${o.occ > 0.5 && o.occ < 1.95 && html`<circle cx=${x0 + w / 2} cy=${y(o.energy_ev) - 4} r="2" fill=${sysColor(c)} />`}
        ${label && html`<text x=${cols.indexOf(c) < cols.length - 1 || cols.length === 1 ? x0 + w + 4 : x0 - 4} y=${ly}
          text-anchor=${cols.indexOf(c) < cols.length - 1 || cols.length === 1 ? 'start' : 'end'} class="lab">${orbName(o)}</text>`}
      </g>`;
    }))}
  </svg>`;
}

// ---- correlation diagram ----------------------------------------------------------
function Correlation({ data, k, picks, more, hidden, frags, sysColor, sysLabel, onFrame, onPick }) {
  const W = 900, H = 240, left = 44, right = 8, top = 8, bottom = 20;
  const svg = useRef(null);
  const n = data.frames.length;
  const lo = more ? -4 : -1, hi = more ? 5 : 2;
  const systems = ['w', ...frags.map((_, i) => i)].filter((s) => !hidden[s]);
  const orbAt = (sys, kk, i) => {
    const fr = data.frames[kk];
    const s = fr && (sys === 'w' ? fr.whole : fr.fragments[sys]);
    return s?.orbitals.find((o) => o.i === i) || null;
  };
  // Lines to draw: those inside the level window somewhere along the path, and (unless "more levels")
  // inside the frontier region (every HOMO..LUMO ± 2.5 eV) somewhere.
  const all = [];
  for (const sys of systems) {
    const lines = sys === 'w' ? data.lines.whole : data.lines.fragments[sys];
    lines.forEach((ln, li) => {
      const pts = ln.points.map(([kk, i]) => ({ kk, o: orbAt(sys, kk, i) })).filter((p) => p.o);
      if (pts.some((p) => p.o.rel >= lo && p.o.rel <= hi)) all.push({ sys, li, pts });
    });
  }
  const fe = all.flatMap((d) => d.pts.filter((p) => p.o.rel === 0 || p.o.rel === 1).map((p) => p.o.energy_ev));
  const near = (e) => more || !fe.length || (e >= Math.min(...fe) - 2.5 && e <= Math.max(...fe) + 2.5);
  const drawn = all.filter((d) => d.pts.some((p) => p.o.rel >= lo && p.o.rel <= hi && near(p.o.energy_ev)));
  const es = drawn.flatMap((d) => d.pts.filter((p) => p.o.rel >= lo && p.o.rel <= hi && near(p.o.energy_ev)).map((p) => p.o.energy_ev));
  if (!es.length) return html`<div class="muted small">No orbitals to follow.</div>`;
  const eMin = Math.min(...es) - 0.3, eMax = Math.max(...es) + 0.3;
  const x = (kk) => left + (n > 1 ? (kk / (n - 1)) * (W - left - right) : 0);
  const y = (e) => top + ((eMax - Math.min(eMax, Math.max(eMin, e))) / (eMax - eMin || 1)) * (H - top - bottom);
  const isPicked = (sys, li) => picks.findIndex((p) => p.sys === sys && p.line === li);
  const ticks = [];
  for (let e = Math.ceil(eMin); e <= Math.floor(eMax); e += Math.max(1, Math.round((eMax - eMin) / 5))) ticks.push(e);
  const frameAt = (ev) => {
    const r = svg.current.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    return Math.max(0, Math.min(n - 1, Math.round(((px - left) / (W - left - right)) * (n - 1))));
  };
  // Segments: solid where the orbital is filled, dashed where empty (a filled orbital becoming empty along the path
  // is an orbital-symmetry crossing).
  const segs = (d) => {
    const out = [];
    for (let j = 1; j < d.pts.length; j += 1) {
      const a = d.pts[j - 1], b = d.pts[j];
      if (b.kk !== a.kk + 1) continue;
      const inside = (e) => e >= eMin && e <= eMax;
      if (!inside(a.o.energy_ev) || !inside(b.o.energy_ev)) continue;   // off the diagram's energy range
      out.push({ a, b, filled: a.o.occ > 0.5 && b.o.occ > 0.5 });
    }
    return out;
  };
  return html`<svg ref=${svg} class="orbx-svg corr" viewBox=${`0 0 ${W} ${H}`} preserveAspectRatio="none"
    onClick=${(e) => { if (e.target === svg.current || e.target.classList.contains('bg')) onFrame(frameAt(e)); }}>
    <rect class="bg" x=${left} y=${top} width=${W - left - right} height=${H - top - bottom} fill="transparent" />
    ${ticks.map((e) => html`<g><line x1=${left} x2=${W - right} y1=${y(e)} y2=${y(e)} class="grid" />
      <text x=${left - 6} y=${y(e) + 3} class="tick" text-anchor="end">${fmtE(e).replace('.00', '')}</text></g>`)}
    <text x=${left} y=${H - 4} class="tick">start</text>
    <text x=${W - right} y=${H - 4} class="tick" text-anchor="end">end</text>
    ${drawn.map((d) => {
      const sel = isPicked(d.sys, d.li);
      const first = d.pts.find((p) => p.kk === k) || d.pts[0];
      return html`<g class="trk" onClick=${(e) => { e.stopPropagation(); onPick(d.sys, d.li, d.pts.some((p) => p.kk === k) ? k : first.kk, e.shiftKey || e.metaKey || e.ctrlKey); }}>
        <title>${`${sysLabel(d.sys)} ${orbName(first.o)} (at ${d.pts.some((p) => p.kk === k) ? 'this frame' : `frame ${first.kk + 1}`}): click to draw, shift-click to add`}</title>
        ${segs(d).map((s) => html`<line x1=${x(s.a.kk)} y1=${y(s.a.o.energy_ev)} x2=${x(s.b.kk)} y2=${y(s.b.o.energy_ev)}
          stroke=${sysColor(d.sys)} stroke-width=${sel >= 0 ? 3 : 1.3} stroke-dasharray=${s.filled ? '' : '4 3'}
          stroke-opacity=${sel >= 0 || !picks.length ? 0.95 : 0.55} />`)}
        ${segs(d).map((s) => html`<line x1=${x(s.a.kk)} y1=${y(s.a.o.energy_ev)} x2=${x(s.b.kk)} y2=${y(s.b.o.energy_ev)}
          stroke="transparent" stroke-width="8" />`)}
      </g>`;
    })}
    <line x1=${x(k)} x2=${x(k)} y1=${top} y2=${H - bottom} class="now" />
    ${picks.map((p, slot) => {
      if (p.line == null) return null;
      const d = drawn.find((q) => q.sys === p.sys && q.li === p.line);
      const pt = d?.pts.find((q) => q.kk === k);
      return pt && html`<circle cx=${x(k)} cy=${y(pt.o.energy_ev)} r="4" class=${`dot ${slot ? 'pale' : ''}`} />`;
    })}
  </svg>`;
}
