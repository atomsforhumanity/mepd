// The interactive reactor (mepd.web.sandbox): a live MD you steer. Drag an
// atom and it is pulled toward the pointer (the slider sets how hard, Shift
// pulls harder); the scroll wheel grows or shrinks the wall (Ctrl+scroll
// zooms); temperature, pause, and the current frame into Explore.
import { html, useEffect, useRef, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, set, state, useStore } from '../store.js';
import { depictUrl } from '../util.js';
import { FULL, cssVar, useStage, xyzText } from './ReactorLive.js';
import { StepBar, markStep } from './Welcome.js';
import { Why } from './Why.js';

// After this long with no reaction, a hint on how to make one (left alone, few do).
const NUDGE_FS = 4000;

const PICK_PX = 18;      // how close (screen pixels) a press must be to an atom to grab it
const RECENT_FS = 400;   // an event stays marked in the list this long after it starts (and while it is unfolding)

const logSlider = (v, lo, hi) => Math.round(Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * v));
// The wall's radius (Angstrom) on a log slider 0..1, the same range the scroll wheel covers.
const WALL_MIN = 3, WALL_MAX = 60;
const COARSE = window.matchMedia?.('(pointer: coarse)').matches;   // a touch screen: no scroll wheel
const wallOf = (v) => WALL_MIN * Math.exp(Math.log(WALL_MAX / WALL_MIN) * v);
const sliderOf = (r) => Math.log(r / WALL_MIN) / Math.log(WALL_MAX / WALL_MIN);

// "C=O + [H] -> [CH]=O + [H][H]" (a reaction label) -> each side's molecules,
// a count each, with the ones on both sides (water carrying an H across) left out.
function sidesOf(label) {
  const side = (t) => t.split(' + ').map((x) => { const m = x.trim().match(/^(\d+) (.+)$/); return m ? { n: +m[1], smi: m[2] } : { n: 1, smi: x.trim() }; });
  const [l, r] = (label || '').split(' -> ');
  if (r == null) return null;
  const a = side(l), b = side(r);
  const both = new Set(a.map((x) => x.smi).filter((smi) => b.some((y) => y.smi === smi)));
  const keep = (xs) => xs.filter((x) => !both.has(x.smi));
  return keep(a).length && keep(b).length ? [keep(a), keep(b)] : [a, b];
}
function EventPics({ label }) {
  const sides = sidesOf(label);
  if (!sides) return html`<span class="rx-eq">${label}</span>`;
  const mol = (x) => { const u = depictUrl(x.smi, 120, 72); return html`<span class="rx-mol" title=${x.smi}>${x.n > 1 && html`<span class="rx-n">${x.n}×</span>`}${u ? html`<img src=${u} alt=${x.smi} loading="lazy" />` : x.smi}</span>`; };
  return html`<span class="rx-pics">${sides[0].map(mol)}<span class="rx-arrow">→</span>${sides[1].map(mol)}</span>`;
}

export function SandboxView() {
  const sb = useStore((s) => s.view.sandbox);
  const host = useRef(null);
  const stage = useStage(host);
  const frame = useRef(null);
  const drag = useRef(null);            // {atom, target: [x,y,z], k}
  const radius = useRef(null);
  const sentAt = useRef(0);
  const [hud, setHud] = useState(null);
  const [strength, setStrength] = useState(0.5);      // slider 0..1 -> 2..300 kcal/mol/A^2
  const [temperature, setTemperature] = useState(() => sb?.temperature ?? 800);
  const [paused, setPaused] = useState(false);
  const [tipGone, setTipGone] = useState(false);   // the "Nothing yet?" hint, once tapped away
  const [wall, setWall] = useState(null);             // the wall radius shown by its slider (radius.current is the truth)
  const [dead, setDead] = useState(null);
  const tMean = useRef(null);
  const [events, setEvents] = useState([]);
  const eventsRef = useRef([]);
  // Until someone has caught a first reaction, Collide keeps going by itself.
  const [autoCollide, setAutoCollide] = useState(() => !state.tour?.reaction);
  // The piston: off at first; its cycle (as the nanoreactor's) behind "cycle".
  const [piston, setPiston] = useState({ on: false, period_ps: 1, compress: 0.7, duty: 0.75 });
  const [cycleOpen, setCycleOpen] = useState(false);
  const setPistonTo = (next) => {
    setPiston(next);
    send({ op: 'piston', value: next.on, period_ps: next.period_ps, compress: next.compress, duty: next.duty });
  };
  const strengthRef = useRef(strength);
  strengthRef.current = strength;

  const send = (cmd) => api.post(`/api/sandbox/${sb.id}/command`, cmd).catch((e) => setDead(String(e.message || e)));
  // Collide is the unbiased network generator's move, run inside the reactor
  // (mepd/web/sandbox.py, reactor_generator.collision_pair + langevin.pull_pairs):
  // two molecules picked at random are pushed into each other as wholes for
  // 300 fs of dynamics (a capped force along one atom of each, spread over each
  // molecule by mass), or, with one molecule or 30% of the time, two of its atoms
  // pulled together. "keep colliding" repeats it on the generator's own schedule.
  const collide = () => send({ op: 'collide' });
  useEffect(() => {
    if (!sb) return undefined;
    send({ op: 'auto_collide', value: !!autoCollide });
    return undefined;
  }, [autoCollide, sb?.id]);

  // Draw the latest frame (positions, the wall, the pull lines).
  const draw = () => {
    const f = frame.current;
    const v = stage.viewer() || null;
    if (!f || !f.pos) return;
    const flat = f.pos.flat();
    const wide = radius.current ?? f.radius;
    const r = f.piston && f.wall_now != null ? f.wall_now : wide;   // the piston's wall, in and out
    const pulls = { ...(f.pulls || {}) };
    if (drag.current) pulls[drag.current.atom] = drag.current.target;
    stage.draw((vv) => {
      vv.addModel(xyzText(sb.symbols, flat), 'xyz');
      vv.setStyle({}, FULL);
      if (drag.current) vv.setStyle({ index: drag.current.atom }, { stick: { radius: 0.12 }, sphere: { scale: 0.35, color: cssVar('--warn', '#b7791f') } });
      if (f.collide?.a) {   // the two molecules being pushed together
        vv.setStyle({ index: f.collide.a }, { stick: { radius: 0.12, color: cssVar('--chart-1', '#3868b8') }, sphere: { scale: 0.26 } });
        vv.setStyle({ index: f.collide.b }, { stick: { radius: 0.12, color: cssVar('--chart-2', '#c26a1f') }, sphere: { scale: 0.26 } });
      }
    }, { extent: wide + 1, shapes: Object.keys(pulls).length > 0 || f.collide?.i != null });
    stage.setWall(r, cssVar('--accent', '#3868b8'));
    const viewer = stage.viewer();
    if (viewer && f.collide?.i != null) {   // the line the push acts along
      const p = f.pos[f.collide.i], q = f.pos[f.collide.j];
      viewer.addCylinder({ start: { x: p[0], y: p[1], z: p[2] }, end: { x: q[0], y: q[1], z: q[2] }, radius: 0.07,
        dashed: true, color: cssVar('--accent', '#3868b8'), fromCap: 1, toCap: 1 });
      if (!Object.keys(pulls).length) viewer.render();
    }
    if (viewer && Object.keys(pulls).length) {
      for (const [atom, t] of Object.entries(pulls)) {
        const p = f.pos[+atom];
        viewer.addCylinder({ start: { x: p[0], y: p[1], z: p[2] }, end: { x: t[0], y: t[1], z: t[2] }, radius: 0.06,
          dashed: true, color: cssVar('--warn', '#b7791f'), fromCap: 1, toCap: 1 });
      }
      viewer.render();
    }
    if (host.current) host.current._sandbox = { viewer: stage.viewer(), frame: f };   // for tests and debugging
    void v;
  };

  // Frames: a long-poll loop for as long as the view is open.
  useEffect(() => {
    if (!sb) return undefined;
    let alive = true, seq = 0, lastHud = 0, eventsRev = 0;
    (async () => {
      while (alive) {
        try {
          const f = await api.get(`/api/sandbox/${sb.id}/frame?since=${seq}`);
          if (!alive) break;
          if (f.error || !f.alive) { setDead(f.error || 'This interactive reactor has stopped.'); break; }
          if ((f.events_rev || 0) > eventsRev) {
            eventsRev = f.events_rev;
            api.get(`/api/sandbox/${sb.id}/events`).then((ev) => { eventsRef.current = ev.events; setEvents(ev.events); }).catch(() => {});
          }
          if (f.seq > seq) {
            seq = f.seq;
            frame.current = f;
            if (radius.current == null) {   // the first frame: the wall and the piston as the reactor has them
              radius.current = f.radius; setWall(f.radius);
              if (f.piston) setPiston({ on: true, ...f.piston });
            }
            draw();
            // The thermometer swings far around the setting (and Collide heats it): show a running mean.
            if (f.T_inst != null) tMean.current = tMean.current == null ? f.T_inst : 0.92 * tMean.current + 0.08 * f.T_inst;
            if (Date.now() - lastHud > 250) { lastHud = Date.now(); setHud(f); }
          }
        } catch (e) {
          if (alive) setDead(String(e.message || e));
          break;
        }
      }
    })();
    return () => { alive = false; };
  }, [sb?.id]);

  // Pointer: grab the nearest atom (else let 3Dmol rotate), pull it toward
  // the pointer in the screen plane through the atom; scroll = wall radius.
  useEffect(() => {
    const el = host.current;
    if (!el || !sb) return undefined;
    // 3Dmol's modelToScreen gives page coordinates (CSS pixels, the canvas's
    // offset on the page included): compare them with the pointer's page
    // position, not with one measured from the viewer's corner.
    const local = (e) => [e.clientX + window.scrollX, e.clientY + window.scrollY];
    const screenOf = (v, p) => { const s = v.modelToScreen({ x: p[0], y: p[1], z: p[2] }); return [s.x, s.y]; };
    const kNow = (e) => logSlider(strengthRef.current, 2, 300) * (e.shiftKey ? 5 : 1);
    const targetFor = (v, atom, e) => {
      const p = frame.current.pos[atom];
      const [ax, ay] = screenOf(v, p);
      const [mx, my] = local(e);
      const off = v.screenOffsetToModel(mx - ax, my - ay, p[2]);
      return [p[0] + off.x, p[1] + off.y, p[2] + off.z];
    };
    // Pointer events: a mouse, a pen or a finger. A finger also makes touch
    // events, which 3Dmol turns into rotation: swallow them while an atom is held.
    const down = (e) => {
      const v = stage.viewer();
      const f = frame.current;
      if (!v || !f?.pos || e.button !== 0 || !e.isPrimary) return;
      const [mx, my] = local(e);
      let best = -1, bestD = e.pointerType === 'touch' ? PICK_PX * 1.6 : PICK_PX;   // fingers are wider
      f.pos.forEach((p, i) => {
        const [sx, sy] = screenOf(v, p);
        const d = Math.hypot(sx - mx, sy - my);
        if (d < bestD) { bestD = d; best = i; }
      });
      if (best < 0) return;                    // not on an atom: 3Dmol rotates
      e.stopPropagation(); e.preventDefault();
      drag.current = { atom: best, target: f.pos[best].slice(), k: kNow(e) };
      draw();
    };
    const move = (e) => {
      if (!drag.current) return;
      e.stopPropagation(); e.preventDefault();
      const v = stage.viewer();
      drag.current.target = targetFor(v, drag.current.atom, e);
      drag.current.k = kNow(e);
      if (Date.now() - sentAt.current > 40) {
        sentAt.current = Date.now();
        send({ op: 'pull', atom: drag.current.atom, target: drag.current.target, k: drag.current.k });
      }
      draw();
    };
    const up = (e) => {
      if (!drag.current) return;
      e.stopPropagation(); e.preventDefault();
      send({ op: 'release', atom: drag.current.atom });
      drag.current = null;
      draw();
    };
    const wheel = (e) => {
      if (e.ctrlKey || radius.current == null) return;    // Ctrl+scroll: 3Dmol zooms
      e.stopPropagation(); e.preventDefault();
      radius.current = Math.min(WALL_MAX, Math.max(WALL_MIN, radius.current * Math.exp(-Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 120) * 0.0004)));
      setWall(radius.current);
      if (Date.now() - sentAt.current > 40) { sentAt.current = Date.now(); send({ op: 'radius', value: radius.current }); }
      draw();
    };
    const wheelEnd = () => { if (radius.current != null) send({ op: 'radius', value: radius.current }); };
    let wheelTimer = null;
    const wheelAll = (e) => { wheel(e); clearTimeout(wheelTimer); wheelTimer = setTimeout(wheelEnd, 120); };
    const holdTouch = (e) => { if (drag.current) { e.stopPropagation(); e.preventDefault(); } };
    el.addEventListener('pointerdown', down, true);
    window.addEventListener('pointermove', move, true);
    window.addEventListener('pointerup', up, true);
    window.addEventListener('pointercancel', up, true);
    el.addEventListener('touchstart', holdTouch, { capture: true, passive: false });
    el.addEventListener('touchmove', holdTouch, { capture: true, passive: false });
    el.addEventListener('wheel', wheelAll, { capture: true, passive: false });
    return () => {
      el.removeEventListener('pointerdown', down, true);
      window.removeEventListener('pointermove', move, true);
      window.removeEventListener('pointerup', up, true);
      window.removeEventListener('pointercancel', up, true);
      el.removeEventListener('touchstart', holdTouch, true);
      el.removeEventListener('touchmove', holdTouch, true);
      el.removeEventListener('wheel', wheelAll, true);
    };
  }, [sb?.id, host.current]);

  // A reaction (bonds that changed and stayed changed): the third step is done.
  const reactions = events.filter((e) => !e.tentative && e.reaction != null);
  useEffect(() => { if (reactions.length) markStep('reaction'); }, [reactions.length]);

  const shown = [...events].filter((e) => e.tentative || e.reaction != null).sort((a, b) => b.start_fs - a.start_fs);
  const swaps = events.filter((e) => !e.tentative && e.reaction == null).length;
  if (!sb) return html`<div class="empty-hint"><p>No interactive reactor is open.</p></div>`;
  const k = logSlider(strength, 2, 300);
  const gone = () => ({ sandboxes: (state.sandboxes || []).filter((x) => x.id !== sb.id) });
  const discard = () => {
    if (events.length && !window.confirm(`Discard this run and its ${events.length} reaction event(s)? Nothing is kept.`)) return;
    api.del(`/api/sandbox/${sb.id}`).catch(() => {}).then(() => set({ view: { tab: 'graph', jobId: null }, ...gone() }));
  };
  const analyze = async () => {
    const out = await attempt(() => api.post(`/api/sandbox/${sb.id}/analyze`),
      'Analyzing your run');
    if (out) { set(gone()); openJob(out.job, 'story'); }
  };
  return html`<div class="job-view sandbox">
    <div class="view-head">
      <button class="btn-icon" title="Back to Explore (it keeps running for 15 minutes; the ◉ button in the top bar brings you back)" onClick=${() => set({ view: { tab: 'graph', jobId: null } })}>←</button>
      <div class="job-head-main"><h2>Interactive reactor<${Why} id="other.reactor" label="The interactive reactor" /></h2>
        <div class="small muted"><b class="sb-names">${sb.names}</b> · <span class="has-tip" title="The reactor runs GFN2-xTB because it is fast. Barriers and energies afterwards use your session's level of theory.">GFN2-xTB</span> · ${sb.symbols.length} atoms · charge ${sb.charge}${sb.multiplicity > 1 ? ` · multiplicity ${sb.multiplicity}` : ''}</div>
        <${StepBar} at=${reactions.length ? 'barrier' : 'reactor'} /></div>
      <div class="row-actions">
        ${!reactions.length && html`<button class="btn" onClick=${analyze}
          title="Stop and keep the run: any reactions are refined, each gets a barrier (TS search), and all go to Explore">Stop & analyze</button>`}
        <button class="btn-link small danger-text" onClick=${discard} title="Stop without keeping anything">Discard</button>
      </div>
    </div>
    ${dead && html`<p class="warn-box small">${dead}</p>`}
    <div class="rx-live">
      <div class="rx-main">
        <div class="rx-stage-wrap">
          <div class="rx-stage sandbox-stage" ref=${host}></div>
          <div class="rx-hud" title=${hud ? `Simulated time ${(hud.t_fs / 1000).toFixed(2)} ps, running at ${hud.rate ?? 0} fs per second. Temperature ≈${tMean.current != null ? Math.round(tMean.current) : '–'} K averaged (right now ${hud.T_inst ?? '–'} K: it swings around the setting, and Collide adds heat). Energy ${hud.e_rel ?? 0} kcal/mol above the start. Wall radius ${(radius.current ?? hud.radius).toFixed(1)} Å.` : ''}>${hud ? `${(hud.t_fs / 1000).toFixed(1)} ps · ≈${tMean.current != null ? Math.round(tMean.current) : '–'} K${hud.paused ? ' · paused' : ''}` : 'starting…'}</div>
          ${reactions.length > 0 && html`<button class="rx-count" title="See the reactions"
            onClick=${(e) => e.currentTarget.closest('.rx-live').querySelector('.rx-events').scrollIntoView({ behavior: 'smooth', block: 'start' })}>
            ⚡ ${reactions.length} reaction${reactions.length > 1 ? 's' : ''}</button>`}
          ${hud && !events.length && !tipGone && hud.t_fs > NUDGE_FS && html`<div class="rx-tip" role="status" title="Tap to hide" onClick=${() => setTipGone(true)}>
            <b>Nothing yet?</b> Molecules rarely react by just bumping around. Press <b>Collide</b>, or drag an atom
            of one molecule into another and hold it there.</div>`}
          <div class="rx-legend small">drag an atom into another: pull it · drag elsewhere: rotate</div>
          <div class="rx-collide">
            <button class="btn primary key" onClick=${collide} title="Pull two atoms together for a moment, as you would by dragging one: of two molecules, or far apart in one">Collide</button>
            <label class="small" title="Push molecules together again and again, on the network generator's schedule (300 fs pushes, ~200 fs apart)"><input type="checkbox" checked=${autoCollide}
              onChange=${(e) => setAutoCollide(e.target.checked)} /> keep colliding</label>
          </div>
        </div>
        <div class="sandbox-controls">
          <label class="small">Pull strength <input type="range" min="0" max="1" step="0.01" value=${strength}
            onInput=${(e) => setStrength(+e.target.value)} /> <span class="mono">${k} kcal/mol/Å²</span></label>
          <label class="small">Temperature <input type="range" min="0" max="3000" step="50" value=${temperature}
            onInput=${(e) => setTemperature(+e.target.value)} onChange=${(e) => send({ op: 'temperature', value: +e.target.value })} />
            <span class="mono">${temperature} K</span></label>
          <label class="small" title="The wall's radius: smaller squeezes the molecules together (the scroll wheel does the same)">Wall
            <input type="range" min="0" max="1" step="0.005" value=${wall != null ? sliderOf(wall) : 0.3} disabled=${wall == null}
              onInput=${(e) => {
                radius.current = wallOf(+e.target.value);
                setWall(radius.current);
                if (Date.now() - sentAt.current > 40) { sentAt.current = Date.now(); send({ op: 'radius', value: radius.current }); }
                draw();
              }}
              onChange=${(e) => send({ op: 'radius', value: wallOf(+e.target.value) })} />
            <span class="mono">${wall != null ? wall.toFixed(1) : '–'} Å</span></label>
          <span class="piston-ctl">
            <label class="small" title="Squeeze the wall in and out, again and again, as the nanoreactor does"><input type="checkbox" class="switch"
              checked=${piston.on} onChange=${(e) => setPistonTo({ ...piston, on: e.target.checked })} /> Piston</label>
            ${piston.on && html`<button class="btn-link small" aria-expanded=${cycleOpen} onClick=${() => setCycleOpen(!cycleOpen)}
              title="How often it squeezes, how far, and for how long">cycle ${cycleOpen ? '▾' : '▸'}</button>`}
            ${piston.on && cycleOpen && html`<span class="piston-fields small">
              <label title="One squeeze and release">every <input type="number" class="tiny" min="0.05" step="0.1" value=${piston.period_ps}
                onChange=${(e) => +e.target.value > 0 && setPistonTo({ ...piston, period_ps: +e.target.value })} /> ps</label>
              <label title="The squeezed wall, as a share of the Wall radius">in to <input type="number" class="tiny" min="5" max="100" step="5" value=${Math.round(piston.compress * 100)}
                onChange=${(e) => +e.target.value > 0 && +e.target.value <= 100 && setPistonTo({ ...piston, compress: +e.target.value / 100 })} /> %</label>
              <label title="How much of each cycle the wall stays open">open <input type="number" class="tiny" min="5" max="95" step="5" value=${Math.round(piston.duty * 100)}
                onChange=${(e) => +e.target.value > 0 && +e.target.value < 100 && setPistonTo({ ...piston, duty: +e.target.value / 100 })} /> % of it</label>
            </span>`}
          </span>
          <button class="btn small" onClick=${() => { send({ op: 'pause', value: !paused }); setPaused(!paused); }}>${paused ? '▶ Resume' : '❚❚ Pause'}</button>
          <button class="btn small" onClick=${() => { setAutoCollide(false); send({ op: 'release_all' }); }}>Release all</button>
          <button class="btn small" title="Keep this geometry in Explore (not minimized; several molecules become a complex)"
            onClick=${() => attempt(() => api.post(`/api/sandbox/${sb.id}/snapshot`), 'This frame added to Explore')}>Save this frame</button>
        </div>
      </div>
      <aside class="rx-events">
        <div class="section-title">Reactions · ${reactions.length}</div>
        ${!shown.length && html`<p class="small muted">None yet. When bonds break or form and stay that way, the reaction shows up here.</p>`}
        <ol>
          ${shown.map((e) => html`<li class=${hud && (e.tentative || hud.t_fs - e.start_fs < RECENT_FS) ? 'on' : ''}>
            <div class="rx-ev" title=${`${e.label} · at ${(e.start_fs / 1000).toFixed(2)} ps`}>
              <${EventPics} label=${e.label} />
              ${e.tentative && html`<span class="small rx-analyzing" title="Still unfolding: its bonds may change again">forming…</span>`}
            </div></li>`)}
        </ol>
        ${swaps > 0 && html`<p class="small muted" title="An atom (usually H) passed from one molecule to an identical one: nothing new was made">+ ${swaps} H swap${swaps > 1 ? 's' : ''} between identical molecules</p>`}
        ${reactions.length > 0 && html`<button class="btn primary rx-find" onClick=${analyze}
          title="Stop and keep the run: its reactions are refined, each gets a barrier (TS search), and all go to Explore">
          Find ${reactions.length > 1 ? 'their barriers' : 'its barrier'}</button>`}
        <details class="small"><summary>How to play</summary>
        <ul class="small muted how-to-play">
          <li>Press Collide, or drag an atom into another molecule (Shift: harder).</li>
          <li>Shrink the Wall (or scroll) to squeeze them.</li>
          <li>Heat it up with Temperature.</li>
          <li>Turn on Piston to squeeze them again and again.</li>
        </ul>
        </details>
      </aside>
    </div>
  </div>`;
}
