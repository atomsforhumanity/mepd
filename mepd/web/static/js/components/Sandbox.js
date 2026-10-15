// The interactive reactor (mepd.web.sandbox): a live MD you steer. Drag an
// atom and it is pulled toward the pointer (the slider sets how hard, Shift
// pulls harder); the scroll wheel grows or shrinks the wall (Ctrl+scroll
// zooms); temperature, pause, and the current frame into Explore.
import { html, useEffect, useRef, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, set, state, useStore } from '../store.js';
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
  const [events, setEvents] = useState([]);
  const eventsRef = useRef([]);
  const strengthRef = useRef(strength);
  strengthRef.current = strength;

  const send = (cmd) => api.post(`/api/sandbox/${sb.id}/command`, cmd).catch((e) => setDead(String(e.message || e)));

  // Draw the latest frame (positions, the wall, the pull lines).
  const draw = () => {
    const f = frame.current;
    const v = stage.viewer() || null;
    if (!f || !f.pos) return;
    const flat = f.pos.flat();
    const r = radius.current ?? f.radius;
    const pulls = { ...(f.pulls || {}) };
    if (drag.current) pulls[drag.current.atom] = drag.current.target;
    stage.draw((vv) => {
      vv.addModel(xyzText(sb.symbols, flat), 'xyz');
      vv.setStyle({}, FULL);
      if (drag.current) vv.setStyle({ index: drag.current.atom }, { stick: { radius: 0.12 }, sphere: { scale: 0.35, color: cssVar('--warn', '#b7791f') } });
    }, { extent: r + 1, shapes: Object.keys(pulls).length > 0 });
    stage.setWall(r, cssVar('--accent', '#3868b8'));
    const viewer = stage.viewer();
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
            if (radius.current == null) { radius.current = f.radius; setWall(f.radius); }
            draw();
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

  if (!sb) return html`<div class="empty-hint"><p>No interactive reactor is open.</p></div>`;
  const k = logSlider(strength, 2, 300);
  const gone = () => ({ sandboxes: (state.sandboxes || []).filter((x) => x.id !== sb.id) });
  const discard = () => {
    if (events.length && !window.confirm(`Discard this run and its ${events.length} reaction event(s)? Nothing is kept.`)) return;
    api.del(`/api/sandbox/${sb.id}`).catch(() => {}).then(() => set({ view: { tab: 'graph', jobId: null }, ...gone() }));
  };
  const analyze = async () => {
    const out = await attempt(() => api.post(`/api/sandbox/${sb.id}/analyze`),
      'Analyzing the run: watch each species and reaction being optimized; each reaction then gets its own TS search');
    if (out) { set(gone()); openJob(out.job, 'story'); }
  };
  return html`<div class="job-view sandbox">
    <div class="view-head">
      <button class="btn-icon" title="Back to Explore (it keeps running for 15 minutes; the ◉ button in the top bar brings you back)" onClick=${() => set({ view: { tab: 'graph', jobId: null } })}>←</button>
      <div class="job-head-main"><h2>Interactive reactor<${Why} id="other.reactor" label="The interactive reactor" /></h2>
        <div class="small muted"><b class="sb-names">${sb.names}</b> · GFN2-xTB · ${sb.symbols.length} atoms · charge ${sb.charge}${sb.multiplicity > 1 ? ` · multiplicity ${sb.multiplicity}` : ''}</div>
        <${StepBar} at=${reactions.length ? 'reaction' : 'reactor'} /></div>
      <div class="row-actions">
        <button class="btn" title="Keep this geometry (not minimized; several molecules become a complex)"
          onClick=${() => attempt(() => api.post(`/api/sandbox/${sb.id}/snapshot`), 'Frame added to Explore')}>Add frame to Explore</button>
        <button class="btn primary" onClick=${analyze}
          title="Stop and keep the run: its reactions are refined, each gets a barrier (TS search), and all go to Explore">Stop & analyze</button>
        <button class="btn danger-outline" onClick=${discard} title="Stop without keeping anything">Discard</button>
      </div>
    </div>
    ${dead && html`<p class="warn-box small">${dead}</p>`}
    <div class="rx-live">
      <div class="rx-main">
        <div class="rx-stage-wrap">
          <div class="rx-stage sandbox-stage" ref=${host}></div>
          <div class="rx-hud">${hud ? `${(hud.t_fs / 1000).toFixed(2)} ps · ${hud.rate ?? 0} fs/s · ${hud.T_inst ?? '–'} K · ΔE ${hud.e_rel ?? 0} kcal/mol · wall ${(radius.current ?? hud.radius).toFixed(1)} Å${hud.paused ? ' · paused' : ''}` : 'starting…'}</div>
          ${events.length > 0 && html`<button class="rx-count" title="See the reaction events"
            onClick=${(e) => e.currentTarget.closest('.rx-live').querySelector('.rx-events').scrollIntoView({ behavior: 'smooth', block: 'start' })}>
            ${reactions.length ? `⚡ ${reactions.length} reaction${reactions.length > 1 ? 's' : ''}` : `${events.length} event${events.length > 1 ? 's' : ''}`}</button>`}
          ${hud && !events.length && !tipGone && hud.t_fs > NUDGE_FS && html`<div class="rx-tip" role="status" title="Tap to hide" onClick=${() => setTipGone(true)}>
            <b>Nothing yet?</b> Molecules rarely react by just bumping around. Drag an atom of one molecule
            into another (hold it there a moment), or shrink the Wall to squeeze them.</div>`}
          <div class="rx-legend small">drag an atom: pull it · drag elsewhere: rotate · ${COARSE ? 'pinch: zoom · Wall: squeeze' : 'scroll or Wall: squeeze'}</div>
        </div>
        ${reactions.length > 0 && html`<button class="rx-bulb phone-only" onClick=${analyze}>
          <span class="bulb-dot"></span><span><b>Find ${reactions.length > 1 ? 'their barriers' : 'its barrier'}</b>
          <span class="small muted"> · stops the reactor</span></span></button>`}
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
          <button class="btn small" onClick=${() => { send({ op: 'pause', value: !paused }); setPaused(!paused); }}>${paused ? '▶ Resume' : '❚❚ Pause'}</button>
          <button class="btn small" onClick=${() => send({ op: 'release_all' })}>Release all</button>
        </div>
      </div>
      <aside class="rx-events">
        <div class="section-title">Reaction events · ${events.length}</div>
        ${!events.length && html`<p class="small muted">None yet. Bonds that change and stay changed show up here.</p>`}
        <ol>
          ${[...events].sort((a, b) => b.start_fs - a.start_fs).map((e) => html`<li class=${hud && (e.tentative || hud.t_fs - e.start_fs < RECENT_FS) ? 'on' : ''}>
            <div class="rx-ev" title=${e.label}>
              <span class="rx-t">${(e.start_fs / 1000).toFixed(2)} ps</span>
              <span class="rx-eq">${e.label}</span>
              ${e.reaction == null && !e.tentative && html`<span class="small muted">exchange</span>`}
              ${e.tentative && html`<span class="small rx-analyzing" title="Still unfolding: its bonds may change again">analyzing…</span>`}
            </div></li>`)}
        </ol>
        ${reactions.length > 0 && html`<button class="rx-bulb" onClick=${analyze}
          title="Stops the reactor. Each reaction is refined and gets its own TS search; the results land in Explore.">
          <span class="bulb-dot"></span><span><b>Find ${reactions.length > 1 ? 'their barriers' : 'its barrier'}</b>
          <span class="small muted"> · stops the reactor</span></span></button>`}
        <details class="small"><summary>How to play</summary>
        <ul class="small muted how-to-play">
          <li>Drag an atom to pull it (Shift: harder).</li>
          <li>Shrink the Wall (or scroll) to squeeze them.</li>
          <li>Heat it up with Temperature.</li>
        </ul>
        </details>
      </aside>
    </div>
  </div>`;
}
