// The live network (#/network/<job> or #/network/example/<slug>): an explorer run as it grows,
// corrects itself and moves material (mepd/web/live_network.py), drawn as a growing reaction scheme
// (LiveScheme.js). A reacting mixture's page is its own (MixtureNetwork.js), drawn the same way.
// The page polls while the run goes: only the frames it lacks (?since=), the rest of the timeline
// (names, events, what runs now) whole; a replay slider plays any moment again.
import { html, useEffect, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { openJob, set, useStore } from '../store.js';
import { MixtureView } from './MixtureNetwork.js';
import { Closer, LiveScheme } from './LiveScheme.js';

const fmtMin = (t) => `${(t / 60).toFixed(1)} min`;

export function LiveNetworkView({ jobId = null, example = null }) {
  // A reacting mixture's page is a network of molecules (its own view).
  const op = useStore((s) => (jobId ? s.jobs[jobId]?.op : null));
  if (op === 'network-mixture') return html`<${MixtureView} jobId=${jobId} />`;
  return html`<${ExpansionNetwork} jobId=${jobId} example=${example} />`;
}

// The timeline, kept up to date: a running job's polls ask only for the frames recorded since the
// last one (the network as it stands, after them, is replaced each time); a finished job or an
// example is fetched whole, once.
function useTimeline(jobId, example, running) {
  const [tl, setTl] = useState(null);
  const [error, setError] = useState(null);
  const have = useRef(null);
  useEffect(() => {
    let alive = true, timer = null;
    have.current = null;
    const base = example ? `/api/examples/${encodeURIComponent(example)}/timeline` : `/api/jobs/${jobId}/network-timeline`;
    const load = async () => {
      const prev = have.current;
      try {
        const d = await api.get(running && prev?.recorded != null ? `${base}?since=${prev.recorded}` : base);
        if (!alive) return;
        const merged = d.since != null && prev ? { ...d, frames: [...prev.frames.slice(0, d.since), ...d.frames] } : d;
        have.current = merged;
        setTl(merged);
        setError(null);
      } catch (e) {
        if (alive) setError(e.message);
      }
      if (alive && running) timer = setTimeout(load, 4000);
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, [jobId, example, running]);
  return [tl, error];
}

function ExpansionNetwork({ jobId = null, example = null }) {
  const [i, setI] = useState(null);           // frame shown (null: follow the latest / the final)
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const job = useStore((s) => (jobId ? s.jobs[jobId] : null));
  const running = ['running', 'queued'].includes(job?.status);
  const [tl, error] = useTimeline(jobId, example, running);
  const n = tl?.frames?.length || 0;
  const cur = i == null ? (running ? n - 1 : tl?.final ?? n - 1) : Math.min(i, n - 1);
  useEffect(() => {
    if (!playing || !n) return undefined;
    const id = setInterval(() => setI((k) => {
      const nk = (k == null ? 0 : k) + 1;
      if (nk >= n) { setPlaying(false); return n - 1; }
      return nk;
    }), 450 / speed);
    return () => clearInterval(id);
  }, [playing, n, speed]);

  if (error) return html`<div class="ln-empty"><p>${error}</p>${jobId && html`<button class="btn" onClick=${() => openJob(jobId)}>Back to the calculation</button>`}</div>`;
  if (!tl) return html`<p class="boot">Loading the network…</p>`;
  if (!n) {
    return html`<div class="ld-page">
      <div class="ln-head"><h2>Live network</h2>${running && html`<span class="ln-live">● live</span>`}</div>
      <div class="ln-stage ln-waiting">
        <div class="ln-seed-pulse"></div>
        <p>${running ? 'Looking for the first reactions…' : 'This calculation found no network.'}</p>
        <p class="small muted">${running ? 'Species appear here as the explorer finds them.' : ''}</p>
      </div>
      ${jobId && html`<div class="ld-details"><${Closer} jobId=${jobId} job=${job} activity=${tl.activity} running=${running} /></div>`}
    </div>`;
  }
  const f = tl.frames[cur];
  const controls = html`<div class="ln-controls">
    <button class="btn primary" onClick=${() => { if (!playing && cur >= n - 1) setI(0); setPlaying(!playing); }}>${playing ? 'Pause' : 'Replay'}</button>
    <input type="range" min="0" max=${n - 1} value=${cur} onInput=${(e) => { setPlaying(false); setI(+e.target.value); }} aria-label="Run time shown" />
    <span class="mono small">${fmtMin(f.t)} / ${fmtMin(tl.frames[n - 1].t)}</span>
    <div class="segmented">${[1, 4].map((s) => html`<button class=${speed === s ? 'on' : ''} onClick=${() => setSpeed(s)}>${s}×</button>`)}</div>
    ${running && i != null && html`<button class="btn small" onClick=${() => { setPlaying(false); setI(null); }}>Back to live</button>`}
    ${!running && cur !== tl.final && html`<button class="btn small" onClick=${() => { setPlaying(false); setI(tl.final); }} title="The end state: the last frame with every energy steered">End</button>`}
  </div>`;
  return html`<${LiveScheme} tl=${tl} cur=${cur} jobId=${jobId} example=${example} job=${job} running=${running}
    follow=${i == null || playing} controls=${controls} />`;
}

export function openLiveNetwork({ jobId = null, example = null }) {
  set({ view: { tab: 'network', jobId, example } });
}
