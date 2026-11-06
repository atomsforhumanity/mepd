// The Workbench frame around every page: a thin top bar, the canvas header
// (where you are, and the views of it), the Calculations bar along the
// bottom, and the Learn and Settings drawers that open over the canvas.
// The pages themselves (Explore, a calculation, Design, the reactor) are the
// components they always were; this file only places them.
import { html, useEffect, useState } from '../lib.js';
import { api } from '../api.js';
import { closeDrawer, openTab, replaceView, set, state, upLevel, useStore } from '../store.js';
import { cls } from '../util.js';
import { useMap } from '../quests.js';
import { ProfilesView } from './Profiles.js';
import { ReferencesView } from './References.js';
import { QuestMap } from './Welcome.js';

let backToReactor = false;   // the first look at the running reactors has been taken

const Icon = {
  learn: html`<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14"/><circle cx="12" cy="17.3" r=".6"/></svg>`,
  gear: html`<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/></svg>`,
};

// Interactive reactors still running in this workspace: a way back to them
// (and, on a fresh load while one runs, straight back into it, once).
function useSandboxes() {
  const root = useStore((s) => s.workspace.root);
  const tab = useStore((s) => s.view.tab);
  const sandboxes = useStore((s) => s.sandboxes || []);
  useEffect(() => {
    let alive = true;
    const refresh = () => api.get('/api/sandboxes').then((list) => {
      if (!alive) return;
      const v = state.view;
      if (v.tab === 'sandbox' && v.sandbox && !v.sandbox.symbols) {
        // An address (#/reactor/<id>): the reactor it names, if it still runs.
        const sb = list.find((x) => x.id === v.sandbox.id);
        replaceView(sb ? { ...v, sandbox: sb } : { tab: 'graph', jobId: null }, { sandboxes: list });
      } else if (!backToReactor && list.length && ['welcome', 'graph'].includes(v.tab)) {
        replaceView({ tab: 'sandbox', jobId: null, sandbox: list[0] }, { sandboxes: list });
      } else set({ sandboxes: list });
      backToReactor = true;
    }).catch(() => {});
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => { alive = false; clearInterval(timer); };
  }, [root, tab]);
  return sandboxes;
}

export function TopBar() {
  const view = useStore((s) => s.view);
  const connected = useStore((s) => s.connected);
  const root = useStore((s) => s.workspace.root);
  const demo = useStore((s) => s.demo);
  const auth = useStore((s) => s.auth);
  const sandboxes = useSandboxes();
  const toggle = (d) => (view.drawer === d ? closeDrawer() : set({ view: { ...view, drawer: d } }));
  return html`
    <header class="topbar wb-topbar">
      <a class="brand key" href="#/home" title="Home" onClick=${(e) => { e.preventDefault(); openTab('welcome'); }}>
        <svg class="logo" viewBox="0 0 32 32" aria-hidden="true"><polygon points="16,3 27,9.5 27,22.5 16,29 5,22.5 5,9.5" fill="none" stroke="currentColor" stroke-width="2.6"/><circle cx="16" cy="16" r="3.2" fill="currentColor"/></svg>
        mepd
      </a>
      <button class="session-name" title=${`Session folder: ${root}\nSwitch, open or start a new session`}
        onClick=${() => set({ modal: { kind: 'sessions' } })}>
        <span class=${`conn ${connected ? 'on' : 'off'}`} title=${connected ? 'Connected' : 'Reconnecting…'}></span>
        <span class="session-label">${root.split('/').pop()}</span><span class="caret">▾</span>
      </button>
      ${demo && html`<span class="badge demo-badge" title="Public demo: private workspace, limited sizes and run times">Demo</span>`}
      <span class="spacer"></span>
      ${sandboxes.length > 0 && view.tab !== 'sandbox' && html`<button class="btn small sandbox-return"
        title=${`Back to the reactor: ${sandboxes[0].names}${sandboxes.length > 1 ? ` (and ${sandboxes.length - 1} more)` : ''}`}
        onClick=${() => set({ view: { tab: 'sandbox', jobId: null, sandbox: sandboxes[0] } })}>◉ Reactor</button>`}
      <button class=${cls('wb-top-btn', view.drawer === 'learn' && 'on')} onClick=${() => toggle('learn')} title="How things work, words, references, your tools">
        ${Icon.learn}<span>Learn</span></button>
      <button class=${cls('wb-top-btn icon-only', view.drawer === 'settings' && 'on')} onClick=${() => toggle('settings')} aria-label="Settings" title="Settings: level of theory and preferences">
        ${Icon.gear}</button>
      ${auth && html`<a class="topbar-link small" href="/logout"
        title=${demo ? `${demo.open ? 'Delete' : 'Log out and delete'} your demo workspace (asks first)` : 'Log this device out'}>
        ${demo?.open ? 'Delete workspace' : 'Log out'}</a>`}
    </header>`;
}

// Where you are (Explore › a calculation) and the views of what is open.
export function CanvasHeader() {
  const view = useStore((s) => s.view);
  const analyzeView = useStore((s) => s.analyze?.view || 'reactions');
  const map = useMap();
  if (!['graph', 'analyze'].includes(view.tab)) return null;
  const at = view.tab === 'graph' ? 'explore' : ['kinetics', 'landscape'].includes(analyzeView) ? analyzeView : 'reactions';
  const go = (k) => {
    if (k === 'explore') openTab('graph');
    else set({ analyze: { ...(state.analyze || {}), compose: false, view: k }, view: { tab: 'analyze', jobId: null } });
  };
  const views = [['explore', 'Explore'], ...(map.tab('analyze') ? [['reactions', 'Reactions'], ['kinetics', 'Kinetics'], ['landscape', 'Landscape']] : [])];
  return html`<nav class="canvas-tabs" aria-label="Views">
    ${views.map(([k, l]) => html`<button class=${cls(at === k && 'on')} onClick=${() => go(k)}>${l}</button>`)}
  </nav>`;
}

// Design and the reactor are modes over the canvas: one bar, one way out.
export function ModeBar() {
  const view = useStore((s) => s.view);
  if (view.tab === 'design') {
    return html`<div class="mode-bar">
      <strong>Design</strong>
      <span class="spacer"></span>
      <button class="mode-done" onClick=${() => openTab('graph')} title="Back to Explore (the design is kept)">Done</button>
    </div>`;
  }
  if (view.tab === 'sandbox') {
    return html`<div class="mode-bar">
      <strong>Reactor</strong>
      <span class="spacer"></span>
      <button class="mode-done" onClick=${() => openTab('graph')} title="Back to Explore (the reactor keeps running for 15 minutes; ◉ Reactor brings you back)">Exit</button>
    </div>`;
  }
  return null;
}

// The bottom bar: what is running, the workspace's level of theory, units.
// Click it for the full list of calculations.
export function ActivityBar() {
  const view = useStore((s) => s.view);
  const run = useStore((s) => {
    const js = Object.values(s.jobs);
    const running = js.filter((j) => j.status === 'running').sort((a, b) => (b.started || b.created || 0) - (a.started || a.created || 0));
    const queued = js.filter((j) => j.status === 'queued').length;
    const top = running[0];
    return { n: running.length, queued, total: js.length, title: top?.title || '', line: top ? (s.progress[top.id]?.last_line ?? top.last_line ?? '') : '' };
  });
  const level = useStore((s) => s.levels[s.levelProfile ?? '']?.label || '');
  const hint = useStore((s) => s.hint || '');
  const busy = run.n + run.queued > 0;
  const onList = view.tab === 'jobs';
  return html`<footer class="activity-bar">
    <button class=${cls('ab-calcs', onList && 'on')} onClick=${() => (onList ? upLevel() : openTab('jobs'))}
      title=${onList ? 'Back' : 'All calculations'}>
      <span class=${cls('ab-dot', busy && 'busy')}></span>
      ${busy ? `${run.n} running${run.queued ? ` · ${run.queued} queued` : ''}` : `Calculations${run.total ? ` · ${run.total}` : ''}`}
    </button>
    ${run.n > 0 && html`<span class="ab-run" title=${run.line}>${run.title}${run.line ? html` <span class="muted">· ${run.line}</span>` : ''}</span>`}
    <span class="ab-hint muted">${hint}</span>
    ${level && html`<button class="ab-level" title="Level of theory for this workspace" onClick=${() => set({ view: { ...state.view, drawer: 'settings' } })}>${level}</button>`}
    <span class="ab-units muted">kcal/mol</span>
  </footer>`;
}

// Learn: what the manual says, the words, the references, your tools.
function LearnDrawer() {
  const focus = useStore((s) => s.refsFocus);
  const [tab, setTab] = useState(focus ? 'refs' : 'manual');
  useEffect(() => { if (focus) setTab('refs'); }, [focus]);
  const tabs = [['manual', 'Manual'], ['refs', 'References'], ['tools', 'Your tools']];
  return html`<div class="drawer-body">
    <nav class="drawer-tabs">${tabs.map(([k, l]) => html`<button class=${cls(tab === k && 'on')} onClick=${() => setTab(k)}>${l}</button>`)}</nav>
    <div class="drawer-scroll">
      ${tab === 'manual' && html`<iframe class="manual-frame" src="/manual" title="mepd manual"></iframe>`}
      ${tab === 'refs' && html`<${ReferencesView} />`}
      ${tab === 'tools' && html`<${QuestMap} />`}
    </div>
  </div>`;
}

export function Drawer() {
  const drawer = useStore((s) => s.view.drawer);
  if (!drawer) return null;
  return html`<aside class=${`drawer drawer-${drawer}`} aria-label=${drawer === 'learn' ? 'Learn' : 'Settings'}>
    <header class="drawer-head">
      <h2>${drawer === 'learn' ? 'Learn' : 'Settings'}</h2>
      <button class="btn-icon" aria-label="Close" title="Close (Esc)" onClick=${() => closeDrawer()}>✕</button>
    </header>
    ${drawer === 'learn' ? html`<${LearnDrawer} />` : html`<div class="drawer-scroll"><${ProfilesView} /></div>`}
  </aside>`;
}
