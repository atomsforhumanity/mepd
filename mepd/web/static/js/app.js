import { html, render, useEffect } from './lib.js';
import { api, connect, deleteSelection } from './api.js';
import { clearSelection, openTab, replaceView, set, state, useStore } from './store.js';
import { Graph } from './components/Graph.js';
import { Inspector } from './components/Inspector.js';
import { JobView } from './components/JobView.js';
import { JobsView } from './components/Jobs.js';
import { Library } from './components/Library.js';
import { Modals } from './components/Modals.js';
import { ProfilesView } from './components/Profiles.js';
import { ReferencesView } from './components/References.js';
import { DesignView } from './components/Design.js';
import { AnalyzeView } from './components/Analyze.js';
import { SandboxView } from './components/Sandbox.js';
import { PlaygroundApp, layoutToggle } from './components/Playground.js';
import { WelcomeView } from './components/Welcome.js';
import { announceUnlocks, markQuest, mapOf, useMap } from './quests.js';

let backToReactor = false;   // the first look at the running reactors has been taken

function TopBar() {
  const view = useStore((s) => s.view);
  const connected = useStore((s) => s.connected);
  const root = useStore((s) => s.workspace.root);
  const counts = useStore((s) => {
    const js = Object.values(s.jobs);
    return { running: js.filter((j) => j.status === 'running').length, queued: js.filter((j) => j.status === 'queued').length };
  });
  const lastJob = useStore((s) => s.view.jobId);
  const map = useMap();
  // Tabs appear as the map opens them (quests.js).
  const tabs = [['design', 'Design'], ['graph', 'Explore'], ['analyze', 'Analyze'], ['jobs', 'Calculations'], ['profiles', 'Settings'], ['refs', 'References']]
    .filter(([k]) => map.tab(k));
  const demo = useStore((s) => s.demo);
  const auth = useStore((s) => s.auth);
  const active = counts.running + counts.queued;
  // Interactive reactors still running in this workspace: a way back to them.
  const sandboxes = useStore((s) => s.sandboxes || []);
  useEffect(() => {   // (a demo visitor's: only their own)
    let alive = true;
    const refresh = () => api.get('/api/sandboxes').then((list) => {
      if (!alive) return;
      // Loaded while a reactor runs (a reload, a phone woken up): back to it, once.
      if (!backToReactor && list.length && ['welcome', 'graph'].includes(state.view.tab)) {
        replaceView({ tab: 'sandbox', jobId: null, sandbox: list[0] }, { sandboxes: list });
      } else set({ sandboxes: list });
      backToReactor = true;
    }).catch(() => {});
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => { alive = false; clearInterval(timer); };
  }, [root, view.tab]);
  return html`
    <header class="topbar">
      <div class="brand" role="button" tabindex="0" title="Your tools and the first steps" onClick=${() => openTab('welcome')}
        onKeyDown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openTab('welcome'); } }}>
        <svg class="logo" viewBox="0 0 32 32" aria-hidden="true"><polygon points="16,3 27,9.5 27,22.5 16,29 5,22.5 5,9.5" fill="none" stroke="currentColor" stroke-width="2.6"/><circle cx="16" cy="16" r="3.2" fill="currentColor"/></svg>
        mepd
      </div>
      <nav class="main-tabs">
        ${tabs.map(([k, l]) => html`<button class=${view.tab === k || (k === 'jobs' && view.tab === 'job') ? 'on' : ''}
          onClick=${() => (k === 'jobs' && view.tab === 'jobs' && lastJob ? set({ view: { tab: 'job', jobId: lastJob } }) : openTab(k))}>
          ${l}${k === 'jobs' && active > 0 && html` <span class="tab-count" title=${`${counts.running} running, ${counts.queued} queued`}>${active}</span>`}
        </button>`)}
      </nav>
      <span class="spacer"></span>
      <a class="manual-link small" href="/manual" target="_blank" rel="noopener" title="How mepd's features work, the choices made and why">Manual</a>
      ${sandboxes.length > 0 && view.tab !== 'sandbox' && html`<button class="btn small sandbox-return"
        title=${`Back to the interactive reactor: ${sandboxes[0].names}${sandboxes.length > 1 ? ` (and ${sandboxes.length - 1} more)` : ''}`}
        onClick=${() => set({ view: { tab: 'sandbox', jobId: null, sandbox: sandboxes[0] } })}>◉ Interactive reactor</button>`}
      <button class="session-name" title=${`Session folder: ${root}\nSwitch, open or start a new session`}
        onClick=${() => set({ modal: { kind: 'sessions' } })}>
        <span class=${`conn ${connected ? 'on' : 'off'}`} title=${connected ? 'Connected' : 'Reconnecting…'}></span>
        <span class="session-label">${root.split('/').pop()}</span><span class="caret">▾</span>
      </button>
      ${demo && html`<span class="badge demo-badge" title="Public demo: private workspace, limited sizes and run times">Demo</span>`}
      ${(map.all || state.layout === 'playground') && layoutToggle()}
      ${auth && html`<a class="topbar-link small" href="/logout"
        title=${demo ? `${demo.open ? 'Delete' : 'Log out and delete'} your demo workspace (asks first)` : 'Log this device out'}>
        ${demo?.open ? 'Delete workspace' : 'Log out'}</a>`}
    </header>`;
}

function Toasts() {
  const toasts = useStore((s) => s.toasts);
  return html`<div class="toasts" aria-live="polite">
    ${toasts.map((t) => html`<div class=${`toast ${t.kind}`} onClick=${() => set({ toasts: state.toasts.filter((x) => x.id !== t.id) })}>
      ${t.message}${t.action && html` <button class="toast-action" onClick=${(e) => { e.stopPropagation(); t.action.run(); set({ toasts: state.toasts.filter((x) => x.id !== t.id) }); }}>${t.action.label}</button>`}</div>`)}
  </div>`;
}

function App() {
  const view = useStore((s) => s.view);
  const loaded = useStore((s) => s.loaded);
  const howToHidden = useStore((s) => s.howToHidden);
  const layout = useStore((s) => s.layout);
  const pgBg = useStore((s) => s.pgBg);
  useStore((s) => s.selection);
  // The map: announce what just opened; visiting a quest's tab completes it;
  // a tab the map has not opened (e.g. the landing one) gives way to the first screen.
  const map = useMap();
  const mapKey = `${[...map.done].join()}|${[...map.open].join()}`;
  useEffect(() => { if (loaded) announceUnlocks(); }, [loaded, mapKey]);
  // A new page starts at its top (on a phone the whole layout scrolls, and kept the last page's place).
  useEffect(() => { const el = document.querySelector('.layout'); if (el) el.scrollTop = 0; }, [view.tab, view.jobId]);
  useEffect(() => {
    if (!loaded) return;
    // (Method and Kinetics are done by using them, not by opening their tab.)
    const visit = { graph: 'network' }[view.tab];
    if (visit && mapOf().open.has(visit) && map.tab(view.tab)) markQuest(visit);
    if (['graph', 'analyze', 'jobs', 'profiles'].includes(view.tab) && !map.tab(view.tab)) replaceView({ tab: 'welcome', jobId: null });
  }, [loaded, view.tab, mapKey]);
  // The layout's palette lives on <html>, so the graph (drawn from CSS
  // variables) and everything else pick it up.
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.layout = layout;
    root.dataset.pgbg = pgBg;
    // The graph draws with the CSS colours: redraw it in the new palette.
    requestAnimationFrame(() => window.dispatchEvent(new CustomEvent('mepd:graph', { detail: 'restyle' })));
  }, [layout, pgBg]);

  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || state.modal) return;
      if (e.key === 'Escape') {
        // Playground: Escape also closes a page (Calculations, Profiles, a job) back to the graph.
        if (state.layout === 'playground' && !state.pgPanel && state.view.tab !== 'graph') { openTab('graph'); return; }
        set({ connectMode: false, pgPanel: null });
        clearSelection();
      }
      if (e.key === 'c' && state.view.tab === 'graph') set({ connectMode: !state.connectMode });
      if ((e.key === 'Delete' || e.key === 'Backspace') && state.view.tab === 'graph') { e.preventDefault(); deleteSelection(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  if (!loaded) return html`<div class="boot">Connecting to mepd…</div>`;
  if (layout === 'playground' && view.tab !== 'welcome') {
    return html`<${PlaygroundApp} /><${Modals} /><${Toasts} />`;
  }
  const demo = state.demo;
  const nothingSelected = !state.selection.structures.length && !state.selection.edges.length;
  const showInspector = (view.tab === 'graph' || view.tab === 'jobs') && !(nothingSelected && howToHidden);
  return html`
    <${TopBar} />
    ${demo && html`<div class="demo-banner small" title=${`Limits: ${demo.max_atoms} atoms per structure, ${demo.max_active_jobs} calculations at a time, ${Math.round(demo.job_timeout_s / 60)} min per calculation${demo.op_timeout_s?.vri ? ` (VRI searches ${Math.round(demo.op_timeout_s.vri / 60)} min)` : ''}. The built-in profiles are fixed; Settings › Save as… makes your own.`}>${`Demo · your private workspace · up to ${demo.max_atoms} atoms, ${demo.max_active_jobs} calculations at a time`}</div>`}
    <main class=${`layout tab-${view.tab} ${showInspector ? '' : 'no-inspector'}`}>
      <${Library} />
      <section class="center">
        <div class="center-view" hidden=${view.tab !== 'graph'}><${Graph} /></div>
        ${view.tab === 'jobs' && html`<${JobsView} />`}
        ${view.tab === 'job' && html`<${JobView} jobId=${view.jobId} />`}
        ${view.tab === 'profiles' && html`<${ProfilesView} />`}
        ${view.tab === 'refs' && html`<${ReferencesView} />`}
        ${view.tab === 'design' && html`<${DesignView} />`}
        ${view.tab === 'analyze' && html`<${AnalyzeView} />`}
        ${view.tab === 'sandbox' && html`<${SandboxView} />`}
        ${view.tab === 'welcome' && html`<${WelcomeView} />`}
      </section>
      ${showInspector && html`<${Inspector} />`}
    </main>
    <${Modals} />
    <${Toasts} />`;
}

connect();
render(html`<${App} />`, document.getElementById('app'));
