import { html, render, useEffect } from './lib.js';
import { connect, deleteSelection } from './api.js';
import { clearSelection, set, state, upLevel, useStore } from './store.js';
import { Graph } from './components/Graph.js';
import { Inspector } from './components/Inspector.js';
import { JobView } from './components/JobView.js';
import { JobsView } from './components/Jobs.js';
import { Library } from './components/Library.js';
import { Modals } from './components/Modals.js';
import { DesignView } from './components/Design.js';
import { AnalyzeView } from './components/Analyze.js';
import { SandboxView } from './components/Sandbox.js';
import { WelcomeView } from './components/Welcome.js';
import { ReactionFocus } from './components/ReactionPlayer.js';
import { LiveNetworkView } from './components/LiveNetwork.js';
import { ActivityBar, CanvasHeader, Drawer, ModeBar, TopBar } from './components/Shell.js';
import { announceUnlocks, markQuest, mapOf, useMap } from './quests.js';

function Toasts() {
  const toasts = useStore((s) => s.toasts);
  return html`<div class="toasts" aria-live="polite">
    ${toasts.map((t) => html`<div class=${`toast ${t.kind}`} onClick=${() => set({ toasts: state.toasts.filter((x) => x.id !== t.id) })}>
      ${t.message}${t.action && html` <button class="toast-action" onClick=${(e) => { e.stopPropagation(); t.action.run(); set({ toasts: state.toasts.filter((x) => x.id !== t.id) }); }}>${t.action.label}</button>`}</div>`)}
  </div>`;
}

// Which panes each page has: the browser belongs to the workspace pages;
// Home and the two modes (Design, the reactor) use the whole width.
const WITH_BROWSER = new Set(['graph', 'analyze', 'jobs', 'job', 'reaction']);

function App() {
  const view = useStore((s) => s.view);
  const loaded = useStore((s) => s.loaded);
  const howToHidden = useStore((s) => s.howToHidden);
  const theme = useStore((s) => s.theme);
  useStore((s) => s.selection);
  // The map: announce what just opened; visiting Explore completes its quest.
  const map = useMap();
  const mapKey = `${[...map.done].join()}|${[...map.open].join()}`;
  useEffect(() => { if (loaded) announceUnlocks(); }, [loaded, mapKey]);
  // A new page starts at its top (on a phone the whole layout scrolls, and kept the last page's place).
  useEffect(() => { const el = document.querySelector('.layout'); if (el) el.scrollTop = 0; }, [view.tab, view.jobId]);
  useEffect(() => {
    if (!loaded) return;
    if (view.tab === 'graph' && mapOf().open.has('network')) markQuest('network');
  }, [loaded, view.tab, mapKey]);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'auto') delete root.dataset.theme; else root.dataset.theme = theme;
    // The graph draws with the CSS colours: redraw it in the new palette.
    requestAnimationFrame(() => window.dispatchEvent(new CustomEvent('mepd:graph', { detail: 'restyle' })));
  }, [theme]);

  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || state.modal || e.defaultPrevented) return;
      if (e.key === 'Escape') {
        // Up one level: the selection first, then a drawer, then the page (modes leave by their own button).
        if (state.connectMode || state.selection.structures.length || state.selection.edges.length) {
          set({ connectMode: false });
          clearSelection();
        } else if (state.view.drawer || !['design', 'sandbox'].includes(state.view.tab)) upLevel();
      }
      if (e.key === 'c' && state.view.tab === 'graph') set({ connectMode: !state.connectMode });
      if ((e.key === 'Delete' || e.key === 'Backspace') && state.view.tab === 'graph') { e.preventDefault(); deleteSelection(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  if (!loaded) return html`<div class="boot">Connecting to mepd…</div>`;
  const demo = state.demo;
  const nothingSelected = !state.selection.structures.length && !state.selection.edges.length;
  const browser = WITH_BROWSER.has(view.tab);
  const showInspector = (view.tab === 'graph' && !(nothingSelected && howToHidden)) || view.tab === 'reaction';
  return html`
    <${TopBar} />
    ${demo && html`<div class="demo-banner small" title=${`Limits: ${demo.max_atoms} atoms per structure, ${demo.max_active_jobs} calculations at a time, ${Math.round(demo.job_timeout_s / 60)} min per calculation${demo.op_timeout_s?.vri ? ` (VRI searches ${Math.round(demo.op_timeout_s.vri / 60)} min)` : ''}. The built-in profiles are fixed; Settings › Save as… makes your own.`}>${`Demo · your private workspace · up to ${demo.max_atoms} atoms, ${demo.max_active_jobs} calculations at a time`}</div>`}
    <div class="wb-body">
      <main class=${`layout tab-${view.tab} ${browser ? '' : 'no-browser'} ${showInspector ? '' : 'no-inspector'}`}>
        ${browser && html`<${Library} />`}
        <section class="center">
          <${ModeBar} />
          <${CanvasHeader} />
          <div class="center-view" hidden=${view.tab !== 'graph'}><${Graph} /></div>
          ${view.tab === 'jobs' && html`<${JobsView} />`}
          ${view.tab === 'job' && html`<${JobView} jobId=${view.jobId} />`}
          ${view.tab === 'reaction' && html`<${ReactionFocus} edgeId=${view.edgeId} />`}
          ${view.tab === 'network' && html`<${LiveNetworkView} key=${view.jobId || view.example} jobId=${view.jobId} example=${view.example} />`}
          ${view.tab === 'design' && html`<${DesignView} />`}
          ${view.tab === 'analyze' && html`<${AnalyzeView} />`}
          ${view.tab === 'sandbox' && (view.sandbox?.symbols ? html`<${SandboxView} />` : html`<p class="boot">Finding the reactor…</p>`)}
          ${view.tab === 'welcome' && html`<${WelcomeView} />`}
        </section>
        ${showInspector && html`<${Inspector} />`}
      </main>
      <${Drawer} />
    </div>
    <${ActivityBar} />
    <${Modals} />
    <${Toasts} />`;
}

connect();
render(html`<${App} />`, document.getElementById('app'));
