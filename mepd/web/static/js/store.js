// App state: one plain object, mutated only through `set`/`update`, with a
// hook that re-renders a component when the slice it selected changes.
// The server is the source of truth for workspace/jobs; this store mirrors
// /api/state and then applies SSE events.
import { useEffect, useRef, useState } from './lib.js';

const listeners = new Set();

const _pref = (k, d) => { try { const v = localStorage.getItem('mepd:' + k); return v === null ? d : JSON.parse(v); } catch { return d; } };

export const state = {
  connected: false,
  theme: _pref('theme', 'auto'),         // 'auto' | 'light' | 'dark'
  inspectorTab: _pref('inspectorTab', 'tools'),   // the inspector's tab: 'tools' | 'results' | 'method'
  // The 'How it works' panel, once dismissed, stays away (per browser).
  howToHidden: (() => { try { return localStorage.getItem('mepd:hideHowTo') === 'true'; } catch { return false; } })(),
  // Getting-started steps done (molecule, reactor, reaction) and whether they are hidden (Welcome.js).
  tour: _pref('tour', {}),
  everything: _pref('everything', false),   // "Show everything": skip the map (quests.js)
  seen: null,            // what has been looked at (isNew); seenTick re-renders on change
  seenTick: 0,
  focusOp: null,         // {key, at}: an operation card to open in the Inspector (a quest's "Try it")
  loaded: false,
  workspace: { structures: {}, edges: {}, positions: {}, root: '' },
  jobs: {},
  operations: [],
  profiles: [],
  levelProfile: null,    // the workspace's level of theory (a profile name)
  validateMinima: true,  // Hessian-check structures optimized on entry
  levels: {},            // profile name ('' = built-in defaults) -> {profile, key, label}
  pathSummaries: {},     // profile name -> {text, warnings}: what its path search will really do
  profileEditor: null,   // {name, dirty, save()} while the Profiles tab holds edits
  demo: null,            // public demo policy (limits, allowed ops) or null
  cpus: 1,
  maxConcurrent: 1,
  progress: {},          // job id -> latest progress payload (merged)
  selection: { structures: [], edges: [] },  // click order preserved
  // What the canvas shows (tab: 'welcome' Home, 'graph' Explore, 'analyze', 'jobs', 'job',
  // 'design' and 'sandbox' modes) and which drawer is open over it ('settings' | 'learn' | null).
  // The old Settings ('profiles') and References ('refs') pages open as drawers (see `set`).
  view: { tab: 'graph', jobId: null, drawer: null },
  connectMode: false,
  network: '',           // which network Explore shows: '' the parent, else a variant id (mepd.web.variants)
  toasts: [],
  modal: null,           // {kind, ...}
};

export function update(fn) {
  fn(state);
  for (const l of [...listeners]) l();
}

// Settings and References are drawers over the canvas now, not pages: a view
// asking for one keeps the page underneath and opens the drawer instead.
const DRAWER_OF = { profiles: 'settings', refs: 'learn' };
export function normalizeView(v, current = state.view) {
  if (!v) return v;
  const drawer = DRAWER_OF[v.tab];
  if (drawer) {
    const under = current && !DRAWER_OF[current.tab] ? current : { tab: 'graph', jobId: null };
    return { ...under, drawer };
  }
  return { drawer: null, ...v };
}

export function set(patch) {
  if (patch.view) patch = { ...patch, view: normalizeView(patch.view) };
  update((s) => Object.assign(s, patch));
}

function shallowEqual(a, b) {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || !a || !b) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => Object.is(a[k], b[k]));
}

export function useStore(selector) {
  const [, force] = useState(0);
  const sel = useRef(selector);
  sel.current = selector;
  const last = useRef(selector(state));
  last.current = selector(state);
  useEffect(() => {
    const l = () => {
      const next = sel.current(state);
      if (!shallowEqual(next, last.current)) {
        last.current = next;
        force((n) => n + 1);
      }
    };
    listeners.add(l);
    // State may have changed between render and subscription (e.g. the
    // initial /api/state landing first) -- catch up now or we'd never hear it.
    l();
    return () => listeners.delete(l);
  }, []);
  return last.current;
}

// ---- selection helpers -------------------------------------------------
export function select({ structures = [], edges = [] }, additive = false) {
  markSeen([...structures, ...edges]);
  update((s) => {
    if (!additive) {
      s.selection = { structures: [...structures], edges: [...edges] };
      return;
    }
    const toggle = (arr, ids) => {
      const out = [...arr];
      for (const id of ids) {
        const i = out.indexOf(id);
        if (i >= 0) out.splice(i, 1); else out.push(id);
      }
      return out;
    };
    s.selection = {
      structures: toggle(s.selection.structures, structures),
      edges: toggle(s.selection.edges, edges),
    };
  });
}

// ---- what is new --------------------------------------------------------
// Structures and reactions a calculation added since the person last looked
// are "new": a dot in the lists and a glow in Explore, until selected (or all
// marked seen). Kept per workspace in this browser. The first look at a
// workspace counts everything already in it as seen; no clocks compared, only
// the server's own `created` times.
const seenKey = () => `mepd:seen:${state.workspace.root}`;
const createdTimes = () => [...Object.values(state.workspace.structures || {}), ...Object.values(state.workspace.edges || {})]
  .map((r) => r.created || 0);
function seenNow() {
  if (state.seen?.root === state.workspace.root) return state.seen;
  let rec = null;
  try { rec = JSON.parse(localStorage.getItem(seenKey())); } catch { /* private mode */ }
  if (!rec) {
    rec = { since: Math.max(0, ...createdTimes()), ids: [] };
    try { localStorage.setItem(seenKey(), JSON.stringify(rec)); } catch { /* not kept */ }
  }
  state.seen = { root: state.workspace.root, since: rec.since, ids: new Set(rec.ids) };
  return state.seen;
}
function saveSeen(seen) {
  try { localStorage.setItem(seenKey(), JSON.stringify({ since: seen.since, ids: [...seen.ids] })); } catch { /* not kept */ }
  update((s) => { s.seenTick = (s.seenTick || 0) + 1; });
}
// A structure or edge counts as new when a calculation made it (not one the
// person drew or typed in) after the last look.
export function isNew(rec) {
  if (!rec || !state.workspace.root) return false;
  if (rec.source == null && rec.origin?.kind !== 'job') return false;      // a structure: made by a calculation
  if (rec.source != null && (rec.origin?.kind ?? 'manual') === 'manual') return false;   // an edge: not drawn by hand
  const seen = seenNow();
  return (rec.created || 0) > seen.since && !seen.ids.has(rec.id);
}
export function markSeen(ids) {
  if (!state.workspace.root) return;
  const seen = seenNow();
  const fresh = ids.filter((id) => !seen.ids.has(id)
    && isNew(state.workspace.structures?.[id] || state.workspace.edges?.[id]));
  if (!fresh.length) return;
  fresh.forEach((id) => seen.ids.add(id));
  saveSeen(seen);
}
export function markAllSeen() {
  const seen = seenNow();
  seen.since = Math.max(seen.since, ...createdTimes());
  seen.ids = new Set();
  saveSeen(seen);
}

export function clearSelection() {
  set({ selection: { structures: [], edges: [] } });
}

// ---- toasts -------------------------------------------------------------
let toastId = 0;
// `action`: optional {label, run} rendered as a button in the toast.
export function toast(message, kind = 'info', ms = 4500, action = null) {
  const id = ++toastId;
  update((s) => { s.toasts = [...s.toasts, { id, message, kind, action }]; });
  if (ms) setTimeout(() => update((s) => { s.toasts = s.toasts.filter((t) => t.id !== id); }), ms);
}

// ---- navigation ---------------------------------------------------------
// "Sample more paths" runs add to the run they start from and show on its
// page (mepd/web/jobs.py: JobManager.page_job).
function extendsJob(j) { return j?.extends || (['channels-more', 'nanoreactor-more'].includes(j?.op) ? j.source_job : null); }
export function pageJobId(id) {
  let j = state.jobs[id];
  const seen = new Set([id]);
  for (let up = extendsJob(j); j && up && state.jobs[up] && !seen.has(up); up = extendsJob(j)) {
    seen.add(up);
    j = state.jobs[up];
  }
  return j ? j.id : id;
}
// The page's runs, oldest (the page's own) first.
export function familyOf(jobs, id) {
  const base = pageJobId(id);
  return Object.values(jobs).filter((j) => j.id === base || pageJobId(j.id) === base).sort((a, b) => a.created - b.created);
}
// `tab`: open the page on that tab, showing that run (a Sample more paths run's log).
// Pages as browser history: every change of view (tab, calculation, run) is
// a step that the browser's back and forward (and a phone's back gesture)
// walk through; the job page's ← does the same while there is a step to go
// back to.
// Every page has an address (#/explore, #/calc/<id>/live, #/design?d=settings),
// so a page can be linked, bookmarked and reloaded.
const sandboxId = (v) => v.sandbox?.id || v.sandbox || '';
export function formatHash(v, analyze = state.analyze) {
  let path;
  switch (v.tab) {
    case 'welcome': path = '/home'; break;
    case 'analyze': path = `/${['kinetics', 'landscape'].includes(analyze?.view) ? analyze.view : 'reactions'}`; break;
    case 'jobs': path = '/calculations'; break;
    case 'job': path = `/calc/${encodeURIComponent(v.jobId || '')}${v.jobTab ? `/${v.jobTab}` : ''}`; break;
    case 'reaction': path = `/reaction/${encodeURIComponent(v.edgeId || '')}`; break;
    case 'network': path = v.example ? `/network/example/${encodeURIComponent(v.example)}` : `/network/${encodeURIComponent(v.jobId || '')}`; break;
    case 'design': path = '/design'; break;
    case 'sandbox': path = `/reactor/${encodeURIComponent(sandboxId(v))}`; break;
    default: path = '/explore';
  }
  const q = [];
  if (v.tab === 'job' && v.run) q.push(`run=${encodeURIComponent(v.run)}`);
  if (v.drawer) q.push(`d=${v.drawer}`);
  return `#${path}${q.length ? `?${q.join('&')}` : ''}`;
}
// The view a hash names (null when it names none). A reactor link only keeps
// its id: the app looks the reactor up again (it may have ended).
export function parseHash(hash) {
  const m = /^#\/([^?]*)(?:\?(.*))?$/.exec(hash || '');
  if (!m) return null;
  const parts = m[1].split('/').filter(Boolean).map(decodeURIComponent);
  const q = Object.fromEntries((m[2] || '').split('&').filter(Boolean).map((kv) => kv.split('=').map(decodeURIComponent)));
  const drawer = ['settings', 'learn'].includes(q.d) ? q.d : null;
  const base = { jobId: null, drawer };
  switch (parts[0]) {
    case 'home': return { ...base, tab: 'welcome' };
    case 'explore': return { ...base, tab: 'graph' };
    case 'reactions': return { ...base, tab: 'analyze', analyze: 'reactions' };
    case 'kinetics': return { ...base, tab: 'analyze', analyze: 'kinetics' };
    case 'landscape': return { ...base, tab: 'analyze', analyze: 'landscape' };
    case 'calculations': return { ...base, tab: 'jobs' };
    case 'calc': return parts[1] ? { ...base, tab: 'job', jobId: parts[1], jobTab: parts[2] || null, run: q.run || null } : { ...base, tab: 'jobs' };
    case 'reaction': return parts[1] ? { ...base, tab: 'reaction', edgeId: parts[1] } : { ...base, tab: 'graph' };
    case 'network': return parts[1] === 'example' && parts[2] ? { ...base, tab: 'network', example: parts[2] }
      : parts[1] ? { ...base, tab: 'network', jobId: parts[1] } : { ...base, tab: 'graph' };
    case 'design': return { ...base, tab: 'design' };
    case 'reactor': return { ...base, tab: 'sandbox', sandbox: parts[1] ? { id: parts[1] } : null };
    default: return null;
  }
}
function viewFromHash(hash) {
  const v = parseHash(hash);
  if (!v) return null;
  const { analyze, ...view } = v;
  if (analyze) state.analyze = { ...(state.analyze || {}), view: analyze };
  return view;
}
const viewKey = (v) => formatHash(v);
let shownKey = null, restoring = false, replacing = false;
export function historyDepth() { return (typeof history !== 'undefined' && history.state?.mepdDepth) || 0; }
// A redirect the app makes on its own (first screen, a locked tab, back to a
// running reactor): it replaces the step instead of adding one, so Back doesn't bounce.
export function asRedirect(fn) {
  replacing = true;
  try { fn(); } finally { replacing = false; }
}
export function replaceView(view, extra = {}) { asRedirect(() => set({ ...extra, view })); }
export function goBack(fallback) {
  if (historyDepth() > 0) history.back(); else set({ view: fallback });
}
// Up one level: a drawer closes; a calculation goes back to the list; a mode or
// any other page goes back to Explore.
export function upLevel() {
  const v = state.view;
  if (v.drawer) { set({ view: { ...v, drawer: null } }); return; }
  if (v.tab === 'job') { set({ view: { tab: 'jobs', jobId: null } }); return; }
  if (v.tab === 'reaction') { set({ view: { tab: 'graph', jobId: null } }); return; }
  if (v.tab === 'network') { set({ view: v.jobId ? { tab: 'graph', jobId: null } : { tab: 'welcome', jobId: null } }); return; }
  if (v.tab !== 'graph') set({ view: { tab: 'graph', jobId: null } });
}
if (typeof window !== 'undefined' && window.history) {
  // The address wins (a link, a bookmark, a reload); else the history step's own view.
  const fromHash = viewFromHash(location.hash);
  const kept = history.state?.mepdView;
  if (fromHash) state.view = fromHash;
  else if (kept && kept.tab && !['welcome', 'sandbox'].includes(kept.tab)) state.view = normalizeView(kept, null);
  try { history.replaceState({ mepdView: state.view, mepdDepth: historyDepth() }, '', formatHash(state.view)); } catch { /* not cloneable */ }
  shownKey = viewKey(state.view);
  listeners.add(() => {
    const k = viewKey(state.view);
    if (k === shownKey) return;
    shownKey = k;
    if (restoring) return;
    if (replacing) { try { history.replaceState({ mepdView: state.view, mepdDepth: historyDepth() }, '', k); } catch { /* not cloneable */ } return; }
    try { history.pushState({ mepdView: state.view, mepdDepth: historyDepth() + 1 }, '', k); } catch { /* not cloneable */ }
  });
  window.addEventListener('popstate', (e) => {
    // A step this app pushed carries its view; an address typed or edited by hand only its hash.
    const v = e.state?.mepdView || viewFromHash(location.hash);
    if (!v) return;
    restoring = true;
    try { set({ view: v }); } finally { restoring = false; }
    shownKey = viewKey(state.view);
    if (!e.state?.mepdView) { try { history.replaceState({ mepdView: state.view, mepdDepth: historyDepth() }, '', shownKey); } catch { /* not cloneable */ } }
  });
}

// A reaction's own page: its barriers and its path as a movie (ReactionPlayer.js).
export function openReaction(edgeId) {
  set({ view: { tab: 'reaction', jobId: null, edgeId } });
}

export function openJob(jobId, tab = null) {
  // A network expansion lands on its live network; the calculation's own tabs are a click away.
  if (!tab && state.jobs[jobId]?.op === 'graph-enumeration') { set({ view: { tab: 'network', jobId } }); return; }
  const page = pageJobId(jobId);
  set({ view: { tab: 'job', jobId: page, run: page !== jobId ? jobId : null, jobTab: tab } });
}
// Leaving the Profiles tab with unsaved edits would silently drop them
// (jobs copy the profile *file*), so ask first.
export async function confirmProfileEdits() {
  const ed = state.profileEditor;
  if (!ed?.dirty) return true;
  if (confirm(`Save your changes to profile "${ed.name}"?\n\nOK saves them. Cancel discards them.`)) {
    return (await ed.save()) !== false;
  }
  set({ profileEditor: { ...ed, dirty: false } });
  return true;
}

export async function openTab(tab) {
  if (state.view.drawer === 'settings' && tab !== 'profiles' && !(await confirmProfileEdits())) return;
  set({ view: { tab, jobId: state.view.jobId } });
}
export async function closeDrawer() {
  if (state.view.drawer === 'settings' && !(await confirmProfileEdits())) return;
  set({ view: { ...state.view, drawer: null } });
}

// ---- per-viewer conveniences (never required for correctness) ----------
export const prefs = {
  get(key, fallback) {
    try { const v = localStorage.getItem('mepd:' + key); return v === null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem('mepd:' + key, JSON.stringify(value)); } catch { /* private mode */ }
  },
};
