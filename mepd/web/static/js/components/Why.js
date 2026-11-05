// "?": what an option really does, for whoever wants to know. Nothing shows
// until it is clicked: the entry (static/explain.json, keyed op.<key>,
// param.<op>.<field>, profile.<key>, kinetics.<name>, level.<name>, other.<name>)
// says what it changes, how mepd does it and with which program, the
// default, what else could stand in and where, and the code behind it;
// `refs` links its References section, `manual` its manual page (page#anchor).
import { html, useEffect, useRef, useState } from '../lib.js';
import { set } from '../store.js';

let entries = null;
const waiting = new Set();
const load = fetch('/static/explain.json').then((r) => (r.ok ? r.json() : {})).catch(() => ({}))
  .then((d) => { entries = d; for (const f of waiting) f(); waiting.clear(); });

function useEntries() {
  const [, force] = useState(0);
  useEffect(() => { if (!entries) { const f = () => force((n) => n + 1); waiting.add(f); return () => waiting.delete(f); } return undefined; }, []);
  return entries;
}

export function hasWhy(id) { return !!entries?.[id]; }

// The first id that has an entry (e.g. a field of this operation, else the field anywhere).
export function Why({ id, ids, label, fallback = null }) {
  const all = useEntries();
  const [open, setOpen] = useState(false);
  const btn = useRef(null);
  const pop = useRef(null);
  const key = (ids || [id]).find((k) => all?.[k]);
  useEffect(() => {
    if (!open) return undefined;
    const away = (e) => { if (!pop.current?.contains(e.target) && !btn.current?.contains(e.target)) setOpen(false); };
    const esc = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('pointerdown', away, true);
    document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('pointerdown', away, true); document.removeEventListener('keydown', esc); };
  }, [open]);
  // Place it under the button, inside the window.
  useEffect(() => {
    if (!open || !pop.current || !btn.current) return;
    const b = btn.current.getBoundingClientRect(), p = pop.current;
    const w = Math.min(380, window.innerWidth - 24);
    p.style.width = `${w}px`;
    p.style.left = `${Math.max(12, Math.min(b.left - 16, window.innerWidth - w - 12))}px`;
    const below = b.bottom + 6, h = p.offsetHeight;
    p.style.top = `${below + h > window.innerHeight - 12 && b.top - h - 6 > 12 ? b.top - h - 6 : Math.min(below, Math.max(12, window.innerHeight - h - 12))}px`;
  }, [open]);
  if (!key) return fallback;
  const e = all[key];
  const stop = (ev) => { ev.preventDefault(); ev.stopPropagation(); };
  return html`<span class="why" onClick=${stop}>
    <button type="button" ref=${btn} class=${`why-btn ${open ? 'on' : ''}`} aria-expanded=${open}
      aria-label=${`How ${label || e.title} works`} title="How it works" onClick=${(ev) => { stop(ev); setOpen(!open); }}>?</button>
    ${open && html`<div class="why-pop" ref=${pop} role="dialog" aria-label=${e.title}>
      <div class="why-head"><b>${e.title}</b><button type="button" class="btn-icon" aria-label="Close" onClick=${() => setOpen(false)}>✕</button></div>
      ${e.does && html`<p>${e.does}</p>`}
      ${e.how?.length > 0 && html`<h5>How mepd does it</h5><ol>${e.how.map((s) => html`<li>${s}</li>`)}</ol>`}
      ${e.default && html`<h5>Default</h5><p>${e.default}</p>`}
      ${e.swap?.length > 0 && html`<h5>Other choices</h5><ul class="why-swap">${e.swap.map((s) => html`<li><b>${s.option}</b><span>${s.where}</span></li>`)}</ul>`}
      ${e.caveats?.length > 0 && html`<h5>Keep in mind</h5><ul>${e.caveats.map((s) => html`<li>${s}</li>`)}</ul>`}
      <div class="why-foot">
        ${e.code?.length > 0 && html`<div><span class="muted">Code</span> ${e.code.map((c) => html`<code>${c}</code>`)}</div>`}
        ${e.cli && html`<div><span class="muted">Command</span> <code>${e.cli}</code></div>`}
        ${e.refs && html`<div><a href="#" onClick=${(ev) => { ev.preventDefault(); setOpen(false); set({ view: { tab: 'refs', jobId: null }, refsFocus: e.refs }); }}>What to cite ›</a></div>`}
        ${e.manual && html`<div><a href=${`/manual/${e.manual}`} target="_blank" rel="noopener">Read more in the manual ›</a></div>`}
      </div>
    </div>`}
  </span>`;
}

export const explainReady = load;
