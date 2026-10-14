// One small drawing per quest (quests.js): what it does, at a glance.
// 24×24, drawn with the current text colour.
import { html } from './lib.js';

const ring = (r, n, d = 1) => Array.from({ length: n }, (_, i) => {
  const a = (i / n) * Math.PI * 2;
  return html`<circle cx=${12 + r * Math.cos(a)} cy=${12 + r * Math.sin(a)} r=${d} class="fill" />`;
});

const DRAW = {
  molecule: html`<circle cx="7" cy="15" r="3"/><circle cx="17" cy="9" r="3"/><path d="M9.6 13.4l4.8-2.8"/>`,
  reactor: html`<circle cx="12" cy="12" r="9.5" stroke-dasharray="2.5 2"/><circle cx="9" cy="10" r="1.8" class="fill"/>
    <circle cx="14.5" cy="13.5" r="1.8" class="fill"/><circle cx="9.5" cy="15.5" r="1.1" class="fill"/><circle cx="15" cy="8.2" r="1.1" class="fill"/>`,
  reaction: html`<circle cx="5" cy="12" r="2.6"/><circle cx="19.5" cy="7.5" r="2"/><circle cx="19.5" cy="16.5" r="2"/>
    <path d="M9 12h5.5M12.5 9.5L15 12l-2.5 2.5"/>`,
  barrier: html`<path d="M2 19h3.5c3.5 0 4-12 6.5-12s3 8.5 6.5 8.5H22"/><path d="M12 3v2.2M10.9 4.1h2.2"/>`,
  network: html`<circle cx="5" cy="6" r="2"/><circle cx="19" cy="6.5" r="2"/><circle cx="7.5" cy="18" r="2"/><circle cx="18" cy="17.5" r="2"/>
    <path d="M7 6.1l10 .3M5.4 8l1.7 8M9.5 18h6.5M18.8 8.5l-.6 7"/>`,
  connect: html`<circle cx="5" cy="14" r="3"/><circle cx="19" cy="14" r="3"/><path d="M7.5 11.5c3-5 6-5 9 0" stroke-dasharray="2 2"/>
    <path d="M14.2 10.6l2.3.9.6-2.4"/>`,
  product: html`<path d="M8 3.5l5 2.9v5.8l-5 2.9-5-2.9V6.4z"/><path d="M13.5 20.5l7-7 2 2-7 7h-2z"/>`,
  shapes: html`<path d="M3 8l4.5-3 4.5 3 4.5-3L21 8"/><path d="M3 17l4.5 2.5 3-5 4.5 3 6-4" class="faint"/>`,
  method: html`<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2" class="bg"/><circle cx="16" cy="12" r="2" class="bg"/>
    <circle cx="7" cy="18" r="2" class="bg"/>`,
  kinetics: html`<path d="M4 3.5V20h16.5"/><path d="M5 6c5 0 6 11 14.5 11.5"/><path d="M5 18.5C10 18 11 8 19.5 7.5" class="faint"/>`,
  discover: html`<circle cx="10" cy="10" r="6"/><path d="M14.5 14.5L20.5 20.5M10 7v6M7 10h6"/>`,
  solvent: html`<path d="M12 2.8c4.2 5 6.2 8.2 6.2 11.4a6.2 6.2 0 0 1-12.4 0c0-3.2 2-6.4 6.2-11.4z"/><path d="M9 15a3 3 0 0 0 3 3" class="faint"/>`,
  channels: html`<circle cx="3.5" cy="12" r="2"/><circle cx="20.5" cy="12" r="2"/><path d="M5.5 12h13M5.5 11C9 4 15 4 18.5 11M5.5 13C9 20 15 20 18.5 13"/>`,
  catalyst: html`<path d="M2 19.5c4 0 5-14 10-14s6 14 10 14"/><path d="M4 19.5c3.5 0 5-7 8-7s4.5 7 8 7" class="accent" stroke-dasharray="2 1.6"/>`,
  retro: html`<rect x="2" y="4.5" width="4.5" height="4.5"/><rect x="2" y="15" width="4.5" height="4.5"/>
    <path d="M17 6.5l4.2 2.5v5L17 16.5 12.8 14V9z"/><path d="M11 12H8M9.6 10.2L8 12l1.6 1.8"/>`,
  variants: html`<path d="M7 5.5l4.3 2.5v5L7 15.5 2.7 13V8z"/><circle cx="18.5" cy="8" r="2.4"/><circle cx="18.5" cy="17.5" r="2.4" class="fill"/>
    <path d="M11.5 13.5l4.8 2.8M18.5 10.5v4.5" stroke-dasharray="1.6 1.6"/>`,
  force: html`<circle cx="9" cy="12" r="2.5"/><circle cx="15" cy="12" r="2.5"/><path d="M11.5 12h1M1.8 12h4.2M3.8 10l-2 2 2 2M22.2 12H18M20.2 10l2 2-2 2"/>`,
  explicit: html`<circle cx="12" cy="12" r="3.2"/>${ring(8.2, 10, 1.05)}`,
  optimize: html`<path d="M2.5 5.5c4.5 0 5 13 9.5 13s5-13 9.5-13"/><circle cx="12" cy="18.5" r="1.7" class="fill"/>`,
  complex: html`<circle cx="8.5" cy="12" r="4.5"/><circle cx="16.5" cy="12" r="3.5"/><path d="M12.8 10.2l.9.6" stroke-dasharray="1 1.4"/>`,
  protein: html`<path d="M4 16c0-6 3-11 8-11s8 5 8 11"/><path d="M4 16c2 2 3.5 2 5 0M15 16c1.5 2 3 2 5 0"/><circle cx="12" cy="14" r="2.2" class="fill"/>`,
  bifurcation: html`<path d="M2.5 12H10c4 0 6-6.5 11.5-6.5M10 12c4 0 6 6.5 11.5 6.5"/><circle cx="10" cy="12" r="1.6" class="fill"/>`,
};

export function QuestIcon({ id, size = 34 }) {
  return html`<svg class="q-icon" viewBox="0 0 24 24" width=${size} height=${size} aria-hidden="true"
    fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${DRAW[id] || ''}</svg>`;
}
