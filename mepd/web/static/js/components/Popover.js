// Hover cards that are always fully visible (the live views: species boxes, steps, reactions).
// A card is placed beside what it describes, on the side with the most room, never over it; it flips
// or shifts to stay inside the page (the area under the top bar), and caps its height and scrolls when
// even that is too small. It is measured after it renders and again whenever its size changes (a
// structure image loads late). It stays open while the pointer is on it, so its links can be used.
// Keyboard: focus opens it, Escape closes it. Touch (or a click): tap pins it, a tap elsewhere closes it.
import { html, useEffect, useLayoutEffect, useRef, useState } from '../lib.js';

const PAD = 8, GAP = 10;

function boundsOf() {
  const b = { left: PAD, top: PAD, right: window.innerWidth - PAD, bottom: window.innerHeight - PAD };
  const body = document.querySelector('.wb-body')?.getBoundingClientRect();
  if (body && body.height > 100) { b.top = Math.max(b.top, body.top + PAD); b.bottom = Math.min(b.bottom, body.bottom - PAD); }
  return b;
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, Math.max(lo, hi)));

// Where a card of size w x h goes beside `a` (a DOMRect-like box) inside bounds `b`: {x, y, maxW, maxH}.
export function placeBeside(a, w, h, b) {
  const room = { right: b.right - a.right - GAP, left: a.left - GAP - b.left, below: b.bottom - a.bottom - GAP, above: a.top - GAP - b.top };
  const fits = { right: room.right >= w && b.bottom - b.top >= h, left: room.left >= w && b.bottom - b.top >= h,
    below: room.below >= h && b.right - b.left >= w, above: room.above >= h && b.right - b.left >= w };
  // the side where it fits with the most room; if none fits, the side with the most room (and a cap)
  const sides = ['right', 'left', 'below', 'above'];
  const fitting = sides.filter((s) => fits[s]);
  const side = (fitting.length ? fitting : sides).reduce((x, y) => {
    const rx = x === 'right' || x === 'left' ? room[x] / Math.max(1, w) : room[x] / Math.max(1, h);
    const ry = y === 'right' || y === 'left' ? room[y] / Math.max(1, w) : room[y] / Math.max(1, h);
    return ry > rx ? y : x;
  });
  let maxW = b.right - b.left, maxH = b.bottom - b.top;
  let x, y;
  if (side === 'right' || side === 'left') {
    if (!fits[side]) maxW = Math.max(160, room[side]);
    const cw = Math.min(w, maxW), ch = Math.min(h, maxH);
    x = side === 'right' ? a.right + GAP : a.left - GAP - cw;
    y = clamp(a.top + a.height / 2 - ch / 2, b.top, b.bottom - ch);
  } else {
    if (!fits[side]) maxH = Math.max(120, room[side]);
    const cw = Math.min(w, maxW), ch = Math.min(h, maxH);
    y = side === 'below' ? a.bottom + GAP : a.top - GAP - ch;
    x = clamp(a.left + a.width / 2 - cw / 2, b.left, b.right - cw);
  }
  return { x: clamp(x, b.left, b.right - Math.min(w, maxW)), y: clamp(y, b.top, b.bottom - Math.min(h, maxH)), maxW, maxH, side };
}

// The card itself, fixed in the window; `anchor` is a box in window coordinates.
export function Popover({ anchor, className = '', onEnter, onLeave, children }) {
  const ref = useRef(null);
  // placed by hand (not by render): measured at its natural size, then put where it fits
  const place = () => {
    const el = ref.current;
    if (!el || !anchor) return;
    el.style.maxWidth = 'none';
    el.style.maxHeight = 'none';
    const p = placeBeside(anchor, el.offsetWidth, el.offsetHeight, boundsOf());
    Object.assign(el.style, { left: `${p.x}px`, top: `${p.y}px`, maxWidth: `${p.maxW}px`, maxHeight: `${p.maxH}px`, visibility: 'visible' });
  };
  useLayoutEffect(place, [anchor?.left, anchor?.top, anchor?.width, anchor?.height, children]);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => place());
    ro.observe(el);
    const imgs = [...el.querySelectorAll('img')];
    for (const im of imgs) im.addEventListener('load', place);
    window.addEventListener('resize', place);
    return () => { ro.disconnect(); for (const im of imgs) im.removeEventListener('load', place); window.removeEventListener('resize', place); };
  });
  return html`<div ref=${ref} class=${`popover ${className}`} role="tooltip" onMouseEnter=${onEnter} onMouseLeave=${onLeave}
    style="left:0;top:0;visibility:hidden">${children}</div>`;
}

const boxOf = (r) => ({ left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height });
// A small box around a point (a step's middle, where the pointer is).
export const pointBox = (x, y, r = 6) => ({ left: x - r, top: y - r, right: x + r, bottom: y + r, width: 2 * r, height: 2 * r });

// Which card is open and where: bind(key, data, anchorOf?) spreads onto what is hovered (an SVG <g> or
// an element); `card` onto the Popover. anchorOf(element, event) -> box, default the element's own box.
export function useHoverCard() {
  const [cur, setCur] = useState(null);   // {key, data, anchor, pinned}
  const timer = useRef(null);
  const curRef = useRef(null);
  curRef.current = cur;
  const cancel = () => clearTimeout(timer.current);
  const closeSoon = () => { cancel(); timer.current = setTimeout(() => setCur((c) => (c && !c.pinned ? null : c)), 180); };
  const open = (key, data, anchor, pinned = false) => { cancel(); setCur({ key, data, anchor, pinned }); };
  useEffect(() => {
    // Escape closes an open card first (before the page's own Escape, which leaves the page).
    const onKey = (e) => { if (e.key === 'Escape' && curRef.current) { e.preventDefault(); e.stopPropagation(); setCur(null); } };
    // a tap or click elsewhere closes a pinned card
    const onDown = (e) => {
      const c = curRef.current;
      if (c?.pinned && !e.target.closest?.('.popover') && !e.target.closest?.('[data-hc]')) setCur(null);
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('pointerdown', onDown, true);
    return () => { window.removeEventListener('keydown', onKey, true); window.removeEventListener('pointerdown', onDown, true); cancel(); };
  }, []);
  const bind = (key, data, anchorOf = null) => {
    const at = (el, e) => (anchorOf ? anchorOf(el, e) : boxOf(el.getBoundingClientRect()));
    return {
      'data-hc': key, tabindex: 0,
      onMouseEnter: (e) => { if (!curRef.current?.pinned) open(key, data, at(e.currentTarget, e)); else cancel(); },
      onMouseLeave: () => { if (!curRef.current?.pinned) closeSoon(); },
      onFocus: (e) => open(key, data, at(e.currentTarget, null)),
      onBlur: () => { if (!curRef.current?.pinned) closeSoon(); },
      onClick: (e) => {
        const el = e.currentTarget;
        setCur((c) => (c?.key === key && c.pinned ? null : { key, data, anchor: at(el, e), pinned: true }));
      },
    };
  };
  const card = { onEnter: cancel, onLeave: () => { if (!curRef.current?.pinned) closeSoon(); } };
  return { cur, bind, card, close: () => setCur(null), is: (key) => cur?.key === key };
}
