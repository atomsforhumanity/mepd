// A reacting mixture (`mepd discovery mixture`) as it stands, at #/network/<job>:
// a network of molecules, not of one system's states. Every combination of
// the feed molecules (copies of one included: A + A) and, generation by
// generation, every populated product next to each feed molecule was an
// explorer run; their steps are merged into reactions between molecules
// with mass-action kinetics from the feed's concentrations.
// - a molecule is its structure, on a card whose area grows with the most of
//   it there ever was (the feed: fixed, marked);
// - a reaction is a dot joining its reactants and products, coloured by its
//   barrier the way material went, as large as the amount converted; a
//   molecule on both sides (a proton shuttle, a catalyst) joins it dashed.
// Experimental: the mixture explorer is not yet benchmarked.
import { html, useEffect, useMemo, useState } from '../lib.js';
import { api } from '../api.js';
import { openJob, useStore } from '../store.js';
import { depictUrl } from '../util.js';
import { Popover, pointBox, useHoverCard } from './Popover.js';
import { MixtureDesign, designParam } from './LiveDesigns.js';

const W = 1250, H = 820, CX = W / 2, CY = H / 2 - 20, M = 72;
const BOX_MIN = 46, BOX_MAX = 112, BOX_FEED = 92;
const band = (v) => (v == null ? 'var(--edge-idle)' : v < 10 ? 'var(--chart-3)' : v < 20 ? 'var(--chart-4)' : 'var(--chart-5)');
const MAX_MOLECULES = 40, MAX_REACTIONS = 60;

function springLayout(ids, links, sizes, fixed) {
  const n = ids.length;
  const idx = Object.fromEntries(ids.map((id, i) => [id, i]));
  const E = links.filter(([a, b]) => a in idx && b in idx).map(([a, b]) => [idx[a], idx[b]]);
  const deg = ids.map(() => 0);
  for (const [a, b] of E) { deg[a] += 1; deg[b] += 1; }
  const pos = ids.map((id, i) => {
    const a = i * 2.39996, r = 60 + 260 * Math.sqrt(i / Math.max(1, n));
    return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
  });
  const k = Math.min(150, Math.sqrt((W * H) / Math.max(1, n)) * 0.8);
  for (let it = 0; it < 300; it += 1) {
    const t = 16 * (1 - it / 300) + 0.5;
    const d = pos.map(() => [0, 0]);
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
        const dd = Math.max(1, Math.hypot(dx, dy));
        const clear = (sizes[i] + sizes[j]) / 2 + 30;
        const f = (dd < 3 * k ? (k * k) / dd : 0) + (dd < clear ? (clear - dd) * 6 : 0);
        if (!f) continue;
        d[i][0] += (dx / dd) * f; d[i][1] += (dy / dd) * f; d[j][0] -= (dx / dd) * f; d[j][1] -= (dy / dd) * f;
      }
    }
    for (const [a, b] of E) {
      const dx = pos[a][0] - pos[b][0], dy = pos[a][1] - pos[b][1];
      const dd = Math.max(1, Math.hypot(dx, dy)), f = (dd * dd) / k / Math.sqrt(Math.max(deg[a], deg[b]));
      d[a][0] -= (dx / dd) * f; d[a][1] -= (dy / dd) * f; d[b][0] += (dx / dd) * f; d[b][1] += (dy / dd) * f;
    }
    for (let i = 0; i < n; i += 1) {
      const g = deg[i] ? 0.04 : 0.3;
      d[i][0] += (CX - pos[i][0]) * g; d[i][1] += (CY - pos[i][1]) * g * 1.5;
      if (fixed.has(ids[i])) continue;
      const dl = Math.max(1, Math.hypot(d[i][0], d[i][1]));
      pos[i][0] = Math.min(W - M, Math.max(M, pos[i][0] + (d[i][0] / dl) * Math.min(dl, t)));
      pos[i][1] = Math.min(H - M - 40, Math.max(M, pos[i][1] + (d[i][1] / dl) * Math.min(dl, t)));
    }
  }
  return Object.fromEntries(ids.map((id, i) => [id, pos[i]]));
}

function fmtC(x) {
  if (x == null) return '—';
  if (x === 0) return '0';
  return Math.abs(x) >= 0.01 && Math.abs(x) < 1000 ? x.toPrecision(2) : x.toExponential(1);
}

export function MixtureView({ jobId }) {
  const job = useStore((s) => s.jobs[jobId]);
  const running = ['running', 'queued'].includes(job?.status);
  const [d, setD] = useState(null);
  const [error, setError] = useState(null);
  // the reaction under the pointer (or focused, or pinned by a tap), its card beside it (Popover.js)
  const hc = useHoverCard();
  const hover = hc.cur?.data ?? null;
  useEffect(() => {
    let alive = true;
    const load = () => api.get(`/api/jobs/${jobId}/mixture`).then((x) => { if (alive) { setD(x); setError(null); } })
      .catch((e) => alive && setError(e.message));
    load();
    const t = running ? setInterval(load, 5000) : null;
    return () => { alive = false; if (t) clearInterval(t); };
  }, [jobId, running]);

  const m = d?.mixture;
  const view = useMemo(() => {
    if (!m) return null;
    const feed = new Set((m.feed || []).map((f) => f.molecule));
    const mols = m.molecules || [];
    const scale = Math.max(1e-30, ...mols.filter((x) => feed.has(x.molecule)).map((x) => x.max_concentration || 0));
    const made = mols.filter((x) => !feed.has(x.molecule) && (x.max_concentration || 0) > 0)
      .sort((a, b) => b.max_concentration - a.max_concentration).slice(0, MAX_MOLECULES);
    const shown = [...mols.filter((x) => feed.has(x.molecule)), ...made];
    const keep = new Set(shown.map((x) => x.molecule));
    const rx = (m.reactions || []).filter((r) => [...r.reactants, ...r.products].every((k) => keep.has(k)))
      .sort((a, b) => Math.abs(b.net_extent || 0) - Math.abs(a.net_extent || 0)).slice(0, MAX_REACTIONS)
      .map((r, i) => ({ ...r, id: `r${i}` }));
    const peakMade = Math.max(1e-30, ...made.map((x) => x.max_concentration || 0));
    const size = (x) => (feed.has(x.molecule) ? BOX_FEED
      : Math.min(BOX_MAX, Math.max(BOX_MIN, BOX_MAX * Math.sqrt((x.max_concentration || 0) / peakMade))));
    const ids = [...shown.map((x) => x.molecule), ...rx.map((r) => r.id)];
    const sizes = [...shown.map(size), ...rx.map(() => 14)];
    const links = rx.flatMap((r) => [...r.reactants.map((k) => [k, r.id]), ...r.products.map((k) => [r.id, k])]);
    const pos = springLayout(ids, links, sizes, new Set());
    const maxExt = Math.max(1e-30, ...rx.map((r) => Math.abs(r.net_extent || 0)));
    return { feed, shown, rx, size, pos, maxExt, scale };
  }, [m]);

  if (error) return html`<div class="ln-empty"><p>${error}</p></div>`;
  if (!d) return html`<p class="boot">Loading the mixture…</p>`;
  // The feed's own names (as given to the mixture), else everyday names, else the SMILES.
  const names = { ...(d.names || {}), ...Object.fromEntries(((m && m.feed) || []).filter((f) => f.name && f.name !== f.molecule).map((f) => [f.molecule, f.name])) };
  if (m && (m.molecules || []).length && ['a', 'b', 'c'].includes(designParam())) {   // prototype designs (dev): LiveDesigns.js
    return html`<${MixtureDesign} mixture=${m} names=${names} design=${designParam()} jobId=${jobId} job=${job} running=${running} />`;
  }
  const nameOf = (k) => names[k] || k.split('|')[0];   // "A|B": forms counted as one species
  const unit = m?.settings?.standard_state === '1atm' ? 'atm' : 'M';
  const seeds = m?.seeds || d.seeds || [];
  const doneSeeds = seeds.filter((s) => String(s.status || '').startsWith('done')).length;
  const doors = html`<div class="ln-doors">
    <button class="btn small" onClick=${() => openJob(jobId, 'live')}>Seeds' progress</button>
    <button class="btn small" onClick=${() => openJob(jobId, 'result')} disabled=${job?.status === 'queued'}>Results</button>
    <button class="btn small" onClick=${() => openJob(jobId, 'log')}>Log</button></div>`;
  const head = html`<div class="ln-head"><h2>${job?.title?.replace(/^Reacting mixture[^:]*:\s*/, 'Mixture: ') || 'Reacting mixture'}</h2>
    <span class="ln-exp" title="The mixture explorer is not yet benchmarked against experiment">experimental</span>
    ${running && html`<span class="ln-live">● live</span>`}<span class="spacer"></span>
    ${m && html`<span class="small muted">${m.settings?.temperature_K} K · ${m.settings?.time_s} s · generation ${m.generation ?? 0}</span>`}</div>`;
  if (!m || !view) {
    return html`<div class="live-network"><div class="ln-main">${head}
      <div class="ln-stage ln-waiting"><div class="ln-seed-pulse"></div>
        <p>${running ? 'The first seeds are running; the merged network appears after the first one finishes.' : 'This mixture has no merged network.'}</p>
        <p class="small muted">${seeds.length} seed${seeds.length === 1 ? '' : 's'} started · ${doneSeeds} done</p></div></div>
      <aside class="ln-side"><section><span class="section-title">Look closer</span>${doors}</section></aside></div>`;
  }
  const { feed, shown, rx, size, pos, maxExt, scale } = view;
  const formed = shown.filter((x) => !feed.has(x.molecule) && (x.final_concentration || 0) > 0)
    .sort((a, b) => b.final_concentration - a.final_concentration);
  const formedTotal = formed.reduce((t, x) => t + x.final_concentration, 0);
  const hov = hover && rx.find((r) => r.id === hover);
  const label = (k) => nameOf(k).length > 26 ? `${nameOf(k).slice(0, 24)}…` : nameOf(k);
  return html`<div class="live-network">
    <div class="ln-main">
      ${head}
      <div class="ln-stage">
        <svg viewBox=${`0 0 ${W} ${H}`} width="100%" role="img" aria-label="The mixture's molecules as their structures, sized by how much of each there was; its reactions as dots joining reactants and products, coloured by barrier and sized by the amount converted">
          <defs>${['low', 'mid', 'high', 'idle'].map((k) => html`<marker id=${`mx-arr-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" class=${`ln-arr ${k}`} /></marker>`)}</defs>
          ${rx.map((r) => {
            const fwd = (r.net_extent || 0) >= 0;
            const barrier = fwd ? r.barrier_kcal : r.reverse_barrier_kcal;
            const k = barrier == null ? 'idle' : barrier < 10 ? 'low' : barrier < 20 ? 'mid' : 'high';
            const ins = fwd ? r.reactants : r.products, outs = fwd ? r.products : r.reactants;
            const both = new Set(r.reactants.filter((x) => r.products.includes(x)));
            const share = Math.abs(r.net_extent || 0) / maxExt;
            const strong = share >= 0.01;
            const p = pos[r.id], col = band(barrier), on = hover === r.id;
            return html`<g class=${`mx-rx ${strong ? 'main' : 'minor'} ${on ? 'on' : ''}`}
                ...${hc.bind(`r:${r.id}`, r.id, (el) => { const c = el.querySelector('.mx-dot').getBoundingClientRect(); return pointBox(c.left + c.width / 2, c.top + c.height / 2, Math.max(8, c.width / 2)); })}>
              ${[...new Set(ins)].map((x) => pos[x] && html`<g><line x1=${pos[x][0]} y1=${pos[x][1]} x2=${p[0]} y2=${p[1]}
                style=${`stroke:${col}`} class=${`mx-link ${both.has(x) ? 'shuttle' : ''}`} />
                ${ins.filter((y) => y === x).length > 1 && html`<text x=${(pos[x][0] + p[0]) / 2} y=${(pos[x][1] + p[1]) / 2 - 6} class="mx-times">×${ins.filter((y) => y === x).length}</text>`}</g>`)}
              ${[...new Set(outs)].filter((x) => !both.has(x)).map((x) => pos[x] && html`<line x1=${p[0]} y1=${p[1]}
                x2=${p[0] + (pos[x][0] - p[0]) * 0.8} y2=${p[1] + (pos[x][1] - p[1]) * 0.8}
                style=${`stroke:${col}`} class="mx-link" marker-end=${strong || on ? `url(#mx-arr-${k})` : null} />`)}
              <circle cx=${p[0]} cy=${p[1]} r=${4 + 7 * Math.sqrt(share)} style=${`fill:${col}`} class="mx-dot" />
              ${(r.channels || 1) > 1 && html`<text x=${p[0] + 7 + 7 * Math.sqrt(share)} y=${p[1] + 4} class="mx-times">×${r.channels}</text>`}
              <circle cx=${p[0]} cy=${p[1]} r="12" class="ln-hit" />
            </g>`;
          })}
          ${[...shown].sort((a, b) => size(a) - size(b)).map((x) => {
            const w = size(x), h = w * 0.75, at = pos[x.molecule];
            if (!at) return null;
            return html`<g class="ln-node" transform=${`translate(${at[0]},${at[1]})`}>
              <rect x=${-w / 2} y=${-h / 2} width=${w} height=${h} rx="4" class=${`ln-card ${feed.has(x.molecule) ? 'seed' : (x.max_concentration || 0) >= 0.01 * scale ? 'pop' : ''}`} />
              <image href=${depictUrl(x.molecule, 160, 120)} x=${-w / 2 + 2} y=${-h / 2 + 2} width=${w - 4} height=${h - 4} preserveAspectRatio="xMidYMid meet" />
              <title>${nameOf(x.molecule)} · peak ${fmtC(x.max_concentration)} ${unit} · end ${fmtC(x.final_concentration)} ${unit}${feed.has(x.molecule) ? ' · feed' : ''}</title>
            </g>`;
          })}
          ${shown.filter((x) => feed.has(x.molecule) || names[x.molecule] || (x.max_concentration || 0) >= 0.05 * scale).map((x) => pos[x.molecule] && html`<text
            x=${pos[x.molecule][0]} y=${pos[x.molecule][1] + size(x) * 0.375 + 15} class=${`ln-label ${feed.has(x.molecule) ? 'product' : ''}`}>${label(x.molecule)}</text>`)}
        </svg>
        ${hov && (() => {
          const fwd = (hov.net_extent || 0) >= 0;
          const [a, b] = fwd ? [hov.reactants, hov.products] : [hov.products, hov.reactants];
          return html`<${Popover} anchor=${hc.cur.anchor} className="ln-edge-card" onEnter=${hc.card.onEnter} onLeave=${hc.card.onLeave}>
            <div class="ln-ec-route"><b>${a.map(nameOf).join(' + ')}</b> <span class="muted">→</span> <b>${b.map(nameOf).join(' + ')}</b></div>
            <div class="ln-ec-nums">
              <span><b style=${`color:${band(fwd ? hov.barrier_kcal : hov.reverse_barrier_kcal)}`}>${(fwd ? hov.barrier_kcal : hov.reverse_barrier_kcal)?.toFixed(1)}</b>barrier</span>
              <span><b>${(fwd ? hov.reverse_barrier_kcal : hov.barrier_kcal)?.toFixed(1)}</b>back</span>
              <span><b>${fmtC(Math.abs(hov.net_extent || 0))}</b>${unit} converted</span>
            </div>
            <div class="small muted">free energy, kcal/mol, from the molecules apart${hov.source ? ` · found in ${hov.source}` : ''}</div>
            ${(hov.channels || 1) > 1 && html`<div class="small"><b>×${hov.channels}</b> ${hov.mirror_images ? 'mirror images: two saddles of equal energy, both counted' : 'channels, counted together'}</div>`}
          <//>`;
        })()}
      </div>
      <div class="ln-legend small">
        <span class="ln-lg"><svg width="64" height="14"><line x1="2" y1="7" x2="18" y2="7" class="lg-l low" /><line x1="23" y1="7" x2="39" y2="7" class="lg-l mid" /><line x1="44" y1="7" x2="60" y2="7" class="lg-l high" /></svg>${'barrier: low (<10) · medium · high (>20 kcal/mol)'}</span>
        <span class="ln-lg"><svg width="20" height="14"><circle cx="10" cy="7" r="5" class="lg-dot" /></svg>a reaction; larger = more converted</span>
        <span class="ln-lg"><svg width="30" height="14"><line x1="2" y1="7" x2="28" y2="7" class="lg-l dash" /></svg>on both sides (a shuttle)</span>
        <span class="ln-lg"><svg width="40" height="16"><rect x="1" y="5" width="12" height="9" rx="2" class="lg-card" /><rect x="18" y="1" width="20" height="14" rx="2" class="lg-card" /></svg>bigger card: more of it at its peak</span>
        <span class="ln-lg muted">${(m.molecules || []).length} molecules · ${(m.reactions || []).length} reactions in all${(m.channels || 0) > (m.reactions || []).length ? ` (${m.channels} channels with mirror images)` : ''}; the most populated shown</span>
      </div>
    </div>
    <aside class="ln-side">
      <section><span class="section-title">Feed</span>
        ${(m.feed || []).map((f) => html`<div class="ln-prod" title=${f.molecule}>
          <img class="ln-prod-mol" src=${depictUrl(f.molecule, 160, 120)} alt=${f.molecule} />
          <span class="ln-prod-name">${nameOf(f.molecule)}</span><b>${fmtC(f.concentration)} ${unit}</b></div>`)}
      </section>
      <section><span class="section-title">What forms</span>
        ${formed.slice(0, 5).map((x) => html`<div class="ln-prod" title=${x.molecule}>
          <img class="ln-prod-mol" src=${depictUrl(x.molecule, 160, 120)} alt=${x.molecule} />
          <span class="ln-prod-name">${nameOf(x.molecule)}</span>
          <b>${formedTotal > 0 ? (x.final_concentration / formedTotal >= 0.995 ? '>99%' : x.final_concentration / formedTotal >= 0.01 ? `${Math.round(100 * x.final_concentration / formedTotal)}%` : (x.final_concentration / formedTotal >= 1e-4 ? `${(100 * x.final_concentration / formedTotal).toPrecision(1)}%` : '<0.01%')) : '—'}</b></div>`)}
        ${!formed.length && html`<span class="small muted">Nothing yet</span>`}
      </section>
      <section><span class="section-title">Seeds</span>
        <div class="big-counts ln-counts"><span><b>${seeds.length}</b>seeds</span><span><b>${doneSeeds}</b>done</span><span><b>${m.generation ?? 0}</b>generation</span></div>
        <span class="small muted">Each seed is one combination of molecules explored on its own; its steps join this network.</span>
      </section>
      ${(m.notes || []).length > 0 && html`<section><span class="section-title">Notes</span>
        <ul class="ln-feed">${m.notes.slice(0, 4).map((n) => html`<li class="muted">${n}</li>`)}</ul></section>`}
      <section><span class="section-title">Look closer</span>${doors}</section>
    </aside>
  </div>`;
}
