// The Nanoreactor setup (#/reactor/new): what goes in the box (each species' SMILES, how many, its charge
// and spin), the conditions, and two ways to run it. Launch nanoreactor starts the unattended Nanoreactor job
// (piston MD, then each reaction cut out and refined; it opens on that calculation). Steer by hand starts
// the interactive reactor (Sandbox.js) on the same box. Nothing is added to the session until one of them
// is pressed; the rows are read by /api/species/preview as you type.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, openReactorList, set, toast, useStore } from '../store.js';
import { depictUrl } from '../util.js';
import { ProfilePicker, defaultProfile } from './Actions.js';

const SPINS = [[1, 'singlet'], [2, 'doublet'], [3, 'triplet'], [4, 'quartet'], [5, 'quintet']];
const SPIN_OF_WORD = Object.fromEntries(SPINS.map(([m, w]) => [w, m]));
const spinName = (m) => SPINS.find(([k]) => k === m)?.[1] || `multiplicity ${m}`;
const STEER_MAX_ATOMS = 500;   // the most the interactive reactor takes (sandbox.MAX_ATOMS)
// What Steer by hand runs at this size here (GET /api/sandbox-methods: a GPU? sandbox.pick_method's sizes), in
// words, with the speeds measured on the laptop (2026-11-27, ReactorSetup's box).
function steerSpeed(atoms, m, chosen = 'auto') {
  if (!m || atoms > m.max_atoms) return null;
  if (chosen === 'uma' && !m.gpu) return { slow: true, text: 'There is no GPU here for UMA: Steer by hand would run GFN2-xTB instead.' };
  if (chosen === 'uma' || (chosen === 'auto' && m.gpu && atoms >= m.uma_from_atoms)) {
    return { slow: false, text: `Steer by hand runs UMA (a machine-learned potential) on the GPU at this size: about 40 MD steps a second at 120 atoms, 11 at 480. It takes ~10 s to load.` };
  }
  if (atoms >= m.threads_from_atoms) {
    return { slow: true, text: `Steer by hand runs GFN2-xTB on ${m.big_threads} threads at this size, and slows down a lot: about 1.5 MD steps a second at 240 atoms, one every 4 s at 480${m.gpu ? '' : ' (no GPU here for UMA)'}.` };
  }
  return atoms >= 100 ? { slow: false, text: 'Steer by hand runs GFN2-xTB: about 6 MD steps a second at 120 atoms.' } : null;
}

let lastKey = 0;
const newRow = (smiles = '', n = 1, mult = null) => ({ key: ++lastKey, smiles, n, mult });

// '2 CC=O, 4 O, [OH-], O=O triplet' -> rows. One species per comma, semicolon or line; a count before or
// after it ('2 CC=O', '2x CC=O', 'CC=O*2', '2*CC=O', 'CC=O 2'); a spin word anywhere (singlet ... quintet);
// everyday names ('water', 'hydroxide') are read by the server.
export function parseMix(text) {
  const rows = [];
  for (const item of String(text || '').split(/[,;\n]+/)) {
    let n = 1, mult = null;
    let words = item.trim().split(/\s+/).filter(Boolean).filter((w) => {
      const m = SPIN_OF_WORD[w.toLowerCase()];
      if (m) mult = m;
      return !m;
    });
    if (words.length > 1 && /^\d+[x×*]?$/i.test(words[0])) { n = parseInt(words[0], 10); words = words.slice(1); }
    else if (words.length > 1 && /^[x×*]?\d+$/i.test(words[words.length - 1])) { n = parseInt(words[words.length - 1].replace(/^\D/, ''), 10); words = words.slice(0, -1); }
    let smiles = words.join(' ');
    let m;
    if ((m = /^(\d+)\s*[*×]\s*(.+)$/.exec(smiles))) { n = parseInt(m[1], 10); smiles = m[2]; }
    else if ((m = /^(.+?)\s*[*×]\s*(\d+)$/.exec(smiles))) { n = parseInt(m[2], 10); smiles = m[1]; }
    if (smiles) rows.push(newRow(smiles, n, mult));
  }
  return rows;
}

// The species' preview, by what was typed (one request for every row not read yet).
function usePreview(rows) {
  const [info, setInfo] = useState({});
  const wanted = [...new Set(rows.map((r) => r.smiles.trim()).filter(Boolean))].filter((s) => !(s in info));
  const key = wanted.join('\n');
  useEffect(() => {
    if (!wanted.length) return undefined;
    const timer = setTimeout(() => {
      api.post('/api/species/preview', { smiles: wanted })
        .then((list) => setInfo((old) => ({ ...old, ...Object.fromEntries(list.map((x, i) => [wanted[i], x])) })))
        .catch(() => {});
    }, 250);
    return () => clearTimeout(timer);
  }, [key]);
  return info;
}

function SpeciesRow({ row, info, mult, onChange, onRemove, autofocus }) {
  const ok = info && !info.error;
  const badSpin = ok && (info.electrons + mult - 1) % 2 !== 0;
  const name = ok ? (info.name || info.canonical) : '';
  return html`<div class="nrs-row">
    <div class="nrs-who">
      <span class="nrs-pic">${ok && html`<img src=${depictUrl(info.smiles, 96, 64)} alt="" />`}</span>
      <span class="nrs-name">${name || html`<span class="muted">${row.smiles.trim() ? (info ? '' : '…') : 'new species'}</span>`}</span>
    </div>
    <label class="nrs-smiles"><span class="sr-only">SMILES or name</span>
      <input class=${`mono ${info?.error ? 'bad' : ''}`} spellcheck="false" autocomplete="off" placeholder="SMILES or name"
        value=${row.smiles} autofocus=${autofocus} onInput=${(e) => onChange({ smiles: e.target.value, mult: null })} />
      ${info?.error && html`<span class="nrs-err small">${info.error}</span>`}
    </label>
    <div class="nrs-count" role="group" aria-label=${`How many ${name || 'of this species'}`}>
      <button type="button" aria-label="One fewer" onClick=${() => onChange({ n: Math.max(0, row.n - 1) })}>−</button>
      <input type="number" min="0" step="1" aria-label="How many" value=${row.n}
        onInput=${(e) => onChange({ n: Math.max(0, parseInt(e.target.value, 10) || 0) })} />
      <button type="button" aria-label="One more" onClick=${() => onChange({ n: row.n + 1 })}>+</button>
    </div>
    <span class="nrs-charge mono" title="From the SMILES: write [OH-], [NH4+], ...">${ok ? (info.charge < 0 ? `−${-info.charge}` : info.charge > 0 ? `+${info.charge}` : '0') : ''}</span>
    <label class="nrs-spin"><span class="sr-only">Spin</span>
      <select value=${mult} class=${badSpin ? 'bad' : ''} title=${info?.spin_note || (badSpin ? `${info.electrons} electrons cannot be a ${spinName(mult)}` : 'Spin multiplicity of each of these molecules')}
        onChange=${(e) => onChange({ mult: +e.target.value })}>
        ${SPINS.map(([m, w]) => html`<option value=${m}>${m} · ${w}</option>`)}
        ${mult > SPINS.length && html`<option value=${mult}>${mult}</option>`}
      </select>
    </label>
    <button type="button" class="nrs-x btn-icon" aria-label=${`Remove ${name || 'this species'}`} title="Remove" onClick=${onRemove}>✕</button>
  </div>`;
}

export function ReactorSetup() {
  const mix = useStore((s) => s.view.mix || '');
  const demo = useStore((s) => s.demo);
  const [rows, setRows] = useState(() => { const r = parseMix(mix); return r.length ? r : [newRow()]; });
  const [paste, setPaste] = useState('');
  const [temps, setTemps] = useState('2000');          // one temperature, or several: one run each
  const [repeats, setRepeats] = useState(1);            // independent runs per temperature (another packing each)
  const [timePs, setTimePs] = useState(5);
  const [compress, setCompress] = useState(0.6);
  const [connect, setConnect] = useState(false);
  const [collide, setCollide] = useState(false);   // the Reactor's pulls as well as the piston (mepd reactor-run)
  const [profile, setProfile] = useState(() => defaultProfile());   // refines species, reactions, TSs
  // What drives the unattended run's MD. GFN2-xTB by default: measured 2026-11-28 on 56 atoms (one thread), ~1 min per
  // simulated ps against ~2 for the profile's g-xTB and ~4 for xtb's own g-xTB MD; the reactions it finds are refined
  // at the compute profile's level whatever drives the MD.
  const [mdMethod, setMdMethod] = useState('gfn2');
  const [steerMethod, setSteerMethod] = useState('auto');            // what Steer by hand runs
  const levels = useStore((s) => s.levels);
  const [spinMode, setSpinMode] = useState('species');
  const [otherMult, setOtherMult] = useState(1);
  const [busy, setBusy] = useState(null);
  const info = usePreview(rows);
  const [methods, setMethods] = useState(null);
  useEffect(() => { api.get('/api/sandbox-methods').then(setMethods).catch(() => {}); }, []);
  // A new address (the landing's Nanoreactor button again) fills the page anew.
  useEffect(() => { const r = parseMix(mix); if (r.length) setRows(r); }, [mix]);

  // '1800, 2200' or '1800 K; 2200K': numbers (a K after one is fine); anything else is flagged, never dropped.
  const tempTokens = temps.replace(/(\d)\s*k\b/gi, '$1').split(/[\s,;]+/).filter(Boolean);
  const badTemps = tempTokens.filter((x) => !(Number(x) > 0));
  const temperatures = badTemps.length ? [] : [...new Set(tempTokens.map(Number))];
  const nRuns = temperatures.length * repeats;
  const infoOf = (r) => info[r.smiles.trim()];
  const multOf = (r) => r.mult ?? infoOf(r)?.multiplicity ?? 1;
  const live = rows.filter((r) => r.n > 0 && r.smiles.trim());
  const ready = live.length > 0 && live.every((r) => infoOf(r) && !infoOf(r).error);
  const sum = (f) => live.reduce((t, r) => t + (infoOf(r) && !infoOf(r).error ? f(infoOf(r), r) * r.n : 0), 0);
  const atoms = sum((x) => x.atoms);
  const charge = sum((x) => x.charge);
  const electrons = sum((x) => x.electrons);
  const fromSpecies = 1 + sum((x, r) => Math.max(0, multOf(r) - 1));
  const lowest = electrons % 2 ? 2 : 1;
  const boxMult = spinMode === 'species' ? fromSpecies : spinMode === 'lowest' ? lowest : Math.max(1, otherMult | 0);
  const spinOk = !ready || (electrons + boxMult - 1) % 2 === 0;
  const rowSpinBad = live.some((r) => infoOf(r) && !infoOf(r).error && (infoOf(r).electrons + multOf(r) - 1) % 2 !== 0);

  const change = (key, patch) => setRows(rows.map((r) => (r.key === key ? { ...r, ...patch } : r)));
  const remove = (key) => { const left = rows.filter((r) => r.key !== key); setRows(left.length ? left : [newRow()]); };
  const addPasted = () => {
    const more = parseMix(paste);
    if (!more.length) return;
    setRows([...rows.filter((r) => r.smiles.trim()), ...more]);
    setPaste('');
  };

  // The rows as session structures (one per SMILES and spin; repeated rows add up), in row order.
  const addSpecies = async () => {
    const ids = [], counts = [];
    for (const r of live) {
      const x = infoOf(r);
      const [rec] = await api.post('/api/structures', { text: x.smiles, multiplicity: multOf(r), optimize: false });
      const k = ids.indexOf(rec.id);
      if (k >= 0) counts[k] += r.n; else { ids.push(rec.id); counts.push(r.n); }
    }
    return { ids, counts };
  };
  const title = live.map((r) => `${r.n > 1 ? `${r.n} ` : ''}${infoOf(r)?.name || infoOf(r)?.canonical || r.smiles.trim()}`).join(' + ');
  const go = (which, fn) => async () => { setBusy(which); await attempt(fn); setBusy(null); };
  // One run per temperature and repeat, each packed anew (its own seed, kept with the run); several open
  // the one list of what they find (Reactions › Nanoreactor runs).
  const launch = go('launch', async () => {
    const { ids, counts } = await addSpecies();
    const seed = Math.floor(Math.random() * 100000);
    const plan = temperatures.flatMap((T) => Array.from({ length: repeats }, (_, k) => ({ T, k })));
    const made = [];
    for (const { T, k } of plan) {
      const params = { copies: counts.join(', '), multiplicity: boxMult, temperature: T, time_ps: timePs, compress,
        connect: plan.length === 1 && connect, refine_live: true, seed: seed + made.length,
        md_method: collide && !['gfn2', 'gxtb', 'uma'].includes(mdMethod) ? 'gfn2' : mdMethod,
        forces: collide ? ['piston', 'collide'] : ['piston'] };
      const label = `Nanoreactor ${T} K${repeats > 1 ? `, run ${k + 1}` : ''}: ${title}`;
      try {
        made.push(...await api.post('/api/jobs', { op: 'nanoreactor', structures: ids, params, label, profile }));
      } catch (err) {
        if (!made.length) throw err;
        toast(`Launched ${made.length} of ${plan.length} runs: ${err.message || err}`, 'error', 12000);
        break;
      }
    }
    if (plan.length === 1) openJob(made[0].id); else openReactorList();
  });
  const steer = go('steer', async () => {
    const { ids, counts } = await addSpecies();
    const box = await api.post('/api/sandbox', { counts: Object.fromEntries(ids.map((id, k) => [id, counts[k]])),
      temperature: temperatures[0], multiplicity: boxMult, method: steerMethod });
    set({ view: { tab: 'sandbox', jobId: null, sandbox: box } });
  });

  const maxPs = demo?.limit_compute ? 10 : null;
  const atOnce = demo?.limit_compute ? demo.max_active_jobs : null;
  const steerNote = steerSpeed(atoms, methods, steerMethod);
  return html`<div class="nrs">
    <section class="nrs-recipe" aria-labelledby="nrs-in">
      <h2 id="nrs-in">What goes in</h2>
      <div class="nrs-head small" aria-hidden="true"><span>Species</span><span>SMILES</span><span>How many</span><span>Charge</span><span>Spin</span><span></span></div>
      ${rows.map((r, i) => html`<${SpeciesRow} key=${r.key} row=${r} info=${infoOf(r)} mult=${multOf(r)}
        autofocus=${i === rows.length - 1 && !r.smiles && rows.length > 1}
        onChange=${(p) => change(r.key, p)} onRemove=${() => remove(r.key)} />`)}
      <div class="nrs-more">
        <button type="button" class="btn" onClick=${() => setRows([...rows, newRow()])}>+ Species</button>
        <form class="nrs-paste" onSubmit=${(e) => { e.preventDefault(); addPasted(); }}>
          <input class="mono" spellcheck="false" aria-label="Paste a list" placeholder="or paste a list: 2 CC=O, 4 O, [OH-], O=O triplet"
            value=${paste} onInput=${(e) => setPaste(e.target.value)} />
          ${paste.trim() && html`<button class="btn" type="submit">Add</button>`}
        </form>
      </div>
      <div class="nrs-totals">
        <div><span class="nrs-label">Atoms</span><b>${atoms}</b></div>
        <div><span class="nrs-label">Charge</span><b>${charge < 0 ? `−${-charge}` : charge}</b></div>
        <div class="nrs-boxspin"><span class="nrs-label">Spin of the box</span>
          <span class="nrs-boxspin-row">
            <select value=${spinMode} onChange=${(e) => { setSpinMode(e.target.value); if (e.target.value === 'other') setOtherMult(boxMult); }}
              aria-label="Spin of the box">
              <option value="species">From the species: ${spinName(fromSpecies)} (${fromSpecies})</option>
              <option value="lowest">Lowest: ${spinName(lowest)} (${lowest})</option>
              <option value="other">Other…</option>
            </select>
            ${spinMode === 'other' && html`<input type="number" class="tiny" min="1" step="1" aria-label="Multiplicity" value=${otherMult}
              onInput=${(e) => setOtherMult(parseInt(e.target.value, 10) || 1)} />`}
          </span>
        </div>
      </div>
      ${(!spinOk || rowSpinBad) && html`<p class="nrs-warn small">${!spinOk
        ? `${electrons} electrons cannot be a ${spinName(boxMult)}: an ${electrons % 2 ? 'odd' : 'even'} count takes an ${electrons % 2 ? 'even' : 'odd'} multiplicity.`
        : 'A species has a spin its electrons cannot have (marked).'}</p>`}
    </section>
    <aside class="nrs-run" aria-label="Conditions and run">
      <div class="nrs-level">
        <${ProfilePicker} value=${profile} onChange=${setProfile} />
        <label class="field"><span class="field-label">MD driven by</span>
          <select value=${collide && !['gfn2', 'gxtb', 'uma'].includes(mdMethod) ? 'gfn2' : mdMethod} onChange=${(e) => setMdMethod(e.target.value)}
            title="What computes the forces of Launch nanoreactor's MD; its reactions, species and TSs are refined at the compute profile's level either way. On 56 atoms, one thread: GFN2-xTB ~1 min per simulated ps, the profile's g-xTB ~2, xtb's g-xTB ~4.">
            ${!collide && html`<option value="level">the profile's level (${levels[profile ?? '']?.label || 'its engine'})</option>`}
            <option value="gfn2">GFN2-xTB (fastest)</option>
            <option value="gxtb">g-xTB (slower)</option>
            <option value="uma">UMA (GPU${methods?.gpu ? '' : ': none here'})</option>
            ${!collide && html`<option value="gfn1">GFN1-xTB</option>`}
            ${!collide && html`<option value="auto">Automatic (GFN2-xTB if xtb is installed)</option>`}
          </select></label>
        <label class="field"><span class="field-label">Steer by hand runs</span>
          <select value=${steerMethod} onChange=${(e) => setSteerMethod(e.target.value)}
            title="The live MD you drag and squeeze. Automatic: UMA on the GPU from ${methods?.uma_from_atoms ?? 100} atoms when there is one, else GFN2-xTB.">
            <option value="auto">Automatic (${methods?.gpu && atoms >= (methods?.uma_from_atoms ?? 100) ? 'UMA here' : 'GFN2-xTB here'})</option>
            <option value="gfn2">GFN2-xTB</option>
            <option value="uma">UMA (needs a GPU)</option>
          </select></label>
      </div>
      <div class="nrs-fields">
        <label class="nrs-temps" title="Several, separated by commas, launch one run at each">Temperature<span><input class=${temperatures.length ? '' : 'bad'} inputmode="decimal" value=${temps}
          onInput=${(e) => setTemps(e.target.value)} /> K</span>
          ${badTemps.length > 0 && html`<span class="nrs-err small">Not a temperature: ${badTemps.join(', ')}</span>`}</label>
        <label>Simulated time<span><input type="number" min="0.1" step="1" value=${timePs} onInput=${(e) => setTimePs(+e.target.value || 0)} /> ps</span></label>
        <label title="The wall squeezes in to this fraction of its radius once a picosecond: smaller pushes the molecules harder together">Piston squeeze<span><input type="number" min="0.1" max="1" step="0.05" value=${compress} onInput=${(e) => setCompress(+e.target.value || 0)} /></span></label>
        <label title="Independent runs at each temperature, each with the molecules packed anew">Repeats<span><input type="number" min="1" step="1" value=${repeats} onInput=${(e) => setRepeats(Math.max(1, parseInt(e.target.value, 10) || 1))} /></span></label>
      </div>
      ${nRuns > 1 && html`<p class="small nrs-plan">${nRuns} runs: ${temperatures.map((T) => `${T} K`).join(', ')}${repeats > 1 ? `, ${repeats} of each` : ''}. Their reactions come together in one list.</p>`}
      <div class="nrs-forces">
        <span class="small muted">Forces</span>
        <label class="nrs-toggle" title="The wall squeezes in and out once a picosecond (always on)"><input type="checkbox" class="switch" checked disabled /> <span>Piston</span></label>
        <label class="nrs-toggle" title="Every so often two molecules (or two atoms of one) are pulled together, as network growth and Steer by hand do it"><input type="checkbox" class="switch" checked=${collide} onChange=${(e) => setCollide(e.target.checked)} /> <span>Collide</span></label>
      </div>
      ${(collide || mdMethod === 'uma') && html`<p class="small muted nrs-note">${collide ? 'With Collide' : 'On UMA'} the run's reactions are found and refined once the MD ends (not as it goes), and Run longer cannot continue it.${mdMethod === 'uma' ? ' UMA runs on the GPU: below ~60 atoms, or while the GPU is busy with another job, GFN2-xTB is faster and is used instead (the log says so).' : ''}</p>`}
      ${nRuns === 1 ? html`<label class="nrs-toggle"><input type="checkbox" class="switch" checked=${connect} onChange=${(e) => setConnect(e.target.checked)} />
        <span>Find each reaction's TS<span class="small muted"> as it is found</span></span></label>`
        : html`<p class="small muted nrs-note">With several runs, find TSs from their list: one search per reaction, however many runs found it.</p>`}
      ${atOnce && nRuns > atOnce && html`<p class="nrs-warn small">The demo runs ${atOnce} calculations at a time: the first ${atOnce} start now.</p>`}
      ${maxPs && timePs > maxPs && html`<p class="nrs-warn small">The demo runs at most ${maxPs} ps.</p>`}
      <div class="nrs-go">
        <button class="btn primary nrs-launch" disabled=${!ready || !spinOk || rowSpinBad || busy || !nRuns}
          title="Runs on its own: hot MD inside a squeezing wall, then each reaction is cut out and refined at your level of theory. Reactions join Explore as they are found."
          onClick=${launch}>${busy === 'launch' ? 'Launching…' : nRuns > 1 ? `Launch ${nRuns} runs ›` : 'Launch nanoreactor ›'}</button>
        <button class="btn" disabled=${!ready || !spinOk || rowSpinBad || busy || atoms > STEER_MAX_ATOMS || !nRuns}
          title=${atoms > STEER_MAX_ATOMS ? `Steering by hand takes ${STEER_MAX_ATOMS} atoms at most` : 'A live GFN2-xTB MD you drag and squeeze; Stop & analyze pulls out its reactions'}
          onClick=${steer}>${busy === 'steer' ? 'Starting…' : 'Steer by hand'}</button>
      </div>
      ${steerNote && html`<p class=${`small ${steerNote.slow ? 'nrs-warn' : 'muted nrs-note'}`}>${steerNote.text}</p>`}
      <p class="small muted nrs-note">${atoms > STEER_MAX_ATOMS ? `Steer by hand takes ${STEER_MAX_ATOMS} atoms at most. ` : ''}The MD's speed depends on what drives it: on 56 atoms, GFN2-xTB took about 1 min per simulated ps, g-xTB 2 to 4.${' '}
        <a href="/manual/nanoreactor" target="_blank" rel="noopener">How it works ›</a></p>
    </aside>
  </div>`;
}
