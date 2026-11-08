// Renders any operation's parameter form from its pydantic JSON schema.
// Field extras from mepd.web.operations: group, advanced, cli, cli_kind.
import { html, useEffect, useState } from '../lib.js';
import { prefs, useStore } from '../store.js';
import { Why } from './Why.js';

// A number field that lets you type: it keeps its own text while you edit
// ("0.", "-", "1e-" are fine half-way) and reports a number only when the text
// parses as one. A number input re-rendered from the parsed value on every
// keystroke loses the "." (Chrome reports "0." as empty, which reset the field).
export function NumberInput({ value, onChange, integer = false, min, max, placeholder = '', id, className = '', title }) {
  const shown = (v) => (v == null || Number.isNaN(v) ? '' : String(v));
  const [text, setText] = useState(shown(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => { if (!editing) setText(shown(value)); }, [value, editing]);
  const pattern = integer ? /^-?\d+$/ : /^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
  const bad = text.trim() !== '' && (!pattern.test(text.trim())
    || (min != null && +text < min) || (max != null && +text > max));
  return html`<input id=${id} type="text" inputmode=${integer ? 'numeric' : 'decimal'} class=${`${className} ${bad ? 'invalid' : ''}`}
    value=${text} placeholder=${placeholder} title=${bad ? `Not a valid number${min != null ? ` (at least ${min})` : ''}${max != null ? ` (at most ${max})` : ''}` : title}
    onFocus=${() => setEditing(true)}
    onBlur=${() => { setEditing(false); setText(shown(value)); }}
    onInput=${(e) => {
      const raw = e.target.value;
      setText(raw);
      const t = raw.trim();
      if (t === '') { onChange(null); return; }
      if (pattern.test(t)) {
        const v = integer ? parseInt(t, 10) : parseFloat(t);
        if ((min == null || v >= min) && (max == null || v <= max)) onChange(v);
      }
    }} />`;
}

function fieldType(prop) {
  const variants = prop.anyOf ? prop.anyOf.filter((v) => v.type !== 'null') : [prop];
  const base = variants[0] || {};
  const nullable = !!prop.anyOf?.some((v) => v.type === 'null');
  if (base.enum || prop.enum) return { kind: 'enum', options: base.enum || prop.enum, nullable };
  if (base.type === 'array' && base.items?.enum) return { kind: 'multi', options: base.items.enum };
  return { kind: base.type || 'string', nullable, min: base.minimum ?? base.exclusiveMinimum, max: base.maximum };
}

function Field({ name, prop, value, onChange, op }) {
  const info = html`<${Why} ids=${[`param.${op}.${name}`, `param.${name}`]} label=${prop.title || name}
    fallback=${prop.description ? html` <span class="field-info" aria-hidden="true">ⓘ</span>` : null} />`;
  const t = fieldType(prop);
  const hint = [prop.description, prop.cli ? `CLI: ${prop.cli}` : ''].filter(Boolean).join('\n');
  const id = `f-${name}`;
  const profiles = useStore((s) => s.profiles) || [];
  let control;
  if (prop.choices_from === 'profiles') {   // one of the workspace's saved profiles, or none
    control = html`<select id=${id} value=${value ?? ''} onChange=${(e) => onChange(e.target.value)}>
      <option value="">none</option>
      ${profiles.map((o) => html`<option value=${o}>${o}</option>`)}
    </select>`;
    return html`
      <div class="field" title=${hint}>
        <label class="field-label" for=${id}>${prop.title || name}${info}</label>
        ${control}
      </div>`;
  }
  if (t.kind === 'boolean') {
    return html`
      <label class="field field-bool" title=${hint}>
        <input type="checkbox" class="switch" checked=${!!value} onChange=${(e) => onChange(e.target.checked)} />
        <span class="field-label">${prop.title || name}${info}</span>
      </label>`;
  }
  // Field extras may name what each value is shown as (`labels`).
  const shown = (o) => prop.labels?.[o] ?? o;
  if (t.kind === 'multi') {
    const on = new Set(value || []);
    control = html`<div class="chips" role="group">
      ${t.options.map((o) => html`<button type="button" aria-pressed=${on.has(o)} class=${`chip ${on.has(o) ? 'on' : ''}`}
        title=${shown(o)} onClick=${() => {
          const next = on.has(o) ? (value || []).filter((v) => v !== o) : [...(value || []), o];
          if (next.length >= (prop.minItems || 0)) onChange(next);   // e.g. the Nanoreactor keeps one force on
        }}>
        ${String(shown(o)).replace(/\s*\(.*\)$/, '')}</button>`)}
    </div>`;
  } else if (t.kind === 'enum') {
    const options = t.options;
    control = options.length <= 3 && options.every((o) => String(shown(o)).length <= 24)
      ? html`<div class="segmented" role="radiogroup">
          ${options.map((o) => html`<button type="button" role="radio" aria-checked=${value === o}
            class=${value === o ? 'on' : ''} onClick=${() => onChange(o)}>${shown(o)}</button>`)}
        </div>`
      : html`<select id=${id} value=${value ?? ''} onChange=${(e) => onChange(e.target.value)}>
          ${options.map((o) => html`<option value=${o}>${shown(o)}</option>`)}
        </select>`;
  } else if (t.kind === 'integer' || t.kind === 'number') {
    control = html`<${NumberInput} id=${id} integer=${t.kind === 'integer'} min=${t.min} max=${t.max} value=${value}
      placeholder=${t.nullable ? 'default' : String(prop.default ?? '')}
      onChange=${(v) => onChange(v == null ? (t.nullable ? null : prop.default) : v)} />`;
  } else {
    control = html`<input id=${id} type="text" value=${value ?? ''} placeholder=${t.nullable ? 'default' : ''}
      onInput=${(e) => onChange(e.target.value === '' && t.nullable ? null : e.target.value)} />`;
  }
  return html`
    <div class="field" title=${hint}>
      <label class="field-label" for=${id}>${prop.title || name}${info}</label>
      ${control}
    </div>`;
}

// Pull remembered values back inside the schema's limits (e.g. demo caps
// that were added after the value was saved in this browser).
export function clampToSchema(schema, values) {
  const out = { ...values };
  for (const [k, p] of Object.entries(schema?.properties || {})) {
    const max = p.maximum ?? p.anyOf?.find((v) => v.maximum != null)?.maximum;
    if (max != null && typeof out[k] === 'number' && out[k] > max) out[k] = max;
  }
  return out;
}

// A calculation form remembers only the settings you changed from its
// defaults, so a default that changes later still reaches every setting you
// never touched. (Version 2 of the stored form: version 1 kept every value,
// defaults included, so one changed default stayed stale forever; those are
// not read, which starts every form once from the current defaults.)
const REMEMBERED = (key) => `params2:${key}`;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export function rememberedValues(op, extra = {}) {
  const defaults = defaultsFor(op.schema);
  return clampToSchema(op.schema, movedCards(op.schema, { ...defaults, ...prefs.get(REMEMBERED(op.key), {}), ...extra }));
}

// A card that became another card's setting (Reactor MD -> the Nanoreactor with Piston and Collide) opens as that
// card: remembered forms and Run again of older jobs land on the choice the form offers now.
function movedCards(schema, values) {
  let out = values;
  for (const [k, p] of Object.entries(schema?.properties || {})) {
    const moved = p.cards?.[out[k]]?.moved_to;
    if (moved) out = { ...out, ...moved };
  }
  return out;
}

export function rememberValues(op, values) {
  const defaults = defaultsFor(op.schema);
  const changed = Object.fromEntries(Object.entries(values).filter(([k, v]) => k in defaults && !same(v, defaults[k])));
  prefs.set(REMEMBERED(op.key), changed);
}

export function defaultsFor(schema) {
  const out = {};
  for (const [k, p] of Object.entries(schema?.properties || {})) out[k] = p.default ?? null;
  return out;
}

// Mirrors operations.requirement_met: "other" or "other=a|b" (a card's `needs` may join several with ";").
// A list value reads as its sorted items joined by "," ("collide,piston").
function requirementMet(values, requires) {
  if (!requires) return true;
  return requires.split(';').every((one) => {
    const [name, allowed] = one.split('=');
    const v = Array.isArray(values[name]) ? [...values[name]].sort().join(',') : values[name];
    return allowed === undefined ? (Array.isArray(values[name]) ? values[name].length > 0 : !!v) : allowed.split('|').includes(String(v));
  });
}

// What each stage of a staged operation decides (network expansion).
const STAGE_TEXT = {
  Grow: 'how new species are proposed',
  Connect: 'how each proposed reaction gets its transition state',
  Steer: 'which species grow next',
  Run: 'how the work is scheduled and when it stops',
};

// A card shows its blurb's first sentence; the whole is its tooltip.
const firstSentence = (t) => (t || '').split(/(?<=\.)\s/)[0];

// A choice shown as cards: one per value, greyed out (with why) when another stage rules it out.
function Cards({ name, prop, value, values, onChange, context, op }) {
  // hidden: an older choice kept for older jobs, shown only while it is the one chosen
  const cards = Object.entries(prop.cards).filter(([v, c]) => (!c.qmmm_only || context?.qmmm) && (!c.hidden || v === value));
  return html`<div class="stage-cards" role="radiogroup" aria-label=${prop.title || name}>
    ${cards.map(([v, c]) => {
      const blocked = c.unavailable || (!requirementMet(values, c.needs) ? c.needs_text || `needs ${c.needs}` : '');
      return html`<button type="button" role="radio" aria-checked=${value === v} disabled=${!!blocked}
          class=${`stage-card ${value === v ? 'on' : ''} ${blocked ? 'blocked' : ''}`}
          title=${blocked ? `Not with the other choices: ${blocked}` : c.blurb} onClick=${() => onChange(v)}>
        <b>${c.title}</b><span>${firstSentence(c.blurb)}</span>${blocked && html`<i>${blocked}</i>`}
      </button>`;
    })}
  </div>`;
}

// Choices ruled out by a change elsewhere move to the first card still allowed; the form says so.
function settleCards(schema, values, context) {
  let out = values;
  const notes = [];
  for (let pass = 0; pass < 3; pass += 1) {
    let moved = false;
    for (const [k, p] of Object.entries(schema.properties)) {
      if (!p.cards || !requirementMet(out, p.requires)) continue;
      const ok = (v) => { const c = p.cards[v]; return c && !c.unavailable && (!c.qmmm_only || context?.qmmm) && requirementMet(out, c.needs); };
      if (ok(out[k])) continue;
      const next = Object.keys(p.cards).find(ok);
      if (next && next !== out[k]) {
        notes.push({ stage: p.stage, text: `${p.title}: ${p.cards[next].title} (${p.cards[out[k]]?.title || out[k]} ${p.cards[out[k]]?.needs_text || 'is ruled out'})` });
        out = { ...out, [k]: next };
        moved = true;
      }
    }
    if (!moved) break;
  }
  return { values: out, notes };
}

function StagedForm({ schema, values, onChange, op, context }) {
  const [notes, setNotes] = useState([]);
  const all = Object.entries(schema.properties).filter(([, p]) => p.group !== 'Hidden' && p.stage);
  const stages = [...new Set(all.map(([, p]) => p.stage))];
  const change = (k) => (v) => {
    const { values: next, notes: n } = settleCards(schema, { ...values, [k]: v }, context);
    setNotes(n);
    onChange(next);
  };
  return html`<div class="param-form staged">
    ${stages.map((st, i) => {
      const fields = all.filter(([, p]) => p.stage === st && requirementMet(values, p.requires) && (!p.qmmm_only || context?.qmmm));
      const choice = fields.find(([, p]) => p.cards);
      const rest = fields.filter(([, p]) => !p.cards);
      const basic = rest.filter(([, p]) => !p.advanced), more = rest.filter(([, p]) => p.advanced);
      const changedMore = more.filter(([k, p]) => values[k] !== (p.default ?? null)).length;
      return html`<section class="stage">
        <header><span class="stage-n">${i + 1}</span><b>${st}</b><span class="muted small">${STAGE_TEXT[st] || ''}</span>
          ${op === 'graph-enumeration' && html`<a class="stage-help small" href=${`/manual/network-expansion#${st.toLowerCase()}`} target="_blank" rel="noopener" title="What each choice does">?</a>`}</header>
        ${choice && html`<${Cards} name=${choice[0]} prop=${choice[1]} value=${values[choice[0]]} values=${values}
          onChange=${change(choice[0])} context=${context} op=${op} />`}
        ${notes.filter((n) => n.stage === st).map((n) => html`<p class="stage-note small">↻ ${n.text}</p>`)}
        ${basic.length > 0 && html`<div class="stage-fields">${basic.map(([k, p]) => html`<${Field} key=${k} op=${op} name=${k} prop=${p} value=${values[k]} onChange=${change(k)} />`)}</div>`}
        ${more.length > 0 && html`<details class="advanced"><summary>More${changedMore ? html` <span class="badge">${changedMore} changed</span>` : ''}</summary>
          ${more.map(([k, p]) => html`<${Field} key=${k} op=${op} name=${k} prop=${p} value=${values[k]} onChange=${change(k)} />`)}</details>`}
      </section>`;
    })}
    <button type="button" class="btn-link small" onClick=${() => { setNotes([]); onChange(defaultsFor(schema)); }}>Reset all to defaults</button>
  </div>`;
}

export function isStaged(schema) {
  return Object.values(schema?.properties || {}).some((p) => p.stage);
}

export function ParamForm({ schema, values, onChange, op = '', context = null }) {
  if (!schema?.properties || !Object.keys(schema.properties).length) {
    return html`<p class="muted small">No parameters.</p>`;
  }
  if (isStaged(schema)) return html`<${StagedForm} schema=${schema} values=${values} onChange=${onChange} op=${op} context=${context} />`;
  // group "Hidden": set by the page itself (e.g. Sandbox › Stop & analyze), never a form field.
  const entries = Object.entries(schema.properties).filter(([, p]) => p.group !== 'Hidden' && requirementMet(values, p.requires));
  const basic = entries.filter(([, p]) => !p.advanced);
  const advanced = entries.filter(([, p]) => p.advanced);
  const groups = {};
  for (const [k, p] of advanced) (groups[p.group || 'Advanced'] ||= []).push([k, p]);
  const changed = (k) => values[k] !== (schema.properties[k].default ?? null);
  const nChanged = advanced.filter(([k]) => changed(k)).length;
  const set = (k) => (v) => onChange({ ...values, [k]: v });
  return html`
    <div class="param-form">
      ${basic.map(([k, p]) => html`<${Field} key=${k} op=${op} name=${k} prop=${p} value=${values[k]} onChange=${set(k)} />`)}
      ${advanced.length > 0 && html`
        <details class="advanced">
          <summary>Advanced${nChanged ? html` <span class="badge">${nChanged} changed</span>` : ''}</summary>
          ${Object.entries(groups).map(([g, fields]) => html`
            <fieldset>
              <legend>${g}</legend>
              ${fields.map(([k, p]) => html`<${Field} key=${k} op=${op} name=${k} prop=${p} value=${values[k]} onChange=${set(k)} />`)}
            </fieldset>`)}
          <button type="button" class="btn-link small" onClick=${() => onChange(defaultsFor(schema))}>Reset all to defaults</button>
        </details>`}
    </div>`;
}
