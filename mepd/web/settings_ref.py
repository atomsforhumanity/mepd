"""The manual's settings reference (/manual/settings), written from the code.

Every calculation the web app offers, with each setting its form shows (the
default the form starts from, what it does, when it applies) and each option
of the mepd command it runs that the form leaves at mepd's default. It is
built from the operations registry and the CLI, so it can't fall behind them.
Anchors: `#<operation key>` for a calculation, `#<key>--<field>` for one of its
settings, `#reactor` for the interactive Reactor.
"""
from __future__ import annotations

import inspect
import re
from functools import lru_cache
from typing import Any, Optional

# Options every command shares and the web sets for each job (structures, level of theory, output folder).
_SET_BY_THE_APP = {"--inputs", "-i", "--charge", "--multiplicity", "--output", "-o", "--help",
                   "--start", "--end", "--guess", "--log-level", "--verbose", "-v"}
_ORDER = ["Connect two structures", "Explore around a structure", "Across a set of structures", "Reaction discovery",
          "Past the transition state", "Reaction conditions", "Complexes", "QM/MM", "Design"]


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _value(v: Any, labels: Optional[dict] = None) -> str:
    if labels and str(v) in {str(k) for k in labels}:
        shown = {str(k): lab for k, lab in labels.items()}[str(v)]
        return f"{shown} (`{v}`)"
    if v is None or v == "":
        return "empty"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (list, tuple)):
        return ", ".join(f"`{x}`" for x in v) if v else "none"
    if isinstance(v, float):
        return f"`{v:g}`"
    return f"`{v}`"


def _requires(req: Optional[str]) -> str:
    if not req:
        return ""
    parts = []
    for one in req.split(";"):
        name, _, allowed = one.partition("=")
        parts.append(f"`{name}` is {' or '.join(f'`{a}`' for a in allowed.split('|'))}" if allowed else f"`{name}` is on")
    return " Only when " + " and ".join(parts) + "."


def command_of(op) -> tuple:
    """The mepd command an operation runs (`("discovery", "nanoreactor")`), from its declaration or its builder."""
    if op.cli_path:
        return tuple(op.cli_path)
    if not op.build:
        return ()
    free = dict(zip(op.build.__code__.co_freevars, [c.cell_contents for c in (op.build.__closure__ or ())]))
    src = inspect.getsource(op.build)
    m = re.search(r'(?:return|argv\s*=)\s*\[\s*"([a-z][a-z-]*)"(?:\s*,\s*(?:"([a-z][a-z-]*)"|(command)\b))?', src)
    if m:
        first, second, var = m.groups()
        return (first, second or free.get(var)) if (second or free.get(var)) else (first,)
    return ()


def _click_command(path: tuple):
    import click
    import typer

    from mepd.cli import app

    cmd = typer.main.get_command(app)
    for word in path:
        cmd = cmd.get_command(click.Context(cmd), word) if hasattr(cmd, "get_command") else None
        if cmd is None:
            return None
    return cmd


def _set_by_the_form(op) -> set:
    """The command-line flags this operation's job sets itself: its fields', its extra flags, and any its builder writes."""
    flags = set(op.cli_extra_flags) | _SET_BY_THE_APP
    for name, finfo in (op.params_model.model_fields.items() if op.params_model else []):
        cli = (finfo.json_schema_extra or {}).get("cli")
        # (a field with no flag of its own is written by the builder, usually under its own name)
        for flag in filter(None, [cli, "--" + name.replace("_", "-")]):
            flags |= {flag, "--no-" + flag.removeprefix("--")}
    for src in _builder_sources(op.build):
        flags |= set(re.findall(r'"(--[a-z0-9-]+)"', src))
    return flags


def _builder_sources(fn) -> list[str]:
    """The source of a builder and of every function of mepd.web it calls, however deep."""
    import types

    seen, out, todo = set(), [], [fn] if fn else []
    while todo:
        f = todo.pop()
        f = getattr(f, "__func__", f)
        if not isinstance(f, types.FunctionType) or f in seen or not (f.__module__ or "").startswith("mepd.web"):
            continue
        seen.add(f)
        try:
            out.append(inspect.getsource(f))
        except (OSError, TypeError):
            continue
        cells = [c.cell_contents for c in (f.__closure__ or ()) if _filled(c)]
        codes = [f.__code__]
        while codes:
            code = codes.pop()
            codes += [c for c in code.co_consts if isinstance(c, types.CodeType)]
            todo += [f.__globals__[n] for n in code.co_names if n in f.__globals__]
            todo += [getattr(f.__globals__.get(a), b, None) for a, b in zip(code.co_names, code.co_names[1:])]
        todo += [c for c in cells if callable(c)]
    return out


def _filled(cell) -> bool:
    try:
        cell.cell_contents
        return True
    except ValueError:
        return False


def mepd_defaults(op) -> list[dict]:
    """The options of the operation's mepd command that the form never sets: they run at mepd's defaults."""
    cmd = _click_command(command_of(op))
    if cmd is None:
        return []
    taken = _set_by_the_form(op)
    out = []
    for prm in cmd.params:
        opts = [*getattr(prm, "opts", []), *getattr(prm, "secondary_opts", [])]
        if getattr(prm, "param_type_name", "") != "option" or getattr(prm, "hidden", False) or taken & set(opts):
            continue
        name = max(prm.opts, key=len)
        default = prm.default
        if getattr(prm, "is_flag", False) and prm.secondary_opts:
            default = bool(default)
        out.append({"flag": name, "default": default, "help": (prm.help or "").strip()})
    return out


def _form_rows(op) -> list[str]:
    if not op.params_model:
        return []
    fixed = {}
    for m in op.methods:
        fixed.update({k: None for k in (m.get("fixed") or {})})
    rows = []
    for name, finfo in op.params_model.model_fields.items():
        extra = finfo.json_schema_extra or {}
        where = extra.get("stage") or extra.get("group") or ""
        if name in fixed or where == "Hidden":
            continue
        if extra.get("advanced") and where.lower() != "advanced":
            where = f"{where}, advanced" if where else "Advanced"
        flag = extra.get("cli")
        help_ = _cell(finfo.description or "") + _requires(extra.get("requires"))
        if flag:
            help_ += f" (`{flag}`)"
        rows.append(f"| <span id=\"{op.key}--{name}\"></span>{_cell(finfo.title or name)} | "
                    f"{_value(finfo.default, extra.get('labels'))} | {_cell(where)} | {help_.strip()} |")
    return rows


def _operation(op) -> list[str]:
    cmd = command_of(op)
    lines = [f"### {op.title} {{#{op.key}}}", "", _cell(op.summary), ""]
    if cmd and "`mepd " not in op.summary:
        lines += [f"Runs `mepd {' '.join(cmd)}`.", ""]
    methods = [m for m in op.methods if m.get("fixed")]
    if methods:
        lines += ["Methods on this card:", ""]
        lines += [f"- **{m['label']}**: " + ", ".join(f"`{k}` = {_value(v)}" for k, v in m["fixed"].items())
                  for m in methods]
        lines.append("")
    rows = _form_rows(op)
    if rows:
        lines += ["| Setting | Default | Where | What it does |", "|---|---|---|---|", *rows, ""]
    else:
        lines += ["No settings: it runs as it is.", ""]
    rest = mepd_defaults(op) if op.target != "job" else []
    if op.target == "job":
        lines += ["Everything else is as in the job it continues.", ""]
    if rest:
        lines += ['<details markdown="1"><summary>' "Not in the form: mepd's defaults "
                  f"({len(rest)})</summary>", "",
                  "| Option | Default | What it does |", "|---|---|---|"]
        lines += [f"| `{r['flag']}` | {_value(r['default'])} | {_cell(r['help'])} |" for r in rest]
        lines += ["", "</details>", ""]
    return lines


def _reactor() -> list[str]:
    """The interactive Reactor (mepd/web/sandbox.py), which is not a job: its settings live in its code."""
    from mepd.discovery import langevin, reactor_generator
    from mepd.web import app as _app, sandbox

    md = inspect.signature(langevin.LangevinMD.__init__).parameters
    start = getattr(_app, "SandboxIn", None)
    t0 = start.model_fields["temperature"].default if start else inspect.signature(sandbox.start).parameters[
        "temperature"].default
    p = sandbox.PISTON
    rows = [
        ("Starting temperature", f"{t0:g} K", "Set before starting (Reactor card) and with the slider while it runs (0-3000 K)."),
        ("Forces", f"`{md['method'].default}`", "GFN2-xTB through axtb, one thread."),
        ("Thermostat", f"Langevin, friction {md['friction'].default:g} /fs", "Holds the set temperature."),
        ("Time step", f"{md['dt'].default:g} fs", ""),
        ("Fermi smearing", f"{langevin.ETEMPS[0]:g} K", "Lets bonds break cleanly; a step whose SCF fails retries at "
         + " then ".join(f"{t:g} K" for t in langevin.ETEMPS[1:]) + "."),
        ("Wall", "soft sphere, 20 kcal/mol/Å per atom outside",
         "Radius: the packing's, plus 15%; for one structure as it is, its farthest atom plus 2 Å. Scroll to change it."),
        ("Piston (when on)", f"period {p['period_ps']:g} ps, squeeze to {p['compress']:g} of the radius, "
         f"wide {p['duty']:.0%} of each period", f"The wall closes over {sandbox.PISTON_RAMP_FS:g} fs."),
        ("Collide pull", f"k = {reactor_generator.OPTIONS['k']:g} kcal/mol/Å², capped at about 100 kcal/mol/Å",
         "Two molecules pushed together, or two atoms of one molecule pulled toward each other."),
        ("Trajectory kept", f"one frame every {sandbox.TRAJ_FS:g} fs", "What the reaction events are read from."),
        ("Reaction events", f"checked every {sandbox.EVENT_EVERY_S:g} s, over the last "
         f"{sandbox.EVENT_WINDOW * sandbox.TRAJ_FS / 1000:g} ps", ""),
        ("Idle stop", f"{sandbox.IDLE_STOP_S / 60:g} min", "Stops when nobody has looked at it for this long."),
    ]
    return ["### Interactive Reactor {#reactor}", "",
            "The Reactor you drive by hand (drag atoms, Collide, Piston, temperature). It is not a job and has no "
            "form; these are its settings. The [Nanoreactor](#nanoreactor) is the job version, and the MD of [Reaction network expansion](#graph-enumeration) grows networks.", "",
            "| Setting | Value | Notes |", "|---|---|---|",
            *[f"| {a} | {b} | {c} |" for a, b, c in rows], ""]


@lru_cache(maxsize=1)
def markdown() -> str:
    from mepd.web.operations import OPERATIONS

    ops = list(OPERATIONS.values())
    cats = sorted({op.category for op in ops}, key=lambda c: (_ORDER.index(c) if c in _ORDER else len(_ORDER), c))
    lines = ["# Settings reference", "",
             "Every calculation mepd's web app runs, with each setting its form shows and the default it starts "
             "from, and the options of the mepd command behind it that the form leaves at mepd's defaults. This page "
             "is written from the code, so it lists exactly what this version of mepd does.", "",
             "The level of theory (method, optimizers, path-search settings) is the compute profile, set under "
             "**Settings**; every calculation below runs at the profile chosen in its form.", "",
             "## Contents", ""]
    for cat in cats:
        lines.append(f"- **{cat}**: " + ", ".join(f"[{op.title}](#{op.key})" for op in ops if op.category == cat))
    lines.append(f"- **Molecular dynamics by hand**: [Interactive Reactor](#reactor)")
    lines.append("")
    for cat in cats:
        lines += [f"## {cat}", ""]
        for op in ops:
            if op.category == cat:
                lines += _operation(op)
    lines += ["## Molecular dynamics by hand", "", *_reactor()]
    lines += ["## Limits and possible improvements", "",
              "- A default here is the form's starting value. Some settings change with others (shown as \"Only "
              "when\"), and an empty value means the calculation picks one, as its description says.",
              "- What each setting does is written in one line; the feature pages of this manual explain the "
              "reasoning where one exists.", ""]
    return "\n".join(lines)
