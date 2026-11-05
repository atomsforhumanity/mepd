"""The mepd manual (`mepd/manual/*.md`), served by the web app at /manual.

How each feature works, the choices made and why, and its limits: the UI
links to the section that explains what it shows (e.g. /manual/orbitals#
fragment-orbitals). Pages ship inside the package, so the deployed image and
the demo serve the manual of the code they run. Each page links to
SUGGEST_URL to propose a change.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

PAGES = Path(__file__).resolve().parent.parent / "manual"
SUGGEST_URL = "https://github.com/atomsforhumanity/mepd/edit/main/mepd/manual/{page}.md"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")

_STYLE = """
body { background: var(--bg); color: var(--text); margin: 0; }
.manual { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; line-height: 1.55; font-size: 16px; }
.manual nav.top { display: flex; gap: 14px; align-items: baseline; font-size: 14px; margin-bottom: 18px; }
.manual nav.top a { color: var(--accent); text-decoration: none; }
.manual h1, .manual h2, .manual h3 { font-family: "Source Serif 4 Variable", Georgia, serif; color: var(--heading, var(--text)); line-height: 1.25; }
.manual h1 { font-size: 30px; margin: 8px 0 14px; }
.manual h2 { font-size: 22px; margin: 34px 0 10px; padding-top: 10px; border-top: 1px solid var(--border); }
.manual h3 { font-size: 17px; margin: 22px 0 6px; }
.manual a.headerlink { color: var(--muted); text-decoration: none; margin-left: 6px; opacity: 0; font-size: 0.8em; }
.manual h2:hover a.headerlink, .manual h3:hover a.headerlink { opacity: 1; }
.manual code { font-family: var(--font-mono); font-size: 0.88em; background: var(--accent-soft); padding: 1px 4px; border-radius: 3px; }
.manual :target { scroll-margin-top: 12px; }
.manual h2:target, .manual h3:target { background: var(--accent-soft); border-radius: 4px; }
.manual footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 14px; color: var(--muted); }
.manual footer a { color: var(--accent); }
.manual ul { padding-left: 22px; }
.manual li { margin: 3px 0; }
.manual figure { margin: 20px 0; }
.manual figure svg { display: block; width: 100%; height: auto; }
.manual figcaption { font-size: 14px; color: var(--muted); margin-top: 6px; line-height: 1.45; }
.manual table { border-collapse: collapse; margin: 12px 0; font-size: 15px; }
.manual th, .manual td { text-align: left; vertical-align: top; padding: 5px 10px; border-bottom: 1px solid var(--border); }
"""


class ManualError(LookupError):
    pass


def pages() -> list[str]:
    return sorted(p.stem for p in PAGES.glob("*.md") if p.stem != "index")


def _title(text: str, page: str) -> str:
    m = re.search(r"^# (.+)$", text, re.M)
    return m.group(1).strip() if m else page


@lru_cache(maxsize=64)
def _render(path: str, mtime: float) -> tuple[str, str]:
    import markdown

    text = Path(path).read_text()
    body = markdown.markdown(text, extensions=["toc", "tables", "fenced_code", "sane_lists"],
                             extension_configs={"toc": {"permalink": "#"}})
    return _title(text, Path(path).stem), body


def render(page: Optional[str] = None) -> str:
    """The HTML of a manual page (`None`: the index)."""
    page = page or "index"
    if not _NAME.fullmatch(page):
        raise ManualError(page)
    path = PAGES / f"{page}.md"
    if not path.is_file():
        raise ManualError(page)
    title, body = _render(str(path), path.stat().st_mtime)
    suggest = SUGGEST_URL.format(page=page)
    crumbs = '<a href="/manual">Manual</a>' if page != "index" else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · mepd manual</title>
<link rel="stylesheet" href="/static/style.css"><style>{_STYLE}</style></head>
<body><main class="manual">
<nav class="top"><a href="/">mepd</a>{crumbs}</nav>
{body}
<footer>Wrong, unclear, or could this work better? <a href="{suggest}" target="_blank" rel="noopener">Suggest a change</a>
to this page.</footer>
</main></body></html>"""
