"""The manual (mepd/manual, /manual): every page renders, and every link the
UI makes into it lands on a section that exists."""
import re
from pathlib import Path

import mepd.web
from mepd.web import manual

JS = Path(mepd.web.__file__).resolve().parent / "static" / "js"   # (the installed package's, as in the image)


def test_every_page_renders_with_a_way_to_suggest_changes():
    assert "orbitals" in manual.pages()
    for page in [None, *manual.pages()]:
        out = manual.render(page)
        assert "<h1" in out and "Suggest a change" in out


def test_ui_links_point_at_existing_sections():
    links = set()
    assert any(JS.rglob("*.js")), JS
    for js in JS.rglob("*.js"):
        links |= set(re.findall(r"/manual/([a-z0-9-]+)#([a-z0-9-]+)", js.read_text()))
        # (links built from a choice: /manual/orbitals#${x ? 'a' : 'b'})
        for page, rest in re.findall(r"/manual/([a-z0-9-]+)#\$\{([^}]*)\}", js.read_text()):
            links |= {(page, a) for a in re.findall(r"'([a-z0-9-]+)'", rest)}
    assert ("orbitals", "fragment-orbitals") in links and ("orbitals", "whole-orbitals") in links
    for page, anchor in links:
        assert f'id="{anchor}"' in manual.render(page), (page, anchor)


def test_unknown_pages_are_refused():
    import pytest

    for bad in ["../app", "nope", "Index"]:
        with pytest.raises(manual.ManualError):
            manual.render(bad)


def test_the_route(tmp_path):
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        assert c.get("/manual").status_code == 200
        r = c.get("/manual/orbitals")
        assert r.status_code == 200 and "Fragment orbitals" in r.text
        assert c.get("/manual/missing").status_code == 404
