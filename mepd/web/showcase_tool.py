"""Pack a finished workspace as a Home "Examples" study (mepd/web/showcase).

    python -m mepd.web.showcase_tool pack WORKSPACE SLUG --title ... --tag ... [--blurb ...]

The archive keeps what the results pages read and drops what is rebuilt on
demand or too big to ship: the orbital caches (rebuilt when a view needs
them) and reactor MD trajectories. Paths inside are made relative to the
workspace so a copy can live anywhere (`unpack` puts them back).
"""
from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

SHOWCASE = Path(__file__).parent / "showcase"
_SKIP_DIRS = {"orbitals", "reactor", "__pycache__"}
_ROOT_TOKEN = "@WORKSPACE@"


def _keep(rel: Path) -> bool:
    return not any(part in _SKIP_DIRS for part in rel.parts)


def _rewrite(obj, old: str, new: str):
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_rewrite(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v, old, new) for k, v in obj.items()}
    return obj


def _json_files(root: Path):
    yield root / "workspace.json"
    yield from (root / "jobs").glob("*/job.json")


def pack(workspace: Path, slug: str, title: str, tag: str, blurb: str = "") -> Path:
    workspace = Path(workspace).resolve()
    if not (workspace / "workspace.json").exists():
        raise SystemExit(f"{workspace} is not an mepd workspace")
    SHOWCASE.mkdir(exist_ok=True)
    out = SHOWCASE / f"{slug}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / slug
        for fp in workspace.rglob("*"):
            rel = fp.relative_to(workspace)
            if fp.is_dir() or not _keep(rel):
                continue
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, dest)
        for jf in _json_files(stage):
            if jf.exists():
                jf.write_text(json.dumps(_rewrite(json.loads(jf.read_text()), str(workspace), _ROOT_TOKEN)))
        with tarfile.open(out, "w:gz") as tar:
            tar.add(stage, arcname=".")
    (SHOWCASE / f"{slug}.json").write_text(json.dumps({"slug": slug, "title": title, "tag": tag, "blurb": blurb}, indent=1) + "\n")
    return out


def listing() -> list[dict]:
    items = []
    for meta in sorted(SHOWCASE.glob("*.json")):
        info = json.loads(meta.read_text())
        if (SHOWCASE / f"{info['slug']}.tar.gz").exists():
            items.append(info)
    return items


def unpack(slug: str, dest: Path) -> Path:
    """Copy the study into `dest` (a new, empty directory) as a workspace of its own."""
    archive = SHOWCASE / f"{slug}.tar.gz"
    if not archive.exists() or "/" in slug or slug.startswith("."):
        raise FileNotFoundError(slug)
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    for jf in _json_files(dest):
        if jf.exists():
            jf.write_text(json.dumps(_rewrite(json.loads(jf.read_text()), _ROOT_TOKEN, str(dest))))
    return dest


def free_name(parent: Path, slug: str, exists=lambda p: p.exists()) -> Path:
    for i in range(1, 1000):
        cand = parent / (f"example-{slug}" if i == 1 else f"example-{slug}-{i}")
        if not exists(cand):
            return cand
    raise RuntimeError("no free name")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("workspace", type=Path)
    p.add_argument("slug")
    p.add_argument("--title", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--blurb", default="")
    a = ap.parse_args()
    path = pack(a.workspace, a.slug, a.title, a.tag, a.blurb)
    print(f"{path} {path.stat().st_size / 1e6:.2f} MB")
