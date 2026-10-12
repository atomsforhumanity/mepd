"""Fetch the files of a data manifest (deploy/docker/retro-data.txt), checked
against the SHA-256s it pins. Standard library only.

    fetch_data.py MANIFEST DEST [--mirror URL]     # into place, as the image needs them
    fetch_data.py MANIFEST DIR --flat [--mirror URL] [--from DIR]
                                                   # a mirror: one file per name,
                                                   # SHA256SUMS and README.md

Each file comes from the mirror (URL/<name>) if one is given, else (or if
the mirror fails) from its upstream URL; --from DIR takes it from a local
copy first (e.g. ~/.cache/mepd/retro). A file whose checksum does not match
is never used, wherever it came from: the checksums in the manifest, kept
in git, are what the mirror is trusted against.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            sha, name, url, dests = line.split()
            entries.append({"sha256": sha, "name": name, "url": url, "dests": dests.split(",")})
    return entries


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, out: Path, attempts: int = 4) -> None:
    """`url` into `out`, retrying server errors and timeouts (Zenodo answers
    504 under load) after 5, 15 and 45 s."""
    req = urllib.request.Request(url, headers={"User-Agent": "mepd-fetch-data"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            client_error = isinstance(exc, urllib.error.HTTPError) and exc.code < 500
            if client_error or attempt == attempts - 1:
                raise
            wait = 5 * 3 ** attempt
            print(f"{url}: {exc}; retrying in {wait} s", file=sys.stderr, flush=True)
            time.sleep(wait)


def fetch(entry: dict, out: Path, mirror: str | None, local: Path | None) -> None:
    """`entry` into `out`, from the first source whose copy matches."""
    sources = []
    if local:
        # Under its mirror name, or a name it is installed as (an image's or
        # mepd's cache layout, e.g. aizynth-data/uspto_templates.csv.gz).
        names = {entry["name"], *(Path(d).name for d in entry["dests"])}
        sources += [("local", p) for n in sorted(names) for p in sorted(local.rglob(n))]
    if mirror:
        sources.append(("mirror", f"{mirror.rstrip('/')}/{entry['name']}"))
    sources.append(("upstream", entry["url"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    for kind, src in sources:
        fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{entry['name']}.")
        os.close(fd)
        tmp = Path(tmp)
        try:
            if kind == "local":
                shutil.copyfile(src, tmp)
            else:
                _download(src, tmp)
            got = sha256(tmp)
            if got == entry["sha256"]:
                tmp.chmod(0o644)
                tmp.replace(out)
                print(f"{entry['name']}: from {kind} ({src})", flush=True)
                return
            print(f"{entry['name']}: {kind} copy has sha256 {got}, expected {entry['sha256']}; skipped",
                  file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"{entry['name']}: {kind} failed ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        finally:
            tmp.unlink(missing_ok=True)
    sys.exit(f"{entry['name']}: no source had the expected file")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--mirror", default=os.environ.get("DATA_MIRROR") or None)
    ap.add_argument("--flat", action="store_true", help="lay out a mirror instead of the image's tree")
    ap.add_argument("--from", dest="local", type=Path, help="a directory to take matching copies from first")
    args = ap.parse_args()

    entries = read_manifest(args.manifest)
    if args.flat:
        for e in entries:
            out = args.dest / e["name"]
            if out.exists() and sha256(out) == e["sha256"]:
                print(f"{e['name']}: already there", flush=True)
            else:
                fetch(e, out, args.mirror, args.local)
        (args.dest / "SHA256SUMS").write_text("".join(f"{e['sha256']}  {e['name']}\n" for e in entries))
        (args.dest / "README.md").write_text(
            "# mepd retrosynthesis data (mirror)\n\nUnmodified copies of public files, mirrored so that "
            "building the mepd image does not depend on the original hosts. Each file, its source and "
            "license:\n\n" + "".join(f"- `{e['name']}`: {e['url']}\n" for e in entries) +
            "\nZenodo records 7341155 (PaRoutes: Genheden & Bjerrum, Digital Discovery 2022) and 7797465 "
            "(AiZynthFinder models: Genheden et al., J. Cheminform. 2020; Saigiridharan et al., "
            "J. Cheminform. 2024) are licensed CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). "
            "The ZINC stock (figshare 23086469) is distributed with AiZynthFinder (MIT); ZINC: Irwin "
            "et al., J. Chem. Inf. Model. 2020.\n\nChecksums: SHA256SUMS.\n")
        return
    for e in entries:
        first = args.dest / e["dests"][0]
        fetch(e, first, args.mirror, args.local)
        for other in e["dests"][1:]:
            link = args.dest / other
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            os.link(first, link)


if __name__ == "__main__":
    main()
