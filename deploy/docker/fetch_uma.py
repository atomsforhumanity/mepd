"""Download FAIR-Chem UMA weights into fairchem's cache, without fairchem.

Runs in the Dockerfile's `uma` stage, which has only Python and
huggingface_hub, so the weights' layer does not depend on uv.lock: a
dependency change neither downloads nor pushes them again.

    fetch_uma.py uma-s-1p2p1 [...]      (HF_TOKEN: an account with access to facebook/UMA)

The files and arguments are fairchem's own (fairchem.core.calculate.
pretrained_mlip: the checkpoint, and the reference energies it loads with
it), so the cache is laid out exactly as fairchem reads it. The image
check (`mepd-entrypoint check`) loads each model through fairchem with
Hugging Face offline: if a fairchem release ever wants other files, the
build fails there.
"""
from __future__ import annotations

import os
import sys

REPO = "facebook/UMA"
FILES = [("checkpoints", "{model}.pt"),
         ("references", "iso_atom_elem_refs.yaml"),
         ("references", "form_elem_refs.yaml")]


def main(models: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    cache = os.environ.get("FAIRCHEM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "fairchem"))
    for model in models:
        if not model.startswith("uma-"):
            sys.exit(f"{model}: only UMA models ({REPO}) are fetched here")
        for subfolder, name in FILES:
            path = hf_hub_download(repo_id=REPO, filename=name.format(model=model), subfolder=subfolder,
                                   cache_dir=cache)
            print(f"{model}: {subfolder}/{name.format(model=model)} -> {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
