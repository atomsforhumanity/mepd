"""Download the ReactionT5 models into Hugging Face's cache, without transformers.

Runs in the Dockerfile's `reactiont5` stage, which has only Python and
huggingface_hub, so the weights' layer does not depend on uv.lock: a
dependency change neither downloads nor pushes them again.

    fetch_reactiont5.py

They land where transformers looks for them ($HF_HOME, ~/.cache/huggingface),
at the revision `from_pretrained` asks for (main). The image check loads each
one with Hugging Face offline, so a misplaced download fails the build.
"""
from __future__ import annotations

# mepd.retro.proposers (_ReactionT5): the retrosynthesis model and the forward
# model that checks its proposals.
MODELS = ["sagawa/ReactionT5v2-retrosynthesis", "sagawa/ReactionT5v2-forward"]


def main() -> None:
    from huggingface_hub import snapshot_download

    for repo in MODELS:
        print(f"{repo}: {snapshot_download(repo)}", flush=True)


if __name__ == "__main__":
    main()
