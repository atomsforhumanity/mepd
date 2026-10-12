"""Download, at image build time, what mepd otherwise fetches on first use.

Runs in the Dockerfile's `models` stage, which has mepd's dependencies but
not mepd itself (so editing mepd does not re-download any of this):

    prefetch.py models              # AIMNet2, ANI-2x, ReactionT5

Everything lands where mepd looks for it at run time, the caches under $HOME
(~/.cache/aimnet, ~/.cache/huggingface, ~/.local/share/Torchani). The
retrosynthesis data has a stage of its own (the Dockerfile's `retro`).
"""
from __future__ import annotations

import sys

# The no-account models of mepd.engines.mlip.MODELS whose packages are in
# mepd's environment (Orb and MACE are not: they conflict with fairchem-core).
AIMNET2_MODELS = ["aimnet2-rxn", "aimnet2", "aimnet2-nse", "aimnet2-2025"]
# mepd.retro.proposers: the ReactionT5 method's retro and forward models.
REACTIONT5_MODELS = ["sagawa/ReactionT5v2-retrosynthesis", "sagawa/ReactionT5v2-forward"]


def _water():
    from ase import Atoms

    return Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])


def models() -> None:
    from aimnet.calculators import AIMNet2Calculator
    from aimnet.calculators.aimnet2ase import AIMNet2ASE

    for name in AIMNET2_MODELS:
        atoms = _water()
        atoms.calc = AIMNet2ASE(AIMNet2Calculator(name, device="cpu"))
        print(f"{name}: E(water) = {atoms.get_potential_energy():.4f} eV", flush=True)

    import torchani

    atoms = _water()
    atoms.calc = torchani.models.ANI2x(periodic_table_index=True).ase()
    print(f"ani-2x: E(water) = {atoms.get_potential_energy():.4f} eV", flush=True)

    from huggingface_hub import snapshot_download

    for repo in REACTIONT5_MODELS:
        print(f"{repo}: {snapshot_download(repo)}", flush=True)


if __name__ == "__main__":
    what, args = sys.argv[1], sys.argv[2:]
    if what == "models":
        models()
    else:
        sys.exit(f"unknown: {what}")
