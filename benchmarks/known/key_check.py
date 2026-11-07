"""Do an earlier expansion's cache keys survive into a follow-up? Imports a real `discovery expand` folder as
`--known` (expand_network's own import, with the real engines; nothing is computed: the run stops right after
the import) and writes what it took: species, steps, steering single points and Hessians, against what the
folder holds.

    python benchmarks/known/key_check.py KNOWN_FOLDER SEED_INDEX EXPLORER.toml STEER.toml T_K OUT.json [--permute]

--permute: the seed's atoms in another order (reversed), so the folder's must be mapped onto it.
"""
import json
import sys
from pathlib import Path

import numpy as np


class _Stop(Exception):
    pass


def main(argv):
    from mepd.cli_common import _open_run_inputs
    from mepd.discovery.known import read_known
    from mepd.discovery.network_expansion import expand_network
    from mepd.nodes.node import StructureNode
    from mepd.web.workspace import level_key
    from qcdata import Structure

    folder, seed_index, explorer, steer, temperature, out = (Path(argv[0]), int(argv[1]), Path(argv[2]),
                                                             Path(argv[3]), float(argv[4]), Path(argv[5]))
    permute = "--permute" in argv
    runs, skipped = read_known([folder], level_key(explorer.read_text()))
    s = Structure.open(str(folder / "species" / f"species_{seed_index}.xyz"))
    if permute:
        order = list(range(len(s.symbols)))[::-1]
        s = s.model_copy(update={"symbols": [s.symbols[i] for i in order], "geometry": np.asarray(s.geometry)[order]})
    said = []

    def on_event(event, payload):
        if event == "known":
            said.append(payload["message"])
            if payload["message"].startswith(f"known run {folder}:") or " not used:" in payload["message"]:
                raise _Stop()

    kin = {"temperature": temperature, "time_s": 3600.0, "threshold": 0.01, "free_energy": True, "standard_state": "1M"}
    try:
        expand_network(StructureNode(structure=s), _open_run_inputs(explorer).engine, steer="access",
                       access_cutoff_kcal=40.0, connect=lambda pairs, nodes: [], kinetics=kin,
                       steer_engine=_open_run_inputs(steer).engine, steer_save_dir=out.parent / "steer_single_points",
                       known={"runs": runs, "skipped": skipped, "dir": out.parent / "ts"}, on_event=on_event)
    except _Stop:
        pass
    summary = json.loads((folder / "summary.json").read_text())
    held = {"species": len(summary["species"]), "steps": len(summary["steps"]),
            "single_points": len(list((folder / "steer_single_points" / "single_points").glob("*.json"))),
            "hessians": len(runs[0].free_cache) if runs else 0}
    out.write_text(json.dumps({"folder": str(folder), "seed_index": seed_index, "permuted": permute,
                               "skipped": skipped, "said": said, "folder_holds": held}, indent=1))
    print("\n".join(said))
    print("folder holds:", held)


if __name__ == "__main__":
    main(sys.argv[1:])
