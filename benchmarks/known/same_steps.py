"""identical.py for an expansion whose steps come back from a job queue (--steer access, the fast connector): the
order of `steps` then follows which job ended first and differs between two runs of one commit. Compares two runs
identical.py made (OUT_DIR/run-<commit>/summary.json), everything but the step order: species, reactions and
settings exactly; the steps as a set by label (ends, TS energy, barriers, files); the final kinetics' molecules,
their free energies and final amounts; the mirror-image and molecule records' sizes.

    python benchmarks/known/same_steps.py OUT_DIR COMMIT_A COMMIT_B
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identical import strip, without_record  # noqa: E402


def load(out: Path, commit: str) -> dict:
    o = out / f"run-{commit}"
    return without_record(strip(json.loads((o / "summary.json").read_text()), str(o)))


def unordered(summary: dict) -> dict:
    steps = {st["label"]: {k: v for k, v in st.items() if k != "warning"} for st in summary.get("steps", [])}
    assert len(steps) == len(summary.get("steps", [])), "two steps with one label"
    mol = ((summary.get("rounds") or [{}])[-1].get("kinetics") or {}).get("molecules") or {}
    mirrors = summary.get("mirror_images") or {}
    rest = {k: v for k, v in summary.items() if k not in ("steps", "rounds", "mirror_images", "warnings")}
    return {"rest": rest, "steps": steps, "warnings": sorted(summary.get("warnings") or []),
            "rounds": [{k: v for k, v in r.items() if k != "kinetics"} for r in summary.get("rounds", [])],
            "kinetics": [{k: v for k, v in (r.get("kinetics") or {}).items() if k not in ("molecules", "mirror_images")}
                         for r in summary.get("rounds", [])],
            "molecules": {"names": mol.get("names"), "g_kcal": mol.get("g_kcal"), "final": mol.get("final"),
                          "max": mol.get("max"), "counts": mol.get("counts"),
                          "ts": sorted((x["r"], x["p"], x["g_ts_kcal"], x["twin"]) for x in mol.get("reactions", []))},
            "mirrors": {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in mirrors.items()
                        if k != "species_mirror"} | {"species_mirror": mirrors.get("species_mirror")}}


def main(argv):
    out, a, b = Path(argv[0]).resolve(), argv[1], argv[2]
    x, y = unordered(load(out, a)), unordered(load(out, b))
    differ = sorted(k for k in x if x[k] != y[k])
    order = [st["label"] for st in load(out, a).get("steps", [])] == [st["label"] for st in load(out, b).get("steps", [])]
    rec = {"a": a, "b": b, "same_but_step_order": not differ, "differ": differ, "same_step_order": order,
           "steps": len(x["steps"])}
    (out / f"same_steps-{a}-{b}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return 0 if not differ else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
