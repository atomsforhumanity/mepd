"""What a run's summary.json says of the staged procedure's two changes (read from summary.json only): how the
run ended, each step from the seed with its product, heights, whether the kinetics use it and why not, its
window and steering terms; the steering single points the final kinetics use; the answer.

    python benchmarks/staged/read_summary.py RUN_DIR [RUN_DIR ...] [--json OUT.json]
"""
import collections
import json
import sys
from pathlib import Path

KCAL = 627.5094740631


def read(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    st, sp = s["steering"], s["species"]
    seed_steps = []
    for x in s["steps"]:
        if 0 not in (x["a"], x["b"]):
            continue
        other = x["b"] if x["a"] == 0 else x["a"]
        side = 0 if x["a"] == 0 else 1
        closed = x.get("closed") or {}
        seed_steps.append({
            "label": x["label"], "product": sp[other]["smiles"],
            "ts_electronic_above_seed_kcal": round((x["ts_energy"] - s["seed_energy"]) * KCAL, 2),
            "free_barrier_from_seed_kcal": (round(x["free_energy_barrier_kcal"][side], 2)
                                            if x.get("free_energy_barrier_kcal") else None),
            "used": not closed, "closed": closed.get("reason"), "closed_value_kcal": closed.get("value_kcal"),
            "closed_on": closed.get("energy_used"), "provisional": closed.get("provisional"),
            "verified": x["files"].get("verified"), "kind": x["files"].get("ts_kind", "ts"),
            "in_window": None if "deciding_window" not in x else x["deciding_window"] is not None,
            "steer": x["ts_energy_terms"]["steer"], "shift_lent": x.get("shift_lent")})
    mol = (s["rounds"][-1].get("kinetics") or {}).get("molecules") or {}
    answer = {}
    if mol.get("names"):
        seed = {mol["names"][i] for i in mol["seed"]}
        made = {k: c for k, c in zip(mol["names"], mol["final"]) if k not in seed}
        tot = sum(made.values())
        answer = {k: c / tot for k, c in sorted(made.items(), key=lambda kv: -kv[1]) if tot > 0 and c / tot >= 1e-4}
    steps = collections.Counter(((x.get("closed") or {}).get("reason") or "used")
                                + ("/provisional" if (x.get("closed") or {}).get("provisional") else "")
                                for x in s["steps"])
    return {
        "run": str(run), "procedure": st.get("procedure", "classic"), "stopped": s["rounds"][-1].get("stopped"),
        "cutoff_kcal": st.get("cutoff_kcal"), "species": len(sp), "steps": len(s["steps"]), "steps_by_state": dict(steps),
        "kinetics_energy_basis": st.get("kinetics_energy_basis"),
        "single_points_used": {"species": sum(x["energy_terms"]["steer"] == "single_point" for x in sp),
                               "ts": sum(x["ts_energy_terms"]["steer"] == "single_point" for x in s["steps"])},
        "single_points_computed": (st.get("single_points") or {}).get("computed"),
        "shift_lent": {k: len(v) for k, v in (st.get("shift_lent") or {}).items()},
        "open_steps_on_a_lent_shift": [x["label"] for x in s["steps"] if x.get("shift_lent")],
        "open_steps_with_an_end_on_a_lent_shift": [x["label"] for x in s["steps"] if x.get("end_shift_lent")],
        "provisional_closures": st.get("provisional_closures"),
        "steer_fallbacks": len(st.get("steer_fallbacks") or []),
        "deciding_window_steps": st.get("deciding_window_steps"),
        "seed_steps": seed_steps, "answer": answer}


def main(argv):
    out = None
    if "--json" in argv:
        out = argv[argv.index("--json") + 1]
        argv = [a for a in argv if a not in ("--json", out)]
    rows = [read(Path(a)) for a in argv]
    for r in rows:
        print(f"== {r['run']}: {r['procedure']}, stopped {r['stopped']}, {r['species']} species, {r['steps']} steps "
              f"{r['steps_by_state']}, cutoff {r['cutoff_kcal']:.2f}")
        print(f"   basis: {r['kinetics_energy_basis']}")
        print(f"   single points used {r['single_points_used']}, computed {r['single_points_computed']}; lent "
              f"{r['shift_lent']}; provisional {r['provisional_closures']}; fallbacks {r['steer_fallbacks']}")
        for x in r["seed_steps"]:
            print(f"   {x['label']:24s} {x['product']:28s} E {x['ts_electronic_above_seed_kcal']:7.2f}  G "
                  f"{x['free_barrier_from_seed_kcal']}  {'used' if x['used'] else x['closed']} "
                  f"{'' if x['used'] or x['closed_value_kcal'] is None else round(x['closed_value_kcal'], 2)} "
                  f"window={x['in_window']} steer={x['steer']} verified={x['verified']} {x['kind']}")
        print(f"   answer: { {k: round(v, 4) for k, v in r['answer'].items()} }")
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
