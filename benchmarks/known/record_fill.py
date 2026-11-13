"""How much of its record a run filled in (docs/network-expansion.md, "What a run records"): for each recorded field
of summary.json, in how many rows it is there and not null; the reasons given (closed steps, removed steps, steering
single points); and checks of the record against the numbers the summary already had.

    python benchmarks/known/record_fill.py OUT.json SUMMARY.json [SUMMARY.json ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identical import RECORDED  # noqa: E402


def fill(rows: list, keys) -> dict:
    return {k: {"present": sum(k in r for r in rows), "not_null": sum(r.get(k) is not None for r in rows),
                "rows": len(rows)} for k in keys}


def one(fp: Path) -> dict:
    s = json.loads(Path(fp).read_text())
    steps, species, reactions = s.get("steps", []), s.get("species", []), s.get("reactions", [])
    out = {"summary": str(fp), "species": fill(species, RECORDED["species"]), "steps": fill(steps, RECORDED["steps"]),
           "reactions": fill(reactions, RECORDED["reactions"]),
           "steering": {k: s.get("steering", {}).get(k) for k in RECORDED["steering"]},
           "record_error": s.get("record_error"), "steps_removed": len(s.get("steps_removed", []))}
    out["closed_reasons"] = dict(Counter((st.get("closed") or {}).get("reason", "open") for st in steps))
    out["removed_reasons"] = dict(Counter(r["reason"] for r in s.get("steps_removed", [])))
    out["irc_failed_with_reason"] = sum(1 for r in s.get("steps_removed", [])
                                        if r["reason"] == "irc_failed" and (r.get("irc") or {}).get("why"))
    out["ts_steer"] = dict(Counter(str((st.get("ts_energy_terms") or {}).get("steer")) for st in steps))
    out["species_steer"] = dict(Counter(str((sp.get("energy_terms") or {}).get("steer")) for sp in species))
    out["barrier_missing"] = dict(Counter(m.rstrip("0123456789").strip() for st in steps
                                          for m in (st.get("free_energy_barrier_missing") or [])))
    out["placeholder_why"] = dict(Counter((st.get("placeholder_reason") or {}).get("why") for st in steps
                                          if st.get("placeholder_reason")))
    out["not_searched"] = dict(Counter(r["not_searched"]["reason"] for r in reactions if r.get("not_searched")))
    out["with_spectators"] = sum(1 for st in steps if (st.get("reacting") or {}).get("spectators"))
    out["unreadable_reacting"] = dict(Counter(st["reacting"]["unreadable"] for st in steps
                                              if "unreadable" in (st.get("reacting") or {})))
    groups = Counter(st.get("ts_group") for st in steps)
    out["ts_groups_of_several"] = sum(1 for n in groups.values() if n > 1)
    # checks against what the summary already had
    used = [st for st in steps if st.get("kinetics_barrier_kcal") is not None]
    # (where the TS lies above both ends: below one, the kinetics lift it to that end)
    diffs = [abs(f - k) for st in used
             if st.get("free_energy_barrier_kcal") and min(st["free_energy_barrier_kcal"]) >= 0
             for f, k in zip(st["free_energy_barrier_kcal"], st["kinetics_barrier_kcal"])]
    out["checks"] = {
        "steps_the_kinetics_use": len(used),
        "of_them_with_a_barrier_recorded": sum(st.get("free_energy_barrier_kcal") is not None for st in used),
        "max_abs_difference_to_kinetics_barrier_kcal": max(diffs) if diffs else None,
        "box_closed_steps": sum(st.get("kinetics_barrier_kcal") is None for st in steps),
        "of_them_with_a_closing_record": sum(st.get("kinetics_barrier_kcal") is None and st.get("closed") is not None
                                             for st in steps),
        "closed_reason_unknown": sum((st.get("closed") or {}).get("reason") == "unknown" for st in steps),
        "opened_before_found": sum(1 for st in steps if st.get("t_opened") and st.get("t_found")
                                   and st["t_opened"] < st["t_found"]),
        "verified_fast_or_leaf_steps_without_irc_record": sum(
            1 for st in steps if st["files"].get("verified") is True and st["files"].get("ts_kind") != "path_peak"
            and not st["files"].get("channels") and not st["files"].get("known") and st.get("irc") is None)}
    mol = ((s.get("rounds") or [{}])[-1].get("kinetics") or {}).get("molecules") or {}
    in_reactions = {x["step"] for x in mol.get("reactions", [])}
    out["checks"]["molecule_reactions_on_steps_marked_closed"] = sum(
        1 for m in in_reactions if m < len(steps) and steps[m].get("closed") is not None)
    out["checks"]["open_steps_that_are_no_molecule_reaction"] = (
        sum(1 for m, st in enumerate(steps) if st.get("closed") is None and m not in in_reactions) if mol else None)
    return out


def main(argv):
    out = [one(Path(fp)) for fp in argv[1:]]
    Path(argv[0]).write_text(json.dumps(out, indent=1))
    for o in out:
        print(json.dumps({k: o[k] for k in ("summary", "closed_reasons", "removed_reasons", "ts_steer", "checks",
                                            "record_error")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
