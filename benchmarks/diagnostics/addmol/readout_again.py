"""The read-out of a finished `mepd addmol test` result, computed again with the code as it is (no new rows):
the steps whose apparent barrier changes most, before and now.

    readout_again.py QUERY.json RESULT.json [N]
"""
import json
import sys


def main(query_fp, result_fp, n=6):
    from mepd import addmol

    q = json.load(open(query_fp))
    d = json.load(open(result_fp))
    new = addmol.result(q, d["candidate"], d["estimates"], status=d["status"])["readout"]
    old = d["readout"]

    def top(r):
        rows = [x for x in r["steps"] if x.get("change") is not None]
        rows.sort(key=lambda x: x["change"])
        return [{"step": x["step"], "label": x["label"], "before": round(x["before"], 1), "after": round(x["after"], 1),
                 "change": round(x["change"], 1), "group": x["group"], "through": x["through"],
                 **({"before_this_step": round(x["before_this_step"], 1)} if "before_this_step" in x else {})}
                for x in rows[:int(n)]]

    print(json.dumps({"file": result_fp.split("/")[-1], "candidate": d["candidate"]["name"],
                      "largest_changes_as_written": top(old), "largest_changes_now": top(new),
                      "through_rows_as_written": sum(x["group"] == "through" for x in old["steps"]),
                      "through_rows_now": sum(x["group"] == "through" for x in new["steps"]),
                      "target_as_written": old["target"]["with"]["final"], "target_now": new["target"]["with"]["final"],
                      "route_peak_now": new["peak_routes"]["with"][0]["peak"]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
