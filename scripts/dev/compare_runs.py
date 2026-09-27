"""Compare every number in results/*.json between two runs of the pipeline (reproducibility check).

Usage: python scripts/dev/compare_runs.py <run_a_root> <run_b_root> [--out results/repro_check.json]
(each root is a directory containing results/*.json; files present only in run A are listed)
Keys ignored: _provenance, seconds (wall-clock). Numbers are compared exactly and with a relative tolerance.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
other = Path(sys.argv[2])
out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
IGN = {"_provenance", "seconds"}


def walk(a, b, p, acc):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in IGN or k.endswith("_seconds"):
                continue
            if k not in a or k not in b:
                acc["missing"].append(f"{p}/{k}")
                continue
            walk(a[k], b[k], f"{p}/{k}", acc)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["diff"].append((p, f"len {len(a)} vs {len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{p}[{i}]", acc)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        acc["n"] += 1
        if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            acc["exact"] += 1
        elif math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            acc["close"] += 1
        else:
            acc["diff"].append((p, a, b))
    else:
        acc["n_other"] += 1
        if a != b:
            acc["diff"].append((p, str(a)[:80], str(b)[:80]))


report = {}
for f in sorted((ROOT / "results").glob("*.json")):
    g = other / "results" / f.name
    if not g.exists():
        report[f.name] = {"status": "only in this run"}
        continue
    acc = {"n": 0, "exact": 0, "close": 0, "n_other": 0, "diff": [], "missing": []}
    walk(json.load(open(f)), json.load(open(g)), "", acc)
    report[f.name] = {"status": "identical" if not acc["diff"] and not acc["missing"] else "differs", "n_numbers": acc["n"],
                      "n_exact": acc["exact"], "n_close": acc["close"], "n_diff": len(acc["diff"]), "diff_examples": acc["diff"][:10],
                      "missing_keys": acc["missing"][:10], "n_missing": len(acc["missing"])}
summary = {k: v["status"] for k, v in report.items()}
print(json.dumps(summary, indent=1))
for k, v in report.items():
    if v["status"] == "differs":
        print(k, v["n_diff"], v["diff_examples"][:5], v["missing_keys"][:5])
if out:
    json.dump({"other_run": str(other), "files": report}, open(out, "w"), indent=1)
