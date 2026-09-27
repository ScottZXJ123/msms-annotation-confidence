"""Build the per-molecule ClassyFire family table (superclass / class / subclass) from the lookup cache.

Handling (logged in LOG.md):
 - molecule found, class missing       -> class label '<superclass> [no class]' (it IS classified, just not at class level)
 - molecule found, subclass missing    -> subclass label '<class label> [no subclass]'
 - molecule not found in any ClassyFire source -> 'unclassified' at every level (reported fraction)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402

cfg = load_config()
import os  # noqa: E402

der = Path(os.environ["MSCONF_DERIVED"]) if os.environ.get("MSCONF_DERIVED") else path(cfg, "derived")
mols = pd.read_parquet(der / "molecules.parquet")
recs = {}
for line in open(path(cfg, "cache") / "classyfire.jsonl"):
    r = json.loads(line)
    if r["inchikey"] not in recs or r.get("status") == "ok":
        recs[r["inchikey"]] = r
UN = cfg["family"]["unclassified_label"]
rows = []
for ik in mols.inchikey_full:
    r = recs.get(ik, {"status": "missing"})
    if r.get("status") != "ok" or not r.get("superclass"):
        rows.append({"superclass": UN, "class": UN, "subclass": UN, "cf_status": r.get("status", "missing")})
        continue
    sup = r["superclass"]
    cls = r["class"] or f"{sup} [no class]"
    sub = r["subclass"] or f"{cls} [no subclass]"
    rows.append({"superclass": sup, "class": cls, "subclass": sub, "cf_status": "ok"})
fam = pd.concat([mols[["mol_idx", "ik14", "inchikey_full", "fold"]], pd.DataFrame(rows)], axis=1)
fam.to_parquet(der / "families.parquet")
summ = {"n_molecules": len(fam), "cf_status": fam.cf_status.value_counts().to_dict(),
        "frac_unclassified": float((fam["class"] == UN).mean()),
        "frac_no_class_level": float(fam["class"].str.endswith("[no class]").mean()),
        "n_groups": {lv: int(fam[lv].nunique()) for lv in ["superclass", "class", "subclass"]},
        "frac_unclassified_by_fold": fam.assign(u=fam["class"] == UN).groupby("fold").u.mean().to_dict(),
        "top_classes": fam["class"].value_counts().head(25).to_dict()}
write_result(cfg, "families_summary" + ("_" + der.name if os.environ.get("MSCONF_DERIVED") else ""), summ)
print(json.dumps(summ, indent=1)[:3000])
