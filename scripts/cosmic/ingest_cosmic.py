"""Join SIRIUS/COSMIC summaries with MassSpecGym truth and compare selective accuracy with the proposed
system on the same subset. COSMIC scores refer to CSI:FingerID's own top-1 (structure-database search);
the proposed system scores the spectral-library top-1 in scenario (a). Each method is evaluated on its own
top-1 (risk-coverage). Caveat: CSI:FingerID/COSMIC were trained on public spectra that may include
MassSpecGym molecules, so COSMIC may be optimistic here."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evaluation import risk_coverage  # noqa: E402
from msconf.interfaces import read_sirius_structure_candidates  # noqa: E402
from msconf.chem import Standardizer  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
summ = der / "cosmic" / "summaries" / "structure_identifications.tsv"
if not summ.exists():
    raise SystemExit(f"missing {summ}: run scripts/cosmic/run_sirius.sh first (needs a SIRIUS license login)")
s = pd.read_csv(summ, sep="\t")
idc = next(c for c in ["mappingFeatureId", "featureId", "id"] if c in s.columns)
conf = next(c for c in ["ConfidenceScoreApproximate", "ConfidenceScoreExact", "ConfidenceScore"] if c in s.columns)
std = Standardizer(cfg["structure"])
s["ik14_pred"] = s.smiles.map(lambda x: std.record(x)["ik14"])
q = pd.read_parquet(der / "queries.parquet")
mols = pd.read_parquet(der / "molecules.parquet").set_index("mol_idx")
q["ik14_true"] = q.mol_idx.map(mols.ik14)
m = s.merge(q[["identifier", "ik14_true"]], left_on=idc, right_on="identifier", how="inner")
m["correct"] = (m.ik14_pred == m.ik14_true).astype(float)
m[conf] = pd.to_numeric(m[conf], errors="coerce").fillna(-1)
res = {"n": int(len(m)), "cosmic_top1_acc": float(m.correct.mean()), "cosmic_rc": risk_coverage(m[conf].values, m.correct.values)}
write_result(cfg, "eval_cosmic_subset", res)
print(res)
