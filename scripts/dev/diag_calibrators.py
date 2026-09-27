"""Validation-only: compare calibrators for the two components by cross-fitting between val-A and val-B
(fit on one half, evaluate on the other). Output: results/dev_calibrators.json"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evaluation import calibration_metrics  # noqa: E402
from msconf.pipeline import load_rows, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
seed = cfg["project"]["seed"]
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
rows = load_rows(der, "entropy")
val = rows[rows.fold == "val"].copy()
val["vsplit"] = val.mol.map(split)
sysm = pickle.loads((der / "models" / "entropy" / "M_full_minus_family.pkl").read_bytes())


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def mid_gap(p, y, w, lo=0.2, hi=0.9):
    m = (p > lo) & (p < hi)
    return float(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m])) if m.any() else None, float(w[m].sum() / w.sum())


out = {}
for comp, sc, tgt, model in [("C", ["a"], "correct", sysm.mC), ("I", ["a", "b"], "in_lib", sysm.mI)]:
    d = val[val.scenario.isin(sc)]
    res = {}
    for fit_h, ev_h in [("A", "B"), ("B", "A")]:
        F, E = d[d.vsplit == fit_h], d[d.vsplit == ev_h]
        rf, re_ = model.raw(F), model.raw(E)
        cands = {}
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(rf, F[tgt], sample_weight=F.w)
        cands["isotonic"] = iso.predict(re_)
        pl = LogisticRegression(C=1e6).fit(logit(rf)[:, None], F[tgt], sample_weight=F.w)
        cands["platt"] = pl.predict_proba(logit(re_)[:, None])[:, 1]
        cands["raw_uncalibrated"] = re_
        cands["avg_iso_platt"] = 0.5 * (cands["isotonic"] + cands["platt"])
        for k, p in cands.items():
            m = calibration_metrics(p, E[tgt].values, E.w.values)
            g, frac = mid_gap(p, E[tgt].values, E.w.values)
            res.setdefault(k, []).append({"fit": fit_h, "logloss": m["logloss"], "ece": m["ece"], "mid_gap_obs_minus_pred": g, "mid_frac": frac})
    out[comp] = {k: {"mean_logloss": float(np.mean([r["logloss"] for r in v])), "mean_ece": float(np.mean([r["ece"] for r in v])),
                     "mean_mid_gap": float(np.mean([r["mid_gap_obs_minus_pred"] for r in v])), "folds": v} for k, v in res.items()}
    for k, v in out[comp].items():
        print(comp, f"{k:18s} ll {v['mean_logloss']:.4f} ece {v['mean_ece']:.4f} mid gap {v['mean_mid_gap']:+.3f}", [round(f['mid_gap_obs_minus_pred'], 3) for f in v['folds']])
write_result(cfg, "dev_calibrators", out)
