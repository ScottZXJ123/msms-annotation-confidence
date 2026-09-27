"""Import SIRIUS 6 / COSMIC results for the public subset and compare confidence quality with the proposed system.

Usage (repository root):  python scripts/cosmic/ingest_cosmic.py cosmic/results

What is compared: each system's confidence ranking and calibration for ITS OWN top-1 annotation.
COSMIC scores CSI:FingerID's top-1 from a structure database (PubChem); the proposed system scores the top-1
of spectral-library search (scenario (a): the truth is in the library, the analogue of PubChem containing it).
Strata: molecule in the published CSI:FingerID training structures (leaky reference) vs not (primary).
Main: COSMIC exact mode vs exact correctness (InChIKey first block after the shared RDKit standardization).
Supplement: COSMIC approximate mode vs "correct or MCES distance <= 2" (myopic_mces), and the proposed system's
score against the same approximate labels. COSMIC scores are not probabilities (SIRIUS documentation); they are
reported raw and after calibration on the validation subset with the same rule as every other system.
Output: results/eval_cosmic_subset.json
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.chem import Standardizer  # noqa: E402
from msconf.config import ROOT, load_config, path, write_result  # noqa: E402
from msconf.evalutil import components, probs  # noqa: E402
from msconf.evaluation import auroc, calibration_metrics, risk_coverage  # noqa: E402
from msconf.models import ScoreModel, choose_calibrator  # noqa: E402
from msconf.pipeline import load_rows, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
seed = cfg["project"]["seed"]
res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cosmic" / "results"
man = pd.read_csv(ROOT / "cosmic" / "subset_manifest.csv")
std = Standardizer(cfg["structure"])
ID = re.compile(r"MassSpecGymID\d+")

# ------------------------------------------------------------------ read SIRIUS summaries
frames = []
for f in sorted(res_dir.glob("summaries_*/structure_identifications.*")):
    sep = "\t" if f.suffix == ".tsv" else ","
    d = pd.read_csv(f, sep=sep)
    ids = []
    for _, r in d.iterrows():
        m = next((ID.search(str(v)) for v in r.values if isinstance(v, str) and ID.search(v)), None)
        ids.append(m.group(0) if m else None)
    d["identifier"] = ids
    frames.append(d)
if not frames:
    raise SystemExit(f"no structure_identifications.* under {res_dir}")
S = pd.concat(frames, ignore_index=True)
S = S[S.identifier.notna()].drop_duplicates("identifier")
cex = next(c for c in S.columns if c.lower().replace("_", "") in ("confidencescoreexact", "siriusconfidencescoreexact"))
cap = next(c for c in S.columns if c.lower().replace("_", "") in ("confidencescoreapproximate", "siriusconfidencescoreapproximate"))
smi_col = next(c for c in S.columns if c.lower() in ("smiles", "sirius_smiles"))
S["cosmic_exact"] = pd.to_numeric(S[cex], errors="coerce").replace([np.inf, -np.inf], np.nan)  # SIRIUS writes -Infinity when not computed
S["cosmic_approx"] = pd.to_numeric(S[cap], errors="coerce").replace([np.inf, -np.inf], np.nan)
S["pred_ik14"] = S[smi_col].map(lambda s: std.record(s)["ik14"] if isinstance(s, str) else None)
S["pred_smiles"] = S[smi_col]

# ------------------------------------------------------------------ truth + proposed system on the same spectra
mols = pd.read_parquet(der / "molecules.parquet").set_index("mol_idx")
Q = pd.read_parquet(der / "queries.parquet")[["identifier", "pos", "mol_idx", "fold"]]
D = man.merge(Q, on="identifier", how="left").merge(S[["identifier", "cosmic_exact", "cosmic_approx", "pred_ik14", "pred_smiles"]],
                                                    on="identifier", how="left")
D["true_ik14"] = D.mol_idx.map(mols.ik14)
D["true_smiles"] = D.mol_idx.map(mols.std_smiles.fillna(mols.smiles))
D["annotated"] = D.pred_ik14.notna()
D["cosmic_correct"] = (D.pred_ik14 == D.true_ik14).astype(float)
rows = load_rows(der, "entropy")
ra = rows[rows.scenario == "a"][["qid", "top1", "correct", "w"] + [c for c in rows.columns if c not in ("qid", "top1", "correct", "w")]]
D = D.merge(ra, left_on="pos", right_on="qid", how="left", suffixes=("", "_row"))
D["ours_top1_smiles"] = D.top1.map(lambda t: mols.std_smiles.fillna(mols.smiles).get(int(t)) if pd.notna(t) else None)
prim = json.load(open(path(cfg, "results") / "selection.json"))["primary_system"]
sysm = pickle.loads((der / "models" / "entropy" / f"{prim}.pkl").read_bytes())
ok = D.qid.notna()
D.loc[ok, "ours_p"] = sysm.p_correct_given_in(D[ok])  # scenario (a): P(correct | in) = p_correct at prevalence 1

# ------------------------------------------------------------------ approximate labels (MCES <= 2)
try:
    from myopic_mces import MCES

    def mces_le2(a, b):
        if not isinstance(a, str) or not isinstance(b, str):
            return np.nan
        try:
            return float(MCES(a, b, threshold=2, solver="PULP_CBC_CMD", solver_options={"msg": False})[1] <= 2)
        except Exception:
            return np.nan
    approx_method = "myopic_mces, threshold 2"
except ImportError:  # proxy, reported as such
    def mces_le2(a, b):
        return np.nan
    approx_method = "unavailable"
D["cosmic_approx_correct"] = [1.0 if c == 1 else mces_le2(p, t) for c, p, t in zip(D.cosmic_correct, D.pred_smiles, D.true_smiles)]
D["ours_approx_correct"] = [1.0 if c == 1 else mces_le2(p, t) for c, p, t in zip(D.correct, D.ours_top1_smiles, D.true_smiles)]

# ------------------------------------------------------------------ calibration of COSMIC on the validation subset
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
V = D[(D.split == "val") & D.annotated].copy()
V["vsplit"] = V.mol_idx.map(split)
V["w"] = 1.0
cal = {}
for mode, score, label in [("exact", "cosmic_exact", "cosmic_correct"), ("approx", "cosmic_approx", "cosmic_approx_correct")]:
    vv = V[V[label].notna() & V[score].notna()]
    if len(vv) >= 40 and vv[label].nunique() == 2:
        cal[mode], info = choose_calibrator(ScoreModel(score), vv, label)
    else:
        cal[mode], info = None, {"note": "too few validation results"}
    cal[mode + "_info"] = info

# ------------------------------------------------------------------ metrics with bootstrap CIs
T = D[D.split == "test"].copy()
rng = np.random.default_rng(seed)


def metrics(d, score, label, prob=None):
    y = d[label].values.astype(float)
    s = d[score].values.astype(float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {"n": int(len(y))}
    out = {"n": int(len(y)), "accuracy": float(y.mean()), **{k: v for k, v in risk_coverage(s, y).items() if k in ("aurc", "e_aurc", "cov@risk0.1", "risk@cov0.5")},
           "auroc": auroc(s, y)}
    if prob is not None:
        p = prob(d[m]) if callable(prob) else d[prob].values[m]
        c = calibration_metrics(p, y)
        out.update({"ece": c["ece"], "logloss": c["logloss"], "mean_pred": c["mean_p"]})
    return out


def boot(d, fn, B=2000):
    vals = []
    for _ in range(B):
        ix = rng.integers(0, len(d), len(d))
        vals.append(fn(d.iloc[ix]))
    return vals


def summarize(d, label_c, label_o):
    """COSMIC (all test queries: unannotated -> lowest confidence, counted wrong) vs proposed."""
    d = d.copy()
    d["cosmic_score_all"] = d["cosmic_" + ("exact" if label_c == "cosmic_correct" else "approx")].fillna(-1.0)
    d[label_c] = d[label_c].fillna(0.0)
    mode = "exact" if label_c == "cosmic_correct" else "approx"
    calf = cal[mode]
    d["cosmic_p_raw"] = d["cosmic_score_all"].clip(0, 1)
    d["cosmic_p_cal"] = calf(d["cosmic_score_all"].values) if calf is not None else np.nan
    out = {"cosmic": metrics(d, "cosmic_score_all", label_c, "cosmic_p_raw"),
           "cosmic_calibrated": metrics(d, "cosmic_score_all", label_c, "cosmic_p_cal") if calf is not None else None,
           "proposed": metrics(d, "ours_p", label_o, "ours_p"), "n_cosmic_unannotated": int((~d.annotated).sum())}
    for k in ["auroc", "e_aurc", "aurc", "accuracy"]:
        def diff(x, k=k):
            a, b = metrics(x, "ours_p", label_o).get(k), metrics(x, "cosmic_score_all", label_c).get(k)
            return np.nan if a is None or b is None else a - b
        v = np.array(boot(d, diff, 1000), float)
        out[f"proposed_minus_cosmic_{k}"] = {"point": diff(d), "ci95": np.nanpercentile(v, [2.5, 97.5]).tolist()}
    for who, sc, lab, pr in [("cosmic", "cosmic_score_all", label_c, "cosmic_p_raw"), ("proposed", "ours_p", label_o, "ours_p")]:
        for k in ["accuracy", "auroc", "e_aurc", "ece"]:
            v = np.array([metrics(x, sc, lab, pr).get(k, np.nan) for x in [d.iloc[rng.integers(0, len(d), len(d))] for _ in range(1000)]], float)
            out[who][f"{k}_ci95"] = np.nanpercentile(v, [2.5, 97.5]).tolist()
    return out


res = {"n_test": int(len(T)), "n_test_annotated_by_sirius": int(T.annotated.sum()),
       "n_test_confidence_missing": int((T.annotated & T.cosmic_exact.isna()).sum()), "n_val": int((D.split == "val").sum()), "approx_label_method": approx_method,
       "calibration": {k: v for k, v in cal.items() if k.endswith("_info")},
       "note": "COSMIC ranks structure-database (PubChem) hits; the proposed system ranks spectral-library hits "
               "(scenario a). Each system is evaluated on its own annotations.", "strata": {}}
for name, sub in [("not_in_csi_training (primary)", T[~T.in_csi_training]), ("in_csi_training (leaky reference)", T[T.in_csi_training])]:
    res["strata"][name] = {"exact": summarize(sub, "cosmic_correct", "correct"),
                           "approximate": summarize(sub, "cosmic_approx_correct", "ours_approx_correct")}
write_result(cfg, "eval_cosmic_subset", res)
D.drop(columns=[c for c in D.columns if c.startswith("ana_")]).to_csv(path(cfg, "tables") / "cosmic_subset_rows.csv", index=False)
print(json.dumps(res, indent=1)[:4000])
