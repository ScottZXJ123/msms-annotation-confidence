"""Two additional baselines requested after the main test evaluation (the proposed method is NOT changed):

 1. MS2Query-style neighbourhood-consistency scorer (re-implemented without leakage, trained on the train fold,
    calibrated on validation with the same rule as every other system), in three uses:
      B_ms2q_rerank_direct  MS2Query's own top-1 (argmax RF score over window + analogue candidates) and its score
      B_ms2q_conf_direct    RF score of OUR engine's top-1 (same annotations as the proposed system)
      B_ms2q_dec            MS2Query's neighbourhood features for our top-1 inside OUR decomposed framework
                            (LightGBM P(correct|in), P(in), label-shift) — isolates "how neighbourhood information
                            is used" from "how absence/prevalence are handled".
 2. Target-decoy competition (TDA):
      B_tda_margin_direct   per-query score s1 − d1 (best target minus best decoy in the window), calibrated on val
      TDA-FDR (set level)   per batch: concatenated target-decoy competition, q-values, accept q <= 0.10; compared
                            with the proposed trusted tier (nominal <= 10% error) on the SAME batches.
Every configuration is fixed on train/validation only; the test fold is evaluated once.
Output: results/eval_new_baselines.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf import evalutil  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evalutil import class_weights, components, evaluate_weighted, probs  # noqa: E402
from msconf.evaluation import assign_tiers, em_prevalence, risk_coverage  # noqa: E402
from msconf.models import DirectSystem, ScoreModel, choose_calibrator  # noqa: E402
from msconf.pipeline import load_rows, train_decomposed, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
ev = cfg["evaluation"]
seed = cfg["project"]["seed"]
T_TR, T_AB = cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"]
evalutil.set_thresholds(T_TR, T_AB)
PRIMARY = json.load(open(path(cfg, "results") / "selection.json"))["primary_system"]
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
t0 = time.time()
rows = load_rows(der, "entropy")
rows = rows[rows.scenario.isin(["a", "b"])].copy()
rows["vsplit"] = rows.mol.map(split)
mdir = der / "models" / "entropy_new_baselines"
mdir.mkdir(parents=True, exist_ok=True)
log = {}

# ============================================================== MS2Query-style
C = pd.read_parquet(der / "ms2q_candidates.parquet")
key = rows[["qid", "scenario", "fold"]].drop_duplicates()
C = C.merge(key, on=["qid", "scenario"], how="inner")  # study population only (>= 1 window candidate)
FE = ["sim", "precursor_mz", "mz_diff", "nbr_sim", "nbr_tani"]
tr = C[C.fold == "train"]
sub = tr.sample(n=min(500_000, len(tr)), random_state=seed)
rf = RandomForestRegressor(n_estimators=250, max_depth=5, n_jobs=2, random_state=seed)  # MS2Query's settings
rf.fit(sub[FE].values, sub.tani_truth.values)
C["ms2q_score"] = np.concatenate([rf.predict(C[FE].values[a:a + 300_000]) for a in range(0, len(C), 300_000)])
log["ms2q_rf"] = {"n_train_pairs": int(len(sub)), "features": FE, "target": "Tanimoto(candidate, truth)",
                  "feature_importance": dict(zip(FE, rf.feature_importances_.round(4).tolist()))}
print("rf done", f"{time.time()-t0:.0f}s", flush=True)
# (i) MS2Query's own top-1
C = C.sort_values(["qid", "scenario", "ms2q_score", "sim"], ascending=[True, True, False, False])
own = C.drop_duplicates(["qid", "scenario"])[["qid", "scenario", "cand", "ms2q_score", "exact", "in_window"]]
rr = rows.merge(own, on=["qid", "scenario"], how="inner").rename(columns={"ms2q_score": "ms2q_top_score"})
rr["correct_own"] = rr.exact.astype(int)
log["ms2q_rerank_frac_top1_is_analog"] = float(1 - rr.in_window.mean())
# (ii) RF score of our top-1
ours = C[["qid", "scenario", "cand", "ms2q_score", "nbr_sim", "nbr_tani"]].rename(columns={"cand": "top1"})
rows = rows.merge(ours, on=["qid", "scenario", "top1"], how="left")
rows = rows.merge(rr[["qid", "scenario", "ms2q_top_score", "correct_own"]], on=["qid", "scenario"], how="left")
log["ms2q_missing_own_top1"] = int(rows.ms2q_score.isna().sum())

# ============================================================== TDA
T = pd.read_parquet(der / "tda.parquet")
rows = rows.merge(T, on=["qid", "scenario"], how="left")
rows["tda_margin"] = rows.s1 - rows.d1

val = rows[rows.fold == "val"]
systems = {}
# direct systems calibrated with the shared rule (cross-fit val-A <-> val-B, refit on all val)
for name, col, target in [("B_ms2q_conf_direct", "ms2q_score", "correct"), ("B_tda_margin_direct", "tda_margin", "correct"),
                          ("B_ms2q_rerank_direct", "ms2q_top_score", "correct_own")]:
    m = ScoreModel(col)
    cal, info = choose_calibrator(m, val, target)
    systems[name] = DirectSystem(name, m, cal)
    log[name] = {"calibrator": info}
# decomposed MS2Query features inside our framework (same LightGBM budget as the proposed system)
cols = ["s1", "precursor_mz", "nbr_sim", "nbr_tani"]
trn = rows[rows.fold == "train"]
vA, vB = val[val.vsplit == "A"], val[val.vsplit == "B"]
sd, lg = train_decomposed("B_ms2q_dec", "lgbm", cols, trn, vA, vB, cfg, seed)
sd.calC, iC = choose_calibrator(sd.mC, val[val.scenario == "a"], "correct")
sd.calI, iI = choose_calibrator(sd.mI, val, "in_lib")
sd.pi_cal = float(np.average(val.in_lib, weights=val.w))
systems["B_ms2q_dec"] = sd
log["B_ms2q_dec"] = {"tuning_C": lg["tuning_C"]["best"], "tuning_I": lg["tuning_I"]["best"], "calibrator_C": iC, "calibrator_I": iI}
for n, s in systems.items():
    (mdir / f"{n}.pkl").write_bytes(pickle.dumps(s))
print("systems ready", f"{time.time()-t0:.0f}s", flush=True)

# ============================================================== single test evaluation
test = rows[rows.fold == "test"].reset_index(drop=True)
prim = pickle.loads((der / "models" / "entropy" / f"{PRIMARY}.pkl").read_bytes())
S = {PRIMARY: prim, **systems}
LABEL = {n: ("correct_own" if n == "B_ms2q_rerank_direct" else "correct") for n in S}
COMP = {n: components(s, test) for n, s in S.items()}
res = {"primary": PRIMARY, "log": log, "sweep": {}}
for n in S:
    tt = test.assign(correct=test[LABEL[n]])
    res["sweep"][n] = {str(pi): {m: evaluate_weighted(COMP[n], tt, pi, m) for m in ["oracle", "em", "none"]} for pi in ev["prevalence_grid"]}

# paired molecule-cluster bootstrap: primary minus each new baseline (AURC, log loss), oracle prevalence
rng = np.random.default_rng(seed)
mol_idx = test.groupby("mol").indices
mk = np.array(list(mol_idx))


def ll(p, y, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def pm(ix, pi):
    d = test.iloc[ix]
    yin = d.in_lib.values.astype(float)
    w = class_weights(yin, d.w.values, pi)
    out = {}
    for n in S:
        c = COMP[n]
        pc, _ = probs({"direct": c["direct"], "pc": c["pc"][ix], "pin_cal": c["pin_cal"][ix], "pi_cal": c["pi_cal"]}, pi)
        y = d[LABEL[n]].values.astype(float)
        out[n] = (risk_coverage(pc, y, w)["aurc"], ll(pc, y, w))
    return out


res["bootstrap"] = {}
for pi in ev["prevalence_headline"]:
    point = pm(np.arange(len(test)), pi)
    draws = [pm(np.concatenate([mol_idx[k] for k in mk[rng.integers(0, len(mk), len(mk))]]), pi) for _ in range(ev["bootstrap_repeats"])]
    res["bootstrap"][str(pi)] = {f"{PRIMARY} - {n}": {
        "d_aurc": point[PRIMARY][0] - point[n][0], "d_aurc_ci": np.percentile([d[PRIMARY][0] - d[n][0] for d in draws], [2.5, 97.5]).tolist(),
        "d_logloss": point[PRIMARY][1] - point[n][1], "d_logloss_ci": np.percentile([d[PRIMARY][1] - d[n][1] for d in draws], [2.5, 97.5]).tolist()}
        for n in S if n != PRIMARY}
print("bootstrap done", f"{time.time()-t0:.0f}s", flush=True)

# set-level TDA-FDR vs proposed trusted tier on identical deployment-like batches (same sampler/seeds as 09)
Y, YIN = test.correct.values, test.in_lib.values
S1, D1 = test.s1.values, test.d1.values


def _pos_by_mol(mask):
    pos = np.nonzero(mask)[0]
    return {k: pos[ix] for k, ix in test.iloc[pos].groupby("mol").indices.items()}


a_by, b_by = _pos_by_mol(test.scenario.values == "a"), _pos_by_mol(test.scenario.values == "b")
ALLM = np.array(sorted(set(a_by) | set(b_by)))


def batch(pi, r):
    m = ALLM.copy()
    r.shuffle(m)
    h = len(m) // 2
    A = [a_by[k][r.integers(len(a_by[k]))] for k in m[:h] if k in a_by]
    B = [b_by[k][r.integers(len(b_by[k]))] for k in m[h:] if k in b_by]
    nA, nB = len(A), len(B)
    if nA / (nA + nB) > pi:
        nA = int(round(pi / (1 - pi) * nB))
    else:
        nB = int(round((1 - pi) / pi * nA))
    return np.array(A[:nA] + B[:nB])


def tda_qvalues(s, d):
    """Concatenated target-decoy competition within a batch; q-values for target wins (NaN for decoy wins)."""
    win_t = s > d
    score = np.where(win_t, s, d)
    o = np.argsort(-score, kind="stable")
    t_cum = np.cumsum(win_t[o])
    d_cum = np.cumsum(~win_t[o])
    fdr = (d_cum + 1) / np.maximum(t_cum, 1)
    qv = np.minimum.accumulate(fdr[::-1])[::-1]
    q = np.full(len(s), np.nan)
    q[o] = qv
    q[~win_t] = np.nan
    return q


setlvl = {}
for pi in [0.1, 0.2, 0.3, 0.5, 0.7]:
    r = np.random.default_rng(seed + int(pi * 100))
    recs = []
    for _ in range(ev["wetlab_repeats"]):
        ix = batch(pi, r)
        q = tda_qvalues(S1[ix], D1[ix])
        acc = np.nan_to_num(q, nan=np.inf) <= 0.10
        c = COMP[PRIMARY]
        pu = em_prevalence(c["pin_cal"][ix], c["pi_cal"])
        pc, pin = probs({"direct": False, "pc": c["pc"][ix], "pin_cal": c["pin_cal"][ix], "pi_cal": c["pi_cal"]}, pu)
        tr_ = assign_tiers(pc, pin, T_TR, T_AB) == "trusted"
        y = Y[ix]
        conf_tda = -np.nan_to_num(q, nan=np.inf)
        recs.append({"tda_cov": acc.mean(), "tda_err": 1 - y[acc].mean() if acc.any() else np.nan,
                     "tda_est_fdr": float(np.nanmax(q[acc])) if acc.any() else np.nan,
                     "prop_cov": tr_.mean(), "prop_err": 1 - y[tr_].mean() if tr_.any() else np.nan,
                     "prop_est_err": 1 - pc[tr_].mean() if tr_.any() else np.nan,
                     "tda_aurc": risk_coverage(conf_tda, y)["aurc"], "prop_aurc": risk_coverage(pc, y)["aurc"]})
    d = pd.DataFrame(recs)
    setlvl[str(pi)] = {k: {"mean": float(d[k].mean()), "q025": float(d[k].quantile(0.025)), "q975": float(d[k].quantile(0.975))} for k in d}
    setlvl[str(pi)]["P(tda_err > 0.10)"] = float((d.tda_err > 0.10).mean())
    setlvl[str(pi)]["P(prop_err > 0.10)"] = float((d.prop_err > 0.10).mean())
res["tda_set_level"] = setlvl
res["seconds"] = time.time() - t0
write_result(cfg, "eval_new_baselines", res)
print(json.dumps({k: v for k, v in res.items() if k in ("tda_set_level",)}, indent=1)[:3000])
