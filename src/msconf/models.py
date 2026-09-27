"""Confidence models: P(correct | x, in-library), P(in-library | x), their calibration, label-shift
adjustment to a target prevalence, and baselines with the same tuning budget.

p_correct(x; pi) = P(in | x; pi) * P(correct | x, in)
  - P(correct | x, in) is invariant to the in-library prevalence pi (label shift on in/absent);
  - P(in | x; pi) is obtained from the calibrated P(in | x; pi_cal) by Bayes prior shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import prior_shift
from .features import GROUPS

FEATURE_SETS = {
    "meta_only": ["meta"],
    "sim_only": ["sim"],
    "base": ["meta", "sim", "sim_extra", "analog"],
    "full": ["meta", "sim", "sim_extra", "analog", "crowd_spec", "crowd_struct", "crowd_family"],
    "full_minus_family": ["meta", "sim", "sim_extra", "analog", "crowd_spec", "crowd_struct"],
    "full_minus_spec": ["meta", "sim", "sim_extra", "analog", "crowd_struct", "crowd_family"],
    "full_minus_struct": ["meta", "sim", "sim_extra", "analog", "crowd_spec", "crowd_family"],
    "sim_plus_crowd": ["sim", "crowd_spec", "crowd_struct", "crowd_family"],
}
# Engine-agnostic variant: only what an external engine's candidate list + structures give us
# (no access to the engine's reference library).
CANDIDATE_ONLY = ["precursor_mz", "adduct_na", "instrument", "collision_energy", "n_peaks", "spec_entropy", "n_win_mol",
                  "s1", "s2", "s3", "gap12", "gap13",
                  "n_near_tie", "n_win_ge05", "tani12", "tani_mean_top5", "n_win_struct_nbrs",
                  "fam_ent_topk", "fam_frac_topk_same", "fam_n_win_same", "fam_spec_crowd_train", "fam_struct_crowd_train"]


def columns(set_name: str) -> list[str]:
    if set_name == "candidate_only":
        return list(CANDIDATE_ONLY)
    cols = []
    for g in FEATURE_SETS[set_name]:
        cols += GROUPS[g]
    for c in cols:  # guard: truth-derived columns can never become features (failure mode #3)
        assert not c.startswith("ana_") and c not in {"mol", "correct", "in_lib", "scenario", "top1", "top5"}, c
    return cols


# ----------------------------------------------------------------------------- LightGBM with budget
def lgbm_space(rng, budget):
    cfgs = []
    for _ in range(budget):
        cfgs.append({
            "num_leaves": int(rng.choice([7, 15, 31, 63])),
            "learning_rate": float(rng.choice([0.02, 0.05, 0.1])),
            "n_estimators": int(rng.choice([200, 400, 800])),
            "min_child_samples": int(rng.choice([20, 50, 100, 200])),
            "subsample": float(rng.choice([0.7, 0.9, 1.0])),
            "subsample_freq": 1,
            "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
            "reg_lambda": float(rng.choice([0.0, 1.0, 10.0])),
        })
    return cfgs


def logloss(p, y, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def fit_lgbm(Xtr, ytr, wtr, Xva, yva, wva, budget, seed):
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    best = None
    trials = []
    for k, hp in enumerate(lgbm_space(rng, budget)):
        m = lgb.LGBMClassifier(objective="binary", random_state=seed + k, verbose=-1, n_jobs=2, **hp)
        m.fit(Xtr, ytr, sample_weight=wtr)
        ll = logloss(m.predict_proba(Xva)[:, 1], yva, wva)
        trials.append({**hp, "val_logloss": ll})
        if best is None or ll < best[0]:
            best = (ll, m, hp)
    return best[1], {"best": best[2], "best_val_logloss": best[0], "trials": trials}


def fit_lr(Xtr, ytr, wtr, Xva, yva, wva, budget, seed):
    Cs = np.logspace(-4, 3, budget)
    best = None
    trials = []
    for C in Cs:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        m.fit(np.nan_to_num(Xtr), ytr, logisticregression__sample_weight=wtr)
        ll = logloss(m.predict_proba(np.nan_to_num(Xva))[:, 1], yva, wva)
        trials.append({"C": float(C), "val_logloss": ll})
        if best is None or ll < best[0]:
            best = (ll, m, C)
    return best[1], {"best": {"C": float(best[2])}, "best_val_logloss": best[0], "trials": trials}


class ScoreModel:
    """Untrained single-score 'model' (raw threshold baselines): the score itself is the confidence."""

    def __init__(self, col):
        self.col = col

    def raw(self, df):
        return df[self.col].values.astype(float)


class TrainedModel:
    def __init__(self, est, cols, kind):
        self.est, self.cols, self.kind = est, cols, kind

    def raw(self, df):
        X = df[self.cols].values.astype(float)
        if self.kind == "lr":
            X = np.nan_to_num(X)
        return self.est.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- calibration
class Isotonic:
    def fit(self, raw, y, w):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(raw, y, sample_weight=w)
        return self

    def __call__(self, raw):
        return self.iso.predict(raw)


class GroupOffset:
    """Per-group logit offsets on top of a global calibrator, shrunk toward 0 (ridge), fitted only for
    groups with at least `min_n` calibration rows; smaller groups share offset 0 (merged)."""

    def __init__(self, min_n: int, lam: float = 20.0):
        self.min_n, self.lam = min_n, lam
        self.off = {}

    def fit(self, p, y, w, groups):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        z = np.log(p / (1 - p))
        for g in pd.unique(groups):
            m = groups == g
            if m.sum() < self.min_n:
                continue
            # 1-D Newton on penalized weighted log-likelihood for offset b
            b = 0.0
            for _ in range(50):
                q = 1 / (1 + np.exp(-(z[m] + b)))
                grad = np.sum(w[m] * (y[m] - q)) - self.lam * b
                hess = -np.sum(w[m] * q * (1 - q)) - self.lam
                step = grad / hess
                b -= step
                if abs(step) < 1e-8:
                    break
            self.off[g] = float(b)
        return self

    def __call__(self, p, groups):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        z = np.log(p / (1 - p)) + np.array([self.off.get(g, 0.0) for g in groups])
        return 1 / (1 + np.exp(-z))


class ConfidenceSystem:
    """Pair of calibrated models producing p_correct and p_in at any target prevalence."""

    def __init__(self, name, mC, mI, calC, calI, pi_cal, groupC=None, groupI=None, group_col=None):
        self.name, self.mC, self.mI, self.calC, self.calI = name, mC, mI, calC, calI
        self.pi_cal = pi_cal
        self.groupC, self.groupI, self.group_col = groupC, groupI, group_col

    def p_correct_given_in(self, df):
        p = self.calC(self.mC.raw(df))
        if self.groupC is not None:
            p = self.groupC(p, df[self.group_col].values)
        return p

    def p_in_cal(self, df):
        p = self.calI(self.mI.raw(df))
        if self.groupI is not None:
            p = self.groupI(p, df[self.group_col].values)
        return p

    def predict(self, df, pi_target=None):
        """pi_target None -> no label-shift correction (probabilities at the calibration prevalence)."""
        pc = self.p_correct_given_in(df)
        pin = self.p_in_cal(df)
        if pi_target is not None:
            pin = prior_shift(pin, self.pi_cal, pi_target)
        pin = np.maximum(pin, 0)
        return np.clip(pin * pc, 0, 1), pin


class DirectSystem:
    """A single calibrated P(correct|x) at the calibration prevalence (what a practitioner would fit
    without modelling library absence). P(in) is not modelled; for tiering it uses p_in = p_correct."""

    def __init__(self, name, m, cal):
        self.name, self.m, self.cal = name, m, cal

    def predict(self, df, pi_target=None):
        p = self.cal(self.m.raw(df))
        return p, p
