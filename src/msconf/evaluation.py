"""Final-form evaluation harness (§2). Written before any confidence model.

Everything here consumes a table of scored query rows with columns:
  qid        query spectrum id (library position of the query spectrum)
  mol        true molecule index (used ONLY for labels, grouping and resampling — never as a feature)
  scenario   'a' (truth in library), 'b' (truth removed), 'c' (truth + structural neighbours removed)
  correct    1 if top-1 InChIKey first block == truth first block
  in_lib     1 if the truth is among the engine's candidates (a), else 0
  p_correct  predicted probability that top-1 is correct (at the batch's prevalence)
  p_in       predicted probability that the truth is in the library
plus any grouping columns (e.g. top-1 family, crowding bin).

Deployment-like batches: one spectrum per molecule; each molecule is either in the library or absent;
the prevalence pi (fraction of in-library queries in the study population) is a sweep variable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------- intervals
def wilson(k, n, level=0.95):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def clopper_pearson_upper(k, n, delta):
    """One-sided (1-delta) upper bound on a binomial proportion."""
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(stats.beta.ppf(1 - delta, k + 1, n - k))


# ----------------------------------------------------------------------------- batches
def make_batch(rows: pd.DataFrame, pi: float, rng: np.random.Generator, absent_scenario: str = "b",
               max_size: int | None = None) -> pd.DataFrame:
    """Deployment-like batch with in-library prevalence exactly ~pi among the study population.

    rows must contain scenario 'a' rows and `absent_scenario` rows (rows without any candidate are
    assumed already removed = outside the study population). One spectrum per molecule; a molecule is
    either in-library or absent within a batch (never both)."""
    a = rows[rows.scenario == "a"]
    b = rows[rows.scenario == absent_scenario]
    # one random spectrum per molecule for each scenario
    a1 = a.sample(frac=1, random_state=rng.integers(1 << 31)).drop_duplicates("mol")
    b1 = b.sample(frac=1, random_state=rng.integers(1 << 31)).drop_duplicates("mol")
    mols = np.union1d(a1.mol.values, b1.mol.values)
    rng.shuffle(mols)
    # randomly split molecules into an in-library group and an absent group (disjoint)
    half = len(mols) // 2
    in_m, ab_m = set(mols[:half]), set(mols[half:])
    A = a1[a1.mol.isin(in_m)]
    B = b1[b1.mol.isin(ab_m)]
    nA, nB = len(A), len(B)
    if pi <= 0:
        return B.copy()
    if pi >= 1:
        return A.copy()
    # largest batch with the requested prevalence
    if nA / (nA + nB) > pi:
        nA = int(round(pi / (1 - pi) * nB))
    else:
        nB = int(round((1 - pi) / pi * nA))
    if max_size and nA + nB > max_size:
        s = max_size / (nA + nB)
        nA, nB = int(round(nA * s)), int(round(nB * s))
    out = pd.concat([A.iloc[:nA], B.iloc[:nB]])
    return out.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(drop=True)


# ----------------------------------------------------------------------------- tiers
TIERS = ["trusted", "suspicious", "absent"]


def assign_tiers(p_correct, p_in, t_trusted, t_absent):
    p_correct = np.asarray(p_correct)
    p_in = np.asarray(p_in)
    tier = np.full(len(p_correct), "suspicious", dtype=object)
    tier[p_in <= t_absent] = "absent"
    tier[p_correct >= t_trusted] = "trusted"  # p_correct <= p_in, so trusted and absent cannot overlap
    return tier


# ----------------------------------------------------------------------------- wet-lab simulation
def wetlab_once(batch: pd.DataFrame, n_per_tier: int, rng, level=0.95) -> list[dict]:
    out = []
    for t in TIERS:
        sub = batch[batch.tier == t]
        n_avail = len(sub)
        if n_avail == 0:
            out.append({"tier": t, "n_avail": 0, "n": 0})
            continue
        s = sub.sample(n=min(n_per_tier, n_avail), random_state=rng.integers(1 << 31))
        n = len(s)
        k_c = int(s.correct.sum())
        k_in = int(s.in_lib.sum())
        lo, hi = wilson(k_c, n, level)
        lo_in, hi_in = wilson(k_in, n, level)
        pred_c = float(s.p_correct.mean())
        pred_in = float(s.p_in.mean())
        out.append({"tier": t, "n_avail": n_avail, "n": n,
                    "obs_acc": k_c / n, "acc_lo": lo, "acc_hi": hi, "pred_acc": pred_c,
                    "acc_consistent": bool(lo <= pred_c <= hi),
                    "obs_in": k_in / n, "in_lo": lo_in, "in_hi": hi_in, "pred_in": pred_in,
                    "in_consistent": bool(lo_in <= pred_in <= hi_in)})
    return out


def summarize_wetlab(records: list[dict]) -> dict:
    df = pd.DataFrame(records)
    res = {}
    for t, g in df.groupby("tier"):
        g = g[g.n > 0]
        if len(g) == 0:
            res[t] = {"n_repeats_nonempty": 0}
            continue
        res[t] = {
            "n_repeats_nonempty": int(len(g)),
            "frac_repeats_full_n": float((g.n == g.n.max()).mean()),
            "mean_tier_size_in_batch": float(g.n_avail.mean()),
            "obs_acc_mean": float(g.obs_acc.mean()),
            "obs_acc_q025": float(g.obs_acc.quantile(0.025)), "obs_acc_q975": float(g.obs_acc.quantile(0.975)),
            "pred_acc_mean": float(g.pred_acc.mean()),
            "acc_consistent_rate": float(g.acc_consistent.mean()),
            "mean_abs_gap_acc": float((g.obs_acc - g.pred_acc).abs().mean()),
            "obs_in_mean": float(g.obs_in.mean()), "pred_in_mean": float(g.pred_in.mean()),
            "in_consistent_rate": float(g.in_consistent.mean()),
        }
    return res


# ----------------------------------------------------------------------------- selective prediction
def risk_coverage(conf, correct, weights=None, grid=None):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    w = np.ones_like(conf) if weights is None else np.asarray(weights, float)
    o = np.argsort(-conf, kind="stable")
    c, w_ = correct[o], w[o]
    cw = np.cumsum(w_)
    err = np.cumsum(w_ * (1 - c))
    cov = cw / cw[-1]
    risk = err / cw
    # AURC: mean selective risk over coverage (weighted steps)
    aurc = float(np.sum(risk * w_) / cw[-1])
    # oracle AURC for e-AURC
    oo = np.argsort(-c, kind="stable")
    cwo = np.cumsum(w[oo])
    risko = np.cumsum(w[oo] * (1 - correct[oo])) / cwo
    aurc_oracle = float(np.sum(risko * w[oo]) / cwo[-1])
    out = {"aurc": aurc, "e_aurc": aurc - aurc_oracle, "base_risk": float(risk[-1])}
    grid = grid if grid is not None else [0.1, 0.2, 0.3, 0.5, 0.7]
    for g in grid:
        i = min(np.searchsorted(cov, g), len(cov) - 1)
        out[f"risk@cov{g}"] = float(risk[i])
    for r in [0.05, 0.1, 0.2]:
        ok = np.nonzero(risk <= r)[0]
        out[f"cov@risk{r}"] = float(cov[ok.max()]) if len(ok) else 0.0
    return out


def rc_curve(conf, correct, n_points=101):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    o = np.argsort(-conf, kind="stable")
    c = correct[o]
    cov = np.arange(1, len(c) + 1) / len(c)
    risk = np.cumsum(1 - c) / np.arange(1, len(c) + 1)
    g = np.linspace(0.01, 1, n_points)
    idx = np.minimum(np.searchsorted(cov, g), len(c) - 1)
    return g, risk[idx]


# ----------------------------------------------------------------------------- calibration
def calibration_metrics(p, y, weights=None, n_bins=10):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, float)
    W = w.sum()
    brier = float(np.sum(w * (p - y) ** 2) / W)
    ll = float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / W)
    # equal-mass bins
    o = np.argsort(p)
    cw = np.cumsum(w[o]) / W
    b = np.minimum((cw * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            ww = w[o][m]
            ece += ww.sum() / W * abs(np.average(p[o][m], weights=ww) - np.average(y[o][m], weights=ww))
    # calibration-in-the-large and slope (logistic recalibration)
    lg = np.log(p / (1 - p))
    try:
        import statsmodels.api as sm  # optional

        fit = sm.GLM(y, sm.add_constant(lg), family=sm.families.Binomial(), freq_weights=w).fit()
        intercept, slope = float(fit.params[0]), float(fit.params[1])
    except Exception:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6).fit(lg[:, None], y, sample_weight=w) if len(np.unique(y)) > 1 else None
        intercept, slope = (float(lr.intercept_[0]), float(lr.coef_[0, 0])) if lr is not None else (np.nan, np.nan)
    return {"n": int(len(p)), "mean_p": float(np.average(p, weights=w)), "mean_y": float(np.average(y, weights=w)),
            "ece": float(ece), "brier": brier, "logloss": ll, "cal_intercept": intercept, "cal_slope": slope}


def auroc(score, y, weights=None):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, score, sample_weight=weights))


def group_calibration(df: pd.DataFrame, group_col: str, p_col: str, y_col: str, min_n: int, level=0.95):
    """Per-group observed vs predicted with Wilson CIs. Groups below min_n are reported but flagged
    (never claimed to be calibrated — failure mode #7)."""
    rows = []
    for g, sub in df.groupby(group_col):
        n = len(sub)
        k = int(sub[y_col].sum())
        lo, hi = wilson(k, n, level)
        pm = float(sub[p_col].mean())
        rows.append({"group": g, "n": n, "n_mol": int(sub.mol.nunique()) if "mol" in sub else np.nan,
                     "obs": k / n, "lo": lo, "hi": hi, "pred": pm, "consistent": bool(lo <= pm <= hi),
                     "sufficient": bool(n >= min_n)})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ----------------------------------------------------------------------------- label shift
def prior_shift(p, pi_from, pi_to):
    """Bayes prior-shift adjustment of P(in|x) from training prevalence pi_from to pi_to."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    num = p * pi_to / pi_from
    den = num + (1 - p) * (1 - pi_to) / (1 - pi_from)
    return num / den


def em_prevalence(p_in, pi_train, n_iter=200, tol=1e-7):
    """Saerens et al. (2002) EM estimate of the in-library prevalence of an unlabeled batch."""
    pi = pi_train
    for _ in range(n_iter):
        new = float(np.mean(prior_shift(p_in, pi_train, pi)))
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi


# ----------------------------------------------------------------------------- bootstrap
def cluster_bootstrap(df: pd.DataFrame, fn, n_boot: int, rng, cluster_col="mol"):
    """Bootstrap over molecules (spectra of one molecule are correlated)."""
    groups = df.groupby(cluster_col).indices
    keys = np.array(list(groups.keys()))
    idx_lists = [groups[k] for k in keys]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        ix = np.concatenate([idx_lists[i] for i in pick])
        vals.append(fn(df.iloc[ix]))
    return np.array(vals)
