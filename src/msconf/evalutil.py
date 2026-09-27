"""Shared evaluation utilities (weighted prevalence evaluation of frozen systems)."""
from __future__ import annotations

import numpy as np

from .evaluation import (assign_tiers, auroc, calibration_metrics, prior_shift, risk_coverage)
from .models import DirectSystem

T_TR = T_AB = None


def set_thresholds(t_tr, t_ab):
    global T_TR, T_AB
    T_TR, T_AB = t_tr, t_ab


def components(sysm, df):
    if isinstance(sysm, DirectSystem):
        p = sysm.predict(df)[0]
        return {"direct": True, "pc": p, "pin_cal": p, "pi_cal": None}
    return {"direct": False, "pc": sysm.p_correct_given_in(df), "pin_cal": sysm.p_in_cal(df), "pi_cal": sysm.pi_cal}


def probs(comp, pi):
    """p_correct, p_in at prevalence pi (None = no label-shift correction)."""
    if comp["direct"]:
        return comp["pc"], comp["pin_cal"]
    pin = comp["pin_cal"] if pi is None else prior_shift(comp["pin_cal"], comp["pi_cal"], pi)
    return np.clip(pin * comp["pc"], 0, 1), pin


def class_weights(in_lib, w, pi):
    """Importance weights giving in-library prevalence pi (exactly) within the given rows."""
    wi = w * in_lib
    wa = w * (1 - in_lib)
    return wi * (pi / wi.sum()) + wa * ((1 - pi) / wa.sum())


def em_w(pin_cal, pi_cal, w, n_iter=300):
    pi = pi_cal
    for _ in range(n_iter):
        new = float(np.sum(w * prior_shift(pin_cal, pi_cal, pi)) / np.sum(w))
        if abs(new - pi) < 1e-8:
            return new
        pi = new
    return pi


def tier_summary(pc, pin, correct, in_lib, w):
    t = assign_tiers(pc, pin, T_TR, T_AB)
    out = {}
    for tt in ["trusted", "suspicious", "absent"]:
        m = t == tt
        if not m.any():
            out[tt] = {"frac": 0.0}
            continue
        out[tt] = {"frac": float(w[m].sum() / w.sum()), "acc": float(np.average(correct[m], weights=w[m])),
                   "pred_acc": float(np.average(pc[m], weights=w[m])), "in_rate": float(np.average(in_lib[m], weights=w[m])),
                   "pred_in": float(np.average(pin[m], weights=w[m]))}
    return out


def evaluate_weighted(comp, df, pi, mode):
    y, yin = df.correct.values.astype(float), df.in_lib.values.astype(float)
    w = class_weights(yin, df.w.values, pi)
    if mode == "oracle":
        pi_use = pi
    elif mode == "em":
        pi_use = None if comp["direct"] else em_w(comp["pin_cal"], comp["pi_cal"], w)
    else:
        pi_use = None
    pc, pin = probs(comp, pi_use)
    r = {"pi_used": pi_use, **risk_coverage(pc, y, w), **{f"cal_{k}": v for k, v in calibration_metrics(pc, y, w).items()},
         "auroc_correct": auroc(pc, y, w), "auroc_in": auroc(pin, yin, w),
         "cal_in": calibration_metrics(pin, yin, w), "tiers": tier_summary(pc, pin, y, yin, w)}
    return r


