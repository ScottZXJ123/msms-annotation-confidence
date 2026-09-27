"""Training / calibration pipeline shared by the training, shift-test and evaluation scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .evaluation import calibration_metrics
from .models import (ConfidenceSystem, DirectSystem, GroupOffset, Isotonic, ScoreModel, TrainedModel, columns,
                     fit_lgbm, fit_lr)


def load_rows(der, engine="entropy"):
    df = pd.read_parquet(der / f"rows_{engine}.parquet")
    df = df[df.has_candidate == 1].copy()  # study population (§4): >=1 candidate in the precursor window
    return df


def val_split(fam_or_mols: pd.DataFrame, struct_nbrs: dict, thr: float, frac_calib: float, seed: int) -> dict:
    """Split val-fold molecules into model-selection (A) and calibration (B) halves by connected
    components of the Tanimoto >= thr graph (structurally close molecules stay on the same side)."""
    fold = fam_or_mols.fold.values
    n = len(fold)
    val = np.nonzero(fold == "val")[0]
    isval = np.zeros(n, bool)
    isval[val] = True
    r, c = [], []
    for i in val:
        a, b = struct_nbrs["indptr"][i], struct_nbrs["indptr"][i + 1]
        j = struct_nbrs["indices"][a:b]
        s = struct_nbrs["sims"][a:b]
        j = j[(s >= thr) & isval[j]]
        r += [i] * len(j)
        c += list(j)
    G = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    _, lab = connected_components(G, directed=False)
    comps = np.unique(lab[val])
    rng = np.random.default_rng(seed)
    rng.shuffle(comps)
    sizes = pd.Series(lab[val]).value_counts()
    tot, acc, B = len(val), 0, set()
    for cpt in comps:
        if acc >= frac_calib * tot:
            break
        B.add(cpt)
        acc += sizes[cpt]
    return {int(m): ("B" if lab[m] in B else "A") for m in val}


def _w(df):
    return df.w.values


def fit_component(kind, cols, tr, ytr, va, yva, budget, seed):
    if kind == "lgbm":
        est, log = fit_lgbm(tr[cols].values.astype(float), ytr, _w(tr), va[cols].values.astype(float), yva, _w(va), budget, seed)
        return TrainedModel(est, cols, "lgbm"), log
    if kind == "lr":
        est, log = fit_lr(tr[cols].values.astype(float), ytr, _w(tr), va[cols].values.astype(float), yva, _w(va), budget, seed)
        return TrainedModel(est, cols, "lr"), log
    if kind == "score":
        assert len(cols) == 1
        return ScoreModel(cols[0]), {"best": None, "trials": []}
    raise ValueError(kind)


def train_decomposed(name, kind, cols, train, valA, valB, cfg, seed, group_col=None, min_group=None):
    """P(correct|x,in) on scenario-(a) rows; P(in|x) on (a)+(b) rows; isotonic calibration on val-B;
    optional per-family logit offsets (shrunk, min group size) on val-B."""
    b = cfg["models"]["lgbm_budget"] if kind == "lgbm" else cfg["models"]["lr_budget"]
    trA, vaA, vbA = (d[d.scenario == "a"] for d in (train, valA, valB))
    trAB, vaAB, vbAB = (d[d.scenario.isin(["a", "b"])] for d in (train, valA, valB))
    mC, logC = fit_component(kind, cols, trA, trA.correct.values, vaA, vaA.correct.values, b, seed)
    mI, logI = fit_component(kind, cols, trAB, trAB.in_lib.values, vaAB, vaAB.in_lib.values, b, seed + 1)
    calC = Isotonic().fit(mC.raw(vbA), vbA.correct.values, _w(vbA))
    calI = Isotonic().fit(mI.raw(vbAB), vbAB.in_lib.values, _w(vbAB))
    pi_cal = float(np.sum(_w(vbAB) * vbAB.in_lib.values) / np.sum(_w(vbAB)))
    pi_train = float(np.sum(_w(trAB) * trAB.in_lib.values) / np.sum(_w(trAB)))
    gC = gI = None
    if group_col is not None:
        gC = GroupOffset(min_group).fit(calC(mC.raw(vbA)), vbA.correct.values, _w(vbA), vbA[group_col].values)
        gI = GroupOffset(min_group).fit(calI(mI.raw(vbAB)), vbAB.in_lib.values, _w(vbAB), vbAB[group_col].values)
    sysm = ConfidenceSystem(name, mC, mI, calC, calI, pi_cal, gC, gI, group_col)
    log = {"name": name, "kind": kind, "cols": cols, "pi_train": pi_train, "pi_cal": pi_cal,
           "tuning_C": logC, "tuning_I": logI,
           "n_train_rows": {"C": len(trA), "I": len(trAB)}, "n_valB_rows": {"C": len(vbA), "I": len(vbAB)},
           "group_offsets": {"C": gC.off if gC else None, "I": gI.off if gI else None}}
    return sysm, log


def train_direct(name, kind, cols, train, valA, valB, cfg, seed):
    """Single P(correct|x) fitted on the (a)+(b) mixture, no absence model, no label-shift correction."""
    b = cfg["models"]["lgbm_budget"] if kind == "lgbm" else cfg["models"]["lr_budget"]
    trAB, vaAB, vbAB = (d[d.scenario.isin(["a", "b"])] for d in (train, valA, valB))
    m, log = fit_component(kind, cols, trAB, trAB.correct.values, vaAB, vaAB.correct.values, b, seed)
    cal = Isotonic().fit(m.raw(vbAB), vbAB.correct.values, _w(vbAB))
    return DirectSystem(name, m, cal), {"name": name, "kind": kind, "cols": cols, "tuning": log, "direct": True}


SYSTEMS = [
    # name, builder, kind, feature set / column
    ("B_s1_direct", "direct", "score", ["s1"]),
    ("B_gap_direct", "direct", "score", ["gap12"]),
    ("B_lr_direct", "direct", "lr", ["s1", "gap12"]),
    ("B_s1_dec", "dec", "score", ["s1"]),
    ("B_gap_dec", "dec", "score", ["gap12"]),
    ("B_lr_dec", "dec", "lr", ["s1", "gap12"]),
    ("F_meta", "dec", "lgbm", "meta_only"),
    ("F_sim", "dec", "lgbm", "sim_only"),
    ("M_base", "dec", "lgbm", "base"),
    ("M_full", "dec", "lgbm", "full"),
    ("M_full_nocounts", "dec", "lgbm", "full_nocounts"),
    ("M_full_with_libsize", "dec", "lgbm", "full_with_libsize"),
    ("M_sim_plus_crowd", "dec", "lgbm", "sim_plus_crowd"),
    ("M_full_minus_family", "dec", "lgbm", "full_minus_family"),
    ("M_full_minus_spec", "dec", "lgbm", "full_minus_spec"),
    ("M_full_minus_struct", "dec", "lgbm", "full_minus_struct"),
    ("M_candidate_only", "dec", "lgbm", "candidate_only"),
    ("M_full_direct", "direct", "lgbm", "full"),
    ("M_full_lr", "dec", "lr", "full"),
]


def resolve_cols(spec):
    return spec if isinstance(spec, list) else columns(spec)


def val_diagnostics(sysm, valB, pis=(0.1, 0.2, 0.5)):
    """Validation-only diagnostics (used for selecting the primary configuration)."""
    out = {}
    vb = valB[valB.scenario.isin(["a", "b"])]
    for pi in [None, *pis]:
        pc, pin = sysm.predict(vb, pi)
        out[f"pi={pi}"] = {"p_correct_vs_correct": calibration_metrics(pc, vb.correct.values, vb.w.values)}
    return out
