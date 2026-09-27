"""Every baseline under the SAME EM prior correction as the proposed system.

Added after the main test evaluation (owner's request); nothing is re-fitted and the proposed method is unchanged.
In the main tables a "direct" system (one calibrated P(correct), no absence model) is used as practitioners use it,
i.e. without any prevalence correction. Here every direct system additionally gets the Saerens et al. (2002) EM
prior correction, applied to its P(correct): the batch prior of "correct" is estimated by EM on the unlabeled batch
and the probabilities are moved there by Bayes' rule, starting from the calibration prior (weighted fraction of
correct top-1 annotations in the validation (a)+(b) rows the calibrator was fitted on). Decomposed systems keep
their usual EM on P(in). The correction is monotone within a batch, so ranking metrics (AURC, e-AURC, AUROC) of a
direct system do not change; calibration, log loss and tiers do.
Test fold, (a)+(b), prevalence sweep; wet-lab simulation (40 per tier, per-batch EM) at pi = 0.1, 0.2, 0.5 for the
main baselines. External set (v3) and scenario a_x are included for the direct systems that exist there.
Output: results/eval_baselines_em.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf import evalutil  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evalutil import class_weights, components, em_w, probs, tier_summary  # noqa: E402
from msconf.evaluation import (assign_tiers, auroc, calibration_metrics, em_prevalence, prior_shift,  # noqa: E402
                               risk_coverage, summarize_wetlab, wilson)
from msconf.pipeline import load_rows, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
ev = cfg["evaluation"]
seed = cfg["project"]["seed"]
T_TR, T_AB = cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"]
evalutil.set_thresholds(T_TR, T_AB)
PRIMARY = json.load(open(path(cfg, "results") / "selection.json"))["primary_system"]
t0 = time.time()


def em_direct(p, prior_cal, w, n_iter=500):
    pi = prior_cal
    for _ in range(n_iter):
        new = float(np.sum(w * prior_shift(p, prior_cal, pi)) / np.sum(w))
        if abs(new - pi) < 1e-9:
            pi = new
            break
        pi = new
    return prior_shift(p, prior_cal, pi), pi


def evaluate(comp, df, pi, mode, prior_cal=None, label="correct"):
    y, yin = df[label].values.astype(float), df.in_lib.values.astype(float)
    w = class_weights(yin, df.w.values, pi)
    if comp["direct"]:
        pc = comp["pc"]
        est = None
        if mode == "em":
            pc, est = em_direct(pc, prior_cal, w)
        pin = pc
    else:
        pu = em_w(comp["pin_cal"], comp["pi_cal"], w) if mode == "em" else None
        pc, pin = probs(comp, pu)
        est = pu
    c = calibration_metrics(pc, y, w)
    rc = risk_coverage(pc, y, w)
    return {"estimate": est, "aurc": rc["aurc"], "e_aurc": rc["e_aurc"], "cov@risk0.05": rc["cov@risk0.05"],
            "auroc": auroc(pc, y, w), "logloss": c["logloss"], "ece": c["ece"], "mean_p": c["mean_p"], "mean_y": c["mean_y"],
            "tiers": tier_summary(pc, pin, y, yin, w)}


# ---------------------------------------------------------------- main systems (models/entropy)
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
rows = load_rows(der, "entropy")
rows = rows[rows.scenario.isin(["a", "b"])]
val = rows[rows.fold == "val"]
test = rows[rows.fold == "test"].reset_index(drop=True)
PRIOR_MAIN = float(np.average(val.correct, weights=val.w))  # calibration prior of "correct" (val (a)+(b), weights 1/n)
S = {f.stem: pickle.loads(f.read_bytes()) for f in sorted((der / "models" / "entropy").glob("*.pkl"))}
COMP = {n: components(s, test) for n, s in S.items()}
res = {"prior_correct_calibration_main": PRIOR_MAIN, "sweep": {}, "added_after_main_test": True}
for n, c in COMP.items():
    res["sweep"][n] = {"direct": c["direct"]}
    for pi in ev["prevalence_grid"]:
        res["sweep"][n][str(pi)] = {m: evaluate(c, test, pi, m, PRIOR_MAIN) for m in ["em", "none"]}
print("main sweep", f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- baselines added after the main test (MS2Query-style, TDA)
NB = pd.read_parquet(der / "rows_new_baselines.parquet")
nbv, nbt = NB[NB.fold == "val"], NB[NB.fold == "test"].reset_index(drop=True)
for f in sorted((der / "models" / "entropy_new_baselines").glob("*.pkl")):
    s = pickle.loads(f.read_bytes())
    lab = "correct_own" if f.stem == "B_ms2q_rerank_direct" else "correct"
    c = components(s, nbt)
    prior = float(np.average(nbv[lab], weights=nbv.w))
    res["sweep"][f.stem] = {"direct": c["direct"], "prior_correct_calibration": prior, "label": lab}
    for pi in ev["prevalence_grid"]:
        res["sweep"][f.stem][str(pi)] = {m: evaluate(c, nbt, pi, m, prior, lab) for m in ["em", "none"]}
print("new baselines sweep", f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- wet-lab simulation with per-batch EM for direct systems
a_pos = test.index[test.scenario == "a"].values
b_pos = test.index[test.scenario == "b"].values
a_by = {k: a_pos[v] for k, v in test.loc[a_pos].groupby("mol").indices.items()}
b_by = {k: b_pos[v] for k, v in test.loc[b_pos].groupby("mol").indices.items()}
ALLM = np.array(sorted(set(a_by) | set(b_by)))
Y, YIN = test.correct.values, test.in_lib.values


def batch(pi, rng):  # identical construction to scripts/09_evaluate.py
    m = ALLM.copy()
    rng.shuffle(m)
    half = len(m) // 2
    A = [a_by[k][rng.integers(len(a_by[k]))] for k in m[:half] if k in a_by]
    B = [b_by[k][rng.integers(len(b_by[k]))] for k in m[half:] if k in b_by]
    nA, nB = len(A), len(B)
    if nA / (nA + nB) > pi:
        nA = int(round(pi / (1 - pi) * nB))
    else:
        nB = int(round((1 - pi) / pi * nA))
    return np.array(A[:nA] + B[:nB])


def wetlab(name, mode, pi, repeats, rng):
    comp = COMP[name]
    recs = []
    for _ in range(repeats):
        ix = batch(pi, rng)
        if comp["direct"]:
            pc = comp["pc"][ix]
            if mode == "em":
                pc, _ = em_direct(pc, PRIOR_MAIN, np.ones(len(ix)))
            pin = pc
        else:
            sub = {"direct": False, "pc": comp["pc"][ix], "pin_cal": comp["pin_cal"][ix], "pi_cal": comp["pi_cal"]}
            pc, pin = probs(sub, em_prevalence(sub["pin_cal"], comp["pi_cal"]) if mode == "em" else None)
        tier = assign_tiers(pc, pin, T_TR, T_AB)
        for tt in ["trusted", "suspicious", "absent"]:
            cand = np.nonzero(tier == tt)[0]
            if len(cand) == 0:
                recs.append({"tier": tt, "n_avail": 0, "n": 0})
                continue
            take = rng.choice(cand, size=min(ev["wetlab_per_tier"], len(cand)), replace=False)
            n = len(take)
            kc, ki = int(Y[ix][take].sum()), int(YIN[ix][take].sum())
            lo, hi = wilson(kc, n)
            li, hi_i = wilson(ki, n)
            prc, pri = float(pc[take].mean()), float(pin[take].mean())
            recs.append({"tier": tt, "n_avail": len(cand), "n": n, "obs_acc": kc / n, "acc_lo": lo, "acc_hi": hi, "pred_acc": prc,
                         "acc_consistent": lo <= prc <= hi, "obs_in": ki / n, "in_lo": li, "in_hi": hi_i, "pred_in": pri,
                         "in_consistent": li <= pri <= hi_i, "batch_size": len(ix)})
    return summarize_wetlab(recs)


WL = [(PRIMARY, "em"), ("B_lr_dec", "em"), ("B_lr_direct", "em"), ("B_lr_direct", "none"), ("M_full_direct", "em"), ("M_full_direct", "none"),
      ("B_s1_direct", "em"), ("B_gap_direct", "em")]
res["wetlab"] = {}
for pi in [0.1, 0.2, 0.5]:
    res["wetlab"][str(pi)] = {f"{s}|{m}": wetlab(s, m, pi, ev["wetlab_repeats"], np.random.default_rng(seed + int(pi * 100))) for s, m in WL if s in COMP}
print("wetlab", f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- group calibration under the same EM correction (as in 09_evaluate)
CROWD = ["top1_struct_nbrs", "fam_spec_crowd_train", "n_open_hi", "top1_specdens", "n_win_mol"]
vB = val[val.mol.map(split) == "B"]
edges = {c: np.unique(np.nanquantile(vB[c].values, np.linspace(0, 1, ev["crowding_bins"] + 1)[1:-1])) for c in CROWD}
for c in CROWD:
    test[f"bin_{c}"] = np.searchsorted(edges[c], test[c].values, side="right")


def group_summary(p, y, w, col):
    rs = []
    for g, ix in test.groupby(col).indices.items():
        ww = w[ix]
        neff = ww.sum() ** 2 / np.sum(ww ** 2)
        obs, pred = float(np.average(y[ix], weights=ww)), float(np.average(p[ix], weights=ww))
        lo, hi = wilson(obs * neff, neff)
        nm = len(np.unique(test.mol.values[ix]))
        rs.append({"obs": obs, "pred": pred, "consistent": lo <= pred <= hi,
                   "sufficient": neff >= ev["min_group_n_test"] and nm >= ev["min_group_mol_test"]})
    g = pd.DataFrame(rs)
    sf = g[g.sufficient]
    return {"n_groups": int(len(g)), "n_sufficient": int(len(sf)),
            "frac_consistent_sufficient": float(sf.consistent.mean()) if len(sf) else None,
            "mean_abs_gap_sufficient": float((sf.obs - sf.pred).abs().mean()) if len(sf) else None,
            "max_abs_gap_sufficient": float((sf.obs - sf.pred).abs().max()) if len(sf) else None}


res["groups_em"] = {}
for n in [PRIMARY, "B_lr_dec", "B_lr_direct", "M_full_direct", "B_s1_direct", "B_gap_direct"]:
    if n not in COMP:
        continue
    c = COMP[n]
    res["groups_em"][n] = {}
    for pi in ev["prevalence_headline"]:
        y, yin = test.correct.values.astype(float), test.in_lib.values.astype(float)
        w = class_weights(yin, test.w.values, pi)
        if c["direct"]:
            pc, _ = em_direct(c["pc"], PRIOR_MAIN, w)
        else:
            pc, _ = probs(c, em_w(c["pin_cal"], c["pi_cal"], w))
        res["groups_em"][n][str(pi)] = {g: group_summary(pc, y, w, g) for g in ["top1_class", "top1_superclass"] + [f"bin_{x}" for x in CROWD]}
print("groups", f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- external set v3 and scenario a_x (direct systems)
extra = {}
for tag in ["external_v3", "a_x"]:
    if tag == "external_v3":
        f = der / "ext_v3" / "rows_entropy.parquet"
        if not f.exists():
            extra[tag] = {"error": "rows not found"}
            continue
        d = pd.read_parquet(f)
        d = d[(d.fold == "ext") & (d.has_candidate == 1) & d.scenario.isin(["a", "b"])].reset_index(drop=True)
    else:  # a_x rows paired with the same queries' (b) rows, exactly as scripts/16 and 21
        ax = pd.read_parquet(der / "rows_ax_entropy.parquet").assign(scenario="a")
        d = pd.concat([ax, test[(test.scenario == "b") & test.qid.isin(ax.qid)]], ignore_index=True)
    extra[tag] = {}
    for n in [PRIMARY, "B_lr_dec", "B_lr_direct", "M_full_direct", "B_s1_direct"]:
        if n not in S:
            continue
        c = components(S[n], d)
        extra[tag][n] = {str(pi): {m: evaluate(c, d, pi, m, PRIOR_MAIN) for m in ["em", "none"]} for pi in ev["prevalence_headline"]}
res["shifted"] = extra
res["seconds"] = time.time() - t0
write_result(cfg, "eval_baselines_em", res)
for pi in ["0.1", "0.2", "0.5"]:
    print(pi, {n: (round(v[pi]["none"]["logloss"], 4), round(v[pi]["em"]["logloss"], 4), round(v[pi]["em"]["e_aurc"], 4)) for n, v in res["sweep"].items()})
