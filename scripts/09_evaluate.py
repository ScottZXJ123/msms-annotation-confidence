"""FINAL evaluation on the test fold (run once, after all selections were frozen on validation).

Reads: frozen models (data/derived/models/<engine>/), results/selection.json (primary system + thresholds).
Writes: results/eval_*.json and reports/tables/*.csv
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import ROOT, load_config, path, write_result  # noqa: E402
from msconf.evaluation import (assign_tiers, auroc, calibration_metrics, prior_shift, risk_coverage,  # noqa: E402
                               summarize_wetlab, wilson)
from msconf.models import DirectSystem  # noqa: E402
from msconf import evalutil  # noqa: E402
from msconf.evalutil import class_weights, components, em_w, evaluate_weighted, probs, tier_summary  # noqa: E402
from msconf.pipeline import SYSTEMS, load_rows, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
tables = path(cfg, "tables")
ev = cfg["evaluation"]
seed = cfg["project"]["seed"]
sel = json.load(open(path(cfg, "results") / "selection.json"))
PRIMARY = sel["primary_system"]
T_TR, T_AB = cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"]
evalutil.set_thresholds(T_TR, T_AB)
HEAD = ev["prevalence_headline"]
GRID = ev["prevalence_grid"]
t0 = time.time()


# ----------------------------------------------------------------------------- helpers
def load_systems(engine):
    out = {}
    for f in sorted((der / "models" / engine).glob("*.pkl")):
        out[f.stem] = pickle.loads(f.read_bytes())
    return out


# ----------------------------------------------------------------------------- data
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"],
                  cfg["models"]["val_split_frac_calib"], seed)
R = {}
for engine in ["entropy", "cosine"]:
    rows = load_rows(der, engine)
    rows["vsplit"] = rows.mol.map(split)
    R[engine] = rows
rows = R["entropy"]
import os  # noqa: E402

DRY = os.environ.get("MSCONF_DRYRUN_VALA") == "1"  # debug the harness on val-A without touching the test fold
if os.environ.get("MSCONF_FAST") == "1":
    ev = dict(ev, bootstrap_repeats=30, wetlab_repeats=30)
if DRY:
    test = rows[(rows.fold == "val") & (rows.vsplit == "A")].reset_index(drop=True)
    _wr = write_result
    tables = path(cfg, "tables") / "dryrun_valA"
    tables.mkdir(parents=True, exist_ok=True)

    def write_result(cfg, name, payload, **k):  # noqa: F811
        return _wr(cfg, "dryrun_valA_" + name, payload, **k)
else:
    test = rows[rows.fold == "test"].reset_index(drop=True)
valB = rows[(rows.fold == "val") & (rows.vsplit == "B")]
testAB = test[test.scenario.isin(["a", "b"])].reset_index(drop=True)
testA = test[test.scenario == "a"].reset_index(drop=True)
allrows_test = pd.read_parquet(der / "rows_entropy.parquet")
allrows_test = allrows_test[allrows_test.fold == "test"] if not DRY else allrows_test[(allrows_test.fold == "val") & (allrows_test.mol.map(split) == "A")]
population = {sc: {"n_rows": int((allrows_test.scenario == sc).sum()),
                   "frac_no_candidate": float(1 - allrows_test[allrows_test.scenario == sc].has_candidate.mean()),
                   "n_molecules": int(allrows_test[allrows_test.scenario == sc].mol.nunique())} for sc in ["a", "b", "c"]}
pi_test_native = float(np.average(testAB.in_lib, weights=testAB.w))
S = load_systems("entropy")
COMP = {k: components(v, testAB) for k, v in S.items()}
print("loaded", f"{time.time()-t0:.0f}s", flush=True)

# ============================================================================= 1. prevalence sweep, all systems
sweep = {}
for name, comp in COMP.items():
    sweep[name] = {}
    for pi in GRID:
        sweep[name][str(pi)] = {mode: evaluate_weighted(comp, testAB, pi, mode) for mode in ["oracle", "em", "none"]}
write_result(cfg, "eval_sweep", {"primary": PRIMARY, "population_test": population, "pi_test_native": pi_test_native,
                                 "thresholds": {"trusted": T_TR, "absent": T_AB}, "sweep": sweep})
print("sweep done", f"{time.time()-t0:.0f}s", flush=True)

# ============================================================================= 2. bootstrap CIs for key comparisons
rng = np.random.default_rng(seed)
mol_idx = testAB.groupby("mol").indices
mkeys = np.array(list(mol_idx.keys()))
COMPARE = [c for c in ["B_s1_direct", "B_gap_direct", "B_lr_direct", "B_s1_dec", "B_gap_dec", "B_lr_dec", "F_meta", "F_sim",
                       "M_base", "M_full", "M_full_nocounts", "M_sim_plus_crowd", "M_full_minus_family", "M_full_minus_spec",
                       "M_full_minus_struct", "M_candidate_only", "M_full_direct", "M_full_lr"] if c in COMP and c != PRIMARY]
PAIRS = [(PRIMARY, c) for c in COMPARE] + [("M_full", "M_base"), ("M_sim_plus_crowd", "F_sim"), ("M_full_with_libsize", PRIMARY)]
PAIRS = [p for p in PAIRS if p[0] in COMP and p[1] in COMP and p[0] != p[1]]


def fast_ll(p, y, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def point_metrics(ix, pi, names):
    d = testAB.iloc[ix]
    y, yin = d.correct.values.astype(float), d.in_lib.values.astype(float)
    if yin.sum() == 0 or yin.sum() == len(yin):
        return None
    w = class_weights(yin, d.w.values, pi)
    out = {}
    for n in names:
        c = COMP[n]
        sub = {"direct": c["direct"], "pc": c["pc"][ix], "pin_cal": c["pin_cal"][ix], "pi_cal": c["pi_cal"]}
        pc, pin = probs(sub, pi)
        out[n] = (risk_coverage(pc, y, w)["aurc"], fast_ll(pc, y, w))
    return out


boot = {}
names = sorted({n for p in PAIRS for n in p})
for pi in HEAD:
    base = point_metrics(np.arange(len(testAB)), pi, names)
    draws = []
    for b in range(ev["bootstrap_repeats"]):
        pick = rng.integers(0, len(mkeys), len(mkeys))
        ix = np.concatenate([mol_idx[mkeys[i]] for i in pick])
        m = point_metrics(ix, pi, names)
        if m is not None:
            draws.append(m)
    res = {}
    for a, c in PAIRS:
        da = np.array([d[a][0] - d[c][0] for d in draws])
        dl = np.array([d[a][1] - d[c][1] for d in draws])
        res[f"{a} - {c}"] = {"d_aurc": base[a][0] - base[c][0], "d_aurc_ci": np.percentile(da, [2.5, 97.5]).tolist(),
                             "d_logloss": base[a][1] - base[c][1], "d_logloss_ci": np.percentile(dl, [2.5, 97.5]).tolist(),
                             "p_aurc_not_better": float(np.mean(da >= 0)), "p_logloss_not_better": float(np.mean(dl >= 0))}
    boot[str(pi)] = {"point": {n: {"aurc": base[n][0], "logloss": base[n][1]} for n in names}, "diffs": res,
                     "n_boot": len(draws)}
write_result(cfg, "eval_bootstrap", {"primary": PRIMARY, "mode": "oracle prevalence", "boot": boot})
print("bootstrap done", f"{time.time()-t0:.0f}s", flush=True)

# ============================================================================= 3. wet-lab simulation
def _pos_by_mol(mask):
    pos = np.nonzero(mask)[0]  # positions in testAB (not in the filtered frame)
    return {k: pos[ix] for k, ix in testAB.iloc[pos].groupby("mol").indices.items()}


a_by_mol = _pos_by_mol(testAB.scenario.values == "a")
b_by_mol = _pos_by_mol(testAB.scenario.values == "b")
ALLM = np.array(sorted(set(a_by_mol) | set(b_by_mol)))
Y = testAB.correct.values
YIN = testAB.in_lib.values


def batch(pi, rng):
    m = ALLM.copy()
    rng.shuffle(m)
    half = len(m) // 2
    A = [a_by_mol[k][rng.integers(len(a_by_mol[k]))] for k in m[:half] if k in a_by_mol]
    B = [b_by_mol[k][rng.integers(len(b_by_mol[k]))] for k in m[half:] if k in b_by_mol]
    nA, nB = len(A), len(B)
    if nA / (nA + nB) > pi:
        nA = int(round(pi / (1 - pi) * nB))
    else:
        nB = int(round((1 - pi) / pi * nA))
    return np.array(A[:nA] + B[:nB])


def wetlab(system, mode, pi, repeats, rng):
    comp = COMP[system]
    recs, pis = [], []
    for _ in range(repeats):
        ix = batch(pi, rng)
        sub = {"direct": comp["direct"], "pc": comp["pc"][ix], "pin_cal": comp["pin_cal"][ix], "pi_cal": comp["pi_cal"]}
        if mode == "em" and not comp["direct"]:
            from msconf.evaluation import em_prevalence
            pu = em_prevalence(sub["pin_cal"], comp["pi_cal"])
        elif mode == "oracle":
            pu = pi
        else:
            pu = None
        pis.append(pu if pu is not None else np.nan)
        pc, pin = probs(sub, pu)
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
            recs.append({"tier": tt, "n_avail": len(cand), "n": n, "obs_acc": kc / n, "acc_lo": lo, "acc_hi": hi,
                         "pred_acc": prc, "acc_consistent": lo <= prc <= hi, "obs_in": ki / n, "in_lo": li, "in_hi": hi_i,
                         "pred_in": pri, "in_consistent": li <= pri <= hi_i, "batch_size": len(ix)})
    s = summarize_wetlab(recs)
    s["_pi_used"] = {"mean": float(np.nanmean(pis)) if not np.all(np.isnan(pis)) else None,
                     "q025": float(np.nanpercentile(pis, 2.5)) if not np.all(np.isnan(pis)) else None,
                     "q975": float(np.nanpercentile(pis, 97.5)) if not np.all(np.isnan(pis)) else None}
    s["_batch_size_mean"] = float(np.mean([r.get("batch_size", np.nan) for r in recs if r.get("n", 0) > 0]))
    return s


wl = {}
wl_systems = [(PRIMARY, "em"), (PRIMARY, "oracle"), (PRIMARY, "none"), ("B_lr_dec", "em"), ("M_full_direct", "none"),
              ("B_lr_direct", "none")]
for pi in [0.1, 0.2, 0.3, 0.5, 0.7]:
    wl[str(pi)] = {}
    for s, mode in wl_systems:
        if s in COMP:
            wl[str(pi)][f"{s}|{mode}"] = wetlab(s, mode, pi, ev["wetlab_repeats"], np.random.default_rng(seed + int(pi * 100)))
write_result(cfg, "eval_wetlab", {"primary": PRIMARY, "per_tier": ev["wetlab_per_tier"], "repeats": ev["wetlab_repeats"],
                                  "thresholds": {"trusted": T_TR, "absent": T_AB}, "wetlab": wl})
print("wetlab done", f"{time.time()-t0:.0f}s", flush=True)

# ============================================================================= 4. calibration by family / crowding group
CROWD = ["top1_struct_nbrs", "fam_spec_crowd_train", "n_open_hi", "top1_specdens", "n_win_mol"]
vbAB = valB[valB.scenario.isin(["a", "b"])]
edges = {c: np.unique(np.nanquantile(vbAB[c].values, np.linspace(0, 1, ev["crowding_bins"] + 1)[1:-1])) for c in CROWD}
for c in CROWD:
    testAB[f"bin_{c}"] = np.searchsorted(edges[c], testAB[c].values, side="right")
    testA[f"bin_{c}"] = np.searchsorted(edges[c], testA[c].values, side="right")


def group_table(df, p, y, w, col):
    rows_ = []
    for g, ix in df.groupby(col).indices.items():
        ww = w[ix]
        neff = ww.sum() ** 2 / np.sum(ww ** 2)
        obs = float(np.average(y[ix], weights=ww))
        pred = float(np.average(p[ix], weights=ww))
        lo, hi = wilson(obs * neff, neff)
        nm = int(df.mol.values[ix].size and len(np.unique(df.mol.values[ix])))
        rows_.append({"group": g, "n_rows": len(ix), "n_eff": float(neff), "n_mol": nm, "obs": obs, "lo": lo, "hi": hi,
                      "pred": pred, "consistent": bool(lo <= pred <= hi),
                      "sufficient": bool(neff >= ev["min_group_n_test"] and nm >= ev["min_group_mol_test"])})
    return pd.DataFrame(rows_)


def summarize_groups(gt):
    s = gt[gt.sufficient]
    return {"n_groups": int(len(gt)), "n_sufficient": int(len(s)),
            "frac_rows_in_sufficient": float(gt[gt.sufficient].n_rows.sum() / max(gt.n_rows.sum(), 1)),
            "frac_consistent_sufficient": float(s.consistent.mean()) if len(s) else None,
            "mean_abs_gap_sufficient": float((s.obs - s.pred).abs().mean()) if len(s) else None,
            "max_abs_gap_sufficient": float((s.obs - s.pred).abs().max()) if len(s) else None}


grp = {}
GROUPINGS = ["top1_class", "top1_superclass", "top1_subclass"] + [f"bin_{c}" for c in CROWD] + ["ana_true_class"]
for sname in [PRIMARY, "M_full", "M_base", "B_lr_dec", "B_lr_direct", "M_full_direct"]:
    if sname not in COMP or sname in grp:
        continue
    comp = COMP[sname]
    grp[sname] = {}
    for pi in HEAD:
        w = class_weights(testAB.in_lib.values.astype(float), testAB.w.values, pi)
        pc, pin = probs(comp, pi)
        grp[sname][str(pi)] = {}
        for gcol in GROUPINGS:
            gt = group_table(testAB, pc, testAB.correct.values.astype(float), w, gcol)
            grp[sname][str(pi)][gcol] = summarize_groups(gt)
            if sname == PRIMARY and gcol in ("top1_class", "ana_true_class", "bin_fam_spec_crowd_train", "bin_top1_struct_nbrs"):
                gt.sort_values("n_eff", ascending=False).to_csv(tables / f"groups_{gcol}_pi{pi}.csv", index=False)
    # component-level: P(correct | in) on (a) rows (prevalence-free), P(in) at pi=0.5
    if not comp["direct"]:
        pcA = S[sname].p_correct_given_in(testA)
        grp[sname]["component_C"] = {g: summarize_groups(group_table(testA, pcA, testA.correct.values.astype(float), testA.w.values, g))
                                     for g in ["top1_class", "top1_superclass", "top1_subclass", "bin_fam_spec_crowd_train",
                                               "bin_top1_struct_nbrs"]}
        w = class_weights(testAB.in_lib.values.astype(float), testAB.w.values, 0.5)
        _, pin = probs(comp, 0.5)
        grp[sname]["component_I_pi0.5"] = {g: summarize_groups(group_table(testAB, pin, testAB.in_lib.values.astype(float), w, g))
                                           for g in ["top1_class", "top1_superclass", "top1_subclass"]}
write_result(cfg, "eval_groups", {"primary": PRIMARY, "bin_edges_from_valB": {k: v.tolist() for k, v in edges.items()},
                                  "groups": grp})
print("groups done", f"{time.time()-t0:.0f}s", flush=True)

# ============================================================================= 5. incremental information of crowding + prior check
inc = {}
for a, c in dict.fromkeys([(PRIMARY, "M_base"), ("M_full", "M_base"), ("M_full", PRIMARY), ("M_sim_plus_crowd", "F_sim")]):
    if a == c:
        continue
    if a not in S or c not in S:
        continue
    pa, pcmp = S[a].p_correct_given_in(testA), S[c].p_correct_given_in(testA)
    ya = testA.correct.values.astype(float)
    ia, ic = S[a].p_in_cal(testAB), S[c].p_in_cal(testAB)
    yi = testAB.in_lib.values.astype(float)
    inc[f"{a} vs {c}"] = {
        "C_logloss": [calibration_metrics(pa, ya, testA.w.values)["logloss"], calibration_metrics(pcmp, ya, testA.w.values)["logloss"]],
        "C_auroc": [auroc(pa, ya, testA.w.values), auroc(pcmp, ya, testA.w.values)],
        "I_logloss": [calibration_metrics(ia, yi, testAB.w.values)["logloss"], calibration_metrics(ic, yi, testAB.w.values)["logloss"]],
        "I_auroc": [auroc(ia, yi, testAB.w.values), auroc(ic, yi, testAB.w.values)]}
    # cluster bootstrap of component differences
    ma = testA.groupby("mol").indices
    mab = testAB.groupby("mol").indices
    keys = np.array(sorted(set(ma) & set(mab)))
    dC, dI, dCa = [], [], []
    for b in range(ev["bootstrap_repeats"]):
        pick = keys[rng.integers(0, len(keys), len(keys))]
        ixa = np.concatenate([ma[k] for k in pick])
        ixb = np.concatenate([mab[k] for k in pick])
        dC.append(fast_ll(pa[ixa], ya[ixa], testA.w.values[ixa]) - fast_ll(pcmp[ixa], ya[ixa], testA.w.values[ixa]))
        dCa.append(auroc(pa[ixa], ya[ixa], testA.w.values[ixa]) - auroc(pcmp[ixa], ya[ixa], testA.w.values[ixa]))
        dI.append(fast_ll(ia[ixb], yi[ixb], testAB.w.values[ixb]) - fast_ll(ic[ixb], yi[ixb], testAB.w.values[ixb]))
    inc[f"{a} vs {c}"]["C_dlogloss_ci"] = np.percentile(dC, [2.5, 97.5]).tolist()
    inc[f"{a} vs {c}"]["C_dauroc_ci"] = np.nanpercentile(dCa, [2.5, 97.5]).tolist()
    inc[f"{a} vs {c}"]["I_dlogloss_ci"] = np.percentile(dI, [2.5, 97.5]).tolist()

# Mantel-Haenszel: error vs crowding within similarity-score strata (scenario a; edges from val-B)
s1_edges = np.unique(np.nanquantile(valB[valB.scenario == "a"].s1.values, np.linspace(0, 1, 11)[1:-1]))
testA["s1_stratum"] = np.searchsorted(s1_edges, testA.s1.values, side="right")
mh = {}
for c in ["top1_struct_nbrs", "fam_spec_crowd_train", "n_open_hi", "top1_specdens", "n_near_tie", "fam_ent_topk", "tani12"]:
    med = float(np.nanmedian(valB[valB.scenario == "a"][c].values))
    expo = (testA[c].values > med).astype(int)
    err = 1 - testA.correct.values
    num = den = 0.0
    P = Q = PR = PS_QR = QS = 0.0
    tabs = []
    for st, ix in testA.groupby("s1_stratum").indices.items():
        a_ = np.sum((expo[ix] == 1) & (err[ix] == 1)); b_ = np.sum((expo[ix] == 1) & (err[ix] == 0))
        c_ = np.sum((expo[ix] == 0) & (err[ix] == 1)); d_ = np.sum((expo[ix] == 0) & (err[ix] == 0))
        n = a_ + b_ + c_ + d_
        if n == 0:
            continue
        R_, S_ = a_ * d_ / n, b_ * c_ / n
        Pk, Qk = (a_ + d_) / n, (b_ + c_) / n
        num += R_; den += S_
        PR += Pk * R_; PS_QR += Pk * S_ + Qk * R_; QS += Qk * S_
        tabs.append({"stratum": int(st), "err_exposed": float(a_ / max(a_ + b_, 1)), "err_unexposed": float(c_ / max(c_ + d_, 1)),
                     "n_exposed": int(a_ + b_), "n_unexposed": int(c_ + d_)})
    orr = num / den if den > 0 else np.nan
    var = PR / (2 * num ** 2) + PS_QR / (2 * num * den) + QS / (2 * den ** 2) if num > 0 and den > 0 else np.nan
    ci = (float(np.exp(np.log(orr) - 1.96 * np.sqrt(var))), float(np.exp(np.log(orr) + 1.96 * np.sqrt(var)))) if np.isfinite(var) else (None, None)
    mh[c] = {"threshold_val_median": med, "mh_odds_ratio_error": float(orr) if np.isfinite(orr) else None, "ci95": ci, "strata": tabs}

# collaborator prior: family-level crowding (train library) vs test error rate by family
fam_crowd = testAB.drop_duplicates("top1_class").set_index("top1_class").fam_spec_crowd_train.to_dict()
prior = {}
from scipy.stats import spearmanr  # noqa: E402

for gcol in ["top1_class", "ana_true_class"]:
    g = testA.groupby(gcol).agg(n=("correct", "size"), n_mol=("mol", "nunique"),
                                err=("correct", lambda s: 1 - s.mean())).reset_index()
    g["crowd"] = g[gcol].map(fam_crowd)
    g = g[(g.n >= 50) & (g.n_mol >= 10) & g.crowd.notna()]
    if len(g) >= 5:
        rho, p = spearmanr(g.crowd, g.err)
        bs = []
        for b in range(ev["bootstrap_repeats"]):
            gg = g.iloc[rng.integers(0, len(g), len(g))]
            bs.append(spearmanr(gg.crowd, gg.err)[0])
        prior[gcol] = {"n_families": int(len(g)), "spearman_rho": float(rho), "p": float(p),
                       "rho_ci95_family_bootstrap": np.nanpercentile(bs, [2.5, 97.5]).tolist(),
                       "table": g.sort_values("crowd").to_dict("records")}
# query-level: error by crowding quartile
qlev = {}
for c in CROWD:
    qlev[c] = testA.groupby(f"bin_{c}").agg(n=("correct", "size"), err=("correct", lambda s: 1 - s.mean()),
                                            mean_s1=("s1", "mean")).reset_index().to_dict("records")
write_result(cfg, "eval_incremental", {"component_comparisons": inc, "mantel_haenszel_error_vs_crowding": mh,
                                       "family_level_prior": prior, "query_level_error_by_crowding_bin": qlev})
print("incremental done", f"{time.time()-t0:.0f}s", flush=True)

# ============================================================================= 6. scenario (c), near-correct, engine swap, errors
extra = {}
testAC = test[test.scenario.isin(["a", "c"])].reset_index(drop=True)
for sname in [PRIMARY, "M_base", "B_lr_dec"]:
    if sname not in S:
        continue
    comp = components(S[sname], testAC)
    extra.setdefault("scenario_c", {})[sname] = {str(pi): evaluate_weighted(comp, testAC, pi, "em") for pi in HEAD}
    for pi in HEAD:
        extra["scenario_c"][sname][str(pi) + "_oracle"] = evaluate_weighted(comp, testAC, pi, "oracle")
near = testAB.ana_tani_top1_truth.values >= ev["approx_correct_tanimoto"]
extra["near_correct"] = {}
for sname in [PRIMARY, "M_base", "B_lr_dec", "B_s1_direct"]:
    if sname in COMP:
        extra["near_correct"][sname] = {}
        for pi in HEAD:
            w = class_weights(testAB.in_lib.values.astype(float), testAB.w.values, pi)
            pc, _ = probs(COMP[sname], pi)
            extra["near_correct"][sname][str(pi)] = {**risk_coverage(pc, near.astype(float), w),
                                                     "frac_near_correct": float(np.average(near, weights=w))}
# engine swap (cosine)
cos = R["cosine"]
ctest = cos[((cos.fold == "test") if not DRY else ((cos.fold == "val") & (cos.vsplit == "A"))) & cos.scenario.isin(["a", "b"])].reset_index(drop=True)
SC = load_systems("cosine")
extra["engine_cosine"] = {}
for sname, sysm in SC.items():
    comp = components(sysm, ctest)
    extra["engine_cosine"][sname] = {str(pi): {m: evaluate_weighted(comp, ctest, pi, m) for m in ["oracle", "em"]} for pi in HEAD}
extra["engine_cosine_top1_acc_a"] = float(np.average(ctest[ctest.scenario == "a"].correct, weights=ctest[ctest.scenario == "a"].w))
extra["engine_entropy_top1_acc_a"] = float(np.average(testA.correct, weights=testA.w))
# error anatomy (scenario a errors, scenario b top-1)
errA = testA[testA.correct == 0]
extra["errors_a"] = {"n_err": int(len(errA)), "err_rate_weighted": float(1 - np.average(testA.correct, weights=testA.w)),
                     "tani_top1_truth_quantiles": np.percentile(errA.ana_tani_top1_truth, [10, 25, 50, 75, 90]).tolist() if len(errA) else None,
                     "frac_same_class_as_truth": float((errA.top1_class == errA.ana_true_class).mean()) if len(errA) else None,
                     "frac_same_superclass_as_truth": float((errA.top1_superclass == errA.ana_true_superclass).mean()) if len(errA) else None,
                     "frac_tani_ge_0.7": float((errA.ana_tani_top1_truth >= 0.7).mean()) if len(errA) else None}
tb = test[test.scenario == "b"]
extra["top1_in_b"] = {"tani_top1_truth_quantiles": np.percentile(tb.ana_tani_top1_truth, [10, 25, 50, 75, 90]).tolist(),
                      "frac_same_class_as_truth": float((tb.top1_class == tb.ana_true_class).mean()),
                      "frac_tani_ge_0.7": float((tb.ana_tani_top1_truth >= 0.7).mean()),
                      "s1_quantiles_b": np.percentile(tb.s1, [10, 25, 50, 75, 90]).tolist(),
                      "s1_quantiles_a": np.percentile(testA.s1, [10, 25, 50, 75, 90]).tolist()}
hard = testA.groupby("top1_class").agg(n=("correct", "size"), n_mol=("mol", "nunique"), err=("correct", lambda s: 1 - s.mean()))
hard = hard[(hard.n >= 50) & (hard.n_mol >= 10)].sort_values("err", ascending=False)
hard["lo"], hard["hi"] = zip(*[wilson(e * n, n) for e, n in zip(hard.err, hard.n)])
hard.reset_index().to_csv(tables / "hardest_families_top1_class.csv", index=False)
extra["hardest_families"] = hard.head(15).reset_index().to_dict("records")
write_result(cfg, "eval_extra", extra)
print("all done", f"{time.time()-t0:.0f}s", flush=True)
