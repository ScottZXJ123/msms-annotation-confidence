"""Train all systems (baselines, floors, method, ablations) on the TRAIN fold, select hyperparameters on
val-A, calibrate on val-B. The test fold is not read here.

Also makes the two pre-registered validation-only decisions:
  (1) whether the primary system uses per-family calibration offsets (2-fold cross-fit inside val-B);
  (2) sanity of the frozen tier thresholds on val-B (tier sizes / accuracies at the calibration prevalence).
Outputs: data/derived/models/<engine>/<system>.pkl, results/train_<engine>.json, results/selection.json
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evaluation import assign_tiers, calibration_metrics, group_calibration, prior_shift  # noqa: E402
from msconf.models import GroupOffset, Isotonic  # noqa: E402
from msconf.pipeline import (SYSTEMS, load_rows, resolve_cols, train_decomposed, train_direct,  # noqa: E402
                             val_split)

cfg = load_config()
der = path(cfg, "derived")
seed = cfg["project"]["seed"]
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"],
                  cfg["models"]["val_split_frac_calib"], seed)
DECIDE_ONLY = "--decide-only" in sys.argv
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
PRIMARY_CANDIDATES = ["M_full", "M_full_nocounts", "M_full_minus_family", "M_base"]
engines = [a for a in sys.argv[1:] if not a.startswith("--")] or ["entropy", "cosine"]
GROUP = "top1_" + cfg["family"]["primary_level"]


def reweight(df, pi, pi_cal):
    """Importance weights that move a labelled batch from prevalence pi_cal to pi (validation diagnostics)."""
    w = df.w.values * np.where(df.in_lib.values == 1, pi / pi_cal, (1 - pi) / (1 - pi_cal))
    return w


for engine in engines:
    t0 = time.time()
    rows = load_rows(der, engine)
    rows = rows[rows.fold != "test"]  # the test fold is never touched during training/selection
    rows["vsplit"] = rows.mol.map(split)
    train = rows[rows.fold == "train"]
    valA = rows[(rows.fold == "val") & (rows.vsplit == "A")]
    valB = rows[(rows.fold == "val") & (rows.vsplit == "B")]
    mdir = der / "models" / engine
    mdir.mkdir(parents=True, exist_ok=True)
    prev = path(cfg, "results") / f"train_{engine}.json"
    logs = json.load(open(prev))["systems"] if (ONLY and prev.exists()) else {}
    systems = SYSTEMS if engine == "entropy" else [s for s in SYSTEMS if s[0] in
                                                   {"B_s1_direct", "B_gap_direct", "B_lr_direct", "B_s1_dec", "B_gap_dec",
                                                    "B_lr_dec", "F_meta", "F_sim", "M_base", "M_full",
                                                    "M_full_minus_family"}]
    for name, builder, kind, spec in (systems if not DECIDE_ONLY else []):
        if ONLY and name not in ONLY:
            continue
        t1 = time.time()
        cols = resolve_cols(spec)
        if builder == "dec":
            sysm, log = train_decomposed(name, kind, cols, train, valA, valB, cfg, seed)
        else:
            sysm, log = train_direct(name, kind, cols, train, valA, valB, cfg, seed)
        # validation-only diagnostics at the calibration prevalence and reweighted prevalences
        vb = valB[valB.scenario.isin(["a", "b"])]
        diag = {}
        pi_cal = log.get("pi_cal", float(np.average(vb.in_lib, weights=vb.w)))
        for pi in [None, 0.1, 0.2, 0.5]:
            pc, pin = sysm.predict(vb, pi)
            w = vb.w.values if pi is None else reweight(vb, pi, pi_cal)
            diag[f"pi={pi}"] = calibration_metrics(pc, vb.correct.values, w)
        log["valB_diag"] = diag
        log["seconds"] = time.time() - t1
        blob = pickle.dumps(sysm)
        (mdir / f"{name}.pkl").write_bytes(blob)
        log["model_sha256"] = hashlib.sha256(blob).hexdigest()[:16]
        logs[name] = log
        print(engine, name, f"{time.time()-t1:.0f}s", "valB ll", round(diag["pi=None"]["logloss"], 4),
              "ece", round(diag["pi=None"]["ece"], 4), flush=True)
    if not DECIDE_ONLY:
        write_result(cfg, f"train_{engine}", {"val_split_sizes": pd.Series(split).value_counts().to_dict(),
                                              "systems": logs, "seconds": time.time() - t0})

    if engine != "entropy":
        continue
    # ------------------------------------------------------------------ decision (0): primary feature set
    # rule: lowest val-A log loss of the combined p_correct at pi = 0.5 (calibration fitted on val-B)
    vaAB = valA[valA.scenario.isin(["a", "b"])]
    cand = {}
    for name in PRIMARY_CANDIDATES:
        sm = pickle.loads((mdir / f"{name}.pkl").read_bytes())
        pc, _ = sm.predict(vaAB, 0.5)
        w = reweight(vaAB, 0.5, float(np.average(vaAB.in_lib, weights=vaAB.w)))
        cand[name] = {"valA_logloss_pi0.5": calibration_metrics(pc, vaAB.correct.values, w)["logloss"]}
        pc2, _ = sm.predict(vaAB, 0.2)
        w2 = reweight(vaAB, 0.2, float(np.average(vaAB.in_lib, weights=vaAB.w)))
        cand[name]["valA_logloss_pi0.2"] = calibration_metrics(pc2, vaAB.correct.values, w2)["logloss"]
    base_primary = min(cand, key=lambda k: cand[k]["valA_logloss_pi0.5"])
    print("primary candidates (val-A):", json.dumps(cand), "->", base_primary, flush=True)
    # ------------------------------------------------------------------ decision (1): family offsets?
    sysm = pickle.loads((mdir / f"{base_primary}.pkl").read_bytes())
    minn = cfg["family"]["min_group_size_val"]
    vbmols = np.array(sorted(valB.mol.unique()))
    sub_split = val_split(fam.assign(fold=np.where(fam.mol_idx.isin(vbmols), "val", "x")),
                          dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], 0.5, seed + 7)
    valB = valB.assign(half=valB.mol.map(sub_split))
    res = {"global": [], "group": []}
    for fit_h, ev_h in [("A", "B"), ("B", "A")]:
        F, E = valB[valB.half == fit_h], valB[valB.half == ev_h]
        for comp, target, sc in [("C", "correct", ["a"]), ("I", "in_lib", ["a", "b"])]:
            Fm, Em = F[F.scenario.isin(sc)], E[E.scenario.isin(sc)]
            m = sysm.mC if comp == "C" else sysm.mI
            iso = Isotonic().fit(m.raw(Fm), Fm[target].values, Fm.w.values)
            pg = iso(m.raw(Em))
            go = GroupOffset(minn // 2).fit(iso(m.raw(Fm)), Fm[target].values, Fm.w.values, Fm[GROUP].values)
            pgo = go(pg, Em[GROUP].values)
            for lab, p in [("global", pg), ("group", pgo)]:
                gc = group_calibration(Em.assign(p=p), GROUP, "p", target, minn // 2)
                suff = gc[gc.sufficient]
                res[lab].append({"comp": comp, "fit": fit_h, "mean_abs_gap_sufficient": float((suff.obs - suff.pred).abs().mean()),
                                 "frac_consistent_sufficient": float(suff.consistent.mean()) if len(suff) else None,
                                 "n_sufficient_groups": int(len(suff)),
                                 "logloss": calibration_metrics(p, Em[target].values, Em.w.values)["logloss"]})
    g_gap = np.mean([r["mean_abs_gap_sufficient"] for r in res["group"]])
    n_gap = np.mean([r["mean_abs_gap_sufficient"] for r in res["global"]])
    g_ll = np.mean([r["logloss"] for r in res["group"]])
    n_ll = np.mean([r["logloss"] for r in res["global"]])
    use_group = bool(g_gap < 0.9 * n_gap and g_ll <= n_ll + 1e-4)
    decision = {"rule": "use per-family offsets iff cross-fitted mean |obs-pred| over sufficient groups drops by >10% "
                        "and log loss does not increase", "global": res["global"], "group": res["group"],
                "mean_gap_global": n_gap, "mean_gap_group": g_gap, "ll_global": n_ll, "ll_group": g_ll,
                "use_group_offsets": use_group}
    spec_of = {n: sp for n, _, _, sp in SYSTEMS}
    if use_group:
        vtr = rows[rows.fold == "train"]
        vA = rows[(rows.fold == "val") & (rows.vsplit == "A")]
        vB = rows[(rows.fold == "val") & (rows.vsplit == "B")]
        sg, lg = train_decomposed(base_primary + "_gc", "lgbm", resolve_cols(spec_of[base_primary]), vtr, vA, vB, cfg, seed,
                                  group_col=GROUP, min_group=minn)
        (mdir / f"{base_primary}_gc.pkl").write_bytes(pickle.dumps(sg))
    primary = base_primary + "_gc" if use_group else base_primary
    # ------------------------------------------------------------------ decision (2): tier sanity on val-B
    psys = pickle.loads((mdir / f"{primary}.pkl").read_bytes())
    vb = rows[(rows.fold == "val") & (rows.vsplit == "B") & rows.scenario.isin(["a", "b"])]
    tiers = {}
    for pi in [0.1, 0.2, 0.5]:
        pc, pin = psys.predict(vb, pi)
        t = assign_tiers(pc, pin, cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"])
        w = reweight(vb, pi, psys.pi_cal)
        tiers[f"pi={pi}"] = {tt: {"weight_frac": float(w[t == tt].sum() / w.sum()),
                                  "acc": float(np.average(vb.correct.values[t == tt], weights=w[t == tt])) if (t == tt).any() else None,
                                  "pred": float(np.average(pc[t == tt], weights=w[t == tt])) if (t == tt).any() else None,
                                  "in_rate": float(np.average(vb.in_lib.values[t == tt], weights=w[t == tt])) if (t == tt).any() else None}
                             for tt in ["trusted", "suspicious", "absent"]}
    write_result(cfg, "selection", {"primary_candidates_valA": cand, "primary_rule": "min val-A log loss of p_correct at pi=0.5",
                                    "group_offsets_decision": decision, "primary_system": primary,
                                    "frozen_thresholds": {"trusted_min_p_correct": cfg["tiers"]["trusted_min_p_correct"],
                                                          "absent_max_p_in": cfg["tiers"]["absent_max_p_in"]},
                                    "valB_tiers_reweighted": tiers})
    print(json.dumps({"primary": primary, "use_group": use_group, "gap": [n_gap, g_gap], "ll": [n_ll, g_ll]}, indent=1))
    print(json.dumps(tiers, indent=1))
