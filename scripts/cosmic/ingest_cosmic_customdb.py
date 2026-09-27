"""MAIN COSMIC comparison: CSI:FingerID/COSMIC and the proposed system on the SAME candidate set.

Usage (repository root):  python scripts/cosmic/ingest_cosmic_customdb.py cosmic/customdb/results

SIRIUS searched custom structure databases made of our library's molecules (cosmic/customdb/README.md):
  (a) msg_test_a  = test-phase library molecules (truth present);  our side: leave-one-spectrum-out library
  (b) msg_test_b  = the same minus the 300 query molecules;        our side: all spectra of those molecules removed
Both systems therefore choose among the same molecules in each scenario. Every returned SIRIUS structure is
checked to belong to the searched database.
Evaluation (rules pre-declared in cosmic_common.py before these results existed):
  - per stratum: not in CSI:FingerID training structures (primary) / in them (leaky reference);
  - scenario (a) alone, and (a)+(b) mixed at pi in config evaluation.prevalence_headline;
  - COSMIC exact mode = main, approximate mode = supplement; COSMIC scores calibrated on the validation subset
    ((a)+(b) at the proposed system's calibration prevalence) with the shared calibrator rule, then the same
    Saerens EM prior correction that the proposed system uses (applied to COSMIC's P(correct));
  - no annotation -> confidence 0, not correct (both systems);
  - positive/negative counts for every stratum and system; minority class < 20 -> "unstable";
  - query bootstrap (both scenario rows of a query together), 1000 draws.
Output: results/eval_cosmic_customdb.json, reports/tables/cosmic_customdb_rows.csv
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosmic_common import (PIS, ROOT, MCESCache, attach_components, bootstrap, cfg, der, em_direct,  # noqa: E402
                           mix_weights, path, primary_system, proposed_eval, proposed_rows, read_summaries, wmetrics)
from msconf.chem import Standardizer  # noqa: E402
from msconf.config import write_result  # noqa: E402
from msconf.models import ScoreModel, choose_calibrator  # noqa: E402
from msconf.pipeline import val_split  # noqa: E402

seed = cfg["project"]["seed"]
B = int(os.environ.get("COSMIC_BOOT", cfg["evaluation"]["bootstrap_repeats"]))  # env only for smoke tests
res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cosmic" / "customdb" / "results"
man = pd.read_csv(ROOT / "cosmic" / "subset_manifest.csv")
std = Standardizer(cfg["structure"])
DIR = re.compile(r"summaries_(test|val)_(orbitrap|qtof)_([ab])$")

# ------------------------------------------------------------------ SIRIUS results per (split, scenario)
frames = []
for d in sorted(res_dir.glob("summaries_*")):
    m = DIR.search(d.name)
    f = next(iter(sorted(d.glob("structure_identifications.*"))), None)
    if not m or f is None:
        continue
    s = read_summaries([f], std)
    if s is None or not len(s):
        continue
    frames.append(s.assign(split=m.group(1), scenario=m.group(3)))
if not frames:
    raise SystemExit(f"no summaries_<split>_<instrument>_<a|b>/structure_identifications.* under {res_dir}")
S = pd.concat(frames, ignore_index=True).drop_duplicates(["identifier", "scenario"])

dbinfo = {}
for f in sorted(res_dir.glob("dbinfo_msg_*.txt")):
    t = f.read_text()
    n = re.findall(r"Number of Structures: (\d+)", t)
    dbinfo[f.stem.replace("dbinfo_", "")] = int(n[-1]) if n else None
dbman = json.load(open(ROOT / "cosmic" / "customdb" / "db_manifest.json"))
DB = {}
for key in ["test_a", "test_b", "val_a", "val_b"]:
    t = pd.read_csv(ROOT / "cosmic" / "customdb" / f"db_{key}.tsv", sep="\t", header=None, names=["smiles", "ik14", "name"])
    DB[key] = set(t.ik14)


def returned_ik14(r, dbset):
    """Structure identity of a SIRIUS hit in OUR identity space: SIRIUS's own InChIKey 2-D block if it is one of the
    database's keys (these were computed with our standardization), else the first block of the InChIKey of the
    returned SMILES after our standardization."""
    k = r.get("sirius_ik14")
    if isinstance(k, str) and k in dbset:
        return k, "sirius_inchikey2d"
    return r["pred_ik14"], "restandardized_smiles"


ids, how = [], []
for _, r in S.iterrows():
    a, b = returned_ik14(r, DB[f"{r['split']}_{r['scenario']}"])
    ids.append(a)
    how.append(b)
S["hit_ik14"], S["hit_id_source"] = ids, how
S["links_has_db"] = [isinstance(l, str) and f"msg_{sp}_{sc}" in l.split(";") for l, sp, sc in zip(S["links"], S.split, S.scenario)]
S["in_searched_db"] = [(h in DB[f"{sp}_{sc}"]) if isinstance(h, str) else np.nan for h, sp, sc in zip(S.hit_ik14, S.split, S.scenario)]

# ------------------------------------------------------------------ one row per (query, scenario) for COSMIC
mols = pd.read_parquet(der / "molecules.parquet").set_index("mol_idx")
SMI = mols.std_smiles.fillna(mols.smiles)
Q = pd.read_parquet(der / "queries.parquet")[["identifier", "mol_idx"]]
base = man.merge(Q, on="identifier", how="left")
base["true_ik14"] = base.mol_idx.map(mols.ik14)
base["true_smiles"] = base.mol_idx.map(SMI)
C = pd.concat([base.assign(scenario=sc) for sc in ["a", "b"]], ignore_index=True)
C = C.merge(S[["identifier", "scenario", "cosmic_exact", "cosmic_approx", "hit_ik14", "hit_id_source", "pred_smiles", "in_searched_db", "links_has_db"]],
            on=["identifier", "scenario"], how="left")
C["annotated"] = C.hit_ik14.notna()
C["cosmic_correct"] = (C.hit_ik14 == C.true_ik14).astype(float)
mces = MCESCache(der / "mces_cache.json")
C["cosmic_approx_correct"] = [1.0 if c == 1 else (mces(p, t) if a else 0.0)
                              for c, p, t, a in zip(C.cosmic_correct, C.pred_smiles, C.true_smiles, C.annotated)]
checks = {"db_structures_reported_by_sirius": dbinfo, "db_structures_expected": {k: v["n_molecules"] for k, v in dbman.items()},
          "hits_outside_searched_db": int((C.in_searched_db == False).sum()),  # noqa: E712
          "hits_without_db_link": int((C.annotated & (C.links_has_db == False)).sum()),  # noqa: E712
          "hit_id_source": C.hit_id_source.value_counts().to_dict(),
          "b_rows_correct_by_cosmic": int(C[C.scenario == "b"].cosmic_correct.sum()),
          "rows_with_results": {f"{sp}_{sc}": int(((C.split == sp) & (C.scenario == sc) & C.cosmic_exact.notna()).sum())
                                for sp in ["test", "val"] for sc in ["a", "b"]}}
# a hit outside the searched database would mean a PubChem fallback; such rows are treated as not annotated
bad = C.in_searched_db == False  # noqa: E712
C.loc[bad, ["annotated", "cosmic_exact", "cosmic_approx"]] = [False, np.nan, np.nan]
C.loc[bad, ["cosmic_correct", "cosmic_approx_correct"]] = 0.0

# ------------------------------------------------------------------ COSMIC calibration on validation (a)+(b)
prim, sysm = primary_system()
PI_CAL = float(sysm.pi_cal)
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
V = C[C.split == "val"].copy()
V["vsplit"] = V.mol_idx.map(split)
V["w"] = mix_weights(V, PI_CAL) * len(V)
cal, cal_info, prior = {}, {}, {}
for mode, score, label in [("exact", "cosmic_exact", "cosmic_correct"), ("approx", "cosmic_approx", "cosmic_approx_correct")]:
    vv = V.assign(_s=V[score].fillna(-1.0), _y=V[label].fillna(0.0))
    if vv._y.nunique() == 2 and len(vv) >= 40:
        cal[mode], cal_info[mode] = choose_calibrator(ScoreModel("_s"), vv, "_y")
        prior[mode] = float(np.average(vv._y, weights=vv.w))
    else:
        cal[mode], cal_info[mode], prior[mode] = None, {"note": "too few validation results"}, None

# ------------------------------------------------------------------ proposed system on the same queries
T = C[C.split == "test"].copy()
R = attach_components(sysm, proposed_rows(base[base.split == "test"].identifier.tolist(), "test"))
R = R.merge(base[["identifier", "in_csi_training", "true_smiles"]], on="identifier", how="left")
R["top1_smiles"] = R.top1.map(lambda t: SMI.get(int(t)) if pd.notna(t) else None)
R["approx_correct"] = [1.0 if c == 1 else (mces(p, t) if h == 1 else 0.0)
                       for c, p, t, h in zip(R.correct, R.top1_smiles, R.true_smiles, R.has_candidate)]
mces.save()
rng = np.random.default_rng(seed + 505)


def cosmic_eval(x, pi, mode="exact", correction="em", calset=None):
    cal_, prior_ = calset if calset is not None else (cal, prior)
    sc = x["cosmic_" + mode].fillna(-1.0).values
    y = x["cosmic_correct" if mode == "exact" else "cosmic_approx_correct"].fillna(0.0).values
    w = mix_weights(x, pi)
    if correction == "raw":
        return wmetrics(sc, y, w, np.clip(sc, 0, 1))
    p = cal_[mode](sc)
    est = None
    if correction == "em":
        p, est = em_direct(p, prior_[mode], w)
    r = wmetrics(sc, y, w, p)
    r["prior_estimate"] = est
    return r


def PE(x, pi, lab, correction="em"):
    return proposed_eval(x, pi, lab, correction, pi_cal=PI_CAL)


KEYS = ["accuracy", "aurc", "e_aurc", "auroc", "ece", "logloss"]


def with_ci(fn, d):
    out = fn(d)
    ci = bootstrap(d, lambda x: {k: fn(x).get(k) for k in KEYS}, B, rng)
    out.update({f"{k}_ci95": v for k, v in ci.items() if v is not None})
    return out


def paired(tsub, rsub, pi, mode, lab, calset=None):
    """Paired query bootstrap of proposed − COSMIC (val-calibrated + EM)."""
    idx_t = {k: v for k, v in tsub.groupby("identifier").indices.items()}
    idx_r = {k: v for k, v in rsub.groupby("identifier").indices.items()}
    keys = np.array(sorted(set(idx_t) & set(idx_r)))

    def both(kk):
        xt = tsub.iloc[np.concatenate([idx_t[k] for k in kk])]
        xr = rsub.iloc[np.concatenate([idx_r[k] for k in kk])]
        a, c = PE(xr, pi, lab), cosmic_eval(xt, pi, mode, "em", calset)
        return {k: (a.get(k, np.nan) - c.get(k, np.nan)) if a.get(k) is not None and c.get(k) is not None else np.nan
                for k in ["aurc", "e_aurc", "ece", "logloss", "auroc"]}
    pt = both(keys)
    vals = [both(keys[rng.integers(0, len(keys), len(keys))]) for _ in range(B)]
    return {k: {"point": pt[k], "ci95": np.nanpercentile([v[k] for v in vals], [2.5, 97.5]).tolist()} for k in pt}


res = {"primary_system": prim, "pi_cal": PI_CAL, "mixture_pi": PIS, "approx_label_method": mces.method, "checks": checks,
       "calibration": cal_info, "calibration_prior_correct": prior,
       "min_class_rows_for_stable_metrics": int(cfg["evaluation"]["min_group_mol_test"]),
       "n_test_queries": int(T.identifier.nunique()), "strata": {}}
for name, sel in [("not_in_csi_training (primary)", False), ("in_csi_training (leaky reference)", True)]:
    ts = T[T.in_csi_training == sel].reset_index(drop=True)
    rs = R[R.in_csi_training == sel].reset_index(drop=True)
    block = {"n_queries": int(ts.identifier.nunique()),
             "cosmic_unannotated": {sc: int((~ts[ts.scenario == sc].annotated).sum()) for sc in ["a", "b"]},
             "proposed_no_candidate": {sc: int((rs[rs.scenario == sc].has_candidate == 0).sum()) for sc in ["a", "b"]}}
    for mode, lab in [("exact", "correct"), ("approx", "approx_correct")]:
        mb = {}
        # scenario (a) alone (known in-library fraction 1 for the proposed system)
        ta, ra = ts[ts.scenario == "a"], rs[rs.scenario == "a"]
        mb["a_only"] = {"cosmic_raw": with_ci(lambda x: cosmic_eval(x, None, mode, "raw"), ta),
                        "cosmic_calibrated": with_ci(lambda x: cosmic_eval(x, None, mode, None), ta) if cal[mode] else None,
                        "proposed": with_ci(lambda x: PE(x, None, lab, 1.0), ra)}
        for pi in PIS:
            mb[f"mix_pi{pi}"] = {"cosmic_raw": with_ci(lambda x, pi=pi: cosmic_eval(x, pi, mode, "raw"), ts),
                                 "cosmic_calibrated_em": with_ci(lambda x, pi=pi: cosmic_eval(x, pi, mode, "em"), ts) if cal[mode] else None,
                                 "cosmic_calibrated_none": cosmic_eval(ts, pi, mode, None) if cal[mode] else None,
                                 "proposed_em": with_ci(lambda x, pi=pi: PE(x, pi, lab), rs),
                                 "proposed_none": PE(rs, pi, lab, None),
                                 "proposed_minus_cosmic": paired(ts, rs, pi, mode, lab) if cal[mode] else None}
        block[mode] = mb
    res["strata"][name] = block

# ------------------------------------------------------------------ a_x version (owner's request after the main import)
# Same 300 queries and the same SIRIUS results; on our side the library keeps the truth ONLY through spectra measured on
# the other instrument type (scenario a_x, scripts/16_scenario_cross_instrument.py). The candidate universe is unchanged
# (the truth molecule stays in the library), so COSMIC's (a) database is still the matching candidate set.
Qf = pd.read_parquet(der / "queries.parquet")[["identifier", "row"]]
AX = pd.read_parquet(der / "rows_ax_entropy.parquet")
AX = AX[AX.row.isin(R.row)].copy()
AX["identifier"] = AX.row.map(dict(zip(Qf.row, Qf.identifier)))
AX = AX.assign(scenario="a", w=1.0).merge(base[["identifier", "in_csi_training", "true_smiles"]], on="identifier", how="left")
AX = attach_components(sysm, AX)
AX["top1_smiles"] = AX.top1.map(lambda t: SMI.get(int(t)) if pd.notna(t) else None)
AX["approx_correct"] = [1.0 if c == 1 else mces(p_, t) for c, p_, t in zip(AX.correct, AX.top1_smiles, AX.true_smiles)]
mces.save()
ax_ids = set(AX.identifier)
Rx = pd.concat([AX, R[(R.scenario == "b") & R.identifier.isin(ax_ids)]], ignore_index=True)
Rd = R[R.identifier.isin(ax_ids)].reset_index(drop=True)
Tx = T[T.identifier.isin(ax_ids)].reset_index(drop=True)
res["a_x"] = {"n_eligible_queries": int(len(ax_ids)),
              "n_eligible_by_stratum": {k: int(v) for k, v in base[base.identifier.isin(ax_ids)].in_csi_training.map(
                  {False: "not_in_csi_training (primary)", True: "in_csi_training (leaky reference)"}).value_counts().items()},
              "eligibility": "truth has >= 1 spectrum from the other instrument type inside the precursor window", "strata": {}}
for name, sel in [("all eligible", None), ("not_in_csi_training (primary)", False), ("in_csi_training (leaky reference)", True)]:
    f = (lambda d: d) if sel is None else (lambda d, sel=sel: d[d.in_csi_training == sel].reset_index(drop=True))
    tx, rx, rd = f(Tx), f(Rx), f(Rd)
    if tx.identifier.nunique() == 0:
        continue
    blk = {"n_queries": int(tx.identifier.nunique()),
           "a_only": {"cosmic_calibrated": with_ci(lambda x: cosmic_eval(x, None, "exact", None), tx[tx.scenario == "a"]),
                      "proposed_default_a": with_ci(lambda x: PE(x, None, "correct", 1.0), rd[rd.scenario == "a"]),
                      "proposed_a_x": with_ci(lambda x: PE(x, None, "correct", 1.0), rx[rx.scenario == "a"])}}
    for pi in PIS:
        blk[f"mix_pi{pi}"] = {"cosmic_calibrated_em": with_ci(lambda x, pi=pi: cosmic_eval(x, pi, "exact"), tx),
                              "proposed_default_em": with_ci(lambda x, pi=pi: PE(x, pi, "correct"), rd),
                              "proposed_a_x_em": with_ci(lambda x, pi=pi: PE(x, pi, "correct"), rx),
                              "a_x_minus_cosmic": paired(tx, rx, pi, "exact", "correct"),
                              "default_minus_cosmic": paired(tx, rd, pi, "exact", "correct")}
    res["a_x"]["strata"][name] = blk

# ------------------------------------------------------------------ sensitivity: validation query whose (b) database holds a tautomer
# of the truth (scripts/cosmic/audit_customdb.py) removed from COSMIC's calibration; main analysis unchanged
# (computed here from the databases and SIRIUS hits, so this script does not depend on scripts/cosmic/audit_customdb.py)
tk = pd.read_parquet(der / "ext" / "tautomer_keys.parquet")
ik2tk = dict(zip(tk.ik14, tk.tkey))
vb = base[base.split == "val"]
vb_true_tk = vb.mol_idx.map(mols.ik14).map(ik2tk)
left = set(pd.Series(sorted(DB["val_b"])).map(ik2tk)) & set(vb_true_tk)  # tautomers of validation truths left in val (b)
drop = set(vb[vb_true_tk.isin(left)].identifier)
cv = C[(C.split == "val") & (C.scenario == "b") & C.annotated]
drop |= set(cv[(cv.hit_ik14.map(ik2tk) == cv.mol_idx.map(mols.ik14).map(ik2tk)) & (cv.cosmic_correct != 1)].identifier)
if drop:
    V2 = V[~V.identifier.isin(drop)]
    cal2, prior2 = {}, {}
    for mode, score, label in [("exact", "cosmic_exact", "cosmic_correct"), ("approx", "cosmic_approx", "cosmic_approx_correct")]:
        vv = V2.assign(_s=V2[score].fillna(-1.0), _y=V2[label].fillna(0.0))
        cal2[mode], _ = choose_calibrator(ScoreModel("_s"), vv, "_y")
        prior2[mode] = float(np.average(vv._y, weights=vv.w))
    ts = T[~T.in_csi_training].reset_index(drop=True)
    rs = R[~R.in_csi_training].reset_index(drop=True)
    res["sensitivity_val_tautomer"] = {"dropped_validation_queries": sorted(drop), "primary_stratum": {
        f"mix_pi{pi}": {"cosmic_calibrated_em": cosmic_eval(ts, pi, "exact", "em", (cal2, prior2)),
                        "proposed_minus_cosmic": paired(ts, rs, pi, "exact", "correct", (cal2, prior2))} for pi in PIS}}

SMOKE = "COSMIC_BOOT" in os.environ  # smoke test: never overwrite the real result
write_result(cfg, "smoketest_cosmic_customdb" if SMOKE else "eval_cosmic_customdb", res)
out = T.merge(R[["identifier", "scenario", "top1", "correct", "approx_correct", "has_candidate"]], on=["identifier", "scenario"], how="left")
out.to_csv(path(cfg, "tables") / ("smoketest_cosmic_customdb_rows.csv" if SMOKE else "cosmic_customdb_rows.csv"), index=False)
print(json.dumps({k: v for k, v in res.items() if k != "strata"}, indent=1))
for n, v in res["strata"].items():
    for key in ["a_only"] + [f"mix_pi{p}" for p in PIS]:
        e = v["exact"][key]
        print(n, key, {k: (x.get("n_pos"), x.get("n_neg"), round(x.get("e_aurc", np.nan), 4), round(x.get("ece", np.nan), 4), x.get("unstable"))
                       for k, x in e.items() if isinstance(x, dict) and "n_pos" in x})
