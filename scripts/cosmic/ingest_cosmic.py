"""Import the SIRIUS 6 / COSMIC PubChem run on the public subset (SUPPLEMENTARY background comparison).

Usage (repository root):  python scripts/cosmic/ingest_cosmic.py cosmic/results

The main COSMIC comparison is the same-candidate-set head-to-head (ingest_cosmic_customdb.py). Here COSMIC scores
CSI:FingerID's top-1 from PubChem, which contains almost every test molecule, so COSMIC has only scenario (a).
The proposed system is scored on its own spectral-library top-1:
  - scenario (a) alone (truth in the library) -- only a handful of wrong top-1 annotations, so every metric that
    depends on the negatives is flagged "unstable" (rule in cosmic_common.py);
  - (a) and (b) mixed at the pre-declared in-library fractions pi (config evaluation.prevalence_headline), with
    the same EM prior correction as in the main evaluation, so that the proposed side has enough negatives too.
    (b) removes all spectra of all 300 subset molecules from the library at once.
Strata: molecule in the published CSI:FingerID training structures (leaky reference) vs not (primary).
Exact mode is the main result; approximate mode ("correct or MCES distance <= 2") is a supplement.
Positive and negative counts are reported for every stratum and system. The AUROC difference is reported for
completeness only; it is not a main conclusion (few negatives on the proposed side in (a), few positives on
the COSMIC side in the primary stratum).
Output: results/eval_cosmic_subset.json, reports/tables/cosmic_subset_rows.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosmic_common import (PIS, ROOT, MCESCache, attach_components, bootstrap, cfg, der, path,  # noqa: E402
                           primary_system, proposed_eval, proposed_rows, read_summaries, wmetrics)
from msconf.chem import Standardizer  # noqa: E402
from msconf.config import write_result  # noqa: E402
from msconf.models import ScoreModel, choose_calibrator  # noqa: E402
from msconf.pipeline import val_split  # noqa: E402

seed = cfg["project"]["seed"]
B = cfg["evaluation"]["bootstrap_repeats"]
res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cosmic" / "results"
man = pd.read_csv(ROOT / "cosmic" / "subset_manifest.csv")
std = Standardizer(cfg["structure"])
S = read_summaries(sorted(res_dir.glob("summaries_*/structure_identifications.*")), std)
if S is None:
    raise SystemExit(f"no structure_identifications.* under {res_dir}")
S = S.drop_duplicates("identifier")

# ------------------------------------------------------------------ truth, COSMIC labels
mols = pd.read_parquet(der / "molecules.parquet").set_index("mol_idx")
SMI = mols.std_smiles.fillna(mols.smiles)
Q = pd.read_parquet(der / "queries.parquet")[["identifier", "pos", "mol_idx", "fold"]]
D = man.merge(Q, on="identifier", how="left").merge(S[["identifier", "cosmic_exact", "cosmic_approx", "pred_ik14", "pred_smiles"]],
                                                    on="identifier", how="left")
D["true_ik14"] = D.mol_idx.map(mols.ik14)
D["true_smiles"] = D.mol_idx.map(SMI)
D["annotated"] = D.pred_ik14.notna()
D["cosmic_correct"] = (D.pred_ik14 == D.true_ik14).astype(float)
mces = MCESCache(der / "mces_cache.json")
approx_method = mces.method
D["cosmic_approx_correct"] = [1.0 if c == 1 else (mces(p, t) if a else 0.0)
                              for c, p, t, a in zip(D.cosmic_correct, D.pred_smiles, D.true_smiles, D.annotated)]

# ------------------------------------------------------------------ calibration of COSMIC on the validation subset
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
V = D[(D.split == "val") & D.annotated].copy()
V["vsplit"] = V.mol_idx.map(split)
V["w"] = 1.0
cal, cal_info = {}, {}
for mode, score, label in [("exact", "cosmic_exact", "cosmic_correct"), ("approx", "cosmic_approx", "cosmic_approx_correct")]:
    vv = V[V[label].notna() & V[score].notna()]
    if len(vv) >= 40 and vv[label].nunique() == 2:
        cal[mode], cal_info[mode] = choose_calibrator(ScoreModel(score), vv, label)
    else:
        cal[mode], cal_info[mode] = None, {"note": "too few validation results"}

# ------------------------------------------------------------------ proposed system: scenario (a) and (b) on the same 300 queries
T = D[D.split == "test"].copy()
prim, sysm = primary_system()
R = attach_components(sysm, proposed_rows(T.identifier.tolist(), "test"))
R = R.merge(T[["identifier", "in_csi_training", "true_smiles"]], on="identifier", how="left")
R["top1_smiles"] = R.top1.map(lambda t: SMI.get(int(t)) if pd.notna(t) else None)
R["approx_correct"] = [1.0 if c == 1 else (mces(p, t) if h == 1 else 0.0)
                       for c, p, t, h in zip(R.correct, R.top1_smiles, R.true_smiles, R.has_candidate)]
mces.save()
rng = np.random.default_rng(seed)


def PE(x, pi, lab, correction="em"):
    return proposed_eval(x, pi, lab, correction, pi_cal=sysm.pi_cal)


def cosmic_eval(d, mode, calibrated):
    score = d["cosmic_" + ("exact" if mode == "exact" else "approx")].fillna(-1.0).values  # unannotated -> lowest
    y = d["cosmic_correct" if mode == "exact" else "cosmic_approx_correct"].fillna(0.0).values
    prob = (cal[mode](score) if cal[mode] is not None else np.full(len(d), np.nan)) if calibrated else np.clip(score, 0, 1)
    return wmetrics(score, y, np.ones(len(d)), prob)


KEYS = ["accuracy", "aurc", "e_aurc", "auroc", "ece", "logloss"]


def with_ci(fn, d):
    out = fn(d)
    ci = bootstrap(d, lambda x: {k: fn(x).get(k) for k in KEYS}, B, rng)
    out.update({f"{k}_ci95": v for k, v in ci.items() if v is not None})
    return out


res = {"n_test": int(len(T)), "n_test_annotated_by_sirius": int(T.annotated.sum()),
       "n_test_confidence_missing": int((T.annotated & T.cosmic_exact.isna()).sum()), "n_val": int((D.split == "val").sum()),
       "approx_label_method": approx_method, "primary_system": prim, "mixture_pi": PIS,
       "calibration": cal_info, "min_class_rows_for_stable_metrics": int(cfg["evaluation"]["min_group_mol_test"]),
       "note": "Supplementary. COSMIC ranks PubChem hits (scenario a only); the proposed system ranks spectral-library hits "
               "in scenario (a) and in (a)+(b) mixtures. Each system is evaluated on its own annotations.", "strata": {}}
for name, sel in [("not_in_csi_training (primary)", ~T.in_csi_training), ("in_csi_training (leaky reference)", T.in_csi_training)]:
    sub = T[sel.values]
    rs = R[R.identifier.isin(sub.identifier)]
    ra = rs[rs.scenario == "a"]
    block = {}
    for mode, lab in [("exact", "correct"), ("approximate", "approx_correct")]:
        m = "exact" if mode == "exact" else "approx"
        b = {"cosmic": with_ci(lambda x: cosmic_eval(x, m, False), sub),
             "cosmic_calibrated": with_ci(lambda x: cosmic_eval(x, m, True), sub) if cal[m] is not None else None,
             "proposed_a_only": with_ci(lambda x: PE(x, None, lab, correction=1.0), ra),
             "n_cosmic_unannotated": int((~sub.annotated).sum())}
        for pi in PIS:
            b[f"proposed_mix_pi{pi}"] = with_ci(lambda x, pi=pi: PE(x, pi, lab), rs)
        # (a)-only paired differences, kept for completeness; not a main conclusion
        for k in ["auroc", "e_aurc"]:
            RA = R[R.scenario == "a"].set_index("identifier")

            def diff(x, k=k, m=m, lab=lab, RA=RA):
                xa = RA.loc[x.identifier.values].reset_index()
                a = PE(xa, None, lab, correction=1.0).get(k)
                c = cosmic_eval(x, m, False).get(k)
                return {"d": np.nan if a is None or c is None else a - c}
            pt = diff(sub)["d"]
            ci = bootstrap(sub, diff, B, rng).get("d")
            b[f"a_only_proposed_minus_cosmic_{k}"] = {"point": pt, "ci95": ci, "not_a_main_conclusion": True}
        block[mode] = b
    res["strata"][name] = block
write_result(cfg, "eval_cosmic_subset", res)
D.to_csv(path(cfg, "tables") / "cosmic_subset_rows.csv", index=False)
print(json.dumps({k: v for k, v in res.items() if k != "strata"}, indent=1))
for n, v in res["strata"].items():
    e = v["exact"]
    print(n, {k: (e[k].get("n_pos"), e[k].get("n_neg"), round(e[k].get("e_aurc", np.nan), 4), e[k].get("unstable"))
              for k in e if isinstance(e[k], dict) and "n_pos" in e[k]})
