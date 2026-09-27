"""Validation-only diagnostic: which feature groups hurt P(in) / P(correct|in) out of sample?
Fixed LightGBM hyperparameters (M_base's selected config) so only the feature set varies.
Output: results/dev_feature_variants.json (all variants logged, §5.8)."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.features import GROUPS  # noqa: E402
from msconf.models import logloss  # noqa: E402
from msconf.pipeline import load_rows, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
seed = cfg["project"]["seed"]
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
rows = load_rows(der, "entropy")
rows = rows[rows.fold != "test"]
rows["vsplit"] = rows.mol.map(split)
tr, vA = rows[rows.fold == "train"], rows[(rows.fold == "val") & (rows.vsplit == "A")]
tl = json.load(open(path(cfg, "results") / "train_entropy.json"))["systems"]["M_base"]
hpC, hpI = tl["tuning_C"]["best"], tl["tuning_I"]["best"]
base = GROUPS["meta"] + GROUPS["sim"] + GROUPS["sim_extra"] + GROUPS["analog"]
fam_all = GROUPS["crowd_family"]
fam_query = ["fam_ent_topk", "fam_frac_topk_same", "fam_n_win_same", "fam_frac_open_same"]
spec, struct = GROUPS["crowd_spec"], GROUPS["crowd_struct"]
counts = ["n_win_mol", "n_win_spec"]
V = {
    "base": base,
    "base+family_all": base + fam_all,
    "base+family_no_libsize": base + [c for c in fam_all if c != "fam_lib_size"],
    "base+family_query_level": base + fam_query,
    "base+family_tables_only": base + ["fam_spec_crowd_train", "fam_struct_crowd_train"],
    "base+fam_lib_size_only": base + ["fam_lib_size"],
    "base+spec+struct": base + spec + struct,
    "base+spec+struct+family_query_level": base + spec + struct + fam_query,
    "base+spec+struct+family_no_libsize": base + spec + struct + [c for c in fam_all if c != "fam_lib_size"],
    "full": base + spec + struct + fam_all,
    "base+spec+struct-no_window_counts": [c for c in base + spec + struct if c not in counts],
    "base+spec+struct-no_libcounts": [c for c in base + spec + struct if c not in ["top1_struct_nbrs", "top1_specdens"]],
}
out = {}
for name, cols in V.items():
    r = {}
    for comp, hp, sc, y in [("C", hpC, ["a"], "correct"), ("I", hpI, ["a", "b"], "in_lib")]:
        t, v = tr[tr.scenario.isin(sc)], vA[vA.scenario.isin(sc)]
        m = lgb.LGBMClassifier(objective="binary", random_state=seed, verbose=-1, n_jobs=2, **hp)
        m.fit(t[cols].values.astype(float), t[y].values, sample_weight=t.w.values)
        p = m.predict_proba(v[cols].values.astype(float))[:, 1]
        ptr = m.predict_proba(t[cols].values.astype(float))[:, 1]
        r[comp] = {"valA_logloss": logloss(p, v[y].values, v.w.values), "train_logloss": logloss(ptr, t[y].values, t.w.values)}
        if name == "full":
            imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
            r[comp]["top_gain"] = (imp / imp.sum()).head(12).round(4).to_dict()
    out[name] = r
    print(f"{name:40s} C val {r['C']['valA_logloss']:.4f} (train {r['C']['train_logloss']:.4f})  I val {r['I']['valA_logloss']:.4f} (train {r['I']['train_logloss']:.4f})", flush=True)
write_result(cfg, "dev_feature_variants", out)
print(json.dumps(out["full"], indent=1))
