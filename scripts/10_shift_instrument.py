"""Leave-one-instrument-type shift test (§2): train + calibrate on one instrument type, test on the other.

The reference library is unchanged (all instrument types); only the confidence model's training and
calibration queries are restricted. Compared on the SAME test rows against the frozen systems trained on
all instrument types. Runs after the main evaluation (test fold used for evaluation only).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf import evalutil  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evalutil import components, evaluate_weighted  # noqa: E402
from msconf.models import choose_calibrator  # noqa: E402
from msconf.pipeline import load_rows, resolve_cols, train_decomposed, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
seed = cfg["project"]["seed"]
sel = json.load(open(path(cfg, "results") / "selection.json"))
PRIMARY = sel["primary_system"]
evalutil.set_thresholds(cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"])
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"],
                  cfg["models"]["val_split_frac_calib"], seed)
rows = load_rows(der, "entropy")
rows["vsplit"] = rows.mol.map(split)
GROUP = "top1_" + cfg["family"]["primary_level"]
INST = {0.0: "Orbitrap", 1.0: "QTOF"}
from msconf.pipeline import SYSTEMS  # noqa: E402

spec_of = {n: sp for n, _, _, sp in SYSTEMS}
base_primary = PRIMARY.replace("_gc", "")
specs = {base_primary: spec_of[base_primary], "M_base": "base", "B_lr_dec": ["s1", "gap12"]}
res = {"primary": PRIMARY, "note": "library unchanged; model trained/calibrated on the other instrument type"}
for held, hname in INST.items():
    other = 1.0 - held
    tr = rows[(rows.fold == "train") & (rows.instrument == other)]
    vA = rows[(rows.fold == "val") & (rows.vsplit == "A") & (rows.instrument == other)]
    vB = rows[(rows.fold == "val") & (rows.vsplit == "B") & (rows.instrument == other)]
    te = rows[(rows.fold == "test") & (rows.instrument == held) & rows.scenario.isin(["a", "b"])].reset_index(drop=True)
    out = {"n_train_rows": int(len(tr)), "n_test_rows": int(len(te)), "n_test_mol": int(te.mol.nunique())}
    for name, spec in specs.items():
        kind = "lr" if isinstance(spec, list) else "lgbm"
        use_gc = (name == base_primary and PRIMARY.endswith("_gc"))
        sysm, log = train_decomposed(name + ("_gc" if use_gc else ""), kind, resolve_cols(spec), tr, vA, vB, cfg, seed,
                                     group_col=GROUP if use_gc else None,
                                     min_group=cfg["family"]["min_group_size_val"] if use_gc else None)
        va = rows[(rows.fold == "val") & (rows.instrument == other)]
        if not use_gc:
            sysm.calC, _ = choose_calibrator(sysm.mC, va[va.scenario == "a"], "correct")
            sysm.calI, _ = choose_calibrator(sysm.mI, va[va.scenario.isin(["a", "b"])], "in_lib")
            sysm.pi_cal = float(np.average(va[va.scenario.isin(["a", "b"])].in_lib, weights=va[va.scenario.isin(["a", "b"])].w))
        comp = components(sysm, te)
        ref_name = (name + "_gc") if use_gc else name
        ref = pickle.loads((der / "models" / "entropy" / f"{ref_name}.pkl").read_bytes())
        comp_ref = components(ref, te)
        out[name] = {"cross_instrument": {str(pi): {m: evaluate_weighted(comp, te, pi, m) for m in ["oracle", "em"]} for pi in cfg["evaluation"]["prevalence_headline"]},
                     "all_instruments_reference": {str(pi): {m: evaluate_weighted(comp_ref, te, pi, m) for m in ["oracle", "em"]} for pi in cfg["evaluation"]["prevalence_headline"]},
                     "pi_cal": log["pi_cal"]}
        print(hname, name, "done", flush=True)
    res[f"test_on_{hname}"] = out
write_result(cfg, "eval_shift_instrument", res)
