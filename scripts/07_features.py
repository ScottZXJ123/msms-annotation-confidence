"""Compute observable-only feature rows for every (query, scenario, engine).

Scenarios: train fold -> (a), (b); val/test folds -> (a), (b), (c).
Engines: entropy (primary), cosine (secondary; same library, same processed peaks).
Output: data/derived/rows_<engine>.parquet
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.features import FeatureContext  # noqa: E402

cfg = load_config()
import os  # noqa: E402

der = Path(os.environ["MSCONF_DERIVED"]) if os.environ.get("MSCONF_DERIVED") else path(cfg, "derived")
Q = pd.read_parquet(der / "queries.parquet")
L = dict(np.load(der / "qlists.npz"))
CTX = FeatureContext(cfg, pd.read_parquet(der / "lib_index.parquet"), dict(np.load(der / "spec_nbrs07.npz")),
                     dict(np.load(der / "struct_nbrs.npz")), np.load(der / "fps.npy"),
                     pd.read_parquet(der / "families.parquet"), Q, L)
FAM = pd.read_parquet(der / "families.parquet")
PRIMARY = cfg["family"]["primary_level"]


def work(ix):
    out = []
    for i in ix:
        sc = ("a", "b") if Q.fold.iat[i] in ("train", "ext") else ("a", "b", "c")
        for eng in ("entropy", "cosine"):
            out.extend(CTX.rows_for_query(i, scenarios=sc, engine=eng, fam_level=PRIMARY))
    return out


if __name__ == "__main__":
    t0 = time.time()
    lim = int(os.environ.get("MSCONF_LIMIT", "0"))
    idx = np.arange(len(Q)) if not lim else np.random.default_rng(0).choice(len(Q), lim, replace=False)
    rows = []
    with Pool(2) as p:
        for k, r in enumerate(p.imap(work, np.array_split(idx, 200))):
            rows.extend(r)
            if k % 20 == 0:
                print(k, len(rows), f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    # family labels of the observable top-1 at all levels (grouping for calibration + sensitivity)
    for lv in ["superclass", "class", "subclass"]:
        lab = FAM[lv].values
        df[f"top1_{lv}"] = np.where(df.has_candidate == 1, lab[df.top1.fillna(0).astype(int)], None)
        df[f"ana_true_{lv}"] = lab[df.mol.values]
    # molecule weights: each molecule contributes total weight 1 per (fold, scenario, engine)
    df["w"] = 1.0 / df.groupby(["fold", "scenario", "engine", "mol"]).qid.transform("size")
    suffix = "" if not lim else "_debug"
    for eng, g in df.groupby("engine"):
        g.reset_index(drop=True).to_parquet(der / f"rows_{eng}{suffix}.parquet")
    summ = {"seconds": time.time() - t0, "n_rows": len(df)}
    for (eng, f, sc), g in df.groupby(["engine", "fold", "scenario"]):
        gg = g[g.has_candidate == 1]
        summ[f"{eng}/{f}/{sc}"] = {"n_queries": int(len(g)), "frac_no_candidate": float(1 - g.has_candidate.mean()),
                                   "n_mol": int(g.mol.nunique()),
                                   # engine accuracy is not summarized for the test fold here (test is read only by 09_evaluate)
                                   "top1_acc_unweighted": (float(gg.correct.mean()) if len(gg) else None) if f != "test" else "withheld"}
    if not lim:
        write_result(cfg, "features_summary" + ("_" + der.name if os.environ.get("MSCONF_DERIVED") else ""), summ)
    import json
    print(json.dumps(summ, indent=1))
