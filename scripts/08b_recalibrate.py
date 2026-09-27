"""Calibration stage (validation only), run after 08_train.py.

Rule (fixed on validation, see LOG 23:20): for every system and each component (P(correct|in), P(in), or the
direct P(correct)), choose between isotonic regression and Platt scaling by 2-fold cross-fitting between
val-A and val-B (lower mean log loss), then refit the chosen calibrator on the whole validation fold.
Every system — baselines included — gets the same calibration budget.
Also re-evaluates the per-family offset rule under the new calibrators (cross-fit val-A <-> val-B).
Overwrites the calibrators inside data/derived/models/<engine>/*.pkl and writes results/calibration_<engine>.json.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evaluation import group_calibration  # noqa: E402
from msconf.models import DirectSystem, GroupOffset, Iso, Platt, ScoreModel, logloss  # noqa: E402
from msconf.pipeline import load_rows, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
seed = cfg["project"]["seed"]
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
GROUP = "top1_" + cfg["family"]["primary_level"]


def choose(model, d, target):
    on_logit = not isinstance(model, ScoreModel)
    scores = {"isotonic": [], "platt": []}
    for fh, eh in [("A", "B"), ("B", "A")]:
        F, E = d[d.vsplit == fh], d[d.vsplit == eh]
        for name, C in [("isotonic", Iso()), ("platt", Platt(on_logit))]:
            c = C.fit(model.raw(F), F[target].values, F.w.values)
            scores[name].append(logloss(c(model.raw(E)), E[target].values, E.w.values))
    best = min(scores, key=lambda k: np.mean(scores[k]))
    C = Iso() if best == "isotonic" else Platt(on_logit)
    C.fit(model.raw(d), d[target].values, d.w.values)
    return C, {"choice": best, "crossfit_logloss": {k: float(np.mean(v)) for k, v in scores.items()}}


for engine in [a for a in sys.argv[1:] if not a.startswith("--")] or ["entropy", "cosine"]:
    rows = load_rows(der, engine)
    val = rows[rows.fold == "val"].copy()
    val["vsplit"] = val.mol.map(split)
    vA_, vAB = val[val.scenario == "a"], val[val.scenario.isin(["a", "b"])]
    log = {}
    for f in sorted((der / "models" / engine).glob("*.pkl")):
        s = pickle.loads(f.read_bytes())
        if f.stem.endswith("_gc"):
            continue
        if isinstance(s, DirectSystem):
            s.cal, info = choose(s.m, vAB, "correct")
            log[f.stem] = {"direct": info}
        else:
            s.calC, iC = choose(s.mC, vA_, "correct")
            s.calI, iI = choose(s.mI, vAB, "in_lib")
            s.pi_cal = float(np.average(vAB.in_lib, weights=vAB.w))
            s.groupC = s.groupI = None
            log[f.stem] = {"C": iC, "I": iI, "pi_cal": s.pi_cal}
        blob = pickle.dumps(s)
        f.write_bytes(blob)
        log[f.stem]["model_sha256"] = hashlib.sha256(blob).hexdigest()[:16]
        print(engine, f.stem, json.dumps({k: (v["choice"] if isinstance(v, dict) and "choice" in v else v) for k, v in log[f.stem].items()}), flush=True)
    out = {"calibrators": log}
    if engine == "entropy":
        sel = json.load(open(path(cfg, "results") / "selection.json"))
        prim = sel["primary_system"].replace("_gc", "")
        s = pickle.loads((der / "models" / engine / f"{prim}.pkl").read_bytes())
        minn = cfg["family"]["min_group_size_val"] // 2
        gaps = {"global": [], "group": []}
        for fh, eh in [("A", "B"), ("B", "A")]:
            for comp, d, tgt, m in [("C", vA_, "correct", s.mC), ("I", vAB, "in_lib", s.mI)]:
                F, E = d[d.vsplit == fh], d[d.vsplit == eh]
                on_logit = True
                c = Platt(on_logit).fit(m.raw(F), F[tgt].values, F.w.values)
                pg = c(m.raw(E))
                go = GroupOffset(minn).fit(c(m.raw(F)), F[tgt].values, F.w.values, F[GROUP].values)
                po = go(pg, E[GROUP].values)
                for lab, p in [("global", pg), ("group", po)]:
                    gc = group_calibration(E.assign(p=p), GROUP, "p", tgt, minn)
                    sf = gc[gc.sufficient]
                    gaps[lab].append({"comp": comp, "fit": fh, "gap": float((sf.obs - sf.pred).abs().mean()), "ll": logloss(p, E[tgt].values, E.w.values)})
        g, n = np.mean([x["gap"] for x in gaps["group"]]), np.mean([x["gap"] for x in gaps["global"]])
        lg, ln = np.mean([x["ll"] for x in gaps["group"]]), np.mean([x["ll"] for x in gaps["global"]])
        out["group_offsets_recheck"] = {"mean_gap_global": n, "mean_gap_group": g, "ll_global": ln, "ll_group": lg,
                                        "use_group_offsets": bool(g < 0.9 * n and lg <= ln + 1e-4), "folds": gaps}
        print("group offsets recheck:", out["group_offsets_recheck"]["use_group_offsets"], n, g, ln, lg)
    write_result(cfg, f"calibration_{engine}", out)
