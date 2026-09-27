"""Like-for-like comparison of HOW neighbourhood information is used (evaluation of frozen systems only).

All inside the same decomposed / label-shift framework with the same LightGBM budget and calibration rule:
  F_sim             similarity scores only
  B_ms2q_dec        similarity + MS2Query's neighbourhood CONSISTENCY features (mean similarity of the query to the
                    candidate's 10 structural library neighbours; their mean Tanimoto) + precursor m/z
  M_sim_plus_crowd  similarity + our COMPETITION/crowding features (near-ties, structural neighbours of the top-1,
                    spectral neighbourhood density, family entropy of the top-k)
  M_base / primary  the full systems without / with crowding
Paired molecule-cluster bootstrap on the test fold. Output: results/eval_neighbourhood_comparison.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evalutil import class_weights, components, probs  # noqa: E402
from msconf.evaluation import risk_coverage  # noqa: E402
from msconf.pipeline import load_rows  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
seed = cfg["project"]["seed"]
P = json.load(open(path(cfg, "results") / "selection.json"))["primary_system"]
rows = load_rows(der, "entropy")
rows = rows[rows.scenario.isin(["a", "b"])]
C = pd.read_parquet(der / "ms2q_candidates.parquet")[["qid", "scenario", "cand", "nbr_sim", "nbr_tani"]].rename(columns={"cand": "top1"})
test = rows[rows.fold == "test"].merge(C, on=["qid", "scenario", "top1"], how="left").reset_index(drop=True)
S = {n: pickle.loads((der / "models" / "entropy" / f"{n}.pkl").read_bytes()) for n in ["F_sim", "M_sim_plus_crowd", "M_base", P]}
S["B_ms2q_dec"] = pickle.loads((der / "models" / "entropy_new_baselines" / "B_ms2q_dec.pkl").read_bytes())
COMP = {n: components(s, test) for n, s in S.items()}
PAIRS = [("M_sim_plus_crowd", "B_ms2q_dec"), ("B_ms2q_dec", "F_sim"), ("M_sim_plus_crowd", "F_sim"), (P, "B_ms2q_dec"), (P, "M_base")]
mi = test.groupby("mol").indices
mk = np.array(list(mi))
rng = np.random.default_rng(seed)


def ll(p, y, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def met(ix, pi):
    d = test.iloc[ix]
    y, yin = d.correct.values.astype(float), d.in_lib.values.astype(float)
    w = class_weights(yin, d.w.values, pi)
    o = {}
    for n, c in COMP.items():
        pc, _ = probs({"direct": c["direct"], "pc": c["pc"][ix], "pin_cal": c["pin_cal"][ix], "pi_cal": c["pi_cal"]}, pi)
        rc = risk_coverage(pc, y, w)
        o[n] = {"e_aurc": rc["e_aurc"], "logloss": ll(pc, y, w)}
    return o


res = {"primary": P}
for pi in cfg["evaluation"]["prevalence_headline"]:
    pt = met(np.arange(len(test)), pi)
    dr = [met(np.concatenate([mi[k] for k in mk[rng.integers(0, len(mk), len(mk))]]), pi) for _ in range(cfg["evaluation"]["bootstrap_repeats"])]
    res[str(pi)] = {"point": pt, "diffs": {f"{a} - {b}": {k: {"d": pt[a][k] - pt[b][k],
                                                                "ci95": np.percentile([x[a][k] - x[b][k] for x in dr], [2.5, 97.5]).tolist()}
                                                            for k in ["e_aurc", "logloss"]} for a, b in PAIRS}}
write_result(cfg, "eval_neighbourhood_comparison", res)
print(json.dumps(res, indent=1)[:3000])
