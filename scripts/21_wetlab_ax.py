"""Wet-lab simulation under scenario a_x (truth in the library only via other-instrument spectra), i.e. the
planning condition for a lab querying a public library. Frozen primary system, EM prevalence; in-library
queries from a_x rows, absent queries from the (b) rows of the same test queries. Test fold only; nothing is
fitted. Output: results/eval_wetlab_ax.json"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evalutil import components, probs  # noqa: E402
from msconf.evaluation import assign_tiers, em_prevalence, summarize_wetlab, wilson  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
ev = cfg["evaluation"]
seed = cfg["project"]["seed"]
T_TR, T_AB = cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"]
PRIMARY = json.load(open(path(cfg, "results") / "selection.json"))["primary_system"]
ax = pd.read_parquet(der / "rows_ax_entropy.parquet").assign(scenario="a")
main = pd.read_parquet(der / "rows_entropy.parquet")
b = main[(main.fold == "test") & (main.scenario == "b") & (main.has_candidate == 1) & main.qid.isin(ax.qid)]
df = pd.concat([ax, b], ignore_index=True)
sysm = pickle.loads((der / "models" / "entropy" / f"{PRIMARY}.pkl").read_bytes())
comp = components(sysm, df)
Y, YIN = df.correct.values, df.in_lib.values


def by_mol(mask):
    pos = np.nonzero(mask)[0]
    return {k: pos[ix] for k, ix in df.iloc[pos].groupby("mol").indices.items()}


A, B = by_mol(df.scenario.values == "a"), by_mol(df.scenario.values == "b")
ALLM = np.array(sorted(set(A) | set(B)))


def batch(pi, r):
    m = ALLM.copy()
    r.shuffle(m)
    h = len(m) // 2
    a = [A[k][r.integers(len(A[k]))] for k in m[:h] if k in A]
    bb = [B[k][r.integers(len(B[k]))] for k in m[h:] if k in B]
    nA, nB = len(a), len(bb)
    if nA / (nA + nB) > pi:
        nA = int(round(pi / (1 - pi) * nB))
    else:
        nB = int(round((1 - pi) / pi * nA))
    return np.array(a[:nA] + bb[:nB])


out = {"primary": PRIMARY, "n_in_molecules": len(A), "n_absent_molecules": len(B), "wetlab": {}}
for pi in [0.1, 0.2, 0.3, 0.5]:
    r = np.random.default_rng(seed + int(pi * 100))
    recs, pis = [], []
    for _ in range(ev["wetlab_repeats"]):
        ix = batch(pi, r)
        pu = em_prevalence(comp["pin_cal"][ix], comp["pi_cal"])
        pis.append(pu)
        pc, pin = probs({"direct": False, "pc": comp["pc"][ix], "pin_cal": comp["pin_cal"][ix], "pi_cal": comp["pi_cal"]}, pu)
        tier = assign_tiers(pc, pin, T_TR, T_AB)
        for t in ["trusted", "suspicious", "absent"]:
            cand = np.nonzero(tier == t)[0]
            if len(cand) == 0:
                recs.append({"tier": t, "n_avail": 0, "n": 0})
                continue
            take = r.choice(cand, size=min(ev["wetlab_per_tier"], len(cand)), replace=False)
            n = len(take)
            kc, ki = int(Y[ix][take].sum()), int(YIN[ix][take].sum())
            lo, hi = wilson(kc, n)
            li, hi_i = wilson(ki, n)
            pc_, pi_ = float(pc[take].mean()), float(pin[take].mean())
            recs.append({"tier": t, "n_avail": len(cand), "n": n, "obs_acc": kc / n, "acc_lo": lo, "acc_hi": hi, "pred_acc": pc_,
                         "acc_consistent": lo <= pc_ <= hi, "obs_in": ki / n, "in_lo": li, "in_hi": hi_i, "pred_in": pi_,
                         "in_consistent": li <= pi_ <= hi_i})
    s = summarize_wetlab(recs)
    s["_pi_used"] = {"mean": float(np.mean(pis)), "q025": float(np.percentile(pis, 2.5)), "q975": float(np.percentile(pis, 97.5))}
    out["wetlab"][str(pi)] = s
write_result(cfg, "eval_wetlab_ax", out)
print(json.dumps({k: {t: (round(v[t]["obs_acc_mean"], 3), round(v[t]["pred_acc_mean"], 3), round(v[t]["acc_consistent_rate"], 2))
                      for t in ["trusted", "suspicious", "absent"] if t in v} for k, v in out["wetlab"].items()}, indent=1))
