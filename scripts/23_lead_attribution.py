"""Where does the proposed system's lead over the score baselines come from? Breakdown by feature group.

POST HOC (added after the main test evaluation, at the project owner's request; the proposed method is NOT
changed and nothing here feeds back into any choice). Ablation systems are trained exactly like every other
system: LightGBM, the same 16-configuration budget selected on val-A, calibrator chosen by val-A <-> val-B
cross-fit and refitted on the whole validation fold, Bayes prior shift with EM on test.

Two views on the test fold (pi = 0.2 and 0.5):
  path:     logistic(s1, gap) decomposed -> sim -> + sim_extra -> + analog -> + meta (= no-crowding model)
            -> + crowd_spec + crowd_struct (= proposed). Step differences telescope to the total lead.
  drop-one: the proposed system minus one feature group at a time (order-free).
Molecule-cluster bootstrap of every difference (oracle prevalence, as in results/eval_bootstrap.json).
Models are stored in data/derived/models/entropy_lead/ (separate from the main systems).
Output: results/eval_lead_attribution.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf import evalutil  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evalutil import class_weights, components, evaluate_weighted, probs  # noqa: E402
from msconf.evaluation import risk_coverage  # noqa: E402
from msconf.features import GROUPS  # noqa: E402
from msconf.models import choose_calibrator  # noqa: E402
from msconf.pipeline import load_rows, train_decomposed, val_split  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
ev = cfg["evaluation"]
seed = cfg["project"]["seed"]
evalutil.set_thresholds(cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"])
PRIMARY = json.load(open(path(cfg, "results") / "selection.json"))["primary_system"]
PGROUPS = ["meta", "sim", "sim_extra", "analog", "crowd_spec", "crowd_struct"]  # the primary system's groups
NEW = {f"L_minus_{g}": [x for x in PGROUPS if x != g] for g in PGROUPS}
NEW.update({"L_path_sim_simx": ["sim", "sim_extra"], "L_path_sim_simx_analog": ["sim", "sim_extra", "analog"]})
PATH = [("B_lr_dec", "logistic(s1, gap), decomposed"), ("F_sim", "+ LightGBM on sim (s1, s2, s3, gaps)"),
        ("L_path_sim_simx", "+ sim_extra"), ("L_path_sim_simx_analog", "+ analog"), ("M_base", "+ meta (= no-crowding model)"),
        (PRIMARY, "+ crowd_spec + crowd_struct (= proposed)")]

t0 = time.time()
fam = pd.read_parquet(der / "families.parquet")
split = val_split(fam, dict(np.load(der / "struct_nbrs.npz")), cfg["models"]["val_cluster_tanimoto"], cfg["models"]["val_split_frac_calib"], seed)
rows = load_rows(der, "entropy")
rows = rows[rows.scenario.isin(["a", "b"])].copy()
rows["vsplit"] = rows.mol.map(split)
train = rows[rows.fold == "train"]
val = rows[rows.fold == "val"]
valA, valB = val[val.vsplit == "A"], val[val.vsplit == "B"]
vA_, vAB = val[val.scenario == "a"], val
mdir = der / "models" / "entropy_lead"
mdir.mkdir(parents=True, exist_ok=True)
log = {}
for name, groups in NEW.items():
    f = mdir / f"{name}.pkl"
    if f.exists():
        continue
    t1 = time.time()
    cols = [c for g in groups for c in GROUPS[g]]
    sysm, lg = train_decomposed(name, "lgbm", cols, train, valA, valB, cfg, seed)
    sysm.calC, iC = choose_calibrator(sysm.mC, vA_, "correct")  # same calibration stage as 08b_recalibrate.py
    sysm.calI, iI = choose_calibrator(sysm.mI, vAB, "in_lib")
    sysm.pi_cal = float(np.average(vAB.in_lib, weights=vAB.w))
    f.write_bytes(pickle.dumps(sysm))
    log[name] = {"groups": groups, "calibrators": {"C": iC["choice"], "I": iI["choice"]}, "seconds": time.time() - t1}
    print(name, f"{time.time()-t1:.0f}s", flush=True)

# ---------------------------------------------------------------- test evaluation
S = {n: pickle.loads((der / "models" / "entropy" / f"{n}.pkl").read_bytes()) for n in ["B_lr_dec", "F_sim", "M_base", PRIMARY]}
S.update({n: pickle.loads((mdir / f"{n}.pkl").read_bytes()) for n in NEW})
test = rows[rows.fold == "test"].reset_index(drop=True)
COMP = {n: components(s, test) for n, s in S.items()}
point = {}
for pi in ev["prevalence_headline"]:
    point[str(pi)] = {}
    for n, c in COMP.items():
        r = evaluate_weighted(c, test, pi, "em")
        point[str(pi)][n] = {k: r[k] for k in ["aurc", "e_aurc", "cal_logloss", "cal_ece", "auroc_correct", "auroc_in"]}


def fast_ll(p, y, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def metrics(ix, pi):
    d = test.iloc[ix]
    y, yin = d.correct.values.astype(float), d.in_lib.values.astype(float)
    w = class_weights(yin, d.w.values, pi)
    out = {}
    for n, c in COMP.items():
        sub = {"direct": c["direct"], "pc": c["pc"][ix], "pin_cal": c["pin_cal"][ix], "pi_cal": c["pi_cal"]}
        pc, _ = probs(sub, pi)
        out[n] = (risk_coverage(pc, y, w)["aurc"], fast_ll(pc, y, w))
    return out


rng = np.random.default_rng(seed + 23)
mol_idx = test.groupby("mol").indices
mkeys = np.array(list(mol_idx.keys()))
PAIRS = [(PATH[i + 1][0], PATH[i][0]) for i in range(len(PATH) - 1)] + [(PRIMARY, "B_lr_dec")] + [(PRIMARY, n) for n in NEW if n.startswith("L_minus_")] + [(PRIMARY, "M_base")]
boot = {}
for pi in ev["prevalence_headline"]:
    base = metrics(np.arange(len(test)), pi)
    draws = []
    for b in range(ev["bootstrap_repeats"]):
        pick = rng.integers(0, len(mkeys), len(mkeys))
        draws.append(metrics(np.concatenate([mol_idx[mkeys[i]] for i in pick]), pi))
    res = {}
    for a, c in PAIRS:
        da = np.array([d[a][0] - d[c][0] for d in draws])
        dl = np.array([d[a][1] - d[c][1] for d in draws])
        res[f"{a} - {c}"] = {"d_aurc": base[a][0] - base[c][0], "d_aurc_ci": np.percentile(da, [2.5, 97.5]).tolist(),
                             "d_logloss": base[a][1] - base[c][1], "d_logloss_ci": np.percentile(dl, [2.5, 97.5]).tolist()}
    lead_a = base["B_lr_dec"][0] - base[PRIMARY][0]
    lead_l = base["B_lr_dec"][1] - base[PRIMARY][1]
    steps = []
    for i in range(len(PATH) - 1):
        a, c = PATH[i + 1][0], PATH[i][0]
        steps.append({"step": PATH[i + 1][1], "from": c, "to": a, **res[f"{a} - {c}"],
                      "share_of_lead_aurc": (base[c][0] - base[a][0]) / lead_a, "share_of_lead_logloss": (base[c][1] - base[a][1]) / lead_l})
    boot[str(pi)] = {"point_oracle": {n: {"aurc": v[0], "logloss": v[1]} for n, v in base.items()}, "diffs": res, "path": steps,
                     "lead_aurc": lead_a, "lead_logloss": lead_l, "n_boot": len(draws)}
write_result(cfg, "eval_lead_attribution", {"post_hoc": True, "primary": PRIMARY, "new_systems": {k: v for k, v in NEW.items()},
                                            "training": log, "point_em": point, "boot": boot, "seconds": time.time() - t0})
print(json.dumps({pi: [(s["step"], round(s["d_aurc"], 5), round(s["share_of_lead_aurc"], 3), round(s["share_of_lead_logloss"], 3)) for s in b["path"]]
                  for pi, b in boot.items()}, indent=1))
