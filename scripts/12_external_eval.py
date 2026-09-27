"""Data-source shift: frozen systems (trained on MassSpecGym train, calibrated on MassSpecGym val-B) applied
to external MassBank-2026.03 queries whose molecules are absent from MassSpecGym and have Tanimoto < 0.7
to every train molecule. Reported overall and leave-one-source (per MassBank contributor)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf import evalutil  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.evalutil import components, evaluate_weighted, probs  # noqa: E402
from msconf.evaluation import assign_tiers, wilson  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
import os  # noqa: E402

VARIANT = os.environ.get("MSCONF_EXT_VARIANT", "v1")
ext = der / ("ext" if VARIANT == "v1" else f"ext_{VARIANT}")
sel = json.load(open(path(cfg, "results") / "selection.json"))
PRIMARY = sel["primary_system"]
evalutil.set_thresholds(cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"])
rows = pd.read_parquet(ext / "rows_entropy.parquet")
spec = pd.read_parquet(ext / "spectra.parquet")
all_ext = rows[rows.fold == "ext"]
pop = {sc: {"n_rows": int((all_ext.scenario == sc).sum()), "frac_no_candidate": float(1 - all_ext[all_ext.scenario == sc].has_candidate.mean()),
            "n_mol": int(all_ext[all_ext.scenario == sc].mol.nunique())} for sc in ["a", "b"]}
te = all_ext[(all_ext.has_candidate == 1) & all_ext.scenario.isin(["a", "b"])].reset_index(drop=True)
te["contributor"] = te.row.map(spec.set_index("row").contributor)
res = {"primary": PRIMARY, "population": pop, "n_rows": int(len(te)), "n_mol": int(te.mol.nunique()),
       "top1_acc_a": float(np.average(te[te.scenario == "a"].correct, weights=te[te.scenario == "a"].w)),
       "contributors": te.groupby("contributor").mol.nunique().to_dict()}
out = {}
for name in [PRIMARY, "M_full", "M_base", "B_lr_dec", "B_lr_direct", "B_s1_direct", "F_sim", "F_meta"]:
    f = der / "models" / "entropy" / f"{name}.pkl"
    if not f.exists() or name in out:
        continue
    comp = components(pickle.loads(f.read_bytes()), te)
    out[name] = {str(pi): {m: evaluate_weighted(comp, te, pi, m) for m in ["oracle", "em"]} for pi in cfg["evaluation"]["prevalence_headline"]}
    if name == PRIMARY:
        # leave-one-source: per contributor with enough molecules; everything else pooled as 'other'
        vc = te.groupby("contributor").mol.nunique()
        big = vc[vc >= 30].index
        te["source"] = np.where(te.contributor.isin(big), te.contributor, "other")
        per = {}
        for src, g in te.groupby("source"):
            g = g.reset_index(drop=True)
            if g.in_lib.nunique() < 2:
                continue
            c = components(pickle.loads(f.read_bytes()), g)
            per[src] = {"n_mol": int(g.mol.nunique()), **{str(pi): evaluate_weighted(c, g, pi, "oracle") for pi in [0.5]}}
        out["_per_source_primary"] = per
res["systems"] = out

# paired molecule-cluster bootstrap: primary minus each other system (AURC, log loss; oracle prevalence)
from msconf.evalutil import class_weights  # noqa: E402
from msconf.evaluation import risk_coverage  # noqa: E402

COMP = {n: components(pickle.loads((der / "models" / "entropy" / f"{n}.pkl").read_bytes()), te)
        for n in [PRIMARY, "M_base", "B_lr_dec", "B_lr_direct", "B_s1_direct", "F_sim"] if (der / "models" / "entropy" / f"{n}.pkl").exists()}
mi = te.groupby("mol").indices
mk = np.array(list(mi))
rng = np.random.default_rng(cfg["project"]["seed"])


def _ll(p, y, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def _met(ix, pi):
    d = te.iloc[ix]
    y, yin = d.correct.values.astype(float), d.in_lib.values.astype(float)
    if yin.sum() == 0 or yin.sum() == len(yin):
        return None
    w = class_weights(yin, d.w.values, pi)
    o = {}
    for n, c in COMP.items():
        pc, _ = probs({"direct": c["direct"], "pc": c["pc"][ix], "pin_cal": c["pin_cal"][ix], "pi_cal": c["pi_cal"]}, pi)
        o[n] = (risk_coverage(pc, y, w)["e_aurc"], _ll(pc, y, w))
    return o


res["bootstrap"] = {}
for pi in cfg["evaluation"]["prevalence_headline"]:
    pt = _met(np.arange(len(te)), pi)
    dr = [x for x in (_met(np.concatenate([mi[k] for k in mk[rng.integers(0, len(mk), len(mk))]]), pi) for _ in range(1000)) if x]
    res["bootstrap"][str(pi)] = {f"{PRIMARY} - {n}": {"d_e_aurc": pt[PRIMARY][0] - pt[n][0],
                                                     "d_e_aurc_ci": np.percentile([x[PRIMARY][0] - x[n][0] for x in dr], [2.5, 97.5]).tolist(),
                                                     "d_logloss": pt[PRIMARY][1] - pt[n][1],
                                                     "d_logloss_ci": np.percentile([x[PRIMARY][1] - x[n][1] for x in dr], [2.5, 97.5]).tolist()}
                                 for n in COMP if n != PRIMARY}

# anatomy of confident errors: wrong top-1 with proposed p_correct >= trusted threshold (pi = 0.5, oracle)
from rdkit import Chem  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

mols = pd.read_parquet(ext / "molecules.parquet").set_index("mol_idx")
smi = mols.std_smiles.fillna(mols.smiles)


def formula(m):
    try:
        return CalcMolFormula(Chem.MolFromSmiles(smi.loc[m]))
    except Exception:
        return None


c = COMP[PRIMARY]
pc, _ = probs(c, 0.5)
bad = te[(pc >= cfg["tiers"]["trusted_min_p_correct"]) & (te.correct.values == 0)]
if len(bad):
    same_formula = np.mean([formula(a) == formula(b) for a, b in zip(bad.top1.astype(int), bad.mol)])
    res["confident_errors_pi0.5"] = {"n": int(len(bad)), "n_mol": int(bad.mol.nunique()), "frac_scenario_b": float((bad.scenario == "b").mean()),
                                     "frac_same_formula_as_truth": float(same_formula),
                                     "frac_s1_ge_0.95": float((bad.s1 >= 0.95).mean()),
                                     "median_tani_top1_truth": float(bad.ana_tani_top1_truth.median()),
                                     "frac_top1_from_external_source": float(np.mean(mols.loc[bad.top1.astype(int), "fold"].values == "ext")),
                                     "by_contributor": bad.contributor.value_counts().head(8).to_dict()}
res["variant"] = VARIANT
write_result(cfg, "eval_external" + ("" if VARIANT == "v1" else "_" + VARIANT), res)
print(json.dumps({k: v for k, v in res.items() if k != "systems"}, indent=1))
