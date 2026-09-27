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
ext = der / "ext"
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
write_result(cfg, "eval_external", res)
print(json.dumps({k: v for k, v in res.items() if k != "systems"}, indent=1))
