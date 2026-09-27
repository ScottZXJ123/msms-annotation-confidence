"""Automated leakage / integrity audits. Fails loudly if any check fails; writes results/checks.json."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import ROOT, load_config, path, write_result  # noqa: E402
from msconf.models import CANDIDATE_ONLY, FEATURE_SETS, columns  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
checks = {}


def ok(name, cond, detail=None):
    checks[name] = {"pass": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name, "" if detail is None else detail)


mols = pd.read_parquet(der / "molecules.parquet")
fold_of = mols.set_index("mol_idx").fold
rows = pd.read_parquet(der / "rows_entropy.parquet")
r = rows[rows.has_candidate == 1]

# 1. molecule-level disjointness of query folds
f = rows.groupby("mol").fold.nunique()
ok("query_molecules_single_fold", (f == 1).all(), int((f > 1).sum()))

# 2. candidates for train/val queries never come from later folds (nested libraries)
allowed = {"train": {"train"}, "val": {"train", "val"}, "test": {"train", "val", "test"}}
bad = 0
for fo, g in r.groupby("fold"):
    c = np.concatenate([np.array(s.split(","), dtype=int) for s in g.top5.values])
    bad += int((~fold_of.loc[c].isin(allowed[fo]).values).sum())
ok("candidates_respect_nested_libraries", bad == 0, bad)

# 3. the query spectrum itself is never in its own library (scenario a): the truth's spectrum count in the
#    window equals (#truth spectra in window in the phase library) - 1
Q = pd.read_parquet(der / "queries.parquet")
L = np.load(der / "qlists.npz")
lib = pd.read_parquet(der / "lib_index.parquet")
ppm = cfg["engine"]["precursor_tol_ppm"]
rng = np.random.default_rng(0)
viol = 0
phase = {k: set(v) for k, v in cfg["scenarios"]["phases"].items()}
for i in rng.choice(len(Q), 2000, replace=False):
    q = Q.iloc[i]
    w0, w1 = L["W_ptr"][i], L["W_ptr"][i + 1]
    wm, wn = L["W_mol"][w0:w1], L["W_n"][w0:w1]
    n_stored = int(wn[wm == q.mol_idx].sum())
    same = lib[(lib.mol_idx == q.mol_idx) & lib.fold.isin(phase[q.fold])]
    tol = q.precursor_mz * ppm * 1e-6
    n_true = int((np.abs(same.precursor_mz - q.precursor_mz) <= tol).sum())
    viol += int(n_stored != n_true - 1)
ok("query_spectrum_excluded_from_own_library", viol == 0, f"{viol} violations in 2000 sampled queries")

# 4. scenario (b)/(c): truth never among candidates
ok("truth_absent_in_b_c", int(r[r.scenario.isin(["b", "c"])].in_lib.sum()) == 0)
ok("truth_present_in_a", bool(r[r.scenario == "a"].in_lib.all()))

# 5. no truth-derived column can be a feature
cols = set(CANDIDATE_ONLY)
for s in FEATURE_SETS:
    cols |= set(columns(s))
leaky = [c for c in cols if c.startswith("ana_") or c in {"mol", "correct", "in_lib", "scenario", "top1", "top5", "qid", "row", "fold"}]
ok("no_truth_columns_in_feature_sets", not leaky, leaky)

# 6. training/selection scripts never read the test fold
src = (ROOT / "scripts" / "08_train.py").read_text()
ok("train_script_drops_test_fold", 'rows = rows[rows.fold != "test"]' in src)

# 7. lab data protections
ig = subprocess.run(["git", "check-ignore", "-q", "data/lab/example.mzML"], cwd=ROOT).returncode == 0
ok("data_lab_is_git_ignored", ig)
tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT).decode().splitlines()
ok("no_lab_files_tracked", not any(t.startswith("data/lab") for t in tracked))

# 8. no AI co-author trailers in history
log = subprocess.check_output(["git", "log", "--format=%an %ae%n%B"], cwd=ROOT).decode()
ok("no_ai_coauthor_in_git_history", not re.search(r"co-authored-by:.*(claude|anthropic)|noreply@anthropic", log, re.I))

# 9. paired design: query-side metadata identical between the (a) and (b) versions of each query
qa = r[r.scenario == "a"].set_index("qid")
qb = r[r.scenario == "b"].set_index("qid")
common = qa.index.intersection(qb.index)
qcols = ["precursor_mz", "adduct_na", "instrument", "collision_energy", "n_peaks", "spec_entropy"]
same = all(np.allclose(qa.loc[common, c].fillna(-1).values, qb.loc[common, c].fillna(-1).values) for c in qcols)
ok("paired_a_b_rows_share_query_metadata", same, f"{len(common)} paired queries")
try:
    sw = json.load(open(path(cfg, "results") / "eval_sweep.json"))["sweep"]
    checks["info_metadata_only_floor_auroc_in_library_pi0.5"] = {"pass": True, "detail": sw["F_meta"]["0.5"]["oracle"]["auroc_in"]}
    checks["info_metadata_only_floor_auroc_correct_pi0.5"] = {"pass": True, "detail": sw["F_meta"]["0.5"]["oracle"]["auroc_correct"]}
except FileNotFoundError:
    pass

write_result(cfg, "checks", checks)
if not all(v["pass"] for v in checks.values()):
    sys.exit(1)
