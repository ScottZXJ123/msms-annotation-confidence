"""Train-fold diagnostic behind the precursor tolerance (config engine.precursor_tol_ppm; LOG 2026-09-26 19:58).

Rule (fixed before looking): the smallest tolerance in {5, 10, 15, 20, 30} ppm that covers >= 99% of the
|delta m/z| between spectra of the same molecule and the same adduct in the TRAIN fold.
Re-computation written after the fact (2026-09-27): the 19:58 numbers in LOG.md came from an ad hoc computation
whose code was not kept and whose exact definition could not be reconstructed. Two definitions are reported:
  all_pairs: within each (molecule, adduct) group of kept train spectra, every spectrum against every other one
             (groups larger than 200 spectra subsampled to 200, fixed seed); ppm = |mz_i - mz_j| / mz_j * 1e6;
  nearest:   for each spectrum, the nearest other spectrum of the same molecule and adduct (what matters for
             retrieving the truth in scenario (a)).
Output: results/dev_precursor_tolerance.json
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
sp = pd.read_parquet(der / "spectra.parquet")
sp = sp[(sp.fold == "train") & sp.keep]
rng = np.random.default_rng(cfg["project"]["seed"])
vals = []
for _, g in sp.groupby(["mol_idx", "adduct"]):
    mz = g.precursor_mz.values
    if len(mz) < 2:
        continue
    if len(mz) > 200:
        mz = rng.choice(mz, 200, replace=False)
    d = np.abs(mz[:, None] - mz[None, :]) / mz[None, :] * 1e6
    vals.append(d[~np.eye(len(mz), dtype=bool)])
v = np.concatenate(vals)
nn = []
for _, g in sp.groupby(["mol_idx", "adduct"]):
    mz = np.sort(g.precursor_mz.values)
    if len(mz) < 2:
        continue
    d = np.diff(mz)
    nn.append(np.minimum(np.r_[np.inf, d], np.r_[d, np.inf]) / mz * 1e6)
nn = np.concatenate(nn)
grid = [5, 10, 15, 20, 30]
cov = {str(t): float((v <= t).mean()) for t in grid}
cov_nn = {str(t): float((nn <= t).mean()) for t in grid}
chosen = next((t for t in grid if cov[str(t)] >= 0.99), None)
res = {"n_pairs": int(len(v)), "n_groups": int(len(vals)), "coverage_all_pairs": cov, "coverage_nearest": cov_nn,
       "rule": "smallest tolerance with >= 99% coverage", "chosen_ppm_all_pairs": chosen,
       "chosen_ppm_nearest": next((t for t in grid if cov_nn[str(t)] >= 0.99), None),
       "config_ppm": cfg["engine"]["precursor_tol_ppm"]}
write_result(cfg, "dev_precursor_tolerance", res)
print(res)
