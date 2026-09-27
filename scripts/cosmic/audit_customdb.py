"""Audits of the same-candidate-set COSMIC comparison (owner's questions, 2026-09-27).

1. Database pairs: (a) minus (b) must be exactly the subset's query molecules of that split
   (test: 300 molecules; validation: 200 molecules, one spectrum per molecule).
2. Tautomers. The (b) databases and our (b) library remove the query molecules by our identity key
   (InChIKey first block after RDKit standardization), which is NOT tautomer-insensitive. Using the
   tautomer-insensitive key of scripts/20_tautomer_keys.py (RDKit canonical tautomer -> InChIKey first block):
   - how many molecules left in each (b) database are tautomers of a removed query molecule;
   - for every (b) row, whether either system's top-1 is a tautomer of the true molecule;
   - for every (a) row, whether a "wrong" top-1 is a tautomer of the truth.
Output: results/cosmic_customdb_audit.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosmic_common import ROOT, cfg, der  # noqa: E402
from msconf.config import path, write_result  # noqa: E402

man = pd.read_csv(ROOT / "cosmic" / "subset_manifest.csv")
Q = pd.read_parquet(der / "queries.parquet")[["identifier", "mol_idx"]]
man = man.merge(Q, on="identifier", how="left")
mols = pd.read_parquet(der / "molecules.parquet").set_index("mol_idx")
tk = pd.read_parquet(der / "ext" / "tautomer_keys.parquet")
tk = tk[tk.fold != "ext"].set_index("mol_idx")
# the MassSpecGym part of the external molecule table must use the same indices and keys
common = mols.index.intersection(tk.index)
assert (mols.loc[common, "ik14"].values == tk.loc[common, "ik14"].values).all(), "molecule index mismatch"
TK = tk.tkey
IK2TK = dict(zip(tk.ik14, tk.tkey))
out = {"identity_key_used_for_removal": "InChIKey first block after RDKit standardization (Cleanup -> LargestFragment -> Uncharge); not tautomer-insensitive",
       "tautomer_key": "RDKit TautomerEnumerator canonical tautomer -> InChIKey first block (scripts/20_tautomer_keys.py)"}

# ---------------------------------------------------------------- 1. database pairs
db = {k: pd.read_csv(ROOT / "cosmic" / "customdb" / f"db_{k}.tsv", sep="\t", header=None, names=["smiles", "ik14", "name"])
      for k in ["test_a", "test_b", "val_a", "val_b"]}
for split in ["test", "val"]:
    a, b = set(db[f"{split}_a"].ik14), set(db[f"{split}_b"].ik14)
    qmol = man[man.split == split].mol_idx
    qik = set(mols.loc[qmol, "ik14"])
    out[f"db_pair_{split}"] = {"n_a": len(a), "n_b": len(b), "a_minus_b": len(a - b), "b_minus_a": len(b - a),
                               "n_query_spectra": int(len(qmol)), "n_distinct_query_molecules": int(qmol.nunique()),
                               "a_minus_b_equals_query_molecules": (a - b) == qik}

# ---------------------------------------------------------------- 2a. tautomers of removed molecules left in (b)
for split in ["test", "val"]:
    qmol = man[man.split == split].mol_idx
    qtk = set(TK.loc[qmol])
    b = db[f"{split}_b"]
    btk = b.ik14.map(IK2TK)
    hit = b[btk.isin(qtk)]
    out[f"{split}_b_molecules_tautomer_equal_to_a_query_molecule"] = {"n": int(len(hit)), "ik14": hit.ik14.tolist()}

# ---------------------------------------------------------------- 2b/2c. top-1 vs truth under the tautomer key
rows = pd.read_csv(path(cfg, "tables") / "cosmic_customdb_rows.csv")
rows["true_tk"] = rows.mol_idx.map(TK)
rows["cosmic_tk"] = rows.hit_ik14.map(IK2TK)
rows["ours_tk"] = rows.top1.map(lambda t: TK.get(int(t)) if pd.notna(t) else None)
res = {}
for sc in ["a", "b"]:
    r = rows[rows.scenario == sc]
    res[sc] = {"cosmic_top1_tautomer_of_truth_but_not_identical": int(((r.cosmic_tk == r.true_tk) & (r.cosmic_correct != 1)).sum()),
               "proposed_top1_tautomer_of_truth_but_not_identical": int(((r.ours_tk == r.true_tk) & (r.correct != 1) & r.top1.notna()).sum()),
               "identifiers": r[((r.cosmic_tk == r.true_tk) & (r.cosmic_correct != 1)) | ((r.ours_tk == r.true_tk) & (r.correct != 1) & r.top1.notna())].identifier.tolist()}
out["top1_vs_truth_tautomer_key"] = res

# ---------------------------------------------------------------- 2d. validation (COSMIC calibration rows)
from cosmic_common import read_summaries  # noqa: E402
from msconf.chem import Standardizer  # noqa: E402

std = Standardizer(cfg["structure"])
vres = {}
for sc in ["a", "b"]:
    files = sorted((ROOT / "cosmic" / "customdb" / "results").glob(f"summaries_val_*_{sc}/structure_identifications.tsv"))
    S = read_summaries(files, std).merge(man[["identifier", "mol_idx"]], on="identifier", how="left")
    S["hit_tk"] = S.sirius_ik14.map(IK2TK)
    S["true_tk"] = S.mol_idx.map(TK)
    S["true_ik"] = S.mol_idx.map(mols.ik14)
    m = (S.hit_tk == S.true_tk) & (S.sirius_ik14 != S.true_ik)
    vres[sc] = {"cosmic_top1_tautomer_of_truth_but_not_identical": int(m.sum()), "identifiers": S[m].identifier.tolist()}
out["validation_cosmic_top1_vs_truth_tautomer_key"] = vres
write_result(cfg, "cosmic_customdb_audit", out)
print(json.dumps(out, indent=1, default=str))
