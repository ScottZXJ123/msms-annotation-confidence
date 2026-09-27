"""Export a seeded random subset of TEST-fold query spectra (one per molecule) as MGF for SIRIUS/COSMIC.

Public MassSpecGym spectra only (allowed to be sent to the SIRIUS web service, §6). Lab data must never be
exported with this script. Output: data/derived/cosmic/subset.mgf and subset_ids.csv (not the truth labels;
labels are joined only in ingest_cosmic.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.config import load_config, path  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
out = der / "cosmic"
out.mkdir(exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
q = pd.read_parquet(der / "queries.parquet")
q = q[q.fold == "test"]
rng = np.random.default_rng(cfg["project"]["seed"])
mols = rng.choice(q.mol_idx.unique(), size=min(N, q.mol_idx.nunique()), replace=False)
pick = q[q.mol_idx.isin(mols)].groupby("mol_idx").sample(1, random_state=cfg["project"]["seed"])
raw = pd.read_csv(path(cfg, "raw") / "massspecgym" / "MassSpecGym1.5.tsv", sep="\t", usecols=["identifier", "mzs", "intensities"])
raw = raw.set_index("identifier")
with open(out / "subset.mgf", "w") as f:
    for _, r in pick.iterrows():
        pk = raw.loc[r.identifier]
        f.write("BEGIN IONS\n")
        f.write(f"FEATURE_ID={r.identifier}\nTITLE={r.identifier}\nPEPMASS={r.precursor_mz}\nCHARGE=1+\n")
        f.write(f"ION={r.adduct}\nMSLEVEL=2\n")
        for mz, it in zip(pk.mzs.split(","), pk.intensities.split(",")):
            f.write(f"{mz} {it}\n")
        f.write("END IONS\n\n")
pick[["identifier", "qid" if "qid" in pick else "pos"]].to_csv(out / "subset_ids.csv", index=False)
print("wrote", len(pick), "spectra to", out / "subset.mgf")
