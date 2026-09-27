"""Build the COSMIC (SIRIUS 6) input package in cosmic/ (public MassSpecGym spectra only).

Test subset: 300 test-fold molecules (one query spectrum each; Orbitrap/QTOF), stratified by whether the
molecule's InChIKey first block is in the published CSI:FingerID positive-mode training structures
(https://csi.bright-giant.com/v3.0/api/fingerid/trainingstructures?predictor=1): 150 in the list
("leaky reference" stratum) + 150 not in the list (primary stratum).
Calibration subset: 200 validation-fold molecules NOT in the list (for calibrating COSMIC scores with the same
validation-only rule used for every other system).
Spectra are the raw MassSpecGym peak lists (SIRIUS does its own processing); no structure or formula is written
into the input files. Written as multi-compound .ms files, one per (split, instrument) so SIRIUS can use the
matching instrument profile.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.config import ROOT, load_config, path  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
seed = cfg["project"]["seed"] + 505
out = ROOT / "cosmic"
(out / "input").mkdir(parents=True, exist_ok=True)
train_ik = set(pd.read_csv(der / "csi_training_ik14.csv").ik14)
mols = pd.read_parquet(der / "molecules.parquet").set_index("mol_idx")
q = pd.read_parquet(der / "queries.parquet")
q = q[q.instrument_type.isin(["Orbitrap", "QTOF"])].copy()
q["ik14"] = q.mol_idx.map(mols.ik14)
q["in_csi_training"] = q.ik14.isin(train_ik)
rng = np.random.default_rng(seed)


def pick(df, n):
    m = df.mol_idx.unique()
    m = rng.choice(m, size=min(n, len(m)), replace=False)
    sub = df[df.mol_idx.isin(m)]
    return sub.groupby("mol_idx", group_keys=False).apply(lambda g: g.iloc[[rng.integers(len(g))]])


te = q[q.fold == "test"]
sel = pd.concat([pick(te[te.in_csi_training], 150).assign(split="test"),
                 pick(te[~te.in_csi_training], 150).assign(split="test"),
                 pick(q[(q.fold == "val") & ~q.in_csi_training], 200).assign(split="val")], ignore_index=True)
raw = pd.read_csv(path(cfg, "raw") / "massspecgym" / "MassSpecGym1.5.tsv", sep="\t",
                  usecols=["identifier", "mzs", "intensities"]).set_index("identifier")
for (split, inst), g in sel.groupby(["split", "instrument_type"]):
    fn = out / "input" / f"{split}_{inst.lower()}.ms"
    with open(fn, "w") as f:
        for _, r in g.iterrows():
            pk = raw.loc[r.identifier]
            f.write(f">compound {r.identifier}\n>parentmass {r.precursor_mz:.6f}\n>ionization {r.adduct}\n")
            f.write(f">instrumentation {inst}\n")
            f.write(f">collision {r.collision_energy:g}\n" if pd.notna(r.collision_energy) else ">ms2\n")
            for mz, it in zip(pk.mzs.split(","), pk.intensities.split(",")):
                f.write(f"{mz} {it}\n")
            f.write("\n")
    print(fn.name, len(g))
man = sel[["identifier", "split", "fold", "instrument_type", "adduct", "precursor_mz", "collision_energy", "in_csi_training"]]
man.to_csv(out / "subset_manifest.csv", index=False)
print(man.groupby(["split", "in_csi_training", "instrument_type"]).size())
