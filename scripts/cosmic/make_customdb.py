"""Head-to-head COSMIC comparison on the SAME candidate universe as our spectral library.

Builds SIRIUS custom structure databases from the molecules of our reference library, so CSI:FingerID/COSMIC
searches the same molecules our library search can return:
  test_a : molecules of the test-phase library (train+val+test folds; kept spectra)    -> scenario (a), truth present
  test_b : test_a minus ALL 300 test query molecules of the COSMIC subset              -> scenario (b), truth absent
  val_a  : molecules of the validation-phase library (train+val folds)                  -> calibration, (a)
  val_b  : val_a minus ALL 200 validation query molecules                               -> calibration, (b)
On our side the same removals are applied (for (b), all subset query molecules are removed at once), so both
systems face one identical candidate universe per scenario.
Size reduction without changing any retrieval: CSI:FingerID only retrieves database structures whose molecular
formula matches a formula candidate of the query, i.e. whose mass matches the query's neutral mass. Molecules
whose monoisotopic mass is farther than 0.05 Da from every query neutral mass of the split can never be retrieved
and are left out (0.05 Da exceeds SIRIUS's 5-10 ppm formula tolerance for every precursor here). The (a) databases
always contain every query molecule, also the rare one whose recorded precursor m/z is > 0.05 Da off (counted in
the manifest; SIRIUS cannot propose its formula, and the query counts as unannotated/wrong for COSMIC).
Output (public structures only): cosmic/customdb/db_<name>.tsv (SMILES <tab> id <tab> name, no header),
cosmic/customdb/db_manifest.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.config import ROOT, load_config, path  # noqa: E402

RDLogger.DisableLog("rdApp.*")
cfg = load_config()
der = path(cfg, "derived")
out = ROOT / "cosmic" / "customdb"
out.mkdir(parents=True, exist_ok=True)
ADDUCT = {"[M+H]+": 1.007276467, "[M+Na]+": 22.989218}
TOL = 0.05

man = pd.read_csv(ROOT / "cosmic" / "subset_manifest.csv")
Q = pd.read_parquet(der / "queries.parquet")[["identifier", "mol_idx"]]
man = man.merge(Q, on="identifier", how="left")
lib = pd.read_parquet(der / "lib_index.parquet")
mols = pd.read_parquet(der / "molecules.parquet").set_index("mol_idx")
smi = mols.std_smiles.fillna(mols.smiles)
mass = {}
for m in lib.mol_idx.unique():
    mo = Chem.MolFromSmiles(smi.loc[m])
    mass[m] = ExactMolWt(mo) if mo is not None else np.nan
mass = pd.Series(mass)
phase = {"test": ["train", "val", "test"], "val": ["train", "val"]}
manifest = {}
for split in ["test", "val"]:
    q = man[man.split == split]
    neutral = np.sort((q.precursor_mz - q.adduct.map(ADDUCT)).values)
    lib_m = pd.Index(lib[lib.fold.isin(phase[split])].mol_idx.unique())
    mm = mass.loc[lib_m].dropna()
    pos = np.searchsorted(neutral, mm.values)
    d = np.minimum(np.abs(mm.values - neutral[np.clip(pos - 1, 0, len(neutral) - 1)]),
                   np.abs(mm.values - neutral[np.clip(pos, 0, len(neutral) - 1)]))
    keep = mm.index[d <= TOL]
    forced = sorted(set(q.mol_idx) - set(keep))  # query molecules whose recorded precursor m/z is off by > TOL
    keep = keep.union(pd.Index(forced))           # the (a) database always contains every query molecule
    for sc, mset in [("a", keep), ("b", keep[~keep.isin(q.mol_idx)])]:
        df = pd.DataFrame({"smiles": smi.loc[mset].values, "id": mols.loc[mset, "ik14"].values, "name": [f"msg_mol_{m}" for m in mset]})
        df.to_csv(out / f"db_{split}_{sc}.tsv", sep="\t", header=False, index=False)
        manifest[f"{split}_{sc}"] = {"n_molecules": int(len(df)), "n_query_molecules_included": int(q.mol_idx.isin(mset).sum()),
                                     "phase_folds": phase[split], "mass_window_da": TOL,
                                     "query_molecules_outside_window_added": len(forced) if sc == "a" else 0}
json.dump(manifest, open(out / "db_manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
