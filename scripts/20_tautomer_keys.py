"""Tautomer-insensitive structure keys (RDKit canonical tautomer -> InChIKey first block) for every
MassSpecGym and external molecule. Used for (i) cross-library de-duplication of the external source set
(failure mode #11), (ii) a leakage audit of the official folds, (iii) a supplementary tautomer-insensitive
accuracy. The primary identity definition (§4, InChIKey first block) is NOT changed.
Output: data/derived/ext/tautomer_keys.parquet, results/tautomer_audit.json
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.chem import Standardizer  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402

RDLogger.DisableLog("rdApp.*")
cfg = load_config()
der = path(cfg, "derived")
mols = pd.read_parquet(der / "ext" / "molecules.parquet")
STD = Standardizer(cfg["structure"])


def tkey(smi):
    try:
        m = STD.mol(smi)
        if m is None:
            return None
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(256)
        te.SetMaxTransforms(512)
        c = te.Canonicalize(m)
        k = Chem.MolToInchiKey(c)
        return k[:14] if k else None
    except Exception:
        return None


if __name__ == "__main__":
    t0 = time.time()
    smis = mols.std_smiles.fillna(mols.smiles).tolist()
    with Pool(2) as p:
        keys = p.map(tkey, smis, chunksize=200)
    mols["tkey"] = keys
    mols["tkey"] = mols.tkey.fillna(mols.ik14)
    mols[["mol_idx", "ik14", "tkey", "fold"]].to_parquet(der / "ext" / "tautomer_keys.parquet")
    msg = mols[mols.fold != "ext"]
    g = msg.groupby("tkey").agg(n=("ik14", "nunique"), folds=("fold", "nunique"))
    ext = mols[mols.fold == "ext"]
    ge = ext.groupby("tkey").ik14.nunique()
    audit = {"seconds": time.time() - t0,
             "massspecgym_tkeys_with_multiple_ik14": int((g.n > 1).sum()),
             "massspecgym_molecules_in_such_groups": int(g[g.n > 1].n.sum()),
             "massspecgym_tautomer_groups_spanning_folds": int(((g.n > 1) & (g.folds > 1)).sum()),
             "external_molecules_tautomer_equal_to_massspecgym": int(ext.tkey.isin(set(msg.tkey)).sum()),
             "external_molecules_tautomer_equal_to_other_external": int(ext.tkey.map(ge).gt(1).sum()),
             "n_external_molecules": int(len(ext))}
    write_result(cfg, "tautomer_audit", audit)
    print(audit)
