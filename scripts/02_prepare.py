"""Prepare derived data: processed spectra, standardized structures, fingerprints, fold audit.

Outputs (data/derived/):
  spectra.parquet   one row per MassSpecGym spectrum with metadata, processed-peak counts, keep flag
  peaks.npz         processed peaks (concatenated) + offsets, for kept and dropped spectra alike
  molecules.parquet one row per standardized molecule (ik14) with fold
  fps.npy           bit-packed Morgan fingerprints aligned with molecules.parquet
Results: results/prepare_audit.json
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.chem import Standardizer, pack_fps  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.spectra import parse_peaks, process, spectral_entropy  # noqa: E402

cfg = load_config()
raw = path(cfg, "raw") / "massspecgym"
der = path(cfg, "derived")
sp = cfg["spectrum_processing"]

df = pd.read_csv(raw / Path(cfg["data"]["files"]["spectra_tsv"]).name, sep="\t")
df["row"] = np.arange(len(df))
audit = {"n_spectra_raw": len(df), "adducts": df.adduct.value_counts().to_dict(),
         "instrument_type": df.instrument_type.fillna("NA").value_counts().to_dict(),
         "folds_spectra": df.fold.value_counts().to_dict()}

# ---------------- structures ----------------
std = Standardizer(cfg["structure"])
smi = df.drop_duplicates("inchikey")[["inchikey", "smiles"]]
recs = [std.record(s) for s in smi.smiles]
smi = smi.rename(columns={"inchikey": "ik14_provided"}).reset_index(drop=True)
smi = pd.concat([smi, pd.DataFrame(recs).rename(columns={"inchikey": "inchikey_full"})], axis=1)
n_fail = int(smi.ik14.isna().sum())
mism = smi[smi.ik14.notna() & (smi.ik14 != smi.ik14_provided)]
audit["structures"] = {"n_unique_provided_ik14": len(smi), "n_standardization_failed": n_fail,
                       "n_ik14_changed_by_standardization": len(mism),
                       "examples_changed": mism.head(10)[["ik14_provided", "ik14", "smiles", "std_smiles"]].to_dict("records")}
smi.loc[smi.ik14.isna(), "ik14"] = smi.loc[smi.ik14.isna(), "ik14_provided"]  # fall back to provided key (logged)
df = df.merge(smi[["ik14_provided", "ik14", "std_smiles", "inchikey_full"]], left_on="inchikey", right_on="ik14_provided", how="left")
df = df.sort_values("row").reset_index(drop=True)

# fold audit on standardized keys: a molecule must not span folds (failure mode #1)
span = df.groupby("ik14").fold.nunique()
bad = span[span > 1].index
audit["structures"]["n_std_molecules"] = int(df.ik14.nunique())
audit["structures"]["n_std_molecules_spanning_folds"] = int(len(bad))
if len(bad):
    # Stricter-than-official handling: drop spectra of molecules that collapse across folds.
    audit["structures"]["dropped_spectra_cross_fold_merge"] = int(df.ik14.isin(bad).sum())
    df = df[~df.ik14.isin(bad)].reset_index(drop=True)

mols = df.groupby("ik14").agg(fold=("fold", "first"), std_smiles=("std_smiles", "first"), inchikey_full=("inchikey_full", "first"), smiles=("smiles", "first"),
                              formula=("formula", "first"), n_spectra=("row", "size")).reset_index()
mols["mol_idx"] = np.arange(len(mols))
df = df.merge(mols[["ik14", "mol_idx"]], on="ik14", how="left")
audit["folds_molecules"] = mols.fold.value_counts().to_dict()

fps = []
for s in mols.std_smiles.fillna(mols.smiles):
    fp = std.fingerprint(s)
    fps.append(fp)
assert all(f is not None for f in fps)
np.save(der / "fps.npy", pack_fps(fps, cfg["structure"]["fingerprint"]["nbits"]))
mols.to_parquet(der / "molecules.parquet")

# ---------------- spectra ----------------
peaks_all, offs, npk_raw, npk, ent = [], [0], [], [], []
for mzs, its, pmz in zip(df.mzs.values, df.intensities.values, df.precursor_mz.values):
    p = parse_peaks(mzs, its)
    npk_raw.append(len(p))
    q = process(p, pmz, sp)
    peaks_all.append(q)
    offs.append(offs[-1] + len(q))
    npk.append(len(q))
    ent.append(spectral_entropy(q) if len(q) else 0.0)
cat = np.concatenate(peaks_all).astype(np.float32) if peaks_all else np.zeros((0, 2), np.float32)
np.savez(der / "peaks.npz", peaks=cat, offsets=np.array(offs, dtype=np.int64))
df["n_peaks_raw"] = npk_raw
df["n_peaks"] = npk
df["spec_entropy"] = ent
df["keep"] = df.n_peaks >= sp["min_peaks"]
audit["spectra"] = {"n_after_structure_audit": len(df), "n_kept_min_peaks": int(df.keep.sum()),
                    "frac_dropped_min_peaks": float(1 - df.keep.mean()),
                    "frac_dropped_by_fold": (1 - df.groupby("fold").keep.mean()).to_dict()}

# ---------------- precursor m/z accuracy check (to justify the tolerance a priori) ----------------
MASS = {"C": 12.0, "H": 1.00782503223, "N": 14.00307400443, "O": 15.99491461957, "P": 30.97376199842,
        "S": 31.9720711744, "F": 18.99840316273, "Cl": 34.968852682, "Br": 78.9183376, "I": 126.9044719,
        "Na": 22.989769282, "K": 38.9637064864, "Si": 27.97692653465, "B": 11.0093054, "Se": 79.9165218,
        "As": 74.92159457}
ELECTRON = 0.000548579909


def formula_mass(f):
    tot = 0.0
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if el not in MASS:
            return np.nan
        tot += MASS[el] * (int(n) if n else 1)
    return tot


theo = df.precursor_formula.map(formula_mass) - ELECTRON
ppm = (df.precursor_mz - theo) / theo * 1e6
audit["precursor_ppm_error"] = {"n_eval": int(ppm.notna().sum()),
                                "abs_quantiles": np.nanpercentile(ppm.abs(), [50, 90, 95, 99, 99.9]).tolist(),
                                "frac_within_tol": float((ppm.abs() <= cfg["engine"]["precursor_tol_ppm"]).mean())}
df["precursor_ppm_err"] = ppm

keep_cols = ["row", "identifier", "fold", "ik14", "mol_idx", "precursor_mz", "adduct", "instrument_type",
             "collision_energy", "simulation_challenge", "n_peaks_raw", "n_peaks", "spec_entropy", "keep", "precursor_ppm_err"]
df[keep_cols].to_parquet(der / "spectra.parquet")
write_result(cfg, "prepare_audit", audit)
print(pd.Series(audit["structures"]).drop("examples_changed"))
print(audit["spectra"]); print(audit["precursor_ppm_error"])
