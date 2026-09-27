"""External data-source shift set from MassBank (release 2026.03, Zenodo 10.5281/zenodo.19590880).

Construction (all choices logged):
  * positive-mode MS2 records, precursor type [M+H]+ or [M+Na]+, high-resolution instrument types that
    map onto MassSpecGym's categories (QTOF: *QTOF*; Orbitrap: *ITFT*, *QFT*, *Orbitrap*); deprecated
    records dropped;
  * same RDKit standardization and the same spectrum processing as MassSpecGym;
  * cross-library de-duplication at the molecule level: records whose standardized InChIKey first
    block occurs anywhere in MassSpecGym (any fold) are removed, so every external query molecule is
    new to both the confidence model and the MassSpecGym library;
  * structural distance proxy to the training fold: molecules with Tanimoto >= 0.7 (Morgan r2/2048)
    to any MassSpecGym TRAIN molecule are removed (MCES was not computed for this set — stated in REPORT);
  * the external spectra are ADDED to the reference library (as a new source), so scenario (a) exists
    when a molecule has another external spectrum in the precursor window.
Outputs in data/derived/ext/: spectra.parquet, peaks.npz, molecules.parquet, fps.npy (MassSpecGym first,
external appended; MassSpecGym indices unchanged), plus results/external_prepare.json.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.chem import Standardizer, pack_fps, unpack_to_bitvects  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.spectra import process, spectral_entropy  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
import os  # noqa: E402

VARIANT = os.environ.get("MSCONF_EXT_VARIANT", "v1")  # v1: InChIKey-block dedup; v2: + canonical-tautomer dedup;
# v3: + removal of external molecules having an (almost) identical spectrum (entropy >= 0.99, same precursor
# window) of a DIFFERENT MassSpecGym molecule (cross-library duplicate deposits / indistinguishable isomers)
ext = der / ("ext" if VARIANT == "v1" else f"ext_{VARIANT}")
ext.mkdir(exist_ok=True)
zf = zipfile.ZipFile(path(cfg, "raw") / "massbank" / "MassBank-data-2026.03.zip")


def inst_cat(t):
    t = (t or "").upper()
    if "QTOF" in t or "Q-TOF" in t:
        return "QTOF"
    if "ITFT" in t or "QFT" in t or "ORBITRAP" in t:
        return "Orbitrap"
    return None


def parse(txt):
    r = {"peaks": []}
    inpk = False
    for line in txt.splitlines():
        if inpk:
            if line.startswith("  "):
                parts = line.split()
                if len(parts) >= 2:
                    r["peaks"].append((float(parts[0]), float(parts[1])))
                continue
            inpk = False
        if line.startswith("ACCESSION:"):
            r["accession"] = line.split(":", 1)[1].strip()
        elif line.startswith("DEPRECATED:"):
            r["deprecated"] = True
        elif line.startswith("DATE:"):
            r["date"] = line.split(":", 1)[1].strip()
        elif line.startswith("CH$SMILES:"):
            r["smiles"] = line.split(":", 1)[1].strip()
        elif line.startswith("AC$INSTRUMENT_TYPE:"):
            r["inst_raw"] = line.split(":", 1)[1].strip()
        elif line.startswith("AC$MASS_SPECTROMETRY: MS_TYPE"):
            r["ms_type"] = line.split()[-1]
        elif line.startswith("AC$MASS_SPECTROMETRY: ION_MODE"):
            r["ion_mode"] = line.split()[-1]
        elif line.startswith("AC$MASS_SPECTROMETRY: COLLISION_ENERGY"):
            m = re.search(r"(\d+(?:\.\d+)?)", line.split("COLLISION_ENERGY", 1)[1])
            r["ce_raw"] = line.split("COLLISION_ENERGY", 1)[1].strip()
            r["collision_energy"] = float(m.group(1)) if m and "%" not in line and "NCE" not in line.upper() else np.nan
        elif line.startswith("MS$FOCUSED_ION: PRECURSOR_M/Z"):
            try:
                r["precursor_mz"] = float(line.split()[-1])
            except ValueError:
                pass
        elif line.startswith("MS$FOCUSED_ION: PRECURSOR_TYPE"):
            r["adduct"] = line.split()[-1]
        elif line.startswith("PK$PEAK:"):
            inpk = True
    return r


recs = []
n_files = 0
for name in zf.namelist():
    if not (name.endswith(".txt") and name.split("/")[-1].startswith("MSBNK")):
        continue
    n_files += 1
    r = parse(zf.read(name).decode("utf-8", "replace"))
    r["contributor"] = name.split("/")[-2] if "/" in name else "?"
    recs.append(r)
df = pd.DataFrame(recs)
audit = {"n_record_files": n_files}
f = (df.get("deprecated").isna() if "deprecated" in df else True) & (df.ms_type == "MS2") & (df.ion_mode == "POSITIVE") \
    & df.adduct.isin(["[M+H]+", "[M+Na]+"]) & df.precursor_mz.notna() & df.smiles.notna()
df = df[f].copy()
df["instrument_type"] = df.inst_raw.map(inst_cat)
audit["n_pos_ms2_hplus_na"] = int(len(df))
df = df[df.instrument_type.notna()].copy()
audit["n_highres_mapped"] = int(len(df))

std = Standardizer(cfg["structure"])
u = df.smiles.drop_duplicates()
rec = {s: std.record(s) for s in u}
df["ik14"] = df.smiles.map(lambda s: rec[s]["ik14"])
df["inchikey_full"] = df.smiles.map(lambda s: rec[s]["inchikey"])
df["std_smiles"] = df.smiles.map(lambda s: rec[s]["std_smiles"])
df = df[df.ik14.notna()]
msg_mols = pd.read_parquet(der / "molecules.parquet")
in_msg = df.ik14.isin(set(msg_mols.ik14))
audit["n_records_molecule_in_massspecgym_removed"] = int(in_msg.sum())
df = df[~in_msg].copy()
# structural distance proxy to the TRAIN fold
new = df.drop_duplicates("ik14")[["ik14", "std_smiles", "inchikey_full"]].reset_index(drop=True)
fps_new = [std.fingerprint(s) for s in new.std_smiles]
ok = np.array([fp is not None for fp in fps_new])
new, fps_new = new[ok].reset_index(drop=True), [fp for fp in fps_new if fp is not None]
train_fps = unpack_to_bitvects(np.load(der / "fps.npy")[msg_mols.fold.values == "train"])
maxt = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in fps_new])
new["max_tani_train"] = maxt
far = new[new.max_tani_train < 0.7]
if VARIANT in ("v2", "v3"):
    tk = pd.read_parquet(der / "ext" / "tautomer_keys.parquet")  # computed by 20_tautomer_keys.py on the v1 set
    msg_t = set(tk[tk.fold != "ext"].tkey)
    et = tk[tk.fold == "ext"].set_index("ik14").tkey
    dup_ext = et[et.duplicated(keep=False)].index
    bad = set(et[et.isin(msg_t)].index) | set(dup_ext)
    audit["n_removed_tautomer_equal_to_massspecgym_or_other_external"] = int(far.ik14.isin(bad).sum())
    far = far[~far.ik14.isin(bad)]
if VARIANT == "v3":
    q1 = pd.read_parquet(der / "ext" / "queries.parquet")
    L1 = np.load(der / "ext" / "qlists.npz")
    m1 = pd.read_parquet(der / "ext" / "molecules.parquet")
    msg_mol = set(m1[m1.fold != "ext"].mol_idx)
    dup = set()
    for i in range(len(q1)):
        a, b = L1["W_ptr"][i], L1["W_ptr"][i + 1]
        wm, ws = L1["W_mol"][a:b], L1["W_s"][a:b]
        if any((m in msg_mol) and (s_ >= 0.99) for m, s_ in zip(wm, ws)):
            dup.add(m1.ik14.values[q1.mol_idx.values[i]])
    audit["n_removed_spectral_duplicate_of_massspecgym"] = int(far.ik14.isin(dup).sum())
    far = far[~far.ik14.isin(dup)]
audit["n_new_molecules"] = int(len(new))
audit["n_new_molecules_tanimoto_lt0.7_to_train"] = int(len(far))
df = df[df.ik14.isin(set(far.ik14))].copy()
far = far.reset_index(drop=True)
far["mol_idx"] = np.arange(len(msg_mols), len(msg_mols) + len(far))
df = df.merge(far[["ik14", "mol_idx"]], on="ik14", how="left").reset_index(drop=True)

# spectra processing (identical to MassSpecGym)
sp = cfg["spectrum_processing"]
peaks, npk, ent = [], [], []
for pk, pmz in zip(df.peaks.values, df.precursor_mz.values):
    arr = np.array(pk, dtype=np.float32).reshape(-1, 2)
    q = process(arr, pmz, sp) if len(arr) else np.zeros((0, 2), np.float32)
    peaks.append(q)
    npk.append(len(q))
    ent.append(spectral_entropy(q) if len(q) else 0.0)
df["n_peaks"] = npk
df["spec_entropy"] = ent
df["n_peaks_raw"] = df.peaks.map(len)
df["keep"] = df.n_peaks >= sp["min_peaks"]
audit["n_external_spectra"] = int(len(df))
audit["n_external_kept"] = int(df.keep.sum())
audit["by_contributor"] = df.contributor.value_counts().head(30).to_dict()
audit["by_instrument"] = df.instrument_type.value_counts().to_dict()

# ------ combined tables (MassSpecGym first; external appended) ------
msg_spec = pd.read_parquet(der / "spectra.parquet")
pk0 = np.load(der / "peaks.npz")
n0 = len(msg_spec)
df["row"] = np.arange(n0, n0 + len(df))
df["fold"] = "ext"
df["identifier"] = df.accession
df["simulation_challenge"] = False
df["precursor_ppm_err"] = np.nan
cols = list(msg_spec.columns)
comb = pd.concat([msg_spec, df.assign(contributor=df.contributor)[cols + ["contributor"]]], ignore_index=True)
comb.to_parquet(ext / "spectra.parquet")
ext_peaks = peaks
offs = list(pk0["offsets"])
cat = [pk0["peaks"]]
for q in ext_peaks:
    cat.append(q.astype(np.float32))
    offs.append(offs[-1] + len(q))
np.savez(ext / "peaks.npz", peaks=np.concatenate(cat), offsets=np.array(offs, np.int64))
mcomb = pd.concat([msg_mols, far.assign(fold="ext", smiles=far.std_smiles, formula=None,
                                       n_spectra=far.ik14.map(df.ik14.value_counts()))[list(msg_mols.columns)]],
                  ignore_index=True)
mcomb.to_parquet(ext / "molecules.parquet")
keep_ik = set(far.ik14)
fps_far = [fps_new[i] for i in range(len(new)) if new.ik14.values[i] in keep_ik]
np.save(ext / "fps.npy", np.concatenate([np.load(der / "fps.npy"), pack_fps(fps_far, cfg["structure"]["fingerprint"]["nbits"])]))
write_result(cfg, "external_prepare" + ("" if VARIANT == "v1" else "_" + VARIANT), audit)
print(audit)
