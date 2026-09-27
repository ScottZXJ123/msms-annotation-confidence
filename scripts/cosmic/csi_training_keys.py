"""Download the published CSI:FingerID positive-mode training structures and convert them to InChIKey first
blocks after the shared RDKit standardization (used only to stratify the COSMIC subset).
Source: https://csi.bright-giant.com/v3.0/api/fingerid/trainingstructures?predictor=1 (gzip-compressed InChI list).
Output: data/derived/csi_training_ik14.csv, results/csi_training_keys.json"""
import gzip
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.chem import Standardizer  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402

RDLogger.DisableLog("rdApp.*")
cfg = load_config()
raw = path(cfg, "raw") / "csifingerid"
raw.mkdir(exist_ok=True)
f = raw / "trainingstructures_pos.csv"
if not f.exists():
    urllib.request.urlretrieve("https://csi.bright-giant.com/v3.0/api/fingerid/trainingstructures?predictor=1", f)
blob = f.read_bytes()
text = gzip.decompress(blob).decode() if blob[:2] == b"\x1f\x8b" else blob.decode()
inchis = [l.strip() for l in io.StringIO(text) if l.strip()]
std = Standardizer(cfg["structure"])
keys, fail = set(), 0
for s in inchis:
    m = Chem.MolFromInchi(s)
    if m is None:
        fail += 1
        continue
    keys.add(Chem.MolToInchiKey(m)[:14])
    r = std.record(Chem.MolToSmiles(m))
    if r["ik14"]:
        keys.add(r["ik14"])
pd.Series(sorted(keys)).to_csv(path(cfg, "derived") / "csi_training_ik14.csv", index=False, header=["ik14"])
mols = pd.read_parquet(path(cfg, "derived") / "molecules.parquet")
write_result(cfg, "csi_training_keys", {"n_inchi": len(inchis), "n_parse_fail": fail, "n_keys": len(keys),
                                        "frac_in_list_by_fold": mols.assign(i=mols.ik14.isin(keys)).groupby("fold").i.mean().to_dict()})
print(len(inchis), fail, len(keys))
