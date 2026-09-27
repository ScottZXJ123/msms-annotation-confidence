"""Structural neighbor graph over all MassSpecGym molecules (Morgan r=2, 2048 bits, Tanimoto).

Stored as CSR with Tanimoto >= 0.4 (data/derived/struct_nbrs.npz). Used for:
 - scenario (c) (remove truth + library molecules with Tanimoto >= c_struct_neighbor_tanimoto)
 - structural crowding features of the observable top-1 candidate
 - clustering val molecules into calibration / model-selection halves
Also records each molecule's max Tanimoto to the train fold (analysis only; never a feature).
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.chem import unpack_to_bitvects  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402

cfg = load_config()
import os  # noqa: E402

der = Path(os.environ["MSCONF_DERIVED"]) if os.environ.get("MSCONF_DERIVED") else path(cfg, "derived")
mols = pd.read_parquet(der / "molecules.parquet")
FPS = unpack_to_bitvects(np.load(der / "fps.npy"))
IS_TRAIN = (mols.fold.values == "train")
THR = 0.4


def rows(ix):
    out = []
    for i in ix:
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(FPS[i], FPS), dtype=np.float32)
        s[i] = -1.0
        j = np.nonzero(s >= THR)[0].astype(np.int32)
        mt = s[IS_TRAIN].max()
        out.append((i, j, s[j].astype(np.float16), mt))
    return out


if __name__ == "__main__":
    t0 = time.time()
    n = len(FPS)
    chunks = np.array_split(np.arange(n), 200)
    res = [None] * n
    maxtrain = np.zeros(n, np.float32)
    with Pool(2) as p:
        for out in p.imap(rows, chunks):
            for i, j, s, mt in out:
                res[i] = (j, s)
                maxtrain[i] = mt
    indptr = np.zeros(n + 1, np.int64)
    indptr[1:] = np.cumsum([len(r[0]) for r in res])
    np.savez(der / "struct_nbrs.npz", indptr=indptr, indices=np.concatenate([r[0] for r in res]),
             sims=np.concatenate([r[1] for r in res]), max_tani_to_train=maxtrain, threshold=THR)
    deg7 = np.array([(r[1] >= 0.7).sum() for r in res])
    summ = {"n_molecules": n, "seconds": time.time() - t0, "mean_deg_ge0.4": float(np.mean(np.diff(indptr))),
            "mean_deg_ge0.7": float(deg7.mean()),
            "max_tani_to_train_by_fold": {f: np.percentile(maxtrain[mols.fold.values == f], [5, 25, 50, 75, 95]).tolist()
                                          for f in ["val", "test"]}}
    write_result(cfg, "struct_neighbors_summary" + ("_ext" if os.environ.get("MSCONF_DERIVED") else ""), summ)
    print(summ)
