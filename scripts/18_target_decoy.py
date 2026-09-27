"""Target-decoy baseline (added AFTER the main test evaluation; see LOG).

Decoy library (fixed a priori, never tuned on test): one decoy per library spectrum, with the same precursor
m/z, the same number of peaks and the same intensity values in random order; each decoy fragment m/z is
drawn from a real fragment of a random library spectrum of a DIFFERENT molecule whose precursor lies within
±5 Da (conditional, "spectrum-based" decoys in the spirit of Scheubert et al. 2017), keeping only fragments
below precursor − 1.6 Da and merging fragments closer than 0.05 Da (same processing as targets).
Decoys mirror the scenario library: the decoy of the query spectrum is removed in (a), decoys of every truth
spectrum are removed in (b).

Output: data/derived/tda.parquet with, for val/test queries and scenarios a/b: best decoy score in the
precursor window (d1).
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from ms_entropy import FlashEntropySearch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path, write_result  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
eng, sp = cfg["engine"], cfg["spectrum_processing"]
PPM, MS2 = eng["precursor_tol_ppm"], eng["ms2_tol_da"]
lib = pd.read_parquet(der / "lib_index.parquet")
pk = np.load(der / "peaks.npz")
PEAKS, OFFS = pk["peaks"], pk["offsets"]
ROWS = lib.row.values
PMZ = lib.precursor_mz.values
MOL = lib.mol_idx.values
N = len(lib)
rng = np.random.default_rng(cfg["project"]["seed"] + 101)


def make_decoy(j):
    pj = PMZ[j]
    peaks = PEAKS[OFFS[ROWS[j]]:OFFS[ROWS[j] + 1]]
    n = len(peaks)
    lo, hi = np.searchsorted(PMZ, pj - 5.0, "left"), np.searchsorted(PMZ, pj + 5.0, "right")
    mzs = []
    tries = 0
    while len(mzs) < n and tries < 60 * n:
        tries += 1
        k = rng.integers(lo, hi)
        if MOL[k] == MOL[j]:
            continue
        pk_ = PEAKS[OFFS[ROWS[k]]:OFFS[ROWS[k] + 1]]
        cand = pk_[pk_[:, 0] < pj - sp["remove_precursor_region_da"], 0]
        if len(cand) == 0:
            continue
        mz = float(cand[rng.integers(len(cand))])
        if all(abs(mz - x) >= sp["merge_peaks_within_da"] for x in mzs):
            mzs.append(mz)
    if len(mzs) == 0:
        return np.zeros((0, 2), np.float32)
    it = rng.permutation(peaks[:, 1])[: len(mzs)]
    d = np.stack([np.array(mzs), it], 1).astype(np.float32)
    d = d[np.argsort(d[:, 0])]
    d[:, 1] /= d[:, 1].sum()
    return d


t0 = time.time()
decoys = [{"idx": j, "precursor_mz": float(PMZ[j]), "peaks": make_decoy(j)} for j in range(N)]
ok = np.array([len(d["peaks"]) > 0 for d in decoys])
FES = FlashEntropySearch(max_ms2_tolerance_in_da=max(0.024, MS2))
DL = FES.build_index([d for d in decoys if len(d["peaks"])], max_indexed_mz=sp["max_mz_indexed"], clean_spectra=False)
DIDX = np.array([d["idx"] for d in DL])  # decoy index position -> target library position it mirrors
DPMZ = np.array([d["precursor_mz"] for d in DL])
print("decoys built", ok.mean(), f"{time.time()-t0:.0f}s", flush=True)
FC = lib.fold.map({"train": 0, "val": 1, "test": 2}).values
PHASE = {ph: np.isin(FC, [{"train": 0, "val": 1, "test": 2}[f] for f in fs]) for ph, fs in cfg["scenarios"]["phases"].items() if ph != "ext"}
Q = pd.read_parquet(der / "queries.parquet")
Q = Q[Q.fold.isin(["val", "test"])].reset_index(drop=True)
qpk = {int(p): PEAKS[OFFS[ROWS[p]]:OFFS[ROWS[p] + 1]] for p in Q.pos.values}


def work(ix):
    out = []
    for i in ix:
        q = Q.iloc[i]
        qp = int(q.pos)
        tol = q.precursor_mz * PPM * 1e-6
        lo, up = np.searchsorted(DPMZ, q.precursor_mz - tol, "left"), np.searchsorted(DPMZ, q.precursor_mz + tol, "right")
        if up <= lo:
            out += [(qp, "a", 0.0), (qp, "b", 0.0)]
            continue
        sim = FES.open_search(peaks=qpk[qp], ms2_tolerance_in_da=MS2)[lo:up]
        mirror = DIDX[lo:up]
        base = PHASE[q.fold][mirror] & (mirror != qp)
        for sc in ("a", "b"):
            m = base & ((MOL[mirror] != q.mol_idx) if sc == "b" else True)
            out.append((qp, sc, float(sim[m].max()) if m.any() else 0.0))
    return out


if __name__ == "__main__":
    rows = []
    with Pool(2) as p:
        for r in p.imap(work, np.array_split(np.arange(len(Q)), 100)):
            rows.extend(r)
    df = pd.DataFrame(rows, columns=["qid", "scenario", "d1"])
    df.to_parquet(der / "tda.parquet")
    write_result(cfg, "tda_summary", {"n_decoys": int(ok.sum()), "frac_decoy_nonempty": float(ok.mean()), "n_rows": len(df),
                                      "seconds": time.time() - t0})
    print("done", len(df), f"{time.time()-t0:.0f}s")
