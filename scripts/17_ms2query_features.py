"""MS2Query-style baseline, re-implemented without leakage (added AFTER the main test evaluation; see LOG).

MS2Query (de Jonge et al., Nat Commun 2023) scores each library candidate with a random forest (250 trees,
max depth 5, regression on the Tanimoto between candidate and true molecule) using five features:
  (1) spectral similarity query-candidate (Spec2Vec in the original),
  (2) query precursor m/z, (3) precursor m/z difference query-candidate,
  (4) average similarity (MS2Deepscore in the original) between the query and ALL spectra of the 10 library
      molecules structurally most similar (Tanimoto) to the candidate  -> "neighbourhood consistency",
  (5) average Tanimoto between the candidate and those 10 library molecules.
Its official Spec2Vec / MS2Deepscore / random-forest models were trained on public libraries that overlap
the MassSpecGym test molecules, so they are NOT used. Here every similarity is our entropy similarity
(no learned embedding), the random forest is trained on the TRAIN fold only, and candidates are the
precursor-window molecules plus the 10 best analogue molecules outside the window (MS2Query is an
analogue search). Libraries and scenario exclusions are identical to the main pipeline.

Output: data/derived/ms2q_candidates.parquet (one row per query x scenario x candidate).
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
from msconf.features import tanimoto_packed  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
eng = cfg["engine"]
PPM, MS2 = eng["precursor_tol_ppm"], eng["ms2_tol_da"]
K_NBR, K_ANALOG = 10, 10
lib = pd.read_parquet(der / "lib_index.parquet")
pk = np.load(der / "peaks.npz")
PEAKS, OFFS = pk["peaks"], pk["offsets"]
entries = [{"row": int(r), "precursor_mz": float(m), "peaks": PEAKS[OFFS[r]:OFFS[r + 1]]} for r, m in zip(lib.row.values, lib.precursor_mz.values)]
FES = FlashEntropySearch(max_ms2_tolerance_in_da=max(0.024, MS2))
LIB = FES.build_index(entries, max_indexed_mz=cfg["spectrum_processing"]["max_mz_indexed"], clean_spectra=False)
assert np.array_equal(np.array([e["row"] for e in LIB]), lib.row.values)
PMZ = lib.precursor_mz.values
MOL = lib.mol_idx.values.astype(np.int64)
FC = lib.fold.map({"train": 0, "val": 1, "test": 2}).values
PHASE = {ph: np.isin(FC, [{"train": 0, "val": 1, "test": 2}[f] for f in fs]) for ph, fs in cfg["scenarios"]["phases"].items() if ph != "ext"}
NMOL = int(MOL.max()) + 1
o = np.argsort(MOL, kind="stable")
MPTR = np.searchsorted(MOL[o], np.arange(NMOL + 1))
MPOS = o
SN = dict(np.load(der / "struct_nbrs.npz"))  # eager: lazy NpzFile is not fork-safe
FPS = np.load(der / "fps.npy")
Q = pd.read_parquet(der / "queries.parquet")


def mol_positions(m):
    return MPOS[MPTR[m]:MPTR[m + 1]]


def nbrs_sorted(m):
    a, b = SN["indptr"][m], SN["indptr"][m + 1]
    j, s = SN["indices"][a:b], SN["sims"][a:b].astype(np.float32)
    k = np.argsort(-s, kind="stable")
    return j[k], s[k]


def work(ix):
    out = []
    for i in ix:
        q = Q.iloc[i]
        qp, mt, ph = int(q.pos), int(q.mol_idx), q.fold
        sim = FES.open_search(peaks=LIB[qp]["peaks"], ms2_tolerance_in_da=MS2)
        base = PHASE[ph].copy()
        base[qp] = False
        tol = q.precursor_mz * PPM * 1e-6
        lo, up = np.searchsorted(PMZ, q.precursor_mz - tol, "left"), np.searchsorted(PMZ, q.precursor_mz + tol, "right")
        for sc in ("a", "b"):
            libm = base.copy()
            if sc == "b":
                libm[mol_positions(mt)] = False
            molpresent = np.zeros(NMOL, bool)
            molpresent[MOL[libm]] = True
            # window candidates
            wpos = np.arange(lo, up)[libm[lo:up]]
            if len(wpos) == 0:
                continue
            # best spectrum per molecule (window)
            cand = {}
            for p in wpos:
                m = MOL[p]
                if m not in cand or sim[p] > sim[cand[m]]:
                    cand[m] = p
            # analogue candidates: best molecules outside the window
            outside = libm.copy()
            outside[lo:up] = False
            op = np.nonzero((sim > 0) & outside)[0]
            if len(op):
                op = op[np.argsort(-sim[op], kind="stable")]
                seen = 0
                for p in op:
                    m = MOL[p]
                    if m in cand:
                        continue
                    cand[m] = p
                    seen += 1
                    if seen >= K_ANALOG:
                        break
            for m, p in cand.items():
                j, s = nbrs_sorted(m)
                keep = molpresent[j]
                j, s = j[keep][:K_NBR], s[keep][:K_NBR]
                if len(j):
                    vals = []
                    for u in j:
                        up_ = mol_positions(u)
                        up_ = up_[libm[up_]]
                        if len(up_):
                            vals.append(sim[up_].mean())
                    nbr_sim = float(np.mean(vals)) if vals else 0.0
                    nbr_tani = float(s.mean())
                else:
                    nbr_sim, nbr_tani = 0.0, 0.0
                out.append((qp, sc, int(m), float(sim[p]), float(q.precursor_mz), float(abs(q.precursor_mz - PMZ[p])),
                            nbr_sim, nbr_tani, int(len(j)), int(lo <= p < up), mt))
    return out


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    with Pool(2) as pool:
        for k, r in enumerate(pool.imap(work, np.array_split(np.arange(len(Q)), 300))):
            rows.extend(r)
            if k % 30 == 0:
                print(k, len(rows), f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["qid", "scenario", "cand", "sim", "precursor_mz", "mz_diff", "nbr_sim", "nbr_tani",
                                     "n_nbr", "in_window", "truth"])
    POP = np.array([bin(i).count("1") for i in range(256)], np.uint16)
    tt = np.zeros(len(df), np.float32)
    cv, tv = df.cand.values, df.truth.values
    for a in range(0, len(df), 200000):
        fc, ft = FPS[cv[a:a + 200000]], FPS[tv[a:a + 200000]]
        inter = POP[fc & ft].sum(1)
        union = POP[fc | ft].sum(1)
        tt[a:a + 200000] = np.where(union > 0, inter / np.maximum(union, 1), 0)
    df["tani_truth"] = tt
    df["exact"] = (df.cand == df.truth).astype(np.int8)
    df = df.drop(columns=["truth"])
    df.to_parquet(der / "ms2q_candidates.parquet")
    write_result(cfg, "ms2q_features_summary", {"n_rows": len(df), "seconds": time.time() - t0,
                                                "mean_candidates_per_query_scenario": float(df.groupby(["qid", "scenario"]).size().mean())})
    print("done", len(df), time.time() - t0)
