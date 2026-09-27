"""Annotation engine pass (spectral library search) + library neighborhood pass.

One Flash-entropy open search per kept spectrum over the full processed library. Because the entropy
score is pairwise, the score of any library spectrum does not depend on which other spectra are in the
library; library composition (phase-nested libraries, leave-one-out exclusions) is therefore applied
afterwards as masks. Verified: open-search score == identity-search score for spectra in the precursor
window (scripts/checks/check_identity_equals_open.py).

Outputs (data/derived/):
  lib_index.parquet   library position -> spectrum row / molecule / fold / precursor m/z
  spec_nbrs07.npz     for every library spectrum: library positions with entropy sim >= crowding.spec_sim_hi
  queries.parquet     eligible query spectra (scenario (a) possible: another spectrum of the same molecule in the window)
  qlists.npz          per query, phase-masked and excluding the query spectrum itself:
                        W: molecules in the precursor window (max entropy, n spectra, mean entropy, matched peaks of best, max cosine)
                        O: molecules at any precursor with max entropy >= crowding.spec_sim_lo
                        X: top-50 molecules OUTSIDE the window by max entropy (analog hits)
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
from msconf.cosine import cosine_greedy  # noqa: E402

cfg = load_config()
import os  # noqa: E402

der = Path(os.environ["MSCONF_DERIVED"]) if os.environ.get("MSCONF_DERIVED") else path(cfg, "derived")
QUERY_FOLDS = os.environ.get("MSCONF_QUERY_FOLDS", "train,val,test").split(",")
eng = cfg["engine"]
cr = cfg["crowding"]
PPM = eng["precursor_tol_ppm"]
MS2 = eng["ms2_tol_da"]
TOPX = 50

spec = pd.read_parquet(der / "spectra.parquet")
pk = np.load(der / "peaks.npz")
PEAKS, OFFS = pk["peaks"], pk["offsets"]
kept = spec[spec.keep].copy()
entries = [{"row": int(r), "precursor_mz": float(m), "peaks": PEAKS[OFFS[r]:OFFS[r + 1]]}
           for r, m in zip(kept.row.values, kept.precursor_mz.values)]
FES = FlashEntropySearch(max_ms2_tolerance_in_da=max(0.024, MS2))
LIB = FES.build_index(entries, max_indexed_mz=cfg["spectrum_processing"]["max_mz_indexed"], clean_spectra=False)
N = len(LIB)
POS_ROW = np.array([e["row"] for e in LIB], np.int64)
ROW_POS = np.full(len(spec), -1, np.int64)
ROW_POS[POS_ROW] = np.arange(N)
PMZ = np.array([e["precursor_mz"] for e in LIB], np.float64)
MOL = spec.mol_idx.values[POS_ROW].astype(np.int32)
FOLD = spec.fold.values[POS_ROW]
FOLDCODE = {"train": 0, "val": 1, "test": 2, "ext": 3}
FC = np.array([FOLDCODE[f] for f in FOLD], np.int8)
PHASE_MASK = {ph: np.isin(FC, [FOLDCODE[f] for f in folds]) for ph, folds in cfg["scenarios"]["phases"].items()}
NMOL = int(spec.mol_idx.max()) + 1


def window(pmz):
    tol = max(pmz * PPM * 1e-6, eng["precursor_tol_min_da"])
    return np.searchsorted(PMZ, pmz - tol, "left"), np.searchsorted(PMZ, pmz + tol, "right")


def mol_max(pos, score):
    """Aggregate spectrum-level scores to molecule level: (mols, max, count, mean, argmax_pos)."""
    if len(pos) == 0:
        e = np.zeros(0)
        return e.astype(np.int32), e, e.astype(np.int32), e, e.astype(np.int64)
    m = MOL[pos]
    o = np.lexsort((-score, m))
    m, s, p = m[o], score[o], pos[o]
    first = np.r_[True, m[1:] != m[:-1]]
    starts = np.nonzero(first)[0]
    cnt = np.diff(np.r_[starts, len(m)])
    mean = np.add.reduceat(s, starts) / cnt
    return m[starts], s[starts], cnt.astype(np.int32), mean, p[starts]


def work(positions):
    out = {"nbr": [], "q": []}
    for qp in positions:
        e = LIB[qp]
        sim, mp = FES.open_search(peaks=e["peaks"], ms2_tolerance_in_da=MS2, output_matched_peak_number=True)
        hi = np.nonzero(sim >= cr["spec_sim_hi"])[0]
        hi = hi[hi != qp].astype(np.int32)
        out["nbr"].append((qp, hi))
        if not IS_QUERY[qp]:
            continue
        ph = QPHASE[qp]
        lib = PHASE_MASK[ph].copy()
        lib[qp] = False
        lo, up = window(e["precursor_mz"])
        wpos = np.arange(lo, up)
        wpos = wpos[lib[lo:up]]
        ws = sim[wpos]
        wm, wmax, wcnt, wmean, wbest = mol_max(wpos, ws)
        wmp = mp[wbest].astype(np.int32)
        # cosine (secondary engine): max over the molecule's window spectra
        cos = np.array([cosine_greedy(e["peaks"].astype(np.float64), LIB[j]["peaks"].astype(np.float64), MS2, 1.0) for j in wpos])
        if len(wpos):
            cm = MOL[wpos]
            o = np.lexsort((-cos, cm))
            cm_s, cos_s = cm[o], cos[o]
            st = np.nonzero(np.r_[True, cm_s[1:] != cm_s[:-1]])[0]
            cosmax = dict(zip(cm_s[st].tolist(), cos_s[st].tolist()))
            wcos = np.array([cosmax[m] for m in wm])
        else:
            wcos = np.zeros(0)
        # open neighborhood (all precursor masses), phase-masked, excluding the query spectrum
        opos = np.nonzero((sim >= cr["spec_sim_lo"]) & lib)[0]
        om, omax, _, _, _ = mol_max(opos, sim[opos])
        # best analogs outside the precursor window
        outside = lib.copy()
        outside[lo:up] = False
        cand = np.nonzero((sim > 0) & outside)[0]
        if len(cand) > 4000:
            cand = cand[np.argpartition(-sim[cand], 4000)[:4000]]
        xm, xmax, _, _, _ = mol_max(cand, sim[cand])
        if len(xm) > TOPX:
            k = np.argsort(-xmax, kind="stable")[:TOPX]
            xm, xmax = xm[k], xmax[k]
        out["q"].append((qp, wm, wmax.astype(np.float32), wcnt, wmean.astype(np.float32), wmp, wcos.astype(np.float32),
                         om, omax.astype(np.float32), xm, xmax.astype(np.float32)))
    return out


# ---------------- eligible queries ----------------
rng = np.random.default_rng(cfg["project"]["seed"])
elig = np.zeros(N, bool)
for m, grp in pd.DataFrame({"pos": np.arange(N), "mol": MOL}).groupby("mol").pos:
    p = grp.values
    if len(p) < 2:
        continue
    pm = PMZ[p]
    tol = np.maximum(pm * PPM * 1e-6, eng["precursor_tol_min_da"])
    d = np.abs(pm[:, None] - pm[None, :]) <= tol[:, None]
    np.fill_diagonal(d, False)
    elig[p[d.any(1)]] = True
IS_QUERY = elig & np.isin(FC, [FOLDCODE[f] for f in QUERY_FOLDS])
cap = cfg["scenarios"]["max_query_spectra_per_molecule_train"]
tr = np.nonzero(elig & (FC == 0))[0]
dfq = pd.DataFrame({"pos": tr, "mol": MOL[tr]})
keep_tr = dfq.groupby("mol").pos.apply(lambda s: rng.choice(s.values, size=min(cap, len(s)), replace=False)).explode().astype(int).values
if "train" in QUERY_FOLDS:
    IS_QUERY[tr] = False
    IS_QUERY[keep_tr] = True
QPHASE = np.array(["train", "val", "test", "ext"])[FC]

if __name__ == "__main__":
    t0 = time.time()
    import os
    lim = int(os.environ.get("MSCONF_LIMIT", "0"))
    todo = np.arange(N) if not lim else np.sort(np.r_[np.nonzero(IS_QUERY)[0][:lim], np.arange(lim)])
    chunks = np.array_split(np.unique(todo), 400 if not lim else 8)
    if lim:
        der = der / "debug"
        der.mkdir(exist_ok=True)
    nbr = [None] * N
    qrecs = []
    with Pool(2) as pool:
        for i, out in enumerate(pool.imap_unordered(work, chunks)):
            for qp, hi in out["nbr"]:
                nbr[qp] = hi
            qrecs.extend(out["q"])
            if i % 20 == 0:
                print(i, len(qrecs), f"{time.time()-t0:.0f}s", flush=True)
    indptr = np.zeros(N + 1, np.int64)
    nbr = [x if x is not None else np.zeros(0, np.int32) for x in nbr]
    indptr[1:] = np.cumsum([len(x) for x in nbr])
    np.savez(der / "spec_nbrs07.npz", indptr=indptr, indices=np.concatenate(nbr).astype(np.int32))
    pd.DataFrame({"pos": np.arange(N), "row": POS_ROW, "mol_idx": MOL, "fold": FOLD, "precursor_mz": PMZ}).to_parquet(der / "lib_index.parquet")
    qrecs.sort(key=lambda r: r[0])
    qpos = np.array([r[0] for r in qrecs])

    def csr(cols):
        lens = np.array([len(r[cols[0]]) for r in qrecs])
        ip = np.zeros(len(qrecs) + 1, np.int64)
        ip[1:] = np.cumsum(lens)
        return ip, [np.concatenate([r[c] for r in qrecs]) if len(qrecs) else np.zeros(0) for c in cols]

    wi, (wm, ws, wc, wmean, wmp, wcos) = csr([1, 2, 3, 4, 5, 6])
    oi, (om, os_) = csr([7, 8])
    xi, (xm, xs) = csr([9, 10])
    np.savez(der / "qlists.npz", qpos=qpos, W_ptr=wi, W_mol=wm, W_s=ws, W_n=wc, W_mean=wmean, W_mp=wmp, W_cos=wcos,
             O_ptr=oi, O_mol=om, O_s=os_, X_ptr=xi, X_mol=xm, X_s=xs)
    q = spec.iloc[POS_ROW[qpos]].reset_index(drop=True).copy()
    q["pos"] = qpos
    q.to_parquet(der / "queries.parquet")
    summ = {"n_library_spectra": int(N), "n_eligible_by_fold": {f: int((elig & (FC == c)).sum()) for f, c in FOLDCODE.items()},
            "n_kept_by_fold": {f: int((FC == c).sum()) for f, c in FOLDCODE.items()},
            "n_queries_by_fold": q.fold.value_counts().to_dict(), "n_query_molecules_by_fold": q.groupby("fold").mol_idx.nunique().to_dict(),
            "mean_window_molecules": float(np.diff(wi).mean()), "seconds": time.time() - t0}
    if not lim:
        write_result(cfg, "search_summary" + ("_" + der.name if os.environ.get("MSCONF_DERIVED") else ""), summ)
    print(summ)
