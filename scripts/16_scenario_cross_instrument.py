"""Scenario (a_x): truth in the library ONLY via spectra from the OTHER instrument type.

In MassSpecGym, scenario (a) usually leaves same-instrument spectra of the truth (other collision energies)
in the library, which makes in-library retrieval easy. A lab querying a public library is closer to a_x:
the library entry for its compound was measured elsewhere. This is a library-level distribution-shift test
evaluated with the frozen models (test fold only; nothing is fitted here).

Output: data/derived/rows_ax_entropy.parquet (test queries with Orbitrap/QTOF instrument whose truth has
>= 1 other-instrument spectrum in the precursor window) and results/eval_cross_instrument_library.json.
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from ms_entropy import FlashEntropySearch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf import evalutil  # noqa: E402
from msconf.config import load_config, path, write_result  # noqa: E402
from msconf.cosine import cosine_greedy  # noqa: E402
from msconf.evalutil import components, evaluate_weighted  # noqa: E402
from msconf.features import FeatureContext  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
eng, cr = cfg["engine"], cfg["crowding"]
PPM, MS2 = eng["precursor_tol_ppm"], eng["ms2_tol_da"]
spec = pd.read_parquet(der / "spectra.parquet")
lib = pd.read_parquet(der / "lib_index.parquet")
pk = np.load(der / "peaks.npz")
PEAKS, OFFS = pk["peaks"], pk["offsets"]
entries = [{"row": int(r), "precursor_mz": float(m), "peaks": PEAKS[OFFS[r]:OFFS[r + 1]]} for r, m in zip(lib.row.values, lib.precursor_mz.values)]
FES = FlashEntropySearch(max_ms2_tolerance_in_da=max(0.024, MS2))
LIB = FES.build_index(entries, max_indexed_mz=cfg["spectrum_processing"]["max_mz_indexed"], clean_spectra=False)
assert np.array_equal(np.array([e["row"] for e in LIB]), lib.row.values), "library order must match lib_index"
PMZ = lib.precursor_mz.values
MOL = lib.mol_idx.values.astype(np.int32)
INST = spec.instrument_type.values[lib.row.values]
PHASE = lib.fold.isin(cfg["scenarios"]["phases"]["test"]).values
Q = pd.read_parquet(der / "queries.parquet")
Q = Q[(Q.fold == "test") & Q.instrument_type.isin(["Orbitrap", "QTOF"])].reset_index(drop=True)


def window(pmz):
    tol = pmz * PPM * 1e-6
    return np.searchsorted(PMZ, pmz - tol, "left"), np.searchsorted(PMZ, pmz + tol, "right")


def mol_max(pos, score):
    if len(pos) == 0:
        e = np.zeros(0)
        return e.astype(np.int32), e, e.astype(np.int32), e, e.astype(np.int64)
    m = MOL[pos]
    o = np.lexsort((-score, m))
    m, s, p = m[o], score[o], pos[o]
    st = np.nonzero(np.r_[True, m[1:] != m[:-1]])[0]
    cnt = np.diff(np.r_[st, len(m)])
    return m[st], s[st], cnt.astype(np.int32), np.add.reduceat(s, st) / cnt, p[st]


def work(ix):
    out = []
    for i in ix:
        q = Q.iloc[i]
        qp = int(q.pos)
        libm = PHASE.copy()
        libm[qp] = False
        same = np.nonzero((MOL == q.mol_idx) & (INST == q.instrument_type))[0]
        libm[same] = False  # truth reachable only through other-instrument spectra
        lo, up = window(q.precursor_mz)
        wpos = np.arange(lo, up)[libm[lo:up]]
        if not np.any(MOL[wpos] == q.mol_idx):
            continue  # a_x not possible for this query
        e = LIB[qp]
        sim, mp = FES.open_search(peaks=e["peaks"], ms2_tolerance_in_da=MS2, output_matched_peak_number=True)
        wm, wmax, wcnt, wmean, wbest = mol_max(wpos, sim[wpos])
        cos = np.array([cosine_greedy(e["peaks"].astype(np.float64), LIB[j]["peaks"].astype(np.float64), MS2, 1.0) for j in wpos])
        cm = MOL[wpos]
        o = np.lexsort((-cos, cm))
        st = np.nonzero(np.r_[True, cm[o][1:] != cm[o][:-1]])[0]
        cmax = dict(zip(cm[o][st].tolist(), cos[o][st].tolist()))
        opos = np.nonzero((sim >= cr["spec_sim_lo"]) & libm)[0]
        om, omax, _, _, _ = mol_max(opos, sim[opos])
        outside = libm.copy()
        outside[lo:up] = False
        cand = np.nonzero((sim > 0) & outside)[0]
        if len(cand) > 4000:
            cand = cand[np.argpartition(-sim[cand], 4000)[:4000]]
        xm, xmax, _, _, _ = mol_max(cand, sim[cand])
        k = np.argsort(-xmax, kind="stable")[:50]
        out.append((i, wm, wmax.astype(np.float32), wcnt, wmean.astype(np.float32), mp[wbest].astype(np.int32),
                    np.array([cmax[m] for m in wm], np.float32), om, omax.astype(np.float32), xm[k], xmax[k].astype(np.float32),
                    same))
    return out


if __name__ == "__main__":
    t0 = time.time()
    recs = []
    with Pool(2) as p:
        for r in p.imap(work, np.array_split(np.arange(len(Q)), 100)):
            recs.extend(r)
    recs.sort(key=lambda r: r[0])
    Qx = Q.iloc[[r[0] for r in recs]].reset_index(drop=True)

    def csr(c):
        ip = np.zeros(len(recs) + 1, np.int64)
        ip[1:] = np.cumsum([len(r[c[0]]) for r in recs])
        return ip, [np.concatenate([r[j] for r in recs]) for j in c]

    wi, (wm, ws, wc, wmean, wmp, wcos) = csr([1, 2, 3, 4, 5, 6])
    oi, (om, os_) = csr([7, 8])
    xi, (xm, xs) = csr([9, 10])
    Lx = {"W_ptr": wi, "W_mol": wm, "W_s": ws, "W_n": wc, "W_mean": wmean, "W_mp": wmp, "W_cos": wcos,
          "O_ptr": oi, "O_mol": om, "O_s": os_, "X_ptr": xi, "X_mol": xm, "X_s": xs}
    ctx = FeatureContext(cfg, lib, dict(np.load(der / "spec_nbrs07.npz")), dict(np.load(der / "struct_nbrs.npz")),
                         np.load(der / "fps.npy"), pd.read_parquet(der / "families.parquet"), Qx, Lx)
    ctx.extra_excl_pos = {k: r[11] for k, r in enumerate(recs)}
    rows = []
    for i in range(len(Qx)):
        rows.extend(ctx.rows_for_query(i, scenarios=("a",), engine="entropy", fam_level=cfg["family"]["primary_level"]))
    ax = pd.DataFrame(rows)
    ax["scenario"] = "a_x"
    fam = pd.read_parquet(der / "families.parquet")
    for lv in ["superclass", "class", "subclass"]:
        ax[f"top1_{lv}"] = fam[lv].values[ax.top1.astype(int)]
        ax[f"ana_true_{lv}"] = fam[lv].values[ax.mol.values]
    ax["w"] = 1.0 / ax.groupby("mol").qid.transform("size")
    ax.to_parquet(der / "rows_ax_entropy.parquet")
    # evaluation with frozen models: in-library rows = a_x, absent rows = (b) rows of the same queries
    evalutil.set_thresholds(cfg["tiers"]["trusted_min_p_correct"], cfg["tiers"]["absent_max_p_in"])
    main = pd.read_parquet(der / "rows_entropy.parquet")
    b = main[(main.fold == "test") & (main.scenario == "b") & (main.has_candidate == 1) & main.qid.isin(ax.qid)]
    b = b.assign(w=1.0 / b.groupby("mol").qid.transform("size"))
    a = main[(main.fold == "test") & (main.scenario == "a") & main.qid.isin(ax.qid)]
    a = a.assign(w=1.0 / a.groupby("mol").qid.transform("size"))
    ev = pd.concat([ax.assign(scenario="a"), b], ignore_index=True)
    ev_ref = pd.concat([a, b], ignore_index=True)
    sel = json.load(open(path(cfg, "results") / "selection.json"))
    res = {"n_queries_ax": int(len(ax)), "n_mol_ax": int(ax.mol.nunique()),
           "top1_acc_ax": float(np.average(ax.correct, weights=ax.w)), "top1_acc_a_same_queries": float(np.average(a.correct, weights=a.w)),
           "seconds": time.time() - t0, "systems": {}}
    for name in dict.fromkeys([sel["primary_system"], "M_full", "M_base", "B_lr_dec", "B_lr_direct"]):
        f = der / "models" / "entropy" / f"{name}.pkl"
        if not f.exists():
            continue
        sysm = pickle.loads(f.read_bytes())
        res["systems"][name] = {
            "a_x": {str(pi): {m: evaluate_weighted(components(sysm, ev), ev, pi, m) for m in ["oracle", "em"]} for pi in cfg["evaluation"]["prevalence_headline"]},
            "a_same_queries": {str(pi): {m: evaluate_weighted(components(sysm, ev_ref), ev_ref, pi, m) for m in ["oracle", "em"]} for pi in cfg["evaluation"]["prevalence_headline"]}}
        pcA = sysm.p_correct_given_in(ax) if hasattr(sysm, "p_correct_given_in") else None
        if pcA is not None:
            res["systems"][name]["C_component_ax"] = {"mean_pred": float(np.average(pcA, weights=ax.w)), "obs": float(np.average(ax.correct, weights=ax.w))}
    write_result(cfg, "eval_cross_instrument_library", res)
    print(json.dumps({k: v for k, v in res.items() if k != "systems"}, indent=1))
