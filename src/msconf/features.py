"""Observable-only features for a (query, scenario) pair.

HARD RULE (§4, failure mode #3): features may use only the query spectrum and its metadata, the engine's
candidate list, and the reference library as it exists in that scenario. The truth molecule is used for
(i) deciding which library entries exist in the scenario (simulation, not a feature), (ii) labels, and
(iii) analysis-only columns prefixed `ana_`, which are never passed to a model (enforced in models.py).

Feature groups
  meta         precursor m/z, adduct, instrument, collision energy, peak count, spectral entropy,
               number of library spectra / molecules in the precursor window
  sim          engine similarity scores only: s1, s2, s3, s1-s2, s1-s3
  sim_extra    other score-derived quantities: matched peaks of top-1, mean score over top-1's
               library spectra, number of top-1 library spectra in the window
  analog       best similarity to library spectra OUTSIDE the precursor window (analog hit)
  crowd_spec   spectral-space crowding: open-search neighbourhood density of the query, near-ties
               among candidates, spectral neighbourhood density of the top-1 candidate in the library
  crowd_struct fingerprint-space crowding: Tanimoto among top candidates, structural neighbours of
               top-1 in the library / in the precursor window
  crowd_family ClassyFire-family crowding of the observable top-k candidates: family entropy of top-k,
               same-family counts, family coverage in the library, family-level spectral crowding
               (fixed table computed on the train-fold library only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

GROUPS = {
    "meta": ["precursor_mz", "adduct_na", "instrument", "collision_energy", "n_peaks", "spec_entropy",
             "n_win_mol", "n_win_spec"],
    "sim": ["s1", "s2", "s3", "gap12", "gap13"],
    "sim_extra": ["mp1", "mp1_frac", "s1_mean", "n_top1_spec"],
    "analog": ["x1", "x1_minus_s1"],
    "crowd_spec": ["n_open_lo", "n_open_hi", "n_near_tie", "n_win_ge05", "top1_specdens"],
    "crowd_struct": ["tani12", "tani_mean_top5", "top1_struct_nbrs", "n_win_struct_nbrs"],
    "crowd_family": ["fam_ent_topk", "fam_frac_topk_same", "fam_n_win_same",
                     "fam_frac_open_same", "fam_spec_crowd_train", "fam_struct_crowd_train"],
    # EXCLUDED from all proposed feature sets (see LOG 2026-09-26 22:15): with the family-level tables
    # identifying the class, the library count of the class differs by exactly one between the paired (a)/(b)
    # rows of training queries, and the model memorizes per-class counts of the training library — a
    # construction shortcut that does not transfer (val-A log loss of P(in) 0.083 -> 0.171).
    "crowd_family_libsize": ["fam_lib_size"],
}


def tanimoto_packed(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Tanimoto between one packed fp (nbytes,) and many (n, nbytes)."""
    inter = POPCNT[np.bitwise_and(fp_b, fp_a)].sum(1).astype(np.float32)
    union = POPCNT[np.bitwise_or(fp_b, fp_a)].sum(1).astype(np.float32)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


class FeatureContext:
    """Holds library-level structures; phase-specific masks are selected per query."""

    def __init__(self, cfg, lib_index: pd.DataFrame, spec_nbrs, struct_nbrs, fps: np.ndarray, fam: pd.DataFrame,
                 queries: pd.DataFrame, qlists):
        self.cfg = cfg
        cr = cfg["crowding"]
        self.hi, self.lo = cr["spec_sim_hi"], cr["spec_sim_lo"]
        self.tan, self.dtie, self.topk = cr["struct_tanimoto"], cr["near_tie_delta"], cr["topk_family"]
        self.c_tan = cfg["scenarios"]["c_struct_neighbor_tanimoto"]
        self.fps = fps
        self.nmol = len(fps)
        self.lib_mol = lib_index.mol_idx.values.astype(np.int64)
        fold_code = {"train": 0, "val": 1, "test": 2, "ext": 3}
        self.lib_fc = lib_index.fold.map(fold_code).values.astype(np.int8)
        self.phase_pos = {ph: np.isin(self.lib_fc, [fold_code[f] for f in fs]) for ph, fs in cfg["scenarios"]["phases"].items()}
        self.phase_mol = {}
        for ph, m in self.phase_pos.items():
            pm = np.zeros(self.nmol, bool)
            pm[self.lib_mol[m]] = True
            self.phase_mol[ph] = pm
        # molecule -> library positions
        o = np.argsort(self.lib_mol, kind="stable")
        self.mol_pos_sorted = o
        self.mol_pos_ptr = np.searchsorted(self.lib_mol[o], np.arange(self.nmol + 1))
        self.sn_ptr, self.sn_idx = spec_nbrs["indptr"], spec_nbrs["indices"]
        self.st_ptr, self.st_idx, self.st_sim = struct_nbrs["indptr"], struct_nbrs["indices"], struct_nbrs["sims"].astype(np.float32)
        # families (codes per level)
        self.fam_levels = ["superclass", "class", "subclass"]
        self.fam_code = {}
        self.fam_names = {}
        for lv in self.fam_levels:
            cats = pd.Categorical(fam[lv].values)
            self.fam_code[lv] = np.asarray(cats.codes, np.int32)
            self.fam_names[lv] = list(cats.categories)
        self.fam_tables = {lv: self._family_tables(lv) for lv in self.fam_levels}
        self._fam_cache = {}
        self.extra_excl_pos = None
        self.set_excl_mols = np.zeros(0, np.int64)  # scenario "bset" only
        self.hashperm = np.random.default_rng(12345).permutation(self.nmol)  # fixed tie-break, label-free
        self.q = queries
        self.L = qlists

    # ------------------------------------------------------------------ helpers
    def mol_positions(self, m):
        return self.mol_pos_sorted[self.mol_pos_ptr[m]:self.mol_pos_ptr[m + 1]]

    def struct_nbrs(self, m, thr):
        a, b = self.st_ptr[m], self.st_ptr[m + 1]
        j = self.st_idx[a:b]
        return j[self.st_sim[a:b] >= thr]

    def _family_tables(self, lv):
        """Family-level crowding computed on the TRAIN-fold library only (fixed lookup table)."""
        code = self.fam_code[lv]
        ncls = code.max() + 1
        train_pos = self.lib_fc == 0
        train_mol = np.zeros(self.nmol, bool)
        train_mol[self.lib_mol[train_pos]] = True
        # molecule-level spectral edges within the train library
        spec_deg_same = np.zeros(self.nmol, np.float64)
        for m in np.nonzero(train_mol)[0]:
            pos = self.mol_positions(m)
            pos = pos[train_pos[pos]]
            if len(pos) == 0:
                continue
            nb = np.concatenate([self.sn_idx[self.sn_ptr[p]:self.sn_ptr[p + 1]] for p in pos])
            nb = nb[train_pos[nb]]
            nm = np.unique(self.lib_mol[nb])
            nm = nm[nm != m]
            spec_deg_same[m] = np.sum(code[nm] == code[m])
        struct_deg_same = np.zeros(self.nmol, np.float64)
        for m in np.nonzero(train_mol)[0]:
            j = self.struct_nbrs(m, self.tan)
            j = j[train_mol[j]]
            struct_deg_same[m] = np.sum(code[j] == code[m])
        size = np.bincount(code[train_mol], minlength=ncls).astype(np.float64)
        spec_c = np.bincount(code[train_mol], weights=spec_deg_same[train_mol], minlength=ncls)
        struct_c = np.bincount(code[train_mol], weights=struct_deg_same[train_mol], minlength=ncls)
        with np.errstate(invalid="ignore", divide="ignore"):
            spec_crowd = np.where(size > 1, spec_c / (size * np.maximum(size - 1, 1)), np.nan)
            struct_crowd = np.where(size > 1, struct_c / (size * np.maximum(size - 1, 1)), np.nan)
        # scale to "expected number of same-family spectral/structural neighbours" per molecule
        return {"size_train": size, "spec_crowd": spec_crowd * np.maximum(size - 1, 0),
                "struct_crowd": struct_crowd * np.maximum(size - 1, 0)}

    # ------------------------------------------------------------------ main
    def rows_for_query(self, i: int, scenarios=("a", "b"), engine="entropy", fam_level="class"):
        q = self.q.iloc[i]
        L = self.L
        m_true = int(q.mol_idx)
        ph = q.fold
        qpos = int(q.pos)
        phase_pos, phase_mol = self.phase_pos[ph], self.phase_mol[ph]
        w0, w1 = L["W_ptr"][i], L["W_ptr"][i + 1]
        W = {k: L[f"W_{k}"][w0:w1] for k in ["mol", "s", "n", "mean", "mp", "cos"]}
        o0, o1 = L["O_ptr"][i], L["O_ptr"][i + 1]
        Om, Os = L["O_mol"][o0:o1], L["O_s"][o0:o1]
        x0, x1 = L["X_ptr"][i], L["X_ptr"][i + 1]
        Xm, Xs = L["X_mol"][x0:x1], L["X_s"][x0:x1]
        out = []
        for sc in scenarios:
            if sc == "a":
                excl = np.zeros(0, np.int64)
            elif sc == "b":
                excl = np.array([m_true])
            elif sc == "c":
                excl = np.r_[m_true, self.struct_nbrs(m_true, self.c_tan)]
            elif sc == "bset":  # (b) with a whole set of molecules removed at once (COSMIC head-to-head)
                excl = np.union1d([m_true], self.set_excl_mols).astype(np.int64)
            else:
                raise ValueError(sc)
            keepW = ~np.isin(W["mol"], excl)
            wm = W["mol"][keepW]
            base = {"qid": qpos, "row": int(q.row), "mol": m_true, "fold": ph, "scenario": sc, "engine": engine}
            if len(wm) == 0:
                out.append({**base, "has_candidate": 0})
                continue
            s = (W["s"] if engine == "entropy" else W["cos"])[keepW].astype(np.float64)
            wn, wmean, wmp = W["n"][keepW], W["mean"][keepW], W["mp"][keepW]
            order = np.lexsort((self.hashperm[wm], -wn, -s))
            wm, s, wn, wmean, wmp = wm[order], s[order], wn[order], wmean[order], wmp[order]
            t = int(wm[0])
            r = dict(base)
            r["has_candidate"] = 1
            r["top1"] = t
            r["top5"] = ",".join(map(str, wm[:5].tolist()))
            r["correct"] = int(t == m_true)
            r["in_lib"] = int(m_true in wm)
            # ---- meta
            r["precursor_mz"] = float(q.precursor_mz)
            r["adduct_na"] = int(q.adduct == "[M+Na]+")
            r["instrument"] = {"Orbitrap": 0.0, "QTOF": 1.0}.get(q.instrument_type, np.nan)
            r["collision_energy"] = float(q.collision_energy) if pd.notna(q.collision_energy) else np.nan
            r["n_peaks"] = int(q.n_peaks)
            r["spec_entropy"] = float(q.spec_entropy)
            r["n_win_mol"] = int(len(wm))
            r["n_win_spec"] = int(wn.sum())
            # ---- sim
            r["s1"] = s[0]
            r["s2"] = s[1] if len(s) > 1 else 0.0
            r["s3"] = s[2] if len(s) > 2 else 0.0
            r["gap12"] = r["s1"] - r["s2"]
            r["gap13"] = r["s1"] - r["s3"]
            # ---- sim_extra
            r["mp1"] = int(wmp[0])
            r["mp1_frac"] = wmp[0] / max(q.n_peaks, 1)
            r["s1_mean"] = float(wmean[0])
            r["n_top1_spec"] = int(wn[0])
            # ---- analog (outside window)
            keepX = ~np.isin(Xm, excl)
            r["x1"] = float(Xs[keepX].max()) if keepX.any() else 0.0
            r["x1_minus_s1"] = r["x1"] - r["s1"]
            # ---- crowd_spec
            keepO = ~np.isin(Om, excl)
            om, os_ = Om[keepO], Os[keepO]
            r["n_open_lo"] = int(len(om))
            r["n_open_hi"] = int((os_ >= self.hi).sum())
            r["n_near_tie"] = int((s[1:] >= s[0] - self.dtie).sum())
            r["n_win_ge05"] = int((s >= self.lo).sum())
            tpos = self.mol_positions(t)
            tpos = tpos[phase_pos[tpos] & (tpos != qpos)]
            xpos = self.extra_excl_pos.get(i) if self.extra_excl_pos else None  # scenario a_x only
            if xpos is not None and len(xpos):
                tpos = tpos[~np.isin(tpos, xpos)]
            if len(tpos):
                nb = np.concatenate([self.sn_idx[self.sn_ptr[p]:self.sn_ptr[p + 1]] for p in tpos])
                nb = nb[phase_pos[nb] & (nb != qpos)]
                if xpos is not None and len(xpos):
                    nb = nb[~np.isin(nb, xpos)]
                nm = np.unique(self.lib_mol[nb])
                nm = nm[(nm != t) & ~np.isin(nm, excl)]
                r["top1_specdens"] = int(len(nm))
            else:
                r["top1_specdens"] = 0
            # ---- crowd_struct
            tn = self.struct_nbrs(t, self.tan)
            tn = tn[phase_mol[tn] & ~np.isin(tn, excl)]
            r["top1_struct_nbrs"] = int(len(tn))
            tan_w = tanimoto_packed(self.fps[t], self.fps[wm[1:]]) if len(wm) > 1 else np.zeros(0)
            r["tani12"] = float(tan_w[0]) if len(tan_w) else 0.0
            r["tani_mean_top5"] = float(tan_w[:4].mean()) if len(tan_w) else 0.0
            r["n_win_struct_nbrs"] = int((tan_w >= self.tan).sum())
            # ---- crowd_family (observable: families of candidates / library)
            code = self.fam_code[fam_level]
            ft = code[t]
            topk = wm[: self.topk]
            wts = s[: self.topk] + 1e-3
            cc = code[topk]
            u, inv = np.unique(cc, return_inverse=True)
            pw = np.bincount(inv, weights=wts) / wts.sum()
            r["fam_top1"] = self.fam_names[fam_level][ft]
            r["fam_ent_topk"] = float(-(pw * np.log(pw)).sum())
            r["fam_frac_topk_same"] = float((cc[1:] == ft).mean()) if len(cc) > 1 else 1.0
            r["fam_n_win_same"] = int((code[wm[1:]] == ft).sum())
            cnt = int(self._fam_lib_count(ph, fam_level, ft))
            if len(excl):
                ex = excl[phase_mol[excl]]
                cnt -= int(np.sum(code[ex] == ft))
            r["fam_lib_size"] = cnt
            r["fam_frac_open_same"] = float((code[om] == ft).mean()) if len(om) else 0.0
            tab = self.fam_tables[fam_level]
            r["fam_spec_crowd_train"] = float(tab["spec_crowd"][ft])
            r["fam_struct_crowd_train"] = float(tab["struct_crowd"][ft])
            # ---- analysis-only (truth-derived; never features)
            r["ana_true_fam"] = self.fam_names[fam_level][code[m_true]]
            r["ana_tani_top1_truth"] = float(tanimoto_packed(self.fps[m_true], self.fps[[t]])[0])
            out.append(r)
        return out

    def _fam_lib_count(self, ph, lv, ft):
        key = (ph, lv)
        if key not in self._fam_cache:
            self._fam_cache[key] = np.bincount(self.fam_code[lv][self.phase_mol[ph]], minlength=len(self.fam_names[lv]))
        return self._fam_cache[key][ft]


def feature_columns(groups):
    cols = []
    for g in groups:
        cols += GROUPS[g]
    return cols
