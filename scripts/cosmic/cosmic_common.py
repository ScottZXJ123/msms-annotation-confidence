"""Shared helpers for the COSMIC comparisons (PubChem background run and same-candidate-set head-to-head).

Pre-declared evaluation rules (fixed before the custom-database results existed; LOG.md 2026-09-27):
- in-library mixtures: pi in config evaluation.prevalence_headline (0.2, 0.5); scenario (a) rows get total weight
  pi, scenario (b) rows 1 - pi, within each stratum;
- a metric that depends on the minority class is flagged "unstable" when that class has < MIN_CLASS rows
  (unweighted), MIN_CLASS = evaluation.min_group_mol_test (20), the same minimum used for group calibration;
- a system that returns no annotation for a query scores confidence 0 and counts as not correct;
- bootstrap: queries are resampled (both scenario rows of a query stay together).
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from msconf.config import ROOT, load_config, path  # noqa: E402
from msconf.evalutil import class_weights, components, em_w, probs  # noqa: E402
from msconf.evaluation import auroc, calibration_metrics, prior_shift, risk_coverage  # noqa: E402

cfg = load_config()
der = path(cfg, "derived")
MIN_CLASS = int(cfg["evaluation"]["min_group_mol_test"])
PIS = [float(p) for p in cfg["evaluation"]["prevalence_headline"]]
ID = re.compile(r"MassSpecGymID\d+")


# ------------------------------------------------------------------ SIRIUS summaries
def read_summaries(files, std):
    frames = []
    for f in files:
        sep = "\t" if f.suffix == ".tsv" else ","
        d = pd.read_csv(f, sep=sep)
        ids = []
        for _, r in d.iterrows():
            m = next((ID.search(str(v)) for v in r.values if isinstance(v, str) and ID.search(v)), None)
            ids.append(m.group(0) if m else None)
        d["identifier"] = ids
        d["_source"] = f.parent.name
        frames.append(d)
    if not frames:
        return None
    S = pd.concat(frames, ignore_index=True)
    S = S[S.identifier.notna()]
    cex = next(c for c in S.columns if c.lower().replace("_", "") in ("confidencescoreexact", "siriusconfidencescoreexact"))
    cap = next(c for c in S.columns if c.lower().replace("_", "") in ("confidencescoreapproximate", "siriusconfidencescoreapproximate"))
    smi_col = next(c for c in S.columns if c.lower() in ("smiles", "sirius_smiles"))
    S["cosmic_exact"] = pd.to_numeric(S[cex], errors="coerce").replace([np.inf, -np.inf], np.nan)  # -Infinity = not computed
    S["cosmic_approx"] = pd.to_numeric(S[cap], errors="coerce").replace([np.inf, -np.inf], np.nan)
    S["pred_smiles"] = S[smi_col]
    S["pred_ik14"] = S[smi_col].map(lambda s: std.record(s)["ik14"] if isinstance(s, str) else None)
    ik2d = next((c for c in S.columns if c.lower() == "inchikey2d"), None)
    S["sirius_ik14"] = S[ik2d] if ik2d else None
    S["links"] = S["links"] if "links" in S.columns else None
    return S


# ------------------------------------------------------------------ proposed system on the same queries
def primary_system():
    import json

    prim = json.load(open(path(cfg, "results") / "selection.json"))["primary_system"]
    return prim, pickle.loads((der / "models" / "entropy" / f"{prim}.pkl").read_bytes())


def proposed_rows(identifiers, split="test"):
    """Feature rows of the proposed system for the given queries: scenario (a) (leave-one-spectrum-out) and
    scenario "bset" (ALL spectra of ALL given query molecules removed from the library at once).
    Computed from the stored per-query search lists; nothing is re-fitted."""
    from msconf.features import FeatureContext

    Q = pd.read_parquet(der / "queries.parquet")
    L = dict(np.load(der / "qlists.npz"))
    ctx = FeatureContext(cfg, pd.read_parquet(der / "lib_index.parquet"), dict(np.load(der / "spec_nbrs07.npz")),
                         dict(np.load(der / "struct_nbrs.npz")), np.load(der / "fps.npy"),
                         pd.read_parquet(der / "families.parquet"), Q, L)
    sel = Q.index[Q.identifier.isin(set(identifiers))].values
    assert len(sel) == len(set(identifiers)), "every query must be in queries.parquet"
    assert set(Q.fold.values[sel]) == {split}
    ctx.set_excl_mols = np.unique(Q.mol_idx.values[sel]).astype(np.int64)
    rows = []
    for i in sel:
        rows.extend(ctx.rows_for_query(int(i), scenarios=("a", "bset"), engine="entropy", fam_level=cfg["family"]["primary_level"]))
    df = pd.DataFrame(rows)
    fam = pd.read_parquet(der / "families.parquet")
    for lv in ["superclass", "class", "subclass"]:
        lab = fam[lv].values
        df[f"top1_{lv}"] = np.where(df.has_candidate == 1, lab[df.top1.fillna(0).astype(int)], None)
    df["identifier"] = df.row.map(dict(zip(Q.row.values, Q.identifier.values)))
    df["w"] = 1.0
    df["scenario"] = df.scenario.replace({"bset": "b"})
    for c in ["correct", "in_lib"]:
        df[c] = df[c].fillna(0).astype(int)
    return df


# ------------------------------------------------------------------ metrics
def em_direct(p, prior_cal, w, n_iter=300):
    """Saerens EM on a direct P(correct): estimate the batch prior of 'correct' and shift the probabilities.
    This is the same prior correction applied to every direct system (REPORT §5.2)."""
    pi = prior_cal
    for _ in range(n_iter):
        new = float(np.sum(w * prior_shift(p, prior_cal, pi)) / np.sum(w))
        if abs(new - pi) < 1e-8:
            pi = new
            break
        pi = new
    return prior_shift(p, prior_cal, pi), pi


def fast_cal(p, y, w, n_bins=10):
    """ECE (equal-mass bins), log loss, Brier and mean prediction exactly as msconf.evaluation.calibration_metrics,
    without its (slow) logistic recalibration fit, which is not used here."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    W = w.sum()
    o = np.argsort(p, kind="stable")
    cw = np.cumsum(w[o]) / W
    b = np.minimum((cw * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            ww = w[o][m]
            ece += ww.sum() / W * abs(np.average(p[o][m], weights=ww) - np.average(y[o][m], weights=ww))
    return {"ece": float(ece), "logloss": float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / W),
            "brier": float(np.sum(w * (p - y) ** 2) / W), "mean_p": float(np.average(p, weights=w))}


def wmetrics(conf, y, w, prob=None):
    """Weighted metrics + unweighted class counts + instability flag."""
    y = np.asarray(y, float)
    conf = np.asarray(conf, float)
    w = np.asarray(w, float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    out = {"n": int(len(y)), "n_pos": n_pos, "n_neg": n_neg, "unstable": bool(min(n_pos, n_neg) < MIN_CLASS)}
    if min(n_pos, n_neg) == 0:
        out["accuracy"] = float(np.average(y, weights=w))
        return out
    rc = risk_coverage(conf, y, w)
    out.update({"accuracy": float(np.average(y, weights=w)), "aurc": rc["aurc"], "e_aurc": rc["e_aurc"],
                "cov@risk0.1": rc["cov@risk0.1"], "auroc": auroc(conf, y, w)})
    if prob is not None:
        c = fast_cal(prob, y, w)
        out.update({"ece": c["ece"], "logloss": c["logloss"], "brier": c["brier"], "mean_pred": c["mean_p"]})
        tr = np.asarray(prob) >= cfg["tiers"]["trusted_min_p_correct"]
        out["trusted"] = {"frac": float(w[tr].sum() / w.sum()), "n": int(tr.sum()),
                          "acc": float(np.average(y[tr], weights=w[tr])) if tr.any() else None,
                          "pred": float(np.average(np.asarray(prob)[tr], weights=w[tr])) if tr.any() else None}
    return out


def boot_ci(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5]).tolist() if len(v) >= 50 else None


def bootstrap(d, fn, B, rng, key="identifier"):
    """Percentile bootstrap over queries; fn(sub) -> dict of floats. Returns {metric: [lo, hi]}."""
    groups = d.groupby(key).indices
    keys = np.array(list(groups.keys()))
    vals = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys))
        ix = np.concatenate([groups[keys[k]] for k in pick])
        try:
            vals.append(fn(d.iloc[ix]))
        except Exception:
            continue
    out = {}
    for k in (vals[0].keys() if vals else []):
        v = [x.get(k, np.nan) for x in vals]
        if all(isinstance(z, (int, float, np.floating)) or z is None for z in v):
            out[k] = boot_ci([np.nan if z is None else z for z in v])
    return out


def mces_le2_factory():
    try:
        from myopic_mces import MCES

        def f(a, b):
            if not isinstance(a, str) or not isinstance(b, str):
                return np.nan
            try:
                return float(MCES(a, b, threshold=2, solver="PULP_CBC_CMD", solver_options={"msg": False})[1] <= 2)
            except Exception:
                return np.nan
        return f, "myopic_mces, threshold 2"
    except ImportError:
        return (lambda a, b: np.nan), "unavailable"


def mix_weights(x, pi):
    """Scenario (a) rows get total weight pi, (b) rows 1 - pi (None: equal weight per row)."""
    if pi is None:
        return np.ones(len(x))
    return class_weights(x.scenario.eq("a").values.astype(float), np.ones(len(x)), pi)


def attach_components(sysm, x):
    """Pre-compute the frozen system's components once per row (P(correct|in), calibrated P(in))."""
    x = x.copy()
    has = x.has_candidate.values == 1
    comp = components(sysm, x[has])
    x["_pc"], x["_pincal"] = 0.0, 0.0
    x.loc[has, "_pc"], x.loc[has, "_pincal"] = comp["pc"], comp["pin_cal"]
    x.attrs["pi_cal"] = comp["pi_cal"]
    return x


def proposed_eval(x, pi, label="correct", correction="em", pi_cal=None):
    """Proposed system on rows x (scenario a and/or b; attach_components() applied) at in-library mixture pi,
    with the EM prior correction (correction="em"), none (None), or a known in-library fraction (number; 1.0 for
    scenario (a) alone). Rows without any library candidate: no annotation, confidence 0."""
    pi_cal = pi_cal if pi_cal is not None else x.attrs.get("pi_cal")
    w = mix_weights(x, pi)
    has = x.has_candidate.values == 1
    comp = {"direct": False, "pc": x._pc.values[has], "pin_cal": x._pincal.values[has], "pi_cal": pi_cal}
    if correction == "em":
        pi_use = em_w(comp["pin_cal"], pi_cal, w[has])
    else:
        pi_use = correction  # None = no correction; a number = known (oracle) in-library fraction
    p = np.zeros(len(x))
    pin = np.zeros(len(x))
    p[has], pin[has] = probs(comp, pi_use)
    y = x[label].fillna(0).values.astype(float)
    r = wmetrics(p, y, w, p)
    r["pi_hat"] = pi_use
    return r


class MCESCache:
    """Disk cache of MCES<=2 decisions (public structures only)."""

    def __init__(self, file):
        import json

        self.file = file
        self.d = json.load(open(file)) if Path(file).exists() else {}
        self.f, self.method = mces_le2_factory()

    def __call__(self, a, b):
        if not isinstance(a, str) or not isinstance(b, str):
            return np.nan
        k = a + " >> " + b
        if k not in self.d:
            v = self.f(a, b)
            self.d[k] = None if v != v else v
        v = self.d[k]
        return np.nan if v is None else v

    def save(self):
        import json

        json.dump(self.d, open(self.file, "w"))
