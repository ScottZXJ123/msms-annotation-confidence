"""Engine-agnostic interface: ingest any annotation engine's candidate ranking.

A candidate table has one row per (query, candidate) with at least:
    query_id   str    identifier of the MS/MS query (e.g. mzML scan / feature id)
    rank       int    1 = top-1
    score      float  engine score (higher = better); engine-specific scale
and one structure column: `smiles` (preferred) or `inchikey`.
Query metadata (one row per query): query_id, precursor_mz, adduct, instrument_type, collision_energy,
and optionally the processed peak list (needed only for library-neighbourhood features).

Adapters below map known export formats onto this schema. Column names for external tools are the ones
documented by those tools; they are validated at load time and a clear error is raised if a column is
missing (no silent guessing).

Data-protection rule (§6): in `lab_mode`, nothing may be sent to any network service. Families of
candidate structures are looked up ONLY in the local ClassyFire cache (pre-computed for public reference
libraries); unknown structures become 'unclassified'.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import Standardizer
from .features import tanimoto_packed

REQUIRED = ["query_id", "rank", "score"]


class LabDataGuard:
    """Refuses any attempt to use network lookups while lab data are loaded."""

    def __init__(self, lab_mode: bool):
        self.lab_mode = lab_mode

    def network_allowed(self) -> bool:
        return not self.lab_mode


def validate(cands: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in cands.columns]
    if missing or not ({"smiles", "inchikey"} & set(cands.columns)):
        raise ValueError(f"candidate table lacks columns {missing} or a structure column (smiles/inchikey)")
    return cands.sort_values(["query_id", "rank"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- adapters
def read_generic_csv(path) -> pd.DataFrame:
    return validate(pd.read_csv(path))


def read_gnps_library_search(path) -> pd.DataFrame:
    """GNPS classic library-search result TSV (one row per query-hit)."""
    df = pd.read_csv(path, sep="\t")
    need = ["#Scan#", "MQScore", "Smiles"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"GNPS export lacks column {c}")
    df = df.rename(columns={"#Scan#": "query_id", "MQScore": "score", "Smiles": "smiles"})
    df["rank"] = df.groupby("query_id").score.rank(ascending=False, method="first").astype(int)
    return validate(df[["query_id", "rank", "score", "smiles"]])


def read_sirius_structure_candidates(path) -> pd.DataFrame:
    """SIRIUS 5/6 `structure_candidates.tsv` (per compound folder) or the project-wide summary."""
    df = pd.read_csv(path, sep="\t")
    score_col = next((c for c in ["CSI:FingerIDScore", "csiScore", "score"] if c in df.columns), None)
    id_col = next((c for c in ["id", "featureId", "mappingFeatureId", "alignedFeatureId"] if c in df.columns), None)
    if score_col is None or id_col is None or "smiles" not in df.columns:
        raise ValueError("SIRIUS export: need an id column, a CSI:FingerID score column and 'smiles'")
    df = df.rename(columns={id_col: "query_id", score_col: "score"})
    if "rank" not in df.columns:
        df["rank"] = df.groupby("query_id").score.rank(ascending=False, method="first").astype(int)
    keep = ["query_id", "rank", "score", "smiles"] + [c for c in ["ConfidenceScoreExact", "ConfidenceScoreApproximate"] if c in df.columns]
    return validate(df[keep])


def read_msdial_alignment(path) -> pd.DataFrame:
    """MS-DIAL alignment export: top-1 only (no candidate list). Only top-1-based features can be computed;
    a model trained on top-1-only features must be used (documented limitation)."""
    df = pd.read_csv(path, sep="\t", skiprows=4)
    for c in ["Alignment ID", "SMILES", "Total score"]:
        if c not in df.columns:
            raise ValueError(f"MS-DIAL export lacks column {c}")
    df = df.rename(columns={"Alignment ID": "query_id", "SMILES": "smiles", "Total score": "score"})
    df["rank"] = 1
    return validate(df[["query_id", "rank", "score", "smiles"]].dropna(subset=["smiles"]))


# ----------------------------------------------------------------------------- candidate-only features
class CandidateFeaturizer:
    """Computes the CANDIDATE_ONLY feature set from a candidate table (no access to the engine's library).

    family_lookup: dict InChIKey(full or first block) -> {'class': ...}; built from the local cache only.
    family_tables: dict class name -> {'fam_spec_crowd_train', 'fam_struct_crowd_train'} computed on the
    public train-fold library (shipped with the frozen model).
    """

    def __init__(self, cfg, family_lookup: dict, family_tables: dict, guard: LabDataGuard):
        self.cfg = cfg
        self.std = Standardizer(cfg["structure"])
        self.fam = family_lookup
        self.tabs = family_tables
        self.guard = guard
        cr = cfg["crowding"]
        self.tan, self.dtie, self.lo, self.topk = cr["struct_tanimoto"], cr["near_tie_delta"], cr["spec_sim_lo"], cr["topk_family"]
        self._fp_cache = {}

    def _struct(self, smiles):
        if smiles not in self._fp_cache:
            rec = self.std.record(smiles)
            fp = self.std.fingerprint(smiles)
            if fp is None:
                self._fp_cache[smiles] = (None, None)
            else:
                arr = np.zeros((self.cfg["structure"]["fingerprint"]["nbits"],), np.uint8)
                from rdkit import DataStructs

                DataStructs.ConvertToNumpyArray(fp, arr)
                self._fp_cache[smiles] = (rec, np.packbits(arr))
        return self._fp_cache[smiles]

    def family(self, rec):
        if rec is None or rec.get("inchikey") is None:
            return "unclassified"
        v = self.fam.get(rec["inchikey"]) or self.fam.get(rec["ik14"])
        return v["class"] if v else "unclassified"

    def query_features(self, meta: dict, cands: pd.DataFrame) -> dict:
        c = cands.sort_values("rank")
        recs = [self._struct(s) for s in c.smiles]
        ok = [i for i, (r, fp) in enumerate(recs) if fp is not None]
        c = c.iloc[ok]
        recs = [recs[i] for i in ok]
        # collapse stereo-isomers/duplicates to the first-block key (identity definition, §4)
        seen, keep = set(), []
        for i, (r, _) in enumerate(recs):
            if r["ik14"] not in seen:
                seen.add(r["ik14"])
                keep.append(i)
        c = c.iloc[keep]
        recs = [recs[i] for i in keep]
        s = c.score.values.astype(float)
        fps = np.stack([fp for _, fp in recs]) if recs else np.zeros((0, 1), np.uint8)
        fam = [self.family(r) for r, _ in recs]
        f = {"precursor_mz": meta.get("precursor_mz"), "adduct_na": int(meta.get("adduct") == "[M+Na]+"),
             "instrument": {"Orbitrap": 0.0, "QTOF": 1.0}.get(meta.get("instrument_type"), np.nan),
             "collision_energy": meta.get("collision_energy", np.nan), "n_peaks": meta.get("n_peaks", np.nan),
             "spec_entropy": meta.get("spec_entropy", np.nan), "n_win_mol": len(s)}
        if len(s) == 0:
            return {**f, "has_candidate": 0}
        f.update({"has_candidate": 1, "top1_ik14": recs[0][0]["ik14"], "top1_class": fam[0],
                  "s1": s[0], "s2": s[1] if len(s) > 1 else 0.0, "s3": s[2] if len(s) > 2 else 0.0})
        f["gap12"], f["gap13"] = f["s1"] - f["s2"], f["s1"] - f["s3"]
        f["n_near_tie"] = int((s[1:] >= s[0] - self.dtie).sum())
        f["n_win_ge05"] = int((s >= self.lo).sum())
        tw = tanimoto_packed(fps[0], fps[1:]) if len(s) > 1 else np.zeros(0)
        f["tani12"] = float(tw[0]) if len(tw) else 0.0
        f["tani_mean_top5"] = float(tw[:4].mean()) if len(tw) else 0.0
        f["n_win_struct_nbrs"] = int((tw >= self.tan).sum())
        fk = fam[: self.topk]
        w = s[: self.topk] + 1e-3
        u = {}
        for name, ww in zip(fk, w):
            u[name] = u.get(name, 0.0) + ww
        p = np.array(list(u.values())) / w.sum()
        f["fam_ent_topk"] = float(-(p * np.log(p)).sum())
        f["fam_frac_topk_same"] = float(np.mean([x == fam[0] for x in fk[1:]])) if len(fk) > 1 else 1.0
        f["fam_n_win_same"] = int(sum(x == fam[0] for x in fam[1:]))
        t = self.tabs.get(fam[0], {})
        f["fam_spec_crowd_train"] = t.get("fam_spec_crowd_train", np.nan)
        f["fam_struct_crowd_train"] = t.get("fam_struct_crowd_train", np.nan)
        return f

    def transform(self, meta: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
        cands = validate(cands)
        g = cands.groupby("query_id")
        rows = []
        for _, m in meta.iterrows():
            cc = g.get_group(m.query_id) if m.query_id in g.groups else cands.iloc[:0]
            rows.append({"query_id": m.query_id, **self.query_features(m.to_dict(), cc)})
        return pd.DataFrame(rows)


def load_local_family_lookup(cache_file: Path) -> dict:
    out = {}
    for line in open(cache_file):
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("superclass"):
            cls = r["class"] or f"{r['superclass']} [no class]"
            out[r["inchikey"]] = {"class": cls}
            out.setdefault(r["inchikey"][:14], {"class": cls})
    return out
