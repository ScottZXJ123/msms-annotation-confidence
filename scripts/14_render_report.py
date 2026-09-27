"""Render REPORT.md and WETLAB_PREREG.md from templates + results/*.json.

Every number in the rendered documents comes from a results file through an expression in the template
(`{{ expr }}` or `{{ expr :fmt }}`); nothing is typed in by hand. Tables are produced by the helper
functions below.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import ROOT, git_commit, load_config, path  # noqa: E402

cfg = load_config()
RES = path(cfg, "results")
TAB = path(cfg, "tables")


class Results(dict):
    def __missing__(self, k):
        f = RES / f"{k}.json"
        if not f.exists():
            raise KeyError(f"results/{k}.json not found")
        self[k] = json.load(open(f))
        return self[k]


J = Results()
P = J["selection"]["primary_system"]
NAMES = {P: "**Proposed**", "B_ms2q_rerank_direct": "MS2Query-style, own top-1", "B_ms2q_conf_direct": "MS2Query-style score of our top-1",
         "B_ms2q_dec": "MS2Query features in our decomposed framework", "B_tda_margin_direct": "Target−decoy margin (s1 − d1)", "M_full": "Full features (decomposed)", "M_full_gc": "Full + family offsets",
         "M_base": "No crowding (base)", "M_sim_plus_crowd": "Similarity + crowding", "M_full_minus_family": "Full − family crowding",
         "M_full_minus_spec": "Full − spectral crowding", "M_full_minus_struct": "Full − structural crowding",
         "M_candidate_only": "Candidate-list-only (engine-agnostic)", "M_full_direct": "Full features, direct P(correct)",
         "M_full_lr": "Full features, logistic", "B_s1_direct": "Score threshold s1", "B_gap_direct": "Gap s1−s2",
         "B_lr_direct": "Logistic(s1, gap)", "B_s1_dec": "s1, decomposed", "B_gap_dec": "gap, decomposed",
         "B_lr_dec": "Logistic(s1, gap), decomposed", "F_meta": "Floor: metadata only", "F_sim": "Floor: similarity only"}


def short_commit():
    return git_commit()["commit"][:12]


NAMES["M_full_nocounts"] = "Full without window counts"
NAMES["M_full_with_libsize"] = "Full + fam_lib_size (shortcut demo)"
NAMES[P] = "**Proposed** (full − family features)"


def f3(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def ci(v, d=3):
    return f"[{v[0]:.{d}f}, {v[1]:.{d}f}]"


def _esc(x):
    return str(x).replace("|", "\\|")


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(_esc(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_esc(r[c]) for c in cols) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- tables
def table_main(pi, mode="em", systems=None):
    sw = J["eval_sweep"]["sweep"]
    systems = systems or [P, "M_full", "M_base", "M_candidate_only", "M_full_direct", "M_full_lr", "B_lr_dec", "B_lr_direct",
                          "B_s1_direct", "B_gap_direct", "B_s1_dec", "B_gap_dec", "F_sim", "F_meta"]
    rows = []
    for s in dict.fromkeys(systems):
        if s not in sw:
            continue
        r = sw[s][str(pi)][mode]
        rows.append({"System": NAMES.get(s, s), "AURC ↓": f3(r["aurc"]), "e-AURC ↓": f"{r['e_aurc']:.4f}",
                     "Cov@risk≤5%": pct(r["cov@risk0.05"]), "Cov@risk≤10%": pct(r["cov@risk0.1"]),
                     "AUROC(correct)": f3(r["auroc_correct"]), "AUROC(in-lib)": f3(r["auroc_in"]),
                     "Log loss ↓": f3(r["cal_logloss"]), "ECE ↓": f3(r["cal_ece"]),
                     "Mean pred / obs": f"{r['cal_mean_p']:.3f} / {r['cal_mean_y']:.3f}"})
    return md(pd.DataFrame(rows))


def table_boot(pi):
    b = J["eval_bootstrap"]["boot"][str(pi)]["diffs"]
    rows = []
    for k, v in b.items():
        a, c = k.split(" - ")
        rows.append({"Comparison": f"{NAMES.get(a, a)} − {NAMES.get(c, c)}", "ΔAURC": f"{v['d_aurc']:+.4f}",
                     "95% CI": ci(v["d_aurc_ci"], 4), "ΔLog loss": f"{v['d_logloss']:+.4f}", "95% CI ": ci(v["d_logloss_ci"], 4)})
    return md(pd.DataFrame(rows))


def table_prevalence(system=None):
    sw = J["eval_sweep"]["sweep"]
    system = system or P
    rows = []
    for pi in cfg["evaluation"]["prevalence_grid"]:
        r = sw[system][str(pi)]
        rows.append({"π": pi, "obs. acc": f3(r["oracle"]["cal_mean_y"]), "pred (true π)": f3(r["oracle"]["cal_mean_p"]),
                     "pred (EM π̂)": f3(r["em"]["cal_mean_p"]), "π̂ (EM)": f3(r["em"]["pi_used"]),
                     "pred (no corr.)": f3(r["none"]["cal_mean_p"]),
                     "ECE true/EM/none": f"{f3(r['oracle']['cal_ece'])} / {f3(r['em']['cal_ece'])} / {f3(r['none']['cal_ece'])}",
                     "Direct model pred": f3(sw["M_full_direct"][str(pi)]["none"]["cal_mean_p"]) if "M_full_direct" in sw else "–"})
    return md(pd.DataFrame(rows))


def table_tiers(pi, system=None, mode="em"):
    r = J["eval_sweep"]["sweep"][system or P][str(pi)][mode]["tiers"]
    rows = []
    for t in ["trusted", "suspicious", "absent"]:
        v = r[t]
        rows.append({"Tier": t, "Share of queries": pct(v.get("frac")), "Top-1 accuracy (obs)": pct(v.get("acc")),
                     "Predicted": pct(v.get("pred_acc")), "In-library rate (obs)": pct(v.get("in_rate")),
                     "Predicted in-lib": pct(v.get("pred_in"))})
    return md(pd.DataFrame(rows))


def table_wetlab(pi, key=None):
    w = J["eval_wetlab"]["wetlab"][str(pi)][key or f"{P}|em"]
    rows = []
    for t in ["trusted", "suspicious", "absent"]:
        v = w.get(t, {})
        if not v or not v.get("n_repeats_nonempty"):
            rows.append({"Tier": t, "Note": "empty"})
            continue
        rows.append({"Tier": t, "Mean tier size / batch": f"{v['mean_tier_size_in_batch']:.0f}",
                     "Obs. acc (mean, 2.5–97.5%)": f"{pct(v['obs_acc_mean'])} ({pct(v['obs_acc_q025'])}–{pct(v['obs_acc_q975'])})",
                     "Pred. acc": pct(v["pred_acc_mean"]), "P(pred ∈ 95% CI)": pct(v["acc_consistent_rate"]),
                     "Obs. in-lib": pct(v["obs_in_mean"]), "Pred. in-lib": pct(v["pred_in_mean"]),
                     "P(pred in-lib ∈ CI)": pct(v["in_consistent_rate"])})
    return md(pd.DataFrame(rows))


def table_groups(pi, systems=None, groupings=None):
    g = J["eval_groups"]["groups"]
    systems = systems or [P, "M_base", "B_lr_dec", "B_lr_direct", "M_full_direct"]
    groupings = groupings or ["top1_class", "top1_superclass", "top1_subclass", "bin_top1_struct_nbrs",
                              "bin_fam_spec_crowd_train", "bin_n_open_hi", "bin_top1_specdens", "ana_true_class"]
    rows = []
    for s in dict.fromkeys(systems):
        if s not in g:
            continue
        for gc in groupings:
            v = g[s][str(pi)][gc]
            rows.append({"System": NAMES.get(s, s), "Grouping": gc, "Groups (sufficient/all)": f"{v['n_sufficient']}/{v['n_groups']}",
                         "Rows in sufficient groups": pct(v["frac_rows_in_sufficient"]),
                         "Pred ∈ 95% CI": pct(v["frac_consistent_sufficient"]), "Mean |obs−pred|": f3(v["mean_abs_gap_sufficient"]),
                         "Max |obs−pred|": f3(v["max_abs_gap_sufficient"])})
    return md(pd.DataFrame(rows))


def table_component_groups(system=None):
    g = J["eval_groups"]["groups"][system or P]
    rows = []
    for comp, key in [("P(correct | in)", "component_C"), ("P(in), π=0.5", "component_I_pi0.5")]:
        for gc, v in g.get(key, {}).items():
            rows.append({"Component": comp, "Grouping": gc, "Groups (sufficient/all)": f"{v['n_sufficient']}/{v['n_groups']}",
                         "Pred ∈ 95% CI": pct(v["frac_consistent_sufficient"]), "Mean |obs−pred|": f3(v["mean_abs_gap_sufficient"]),
                         "Max |obs−pred|": f3(v["max_abs_gap_sufficient"])})
    return md(pd.DataFrame(rows))


def table_incremental():
    inc = J["eval_incremental"]["component_comparisons"]
    rows = []
    for k, v in inc.items():
        a, c = k.split(" vs ")
        rows.append({"Comparison": f"{NAMES.get(a, a)} vs {NAMES.get(c, c)}",
                     "P(correct|in) log loss": f"{v['C_logloss'][0]:.4f} vs {v['C_logloss'][1]:.4f}", "Δ 95% CI": ci(v["C_dlogloss_ci"], 4),
                     "P(correct|in) AUROC": f"{v['C_auroc'][0]:.3f} vs {v['C_auroc'][1]:.3f}", "ΔAUROC 95% CI": ci(v["C_dauroc_ci"], 3),
                     "P(in) log loss": f"{v['I_logloss'][0]:.4f} vs {v['I_logloss'][1]:.4f}", "Δ 95% CI ": ci(v["I_dlogloss_ci"], 4),
                     "P(in) AUROC": f"{v['I_auroc'][0]:.3f} vs {v['I_auroc'][1]:.3f}"})
    return md(pd.DataFrame(rows))


def table_mh():
    mh = J["eval_incremental"]["mantel_haenszel_error_vs_crowding"]
    rows = []
    for k, v in mh.items():
        rows.append({"Crowding measure (> val median)": k, "MH odds ratio of error": f3(v["mh_odds_ratio_error"]),
                     "95% CI": ci(v["ci95"], 2) if v["ci95"][0] is not None else "–"})
    return md(pd.DataFrame(rows))


def table_shift_instrument(pi="0.2", mode="em"):
    s = J["eval_shift_instrument"]
    rows = []
    for key in [k for k in s if k.startswith("test_on_")]:
        for name in [P.replace("_gc", ""), "M_base", "B_lr_dec"]:
            if name not in s[key]:
                continue
            x, r = s[key][name]["cross_instrument"][pi][mode], s[key][name]["all_instruments_reference"][pi][mode]
            rows.append({"Test instrument": key.replace("test_on_", ""), "System": NAMES.get(name, name),
                         "AURC cross / ref": f"{f3(x['aurc'])} / {f3(r['aurc'])}",
                         "ECE cross / ref": f"{f3(x['cal_ece'])} / {f3(r['cal_ece'])}",
                         "Pred/obs acc (cross)": f"{f3(x['cal_mean_p'])} / {f3(x['cal_mean_y'])}",
                         "Trusted-tier acc (cross)": pct(x["tiers"]["trusted"].get("acc"))})
    return md(pd.DataFrame(rows))


def table_external(pi="0.2", mode="em"):
    e = J["eval_external"]["systems"]
    rows = []
    for name, v in e.items():
        if name.startswith("_"):
            continue
        r = v[pi][mode]
        rows.append({"System": NAMES.get(name, name), "AURC": f3(r["aurc"]), "AUROC(in-lib)": f3(r["auroc_in"]),
                     "ECE": f3(r["cal_ece"]), "Pred/obs acc": f"{f3(r['cal_mean_p'])} / {f3(r['cal_mean_y'])}",
                     "Trusted acc (share)": f"{pct(r['tiers']['trusted'].get('acc'))} ({pct(r['tiers']['trusted'].get('frac'))})"})
    return md(pd.DataFrame(rows))


def table_external_sources():
    e = J["eval_external"]["systems"].get("_per_source_primary", {})
    rows = []
    for src, v in e.items():
        r = v["0.5"]
        rows.append({"Source (MassBank contributor)": src, "Molecules": v["n_mol"], "AURC": f3(r["aurc"]), "ECE": f3(r["cal_ece"]),
                     "Pred/obs acc": f"{f3(r['cal_mean_p'])} / {f3(r['cal_mean_y'])}", "AUROC(in-lib)": f3(r["auroc_in"])})
    return md(pd.DataFrame(rows))


def table_ax(pi="0.2"):
    x = J["eval_cross_instrument_library"]["systems"]
    rows = []
    for name, v in x.items():
        for scen in ["a_same_queries", "a_x"]:
            for mode in ["oracle", "em"]:
                r = v[scen][pi][mode]
                rows.append({"System": NAMES.get(name, name), "In-library scenario": scen, "Prevalence": mode,
                             "AURC": f3(r["aurc"]), "ECE": f3(r["cal_ece"]), "Pred/obs acc": f"{f3(r['cal_mean_p'])} / {f3(r['cal_mean_y'])}",
                             "Trusted acc (share)": f"{pct(r['tiers']['trusted'].get('acc'))} ({pct(r['tiers']['trusted'].get('frac'))})"})
    return md(pd.DataFrame(rows))


def table_engine_cosine(pi="0.2", mode="em"):
    e = J["eval_extra"]["engine_cosine"]
    rows = []
    for name, v in e.items():
        r = v[pi][mode]
        rows.append({"System (cosine engine)": NAMES.get(name, name), "AURC": f3(r["aurc"]), "AUROC(in-lib)": f3(r["auroc_in"]),
                     "Log loss": f3(r["cal_logloss"]), "ECE": f3(r["cal_ece"])})
    return md(pd.DataFrame(sorted(rows, key=lambda d: d["AURC"])))


def table_hardest(n=10):
    t = pd.read_csv(TAB / "hardest_families_top1_class.csv").head(n)
    t["error (95% CI)"] = [f"{pct(e)} ({pct(l)}–{pct(h)})" for e, l, h in zip(t.err, t.lo, t.hi)]
    return md(t[["top1_class", "n", "n_mol", "error (95% CI)"]].rename(columns={"top1_class": "Top-1 family (class)", "n": "Queries", "n_mol": "Molecules"}))


def table_checks():
    c = J["checks"]
    return md(pd.DataFrame([{"Check": k, "Result": ("PASS" if v["pass"] else "FAIL") if not k.startswith("info_") else "info",
                             "Detail": "" if v["detail"] is None else (f"{v['detail']:.3f}" if isinstance(v["detail"], float) else v["detail"])}
                            for k, v in c.items() if not k.startswith("_")]))


def table_ext_variants(pi="0.2", mode="em"):
    rows = []
    for v, name in [("v1", "eval_external"), ("v2", "eval_external_v2"), ("v3", "eval_external_v3")]:
        if not (RES / f"{name}.json").exists():
            continue
        e = J[name]
        for s_ in [P, "M_base", "B_lr_dec", "B_lr_direct", "B_s1_direct"]:
            if s_ not in e["systems"]:
                continue
            r = e["systems"][s_][pi][mode]
            t = r["tiers"]["trusted"]
            rows.append({"set": v, "molecules": e["n_mol"], "engine top-1 acc (a)": pct(e["top1_acc_a"]), "system": NAMES.get(s_, s_),
                         "AUROC(in-lib)": f3(r["auroc_in"]), "AURC": f3(r["aurc"]), "ECE": f3(r["cal_ece"]),
                         "pred/obs acc": f"{f3(r['cal_mean_p'])} / {f3(r['cal_mean_y'])}", "π̂": f3(r["pi_used"]),
                         "trusted: obs (pred), share": f"{pct(t.get('acc'))} ({pct(t.get('pred_acc'))}), {pct(t.get('frac'))}"})
    return md(pd.DataFrame(rows))


def table_ext_sources(name="eval_external_v3"):
    e = J[name]["systems"].get("_per_source_primary", {})
    rows = []
    for src, v in e.items():
        r = v["0.5"]
        rows.append({"source (MassBank contributor)": src, "molecules": v["n_mol"], "AUROC(in-lib)": f3(r["auroc_in"]),
                     "AURC": f3(r["aurc"]), "ECE": f3(r["cal_ece"]), "pred/obs acc": f"{f3(r['cal_mean_p'])} / {f3(r['cal_mean_y'])}",
                     "trusted obs (pred)": f"{pct(r['tiers']['trusted'].get('acc'))} ({pct(r['tiers']['trusted'].get('pred_acc'))})"})
    return md(pd.DataFrame(rows))


def table_new_baselines(pi="0.2", mode="em"):
    nbj = J["eval_new_baselines"]
    rows = []
    for n, v in nbj["sweep"].items():
        r = v[pi][mode]
        t = r["tiers"]["trusted"]
        d = nbj["bootstrap"][pi].get(f"{P} - {n}")
        rows.append({"system": NAMES.get(n, n), "e-AURC": f"{r['e_aurc']:.4f}", "Cov@risk≤5%": pct(r["cov@risk0.05"]),
                     "AUROC(correct)": f3(r["auroc_correct"]), "Log loss": f3(r["cal_logloss"]), "ECE": f3(r["cal_ece"]),
                     "pred/obs acc": f"{f3(r['cal_mean_p'])} / {f3(r['cal_mean_y'])}",
                     "trusted obs (pred), share": f"{pct(t.get('acc'))} ({pct(t.get('pred_acc'))}), {pct(t.get('frac'))}",
                     "ΔAURC proposed − this [95% CI]": "—" if d is None else f"{d['d_aurc']:+.4f} {ci(d['d_aurc_ci'], 4)}",
                     "ΔLog loss [95% CI]": "—" if d is None else f"{d['d_logloss']:+.4f} {ci(d['d_logloss_ci'], 4)}"})
    return md(pd.DataFrame(rows))


def table_tda():
    t = J["eval_new_baselines"]["tda_set_level"]
    rows = []
    for pi, v in t.items():
        rows.append({"π": pi, "TDA q≤0.10: coverage": pct(v["tda_cov"]["mean"]), "TDA realized error": pct(v["tda_err"]["mean"]),
                     "TDA est. FDR": pct(v["tda_est_fdr"]["mean"]), "P(TDA error > 10%)": pct(v["P(tda_err > 0.10)"]),
                     "Proposed trusted: coverage": pct(v["prop_cov"]["mean"]), "Proposed realized error": pct(v["prop_err"]["mean"]),
                     "Proposed predicted error": pct(v["prop_est_err"]["mean"]), "P(proposed error > 10%)": pct(v["P(prop_err > 0.10)"]),
                     "AURC TDA / proposed": f"{v['tda_aurc']['mean']:.3f} / {v['prop_aurc']['mean']:.3f}"})
    return md(pd.DataFrame(rows))


def table_ext_boot(name="eval_external_v3"):
    b = J[name]["bootstrap"]
    rows = []
    for pi, v in b.items():
        for k, x in v.items():
            a, c = k.split(" - ")
            rows.append({"π": pi, "comparison": f"{NAMES.get(a, a)} − {NAMES.get(c, c)}",
                         "Δ e-AURC [95% CI]": f"{x['d_e_aurc']:+.4f} {ci(x['d_e_aurc_ci'], 4)}",
                         "Δ log loss [95% CI]": f"{x['d_logloss']:+.4f} {ci(x['d_logloss_ci'], 4)}"})
    return md(pd.DataFrame(rows))


def prereg_table(resname):
    w = J[resname]["wetlab"]
    key = f"{P}|em"
    rows = []
    for pi in ["0.1", "0.2", "0.3", "0.5"]:
        if pi not in w:
            continue
        v = w[pi][key] if key in w[pi] else w[pi]
        for t in ["trusted", "suspicious", "absent"]:
            x = v.get(t, {})
            if not x or not x.get("n_repeats_nonempty"):
                continue
            rows.append({"batch in-library fraction π": pi, "tier": t, "tier size / batch": f"{x['mean_tier_size_in_batch']:.0f}",
                         "predicted accuracy": pct(x["pred_acc_mean"]),
                         "obs. accuracy, mean (2.5–97.5%)": f"{pct(x['obs_acc_mean'])} ({pct(x['obs_acc_q025'])}–{pct(x['obs_acc_q975'])})",
                         "P(pred ∈ 95% CI)": pct(x["acc_consistent_rate"]),
                         "obs. in-library (pred.)": f"{pct(x['obs_in_mean'])} ({pct(x['pred_in_mean'])})"})
    return md(pd.DataFrame(rows))


# ----------------------------------------------------------------------------- scalar accessors
def sw(s, pi, key, mode="em"):
    return J["eval_sweep"]["sweep"][s][str(pi)][mode][key]


def tier(s, pi, t, key, mode="em"):
    return J["eval_sweep"]["sweep"][s][str(pi)][mode]["tiers"][t].get(key)


def wl(pi, t, key, sysmode=None):
    return J["eval_wetlab"]["wetlab"][str(pi)][sysmode or f"{P}|em"][t].get(key)


def bd(pi, a, c, key):
    return J["eval_bootstrap"]["boot"][str(pi)]["diffs"][f"{a} - {c}"][key]


def inc(pair, key):
    return J["eval_incremental"]["component_comparisons"][pair][key]


def mh(c, key):
    return J["eval_incremental"]["mantel_haenszel_error_vs_crowding"][c][key]


def grp(s, pi, g, key):
    return J["eval_groups"]["groups"][s][str(pi)][g][key]


def grp_max_crowd(s, pi):
    g = J["eval_groups"]["groups"][s][str(pi)]
    return max(v["max_abs_gap_sufficient"] for k, v in g.items() if k.startswith("bin_") and v.get("n_sufficient"))


def nb(n, pi, key, mode="em"):
    return J["eval_new_baselines"]["sweep"][n][str(pi)][mode][key]


def nbd(pi, n, key):
    return J["eval_new_baselines"]["bootstrap"][str(pi)][f"{P} - {n}"][key]


def tda(pi, key, stat="mean"):
    v = J["eval_new_baselines"]["tda_set_level"][str(pi)][key]
    return v[stat] if isinstance(v, dict) else v


def ax(s, scen, pi, key, mode="em"):
    return J["eval_cross_instrument_library"]["systems"][s][scen][str(pi)][mode][key]


def ext(s, pi, key, mode="em"):
    return J["eval_external"]["systems"][s][str(pi)][mode][key]


def shift(test_on, s, which, pi, key, mode="em"):
    return J["eval_shift_instrument"][f"test_on_{test_on}"][s][which][str(pi)][mode][key]


def nc(pi, pair, key):
    v = J["eval_neighbourhood_comparison"][str(pi)]["diffs"][pair][key]
    return f"{v['d']:+.4f} {ci(v['ci95'], 4)}"


def ext3(s, pi, mode="em"):
    return J["eval_external_v3"]["systems"][s][str(pi)][mode]


def cosmic_h2h_section():
    if not cosmic_h2h_available():
        return ("**Pending.** The same-candidate-set SIRIUS run (`cosmic/customdb/`) has not been imported; no number from it appears "
                "in this report until `scripts/cosmic/ingest_cosmic_customdb.py` has written `results/eval_cosmic_customdb.json`.")
    c = J["eval_cosmic_customdb"]
    ch = c["checks"]
    man = json.load(open(ROOT / "cosmic" / "customdb" / "db_manifest.json"))
    txt = [
        "**Design.**",
        "- **Candidate set.** CSI:FingerID/COSMIC searched custom structure databases built from the molecules of our spectral library "
        "(`scripts/cosmic/make_customdb.py`), so both systems choose among the same molecules.",
        f"  - (a): {man['test_a']['n_molecules']:,} test-phase library molecules, including every query molecule. Our side uses the usual leave-one-spectrum-out library.",
        f"  - (b): the same database without the 300 query molecules ({man['test_b']['n_molecules']:,}). Our side removes all spectra of those molecules at once.",
        "  - Only molecules within 0.05 Da of a query's neutral mass are kept. CSI:FingerID retrieves formula-matching structures only, so this filter changes no retrieval.",
        "- **SIRIUS settings.** Expansive search (PubChem fallback) was switched off. Formulas, fingerprints and classes were computed once and shared by (a) and (b).",
        f"- **Checks.** SIRIUS reported {', '.join(f'{k}: {v:,}' for k, v in ch['db_structures_reported_by_sirius'].items() if v)} structures "
        f"(expected {', '.join(f'{k}: {v:,}' for k, v in ch['db_structures_expected'].items())}). "
        f"{ch['hits_outside_searched_db']} returned structures lay outside the searched database, and {ch['b_rows_correct_by_cosmic']} (b) rows were scored correct by COSMIC "
        "(both must be 0 by construction).",
        "- **Task.** Each system ranks and calibrates its own top-1: CSI:FingerID's structure-database hit vs our spectral-library hit. "
        "The candidate universe is now the same, but the evidence each system uses differs: predicted fingerprints vs reference spectra.",
        "",
    ]
    for st in [PRIM_S, LEAK_S]:
        v = c["strata"][st]
        txt.append(f"**{st}** ({v['n_queries']} queries; COSMIC without annotation: (a) {v['cosmic_unannotated']['a']}, (b) {v['cosmic_unannotated']['b']}; "
                   f"proposed without any library candidate: (a) {v['proposed_no_candidate']['a']}, (b) {v['proposed_no_candidate']['b']}), exact identity:")
        txt.append("")
        txt.append(table_cosmic_h2h(st))
        txt.append("")
    txt.append("In the (a)-only rows, calibrated COSMIC probabilities refer to the validation mixture (no prior correction is possible for a population that is "
               "100% in-library with a direct model), so their ECE and log loss are not comparable; e-AURC is. The proposed system's (a)-only probability is P(correct | in).")
    txt.append("")
    txt.append("Paired differences, proposed − COSMIC (val-calibrated + EM), exact identity; negative favours the proposed system:")
    txt.append("")
    txt.append(table_cosmic_h2h_diffs())
    txt.append("")
    txt.append(cosmic_ax_block())
    txt.append("")
    txt.append(cosmic_audit_block())
    txt.append("")
    txt.append("Approximate mode (supplement; label = correct or MCES distance ≤ 2, " + c["approx_label_method"] + "), primary stratum:")
    txt.append("")
    txt.append(table_cosmic_h2h(PRIM_S, "approx"))
    txt.append("")
    txt.append(cosmic_h2h_reading())
    return "\n".join(txt)


def cosmic_ax_block():
    c = J["eval_cosmic_customdb"]
    ax = c.get("a_x")
    if not ax:
        return ""
    rows = []
    for st, b in ax["strata"].items():
        for cond, key in [("(a) only", "a_only")] + [(f"(a)+(b), π = {pi}", f"mix_pi{pi}") for pi in c["mixture_pi"]]:
            e = b[key]
            names = ([("COSMIC, val-calibrated", "cosmic_calibrated"), ("Proposed, default (a)", "proposed_default_a"), ("Proposed, a_x", "proposed_a_x")]
                     if key == "a_only" else
                     [("COSMIC, val-calibrated + EM", "cosmic_calibrated_em"), ("Proposed, default (a), EM", "proposed_default_em"), ("Proposed, a_x, EM", "proposed_a_x_em")])
            for lab, k in names:
                x = e[k]
                rows.append({"stratum": f"{st} ({b['n_queries']})", "condition": cond, "system": lab, "n correct / wrong": _cnt(x) + _flag(x),
                             "top-1 acc": pct(x.get("accuracy")), "e-AURC": _m(x, "e_aurc", 4), "ECE": _m(x, "ece")})
    diffs = []
    for st, b in ax["strata"].items():
        if b["n_queries"] < cfg["evaluation"]["min_group_mol_test"]:
            continue  # too few queries; shown above with counts only
        for pi in c["mixture_pi"]:
            for lab, k in [("a_x − COSMIC", "a_x_minus_cosmic"), ("default (a) − COSMIC", "default_minus_cosmic")]:
                d = b[f"mix_pi{pi}"][k]
                diffs.append({"stratum": st, "π": pi, "comparison": lab, "Δ e-AURC": f"{d['e_aurc']['point']:+.4f} {ci(d['e_aurc']['ci95'], 4)}",
                              "Δ ECE": f"{d['ece']['point']:+.3f} {ci(d['ece']['ci95'], 3)}"})
    strat = ", ".join(f"{k}: {v}" for k, v in ax["n_eligible_by_stratum"].items())
    pa = ax["strata"].get("all eligible", {})
    text = [f"**a_x version** (added at the owner's request after the main import; same 300 queries, same SIRIUS results, no SIRIUS re-run). "
            f"On our side the library keeps the truth only through spectra measured on the **other** instrument type. The candidate molecules are unchanged, "
            f"so COSMIC's (a) database is still the matching candidate set. **{ax['n_eligible_queries']} of the 300 queries are a_x-eligible** ({strat}): "
            f"the truth must have at least one other-instrument spectrum in the precursor window. The 'default (a)' rows are the usual leave-one-spectrum-out "
            f"condition on the same eligible queries, and COSMIC is evaluated on the same queries. "
            f"The primary stratum has too few eligible queries to be interpreted; the pooled rows are dominated by the leaky stratum.", "",
            md(pd.DataFrame(rows)), "", "Paired differences (negative favours the proposed pipeline; strata with fewer than "
            f"{cfg['evaluation']['min_group_mol_test']} eligible queries are omitted):", "", md(pd.DataFrame(diffs)), ""]
    if pa:
        def sig(d):
            return "significant" if d["ci95"][1] < 0 or d["ci95"][0] > 0 else "not significant"
        r = ["Reading (a_x):"]
        for pi in c["mixture_pi"]:
            m = pa[f"mix_pi{pi}"]
            dx, dd = m["a_x_minus_cosmic"], m["default_minus_cosmic"]
            r.append(f"- All {pa['n_queries']} eligible queries, π = {pi}: e-AURC {_m(m['proposed_a_x_em'], 'e_aurc', 4)} (a_x) and "
                     f"{_m(m['proposed_default_em'], 'e_aurc', 4)} (default (a)) vs {_m(m['cosmic_calibrated_em'], 'e_aurc', 4)} (COSMIC). "
                     f"Paired Δe-AURC: a_x {dx['e_aurc']['point']:+.4f} {ci(dx['e_aurc']['ci95'], 4)} ({sig(dx['e_aurc'])}); "
                     f"default {dd['e_aurc']['point']:+.4f} {ci(dd['e_aurc']['ci95'], 4)} ({sig(dd['e_aurc'])}). "
                     f"ΔECE: a_x {dx['ece']['point']:+.3f} {ci(dx['ece']['ci95'], 3)} ({sig(dx['ece'])}).")
        r.append("- **Under a_x the pipeline-level advantage shrinks.** When the library holds the compound only as measured on another instrument type, "
                 "the proposed pipeline's ranking gets worse (higher e-AURC than under default (a) on the same queries). "
                 "Its advantage over CSI:FingerID + COSMIC is then smaller and, for ranking in the pooled set, not significant at the 95% level. "
                 "The eligible queries are mostly molecules inside CSI:FingerID's training set, which favours COSMIC. "
                 "The primary stratum has only " + str(ax["strata"].get(PRIM_S, {}).get("n_queries", 0)) + " eligible queries and is not interpreted.")
        text += r
    return "\n".join(text)


def cosmic_audit_block():
    f = RES / "cosmic_customdb_audit.json"
    if not f.exists():
        return ""
    a = J["cosmic_customdb_audit"]
    t = a["top1_vs_truth_tautomer_key"]
    v = a.get("validation_cosmic_top1_vs_truth_tautomer_key", {})
    sens = J["eval_cosmic_customdb"].get("sensitivity_val_tautomer")
    lines = ["**Audits** (`results/cosmic_customdb_audit.json`).",
             f"- **Database pairs.** Test (a) − (b) = exactly the {a['db_pair_test']['a_minus_b']} test query molecules. "
             f"Validation (a) − (b) = exactly the {a['db_pair_val']['a_minus_b']} validation query molecules; the validation subset has 200 molecules, hence the smaller difference.",
             "- **Tautomers.** Removal used our identity key (InChIKey first block after RDKit standardization), which is not tautomer-insensitive. "
             "The removal was therefore re-checked with the tautomer-insensitive key.",
             f"  - The test (b) database contains {a['test_b_molecules_tautomer_equal_to_a_query_molecule']['n']} tautomers of test query molecules. "
             f"No test top-1 of either system is a non-identical tautomer of the truth: (a) {t['a']['cosmic_top1_tautomer_of_truth_but_not_identical']} / "
             f"{t['a']['proposed_top1_tautomer_of_truth_but_not_identical']}, (b) {t['b']['cosmic_top1_tautomer_of_truth_but_not_identical']} / "
             f"{t['b']['proposed_top1_tautomer_of_truth_but_not_identical']} (COSMIC / proposed).",
             f"  - The validation (b) database contains {a['val_b_molecules_tautomer_equal_to_a_query_molecule']['n']} tautomers of validation query molecules. "
             f"For {v.get('b', {}).get('cosmic_top1_tautomer_of_truth_but_not_identical', 0)} validation (b) query, COSMIC's top-1 is such a tautomer and was labelled wrong "
             "in COSMIC's calibration data."]
    if sens:
        x = sens["primary_stratum"]["mix_pi0.2"]
        lines.append(f"  - Sensitivity: with the {len(sens['dropped_validation_queries'])} affected validation queries removed from COSMIC's calibration, the primary-stratum "
                     f"π = 0.2 comparison gives COSMIC ECE {x['cosmic_calibrated_em']['ece']:.3f}. The paired Δe-AURC is {x['proposed_minus_cosmic']['e_aurc']['point']:+.4f} "
                     f"{ci(x['proposed_minus_cosmic']['e_aurc']['ci95'], 4)} and ΔECE {x['proposed_minus_cosmic']['ece']['point']:+.3f} "
                     f"{ci(x['proposed_minus_cosmic']['ece']['ci95'], 3)}; "
                     + ("the conclusion is unchanged." if x['proposed_minus_cosmic']['e_aurc']['ci95'][1] < 0 and x['proposed_minus_cosmic']['ece']['ci95'][1] < 0
                        else "**the conclusion changes; see LOG.**"))
    return "\n".join(lines)


def cosmic_pubchem_section():
    f = RES / "eval_cosmic_subset.json"
    if not f.exists():
        return "**Pending.**"
    c = J["eval_cosmic_subset"]
    txt = ["This is the first COSMIC run, kept as background. CSI:FingerID searched **PubChem**, which contains almost every test molecule, so COSMIC has only a scenario-(a) version here. "
           "The proposed system is shown both on (a) alone and on (a)+(b) mixtures of the same 300 queries. In the mixtures both sides have enough wrong annotations for the metrics to be stable. "
           "The candidate sets differ (PubChem vs our library), so these rows describe each system on its own task. They are not a head-to-head comparison.", ""]
    for st in [PRIM_S, LEAK_S]:
        txt.append(f"**{st}**, exact identity (COSMIC without any annotation: {c['strata'][st]['exact']['n_cosmic_unannotated']}):")
        txt.append("")
        txt.append(table_cosmic_pubchem(st))
        txt.append("")
    txt.append(cosmic_pubchem_reading())
    return "\n".join(txt)


def cosmic_h2h_reading():
    c = J["eval_cosmic_customdb"]["strata"]
    pr, lk = c[PRIM_S], c[LEAK_S]
    pa, la = pr["exact"]["a_only"], lk["exact"]["a_only"]
    p2, p5 = pr["exact"]["mix_pi0.2"], pr["exact"]["mix_pi0.5"]
    l2, l5 = lk["exact"]["mix_pi0.2"], lk["exact"]["mix_pi0.5"]
    d2, d5 = p2["proposed_minus_cosmic"], p5["proposed_minus_cosmic"]
    ld2 = l2["proposed_minus_cosmic"]
    ap2, al2 = pr["approx"]["mix_pi0.2"], lk["approx"]["mix_pi0.2"]
    k = cfg["evaluation"]["min_group_mol_test"]
    return "\n".join([
        "Reading:",
        f"- **Same candidates, very different top-1 accuracy.** Restricted to our library's molecules, CSI:FingerID's top-1 is correct for "
        f"{pct(pa['cosmic_raw']['accuracy'])} of the primary-stratum queries in (a) (vs {pct(J['eval_cosmic_subset']['strata'][PRIM_S]['exact']['cosmic']['accuracy'])} "
        f"when it searched PubChem). The spectral-library top-1 is correct for {pct(pa['proposed']['accuracy'])}. The candidate universe is identical, "
        "but the evidence is not. A spectral library that contains other spectra of the compound gives far stronger evidence than a fingerprint predicted from the spectrum.",
        f"- **Scenario (a) alone.** COSMIC now has enough of both classes ({_cnt(pa['cosmic_calibrated'])}): e-AURC {_m(pa['cosmic_calibrated'], 'e_aurc', 4)}, "
        f"AUROC {_m(pa['cosmic_calibrated'], 'auroc')}. The proposed system has only {pa['proposed']['n_neg']} wrong top-1 annotations, below the pre-declared "
        f"minimum of {k}. Its (a)-only e-AURC, AUROC and ECE are therefore unstable and are not compared.",
        f"- **Main result: mixtures with enough negatives on both sides.** In the primary stratum at π = 0.2 (EM for both systems), the proposed system has "
        f"e-AURC {_m(p2['proposed_em'], 'e_aurc', 4)} and COSMIC {_m(p2['cosmic_calibrated_em'], 'e_aurc', 4)}. The paired difference is "
        f"{d2['e_aurc']['point']:+.4f} {ci(d2['e_aurc']['ci95'], 4)}. For calibration, ECE is {_m(p2['proposed_em'], 'ece')} vs "
        f"{_m(p2['cosmic_calibrated_em'], 'ece')} (difference {d2['ece']['point']:+.3f} {ci(d2['ece']['ci95'], 3)}), and log loss differs by "
        f"{d2['logloss']['point']:+.3f} {ci(d2['logloss']['ci95'], 3)}. At π = 0.5 the picture is the same: Δe-AURC {d5['e_aurc']['point']:+.4f} "
        f"{ci(d5['e_aurc']['ci95'], 4)}, ΔECE {d5['ece']['point']:+.3f} {ci(d5['ece']['ci95'], 3)}.",
        f"- **Trusted tier.** Queries with p ≥ 0.9: the proposed system placed {p2['proposed_em']['trusted']['n']} rows there at π = 0.2, all correct "
        f"({pct(p2['proposed_em']['trusted']['acc'])}; predicted {pct(p2['proposed_em']['trusted']['pred'])}). Calibrated COSMIC placed "
        f"{p2['cosmic_calibrated_em']['trusted']['n']} there.",
        f"- **Leaky stratum (reference only).** Here CSI:FingerID has seen the structures in training. COSMIC ranks its (a) annotations almost perfectly "
        f"(e-AURC {_m(la['cosmic_calibrated'], 'e_aurc', 4)}, {_cnt(la['cosmic_calibrated'])}). In the mixtures the proposed system is still ahead "
        f"at π = 0.2 (Δe-AURC {ld2['e_aurc']['point']:+.4f} {ci(ld2['e_aurc']['ci95'], 4)}). "
        f"COSMIC's trusted tier, however, is over-confident: {pct(l2['cosmic_calibrated_em']['trusted']['acc'])} observed vs "
        f"{pct(l2['cosmic_calibrated_em']['trusted']['pred'])} predicted ({l2['cosmic_calibrated_em']['trusted']['n']} rows).",
        f"- **Approximate mode (supplement).** Correct means MCES distance ≤ 2. In the primary stratum at π = 0.2, e-AURC is "
        f"{_m(ap2['proposed_em'], 'e_aurc', 4)} (proposed) vs {_m(ap2['cosmic_calibrated_em'], 'e_aurc', 4)} (COSMIC). "
        f"**In the leaky stratum, COSMIC's approximate-mode e-AURC is lower than the proposed system's** "
        f"({_m(al2['cosmic_calibrated_em'], 'e_aurc', 4)} vs {_m(al2['proposed_em'], 'e_aurc', 4)} at π = 0.2; paired Δ "
        f"{al2['proposed_minus_cosmic']['e_aurc']['point']:+.4f} {ci(al2['proposed_minus_cosmic']['e_aurc']['ci95'], 4)}, not significant). "
        "The proposed system was never trained for near-miss correctness. This is reported as observed.",
        "- **Conclusion (pipeline level).** On the same candidate set, the proposed pipeline (spectral-library search + confidence model) gives better confidence ranking "
        "and calibration than CSI:FingerID + COSMIC, for molecules outside CSI:FingerID's published training set and with enough negatives on both sides. "
        "**Because the two pipelines use different annotation engines, this result cannot be attributed to the confidence estimation alone.** "
        "Part or all of the advantage may come from the library-search engine, which has reference spectra of the compounds. "
        "The comparison does not show that the proposed confidence model would beat COSMIC if both scored the same annotations. "
        "Sample size is 150 molecules per stratum, and the intervals are wide.",
    ])


def cosmic_summary():
    if not cosmic_h2h_available():
        return "Pending (§6.11)."
    c = J["eval_cosmic_customdb"]["strata"][PRIM_S]["exact"]
    d = c["mix_pi0.2"]["proposed_minus_cosmic"]
    return (f"On the same candidate set, the proposed pipeline (spectral-library search + confidence model) gives better confidence ranking and calibration "
            f"than CSI:FingerID + COSMIC. Setting: SIRIUS restricted to our library's molecules; 150 molecules outside CSI:FingerID's published training set; "
            f"(a) and (b) mixed at π = 0.2. e-AURC is {c['mix_pi0.2']['proposed_em']['e_aurc']:.4f} vs {c['mix_pi0.2']['cosmic_calibrated_em']['e_aurc']:.4f} "
            f"(Δ {d['e_aurc']['point']:+.4f} {ci(d['e_aurc']['ci95'], 4)}). ECE is {c['mix_pi0.2']['proposed_em']['ece']:.3f} vs "
            f"{c['mix_pi0.2']['cosmic_calibrated_em']['ece']:.3f}. The engines differ, so this cannot be attributed to the confidence estimation alone. "
            f"In scenario (a) alone the proposed system makes only {c['a_only']['proposed']['n_neg']} errors, so its (a)-only metrics are marked unstable "
            f"and not compared (§6.11)." + cosmic_ax_summary())


def cosmic_ax_summary():
    ax = J["eval_cosmic_customdb"].get("a_x")
    if not ax or "all eligible" not in ax["strata"]:
        return ""
    d = ax["strata"]["all eligible"]["mix_pi0.2"]["a_x_minus_cosmic"]["e_aurc"]
    s_ = "not significant" if d["ci95"][0] <= 0 <= d["ci95"][1] else "significant"
    return (f" When our library holds each compound only as measured on another instrument type (a_x; {ax['n_eligible_queries']} eligible queries, "
            f"mostly inside CSI:FingerID's training set), the ranking advantage shrinks: Δe-AURC {d['point']:+.4f} {ci(d['ci95'], 4)}, {s_}.")


def repro_summary():
    f = RES / "repro_check.json"
    if not f.exists():
        return ""
    r = json.load(open(f))
    files = r["files"]
    both = {k: v for k, v in files.items() if v["status"] != "only in this run"}
    ident = [k for k, v in both.items() if v["status"] == "identical"]
    diff = {k: v for k, v in both.items() if v["status"] != "identical"}
    n_num = sum(v.get("n_numbers", 0) for v in both.values())
    n_bad = sum(v.get("n_diff", 0) for v in both.values())
    txt = (f"**Independent re-run.** The whole pipeline was re-run from scratch from the frozen commit `{r.get('commit', '')[:10]}` in a fresh directory "
           f"(`make all`, {r.get('when', '')}). Of {len(both)} result files produced by both runs, {len(ident)} are identical in every number. "
           f"Over all files, {n_num - n_bad:,} of {n_num:,} numbers are equal.")
    if diff:
        txt += " Differences: " + "; ".join(f"`{k}` ({v['n_diff']} values; {v.get('note', 'see results/repro_check.json')})" for k, v in diff.items()) + "."
    return txt


def em_reading():
    if not (RES / "eval_baselines_em.json").exists():
        return "Pending."
    e = J["eval_baselines_em"]
    sw_ = e["sweep"]
    w = e["wetlab"]
    direct = [n for n in EM_SYSTEMS if n in sw_ and sw_[n]["direct"]]
    improved = sum(sw_[n]["0.2"]["em"]["logloss"] < sw_[n]["0.2"]["none"]["logloss"] for n in direct)
    score = [n for n in direct if n not in ("M_full_direct",)]
    best = min(score, key=lambda n: sw_[n]["0.2"]["em"]["logloss"])
    def ts(key, pi):
        v = w[pi].get(key, {}).get("trusted", {})
        return f"{v['mean_tier_size_in_batch']:.0f}" if v.get("n_repeats_nonempty") else "0"
    return "\n".join([
        f"- The EM correction lowers the log loss of {improved} of {len(direct)} direct baselines at π = 0.2 and brings their overall ECE close to that of the proposed system. Ranking is unchanged by construction.",
        f"- **The score-based baselines stay far behind in log loss.** The best of them after EM is {NAMES.get(best, best)}: {sw_[best]['0.2']['em']['logloss']:.3f} vs "
        f"{sw_[P]['0.2']['em']['logloss']:.3f} at π = 0.2. Once calibrated at low π, their trusted tier nearly disappears. In the simulated batches at π = 0.1, "
        f"logistic(s1, gap) + EM puts {ts('B_lr_direct|em', '0.1')} query(ies) per batch in the trusted tier, against {ts(P + '|em', '0.1')} for the proposed system. "
        f"The corresponding numbers at π = 0.2 are {ts('B_lr_direct|em', '0.2')} and {ts(P + '|em', '0.2')}. So calibration alone does not make them useful: their ranking is too weak.",
        f"- **The direct full-feature model with EM is as good as the proposed system.** Log loss {sw_['M_full_direct']['0.2']['em']['logloss']:.4f} vs "
        f"{sw_[P]['0.2']['em']['logloss']:.4f}, ECE {sw_['M_full_direct']['0.2']['em']['ece']:.4f} vs {sw_[P]['0.2']['em']['ece']:.4f}, and a trusted tier of "
        f"{ts('M_full_direct|em', '0.1')} vs {ts(P + '|em', '0.1')} queries per batch at π = 0.1. "
        "It passes the wet-lab simulation and the group calibration equally well (§6.2). On this benchmark the decomposition is therefore not needed for calibration under prevalence shift.",
        shifted_line(),
    ])


def shifted_line():
    sh = J["eval_baselines_em"]["shifted"]
    def tr(tag, n):
        t = sh[tag][n]["0.2"]["em"]["tiers"]["trusted"]
        return f"{pct(t['acc'])} observed vs {pct(t['pred_acc'])} predicted"
    return ("- **Under the shifts, the proposed system and the direct full-feature model fail alike.** With EM, their trusted tiers are over-confident: "
            f"under a_x, proposed {tr('a_x', P)} and direct {tr('a_x', 'M_full_direct')}; on the external source, proposed {tr('external_v3', P)} and "
            f"direct {tr('external_v3', 'M_full_direct')}. On the external source, the EM-corrected logistic(s1, gap) trusted tier is less over-confident "
            f"({tr('external_v3', 'B_lr_direct')}) but still fails, consistent with its better ranking there (§6.7).")


def lead_reading():
    if not (RES / "eval_lead_attribution.json").exists():
        return "Pending."
    out = []
    for pi in ["0.2", "0.5"]:
        b = J["eval_lead_attribution"]["boot"][pi]
        top = max(b["path"], key=lambda x: x["share_of_lead_aurc"])
        crowd = b["path"][-1]
        out.append(f"- π = {pi}: the largest share of the AURC lead comes from '{STEP_NAMES.get(top['step'], top['step'])}' ({pct(top['share_of_lead_aurc'], 0)}). "
                   f"The last step, adding query-level crowding, contributes {pct(crowd['share_of_lead_aurc'], 0)} of the AURC lead "
                   f"(step ΔAURC {crowd['d_aurc']:+.4f} {ci(crowd['d_aurc_ci'], 4)}) and {pct(crowd['share_of_lead_logloss'], 0)} of the log-loss lead.")
    return "Reading (post hoc):\n" + "\n".join(out)


def cosmic_pubchem_reading():
    c = J["eval_cosmic_subset"]["strata"]
    pe, le = c[PRIM_S]["exact"], c[LEAK_S]["exact"]
    k = cfg["evaluation"]["min_group_mol_test"]
    return "\n".join([
        "Reading:",
        f"- **Primary stratum, (a) alone: too few negatives or positives to rank-compare.** COSMIC has {pe['cosmic']['n_pos']} correct top-1 annotations "
        f"out of 150 (accuracy {pct(pe['cosmic']['accuracy'])}). The proposed system has only {pe['proposed_a_only']['n_neg']} wrong ones. "
        f"Both fall below the pre-declared minimum of {k} per class. Their AUROC, e-AURC and ECE are therefore marked unstable and are not interpreted. "
        f"The earlier AUROC difference (proposed − COSMIC {pe['a_only_proposed_minus_cosmic_auroc']['point']:+.3f}, "
        f"CI {ci(pe['a_only_proposed_minus_cosmic_auroc']['ci95'], 3)}) is **not** a conclusion of this report.",
        f"- **Proposed system with enough negatives.** Mixing (a) and (b) of the same 150 molecules gives {pe['proposed_mix_pi0.2']['n_pos']} correct and "
        f"{pe['proposed_mix_pi0.2']['n_neg']} wrong rows. At π = 0.2 (EM): e-AURC {_m(pe['proposed_mix_pi0.2'], 'e_aurc', 4)}, ECE {_m(pe['proposed_mix_pi0.2'], 'ece')}. "
        f"Its trusted tier held {pe['proposed_mix_pi0.2']['trusted']['n']} rows with observed accuracy {pct(pe['proposed_mix_pi0.2']['trusted']['acc'])} "
        f"(predicted {pct(pe['proposed_mix_pi0.2']['trusted']['pred'])}).",
        f"- **Leakage is visible in COSMIC's own numbers.** CSI:FingerID's exact top-1 accuracy is {pct(le['cosmic']['accuracy'])} for molecules in its published "
        f"training structures, vs {pct(pe['cosmic']['accuracy'])} for the others. In the leaky stratum COSMIC has enough of both classes "
        f"({_cnt(le['cosmic'])}): AUROC {_m(le['cosmic'], 'auroc')}, e-AURC {_m(le['cosmic'], 'e_aurc', 4)}, ECE {_m(le['cosmic'], 'ece')} raw and "
        f"{_m(le['cosmic_calibrated'], 'ece')} after validation calibration. This stratum is a reference only.",
        "- **What this supplement can and cannot show.** The two systems search different candidate sets: PubChem, which contains nearly every test molecule, versus our library. "
        "The rows therefore describe each system on its own task. The like-for-like comparison is §6.11.1.",
    ])


def cosmic_status():
    return ("Results: §6.11. The main comparison restricts SIRIUS to our library's molecules (same candidate set, (a) and (b)); "
            "a first run against PubChem is kept as background.")


def mid_gaps():
    f = J["dev_calibrators"]["I"]["isotonic"]["folds"]
    return " and ".join(f"{x['mid_gap_obs_minus_pred']:+.2f}" for x in f)


def mid_gap_max():
    return max(abs(x["mid_gap_obs_minus_pred"]) for x in J["dev_calibrators"]["I"]["isotonic"]["folds"])


def dev(v, comp, key="valA_logloss"):
    return J["dev_feature_variants"][v][comp][key]


# ----------------------------------------------------------------------------- added 2026-09-27
def table_wetlab_ax(pi):
    w = J["eval_wetlab_ax"]["wetlab"][str(pi)]
    rows = []
    for t in ["trusted", "suspicious", "absent"]:
        v = w.get(t, {})
        if not v or not v.get("n_repeats_nonempty"):
            rows.append({"Tier": t, "Note": "empty"})
            continue
        rows.append({"Tier": t, "Mean tier size / batch": f"{v['mean_tier_size_in_batch']:.0f}",
                     "Obs. acc (mean, 2.5–97.5%)": f"{pct(v['obs_acc_mean'])} ({pct(v['obs_acc_q025'])}–{pct(v['obs_acc_q975'])})",
                     "Pred. acc": pct(v["pred_acc_mean"]), "P(pred ∈ 95% CI)": pct(v["acc_consistent_rate"]),
                     "Obs. in-lib": pct(v["obs_in_mean"]), "Pred. in-lib": pct(v["pred_in_mean"])})
    return md(pd.DataFrame(rows))


def ax_wetlab_reading(thr=0.8):
    w = J["eval_wetlab_ax"]["wetlab"]
    bad = []
    for pi in ["0.1", "0.2", "0.3", "0.5"]:
        for t in ["suspicious", "absent"]:
            v = w.get(pi, {}).get(t, {})
            if v.get("n_repeats_nonempty") and v["acc_consistent_rate"] < thr:
                bad.append(f"{t} at π = {pi} (predicted {pct(v['pred_acc_mean'])}, observed {pct(v['obs_acc_mean'])}, "
                           f"consistent in {pct(v['acc_consistent_rate'])} of batches)")
    if not bad:
        return "The other tiers stay consistent (≥ 80% of batches) at every π."
    return "Other tiers below 80% consistency under a_x: " + "; ".join(bad) + "."


LEAD_NAMES = {"L_minus_meta": "− meta", "L_minus_sim": "− sim", "L_minus_sim_extra": "− sim_extra", "L_minus_analog": "− analog",
              "L_minus_crowd_spec": "− crowd_spec", "L_minus_crowd_struct": "− crowd_struct", "M_base": "− crowd_spec − crowd_struct (= no crowding)"}


def lead(pi, key):
    return J["eval_lead_attribution"]["boot"][str(pi)][key]


def lstep(pi, i, key):
    """Step i of the lead path (0 sim model, 1 + sim_extra, 2 + analog, 3 + meta, 4 + crowding)."""
    return J["eval_lead_attribution"]["boot"][str(pi)]["path"][i][key]


STEP_NAMES = {"+ LightGBM on sim (s1, s2, s3, gaps)": "+ non-linear score model (LightGBM on s1, s2, s3, gaps)",
              "+ sim_extra": "+ top-1 library support (sim_extra: matched peaks, mean score over the top-1's spectra, # its spectra in window)",
              "+ analog": "+ analogue hits (analog)", "+ meta (= no-crowding model)": "+ metadata incl. window counts (= no-crowding model)",
              "+ crowd_spec + crowd_struct (= proposed)": "+ query-level crowding (= proposed)"}


def table_lead_path(pi):
    if not (RES / "eval_lead_attribution.json").exists():
        return "Pending."
    b = J["eval_lead_attribution"]["boot"][str(pi)]
    rows = []
    for s in b["path"]:
        rows.append({"step": STEP_NAMES.get(s["step"], s["step"]), "ΔAURC (step)": f"{s['d_aurc']:+.4f}", "95% CI": ci(s["d_aurc_ci"], 4),
                     "share of AURC lead": pct(s["share_of_lead_aurc"], 0), "Δ log loss (step)": f"{s['d_logloss']:+.4f}",
                     "95% CI ": ci(s["d_logloss_ci"], 4), "share of log-loss lead": pct(s["share_of_lead_logloss"], 0)})
    rows.append({"step": "**total lead** (proposed − logistic(s1, gap), decomposed)", "ΔAURC (step)": f"{-b['lead_aurc']:+.4f}",
                 "95% CI": ci(b["diffs"][f"{P} - B_lr_dec"]["d_aurc_ci"], 4), "share of AURC lead": "100%",
                 "Δ log loss (step)": f"{-b['lead_logloss']:+.4f}", "95% CI ": ci(b["diffs"][f"{P} - B_lr_dec"]["d_logloss_ci"], 4),
                 "share of log-loss lead": "100%"})
    return md(pd.DataFrame(rows))


def table_lead_dropone(pi):
    if not (RES / "eval_lead_attribution.json").exists():
        return "Pending."
    b = J["eval_lead_attribution"]["boot"][str(pi)]["diffs"]
    rows = []
    for n, lab in LEAD_NAMES.items():
        v = b.get(f"{P} - {n}")
        if not v:
            continue
        rows.append({"proposed system …": lab, "AURC cost of removing (Δ = removed − proposed)": f"{-v['d_aurc']:+.4f}",
                     "95% CI": ci([-v["d_aurc_ci"][1], -v["d_aurc_ci"][0]], 4),
                     "log-loss cost": f"{-v['d_logloss']:+.4f}", "95% CI ": ci([-v["d_logloss_ci"][1], -v["d_logloss_ci"][0]], 4)})
    return md(pd.DataFrame(rows))


EM_SYSTEMS = [P, "B_lr_dec", "B_s1_dec", "B_gap_dec", "B_ms2q_dec", "B_lr_direct", "B_s1_direct", "B_gap_direct", "M_full_direct",
              "B_ms2q_conf_direct", "B_ms2q_rerank_direct", "B_tda_margin_direct", "F_sim", "F_meta"]


def table_baselines_em(pi):
    sw_ = J["eval_baselines_em"]["sweep"]
    rows = []
    for s in EM_SYSTEMS:
        if s not in sw_:
            continue
        e, n = sw_[s][str(pi)]["em"], sw_[s][str(pi)]["none"]
        tr = e["tiers"].get("trusted", {})
        rows.append({"System": NAMES.get(s, s), "type": "direct" if sw_[s]["direct"] else "decomposed",
                     "e-AURC ↓": f"{e['e_aurc']:.4f}", "Log loss: none → EM": f"{n['logloss']:.3f} → {e['logloss']:.3f}",
                     "ECE: none → EM": f"{n['ece']:.3f} → {e['ece']:.3f}", "Mean pred (EM) / obs": f"{e['mean_p']:.3f} / {e['mean_y']:.3f}",
                     "Trusted (EM): share, obs / pred": (f"{pct(tr.get('frac'))}, {pct(tr.get('acc'))} / {pct(tr.get('pred_acc'))}" if tr.get("acc") is not None else "empty")})
    return md(pd.DataFrame(rows))


def table_wetlab_em(pi):
    w = J["eval_baselines_em"]["wetlab"][str(pi)]
    rows = []
    for key, v in w.items():
        s, mode = key.split("|")
        for t in ["trusted", "suspicious", "absent"]:
            x = v.get(t, {})
            if not x or not x.get("n_repeats_nonempty"):
                continue
            rows.append({"System": NAMES.get(s, s), "correction": "EM" if mode == "em" else "none", "Tier": t,
                         "size / batch": f"{x['mean_tier_size_in_batch']:.0f}", "Pred. acc": pct(x["pred_acc_mean"]),
                         "Obs. acc (mean)": pct(x["obs_acc_mean"]), "P(pred ∈ 95% CI)": pct(x["acc_consistent_rate"])})
    return md(pd.DataFrame(rows))


def table_baselines_em_shifted(pi):
    sh = J["eval_baselines_em"]["shifted"]
    rows = []
    for tag, lab in [("a_x", "library on other instrument (a_x)"), ("external_v3", "external source (MassBank v3)")]:
        for s, v in sh.get(tag, {}).items():
            if not isinstance(v, dict) or str(pi) not in v:
                continue
            e, n = v[str(pi)]["em"], v[str(pi)]["none"]
            tr = e["tiers"].get("trusted", {})
            rows.append({"Shift": lab, "System": NAMES.get(s, s), "e-AURC": f"{e['e_aurc']:.4f}",
                         "Log loss: none → EM": f"{n['logloss']:.3f} → {e['logloss']:.3f}", "ECE: none → EM": f"{n['ece']:.3f} → {e['ece']:.3f}",
                         "Trusted (EM) obs / pred": (f"{pct(tr.get('acc'))} / {pct(tr.get('pred_acc'))}" if tr.get("acc") is not None else "empty")})
    return md(pd.DataFrame(rows))


def table_groups_em(pi):
    g = J["eval_baselines_em"]["groups_em"]
    rows = []
    for s_, v in g.items():
        for gc, x in v[str(pi)].items():
            rows.append({"System (EM for all)": NAMES.get(s_, s_), "Grouping": gc, "Groups (sufficient/all)": f"{x['n_sufficient']}/{x['n_groups']}",
                         "Pred ∈ 95% CI": pct(x["frac_consistent_sufficient"]), "Mean |obs−pred|": f3(x["mean_abs_gap_sufficient"]),
                         "Max |obs−pred|": f3(x["max_abs_gap_sufficient"])})
    return md(pd.DataFrame(rows))


def gem(s, pi, g, key):
    return J["eval_baselines_em"]["groups_em"][s][str(pi)][g][key]


def gem_max_crowd(s, pi):
    g = J["eval_baselines_em"]["groups_em"][s][str(pi)]
    return max(v["max_abs_gap_sufficient"] for k, v in g.items() if k.startswith("bin_") and v.get("n_sufficient"))


def gem_frac_crowd(s, pi):
    g = J["eval_baselines_em"]["groups_em"][s][str(pi)]
    v = [x["frac_consistent_sufficient"] for k, x in g.items() if k.startswith("bin_") and x.get("n_sufficient")]
    return float(np.mean(v)) if v else float("nan")


def bem(s, pi, key, mode="em"):
    return J["eval_baselines_em"]["sweep"][s][str(pi)][mode][key]


# ---- COSMIC
def _cnt(x):
    return f"{x['n_pos']} / {x['n_neg']}"


def _flag(x):
    return " ⚠ unstable" if x.get("unstable") else ""


def _m(x, k, d=3):
    if x is None or k not in x or x[k] is None or (isinstance(x[k], float) and np.isnan(x[k])):
        return "–"
    s = f"{x[k]:.{d}f}"
    if f"{k}_ci95" in x:
        s += f" {ci(x[k + '_ci95'], d)}"
    return s


def _cosmic_rows(block, entries):
    rows = []
    for cond, sysname, x in entries:
        if not x:
            continue
        tr = x.get("trusted") or {}
        rows.append({"condition": cond, "system": sysname, "n correct / wrong": _cnt(x) + _flag(x),
                     "top-1 acc": pct(x.get("accuracy")), "e-AURC": _m(x, "e_aurc", 4), "ECE": _m(x, "ece"),
                     "log loss": _m(x, "logloss"),
                     "trusted (p ≥ 0.9): share, obs / pred": (f"{pct(tr.get('frac'))}, {pct(tr.get('acc'))} / {pct(tr.get('pred'))}"
                                                           if tr.get("acc") is not None else ("none" if tr else "–"))})
    return rows


def cosmic_h2h_available():
    return (RES / "eval_cosmic_customdb.json").exists()


def table_cosmic_h2h(stratum, mode="exact"):
    b = J["eval_cosmic_customdb"]["strata"][stratum][mode]
    ent = [("(a) only", "COSMIC, val-calibrated", b["a_only"]["cosmic_calibrated"]), ("(a) only", "COSMIC, raw score", b["a_only"]["cosmic_raw"]),
           ("(a) only", "Proposed, P(correct | in)", b["a_only"]["proposed"])]
    for pi in J["eval_cosmic_customdb"]["mixture_pi"]:
        m = b[f"mix_pi{pi}"]
        ent += [(f"(a)+(b), π = {pi}", "COSMIC, val-calibrated + EM", m["cosmic_calibrated_em"]),
                (f"(a)+(b), π = {pi}", "COSMIC, val-calibrated, no prior correction", m["cosmic_calibrated_none"]),
                (f"(a)+(b), π = {pi}", "COSMIC, raw score", m["cosmic_raw"]), (f"(a)+(b), π = {pi}", "Proposed, EM", m["proposed_em"])]
    return md(pd.DataFrame(_cosmic_rows(b, ent)))


def table_cosmic_h2h_diffs(mode="exact"):
    rows = []
    for stratum, v in J["eval_cosmic_customdb"]["strata"].items():
        for pi in J["eval_cosmic_customdb"]["mixture_pi"]:
            d = v[mode][f"mix_pi{pi}"].get("proposed_minus_cosmic")
            if not d:
                continue
            rows.append({"stratum": stratum, "π": pi, **{f"Δ {k}": f"{d[k]['point']:+.4f} {ci(d[k]['ci95'], 4)}" for k in ["e_aurc", "aurc", "ece", "logloss"]}})
    return md(pd.DataFrame(rows))


def h2h(stratum, key, sysk, metric, mode="exact"):
    return J["eval_cosmic_customdb"]["strata"][stratum][mode][key][sysk][metric]


def table_cosmic_pubchem(stratum, mode="exact"):
    b = J["eval_cosmic_subset"]["strata"][stratum][mode]
    ent = [("COSMIC: PubChem search", "COSMIC, raw score", b["cosmic"]), ("COSMIC: PubChem search", "COSMIC, val-calibrated", b.get("cosmic_calibrated")),
           ("Proposed: (a) only", "Proposed, P(correct | in)", b["proposed_a_only"])]
    for pi in J["eval_cosmic_subset"]["mixture_pi"]:
        ent.append((f"Proposed: (a)+(b), π = {pi}", "Proposed, EM", b[f"proposed_mix_pi{pi}"]))
    return md(pd.DataFrame(_cosmic_rows(b, ent)))


def cps(stratum, sysk, metric, mode="exact"):
    return J["eval_cosmic_subset"]["strata"][stratum][mode][sysk][metric]


PRIM_S, LEAK_S = "not_in_csi_training (primary)", "in_csi_training (leaky reference)"


# ----------------------------------------------------------------------------- render
NS = {k: v for k, v in globals().items() if not k.startswith("_")}
PAT = re.compile(r"\{\{\s*(.+?)\s*(?::\s*([^}]+?))?\s*\}\}")


def render(tmpl: str) -> str:
    def sub(m):
        val = eval(m.group(1), NS)
        fmt = m.group(2)
        return format(val, fmt) if fmt else str(val)
    return PAT.sub(sub, tmpl)


if __name__ == "__main__":
    prov = git_commit()
    for name in ["REPORT.md", "WETLAB_PREREG.md"]:
        t = ROOT / "reports" / "templates" / (name + ".tmpl")
        if not t.exists():
            print("no template", t)
            continue
        out = render(t.read_text())
        out += f"\n\n---\n<sub>Rendered by scripts/14_render_report.py from results/*.json; config {cfg['_config_hash']}, commit {prov['commit'][:10]}{' (dirty)' if prov['dirty'] else ''}.</sub>\n"
        (ROOT / name).write_text(out)
        print("wrote", name)
