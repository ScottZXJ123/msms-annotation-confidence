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
NAMES = {P: "**Proposed**", "M_full": "Full features (decomposed)", "M_full_gc": "Full + family offsets",
         "M_base": "No crowding (base)", "M_sim_plus_crowd": "Similarity + crowding", "M_full_minus_family": "Full − family crowding",
         "M_full_minus_spec": "Full − spectral crowding", "M_full_minus_struct": "Full − structural crowding",
         "M_candidate_only": "Candidate-list-only (engine-agnostic)", "M_full_direct": "Full features, direct P(correct)",
         "M_full_lr": "Full features, logistic", "B_s1_direct": "Score threshold s1", "B_gap_direct": "Gap s1−s2",
         "B_lr_direct": "Logistic(s1, gap)", "B_s1_dec": "s1, decomposed", "B_gap_dec": "gap, decomposed",
         "B_lr_dec": "Logistic(s1, gap), decomposed", "F_meta": "Floor: metadata only", "F_sim": "Floor: similarity only"}


def f3(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def ci(v, d=3):
    return f"[{v[0]:.{d}f}, {v[1]:.{d}f}]"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
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
        rows.append({"System": NAMES.get(s, s), "AURC ↓": f3(r["aurc"]), "e-AURC ↓": f3(r["e_aurc"]),
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
        for name in ["M_full", "M_base", "B_lr_dec"]:
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
                            for k, v in c.items()]))


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


def dev(v, comp, key="valA_logloss"):
    return J["dev_feature_variants"][v][comp][key]


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
