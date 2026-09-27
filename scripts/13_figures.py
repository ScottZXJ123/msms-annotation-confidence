"""All figures, generated from results/*.json and the frozen models (no hand-entered numbers)."""
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path  # noqa: E402
from msconf.evalutil import class_weights, components, probs  # noqa: E402
from msconf.evaluation import rc_curve  # noqa: E402
from msconf.pipeline import load_rows  # noqa: E402

cfg = load_config()
der, res, figd, tab = path(cfg, "derived"), path(cfg, "results"), path(cfg, "figures"), path(cfg, "tables")
J = lambda n: json.load(open(res / f"{n}.json"))  # noqa: E731
PRIMARY = J("selection")["primary_system"]
# reference categorical palette (validated default instance; light mode for print figures)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight"})
LABEL = {PRIMARY: "Proposed (full, decomposed + label shift)", "B_s1_direct": "Score threshold (s1)",
         "B_gap_direct": "Top1-top2 gap", "B_lr_direct": "Logistic(s1, gap)", "B_lr_dec": "Logistic(s1, gap), decomposed",
         "M_base": "No-crowding model (base)", "F_meta": "Metadata-only floor", "F_sim": "Similarity-only floor",
         "M_full_direct": "Full features, direct (no absence model)", "M_full": "Full (decomposed)"}

# ---------------------------------------------------------------- 1. risk-coverage at headline prevalences
rows = load_rows(der, "entropy")
test = rows[(rows.fold == "test") & rows.scenario.isin(["a", "b"])].reset_index(drop=True)
S = {f.stem: pickle.loads(f.read_bytes()) for f in (der / "models" / "entropy").glob("*.pkl")}
show = [PRIMARY, "M_base", "B_lr_dec", "B_lr_direct", "B_gap_direct", "B_s1_direct", "F_meta"]
show = [s for s in dict.fromkeys(show) if s in S]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=False)
for ax, pi in zip(axes, cfg["evaluation"]["prevalence_headline"]):
    w = class_weights(test.in_lib.values.astype(float), test.w.values, pi)
    y = test.correct.values.astype(float)
    for k, s in enumerate(show):
        pc, _ = probs(components(S[s], test), pi)
        # weighted RC curve: expand by resampling proportional to weights (deterministic grid)
        o = np.argsort(-pc, kind="stable")
        cw = np.cumsum(w[o])
        cov = cw / cw[-1]
        risk = np.cumsum(w[o] * (1 - y[o])) / cw
        ax.plot(cov, risk, color=C[k % len(C)], lw=2 if s == PRIMARY else 1.3, label=LABEL.get(s, s))
    ax.set_title(f"In-library prevalence π = {pi}", fontsize=9, color=INK)
    ax.set_xlabel("Coverage (fraction of queries accepted)")
    ax.set_ylabel("Selective risk (top-1 wrong)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
axes[1].legend(fontsize=7, loc="upper left")
fig.savefig(figd / "fig1_risk_coverage.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. calibration under prevalence shift
sw = J("eval_sweep")["sweep"]
grid = cfg["evaluation"]["prevalence_grid"]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(grid, [sw[PRIMARY][str(p)]["oracle"]["cal_mean_y"] for p in grid], color=INK, lw=1, ls="--", label="Observed accuracy")
series = [(PRIMARY, "em", "Proposed, EM-estimated π"), (PRIMARY, "oracle", "Proposed, true π"),
          (PRIMARY, "none", "Proposed, no correction"), ("M_full_direct", "none", "Direct model (no absence model)"),
          ("B_lr_direct", "none", "Logistic(s1, gap), direct")]
for k, (s, m, lab) in enumerate(series):
    if s in sw:
        ax.plot(grid, [sw[s][str(p)][m]["cal_mean_p"] for p in grid], color=C[k], marker="o", ms=4, lw=1.5, label=lab)
ax.set_xlabel("True in-library prevalence π of the batch")
ax.set_ylabel("Mean predicted P(top-1 correct)")
ax.legend(fontsize=7)
fig.savefig(figd / "fig2_prevalence_shift.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. wet-lab simulation
wl = J("eval_wetlab")["wetlab"]
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
tiers = ["trusted", "suspicious", "absent"]
pis = [p for p in ["0.1", "0.2", "0.3", "0.5", "0.7"] if p in wl]
for ax, t in zip(axes, tiers):
    key = f"{PRIMARY}|em"
    obs = [wl[p][key][t].get("obs_acc_mean", np.nan) for p in pis]
    lo = [wl[p][key][t].get("obs_acc_q025", np.nan) for p in pis]
    hi = [wl[p][key][t].get("obs_acc_q975", np.nan) for p in pis]
    pred = [wl[p][key][t].get("pred_acc_mean", np.nan) for p in pis]
    x = np.arange(len(pis))
    ax.errorbar(x, obs, yerr=[np.array(obs) - np.array(lo), np.array(hi) - np.array(obs)], fmt="o", color=C[0], ms=5,
                capsize=3, lw=1.3, label="Observed (40 per tier; 2.5–97.5% over repeats)")
    ax.plot(x, pred, "D", color=C[1], ms=5, label="Predicted mean P(correct)")
    ax.set_xticks(x, pis)
    ax.set_xlabel("Batch prevalence π")
    ax.set_title(t.capitalize(), fontsize=9)
axes[0].set_ylabel("Top-1 accuracy in tier")
axes[0].set_ylim(-0.02, 1.02)
axes[2].legend(fontsize=7, loc="center right")
fig.savefig(figd / "fig3_wetlab_simulation.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. per-family calibration (primary, pi=0.5)
f = tab / "groups_top1_class_pi0.5.csv"
if f.exists():
    g = pd.read_csv(f)
    g = g[g.sufficient].sort_values("pred")
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.22 * len(g) + 0.8)))
    y = np.arange(len(g))
    ax.errorbar(g.obs, y, xerr=[g.obs - g.lo, g.hi - g.obs], fmt="o", color=C[0], ms=4, capsize=2, lw=1, label="Observed (95% CI)")
    ax.plot(g.pred, y, "D", color=C[1], ms=4, label="Predicted mean")
    ax.set_yticks(y, [str(s)[:38] for s in g.group], fontsize=7)
    ax.set_xlabel("P(top-1 correct), π = 0.5")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(figd / "fig4_family_calibration.png")
    plt.close(fig)

# ---------------------------------------------------------------- 5. crowding vs error (scenario a)
inc = J("eval_incremental")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
ax = axes[0]
for k, c in enumerate(["top1_struct_nbrs", "fam_spec_crowd_train", "n_open_hi", "top1_specdens"]):
    q = pd.DataFrame(inc["query_level_error_by_crowding_bin"][c])
    ax.plot(q.iloc[:, 0], q.err, marker="o", ms=4, color=C[k], label=c)
ax.set_xlabel("Crowding quartile (edges fixed on validation)")
ax.set_ylabel("Top-1 error rate, truth in library")
ax.legend(fontsize=7)
ax = axes[1]
pr = inc["family_level_prior"].get("top1_class")
if pr:
    t = pd.DataFrame(pr["table"])
    ax.scatter(t.crowd, t.err, s=np.clip(t.n / 10, 8, 80), color=C[0], alpha=0.8, edgecolor="white", lw=0.8)
    ax.set_xlabel("Family spectral crowding (train library)")
    ax.set_ylabel("Family top-1 error rate (test, in library)")
    ax.set_title(f"Spearman ρ = {pr['spearman_rho']:.2f} (n = {pr['n_families']} families)", fontsize=9)
fig.savefig(figd / "fig5_crowding_vs_error.png")
plt.close(fig)

# ---------------------------------------------------------------- 6. reliability diagram (primary)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
for ax, pi in zip(axes, cfg["evaluation"]["prevalence_headline"]):
    w = class_weights(test.in_lib.values.astype(float), test.w.values, pi)
    pc, _ = probs(components(S[PRIMARY], test), pi)
    y = test.correct.values.astype(float)
    bins = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(pc, bins) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for k in range(10):
        m = b == k
        if w[m].sum() > 0:
            xs.append(np.average(pc[m], weights=w[m]))
            ys.append(np.average(y[m], weights=w[m]))
            ns.append(w[m].sum() / w.sum())
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.scatter(xs, ys, s=np.clip(np.array(ns) * 600, 10, 200), color=C[0], edgecolor="white", lw=0.8, zorder=3)
    ax.set_title(f"π = {pi}", fontsize=9)
    ax.set_xlabel("Predicted P(top-1 correct)")
    ax.set_ylabel("Observed accuracy")
fig.savefig(figd / "fig6_reliability.png")
plt.close(fig)
print("figures written to", figd)
