#!/usr/bin/env python3
"""
visualize_results.py

Effective visualizations for:
1) Main results (MB–GIB/MB–VIB vs baselines),
2) Ablations (scope, capacity, beta, likelihood, MB misspec),
3) Sensitivity (shift magnitude/type, support mismatch & missingness, N_s curves).

Inputs (optional, load if present in cwd):
  - main_results.csv
  - ablations.csv
  - sensitivity.csv

Outputs:
  - ./figs/*.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
# Improve PDF compatibility with some viewers:
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42


plt.rcParams.update({
    "figure.dpi": 140,
    "axes.grid": True,
    "grid.linestyle": ":",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.max_open_warning": 0,  # suppress warnings; we also close explicitly
})

OUTDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def save_and_close(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
def save_multi_and_close(fig, stem):
    """
    Save both PNG (raster) and PDF (vector; heatmaps rasterized inside).
    fig name stem -> figs/<stem>.png and figs/<stem>.pdf
    """
    png_path = os.path.join(OUTDIR, f"{stem}.png")
    pdf_path = os.path.join(OUTDIR, f"{stem}.pdf")
    fig.tight_layout()
    # High-DPI PNG fallback (always works)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    # Try PDF too
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
    except Exception as e:
        print(f"[warn] PDF save failed for {stem}: {e}")
    plt.close(fig)


METRICS = ["MAE", "RMSE", "R2"]

def agg_mean_se(df, group_cols, metric):
    g = df.groupby(group_cols)[metric]
    return g.mean().reset_index(name=f"{metric}_mean"), g.sem().reset_index(name=f"{metric}_se")


def bar_with_err(ax, xlabels, means, ses, title, ylabel):
    x = np.arange(len(xlabels))
    ax.bar(x, means, yerr=ses, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

def grouped_bar_err(ax, groups, series, mean_tbl, se_tbl, title, ylabel):
    """
    groups: x-axis categories (e.g., subsets)
    series: legend entries (e.g., methods)
    mean_tbl, se_tbl: DataFrames with cols [group, series, metric_mean/metric_se]
    """
    n_groups = len(groups); n_series = len(series)
    x = np.arange(n_groups)
    width = 0.8 / n_series
    for i, s in enumerate(series):
        m, e = [], []
        for g in groups:
            row = mean_tbl[(mean_tbl.iloc[:,0] == g) & (mean_tbl.iloc[:,1] == s)]
            if len(row) == 0:
                m.append(np.nan); e.append(0.0)
            else:
                m.append(float(row.iloc[0, 2]))
                e_row = se_tbl[(se_tbl.iloc[:,0]==g) & (se_tbl.iloc[:,1]==s)]
                e.append(float(e_row.iloc[0, 2]) if len(e_row) else 0.0)
        ax.bar(x + i*width, m, width, yerr=e, capsize=3, label=s)
    ax.set_xticks(x + width*(n_series-1)/2)
    ax.set_xticklabels(groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()

def pivot_matrix(df, row, col, val, agg="mean"):
    if agg == "mean":
        p = df.pivot_table(index=row, columns=col, values=val, aggfunc="mean")
    else:
        p = df.pivot_table(index=row, columns=col, values=val, aggfunc=agg)
    p = p.sort_index().sort_index(axis=1)
    return p

def imshow_matrix(ax, mat, row_labels, col_labels, title, cbar_label, rasterize_for_pdf=True):
    data = np.asarray(mat, dtype=float)
    mask = np.isnan(data)
    M = np.ma.array(data, mask=mask)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")  # NaNs shown as light gray
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap)
    if rasterize_for_pdf:
        # Ensures the image part is rasterized inside the PDF (fixes heavy/fragile PDFs)
        im.set_rasterized(True)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


# -----------------------------
# 1) Main results
# -----------------------------
if os.path.exists("main_results.csv"):
    df = pd.read_csv("main_results.csv")
    scenarios = sorted(df["scenario"].unique())

    for metric in METRICS:
        n = len(scenarios)
        fig, axes = plt.subplots(1, n, figsize=(4.6*n, 3.2), sharey=(metric != "R2"))
        if n == 1:
            axes = [axes]

        for i, scenario in enumerate(scenarios):
            d = df[df["scenario"] == scenario]
            m, s = agg_mean_se(d, ["method"], metric)

            # Emphasize MB-GIB / MB-VIB first (keep others in their original order)
            order = ["MB-GIB", "MB-VIB"] + [mm for mm in m["method"] if mm not in ["MB-GIB", "MB-VIB"]]
            m = m.set_index("method").reindex(order).dropna().reset_index()
            s = s.set_index("method").reindex(order).dropna().reset_index()

            bar_with_err(
                axes[i],
                m["method"].tolist(),
                m[f"{metric}_mean"].values,
                s[f"{metric}_se"].values,
                title=scenario,
                ylabel=metric
            )
            axes[i].tick_params(axis="x", rotation=20)

            # Clamp R^2 axis to [0.95, 1.0]
            if metric == "R2":
                axes[i].set_ylim(0.98, 1.0)

        #fig.suptitle(f"Main Results — {metric}")
        save_and_close(fig, os.path.join("figs", f"main_{metric.lower()}.pdf"))

    print("Saved: figs/main_*.pdf")
else:
    print("main_results.csv not found — skipping main results plots.")

# -----------------------------
# 2) Ablations
# -----------------------------
if os.path.exists("ablations.csv"):
    df = pd.read_csv("ablations.csv")

    # (A) Scope ablation
    d_scope = df[df.get("abl","") == "scope"].copy()
    if not d_scope.empty:
        for metric in METRICS:
            m, s = agg_mean_se(d_scope, ["subset", "method"], metric)
            subsets = ["parent", "mb", "global"]
            methods = sorted(d_scope["method"].unique(), key=lambda x: {"MB-GIB":0,"MB-VIB":1}.get(x, 2))
            fig, ax = plt.subplots(figsize=(8,3.2))
            grouped_bar_err(ax, subsets, methods, m, s,
                            title=f"Ablation: Scope — {metric}", ylabel=metric)
            save_and_close(fig, os.path.join(OUTDIR, f"abl_scope_{metric.lower()}.pdf"))
        print("Saved: figs/abl_scope_*.pdf")

    # (B) Capacity/β/likelihood for VIB (MB scope)
    d_cap = df[df.get("abl","") == "capacity_beta_like"].copy()
    if not d_cap.empty:
        for like in sorted(d_cap["like"].unique()):
            d_like = d_cap[d_cap["like"] == like]
            for metric in ["RMSE"]:  # heatmaps on RMSE are most interpretable
                pivot = pivot_matrix(d_like, row="z_dim", col="beta", val=metric)
                fig, ax = plt.subplots(figsize=(4.2,3.6))
                imshow_matrix(ax,
                              pivot.values,
                              row_labels=pivot.index.astype(str).tolist(),
                              col_labels=[str(c) for c in pivot.columns.tolist()],
                              title=f"VIB {metric} — z_dim × beta ({like})",
                              cbar_label=metric)
                ax.set_xlabel("beta"); ax.set_ylabel("z_dim")
                save_multi_and_close(fig, f"abl_vib_heatmap_{like}_{metric.lower()}")
        print("Saved: figs/abl_vib_heatmap_*.pdf")

    # (C) MB misspecification
    d_mis = df[df.get("abl","") == "mb_misspec"].copy()
    if not d_mis.empty:
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(7,3.2))
            lbls = sorted(d_mis["misspec"].unique())
            methods = sorted(d_mis["method"].unique())
            x = np.arange(len(lbls)); width = 0.8/len(methods)
            # mean ± se per misspec
            for i, method in enumerate(methods):
                dl = d_mis[d_mis["method"] == method]
                m, s = agg_mean_se(dl, ["misspec"], metric)
                m = m.set_index("misspec").loc[lbls].reset_index()
                s = s.set_index("misspec").loc[lbls].reset_index()
                ax.bar(x + i*width, m[f"{metric}_mean"], width, yerr=s[f"{metric}_se"], capsize=3, label=method)
            ax.set_xticks(x + width*(len(methods)-1)/2)
            ax.set_xticklabels(lbls)
            ax.set_title(f"MB Misspecification — {metric}")
            ax.set_ylabel(metric)
            ax.legend()
            save_and_close(fig, os.path.join(OUTDIR, f"abl_mb_misspec_{metric.lower()}.pdf"))
        print("Saved: figs/abl_mb_misspec_*.pdf")

# -----------------------------
# 3) Sensitivity
# -----------------------------
if os.path.exists("sensitivity.csv"):
    df = pd.read_csv("sensitivity.csv")

    # (A) Shift sweeps: RMSE vs each parameter, per scenario (MB-GIB & MB-VIB)
    # Treat any scenario that starts with "Target" (e.g., "Target(gen)") as target-shift.
    d_shift = df[df["scenario"].apply(lambda s: (isinstance(s, str) and (s == "Covariate" or s.startswith("Target"))))].copy()
    if not d_shift.empty:
        # Include generalized target-shift knobs if they exist
        params_all = ["cov_mu","cov_sd","tgt_mu","tgt_sd","tgt_eps_mu","tgt_eps_sd","tgt_add_const"]
        params = [c for c in params_all if c in d_shift.columns]
        for param in params:
            for scenario in sorted(d_shift["scenario"].unique()):
                subset = d_shift[d_shift["scenario"] == scenario]
                # curve: average over seeds/repeats per (method, param)
                m = subset.groupby(["method", param])["RMSE"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(7,3.0))
                for method in ["MB-GIB","MB-VIB"]:
                    mm = m[m["method"] == method].sort_values(param)
                    if not mm.empty:
                        ax.plot(mm[param].values, mm["RMSE"].values, marker="o", label=method)
                ax.set_xlabel(param); ax.set_ylabel("RMSE"); ax.legend()
                # make filenames robust: remove spaces/parens
                scen_tag = scenario.lower().replace("(","").replace(")","").replace(" ","_")
                save_and_close(fig, os.path.join(OUTDIR, f"sens_{scen_tag}_{param}_rmse.pdf"))
        print("Saved: figs/sens_*_rmse.pdf")

    # (B) Support mismatch × missingness (heatmaps), per method
    d_sup = df[df["scenario"] == "support/missing"].copy()
    if not d_sup.empty:
        for method in sorted(d_sup["method"].unique()):
            dd = d_sup[d_sup["method"] == method]
            if "stretch" in dd.columns and "miss" in dd.columns:
                pivot = pivot_matrix(dd, row="stretch", col="miss", val="RMSE")
                fig, ax = plt.subplots(figsize=(4.4,3.6))
                imshow_matrix(ax, pivot.values,
                              row_labels=[str(r) for r in pivot.index.tolist()],
                              col_labels=[str(c) for c in pivot.columns.tolist()],
                              title=f"{method}: RMSE — stretch × miss",
                              cbar_label="RMSE")
                ax.set_xlabel("missing_rate"); ax.set_ylabel("stretch")
                # dual save (PNG+PDF with rasterized heatmap)
                save_multi_and_close(fig, f"sens_support_missing_{method}_rmse")
        print("Saved: figs/sens_support_missing_*_rmse.*")

    # (C) N_s learning curves (MB-GIB vs MB-VIB)
    d_ns = df[df["scenario"] == "Ns_curve"].copy()
    if not d_ns.empty and "Ns" in d_ns.columns:
        for metric in ["RMSE","MAE","R2"]:
            m = d_ns.groupby(["method","Ns"])[metric].mean().reset_index()
            fig, ax = plt.subplots(figsize=(7,3.0))
            for method in ["MB-GIB","MB-VIB"]:
                mm = m[m["method"] == method].sort_values("Ns")
                if not mm.empty:
                    ax.plot(mm["Ns"].values, mm[metric].values, marker="o", label=method)
            ax.set_xlabel("N_s (source sample size)")
            ax.set_ylabel(metric)
            ax.legend()
            save_and_close(fig, os.path.join(OUTDIR, f"sens_learning_{metric.lower()}.pdf"))
        print("Saved: figs/sens_learning_*.pdf")


print("\nDone. Figures are in ./figs/")
