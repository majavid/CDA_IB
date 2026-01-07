# ======== CLEARER DIFFERENCE PLOTS ========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, math, random

def save_and_close(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def bootstrap_ci(x, iters=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan)
    bs = []
    n = len(x)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        bs.append(np.mean(x[idx]))
    lo = np.percentile(bs, 100*alpha/2)
    hi = np.percentile(bs, 100*(1-alpha/2))
    return (float(lo), float(hi))

def dot_whisker(df, metric="R2", out="figs/main_dotwhisker_R2.png", ylim=None):
    # mean ± bootstrap CI per method, per scenario (horizontal)
    scenarios = sorted(df["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(9, 3.0), sharex=True)
    if len(scenarios) == 1: axes = [axes]
    for ax, sc in zip(axes, scenarios):
        d = df[df["scenario"]==sc].copy()
        # aggregate
        rows=[]
        for m in sorted(d["method"].unique(), key=lambda x: {"MB-GIB":0, "MB-VIB":1}.get(x, 99)):
            vals = d.loc[d["method"]==m, metric].values
            mu = float(np.mean(vals))
            lo, hi = bootstrap_ci(vals, rng=np.random.default_rng(1))
            rows.append((m, mu, lo, hi))
        agg = pd.DataFrame(rows, columns=["method","mean","lo","hi"])
        agg = agg.sort_values("mean")  # low→high so best is at top when flipped
        y = np.arange(len(agg))
        ax.hlines(y, agg["lo"], agg["hi"])
        ax.plot(agg["mean"], y, "o")
        ax.set_yticks(y); ax.set_yticklabels(agg["method"])
        ax.invert_yaxis()
        ax.set_title(f"{sc} shift")
        ax.set_xlabel(metric)
        if ylim is not None: ax.set_xlim(ylim)
    save_and_close(fig, out)

def delta_to_bn(df, metric="R2", baseline="BN", out="figs/delta_to_BN_R2.png"):
    # show improvement over BN (mean Δ and CI), per scenario
    scenarios = sorted(df["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(9, 3.0), sharey=True)
    if len(scenarios) == 1: axes = [axes]
    for ax, sc in zip(axes, scenarios):
        d = df[df["scenario"]==sc].copy()
        if baseline not in d["method"].unique():
            ax.set_title(f"{sc}: no {baseline}"); continue
        # compute seed-wise deltas by joining on seed
        base = d[d["method"]==baseline][["seed", metric]].rename(columns={metric:"base"})
        out_rows=[]
        for m in sorted(d["method"].unique(), key=lambda x: {"MB-GIB":0, "MB-VIB":1}.get(x, 99)):
            if m == baseline: continue
            dm = d[d["method"]==m][["seed",metric]].merge(base, on="seed", how="inner")
            delta = (dm[metric] - dm["base"]).values  # positive = better than BN
            mu = float(np.mean(delta)); lo,hi = bootstrap_ci(delta, rng=np.random.default_rng(2))
            out_rows.append((m, mu, lo, hi))
        agg = pd.DataFrame(out_rows, columns=["method","mean","lo","hi"])
        # order by mean descending
        agg = agg.sort_values("mean", ascending=False)
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=[agg["mean"]-agg["lo"], agg["hi"]-agg["mean"]], capsize=3)
        ax.axhline(0, linestyle="--")
        ax.set_xticks(x); ax.set_xticklabels(agg["method"], rotation=15)
        ax.set_title(f"Δ {metric} vs {baseline} — {sc}")
        ax.set_ylabel(f"Δ {metric} (↑ better)")
    save_and_close(fig, out)

def dumbbell_bn_vs_mb(df, metric="R2", out="figs/dumbbell_BN_vs_MB_R2.png"):
    # per scenario, draw BN→MB-GIB and BN→MB-VIB dumbbells
    scenarios = sorted(df["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(9, 3.0), sharey=True)
    if len(scenarios) == 1: axes = [axes]
    targets = ["MB-GIB","MB-VIB"]
    for ax, sc in zip(axes, scenarios):
        d = df[df["scenario"]==sc].copy()
        if not {"BN", *targets}.issubset(set(d["method"].unique())):
            ax.set_title(f"{sc}: missing methods"); continue
        # aggregate means by method
        g = d.groupby("method")[metric].mean()
        bn = g.get("BN", np.nan)
        pts=[]
        for t in targets:
            if t in g:
                pts.append((t, float(bn), float(g[t])))
        # sort by target score
        pts = sorted(pts, key=lambda x: x[2])
        y = np.arange(len(pts))
        for i,(lbl, x0, x1) in enumerate(pts):
            ax.plot([x0, x1], [i, i], "-")
            ax.plot([x0], [i], "o")  # BN
            ax.plot([x1], [i], "o")  # target
        ax.set_yticks(y); ax.set_yticklabels([p[0] for p in pts])
        ax.set_xlabel(metric); ax.set_title(f"{sc}: BN → MB")
    save_and_close(fig, out)

def seed_scatter_overlay(df, metric="R2", out="figs/seed_scatter_overlay_R2.png", xlim=None):
    # per scenario: light seed points + thick mean bars (methods sorted)
    scenarios = sorted(df["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(9,3.0), sharex=True, sharey=True)
    if len(scenarios) == 1: axes=[axes]
    for ax, sc in zip(axes, scenarios):
        d = df[df["scenario"]==sc]
        order = sorted(d["method"].unique(), key=lambda x: {"MB-GIB":0,"MB-VIB":1}.get(x, 99))
        yvals = np.arange(len(order))
        # points
        for i, m in enumerate(order):
            vals = d[d["method"]==m][metric].values
            # jitter on x; plotting horizontally is clearer, but stick vertical here intentionally
            x = np.full_like(vals, i, dtype=float) + (np.random.rand(len(vals))-0.5)*0.1
            ax.scatter(x, vals, s=16, alpha=0.35)
        # means + CIs
        mus, los, his = [], [], []
        for m in order:
            vals = d[d["method"]==m][metric].values
            mu = float(np.mean(vals)); lo,hi = bootstrap_ci(vals, rng=np.random.default_rng(3))
            mus.append(mu); los.append(lo); his.append(hi)
        ax.errorbar(np.arange(len(order)), mus, yerr=[np.array(mus)-np.array(los), np.array(his)-np.array(mus)],
                    fmt="o", capsize=3, lw=2)
        ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order, rotation=15)
        ax.set_title(f"{sc} shift"); ax.set_ylabel(metric)
        if xlim: ax.set_ylim(xlim)
    save_and_close(fig, out)

# ---- Run the enhanced visuals if main_results.csv exists ----
if os.path.exists("main_results.csv"):
    df_main = pd.read_csv("main_results.csv")
    # 1) Horizontal dot-whisker (tighten R² axis if desired)
    dot_whisker(df_main, metric="R2", out="figs/main_dotwhisker_R2.png", ylim=(0.3,1.0))
    # 2) Δ vs BN (R²)
    delta_to_bn(df_main, metric="R2", baseline="BN", out="figs/delta_to_BN_R2.png")
    # 3) BN → MB dumbbells
    dumbbell_bn_vs_mb(df_main, metric="R2", out="figs/dumbbell_BN_vs_MB_R2.png")
    # 4) Seed scatter overlay (variance visibility)
    seed_scatter_overlay(df_main, metric="R2", out="figs/seed_scatter_overlay_R2.png", xlim=(0.3,1.0))
    print("Saved clearer difference plots in figs/:")
    print(" - main_dotwhisker_R2.png")
    print(" - delta_to_BN_R2.png")
    print(" - dumbbell_BN_vs_MB_R2.png")
    print(" - seed_scatter_overlay_R2.png")
