# dag_vib_latent.py
# DAG-aware latent Information Bottleneck (linear Gaussian).
# Learns an encoder Z = (X_S - mu_X) @ P  and a separate decoder Y_hat = Z @ B (+ mu_Y),
# where X_S are DAG-chosen features: Parents(T), MB(T), or Global (no T).
#
# Inputs: same CSV schema as your current script, plus target_var selection.
#   - adjacency_matrix.csv  (index/cols = node names; must include "T")
#   - source_1.csv          (complete source; includes target_var)
#   - tgt_target_missing_1.csv  (target; target_var missing)
#   - tgt_target_true_1.csv     (target; target_var for eval only)
#
# Author: updates to separate Z from the graph node T.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh, inv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Utilities
# ---------------------------

def invsqrt_psd(S, ridge=0.0):
    S = 0.5 * (S + S.T)
    if ridge > 0:
        S = S + ridge * np.eye(S.shape[0])
    vals, vecs = eigh(S)
    vals = np.clip(vals, 0.0, None)
    eps = 1e-12
    inv_sqrt_vals = np.where(vals > eps, 1.0 / np.sqrt(vals), 0.0)
    Sinvhalf = (vecs * inv_sqrt_vals) @ vecs.T
    return Sinvhalf, vals[::-1]

def choose_d_from_eigs(eigs, frac=0.99, max_d=None):
    eigs = np.asarray(eigs)
    if eigs.size == 0:
        return 0
    total = eigs.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(eigs)
    d = int(np.searchsorted(cum / total, frac) + 1)
    if max_d is not None:
        d = min(d, max_d)
    return d

def markov_blanket_indices(amat, idx_t):
    """
    Markov blanket of T: parents(T) ∪ children(T) ∪ parents(children(T)) \ {T}
    amat[j, i] = 1 if j -> i
    """
    p = amat.shape[0]
    parents_T = {j for j in range(p) if amat[j, idx_t] != 0}
    children_T = {i for i in range(p) if amat[idx_t, i] != 0}
    spouses = set()
    for c in children_T:
        for j in range(p):
            if amat[j, c] != 0 and j != idx_t:
                spouses.add(j)
    mb = (parents_T | children_T | spouses) - {idx_t}
    return sorted(mb), sorted(parents_T), sorted(children_T), sorted(spouses)

# ---------------------------
# Core: Linear IB encoder (CCA-style) and decoder
# ---------------------------

def fit_linear_ib_encoder(X, Y, ridge_x=1e-8, ridge_y=1e-8, frac=0.99, d=None):
    """
    Fit a linear IB encoder using a CCA-style eigenproblem between X and Y.
    Returns a projection P (q x d) that maps centered X to Z = (X - muX) @ P.

    X: (n, q)  inputs (parents/MB/global), will be centered
    Y: (n,) or (n, m) targets, will be centered
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n, q = X.shape
    _, m = Y.shape

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    Sxx = (Xc.T @ Xc) / max(n, 1)
    Syy = (Yc.T @ Yc) / max(n, 1)
    Sxy = (Xc.T @ Yc) / max(n, 1)

    # Stabilize
    trace_x = np.trace(Sxx)
    tiny_x = ridge_x if ridge_x > 0 else (1e-8 * trace_x / max(q, 1))
    Sxx_invhalf, _ = invsqrt_psd(Sxx, ridge=tiny_x)

    trace_y = np.trace(Syy)
    tiny_y = ridge_y if ridge_y > 0 else (1e-8 * trace_y / max(m, 1))
    # For Syy^{-1}, use ridge and inverse in the Y-space
    Syy_r = 0.5 * (Syy + Syy.T) + tiny_y * np.eye(m)
    Syy_inv = inv(Syy_r)

    # M = Sxx^{-1/2} Sxy Syy^{-1} Sxy^T Sxx^{-1/2}
    M = Sxx_invhalf @ (Sxy @ (Syy_inv @ Sxy.T)) @ Sxx_invhalf
    M = 0.5 * (M + M.T)
    evals, U = eigh(M)              # ascending
    evals = np.clip(evals, 0.0, None)
    order = np.argsort(evals)[::-1]  # descending
    evals = evals[order]
    U = U[:, order]

    if d is None:
        d = choose_d_from_eigs(evals, frac=frac, max_d=min(q, m))

    Ud = U[:, :d]
    Pd = (Sxx_invhalf @ Ud)         # q x d

    enc = {
        "muX": muX, "muY": muY, "P": Pd, "evals": evals, "d": d
    }
    return enc

def encode_Z(enc, X):
    Xc = np.asarray(X, float) - enc["muX"]
    return Xc @ enc["P"]  # (n, d)

def fit_decoder_from_Z(Z, Y):
    """
    Fit linear decoder Y ≈ Z @ B + muY. Returns B.
    Handles scalar or vector Y (column-wise regression).
    """
    Z = np.asarray(Z, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Ordinary least squares with small ridge for stability
    n, d = Z.shape
    _, m = Y.shape
    lam = 1e-8
    G = Z.T @ Z + lam * np.eye(d)     # d x d
    B = np.linalg.solve(G, Z.T @ Y)   # d x m
    return B

def predict_from_Z(B, Z, muY):
    Yhat = Z @ B + muY
    if Yhat.shape[1] == 1:
        return Yhat.ravel()
    return Yhat

def evaluate_and_plot(y, yhat, title, pdf_path, ylabel="Predicted"):
    y = np.asarray(y).ravel()
    yhat = np.asarray(yhat).ravel()
    mae = float(mean_absolute_error(y, yhat))
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    r2 = float(r2_score(y, yhat))

    plt.figure(figsize=(6, 6))
    plt.scatter(y, yhat, alpha=0.6)
    mn, mx = float(np.nanmin(y)), float(np.nanmax(y))
    plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
    plt.xlabel("True", fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.tick_params(labelsize=24)
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ---------------------------
# Main: Parents vs MB vs Global encoders → latent Z → decoder to Y
# ---------------------------

if __name__ == "__main__":
    # ---- Config ----
    target_var = "T"          # set to any node/target you want to predict
    encoder_scope = ["parent", "mb", "global"]  # which encoders to run
    frac_energy = {"parent": 0.999, "mb": 0.99, "global": 0.99}  # IB energy keep
    bottleneck_cap = None     # or an int to cap latent dim

    # ---- Load data ----
    source_df = pd.read_csv("source_1.csv")
    target_obs_df = pd.read_csv("tgt_target_missing_1.csv")  # target_var missing
    target_true_df = pd.read_csv("tgt_target_true_1.csv")    # target_var for eval
    amat_df = pd.read_csv("adjacency_matrix.csv", index_col=0)

    nodes = list(amat_df.index)
    p = len(nodes)
    amat = amat_df.values.astype(int)

    if "T" not in nodes:
        raise ValueError("Node 'T' not found in adjacency_matrix.csv index/columns.")
    idx_t = nodes.index("T")

    if target_var not in source_df.columns:
        raise ValueError(f"target_var='{target_var}' not found in source_1.csv columns.")

    # Parent lists from adjacency (amat[j, i] = 1 if j -> i)
    parent_list = [[j for j in range(p) if amat[j, i] != 0] for i in range(p)]

    # --- Build feature sets X_S (DAG constraint) ---
    # Global: all variables except T (to avoid label leakage if target_var == 'T')
    global_feats = [k for k in range(p) if k != idx_t]
    parent_feats = parent_list[idx_t]  # parents of T
    mb_feats, parT, chT, spT = markov_blanket_indices(amat, idx_t)

    # Extract matrices
    Xs_global = source_df[nodes].iloc[:, global_feats].values
    Xt_global = target_obs_df[nodes].iloc[:, global_feats].values

    Xs_parent = source_df[nodes].iloc[:, parent_feats].values if len(parent_feats) > 0 else None
    Xt_parent = target_obs_df[nodes].iloc[:, parent_feats].values if len(parent_feats) > 0 else None

    Xs_mb = source_df[nodes].iloc[:, mb_feats].values if len(mb_feats) > 0 else None
    Xt_mb = target_obs_df[nodes].iloc[:, mb_feats].values if len(mb_feats) > 0 else None

    Ys = source_df[target_var].values
    Yt_true = target_true_df[target_var].values

    results = []

    def run_one(scope_name, Xs, Xt):
        if Xs is None or Xt is None or Xs.shape[1] == 0:
            print(f"No features for {scope_name} encoder; degenerates to mean predictor.")
            pred = np.repeat(np.mean(Ys), len(Yt_true))
            r = evaluate_and_plot(Yt_true, pred, f"{scope_name}-VIB (empty → mean)", f"VIB_{target_var}_{scope_name}.pdf", ylabel=f"Predicted {target_var}")
            return r

        enc = fit_linear_ib_encoder(
            X=Xs, Y=Ys,
            ridge_x=1e-8, ridge_y=1e-8,
            frac=frac_energy.get(scope_name, 0.99),
            d=bottleneck_cap
        )
        print(f"{scope_name}: kept d={enc['d']} latent dims; top canonical rho^2 ≈ {enc['evals'][:min(5, len(enc['evals']))]}")

        Zs = encode_Z(enc, Xs)
        Zt = encode_Z(enc, Xt)
        B = fit_decoder_from_Z(Zs, Ys.reshape(-1, 1))
        Yhat = predict_from_Z(B, Zt, enc['muY'])

        r = evaluate_and_plot(Yt_true, Yhat, f"{scope_name}-VIB (d={enc['d']})", f"VIB_{target_var}_{scope_name}.pdf", ylabel=f"Predicted {target_var}")
        return r

    if "parent" in encoder_scope:
        r_parent = run_one("parent", Xs_parent, Xt_parent)
        r_parent["model"] = "tgt_Parent-VIB"
        results.append(r_parent)

    if "mb" in encoder_scope:
        r_mb = run_one("mb", Xs_mb, Xt_mb)
        r_mb["model"] = "tgt_MB-VIB"
        results.append(r_mb)

    if "global" in encoder_scope:
        r_global = run_one("global", Xs_global, Xt_global)
        r_global["model"] = "tgt_Global-VIB"
        results.append(r_global)

    # ---------------- Compare & Save metrics ----------------
    res_df = pd.DataFrame(results, columns=["model", "MAE", "RMSE", "R2"])
    print("\n=== DAG-aware latent IB Summary ===")
    print(res_df.to_string(index=False))
    res_df.to_csv(f"VIB_{target_var}_results.csv", index=False)
    print(f"\nSaved plots: VIB_{target_var}_parent.pdf / _mb.pdf / _global.pdf, and VIB_{target_var}_results.csv")
