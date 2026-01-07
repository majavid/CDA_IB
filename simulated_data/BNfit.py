import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- (helper) OLS fit of a Gaussian DAG node --------------------------------

def lmfit(s: np.ndarray, y: int, x: list[int]) -> np.ndarray:
    """
    Fit regression of node y on parents x using covariance matrix s.
    Returns vector m of length p: m[parent] = beta, m[y] = residual variance.
    """
    p = s.shape[0]
    m = np.zeros(p)
    if not x:
        m[y] = s[y, y]
        return m
    S_xx = s[np.ix_(x, x)]
    S_xy = s[np.ix_(x, [y])].reshape(-1,)
    beta = np.linalg.solve(S_xx, S_xy)
    for idx, parent in enumerate(x):
        m[parent] = beta[idx]
    m[y] = s[y, y] - S_xy @ beta
    return m

def fitdag(amat: np.ndarray, s: np.ndarray, parent_list: list[list[int]]) -> dict:
    """
    Fit a recursive Gaussian DAG via GLS on covariance matrix s.

    Convention used consistently here:
      amat[j, i] = 1  means edge  j → i  (j is a parent of i).

    parent_list[i] = list of j such that amat[j, i] == 1 and j != i
    Returns dict with keys 'A', 'Delta', 'Shat'.
    """
    p = s.shape[0]
    A = np.zeros((p, p))
    Delta = np.zeros(p)

    for i in range(p):
        parents = parent_list[i]
        if not parents:
            Delta[i] = s[i, i]
            A[i, i]  = 1.0
        else:
            m = lmfit(s, y=i, x=parents)
            for parent in parents:
                A[i, parent] = -m[parent]
            A[i, i] = 1.0
            Delta[i] = m[i]

    Khat = A.T @ np.diag(1.0 / Delta) @ A
    Shat = np.linalg.inv(Khat)
    return {"A": A, "Delta": Delta, "Shat": Shat}

# --- (main) “Fit-on-Source” baseline for T using mean-aware form -----------

# 1) Load source data
source_df = pd.read_csv("source_1.csv")
n_source  = source_df.shape[0]

# 2) Load adjacency matrix and prepare nodes / amat
amat_df = pd.read_csv("adjacency_matrix.csv", index_col=0)
amat_df.index   = amat_df.index.astype(str).str.strip()
amat_df.columns = amat_df.columns.astype(str).str.strip()

nodes = list(amat_df.index)   # node order by index rows
p     = len(nodes)
amat  = amat_df.values.astype(int)

# 2a) Build parent_list with the convention amat[j, i] == 1 implies j → i
parent_list = [[j for j in range(p) if amat[j, i] == 1 and j != i] for i in range(p)]

# 2b) Print parents for each node (indices + names)
for i, parents in enumerate(parent_list):
    parent_names = [nodes[j] for j in parents]
    print(f"Node {i} ({nodes[i]}): parents idx={parents}  names={parent_names}")

# 3) Fit the DAG on the source domain (covariance with ddof=1)
S_source = source_df[nodes].cov().values
dag_fit  = fitdag(amat=amat, s=S_source, parent_list=parent_list)
A_src    = dag_fit["A"]
Sigma_src = dag_fit["Shat"]  # available if you later want Σ-based conditionals

# 4) Extract β-coefficients for T in SAME order as X_obs columns (parents of T)
idx_T          = nodes.index("T")
parents_idx_T  = parent_list[idx_T]
parents_T      = [nodes[j] for j in parents_idx_T]

beta_vec_ht = np.array([-A_src[idx_T, j] for j in parents_idx_T], dtype=float)

print("\nRegression coefficients learned for T on the SOURCE domain (aligned with X columns):")
for j, parent in zip(parents_idx_T, parents_T):
    coeff = -A_src[idx_T, j]
    print(f"   beta[{parent} -> T] = {coeff:.6f}")
print()

# 5) Load target (T missing) and impute using mean-aware form
target_df = pd.read_csv("tgt_target_missing_1.csv")  # T is NaN here

# Ensure required parent columns exist
missing_cols = [c for c in parents_T if c not in target_df.columns]
if missing_cols:
    raise ValueError(f"'tgt_target_missing_1.csv' is missing required parent columns: {missing_cols}")

X_obs = target_df[parents_T].values  # (n_target × num_parents), same order as beta_vec_ht

# Mean-aware prediction using ONLY parents:
# E[T | parents] = mu_t + beta^T (x - mu_parents)
# Under zero-mean noise in the SEM, mu_t = beta^T mu_parents (in ANY domain),
# so this collapses to  beta^T x. We compute it both ways for clarity.

mu_parents_tgt = target_df[parents_T].mean(axis=0).values  # (num_parents,)
mu_t_baseline  = float(mu_parents_tgt @ beta_vec_ht)        # = beta^T mu_parents
imputed_T_ma  = mu_t_baseline + (X_obs - mu_parents_tgt) @ beta_vec_ht  # mean-aware
imputed_T     = X_obs @ beta_vec_ht                        # equivalent under the assumption

# Sanity check: the two should be (numerically) identical
max_abs_diff = float(np.max(np.abs(imputed_T_ma - imputed_T)))
print(f"Sanity check (mean-aware vs plain X@beta): max |diff| = {max_abs_diff:.3e}\n")

# 6) Evaluate against true T
targetT_df = pd.read_csv("tgt_target_true_1.csv")
true_T    = targetT_df["T"].values

mask        = (~np.isnan(true_T)) & (~np.isnan(imputed_T_ma))
true_T     = true_T[mask]
imputed_T  = imputed_T_ma[mask]  # use the explicit mean-aware version

mae  = mean_absolute_error(true_T, imputed_T)
mse  = mean_squared_error(true_T, imputed_T)
rmse = np.sqrt(mse)
r2   = r2_score(true_T, imputed_T)

print("“Fit-on-Source (Parents-only)” baseline for T (mean-aware):")
print(f"  MAE  = {mae:.4f}")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R²   = {r2:.4f}\n")

# 7) Scatter plot: True vs. Predicted T
plt.figure(figsize=(6, 6))
plt.scatter(true_T, imputed_T, alpha=0.6, label="Baseline (parents-only)")
mn, mx = np.nanmin(true_T), np.nanmax(true_T)
plt.plot([mn, mx], [mn, mx], "k--", lw=2)
plt.xlabel("True T", fontsize=32)
plt.ylabel("Predicted T", fontsize=32)
plt.tick_params(axis="both", which="major", labelsize=24)
plt.tight_layout()
plt.savefig("T_baseline_scatter.pdf", format="pdf", bbox_inches="tight")
plt.show()
