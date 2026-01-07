import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------
# Configuration (edit as needed)
# -----------------------
seed = 42
k = 5          # number of MB features C
m = 100        # number of spurious features S
r = 5          # number of nuisance features N

n_source = 1000
n_target = 1000

w = np.ones(k, dtype=float)       # shape (k,)
sigma_T = 1.0
sigma_S = 0.1
sigma_N = 1.0

a_source = +1.0   # a_0
a_target = -1.0   # a_1 (flip sign)
b_source = 0.0
b_target = 1.0

out_dir = Path("D:/Paper/AAMAS\AAMAS2026/simulated_data/motivatingExample")
# -----------------------

rng = np.random.default_rng(seed)

# Column ordering for both adjacency matrix and datasets
col_D = ["D"]
col_C = [f"C{i+1}" for i in range(k)]
col_T = ["T"]
col_S = [f"S{j+1}" for j in range(m)]
col_N = [f"N{l+1}" for l in range(r)]
cols = col_D + col_C + col_T + col_S + col_N

# ---------- Adjacency matrix (binary) ----------
# Edges:
# C -> T
# C -> S_j (all j)
# D -> S_j (all j)
# D -> N_l (all l)
p = len(cols)
A = np.zeros((p, p), dtype=int)
idx = {name: i for i, name in enumerate(cols)}

# C -> T
for ci in col_C:
    A[idx[ci], idx["T"]] = 1

# C -> S
for ci in col_C:
    for sj in col_S:
        A[idx[ci], idx[sj]] = 1

# D -> S
for sj in col_S:
    A[idx["D"], idx[sj]] = 1

# D -> N
for nl in col_N:
    A[idx["D"], idx[nl]] = 1

adj_df = pd.DataFrame(A, index=cols, columns=cols)

# ---------- Data generation ----------
def generate_domain(n: int, D_value: int, a_D: float, b_D: float) -> pd.DataFrame:
    # C ~ N(0, I)
    C = rng.normal(0.0, 1.0, size=(n, k))

    # U = w^T C
    U = C @ w.reshape(-1, 1)  # (n,1)

    # T = w^T C + eps_T  (same mechanism in both domains)
    eps_T = rng.normal(0.0, sigma_T, size=(n, 1))
    T = U + eps_T

    # S_j = a_D * U + eps_S (domain-dependent sign flip)
    eps_S = rng.normal(0.0, sigma_S, size=(n, m))
    S = a_D * U + eps_S

    # N_l = b_D + eps_N (domain-dependent mean shift, pure nuisance)
    eps_N = rng.normal(0.0, sigma_N, size=(n, r))
    N = b_D + eps_N

    df = pd.DataFrame(
        np.concatenate(
            [
                np.full((n, 1), float(D_value)),
                C,
                T,
                S,
                N,
            ],
            axis=1,
        ),
        columns=cols,
    )
    return df

source_df = generate_domain(n_source, D_value=0, a_D=a_source, b_D=b_source)
target_df = generate_domain(n_target, D_value=1, a_D=a_target, b_D=b_target)

# ---------- Save ----------
adj_path = out_dir / "adjacency_matrix.csv"
src_path = out_dir / "source.csv"
tgt_path = out_dir / "target.csv"

adj_df.to_csv(adj_path, index=True)
source_df.to_csv(src_path, index=False)
target_df.to_csv(tgt_path, index=False)

adj_path, src_path, tgt_path, adj_df.shape, source_df.shape, target_df.shape, source_df.head(2), target_df.head(2)

