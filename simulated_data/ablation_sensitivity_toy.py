#!/usr/bin/env python3
"""
ablation_sensitivity_toy.py

Ablation & sensitivity study on the 7-node SEM toy example.

Compares: MB–GIB, MB–VIB, BN baseline, Pure DNN, IIB-style
Studies:
  (1) Main results (covariate shift on C2; generalized target shift on C1 and eps_T).
  (2) Ablations: feature scope (Parents/MB/Global), capacity (d or z_dim),
                 beta (VIB), likelihood (VIB), MB misspecification.
  (3) Sensitivity: shift magnitudes/types, support mismatch, missingness in MB,
                  and source sample size (N_s) curves.

Outputs: CSVs under --outdir.

Dependencies: numpy, pandas, sklearn, torch, matplotlib (optional for other scripts).
"""

import os, argparse, itertools
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----- project imports (ensure these files are in the same folder) -----
import vib_dag_latent_ib_nongauss as vibmod        # train_vib, vib_predict
import dag_Gib_latent as gibmod                    # fit_linear_ib_encoder, encode_Z, fit_decoder_from_Z, predict_from_Z
import BNfit as bnmod                              # fitdag (we’ll extract parents(T)->T coeffs)
import compare_methods as cmpmod                   # train_pure_dnn, train_IIB_style

# ---------------------------
# Utilities
# ---------------------------
def metrics(y, yhat) -> Dict[str, float]:
    y = np.asarray(y).ravel(); yhat = np.asarray(yhat).ravel()
    return dict(
        MAE=float(mean_absolute_error(y, yhat)),
        RMSE=float(np.sqrt(mean_squared_error(y, yhat))),
        R2=float(r2_score(y, yhat))
    )

def safe_mkdir(p): os.makedirs(p, exist_ok=True)

# ---------------------------
# DAG: 7-node structure and MB extraction
# Nodes: ["C1","C2","Z","X","T","P","Y"]
# Edges: C1->Z, C2->Z, C1->X, C1->T, X->T, Z->T, T->P, T->Y
# ---------------------------
NODES = ["C1","C2","Z","X","T","P","Y"]
IDX = {n:i for i,n in enumerate(NODES)}

def adjacency_matrix_7():
    p = len(NODES)
    A = np.zeros((p,p), dtype=int)
    def e(u,v): A[IDX[u], IDX[v]] = 1
    e("C1","Z"); e("C2","Z"); e("C1","X"); e("C1","T"); e("X","T"); e("Z","T"); e("T","P"); e("T","Y")
    return A  # amat[j,i]=1 if j->i

def markov_blanket_T(amat: np.ndarray) -> Tuple[List[int], List[int]]:
    """MB(T) = pa(T) ∪ ch(T) ∪ pa(ch(T)) \ {T}; parents(T) returned separately for the 'parent' scope."""
    idx_t = IDX["T"]; p = amat.shape[0]
    parents = {j for j in range(p) if amat[j, idx_t] != 0}
    children = {i for i in range(p) if amat[idx_t, i] != 0}
    spouses = set()
    for c in children:
        for j in range(p):
            if amat[j, c] != 0 and j != idx_t:
                spouses.add(j)
    mb = (parents | children | spouses) - {idx_t}
    return sorted(mb), sorted(parents)

# ---------------------------
# Data generation (in-memory)
# Source SEM:
#   C1~N(0,1), C2~N(0,1)
#   Z = 2*C1 + 3*C2 + eps_Z,  eps_Z~N(0,1)
#   X = 3*C1 + eps_X,         eps_X~N(0,1)
#   T = 1*C1 + 2*X + 3*Z + eps_T, eps_T~N(mu_epsT, sd_epsT)
#   P = 1*T + eps_P,          eps_P~N(0,1)
#   Y = 2*T + eps_Y,          eps_Y~N(0,1)
#
# Shift controls (target):
# - covariate shift: C2 ~ N(mu_cov, sd_cov^2) (C1, eps_T like source)
# - generalized target shift: C1 ~ N(mu_tgt, sd_tgt^2); eps_T ~ N(mu_epsT, sd_epsT^2); optional +const on T
# ---------------------------
def sample_sem(n: int,
               mode: str,
               rng: np.random.Generator,
               mu_cov=5.0, sd_cov=2.0,            # covariate shift on C2
               mu_tgt=5.0, sd_tgt=2.0,            # target shift on C1
               mu_epsT=0.0, sd_epsT=1.0, add_T_const=0.0,  # generalized target shift on T
               support_stretch: float = 1.0       # support mismatch multiplier applied to X_M (later)
               ) -> pd.DataFrame:
    # source or target (cov / tgt)
    if mode == "source":
        C1 = rng.normal(0,1,n)
        C2 = rng.normal(0,1,n)
        mu_epsT_now, sd_epsT_now, add_const_now = 0.0, 1.0, 0.0
    elif mode == "cov":         # covariate shift on C2 only
        C1 = rng.normal(0,1,n)
        C2 = rng.normal(mu_cov, sd_cov, n)
        mu_epsT_now, sd_epsT_now, add_const_now = 0.0, 1.0, 0.0
    elif mode == "tgt":         # generalized target shift on C1 and/or eps_T
        C1 = rng.normal(mu_tgt, sd_tgt, n)
        C2 = rng.normal(0,1,n)
        mu_epsT_now, sd_epsT_now, add_const_now = mu_epsT, sd_epsT, add_T_const
    else:
        raise ValueError("mode must be in {'source','cov','tgt'}")

    eps_Z = rng.normal(0,1,n); Z = 2*C1 + 3*C2 + eps_Z
    eps_X = rng.normal(0,1,n); X = 3*C1 + eps_X
    eps_T = rng.normal(mu_epsT_now, sd_epsT_now, n)
    T = 1*C1 + 2*X + 3*Z + eps_T + add_const_now
    eps_P = rng.normal(0,1,n); P = 1*T + eps_P
    eps_Y = rng.normal(0,1,n); Y = 2*T + eps_Y

    df = pd.DataFrame({"C1":C1,"C2":C2,"Z":Z,"X":X,"T":T,"P":P,"Y":Y})
    if support_stretch != 1.0:
        df.attrs["support_stretch"] = float(support_stretch)
    return df

# ---------------------------
# Feature builders (Parents / MB / Global), missingness, support stretch
# ---------------------------
def build_feature_matrix(df: pd.DataFrame, subset: str, amat: np.ndarray,
                         missing_rate: float = 0.0,
                         mb_misspec: str = "none") -> Tuple[np.ndarray, List[str]]:
    """
    subset in {"parent","mb","global"}.
    missing_rate: randomly mask that fraction of features to NaN (then mean-impute).
    mb_misspec: "none" | "drop_one" | "add_nonmb"
    """
    nodes = NODES[:]  # fixed order
    idx_t = IDX["T"]
    mb_idx, par_idx = markov_blanket_T(amat)

    if subset == "parent":
        feats_idx = par_idx
    elif subset == "mb":
        feats_idx = mb_idx.copy()
        # simple misspec: drop one MB feature OR add a random non-MB non-T feature
        if mb_misspec == "drop_one" and len(feats_idx) > 0:
            drop = np.random.default_rng(0).choice(feats_idx)
            feats_idx = [i for i in feats_idx if i != drop]
        elif mb_misspec == "add_nonmb":
            others = [i for i in range(len(nodes)) if i not in mb_idx+[idx_t]]
            if others:
                add = np.random.default_rng(0).choice(others)
                feats_idx = sorted(set(feats_idx+[add]))
    else:
        feats_idx = [i for i in range(len(nodes)) if i != idx_t]

    feat_names = [nodes[i] for i in feats_idx]
    X = df[feat_names].values.astype(float)

    # Support stretch (if annotated) — stretch ONLY features in X to simulate mismatch
    stretch = df.attrs.get("support_stretch", 1.0)
    if isinstance(stretch, float) and abs(stretch-1.0) > 1e-12:
        X = X * stretch

    # Random missingness (test-time) → mean impute
    if missing_rate > 0.0:
        rng = np.random.default_rng(123)
        mask = rng.uniform(size=X.shape) < missing_rate
        Xm = X.copy()
        Xm[mask] = np.nan
        # mean-impute column-wise
        col_means = np.nanmean(Xm, axis=0)
        inds = np.where(np.isnan(Xm))
        Xm[inds] = np.take(col_means, inds[1])
        X = Xm

    return X, feat_names

# ---------------------------
# Estimators (thin wrappers over your implementations)
# ---------------------------
def run_MBGIB(Xs, Ts, Xt):
    enc = gibmod.fit_linear_ib_encoder(Xs, Ts, ridge_x=1e-8, ridge_y=1e-8, frac=0.99, d=None)
    Zs = gibmod.encode_Z(enc, Xs)
    B = gibmod.fit_decoder_from_Z(Zs, Ts.reshape(-1,1))
    Zt = gibmod.encode_Z(enc, Xt)
    yhat = gibmod.predict_from_Z(B, Zt, enc["muY"])
    return np.asarray(yhat).ravel()

def run_MBVIB(Xs, Ts, Xt, z_dim=8, beta=1e-2, like="student_t", epochs=400, batch=256, lr=1e-3):
    model = vibmod.train_vib(
        Xs, Ts, in_dim=Xs.shape[1],
        z_dim=z_dim, beta=beta, epochs=epochs, batch_size=batch, lr=lr,
        likelihood=like, early_stop_patience=30
    )
    yhat = vibmod.vib_predict(model, Xt)
    return np.asarray(yhat).ravel()

def run_BN_parents(Xs_parents, Ts, Xt_parents, source_df_full_cov):
    """
    Use BNfit.fitdag on source covariance to get A, then extract β for parents(T) order.
    """
    nodes = NODES
    amat = adjacency_matrix_7()
    parent_list = [[j for j in range(len(nodes)) if amat[j, i] == 1 and j != i]
                   for i in range(len(nodes))]
    S = source_df_full_cov[nodes].cov().values
    dag_fit = bnmod.fitdag(amat=amat, s=S, parent_list=parent_list)
    A = dag_fit["A"]; idxT = IDX["T"]; parents_idx_T = parent_list[idxT]
    beta = np.array([-A[idxT, j] for j in parents_idx_T], dtype=float)
    # order X columns to match parents_idx_T (same as build_feature_matrix(..., subset="parent"))
    return Xt_parents @ beta

def run_PureDNN(Xs, Ts, Xt, epochs=500, batch=256, lr=1e-3):
    return np.asarray(cmpmod.train_pure_dnn(Xs, Ts, Xt, epochs=epochs, batch_size=batch, lr=lr)).ravel()

def run_IIB(Xs, Ts, Xt, env_ids, z_dim=8, beta=1e-2, lambda_adv=1.0, epochs=300, batch=256, lr=1e-3, alpha_grl=1.0):
    return np.asarray(cmpmod.train_IIB_style(
        Xs, Ts, Xt, env_ids, z_dim=z_dim, beta=beta, lambda_adv=lambda_adv,
        epochs=epochs, batch_size=batch, lr=lr, alpha_grl=alpha_grl
    )).ravel()

# ---------------------------
# Experiment runner helpers
# ---------------------------
def evaluate_block(Y_true, preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for k, yhat in preds.items():
        m = metrics(Y_true, yhat); m["method"] = k
        rows.append(m)
    return pd.DataFrame(rows)

def pseudo_env_ids(n: int, k: int = 3, seed=123):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ids = np.zeros(n, dtype=int)
    for e, part in enumerate(np.array_split(idx, k)):
        ids[part] = e
    return ids

# ---------------------------
# Main Results
# ---------------------------
def main_results(args, outdir):
    """
    Compare MB–GIB / MB–VIB vs BN, Pure DNN, IIB under:
    - Covariate shift (C2 shift),
    - Generalized target shift (C1 and eps_T shifts + optional additive constant).
    """
    print("\n== Main Results ==")
    rows = []
    amat = adjacency_matrix_7()

    for seed in range(args.seeds):
        rng = np.random.default_rng(args.base_seed + seed)
        # Source & targets
        df_s   = sample_sem(args.Ns, "source", rng)
        df_cov = sample_sem(args.Nt, "cov",    rng, mu_cov=5, sd_cov=2)
        df_tgt = sample_sem(args.Nt, "tgt",    rng,
                            mu_tgt=args.tgt_mu, sd_tgt=args.tgt_sd,
                            mu_epsT=args.tgt_eps_mu, sd_epsT=args.tgt_eps_sd,
                            add_T_const=args.tgt_add_const)

        # --- Feature matrices (MB scope for primary comparison) ---
        Xs_mb,_ = build_feature_matrix(df_s,   "mb", amat)
        Xt_cov,_= build_feature_matrix(df_cov, "mb", amat)
        Xt_tgt,_= build_feature_matrix(df_tgt, "mb", amat)
        Ts = df_s["T"].values; Tc = df_cov["T"].values; Tt = df_tgt["T"].values

        # Pseudo env ids from source for IIB
        env_ids = pseudo_env_ids(len(Ts), k=3, seed=123)

        # --- Covariate shift ---
        preds_cov = {
            "MB-GIB":   run_MBGIB(Xs_mb, Ts, Xt_cov),
            "MB-VIB":   run_MBVIB(Xs_mb, Ts, Xt_cov, z_dim=args.vib_z, beta=args.vib_beta,
                                   like=args.vib_like, epochs=args.vib_epochs, batch=args.batch, lr=args.lr),
            "PureDNN":  run_PureDNN(Xs_mb, Ts, Xt_cov, epochs=args.dnn_epochs, batch=args.batch, lr=args.lr),
            "IIB-style":run_IIB(Xs_mb, Ts, Xt_cov, env_ids, z_dim=args.iib_z, beta=args.iib_beta,
                                   lambda_adv=args.iib_lambda_adv, epochs=args.iib_epochs,
                                   batch=args.batch, lr=args.lr, alpha_grl=args.iib_alpha_grl)
        }
        # BN uses parents only
        Xs_par,_ = build_feature_matrix(df_s, "parent", amat)
        Xt_cov_par,_= build_feature_matrix(df_cov, "parent", amat)
        preds_cov["BN"] = run_BN_parents(Xs_par, Ts, Xt_cov_par, source_df_full_cov=df_s)

        res_cov = evaluate_block(Tc, preds_cov)
        res_cov["scenario"]="Covariate"; res_cov["seed"]=seed
        rows.append(res_cov)

        # --- Generalized target shift ---
        preds_tgt = {
            "MB-GIB":   run_MBGIB(Xs_mb, Ts, Xt_tgt),
            "MB-VIB":   run_MBVIB(Xs_mb, Ts, Xt_tgt, z_dim=args.vib_z, beta=args.vib_beta,
                                   like=args.vib_like, epochs=args.vib_epochs, batch=args.batch, lr=args.lr),
            "PureDNN":  run_PureDNN(Xs_mb, Ts, Xt_tgt, epochs=args.dnn_epochs, batch=args.batch, lr=args.lr),
            "IIB-style":run_IIB(Xs_mb, Ts, Xt_tgt, env_ids, z_dim=args.iib_z, beta=args.iib_beta,
                                   lambda_adv=args.iib_lambda_adv, epochs=args.iib_epochs,
                                   batch=args.batch, lr=args.lr, alpha_grl=args.iib_alpha_grl)
        }
        Xt_tgt_par,_= build_feature_matrix(df_tgt, "parent", amat)
        preds_tgt["BN"] = run_BN_parents(Xs_par, Ts, Xt_tgt_par, source_df_full_cov=df_s)

        res_tgt = evaluate_block(Tt, preds_tgt)
        res_tgt["scenario"]="Target(gen)"; res_tgt["seed"]=seed
        # annotate the chosen generalized shift params
        res_tgt["tgt_mu"]=args.tgt_mu; res_tgt["tgt_sd"]=args.tgt_sd
        res_tgt["tgt_eps_mu"]=args.tgt_eps_mu; res_tgt["tgt_eps_sd"]=args.tgt_eps_sd
        res_tgt["tgt_add_const"]=args.tgt_add_const
        rows.append(res_tgt)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(outdir, "main_results.csv"), index=False)
    print("Saved: main_results.csv")
    return df

# ---------------------------
# Ablations
# ---------------------------
def ablations(args, outdir):
    """
    Scope: parent/mb/global
    Capacity: GIB d (via frac≈ keep energy) and VIB z_dim
    Beta: VIB beta
    Likelihood: VIB {student_t, laplace}
    MB estimation: misspec (drop_one / add_nonmb)
    """
    print("\n== Ablations ==")
    amat = adjacency_matrix_7()
    rows = []

    for seed in range(args.seeds):
        rng = np.random.default_rng(args.base_seed + seed)
        df_s = sample_sem(args.Ns, "source", rng)
        # use covariate shift as default target here
        df_t = sample_sem(args.Nt, "cov", rng, mu_cov=5, sd_cov=2)
        Ts = df_s["T"].values; Ttrue = df_t["T"].values

        # (A) Scope
        for subset in ["parent","mb","global"]:
            Xs,_ = build_feature_matrix(df_s, subset, amat)
            Xt,_ = build_feature_matrix(df_t, subset, amat)
            res = {}
            res["MB-GIB"] = run_MBGIB(Xs, Ts, Xt)
            res["MB-VIB"] = run_MBVIB(Xs, Ts, Xt, z_dim=args.vib_z, beta=args.vib_beta, like=args.vib_like,
                                      epochs=args.vib_epochs, batch=args.batch, lr=args.lr)
            # BN limited to parent subset
            if subset=="parent":
                Xs_par,_=build_feature_matrix(df_s,"parent",amat)
                Xt_par,_=build_feature_matrix(df_t,"parent",amat)
                res["BN"] = run_BN_parents(Xs_par, Ts, Xt_par, df_s)
            else:
                res["BN"] = np.repeat(np.mean(Ts), len(Ttrue))  # benign no-op baseline when non-parent scope
            res["PureDNN"] = run_PureDNN(Xs, Ts, Xt, epochs=args.dnn_epochs, batch=args.batch, lr=args.lr)
            env_ids = pseudo_env_ids(len(Ts))
            res["IIB-style"] = run_IIB(Xs, Ts, Xt, env_ids,
                                       z_dim=args.iib_z, beta=args.iib_beta, lambda_adv=args.iib_lambda_adv,
                                       epochs=args.iib_epochs, batch=args.batch, lr=args.lr, alpha_grl=args.iib_alpha_grl)
            df_scope = evaluate_block(Ttrue, res); df_scope["abl"]="scope"; df_scope["subset"]=subset; df_scope["seed"]=seed
            rows.append(df_scope)

        # (B) Capacity & beta & likelihood (VIB); capacity (GIB handled implicitly by eig cutoff)
        Xs_mb,_= build_feature_matrix(df_s,"mb",amat)
        Xt_mb,_= build_feature_matrix(df_t,"mb",amat)
        for z_dim in args.grid_zdim:
            for beta in args.grid_beta:
                for like in ["student_t","laplace"]:
                    res = {}
                    res["MB-GIB"] = run_MBGIB(Xs_mb, Ts, Xt_mb)  # fixed
                    res["MB-VIB"] = run_MBVIB(Xs_mb, Ts, Xt_mb, z_dim=z_dim, beta=beta, like=like,
                                              epochs=args.vib_epochs, batch=args.batch, lr=args.lr)
                    df_vib = evaluate_block(Ttrue, res)
                    df_vib["abl"]="capacity_beta_like"; df_vib["z_dim"]=z_dim; df_vib["beta"]=beta; df_vib["like"]=like
                    df_vib["seed"]=seed
                    rows.append(df_vib)

        # (C) MB estimation misspec
        for misspec in ["drop_one","add_nonmb"]:
            Xs_bad,_= build_feature_matrix(df_s,"mb",amat, mb_misspec=misspec)
            Xt_bad,_= build_feature_matrix(df_t,"mb",amat, mb_misspec=misspec)
            res={}
            res["MB-GIB(misspec)"] = run_MBGIB(Xs_bad, Ts, Xt_bad)
            res["MB-VIB(misspec)"] = run_MBVIB(Xs_bad, Ts, Xt_bad, z_dim=args.vib_z, beta=args.vib_beta, like=args.vib_like,
                                               epochs=args.vib_epochs, batch=args.batch, lr=args.lr)
            df_m = evaluate_block(Ttrue, res); df_m["abl"]="mb_misspec"; df_m["misspec"]=misspec; df_m["seed"]=seed
            rows.append(df_m)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(outdir, "ablations.csv"), index=False)
    print("Saved: ablations.csv")
    return df

# ---------------------------
# Sensitivity
# ---------------------------
def parse_grid_str(s: str) -> Dict[str, List[float]]:
    """
    "cov_mu=0,2,5 tgt_mu=0,2,5 tgt_eps_mu=0,3 tgt_eps_sd=1,2 tgt_add_const=0,3"
    → dict with lists (floats/ints).
    """
    if not s: return {}
    out = {}
    for chunk in s.split():
        if "=" not in chunk: continue
        k,v = chunk.split("=",1)
        vals=[]
        for t in v.split(","):
            try: vals.append(int(t)); continue
            except: pass
            try: vals.append(float(t)); continue
            except: pass
            vals.append(t)
        out[k] = vals
    return out

def sensitivity(args, outdir):
    """
    Shift magnitude/types: vary cov_mu (C2), tgt_mu (C1), eps_T (mu, sd), and additive constant.
    Support mismatch: scale MB features at test time
    Missingness: random feature-wise missing at test time
    N_s curves: vary source sample size
    """
    print("\n== Sensitivity ==")
    rows=[]
    amat = adjacency_matrix_7()
    grid = parse_grid_str(args.shift_grid)

    cov_mu_vals  = grid.get("cov_mu",[5])
    cov_sd_vals  = grid.get("cov_sd",[2])
    tgt_mu_vals  = grid.get("tgt_mu",[5])
    tgt_sd_vals  = grid.get("tgt_sd",[2])
    tgt_eps_mu   = grid.get("tgt_eps_mu",[0])
    tgt_eps_sd   = grid.get("tgt_eps_sd",[1])
    tgt_addc     = grid.get("tgt_add_const",[0])   # NEW: sweep additive constant on T
    stretch_vals = grid.get("stretch",[1.0, 1.5, 2.0])
    miss_vals    = grid.get("miss",[0.0, 0.2, 0.5])
    Ns_vals      = [int(v) for v in grid.get("Ns", [args.Ns, max(50, args.Ns//2), args.Ns*2])]

    for seed in range(args.seeds):
        rng = np.random.default_rng(args.base_seed + seed)

        # (A) Shift sweep (covariate vs generalized target)
        for mu_cov in cov_mu_vals:
            for sd_cov in cov_sd_vals:
                for mu_tgt in tgt_mu_vals:
                    for sd_tgt in tgt_sd_vals:
                        for mu_eps in tgt_eps_mu:
                            for sd_eps in tgt_eps_sd:
                                for addc in tgt_addc:
                                    df_s = sample_sem(args.Ns, "source", rng)
                                    df_t_cov = sample_sem(args.Nt, "cov", rng, mu_cov=mu_cov, sd_cov=sd_cov)
                                    df_t_tgt = sample_sem(args.Nt, "tgt", rng,
                                                          mu_tgt=mu_tgt, sd_tgt=sd_tgt,
                                                          mu_epsT=mu_eps, sd_epsT=sd_eps,
                                                          add_T_const=addc)
                                    Ts = df_s["T"].values
                                    for scenario, dft in [("Covariate", df_t_cov), ("Target(gen)", df_t_tgt)]:
                                        Xs_mb,_ = build_feature_matrix(df_s,"mb",amat)
                                        Xt_mb,_ = build_feature_matrix(dft,"mb",amat)
                                        y_MBGIB = run_MBGIB(Xs_mb, Ts, Xt_mb)
                                        y_MBVIB = run_MBVIB(Xs_mb, Ts, Xt_mb, z_dim=args.vib_z, beta=args.vib_beta,
                                                            like=args.vib_like, epochs=args.vib_epochs,
                                                            batch=args.batch, lr=args.lr)
                                        row = metrics(dft["T"].values, y_MBGIB); row["method"]="MB-GIB"
                                        row.update(dict(scenario=scenario, seed=seed,
                                                        cov_mu=mu_cov, cov_sd=sd_cov,
                                                        tgt_mu=mu_tgt, tgt_sd=sd_tgt,
                                                        tgt_eps_mu=mu_eps, tgt_eps_sd=sd_eps,
                                                        tgt_add_const=addc))
                                        rows.append(row)
                                        row = metrics(dft["T"].values, y_MBVIB); row["method"]="MB-VIB"
                                        row.update(dict(scenario=scenario, seed=seed,
                                                        cov_mu=mu_cov, cov_sd=sd_cov,
                                                        tgt_mu=mu_tgt, tgt_sd=sd_tgt,
                                                        tgt_eps_mu=mu_eps, tgt_eps_sd=sd_eps,
                                                        tgt_add_const=addc))
                                        rows.append(row)

        # (B) Support mismatch & missingness (on MB features)
        for stretch in stretch_vals:
            for miss in miss_vals:
                df_s = sample_sem(args.Ns, "source", rng)
                df_t = sample_sem(args.Nt, "cov", rng, mu_cov=5, sd_cov=2, support_stretch=stretch)
                Xs_mb,_ = build_feature_matrix(df_s,"mb",amat)
                Xt_mb,_ = build_feature_matrix(df_t,"mb",amat, missing_rate=miss)
                Ts = df_s["T"].values; Tt = df_t["T"].values
                yG = run_MBGIB(Xs_mb, Ts, Xt_mb)
                yV = run_MBVIB(Xs_mb, Ts, Xt_mb, z_dim=args.vib_z, beta=args.vib_beta,
                               like=args.vib_like, epochs=args.vib_epochs, batch=args.batch, lr=args.lr)
                rg = metrics(Tt, yG); rg.update(dict(method="MB-GIB", scenario="support/missing",
                                                     stretch=stretch, miss=miss, seed=seed)); rows.append(rg)
                rv = metrics(Tt, yV); rv.update(dict(method="MB-VIB", scenario="support/missing",
                                                     stretch=stretch, miss=miss, seed=seed)); rows.append(rv)

        # (C) N_s curves
        for Ns in Ns_vals:
            df_s = sample_sem(Ns, "source", rng)
            df_t = sample_sem(args.Nt, "cov", rng, mu_cov=5, sd_cov=2)
            Xs_mb,_ = build_feature_matrix(df_s,"mb",amat)
            Xt_mb,_ = build_feature_matrix(df_t,"mb",amat)
            Ts = df_s["T"].values; Tt = df_t["T"].values
            yG = run_MBGIB(Xs_mb, Ts, Xt_mb)
            yV = run_MBVIB(Xs_mb, Ts, Xt_mb, z_dim=args.vib_z, beta=args.vib_beta,
                           like=args.vib_like, epochs=args.vib_epochs, batch=args.batch, lr=args.lr)
            rg = metrics(Tt, yG); rg.update(dict(method="MB-GIB", scenario="Ns_curve", Ns=Ns, seed=seed)); rows.append(rg)
            rv = metrics(Tt, yV); rv.update(dict(method="MB-VIB", scenario="Ns_curve", Ns=Ns, seed=seed)); rows.append(rv)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "sensitivity.csv"), index=False)
    print("Saved: sensitivity.csv")
    return df

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Toy SEM ablation & sensitivity for MB–GIB/MB–VIB vs baselines")
    # data sizes
    p.add_argument("--Ns", type=int, default=100, help="Source sample size")
    p.add_argument("--Nt", type=int, default=100, help="Target sample size per scenario")
    p.add_argument("--seeds", type=int, default=10, help="Seeds/repeats")
    p.add_argument("--base-seed", type=int, default=42)
    # VIB
    p.add_argument("--vib-z", type=int, default=8)
    p.add_argument("--vib-beta", type=float, default=1e-2)
    p.add_argument("--vib-like", choices=["student_t", "laplace"], default="student_t")
    p.add_argument("--vib-epochs", type=int, default=400)
    # IIB/PureDNN
    p.add_argument("--iib-z", type=int, default=8)
    p.add_argument("--iib-beta", type=float, default=1e-2)
    p.add_argument("--iib-lambda-adv", type=float, default=1.0)
    p.add_argument("--iib-epochs", type=int, default=300)
    p.add_argument("--iib-alpha-grl", type=float, default=1.0)
    p.add_argument("--dnn-epochs", type=int, default=500)
    # shared opt
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    # ablation grids
    p.add_argument("--grid-zdim", type=str, default="4,8,16")
    p.add_argument("--grid-beta", type=str, default="1e-3,3e-3,1e-2")
    # sensitivity grid (space-separated key=comma,separated,vals)
    # Rich defaults so sensitivity produces curves/heatmaps instead of a single point.
    p.add_argument(
        "--shift-grid", type=str,
        default=(
            "cov_mu=0,2,5 cov_sd=1,2 "
            "tgt_mu=0,5,10 tgt_sd=1,2 "
            "tgt_eps_mu=0,3 tgt_eps_sd=1,2,3 tgt_add_const=0,3,6 "
            "stretch=1,1.5,2 miss=0,0.2,0.5 "
            "Ns=100,300,1000"
        )
    )
    # generalized target shift (main results)
    p.add_argument("--tgt-mu", type=float, default=5.0, help="Target shift: mean of C1 in target")
    p.add_argument("--tgt-sd", type=float, default=2.0, help="Target shift: std of C1 in target")
    p.add_argument("--tgt-eps-mu", type=float, default=0.0, help="Target shift: mean of eps_T")
    p.add_argument("--tgt-eps-sd", type=float, default=1.0, help="Target shift: std of eps_T")
    p.add_argument("--tgt-add-const", type=float, default=3.0, help="Target shift: constant added to T")
    # output
    p.add_argument("--outdir", type=str, default="results")
    return p.parse_args()


def as_floats_list(s: str) -> List[float]:
    vals=[]
    for t in s.split(","):
        try: vals.append(int(t)); continue
        except: pass
        try: vals.append(float(t)); continue
        except: pass
        vals.append(t)
    return vals

def main():
    args = parse_args()
    safe_mkdir(args.outdir)
    args.grid_zdim = [int(v) for v in as_floats_list(args.grid_zdim)]
    args.grid_beta = [float(v) for v in as_floats_list(args.grid_beta)]

    # 1) Main results
    df_main = main_results(args, args.outdir)
    # 2) Ablations
    df_abl  = ablations(args, args.outdir)
    # 3) Sensitivity
    df_sens = sensitivity(args, args.outdir)

    # quick summary
    print("\n=== Summary (means over seeds) ===")
    for fname in ["main_results.csv","ablations.csv","sensitivity.csv"]:
        path = os.path.join(args.outdir, fname)
        df = pd.read_csv(path)
        group_cols = [c for c in ["scenario","abl","subset","method"] if c in df.columns]
        if group_cols:
            print(f"\n{fname}:")
            print(df.groupby(group_cols)[["MAE","RMSE","R2"]].mean().round(4))

if __name__ == "__main__":
    main()
