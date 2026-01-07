#!/usr/bin/env python3
# compare_methods.py
# Compare VIB vs IRM vs IIB-style (DANN-IB) vs Pure DNN for causal domain adaptation (imputation of T)
#
# Inputs (override with CLI flags):
#   --adj       adjacency_matrix.csv         (p x p, node names as index/columns; includes "T")
#   --src       source_1.csv                 (complete source with "T"; optional column 'ENV' for environments)
#   --tgt-miss  tgt_target_missing_1.csv     (target without "T")
#   --tgt-true  tgt_target_true_1.csv        (target with "T" for evaluation only)
#
# Choose feature subset with --subset {parent, mb, global}
#
# Outputs:
#   - *_scatter.pdf for each method
#   - compare_metrics.csv (MAE, RMSE, R2 for each method)

import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------
# Repro & device
# -----------------------
def set_seed(seed: int = 123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utilities
# -----------------------
class Standardizer:
    """Fit on source; apply to target."""
    def __init__(self): self.mean_ = None; self.std_ = None
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0, ddof=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self
    def transform(self, X): return (np.asarray(X, float) - self.mean_) / self.std_

def markov_blanket_indices(amat: np.ndarray, idx_t: int):
    """MB(T) = pa(T) ∪ ch(T) ∪ pa(ch(T)) \ {T}; amat[j,i]=1 if j->i"""
    p = amat.shape[0]
    parents_T  = {j for j in range(p) if amat[j, idx_t] != 0}
    children_T = {i for i in range(p) if amat[idx_t, i] != 0}
    spouses = set()
    for c in children_T:
        for j in range(p):
            if amat[j, c] != 0 and j != idx_t:
                spouses.add(j)
    mb = (parents_T | children_T | spouses) - {idx_t}
    return sorted(mb), sorted(parents_T), sorted(children_T), sorted(spouses)

def load_data(adj_path, src_path, tgt_miss_path, tgt_true_path, subset: str):
    amat_df = pd.read_csv(adj_path, index_col=0)
    nodes = list(amat_df.index)
    if "T" not in nodes:
        raise ValueError("Node 'T' not found in adjacency_matrix.csv")
    source_df = pd.read_csv(src_path)
    tgt_obs_df = pd.read_csv(tgt_miss_path)
    tgt_true_df = pd.read_csv(tgt_true_path)

    # Keep only known nodes order
    source_df = source_df[nodes]
    tgt_obs_df = tgt_obs_df[nodes]  # T will be missing; we'll slice features

    p = len(nodes); amat = amat_df.values.astype(int)
    idx_t = nodes.index("T")

    # Feature subsets
    parent_feats = [j for j in range(p) if amat[j, idx_t] != 0]
    mb_feats, _, _, _ = markov_blanket_indices(amat, idx_t)
    global_feats = [k for k in range(p) if k != idx_t]

    if subset == "parent":
        chosen = parent_feats; name = f"Parents of T ({len(parent_feats)})"
    elif subset == "mb":
        chosen = mb_feats; name = f"Markov blanket of T ({len(mb_feats)})"
    else:
        chosen = global_feats; name = f"Global features ({len(global_feats)})"

    if len(chosen) == 0:
        raise ValueError(f"Chosen subset '{subset}' is empty. Check your DAG and subset choice.")

    Xs = source_df.iloc[:, chosen].values.astype(float)
    Ts = source_df["T"].values.astype(float)
    Xt = tgt_obs_df.iloc[:, chosen].values.astype(float)
    Tt_true = tgt_true_df["T"].values.astype(float)

    # Optional source environments for IRM/IIB-style (use column 'ENV' if present, else pseudo-envs)
    env = source_df.columns.to_list()
    env_col = "ENV" if "ENV" in source_df.columns else None  # (kept for clarity; using nodes-only frame)
    if "ENV" in source_df.columns:
        env_ids = source_df["ENV"].values
    else:
        # Make pseudo environments by hashing indices (stable, reproducible)
        n = len(Ts)
        rng = np.random.default_rng(123)
        perm = rng.permutation(n)
        k = 3  # 3 pseudo-envs by default
        env_ids = np.zeros(n, dtype=int)
        splits = np.array_split(perm, k)
        for i, idxs in enumerate(splits): env_ids[idxs] = i

    return {
        "Xs": Xs, "Ts": Ts, "Xt": Xt, "Tt_true": Tt_true,
        "subset_name": name, "subset": subset, "nodes": nodes,
        "env_ids": env_ids
    }

def evaluate_and_plot(y_true, y_pred, title, pdf_path):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{title}: MAE={mae:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}")
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
    plt.xlabel("True T", fontsize=14); plt.ylabel("Predicted T", fontsize=14)
    plt.title(title, fontsize=12)
    plt.tight_layout(); plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# -----------------------
# Common MLP blocks
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden=(128,64), activation=nn.ReLU):
        super().__init__()
        layers = []; d = in_dim
        for h in hidden: layers += [nn.Linear(d, h), activation()]; d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

# -----------------------
# VIB (encoder + predictor)
# -----------------------
class VIB_Encoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden=(128,64), activation=nn.ReLU):
        super().__init__()
        layers = []; d = in_dim
        for h in hidden: layers += [nn.Linear(d, h), activation()]; d = h
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(d, z_dim)
        self.logvar = nn.Linear(d, z_dim)
    def forward(self, x):
        h = self.backbone(x); mu = self.mu(h); logvar = self.logvar(h)
        return mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar); eps = torch.randn_like(std)
    return mu + eps*std

def kl_standard_normal(mu, logvar):
    return 0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - 1.0 - logvar, dim=1)

def train_VIB(Xs, Ts, Xt, z_dim=8, beta=1e-2, epochs=300, batch_size=256, lr=1e-3, fixed_sigma2=1.0):
    scaler = Standardizer().fit(Xs)
    Xsz = scaler.transform(Xs); Xtz = scaler.transform(Xt)
    Xst = torch.tensor(Xsz, dtype=torch.float32); Tst = torch.tensor(Ts, dtype=torch.float32)
    ds = TensorDataset(Xst, Tst); dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    in_dim = Xs.shape[1]
    enc = VIB_Encoder(in_dim, z_dim).to(DEVICE)
    pred = MLP(z_dim, out_dim=1, hidden=(64,32)).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters())+list(pred.parameters()), lr=lr)

    best = (1e18, None, None)
    for ep in range(1, epochs+1):
        enc.train(); pred.train(); losses = []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            mu, logvar = enc(xb)
            z = reparameterize(mu, logvar)
            yhat = pred(z)
            mse = torch.mean((yhat - yb)**2)
            nll = 0.5/fixed_sigma2 * mse
            kl = torch.mean(kl_standard_normal(mu, logvar))
            loss = nll + beta*kl
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        cur = float(np.mean(losses))
        if cur < best[0]: best = (cur, enc.state_dict(), pred.state_dict())
    enc.load_state_dict(best[1]); pred.load_state_dict(best[2])
    enc.eval(); pred.eval()

    with torch.no_grad():
        xt = torch.tensor(Xtz, dtype=torch.float32, device=DEVICE)
        mu, logvar = enc(xt); yhat = pred(mu).cpu().numpy()
    return yhat

# -----------------------
# IRM for regression
# -----------------------
class IRM_Head(nn.Module):
    def __init__(self, in_dim, hidden=(128,64), activation=nn.ReLU):
        super().__init__()
        self.f = MLP(in_dim, out_dim=1, hidden=hidden, activation=activation)
    def forward(self, x): return self.f(x).squeeze(-1)

def irm_penalty(env_losses: List[torch.Tensor], env_outputs: List[torch.Tensor], env_targets: List[torch.Tensor]):
    """
    Arjovsky et al. IRM penalty for regression via dummy scalar w:
    sum_e || d/dw R_e(w * f(x)) |_{w=1} ||^2
    Where R_e is MSE on environment e.
    """
    penalty = 0.0
    for yhat_e, y_e in zip(env_outputs, env_targets):
        scale = torch.tensor(1.0, requires_grad=True, device=yhat_e.device)
        loss_e = torch.mean((y_e - yhat_e*scale)**2)
        grad = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
        penalty = penalty + grad.pow(2)
    return penalty

def train_IRM(Xs, Ts, Xt, env_ids, lambda_irm=1000.0, epochs=1000, batch_size=1024, lr=1e-3):
    scaler = Standardizer().fit(Xs)
    Xsz = scaler.transform(Xs); Xtz = scaler.transform(Xt)
    Xst = torch.tensor(Xsz, dtype=torch.float32); Tst = torch.tensor(Ts, dtype=torch.float32)
    env_ids = np.asarray(env_ids, int)
    in_dim = Xs.shape[1]
    head = IRM_Head(in_dim).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    # Build per-env loaders
    envs = np.unique(env_ids)
    env_tensors = []
    for e in envs:
        idx = np.where(env_ids==e)[0]
        ds = TensorDataset(Xst[idx], Tst[idx]); dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        env_tensors.append(dl)

    best = (1e18, None)
    for ep in range(1, epochs+1):
        head.train()
        # One pass per env minibatch
        total = 0.0
        for dl in env_tensors:
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                yhat = head(xb)
                erm = torch.mean((yhat - yb)**2)
                # For penalty compute on full mini-batch; reuse outputs
                penalty = irm_penalty([erm], [yhat], [yb])  # slight approximation per mini-batch
                loss = erm + lambda_irm * penalty
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss)
        if total < best[0]: best = (total, head.state_dict())

    head.load_state_dict(best[1]); head.eval()
    with torch.no_grad():
        xt = torch.tensor(Xtz, dtype=torch.float32, device=DEVICE)
        yhat = head(xt.to(DEVICE)).cpu().numpy()
    return yhat

# -----------------------
# IIB-style (DANN-IB): IB encoder + task head + domain classifier with GRL
# -----------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return -ctx.alpha * grad_output, None
def grad_reverse(x, alpha=1.0): return GradReverse.apply(x, alpha)

class DomainDisc(nn.Module):
    def __init__(self, z_dim, hidden=(64,32), n_domains=3, activation=nn.ReLU):
        super().__init__()
        layers=[]; d=z_dim
        for h in hidden: layers += [nn.Linear(d,h), activation()]; d=h
        layers += [nn.Linear(d, n_domains)]
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

def train_IIB_style(Xs, Ts, Xt, env_ids, z_dim=8, beta=1e-2, lambda_adv=1.0,
                    epochs=300, batch_size=256, lr=1e-3, fixed_sigma2=1.0, alpha_grl=1.0):
    scaler = Standardizer().fit(Xs)
    Xsz = scaler.transform(Xs); Xtz = scaler.transform(Xt)
    Xst = torch.tensor(Xsz, dtype=torch.float32); Tst = torch.tensor(Ts, dtype=torch.float32)
    env_ids = np.asarray(env_ids, int); env_t = torch.tensor(env_ids, dtype=torch.long)
    ds = TensorDataset(Xst, Tst, env_t); dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    in_dim = Xs.shape[1]; n_domains = int(env_ids.max()) + 1
    enc = VIB_Encoder(in_dim, z_dim).to(DEVICE)
    pred = MLP(z_dim, out_dim=1, hidden=(64,32)).to(DEVICE)
    disc = DomainDisc(z_dim, n_domains=n_domains).to(DEVICE)

    opt = torch.optim.Adam(list(enc.parameters())+list(pred.parameters())+list(disc.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    best = (1e18, None, None, None)
    for ep in range(1, epochs+1):
        enc.train(); pred.train(); disc.train()
        losses=[]
        for xb, yb, db in dl:
            xb, yb, db = xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            mu, logvar = enc(xb)
            z = reparameterize(mu, logvar)
            # task prediction
            yhat = pred(z)
            mse = torch.mean((yhat - yb)**2)
            nll = 0.5/fixed_sigma2 * mse
            kl = torch.mean(kl_standard_normal(mu, logvar))
            # domain adversary with gradient reversal on mu (cleaner than sampling)
            z_det = grad_reverse(mu, alpha=alpha_grl)
            dlogits = disc(z_det)
            dloss = ce(dlogits, db)
            loss = nll + beta*kl + lambda_adv*dloss
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        cur = float(np.mean(losses))
        if cur < best[0]: best = (cur, enc.state_dict(), pred.state_dict(), scaler)

    enc.load_state_dict(best[1]); pred.load_state_dict(best[2]); scaler = best[3]
    enc.eval(); pred.eval()
    with torch.no_grad():
        xt = torch.tensor(Xtz, dtype=torch.float32, device=DEVICE)
        mu, logvar = enc(xt); yhat = pred(mu).cpu().numpy()
    return yhat

# -----------------------
# Pure Deep NN (no IB, no invariance)
# -----------------------
def train_pure_dnn(Xs, Ts, Xt, epochs=500, batch_size=256, lr=1e-3):
    scaler = Standardizer().fit(Xs)
    Xsz = scaler.transform(Xs); Xtz = scaler.transform(Xt)
    Xst = torch.tensor(Xsz, dtype=torch.float32); Tst = torch.tensor(Ts, dtype=torch.float32)
    ds = TensorDataset(Xst, Tst); dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    in_dim = Xs.shape[1]; net = MLP(in_dim, out_dim=1, hidden=(256,128,64)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = (1e18, None)
    for ep in range(1, epochs+1):
        net.train(); losses=[]
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yhat = net(xb)
            loss = torch.mean((yhat - yb)**2)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        cur = float(np.mean(losses))
        if cur < best[0]: best = (cur, net.state_dict())
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        xt = torch.tensor(Xtz, dtype=torch.float32, device=DEVICE)
        yhat = net(xt.to(DEVICE)).cpu().numpy()
    return yhat

# -----------------------
# Runner
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Compare VIB vs IRM vs IIB-style vs Pure DNN for target imputation")
    ap.add_argument("--adj", default="adjacency_matrix.csv")
    ap.add_argument("--src", default="source_1.csv")
    ap.add_argument("--tgt-miss", default="tgt_target_missing_1.csv")
    ap.add_argument("--tgt-true", default="tgt_target_true_1.csv")
    ap.add_argument("--subset", choices=["parent","mb","global"], default="global",
                    help="Feature subset based on DAG")
    ap.add_argument("--seed", type=int, default=123)

    # VIB
    ap.add_argument("--vib", action="store_true")
    ap.add_argument("--vib-z", type=int, default=8)
    ap.add_argument("--vib-beta", type=float, default=1e-2)
    ap.add_argument("--vib-epochs", type=int, default=300)

    # IRM
    ap.add_argument("--irm", action="store_true")
    ap.add_argument("--irm-lambda", type=float, default=1000.0)
    ap.add_argument("--irm-epochs", type=int, default=1000)

    # IIB-style (DANN-IB)
    ap.add_argument("--iib", action="store_true")
    ap.add_argument("--iib-z", type=int, default=8)
    ap.add_argument("--iib-beta", type=float, default=1e-2)
    ap.add_argument("--iib-lambda-adv", type=float, default=1.0)
    ap.add_argument("--iib-epochs", type=int, default=300)
    ap.add_argument("--iib-alpha-grl", type=float, default=1.0)

    # Pure DNN
    ap.add_argument("--dnn", action="store_true")
    ap.add_argument("--dnn-epochs", type=int, default=500)

    # Shared train config
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)

    args = ap.parse_args()
    set_seed(args.seed)

    data = load_data(args.adj, args.src, args.tgt_miss, args.tgt_true, subset=args.subset)
    Xs, Ts, Xt, Tt = data["Xs"], data["Ts"], data["Xt"], data["Tt_true"]
    env_ids = data["env_ids"]

    print(f"Subset: {data['subset_name']}")
    results = []

    # If no flags given, run all four
    run_all = (not args.vib and not args.irm and not args.iib and not args.dnn)
    if args.vib or run_all:
        yhat = train_VIB(Xs, Ts, Xt,
                         z_dim=args.vib_z, beta=args.vib_beta,
                         epochs=args.vib_epochs, batch_size=args.batch_size, lr=args.lr)
        r = evaluate_and_plot(Tt, yhat, f"VIB (z={args.vib_z}, β={args.vib_beta})", "VIB_scatter_G.pdf")
        r["model"]="VIB"; results.append(r)

    if args.irm or run_all:
        yhat = train_IRM(Xs, Ts, Xt, env_ids,
                         lambda_irm=args.irm_lambda,
                         epochs=args.irm_epochs, batch_size=max(256,args.batch_size), lr=args.lr)
        r = evaluate_and_plot(Tt, yhat, f"IRM (λ={args.irm_lambda})", "IRM_scatter.pdf")
        r["model"]="IRM"; results.append(r)

    if args.iib or run_all:
        yhat = train_IIB_style(Xs, Ts, Xt, env_ids,
                               z_dim=args.iib_z, beta=args.iib_beta,
                               lambda_adv=args.iib_lambda_adv,
                               epochs=args.iib_epochs, batch_size=args.batch_size, lr=args.lr,
                               alpha_grl=args.iib_alpha_grl)
        r = evaluate_and_plot(Tt, yhat, f"IIB-style (z={args.iib_z}, β={args.iib_beta}, λ_adv={args.iib_lambda_adv})", "IIB_style_scatter.pdf")
        r["model"]="IIB-style"; results.append(r)

    if args.dnn or run_all:
        yhat = train_pure_dnn(Xs, Ts, Xt,
                              epochs=args.dnn_epochs, batch_size=args.batch_size, lr=args.lr)
        r = evaluate_and_plot(Tt, yhat, f"Pure DNN", "DNN_scatter.pdf")
        r["model"]="PureDNN"; results.append(r)

    df = pd.DataFrame(results, columns=["model","MAE","RMSE","R2"])
    print("\n=== Comparison Summary ===")
    print(df.to_string(index=False))
    df.to_csv("compare_metrics.csv", index=False)
    print("\nSaved: compare_metrics.csv and scatter PDFs for each method.")

if __name__ == "__main__":
    main()
