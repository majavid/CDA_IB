# vib_dag_latent_ib_nongauss.py
# DAG-aware latent IB with NON-LINEAR encoder/decoder and NON-GAUSSIAN likelihoods.
# Likelihoods supported: "laplace", "student_t" (regression), and auto "bernoulli" (binary Y).
#
# Files expected (same schema as your existing pipeline):
#   - adjacency_matrix.csv
#   - source_1.csv
#   - cov_target_missing_1.csv
#   - cov_target_true_1.csv
#
# Usage: python vib_dag_latent_ib_nongauss.py
#   (edit config in __main__ to pick likelihood, β, z-dim, etc.)

import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -----------------------
# Repro & device
# -----------------------
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utilities (scaler + DAG)
# -----------------------
class Standardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0, ddof=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self
    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.std_
    def inverse_transform(self, Xs):
        return np.asarray(Xs, float) * self.std_ + self.mean_

def markov_blanket_indices(amat: np.ndarray, idx_t: int):
    """
    Markov blanket of T: parents(T) ∪ children(T) ∪ parents(children(T)) \ {T}
    Convention: amat[j, i] = 1 if j -> i
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

def load_common_inputs(target_var="T"):
    # Keep column order consistent with adjacency
    amat_df = pd.read_csv("adjacency_matrix.csv", index_col=0)
    nodes = list(amat_df.index)
    if "T" not in nodes:
        raise ValueError("Node 'T' must exist in adjacency_matrix.csv (used to derive Parents/MB).")
    if target_var not in nodes:
        raise ValueError(f"target_var {target_var!r} not found among nodes.")
    source_df = pd.read_csv("source_1.csv")[nodes]
    target_obs_df = pd.read_csv("cov_target_missing_1.csv")[nodes]
    target_true_df = pd.read_csv("cov_target_true_1.csv")  # eval only
    if target_var not in target_true_df.columns:
        raise ValueError(f"target_var {target_var!r} not found in cov_target_true_1.csv")

    p = len(nodes)
    amat = amat_df.values.astype(int)

    idx_t = nodes.index("T")
    idx_y = nodes.index(target_var)

    # Parents / MB feature sets are built WRT node T (graph location you care about)
    parent_feats = [j for j in range(p) if amat[j, idx_t] != 0]
    parent_names = [nodes[j] for j in parent_feats]
    mb_feats, parT, chT, spT = markov_blanket_indices(amat, idx_t)
    mb_names = [nodes[j] for j in mb_feats]
    global_feats = [k for k in range(p) if k != idx_y]  # avoid target leakage

    # Split features/targets
    Xs_global = source_df.iloc[:, global_feats].values
    Xs_parent = source_df.iloc[:, parent_feats].values if len(parent_feats) > 0 else None
    Xs_mb     = source_df.iloc[:, mb_feats].values if len(mb_feats) > 0 else None
    Ys = source_df[target_var].values.astype(float)

    Xt_global = target_obs_df.iloc[:, global_feats].values
    Xt_parent = target_obs_df.iloc[:, parent_feats].values if len(parent_feats) > 0 else None
    Xt_mb     = target_obs_df.iloc[:, mb_feats].values if len(mb_feats) > 0 else None
    Y_true = target_true_df[target_var].values.astype(float)

    return {
        "nodes": nodes,
        "idx_t": idx_t,
        "idx_y": idx_y,
        "parent_feats": parent_feats,
        "parent_names": parent_names,
        "mb_feats": mb_feats,
        "mb_names": mb_names,
        "Xs_global": Xs_global,
        "Xs_parent": Xs_parent,
        "Xs_mb": Xs_mb,
        "Xt_global": Xt_global,
        "Xt_parent": Xt_parent,
        "Xt_mb": Xt_mb,
        "Ys": Ys,
        "Y_true": Y_true,
        "target_var": target_var
    }

# -----------------------
# Encoder & Decoders
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden=(256,128), activation=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), activation()]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(d, z_dim)
        self.logvar = nn.Linear(d, z_dim)
    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

# Laplace decoder: outputs (mu, log_b), with b>0
class LaplaceDecoder(nn.Module):
    def __init__(self, z_dim, hidden=(128,64), activation=nn.ReLU):
        super().__init__()
        layers = []
        d = z_dim
        for h in hidden:
            layers += [nn.Linear(d, h), activation()]
            d = h
        self.core = nn.Sequential(*layers)
        self.mu = nn.Linear(d, 1)
        self.log_b = nn.Linear(d, 1)
    def forward(self, z):
        h = self.core(z)
        mu = self.mu(h).squeeze(-1)
        log_b = self.log_b(h).squeeze(-1)
        return mu, log_b

# Student-t decoder: outputs (mu, log_s, log_nu), with s>0, nu>2
class StudentTDecoder(nn.Module):
    def __init__(self, z_dim, hidden=(128,64), activation=nn.ReLU):
        super().__init__()
        layers = []
        d = z_dim
        for h in hidden:
            layers += [nn.Linear(d, h), activation()]
            d = h
        self.core = nn.Sequential(*layers)
        self.mu = nn.Linear(d, 1)
        self.log_s = nn.Linear(d, 1)
        self.log_nu = nn.Linear(d, 1)
    def forward(self, z):
        h = self.core(z)
        mu = self.mu(h).squeeze(-1)
        log_s = self.log_s(h).squeeze(-1)
        log_nu = self.log_nu(h).squeeze(-1)
        return mu, log_s, log_nu

# Bernoulli decoder (classification)
class BernoulliDecoder(nn.Module):
    def __init__(self, z_dim, hidden=(128,64), activation=nn.ReLU):
        super().__init__()
        layers = []
        d = z_dim
        for h in hidden:
            layers += [nn.Linear(d, h), activation()]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))
    def forward(self, z):
        logits = self.net(z).squeeze(-1)
        return logits

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_standard_normal(mu, logvar):
    # KL( N(mu, diag(exp(logvar))) || N(0, I) )
    return 0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - 1.0 - logvar, dim=1)

# -----------------------
# Likelihoods (NLLs)
# -----------------------
def nll_laplace(y, mu, log_b):
    # Laplace(y|mu,b): loglik = -|y-mu|/b - log(2b)
    b = torch.nn.functional.softplus(log_b) + 1e-6
    abs_res = torch.abs(y - mu)
    return (abs_res / b + torch.log(2*b))


import torch.nn.functional as F

def nll_student_t(y, mu, log_s, log_nu):
    """
    Per-sample negative log-likelihood for Student-t:
      y ~ StudentT(df=nu, loc=mu, scale=s)
    Returns a tensor of shape (batch,).
    """
    # Ensure positive scale and nu > 2
    s  = F.softplus(log_s) + 1e-6               # (batch,)
    nu = 2.0 + F.softplus(log_nu)               # (batch,)

    t = (y - mu) / s                            # (batch,)

    # a = nu/2, b = 1/2 as T distribution constants
    a = 0.5 * nu                                # (batch,)
    b = torch.full_like(nu, 0.5)                # tensor, not float!

    # log Beta(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)
    log_beta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    # 0.5*log(nu*pi) + log s
    const = 0.5 * torch.log(nu * torch.pi) + torch.log(s)

    # (nu+1)/2 * log(1 + t^2 / nu)
    quad = 0.5 * (nu + 1.0) * torch.log1p((t * t) / nu)

    return log_beta + const + quad


def nll_bernoulli(y, logits):
    # BCE with logits
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")

# -----------------------
# Training / Eval
# -----------------------
def train_vib(
    Xs, Ys, in_dim,
    z_dim=8,
    beta=1e-2,
    batch_size=256,
    epochs=300,
    lr=1e-3,
    likelihood="student_t",   # "laplace" | "student_t" | "auto"
    early_stop_patience=30,
):
    # Detect binary target for Bernoulli if all y in {0,1}
    is_binary = np.all(np.isin(np.unique(Ys), [0.0, 1.0]))
    like = "bernoulli" if is_binary else likelihood

    x_scaler = Standardizer().fit(Xs)
    Xs_z = x_scaler.transform(Xs)

    Xs_t = torch.tensor(Xs_z, dtype=torch.float32)
    Ys_t = torch.tensor(Ys, dtype=torch.float32)

    ds = TensorDataset(Xs_t, Ys_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    enc = Encoder(in_dim=in_dim, z_dim=z_dim).to(DEVICE)

    if like == "laplace":
        dec = LaplaceDecoder(z_dim=z_dim).to(DEVICE)
    elif like == "student_t":
        dec = StudentTDecoder(z_dim=z_dim).to(DEVICE)
    elif like == "bernoulli":
        dec = BernoulliDecoder(z_dim=z_dim).to(DEVICE)
    else:
        raise ValueError(f"Unknown likelihood: {likelihood!r}")

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)

    best_loss = float("inf")
    best = None
    patience = 0

    for epoch in range(1, epochs + 1):
        enc.train(); dec.train()
        tot_loss = 0.0; nobs = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)

            mu, logvar = enc(xb)
            z = reparameterize(mu, logvar)

            if like == "laplace":
                pred_mu, pred_logb = dec(z)
                nll = nll_laplace(yb, pred_mu, pred_logb).mean()
            elif like == "student_t":
                pred_mu, pred_logs, pred_lnu = dec(z)
                nll = nll_student_t(yb, pred_mu, pred_logs, pred_lnu).mean()
            else:  # bernoulli
                logits = dec(z)
                nll = nll_bernoulli(yb, logits).mean()

            kl = kl_standard_normal(mu, logvar).mean()
            loss = nll + beta * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = xb.size(0)
            tot_loss += loss.item() * bs
            nobs += bs

        avg_loss = tot_loss / max(1, nobs)

        improved = avg_loss + 1e-9 < best_loss
        if improved:
            best_loss = avg_loss
            best = {
                "enc": enc.state_dict(),
                "dec": dec.state_dict(),
                "x_mean": x_scaler.mean_.copy(),
                "x_std": x_scaler.std_.copy(),
                "like": like
            }
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                break

    # restore best
    if best is not None:
        enc.load_state_dict(best["enc"])
        dec.load_state_dict(best["dec"])
        x_scaler.mean_ = best["x_mean"]; x_scaler.std_ = best["x_std"]

    enc.eval(); dec.eval()
    return {"enc": enc, "dec": dec, "x_scaler": x_scaler, "z_dim": z_dim, "beta": beta, "likelihood": like}

@torch.no_grad()
def vib_predict(model, X):
    x_scaler = model["x_scaler"]
    enc, dec = model["enc"], model["dec"]
    Xz = x_scaler.transform(X)
    xt = torch.tensor(Xz, dtype=torch.float32, device=DEVICE)
    mu, logvar = enc(xt)
    z = mu  # mean of q(z|x) for deterministic prediction

    like = model["likelihood"]
    if like == "laplace":
        pred_mu, pred_logb = dec(z)
        yhat = pred_mu
    elif like == "student_t":
        pred_mu, pred_logs, pred_lnu = dec(z)
        yhat = pred_mu
    else:  # bernoulli
        logits = dec(z)
        yhat = torch.sigmoid(logits)

    return yhat.detach().cpu().numpy().ravel()

def evaluate_and_plot(regression, y_true, y_pred, title, pdf_path, ylabel):
    if regression:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        print(f"{title}: MAE={mae:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}")
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        mn, mx = float(np.min(y_true)), float(np.max(y_true))
        plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
        plt.xlabel("True", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close()
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
    else:
        # simple classification metrics (threshold 0.5)
        ybin = (y_pred >= 0.5).astype(float)
        acc = float(np.mean(ybin == y_true))
        print(f"{title}: Accuracy={acc:.4f}")
        return {"ACC": acc}

# -----------------------
# Parent / MB / Global runners
# -----------------------
def run_parent_mb_global(
    target_var="T",
    z_dim_parent=4,
    z_dim_mb=8,
    z_dim_global=8,
    beta_parent=1e-2,
    beta_mb=7e-3,
    beta_global=5e-3,
    epochs=400,
    batch_size=256,
    lr=1e-3,
    likelihood="student_t"   # "laplace" | "student_t" | "auto"
):
    data = load_common_inputs(target_var=target_var)

    parent_feats = data["parent_feats"]; parent_names = data["parent_names"]
    mb_feats = data["mb_feats"]; mb_names = data["mb_names"]

    Xs_global, Xt_global = data["Xs_global"], data["Xt_global"]
    Xs_parent, Xt_parent = data["Xs_parent"], data["Xt_parent"]
    Xs_mb, Xt_mb         = data["Xs_mb"], data["Xt_mb"]
    Ys, Y_true = data["Ys"], data["Y_true"]

    # regression vs classification
    is_binary = np.all(np.isin(np.unique(Ys), [0.0, 1.0]))
    regression = not is_binary
    like_display = "Bernoulli" if is_binary else likelihood

    results = []

    # ---- Parent ----
    if Xs_parent is None or Xs_parent.shape[1] == 0:
        print("No parents of T → Parent model = mean (regression) / majority (classification).")
        if regression:
            parent_pred = np.repeat(np.mean(Ys), len(Y_true))
        else:
            maj = float(np.round(np.mean(Ys)))  # majority class (0/1)
            parent_pred = np.repeat(maj, len(Y_true))
        r_parent = evaluate_and_plot(regression, Y_true, parent_pred,
                                     f"Parent VIB (no parents → baseline)",
                                     f"VIB_{target_var}_parent.pdf",
                                     ylabel=f"Pred {target_var}")
        r_parent["model"] = "Parent-VIB"
        results.append(r_parent)
    else:
        in_dim_parent = Xs_parent.shape[1]
        print(f"Parents of T ({in_dim_parent}): {parent_names}")
        model_parent = train_vib(
            Xs_parent, Ys, in_dim=in_dim_parent,
            z_dim=z_dim_parent, beta=beta_parent,
            batch_size=batch_size, epochs=epochs, lr=lr,
            likelihood=likelihood
        )
        parent_pred = vib_predict(model_parent, Xt_parent)
        r_parent = evaluate_and_plot(regression, Y_true, parent_pred,
                                     f"Parent VIB [{like_display}] (z={z_dim_parent}, β={beta_parent})",
                                     f"VIB_{target_var}_parent.pdf",
                                     ylabel=f"Pred {target_var}")
        r_parent["model"] = "Parent-VIB"
        results.append(r_parent)

    # ---- MB ----
    if Xs_mb is None or Xs_mb.shape[1] == 0:
        print("Empty Markov blanket → MB model baseline.")
        if regression:
            mb_pred = np.repeat(np.mean(Ys), len(Y_true))
        else:
            maj = float(np.round(np.mean(Ys)))
            mb_pred = np.repeat(maj, len(Y_true))
        r_mb = evaluate_and_plot(regression, Y_true, mb_pred,
                                 "MB VIB (empty MB → baseline)",
                                 f"VIB_{target_var}_mb.pdf",
                                 ylabel=f"Pred {target_var}")
        r_mb["model"] = "MB-VIB"
        results.append(r_mb)
    else:
        in_dim_mb = Xs_mb.shape[1]
        print(f"Markov blanket of T ({in_dim_mb}): {mb_names}")
        model_mb = train_vib(
            Xs_mb, Ys, in_dim=in_dim_mb,
            z_dim=z_dim_mb, beta=beta_mb,
            batch_size=batch_size, epochs=epochs, lr=lr,
            likelihood=likelihood
        )
        mb_pred = vib_predict(model_mb, Xt_mb)
        r_mb = evaluate_and_plot(regression, Y_true, mb_pred,
                                 f"MB VIB [{like_display}] (z={z_dim_mb}, β={beta_mb})",
                                 f"VIB_{target_var}_mb.pdf",
                                 ylabel=f"Pred {target_var}")
        r_mb["model"] = "MB-VIB"
        results.append(r_mb)

    # ---- Global ----
    in_dim_global = Xs_global.shape[1]
    print(f"Global features: {in_dim_global} (all variables except {target_var})")
    model_global = train_vib(
        Xs_global, Ys, in_dim=in_dim_global,
        z_dim=z_dim_global, beta=beta_global,
        batch_size=batch_size, epochs=epochs, lr=lr,
        likelihood=likelihood
    )
    global_pred = vib_predict(model_global, Xt_global)
    r_global = evaluate_and_plot(regression, Y_true, global_pred,
                                 f"Global VIB [{like_display}] (z={z_dim_global}, β={beta_global})",
                                 f"VIB_{target_var}_global.pdf",
                                 ylabel=f"Pred {target_var}")
    r_global["model"] = "Global-VIB"
    results.append(r_global)

    # ---- Save metrics ----
    cols = ["model", "MAE", "RMSE", "R2"] if regression else ["model", "ACC"]
    res_df = pd.DataFrame(results, columns=cols)
    print("\n=== DAG-aware Non-Gaussian VIB Summary ===")
    print(res_df.to_string(index=False))
    res_df.to_csv(f"VIB_{target_var}_results.csv", index=False)
    print(f"\nSaved plots: VIB_{target_var}_parent.pdf / _mb.pdf / _global.pdf, and VIB_{target_var}_results.csv")

if __name__ == "__main__":
    run_parent_mb_global(
        target_var="T",          # <- you can switch this to any node name
        z_dim_parent=4,
        z_dim_mb=8,
        z_dim_global=8,
        beta_parent=1e-2,
        beta_mb=7e-3,
        beta_global=5e-3,
        epochs=400,
        batch_size=256,
        lr=1e-3,
        likelihood="student_t"   # "laplace" | "student_t" ; auto Bernoulli if labels are {0,1}
    )
