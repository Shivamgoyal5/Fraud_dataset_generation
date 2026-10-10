#!/usr/bin/env python3
"""
repro_sampler_bug.py - reproduces the v1 SimpleTabDDPM failure (KS ~0.7-0.9, DCR ratio >> 1)
on an IEEE-shaped toy cell, isolates the cause, and shows the fix.

  python repro_sampler_bug.py             # ~2 min on a laptop CPU; writes sampler_divergence.png

Cause: the cosine schedule's last beta is clipped at 0.9999, so alpha_T = 1e-4 and the first
reverse step multiplies the noise-prediction error by 1/sqrt(alpha_T) = 100. The error compounds,
std(x_t) explodes, every value hits the +-3 clip and decodes to a column min/max.
Fix: predict x0, clip it, then sample from the exact Gaussian posterior q(x_{t-1} | x_t, x0_hat).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, StandardScaler

torch.manual_seed(0)
np.random.seed(0)


def make_cell(n=2746, seed=0):
    """Toy G3-fraud cell: 27 numeric columns shaped like v1's features (discrete amounts,
    heavy-tailed counts, integer day-deltas with heavy missingness)."""
    r = np.random.default_rng(seed)
    d = {"TransactionAmt": np.log1p(r.choice([25, 35, 50, 75, 100, 150, 200, 300], n)
                                    * np.exp(r.normal(0, .05, n))),
         "dist1": np.where(r.random(n) < .92, np.nan, r.exponential(40, n).round()),
         "dist2": np.where(r.random(n) < .70, np.nan, r.exponential(200, n).round()),
         "amt_to_mean_card1": r.normal(1.0, .15, n), "amt_to_mean_product": r.normal(1.05, .12, n)}
    for i in range(1, 15):
        base = r.negative_binomial(1, .4, n) + (i % 3 == 0)
        d[f"C{i}"] = np.where(r.random(n) < .01, r.integers(100, 3000, n), base).astype(float)
    for k, p in {"D1": .02, "D2": .85, "D3": .80, "D4": .45, "D10": .30, "D11": .95, "D15": .30}.items():
        v = np.where(r.random(n) < .55, 0, r.integers(1, 640, n)).astype(float)
        d[k] = np.where(r.random(n) < p, np.nan, v)
    df = pd.DataFrame(d)
    df["d1_to_d15_ratio"] = df["D1"].fillna(0) / (df["D15"].fillna(0) + 1.0)
    return df


# ------------------------------------------------------------------ v1 model (verbatim logic)
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class TabularDenoiserMLP(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.input_layer = nn.Linear(input_dim + embed_dim, 256)
        self.res1, self.res2 = ResidualBlock(256), ResidualBlock(256)
        self.output_layer = nn.Linear(256, input_dim)

    def forward(self, x_t, t):
        h = torch.relu(self.input_layer(torch.cat([x_t, self.time_mlp(t.unsqueeze(-1).float())], -1)))
        return self.output_layer(self.res2(self.res1(h)))


def cosine_betas(T, beta_max):
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + 0.008) / 1.008 * np.pi / 2) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - ac[1:] / ac[:-1], 0.0001, beta_max)


def train_v1(X, T=200, epochs=500, bs=256):
    model = TabularDenoiserMLP(X.shape[1])
    betas = cosine_betas(T, 0.9999)
    ac = torch.cumprod(1 - betas, 0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            x0 = X[perm[i:i + bs]]
            t = torch.randint(0, T, (len(x0),))
            eps = torch.randn_like(x0)
            a = ac[t][:, None]
            loss = nn.functional.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, betas


@torch.no_grad()
def sample_v1(model, betas, n, d):
    """v1 sampler: eps-parameterized ancestral step, no clipping until the very end."""
    alphas, ac = 1 - betas, torch.cumprod(1 - betas, 0)
    x, trace = torch.randn(n, d), []
    for ti in reversed(range(len(betas))):
        eps = model(x, torch.full((n,), ti, dtype=torch.long))
        z = torch.randn_like(x) if ti > 0 else 0
        x = (x - betas[ti] / (1 - ac[ti]).sqrt() * eps) / alphas[ti].sqrt() + betas[ti].sqrt() * z
        trace.append(float(x.std()))
    return torch.clip(x, -3, 3), trace[::-1]


@torch.no_grad()
def sample_stable(model, betas, n, d, clip=5.0):
    """Fix: x0-prediction with clipping + exact Gaussian posterior (same trained network)."""
    alphas, ac = 1 - betas, torch.cumprod(1 - betas, 0)
    ac_prev = torch.cat([torch.ones(1), ac[:-1]])
    x, trace = torch.randn(n, d), []
    for ti in reversed(range(len(betas))):
        eps = model(x, torch.full((n,), ti, dtype=torch.long))
        a, ap, b = ac[ti], ac_prev[ti], betas[ti]
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-clip, clip)
        mean = (ap.sqrt() * b / (1 - a)) * x0 + (alphas[ti].sqrt() * (1 - ap) / (1 - a)) * x
        x = mean + ((b * (1 - ap) / (1 - a)).sqrt() * torch.randn_like(x) if ti > 0 else 0)
        trace.append(float(x.std()))
    return x, trace[::-1]


def v1_audit(tr, ho, sy, cols):
    ks = np.mean([ks_2samp(ho[c].fillna(0), sy[c].fillna(0)).statistic for c in cols])
    sc = StandardScaler().fit(tr[cols].fillna(0))
    nnb = NearestNeighbors(n_neighbors=1).fit(sc.transform(tr[cols].fillna(0)))
    d_s = nnb.kneighbors(sc.transform(sy[cols].fillna(0)))[0]
    d_h = nnb.kneighbors(sc.transform(ho[cols].fillna(0)))[0]
    return ks, float(np.median(d_s) / (np.median(d_h) + 1e-8))


if __name__ == "__main__":
    cell = make_cell()
    cols = list(cell.columns)
    k = int(.8 * len(cell))
    tr, ho = cell.iloc[:k], cell.iloc[k:]
    qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=42)
    X = torch.tensor(qt.fit_transform(tr[cols].fillna(0).values), dtype=torch.float32)
    print("[*] training the v1 network (500 epochs) ...")
    model, betas = train_v1(X)
    model.eval()
    dec = lambda z: pd.DataFrame(qt.inverse_transform(z.numpy()), columns=cols)
    rows = []
    zb, tr_b = sample_v1(model, betas, 2254, len(cols))
    rows.append(("v1 sampler, beta_max=0.9999", *v1_audit(tr, ho, dec(zb), cols),
                 float((zb.abs() >= 2.999).float().mean())))
    zc, _ = sample_v1(model, cosine_betas(200, 0.999), 2254, len(cols))
    rows.append(("v1 sampler, beta_max=0.999", *v1_audit(tr, ho, dec(zc), cols),
                 float((zc.abs() >= 2.999).float().mean())))
    zs, tr_s = sample_stable(model, betas, 2254, len(cols))
    rows.append(("same weights, stable sampler", *v1_audit(tr, ho, dec(zs), cols),
                 float((zs.abs() >= 2.999).float().mean())))
    rows.append(("real rows (noise floor)", *v1_audit(tr, ho, tr.sample(len(ho), random_state=1), cols),
                 np.nan))
    print(pd.DataFrame(rows, columns=["sampler", "mean_KS", "DCR_ratio", "share_at_clip"])
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("    (the noise-floor DCR of 0 is expected: those rows ARE training rows - the "
          "memorization extreme)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.4))
        steps = np.arange(len(tr_b))
        ax.semilogy(steps, tr_b, label="v1 sampler (diverges)")
        ax.semilogy(steps, tr_s, label="x0-clipped posterior sampler")
        ax.axhline(1.0, ls="--", lw=.8, c="grey")
        ax.invert_xaxis()
        ax.set_xlabel("diffusion step t (sampling proceeds left to right)")
        ax.set_ylabel("std of x_t (log scale)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig("sampler_divergence.png", dpi=150)
        print("[*] saved sampler_divergence.png")
    except ImportError:
        pass
