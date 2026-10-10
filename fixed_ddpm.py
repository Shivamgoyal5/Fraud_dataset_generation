"""
fixed_ddpm.py
-------------
Stable tabular DDPM with PyTorch CUDA GPU support:
x0-clipped posterior sampler, sinusoidal time embeddings, EMA,
missing-indicators + NaN re-insertion, integer rounding, one-hot categorical blocks.
"""
import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

# Detect and assign CUDA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sinusoidal(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)

class Denoiser(nn.Module):
    def __init__(self, d_in, n_cond=0, width=256, emb=128):
        super().__init__()
        self.emb = emb
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.c_emb = nn.Embedding(n_cond + 1, emb) if n_cond else None
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width + emb, width),
                          nn.SiLU(), nn.Linear(width, width)) for _ in range(3)
        ])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_in))

    def forward(self, x, t, c=None):
        e = self.t_mlp(sinusoidal(t, self.emb))
        if self.c_emb is not None and c is not None:
            e = e + self.c_emb(c)
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk[3](blk[2](blk[1](torch.cat([blk[0](h), e], -1))))
        return self.out(h)

class StableTabDDPM:
    def __init__(self, num_cols, cat_cols=(), timesteps=200, seed=0):
        self.num_cols, self.cat_cols, self.T = list(num_cols), list(cat_cols), timesteps
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        self.betas = torch.clip(1 - ac[1:] / ac[:-1], 1e-4, 0.999).float().to(device)
        self.alphas = (1 - self.betas).to(device)
        self.ac = torch.cumprod(self.alphas, 0).to(device)
        self.ac_prev = torch.cat([torch.ones(1, device=device), self.ac[:-1]]).to(device)

    def _fit_encoder(self, X):
        self.miss_cols = [c for c in self.num_cols if X[c].isna().any()]
        self.int_cols = [c for c in self.num_cols if np.allclose(X[c].dropna() % 1, 0)]
        self.lo, self.hi = X[self.num_cols].min(), X[self.num_cols].max()
        filled = self._hotdeck(X[self.num_cols])
        self.qt = QuantileTransformer(n_quantiles=min(1000, len(X)), output_distribution="normal",
                                      subsample=200_000, random_state=0).fit(filled.values)
        self.cats = {c: np.sort(X[c].unique()) for c in self.cat_cols}

    def _hotdeck(self, N):
        N = N.copy()
        for c in N.columns:
            m = N[c].isna().values
            if m.any():
                obs = N[c].dropna().values
                N.loc[m, c] = self.rng.choice(obs, m.sum()) if len(obs) else 0.0
        return N

    def _encode(self, X):
        parts = [self.qt.transform(self._hotdeck(X[self.num_cols]).values)]
        if self.miss_cols:
            parts.append(np.where(X[self.miss_cols].isna().values, 1.0, -1.0))
        for c in self.cat_cols:
            parts.append(np.where(X[c].values[:, None] == self.cats[c][None, :], 1.0, -1.0))
        return np.hstack(parts).astype(np.float32)

    def _decode(self, Z):
        k, out = len(self.num_cols), {}
        num = pd.DataFrame(self.qt.inverse_transform(np.clip(Z[:, :k], -5.2, 5.2)), columns=self.num_cols)
        for c in self.int_cols:
            num[c] = num[c].round()
        num = num.clip(self.lo, self.hi, axis=1)
        j = k
        for c in self.miss_cols:
            num.loc[Z[:, j] > 0, c] = np.nan; j += 1
        for c in self.cat_cols:
            m = len(self.cats[c]); out[c] = self.cats[c][Z[:, j:j + m].argmax(1)]; j += m
        return pd.concat([num, pd.DataFrame(out)], axis=1)[self.num_cols + self.cat_cols]

    def fit(self, X, cond=None, epochs=300, batch_size=512, lr=1e-3, p_uncond=0.1, verbose=False):
        self._fit_encoder(X)
        Z = torch.tensor(self._encode(X), dtype=torch.float32, device=device)
        self.cells = sorted(pd.unique(cond)) if cond is not None else []
        c_all = (torch.tensor(pd.Series(cond).map({c: i for i, c in enumerate(self.cells)}).values, device=device)
                 if cond is not None else None)
        self.model = Denoiser(Z.shape[1], n_cond=len(self.cells)).to(device)
        self.ema = copy.deepcopy(self.model).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        steps = epochs * math.ceil(len(Z) / batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

        for ep in range(epochs):
            perm = torch.randperm(len(Z), device=device)
            for i in range(0, len(Z), batch_size):
                idx = perm[i:i + batch_size]; x0 = Z[idx]
                t = torch.randint(0, self.T, (len(idx),), device=device); eps = torch.randn_like(x0)
                a = self.ac[t][:, None]
                xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
                c = None
                if c_all is not None:
                    c = c_all[idx].clone(); c[torch.rand(len(idx), device=device) < p_uncond] = len(self.cells)
                loss = nn.functional.mse_loss(self.model(xt, t, c), eps)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step()
                with torch.no_grad():
                    for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                        pe.mul_(0.995).add_(pm, alpha=0.005)
            if verbose and ep % max(1, epochs // 5) == 0:
                print(f"    epoch {ep:4d} loss {loss.item():.4f}")
        return self

    @torch.no_grad()
    def sample(self, n, cell=None, guidance=1.0, clip=5.0):
        m = self.ema.eval()
        x = torch.randn((n, self.model.out[1].out_features), device=device)
        ci = None if cell is None else torch.full((n,), self.cells.index(cell), dtype=torch.long, device=device)
        null = None if cell is None else torch.full((n,), len(self.cells), dtype=torch.long, device=device)
        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, dtype=torch.long, device=device)
            eps = m(x, t, ci)
            if ci is not None and guidance != 1.0:
                eps = m(x, t, null) + guidance * (eps - m(x, t, null))
            a, ap, b = self.ac[ti], self.ac_prev[ti], self.betas[ti]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-clip, clip)
            mean = (ap.sqrt() * b / (1 - a)) * x0 + (self.alphas[ti].sqrt() * (1 - ap) / (1 - a)) * x
            var = b * (1 - ap) / (1 - a)
            x = mean + (var.sqrt() * torch.randn_like(x) if ti > 0 else 0)
        return self._decode(x.cpu().numpy())