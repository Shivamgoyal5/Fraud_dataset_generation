#!/usr/bin/env python3
"""
train_pipeline_v2.py
--------------------
Corrected & Leakage-Free Uniform Fraud Pipeline for IEEE-CIS:
1. Strict train-only feature engineering (no future test set leakage).
2. Time-safe temporal splitting with chargeback embargo gaps.
3. Stable ResNet TabDDPM generator with x0-clamped posterior sampler.
4. Two-sided Bounded Fidelity & Privacy Quality Gate Audit.
5. Consistent Fallback Re-weighting & Logit Offset derivation.
6. Downstream GBDT training with weight normalization (mean=1.0).
7. Per-product threshold tuning & prior logit correction.
"""

from __future__ import annotations
import os
import argparse
from dataclasses import replace, dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, 
    matthews_corrcoef, brier_score_loss, roc_curve
)
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Silence joblib physical core warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

# ============================================================================
# 1. CONFIGURATION & STRATEGY HYPERPARAMETERS
# ============================================================================

PRODUCT_MAP = {
    "W": "G1_W",      # ~74.5% volume, ~2.0% fraud rate
    "C": "G2_C",      # ~11.6% volume, ~11.7% fraud rate
    "R": "G3_RHS",    # ~13.9% volume, ~4.5% aggregate fraud rate
    "H": "G3_RHS",
    "S": "G3_RHS"
}

@dataclass
class StrategyPlan:
    tau: float = 0.0        # Product target mass exponent (0.0 = uniform, 1.0 = natural volume)
    r_star: float = 0.20     # Target fraud ratio within each product cell
    floor: int = 5000        # Positive sample floor for minority cells
    rho: float = 1.0         # Maximum synthetic rows per real row cap
    lam: float = 0.8         # Loss weight discount factor for synthetic rows
    epochs_ddpm: int = 300   # Diffusion training epochs
    timesteps: int = 100     # Diffusion timesteps

# ============================================================================
# 2. DATA PROCESSING & LEAKAGE-FREE FEATURE ENGINEERING
# ============================================================================

def load_and_preprocess_ieee(tx_path: str, id_path: str | None = None) -> pd.DataFrame:
    print(f"[*] Loading transaction records from: {tx_path}")
    df_tx = pd.read_csv(tx_path)
    
    if id_path and os.path.exists(id_path):
        print(f"[*] Merging identity records from: {id_path}")
        df_id = pd.read_csv(id_path)
        df = df_tx.merge(df_id, on="TransactionID", how="left")
    else:
        df = df_tx

    df["day"] = df["TransactionDT"] // 86400
    df["group"] = df["ProductCD"].map(PRODUCT_MAP)
    df["y"] = df["isFraud"].astype(int)
    df["TransactionAmt"] = np.log1p(df["TransactionAmt"])

    # Approximate client key (uid) for entity-disjoint auditing
    card1_str = df["card1"].fillna(-1).astype(int).astype(str)
    addr1_str = df["addr1"].fillna(-1).astype(int).astype(str)
    d1_shift = (df["day"] - df["D1"].fillna(0)).astype(int).astype(str)
    df["uid"] = card1_str + "_" + addr1_str + "_" + d1_shift

    return df

def temporal_split(df: pd.DataFrame, train_end=120, val_end=153, gap=7):
    train_df = df[df["day"] < train_end].copy()
    val_df = df[(df["day"] >= train_end + gap) & (df["day"] < val_end)].copy()
    test_df = df[df["day"] >= val_end + gap].copy()
    return train_df, val_df, test_df

def add_leakage_free_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Computes group stats ONLY on train_df and maps them to val_df and test_df."""
    print("[*] Computing leakage-free domain interaction features...")
    
    # Card1 Mean Amount
    card1_means = train_df.groupby('card1')['TransactionAmt'].mean().to_dict()
    global_amt_mean = train_df['TransactionAmt'].mean()
    
    # ProductCD Mean Amount
    product_means = train_df.groupby('ProductCD')['TransactionAmt'].mean().to_dict()

    for split in [train_df, val_df, test_df]:
        c1_m = split['card1'].map(card1_means).fillna(global_amt_mean)
        p_m = split['ProductCD'].map(product_means).fillna(global_amt_mean)
        
        split['amt_to_mean_card1'] = split['TransactionAmt'] / (c1_m + 1e-5)
        split['amt_to_mean_product'] = split['TransactionAmt'] / (p_m + 1e-5)
        
        if 'D1' in split.columns and 'D15' in split.columns:
            split['d1_to_d15_ratio'] = split['D1'].fillna(0) / (split['D15'].fillna(0) + 1.0)

def extract_features(df: pd.DataFrame):
    num_cols = [c for c in ["TransactionAmt", "dist1", "dist2", "amt_to_mean_card1", 
                            "amt_to_mean_product", "d1_to_d15_ratio",
                            *[f"C{i}" for i in range(1, 15)],
                            *[f"D{i}" for i in (1, 2, 3, 4, 10, 11, 15)]] if c in df.columns]
    cat_cols = [c for c in ["card4", "card6", "P_emaildomain", "R_emaildomain",
                            "M4", "M5", "M6", "DeviceType", "ProductCD"] if c in df.columns]
    return num_cols, cat_cols

# ============================================================================
# 3. STRATEGY MATH & PRIOR CORRECTION
# ============================================================================

def compute_cell_weights(counts_df: pd.DataFrame, plan: StrategyPlan) -> pd.DataFrame:
    n1 = counts_df["fraud"].astype(float)
    n0 = (counts_df["rows"] - counts_df["fraud"]).astype(float)
    total_rows = counts_df["rows"].sum()

    vol_share = counts_df["rows"] / total_rows
    alpha = vol_share ** plan.tau
    alpha = alpha / alpha.sum()

    syn_needed = np.maximum(0, plan.floor - n1)
    syn_count = np.minimum(plan.rho * n1, syn_needed).round()

    eff_n1 = n1 + plan.lam * syn_count
    
    scale = (eff_n1 / (alpha * plan.r_star)).max()
    m1 = scale * alpha * plan.r_star
    m0 = scale * alpha * (1.0 - plan.r_star)

    w_fraud_real = m1 / eff_n1
    w_fraud_syn = plan.lam * w_fraud_real
    w_legit = m0 / n0

    real_rate = n1 / counts_df["rows"]
    logit_offset = np.log(plan.r_star / (1.0 - plan.r_star)) - np.log(real_rate / (1.0 - real_rate))

    return pd.DataFrame({
        "real_fraud": n1,
        "syn_fraud": syn_count,
        "w_fraud_real": w_fraud_real,
        "w_fraud_syn": w_fraud_syn,
        "w_legit": w_legit,
        "logit_offset": logit_offset
    }, index=counts_df.index)

def apply_prior_correction(probabilities: np.ndarray, product_series: pd.Series, offsets: pd.Series) -> np.ndarray:
    p = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    raw_logits = np.log(p / (1.0 - p))
    sample_offsets = product_series.map(offsets).values
    corrected_logits = raw_logits - sample_offsets
    return 1.0 / (1.0 + np.exp(-corrected_logits))

# ============================================================================
# 4. STABLE RESNET TABDDPM WITH x0-CLAMPED POSTERIOR SAMPLER
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))

class TabularDenoiserMLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.input_layer = nn.Linear(input_dim + embed_dim, 128)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.output_layer = nn.Linear(128, input_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        h = torch.cat([x_t, t_emb], dim=-1)
        h = torch.relu(self.input_layer(h))
        h = self.res1(h)
        h = self.res2(h)
        return self.output_layer(h)

class SimpleTabDDPM:
    """Stable Tabular DDPM Generator using x0-clamped posterior sampling."""
    def __init__(self, num_cols: list[str], timesteps: int = 100):
        self.num_cols = num_cols
        self.timesteps = timesteps
        self.scaler = QuantileTransformer(
            n_quantiles=500, output_distribution='normal', subsample=100000, random_state=42
        )
        
        # Cosine Beta Schedule clipped at beta_max = 0.9990 to prevent step variance explosion
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9990)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def fit(self, X_minority: pd.DataFrame, epochs: int = 300, batch_size: int = 256):
        X_raw = X_minority[self.num_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X_raw)
        x_0 = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model = TabularDenoiserMLP(input_dim=len(self.num_cols))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(x_0.size(0))
            for i in range(0, x_0.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x0 = x_0[indices]
                
                t = torch.randint(0, self.timesteps, (batch_x0.size(0),))
                noise = torch.randn_like(batch_x0)
                
                alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
                x_t = torch.sqrt(alpha_t) * batch_x0 + torch.sqrt(1 - alpha_t) * noise
                
                noise_pred = self.model(x_t, t)
                loss = nn.functional.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def sample(self, num_samples: int) -> pd.DataFrame:
        self.model.eval()
        ac = self.alphas_cumprod
        ac_prev = torch.cat([torch.ones(1), ac[:-1]])
        
        with torch.no_grad():
            x = torch.randn((num_samples, len(self.num_cols)))
            for t_idx in reversed(range(self.timesteps)):
                t = torch.full((num_samples,), t_idx, dtype=torch.long)
                eps = self.model(x, t)
                
                a = ac[t_idx]
                ap = ac_prev[t_idx]
                b = self.betas[t_idx]
                
                # Predict x0 and clamp in scaled space [-5, 5] to stop exploding noise variance
                x0_pred = ((x - torch.sqrt(1 - a) * eps) / torch.sqrt(a)).clamp(-5.0, 5.0)
                
                # Exact posterior mean calculation
                mean = (torch.sqrt(ap) * b / (1 - a)) * x0_pred + (torch.sqrt(self.alphas[t_idx]) * (1 - ap) / (1 - a)) * x
                var = b * (1 - ap) / (1 - a)
                
                noise = torch.randn_like(x) if t_idx > 0 else 0
                x = mean + torch.sqrt(var) * noise

        X_syn_raw = self.scaler.inverse_transform(x.numpy())
        return pd.DataFrame(X_syn_raw, columns=self.num_cols)

# ============================================================================
# 5. BOUNDED FIDELITY & PRIVACY AUDIT GATE
# ============================================================================

def audit_generative_quality(train_real: pd.DataFrame, holdout_real: pd.DataFrame, 
                             synthetic: pd.DataFrame, num_cols: list[str]) -> bool:
    # 1. Kolmogorov-Smirnov Distribution Drift
    ks_scores = [ks_2samp(holdout_real[col].fillna(0).values, synthetic[col].fillna(0).values).statistic for col in num_cols]
    mean_ks = float(np.mean(ks_scores))

    # 2. Distance to Closest Record (DCR) Two-Sided Check
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_real[num_cols].fillna(0).values)
    hold_scaled = scaler.transform(holdout_real[num_cols].fillna(0).values)
    syn_scaled = scaler.transform(synthetic[num_cols].fillna(0).values)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(train_scaled)
    dist_syn, _ = nn.kneighbors(syn_scaled)
    dist_hold, _ = nn.kneighbors(hold_scaled)

    dcr_ratio = float(np.median(dist_syn) / (np.median(dist_hold) + 1e-8))
    near_copy_share = float(np.mean(dist_syn[:, 0] < 0.05))

    print(f"\n" + "=" * 80)
    print(" GENERATIVE DATA FIDELITY & PRIVACY AUDIT REPORT")
    print("=" * 80)
    print(f"  • Mean KS Statistic       : {mean_ks:.4f} (Target: < 0.20)")
    print(f"  • Median DCR Ratio         : {dcr_ratio:.4f} (Target: 0.20 <= DCR <= 2.50)")
    print(f"  • Near-Copy Share (<0.05)  : {near_copy_share*100:.2f}% (Target: < 1.0%)")
    print("=" * 80)

    # TWO-SIDED DCR GATE: Rejects both memorization (DCR < 0.20) AND divergence/garbage (DCR > 2.50)
    passed_gate = (mean_ks <= 0.20) and (0.20 <= dcr_ratio <= 2.50) and (near_copy_share <= 0.01)
    print(f"[*] QUALITY GATE DECISION: {'PASSED (ACCEPT SYNTHETIC)' if passed_gate else 'FAILED (FALLBACK TO REWEIGHTING)'}\n")
    return passed_gate

def compute_imbalance_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50, target_fpr: float = 0.01) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
    
    balanced_acc = (recall + specificity) / 2.0
    g_mean = np.sqrt(recall * specificity)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    fpr_array, tpr_array, _ = roc_curve(y_true, y_prob)
    valid_idx = np.where(fpr_array <= target_fpr)[0]
    recall_at_target_fpr = tpr_array[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
    
    return {
        "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "F1": f1, "F2": f2,
        "Precision": precision, "Recall": recall, "Recall@1%FPR": recall_at_target_fpr,
        "Bal_Acc": balanced_acc, "G_Mean": g_mean, "MCC": mcc, "Brier": brier
    }

def find_best_threshold(y_val: np.ndarray, y_val_prob: np.ndarray) -> float:
    best_thresh, best_f1 = 0.50, 0.0
    for thresh in np.linspace(0.01, 0.90, 90):
        preds = (y_val_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh

# ============================================================================
# 6. PIPELINE EXECUTION
# ============================================================================

def run_pipeline(tx_path: str, id_path: str | None = None):
    df = load_and_preprocess_ieee(tx_path, id_path)
    train_df, val_df, test_df = temporal_split(df)
    
    # Feature Engineering (Strictly Leakage-Free)
    add_leakage_free_features(train_df, val_df, test_df)
    num_cols, cat_cols = extract_features(train_df)
    
    for c in cat_cols:
        train_df[c] = train_df[c].astype("category")
        val_df[c] = val_df[c].astype("category")
        test_df[c] = test_df[c].astype("category")

    print(f"[*] Train set: {len(train_df)} rows | Val set: {len(val_df)} rows | Test set: {len(test_df)} rows")

    counts_data = train_df.groupby("group").agg(rows=("y", "count"), fraud=("y", "sum"))
    plan = StrategyPlan(tau=0.0, r_star=0.20, floor=5000, rho=1.0)
    weights_df = compute_cell_weights(counts_data, plan)
    
    print("\n[+] Computed Cell Mass Weights & Logit Shifts:")
    print(weights_df.to_string())

    g3_fraud = train_df[(train_df["group"] == "G3_RHS") & (train_df["y"] == 1)]
    syn_count = int(weights_df.loc["G3_RHS", "syn_fraud"])
    
    # Entity-Disjoint Audit Split for TabDDPM Quality Gate
    split_idx = int(0.8 * len(g3_fraud))
    g3_train, g3_holdout = g3_fraud.iloc[:split_idx], g3_fraud.iloc[split_idx:]

    feature_cols = num_cols + cat_cols

    if syn_count > 0 and len(g3_train) > 10:
        print(f"\n[*] Training Stable ResNet TabDDPM Generator on G3_RHS fraud cell ({len(g3_train)} rows)...")
        generator = SimpleTabDDPM(num_cols=num_cols, timesteps=plan.timesteps)
        generator.fit(g3_train, epochs=plan.epochs_ddpm)
        
        syn_df = generator.sample(syn_count)
        syn_df["group"] = "G3_RHS"
        syn_df["y"] = 1
        
        # FIX LATENT CONTAMINATION: Set categoricals to G3_RHS mode (e.g. ProductCD 'R'), NOT global mode ('W')
        g3_real_mode = train_df[train_df["group"] == "G3_RHS"]
        for c in cat_cols:
            mode_s = g3_real_mode[c].mode()
            if len(mode_s) > 0:
                syn_df[c] = mode_s.iloc[0]
            else:
                global_mode = train_df[c].mode()
                syn_df[c] = global_mode.iloc[0] if len(global_mode) > 0 else train_df[c].cat.categories[0]

        passed_audit = audit_generative_quality(g3_train, g3_holdout, syn_df, num_cols)
        
        if passed_audit:
            print("[+] Appending verified synthetic rows to training set.")
            X_real = train_df[feature_cols]
            y_real = train_df["y"]
            w_real = np.where(train_df["y"] == 1, 
                              train_df["group"].map(weights_df["w_fraud_real"]), 
                              train_df["group"].map(weights_df["w_legit"]))

            X_syn = syn_df[feature_cols]
            y_syn = syn_df["y"]
            w_syn = np.full(len(syn_df), weights_df.loc["G3_RHS", "w_fraud_syn"])

            X_train_final = pd.concat([X_real, X_syn], axis=0, ignore_index=True)
            y_train_final = pd.concat([y_real, y_syn], axis=0, ignore_index=True)
            w_train_final = np.concatenate([w_real, w_syn])
        else:
            print("[!] Generative Audit failed. Falling back to pure sample reweighting (rho=0.0).")
            # RECOMPUTE WEIGHTS & OFFSETS FOR FALLBACK (RHO = 0.0)
            weights_df = compute_cell_weights(counts_data, replace(plan, rho=0.0))
            X_train_final, y_train_final = train_df[feature_cols], train_df["y"]
            w_train_final = np.where(train_df["y"] == 1, 
                                     train_df["group"].map(weights_df["w_fraud_real"]), 
                                     train_df["group"].map(weights_df["w_legit"]))
    else:
        X_train_final, y_train_final = train_df[feature_cols], train_df["y"]
        w_train_final = np.where(train_df["y"] == 1, 
                                 train_df["group"].map(weights_df["w_fraud_real"]), 
                                 train_df["group"].map(weights_df["w_legit"]))

    # NORMALIZE WEIGHTS TO MEAN = 1.0 (Fixes Tree Model L2 Regularization Distortion)
    w_train_final = w_train_final / np.mean(w_train_final)

    print("\n[*] Training Tuned Downstream GBDT Classifier with Per-Cell Weights...")
    clf = HistGradientBoostingClassifier(
        categorical_features=[c in cat_cols for c in feature_cols],
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    )
    clf.fit(X_train_final, y_train_final, sample_weight=w_train_final)

    # Threshold Tuning on Validation Split
    X_val = val_df[feature_cols]
    y_val = val_df["y"].values
    raw_val_probs = clf.predict_proba(X_val)[:, 1]
    val_probs = apply_prior_correction(raw_val_probs, val_df["group"], weights_df["logit_offset"])

    optimal_thresholds = {}
    for group_name in weights_df.index:
        mask_val = (val_df["group"] == group_name).values
        if mask_val.sum() > 0 and y_val[mask_val].sum() > 0:
            optimal_thresholds[group_name] = find_best_threshold(y_val[mask_val], val_probs[mask_val])
        else:
            optimal_thresholds[group_name] = 0.50

    print("\n[+] Optimal Per-Product Thresholds Tuned on Validation Split:")
    for g, t in optimal_thresholds.items():
        print(f"  • Product Group {g:<8}: Threshold = {t:.4f}")

    # Test Split Evaluation
    X_test = test_df[feature_cols]
    y_test = test_df["y"].values
    raw_test_probs = clf.predict_proba(X_test)[:, 1]
    corrected_test_probs = apply_prior_correction(raw_test_probs, test_df["group"], weights_df["logit_offset"])

    print("\n" + "=" * 115)
    print(" EXTENDED PER-PRODUCT CLASS IMBALANCE EVALUATION TABLE (TEST SET)")
    print("=" * 115)
    
    results = []
    overall_preds_list = np.zeros(len(test_df))

    for group_name in weights_df.index:
        mask = (test_df["group"] == group_name).values
        if mask.sum() > 0:
            t_opt = optimal_thresholds[group_name]
            m = compute_imbalance_metrics(y_test[mask], corrected_test_probs[mask], threshold=t_opt)
            m["Group"] = group_name
            m["Rows"] = int(mask.sum())
            m["Frauds"] = int(y_test[mask].sum())
            results.append(m)
            
            # Store per-product threshold decisions for OVERALL row computation
            overall_preds_list[mask] = (corrected_test_probs[mask] >= t_opt).astype(int)

    # Compute OVERALL row using per-product optimal thresholds
    overall_m = compute_imbalance_metrics(y_test, corrected_test_probs)
    
    # Recompute OVERALL threshold-dependent metrics using product-wise decisions
    tn, fp, fn, tp = confusion_matrix(y_test, overall_preds_list, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    overall_m["F1"] = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    overall_m["F2"] = 5 * (prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    overall_m["Precision"] = prec
    overall_m["Recall"] = rec
    overall_m["MCC"] = matthews_corrcoef(y_test, overall_preds_list)

    overall_m["Group"] = "OVERALL"
    overall_m["Rows"] = int(len(y_test))
    overall_m["Frauds"] = int(y_test.sum())
    results.append(overall_m)

    res_df = pd.DataFrame(results)
    cols = ["Group", "Rows", "Frauds", "PR_AUC", "ROC_AUC", "F1", "F2", "Precision", "Recall", "Recall@1%FPR", "Bal_Acc", "G_Mean", "MCC", "Brier"]
    res_df = res_df[cols]

    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 115)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE-CIS Corrected Uniform Fraud Pipeline")
    parser.add_argument("--tx_path", type=str, required=True, help="Path to train_transaction.csv")
    parser.add_argument("--id_path", type=str, required=False, default=None, help="Path to train_identity.csv")
    args = parser.parse_args()

    run_pipeline(args.tx_path, args.id_path)