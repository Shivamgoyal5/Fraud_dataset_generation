#!/usr/bin/env python3
"""
train_pipeline_v2.py  -  uniform per-product fraud learning on IEEE-CIS (corrected v1)
====================================================================================
  python train_pipeline_v2.py --tx train_transaction.csv --id train_identity.csv            # ablations
  python train_pipeline_v2.py --tx ... --id ... --gen                                       # + synthetic
  python train_pipeline_v2.py --tx ... --id ... --quick                                     # B0 vs B2 only

FIXES vs v1
 F1 Generator sampler diverged: beta_T = 0.9999 makes the first reverse step multiply the noise-
    prediction error ~100x; std(x_t) grows to ~1e5 and every value is clipped at +-3, i.e. decoded
    to a column min/max (reproduced on toy data: KS 0.885, DCR ratio 42; your run: 0.73 / 419).
    -> x0-prediction with clipping + exact Gaussian posterior, beta cap 0.999, sinusoidal time
       embedding, EMA, grad clipping (toy: KS 0.885 -> 0.064, noise floor 0.069). See fixed_ddpm.py
 F2 NaN -> 0 created fake spikes and synthetic rows were never missing
    -> hot-deck imputation + missing-indicator channels; NaN re-inserted at sampling; ints rounded.
 F3 Synthetic categoricals = global mode, incl. ProductCD = 'W' for G3 rows (latent contamination of W)
    -> categoricals generated jointly as one-hot blocks by ONE pooled model conditioned on the cell.
 F4 After a failed gate, the fallback reused weights/offsets that assumed synthetic rows existed
    (G3 fraud mass ~3.7k instead of ~6.1k; G3 logit offset off by ~0.5)
    -> plan recomputed with synthetic rows only for cells that pass; offsets from REALIZED weights.
 F5 Gate: DCR >= 0.2 lets memorization through, and DCR = 419 (off-manifold) passed the DCR check;
    KS on fillna(0) data, numerics only, holdout = later-in-time rows (drift looks like infidelity)
    -> entity-disjoint holdout from the same window; two-sided DCR band; KS on observed values;
       missing-rate gap; categorical TVD; C2ST.
 F6 OVERALL row used threshold 0.5 while product rows used tuned thresholds -> per-product thresholds.
 F7 groupby aggregates over the whole timeline (incl. test) -> past-only (leak-free) features.
 F8 Mean sample weight ~0.22 makes l2_regularization ~4.5x stronger; W legit (w=0.085) hit hardest
    -> weights normalized to mean 1.
 F9 300 trees @ lr 0.03 + random internal early-stopping split -> warm-start early stopping on the
    validation MONTH using mean per-product PR-AUC (the stated objective).
W-FOCUSED UPGRADES
 U1 Past-only client (uid) / card / address velocity + amount-deviation features, D-normalization
    (day - Dx), hour, cents, M1-M9, a reduced V block, identity numerics/categoricals.
 U2 Ablations that show what uniform weighting costs or gains for W: raw, global class weights,
    tau in {0, 0.5}, a lower r* for W only.
 U3 Metrics comparable across base rates: prevalence, AP skill (AP-pi)/(1-pi), ROC-AUC, recall@1%FPR,
    F1 at a common reference prevalence, Brier skill score, entity-bootstrap 95% CIs.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # harmless Windows warning

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

PRODUCT_MAP = {"W": "G1_W", "C": "G2_C", "R": "G3_RHS", "H": "G3_RHS", "S": "G3_RHS"}
GROUPS = ["G1_W", "G2_C", "G3_RHS"]

NUM_BASE = ["log_amt", "amt_cents", "hour", "dist1", "dist2", "card1", "card2", "card3", "card5",
            "addr1", "addr2", *[f"C{i}" for i in range(1, 15)], "D1", "D3", "D15",
            "D1n", "D2n", "D3n", "D4n", "D10n", "D11n", "D15n",
            "uid_cnt_past", "card1_cnt_past", "addr1_cnt_past", "uid_amt_mean_past", "amt_dev_uid",
            "amt_z_uid", "uid_hours_since_prev", "uid_day_cnt_past", "uid_email_changed",
            "id_01", "id_02", "id_05", "id_06", "id_13", "id_17", "id_19", "id_20"]
CAT_BASE = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            *[f"M{i}" for i in range(1, 10)], "DeviceType", "DeviceInfo",
            "id_12", "id_15", "id_16", "id_31", "id_38"]


@dataclass
class Plan:
    tau: float = 0.0                                     # 0 = uniform products, 1 = natural mix
    r_star: float = 0.20                                 # target fraud share within a product
    r_star_by_group: dict = field(default_factory=dict)  # e.g. {"G1_W": 0.05}
    floor: int = 5000                                    # top up fraud cells below this ...
    rho: float = 1.0                                     # ... with <= rho synthetic per real row
    lam: float = 0.8                                     # relative weight of a synthetic row


# ============================================================================ data + features
def _as_str(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "__nan__").astype(str)


def load(tx_path: str, id_path: str | None) -> pd.DataFrame:
    print(f"[*] loading {tx_path}")
    df = pd.read_csv(tx_path)
    if id_path and os.path.exists(id_path):
        df = df.merge(pd.read_csv(id_path), on="TransactionID", how="left")
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    df["day"] = df["TransactionDT"] // 86400
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["group"] = df["ProductCD"].map(PRODUCT_MAP)
    df["y"] = df["isFraud"].astype(int)
    amt = df["TransactionAmt"].astype(float)
    df["amt_cents"] = (amt - np.floor(amt)).round(3)
    df["log_amt"] = np.log1p(amt)
    # approximate client key (top competition write-ups): card1 + addr1 + account-start day
    df["uid"] = _as_str(df["card1"]) + "_" + _as_str(df["addr1"]) + "_" + _as_str(df["day"] - df["D1"])
    return df


def past_only_features(df: pd.DataFrame) -> pd.DataFrame:
    """History features from EARLIER transactions only (df is time-sorted). Leak-free under any
    temporal split, and they are what production fraud systems compute anyway."""
    amt = df["TransactionAmt"].astype(float)
    g = df.groupby("uid")
    f = {"uid_cnt_past": g.cumcount()}
    for key in ["card1", "addr1"]:
        f[f"{key}_cnt_past"] = df.groupby(_as_str(df[key])).cumcount()
    cnt = f["uid_cnt_past"].replace(0, np.nan)
    mean_p = (g["TransactionAmt"].cumsum() - amt) / cnt
    sq_p = ((amt ** 2).groupby(df["uid"]).cumsum() - amt ** 2) / cnt
    sd_p = np.sqrt((sq_p - mean_p ** 2).clip(lower=0))
    f["uid_amt_mean_past"] = np.log1p(mean_p)
    f["amt_dev_uid"] = df["log_amt"] - np.log1p(mean_p)
    f["amt_z_uid"] = (amt - mean_p) / (sd_p + 1.0)
    f["uid_hours_since_prev"] = (df["TransactionDT"] - g["TransactionDT"].shift()) / 3600.0
    f["uid_day_cnt_past"] = df.groupby([df["uid"], df["day"]]).cumcount()
    if "P_emaildomain" in df:
        prev = g["P_emaildomain"].shift()
        f["uid_email_changed"] = pd.Series(np.where(prev.isna(), np.nan, (_as_str(df["P_emaildomain"])
                                                    != _as_str(prev)).astype(float)), index=df.index)
    for c in ["D1", "D2", "D3", "D4", "D10", "D11", "D15"]:      # 'days since X' -> date of X
        if c in df:
            f[f"{c}n"] = df["day"] - df[c]
    return pd.concat([df, pd.DataFrame(f, index=df.index)], axis=1)


def reduce_v(df: pd.DataFrame, mask: pd.Series, per_group=2, max_cols=30) -> list[str]:
    """V1-V339 come in blocks sharing NaN patterns: keep the most granular columns per block.
    Computed on the TRAIN window, column by column (no large temporary copies)."""
    vcols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    if not vcols:
        return []
    m = mask.values
    pos = np.flatnonzero(m)
    smp = np.random.default_rng(0).choice(pos, min(len(pos), 100_000), replace=False)
    nan_cnt = pd.Series({c: int(np.isnan(df[c].values[m]).sum()) for c in vcols})
    nun = pd.Series({c: pd.Series(df[c].values[smp]).nunique() for c in vcols})
    keep = []
    for _, cols in sorted(nan_cnt.groupby(nan_cnt).groups.items()):
        keep += list(nun[list(cols)].sort_values(ascending=False).index[:per_group])
    return keep[:max_cols]


def fit_vocab(train: pd.DataFrame, cat_cols, top_k=30) -> dict:
    return {c: list(_as_str(train[c]).value_counts().drop("__nan__", errors="ignore").index[:top_k])
            for c in cat_cols}


def encode_cats(df: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """Integer codes: kept categories 0..k-1, OTHER = k, missing = k+1 (all < 255 for HGB)."""
    out = {}
    for c, keep in vocab.items():
        s = _as_str(df[c])
        codes = s.map({v: i for i, v in enumerate(keep)}).fillna(len(keep)).values
        out[c] = np.where(s.values == "__nan__", len(keep) + 1, codes).astype(np.int16)
    return pd.DataFrame(out, index=df.index)


def diagnostics(df, masks):
    rows = []
    for g in GROUPS:
        d = df[df["group"] == g]
        r = {"group": g, "rows": len(d)}
        for k, m in masks.items():
            r[f"prev_{k}"] = d.loc[m[d.index], "y"].mean()
        r["identity_cov"] = d["DeviceType"].notna().mean() if "DeviceType" in d else np.nan
        mcols = [c for c in d.columns if c in [f"M{i}" for i in range(1, 10)]]
        r["M_cov"] = d[mcols].notna().any(axis=1).mean() if mcols else np.nan
        r["fraud_uids_train"] = d.loc[masks["train"][d.index] & (d["y"] == 1), "uid"].nunique()
        rows.append(r)
    print("\n[diagnostics] per product (identity_cov = share with identity rows)")
    print(pd.DataFrame(rows).set_index("group").round(4).to_string())


# ============================================================================ weights
def design_plan(counts: pd.DataFrame, plan: Plan, allow_syn=()) -> pd.DataFrame:
    n1 = counts["fraud"].astype(float)
    n0 = (counts["rows"] - counts["fraud"]).astype(float)
    alpha = (counts["rows"] / counts["rows"].sum()) ** plan.tau
    alpha = alpha / alpha.sum()
    r = pd.Series({g: plan.r_star_by_group.get(g, plan.r_star) for g in counts.index})
    syn = np.minimum(plan.rho * n1, np.clip(plan.floor - n1, 0, None)).round()
    syn = syn.where(syn.index.isin(list(allow_syn)), 0.0)          # F4: only cells that passed
    eff1 = n1 + plan.lam * syn
    scale = (eff1 / (alpha * r)).max()
    m1, m0 = scale * alpha * r, scale * alpha * (1 - r)
    return pd.DataFrame({"real_fraud": n1, "syn_fraud": syn, "w_fraud_real": m1 / eff1,
                         "w_fraud_syn": plan.lam * m1 / eff1, "w_legit": m0 / n0})


def row_weights(groups, y, P, is_syn=None):
    w = np.where(y == 1, groups.map(P["w_fraud_real"]), groups.map(P["w_legit"])).astype(float)
    if is_syn is not None:
        w = np.where(is_syn, groups.map(P["w_fraud_syn"]), w)
    return w * len(w) / w.sum()                                    # F8: mean weight = 1


def realized_offsets(groups, y, w, is_syn=None) -> pd.Series:
    """logit shift actually induced by the assembled training set (F4)."""
    d = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y), "w": w})
    real = d if is_syn is None else d[~np.asarray(is_syn)]
    rate = real.groupby("g")["y"].mean()
    wm = d.groupby(["g", "y"])["w"].sum().unstack()
    return np.log(wm[1] / wm[0]) - np.log(rate / (1 - rate))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_offset(p, groups, off):
    return 1 / (1 + np.exp(-(_logit(p) - groups.map(off).values)))


# ============================================================================ classifier
def fit_gbdt(Xtr, ytr, wtr, Xva, yva, gva, cat_mask, max_iter=1000, step=200, patience=2, seed=0):
    """Fast early-stopped GBDT training evaluated on validation month."""
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08, 
        max_leaf_nodes=31,         # Optimized from 127 to 31 for speed
        min_samples_leaf=50,
        l2_regularization=0.5, 
        categorical_features=cat_mask,
        early_stopping=False, 
        warm_start=True, 
        max_iter=step,
        random_state=seed
    )
    best, bad = (-np.inf, 0, None), 0
    for it in range(step, max_iter + 1, step):
        clf.set_params(max_iter=it)
        clf.fit(Xtr, ytr, sample_weight=wtr)
        p = clf.predict_proba(Xva)[:, 1]
        score = np.mean([average_precision_score(yva[gva == g], p[gva == g]) for g in GROUPS])
        if score > best[0] + 1e-4:
            best, bad = (score, it, copy.deepcopy(clf)), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best[2], best[1], best[0]


def fit_platt(p, y, g):
    """Per-product Platt scaling on the logit (strictly monotone: ranking metrics unchanged)."""
    models = {}
    for k in GROUPS:
        m = g == k
        if m.sum() and len(np.unique(y[m])) == 2:
            models[k] = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p[m])[:, None], y[m])
    return models


def apply_platt(p, g, models):
    out = p.copy()
    for k, mdl in models.items():
        m = g == k
        if m.any():
            out[m] = mdl.predict_proba(_logit(p[m])[:, None])[:, 1]
    return out


def f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


# ============================================================================ metrics
def product_metrics(y, p, thr, pi_ref=0.05):
    prev = y.mean()
    fpr_c, tpr_c, _ = roc_curve(y, p)
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    prec_ref = pi_ref * rec / max(pi_ref * rec + (1 - pi_ref) * fpr, 1e-12)   # precision re-based
    brier = brier_score_loss(y, p)
    return {"rows": len(y), "frauds": int(y.sum()), "prevalence": prev,
            "ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "AP_skill": (average_precision_score(y, p) - prev) / (1 - prev),   # 0 random, 1 perfect
            "R@1%FPR": float(tpr_c[fpr_c <= 0.01].max()) if (fpr_c <= 0.01).any() else 0.0,
            "thr": thr, "Precision": prec, "Recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-12),
            f"F1@pi={pi_ref:.0%}": 2 * prec_ref * rec / max(prec_ref + rec, 1e-12),
            "Brier": brier, "BSS": 1 - brier / max(prev * (1 - prev), 1e-12)}


def entity_bootstrap(y, p, uid, B=200, seed=0):
    """95% CIs resampling clients, not rows (rows of one client share a propagated label)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(np.asarray(uid))
    k, roc, ap = len(uniq), [], []
    for _ in range(B):
        w = np.bincount(rng.integers(0, k, k), minlength=k)[codes]
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        roc.append(roc_auc_score(y, p, sample_weight=w))
        ap.append(average_precision_score(y, p, sample_weight=w))
    q = lambda v: f"[{np.quantile(v, .025):.3f}, {np.quantile(v, .975):.3f}]"
    return q(roc), q(ap)


def evaluate(name, p_val_raw, p_te_raw, off, meta_va, meta_te, B=200):
    """offset -> per-product Platt (fit on val) -> per-product F1 threshold (val) -> test report."""
    gva, gte = meta_va["group"].values, meta_te["group"].values
    yva, yte = meta_va["y"].values, meta_te["y"].values
    pv = apply_offset(p_val_raw, meta_va["group"], off)
    pt = apply_offset(p_te_raw, meta_te["group"], off)
    platt = fit_platt(pv, yva, gva)
    pv_c, pt_c = apply_platt(pv, gva, platt), apply_platt(pt, gte, platt)
    rows, pred_all = [], np.zeros(len(yte), bool)
    for g in GROUPS:
        mv, mt = gva == g, gte == g
        thr = f1_threshold(yva[mv], pv_c[mv])
        r = product_metrics(yte[mt], pt_c[mt], thr)
        r["BSS_offset_only"] = 1 - brier_score_loss(yte[mt], pt[mt]) / (yte[mt].mean() * (1 - yte[mt].mean()))
        r["ROC_CI"], r["PR_CI"] = entity_bootstrap(yte[mt], pt_c[mt], meta_te.loc[mt, "uid"].values, B=B)
        r["setup"], r["group"] = name, g
        pred_all[mt] = pt_c[mt] >= thr
        rows.append(r)
    tp = int((pred_all & (yte == 1)).sum()); fp = int((pred_all & (yte == 0)).sum())
    fn = int((~pred_all & (yte == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    rows.append({"setup": name, "group": "OVERALL (per-product thr)", "rows": len(yte), "frauds": int(yte.sum()),
                 "prevalence": yte.mean(), "ROC_AUC": roc_auc_score(yte, pt_c),
                 "PR_AUC": average_precision_score(yte, pt_c), "Precision": prec, "Recall": rec,
                 "F1": 2 * prec * rec / max(prec + rec, 1e-12), "Brier": brier_score_loss(yte, pt_c)})
    return pd.DataFrame(rows)


# ============================================================================ generator audit
def _embed(frames, num_cols, cat_cols, top=20):
    ref, mats = frames[0], [[] for _ in frames]
    for c in num_cols:
        srt = np.sort(ref[c].dropna().values)
        for m, f in zip(mats, frames):
            x = f[c].values.astype(float)
            miss = np.isnan(x)
            u = np.searchsorted(srt, x, side="right") / max(len(srt), 1)
            u[miss] = 0.0
            m += [u, miss.astype(float)]
    for c in cat_cols:
        cats = ref[c].value_counts().index[:top]
        for m, f in zip(mats, frames):
            m += [(f[c].values == k).astype(float) / np.sqrt(2) for k in cats]
    return [np.column_stack(m) for m in mats]


def c2st(R, S, cols, cat_cols, n_max=3000, seed=0):
    n = min(len(R), len(S), n_max)
    X = pd.concat([R.sample(n, random_state=seed)[cols], S.sample(n, random_state=seed)[cols]])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=150, categorical_features=[c in cat_cols for c in cols],
                                         random_state=seed)
    p = cross_val_predict(clf, X, y, method="predict_proba",
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    return roc_auc_score(y, p)


def audit_cell(T, R, S, num_cols, cat_cols):
    """T = generator-train rows, R = entity-disjoint real holdout (same window), S = synthetic."""
    ks = [ks_2samp(R[c].dropna(), S[c].dropna()).statistic for c in num_cols
          if R[c].notna().sum() > 1 and S[c].notna().sum() > 1]
    miss = [abs(R[c].isna().mean() - S[c].isna().mean()) for c in num_cols]
    tvd = [0.5 * R[c].value_counts(normalize=True).subtract(S[c].value_counts(normalize=True),
                                                            fill_value=0).abs().sum() for c in cat_cols]
    Et, Er, Es = _embed([T, R, S], num_cols, cat_cols)
    nnb = NearestNeighbors(n_neighbors=1).fit(Et)
    dcr = np.median(nnb.kneighbors(Es[:3000])[0]) / max(np.median(nnb.kneighbors(Er[:3000])[0]), 1e-9)
    return {"n_hold": len(R), "KS_obs": np.mean(ks), "miss_gap": np.mean(miss), "cat_TVD": np.mean(tvd),
            "DCR_ratio": dcr, "C2ST": c2st(R, S, num_cols + cat_cols, cat_cols)}


def gate(a: dict, n_min=200) -> str:
    """ILLUSTRATIVE thresholds - calibrate them so the two control generators in
    fraud_synth_pipeline.py are rejected. DCR is two-sided: << 1 memorizes, >> 1 is off-manifold."""
    if a["n_hold"] < n_min:
        return "insufficient"
    ok = (a["KS_obs"] <= 0.10 and a["miss_gap"] <= 0.05 and a["cat_TVD"] <= 0.10
          and 0.8 <= a["DCR_ratio"] <= 1.5 and a["C2ST"] <= 0.70)
    return "accept" if ok else "reject"


# ============================================================================ main
def run(args):
    t_start = time.time()
    df = load(args.tx, args.id)
    vcols = reduce_v(df, df["day"] < 120)
    need = set(NUM_BASE) | set(CAT_BASE) | set(vcols) | {
        "TransactionDT", "TransactionAmt", "D1", "D2", "D3", "D4", "D10", "D11", "D15",
        "uid", "day", "group", "y"}
    df = df[[c for c in df.columns if c in need]]                 # lower peak memory
    df = past_only_features(df)
    tr_m = df["day"] < 120
    va_m = (df["day"] >= 127) & (df["day"] < 153)
    te_m = df["day"] >= 160
    diagnostics(df, {"train": tr_m, "val": va_m, "test": te_m})

    num_cols = [c for c in NUM_BASE + vcols if c in df]
    cat_cols = [c for c in CAT_BASE if c in df]
    vocab = fit_vocab(df[tr_m], cat_cols)                          # vocabulary from TRAIN only
    X = pd.concat([df[num_cols].astype(np.float32), encode_cats(df, vocab)], axis=1)
    meta = df[["group", "y", "uid", "day"]].copy()
    del df
    feats = num_cols + cat_cols
    cat_mask = [c in cat_cols for c in feats]
    print(f"\n[*] {len(num_cols)} numeric ({len(vcols)} V) + {len(cat_cols)} categorical features")

    Xtr, Xva, Xte = X[tr_m.values], X[va_m.values], X[te_m.values]
    mtr, mva, mte = meta[tr_m.values], meta[va_m.values], meta[te_m.values]
    counts = mtr.groupby("group")["y"].agg(rows="count", fraud="sum")
    print(f"[*] train/val/test rows = {len(Xtr)}/{len(Xva)}/{len(Xte)}")

    setups = {"B0 raw": None,
              "B1 global class weight": "global",
              "B2 uniform tau=0": Plan(),
              "B2 tau=0.5": Plan(tau=0.5),
              "B2 tau=0, r*_W=0.05": Plan(r_star_by_group={"G1_W": 0.05})}
    if args.quick:
        setups = {k: setups[k] for k in ["B0 raw", "B2 uniform tau=0"]}
    if args.skip_baselines:
        setups = {}

    def weights_for(spec, groups, y, is_syn=None, P=None):
        if spec is None:
            return np.ones(len(y))
        if spec == "global":
            rate = y.mean()
            w = np.where(y == 1, (0.2 / 0.8) / (rate / (1 - rate)), 1.0)
            return w * len(w) / w.sum()
        return row_weights(groups, y, P if P is not None else design_plan(counts, spec), is_syn)

    reports = []
    for name, spec in setups.items():
        t0 = time.time()
        w = weights_for(spec, mtr["group"], mtr["y"].values)
        off = realized_offsets(mtr["group"], mtr["y"], w)
        clf, n_it, sc = fit_gbdt(Xtr, mtr["y"].values, w, Xva, mva["y"].values, mva["group"].values,
                                 cat_mask, max_iter=args.max_iter)
        rep = evaluate(name, clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1], off, mva, mte,
                       B=args.boot)
        reports.append(rep)
        print(f"[+] {name:<26} trees={n_it:<5} val mean-PR-AUC={sc:.4f}  ({time.time() - t0:.0f}s)")

    if args.gen:
        reports.append(run_generator(Xtr, mtr, Xva, mva, Xte, mte, counts, num_cols, cat_cols,
                                     cat_mask, args))

    out = pd.concat(reports, ignore_index=True)
    cols = ["setup", "group", "rows", "frauds", "prevalence", "ROC_AUC", "ROC_CI", "PR_AUC", "PR_CI",
            "AP_skill", "R@1%FPR", "thr", "Precision", "Recall", "F1", "F1@pi=5%", "BSS",
            "BSS_offset_only"]
    pd.set_option("display.width", 250)
    print("\n" + "=" * 120 + "\nTEST (days >= 160), real rows only\n" + "=" * 120)
    print(out[[c for c in cols if c in out]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    prod = out[out["group"].isin(GROUPS)]
    summ = prod.groupby("setup", sort=False).agg(W_ROC=("ROC_AUC", lambda s: s.iloc[0]),
                                                 W_PR_AUC=("PR_AUC", lambda s: s.iloc[0]),
                                                 W_F1=("F1", lambda s: s.iloc[0]),
                                                 mean_ROC=("ROC_AUC", "mean"),
                                                 worst_AP_skill=("AP_skill", "min"),
                                                 mean_R1FPR=("R@1%FPR", "mean"))
    print("\nUNIFORMITY SUMMARY (W = G1_W)\n" + summ.to_string(float_format=lambda v: f"{v:.4f}"))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\n[*] saved {args.out}")
    print(f"[*] total {time.time() - t_start:.0f}s")


def run_generator(Xtr, mtr, Xva, mva, Xte, mte, counts, num_cols, cat_cols, cat_mask, args):
    from fixed_ddpm import StableTabDDPM
    feats = num_cols + cat_cols
    parts = [d if yv == 1 else d.sample(min(len(d), args.gen_legit_cap), random_state=0)
             for (_, yv), d in mtr.groupby(["group", "y"])]      # all fraud + capped legit per group
    pool = pd.concat(parts)
    gi, hi = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(pool, groups=pool["uid"]))
    g_tr, g_ho = pool.iloc[gi], pool.iloc[hi]
    cell = (g_tr["group"] + "|" + g_tr["y"].astype(str)).values
    t0 = time.time()
    gen = StableTabDDPM(num_cols, cat_cols, timesteps=args.gen_T).fit(
        Xtr.loc[g_tr.index, feats], cond=cell, epochs=args.gen_epochs, batch_size=512)
    print(f"\n[*] pooled conditional generator trained on {len(g_tr)} rows ({time.time() - t0:.0f}s)")

    P_full = design_plan(counts, Plan(), allow_syn=GROUPS)       # which cells NEED rows
    passed, audits = [], []
    for g in GROUPS:
        sel_tr = g_tr.index[(g_tr["group"] == g) & (g_tr["y"] == 1)]
        sel_ho = g_ho.index[(g_ho["group"] == g) & (g_ho["y"] == 1)]
        S = gen.sample(max(len(sel_ho), 500), cell=f"{g}|1")
        a = audit_cell(Xtr.loc[sel_tr, feats], Xtr.loc[sel_ho, feats], S, num_cols, cat_cols)
        a["group"], a["needs_syn"], a["decision"] = g, P_full.loc[g, "syn_fraud"] > 0, gate(a)
        audits.append(a)
        if a["needs_syn"] and (a["decision"] == "accept" or args.no_gate):
            passed.append(g)
    print("[audit] fraud cells (holdout is entity-disjoint, same time window):")
    print(pd.DataFrame(audits).set_index("group").round(3).to_string())

    P = design_plan(counts, Plan(), allow_syn=passed)               # F4: consistent fallback
    syn = [gen.sample(int(P.loc[g, "syn_fraud"]), cell=f"{g}|1") for g in passed]
    if syn:
        S = pd.concat(syn, ignore_index=True)[feats].astype(Xtr.dtypes[feats].to_dict())
        ms = pd.DataFrame({"group": np.repeat(passed, [int(P.loc[g, "syn_fraud"]) for g in passed]),
                           "y": 1, "uid": "synthetic", "day": -1})
        X_all = pd.concat([Xtr[feats], S], ignore_index=True)
        m_all = pd.concat([mtr.reset_index(drop=True), ms], ignore_index=True)
        is_syn = np.r_[np.zeros(len(Xtr), bool), np.ones(len(S), bool)]
        name = f"G1 B2 + synthetic {passed}" + (" [NO GATE - ablation]" if args.no_gate else "")
    else:
        X_all, m_all, is_syn = Xtr[feats], mtr, None
        name = "G1 gate failed -> B2 (consistent fallback)"
    w = row_weights(m_all["group"], m_all["y"].values, P, is_syn)
    off = realized_offsets(m_all["group"], m_all["y"], w, is_syn)
    clf, n_it, sc = fit_gbdt(X_all, m_all["y"].values, w, Xva[feats], mva["y"].values,
                             mva["group"].values, cat_mask, max_iter=args.max_iter)
    print(f"[+] {name}: trees={n_it} val mean-PR-AUC={sc:.4f}")
    return evaluate(name, clf.predict_proba(Xva[feats])[:, 1], clf.predict_proba(Xte[feats])[:, 1],
                    off, mva, mte, B=args.boot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IEEE-CIS uniform fraud pipeline v2")
    ap.add_argument("--tx", "--tx_path", dest="tx", required=True)
    ap.add_argument("--id", "--id_path", dest="id", default=None)
    ap.add_argument("--quick", action="store_true", help="only B0 raw vs B2 uniform, fewer trees")
    ap.add_argument("--gen", action="store_true", help="train pooled conditional generator + gate")
    ap.add_argument("--gen_epochs", type=int, default=100)
    ap.add_argument("--gen_T", type=int, default=200)
    ap.add_argument("--gen_legit_cap", type=int, default=20_000)
    ap.add_argument("--boot", type=int, default=200, help="entity-bootstrap replicates for CIs")
    ap.add_argument("--max_iter", type=int, default=1500, help="max trees (early-stopped on val)")
    ap.add_argument("--no_gate", action="store_true",
                    help="ABLATION: use synthetic rows even if the gate rejects them")
    ap.add_argument("--skip_baselines", action="store_true", help="with --gen: run only the G1 setup")
    ap.add_argument("--out", default="results_v2.csv")
    run(ap.parse_args())
