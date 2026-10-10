"""Fake data with IEEE-CIS column names/types, product-dependent missingness and entity-level
fraud labels. ONLY for smoke-testing the pipeline: it says nothing about real performance."""
import numpy as np, pandas as pd

def make(n_ent=30_000, seed=0, out_tx="fake_train_transaction.csv", out_id="fake_train_identity.csv"):
    r = np.random.default_rng(seed)
    prods = np.array(list("WCRHS")); pp = [.745, .116, .064, .056, .019]
    rate = {"W": .02, "C": .117, "R": .038, "H": .048, "S": .059}
    ep = r.choice(prods, n_ent, p=pp)
    ef = r.random(n_ent) < pd.Series(ep).map(rate).values
    card1 = r.integers(1000, 18000, n_ent); addr1 = np.where(r.random(n_ent) < .1, np.nan, r.integers(100, 540, n_ent))
    start = r.integers(-200, 170, n_ent)
    ntx = r.poisson(3, n_ent) + 1 + 2 * ef
    idx = np.repeat(np.arange(n_ent), ntx); n = len(idx)
    day = np.minimum(np.maximum(start[idx], 1) + r.integers(0, 25, n), 182)
    dt = day * 86400 + r.integers(0, 86400, n)
    p = ep[idx]; y = ef[idx].astype(int); W = p == "W"
    amt = np.round(np.exp(r.normal(4, 1, n)) * (1 + .8 * y * (p != "W")), 2)
    df = pd.DataFrame({"TransactionID": np.arange(n) + 2987000, "isFraud": y, "TransactionDT": dt,
                       "TransactionAmt": amt, "ProductCD": p, "card1": card1[idx],
                       "card2": r.integers(100, 600, n).astype(float), "card3": r.choice([150., 185.], n),
                       "card4": r.choice(["visa", "mastercard", "american express", "discover"], n, p=[.65, .32, .02, .01]),
                       "card5": r.integers(100, 240, n).astype(float),
                       "card6": r.choice(["debit", "credit"], n, p=[.75, .25]),
                       "addr1": addr1[idx], "addr2": np.where(np.isnan(addr1[idx]), np.nan, 87.),
                       "dist1": np.where(W & (r.random(n) < .5), r.exponential(30, n).round(), np.nan),
                       "dist2": np.where(~W & (r.random(n) < .3), r.exponential(200, n).round(), np.nan),
                       "P_emaildomain": r.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], n),
                       "R_emaildomain": np.where(W, None, r.choice(["gmail.com", "hotmail.com", None], n))})
    for i in range(1, 15):
        df[f"C{i}"] = (r.negative_binomial(1, .4, n) + 3 * y * (p == "C") * (i % 2)).astype(float)
    for i in range(1, 16):
        d = (day - start[idx]).astype(float) if i == 1 else r.integers(0, 600, n).astype(float)
        df[f"D{i}"] = np.where(r.random(n) < (.05 if i in (1, 10, 15) else .6), np.nan, d)
    for i in range(1, 10):
        vals = ["M0", "M1", "M2"] if i == 4 else ["T", "F"]
        df[f"M{i}"] = np.where(W & (r.random(n) < .9), r.choice(vals, n), None)
    for i in range(1, 41):            # 5 NaN blocks, block 1-2 mostly for W
        block = (i - 1) // 8
        present = (W if block < 2 else ~W) & (r.random(n) < .95)
        v = r.exponential(1, n) + (1.5 * y * W if block == 0 and i % 3 == 0 else 0)
        df[f"V{i}"] = np.where(present, np.round(v, 2), np.nan)
    has_id = np.where(W, r.random(n) < .02, r.random(n) < .9)
    ident = pd.DataFrame({"TransactionID": df["TransactionID"],
                          "id_01": r.choice([-5., -10., -20., 0.], n), "id_02": r.integers(1, 9e5, n).astype(float),
                          "id_05": r.integers(-10, 20, n).astype(float), "id_06": r.integers(-50, 0, n).astype(float),
                          "id_12": r.choice(["Found", "NotFound"], n), "id_13": r.integers(10, 60, n).astype(float),
                          "id_15": r.choice(["New", "Found", "Unknown"], n), "id_16": r.choice(["Found", "NotFound"], n),
                          "id_17": r.integers(100, 230, n).astype(float), "id_19": r.integers(100, 600, n).astype(float),
                          "id_20": r.integers(100, 600, n).astype(float),
                          "id_31": np.where(y & ~W, r.choice(["chrome 63.0", "mobile safari 11.0"], n),
                                            r.choice(["chrome 62.0", "safari generic", "edge 16.0", "firefox"], n)),
                          "id_38": r.choice(["T", "F"], n), "DeviceType": r.choice(["desktop", "mobile"], n),
                          "DeviceInfo": r.choice(["Windows", "iOS Device", "MacOS", "Trident/7.0", "SM-G9650"], n)})
    df.to_csv(out_tx, index=False); ident[has_id].to_csv(out_id, index=False)
    print(f"fake rows={n}  frauds={y.sum()}  identity rows={has_id.sum()}")

if __name__ == "__main__":
    make()
