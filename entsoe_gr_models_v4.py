"""
Προϋπόθεση: έχει τρέξει ήδη το  python entsoe_gr_pipeline_v2.py build <περίοδος>
           (δημιουργεί το dataset_hourly_<περίοδος>.parquet).
Όλα τα αποτελέσματα αποθηκεύονται στον φάκελο results_<περίοδος>/ (και ανά εκδοχή στους υποφακέλους
με_καιρό/ και χωρίς_καιρό/).
"""
import os, sys, json, platform, warnings
from itertools import product
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ============================ ΡΥΘΜΙΣΕΙΣ ============================
SEED = 42
TEST_START = {"crisis": "2022-01-01",   # όλο το 2022 ως σύνολο ελέγχου (walk-forward)
              "recent": "2026-04-01"}
REFIT_EVERY = 30                         # ημέρες ανάμεσα σε επανεκπαιδεύσεις
SCALE = 100.0                            # κλιμάκωση τιμής για το ARIMAX (βοηθά τη σύγκλιση)
GRID = {"p": [0, 1, 2], "q": [0, 1, 2], "P": [0, 1], "Q": [0, 1]}   # d = 1, D = 0, s = 7
VAL_FRAC = 0.2                           # τελευταίο 20% της εκπαίδευσης = επικύρωση (κατώφλι spikes)
# max_features = 1/3: σε κάθε διακλάδωση εξετάζεται τυχαίο 1/3 των μεταβλητών, όπως στη θεωρία (§3.3).
# (Η προεπιλογή του scikit-learn για παλινδρόμηση είναι όλες οι μεταβλητές, δηλαδή απλό bagging.)
RF_PARAMS = {"n_estimators": 400, "min_samples_leaf": 3, "max_features": 1/3, "random_state": SEED, "n_jobs": -1}
CLF_PARAMS = {"n_estimators": 500, "min_samples_leaf": 3, "class_weight": "balanced",
              "random_state": SEED, "n_jobs": -1}
TTF_FILE = "data_raw/ttf_daily.csv"      # προαιρετικό: στήλες date, ttf (€/MWh)

WEATHER = ["w_temperature_2m", "w_wind_speed_100m", "w_shortwave_radiation", "w_cloud_cover"]
EXOG = ["residual_load", "res_share", "net_import", "w_temperature_2m", "w_wind_speed_100m",
        "w_shortwave_radiation", "holiday", "spike_lag1"]
RF_EXTRA = ["load_fc", "res_fc", "w_cloud_cover", "dow", "month", "price_lag1", "price_lag7"]
# Ο καιρός είναι παρατηρούμενος (όχι πρόβλεψη), άρα δεν θα ήταν γνωστός τη στιγμή της πρόβλεψης.
# Η τελική εκδοχή είναι χωρίς καιρό· η εκδοχή με καιρό τρέχει μόνο ως έλεγχος (ablation).
# Κάθε εκδοχή: weather = αν χρησιμοποιούνται καιρικές μεταβλητές, netimp = τι γίνεται με τις καθαρές εισαγωγές:
#   "same" = της ίδιας ημέρας (όπως στο κύριο μοντέλο), "none" = αφαιρούνται, "lag1" = της προηγούμενης ημέρας.
# Η πρώτη είναι η τελική εκδοχή. Οι δύο τελευταίες είναι έλεγχος ευαισθησίας για το net_import: οι προγραμματισμένες
# ανταλλαγές επόμενης ημέρας προκύπτουν από την εκκαθάριση της αγοράς, άρα ΔΕΝ είναι γνωστές πριν από το κλείσιμο.
# ΤΕΛΙΚΗ εκδοχή: χωρίς καιρό και με net_import της ΠΡΟΗΓΟΥΜΕΝΗΣ ημέρας. Επιλέχθηκε με κριτήριο αρχής (η πληροφορία πρέπει να είναι
# γνωστή πριν από το κλείσιμο της αγοράς), όχι με βάση την ακρίβεια: η εκδοχή με net_import της ίδιας ημέρας είναι ακριβέστερη
# για το ARIMAX, αλλά το net_import της ημέρας d προκύπτει από την ίδια εκκαθάριση με την τιμή και δεν είναι γνωστό εκ των προτέρων.
VARIANTS = {
    "τελική": {"weather": False, "netimp": "lag1"},
    "με_καιρό": {"weather": True, "netimp": "lag1"},
    "χωρίς_net_import": {"weather": False, "netimp": "none"},
    "net_import_ίδιας_ημέρας": {"weather": False, "netimp": "same"},
}
FINAL = "τελική"
# Στους ελέγχους ευαισθησίας του net_import χρησιμοποιείται η τάξη ARIMAX της τελικής εκδοχής (χωρίς νέα αναζήτηση AIC).
REUSE_ORDER = {"χωρίς_net_import", "net_import_ίδιας_ημέρας"}
MODELS = ["naive1", "naive7", "arx", "arimax", "rf"]
# Παραλλαγές RF (v4): (στόχος = μεταβολή;, παράθυρο εκπαίδευσης σε ημέρες ή None για διευρυνόμενο)
ROLL_WINDOW = 60          # ημέρες· οι Michalakopoulos κ.ά. (2026) βρίσκουν ότι τα παράθυρα 45–60 ημερών αποδίδουν καλά
EXTRA_RF = {"rf_diff": (True, None), "rf_roll": (False, ROLL_WINDOW), "rf_diff_roll": (True, ROLL_WINDOW)}
EXTRA = list(EXTRA_RF)
COLORS = {"y": "#000000", "arimax": "#1F4E79", "rf": "#C55A11"}   # σταθερή παλέτα σε όλα τα γραφήματα


# ============================ ΔΕΔΟΜΕΝΑ ============================
def load_ttf(index):
    """Τιμή TTF της προηγούμενης ημέρας συναλλαγών (γνωστή πριν από το κλείσιμο της αγοράς)."""
    if not os.path.exists(TTF_FILE):
        return None
    t = pd.read_csv(TTF_FILE, parse_dates=["date"]).set_index("date")["ttf"].sort_index()
    t.index = t.index.tz_localize("Europe/Athens")
    t = t.reindex(pd.date_range(t.index.min(), index.max(), freq="D", tz="Europe/Athens")).ffill()
    return t.shift(1).reindex(index)


def daily_dataset(h):
    h = h.copy()
    h.index = h.index.tz_convert("Europe/Athens")
    d = h.resample("D").agg({"price": "mean", "load_fc": "mean", "res_fc": "mean", "res_share": "mean",
                             "residual_load": "mean", "net_import": "mean", "w_temperature_2m": "mean",
                             "w_wind_speed_100m": "mean", "w_shortwave_radiation": "mean",
                             "w_cloud_cover": "mean", "holiday": "max", "spike": "max"})
    d["n_hours"] = h["price"].resample("D").count()
    d.loc[d["n_hours"] < 20, :] = np.nan
    d = d.drop(columns="n_hours").asfreq("D")
    d["dow"], d["month"] = d.index.dayofweek, d.index.month
    d["price_lag1"], d["price_lag7"] = d["price"].shift(1), d["price"].shift(7)
    d["spike_lag1"] = d["spike"].shift(1)
    for k in range(1, 7):
        d[f"dow_{k}"] = (d["dow"] == k).astype(float)
    ttf = load_ttf(d.index)
    if ttf is not None:
        d["ttf_lag1"] = ttf
    d = d.dropna()
    # Καθαρές εισαγωγές της προηγούμενης ημέρας (ο πίνακας δεν έχει κενές ημέρες, άρα shift(1) = προηγούμενη ημέρα).
    # Η πρώτη γραμμή συμπληρώνεται με την ίδια ημέρα, ώστε να μη χαθεί ημέρα και να μη αλλάξει το μέγεθος του συνόλου.
    d["net_import_lag1"] = d["net_import"].shift(1).fillna(d["net_import"])
    return d


def feature_sets(d, weather=True, netimp="same"):
    keep = (lambda c: True) if weather else (lambda c: c not in WEATHER)
    ex = [c for c in EXOG if keep(c)]
    if netimp == "none":
        ex = [c for c in ex if c != "net_import"]
    elif netimp == "lag1":
        ex = ["net_import_lag1" if c == "net_import" else c for c in ex]
    exog = ex + (["ttf_lag1"] if "ttf_lag1" in d else [])
    rf = exog + [c for c in RF_EXTRA if keep(c)]
    arx = exog + ["price_lag1", "price_lag7"] + [f"dow_{k}" for k in range(1, 7)]
    return exog, rf, arx


# ============================ ΑΡΙΘΜΟΙ ΓΙΑ ΤΟ ΚΕΦ. 1 ============================
def chapter1_numbers(h, d):
    p = h["price"]
    return pd.Series({
        "ώρες": len(p),
        "μέση ωριαία τιμή (€/MWh)": p.mean(),
        "διάμεσος": p.median(),
        "τυπική απόκλιση": p.std(),
        "ελάχιστη ωριαία τιμή": p.min(),
        "μέγιστη ωριαία τιμή": p.max(),
        "5ο εκατοστημόριο": p.quantile(0.05),
        "95ο εκατοστημόριο": p.quantile(0.95),
        "ώρες με τιμή <= 0": int((p <= 0).sum()),
        "ώρες με τιμή < 0": int((p < 0).sum()),
        "ποσοστό ωρών με τιμή <= 0": (p <= 0).mean(),
        "ποσοστό ωρών με spike": h["spike"].mean(),
        "ελάχιστη μέση ημερήσια τιμή": d["price"].min(),
        "μέγιστη μέση ημερήσια τιμή": d["price"].max(),
        "μέσο ημερήσιο εύρος (max-min ώρας)": (h["price"].resample("D").max()
                                              - h["price"].resample("D").min()).mean(),
    })


# ============================ ARIMAX ============================
def fit_sarimax(y, X, order, seasonal):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    res = SARIMAX(y / SCALE, exog=X, order=order, seasonal_order=seasonal,
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False, method="lbfgs", maxiter=500)
    return res, bool(res.mle_retvals.get("converged", False))


def standardize(train, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    return lambda df: (df[cols] - mu) / sd


def arimax_grid(train, exog):
    z = standardize(train, exog)
    rows = []
    for p, q, P, Q in product(GRID["p"], GRID["q"], GRID["P"], GRID["Q"]):
        order, seas = (p, 1, q), (P, 0, Q, 7)
        try:
            res, conv = fit_sarimax(train["price"], z(train), order, seas)
            rows.append({"order": str(order), "seasonal": str(seas), "aic": res.aic, "bic": res.bic,
                         "converged": conv})
        except Exception as e:
            rows.append({"order": str(order), "seasonal": str(seas), "aic": np.nan, "bic": np.nan,
                         "converged": False, "error": str(e)[:80]})
    g = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    ok = g[g["converged"] & g["aic"].notna()]
    best = ok.iloc[0] if len(ok) else g.iloc[0]
    return g, eval(best["order"]), eval(best["seasonal"])


def stationarity(y):
    from statsmodels.tsa.stattools import adfuller, kpss
    adf = adfuller(y.dropna())
    kp = kpss(y.dropna(), regression="c", nlags="auto")
    return pd.Series({"ADF στατιστικό": adf[0], "ADF p": adf[1], "KPSS στατιστικό": kp[0], "KPSS p": kp[1]})


# ============================ WALK-FORWARD ============================
def walk_forward(d, test_start, exog, rf_feats, arx_feats, order=None, seasonal=None, use_arimax=True, extras=False):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    test_idx = d.index[d.index >= pd.Timestamp(test_start, tz=d.index.tz)]
    out = pd.DataFrame(index=test_idx, dtype=float)
    out["y"] = d.loc[test_idx, "price"]
    out["spike"] = d.loc[test_idx, "spike"]
    out["naive1"] = d.loc[test_idx, "price_lag1"]
    out["naive7"] = d.loc[test_idx, "price_lag7"]
    for m in ("arx", "arimax", "rf") + (tuple(EXTRA) if extras else ()):
        out[m] = np.nan
    log = []
    for b in range(0, len(test_idx), REFIT_EVERY):
        blk = test_idx[b:b + REFIT_EVERY]
        train = d.loc[d.index < blk[0]]
        conv = None
        if use_arimax:
            z = standardize(train, exog)
            res, conv = fit_sarimax(train["price"], z(train), order, seasonal)
            for t in blk:                                   # πρόβλεψη μίας ημέρας με ενημέρωση κατάστασης
                out.loc[t, "arimax"] = float(res.forecast(1, exog=z(d.loc[[t]])).iloc[0]) * SCALE
                res = res.append(d.loc[[t], "price"] / SCALE, exog=z(d.loc[[t]]), refit=False)
        rf = RandomForestRegressor(**RF_PARAMS).fit(train[rf_feats], train["price"])
        out.loc[blk, "rf"] = rf.predict(d.loc[blk, rf_feats])
        if extras:
            for name_, (diff_, win_) in EXTRA_RF.items():
                tr_ = train.tail(win_) if win_ else train
                y_tr = (tr_["price"] - tr_["price_lag1"]) if diff_ else tr_["price"]
                m_ = RandomForestRegressor(**RF_PARAMS).fit(tr_[rf_feats], y_tr)
                pred_ = m_.predict(d.loc[blk, rf_feats])
                out.loc[blk, name_] = pred_ + d.loc[blk, "price_lag1"].values if diff_ else pred_
        arx = make_pipeline(StandardScaler(), LinearRegression()).fit(train[arx_feats], train["price"])
        out.loc[blk, "arx"] = arx.predict(d.loc[blk, arx_feats])
        log.append({"block_start": blk[0].date(), "block_end": blk[-1].date(), "train_days": len(train),
                    "arimax_converged": conv})
        print(f"  μπλοκ {blk[0].date()} -> {blk[-1].date()} (εκπαίδευση {len(train)} ημέρες, "
              f"σύγκλιση ARIMAX: {conv})")
    if not use_arimax:
        out = out.drop(columns="arimax")
    return out, pd.DataFrame(log)


# ============================ ΜΕΤΡΙΚΕΣ ============================
def smape(y, p): return 100 * np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9))

def metrics_table(out, train_price, models=None):
    models = models or [m for m in MODELS if m in out]
    mae7 = np.mean(np.abs(out["y"] - out["naive7"]))
    tp = train_price.values
    mase_scale = np.mean(np.abs(tp[7:] - tp[:-7]))          # MAE του naive7 μέσα στο σύνολο εκπαίδευσης
    rows = []
    for m in models:
        e = out["y"] - out[m]
        mae = np.mean(np.abs(e))
        rows.append({"μοντέλο": m, "MAE": mae, "RMSE": np.sqrt(np.mean(e ** 2)), "sMAPE %": smape(out["y"], out[m]),
                     "rMAE": mae / mae7, "MASE": mae / mase_scale, "bias (πρόβλ.-πραγμ.)": -e.mean(),
                     "R2": 1 - np.sum(e ** 2) / np.sum((out["y"] - out["y"].mean()) ** 2)})
    return pd.DataFrame(rows).set_index("μοντέλο")


def dm_test(e1, e2, power=1, h=1):
    """Diebold–Mariano: αρνητικό στατιστικό = μικρότερο σφάλμα για το 1ο μοντέλο.
    Διακύμανση Newey–West με L = floor(n^(1/3)) και διόρθωση Harvey–Leybourne–Newbold (κατανομή t)."""
    dlt = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    n = len(dlt); m = dlt.mean(); c = dlt - m
    L = max(h - 1, int(np.floor(n ** (1 / 3))))
    g = [np.sum(c[k:] * c[:n - k]) / n for k in range(L + 1)]
    var = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    if var <= 0:
        return np.nan, np.nan
    stat = m / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def dm_table(out):
    models = [m for m in MODELS if m in out]
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            ea, eb = out["y"] - out[a], out["y"] - out[b]
            s1, p1 = dm_test(ea, eb, 1); s2, p2 = dm_test(ea, eb, 2)
            rows.append({"σύγκριση": f"{a} – {b}", "DM |e|": s1, "p |e|": p1, "DM e²": s2, "p e²": p2})
    return pd.DataFrame(rows)


def monthly_mae(out, models=None):
    models = models or [m for m in MODELS if m in out]
    g = out.groupby(out.index.to_period("M"))
    return pd.DataFrame({m: g.apply(lambda x: np.mean(np.abs(x["y"] - x[m]))) for m in models})


def spike_regression_eval(out):
    """Ακρίβεια των μοντέλων ξεχωριστά σε ημέρες με spike, χωρίς spike, και στο ακριβότερο 10%."""
    models = [m for m in MODELS if m in out]
    groups = {"ημέρες με spike": out["spike"] == 1, "ημέρες χωρίς spike": out["spike"] == 0,
              "ακριβότερο 10% ημερών": out["y"] >= out["y"].quantile(0.9)}
    rows = []
    for gname, mask in groups.items():
        sub = out[mask]
        for m in models:
            e = sub["y"] - sub[m]
            rows.append({"ομάδα": gname, "ημέρες": int(mask.sum()), "μοντέλο": m,
                         "MAE": np.mean(np.abs(e)), "bias": -e.mean()})
    return pd.DataFrame(rows)


def spike_classifier(d, test_start, feats):
    """Κατώφλι από σύνολο επικύρωσης (μεγιστοποίηση F1) και σύγκριση με μοντέλο εμμονής."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score, f1_score
    cut = pd.Timestamp(test_start, tz=d.index.tz)
    tr, te = d[d.index < cut], d[d.index >= cut]
    k = int(len(tr) * (1 - VAL_FRAC)); itr, val = tr.iloc[:k], tr.iloc[k:]
    if tr["spike"].sum() < 5 or te["spike"].sum() == 0 or itr["spike"].nunique() < 2:
        return pd.DataFrame([{"σημείωση": "πολύ λίγα spikes για αξιόπιστη αξιολόγηση"}]), None, None
    clf = RandomForestClassifier(**CLF_PARAMS).fit(itr[feats], itr["spike"])
    pv = clf.predict_proba(val[feats])[:, 1]
    grid = np.arange(0.05, 0.96, 0.05)
    f1s = [f1_score(val["spike"], pv >= t, zero_division=0) for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    clf = RandomForestClassifier(**CLF_PARAMS).fit(tr[feats], tr["spike"])
    pt = clf.predict_proba(te[feats])[:, 1]
    rows = []
    for name, pred, score in [(f"RF (κατώφλι επικύρωσης {thr:.2f})", pt >= thr, pt),
                              ("RF (κατώφλι 0,50)", pt >= 0.5, pt),
                              ("εμμονή (spike χθες)", te["spike_lag1"].values == 1, te["spike_lag1"].values)]:
        p, r, f, _ = precision_recall_fscore_support(te["spike"], pred, average="binary", zero_division=0)
        rows.append({"μοντέλο": name, "precision": p, "recall": r, "F1": f,
                     "PR-AUC": average_precision_score(te["spike"], score),
                     "ημέρες spike στο test": int(te["spike"].sum()), "ημέρες test": len(te),
                     "PR-AUC τυχαίας πρόβλεψης": te["spike"].mean()})
    probs = pd.DataFrame({"spike": te["spike"].values, "proba": pt, "persistence": te["spike_lag1"].values,
                          "threshold": thr}, index=te.index)
    return pd.DataFrame(rows), thr, probs


def permutation_imp(d, test_start, feats):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    cut = pd.Timestamp(test_start, tz=d.index.tz)
    tr, te = d[d.index < cut], d[d.index >= cut]
    rf = RandomForestRegressor(**RF_PARAMS).fit(tr[feats], tr["price"])
    pi = permutation_importance(rf, te[feats], te["price"], n_repeats=20, random_state=SEED,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    return pd.DataFrame({"αύξηση MAE (μέσος)": pi.importances_mean, "τυπ. απόκλιση": pi.importances_std,
                         "impurity importance": rf.feature_importances_}, index=feats
                        ).sort_values("αύξηση MAE (μέσος)", ascending=False)



def dm_extra(out):
    """DM των επιπλέον παραλλαγών RF έναντι των βασικών μοντέλων (αρνητικό στατιστικό = η παραλλαγή είναι ακριβέστερη)."""
    rows = []
    for a in [m for m in EXTRA if m in out]:
        for b in ("rf", "arimax", "arx", "naive7"):
            if b not in out:
                continue
            ea, eb = out["y"] - out[a], out["y"] - out[b]
            s1, p1 = dm_test(ea, eb, 1); s2, p2 = dm_test(ea, eb, 2)
            rows.append({"σύγκριση": f"{a} – {b}", "DM |e|": s1, "p |e|": p1, "DM e²": s2, "p e²": p2})
    return pd.DataFrame(rows)


def error_by_level(out, d, models):
    """Σφάλμα ανά επίπεδο τιμής: λόγος της τιμής της ημέρας προς τη μέση τιμή του συνόλου εκπαίδευσης του αντίστοιχου μπλοκ
    (διευρυνόμενο παράθυρο, όπως στο walk-forward), και ημέρες πάνω από το μέγιστο της εκπαίδευσης εκείνου του μπλοκ."""
    tm = pd.Series(index=out.index, dtype=float); tx = pd.Series(index=out.index, dtype=float)
    for b in range(0, len(out), REFIT_EVERY):
        blk = out.index[b:b + REFIT_EVERY]
        tr = d.loc[d.index < blk[0], "price"]
        tm[blk], tx[blk] = tr.mean(), tr.max()
    ratio = out["y"] / tm
    labels = ["μέχρι 1×", "1–1,5×", "1,5–2×", "πάνω από 2×"]
    grp = pd.cut(ratio, bins=[0, 1, 1.5, 2, np.inf], labels=labels)
    groups = {l: (grp == l).values for l in labels}
    groups["πάνω από το μέγιστο της εκπαίδευσης"] = (out["y"] > tx).values
    rows = []
    for g, mask in groups.items():
        sub = out[mask]
        for m in models:
            e = sub["y"] - sub[m]
            rows.append({"ομάδα": g, "ημέρες": int(mask.sum()), "μοντέλο": m,
                         "MAE": np.mean(np.abs(e)) if len(sub) else np.nan, "bias": -e.mean() if len(sub) else np.nan})
    long = pd.DataFrame(rows)
    wide = long.assign(τιμή=long.apply(lambda r: "–" if pd.isna(r["MAE"]) else f"{r['MAE']:.1f} ({r['bias']:+.1f})", axis=1)
                       ).pivot(index=["ομάδα", "ημέρες"], columns="μοντέλο", values="τιμή")[models].reset_index()
    wide = wide.set_index("ομάδα").loc[list(groups)].reset_index()
    return long, wide


def ljung_box(x, lag, model_df=0):
    """Ljung–Box: Q = n(n+2) Σ ρ_k² / (n-k), k = 1..lag· p από χ² με (lag - model_df) βαθμούς ελευθερίας."""
    x = np.asarray(x, dtype=float); x = x - x.mean(); n = len(x); den = np.sum(x * x)
    rho = np.array([np.sum(x[k:] * x[:n - k]) / den for k in range(1, lag + 1)])
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lag + 1)))
    return q, stats.chi2.sf(q, df=lag - model_df)


def arimax_diagnostics(train0, exog, order, seasonal):
    """Διαγνωστικά καταλοίπων του ARIMAX στο αρχικό σύνολο εκπαίδευσης: Ljung–Box, Jarque–Bera, ρίζες AR/MA."""
    z = standardize(train0, exog)
    res, conv = fit_sarimax(train0["price"], z(train0), order, seasonal)
    k = order[0] + order[2] + seasonal[0] + seasonal[2]                       # βαθμοί ελευθερίας του μοντέλου (p+q+P+Q)
    resid = pd.Series(np.asarray(res.resid), index=train0.index).iloc[10:]    # οι πρώτες τιμές επηρεάζονται από τη διάχυτη αρχικοποίηση
    rows = []
    for lag in [l for l in (7, 14, 21, 30) if l > k]:
        q, pv = ljung_box(resid, lag, k)
        rows.append({"έλεγχος": f"Ljung–Box (καθυστέρηση {lag})", "στατιστικό": float(q), "p": float(pv)})
    jb, jp = stats.jarque_bera(resid)
    rows.append({"έλεγχος": "Jarque–Bera (κανονικότητα)", "στατιστικό": float(jb), "p": float(jp)})
    try:
        ar, ma = np.abs(res.arroots), np.abs(res.maroots)
    except Exception:
        ar = ma = np.array([])
    rows.append({"έλεγχος": "ελάχιστο μέτρο ρίζας AR (>1: στάσιμο)", "στατιστικό": float(ar.min()) if len(ar) else np.nan, "p": np.nan})
    rows.append({"έλεγχος": "ελάχιστο μέτρο ρίζας MA (>1: αντιστρέψιμο)", "στατιστικό": float(ma.min()) if len(ma) else np.nan, "p": np.nan})
    rows.append({"έλεγχος": "σύγκλιση εκτίμησης (1 = ναι)", "στατιστικό": float(conv), "p": np.nan})
    return pd.DataFrame(rows), resid.rename("κατάλοιπο")


# ============================ ΓΡΑΦΗΜΑ ============================
def plot_forecasts(out, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ΠΡΟΣΟΧΗ: λείπει η matplotlib (pip install matplotlib)· το γράφημα παραλείπεται.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(out.index, out["y"], color=COLORS["y"], lw=1.4, label="Πραγματική τιμή")
    for m, lab in (("arimax", "ARIMAX"), ("rf", "Random Forest")):
        if m in out:
            ax.plot(out.index, out[m], color=COLORS[m], lw=1, label=lab)
    sp = out[out["spike"] == 1]
    ax.scatter(sp.index, sp["y"], s=10, color="#7F7F7F", zorder=3, label="Ημέρα με spike")
    ax.set_ylabel("€/MWh"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(frameon=False, ncol=4)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# ============================ ΑΝΑΠΑΡΑΓΩΓΙΜΟΤΗΤΑ ============================
def versions():
    from importlib.metadata import version, PackageNotFoundError
    v = {"python": platform.python_version(), "σύστημα": platform.platform()}
    for pkg in ("pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "entsoe-py", "pyarrow",
                "holidays", "requests", "matplotlib"):
        try:
            v[pkg] = version(pkg)
        except PackageNotFoundError:
            v[pkg] = "δεν βρέθηκε"
    return v


# ============================ ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ ============================
def weather_ablation(outs, met):
    """Σύγκριση εκδοχών με και χωρίς καιρό για κάθε μοντέλο που χρησιμοποιεί τον καιρό."""
    rows = []
    for m in ("arx", "arimax", "rf"):
        if m not in outs[FINAL]:
            continue
        w_, n_ = met["με_καιρό"].loc[m], met[FINAL].loc[m]
        y = outs[FINAL]["y"]
        stat, p = dm_test(y - outs[FINAL][m], y - outs["με_καιρό"][m], 1)
        rows.append({"μοντέλο": m, "MAE με καιρό": w_["MAE"], "MAE χωρίς καιρό": n_["MAE"],
                     "μεταβολή MAE %": 100 * (n_["MAE"] - w_["MAE"]) / w_["MAE"],
                     "sMAPE με καιρό": w_["sMAPE %"], "sMAPE χωρίς καιρό": n_["sMAPE %"],
                     "DM (χωρίς – με) |e|": stat, "p": p})
    return pd.DataFrame(rows)


def netimport_sensitivity(outs, mets):
    """Έλεγχος ευαισθησίας: τελική εκδοχή (net_import της προηγούμενης ημέρας, γνωστό πριν από το κλείσιμο) έναντι
    εκδοχής χωρίς net_import και έναντι εκδοχής με net_import της ίδιας ημέρας (ΔΕΝ είναι γνωστό εκ των προτέρων)."""
    rows = []
    for var in ("χωρίς_net_import", "net_import_ίδιας_ημέρας"):
        if var not in outs:
            continue
        for m in ("arx", "arimax", "rf"):
            if m not in outs[FINAL]:
                continue
            y = outs[FINAL]["y"]
            f_, v_ = mets[FINAL].loc[m], mets[var].loc[m]
            stat, p = dm_test(y - outs[var][m], y - outs[FINAL][m], 1)   # αρνητικό: η εκδοχή έχει μικρότερο σφάλμα
            rows.append({"εκδοχή": var, "μοντέλο": m, "MAE τελική": f_["MAE"], "MAE εκδοχής": v_["MAE"],
                         "μεταβολή MAE %": 100 * (v_["MAE"] - f_["MAE"]) / f_["MAE"],
                         "sMAPE τελική": f_["sMAPE %"], "sMAPE εκδοχής": v_["sMAPE %"],
                         "rMAE τελική": f_["rMAE"], "rMAE εκδοχής": v_["rMAE"],
                         "DM (εκδοχή – τελική) |e|": stat, "p": p})
    return pd.DataFrame(rows)


def run(period, h, use_arimax=True, use_extras=True):
    np.random.seed(SEED)
    R = f"results_{period}"; os.makedirs(R, exist_ok=True)
    d = daily_dataset(h)
    ts = TEST_START[period]
    train0 = d[d.index < pd.Timestamp(ts, tz=d.index.tz)]
    print(f"\n=== {period}: {len(d)} ημέρες, εκπαίδευση έως {ts}: {len(train0)} ημέρες, "
          f"test: {len(d) - len(train0)} ημέρες, TTF: {'ναι' if 'ttf_lag1' in d else 'όχι'} ===")
    d.to_csv(f"{R}/dataset_daily.csv")
    chapter1_numbers(h, d).to_csv(f"{R}/ch1_numbers.csv", header=["τιμή"])
    if use_arimax:
        stationarity(d["price"]).to_csv(f"{R}/stationarity.csv", header=["τιμή"])

    outs, mets, cfgs = {}, {}, {}
    for name, opt in VARIANTS.items():
        V = f"{R}/{name}"; os.makedirs(V, exist_ok=True)
        exog, rf_feats, arx_feats = feature_sets(d, opt["weather"], opt["netimp"])
        print(f"\n--- Εκδοχή: {name} ({len(exog)} εξωγενείς μεταβλητές) ---")
        order = seasonal = None
        if use_arimax:
            if name in REUSE_ORDER:
                order, seasonal = cfgs[FINAL]["arimax_order"], cfgs[FINAL]["arimax_seasonal"]
                print(f"Χρήση της τάξης της τελικής εκδοχής: SARIMAX{order}x{seasonal}")
            else:
                print("Αναζήτηση τάξης ARIMAX (AIC)…")
                grid, order, seasonal = arimax_grid(train0, exog)
                grid.to_csv(f"{V}/arimax_grid.csv", index=False)
                print(f"  επιλέχθηκε SARIMAX{order}x{seasonal}")
        print("Walk-forward…")
        out, log = walk_forward(d, ts, exog, rf_feats, arx_feats, order, seasonal, use_arimax,
                                extras=(use_extras and name == FINAL))
        out.to_csv(f"{V}/forecasts.csv"); log.to_csv(f"{V}/blocks.csv", index=False)
        met = metrics_table(out, train0["price"]); met.to_csv(f"{V}/metrics.csv")
        outs[name], mets[name] = out, met
        cfgs[name] = {"arimax_order": order, "arimax_seasonal": seasonal, "exog": exog,
                      "rf_features": rf_feats, "arx_features": arx_feats,
                      "arimax_blocks_not_converged": int((log["arimax_converged"] == False).sum())
                      if use_arimax else None}

    abl = weather_ablation(outs, mets); abl.to_csv(f"{R}/weather_ablation.csv", index=False)
    sens = netimport_sensitivity(outs, mets); sens.to_csv(f"{R}/netimport_sensitivity.csv", index=False)

    # ---- τελική εκδοχή (χωρίς καιρό): πλήρης αξιολόγηση ----
    out = outs[FINAL]; _, rf_feats, _ = feature_sets(d, VARIANTS[FINAL]["weather"], VARIANTS[FINAL]["netimp"])
    mets[FINAL].to_csv(f"{R}/metrics.csv")
    dm = dm_table(out); dm.to_csv(f"{R}/diebold_mariano.csv", index=False)
    monthly_mae(out).to_csv(f"{R}/monthly_mae.csv")
    spike_regression_eval(out).to_csv(f"{R}/spike_days_eval.csv", index=False)
    clf, thr, probs = spike_classifier(d, ts, rf_feats); clf.to_csv(f"{R}/spike_classifier.csv", index=False)
    if probs is not None:
        probs.to_csv(f"{R}/spike_proba.csv")
    pi = permutation_imp(d, ts, rf_feats); pi.to_csv(f"{R}/rf_permutation_importance.csv")
    # ---- v4: επιπλέον παραλλαγές RF, σφάλμα ανά επίπεδο τιμής, διαγνωστικά ARIMAX ----
    base_models = [m for m in MODELS if m in out]
    ext_models = [m for m in EXTRA if m in out]
    if ext_models:
        mex = metrics_table(out, train0["price"], base_models + ext_models); mex.to_csv(f"{R}/metrics_extended.csv")
        dmx = dm_extra(out); dmx.to_csv(f"{R}/dm_extra.csv", index=False)
        monthly_mae(out, base_models + ext_models).to_csv(f"{R}/monthly_mae_extended.csv")
    long_, wide_ = error_by_level(out, d, base_models + ext_models)
    long_.to_csv(f"{R}/error_by_level.csv", index=False); wide_.to_csv(f"{R}/error_by_level_wide.csv", index=False)
    diag = None
    if use_arimax:
        diag, resid = arimax_diagnostics(train0, feature_sets(d, VARIANTS[FINAL]["weather"], VARIANTS[FINAL]["netimp"])[0],
                                         cfgs[FINAL]["arimax_order"], cfgs[FINAL]["arimax_seasonal"])
        diag.to_csv(f"{R}/arimax_diagnostics.csv", index=False); resid.to_csv(f"{R}/arimax_residuals.csv")
    plot_forecasts(out, f"{R}/forecasts_{period}.png",
                   f"Μέση ημερήσια τιμή και προβλέψεις, σύνολο ελέγχου ({period})")

    cfg = {"period": period, "test_start": ts, "refit_every_days": REFIT_EVERY, "seed": SEED,
           "final_variant": FINAL, "variants": cfgs, "arimax_grid": GRID, "y_scale": SCALE,
           "rf_params": RF_PARAMS, "clf_params": CLF_PARAMS, "spike_threshold_from_validation": thr,
           "versions": versions()}
    json.dump(cfg, open(f"{R}/config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)

    pd.set_option("display.width", 160)
    print("\n================ ΑΠΟΤΕΛΕΣΜΑΤΑ ================")
    print("\nΈλεγχος αφαίρεσης καιρού:\n", abl.round(3).to_string(index=False))
    sig = abl[(abl["p"] < 0.05) & (abl["μεταβολή MAE %"] > 0)]
    if len(abl) and sig.empty:
        print("=> Η αφαίρεση του καιρού δεν χειροτερεύει στατιστικά σημαντικά (5%) κανένα μοντέλο.")
    elif len(sig):
        print("=> ΠΡΟΣΟΧΗ: η αφαίρεση του καιρού χειροτερεύει σημαντικά τα:", ", ".join(sig["μοντέλο"]),
              "(δες το weather_ablation.csv πριν αποφασίσεις).")
    print("\nΈλεγχος ευαισθησίας για το net_import (σύγκριση με την τελική εκδοχή):\n", sens.round(3).to_string(index=False))
    same = sens[(sens["εκδοχή"] == "net_import_ίδιας_ημέρας") & (sens["p"] < 0.05) & (sens["μεταβολή MAE %"] < 0)]
    if len(same):
        print("=> Το net_import της ίδιας ημέρας (δεν είναι γνωστό πριν το κλείσιμο) βελτιώνει σημαντικά (5%) τα:",
              ", ".join(same["μοντέλο"]), "- δες το netimport_sensitivity.csv.")
    else:
        print("=> Το net_import της ίδιας ημέρας δεν βελτιώνει σημαντικά κανένα μοντέλο σε σχέση με της προηγούμενης.")
    print(f"\nΜετρικές τελικής εκδοχής ({FINAL}):\n", mets[FINAL].round(3).to_string())
    print("\nDiebold–Mariano (|e|):\n", dm[["σύγκριση", "DM |e|", "p |e|"]].round(4).to_string(index=False))
    print("\nΤαξινομητής spikes:\n", clf.round(3).to_string(index=False))
    if ext_models:
        print("\nΠαραλλαγές Random Forest (v4) – μετρικές:\n", mex.round(3).to_string())
        print("\nΠαραλλαγές Random Forest (v4) – Diebold–Mariano:\n", dmx.round(4).to_string(index=False))
    print("\nΣφάλμα ανά επίπεδο τιμής, MAE (bias):\n", wide_.to_string(index=False))
    if diag is not None:
        print("\nΔιαγνωστικά καταλοίπων ARIMAX:\n", diag.round(4).to_string(index=False))
    if use_arimax:
        for n_, c_ in cfgs.items():
            print(f"Μπλοκ χωρίς σύγκλιση ARIMAX ({n_}):", c_["arimax_blocks_not_converged"])
    print(f"\nΌλα τα αποτελέσματα στον φάκελο {R}/")
    return outs


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TEST_START:
        sys.exit("Χρήση: python entsoe_gr_models_v4.py crisis|recent [--no-arimax] [--no-extras]")
    period = sys.argv[1]
    h = pd.read_parquet(f"dataset_hourly_{period}.parquet")
    run(period, h, use_arimax="--no-arimax" not in sys.argv, use_extras="--no-extras" not in sys.argv)
