"""
Pipeline v2 (ανά περίοδο): καιρός -> ενοποίηση + net_import -> μοντέλα
Διαβάζει τα parquet που κατέβασε το entsoe_download.py (data_raw/<περίοδος>/).
"""
import os, sys, time, warnings
import numpy as np, pandas as pd, requests, holidays
warnings.filterwarnings("ignore")

RAW = "data_raw"
# Παράθυρο ανάλυσης ανά περίοδο. Το "recent" ξεκινά 2025-07-10 γιατί τα day-ahead schedules
# Βουλγαρίας/Τουρκίας υπάρχουν μόνο από αρχές Ιουλίου 2025 (βλ. download_summary.csv).
WINDOWS = {"crisis": ("2021-01-01", "2022-12-31"),
           "recent": ("2025-07-10", "2026-09-30")}
TEST_START = {"crisis": "2022-07-01", "recent": "2026-04-01"}     # τελευταίο ~εξάμηνο = test
NB = ["BG", "AL", "MK", "TR", "IT"]

WEATHER_POINTS = [   # (όνομα, lat, lon, βάρος): ενδεικτικά βάρη, τα βάρη αθροίζουν στο 1
    ("athens", 37.98, 23.73, 0.35), ("thessaloniki", 40.64, 22.94, 0.15),
    ("evia", 38.55, 23.95, 0.15), ("thrace", 41.10, 25.40, 0.10),
    ("peloponnese", 37.50, 22.40, 0.15), ("thessaly", 39.60, 22.40, 0.10)]
WEATHER_VARS = "temperature_2m,wind_speed_100m,shortwave_radiation,cloud_cover"


# ============================ ΒΟΗΘΗΤΙΚΑ ============================
def read_hourly(period, name):
    """Διαβάζει parquet, μετατρέπει σε UTC και κάνει ωριαίο μέσο όρο (15λεπτα -> ωριαία)."""
    path = f"{RAW}/{period}/{name}.parquet"
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)      # κάθε αρχείο έχει τοπική ώρα της χώρας του
    return df.apply(pd.to_numeric, errors="coerce").resample("h").mean()


def net_flow(period, kind):
    """Καθαρό import στην Ελλάδα (MW) = Σ(γείτονας->GR) - Σ(GR->γείτονας).
    kind='sched': προγραμματισμένες ανταλλαγές (προτιμά day-ahead, αλλιώς total).
    kind='phys' : πραγματικές ροές (ΜΟΝΟ για EDA)."""
    ins, outs, used = [], [], {}
    for nb in NB:
        for a, b, lst in ((nb, "GR", ins), ("GR", nb, outs)):
            names = ([f"sched_da_{a}_{b}", f"sched_total_{a}_{b}"] if kind == "sched"
                     else [f"phys_{a}_{b}"])
            for n in names:
                s = read_hourly(period, n)
                if s is not None:
                    lst.append(s["value"]); used[n] = s["value"]; break
    imp = pd.concat(ins, axis=1).sum(axis=1) if ins else pd.Series(dtype=float)
    exp = pd.concat(outs, axis=1).sum(axis=1) if outs else pd.Series(dtype=float)
    return imp.sub(exp, fill_value=0.0), used


# ============================ ΚΑΙΡΟΣ ============================
def download_weather(period):
    path = f"{RAW}/{period}/weather.parquet"
    if os.path.exists(path):
        print("Ο καιρός υπάρχει ήδη:", path); return
    s, e = WINDOWS[period]
    e = min(pd.Timestamp(e), pd.Timestamp.today().normalize() - pd.Timedelta(days=7))  # το αρχείο έχει καθυστέρηση ~5 ημερών
    frames = {}
    for name, lat, lon, w in WEATHER_POINTS:
        for attempt in range(4):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": lat, "longitude": lon, "start_date": s, "end_date": str(e.date()),
                "hourly": WEATHER_VARS, "timezone": "UTC"}, timeout=120)
            if r.status_code == 429:
                time.sleep(30); continue
            r.raise_for_status(); break
        d = pd.DataFrame(r.json()["hourly"])
        d["time"] = pd.to_datetime(d["time"], utc=True)
        frames[name] = d.set_index("time") * w
        print("καιρός:", name, "ok"); time.sleep(2)
    wavg = sum(frames.values())                        # σταθμισμένος μέσος όρος
    wavg.columns = [f"w_{c}" for c in wavg.columns]
    wavg.to_parquet(path)


# ============================ ΕΝΟΠΟΙΗΣΗ + FEATURES ============================
def build(period):
    ws, we = WINDOWS[period]
    price = read_hourly(period, "price")["value"].rename("price")
    df = pd.DataFrame(index=pd.date_range(price.index.min(), price.index.max(), freq="h")).join(price)
    df = df.join(read_hourly(period, "load").rename(
        columns={"Forecasted Load": "load_fc", "Actual Load": "load_actual"}))
    df = df.join(read_hourly(period, "wind_solar_fc").rename(
        columns={"Solar": "fc_solar", "Wind Onshore": "fc_wind"}))
    wx = read_hourly(period, "weather")
    if wx is None:
        sys.exit(f"Λείπει ο καιρός. Τρέξε πρώτα:  python entsoe_gr_pipeline_v2.py weather {period}")
    df = df.join(wx)

    # --- μεταφορές ---
    ni, used = net_flow(period, "sched")
    df["net_import"] = ni.reindex(df.index).fillna(0.0)    # ΥΠΟΘΕΣΗ: ώρα χωρίς schedule = 0 MW
    pni, _ = net_flow(period, "phys")
    df["net_import_phys"] = pni.reindex(df.index)          # μόνο για EDA, ΟΧΙ feature

    win = df.loc[ws:we]
    print(f"\n=== {period}: παράθυρο {ws} -> {we}, {len(win)} ώρες ===")
    print("Κάλυψη schedules στο παράθυρο (% ωρών με δεδομένο):")
    for n, s in used.items():
        print(f"  {n:22s} {100 * s.reindex(win.index).notna().mean():5.1f}")
    cols = ["price", "load_fc", "fc_solar", "fc_wind", "w_temperature_2m"]
    print("Ποσοστό ΛΕΙΠΟΝΤΩΝ ωρών (%):\n", (win[cols].isna().mean() * 100).round(2).to_string())

    # --- κενά: παρεμβολή μόνο μικρών κενών (<=3 ώρες) στις εξωγενείς, ΠΟΤΕ στην τιμή ---
    fill = [c for c in df.columns if c not in ("price", "net_import", "net_import_phys")]
    df[fill] = df[fill].interpolate(limit=3, limit_direction="both")

    # --- features ---
    df["res_fc"] = df["fc_solar"] + df["fc_wind"]
    df["res_share"] = df["res_fc"] / df["load_fc"]
    df["residual_load"] = df["load_fc"] - df["res_fc"]
    loc = df.index.tz_convert("Europe/Athens")
    df["hour"], df["dow"], df["month"] = loc.hour, loc.dayofweek, loc.month
    gr_h = holidays.Greece(years=range(loc.year.min(), loc.year.max() + 1))
    df["holiday"] = [int(d in gr_h) for d in loc.date]
    for L in (24, 48, 168):
        df[f"price_lag{L}"] = df["price"].shift(L)          # κανονικός ωριαίος δείκτης: shift(24) = 24 ώρες πίσω
    df["price_roll7d_mean"] = df["price"].shift(24).rolling(168, min_periods=100).mean()
    df["price_roll7d_std"] = df["price"].shift(24).rolling(168, min_periods=100).std()
    mu = df["price"].shift(1).rolling(720, min_periods=240).mean()      # spike: κυλιόμενος μέσος 30 ημερών + 2σ
    sd = df["price"].shift(1).rolling(720, min_periods=240).std()
    df["spike"] = (df["price"] > mu + 2 * sd).astype(int)
    df["neg_price"] = (df["price"] <= 0).astype(int)

    need = ["price", "load_fc", "fc_solar", "fc_wind", "w_temperature_2m",
            "price_lag24", "price_lag48", "price_lag168"]
    out = df.loc[ws:we]
    n0 = len(out)
    out = out.dropna(subset=need)
    print(f"Γραμμές μετά το καθάρισμα: {len(out)} (αφαιρέθηκαν {n0 - len(out)}) | spikes: {100 * out['spike'].mean():.2f}%")
    out.to_parquet(f"dataset_hourly_{period}.parquet")
    return out


# ============================ ΗΜΕΡΗΣΙΟ ΕΠΙΠΕΔΟ ============================
EXOG = ["residual_load", "res_share", "net_import", "w_temperature_2m", "w_wind_speed_100m",
        "w_shortwave_radiation", "holiday", "spike_lag1"]
RF_FEATS = EXOG + ["load_fc", "res_fc", "w_cloud_cover", "dow", "month", "price_lag1", "price_lag7"]


def daily_dataset(h):
    h = h.copy(); h.index = h.index.tz_convert("Europe/Athens")
    d = h.resample("D").agg({"price": "mean", "load_fc": "mean", "res_fc": "mean", "res_share": "mean",
                             "residual_load": "mean", "net_import": "mean", "w_temperature_2m": "mean",
                             "w_wind_speed_100m": "mean", "w_shortwave_radiation": "mean",
                             "w_cloud_cover": "mean", "holiday": "max", "spike": "max"})
    d["n_hours"] = h["price"].resample("D").count()
    d = d[d["n_hours"] >= 20].drop(columns="n_hours").asfreq("D")   # ημέρες με σχεδόν πλήρη δεδομένα
    d["dow"], d["month"] = d.index.dayofweek, d.index.month
    d["price_lag1"], d["price_lag7"] = d["price"].shift(1), d["price"].shift(7)
    d["spike_lag1"] = d["spike"].shift(1)
    return d.dropna()


def stationarity(y):
    from statsmodels.tsa.stattools import adfuller, kpss
    adf = adfuller(y.dropna())[1]; kp = kpss(y.dropna(), regression="c", nlags="auto")[1]
    print(f"ADF p={adf:.4f} (<0.05 => στάσιμη) | KPSS p={kp:.4f} (>0.05 => στάσιμη)")


def metrics(y, p, name):
    mae = np.mean(np.abs(y - p)); rmse = np.sqrt(np.mean((y - p) ** 2))
    smape = 100 * np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9))
    print(f"{name:8s} MAE={mae:7.2f} RMSE={rmse:7.2f} sMAPE={smape:6.2f}%")
    return np.abs(y - p)


def walk_forward(d, test_start, refit_every=30, order=(1, 1, 1), seasonal=(1, 0, 1, 7)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.ensemble import RandomForestRegressor
    test_idx = d.index[d.index >= pd.Timestamp(test_start, tz=d.index.tz)]
    out = pd.DataFrame(index=test_idx, columns=["y", "naive7", "arimax", "rf"], dtype=float)
    out["y"] = d.loc[test_idx, "price"]; out["naive7"] = d["price"].shift(7).loc[test_idx]
    for b in range(0, len(test_idx), refit_every):
        blk = test_idx[b:b + refit_every]
        train = d.loc[d.index < blk[0]]
        mu, sd = train[EXOG].mean(), train[EXOG].std().replace(0, 1)     # τυποποίηση exog (βοηθά τη σύγκλιση)
        Xall = (d[EXOG] - mu) / sd
        res = SARIMAX(train["price"], exog=Xall.loc[train.index], order=order, seasonal_order=seasonal,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        for t in blk:                                                   # one-step-ahead με append (χωρίς refit)
            out.loc[t, "arimax"] = float(res.forecast(1, exog=Xall.loc[[t]]).iloc[0])
            res = res.append(d.loc[[t], "price"], exog=Xall.loc[[t]], refit=False)
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42)
        rf.fit(train[RF_FEATS], train["price"])
        out.loc[blk, "rf"] = rf.predict(d.loc[blk, RF_FEATS])
        print(f"μπλοκ {blk[0].date()} -> {blk[-1].date()} ok")
    return out.dropna(), rf


def spike_classifier(d, test_start):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score
    cut = pd.Timestamp(test_start, tz=d.index.tz)
    tr, te = d[d.index < cut], d[d.index >= cut]
    if te["spike"].sum() == 0 or tr["spike"].sum() == 0:
        print("Spike RF: δεν υπάρχουν spikes στο train ή στο test, παραλείπεται."); return
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced",
                                 n_jobs=-1, random_state=42).fit(tr[RF_FEATS], tr["spike"])
    prob = clf.predict_proba(te[RF_FEATS])[:, 1]
    p, r, f, _ = precision_recall_fscore_support(te["spike"], prob > 0.5, average="binary", zero_division=0)
    print(f"Spike RF: precision={p:.2f} recall={r:.2f} F1={f:.2f} PR-AUC={average_precision_score(te['spike'], prob):.2f}"
          f" | spike-ημέρες test: {int(te['spike'].sum())}/{len(te)}")


def compare(out, rf):
    from scipy import stats
    e = {m: metrics(out["y"], out[m], m) for m in ["naive7", "arimax", "rf"]}
    dd = e["arimax"] - e["rf"]
    dm = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
    print(f"Diebold-Mariano (ARIMAX vs RF): stat={dm:.2f}, p={2 * (1 - stats.norm.cdf(abs(dm))):.4f}")
    print("RF feature importance:\n", pd.Series(rf.feature_importances_, index=RF_FEATS)
          .sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in WINDOWS:
        sys.exit("Χρήση: python entsoe_gr_pipeline_v2.py weather|build|model  crisis|recent")
    stage, period = sys.argv[1], sys.argv[2]
    if stage == "weather":
        download_weather(period)
    elif stage == "build":
        build(period)
    elif stage == "model":
        h = pd.read_parquet(f"dataset_hourly_{period}.parquet")
        d = daily_dataset(h); d.to_parquet(f"dataset_daily_{period}.parquet")
        print(f"Ημερήσιο dataset: {len(d)} ημέρες")
        stationarity(d["price"])
        out, rf = walk_forward(d, TEST_START[period])
        compare(out, rf)
        spike_classifier(d, TEST_START[period])
        out.to_csv(f"forecasts_{period}.csv")
