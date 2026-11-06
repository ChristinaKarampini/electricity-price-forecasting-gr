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


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in WINDOWS:
        sys.exit("Χρήση: python entsoe_gr_pipeline_v2.py weather|build  crisis|recent")
    stage, period = sys.argv[1], sys.argv[2]
    if stage == "weather":
        download_weather(period)
    elif stage == "build":
        build(period)
