"""
Κατέβασμα ENTSO-E μέσω API για την ελληνική αγορά, σε ΔΥΟ περιόδους:
    crisis : 2021-01-01 -> 2022-12-31  (ενεργειακή κρίση)
    recent : 2025-01-01 -> 2026-09-30

Δεδομένα: τιμή day-ahead, φορτίο (πρόβλεψη/πραγματικό), πρόβλεψη αιολικών/φωτοβολταϊκών,
πραγματική παραγωγή ανά τύπο, net position, και μεταφορές (scheduled exchanges + physical flows)
με Βουλγαρία, Αλβανία, Β. Μακεδονία, Τουρκία, Ιταλία.

Χρήση:
    pip install entsoe-py pandas pyarrow
    export ENTSOE_TOKEN="το_token_σου"          (Windows PowerShell:  $env:ENTSOE_TOKEN="...")
    python entsoe_download.py crisis | recent | all

Αποτέλεσμα:   data_raw/<περίοδος>/<όνομα>.parquet   +   data_raw/download_summary.csv
Ξαναξεκινά από εκεί που έμεινε (κάθε τρίμηνο αποθηκεύεται ξεχωριστά στο chunks/).
"""
import os, sys, time
import pandas as pd
from entsoe import EntsoePandasClient

try:
    from entsoe.exceptions import NoMatchingDataError
except Exception:                       # παλαιότερες εκδόσεις
    class NoMatchingDataError(Exception):
        pass

API_KEY = os.environ.get("ENTSOE_TOKEN", "ΒΑΛΕ_ΤΟ_TOKEN")
OUT = "data_raw"
PERIODS = {
    "crisis": ("2021-01-01", "2022-12-31"),
    "recent": ("2025-01-01", "2026-09-30"),
}
# Γείτονες. Για την Ιταλία δοκιμάζουμε διαδοχικά πιθανούς κωδικούς ζώνης και κρατάμε όποιον έχει δεδομένα.
NEIGHBORS = {
    "BG": ["BG"],
    "AL": ["AL"],
    "MK": ["MK"],
    "TR": ["TR"],
    "IT": ["IT", "IT_BRNN", "IT_GR", "IT_SUD"],
}
INCLUDE_PHYSICAL = True      # πραγματικές ροές (μόνο για EDA, ΟΧΙ ως feature πρόβλεψης)
SLEEP = 0.3                  # ευγένεια προς το API (όριο ~400 αιτήματα/λεπτό)


def quarters(start, end):
    edges = pd.date_range(start, end, freq="QS").union(
        pd.DatetimeIndex([start, end + pd.Timedelta(days=1)]))
    return list(zip(edges[:-1], edges[1:]))


def flatten(obj):
    if isinstance(obj, pd.Series):
        return obj.to_frame("value")
    if isinstance(obj.columns, pd.MultiIndex):
        obj.columns = [" | ".join(map(str, c)).strip() for c in obj.columns]
    obj.columns = [str(c) for c in obj.columns]
    return obj


def fetch_chunked(period, name, fn, start, end):
    """Κατεβάζει ανά τρίμηνο, με cache και 3 προσπάθειες. Επιστρέφει DataFrame ή None."""
    cdir = f"{OUT}/{period}/chunks"
    os.makedirs(cdir, exist_ok=True)
    parts = []
    for s, e in quarters(start, end):
        cpath = f"{cdir}/{name}_{s:%Y%m%d}.parquet"
        if os.path.exists(cpath):
            parts.append(pd.read_parquet(cpath))
            continue
        for attempt in range(3):
            try:
                d = flatten(fn(s, e))
                d.to_parquet(cpath)
                parts.append(d)
                time.sleep(SLEEP)
                break
            except NoMatchingDataError:
                break                                   # δεν υπάρχουν δεδομένα για το διάστημα
            except KeyError:
                return None                             # άγνωστος κωδικός ζώνης
            except Exception as ex:
                print(f"  [{name}] {s.date()} προσπάθεια {attempt + 1} απέτυχε: {ex}")
                time.sleep(5)
    if not parts:
        return None
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_parquet(f"{OUT}/{period}/{name}.parquet")
    return df


def download_period(period):
    s0, e0 = PERIODS[period]
    start, end = pd.Timestamp(s0, tz="UTC"), pd.Timestamp(e0, tz="UTC")
    c = EntsoePandasClient(api_key=API_KEY)
    summary = []

    def run(name, fn):
        print(f"[{period}] {name} ...")
        df = fetch_chunked(period, name, fn, start, end)
        if df is None:
            print(f"  -> χωρίς δεδομένα")
        else:
            print(f"  -> {len(df)} γραμμές ({df.index.min()} -> {df.index.max()})")
            summary.append({"period": period, "name": name, "rows": len(df),
                            "first": df.index.min(), "last": df.index.max(),
                            "nan_share": round(float(df.isna().mean().mean()), 4),
                            "columns": " ; ".join(df.columns)[:200]})
        return df

    # ---- Ελλάδα ----
    run("price", lambda s, e: c.query_day_ahead_prices("GR", start=s, end=e))
    run("load", lambda s, e: c.query_load_and_forecast("GR", start=s, end=e))
    run("wind_solar_fc", lambda s, e: c.query_wind_and_solar_forecast("GR", start=s, end=e, psr_type=None))
    run("generation", lambda s, e: c.query_generation("GR", start=s, end=e, psr_type=None))
    run("net_position_da", lambda s, e: c.query_net_position("GR", start=s, end=e, dayahead=True))

    # ---- Μεταφορές με γείτονες ----
    for nb, candidates in NEIGHBORS.items():
        for code in candidates:
            found = False
            for a, b in ((code, "GR"), ("GR", code)):        # και προς τις δύο κατευθύνσεις
                for da in (True, False):                     # πρώτα day-ahead, αλλιώς συνολικές
                    tag = "da" if da else "total"
                    df = run(f"sched_{tag}_{a}_{b}",
                             lambda s, e, a=a, b=b, da=da:
                             c.query_scheduled_exchanges(a, b, start=s, end=e, dayahead=da))
                    if df is not None:
                        found = True
                        break
                if INCLUDE_PHYSICAL:
                    df = run(f"phys_{a}_{b}",
                             lambda s, e, a=a, b=b: c.query_crossborder_flows(a, b, start=s, end=e))
                    found = found or df is not None
            if found:
                print(f"[{period}] {nb}: χρησιμοποιήθηκε ο κωδικός '{code}'")
                break
        else:
            print(f"[{period}] ΠΡΟΣΟΧΗ: δεν βρέθηκαν δεδομένα για {nb}")
    return summary


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if API_KEY.startswith("ΒΑΛΕ"):
        sys.exit("Όρισε πρώτα το token:  export ENTSOE_TOKEN=...")
    todo = list(PERIODS) if which == "all" else [which]
    allsum = []
    for p in todo:
        allsum += download_period(p)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(allsum).to_csv(f"{OUT}/download_summary.csv", index=False)
    print("\nΤέλος. Περίληψη στο", f"{OUT}/download_summary.csv")
    print(pd.DataFrame(allsum)[["period", "name", "rows", "first", "last", "nan_share"]].to_string(index=False))
