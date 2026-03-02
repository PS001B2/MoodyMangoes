import os
import glob
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

NSEI_DIR = "minute_data/NSEI/"
VIX_DIR  = "minute_data/INDIAVIX/"
OUTPUT_FILE = "minute_data/dataset_intraday.csv"

TP = 10
SL = 4
LOOKAHEAD = 30
MARKET_MINUTES = 375   # 9:15 to 15:30

# ==========================================================
# LOADING
# ==========================================================

def load_csv(path):
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Parse datetime (remove timezone)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    # Drop volume (always zero)
    df = df.drop(columns=["volume"], errors="ignore")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def minutes_from_open(df):
    open_time = df["datetime"].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
    return ((df["datetime"] - open_time).dt.total_seconds() / 60)

# ==========================================================
# NSEI FEATURES
# ==========================================================

def build_nsei_features(df):

    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)

    df["range"] = df["high"] - df["low"]

    df["rv_10"] = df["ret_1"].rolling(10).std()
    df["rv_ratio"] = df["rv_10"] / df["rv_10"].rolling(30).mean()

    # Chop ratio
    abs_sum = df["ret_1"].abs().rolling(10).sum()
    net_move = df["close"].diff(10).abs()
    df["chop_ratio"] = net_move / abs_sum

    # Trend strength
    ema10 = df["close"].ewm(span=10).mean()
    ema30 = df["close"].ewm(span=30).mean()
    df["trend_strength"] = (ema10 - ema30) / df["close"]

    # Range position
    high15 = df["high"].rolling(15).max()
    low15  = df["low"].rolling(15).min()
    df["range_pos_15"] = (df["close"] - low15) / (high15 - low15)

    high30 = df["high"].rolling(30).max()
    low30  = df["low"].rolling(30).min()
    df["range_pos_30"] = (df["close"] - low30) / (high30 - low30)

    # Time encoding
    df["minutes_from_open"] = minutes_from_open(df)
    df["time_sin"] = np.sin(2 * np.pi * df["minutes_from_open"] / MARKET_MINUTES)
    df["time_cos"] = np.cos(2 * np.pi * df["minutes_from_open"] / MARKET_MINUTES)

    return df


# ==========================================================
# VIX FEATURES
# ==========================================================

def build_vix_features(df):

    df["vix_ret_1"] = df["close"].pct_change()
    df["vix_ret_5"] = df["close"].pct_change(5)

    df["vix_rv_10"] = df["vix_ret_1"].rolling(10).std()
    df["vix_vol_ratio"] = df["vix_rv_10"] / df["vix_rv_10"].rolling(30).mean()

    return df


# ==========================================================
# TARGETS
# ==========================================================

def build_targets(df):

    future_high = df["high"].rolling(LOOKAHEAD).max().shift(-LOOKAHEAD)
    future_low  = df["low"].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)

    entry = df["close"]

    df["y_call"] = (
        (future_high - entry >= TP) &
        (entry - future_low <= SL)
    ).astype(int)

    df["y_put"] = (
        (entry - future_low >= TP) &
        (future_high - entry <= SL)
    ).astype(int)

    return df


# ==========================================================
# FILE HELPERS
# ==========================================================

def extract_date_from_filename(path):
    name = os.path.basename(path)
    return name.split("_")[-1].replace(".csv", "")


def build_vix_lookup():
    vix_files = glob.glob(os.path.join(VIX_DIR, "*.csv"))
    lookup = {}
    for path in vix_files:
        date = extract_date_from_filename(path)
        lookup[date] = path
    return lookup


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def process_all_days():

    nsei_files = sorted(glob.glob(os.path.join(NSEI_DIR, "*.csv")))
    vix_lookup = build_vix_lookup()

    print(f"Found {len(nsei_files)} NSEI files")
    print(f"Found {len(vix_lookup)} VIX files")

    all_days = []

    for nsei_path in nsei_files:

        date_str = extract_date_from_filename(nsei_path)

        if date_str not in vix_lookup:
            print(f"⚠ No VIX file for {date_str}")
            continue

        nsei = load_csv(nsei_path)
        vix  = load_csv(vix_lookup[date_str])

        if len(nsei) < 200:
            print(f"⚠ Skipping {date_str} (too few rows)")
            continue

        # Build features
        nsei = build_nsei_features(nsei)
        vix  = build_vix_features(vix)

        # Merge
        df = pd.merge(
            nsei,
            vix[["datetime", "vix_ret_1", "vix_ret_5",
                 "vix_rv_10", "vix_vol_ratio"]],
            on="datetime",
            how="inner"
        )

        if len(df) == 0:
            print(f"⚠ Empty merge for {date_str}")
            continue

        # Cross feature
        df["price_vix_div"] = df["ret_1"] - df["vix_ret_1"]

        # Targets
        df = build_targets(df)

        # Remove warmup + tail
        df = df.iloc[40:-LOOKAHEAD]

        df["date"] = df["datetime"].dt.date

        df = df.dropna()

        if len(df) == 0:
            print(f"⚠ Empty after dropna for {date_str}")
            continue

        all_days.append(df)

        print(f"✅ Processed {date_str} ({len(df)} rows)")

    if len(all_days) == 0:
        print("❌ No valid days processed.")
        return

    final_df = pd.concat(all_days).reset_index(drop=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved dataset to:", OUTPUT_FILE)
    print("Total rows:", len(final_df))


if __name__ == "__main__":
    process_all_days()