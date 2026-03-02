import yfinance as yf
import pandas as pd
import datetime as dt
import os
import time

# -------------------- CONFIG --------------------
TICKERS = ["^NSEI", "^INDIAVIX"]   # Add more if needed
OUTPUT_DIR = "minute_data"
BATCH_DAYS = 7          # Max allowed per 1m request
TOTAL_DAYS = 30         # How many days back to fetch

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- DATA FETCH --------------------
def fetch_data(ticker, start_date, end_date):
    """Download 1-minute data from Yahoo Finance."""
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1m",
            progress=False
        )
        return df
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return pd.DataFrame()


# -------------------- DATA CLEAN --------------------
def clean_data(df):
    """Fix timezone and column names."""
    if df.empty:
        return df

    # Fix timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")

    # Reset index and rename columns properly
    df = df.reset_index()
    df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

    return df


# -------------------- SAVE DAILY FILES --------------------
def save_daily_files(df, ticker):
    """Save one CSV per trading day inside ticker-specific folder."""
    if df.empty:
        return

    ticker_name = ticker.replace("^", "").replace(".NS", "")
    
    # Create ticker-specific folder
    ticker_folder = os.path.join(OUTPUT_DIR, ticker_name)
    os.makedirs(ticker_folder, exist_ok=True)

    for date, day_df in df.groupby(df["Datetime"].dt.date):
        filename = f"{ticker_name}_{date}.csv"
        filepath = os.path.join(ticker_folder, filename)
        day_df.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath}")


# -------------------- MAIN LOGIC --------------------
def download_last_n_days(ticker):
    """Download last N days in 7-day batches."""
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=TOTAL_DAYS)

    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + dt.timedelta(days=BATCH_DAYS), end_date)

        print(f"\nFetching {ticker}: {current_start} → {current_end}")

        df = fetch_data(ticker, current_start, current_end)

        if df.empty:
            print("No data found.")
        else:
            df = clean_data(df)
            save_daily_files(df, ticker)

        current_start = current_end
        time.sleep(1)  # Avoid Yahoo rate limits


# -------------------- RUN --------------------
if __name__ == "__main__":
    for ticker in TICKERS:
        download_last_n_days(ticker)

    print("\n🎉 All downloads completed.")