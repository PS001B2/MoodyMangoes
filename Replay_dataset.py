import yfinance as yf
import pandas as pd
import datetime as dt
import os
import time

# -------------------- CONFIG --------------------
TICKER = "^NSEI"  # Correct Yahoo Finance ticker for NIFTY 50
OUTPUT_DIR = "minute_data"
BATCH_DAYS = 7  # Max days per 1m request
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- FETCH & SAVE --------------------
def fetch_minute_data_batch(ticker, start, end):
    """Fetch 1-minute data for the given date range."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1m",
            progress=False,
            prepost=False
        )
        return df
    except Exception as e:
        print(f"Error fetching {start} to {end}: {e}")
        return pd.DataFrame()


def save_daily_csvs(ticker, output_dir):
    """Fetch 1m data in 7-day batches and save separate CSVs for each day."""
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=30)
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + dt.timedelta(days=BATCH_DAYS), end_date)
        print(f"\nFetching: {current_start} to {current_end}...")

        df = fetch_minute_data_batch(ticker, start=current_start, end=current_end)
        if df.empty:
            print(f"No data for {current_start} to {current_end}")
            current_start = current_end
            continue

        # Fix timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("Asia/Kolkata")

        # Manually reset index and set proper column names
        df = df.reset_index()
        df.columns = ["Datetime", "Close", "High", "Low", "Open", "Volume"]

        # Save one CSV per date
        for date, day_df in df.groupby(df["Datetime"].dt.date):
            date_str = date.strftime("%Y-%m-%d")
            filename = f"{ticker.strip('^').replace('.NS', '')}_{date_str}.csv"
            path = os.path.join(output_dir, filename)
            day_df.to_csv(path, index=False)
            print(f"✅ Saved clean file: {path}")

        current_start = current_end
        time.sleep(1)

    print("\n✅ Done fetching all available data.")


# -------------------- RUN --------------------
if __name__ == "__main__":
    save_daily_csvs(TICKER, OUTPUT_DIR)