import yfinance as yf
import numpy as np
import talib
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

# -------------------- Configuration --------------------
DATA_PATH = 'Dataset/Dataset.csv'
SAVE_PATH = DATA_PATH
FETCH_PADDING_DAYS = 100  # Use last 100 rows to ensure RSI/OBV continuity

# -------------------- Helper: Preprocessing --------------------
def preprocess_data(data):
    log_data = np.log(data)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(log_data.values.reshape(-1, 1))
    return data_scaled, scaler

# -------------------- 1. Load Existing Dataset --------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

existing_df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
last_date = existing_df['Date'].max().date()
print(f"✅ Last available date in dataset: {last_date}")

# -------------------- 2. Fetch New Market Data --------------------
start_date = (last_date - timedelta(days=FETCH_PADDING_DAYS)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

print(f"📥 Fetching data from {start_date} to {end_date}...")
nifty = yf.download("^NSEI", start=start_date, end=end_date)
inrusd = yf.download("INRUSD=X", start=start_date, end=end_date)
vix = yf.download("^INDIAVIX", start=start_date, end=end_date)

nifty['Open_Close_Avg'] = (nifty['Open'] + nifty['Close']) / 2
nifty['High_Low_Avg'] = (nifty['High'] + nifty['Low']) / 2

# Multi-period returns
for days in [5, 15, 30, 60]:
    nifty[f'Open_Close_{days}d'] = nifty['Open_Close_Avg'].pct_change(days)
    nifty[f'High_Low_{days}d'] = nifty['High_Low_Avg'].pct_change(days)

nifty['NSEI_Daily_Return'] = nifty['Close'].pct_change()

# RSI and OBV
nifty['RSI'] = talib.RSI(nifty['Close'].astype(float).values.flatten(), timeperiod=14)
nifty['OBV'] = talib.OBV(nifty['Close'].astype(float).values.flatten(), 
                         nifty['Volume'].astype(float).values.flatten())

# Normalize RSI and OBV
rsi_scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
nifty['RSI_Normalized'] = rsi_scaler.fit_transform(nifty[['RSI']])

obv_scaler = MinMaxScaler(feature_range=(-1, 1))
nifty['OBV_Normalized'] = obv_scaler.fit_transform(nifty[['OBV']])

# INR/USD Close
inrusd['INRUSD_X'] = inrusd['Close']

# India VIX daily return
vix['IndiaVIX_Daily_Return'] = vix['Close'].pct_change()

# -------------------- 3. Prepare Final Data --------------------
nifty_features = nifty[[
    'Open_Close_5d', 'Open_Close_15d', 'Open_Close_30d', 'Open_Close_60d',
    'High_Low_5d', 'High_Low_15d', 'High_Low_30d', 'High_Low_60d',
    'RSI_Normalized', 'OBV_Normalized', 'Close', 'NSEI_Daily_Return']]

combined_df = pd.merge(nifty_features, inrusd[['INRUSD_X']],
                       left_index=True, right_index=True, how='left')
combined_df = pd.merge(combined_df, vix[['IndiaVIX_Daily_Return']],
                       left_index=True, right_index=True, how='left')
combined_df.dropna(inplace=True)

# Move index to Date column
combined_df.reset_index(inplace=True)  # 'Date' column now exists

# -------------------- 4. Final Column Cleanup --------------------
# Flatten multi-index column names if any
combined_df.columns = [col if not isinstance(col, tuple) else col[0] for col in combined_df.columns]

# Drop any stray/unnamed columns
combined_df = combined_df.loc[:, ~combined_df.columns.str.contains("^Unnamed")]

# Ensure consistent column order
expected_order = [
    'Date', 'INRUSD_X', 'IndiaVIX_Daily_Return', 
    'Open_Close_5d', 'Open_Close_15d', 'Open_Close_30d', 'Open_Close_60d',
    'High_Low_5d', 'High_Low_15d', 'High_Low_30d', 'High_Low_60d',
    'RSI_Normalized', 'OBV_Normalized', 'Close', 'NSEI_Daily_Return'
]
combined_df = combined_df[expected_order]

# -------------------- 5. Append Only New Rows --------------------
new_data = combined_df[combined_df['Date'] > pd.to_datetime(last_date)]
print(f"🆕 New rows to append: {len(new_data)}")

if not new_data.empty:
    updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    updated_df.to_csv(SAVE_PATH, index=False)
    print(f"✅ Dataset updated with {len(new_data)} new rows.")
else:
    print("📌 No new data to update.")