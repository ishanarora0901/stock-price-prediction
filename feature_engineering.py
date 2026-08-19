import pandas as pd

# Load cleaned data
df = pd.read_csv("data/clean_stock_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# ==========================
# Create Moving Averages
# ==========================

df["MA10"] = df["Close"].rolling(window=10).mean()
df["MA20"] = df["Close"].rolling(window=20).mean()

# ==========================
# Daily Return
# ==========================

df["Daily_Return"] = df["Close"].pct_change()

# ==========================
# Volatility (5-day rolling std)
# ==========================

df["Volatility"] = df["Daily_Return"].rolling(window=5).std()

# ==========================
# Target Variable
# Next day's closing price
# ==========================

df["Target"] = df["Close"].shift(-1)

# Remove NaN values
df.dropna(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

print(df.head())

# Save feature engineered dataset
df.to_csv("data/final_stock_data.csv", index=False)

print("\nFeature Engineering Completed Successfully!")
print("Saved as data/final_stock_data.csv")