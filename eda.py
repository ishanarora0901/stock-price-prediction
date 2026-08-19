import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/clean_stock_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# ==========================
# Dataset Information
# ==========================

print("First 5 Rows")
print(df.head())

print("\nShape :", df.shape)

print("\nColumns")
print(df.columns)

# ==========================
# Closing Price Trend
# ==========================

plt.figure(figsize=(12,5))

plt.plot(df["Date"], df["Close"])

plt.title("Closing Price Over Time")

plt.xlabel("Date")

plt.ylabel("Closing Price")

plt.grid(True)

plt.show()

# ==========================
# Trading Volume
# ==========================

plt.figure(figsize=(12,5))

plt.bar(df["Date"], df["Volume"])

plt.title("Trading Volume")

plt.xlabel("Date")

plt.ylabel("Volume")

plt.xticks(rotation=45)

plt.show()

# ==========================
# Histogram
# ==========================

plt.figure(figsize=(8,5))

plt.hist(df["Close"], bins=20)

plt.title("Distribution of Closing Price")

plt.xlabel("Close Price")

plt.ylabel("Frequency")

plt.show()

# ==========================
# Statistics
# ==========================

print("\nStatistical Summary")

print(df.describe())