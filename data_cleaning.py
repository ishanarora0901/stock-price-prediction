import pandas as pd

# ===============================
# STEP 1: Load Dataset
# ===============================

df = pd.read_csv("data/stock_data.csv")

print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())


# ===============================
# STEP 2: Dataset Information
# ===============================

print("\nDataset Information")
print("=" * 50)
print(df.info())


# ===============================
# STEP 3: Shape of Dataset
# ===============================

print("\nDataset Shape")
print("=" * 50)
print(df.shape)


# ===============================
# STEP 4: Missing Values
# ===============================

print("\nMissing Values")
print("=" * 50)
print(df.isnull().sum())


# ===============================
# STEP 5: Duplicate Values
# ===============================

print("\nDuplicate Rows")
print("=" * 50)
print(df.duplicated().sum())


# ===============================
# STEP 6: Remove Duplicates
# ===============================

df.drop_duplicates(inplace=True)


# ===============================
# STEP 7: Remove Missing Values
# ===============================

df.dropna(inplace=True)


# ===============================
# STEP 8: Convert Date Column
# ===============================

df["Date"] = pd.to_datetime(df["Date"])


# ===============================
# STEP 9: Sort by Date
# ===============================

df.sort_values(by="Date", inplace=True)


# ===============================
# STEP 10: Reset Index
# ===============================

df.reset_index(drop=True, inplace=True)


# ===============================
# STEP 11: Check Data Types
# ===============================

print("\nData Types")
print("=" * 50)
print(df.dtypes)


# ===============================
# STEP 12: Statistical Summary
# ===============================

print("\nStatistical Summary")
print("=" * 50)
print(df.describe())


# ===============================
# STEP 13: Save Cleaned Dataset
# ===============================

df.to_csv("data/clean_stock_data.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned dataset saved as:")
print("data/clean_stock_data.csv")