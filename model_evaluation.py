import matplotlib
matplotlib.use("Agg")
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# ======================================
# Create Graph Folder
# ======================================

os.makedirs("graphs", exist_ok=True)

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("data/final_stock_data.csv")

# ======================================
# Features and Target
# ======================================

X = df[[
    "Open",
    "High",
    "Low",
    "Volume",
    "MA10",
    "MA20",
    "Daily_Return",
    "Volatility"
]]

y = df["Target"]

# ======================================
# Train-Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================
# Load Trained Model
# ======================================

model = joblib.load("models/stock_model.joblib")

# ======================================
# Predictions
# ======================================

predictions = model.predict(X_test)

# ======================================
# Plot Graph
# ======================================

plt.figure(figsize=(12, 6))

plt.plot(y_test.values, label="Actual Price", marker="o", linewidth=2)
plt.plot(predictions, label="Predicted Price", marker="x", linewidth=2)

plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Test Samples")
plt.ylabel("Closing Price")

plt.legend()
plt.grid(True)
plt.tight_layout()

# ======================================
# Save Graph
# ======================================

plt.savefig("graphs/actual_vs_predicted.png", dpi=300)

plt.close()

print("=" * 50)
print("Graph Generated Successfully!")
print("Saved at: graphs/actual_vs_predicted.png")
print("=" * 50)