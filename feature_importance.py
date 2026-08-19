import matplotlib
matplotlib.use("Agg")

import os
import joblib
import matplotlib.pyplot as plt

# ======================================
# Create Graph Folder
# ======================================

os.makedirs("graphs", exist_ok=True)

# ======================================
# Load Trained Model
# ======================================

model = joblib.load("models/stock_model.joblib")

# ======================================
# Feature Names
# ======================================

features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "MA10",
    "MA20",
    "Daily_Return",
    "Volatility"
]

# ======================================
# Feature Importance
# ======================================

importance = model.feature_importances_

# ======================================
# Plot Graph
# ======================================

plt.figure(figsize=(10, 6))

bars = plt.bar(features, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.xticks(rotation=30)

# Display values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

# ======================================
# Save Graph
# ======================================

plt.savefig("graphs/feature_importance.png", dpi=300)

plt.close()

print("=" * 50)
print("Feature Importance Graph Generated Successfully!")
print("Saved at: graphs/feature_importance.png")
print("=" * 50)