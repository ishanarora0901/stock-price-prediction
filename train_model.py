import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("data/final_stock_data.csv")

# ======================================
# Select Features and Target
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
# Create Model
# ======================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# ======================================
# Train Model
# ======================================

model.fit(X_train, y_train)

# ======================================
# Predictions
# ======================================

predictions = model.predict(X_test)

# ======================================
# Evaluation
# ======================================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")

# ======================================
# Save Model
# ======================================

joblib.dump(model, "models/stock_model.joblib")

print("\nModel Saved Successfully!")