import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/stock_model.joblib")

# Load dataset
df = pd.read_csv("data/final_stock_data.csv")

# Take the latest row
latest = df.iloc[-1]

# Prepare input features
input_data = [[
    latest["Open"],
    latest["High"],
    latest["Low"],
    latest["Volume"],
    latest["MA10"],
    latest["MA20"],
    latest["Daily_Return"],
    latest["Volatility"]
]]

# Predict
prediction = model.predict(input_data)

print("=" * 50)
print("NEXT DAY STOCK PRICE PREDICTION")
print("=" * 50)
print(f"Predicted Closing Price : {prediction[0]:.2f}")