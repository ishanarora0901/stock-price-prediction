import streamlit as st
import joblib
import numpy as np
import os

# ===========================
# Page Configuration
# ===========================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Stock Price Prediction")
st.write("Predict the next day's closing stock price using a Machine Learning model.")

# ===========================
# Load Model
# ===========================

model = joblib.load("models/stock_model.joblib")

# ===========================
# User Inputs
# ===========================

open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=102.0)
low_price = st.number_input("Low Price", value=99.0)
volume = st.number_input("Volume", value=1000000)

ma10 = st.number_input("10-Day Moving Average (MA10)", value=100.5)
ma20 = st.number_input("20-Day Moving Average (MA20)", value=100.0)

daily_return = st.number_input(
    "Daily Return",
    value=0.01,
    format="%.5f"
)

volatility = st.number_input(
    "Volatility",
    value=0.02,
    format="%.5f"
)

# ===========================
# Prediction
# ===========================

if st.button("Predict Stock Price"):

    input_data = np.array([[
        open_price,
        high_price,
        low_price,
        volume,
        ma10,
        ma20,
        daily_return,
        volatility
    ]])

    prediction = model.predict(input_data)

    # ===========================
    # Prediction Result
    # ===========================

    st.success(f"✅ Predicted Closing Price: ₹ {prediction[0]:.2f}")

    # ===========================
    # Input Summary
    # ===========================

    st.markdown("---")
    st.subheader("📋 Input Summary")

    st.table({
        "Feature": [
            "Open",
            "High",
            "Low",
            "Volume",
            "MA10",
            "MA20",
            "Daily Return",
            "Volatility"
        ],
        "Value": [
            open_price,
            high_price,
            low_price,
            volume,
            ma10,
            ma20,
            daily_return,
            volatility
        ]
    })

    # ===========================
    # Actual vs Predicted Graph
    # ===========================

    st.markdown("---")
    st.subheader("📈 Actual vs Predicted Graph")

    if os.path.exists("graphs/actual_vs_predicted.png"):
        st.image(
            "graphs/actual_vs_predicted.png",
            caption="Actual vs Predicted Stock Price",
            use_container_width=True
        )
    else:
        st.warning("Actual vs Predicted graph not found.")

    # ===========================
    # Feature Importance Graph
    # ===========================

    st.markdown("---")
    st.subheader("📊 Feature Importance")

    if os.path.exists("graphs/feature_importance.png"):
        st.image(
            "graphs/feature_importance.png",
            caption="Feature Importance",
            use_container_width=True
        )
    else:
        st.warning("Feature Importance graph not found.")

    # ===========================
    # Model Information
    # ===========================

    st.markdown("---")
    st.subheader("🤖 Model Information")

    st.info("""
Model Used : Random Forest Regressor

Evaluation Metrics:
• R² Score : 75.70%
• MAE : 1.7491
• RMSE : 2.2521
""")