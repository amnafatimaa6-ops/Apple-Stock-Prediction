import streamlit as st
import pandas as pd
import numpy as np

from model import train_models, predict_next_day

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Stock Prediction App", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("📈 Apple Stock Prediction Dashboard")
st.markdown("""
### AI-Powered Stock Forecasting System  
Built with Machine Learning • Real-time Data • Predictive Analytics
""")

# -----------------------------
# LOAD MODELS
# -----------------------------
data = train_models()

# -----------------------------
# METRICS
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Linear Regression R²", f"{data['lr_r2']:.3f}")
    st.metric("Linear MAE", f"{data['lr_mae']:.2f}")

with col2:
    st.metric("Bayesian Ridge R²", f"{data['bayes_r2']:.3f}")
    st.metric("Bayesian MAE", f"{data['bayes_mae']:.2f}")

# -----------------------------
# MODEL COMPARISON
# -----------------------------
st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Bayesian Ridge"],
    "MAE": [data["lr_mae"], data["bayes_mae"]],
    "R2": [data["lr_r2"], data["bayes_r2"]]
})

st.dataframe(comparison_df)
st.bar_chart(comparison_df.set_index("Model"))

# -----------------------------
# ACTUAL VS PREDICTED
# -----------------------------
st.subheader("📉 Actual vs Predicted")

chart_df = pd.DataFrame({
    "Actual": data["y_test"].values,
    "Linear": data["lr_pred"],
    "Bayesian": data["bayes_pred"]
})

st.line_chart(chart_df)

# -----------------------------
# RESIDUAL ANALYSIS
# -----------------------------
st.subheader("📉 Residual Analysis")

residuals = data["y_test"].values - data["lr_pred"]

res_df = pd.DataFrame({
    "Residuals": residuals
})

st.line_chart(res_df)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("🧠 Feature Importance (Linear Model)")

coefficients = data["lr_model"].coef_
features = data["X"].columns

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": np.abs(coefficients)
}).sort_values(by="Importance", ascending=False)

st.bar_chart(feat_df.set_index("Feature"))

# -----------------------------
# NEXT DAY PREDICTION
# -----------------------------
st.subheader("🔮 Next Day Prediction")

lr_next, bayes_next = predict_next_day(data)

st.success(f"Linear Prediction: ${lr_next:.2f}")
st.success(f"Bayesian Prediction: ${bayes_next:.2f}")

# -----------------------------
# SMART INSIGHT
# -----------------------------
st.subheader("📌 Insight")

if data["lr_r2"] > 0.9:
    st.success("Model shows strong predictive performance with high reliability.")
else:
    st.warning("Model performance is moderate. Predictions may vary.")

# -----------------------------
# USER INPUT PREDICTION
# -----------------------------
st.subheader("🎛️ Try Your Own Input")

open_price = st.number_input("Open Price", value=150.0)
high_price = st.number_input("High Price", value=155.0)
low_price = st.number_input("Low Price", value=149.0)
volume = st.number_input("Volume", value=100000000)

if st.button("Predict Custom Price"):
    sample = data["X"].iloc[-1:].copy()

    sample["Open"] = open_price
    sample["High"] = high_price
    sample["Low"] = low_price
    sample["Volume"] = volume

    pred = data["lr_model"].predict(sample)[0]

    st.success(f"Predicted Price: ${pred:.2f}")

# -----------------------------
# RAW DATA
# -----------------------------
with st.expander("📂 View Processed Data"):
    st.dataframe(data["X"].tail(20))
