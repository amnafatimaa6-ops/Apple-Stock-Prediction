import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# APP TITLE
# -----------------------------
st.set_page_config(page_title="Stock Prediction App", layout="wide")

st.title("📈 Apple Stock Prediction Dashboard")
st.markdown("Advanced ML-based forecasting using Linear & Bayesian Ridge Models")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

data = load_data()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
data['Target'] = data['Close'].shift(-1)
data['Prev_Close'] = data['Close'].shift(1)
data['Prev_Open'] = data['Open'].shift(1)
data['MA_5'] = data['Close'].rolling(5).mean()
data['MA_10'] = data['Close'].rolling(10).mean()
data['Volatility'] = data['Close'].rolling(5).std()
data['Daily_Return'] = data['Close'].pct_change()

data = data.dropna()

X = data.drop(['Target'], axis=1)
y = data['Target']

# Train-test split
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# TRAIN MODELS
# -----------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

bayes_model = BayesianRidge()
bayes_model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------
lr_pred = lr_model.predict(X_test)
bayes_pred = bayes_model.predict(X_test)

# -----------------------------
# METRICS
# -----------------------------
lr_r2 = r2_score(y_test, lr_pred)
bayes_r2 = r2_score(y_test, bayes_pred)

lr_mae = mean_absolute_error(y_test, lr_pred)
bayes_mae = mean_absolute_error(y_test, bayes_pred)

# -----------------------------
# UI LAYOUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Linear Regression")
    st.metric("R² Score", f"{lr_r2:.3f}")
    st.metric("MAE", f"{lr_mae:.2f}")

with col2:
    st.subheader("📊 Bayesian Ridge")
    st.metric("R² Score", f"{bayes_r2:.3f}")
    st.metric("MAE", f"{bayes_mae:.2f}")

# -----------------------------
# PLOT
# -----------------------------
st.subheader("📉 Actual vs Predicted")

chart_df = pd.DataFrame({
    "Actual": y_test.values,
    "Linear": lr_pred,
    "Bayesian": bayes_pred
})

st.line_chart(chart_df)

# -----------------------------
# NEXT DAY PREDICTION
# -----------------------------
st.subheader("🔮 Next Day Prediction")

latest_data = X.iloc[-1:]

lr_next = lr_model.predict(latest_data)[0]
bayes_next = bayes_model.predict(latest_data)[0]

st.success(f"Linear Prediction: ${lr_next:.2f}")
st.success(f"Bayesian Prediction: ${bayes_next:.2f}")

# -----------------------------
# RAW DATA
# -----------------------------
with st.expander("📂 View Raw Data"):
    st.dataframe(data.tail(20))
