import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score

def train_models():
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

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

    #IMPORTANT FIX
    data = data.dropna()

    # -----------------------------
    # FEATURES
    # -----------------------------
    features = [
        'Open', 'High', 'Low', 'Volume',
        'Prev_Close', 'Prev_Open',
        'MA_5', 'MA_10',
        'Volatility', 'Daily_Return'
    ]

    X = data[features]
    y = data['Target']

    #FORCE NUMERIC (VERY IMPORTANT)
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    #FINAL CLEAN
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------
    split = int(len(X) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # SAFETY CHECK
    if len(X_train) == 0:
        raise ValueError("Training data is empty after preprocessing!")

    # -----------------------------
    # MODELS
    # -----------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    bayes_model = BayesianRidge()
    bayes_model.fit(X_train, y_train)

    # -----------------------------
    # PREDICT
    # -----------------------------
    lr_pred = lr_model.predict(X_test)
    bayes_pred = bayes_model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    bayes_mae = mean_absolute_error(y_test, bayes_pred)
    bayes_r2 = r2_score(y_test, bayes_pred)

    return {
        "X": X,
        "y_test": y_test,
        "lr_model": lr_model,
        "bayes_model": bayes_model,
        "lr_pred": lr_pred,
        "bayes_pred": bayes_pred,
        "lr_mae": lr_mae,
        "lr_r2": lr_r2,
        "bayes_mae": bayes_mae,
        "bayes_r2": bayes_r2
    }


def predict_next_day(data):
    latest = data["X"].iloc[-1:].copy()

    lr_pred = data["lr_model"].predict(latest)[0]
    bayes_pred = data["bayes_model"].predict(latest)[0]

    return lr_pred, bayes_pred
