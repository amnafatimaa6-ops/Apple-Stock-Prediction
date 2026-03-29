# model.py

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score


def load_and_prepare_data():
    # Load data
    data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

    # Fix MultiIndex
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

    # Clean data properly
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Select features
    features = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'Prev_Close', 'Prev_Open',
        'MA_5', 'MA_10',
        'Volatility', 'Daily_Return'
    ]

    X = data[features]
    y = data['Target']

    # Train-test split (time-based)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, X


def train_models():
    X_train, X_test, y_train, y_test, X = load_and_prepare_data()

    # -----------------------------
    # LINEAR REGRESSION
    # -----------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    # -----------------------------
    # BAYESIAN RIDGE
    # -----------------------------
    bayes_model = BayesianRidge()
    bayes_model.fit(X_train, y_train)

    bayes_pred = bayes_model.predict(X_test)

    bayes_mae = mean_absolute_error(y_test, bayes_pred)
    bayes_r2 = r2_score(y_test, bayes_pred)

    return {
        "lr_model": lr_model,
        "bayes_model": bayes_model,
        "lr_pred": lr_pred,
        "bayes_pred": bayes_pred,
        "y_test": y_test,
        "lr_mae": lr_mae,
        "lr_r2": lr_r2,
        "bayes_mae": bayes_mae,
        "bayes_r2": bayes_r2,
        "X": X
    }


def predict_next_day(models_data):
    X = models_data["X"]
    latest_data = X.iloc[-1:]

    lr_pred = models_data["lr_model"].predict(latest_data)[0]
    bayes_pred = models_data["bayes_model"].predict(latest_data)[0]

    return lr_pred, bayes_pred
