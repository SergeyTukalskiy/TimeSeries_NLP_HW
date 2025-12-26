import pandas as pd
import numpy as np

def generate_features(df):
    X = df.copy()

    # Lags
    for lag in [1, 2, 7, 30]:
        X[f"lag_{lag}"] = X["USD_RUB"].shift(lag)

    # Rolling windows
    for w in [7, 30]:
        X[f"mean_{w}"] = X["USD_RUB"].rolling(w).mean()
        X[f"std_{w}"] = X["USD_RUB"].rolling(w).std()
        X[f"min_{w}"] = X["USD_RUB"].rolling(w).min()
        X[f"max_{w}"] = X["USD_RUB"].rolling(w).max()

    # Date features
    X["dayofweek"] = X.index.dayofweek
    X["month"] = X.index.month
    X["quarter"] = X.index.quarter

    # Cyclic
    X["sin_day"] = np.sin(2*np.pi*X.index.dayofyear/365)
    X["cos_day"] = np.cos(2*np.pi*X.index.dayofyear/365)

    return X
