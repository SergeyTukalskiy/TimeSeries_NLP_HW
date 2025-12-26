import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import boxcox
from scipy.stats import boxcox_normmax

from src.preprocessing import make_stationary
from src.feature_engineering import generate_features
from src.models_base import run_arima_like_models
from src.models_volatility import run_garch
from src.models_seasonal import run_prophet_model, run_tbats_model
from src.models_multivar import run_var_vecm
from src.benchmarks import naive_forecast, seasonal_naive, ses_model
from src.cv import expanding_window_cv, sliding_window_cv
from src.metrics import evaluate_all_metrics
from src.diagnostics import full_diagnostics
from src.utils import train_val_test_split

if __name__ == "__main__":
    # ---------------------------
    # LOAD DATA
    # ---------------------------

    df = pd.read_csv("data/market.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    target = "USD_RUB"

    # ---------------------------
    # PREPROCESSING + STATIONARITY
    # ---------------------------

    y = df[target].dropna()

    # Log transform
    y_log = np.log(y)

    # Box–Cox
    lambda_bc = boxcox_normmax(y_log, brack=(-2, 2))
    y_bc = boxcox(y_log, lmbda=lambda_bc)


    # Differencing chain (auto)
    y_st, diff_info = make_stationary(pd.Series(y_bc, index=y.index))

    # ---------------------------
    # TRAIN/VAL/TEST SPLIT
    # ---------------------------

    train, val, test = train_val_test_split(y_st, train_size=0.7, val_size=0.15)

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------

    X = generate_features(df)

    X_train = X.loc[train.index]
    X_val = X.loc[val.index]
    X_test = X.loc[test.index]

    # ---------------------------
    # CROSS-VALIDATION
    # ---------------------------

    expanding_scores = expanding_window_cv(train, horizon_list=[1, 7, 30])
    sliding_scores = sliding_window_cv(train, window=365, horizon_list=[1, 7, 30])

    # ---------------------------
    # RUN MODELS
    # ---------------------------

    results = {}

    # --- Baseline --- #
    results["naive"] = naive_forecast(train, test)
    results["seasonal_naive"] = seasonal_naive(train, test, season=7)
    results["ses"] = ses_model(train, test)

    # --- ARIMA-like --- #
    results.update(run_arima_like_models(train, val, test))

    # --- GARCH --- #
    results["garch"] = run_garch(train, val, test)

    # --- TBATS / Prophet --- #
    results["tbats"] = run_tbats_model(train, test)
    results["prophet"] = run_prophet_model(df[target])

    # --- VAR / VECM (multivariate) --- #
    results.update(run_var_vecm(df[["USD_RUB", "Brent", "MOEX", "Gold"]]))

    # ---------------------------
    # METRICS + RANKING
    # ---------------------------

    comparison_table = {}

    for name, pred in results.items():
        comparison_table[name] = evaluate_all_metrics(test, pred)

    comparison_df = pd.DataFrame(comparison_table).T
    comparison_df.to_csv("results/comparison_tables/models.csv")

    # ---------------------------
    # DIAGNOSTICS (TOP-3 MODELS)
    # ---------------------------

    top3 = comparison_df.sort_values("MASE").head(3).index

    for model_name in top3:
        full_diagnostics(
            y_true=test,
            y_pred=results[model_name],
            model_name=model_name,
            save_path="results/diagnostics/"
        )

    print("Готово! Все этапы выполнены.")
