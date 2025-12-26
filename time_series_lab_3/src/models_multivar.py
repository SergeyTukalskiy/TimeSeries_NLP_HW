import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler

def run_var_vecm(df):
    results = {}

    # ---- 1. Выбор переменных ----
    cols = ["USD_RUB", "Brent", "MOEX"]  # минимально стабильный набор
    df_small = df[cols].copy()

    # ---- 2. Чистка пропусков ----
    df_small = df_small.dropna()

    # ---- 3. Масштабирование ----
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_small),
        index=df_small.index,
        columns=df_small.columns
    )

    # ---- 4. Обучение VAR ----
    try:
        model = VAR(df_scaled)
        sel = model.select_order(10)
        lag = sel.selected_orders["aic"]

        if lag is None:
            lag = 1

        res = model.fit(lag)

        # ---- 5. Прогноз на 30 шагов ----
        fc_scaled = res.forecast(df_scaled.values[-lag:], steps=30)
        fc = scaler.inverse_transform(fc_scaled)

        results["var"] = fc[:, 0]  # прогноз для USD_RUB

    except Exception as e:
        print("VAR failed:", e)
        results["var"] = np.repeat(df_small["USD_RUB"].iloc[-1], 30)

    return results
