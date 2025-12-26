import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mase(y_true, y_pred, y_train=None):
    if y_train is None:
        return np.nan
    d = np.mean(np.abs(np.diff(y_train)))
    return mean_absolute_error(y_true, y_pred) / d

def smape(y_true, y_pred):
    return 100 * np.mean(2*np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def evaluate_all_metrics(y_true, y_pred, y_train=None):
    # --- Приводим длину y_pred к длине y_true ---
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_pred) != len(y_true):
        if len(y_pred) > len(y_true):
            y_pred = y_pred[:len(y_true)]
        else:
            # если предсказаний меньше — заполним последним значением
            y_pred = np.pad(y_pred, (0, len(y_true) - len(y_pred)),
                            mode="edge")

    # теперь длины равны, можно считать метрики
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train),
        "RMSLE": rmsle(y_true, y_pred),
    }

