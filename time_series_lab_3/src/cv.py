import numpy as np

def expanding_window_cv(series, horizon_list=[1,7,30], min_train=200):
    """Expanding window: train=[0:t], test=[t+1:t+h]"""
    scores = {}

    for h in horizon_list:
        errors = []
        for t in range(min_train, len(series) - h):
            train = series[:t]
            true = series[t:t+h]

            # naive baseline
            pred = np.repeat(train.iloc[-1], h)

            err = np.abs(true.values - pred).mean()
            errors.append(err)

        scores[h] = np.mean(errors)

    return scores


def sliding_window_cv(series, window=365, horizon_list=[1,7,30]):
    """Sliding window: train=[t-w:t], test=[t:t+h]"""
    scores = {}

    for h in horizon_list:
        errors = []
        for t in range(window, len(series) - h):
            train = series[t-window:t]
            true = series[t:t+h]

            pred = np.repeat(train.iloc[-1], h)

            err = np.abs(true.values - pred).mean()
            errors.append(err)

        scores[h] = np.mean(errors)

    return scores
