import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def naive_forecast(train, test):
    """Последнее значение → прогноз на весь горизонт."""
    last = train.iloc[-1]
    return np.repeat(last, len(test))

def seasonal_naive(train, test, season=7):
    """Повторяем значение сезонного лага."""
    values = train.iloc[-season:]
    reps = int(np.ceil(len(test) / season))
    return np.tile(values, reps)[:len(test)]

def ses_model(train, test):
    """Simple Exponential Smoothing."""
    model = SimpleExpSmoothing(train)
    m = model.fit()
    return m.forecast(len(test))
