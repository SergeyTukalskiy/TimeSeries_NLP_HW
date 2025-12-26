from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def run_arima_like_models(train, val, test):
    results = {}

    # ARIMA auto
    model = auto_arima(train, seasonal=False, trace=False)
    results["arima_auto"] = model.predict(n_periods=len(test))

    # SARIMA auto
    model_s = auto_arima(train, seasonal=True, m=7, trace=False)
    results["sarima_auto"] = model_s.predict(n_periods=len(test))

    # manual ARIMA (1,1,1)
    m = ARIMA(train, order=(1,1,1)).fit()
    results["arima_111"] = m.forecast(len(test))

    return results
