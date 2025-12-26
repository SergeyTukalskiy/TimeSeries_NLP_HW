from tbats import TBATS
from prophet import Prophet
import pandas as pd

def run_tbats_model(train, test):
    model = TBATS(seasonal_periods=[7])
    m = model.fit(train)
    return m.forecast(steps=len(test))

def run_prophet_model(series):
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=30)
    fc = m.predict(future)
    return fc["yhat"].iloc[-30:].values
