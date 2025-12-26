from arch import arch_model

def run_garch(train, val, test):
    model = arch_model(train, vol="GARCH", p=1, q=1)
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=len(test))
    return forecast.mean.iloc[-1].values
