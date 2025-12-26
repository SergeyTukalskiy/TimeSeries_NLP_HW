import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def adf(x):
    return adfuller(x, autolag="AIC")[1]

def kpss_test(x):
    return kpss(x, regression="c")[1]


def make_stationary(y):
    diff_info = {"d": 0, "sd": 0}

    y_new = y.copy()

    # ADF wants stationary → p < 0.05
    # KPSS wants stationary → p > 0.05

    def ok(z):
        return adf(z) < 0.05 and kpss_test(z) > 0.05

    # difference(1)
    if not ok(y_new):
        y_new = y_new.diff().dropna()
        diff_info["d"] = 1

    # seasonal diff 7
    if not ok(y_new):
        y_new = y_new.diff(7).dropna()
        diff_info["sd"] = 7

    return y_new, diff_info
