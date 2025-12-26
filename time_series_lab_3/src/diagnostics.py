from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
import matplotlib.pyplot as plt

def full_diagnostics(y_true, y_pred, model_name, save_path):

    resid = y_true - y_pred

    lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].values[0]
    sw_p = shapiro(resid)[1]

    plt.figure(figsize=(10,4))
    plt.plot(resid)
    plt.title(f"Residuals: {model_name}")
    plt.savefig(f"{save_path}/{model_name}_residuals.png")

    print(model_name, lb_p, sw_p)
