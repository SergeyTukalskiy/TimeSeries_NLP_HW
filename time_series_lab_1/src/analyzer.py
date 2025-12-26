import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


class TimeSeriesAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results = {}
        os.makedirs('output', exist_ok=True)

    def get_descriptive_stats(self):
        """Дескриптивная статистика"""
        stats_df = self.df.describe()

        # Добавляем дополнительные метрики
        stats_df.loc['variance'] = self.df.var()
        stats_df.loc['skewness'] = self.df.skew()
        stats_df.loc['kurtosis'] = self.df.kurtosis()
        stats_df.loc['cv'] = self.df.std() / self.df.mean()  # Коэффициент вариации

        return stats_df.round(4)

    def test_stationarity(self, column, alpha=0.05):
        """Тесты на стационарность ADF и KPSS"""
        series = self.df[column].dropna()

        # ADF тест
        adf_result = adfuller(series)
        adf_statistic, adf_pvalue = adf_result[0], adf_result[1]

        # KPSS тест
        kpss_result = kpss(series, regression='c')
        kpss_statistic, kpss_pvalue = kpss_result[0], kpss_result[1]

        # Интерпретация
        is_stationary_adf = adf_pvalue < alpha
        is_stationary_kpss = kpss_pvalue > alpha

        print(
            f"ADF: statistic={adf_statistic:.4f}, p-value={adf_pvalue:.4f} -> {'Стационарен' if is_stationary_adf else 'Нестационарен'}")
        print(
            f"KPSS: statistic={kpss_statistic:.4f}, p-value={kpss_pvalue:.4f} -> {'Стационарен' if is_stationary_kpss else 'Нестационарен'}")

        # Сохраняем результаты
        self.results[f'stationarity_{column}'] = {
            'adf': {'statistic': adf_statistic, 'pvalue': adf_pvalue, 'stationary': is_stationary_adf},
            'kpss': {'statistic': kpss_statistic, 'pvalue': kpss_pvalue, 'stationary': is_stationary_kpss}
        }

        return is_stationary_adf, is_stationary_kpss

    def get_correlations(self, method='spearman'):
        """Корреляционный анализ"""
        return self.df.corr(method=method)

    def decompose_time_series(self, column, period=30, model='additive'):
        """Декомпозиция временного ряда"""
        series = self.df[column].dropna()

        try:
            decomposition = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')

            # Визуализация декомпозиции
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            fig.suptitle(f'Декомпозиция ряда: {column}', fontsize=16)

            decomposition.observed.plot(ax=axes[0], title='Исходный ряд')
            decomposition.trend.plot(ax=axes[1], title='Тренд')
            decomposition.seasonal.plot(ax=axes[2], title='Сезонность')
            decomposition.resid.plot(ax=axes[3], title='Остатки')

            for ax in axes:
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'output/decomposition_{column}.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Декомпозиция {column} сохранена в output/decomposition_{column}.png")

        except Exception as e:
            print(f"❌ Ошибка при декомпозиции {column}: {e}")

    def plot_acf_pacf(self, column, lags=40):
        """Графики ACF и PACF"""
        series = self.df[column].dropna()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Автокорреляционные функции: {column}', fontsize=16)

        # ACF
        plot_acf(series, lags=lags, ax=ax1, title='Autocorrelation Function (ACF)')
        # PACF
        plot_pacf(series, lags=lags, ax=ax2, title='Partial Autocorrelation Function (PACF)', method='ywm')

        plt.tight_layout()
        plt.savefig(f'output/acf_pacf_{column}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ ACF/PACF {column} сохранены в output/acf_pacf_{column}.png")

    def create_lagged_features(self, column, lags=[1, 7, 30]):
        """Создание лаговых признаков"""
        df_lagged = self.df.copy()

        for lag in lags:
            df_lagged[f'{column}_lag_{lag}'] = self.df[column].shift(lag)

        return df_lagged

    def calculate_rolling_stats(self, column, windows=[7, 30]):
        """Расчет скользящих статистик"""
        df_rolling = self.df.copy()

        for window in windows:
            df_rolling[f'{column}_rolling_mean_{window}'] = self.df[column].rolling(window=window).mean()
            df_rolling[f'{column}_rolling_std_{window}'] = self.df[column].rolling(window=window).std()
            df_rolling[f'{column}_rolling_min_{window}'] = self.df[column].rolling(window=window).min()
            df_rolling[f'{column}_rolling_max_{window}'] = self.df[column].rolling(window=window).max()

        return df_rolling