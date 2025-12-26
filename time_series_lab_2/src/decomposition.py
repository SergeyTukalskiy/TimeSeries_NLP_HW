import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, List

class DecompositionAnalyzer:
    """Анализ декомпозиции временных рядов"""
    
    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self.results = {}
    
    def decompose(self, period: int = 7, model: str = 'additive') -> Dict:
        """Декомпозиция временного ряда"""
        try:
            decomposition = seasonal_decompose(self.series, model=model, period=period, extrapolate_trend='freq')
            self.results[model] = decomposition
            return decomposition
        except Exception as e:
            print(f"Ошибка декомпозиции: {e}")
            return None
    
    def test_stationarity(self, series: pd.Series) -> Dict:
        """Тесты на стационарность"""
        # ADF тест
        adf_result = adfuller(series.dropna())
        # KPSS тест
        kpss_result = kpss(series.dropna(), regression='c')
        
        return {
            'ADF': {'statistic': adf_result[0], 'pvalue': adf_result[1]},
            'KPSS': {'statistic': kpss_result[0], 'pvalue': kpss_result[1]}
        }
    
    def analyze_residuals(self, decomposition_result, model_type: str) -> Dict:
        """Анализ остатков декомпозиции"""
        residuals = decomposition_result.resid.dropna()
        
        # Стационарность остатков
        stationarity = self.test_stationarity(residuals)
        
        # Нормальность
        normality_test = stats.shapiro(residuals)
        
        # Автокорреляция
        acf_test = self._test_autocorrelation(residuals)
        
        return {
            'stationarity': stationarity,
            'normality': {'statistic': normality_test[0], 'pvalue': normality_test[1]},
            'autocorrelation': acf_test,
            'residuals_mean': residuals.mean(),
            'residuals_std': residuals.std()
        }
    
    def _test_autocorrelation(self, series: pd.Series, lags: int = 20) -> Dict:
        """Тест на автокорреляцию"""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(series, lags=[lags])
        return {'statistic': lb_test['lb_stat'].iloc[0], 'pvalue': lb_test['lb_pvalue'].iloc[0]}
    
    def compare_decompositions(self, periods: List[int] = [7, 30]) -> Dict:
        """Сравнение различных декомпозиций"""
        comparison = {}
        
        for period in periods:
            for model in ['additive', 'multiplicative']:
                if model == 'multiplicative' and (self.series <= 0).any():
                    continue
                
                key = f"{model}_period_{period}"
                decomposition = self.decompose(period, model)
                
                if decomposition is not None:
                    residual_analysis = self.analyze_residuals(decomposition, model)
                    comparison[key] = {
                        'decomposition': decomposition,
                        'residual_analysis': residual_analysis,
                        'period': period,
                        'model': model
                    }
        
        return comparison
    
    def find_best_decomposition(self, periods: List[int] = [7, 30]) -> Tuple[str, Dict]:
        """Поиск лучшей модели декомпозиции"""
        comparisons = self.compare_decompositions(periods)
        
        best_score = float('inf')
        best_key = None
        
        for key, result in comparisons.items():
            # Критерий: минимум стандартного отклонения остатков + p-value теста на автокорреляцию
            resid_std = result['residual_analysis']['residuals_std']
            acf_pvalue = result['residual_analysis']['autocorrelation']['pvalue']
            
            score = resid_std * (1 + (1 - acf_pvalue))  # Штраф за автокорреляцию
            
            if score < best_score:
                best_score = score
                best_key = key
        
        return best_key, comparisons[best_key] if best_key else None